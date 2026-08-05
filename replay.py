#!/usr/bin/env python3
"""
Replay backtester (BUILD_PLAN.md Phase 4).

--dry mode (the only mode implemented so far): steps through historical H4/H1
bars for each configured symbol and reuses features.market_structure() -- the
actual production structure-detection code, not a reimplementation -- to
check how often the strategy brief's entry conjunction actually occurs, via
two conjunctions (see the "both" bucket note below for why they can overlap):

  BOS-origin (continuation, trend-aligned):
    H4 has a directional trend (not ranging)
  + H1 trend agrees with H4's direction
  + H1 shows a break-of-structure retest with retracement_pct in [20, 80]
  + that retest is fresh (bars_since_break <= --max-bars-since-break)

  CHoCH_confirmed-origin (validated reversal):
    H4 has a directional trend (not ranging)
  + H1's choch_confirmed is true -- a second, independent swing has broken
    further in the reversal's direction, with that newer swing's own retest
    in the 20-80% band (see features.market_structure's docstring)

  This path deliberately does NOT require h1_trend == h4_trend: a confirmed
  reversal is, by nature, often early relative to the slower H4 timeframe --
  that is the entire value of catching it. Gating it on H4/H1 alignment would
  filter out exactly the setups it exists to surface.

  The two paths are NOT always mutually exclusive: choch_confirmed requires
  H1's last_event to be a CHoCH (h1_trend != the break direction), but
  h1_trend can still equal h4_trend if h1_trend itself is the *other*
  direction (e.g. a CHoCH_bearish inside an H1 uptrend, where h1_trend is
  "bullish" and h4_trend also happens to be "bullish"). When both criteria
  fire on the same bar-close, it is bucketed as its own third origin, "both"
  -- never silently forced into one bucket or the other, and never dropped.
  Per-origin and combined totals below are computed from these three
  mutually exclusive buckets, so the combined total is an exact sum with
  neither double-counting nor undercounting.

No API calls, no risk.py, no order simulation -- this only answers "how often
does the setup exist," not "would the model have taken it" or "would it have
won." Live replay (calling the model and running the real risk engine) is not
built yet.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

import MetaTrader5 as mt5

from agent.config import Config
from agent import features
from agent.mt5_client import TIMEFRAME_SECONDS


@dataclass
class ConjunctionHit:
    symbol: str
    h1_time: pd.Timestamp
    h4_trend: str
    h1_bars_since_break: int
    h1_retracement_pct: float
    origin: str  # "BOS", "CHoCH_confirmed", or "both" -- see module docstring for the convention


def fetch_all_bars(symbol: str, timeframe_const: int, lookback_days: int,
                    offset_days: int = 0) -> pd.DataFrame:
    """
    Pull `lookback_days` of history for this symbol/timeframe, ending
    `offset_days` ago instead of now -- lets a second, non-overlapping window
    be pulled for a sanity check (e.g. offset_days=lookback_days gives the
    window immediately preceding the default one) without touching the
    default single-window behaviour.

    Deliberately bounded rather than "as much as the broker has": open-ended
    copy_rates_range calls (e.g. from year 2000) were found to hang the MT5
    terminal itself when the requested range isn't already cached locally --
    not just fail -- which twice required a cooldown to recover and risked
    the live scheduled cycle task. A conservative recent window trades sample
    size for not touching a shared resource unpredictably.
    """
    date_to = datetime.now(timezone.utc) - timedelta(days=offset_days)
    date_from = date_to - timedelta(days=lookback_days)
    rates = mt5.copy_rates_range(symbol, timeframe_const, date_from, date_to)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"No history returned for {symbol}: {mt5.last_error()}")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df.set_index("time").sort_index()


def closed_cutoff(ts: pd.Timestamp, grain_period: pd.Timedelta, timeframe: str) -> pd.Timestamp:
    """
    The latest bar-OPEN timestamp for `timeframe` that has FULLY CLOSED by the
    moment a bar at `ts` (on the caller's own evaluation grain, period
    `grain_period`) itself becomes known -- i.e. at ts + grain_period, not at
    ts itself. MT5 bar timestamps are open times, so naively filtering a
    COARSER timeframe with `df.index <= ts` is up to one whole coarser bar too
    permissive: it can show the still-forming current candle's
    already-known-in-hindsight final OHLC before real time would have made it
    available. For `timeframe == ` the caller's own grain this is a no-op
    (reduces to exactly `ts`); for a coarser one it isn't.

    Canonical home for this helper: replay.py doesn't import from
    pnl_replay.py (that would be circular, since pnl_replay.py already
    imports fetch_all_bars from here) so pnl_replay.py imports this from
    replay.py instead of duplicating it. Confirmed empirically before this
    fix: replay_symbol()'s H4-against-H1 grain leaked on 75% of H1
    evaluation steps, up to 3 hours of premature H4 data at the worst case
    (H1 landing exactly on an H4 boundary) -- the identical class of bug
    pnl_replay.py's snapshots had for H1/H4 against an M15 grain.
    """
    true_now = ts + grain_period
    return true_now - pd.Timedelta(seconds=TIMEFRAME_SECONDS[timeframe])


def replay_symbol(symbol: str, h4_df: pd.DataFrame, h1_df: pd.DataFrame,
                   window: int, max_bars_since_break: int) -> tuple[list[ConjunctionHit], int]:
    """
    Slide a `window`-bar trailing window across H1 bar closes, using only H4/H1
    data that would have been available AT that H1 bar's close (no lookahead),
    and evaluate the strategy brief's conjunction at each step.

    Returns (hits, steps_evaluated).
    """
    hits: list[ConjunctionHit] = []
    steps = 0
    h1_period = pd.Timedelta(seconds=TIMEFRAME_SECONDS["H1"])

    for i in range(window, len(h1_df)):
        h1_window = h1_df.iloc[i - window + 1: i + 1]
        h1_now = h1_window.index[-1]

        # No-lookahead guard: every bar in both windows must be <= the
        # evaluation point, and the H1 window's last bar must BE the
        # evaluation point (not something later).
        assert h1_window.index[-1] == h1_now
        assert (h1_window.index <= h1_now).all()

        # H1 is this function's own grain, so h1_now itself becomes known at
        # its own close (h1_now + h1_period) -- H4 must be filtered against
        # THAT moment, not against h1_now's open time. See closed_cutoff().
        h4_cutoff = closed_cutoff(h1_now, h1_period, "H4")

        # pandas' own tz-aware searchsorted -- raw numpy datetime64 silently
        # drops tzinfo and produces wrong (or erroring) comparisons.
        h4_pos = h4_df.index.searchsorted(h4_cutoff, side="right")
        if h4_pos < window:
            continue  # not enough H4 history yet
        h4_window = h4_df.iloc[h4_pos - window: h4_pos]
        assert (h4_window.index <= h4_cutoff).all(), "H4 lookahead leak"

        steps += 1
        h4_structure = features.market_structure(h4_window)
        h1_structure = features.market_structure(h1_window)

        h4_trend = h4_structure["trend"]
        h1_trend = h1_structure["trend"]
        if h4_trend not in ("bullish", "bearish"):
            continue

        bsb = h1_structure["bars_since_break"]
        retr = h1_structure["retracement_pct"]

        # BOS-origin: H1 continuation aligned with H4's established direction.
        is_bos = (h1_trend == h4_trend and bsb is not None and retr is not None
                  and 20 <= retr <= 80 and int(bsb) <= max_bars_since_break)

        # CHoCH_confirmed-origin: H1 reversal validated by a follow-through swing
        # + retest (see module docstring for why this doesn't gate on alignment).
        # bsb/retr here are the ORIGINAL CHoCH's own numbers, not the confirming
        # swing's -- market_structure doesn't expose the latter -- so they're
        # reported for visibility only, not filtered on.
        is_choch_confirmed = (h1_structure.get("choch_confirmed")
                               and bsb is not None and retr is not None)

        # Explicit three-way bucketing (see module docstring): a bar-close
        # satisfying both criteria goes to "both", not silently to whichever
        # check happens to run first.
        if is_bos and is_choch_confirmed:
            origin = "both"
        elif is_bos:
            origin = "BOS"
        elif is_choch_confirmed:
            origin = "CHoCH_confirmed"
        else:
            origin = None

        if origin is not None:
            hits.append(ConjunctionHit(symbol, h1_now, h4_trend, int(bsb), retr, origin=origin))

    return hits, steps


def group_episodes(hits: list[ConjunctionHit]) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    """Collapse consecutive H1-bar hits into contiguous windows (one 'opportunity' each)."""
    if not hits:
        return []
    episodes = []
    start = prev = hits[0].h1_time
    count = 1
    for h in hits[1:]:
        gap_hours = (h.h1_time - prev).total_seconds() / 3600
        if gap_hours <= 1.01:  # contiguous H1 bars
            prev = h.h1_time
            count += 1
        else:
            episodes.append((start, prev, count))
            start = prev = h.h1_time
            count = 1
    episodes.append((start, prev, count))
    return episodes


def get_code_version() -> str:
    """
    Short git commit hash this run executed under, '-dirty' appended if the
    working tree has uncommitted changes -- so replay_history.csv rows stay
    attributable to the market_structure()/replay.py logic that produced
    them without cross-referencing git log by timestamp. Broad "any
    uncommitted change" dirty check rather than scoping to agent/features.py
    specifically: if anything is uncommitted, the run isn't reproducible from
    a clean checkout, which is the property that actually matters here.
    Never raises -- a version-tracking nicety must not crash the whole run.
    """
    try:
        repo_dir = Path(__file__).parent
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, cwd=repo_dir,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=True, cwd=repo_dir,
        ).stdout.strip()
        return f"{commit}-dirty" if dirty else commit
    except Exception:  # noqa: BLE001 -- version tag is best-effort, never fatal
        return "unknown"


HISTORY_COLUMNS = [
    "run_ts", "code_version", "window_start", "window_end", "offset_days",
    "lookback_days", "max_bars_since_break", "symbol", "bos_hits",
    "choch_confirmed_hits", "both_hits", "total_hits", "steps", "combined_rate_pct",
    "bos_episodes", "choch_confirmed_episodes", "both_episodes", "total_episodes",
]


def log_run(path: str, run_ts: datetime, code_version: str, args, per_symbol_window: dict,
            per_symbol_origin_stats: dict) -> None:
    """
    Append this run's per-symbol and combined ('ALL') results to a running
    CSV, so window-to-window comparisons accumulate into a track record
    instead of being re-derived from memory (or old chat scrollback) each
    time replay.py is run. Writes the header only when creating the file.
    """
    file_exists = Path(path).exists()
    rows = []
    all_row = {"bos_hits": 0, "choch_confirmed_hits": 0, "both_hits": 0, "steps": 0,
               "bos_episodes": 0, "choch_confirmed_episodes": 0, "both_episodes": 0}
    all_starts, all_ends = [], []

    for symbol, stats in per_symbol_origin_stats.items():
        start, end = per_symbol_window[symbol]
        all_starts.append(start)
        all_ends.append(end)
        total_hits = stats["bos_hits"] + stats["choch_confirmed_hits"] + stats["both_hits"]
        total_episodes = stats["bos_episodes"] + stats["choch_confirmed_episodes"] + stats["both_episodes"]
        rows.append({
            "run_ts": run_ts.isoformat(), "code_version": code_version,
            "window_start": start.isoformat(),
            "window_end": end.isoformat(), "offset_days": args.offset_days,
            "lookback_days": args.lookback_days,
            "max_bars_since_break": args.max_bars_since_break,
            "symbol": symbol, "bos_hits": stats["bos_hits"],
            "choch_confirmed_hits": stats["choch_confirmed_hits"],
            "both_hits": stats["both_hits"], "total_hits": total_hits,
            "steps": stats["steps"],
            "combined_rate_pct": round(total_hits / stats["steps"] * 100, 2) if stats["steps"] else 0.0,
            "bos_episodes": stats["bos_episodes"],
            "choch_confirmed_episodes": stats["choch_confirmed_episodes"],
            "both_episodes": stats["both_episodes"], "total_episodes": total_episodes,
        })
        for key in all_row:
            all_row[key] += stats[key]

    all_total_hits = all_row["bos_hits"] + all_row["choch_confirmed_hits"] + all_row["both_hits"]
    all_total_episodes = (all_row["bos_episodes"] + all_row["choch_confirmed_episodes"]
                           + all_row["both_episodes"])
    rows.append({
        "run_ts": run_ts.isoformat(), "code_version": code_version,
        "window_start": min(all_starts).isoformat(),
        "window_end": max(all_ends).isoformat(), "offset_days": args.offset_days,
        "lookback_days": args.lookback_days, "max_bars_since_break": args.max_bars_since_break,
        "symbol": "ALL", "bos_hits": all_row["bos_hits"],
        "choch_confirmed_hits": all_row["choch_confirmed_hits"],
        "both_hits": all_row["both_hits"], "total_hits": all_total_hits,
        "steps": all_row["steps"],
        "combined_rate_pct": round(all_total_hits / all_row["steps"] * 100, 2) if all_row["steps"] else 0.0,
        "bos_episodes": all_row["bos_episodes"],
        "choch_confirmed_episodes": all_row["choch_confirmed_episodes"],
        "both_episodes": all_row["both_episodes"], "total_episodes": all_total_episodes,
    })

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dry", action="store_true", required=True,
                         help="Structural-only mode. No API calls. (Only mode implemented.)")
    parser.add_argument("--max-bars-since-break", type=int, default=10,
                         help="Threshold for 'fresh' H1 break, per bar. Brief says only "
                              "'broadly within the last several bars' -- no exact number "
                              "is configured, so this is an explicit assumption, checked "
                              "at multiple values for sensitivity.")
    parser.add_argument("--lookback-days", type=int, default=90,
                         help="How much history to pull per symbol. Kept conservative -- "
                              "see fetch_all_bars() docstring for why.")
    parser.add_argument("--offset-days", type=int, default=0,
                         help="Shift the whole window back by this many days (window ends "
                              "offset_days ago instead of now). Use e.g. "
                              "--offset-days=<lookback-days> to pull a second, "
                              "non-overlapping window immediately preceding the default one.")
    parser.add_argument("--history-file", default="replay_history.csv",
                         help="CSV this run's per-symbol + combined results are appended to, "
                              "so window-to-window comparisons accumulate into a track record "
                              "instead of being re-derived from memory each time.")
    args = parser.parse_args()

    run_ts = datetime.now(timezone.utc)
    code_version = get_code_version()
    cfg = Config.load(args.config)
    if not mt5.initialize():
        raise SystemExit(f"mt5.initialize failed: {mt5.last_error()}")

    window = cfg.bars_per_timeframe
    print(f"Code version: {code_version}")
    print(f"Symbols: {cfg.symbols}")
    print(f"Window: {window} bars per timeframe (matches config.bars_per_timeframe)")
    if args.offset_days:
        print(f"Lookback: {args.lookback_days} days, ending {args.offset_days} days ago "
              f"(shifted window, not the default)")
    print(f"BOS-origin conjunction: H4 trend directional + H1 trend aligned + H1 "
          f"retracement_pct in [20,80] + H1 bars_since_break <= {args.max_bars_since_break}")
    print("CHoCH_confirmed-origin conjunction: H4 trend directional + H1 choch_confirmed "
          "true (independent of H1/H4 alignment -- see module docstring)")
    print("'both' bucket: bar-closes satisfying both criteria simultaneously -- reported "
          "separately, not folded into either origin (see module docstring)")
    print(f"Strategy brief (unmodified, from config.yaml):\n{cfg.strategy_brief}\n")

    all_hits: dict[str, list[ConjunctionHit]] = {}
    all_steps: dict[str, int] = {}
    per_symbol_window: dict[str, tuple] = {}

    for symbol in cfg.symbols:
        mt5.symbol_select(symbol, True)
        h4_df = fetch_all_bars(symbol, mt5.TIMEFRAME_H4, args.lookback_days, args.offset_days)
        h1_df = fetch_all_bars(symbol, mt5.TIMEFRAME_H1, args.lookback_days, args.offset_days)
        print(f"{symbol}: H4 history {h4_df.index[0]} -> {h4_df.index[-1]} "
              f"({len(h4_df)} bars); H1 history {h1_df.index[0]} -> {h1_df.index[-1]} "
              f"({len(h1_df)} bars)")
        per_symbol_window[symbol] = (h1_df.index[0], h1_df.index[-1])

        hits, steps = replay_symbol(symbol, h4_df, h1_df, window, args.max_bars_since_break)
        all_hits[symbol] = hits
        all_steps[symbol] = steps

    mt5.shutdown()

    print("\n" + "=" * 70)
    print(f"RESULTS  (max_bars_since_break = {args.max_bars_since_break})")
    print("=" * 70)

    totals = {}
    for origin in ("BOS", "CHoCH_confirmed", "both"):
        print(f"\n--- {origin}-origin ---")
        total_steps = 0
        total_hit_bars = 0
        total_episodes = 0
        for symbol in cfg.symbols:
            hits = [h for h in all_hits[symbol] if h.origin == origin]
            steps = all_steps[symbol]
            episodes = group_episodes(hits)
            total_steps += steps
            total_hit_bars += len(hits)
            total_episodes += len(episodes)
            pct = len(hits) / steps * 100 if steps else 0.0
            print(f"\n{symbol}: {len(hits)}/{steps} H1 bar-closes matched "
                  f"({pct:.2f}% of evaluated H1 closes) in {len(episodes)} distinct episode(s)")
            for start, end, count in episodes[:15]:
                print(f"    {start} -> {end}  ({count} consecutive H1 bars)")
            if len(episodes) > 15:
                print(f"    ... and {len(episodes) - 15} more episodes")

        print(f"\n{origin} TOTAL across {len(cfg.symbols)} symbols: {total_hit_bars}/{total_steps} "
              f"H1 bar-closes matched, {total_episodes} distinct episodes")
        totals[origin] = (total_hit_bars, total_steps, total_episodes)

    # Sum across the three mutually exclusive buckets -- exactly once each,
    # no double-counting (a "both" hit isn't also in BOS or CHoCH_confirmed)
    # and no dropping (it isn't excluded from either either).
    combined_hits = sum(h for h, _, _ in totals.values())
    combined_steps = totals["BOS"][1]  # same denominator for all three buckets
    combined_episodes = sum(e for _, _, e in totals.values())
    print(f"\n--- combined ---")
    print(f"TOTAL across {len(cfg.symbols)} symbols: {combined_hits}/{combined_steps} "
          f"H1 bar-closes matched, {combined_episodes} distinct episodes "
          f"(BOS {totals['BOS'][0]} + CHoCH_confirmed {totals['CHoCH_confirmed'][0]} "
          f"+ both {totals['both'][0]})")
    # Rough equivalent at the live M15 decision cadence: each matching H1 bar
    # is "live" for up to 4 M15 cycles (assuming H1 structure doesn't change
    # mid-bar, which it can't by construction).
    print(f"Approx. M15-cycle-equivalent exposure: ~{combined_hits * 4} cycles out of "
          f"~{combined_steps * 4} would have seen either conjunction true, at M15 cadence")

    per_symbol_origin_stats = {}
    for symbol in cfg.symbols:
        by_origin = {o: [h for h in all_hits[symbol] if h.origin == o]
                     for o in ("BOS", "CHoCH_confirmed", "both")}
        per_symbol_origin_stats[symbol] = {
            "bos_hits": len(by_origin["BOS"]),
            "choch_confirmed_hits": len(by_origin["CHoCH_confirmed"]),
            "both_hits": len(by_origin["both"]),
            "steps": all_steps[symbol],
            "bos_episodes": len(group_episodes(by_origin["BOS"])),
            "choch_confirmed_episodes": len(group_episodes(by_origin["CHoCH_confirmed"])),
            "both_episodes": len(group_episodes(by_origin["both"])),
        }
    log_run(args.history_file, run_ts, code_version, args, per_symbol_window, per_symbol_origin_stats)
    print(f"\nAppended this run to {args.history_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
