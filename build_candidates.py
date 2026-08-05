#!/usr/bin/env python3
"""
Pre-filtered candidate list for pnl_replay.py --live.

Free (--dry-equivalent, no API calls): reuses replay.py's own fetch_all_bars
and replay_symbol() -- the exact same structural check that produces
replay_history.csv -- across a set of non-overlapping windows, and emits one
candidate (symbol, ts) row per BOS/CHoCH_confirmed/both hit, instead of
pnl_replay.py walking every sequential M15 bar regardless of whether
anything structural is happening. This is what lets a paid --live run spend
its budget on moments a real setup existed, rather than mostly on quiet
snapshots (this session's evidence: 0 opens across 400 real, sequentially-
walked snapshots).

Each hit's h1_time is the H1 bar's OPEN time (MT5 convention); the bar
becomes fully known only at its own close (h1_time + 1H). The earliest M15
decision timestamp at which pnl_replay.py's ReplayClient would actually show
this bar as the latest closed H1 bar is h1_time + 45 minutes (see
replay.py's closed_cutoff() and pnl_replay.py's identical fix -- a decision
at M15 ts sees H1 bars closed by ts + 15min - 1H = ts - 45min, so the bar
first becomes visible at ts = h1_time + 45min). That's the candidate
timestamp written here, per symbol.

Appends to --output by default, same convention as replay.py's
--history-file (a running record), so re-runs across different --windows
accumulate rather than silently overwrite. Pass --fresh to start clean.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd
import MetaTrader5 as mt5

from agent.config import Config
from replay import ConjunctionHit, fetch_all_bars, replay_symbol

CANDIDATE_COLUMNS = ["symbol", "ts", "origin", "h4_trend", "bars_since_break", "retracement_pct",
                     "source_offset_days"]


def build_candidates(cfg: Config, offset_days: int, lookback_days: int,
                      max_bars_since_break: int, visible_offset_minutes: int) -> list[dict]:
    window = cfg.bars_per_timeframe
    rows = []
    for symbol in cfg.symbols:
        mt5.symbol_select(symbol, True)
        h4 = fetch_all_bars(symbol, mt5.TIMEFRAME_H4, lookback_days, offset_days)
        h1 = fetch_all_bars(symbol, mt5.TIMEFRAME_H1, lookback_days, offset_days)
        hits, _steps = replay_symbol(symbol, h4, h1, window, max_bars_since_break)
        for h in hits:
            ts = h.h1_time + pd.Timedelta(minutes=visible_offset_minutes)
            rows.append({
                "symbol": h.symbol, "ts": ts.isoformat(), "origin": h.origin,
                "h4_trend": h.h4_trend, "bars_since_break": h.h1_bars_since_break,
                "retracement_pct": h.h1_retracement_pct, "source_offset_days": offset_days,
            })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--windows", default="0,90,180",
                         help="Comma-separated offset_days values, one per non-overlapping "
                              "window -- matches replay_history.csv's standard 3-window setup.")
    parser.add_argument("--lookback-days", type=int, default=90)
    parser.add_argument("--max-bars-since-break", type=int, default=10,
                         help="Must match the replay.py run that produced the structural "
                              "baseline you're comparing against, or the hit list won't agree.")
    parser.add_argument("--visible-offset-minutes", type=int, default=45,
                         help="Minutes after an H1 hit's open time at which it first becomes "
                              "visible to a M15 decision snapshot -- see module docstring.")
    parser.add_argument("--output", default="pnl_replay_candidates.csv")
    parser.add_argument("--fresh", action="store_true",
                         help="Overwrite --output instead of appending.")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    if not mt5.initialize():
        raise SystemExit(f"mt5.initialize failed: {mt5.last_error()}")

    all_rows = []
    for offset_days in (int(x) for x in args.windows.split(",")):
        rows = build_candidates(cfg, offset_days, args.lookback_days,
                                 args.max_bars_since_break, args.visible_offset_minutes)
        print(f"offset_days={offset_days}: {len(rows)} candidates")
        all_rows.extend(rows)

    mt5.shutdown()

    mode = "w" if args.fresh else "a"
    path = Path(args.output)
    write_header = args.fresh or not path.exists()
    with open(path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CANDIDATE_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(all_rows)

    by_symbol: dict[str, int] = {}
    by_origin: dict[str, int] = {}
    for r in all_rows:
        by_symbol[r["symbol"]] = by_symbol.get(r["symbol"], 0) + 1
        by_origin[r["origin"]] = by_origin.get(r["origin"], 0) + 1

    print(f"\nTotal candidates this run: {len(all_rows)}")
    print(f"By symbol: {by_symbol}")
    print(f"By origin: {by_origin}")
    print(f"Written ({'fresh' if args.fresh else 'appended'}) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
