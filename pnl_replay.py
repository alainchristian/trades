#!/usr/bin/env python3
"""
P&L-simulating replay (BUILD_PLAN.md Phase 4) -- the full harness.

Pulls historical multi-timeframe bars and builds the exact snapshot the live
system would have built at each historical decision point, via
agent.context.build_snapshot() -- completely unmodified, fed by a
ReplayClient standing in for the real MT5Client. Two modes:

  --dry   stubs the model call with a fixed placeholder response instead of
          calling the API, so response caching (by content hash), cost
          estimation, and the stop/target resolution logic all get built and
          verified with ZERO API spend before a single real dollar is on the
          table.

  --live  calls the real model via agent.decision.DecisionEngine, runs every
          proposed 'open' through the real agent.risk.RiskEngine, simulates
          the fill against subsequent price via resolve_outcome(), and
          tracks a simulated equity curve (run_live()/SimEquityState). No
          order_send anywhere in this file or anything it imports --
          agent.execution is the only module in this codebase that calls it
          (grep-verified). Always prints a cost estimate first and stops
          there unless --confirm is also passed, per BUILD_PLAN.md. Scope:
          single position per symbol at a time (no re-entry, no
          close/modify_stop simulation while one is open) -- see run_live()'s
          docstring for why that's faithful to production, not a shortcut.

Distinct from replay.py: that script only checks how often the strategy
brief's entry conjunction structurally occurs (features.market_structure()
alone -- no snapshot, no model, no risk engine). This is the fuller
simulation BUILD_PLAN.md Phase 4 actually describes; replay.py predates it
and remains useful as the cheaper, narrower frequency check (and its
--offset-days / replay_history.csv machinery is reused here for fetching).

Reuses, never reimplements: agent.context.build_snapshot, agent.decision.
DecisionEngine/build_user_content, agent.risk.RiskEngine,
agent.features.hold_confidence, agent.journal.SessionState (day_start/
session_open, extended with an optional historical `at` param so replay can
reuse the exact production gate instead of re-deriving it),
agent.mt5_client.MT5Client (real spec()/connect(), for the same safety
checks the live system gets) and tf_const, agent.analytics.load_cycles (to
calibrate the cost estimate against real observed token counts), and
replay.py's fetch_all_bars for historical OHLCV.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from dotenv import load_dotenv

import MetaTrader5 as mt5

from agent import analytics, context, features
from agent.config import Config
from agent.decision import Decision, DecisionEngine, build_user_content
from agent.journal import SessionState
from agent.mt5_client import MT5Client, SymbolSpec, TIMEFRAME_SECONDS, tf_const
from agent.risk import RiskEngine
from replay import closed_cutoff, fetch_all_bars


def fetch_closed_bars(symbol: str, timeframe_const: int, lookback_days: int,
                       offset_days: int = 0) -> pd.DataFrame:
    """
    fetch_all_bars() fetches via copy_rates_range with date_to essentially
    "now" -- the last bar it returns can be a still-forming candle whose
    OHLC keeps changing as new ticks arrive, confirmed empirically: two
    fetches of the same range a few seconds apart can (intermittently,
    depending on tick activity) return different values for that bar, and
    since it's included as the tail of every timeframe's rolling window,
    ONE new tick in a still-forming H4 candle can silently change the
    snapshot hash for every M15 replay point within that whole 4-hour span
    -- which would defeat cache-hash reproducibility across separate runs
    without ever changing what "now" conceptually means.

    This is the same hazard CLAUDE.md invariant #6 already documents for
    MT5Client.bars()'s drop_forming default; fetch_all_bars() just doesn't
    apply it (reasonable for replay.py's own structural-frequency-only use,
    where near-"now" instability rarely flips a trend classification, but
    not acceptable here where reproducibility is the entire point). Always
    drops the last bar rather than trying to detect whether a given call
    happened to land exactly on a boundary.
    """
    df = fetch_all_bars(symbol, timeframe_const, lookback_days, offset_days)
    return df.iloc[:-1]


# ------------------------------------------------------- stop/target resolution ---

def resolve_outcome(direction: str, entry: float, stop_loss: float,
                     take_profit: float, subsequent_bars: pd.DataFrame) -> dict:
    """
    Walk forward through subsequent_bars (chronological, must have high/low)
    to determine whether stop_loss or take_profit was hit first.

    A bar whose range contains BOTH levels is genuinely ambiguous -- OHLC
    alone can't say which was touched first intra-bar -- and resolves to a
    loss (SL), not a win. Assuming the favourable outcome on an ambiguous
    bar is exactly the kind of quiet optimism that inflates a backtest's
    apparent edge without there being a real edge behind it.
    """
    is_buy = direction == "buy"
    for ts, bar in subsequent_bars.iterrows():
        hi, lo = float(bar["high"]), float(bar["low"])
        if is_buy:
            hit_sl, hit_tp = lo <= stop_loss, hi >= take_profit
        else:
            hit_sl, hit_tp = hi >= stop_loss, lo <= take_profit
        if hit_sl:
            reason = "ambiguous bar (both levels in range), resolved pessimistically to SL" \
                if hit_tp else "stop hit"
            return {"outcome": "loss", "exit_price": stop_loss, "exit_time": ts, "reason": reason}
        if hit_tp:
            return {"outcome": "win", "exit_price": take_profit, "exit_time": ts, "reason": "target hit"}
    return {"outcome": "open", "exit_price": None, "exit_time": None,
            "reason": "no resolution within available history"}


# --------------------------------------------------------------- replay client ---
# closed_cutoff() lives in replay.py (imported above) -- it doesn't depend on
# anything here, and replay.py can't import it back from this file without a
# circular import (this file already imports fetch_all_bars from replay.py).
# See its docstring there: replay_symbol()'s H1/H4 grain had the identical
# class of bug this fixes for M15/H1/H4 snapshots here.


class ReplayClient:
    """
    Stand-in for MT5Client, feeding agent.context.build_snapshot() point-in-
    time historical data instead of live data -- so build_snapshot() runs
    completely unmodified during replay. set_time() moves the replay clock
    for a symbol before each snapshot; bars()/tick() only ever see data that
    has fully closed by the true decision moment (see closed_cutoff()) --
    slices of already-fetched historical arrays, there is nothing later to
    leak, and now no still-forming current-timeframe bar either.

    positions() is fixed empty for both --dry and --live: a symbol is only
    ever evaluated while flat on it (see run_live()'s open-position skip), so
    a symbol's OWN open_positions in its OWN snapshot is always genuinely [].
    Cross-symbol book state (for the risk engine's total-risk-across-the-book
    check) is tracked separately in run_live()'s SimEquityState and passed to
    RiskEngine.evaluate_open() directly -- it doesn't need to flow through
    this client. account() defaults to a fixed $100k for --dry (never opens a
    position, nothing to compound); --live calls set_equity() to keep it
    reflecting the real simulated equity curve as trades close.
    """

    def __init__(self, specs: dict, bars_by_symbol_tf: dict, decision_period: pd.Timedelta):
        self.specs = specs
        self.bars_by_symbol_tf = bars_by_symbol_tf
        self.decision_period = decision_period
        self._now: dict[str, pd.Timestamp] = {}  # decision timeframe's own bar-open ts, per symbol
        self._equity = 100_000.0

    def set_time(self, symbol: str, ts: pd.Timestamp) -> None:
        self._now[symbol] = ts

    def spec(self, symbol: str):
        return self.specs[symbol]

    def bars(self, symbol: str, timeframe: str, count: int, drop_forming: bool = True) -> pd.DataFrame:
        df = self.bars_by_symbol_tf[(symbol, timeframe)]
        cutoff = closed_cutoff(self._now[symbol], self.decision_period, timeframe)
        return df[df.index <= cutoff].tail(count)

    def tick(self, symbol: str):
        cutoff = closed_cutoff(self._now[symbol], self.decision_period, "M15")
        m15 = self.bars_by_symbol_tf[(symbol, "M15")]
        close = float(m15[m15.index <= cutoff]["close"].iloc[-1])
        spec = self.specs[symbol]
        half_spread = spec.spread_points * spec.point / 2
        return SimpleNamespace(bid=close - half_spread, ask=close + half_spread)

    def set_equity(self, equity: float) -> None:
        self._equity = equity

    def account(self) -> dict:
        return {"equity": self._equity, "balance": self._equity, "currency": "USD",
                "margin_free": self._equity, "trade_allowed": True}

    def positions(self, symbol: str | None = None) -> list[dict]:
        return []  # a symbol is only ever evaluated while flat -- see class docstring


def load_or_fetch_specs(client: MT5Client, symbols: list[str], path: str) -> dict:
    """
    Symbol specs fetched fresh from live MT5 differ run to run for anything
    with non-zero live spread -- confirmed empirically: XAUUSD's
    spread_points varied 22 -> 22 -> 21 -> 20 across calls a second apart,
    while this account's FX majors happened to sit at a stable 0. Feeding a
    fresh live spec into tick() every run would silently defeat cache-hash
    reproducibility for any symbol whose spread actually moves. Persist the
    first fetch and reuse it, so replay stays keyed to one fixed spec
    snapshot rather than whatever the live market happens to show right now.
    A real (live/paid) pass will want a proper historical spread model; this
    is the deliberately simple version for the --dry reproducibility proof.
    """
    p = Path(path)
    if p.exists():
        raw = json.loads(p.read_text(encoding="utf-8"))
        return {sym: SymbolSpec(**raw[sym]) for sym in symbols}
    specs = {s: client.spec(s) for s in symbols}
    p.write_text(json.dumps({s: asdict(spec) for s, spec in specs.items()}, indent=2), encoding="utf-8")
    return specs


# ----------------------------------------------------------------------- caching ---

def snapshot_hash(snapshot: dict) -> str:
    """
    Content hash of a snapshot, excluding generated_at. build_snapshot()
    stamps that field with the real wall-clock time on every call, so two
    snapshots built from IDENTICAL historical data at the same replay point
    would otherwise never hash the same -- defeating caching entirely.
    """
    stable = {k: v for k, v in snapshot.items() if k != "generated_at"}
    encoded = json.dumps(stable, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()


def load_cache(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def save_cache(path: str, cache: dict) -> None:
    Path(path).write_text(json.dumps(cache, indent=2, default=str), encoding="utf-8")


# ------------------------------------------------------------------- token/cost ---

def fixed_prompt_overhead_chars() -> int:
    """
    SYSTEM_PROMPT and the DECISION_TOOL schema are sent on every API call,
    not just user_content -- the API's usage.input_tokens covers all of it.
    Missing this entirely was the single biggest source of error in an
    earlier version of this estimate (it compared a user_content-only
    character count against the real, full input_tokens figure and was off
    by ~2.9x as a result).
    """
    return len(context.SYSTEM_PROMPT) + len(json.dumps(context.DECISION_TOOL))


def calibrate_chars_per_token(cfg: Config, logs_dir: str) -> float:
    """
    Empirically derived chars/token ratio from real logged cycles, rather
    than a blind rule of thumb. JSON-heavy structured content tokenizes far
    more densely than plain English: one real logged cycle showed 6616
    snapshot characters for 5317 total input tokens once the fixed
    SYSTEM_PROMPT + tool-schema overhead above is included -- a naive
    "~4 chars/token" guess would have been roughly 2x too generous even
    with the overhead counted, and ~2.9x too generous without it. Falls
    back to 4.0 (a plain-English default) only if no real logs exist yet
    to calibrate against.
    """
    fixed = fixed_prompt_overhead_chars()
    df = analytics.load_cycles(logs_dir)
    ratios = []
    for path in sorted(Path(logs_dir).glob("cycles_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            real_input = (e.get("cost") or {}).get("input_tokens") or 0
            snapshot = e.get("snapshot")
            if not real_input or not snapshot:
                continue
            user_content = build_user_content(cfg.strategy_brief, snapshot)
            ratios.append((fixed + len(user_content)) / real_input)
    if not ratios:
        return 4.0
    return sum(ratios) / len(ratios)


def estimate_input_tokens(user_content: str, chars_per_token: float) -> int:
    total_chars = fixed_prompt_overhead_chars() + len(user_content)
    return max(1, int(total_chars / chars_per_token))


def observed_token_averages(logs_dir: str) -> tuple[float, float]:
    """
    Real average input/output tokens from logs/cycles_*.jsonl (actual
    Sonnet 5 calls from the live dry_run scheduled task) -- used to stand
    in for output tokens, since the --dry stub's own response is a trivial
    placeholder and using ITS length would badly underestimate what a real
    'hold' response (with real reasoning) actually costs.
    """
    df = analytics.load_cycles(logs_dir)
    if df.empty:
        return 0.0, 0.0
    return float(df["input_tokens"].mean()), float(df["output_tokens"].mean())


# ------------------------------------------------------------------------- main ---

def run_dry(cfg: Config, args: argparse.Namespace) -> None:
    client = MT5Client(cfg)
    client.connect()  # same demo-account / login-match safety checks the live system gets

    specs = load_or_fetch_specs(client, cfg.symbols, args.spec_cache_file)

    print(f"Fetching {args.lookback_days}d history (offset {args.offset_days}d) for "
          f"{cfg.symbols} x {cfg.context_timeframes}...")
    bars_by_symbol_tf = {}
    for symbol in cfg.symbols:
        for tf in cfg.context_timeframes:
            bars_by_symbol_tf[(symbol, tf)] = fetch_closed_bars(
                symbol, tf_const(tf), args.lookback_days, args.offset_days)

    client.shutdown()  # everything from here is pure computation, no MT5 needed

    decision_period = pd.Timedelta(seconds=TIMEFRAME_SECONDS[cfg.decision_timeframe])
    replay_client = ReplayClient(specs, bars_by_symbol_tf, decision_period)

    cache = load_cache(args.cache_file)
    cache_size_before = len(cache)

    real_avg_input, real_avg_output = observed_token_averages(args.logs_dir)
    chars_per_token = calibrate_chars_per_token(cfg, args.logs_dir)
    print(f"Calibrated chars/token: {chars_per_token:.2f} "
          f"(from {args.logs_dir}/; falls back to 4.0 if no real logs exist)")

    evaluated = []  # {symbol, ts, hash, cache_hit}
    skipped_warmup = 0

    for symbol in cfg.symbols:
        decision_df = bars_by_symbol_tf[(symbol, cfg.decision_timeframe)]
        candidate_times = decision_df.index[-args.max_snapshots:]

        for ts in candidate_times:
            enough_history = all(
                len(bars_by_symbol_tf[(symbol, tf)][
                    bars_by_symbol_tf[(symbol, tf)].index <= closed_cutoff(ts, decision_period, tf)])
                >= cfg.bars_per_timeframe
                for tf in cfg.context_timeframes
            )
            if not enough_history:
                skipped_warmup += 1
                continue

            replay_client.set_time(symbol, ts)
            snapshot = context.build_snapshot(
                replay_client, symbol, cfg, daily_realised=0.0,
                consecutive_losses=0, recent_trades=[],
            )
            h = snapshot_hash(snapshot)
            hit = h in cache

            if not hit:
                user_content = build_user_content(cfg.strategy_brief, snapshot)
                input_tokens_est = estimate_input_tokens(user_content, chars_per_token)
                # Representative stand-in, not the stub's own trivial length --
                # see observed_token_averages() docstring for why.
                output_tokens_est = real_avg_output if real_avg_output else 750.0
                stub_confidence = features.hold_confidence(snapshot["timeframes"])
                cache[h] = {
                    "symbol": symbol, "ts": ts.isoformat(),
                    "decision": {
                        "action": "hold", "confidence": round(stub_confidence, 3),
                        "reasoning": "[DRY STUB] no real model call made",
                    },
                    "input_tokens_est": input_tokens_est,
                    "output_tokens_est": output_tokens_est,
                    "cached_at": datetime.now(timezone.utc).isoformat(),
                }

            evaluated.append({"symbol": symbol, "ts": ts, "hash": h, "cache_hit": hit})

    save_cache(args.cache_file, cache)
    new_entries = len(cache) - cache_size_before

    print("\n" + "=" * 70)
    print("DRY-RUN PLUMBING REPORT")
    print("=" * 70)
    print(f"Snapshots evaluated: {len(evaluated)}  (skipped for insufficient warmup: {skipped_warmup})")
    print(f"Cache: {sum(1 for e in evaluated if e['cache_hit'])} hit(s), "
          f"{sum(1 for e in evaluated if not e['cache_hit'])} miss(es), "
          f"{new_entries} new entries written to {args.cache_file} "
          f"(had {cache_size_before}, now {len(cache)})")

    misses = [e for e in evaluated if not e["cache_hit"]]
    total_input = sum(cache[e["hash"]]["input_tokens_est"] for e in misses)
    total_output = sum(cache[e["hash"]]["output_tokens_est"] for e in misses)
    n_missed = len(misses)

    print("\n" + "=" * 70)
    print("COST ESTIMATE (using this run's cache-miss snapshots as the sample)")
    print("=" * 70)
    if n_missed == 0:
        print("Every snapshot this run was already cached -- nothing new to estimate from. "
              "Delete/rename the cache file, or use --offset-days for a fresh window, "
              "to get a sample.")
    else:
        avg_input = total_input / n_missed
        avg_output = total_output / n_missed
        print(f"Sampled from {n_missed} cache-miss snapshot(s).")
        print(f"Estimated avg input tokens/snapshot: {avg_input:.0f} (using chars/token "
              f"calibrated against real logs: {chars_per_token:.2f}; that real logged "
              f"average is {real_avg_input:.0f} for comparison)")
        print(f"Assumed avg output tokens/snapshot: {avg_output:.0f} "
              f"(= real logged average from {args.logs_dir}/, since the stub's own "
              f"response is a trivial placeholder, not representative)")
        cost_per_snapshot = (avg_input / 1_000_000 * args.input_price
                              + avg_output / 1_000_000 * args.output_price)
        print(f"Cost per snapshot at ${args.input_price:.2f}/${args.output_price:.2f} "
              f"per MTok (in/out): ${cost_per_snapshot:.4f}")
        for n in (100, 300, 500, 1000):
            print(f"  Extrapolated to {n:>4} snapshots: ${cost_per_snapshot * n:.2f}")

    return evaluated


# ------------------------------------------------------------------ live P&L ledger ---

@dataclass
class OpenTrade:
    symbol: str
    direction: str
    entry: float
    stop_loss: float
    take_profit: float
    volume: float
    opened_at: pd.Timestamp


@dataclass
class PendingResolution:
    trade: OpenTrade
    outcome: str                      # "win" or "loss" ("open" never gets here -- see run_live)
    exit_price_raw: float
    exit_time: pd.Timestamp
    reason: str


@dataclass
class SimEquityState:
    """
    The book run_live() actually trades against. Deliberately separate from
    ReplayClient: ReplayClient only ever needs to know the CURRENT equity
    (via set_equity()) to build a snapshot -- see its docstring. Cross-symbol
    open-position state is passed to RiskEngine.evaluate_open() directly from
    here, not through ReplayClient.
    """
    equity: float
    open_positions: dict = field(default_factory=dict)   # symbol -> OpenTrade
    pending: list = field(default_factory=list)           # unsettled PendingResolution
    daily_pnl: dict = field(default_factory=dict)         # trading-day date -> realised pnl
    consecutive_losses: int = 0
    equity_curve: list = field(default_factory=list)      # (settle_ts, equity) after each close
    recent_trades: list = field(default_factory=list)     # journal.summary()-shaped, most recent last


def slippage_adjusted_exit(direction: str, outcome: str, exit_price_raw: float,
                            spec: SymbolSpec, slippage_spread_multiple: float) -> float:
    """
    Agreed default: slippage applies to stop-loss fills only, scaled to the
    symbol's own spread. A stop is a forced market exit into adverse
    movement and realistically fills worse than the exact level; a
    take-profit is a resting exit and is not slipped.
    """
    if outcome != "loss":
        return exit_price_raw
    slip = spec.spread_points * spec.point * slippage_spread_multiple
    return exit_price_raw - slip if direction == "buy" else exit_price_raw + slip


def realised_pnl(direction: str, entry: float, exit_price: float, volume: float,
                  spec: SymbolSpec) -> float:
    """
    Same tick-value math as risk.calculate_volume's loss_per_lot (CLAUDE.md
    invariant #7: sizing uses tick value, never pips), applied symmetrically
    to wins. SymbolSpec only captures tick_value_loss (mt5_client.spec()'s
    own fallback, not a separate profit-side tick value) -- a real broker's
    win/loss tick-value asymmetry, where it exists, isn't modelled here.
    """
    signed_distance = (exit_price - entry) if direction == "buy" else (entry - exit_price)
    ticks = signed_distance / spec.tick_size
    return round(ticks * spec.tick_value_loss * volume, 2)


def settle_up_to(state: SimEquityState, ts: pd.Timestamp, cfg: Config,
                  specs: dict, args: argparse.Namespace, results_fh) -> None:
    """
    Apply every pending resolution whose exit_time <= ts to the shared
    ledger. Must run before building any snapshot at `ts`: this is what
    stops a trade's outcome from leaking into another symbol's snapshot
    before it has chronologically happened yet -- BUILD_PLAN.md's
    no-lookahead requirement, extended from bars to portfolio state. Trade
    outcomes are computed eagerly at open time (resolve_outcome() looks
    ahead through history, same as demo_resolve_outcome() already does) but
    are only APPLIED to equity/daily-P&L/streak here, gated on exit_time.
    """
    ready = [p for p in state.pending if p.exit_time <= ts]
    state.pending = [p for p in state.pending if p.exit_time > ts]

    for p in sorted(ready, key=lambda p: p.exit_time):
        trade = p.trade
        spec = specs[trade.symbol]
        exit_price = slippage_adjusted_exit(
            trade.direction, p.outcome, p.exit_price_raw, spec, args.slippage_spread_multiple)
        pnl = realised_pnl(trade.direction, trade.entry, exit_price, trade.volume, spec)

        state.equity += pnl
        state.equity_curve.append((p.exit_time, state.equity))
        state.consecutive_losses = state.consecutive_losses + 1 if pnl < 0 else 0
        day = SessionState(cfg, client=None).day_start(at=p.exit_time).date()
        state.daily_pnl[day] = state.daily_pnl.get(day, 0.0) + pnl
        state.recent_trades.append({
            "ts": p.exit_time.isoformat(), "symbol": trade.symbol, "action": "open",
            "direction": trade.direction, "setup": None, "confidence": None, "approved": True,
        })
        state.recent_trades = state.recent_trades[-5:]
        del state.open_positions[trade.symbol]

        record = {
            "settled_at": p.exit_time.isoformat(), "symbol": trade.symbol,
            "direction": trade.direction, "entry": trade.entry, "stop_loss": trade.stop_loss,
            "take_profit": trade.take_profit, "volume": trade.volume,
            "opened_at": trade.opened_at.isoformat(), "outcome": p.outcome,
            "exit_price": exit_price, "reason": p.reason, "realised_pnl": pnl,
            "equity_after": round(state.equity, 2),
        }
        results_fh.write(json.dumps(record, default=str) + "\n")
        results_fh.flush()


def max_drawdown(curve: list[tuple]) -> float:
    peak = -math.inf
    dd = 0.0
    for _, eq in curve:
        peak = max(peak, eq)
        dd = max(dd, peak - eq)
    return dd


def run_live(cfg: Config, args: argparse.Namespace) -> dict:
    """
    The real pass: calls the Claude API and spends real money. Runs every
    proposed 'open' through the real RiskEngine and simulates the fill via
    resolve_outcome() -- reused, not reimplemented, same as --dry reuses
    build_snapshot(). No code path here calls order_send: agent.execution is
    the only module in this codebase that does (grep-verified), and it is
    never imported by this file or anything it imports.

    Single position per symbol: while a symbol has an open simulated
    position, later snapshots for that symbol are skipped -- no re-entry, no
    close/modify_stop simulation -- until it resolves. This mirrors what the
    real risk engine would enforce anyway ("already open on symbol", of
    max_open_positions), and keeps a first backtest pass in the scope
    BUILD_PLAN.md Phase 4 actually asks for ("weak evidence", not a full
    portfolio simulator). Every snapshot built for a symbol is therefore
    built while flat on it, so DecisionEngine's own _sanity_check() would
    downgrade any stray close/modify_stop proposal to hold regardless (no
    matching ticket in that symbol's open_positions) -- this scope decision
    is enforced by production code already, not just convention here.

    Two-phase spend gate: first compute the full candidate step count and a
    cost estimate with ZERO API calls, print it, and stop unless --confirm
    is also passed. BUILD_PLAN.md: "report the cost estimate... and ask me
    to confirm" -- enforced in code, not just as a suggestion.
    """
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY not set (.env) -- required for --live.")

    client = MT5Client(cfg)
    client.connect()  # same demo-account / login-match safety checks the live system gets
    specs = load_or_fetch_specs(client, cfg.symbols, args.spec_cache_file)

    print(f"Fetching {args.lookback_days}d history (offset {args.offset_days}d) for "
          f"{cfg.symbols} x {cfg.context_timeframes}...")
    bars_by_symbol_tf = {}
    for symbol in cfg.symbols:
        for tf in cfg.context_timeframes:
            bars_by_symbol_tf[(symbol, tf)] = fetch_closed_bars(
                symbol, tf_const(tf), args.lookback_days, args.offset_days)
    client.shutdown()  # everything from here is pure computation + API calls, no MT5 needed

    decision_period = pd.Timedelta(seconds=TIMEFRAME_SECONDS[cfg.decision_timeframe])
    replay_client = ReplayClient(specs, bars_by_symbol_tf, decision_period)

    # ---- merged chronological step list across all symbols, warmup-filtered ----
    # --candidates-file replaces the sequential per-symbol walk with an explicit
    # (symbol, ts) list -- e.g. from build_candidates.py, which only emits a
    # candidate where replay.py's own structural check found a real
    # BOS/CHoCH_confirmed hit, instead of paying to evaluate every quiet bar.
    steps: list[tuple] = []  # (ts, symbol)
    skipped_warmup = 0
    skipped_not_found = 0

    if args.candidates_file:
        cand_df = pd.read_csv(args.candidates_file, parse_dates=["ts"])
        if cand_df["ts"].dt.tz is None:
            cand_df["ts"] = cand_df["ts"].dt.tz_localize("UTC")
        seen = set()
        for _, row in cand_df.iterrows():
            symbol, ts = row["symbol"], row["ts"]
            key = (symbol, ts)
            if key in seen:
                continue
            seen.add(key)
            if (symbol, cfg.decision_timeframe) not in bars_by_symbol_tf:
                continue  # candidate's symbol isn't in this run's configured cfg.symbols
            decision_df = bars_by_symbol_tf[(symbol, cfg.decision_timeframe)]
            if ts not in decision_df.index:
                # Falls outside this run's fetched --lookback-days/--offset-days window,
                # or lands off the M15 grid -- --visible-offset-minutes in
                # build_candidates.py should always land on a valid M15 boundary, so
                # this is almost always a window-coverage mismatch, not a grid issue.
                skipped_not_found += 1
                continue
            enough_history = all(
                len(bars_by_symbol_tf[(symbol, tf)][
                    bars_by_symbol_tf[(symbol, tf)].index <= closed_cutoff(ts, decision_period, tf)])
                >= cfg.bars_per_timeframe
                for tf in cfg.context_timeframes
            )
            if not enough_history:
                skipped_warmup += 1
                continue
            steps.append((ts, symbol))
    else:
        for symbol in cfg.symbols:
            decision_df = bars_by_symbol_tf[(symbol, cfg.decision_timeframe)]
            candidate_times = decision_df.index[-args.max_snapshots:]
            for ts in candidate_times:
                enough_history = all(
                    len(bars_by_symbol_tf[(symbol, tf)][
                        bars_by_symbol_tf[(symbol, tf)].index <= closed_cutoff(ts, decision_period, tf)])
                    >= cfg.bars_per_timeframe
                    for tf in cfg.context_timeframes
                )
                if not enough_history:
                    skipped_warmup += 1
                    continue
                steps.append((ts, symbol))
    steps.sort(key=lambda s: (s[0], s[1]))

    # ---- cost estimate, zero API calls, BEFORE anything can spend ----
    real_avg_input, real_avg_output = observed_token_averages(args.logs_dir)
    chars_per_token = calibrate_chars_per_token(cfg, args.logs_dir)
    avg_output = real_avg_output if real_avg_output else 750.0

    if steps:
        sample_ts, sample_symbol = steps[len(steps) // 2]
        replay_client.set_time(sample_symbol, sample_ts)
        replay_client.set_equity(args.initial_equity)
        sample_snapshot = context.build_snapshot(
            replay_client, sample_symbol, cfg, daily_realised=0.0,
            consecutive_losses=0, recent_trades=[])
        avg_input = estimate_input_tokens(
            build_user_content(cfg.strategy_brief, sample_snapshot), chars_per_token)
    else:
        avg_input = 0

    cost_per_snapshot = (avg_input / 1_000_000 * args.input_price
                          + avg_output / 1_000_000 * args.output_price)
    upper_bound_cost = cost_per_snapshot * len(steps)

    print("\n" + "=" * 70)
    print("LIVE COST ESTIMATE (upper bound -- before the open-position skip)")
    print("=" * 70)
    if args.candidates_file:
        print(f"Candidates sourced from {args.candidates_file} -- --max-snapshots is ignored "
              f"in this mode, every row in the file is a candidate")
        print(f"Candidate steps after warmup/coverage filtering: {len(steps)}  "
              f"(skipped for insufficient warmup: {skipped_warmup}; skipped, not found in "
              f"this run's fetched window: {skipped_not_found})")
    else:
        print(f"--max-snapshots is a PER-SYMBOL cap: {args.max_snapshots} per symbol x "
              f"{len(cfg.symbols)} symbols {cfg.symbols} = up to "
              f"{args.max_snapshots * len(cfg.symbols)} total candidate points")
        print(f"Candidate steps after warmup filtering: {len(steps)}  "
              f"(skipped for insufficient warmup: {skipped_warmup})")
    print(f"Estimated input tokens/snapshot: {avg_input:.0f} (chars/token {chars_per_token:.2f}, "
          f"sampled from the median candidate snapshot)")
    print(f"Assumed output tokens/snapshot: {avg_output:.0f} (real logged average from "
          f"{args.logs_dir}/, or 750.0 fallback if no logs exist yet)")
    print(f"Cost per snapshot at ${args.input_price:.2f}/${args.output_price:.2f} per MTok "
          f"(in/out): ${cost_per_snapshot:.4f}")
    print(f"UPPER BOUND for this run: ${upper_bound_cost:.2f} "
          f"(actual will be <= this -- fewer real calls happen once positions are open, "
          f"and any snapshot already in {args.live_cache_file} is reused for free)")

    if not args.confirm:
        print("\nNot spending anything. Re-run with --confirm to actually make these API "
              "calls and spend real money.")
        return {"spent": False, "steps": len(steps), "upper_bound_cost": upper_bound_cost}

    print(f"\n--confirm passed. Proceeding with up to {len(steps)} real API call(s).")

    # ---- the real walk ----
    engine = DecisionEngine(cfg, api_key)
    engine.verify_model()
    risk = RiskEngine(cfg)
    session = SessionState(cfg, client=None)

    cache = load_cache(args.live_cache_file)

    state = SimEquityState(equity=args.initial_equity)
    results_fh = open(args.results_file, "w", encoding="utf-8")  # fresh file per run

    real_calls = cache_hits = opened = win = loss = still_open = 0
    spent_input = spent_output = 0
    rejected_counts: Counter = Counter()

    try:
        for ts, symbol in steps:
            settle_up_to(state, ts, cfg, specs, args, results_fh)

            if symbol in state.open_positions:
                continue  # still open on this symbol -- see scope note in run_live's docstring

            replay_client.set_time(symbol, ts)
            replay_client.set_equity(state.equity)
            today = session.day_start(at=ts).date()
            daily_realised = state.daily_pnl.get(today, 0.0)
            snapshot = context.build_snapshot(
                replay_client, symbol, cfg, daily_realised=daily_realised,
                consecutive_losses=state.consecutive_losses, recent_trades=state.recent_trades,
            )
            h = snapshot_hash(snapshot)
            was_cached = h in cache

            if was_cached:
                cache_hits += 1
                decision = Decision(**cache[h]["decision"])
            else:
                decision = engine.decide(snapshot)  # <-- real API spend
                real_calls += 1
                spent_input += decision.input_tokens
                spent_output += decision.output_tokens
                cache[h] = {"symbol": symbol, "ts": ts.isoformat(), "decision": asdict(decision),
                            "cached_at": datetime.now(timezone.utc).isoformat()}
                save_cache(args.live_cache_file, cache)  # persist as we go -- a crash must not re-spend

            verdict = None
            if decision.action == "open":
                spec = specs[symbol]
                tick = replay_client.tick(symbol)
                entry = tick.ask if decision.direction == "buy" else tick.bid
                session_ok, _ = session.session_open(at=ts)
                verdict = risk.evaluate_open(
                    direction=decision.direction, entry=entry,
                    stop_loss=decision.stop_loss, take_profit=decision.take_profit,
                    risk_fraction=decision.risk_fraction, spec=spec,
                    account=replay_client.account(),
                    open_positions=[
                        {"symbol": s, "open_price": t.entry, "stop_loss": t.stop_loss, "volume": t.volume}
                        for s, t in state.open_positions.items()
                    ],
                    spread_points=float(spec.spread_points),
                    daily_realised=daily_realised, session_ok=session_ok,
                    consecutive_losses=state.consecutive_losses,
                )
                if verdict.approved:
                    opened += 1
                    decision_df = bars_by_symbol_tf[(symbol, cfg.decision_timeframe)]
                    subsequent = decision_df[decision_df.index > ts]
                    outcome = resolve_outcome(decision.direction, entry, decision.stop_loss,
                                               decision.take_profit, subsequent)
                    trade = OpenTrade(symbol, decision.direction, entry, decision.stop_loss,
                                       decision.take_profit, verdict.volume, ts)
                    state.open_positions[symbol] = trade
                    if outcome["outcome"] == "open":
                        still_open += 1
                        results_fh.write(json.dumps({
                            "settled_at": None, "symbol": symbol, "direction": decision.direction,
                            "entry": entry, "stop_loss": decision.stop_loss,
                            "take_profit": decision.take_profit, "volume": verdict.volume,
                            "opened_at": ts.isoformat(), "outcome": "open", "exit_price": None,
                            "reason": outcome["reason"], "realised_pnl": None, "equity_after": None,
                        }, default=str) + "\n")
                        results_fh.flush()
                    else:
                        win += outcome["outcome"] == "win"
                        loss += outcome["outcome"] == "loss"
                        state.pending.append(PendingResolution(
                            trade, outcome["outcome"], outcome["exit_price"],
                            outcome["exit_time"], outcome["reason"]))
                else:
                    rejected_counts.update(verdict.rejections)

            results_fh.write(json.dumps({
                "decision_log": {
                    "ts": ts.isoformat(), "symbol": symbol, "action": decision.action,
                    "confidence": decision.confidence, "setup_name": decision.setup_name,
                    "reasoning": decision.reasoning,
                    "input_tokens": decision.input_tokens, "output_tokens": decision.output_tokens,
                    "from_cache": was_cached,
                    "verdict": {"approved": verdict.approved, "rejections": verdict.rejections}
                               if verdict else None,
                }
            }, default=str, ensure_ascii=False) + "\n")
            results_fh.flush()

        # Final flush: settle anything left pending, far enough past the last step
        # that every trade opened during the walk (even one still open at the very
        # last step) has had its computed exit_time -- if any -- applied.
        if steps:
            settle_up_to(state, steps[-1][0] + pd.Timedelta(days=3650), cfg, specs, args, results_fh)
    finally:
        results_fh.close()

    total_real_cost = (spent_input / 1_000_000 * args.input_price
                        + spent_output / 1_000_000 * args.output_price)
    resolved = win + loss
    win_rate = win / resolved * 100 if resolved else None
    total_pnl = state.equity - args.initial_equity
    dd = max_drawdown(state.equity_curve)

    print("\n" + "=" * 70)
    print("LIVE RUN REPORT")
    print("=" * 70)
    print(f"Steps walked: {len(steps)}  |  Real API calls: {real_calls}  |  "
          f"Cache hits (from {args.live_cache_file}): {cache_hits}")
    print(f"Actual spend this run: ${total_real_cost:.4f} "
          f"({spent_input} input tok, {spent_output} output tok)")
    print(f"Opened: {opened}  |  Resolved: {resolved} (win {win} / loss {loss})  |  "
          f"Still open at window end: {still_open}")
    print(f"Win rate: {win_rate:.1f}%" if win_rate is not None else "Win rate: n/a (no resolved trades)")
    print(f"Starting equity: ${args.initial_equity:,.2f}  |  Final equity: ${state.equity:,.2f}  |  "
          f"P&L: ${total_pnl:,.2f} ({total_pnl / args.initial_equity * 100:.2f}%)")
    print(f"Max drawdown (equity peak-to-trough): ${dd:,.2f}")
    if rejected_counts:
        print("Top rejection reasons:")
        for reason, count in rejected_counts.most_common(10):
            print(f"  {count:>4}x  {reason}")
    print(f"Full per-step log written to {args.results_file}")

    return {
        "spent": True, "real_calls": real_calls, "cache_hits": cache_hits,
        "total_real_cost": total_real_cost, "opened": opened, "win": win, "loss": loss,
        "still_open": still_open, "win_rate": win_rate, "final_equity": state.equity,
        "total_pnl": total_pnl, "max_drawdown": dd, "rejected_counts": dict(rejected_counts),
    }


def demo_resolve_outcome(cfg: Config, symbol: str, lookback_days: int) -> None:
    """
    Hand-verification of resolve_outcome() against real OHLC, per the
    explicit request to confirm win/loss/still-open by eyeballing the chart
    rather than trusting the function on faith.
    """
    client = MT5Client(cfg)
    client.connect()
    tf = cfg.decision_timeframe
    bars = fetch_closed_bars(symbol, tf_const(tf), lookback_days, 0)
    client.shutdown()

    window = bars.tail(40)
    print("\n" + "=" * 70)
    print(f"RESOLVE_OUTCOME HAND-VERIFICATION -- {symbol} {tf}, last 40 bars")
    print("=" * 70)
    print(window[["open", "high", "low", "close"]].to_string())

    entry_idx = 5
    entry_row = window.iloc[entry_idx]
    entry = float(entry_row["close"])
    subsequent = window.iloc[entry_idx + 1:]

    # +/-5 pips -- small enough to actually get touched within this window,
    # unlike a realistic structural stop/target which would mostly resolve
    # "still open" over just 40 bars and prove nothing. Two examples on the
    # SAME data, opposite directions, so both the win and loss branches get
    # a real hand-checkable case.
    for direction, sl_offset, tp_offset in (("buy", -0.0005, 0.0005), ("sell", 0.0005, -0.0005)):
        stop_loss = round(entry + sl_offset, 5)
        take_profit = round(entry + tp_offset, 5)
        result = resolve_outcome(direction, entry, stop_loss, take_profit, subsequent)
        print(f"\nSynthetic decision: {direction} {symbol} @ {entry:.5f}  "
              f"SL={stop_loss:.5f}  TP={take_profit:.5f}  (entry bar: {entry_row.name})")
        print(f"resolve_outcome() result: {result}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry", action="store_true",
                       help="Stub the model call, make zero API calls.")
    mode.add_argument("--live", action="store_true",
                       help="Call the real model and spend real money. Always prints a cost "
                            "estimate first and stops there unless --confirm is also passed.")
    parser.add_argument("--confirm", action="store_true",
                         help="Required alongside --live to actually spend money, after "
                              "reviewing the printed cost estimate. Ignored for --dry.")
    parser.add_argument("--lookback-days", type=int, default=90)
    parser.add_argument("--offset-days", type=int, default=0)
    parser.add_argument("--max-snapshots", type=int, default=50,
                         help="Cap on decision-timeframe bars stepped through per symbol, "
                              "independent of how much history was fetched for warmup. "
                              "Ignored when --candidates-file is set.")
    parser.add_argument("--candidates-file", default=None,
                         help="CSV with symbol,ts columns (e.g. from build_candidates.py) -- "
                              "evaluate exactly these (symbol, ts) points instead of walking "
                              "every sequential decision-timeframe bar. --lookback-days/"
                              "--offset-days must still be wide enough for this run's fetch to "
                              "cover every candidate timestamp, or those rows are skipped "
                              "(reported as 'not found').")
    parser.add_argument("--cache-file", default="pnl_replay_cache.json")
    parser.add_argument("--live-cache-file", default="pnl_replay_live_cache.json",
                         help="Real decisions keyed by snapshot hash, separate from --dry's "
                              "cache -- a --live re-run must never re-spend on a snapshot it "
                              "already paid for.")
    parser.add_argument("--results-file", default="pnl_replay_live_results.jsonl",
                         help="Per-step decision/verdict/resolution log for --live, "
                              "overwritten fresh each run (the cache is what makes re-runs "
                              "free, not this file).")
    parser.add_argument("--initial-equity", type=float, default=100_000.0)
    parser.add_argument("--slippage-spread-multiple", type=float, default=1.0,
                         help="Stop-loss fills only, scaled to the symbol's own spread "
                              "(spread_points * point * this). Take-profit fills are never "
                              "slipped -- see slippage_adjusted_exit().")
    parser.add_argument("--spec-cache-file", default="pnl_replay_specs.json",
                         help="Symbol specs are fetched live once and persisted here, then "
                              "reused -- re-fetching live each run would defeat cache-hash "
                              "reproducibility for any symbol whose spread isn't pinned at 0.")
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--input-price", type=float, default=2.00)
    parser.add_argument("--output-price", type=float, default=10.00)
    parser.add_argument("--verify-resolution", action="store_true",
                         help="Also run resolve_outcome() against one hand-picked real "
                              "example and print the raw OHLC for manual verification.")
    args = parser.parse_args()

    cfg = Config.load(args.config)

    if args.dry:
        run_dry(cfg, args)
    else:
        run_live(cfg, args)

    if args.verify_resolution:
        demo_resolve_outcome(cfg, cfg.symbols[0], args.lookback_days)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
