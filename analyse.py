#!/usr/bin/env python3
"""
Journal analytics (BUILD_PLAN.md Phase 3).

Reads logs/cycles_*.jsonl via agent.analytics -- the same parser dashboard.py
uses, so the numbers here never diverge from what the dashboard shows -- and
reports decision counts, hold rate, risk-rejection breakdown, and API cost/
latency purely offline from the logs.

For executed trades it goes further: win rate, average R multiple,
expectancy, max drawdown, profit factor, confidence calibration (realised
win rate per confidence bucket -- the most important number in this file,
per the build plan), and performance by setup_name and hour of day. That
needs outcomes attached to each 'open' decision, which means matching its
execution ticket against MT5's closed-deal history (agent.analytics
deliberately has no MT5 import, so this half lives here instead) -- the one
part of this script that needs a live terminal connection. If MT5 isn't
reachable, everything else still runs; only the outcome-dependent sections
are skipped, with a clear message saying why.

Under 30 trades, every outcome-dependent section prints a loud warning
instead of numbers that would be noise dressed as signal.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

import pandas as pd

from agent import analytics
from agent.config import Config

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

MIN_TRADES_FOR_STATS = 30


# --------------------------------------------------------- trade matching ---

def match_trade_outcomes(cfg, opens: pd.DataFrame) -> pd.DataFrame:
    """
    For each 'open' decision with a real execution ticket (dry_run fills use
    ticket 0 as a placeholder -- never matched), find the entry deal by order
    ticket, then sum every closing deal sharing that position -- so partial
    closes are handled, not just full ones -- for realised P/L and close time.

    Adds: closed (bool), close_ts, realised_pnl, r_multiple. Unmatched rows
    (still open / dry-run / no history found) get closed=False and NaN.
    """
    opens = opens.copy()
    opens["closed"] = False
    # Plain `pd.NaT` creates a tz-NAIVE datetime64 column; assigning the
    # tz-aware close timestamps below would then raise. Must be tz-aware
    # from the start.
    opens["close_ts"] = pd.Series(pd.NaT, index=opens.index, dtype="datetime64[ns, UTC]")
    opens["realised_pnl"] = float("nan")
    opens["r_multiple"] = float("nan")

    real_tickets = opens.loc[opens["execution_ticket"].fillna(0) > 0, "execution_ticket"]
    if opens.empty or real_tickets.empty:
        return opens

    earliest = opens["ts"].min() - timedelta(days=1)
    deals = mt5.history_deals_get(earliest, datetime.now(timezone.utc) + timedelta(minutes=5))
    if not deals:
        return opens

    deals = [d for d in deals if d.magic == cfg.magic_number]
    entry_position_by_order = {d.order: d.position_id for d in deals if d.entry == mt5.DEAL_ENTRY_IN}

    close_pnl_by_position: dict[int, float] = {}
    close_ts_by_position: dict[int, datetime] = {}
    for d in deals:
        if d.entry not in (mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_OUT_BY):
            continue
        pid = d.position_id
        close_pnl_by_position[pid] = close_pnl_by_position.get(pid, 0.0) + d.profit + d.commission + d.swap
        ts = datetime.fromtimestamp(d.time, tz=timezone.utc)
        if pid not in close_ts_by_position or ts > close_ts_by_position[pid]:
            close_ts_by_position[pid] = ts

    for idx, row in opens.iterrows():
        ticket = row["execution_ticket"]
        if not ticket or ticket <= 0:
            continue
        position_id = entry_position_by_order.get(int(ticket))
        if position_id is None or position_id not in close_pnl_by_position:
            continue  # still open, or history doesn't cover it yet
        pnl = close_pnl_by_position[position_id]
        opens.at[idx, "closed"] = True
        opens.at[idx, "close_ts"] = close_ts_by_position[position_id]
        opens.at[idx, "realised_pnl"] = pnl
        if row.get("risk_amount"):
            opens.at[idx, "r_multiple"] = pnl / row["risk_amount"]

    return opens


# ------------------------------------------------------------- reporting ---

def trade_stats(trades: pd.DataFrame) -> dict:
    closed = trades[trades["closed"]]
    n = len(closed)
    if n == 0:
        return {"n": 0}
    wins = closed[closed["realised_pnl"] > 0]
    losses = closed[closed["realised_pnl"] < 0]
    gross_win = wins["realised_pnl"].sum()
    gross_loss = abs(losses["realised_pnl"].sum())
    if gross_loss > 0:
        profit_factor = gross_win / gross_loss
    else:
        profit_factor = float("inf") if gross_win > 0 else float("nan")

    equity = closed.sort_values("close_ts")["realised_pnl"].cumsum()
    running_max = equity.cummax()
    max_dd = (running_max - equity).max() if not equity.empty else 0.0

    return {
        "n": n,
        "win_rate": len(wins) / n * 100,
        "avg_r": closed["r_multiple"].mean(),
        "expectancy_usd": closed["realised_pnl"].mean(),
        "profit_factor": profit_factor,
        "max_drawdown_usd": max_dd,
    }


def confidence_calibration(trades: pd.DataFrame) -> pd.DataFrame:
    """Realised win rate per confidence bucket -- the point of this whole file."""
    closed = trades[trades["closed"]].copy()
    if closed.empty:
        return pd.DataFrame(columns=["bucket", "n", "win_rate"])
    edges = [i / 10 for i in range(11)]
    labels = [f"{edges[i]:.1f}-{edges[i + 1]:.1f}" for i in range(10)]
    closed["bucket"] = pd.cut(closed["confidence"], bins=edges, labels=labels, include_lowest=True)
    closed["win"] = closed["realised_pnl"] > 0
    grouped = closed.groupby("bucket", observed=False).agg(n=("win", "size"), win_rate=("win", "mean"))
    grouped["win_rate"] = grouped["win_rate"] * 100
    return grouped.reset_index()


def breakdown_by(trades: pd.DataFrame, column: str) -> pd.DataFrame:
    closed = trades[trades["closed"]].copy()
    if closed.empty:
        return pd.DataFrame(columns=[column, "n", "win_rate", "avg_r", "total_pnl"])
    closed["win"] = closed["realised_pnl"] > 0
    grouped = closed.groupby(column, observed=False).agg(
        n=("win", "size"), win_rate=("win", "mean"),
        avg_r=("r_multiple", "mean"), total_pnl=("realised_pnl", "sum"),
    )
    grouped["win_rate"] *= 100
    return grouped.reset_index().sort_values("n", ascending=False)


def print_table(df: pd.DataFrame, float_cols: tuple[str, ...] = ()) -> None:
    if df.empty:
        print("  (none)")
        return
    fmt = {c: "{:.2f}".format for c in float_cols if c in df.columns}
    print(df.to_string(index=False, formatters=fmt))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--input-price", type=float, default=2.00,
                         help="$/MTok input, Sonnet 5 intro rate through 2026-08-31")
    parser.add_argument("--output-price", type=float, default=10.00,
                         help="$/MTok output, Sonnet 5 intro rate through 2026-08-31")
    parser.add_argument("--tz", default=None,
                         help="Timezone for hour-of-day breakdown; defaults to config.yaml's session.timezone")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    tz = args.tz or cfg.session.timezone

    df = analytics.load_cycles(args.logs_dir)
    if df.empty:
        print(f"No cycle logs found under {args.logs_dir}/. Nothing to analyse yet.")
        return 0
    df = analytics.with_cost(df, args.input_price, args.output_price, tz=tz)

    print("=" * 70)
    print("DECISIONS")
    print("=" * 70)
    k = analytics.kpis(df)
    print(f"Total cycles: {k['total_cycles']}")
    print(f"Hold rate: {k['hold_rate']:.1f}%")
    print(f"Decision counts by action:")
    print_table(df["action"].value_counts().rename_axis("action").reset_index(name="count"))
    print(f"Trade proposals (open/close/modify_stop): {k['trade_proposals']} "
          f"({k['approved_count']} approved, {k['rejected_count']} rejected)")

    print("\n" + "=" * 70)
    print("RISK REJECTIONS")
    print("=" * 70)
    rr = analytics.rejection_reasons(df)
    print_table(rr)

    print("\n" + "=" * 70)
    print("COST & LATENCY")
    print("=" * 70)
    print(f"Total API cost: ${k['total_cost_usd']:.2f}")
    print(f"Average latency per cycle: {k['avg_latency_ms']:.0f} ms")
    print(f"Average confidence (all decisions): {k['avg_confidence']:.2f}")

    # ---- everything below here needs realised trade outcomes ----
    opens = df[df["action"] == "open"].copy()
    print(f"\n'open' decisions logged so far: {len(opens)}")

    if mt5 is None:
        print("\nMetaTrader5 package unavailable (non-Windows or not installed) -- "
              "skipping trade-outcome sections (win rate, R multiple, expectancy, "
              "drawdown, profit factor, confidence calibration, setup/hour breakdown).")
        return 0

    if not mt5.initialize():
        print(f"\nmt5.initialize() failed ({mt5.last_error()}) -- is the terminal open and "
              "logged in? Skipping trade-outcome sections.")
        return 0

    try:
        trades = match_trade_outcomes(cfg, opens)
    finally:
        mt5.shutdown()

    n_closed = int(trades["closed"].sum()) if not trades.empty else 0
    print(f"Closed trades matched via MT5 history: {n_closed}")

    if n_closed < MIN_TRADES_FOR_STATS:
        print(f"\n{'!' * 70}")
        print(f"! SAMPLE TOO SMALL: only {n_closed} closed trade(s) -- fewer than "
              f"{MIN_TRADES_FOR_STATS}.")
        print("! Every number below this line, if shown, is noise dressed as signal.")
        print("! Do not tune the strategy or the prompt off of it.")
        print(f"{'!' * 70}")
        if n_closed == 0:
            return 0

    print("\n" + "=" * 70)
    print("TRADE PERFORMANCE")
    print("=" * 70)
    stats = trade_stats(trades)
    print(f"Trades: {stats['n']}")
    print(f"Win rate: {stats['win_rate']:.1f}%")
    print(f"Average R multiple: {stats['avg_r']:.2f}")
    print(f"Expectancy: ${stats['expectancy_usd']:.2f} per trade")
    pf = stats["profit_factor"]
    print(f"Profit factor: {'inf' if pf == float('inf') else f'{pf:.2f}'}")
    print(f"Max drawdown: ${stats['max_drawdown_usd']:.2f}")

    print("\n" + "=" * 70)
    print("CONFIDENCE CALIBRATION (realised win rate per bucket -- most important table)")
    print("=" * 70)
    print_table(confidence_calibration(trades), float_cols=("win_rate",))

    print("\n" + "=" * 70)
    print("PERFORMANCE BY SETUP")
    print("=" * 70)
    print_table(breakdown_by(trades, "setup_name"), float_cols=("win_rate", "avg_r", "total_pnl"))

    print("\n" + "=" * 70)
    print("PERFORMANCE BY HOUR OF DAY")
    print("=" * 70)
    print_table(breakdown_by(trades, "hour"), float_cols=("win_rate", "avg_r", "total_pnl"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
