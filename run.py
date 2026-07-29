#!/usr/bin/env python3
"""
Entry point. Runs the decision loop on bar close.

    python run.py --config config.yaml
    python run.py --once            # single cycle, useful for testing
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone

from dotenv import load_dotenv

from agent.config import Config
from agent.context import build_snapshot
from agent.decision import DecisionEngine
from agent.execution import Executor
from agent.journal import Journal, SessionState
from agent.mt5_client import MT5Client, TIMEFRAME_SECONDS
from agent.risk import RiskEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)-18s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler("logs/agent.log", encoding="utf-8")],
)
log = logging.getLogger("run")

_running = True


def _stop(signum, frame):
    global _running
    log.info("Signal %s received — finishing current cycle then stopping.", signum)
    _running = False


def seconds_to_next_close(timeframe: str, offset: int = 5) -> float:
    """Seconds until `offset` seconds after the next bar closes."""
    period = TIMEFRAME_SECONDS[timeframe]
    now = time.time()
    return (period - (now % period)) + offset


def run_cycle(symbol, client, engine, risk, executor, journal, state, cfg) -> None:
    spec = client.spec(symbol)
    daily = state.daily_realised()
    session_ok, session_msg = state.session_open()

    snapshot = build_snapshot(
        client, symbol, cfg,
        daily_realised=daily,
        consecutive_losses=state.consecutive_losses,
        recent_trades=journal.summary(),
    )

    decision = engine.decide(snapshot)
    log.info("[%s] %s (confidence %.2f) — %s",
             symbol, decision.action.upper(), decision.confidence,
             (decision.setup_name or decision.reasoning[:80]))

    verdict = execution = None

    if decision.action == "open":
        tick = client.tick(symbol)
        entry = tick.ask if decision.direction == "buy" else tick.bid
        verdict = risk.evaluate_open(
            direction=decision.direction,
            entry=entry,
            stop_loss=decision.stop_loss,
            take_profit=decision.take_profit,
            risk_fraction=decision.risk_fraction,
            spec=spec,
            account=client.account(),
            open_positions=client.positions(),
            spread_points=client.spread_points(symbol),
            daily_realised=daily,
            session_ok=session_ok,
            consecutive_losses=state.consecutive_losses,
        )
        if verdict.approved:
            log.info("Risk approved: %.2f lots, risk %.2f, R:R %s",
                     verdict.volume, verdict.risk_amount, verdict.rr_ratio)
            execution = executor.open_position(
                symbol, decision.direction, verdict.volume,
                decision.stop_loss, decision.take_profit,
                comment=(decision.setup_name or "claude")[:31],
            )
        else:
            log.warning("Risk REJECTED: %s", "; ".join(verdict.rejections))

    elif decision.action == "close":
        execution = executor.close_position(decision.position_ticket, decision.setup_name or "model")

    elif decision.action == "modify_stop":
        position = next((p for p in client.positions()
                         if p["ticket"] == decision.position_ticket), None)
        if position:
            tick = client.tick(position["symbol"])
            price = tick.bid if position["direction"] == "buy" else tick.ask
            verdict = risk.validate_stop_modification(
                position, decision.new_stop_loss, spec, price)
            if verdict.approved:
                execution = executor.modify_stop(decision.position_ticket, decision.new_stop_loss)
            else:
                log.warning("Stop modification REJECTED: %s", "; ".join(verdict.rejections))

    journal.record_cycle(symbol=symbol, snapshot=snapshot, decision=decision,
                         verdict=verdict, execution=execution)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    load_dotenv()

    cfg = Config.load(args.config)
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        log.error("ANTHROPIC_API_KEY not set")
        return 1

    client = MT5Client(cfg)
    client.connect()

    engine = DecisionEngine(cfg, api_key)
    engine.verify_model()

    risk = RiskEngine(cfg)
    executor = Executor(client, cfg)
    journal = Journal("logs")
    state = SessionState(cfg, client)
    state.refresh_streak()

    for s in cfg.symbols:
        client.ensure_symbol(s)

    log.info("Running | symbols=%s | timeframe=%s | dry_run=%s | magic=%s",
             cfg.symbols, cfg.decision_timeframe, cfg.dry_run, cfg.magic_number)

    try:
        while _running:
            if state.kill_switch_active:
                log.critical("KILL SWITCH present — flattening and exiting.")
                executor.flatten_all("kill_switch")
                break

            daily = state.daily_realised()
            limit = -abs(client.account()["balance"] * cfg.risk.max_daily_loss_pct / 100)
            if daily <= limit:
                log.critical("Daily loss limit breached (%.2f). Flattening, halting for today.", daily)
                executor.flatten_all("daily_loss_limit")
                if args.once:
                    break
                time.sleep(3600)
                continue

            for symbol in cfg.symbols:
                try:
                    run_cycle(symbol, client, engine, risk, executor, journal, state, cfg)
                except Exception:  # noqa: BLE001
                    log.exception("Cycle failed for %s — continuing", symbol)

            if args.once:
                break

            wait = seconds_to_next_close(cfg.decision_timeframe)
            log.info("Next cycle in %.0fs (%s)", wait,
                     datetime.now(timezone.utc).strftime("%H:%M:%S"))
            # Sleep in slices so Ctrl-C and the kill switch stay responsive.
            end = time.time() + wait
            while _running and time.time() < end and not state.kill_switch_active:
                time.sleep(min(5, max(0, end - time.time())))
    finally:
        client.shutdown()
        log.info("Shut down cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
