"""
Journal and session state.

The journal is the point of the whole exercise. Every cycle writes one JSONL
record containing the full input snapshot, the model's decision, the risk
verdict and the execution result. This is what lets you answer, months later,
whether the system actually has an edge or has just been lucky.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from datetime import datetime, time as dtime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)


def _serialise(obj):
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)


class Journal:
    def __init__(self, directory: str = "logs"):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        return self.dir / f"{name}_{datetime.now(timezone.utc):%Y-%m}.jsonl"

    def record_cycle(self, *, symbol: str, snapshot: dict, decision,
                     verdict=None, execution=None) -> None:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "snapshot": snapshot,
            "decision": _serialise(decision),
            "verdict": _serialise(verdict) if verdict else None,
            "execution": _serialise(execution) if execution else None,
            "cost": {
                "input_tokens": getattr(decision, "input_tokens", 0),
                "output_tokens": getattr(decision, "output_tokens", 0),
                "latency_ms": getattr(decision, "latency_ms", 0),
            },
        }
        with self._path("cycles").open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=_serialise) + "\n")

    def summary(self, entry_limit: int = 5) -> list[dict]:
        """Condensed recent decisions, fed back into the next snapshot."""
        path = self._path("cycles")
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8").strip().splitlines()[-50:]
        out = []
        for line in lines:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            d = e.get("decision", {})
            if d.get("action") == "hold":
                continue
            out.append({
                "ts": e["ts"],
                "symbol": e["symbol"],
                "action": d.get("action"),
                "direction": d.get("direction"),
                "setup": d.get("setup_name"),
                "confidence": d.get("confidence"),
                "approved": (e.get("verdict") or {}).get("approved"),
            })
        return out[-entry_limit:]


class SessionState:
    """Tracks the things a stateless model cannot: streaks, limits, kill switch."""

    def __init__(self, cfg, client):
        self.cfg = cfg
        self.client = client
        self.consecutive_losses = 0
        self._day = None
        self._day_start_balance = None

    @property
    def kill_switch_active(self) -> bool:
        return Path(self.cfg.kill_switch_file).exists()

    def day_start(self) -> datetime:
        tz = ZoneInfo(self.cfg.session.timezone)
        now = datetime.now(tz)
        return datetime.combine(now.date(), dtime.min, tzinfo=tz).astimezone(timezone.utc)

    def daily_realised(self) -> float:
        return self.client.realised_pnl_since(self.day_start())

    def session_open(self) -> tuple[bool, str]:
        tz = ZoneInfo(self.cfg.session.timezone)
        now = datetime.now(tz)

        if now.weekday() not in self.cfg.session.trading_days:
            return False, f"{now:%A} is not a trading day"

        # Avoid the spread blowout around daily rollover.
        buffer = self.cfg.session.avoid_minutes_around_rollover
        midnight_gap = min(
            abs((now - now.replace(hour=0, minute=0, second=0)).total_seconds()),
            abs((now.replace(hour=0, minute=0, second=0) + timedelta(days=1) - now).total_seconds()),
        )
        if midnight_gap < buffer * 60:
            return False, "Within rollover buffer"

        for start_s, end_s in self.cfg.session.trading_windows:
            start = dtime.fromisoformat(start_s)
            end = dtime.fromisoformat(end_s)
            if start <= now.time() <= end:
                return True, "open"
        return False, f"{now:%H:%M} outside trading windows"

    def register_outcome(self, profit: float) -> None:
        if profit < 0:
            self.consecutive_losses += 1
        elif profit > 0:
            self.consecutive_losses = 0

    def refresh_streak(self) -> None:
        """Recompute the loss streak from broker history, so restarts don't reset it."""
        try:
            import MetaTrader5 as mt5
            deals = mt5.history_deals_get(
                datetime.now(timezone.utc) - timedelta(days=7),
                datetime.now(timezone.utc) + timedelta(minutes=5),
            )
            if not deals:
                return
            closes = [d for d in deals
                      if d.magic == self.cfg.magic_number and d.entry == mt5.DEAL_ENTRY_OUT]
            streak = 0
            for d in reversed(closes):
                if (d.profit + d.commission + d.swap) < 0:
                    streak += 1
                else:
                    break
            self.consecutive_losses = streak
            log.info("Loss streak restored from history: %s", streak)
        except Exception as e:  # noqa: BLE001
            log.warning("Could not restore loss streak: %s", e)
