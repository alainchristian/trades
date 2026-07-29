"""
Deterministic risk engine.

This module contains NO AI. It is the gate between what the model proposes and
what actually reaches the broker. Every rejection reason is explicit and logged.

If you change one thing in this project, change it here — not in the prompt.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class RiskVerdict:
    approved: bool
    volume: float = 0.0
    rejections: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    risk_amount: float = 0.0
    rr_ratio: Optional[float] = None

    def reject(self, reason: str) -> "RiskVerdict":
        self.approved = False
        self.rejections.append(reason)
        return self


def round_to_step(volume: float, step: float) -> float:
    """
    Round DOWN to the broker's volume step.

    Down, never nearest — rounding up silently exceeds your risk budget.
    The float-noise guard matters: 0.03/0.01 evaluates to 2.9999... in binary
    floating point, which would floor to 0.02 without it.
    """
    if step <= 0:
        return volume
    steps = math.floor(round(volume / step, 9))
    decimals = max(0, -int(math.floor(math.log10(step))))
    return round(steps * step, decimals)


def calculate_volume(
    equity: float,
    risk_pct: float,
    entry: float,
    stop_loss: float,
    tick_size: float,
    tick_value_loss: float,
    volume_min: float,
    volume_max: float,
    volume_step: float,
) -> tuple[float, float]:
    """
    Size a position from risk, using tick value rather than pips.

    Returns (volume, actual_risk_amount).

    Tick-value sizing is currency-agnostic: it works identically for EURUSD,
    USDJPY, XAUUSD and index CFDs, whereas pip-based math silently mis-sizes
    anything that isn't a 5-digit FX major.
    """
    stop_distance = abs(entry - stop_loss)
    if stop_distance <= 0:
        return 0.0, 0.0
    if tick_size <= 0 or tick_value_loss <= 0:
        return 0.0, 0.0

    risk_budget = equity * (risk_pct / 100.0)
    ticks = stop_distance / tick_size
    loss_per_lot = ticks * tick_value_loss
    if loss_per_lot <= 0:
        return 0.0, 0.0

    raw_volume = risk_budget / loss_per_lot
    volume = round_to_step(raw_volume, volume_step)
    volume = min(volume, volume_max)

    if volume < volume_min:
        # Do NOT bump up to volume_min — that would exceed the risk budget.
        return 0.0, 0.0

    return volume, round(volume * loss_per_lot, 2)


class RiskEngine:
    def __init__(self, cfg):
        self.cfg = cfg

    def evaluate_open(
        self,
        *,
        direction: str,
        entry: float,
        stop_loss: float,
        take_profit: Optional[float],
        risk_fraction: float,
        spec,
        account: dict,
        open_positions: list[dict],
        spread_points: float,
        daily_realised: float,
        session_ok: bool,
        consecutive_losses: int,
    ) -> RiskVerdict:
        v = RiskVerdict(approved=True)
        r = self.cfg.risk

        # ---- account-level circuit breakers (checked first, cheapest to fail) ----
        equity = account["equity"]
        daily_loss_limit = -abs(account["balance"] * r.max_daily_loss_pct / 100.0)
        if daily_realised <= daily_loss_limit:
            v.reject(f"Daily loss limit hit ({daily_realised:.2f} <= {daily_loss_limit:.2f})")
        if len(open_positions) >= r.max_open_positions:
            v.reject(f"Max open positions reached ({len(open_positions)}/{r.max_open_positions})")
        if consecutive_losses >= r.cooldown_after_losses:
            v.reject(f"Cooldown active after {consecutive_losses} consecutive losses")
        if not session_ok:
            v.reject("Outside permitted trading session")
        if not account.get("trade_allowed", True):
            v.reject("Terminal reports trading not allowed")

        # ---- symbol/market conditions ----
        if spread_points > r.max_spread_points:
            v.reject(f"Spread too wide ({spread_points:.1f} > {r.max_spread_points} points)")
        if any(p["symbol"] == spec.name for p in open_positions) and not r.allow_multiple_per_symbol:
            v.reject(f"Position already open on {spec.name}")

        # ---- trade geometry ----
        if direction not in ("buy", "sell"):
            v.reject(f"Invalid direction '{direction}'")
            return v
        if stop_loss is None or stop_loss <= 0:
            v.reject("Stop loss missing — every trade must define its invalidation")
            return v

        # Stop must be on the correct side of entry. This catches the single most
        # common model error: inverting SL and TP.
        if direction == "buy" and stop_loss >= entry:
            v.reject(f"Buy stop loss {stop_loss} is not below entry {entry}")
        if direction == "sell" and stop_loss <= entry:
            v.reject(f"Sell stop loss {stop_loss} is not above entry {entry}")

        if take_profit:
            if direction == "buy" and take_profit <= entry:
                v.reject(f"Buy take profit {take_profit} is not above entry {entry}")
            if direction == "sell" and take_profit >= entry:
                v.reject(f"Sell take profit {take_profit} is not below entry {entry}")

        # Broker minimum stop distance
        stop_distance = abs(entry - stop_loss)
        if stop_distance < spec.min_stop_distance:
            v.reject(
                f"Stop {stop_distance / spec.point:.1f} points is inside broker "
                f"minimum of {spec.stops_level_points} points"
            )

        # Stop must clear the spread by a sensible margin
        if stop_distance < spread_points * spec.point * r.min_stop_spread_multiple:
            v.reject(
                f"Stop is only {stop_distance / spec.point:.1f} points vs "
                f"{spread_points:.1f} points of spread"
            )

        # Reward:risk floor
        if take_profit:
            rr = abs(take_profit - entry) / stop_distance
            v.rr_ratio = round(rr, 2)
            if rr < r.min_rr:
                v.reject(f"Reward:risk {rr:.2f} below minimum {r.min_rr}")
        elif r.require_take_profit:
            v.reject("Take profit required by configuration")

        if v.rejections:
            return v

        # ---- sizing ----
        # The model supplies a fraction of the configured max; it can never
        # request more risk than the config allows.
        risk_fraction = max(0.0, min(1.0, risk_fraction))
        effective_risk_pct = r.max_risk_per_trade_pct * risk_fraction
        if effective_risk_pct < r.min_risk_per_trade_pct:
            effective_risk_pct = r.min_risk_per_trade_pct

        volume, risk_amount = calculate_volume(
            equity=equity,
            risk_pct=effective_risk_pct,
            entry=entry,
            stop_loss=stop_loss,
            tick_size=spec.tick_size,
            tick_value_loss=spec.tick_value_loss,
            volume_min=spec.volume_min,
            volume_max=spec.volume_max,
            volume_step=spec.volume_step,
        )

        if volume <= 0:
            return v.reject(
                f"Computed volume below broker minimum {spec.volume_min} at "
                f"{effective_risk_pct:.2f}% risk — stop is too wide for this account size"
            )

        # Re-verify the realised risk after step rounding.
        max_allowed = equity * (r.max_risk_per_trade_pct / 100.0)
        if risk_amount > max_allowed * 1.001:
            return v.reject(
                f"Post-rounding risk {risk_amount:.2f} exceeds cap {max_allowed:.2f}"
            )

        # Total open risk across the book
        existing_risk = sum(self._position_risk(p, spec) for p in open_positions)
        total_risk_pct = (existing_risk + risk_amount) / equity * 100
        if total_risk_pct > r.max_total_risk_pct:
            return v.reject(
                f"Total portfolio risk {total_risk_pct:.2f}% exceeds "
                f"{r.max_total_risk_pct}% cap"
            )

        if account["margin_free"] < equity * 0.2:
            v.warnings.append("Free margin below 20% of equity")

        v.volume = volume
        v.risk_amount = risk_amount
        return v

    @staticmethod
    def _position_risk(position: dict, spec) -> float:
        """Approximate open risk of an existing position (0 if it has no stop)."""
        if not position.get("stop_loss"):
            return 0.0
        distance = abs(position["open_price"] - position["stop_loss"])
        ticks = distance / spec.tick_size if spec.tick_size else 0
        return ticks * spec.tick_value_loss * position["volume"]

    def validate_stop_modification(
        self, position: dict, new_sl: float, spec, current_price: float
    ) -> RiskVerdict:
        """
        Stop modifications may only reduce risk. A model that can widen stops
        will eventually turn a small loss into an account-ending one.
        """
        v = RiskVerdict(approved=True)
        old_sl = position.get("stop_loss")
        is_buy = position["direction"] == "buy"

        if old_sl:
            if is_buy and new_sl < old_sl:
                v.reject(f"Cannot widen buy stop from {old_sl} to {new_sl}")
            if not is_buy and new_sl > old_sl:
                v.reject(f"Cannot widen sell stop from {old_sl} to {new_sl}")

        if is_buy and new_sl >= current_price:
            v.reject("New buy stop is at or above current price")
        if not is_buy and new_sl <= current_price:
            v.reject("New sell stop is at or below current price")

        if abs(current_price - new_sl) < spec.min_stop_distance:
            v.reject(f"New stop inside broker minimum distance ({spec.stops_level_points} points)")

        return v
