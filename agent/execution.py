"""
Execution layer. Builds and sends orders. Assumes the risk engine has already
approved everything — this module does not second-guess, it just executes cleanly
and reports honestly.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

log = logging.getLogger(__name__)

# Retcodes worth retrying — price moved, not a logic error.
RETRYABLE = {10004, 10021, 10006}  # REQUOTE, PRICE_OFF, REJECT


@dataclass
class ExecutionResult:
    success: bool
    retcode: Optional[int] = None
    ticket: Optional[int] = None
    fill_price: Optional[float] = None
    volume: Optional[float] = None
    message: str = ""
    slippage_points: Optional[float] = None


class Executor:
    def __init__(self, client, cfg):
        self.client = client
        self.cfg = cfg

    def open_position(self, symbol: str, direction: str, volume: float,
                      stop_loss: float, take_profit: Optional[float],
                      comment: str = "") -> ExecutionResult:
        if self.cfg.dry_run:
            tick = self.client.tick(symbol)
            price = tick.ask if direction == "buy" else tick.bid
            log.info("[DRY RUN] %s %s %.2f lots @ %.5f SL=%.5f TP=%s",
                     direction.upper(), symbol, volume, price, stop_loss, take_profit)
            return ExecutionResult(True, ticket=0, fill_price=price, volume=volume,
                                   message="dry_run")

        spec = self.client.spec(symbol)
        filling = self.client.resolve_filling(spec)

        for attempt in range(3):
            tick = self.client.tick(symbol)
            price = tick.ask if direction == "buy" else tick.bid

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": float(volume),
                "type": mt5.ORDER_TYPE_BUY if direction == "buy" else mt5.ORDER_TYPE_SELL,
                "price": price,
                "sl": round(float(stop_loss), spec.digits),
                "deviation": 20,
                "magic": self.cfg.magic_number,
                "comment": comment[:31],  # MT5 truncates beyond 31 chars
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": filling,
            }
            if take_profit:
                request["tp"] = round(float(take_profit), spec.digits)

            # Margin pre-check: cheaper to fail here than to eat a rejection.
            margin = mt5.order_calc_margin(request["type"], symbol, volume, price)
            if margin is not None and margin > self.client.account()["margin_free"]:
                return ExecutionResult(False, message=f"Insufficient free margin (need {margin:.2f})")

            result = mt5.order_send(request)
            if result is None:
                return ExecutionResult(False, message=f"order_send returned None: {mt5.last_error()}")

            if result.retcode == mt5.TRADE_RETCODE_DONE:
                slippage = abs(result.price - price) / spec.point
                log.info("FILLED %s %s %.2f @ %.5f (ticket %s, slip %.1f pts)",
                         direction.upper(), symbol, result.volume, result.price,
                         result.order, slippage)
                return ExecutionResult(
                    True, retcode=result.retcode, ticket=result.order,
                    fill_price=result.price, volume=result.volume,
                    message=result.comment, slippage_points=round(slippage, 1),
                )

            if result.retcode in RETRYABLE and attempt < 2:
                log.warning("Retryable retcode %s (%s), attempt %s",
                            result.retcode, result.comment, attempt + 1)
                time.sleep(0.5)
                continue

            return ExecutionResult(False, retcode=result.retcode,
                                   message=f"{result.retcode}: {result.comment}")

        return ExecutionResult(False, message="Exhausted retries on requote")

    def close_position(self, ticket: int, reason: str = "") -> ExecutionResult:
        if self.cfg.dry_run:
            log.info("[DRY RUN] close ticket %s (%s)", ticket, reason)
            return ExecutionResult(True, ticket=ticket, message="dry_run")

        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return ExecutionResult(False, message=f"Position {ticket} not found")
        p = positions[0]
        if p.magic != self.cfg.magic_number:
            return ExecutionResult(False, message=f"Position {ticket} not owned by this bot")

        spec = self.client.spec(p.symbol)
        tick = self.client.tick(p.symbol)
        is_buy = p.type == mt5.POSITION_TYPE_BUY

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": ticket,
            "symbol": p.symbol,
            "volume": p.volume,
            "type": mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY,
            "price": tick.bid if is_buy else tick.ask,
            "deviation": 20,
            "magic": self.cfg.magic_number,
            "comment": f"close:{reason}"[:31],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self.client.resolve_filling(spec),
        }
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            log.info("CLOSED ticket %s @ %.5f (%s)", ticket, result.price, reason)
            return ExecutionResult(True, retcode=result.retcode, ticket=ticket,
                                   fill_price=result.price, message=reason)
        return ExecutionResult(False, retcode=getattr(result, "retcode", None),
                               message=getattr(result, "comment", "order_send failed"))

    def modify_stop(self, ticket: int, new_sl: float,
                    new_tp: Optional[float] = None) -> ExecutionResult:
        if self.cfg.dry_run:
            log.info("[DRY RUN] modify ticket %s SL -> %.5f", ticket, new_sl)
            return ExecutionResult(True, ticket=ticket, message="dry_run")

        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return ExecutionResult(False, message=f"Position {ticket} not found")
        p = positions[0]
        spec = self.client.spec(p.symbol)

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": p.symbol,
            "sl": round(float(new_sl), spec.digits),
            "tp": round(float(new_tp), spec.digits) if new_tp else p.tp,
        }
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            log.info("MODIFIED ticket %s SL -> %.5f", ticket, new_sl)
            return ExecutionResult(True, retcode=result.retcode, ticket=ticket)
        return ExecutionResult(False, retcode=getattr(result, "retcode", None),
                               message=getattr(result, "comment", "modify failed"))

    def flatten_all(self, reason: str) -> list[ExecutionResult]:
        """Emergency exit — close everything this bot owns."""
        results = []
        for p in self.client.positions():
            results.append(self.close_position(p["ticket"], reason))
        return results
