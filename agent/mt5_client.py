"""
MT5 connection and data access.

Everything that touches the terminal lives here. No trading logic.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:  # allows importing the module on non-Windows for tests
    mt5 = None

log = logging.getLogger(__name__)


TIMEFRAME_MAP = {
    "M1": "TIMEFRAME_M1", "M5": "TIMEFRAME_M5", "M15": "TIMEFRAME_M15",
    "M30": "TIMEFRAME_M30", "H1": "TIMEFRAME_H1", "H4": "TIMEFRAME_H4",
    "D1": "TIMEFRAME_D1", "W1": "TIMEFRAME_W1",
}

TIMEFRAME_SECONDS = {
    "M1": 60, "M5": 300, "M15": 900, "M30": 1800,
    "H1": 3600, "H4": 14400, "D1": 86400, "W1": 604800,
}


def tf_const(name: str) -> int:
    """Map 'M15' -> mt5.TIMEFRAME_M15."""
    if name not in TIMEFRAME_MAP:
        raise ValueError(f"Unsupported timeframe: {name}")
    return getattr(mt5, TIMEFRAME_MAP[name])


@dataclass
class SymbolSpec:
    """Contract specification. Everything needed to size and validate an order."""
    name: str
    digits: int
    point: float
    tick_size: float
    tick_value_loss: float      # account-currency loss per tick per 1.0 lot
    contract_size: float
    volume_min: float
    volume_max: float
    volume_step: float
    stops_level_points: int     # min SL/TP distance from price, in points
    freeze_level_points: int
    spread_points: int
    filling_mode_mask: int
    trade_mode: int

    @property
    def min_stop_distance(self) -> float:
        return self.stops_level_points * self.point


class MT5Client:
    def __init__(self, cfg):
        self.cfg = cfg
        self._connected = False

    # ---------- lifecycle ----------

    def connect(self) -> None:
        if mt5 is None:
            raise RuntimeError(
                "MetaTrader5 package unavailable. This module requires Windows "
                "with the MT5 terminal installed."
            )
        base_kwargs = {}
        if self.cfg.mt5.terminal_path:
            base_kwargs["path"] = self.cfg.mt5.terminal_path

        # Prefer attaching to the terminal's already-logged-in session (fast,
        # matches the documented "terminal must be open and logged in" setup).
        # Passing login/password/server forces a fresh network login handshake
        # even when already logged into that same account, which can hang or
        # IPC-timeout on some servers — so it's only a fallback, not the default.
        if not mt5.initialize(**base_kwargs) and self.cfg.mt5.login:
            login_kwargs = dict(
                base_kwargs,
                login=int(self.cfg.mt5.login),
                password=self.cfg.mt5.password,
                server=self.cfg.mt5.server,
            )
            if not mt5.initialize(**login_kwargs):
                raise RuntimeError(f"mt5.initialize failed: {mt5.last_error()}")
        elif mt5.last_error()[0] != 1:
            raise RuntimeError(f"mt5.initialize failed: {mt5.last_error()}")

        acct = mt5.account_info()
        if acct is None:
            raise RuntimeError(f"account_info failed: {mt5.last_error()}")

        # Never silently operate on an account other than the one configured.
        if self.cfg.mt5.login and acct.login != int(self.cfg.mt5.login):
            mt5.shutdown()
            raise RuntimeError(
                f"Connected account {acct.login} does not match configured "
                f"MT5_LOGIN {self.cfg.mt5.login}. Refusing to start."
            )

        # Hard safety gate: refuse to run on a real account unless explicitly allowed.
        is_demo = acct.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO
        if not is_demo and not self.cfg.allow_live_account:
            mt5.shutdown()
            raise RuntimeError(
                f"Account {acct.login} is NOT a demo account and "
                "allow_live_account is false. Refusing to start."
            )
        if not acct.trade_allowed:
            log.warning("Terminal reports trade_allowed=False (check AutoTrading button).")

        self._connected = True
        log.info(
            "Connected: login=%s server=%s balance=%.2f %s demo=%s",
            acct.login, acct.server, acct.balance, acct.currency, is_demo,
        )

    def shutdown(self) -> None:
        if self._connected:
            mt5.shutdown()
            self._connected = False

    def ensure_symbol(self, symbol: str) -> None:
        info = mt5.symbol_info(symbol)
        if info is None:
            raise RuntimeError(f"Unknown symbol '{symbol}' — check broker naming (e.g. EURUSD.a)")
        if not info.visible and not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"symbol_select failed for {symbol}: {mt5.last_error()}")

    # ---------- market data ----------

    def spec(self, symbol: str) -> SymbolSpec:
        i = mt5.symbol_info(symbol)
        if i is None:
            raise RuntimeError(f"symbol_info failed for {symbol}")
        # tick_value_loss is the correct field for stop-loss sizing; some brokers
        # leave it at 0, in which case fall back to the generic tick value.
        tvl = i.trade_tick_value_loss or i.trade_tick_value
        return SymbolSpec(
            name=i.name,
            digits=i.digits,
            point=i.point,
            tick_size=i.trade_tick_size or i.point,
            tick_value_loss=tvl,
            contract_size=i.trade_contract_size,
            volume_min=i.volume_min,
            volume_max=i.volume_max,
            volume_step=i.volume_step,
            stops_level_points=i.trade_stops_level,
            freeze_level_points=i.trade_freeze_level,
            spread_points=i.spread,
            filling_mode_mask=i.filling_mode,
            trade_mode=i.trade_mode,
        )

    def bars(self, symbol: str, timeframe: str, count: int,
             drop_forming: bool = True) -> pd.DataFrame:
        """
        Return closed OHLCV bars, oldest-first.

        MT5 returns index 0 as the currently-forming bar. We drop it by default so
        indicator values never repaint between calls within the same bar.
        """
        n = count + (1 if drop_forming else 0)
        rates = mt5.copy_rates_from_pos(symbol, tf_const(timeframe), 0, n)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"copy_rates_from_pos({symbol},{timeframe}) failed: {mt5.last_error()}")
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("time").sort_index()
        if drop_forming:
            df = df.iloc[:-1]
        return df.tail(count)

    def tick(self, symbol: str):
        t = mt5.symbol_info_tick(symbol)
        if t is None:
            raise RuntimeError(f"symbol_info_tick failed for {symbol}")
        return t

    def spread_points(self, symbol: str) -> float:
        s = self.spec(symbol)
        t = self.tick(symbol)
        return (t.ask - t.bid) / s.point

    # ---------- account & positions ----------

    def account(self) -> dict:
        a = mt5.account_info()
        return {
            "login": a.login,
            "currency": a.currency,
            "balance": round(a.balance, 2),
            "equity": round(a.equity, 2),
            "margin": round(a.margin, 2),
            "margin_free": round(a.margin_free, 2),
            "margin_level": round(a.margin_level, 2) if a.margin_level else None,
            "leverage": a.leverage,
            "trade_allowed": a.trade_allowed,
        }

    def positions(self, symbol: Optional[str] = None) -> list[dict]:
        """Only positions opened by this bot (filtered by magic number)."""
        raw = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        if raw is None:
            return []
        out = []
        for p in raw:
            if p.magic != self.cfg.magic_number:
                continue
            out.append({
                "ticket": p.ticket,
                "symbol": p.symbol,
                "direction": "buy" if p.type == mt5.POSITION_TYPE_BUY else "sell",
                "volume": p.volume,
                "open_price": p.price_open,
                "current_price": p.price_current,
                "stop_loss": p.sl or None,
                "take_profit": p.tp or None,
                "profit": round(p.profit, 2),
                "swap": round(p.swap, 2),
                "opened_at": datetime.fromtimestamp(p.time, tz=timezone.utc).isoformat(),
                "comment": p.comment,
            })
        return out

    def realised_pnl_since(self, since: datetime) -> float:
        """Closed P/L (profit + commission + swap) for this bot's deals since `since`."""
        deals = mt5.history_deals_get(since, datetime.now(timezone.utc) + timedelta(minutes=5))
        if deals is None:
            log.warning("history_deals_get returned None: %s", mt5.last_error())
            return 0.0
        return round(
            sum(d.profit + d.commission + d.swap for d in deals if d.magic == self.cfg.magic_number),
            2,
        )

    # ---------- helpers ----------

    def resolve_filling(self, spec: SymbolSpec) -> int:
        """
        Pick a filling mode the broker actually accepts for this symbol.

        symbol_info().filling_mode is a bitmask matching MQL5's SYMBOL_FILLING_MODE
        enum (bit 0 = FOK, bit 1 = IOC) — the Python package does not expose
        SYMBOL_FILLING_FOK/IOC constants for it, only the unrelated ORDER_FILLING_*
        order-type constants, so the bits are checked as raw integers here.
        """
        mask = spec.filling_mode_mask
        SYMBOL_FILLING_FOK, SYMBOL_FILLING_IOC = 1, 2
        if mask & SYMBOL_FILLING_IOC:
            return mt5.ORDER_FILLING_IOC
        if mask & SYMBOL_FILLING_FOK:
            return mt5.ORDER_FILLING_FOK
        return mt5.ORDER_FILLING_RETURN
