"""
Indicator + price-action feature extraction.

Pure pandas — no MT5 dependency, so this is unit-testable and replayable
against historical data for backtesting.

MT5's Python API does not expose indicator buffers; everything is computed here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- indicators

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder's smoothing (alpha = 1/period), matching MT5's RSI
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    atr_ = atr(df, period)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr_
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr_
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean().fillna(0)


# ------------------------------------------------------------- price action

def swing_points(df: pd.DataFrame, left: int = 2, right: int = 2) -> pd.DataFrame:
    """
    Fractal swing highs/lows. A bar is a swing high if its high exceeds the
    `left` bars before and `right` bars after it.

    Note the `right` lookahead: the last `right` bars can never be confirmed
    swings. This is correct and intentional — it prevents repainting.
    """
    highs, lows = df["high"], df["low"]
    is_high = pd.Series(True, index=df.index)
    is_low = pd.Series(True, index=df.index)
    for i in range(1, left + 1):
        is_high &= highs > highs.shift(i)
        is_low &= lows < lows.shift(i)
    for i in range(1, right + 1):
        is_high &= highs > highs.shift(-i)
        is_low &= lows < lows.shift(-i)
    # invalidate the unconfirmable tail
    if right:
        is_high.iloc[-right:] = False
        is_low.iloc[-right:] = False
    return pd.DataFrame({"swing_high": is_high, "swing_low": is_low})


def market_structure(df: pd.DataFrame, left: int = 2, right: int = 2) -> dict:
    """
    Classify structure from the last confirmed swings.

    Returns trend ('bullish'/'bearish'/'ranging'), the last two swing highs/lows,
    and whether the most recent move was a break of structure (BOS, continuation)
    or a change of character (CHoCH, potential reversal).
    """
    sw = swing_points(df, left, right)
    sh = df.loc[sw["swing_high"], "high"]
    sl = df.loc[sw["swing_low"], "low"]

    result = {
        "trend": "ranging",
        "last_swing_highs": [round(float(x), 6) for x in sh.tail(2)],
        "last_swing_lows": [round(float(x), 6) for x in sl.tail(2)],
        "last_event": None,
    }
    if len(sh) < 2 or len(sl) < 2:
        return result

    hh = sh.iloc[-1] > sh.iloc[-2]
    hl = sl.iloc[-1] > sl.iloc[-2]
    lh = sh.iloc[-1] < sh.iloc[-2]
    ll = sl.iloc[-1] < sl.iloc[-2]

    if hh and hl:
        result["trend"] = "bullish"
    elif lh and ll:
        result["trend"] = "bearish"

    close = df["close"].iloc[-1]
    if close > sh.iloc[-1]:
        result["last_event"] = "BOS_bullish" if result["trend"] == "bullish" else "CHoCH_bullish"
    elif close < sl.iloc[-1]:
        result["last_event"] = "BOS_bearish" if result["trend"] == "bearish" else "CHoCH_bearish"
    return result


def candle_anatomy(row: pd.Series) -> dict:
    """
    Body/wick proportions of a single bar — lets the model reason about rejection.

    Wicks are clamped at zero: brokers occasionally serve bars where high/low do
    not strictly bound open/close, and a negative "wick percentage" reaching the
    model is worse than a slightly wrong one.
    """
    o, h, l, c = (float(row["open"]), float(row["high"]),
                  float(row["low"]), float(row["close"]))
    rng = h - l
    if rng <= 0:
        return {"body_pct": 0.0, "upper_wick_pct": 0.0,
                "lower_wick_pct": 0.0, "direction": "doji"}
    body = abs(c - o)
    upper = max(0.0, h - max(c, o))
    lower = max(0.0, min(c, o) - l)
    return {
        "body_pct": round(min(body / rng, 1.0) * 100, 1),
        "upper_wick_pct": round(upper / rng * 100, 1),
        "lower_wick_pct": round(lower / rng * 100, 1),
        "direction": "bull" if c > o else "bear" if c < o else "doji",
    }


def key_levels(df: pd.DataFrame, lookback: int = 100, tolerance_atr: float = 0.25) -> list[float]:
    """
    Cluster recent swing points into support/resistance levels, sorted by how
    many times price reacted there.
    """
    window = df.tail(lookback)
    sw = swing_points(window)
    pts = pd.concat([
        window.loc[sw["swing_high"], "high"],
        window.loc[sw["swing_low"], "low"],
    ]).sort_values()
    if pts.empty:
        return []

    tol = float(atr(window).iloc[-1]) * tolerance_atr
    clusters: list[list[float]] = []
    for p in pts:
        if clusters and abs(p - np.mean(clusters[-1])) <= tol:
            clusters[-1].append(float(p))
        else:
            clusters.append([float(p)])

    ranked = sorted(clusters, key=len, reverse=True)
    return [round(float(np.mean(c)), 6) for c in ranked[:6]]


# ---------------------------------------------------------------- assembly

def summarise_timeframe(df: pd.DataFrame, digits: int = 5, recent_bars: int = 6) -> dict:
    """Compact, token-efficient description of one timeframe."""
    close = df["close"]
    a = float(atr(df).iloc[-1])
    last = df.iloc[-1]

    def f(x, d=digits):
        """Native float, rounded. numpy scalars serialise badly into the prompt."""
        return None if x is None or pd.isna(x) else round(float(x), d)

    return {
        "bars_analysed": int(len(df)),
        "last_bar_time": df.index[-1].isoformat(),
        "close": f(close.iloc[-1]),
        "atr_14": f(a),
        "atr_pct_of_price": f(a / float(close.iloc[-1]) * 100, 3),
        "ema_20": f(ema(close, 20).iloc[-1]),
        "ema_50": f(ema(close, 50).iloc[-1]),
        "ema_200": f(ema(close, 200).iloc[-1]) if len(df) >= 200 else None,
        "rsi_14": f(rsi(close).iloc[-1], 1),
        "adx_14": f(adx(df).iloc[-1], 1),
        "structure": market_structure(df),
        "key_levels": key_levels(df),
        "last_candle": candle_anatomy(last),
        "recent_bars": [
            {
                "t": ts.strftime("%m-%d %H:%M"),
                "o": f(r["open"]),
                "h": f(r["high"]),
                "l": f(r["low"]),
                "c": f(r["close"]),
                "v": int(r["tick_volume"]) if "tick_volume" in r else None,
            }
            for ts, r in df.tail(recent_bars).iterrows()
        ],
    }
