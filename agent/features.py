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


def break_recency(close: pd.Series, extreme: pd.Series, level: float,
                   bullish: bool) -> tuple[int | None, float | None]:
    """
    How long ago price broke `level`, and how far it has since retraced back
    toward it -- the piece `last_event` alone can't express, since that field
    only reflects the CURRENT bar's close vs. level and has no memory of when
    the break happened or how far price has travelled since.

    Finds the most recent contiguous run where close was beyond `level` (above
    it if bullish, below if bearish), even if that run has since ended (price
    has pulled back through `level` again). Returns:
      - bars_since_break: bars since that run started (the actual break bar)
      - retracement_pct: 0 at the run's most extreme point, 100 back at
        `level`, >100 if price has since broken back through it entirely

    (None, None) if close has never been beyond `level` in the given window.
    """
    beyond = close > level if bullish else close < level
    if not beyond.any():
        return None, None

    true_idx = np.flatnonzero(beyond.to_numpy())
    run_end = true_idx[-1]
    run_start = run_end
    while run_start > 0 and beyond.iloc[run_start - 1]:
        run_start -= 1
    bars_since = len(close) - 1 - run_start

    if bullish:
        peak = extreme.iloc[run_start:run_end + 1].max()
        span = peak - level
        retracement = (peak - close.iloc[-1]) / span * 100 if span > 0 else 0.0
    else:
        peak = extreme.iloc[run_start:run_end + 1].min()
        span = level - peak
        retracement = (close.iloc[-1] - peak) / span * 100 if span > 0 else 0.0

    return bars_since, round(float(max(retracement, 0.0)), 1)


def _swing_direction(swings: pd.Series, n: int = 4) -> str:
    """
    Direction of the last `n` confirmed swings of one type (all highs, or all
    lows), by majority of consecutive pairwise moves rather than requiring
    every single one to agree. A lone non-conforming swing -- noise, not a
    genuine reversal -- costs one vote instead of flipping the whole read to
    "ranging".

    With exactly 2 swings available (the minimum for any read at all) this
    collapses to a single comparison, identical to the old strict check --
    the majority vote only kicks in once there's enough history for "majority"
    to mean something, so early-window behaviour is no more conservative than
    before. `n=4` (3 pairwise moves) is odd-vote-equivalent: 4 swings never
    tie, they only can with exactly 3 (one tie-breaking "flat" vote short).
    """
    diffs = swings.tail(n).diff().dropna()
    up = int((diffs > 0).sum())
    down = int((diffs < 0).sum())
    if up > down:
        return "up"
    if down > up:
        return "down"
    return "flat"


def market_structure(df: pd.DataFrame, left: int = 2, right: int = 2) -> dict:
    """
    Classify structure from the last confirmed swings.

    trend is a majority vote over the last few swings of each type (see
    _swing_direction), not a strict "every single one must agree" check --
    one non-conforming swing out of several is noise, not evidence the trend
    ended, and treating it as disqualifying was misclassifying visibly
    trending markets as "ranging" on that single swing alone.

    Returns trend ('bullish'/'bearish'/'ranging'), the last two swing highs/lows,
    whether the most recent move was a break of structure (BOS, continuation)
    or a change of character (CHoCH, potential reversal), and -- via
    bars_since_break/retracement_pct -- how long ago that break happened and
    how far price has since retraced, so a retest doesn't require the
    contradiction of "breaking out" and "currently pulled back" at once.

    choch_confirmed answers a question last_event alone cannot: has this CHoCH
    (a single reversal bar) been structurally validated, or is it still just an
    unconfirmed reversal attempt? Confirmation is a *second*, independent swing
    in the reversal's direction breaking beyond the first (hh for a CHoCH_bullish,
    ll for a CHoCH_bearish) -- price didn't just poke through once, it printed a
    fresh confirmed extreme afterwards -- with that newer swing's own retest
    (bars_since_break/retracement_pct, already computed above since it uses
    the same sh.iloc[-1]/sl.iloc[-1] this hh/ll check reads) sitting in the
    20-80% band. Deliberately does not require the *other* swing type to agree
    (i.e. does not wait for `trend` to fully flip to the reversal direction):
    that stricter, both-types-at-once gate is what leaves `trend` at "ranging"
    on a single non-conforming swing, and it would starve confirmation of the
    exact cases -- one side of structure moving before the other -- it exists
    to catch.
    """
    sw = swing_points(df, left, right)
    sh = df.loc[sw["swing_high"], "high"]
    sl = df.loc[sw["swing_low"], "low"]

    result = {
        "trend": "ranging",
        "last_swing_highs": [round(float(x), 6) for x in sh.tail(2)],
        "last_swing_lows": [round(float(x), 6) for x in sl.tail(2)],
        "last_event": None,
        "bars_since_break": None,
        "retracement_pct": None,
        "choch_confirmed": False,
    }
    if len(sh) < 2 or len(sl) < 2:
        return result

    # Strict last-two comparisons: kept for choch_confirmed below, which asks
    # a narrower question ("did the freshest swing beat the one before it?")
    # than the multi-swing trend read this function otherwise computes.
    hh = sh.iloc[-1] > sh.iloc[-2]
    ll = sl.iloc[-1] < sl.iloc[-2]

    high_dir = _swing_direction(sh)
    low_dir = _swing_direction(sl)
    if high_dir == "up" and low_dir == "up":
        result["trend"] = "bullish"
    elif high_dir == "down" and low_dir == "down":
        result["trend"] = "bearish"

    close_last = df["close"].iloc[-1]
    if close_last > sh.iloc[-1]:
        result["last_event"] = "BOS_bullish" if result["trend"] == "bullish" else "CHoCH_bullish"
    elif close_last < sl.iloc[-1]:
        result["last_event"] = "BOS_bearish" if result["trend"] == "bearish" else "CHoCH_bearish"

    if result["last_event"] in ("BOS_bullish", "CHoCH_bullish"):
        bars_since, retracement = break_recency(df["close"], df["high"], sh.iloc[-1], bullish=True)
        result["bars_since_break"] = bars_since
        result["retracement_pct"] = retracement
    elif result["last_event"] in ("BOS_bearish", "CHoCH_bearish"):
        bars_since, retracement = break_recency(df["close"], df["low"], sl.iloc[-1], bullish=False)
        result["bars_since_break"] = bars_since
        result["retracement_pct"] = retracement

    retr = result["retracement_pct"]
    if result["last_event"] == "CHoCH_bullish" and hh and retr is not None and 20 <= retr <= 80:
        result["choch_confirmed"] = True
    elif result["last_event"] == "CHoCH_bearish" and ll and retr is not None and 20 <= retr <= 80:
        result["choch_confirmed"] = True

    return result


def hold_confidence(timeframes: dict) -> float:
    """
    Confidence for a 'hold' decision, derived from cross-timeframe trend agreement
    and ADX -- not from the model's own guess. An LLM asked for an open-ended 0-1
    confidence number clusters in a narrow, "reasonable-sounding" band regardless
    of how different the underlying setups are (empirically: 0.68-0.78 across an
    outright timeframe conflict, a flat/no-structure market, and a genuine
    near-miss). Trend and ADX are already exact numbers computed above, so there
    is nothing left for the model to calibrate -- compute it directly instead.
    """
    order = ("H4", "H1", "M15")
    trends = [timeframes[tf]["structure"]["trend"] for tf in order if tf in timeframes]
    adx_values = [timeframes[tf]["adx_14"] for tf in order if tf in timeframes]

    directional = [t for t in trends if t != "ranging"]
    bullish = directional.count("bullish")
    bearish = directional.count("bearish")

    if not directional:
        agreement = "absent"       # nothing trending anywhere -- unambiguous hold
    elif bullish and bearish:
        agreement = "conflicting"  # timeframes actively disagree -- also a clear hold
    else:
        agreement = "aligned"      # a real trend exists but the trade didn't clear every gate

    avg_adx = sum(adx_values) / len(adx_values) if adx_values else 0.0

    if agreement == "absent":
        return 0.85
    if agreement == "conflicting":
        return 0.7 if avg_adx >= 20 else 0.55
    # "aligned": a genuine near-miss -- the stronger the aligned trend, the
    # shakier the hold call actually is.
    return 0.2 if avg_adx >= 25 else 0.35


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
