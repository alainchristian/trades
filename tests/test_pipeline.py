"""Smoke test: features + config + risk gates, using synthetic bars."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, pandas as pd
from agent import features
from agent.config import Config
from agent.context import DECISION_TOOL
from agent.risk import RiskEngine
from agent.mt5_client import SymbolSpec

# --- synthetic uptrending EURUSD-like series -------------------------------
np.random.seed(42)
n = 300
drift = np.linspace(0, 0.012, n)
noise = np.cumsum(np.random.normal(0, 0.0006, n))
close = 1.0800 + drift + noise
high = close + np.abs(np.random.normal(0, 0.0004, n))
low = close - np.abs(np.random.normal(0, 0.0004, n))
open_ = np.r_[close[0], close[:-1]]
df = pd.DataFrame(
    {"open": open_, "high": high, "low": low, "close": close,
     "tick_volume": np.random.randint(200, 2000, n)},
    index=pd.date_range("2026-07-01", periods=n, freq="15min", tz="UTC"),
)

print("=== features.summarise_timeframe ===")
s = features.summarise_timeframe(df, digits=5)
for k in ["close", "atr_14", "ema_20", "ema_50", "ema_200", "rsi_14", "adx_14"]:
    print(f"  {k:12s} {s[k]}")
print(f"  structure    {s['structure']['trend']} / event={s['structure']['last_event']}")
print(f"  bars_since_break={s['structure']['bars_since_break']} "
      f"retracement_pct={s['structure']['retracement_pct']}")
print(f"  key_levels   {s['key_levels'][:4]}")
print(f"  last_candle  {s['last_candle']}")
print(f"  recent_bars  {len(s['recent_bars'])} bars, keys={list(s['recent_bars'][0])}")

# no-repaint check: last confirmed swing must not use the final 2 bars
sw = features.swing_points(df)
assert not sw["swing_high"].iloc[-2:].any(), "REPAINT: swing marked in unconfirmable tail"
assert not sw["swing_low"].iloc[-2:].any(), "REPAINT: swing marked in unconfirmable tail"
print("  no-repaint check ......... OK")

# RSI bounds
r = features.rsi(df["close"])
assert r.between(0, 100).all(), "RSI out of bounds"
print("  RSI bounds ............... OK")

# --- features.break_recency: bars_since_break / retracement_pct -------------
# last_event alone can only say "breaking out right now" -- it can't say "broke
# out 5 bars ago and has since retraced 70% back toward the origin", which is
# exactly the fact a break-and-retest entry needs. break_recency computes that.
print("\n=== features.break_recency ===")

# still extended: broke above 1.1000 at bar 3, ran to a 1.1050 peak, has only
# pulled back to 1.1015 -- expect bars_since=5, retracement=70% of the way
# back to the origin (not yet there).
close_a = pd.Series([1.0980, 1.0985, 1.0990, 1.1010, 1.1030, 1.1050, 1.1040, 1.1025, 1.1015])
bars_since, retracement = features.break_recency(close_a, close_a, level=1.1000, bullish=True)
assert bars_since == 5, f"expected bars_since=5, got {bars_since}"
assert retracement == 70.0, f"expected retracement=70.0, got {retracement}"
print(f"  still extended, mid-pullback -> bars_since={bars_since} retracement={retracement} OK")

# failed retest: broke above 1.1000 at bar 2, peaked at 1.1030, then fully
# reversed back below the origin to 1.0985 -- retracement must exceed 100
# (the origin has failed, not "a deeper discount entry").
close_b = pd.Series([1.0980, 1.0990, 1.1010, 1.1030, 1.1010, 1.0995, 1.0985])
bars_since, retracement = features.break_recency(close_b, close_b, level=1.1000, bullish=True)
assert bars_since == 4, f"expected bars_since=4, got {bars_since}"
assert retracement == 150.0, f"expected retracement=150.0, got {retracement}"
print(f"  broken back through origin -> bars_since={bars_since} retracement={retracement} OK")

# never broken: close has never exceeded level -> (None, None), not zero
close_c = pd.Series([1.0980, 1.0985, 1.0990])
bars_since, retracement = features.break_recency(close_c, close_c, level=1.1000, bullish=True)
assert (bars_since, retracement) == (None, None), \
    f"expected (None, None) when never broken, got ({bars_since}, {retracement})"
print("  never broken -> (None, None) ... OK")

# bearish mirror: broke below 1.1000 support at bar 2, bottomed at 1.0950,
# has retraced back up 60% of the way to the origin.
close_d = pd.Series([1.1020, 1.1010, 1.0990, 1.0970, 1.0950, 1.0965, 1.0980])
bars_since, retracement = features.break_recency(close_d, close_d, level=1.1000, bullish=False)
assert bars_since == 4, f"expected bars_since=4, got {bars_since}"
assert retracement == 60.0, f"expected retracement=60.0, got {retracement}"
print(f"  bearish mirror -> bars_since={bars_since} retracement={retracement} OK")

# --- config validation ------------------------------------------------------
print("\n=== config validation ===")
import yaml, tempfile
base = yaml.safe_load(open("config.example.yaml"))
cfg = Config.load("config.example.yaml")
print(f"  example config loads ..... OK (risk/trade {cfg.risk.max_risk_per_trade_pct}%)")

bad = dict(base); bad["risk"] = dict(base["risk"], max_risk_per_trade_pct=5.0)
with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
    yaml.dump(bad, f); path = f.name
try:
    Config.load(path); print("  BUG: accepted 5% per-trade risk")
except ValueError as e:
    print(f"  rejects reckless config .. OK ({str(e).splitlines()[1].strip()[:60]})")

# --- risk engine gates ------------------------------------------------------
print("\n=== risk engine gates ===")
spec = SymbolSpec("EURUSD", 5, 0.00001, 0.00001, 1.0, 100000, 0.01, 100, 0.01, 10, 5, 12, 2, 4)
account = {"equity": 10000.0, "balance": 10000.0, "margin_free": 8000.0, "trade_allowed": True}
engine = RiskEngine(cfg)

def gate(label, **kw):
    base_kw = dict(direction="buy", entry=1.09000, stop_loss=1.08800, take_profit=1.09400,
                   risk_fraction=1.0, spec=spec, account=account, open_positions=[],
                   spread_points=12, daily_realised=0.0, session_ok=True,
                   consecutive_losses=0)
    base_kw.update(kw)
    v = engine.evaluate_open(**base_kw)
    status = f"APPROVED {v.volume} lots (${v.risk_amount}, R:R {v.rr_ratio})" if v.approved \
             else f"rejected: {v.rejections[0][:62]}"
    print(f"  {label:30s} {status}")
    return v

v = gate("valid long")
assert v.approved and v.volume > 0
gate("inverted stop (SL above entry)", stop_loss=1.09200)
gate("TP below entry on a buy", take_profit=1.08500)
gate("R:R below minimum", take_profit=1.09100)
gate("stop inside broker minimum", stop_loss=1.08995)
gate("spread too wide", spread_points=99)
gate("daily loss limit hit", daily_realised=-350.0)
gate("outside session", session_ok=False)
gate("loss-streak cooldown", consecutive_losses=3)
gate("risk_fraction hacked to 99", risk_fraction=99)


# --- hold confidence: deterministic, not the model's guess ------------------
# 38/38 real logged cycles, and 9/9 fresh replayed API calls across three very
# different setups (outright conflict, flat market, genuine near-miss), all came
# back confidence=0.3 on "hold" -- an LLM asked for an open-ended 0-1 number
# clusters in a narrow band regardless of how different the underlying market
# structure is. Rewording the prompt only narrowed the cluster (0.68-0.78); it
# never produced real separation. Trend and ADX per timeframe are already exact,
# pure-pandas numbers computed above, so features.hold_confidence derives
# confidence directly from them instead of trusting the model's float.
print("\n=== features.hold_confidence ===")


def tf(trend, adx_14):
    return {"structure": {"trend": trend}, "adx_14": adx_14}


cases = [
    ("absent (all ranging)",
     {"H4": tf("ranging", 15), "H1": tf("ranging", 18), "M15": tf("ranging", 12)}, 0.85),
    ("conflicting, weak ADX",
     {"H4": tf("bearish", 12), "H1": tf("bullish", 15), "M15": tf("bearish", 10)}, 0.55),
    ("conflicting, strong ADX",
     {"H4": tf("bullish", 30), "H1": tf("bearish", 28), "M15": tf("bullish", 25)}, 0.7),
    ("aligned near-miss, moderate ADX",
     {"H4": tf("ranging", 16), "H1": tf("bullish", 22), "M15": tf("bullish", 20)}, 0.35),
    ("aligned near-miss, strong ADX",
     {"H4": tf("ranging", 17), "H1": tf("bullish", 37), "M15": tf("bullish", 30)}, 0.2),
]
for label, timeframes, expected in cases:
    got = features.hold_confidence(timeframes)
    assert got == expected, f"{label}: expected {expected}, got {got}"
    print(f"  {label:34s} -> {got:.2f} OK")

# A genuine near-miss must always score lower than an outright conflict or a
# flat market -- that ordering, not the exact numbers, is the point of the fix.
absent_conf = features.hold_confidence(cases[0][1])
nearmiss_conf = features.hold_confidence(cases[4][1])
assert nearmiss_conf < absent_conf, "near-miss must score lower confidence than a flat market"
print("  near-miss scores lower than flat/conflict ... OK")

# Schema-level: confidence is no longer unconditionally required (only 'open'
# needs it now; decision.py._sanity_check enforces that at the code level).
assert "confidence" not in DECISION_TOOL["input_schema"]["required"], \
    "confidence should not be unconditionally required -- only 'open' needs it"
print("  confidence not globally required in schema ... OK")

print("\nAll pipeline checks completed.")
