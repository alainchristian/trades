"""Smoke test: features + config + risk gates, using synthetic bars."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, pandas as pd
from agent import features
from agent.config import Config
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

print("\nAll pipeline checks completed.")
