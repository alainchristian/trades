"""
Sizing tests with tick_value derived from contract size rather than hardcoded.

For a USD-quoted instrument:  tick_value = contract_size * tick_size
For USDJPY (USD is base):     tick_value = contract_size * tick_size / rate
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.risk import calculate_volume

EQUITY, RISK = 10_000.0, 1.0   # $100 risk budget

specs = [
    # name, entry, sl, tick_size, contract, quote_rate(None=USD quoted), vmin, vmax, vstep, expected
    ("EURUSD 20 pip", 1.08500, 1.08300, 0.00001, 100_000, None,  0.01, 100, 0.01, 0.50),
    ("USDJPY 30 pip", 157.500, 157.200, 0.001,   100_000, 157.5, 0.01, 100, 0.01, 0.52),
    ("XAUUSD $5",     2350.00, 2345.00, 0.01,    100,     None,  0.01,  50, 0.01, 0.20),
    ("US30 100pt",    39000.0, 38900.0, 1.0,     1,       None,  0.10,  20, 0.10, 1.00),
    ("GBPUSD 15 pip", 1.27000, 1.26850, 0.00001, 100_000, None,  0.01, 100, 0.01, 0.66),
    # Pulled live from symbol_info('EURUSD') on the MetaQuotes-Demo account used for
    # this project (2026-07-29): tick_size=1e-05, contract=100000, volume_min=0.01,
    # volume_max=500, volume_step=0.01, trade_tick_value_loss=1.0 — matches the
    # textbook EURUSD row above except for the broker's real volume_max (500, not
    # 100), confirming the sizing assumptions hold on this actual broker.
    ("EURUSD live-spec 20 pip", 1.13950, 1.13750, 0.00001, 100_000, None, 0.01, 500, 0.01, 0.50),
]

print(f"{'Instrument':16s} {'tick_val':>9s} {'lots':>7s} {'risk $':>8s}  verdict")
print("-" * 58)
fails = 0
for name, entry, sl, ts, contract, rate, vmin, vmax, vstep, exp in specs:
    tick_value = contract * ts / (rate if rate else 1.0)
    vol, risk = calculate_volume(EQUITY, RISK, entry, sl, ts, tick_value, vmin, vmax, vstep)
    ok = abs(vol - exp) < 1e-9
    fails += not ok
    print(f"{name:16s} {tick_value:9.4f} {vol:7.2f} {risk:8.2f}  {'OK' if ok else f'FAIL exp {exp}'}")

print("-" * 58)
print("All instruments size to ~$100 risk" if fails == 0 else f"{fails} failures")

# Prove risk never exceeds budget after step rounding, across many random stops
import random
random.seed(7)
over = 0
for _ in range(20_000):
    entry = 1.0 + random.random() * 0.5
    sl = entry - random.uniform(0.0005, 0.02)
    vol, risk = calculate_volume(EQUITY, RISK, entry, sl, 0.00001, 1.0, 0.01, 100, 0.01)
    if risk > 100.0 + 1e-6:
        over += 1
print(f"\nFuzz: 20,000 random stops -> {over} exceeded the $100 budget "
      f"({'OK — rounding is safe' if over == 0 else 'BUG'})")
