# MT5 + Claude Trading Agent

An autonomous trading agent that analyses MT5 market data with Claude and executes
through MetaTrader 5. Claude proposes; a deterministic risk engine disposes.

**Ships configured for demo accounts with `dry_run: true`. It will refuse to start on
a live account until you explicitly change `allow_live_account`.**

---

## Architecture

```
MT5 terminal
     │  MetaTrader5 python package
     ▼
mt5_client.py   connection, bars, specs, positions, account
     ▼
features.py     EMA / RSI / ATR / ADX, swing points, BOS-CHoCH, key levels
     ▼
context.py      compact JSON snapshot + system prompt + tool schema
     ▼
decision.py  ── Claude API (forced tool use → structured decision)
     ▼
risk.py         ◄── THE GATE. No AI. Sizes, validates, rejects.
     ▼
execution.py    order_send, retries, fills
     ▼
journal.py      JSONL record of every cycle: input, decision, verdict, fill
```

The model never calls `order_send`. It returns a proposal; `risk.py` independently
decides whether that proposal becomes an order and at what size. The model cannot
specify lots at all — only a `risk_fraction` between 0 and 1 of a maximum you set
in config. This is the property that makes autonomous operation survivable.

---

## Setup (Windows)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

copy config.example.yaml config.yaml
copy .env.example .env          # then fill in your keys
```

Requirements:
- MT5 terminal installed, logged into your demo account, and **running**
- AutoTrading enabled (the button in the toolbar)
- Tools → Options → Expert Advisors → "Allow algorithmic trading" ticked
- Python 3.10+ (64-bit, matching your terminal)

Set your environment variables, then:

```powershell
python run.py --once      # single cycle, dry run — verify it reads your market
python run.py             # continuous, on bar close
```

`--once` is the right first command. It connects, pulls data, calls the model,
runs the risk gate and logs the whole thing without sending an order.

---

## Rollout order

Do not skip steps. Each one catches a different class of failure.

1. **`dry_run: true`, demo** — a week minimum. You are checking that data pulls
   are clean, the model produces coherent decisions, and the risk engine rejects
   what it should. Read `logs/cycles_*.jsonl` daily.
2. **`dry_run: false`, demo** — real order flow. Now you find broker-specific
   problems: filling modes, minimum stop distances, symbol naming, requotes.
3. **Evaluate.** At least 100 decisions before you conclude anything. Measure:
   win rate, average R, expectancy, and whether the model's stated `confidence`
   correlates with outcomes. If confidence is uncorrelated with results, the
   model isn't adding signal — it's adding variance.
4. **Live, minimum size** — only if step 3 was positive, and with
   `max_risk_per_trade_pct` at 0.25 or lower.

Between steps 3 and 4 there is no shortcut. A demo curve that looks good over
30 trades is indistinguishable from noise.

---

## Safety controls

| Control | Where | Effect |
|---|---|---|
| Demo-only gate | `mt5_client.connect` | Refuses non-demo unless `allow_live_account: true` |
| Dry run | `config.dry_run` | Logs orders, sends nothing |
| Kill switch | create a file named `KILL` | Flattens all positions, exits |
| Daily loss limit | `run.py` main loop | Flattens, halts for 24h |
| Magic number | every order | Bot only ever manages its own positions |
| Per-trade risk cap | `risk.py` | Model cannot exceed it regardless of output |
| Stop-widening ban | `risk.validate_stop_modification` | Stops may only tighten |
| Loss-streak cooldown | `risk.evaluate_open` | Halts after N consecutive losses |
| Spread ceiling | `risk.evaluate_open` | Blocks entries in poor conditions |
| Session windows | `journal.SessionState` | No trading outside your hours or near rollover |

The kill switch is checked during the sleep between cycles, not just at the top,
so `echo. > KILL` takes effect within five seconds.

---

## Running costs (one symbol, 13-hour session)

| Timeframe | Sonnet | Opus |
|---|---|---|
| M5 | ~$33/mo | ~$165/mo |
| M15 | ~$11/mo | ~$55/mo |
| H1 | ~$3/mo | ~$14/mo |

Measured at ~1,100 input tokens per cycle. Multiply by symbol count. M15 or higher
is the sensible range — below that you pay a lot for noise, and 2–20s of model
latency starts to matter relative to the bar.

---

## Tests

```bash
python tests/test_sizing2.py    # sizing across FX, JPY crosses, metals, indices
python tests/test_pipeline.py   # features, config validation, risk gates
```

The sizing test includes a 20,000-case fuzz confirming that step-rounding never
pushes realised risk above the configured budget.

---

## What to watch for

**Broker symbol names vary.** `EURUSD` may be `EURUSD.a`, `EURUSDm`, or `EURUSD.raw`.
Check Market Watch and put the exact string in config.

**`trade_tick_value_loss` is occasionally zero** on some brokers' exotic symbols.
The client falls back to `trade_tick_value`, but verify sizing on a demo trade for
any instrument before trusting it.

**Filling modes are broker-specific.** `resolve_filling` reads the symbol's bitmask
rather than assuming IOC. If you get retcode 10030, that's the culprit.

**Bar indexing.** `bars()` drops the forming bar by default. This is deliberate — it
means indicator values never change within a bar, so decisions are reproducible on
replay. Do not "fix" it.

---

## Honest framing

The parts of this system that are reliably valuable are the unglamorous ones:
consistent position sizing, enforced stops, no revenge trading, no size creep after
a loss, and a complete decision journal you can audit. Those are real edges over
discretionary trading, and they come from `risk.py`, not from the model.

Whether an LLM reading price structure has predictive edge is an open question, and
this system is built to let you answer it honestly rather than to assume it. The
journal exists for that purpose. Look at it before you scale anything.

Trading involves substantial risk of loss. Nothing here is financial advice.
