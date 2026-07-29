# Build Plan — Claude Code Handoff

Work these phases in order. Each has an explicit exit condition; do not start the
next phase until the current one's exit condition is met.

The prompts below are written to be pasted directly into Claude Code, one at a time.
Do not paste the whole file at once — each phase produces information the next phase
depends on.

---

## Setup (do this yourself, once)

```powershell
cd path\to\mt5_claude_agent
git init
git add -A
git commit -m "Initial scaffold"

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

copy config.example.yaml config.yaml
copy .env.example .env
notepad .env          # fill in ANTHROPIC_API_KEY, MT5_LOGIN, MT5_PASSWORD, MT5_SERVER
```

Create `.gitignore` before your first commit of real config:

```
.env
config.yaml
KILL
logs/
.venv/
__pycache__/
```

Then start Claude Code in the project root:

```powershell
claude
```

The git repo matters. It is your undo button for anything Claude Code changes —
`git reset --hard` recovers any state.

---

## Phase 0 — Environment verification

> I have an MT5 trading agent scaffold in this directory. Read CLAUDE.md and
> README.md first.
>
> Before touching any code, verify my environment is capable of running this:
> confirm Python is 64-bit, confirm the MetaTrader5 package imports, confirm
> `mt5.initialize()` succeeds against my running terminal, and print the account
> info so I can confirm it is a demo account. Also confirm my Python version meets
> the requirements.
>
> Report what you find. Do not fix anything yet — I want to see the raw state first.

**Exit condition:** `mt5.initialize()` returns True and `account_info()` shows
`trade_mode` = demo. If not, everything downstream is guesswork.

---

## Phase 1 — First real-data run

> Now run `python run.py --once` with dry_run still true.
>
> This is the first time the code has touched live market data — expect breakage.
> Work through failures one at a time. For each one, tell me the root cause before
> you change anything, because some failures are my broker's configuration rather
> than bugs in the code.
>
> Pay particular attention to: the exact symbol name my broker uses, whether
> `trade_tick_value_loss` is populated, what `trade_stops_level` is, and whether
> the filling mode resolves correctly.
>
> When it completes, show me the JSON snapshot that was sent to the model and the
> decision that came back.

**Exit condition:** one clean cycle, and you have read the actual snapshot with your
own eyes. Check that the indicator values and structure classification match what you
see on your own chart. If they don't, the model is analysing something other than the
market you think it is.

---

## Phase 2 — Broker calibration

> Update config.yaml for my broker based on what we found: exact symbol name,
> appropriate `max_spread_points` for this instrument, and session windows that
> match when I actually want to trade.
>
> Then add a test to tests/test_sizing2.py using my broker's real contract specs
> for the symbols in my config, pulled live from `symbol_info`. I want to confirm
> the sizing math is correct against my actual broker, not against textbook values.
>
> Do not change any risk thresholds.

**Exit condition:** sizing verified against your broker's real specs.

---

## Phase 3 — Journal analytics

> Build `analyse.py`. It reads logs/cycles_*.jsonl and reports:
>
> - Decision counts by action, and the hold rate
> - For executed trades: win rate, average R multiple, expectancy per trade,
>   maximum drawdown, and profit factor
> - **Confidence calibration**: bucket decisions by stated confidence (0.5-0.6,
>   0.6-0.7, etc.) and show the realised win rate in each bucket. This is the most
>   important output in the file.
> - Risk rejection counts grouped by reason
> - Total API cost and average latency per cycle
> - Performance broken down by setup_name and by hour of day
>
> Trades need outcomes attached. Match closed deals from MT5 history back to journal
> entries by ticket, and write the realised P/L and R multiple into the record.
>
> Print a clear summary table. Warn explicitly when the sample is too small to draw
> conclusions from — under 30 trades, say so prominently.

**Exit condition:** you can run one command and see whether the model's confidence
means anything. If confidence is uncorrelated with outcomes, the model is adding
variance rather than signal, and no amount of prompt tuning fixes that.

---

## Phase 4 — Replay backtester

> Build `replay.py`. It should pull historical bars from MT5, step through them bar
> by bar, build the same snapshot `context.build_snapshot` produces, call the model,
> and run the result through the real risk engine — then simulate the fill and track
> the equity curve.
>
> Requirements:
> - Reuse the actual production modules. Do not reimplement features or risk logic;
>   a backtest that tests different code than production is worse than no backtest.
> - Strict no-lookahead: at bar N the snapshot may only contain data through bar N.
>   Assert this rather than assuming it.
> - Simulate spread and a configurable slippage assumption.
> - Support `--dry` mode that builds every snapshot without calling the API, so I
>   can verify correctness for free before spending money.
> - Cache model responses keyed by snapshot hash so re-runs are cheap.
> - Report the cost estimate before starting, and ask me to confirm.

**Exit condition:** a replay over a few hundred bars, with the no-lookahead assertion
passing. Treat the resulting equity curve as weak evidence — one path over one period
is not an edge.

---

## Phase 5 — Operational hardening

> Add the things needed to run this unattended:
>
> - Windows Task Scheduler setup so the agent restarts on boot, with instructions
> - A heartbeat: if a cycle hasn't completed in 2× the expected interval, alert me
> - Notifications on fills, rejections, kill-switch activation and daily-loss halts
>   (Telegram bot or email, your choice — tell me which and why)
> - A `status.py` showing open positions, daily P/L against limit, current streak,
>   and time to next cycle
> - Log rotation so logs/ doesn't grow unbounded
> - Graceful handling of terminal disconnects: reconnect with backoff, and never
>   place an order on a stale connection

**Exit condition:** you can leave it running and find out quickly when it stops.

---

## What NOT to ask Claude Code to do

**Do not ask it to improve win rate by tuning the prompt against your journal.** With
a few hundred decisions you will be fitting noise, and you'll destroy your only honest
evaluation set in the process. If you want to test a prompt change, decide on it in
advance and run it forward.

**Do not ask it to relax a risk gate because the gate is blocking trades.** Gates
blocking trades is gates working. The rejection log in Phase 3 tells you which ones
fire most; if one is genuinely miscalibrated for your instrument, change it
deliberately with a reason written down — not because it was inconvenient.

**Do not ask it to add martingale, grid, averaging down, or "recovery" logic.** These
convert a visible small loss into a hidden large one. If it appears in a suggestion,
reject it.

**Do not let it go live.** `allow_live_account` is a human decision, made once, after
Phase 3 gives you a real answer over at least 100 decisions.

---

## Useful Claude Code habits for this project

- `/clear` between phases. Long contexts drift, and this project has invariants worth
  re-reading from CLAUDE.md rather than remembering fuzzily.
- Commit after every phase. `git reset --hard` is your recovery path.
- When it proposes a change to `risk.py`, read the diff yourself. That file is the
  one standing between a hallucination and your account.
- Ask "what would make this wrong?" before accepting analysis of your results. It is
  easy to get an agreeable read on an equity curve that means nothing.
