# CLAUDE.md

Project context for Claude Code. Read this fully before making any change.

## What this is

An autonomous MT5 trading agent. Claude analyses market structure and proposes a
trade; a deterministic risk engine independently validates and sizes it. The model
never places orders.

Currently: demo account, `dry_run: true`. Not yet run against live market data.

## Environment

- **Windows only.** The `MetaTrader5` package uses local IPC to a running terminal.
  It will not work on Linux, WSL, or in any container.
- MT5 terminal must be **open and logged in** while the agent runs.
- AutoTrading must be enabled (toolbar button + Tools → Options → Expert Advisors).
- Python must be **64-bit**, matching the terminal architecture.
- Virtualenv at `.venv`. Activate with `.venv\Scripts\activate`.

Verify the environment before debugging anything else:

```powershell
python -c "import MetaTrader5 as mt5; print(mt5.initialize(), mt5.version(), mt5.account_info())"
```

If this returns `False`, no application-level fix will help. The problem is the
terminal, the architecture mismatch, or AutoTrading.

---

## Invariants — do not violate these

These are not style preferences. Each one prevents a specific way this system can
lose money. If a change appears to require breaking one, stop and ask.

1. **The model never calls `order_send`.** `decision.py` produces a proposal.
   `risk.py` decides whether it becomes an order. Do not add a shortcut path.

2. **The model never specifies volume.** It returns `risk_fraction` (0–1) of a
   configured maximum. `risk.py` converts that to lots. There must be no code path
   where a model-supplied number reaches the broker as volume.

3. **Volume rounds DOWN to `volume_step`, never to nearest.** Rounding up silently
   exceeds the risk budget. `round_to_step` also guards float-floor error — do not
   "simplify" it to `round()` or `//`.

4. **Never bump a sub-minimum volume up to `volume_min`.** If the correct size is
   below the broker minimum, the trade is refused. That is correct behaviour.

5. **Stops may only tighten.** `validate_stop_modification` rejects any widening.
   No exceptions, no "the model had a good reason" override.

6. **`bars()` drops the forming bar.** This makes indicator values stable within a
   bar and decisions reproducible on replay. It looks like an off-by-one. It is not.
   Do not remove `drop_forming`.

7. **Sizing uses tick value, never pips.** `trade_tick_value_loss` × tick count.
   Pip math mis-sizes JPY crosses, metals and indices.

8. **Every order carries `magic_number`.** All position queries filter on it. The
   bot must never touch a position it did not open.

9. **Secrets live in `.env`, never in YAML or code.** `.env` and `KILL` are
   gitignored. Never commit credentials, never print them in logs.

10. **Do not weaken a risk gate to make a test pass.** If a gate rejects something
    it shouldn't, fix the input or the gate's logic — never lower the threshold to
    get a green run.

---

## Permission boundaries

- You may freely run: `tests/*`, `run.py --once` with `dry_run: true`, linting,
  any read-only MT5 call.
- **Ask before**: setting `dry_run: false`, changing anything under `risk:` in
  config, or running `run.py` in continuous mode.
- **Never**: set `allow_live_account: true`. That is a human decision, made once,
  after the evaluation phase in README.md.

---

## Testing

```powershell
python tests\test_sizing2.py    # sizing across FX, JPY, metals, indices + fuzz
python tests\test_pipeline.py   # features, config validation, risk gates
```

Both must pass before any commit. `test_sizing2.py` includes a 20,000-case fuzz
asserting realised risk never exceeds budget after rounding — if that fails, stop
and fix it before anything else.

When you add a risk gate, add a case to `test_pipeline.py` proving it rejects.
When you touch sizing, add an instrument to `test_sizing2.py`.

---

## Code conventions

- Layers stay separate. `features.py` has no MT5 import. `risk.py` has no AI and no
  MT5 import. This keeps both unit-testable and replayable against historical data.
- All values entering the model snapshot must be native Python types. numpy scalars
  serialise badly into the prompt — use the `f()` helper in `summarise_timeframe`.
- Log rejections with the specific reason, never a generic failure.
- Prefer failing closed. On any uncertainty — API timeout, missing data, unparseable
  response — the correct action is `hold`.

---

## Known broker variability

- Symbol names differ: `EURUSD` may be `EURUSD.a`, `EURUSDm`, `EURUSD.raw`. Read
  Market Watch, put the exact string in config.
- `trade_tick_value_loss` is 0 on some brokers' exotic symbols. Client falls back to
  `trade_tick_value`. Verify sizing with one small demo trade per new instrument.
- Filling modes vary. `resolve_filling` reads the symbol bitmask. Retcode 10030 =
  unsupported filling mode.
- `trade_stops_level` is 0 on some ECN accounts, meaning no minimum. Do not assume
  it is always positive.
