"""
Builds the structured market snapshot handed to the model.

Design note: everything the model sees is pre-computed numbers, never raw
tick streams. This keeps token cost flat, makes the input reproducible for
replay, and stops the model from doing arithmetic it is bad at.
"""
from __future__ import annotations

from datetime import datetime, timezone

from . import features


def build_snapshot(client, symbol: str, cfg, daily_realised: float,
                   consecutive_losses: int, recent_trades: list[dict]) -> dict:
    spec = client.spec(symbol)
    account = client.account()
    positions = client.positions(symbol)
    tick = client.tick(symbol)

    timeframes = {}
    for tf in cfg.context_timeframes:
        df = client.bars(symbol, tf, cfg.bars_per_timeframe)
        timeframes[tf] = features.summarise_timeframe(df, digits=spec.digits)

    equity = account["equity"]
    max_risk = round(equity * cfg.risk.max_risk_per_trade_pct / 100, 2)
    daily_limit = round(equity * cfg.risk.max_daily_loss_pct / 100, 2)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "decision_timeframe": cfg.decision_timeframe,
        "market": {
            "bid": tick.bid,
            "ask": tick.ask,
            "spread_points": round((tick.ask - tick.bid) / spec.point, 1),
            "digits": spec.digits,
            "point": spec.point,
            "min_stop_distance_points": spec.stops_level_points,
        },
        "timeframes": timeframes,
        "account": {
            "equity": equity,
            "balance": account["balance"],
            "currency": account["currency"],
            "free_margin": account["margin_free"],
        },
        "open_positions": positions,
        "risk_budget": {
            "max_risk_per_trade_currency": max_risk,
            "max_risk_per_trade_pct": cfg.risk.max_risk_per_trade_pct,
            "min_reward_risk_ratio": cfg.risk.min_rr,
            "positions_open": len(positions),
            "positions_allowed": cfg.risk.max_open_positions,
            "daily_pnl_so_far": daily_realised,
            "daily_loss_limit": -daily_limit,
            "consecutive_losses": consecutive_losses,
        },
        "recent_decisions": recent_trades[-5:],
    }


SYSTEM_PROMPT = """You are the analysis component of an automated trading system. \
You do not execute trades; you submit a proposal that a deterministic risk engine \
will independently validate, size, and either accept or reject.

Your task: analyse the supplied multi-timeframe market snapshot and submit exactly \
one decision via the submit_decision tool.

Method:
- Establish bias from the highest timeframe first, then look for alignment on lower \
timeframes. Conflicting timeframes are a reason to hold, not to pick a side.
- Trade with structure, not against it. A CHoCH is only tradeable once choch_confirmed is \
true in the snapshot -- this means a second, independent swing has broken further in the \
CHoCH's new direction (beyond the original reversal bar) and that newer swing's own retest \
sits in the 20-80% band. An unconfirmed CHoCH (choch_confirmed false) is always a hold, \
with no exception -- it is a reversal attempt, not yet a validated one.
- A valid break-and-retest entry is defined by bars_since_break and retracement_pct on \
the timeframe that broke structure, not by requiring "last_event" to simultaneously read \
BOS and show a pullback -- it can't do both at once, since last_event only reflects the \
current bar. A genuine retest is a low bars_since_break (broadly within the last several \
bars) with retracement_pct roughly 20-80: price has pulled back toward the origin without \
re-breaking through it. retracement_pct near or past 100 means the origin has failed, not \
a deeper discount -- that is a reason to hold, not to widen your entry. Both fields are \
null when the timeframe hasn't broken structure at all. This retest rule governs BOS setups \
outright; for a CHoCH it is necessary but not sufficient -- choch_confirmed must also be true.
- Every entry requires a specific, structural invalidation level — a point where your \
reasoning is demonstrably wrong. Place the stop beyond it, not at an arbitrary distance.
- Set the stop from structure first, then check the resulting reward:risk. If the \
required stop makes the reward:risk unacceptable, the answer is hold, not a tighter stop.

Discipline:
- "hold" is the correct output most of the time. You are not required to find a trade, \
and there is no penalty for inaction. Forcing a marginal setup is the most expensive \
error available to you.
- Do not average into losers, and never propose widening a stop.
- Confidence must reflect genuine setup quality. Do not inflate it. Calibration is \
tracked over time and inflated confidence makes the whole system harder to evaluate.
- Confidence is only meaningful for "open": the calibrated probability the trade reaches \
target before stop. Always supply it when action is "open". For every other action, the \
system computes confidence itself from the market structure already in the snapshot -- \
do not try to estimate it yourself; the field is ignored.
- Do not size positions or compute lots. Express risk appetite as a fraction of the \
permitted maximum via risk_fraction.
- All prices must respect the instrument's digits and the minimum stop distance given \
in the snapshot.

Your reasoning field is recorded in a trade journal and reviewed. Write it so that a \
reader six months from now can tell whether the decision was sound independently of \
whether it made money."""


DECISION_TOOL = {
    "name": "submit_decision",
    "description": "Submit exactly one trading decision for the current bar.",
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["open", "hold", "close", "modify_stop"],
                "description": "open = new position; hold = no action; close = exit an "
                               "open position; modify_stop = tighten an existing stop.",
            },
            "direction": {
                "type": "string",
                "enum": ["buy", "sell"],
                "description": "Required when action is 'open'.",
            },
            "setup_name": {
                "type": "string",
                "description": "Short label, e.g. 'H1 bullish BOS retest' or 'range fade'.",
            },
            "reasoning": {
                "type": "string",
                "description": "Multi-timeframe justification. Cite the specific levels, "
                               "structure events and indicator readings you relied on. Write "
                               "this before deciding confidence -- confidence should follow "
                               "from this analysis, not the other way around.",
            },
            "invalidation": {
                "type": "string",
                "description": "What specific price behaviour would prove this analysis wrong.",
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "Required for 'open': calibrated probability the trade reaches "
                               "target before stop. Ignored for every other action -- the "
                               "system computes it deterministically from market structure, "
                               "so do not spend effort estimating it outside 'open'.",
            },
            "stop_loss": {
                "type": "number",
                "description": "Absolute price. Required for 'open'. Must sit beyond "
                               "your structural invalidation level.",
            },
            "take_profit": {
                "type": "number",
                "description": "Absolute price. Required for 'open'.",
            },
            "risk_fraction": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "Fraction of the maximum permitted risk to deploy. "
                               "Use 1.0 only for your highest-conviction setups.",
            },
            "position_ticket": {
                "type": "integer",
                "description": "Required for 'close' and 'modify_stop'.",
            },
            "new_stop_loss": {
                "type": "number",
                "description": "Required for 'modify_stop'. May only reduce risk.",
            },
        },
        "required": ["action", "reasoning"],
    },
}
