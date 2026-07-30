"""
Decision layer: the only module that talks to the Claude API.

Uses forced tool use so the response is always schema-conformant structured data
rather than prose we would have to parse.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Optional

import anthropic

from . import features
from .context import DECISION_TOOL, SYSTEM_PROMPT

log = logging.getLogger(__name__)


@dataclass
class Decision:
    action: str
    reasoning: str
    confidence: float = 0.0
    direction: Optional[str] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_fraction: float = 0.5
    position_ticket: Optional[int] = None
    new_stop_loss: Optional[float] = None
    setup_name: Optional[str] = None
    invalidation: Optional[str] = None
    raw: dict = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0

    @classmethod
    def hold(cls, reason: str) -> "Decision":
        return cls(action="hold", confidence=0.0, reasoning=reason, raw={})


class DecisionEngine:
    def __init__(self, cfg, api_key: str):
        self.cfg = cfg
        self.client = anthropic.Anthropic(api_key=api_key, timeout=cfg.model.timeout_seconds)

    def verify_model(self) -> None:
        """Fail fast at startup rather than mid-session on a bad model ID."""
        try:
            m = self.client.models.retrieve(self.cfg.model.name)
            log.info("Model verified: %s (%s)", m.id, m.display_name)
        except Exception as e:
            raise RuntimeError(
                f"Model '{self.cfg.model.name}' unavailable: {e}. "
                "Check the ID against your account's available models."
            ) from e

    def decide(self, snapshot: dict, max_retries: int = 2) -> Decision:
        user_content = (
            (f"Strategy brief:\n{self.cfg.strategy_brief}\n\n" if self.cfg.strategy_brief else "")
            + "Market snapshot:\n"
            + json.dumps(snapshot, indent=2, default=str)
            + "\n\nAnalyse and submit exactly one decision."
        )

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                start = time.time()
                resp = self.client.messages.create(
                    model=self.cfg.model.name,
                    max_tokens=self.cfg.model.max_tokens,
                    system=SYSTEM_PROMPT,
                    tools=[DECISION_TOOL],
                    # Force the tool so we never receive unparseable prose.
                    tool_choice={"type": "tool", "name": "submit_decision"},
                    messages=[{"role": "user", "content": user_content}],
                )
                latency = int((time.time() - start) * 1000)

                if resp.stop_reason == "refusal":
                    return Decision.hold("Model returned a refusal stop_reason")

                block = next(
                    (b for b in resp.content
                     if b.type == "tool_use" and b.name == "submit_decision"),
                    None,
                )
                if block is None:
                    last_error = f"No tool_use block (stop_reason={resp.stop_reason})"
                    continue

                d = Decision(
                    **{k: v for k, v in block.input.items() if k in Decision.__annotations__},
                    raw=block.input,
                )
                d.input_tokens = resp.usage.input_tokens
                d.output_tokens = resp.usage.output_tokens
                d.latency_ms = latency
                d = self._sanity_check(d, snapshot)
                if d.action == "hold":
                    # Deterministic, not the model's guess -- see features.hold_confidence.
                    d.confidence = features.hold_confidence(snapshot["timeframes"])
                return d

            except anthropic.RateLimitError as e:
                last_error = f"Rate limited: {e}"
                time.sleep(2 ** attempt * 5)
            except anthropic.APIError as e:
                last_error = f"API error: {e}"
                time.sleep(2 ** attempt)
            except Exception as e:  # noqa: BLE001
                last_error = f"Unexpected: {e}"
                log.exception("Decision call failed")
                break

        # Never guess when the model is unreachable. Holding is always safe.
        log.error("Decision unavailable after retries: %s", last_error)
        return Decision.hold(f"Model unavailable, defaulting to hold: {last_error}")

    @staticmethod
    def _sanity_check(d: Decision, snapshot: dict) -> Decision:
        """
        Schema-level completeness only. Trading validity is the risk engine's job —
        this just downgrades structurally incoherent responses to 'hold'.
        """
        if d.action == "open":
            if not d.direction or d.stop_loss is None:
                return Decision.hold(
                    f"Model proposed 'open' without direction/stop_loss; downgraded. "
                    f"Original: {d.reasoning[:200]}"
                )
            if d.raw.get("confidence") is None:
                return Decision.hold(
                    f"Model proposed 'open' without confidence; downgraded. "
                    f"Original: {d.reasoning[:200]}"
                )
        if d.action in ("close", "modify_stop") and d.position_ticket is None:
            return Decision.hold("Model proposed position action without a ticket; downgraded.")
        if d.action == "modify_stop" and d.new_stop_loss is None:
            return Decision.hold("modify_stop without new_stop_loss; downgraded.")

        # Guard against the model referencing a position that isn't ours or is gone.
        if d.position_ticket is not None:
            tickets = {p["ticket"] for p in snapshot.get("open_positions", [])}
            if d.position_ticket not in tickets:
                return Decision.hold(f"Ticket {d.position_ticket} not in open positions; downgraded.")
        return d
