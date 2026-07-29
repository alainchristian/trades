"""Configuration: YAML for behaviour, environment variables for secrets."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class MT5Config:
    terminal_path: str | None = None
    login: int | None = None
    password: str | None = None
    server: str | None = None


@dataclass
class RiskConfig:
    max_risk_per_trade_pct: float = 0.5
    min_risk_per_trade_pct: float = 0.1
    max_total_risk_pct: float = 2.0
    max_daily_loss_pct: float = 3.0
    max_open_positions: int = 2
    allow_multiple_per_symbol: bool = False
    min_rr: float = 1.5
    require_take_profit: bool = True
    max_spread_points: float = 30
    min_stop_spread_multiple: float = 3.0
    cooldown_after_losses: int = 3


@dataclass
class SessionConfig:
    timezone: str = "UTC"
    trading_windows: list[list[str]] = field(default_factory=lambda: [["07:00", "20:00"]])
    trading_days: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    avoid_minutes_around_rollover: int = 15


@dataclass
class ModelConfig:
    # Verified against the Claude model catalogue. Alternatives:
    #   claude-opus-5     — stronger reasoning, higher cost/latency
    #   claude-sonnet-4-6 — previous generation, cheapest capable option
    name: str = "claude-sonnet-5"
    max_tokens: int = 2000
    timeout_seconds: int = 90


@dataclass
class Config:
    symbols: list[str] = field(default_factory=lambda: ["EURUSD"])
    decision_timeframe: str = "M15"
    context_timeframes: list[str] = field(default_factory=lambda: ["H4", "H1", "M15"])
    bars_per_timeframe: int = 250
    magic_number: int = 770125
    allow_live_account: bool = False
    dry_run: bool = True
    kill_switch_file: str = "KILL"
    strategy_brief: str = ""
    mt5: MT5Config = field(default_factory=MT5Config)
    risk: RiskConfig = field(default_factory=RiskConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    @classmethod
    def load(cls, path: str = "config.yaml") -> "Config":
        raw = yaml.safe_load(Path(path).read_text()) or {}

        cfg = cls(
            symbols=raw.get("symbols", ["EURUSD"]),
            decision_timeframe=raw.get("decision_timeframe", "M15"),
            context_timeframes=raw.get("context_timeframes", ["H4", "H1", "M15"]),
            bars_per_timeframe=raw.get("bars_per_timeframe", 250),
            magic_number=raw.get("magic_number", 770125),
            allow_live_account=raw.get("allow_live_account", False),
            dry_run=raw.get("dry_run", True),
            kill_switch_file=raw.get("kill_switch_file", "KILL"),
            strategy_brief=raw.get("strategy_brief", ""),
            mt5=MT5Config(**raw.get("mt5", {})),
            risk=RiskConfig(**raw.get("risk", {})),
            session=SessionConfig(**raw.get("session", {})),
            model=ModelConfig(**raw.get("model", {})),
        )

        # Secrets come from the environment, never the YAML file.
        cfg.mt5.password = os.getenv("MT5_PASSWORD") or cfg.mt5.password
        if os.getenv("MT5_LOGIN"):
            cfg.mt5.login = int(os.environ["MT5_LOGIN"])
        cfg.mt5.server = os.getenv("MT5_SERVER") or cfg.mt5.server

        cfg.validate()
        return cfg

    def validate(self) -> None:
        r = self.risk
        problems = []
        if r.max_risk_per_trade_pct > 2:
            problems.append("max_risk_per_trade_pct above 2% is not a risk policy, it's a coin flip")
        if r.max_total_risk_pct < r.max_risk_per_trade_pct:
            problems.append("max_total_risk_pct is below max_risk_per_trade_pct")
        if r.max_daily_loss_pct <= r.max_risk_per_trade_pct:
            problems.append("max_daily_loss_pct must exceed max_risk_per_trade_pct")
        if r.min_rr < 1.0:
            problems.append("min_rr below 1.0 requires a very high win rate to break even")
        if self.decision_timeframe not in self.context_timeframes:
            problems.append("decision_timeframe must be included in context_timeframes")
        if problems:
            raise ValueError("Config validation failed:\n  - " + "\n  - ".join(problems))
