"""
Loads and summarizes cycle journal logs (logs/cycles_*.jsonl) for the analysis
dashboard. Read-only over the journal — no MT5 import, no side effects, so it
works whether or not the terminal is running.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_cycles(logs_dir: str = "logs") -> pd.DataFrame:
    """Flatten every cycle record in logs_dir into one DataFrame, oldest first."""
    rows = []
    for path in sorted(Path(logs_dir).glob("cycles_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            d = e.get("decision") or {}
            v = e.get("verdict") or {}
            c = e.get("cost") or {}
            rows.append({
                "ts": e.get("ts"),
                "symbol": e.get("symbol"),
                "action": d.get("action"),
                "direction": d.get("direction"),
                "setup_name": d.get("setup_name"),
                "confidence": d.get("confidence"),
                "reasoning": d.get("reasoning"),
                "stop_loss": d.get("stop_loss"),
                "take_profit": d.get("take_profit"),
                "approved": v.get("approved"),
                "rejections": "; ".join(v.get("rejections") or []) or None,
                "volume": v.get("volume"),
                "risk_amount": v.get("risk_amount"),
                "rr_ratio": v.get("rr_ratio"),
                "input_tokens": c.get("input_tokens", 0) or 0,
                "output_tokens": c.get("output_tokens", 0) or 0,
                "latency_ms": c.get("latency_ms", 0) or 0,
            })
    if not rows:
        return pd.DataFrame(columns=[
            "ts", "symbol", "action", "direction", "setup_name", "confidence",
            "reasoning", "stop_loss", "take_profit", "approved", "rejections",
            "volume", "risk_amount", "rr_ratio", "input_tokens", "output_tokens",
            "latency_ms",
        ])

    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    return df.sort_values("ts").reset_index(drop=True)


def with_cost(df: pd.DataFrame, input_price_per_mtok: float, output_price_per_mtok: float,
              tz: str = "Africa/Kigali") -> pd.DataFrame:
    """Attach cost_usd, local time, hour-of-day, and date columns."""
    out = df.copy()
    out["cost_usd"] = (
        out["input_tokens"] / 1_000_000 * input_price_per_mtok
        + out["output_tokens"] / 1_000_000 * output_price_per_mtok
    )
    local = out["ts"].dt.tz_convert(tz)
    out["local_ts"] = local
    out["hour"] = local.dt.hour
    out["date"] = local.dt.date
    return out


def kpis(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "total_cycles": 0, "hold_rate": 0.0, "total_cost_usd": 0.0,
            "avg_latency_ms": 0.0, "avg_confidence": 0.0, "trade_proposals": 0,
        }
    total = len(df)
    holds = (df["action"] == "hold").sum()
    proposals = total - holds
    return {
        "total_cycles": total,
        "hold_rate": holds / total * 100,
        "total_cost_usd": df["cost_usd"].sum() if "cost_usd" in df else 0.0,
        "avg_latency_ms": df["latency_ms"].mean(),
        "avg_confidence": df["confidence"].mean(),
        "trade_proposals": proposals,
    }


def confidence_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Bucket decisions by confidence into 0.1-wide bands, count per bucket."""
    if df.empty:
        return pd.DataFrame(columns=["bucket", "count"])
    edges = [i / 10 for i in range(11)]
    labels = [f"{edges[i]:.1f}-{edges[i+1]:.1f}" for i in range(10)]
    cats = pd.cut(df["confidence"], bins=edges, labels=labels, include_lowest=True)
    return cats.value_counts().reindex(labels).rename_axis("bucket").reset_index(name="count")


def rejection_reasons(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "rejections" not in df:
        return pd.DataFrame(columns=["reason", "count"])
    exploded = df["rejections"].dropna().str.split("; ").explode()
    if exploded.empty:
        return pd.DataFrame(columns=["reason", "count"])
    return exploded.value_counts().rename_axis("reason").reset_index(name="count")
