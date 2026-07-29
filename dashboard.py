"""
Local analysis dashboard for the agent's cycle journal (logs/cycles_*.jsonl).

Read-only over the journal — does not import MT5 and never touches the
broker, so it works whether or not the terminal is running.

Run with:
    streamlit run dashboard.py
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from agent.analytics import (
    load_cycles, with_cost, kpis, confidence_buckets, rejection_reasons,
)

# Fixed categorical colors, assigned by identity (action), never re-cycled.
ACTION_COLORS = {
    "hold": "#94A3B8",        # slate — low-info, default action
    "open": "#22A67A",        # green — entering a position
    "close": "#3B82F6",       # blue — exiting a position
    "modify_stop": "#F59E0B",  # amber — adjustment
}

st.set_page_config(page_title="MT5 Agent — Cycle Analysis", layout="wide")

st.title("MT5 Agent — Cycle Analysis")
st.caption("Reads logs/cycles_*.jsonl. Refresh the page to pick up new cycles.")

with st.sidebar:
    st.header("Settings")
    st.caption("Claude Sonnet 5 pricing (intro rate through 2026-08-31; "
               "reverts to $3.00 / $15.00 after)")
    input_price = st.number_input("Input $ / MTok", value=2.00, step=0.5, format="%.2f")
    output_price = st.number_input("Output $ / MTok", value=10.00, step=0.5, format="%.2f")
    tz = st.text_input("Timezone (for hour-of-day breakdown)", value="Africa/Kigali")

raw = load_cycles("logs")

if raw.empty:
    st.info("No cycles logged yet. Run `python run.py --once` (or wait for the "
            "scheduled task) and refresh this page.")
    st.stop()

df = with_cost(raw, input_price, output_price, tz)

symbols = sorted(df["symbol"].dropna().unique().tolist())
with st.sidebar:
    selected_symbols = st.multiselect("Symbols", symbols, default=symbols)
    date_min, date_max = df["date"].min(), df["date"].max()
    date_range = st.date_input("Date range", value=(date_min, date_max),
                                min_value=date_min, max_value=date_max)

mask = df["symbol"].isin(selected_symbols)
if isinstance(date_range, tuple) and len(date_range) == 2:
    mask &= (df["date"] >= date_range[0]) & (df["date"] <= date_range[1])
df = df[mask]

if df.empty:
    st.warning("No cycles match the current filters.")
    st.stop()

# ---------- KPI row ----------
k = kpis(df)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Cycles", f"{k['total_cycles']:,}")
c2.metric("Hold rate", f"{k['hold_rate']:.1f}%")
c3.metric("Trade proposals", f"{k['trade_proposals']:,}")
c4.metric("Total cost", f"${k['total_cost_usd']:.2f}")
c5.metric("Avg latency", f"{k['avg_latency_ms']/1000:.1f}s")

if k["total_cycles"] < 30:
    st.warning(
        f"Only {k['total_cycles']} cycles logged so far. Confidence calibration "
        "and any trade-outcome stats below are not meaningful yet — treat "
        "everything as a pipeline sanity check, not a performance read, until "
        "the sample is much larger (BUILD_PLAN.md flags 30 as a floor for "
        "trades, and cycle counts should be well past that)."
    )

st.divider()

# ---------- Decisions & confidence ----------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Decisions by action")
    counts = df["action"].value_counts().rename_axis("action").reset_index(name="count")
    fig = px.bar(counts, x="action", y="count", color="action",
                 color_discrete_map=ACTION_COLORS)
    fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title="cycles")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Confidence distribution")
    buckets = confidence_buckets(df)
    fig = px.bar(buckets, x="bucket", y="count")
    fig.update_traces(marker_color="#6366F1")
    fig.update_layout(xaxis_title="confidence", yaxis_title="cycles")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------- Cost & latency over time ----------
col3, col4 = st.columns(2)

with col3:
    st.subheader("Cumulative cost")
    by_time = df.sort_values("ts").assign(cum_cost=lambda d: d["cost_usd"].cumsum())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=by_time["local_ts"], y=by_time["cum_cost"],
                              mode="lines", line=dict(color="#6366F1", width=2),
                              fill="tozeroy"))
    fig.update_layout(xaxis_title=None, yaxis_title="$ cumulative")
    st.plotly_chart(fig, use_container_width=True)

with col4:
    st.subheader("Cost by symbol")
    by_symbol = df.groupby("symbol", as_index=False)["cost_usd"].sum()
    fig = px.bar(by_symbol, x="symbol", y="cost_usd")
    fig.update_traces(marker_color="#6366F1")
    fig.update_layout(xaxis_title=None, yaxis_title="$ total")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------- Performance by hour of day ----------
st.subheader("Cycles by hour of day")
by_hour = df.groupby("hour", as_index=False).agg(
    cycles=("action", "count"),
    avg_confidence=("confidence", "mean"),
)
by_hour = by_hour.set_index("hour").reindex(range(24), fill_value=0).reset_index()
fig = px.bar(by_hour, x="hour", y="cycles")
fig.update_traces(marker_color="#94A3B8")
fig.update_layout(xaxis_title=f"hour ({tz})", yaxis_title="cycles",
                   xaxis=dict(dtick=1))
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------- Risk rejections ----------
st.subheader("Risk gate rejections")
rej = rejection_reasons(df)
if rej.empty:
    st.caption("No rejections logged yet — either no trades have been proposed, "
               "or every proposal so far has cleared the risk gates.")
else:
    fig = px.bar(rej, x="count", y="reason", orientation="h")
    fig.update_traces(marker_color="#EF4444")
    fig.update_layout(yaxis_title=None, xaxis_title="count")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------- Recent decisions ----------
st.subheader("Recent decisions")
recent = df.sort_values("ts", ascending=False).head(50)[[
    "local_ts", "symbol", "action", "direction", "confidence", "setup_name",
    "approved", "rejections", "volume", "risk_amount", "rr_ratio", "cost_usd",
    "latency_ms", "reasoning",
]].rename(columns={"local_ts": "time"})
st.dataframe(recent, use_container_width=True, height=400)
