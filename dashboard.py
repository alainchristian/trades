"""
Local analysis dashboard for the agent's cycle journal (logs/cycles_*.jsonl).

Read-only over the journal — does not import MT5 and never touches the
broker, so it works whether or not the terminal is running.

Run with:
    streamlit run dashboard.py
"""
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from agent.analytics import (
    load_cycles, with_cost, kpis, confidence_buckets, rejection_reasons,
)

# ---------------------------------------------------------------------------
# Palette. Matches assets/mt5_agent_dashboard_concept.html's :root variables.
# Fixed categorical order (never re-cycled) for action colors, one sequential
# ramp for magnitude, a reserved status pair for approved/rejected.
# ---------------------------------------------------------------------------
BG = "#080D18"
PANEL = "#0F1729"
PANEL_2 = "#121C33"
BORDER = "rgba(148,163,184,0.10)"
BORDER_GLOW = "rgba(34,211,238,0.35)"
GRID = "rgba(148,163,184,0.10)"
AXIS_LINE = "rgba(148,163,184,0.25)"

CYAN = "#22D3EE"
BLUE = "#3B82F6"
GREEN = "#22C55E"
AMBER = "#F59E0B"
RED = "#F87171"
SLATE = "#8CA0BE"
TEXT = "#E7ECF5"
TEXT_DIM = "#9FB0C9"

# Cyan sequential ramp, light -> dark, for the confidence-bucket ordinal
# encoding. Placeholder pending the chart-restyle pass.
BLUE_RAMP_10 = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
                "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

# Assigned by identity, fixed order, never re-cycled. "hold" is deliberately a
# neutral (no action taken), not a hue -- matches the concept's chip colors.
ACTION_COLORS = {
    "hold": SLATE,
    "open": GREEN,
    "close": BLUE,
    "modify_stop": AMBER,
}

FONT_STACK = "ui-sans-serif, -apple-system, 'Segoe UI', Roboto, Arial, sans-serif"
MONO_STACK = "ui-monospace, 'SF Mono', 'Cascadia Code', 'JetBrains Mono', Consolas, monospace"

st.set_page_config(page_title="MT5 Agent — Cycle Analysis", page_icon="📈", layout="wide")

# ---------------------------------------------------------------------------
# Chrome: dark theme matching assets/mt5_agent_dashboard_concept.html.
# .streamlit/config.toml carries the base Streamlit theme (sidebar, inputs,
# dataframe) as far as its theming API reaches; this block covers what it
# doesn't -- card borders, the custom header/pipeline components, and hiding
# Streamlit's own dev chrome on an internal tool.
# ---------------------------------------------------------------------------
st.markdown(f"""
<style>
html, body, [class*="css"] {{
    font-family: {FONT_STACK};
}}
[data-testid="stAppViewContainer"] {{
    background:
        radial-gradient(ellipse 900px 500px at 15% -10%, rgba(34,211,238,0.10), transparent 60%),
        radial-gradient(ellipse 700px 500px at 100% 0%, rgba(59,130,246,0.10), transparent 60%),
        {BG};
}}
#MainMenu, footer, header [data-testid="stToolbar"], [data-testid="stDecoration"] {{
    visibility: hidden;
    height: 0;
}}
[data-testid="stMetric"] {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 16px;
    padding: 16px 18px 12px 18px;
}}
[data-testid="stMetricLabel"] {{
    color: {TEXT_DIM};
    font-size: 0.78rem;
}}
[data-testid="stMetricValue"] {{
    color: {TEXT};
    font-weight: 700;
    font-family: {MONO_STACK};
}}
h1 {{
    font-weight: 650;
    letter-spacing: -0.01em;
    color: {TEXT};
}}
h2, h3 {{
    font-weight: 600;
    color: {TEXT};
}}
[data-testid="stCaptionContainer"] {{
    color: {TEXT_DIM};
}}
[data-testid="stSidebar"] {{
    background: {PANEL};
    border-right: 1px solid {BORDER};
}}
hr {{
    border-color: {BORDER} !important;
}}

/* ---------- Custom header ---------- */
.dash-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 18px;
    flex-wrap: wrap;
    gap: 12px;
}}
.dash-brand {{
    display: flex;
    align-items: center;
    gap: 12px;
}}
.dash-brand-mark {{
    width: 38px; height: 38px; border-radius: 10px; flex: 0 0 auto;
    background: linear-gradient(135deg, rgba(34,211,238,0.25), rgba(59,130,246,0.15));
    border: 1px solid {BORDER_GLOW};
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 0 18px rgba(34,211,238,0.25);
}}
.dash-brand h1 {{
    font-size: 19px; font-weight: 600; margin: 0; letter-spacing: 0.2px;
}}
.dash-brand h1 span {{
    color: {TEXT_DIM}; font-weight: 400;
}}
.dash-brand-caption {{
    display: block; font-size: 11.5px; color: {TEXT_DIM}; font-weight: 400; margin-top: 2px;
}}
.status-pill {{
    display: flex; align-items: center; gap: 8px;
    padding: 7px 14px 7px 10px; border-radius: 999px;
    font-size: 12.5px; font-weight: 500;
    font-family: {MONO_STACK};
}}
.status-pill.live {{
    background: rgba(34,197,94,0.08); border: 1px solid rgba(34,197,94,0.35); color: #86EFAC;
}}
.status-pill.idle {{
    background: rgba(148,163,184,0.06); border: 1px solid {BORDER}; color: {TEXT_DIM};
}}
.status-pill .dot {{
    width: 7px; height: 7px; border-radius: 50%;
}}
.status-pill.live .dot {{
    background: {GREEN};
    animation: dash-pulse 2s infinite;
}}
.status-pill.idle .dot {{
    background: {SLATE};
}}
@keyframes dash-pulse {{
    0% {{ box-shadow: 0 0 0 0 rgba(34,197,94,0.55); }}
    70% {{ box-shadow: 0 0 0 7px rgba(34,197,94,0); }}
    100% {{ box-shadow: 0 0 0 0 rgba(34,197,94,0); }}
}}
@media (prefers-reduced-motion: reduce) {{
    .status-pill.live .dot {{ animation: none; }}
}}

/* ---------- Pipeline strip ---------- */
.pipeline {{
    display: grid; grid-template-columns: repeat(7, 1fr); gap: 0;
    background: {PANEL}; border: 1px solid {BORDER}; border-radius: 14px;
    padding: 16px 6px; margin-bottom: 18px; position: relative; overflow: hidden;
}}
.pipeline::before {{
    content: ""; position: absolute; top: 29px; left: 6%; right: 6%; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(148,163,184,0.25), transparent);
    z-index: 0;
}}
.p-step {{
    display: flex; flex-direction: column; align-items: center; gap: 7px;
    position: relative; z-index: 1; padding: 0 6px;
}}
.p-num {{
    width: 30px; height: 30px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 12px; font-weight: 600; font-family: {MONO_STACK};
    background: {PANEL_2}; border: 1px solid {BORDER_GLOW}; color: {CYAN};
}}
.p-step.active .p-num {{
    background: rgba(34,211,238,0.18); box-shadow: 0 0 12px rgba(34,211,238,0.4);
}}
.p-label {{
    font-size: 11px; color: {TEXT_DIM}; text-align: center; line-height: 1.3; max-width: 110px;
}}
@media (max-width: 1100px) {{
    .pipeline {{ grid-template-columns: repeat(4, 1fr); row-gap: 16px; }}
    .pipeline::before {{ display: none; }}
}}
</style>
""", unsafe_allow_html=True)


def _themed(fig: go.Figure, *, showlegend: bool = False) -> go.Figure:
    """House chart style: recessive hairline grid, muted axis ink, system font,
    thin bars with capped width, no dev toolbar. Applied to every chart so the
    dashboard reads as one system rather than a pile of default Plotly output."""
    fig.update_layout(
        template="plotly_dark",
        font=dict(family=FONT_STACK, color=TEXT_DIM, size=12.5),
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        margin=dict(l=8, r=8, t=8, b=8),
        showlegend=showlegend,
        legend=dict(font=dict(color=TEXT_DIM), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=False, showline=True, linecolor=AXIS_LINE,
                    tickfont=dict(color=TEXT_DIM)),
        yaxis=dict(showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False,
                    showline=False, tickfont=dict(color=TEXT_DIM)),
        bargap=0.35,
        hoverlabel=dict(bgcolor=PANEL_2, bordercolor=BORDER_GLOW,
                         font=dict(family=FONT_STACK, color=TEXT, size=12.5)),
    )
    fig.update_traces(marker_cornerradius=4, selector=dict(type="bar"))
    return fig


CHART_CONFIG = {"displayModeBar": False}


def chart(fig: go.Figure, **kw):
    st.plotly_chart(fig, width="stretch", config=CHART_CONFIG, **kw)


# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    st.caption("Claude Sonnet 5 pricing (intro rate through 2026-08-31; "
               "reverts to $3.00 / $15.00 after)")
    input_price = st.number_input("Input $ / MTok", value=2.00, step=0.5, format="%.2f")
    output_price = st.number_input("Output $ / MTok", value=10.00, step=0.5, format="%.2f")
    tz = st.text_input("Timezone (for hour-of-day breakdown)", value="Africa/Kigali")

raw = load_cycles("logs")
has_data = not raw.empty

if has_data:
    df_all = with_cost(raw, input_price, output_price, tz)
    last_cycle = df_all["local_ts"].max()
    span_days = (df_all["date"].max() - df_all["date"].min()).days + 1
    symbols_all = sorted(df_all["symbol"].dropna().unique().tolist())

    # "Live" is derived from the real last-cycle timestamp against the ~15-min
    # scheduled cadence -- not a countdown, since we have no scheduler hook to
    # compute next-run time yet.
    minutes_since = (pd.Timestamp.now(tz=tz) - last_cycle).total_seconds() / 60.0
    is_live = minutes_since <= 20
    pill_class = "live" if is_live else "idle"
    pill_label = f"{'Live' if is_live else 'Idle'} · last cycle {last_cycle.strftime('%H:%M')}"
    header_caption = (
        f"Reads logs/cycles_*.jsonl · {', '.join(symbols_all)} · "
        f"{span_days} day{'s' if span_days != 1 else ''} coverage"
    )
else:
    pill_class = "idle"
    pill_label = "No cycles logged yet"
    header_caption = "Reads logs/cycles_*.jsonl · waiting for the first cycle"

BRAND_MARK_SVG = (
    '<svg viewBox="0 0 24 24" fill="none">'
    '<path d="M12 2 L4 6 V12 C4 17 7.5 20.5 12 22 C16.5 20.5 20 17 20 12 V6 Z" '
    'stroke="#67E8F9" stroke-width="1.4"/>'
    '<path d="M9 12 L11 14 L15.5 9" stroke="#67E8F9" stroke-width="1.4" '
    'stroke-linecap="round" stroke-linejoin="round"/></svg>'
)

st.markdown(f"""
<div class="dash-header">
  <div class="dash-brand">
    <div class="dash-brand-mark">{BRAND_MARK_SVG}</div>
    <div>
      <h1>MT5 Agent <span>/ Cycle Analysis</span></h1>
      <span class="dash-brand-caption">{header_caption}</span>
    </div>
  </div>
  <div class="status-pill {pill_class}"><span class="dot"></span>{pill_label}</div>
</div>
""", unsafe_allow_html=True)

# Mirrors the real 7-stage cycle from the architecture page (pages/1_Architecture.py).
# Stages 1-3 run every cycle regardless of outcome, so they're marked active;
# 4-7 depend on what the model proposes. Placeholder until real per-stage
# status is wired in from the journal.
PIPELINE_STEPS = [
    ("1", "Market data", "MT5 bars, forming bar dropped", True),
    ("2", "Feature build", "multi-timeframe snapshot", True),
    ("3", "Claude decision", "action + confidence", True),
    ("4", "Risk gate", "sizing, RR, limits", False),
    ("5", "Execution", "MT5 order", False),
    ("6", "Journal", "cycles_*.jsonl", False),
    ("7", "Feedback", "this dashboard", False),
]
pipeline_html = '<div class="pipeline">' + "".join(
    f'<div class="p-step{" active" if active else ""}">'
    f'<div class="p-num">{n}</div>'
    f'<div class="p-label">{label}<br>{sub}</div>'
    f'</div>'
    for n, label, sub, active in PIPELINE_STEPS
) + '</div>'
st.markdown(pipeline_html, unsafe_allow_html=True)

if not has_data:
    st.info("No cycles logged yet. Run `python run.py --once` (or wait for the "
            "scheduled task) and refresh this page.")
    st.stop()

with st.sidebar:
    st.divider()
    selected_symbols = st.multiselect("Symbols", symbols_all, default=symbols_all)
    date_min, date_max = df_all["date"].min(), df_all["date"].max()
    date_range = st.date_input("Date range", value=(date_min, date_max),
                                min_value=date_min, max_value=date_max)

mask = df_all["symbol"].isin(selected_symbols)
if isinstance(date_range, tuple) and len(date_range) == 2:
    mask &= (df_all["date"] >= date_range[0]) & (df_all["date"] <= date_range[1])
df = df_all[mask]

if df.empty:
    st.warning("No cycles match the current filters.")
    st.stop()

# ---------- KPI rows ----------
k = kpis(df)

r1c1, r1c2, r1c3, r1c4 = st.columns(4)
r1c1.metric("Cycles", f"{k['total_cycles']:,}")
r1c2.metric("Trade proposals", f"{k['trade_proposals']:,}")
r1c3.metric("Approved", f"{k['approved_count']:,}")
r1c4.metric("Rejected", f"{k['rejected_count']:,}")

r2c1, r2c2, r2c3, r2c4 = st.columns(4)
r2c1.metric("Hold rate", f"{k['hold_rate']:.1f}%")
r2c2.metric("Avg confidence", f"{k['avg_confidence']:.2f}")
r2c3.metric("Total cost", f"${k['total_cost_usd']:.2f}")
r2c4.metric("Avg latency", f"{k['avg_latency_ms']/1000:.1f}s")

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
    fig.update_layout(xaxis_title=None, yaxis_title="cycles")
    chart(_themed(fig))

with col2:
    st.subheader("Confidence distribution")
    buckets = confidence_buckets(df)
    fig = px.bar(buckets, x="bucket", y="count", color="bucket",
                 color_discrete_sequence=BLUE_RAMP_10)
    fig.update_layout(xaxis_title="confidence", yaxis_title="cycles")
    chart(_themed(fig))

st.divider()

# ---------- Cost & latency over time ----------
col3, col4 = st.columns(2)

with col3:
    st.subheader("Cumulative cost")
    by_time = df.sort_values("ts").assign(cum_cost=lambda d: d["cost_usd"].cumsum())
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=by_time["local_ts"], y=by_time["cum_cost"], mode="lines",
        line=dict(color=BLUE, width=2), fill="tozeroy",
        fillcolor="rgba(59,130,246,0.10)",
    ))
    fig.update_layout(xaxis_title=None, yaxis_title="$ cumulative")
    chart(_themed(fig))

with col4:
    st.subheader("Cost by symbol")
    by_symbol = df.groupby("symbol", as_index=False)["cost_usd"].sum()
    fig = px.bar(by_symbol, x="symbol", y="cost_usd")
    fig.update_traces(marker_color=BLUE)
    fig.update_layout(xaxis_title=None, yaxis_title="$ total")
    chart(_themed(fig))

st.divider()

# ---------- Performance by hour of day ----------
st.subheader("Cycles by hour of day")
by_hour = df.groupby("hour", as_index=False).agg(
    cycles=("action", "count"),
    avg_confidence=("confidence", "mean"),
)
by_hour = by_hour.set_index("hour").reindex(range(24), fill_value=0).reset_index()
fig = px.bar(by_hour, x="hour", y="cycles")
fig.update_traces(marker_color=SLATE)
fig.update_layout(xaxis_title=f"hour ({tz})", yaxis_title="cycles",
                   xaxis=dict(dtick=1))
chart(_themed(fig))

st.divider()

# ---------- Risk rejections ----------
st.subheader("Risk gate rejections")
rej = rejection_reasons(df)
if rej.empty:
    st.caption("No rejections logged yet — either no trades have been proposed, "
               "or every proposal so far has cleared the risk gates.")
else:
    fig = px.bar(rej, x="count", y="reason", orientation="h")
    fig.update_traces(marker_color=RED)
    fig.update_layout(yaxis_title=None, xaxis_title="count")
    chart(_themed(fig))

st.divider()

# ---------- Recent decisions ----------
st.subheader("Recent decisions")
recent = df.sort_values("ts", ascending=False).head(50)[[
    "local_ts", "symbol", "action", "direction", "confidence", "setup_name",
    "approved", "rejections", "volume", "risk_amount", "rr_ratio", "cost_usd",
    "latency_ms", "reasoning",
]].rename(columns={"local_ts": "time"})
recent = recent.assign(
    status=recent["approved"].map({True: "✅ Approved", False: "❌ Rejected"}).fillna("—"),
).drop(columns=["approved"])
recent = recent[[
    "time", "symbol", "action", "direction", "confidence", "setup_name", "status",
    "rejections", "volume", "risk_amount", "rr_ratio", "cost_usd", "latency_ms",
    "reasoning",
]]

st.dataframe(
    recent,
    width="stretch",
    height=400,
    column_config={
        "action": st.column_config.TextColumn("action"),
        "confidence": st.column_config.ProgressColumn(
            "confidence", min_value=0.0, max_value=1.0, format="%.2f"),
        "cost_usd": st.column_config.NumberColumn("cost ($)", format="$%.4f"),
        "latency_ms": st.column_config.NumberColumn("latency (ms)", format="%d"),
    },
)
