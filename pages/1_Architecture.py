"""
Architecture page for the cycle-analysis dashboard.

Static explainer of the seven-stage decision cycle (MT5 data -> features ->
Claude decision -> risk engine -> execution -> journal -> dashboard feedback)
and the invariants that gate it. No MT5 import, no journal read -- purely
descriptive, so it renders even before any cycles have been logged.
"""
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="MT5 Agent — Architecture", page_icon="🎛️", layout="wide")

html_path = Path(__file__).parent.parent / "assets" / "cycle_engine.html"
# components.html has no auto-fit-to-content option, so this is a fixed guess
# (measured against the rendered page, ~3150px at default sidebar width) with
# a little headroom. scrolling=True is the fallback if a narrower viewport
# reflows the grid taller than that.
components.html(html_path.read_text(encoding="utf-8"), height=3250, scrolling=True)
