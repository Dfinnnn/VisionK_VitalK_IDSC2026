# =============================================================================
#  app.py  —  Vision K Clinical Decision Support System
#  IDSC 2026  |  Streamlit Frontend — Top Navigation Layout
#
#  RUN:  streamlit run app.py
#  URL:  http://localhost:8501
#
#  FIVE PAGES (top nav):
#    1. Clinical Screening      — main inference page
#    2. Session History   — all scans this session
#    3. Batch Processing  — ZIP upload for multiple images
#    4. Model Transparency— performance metrics + XAI evidence
#    5. About             — project context + disclaimer
# =============================================================================

import io
import json
import os
import zipfile
import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ── Page config MUST be the first Streamlit call
st.set_page_config(
    page_title   = "Vision K — IDSC 2026",
    page_icon    = "👁",
    layout       = "wide",
    initial_sidebar_state = "collapsed",
)

# ── Load engine (cached — loads model once per session)
@st.cache_resource(show_spinner="Loading AI model…")
def load_engine():
    from engine import GlaucomaEngine
    return GlaucomaEngine()

engine = load_engine()

# ── Base paths
BASE_DIR    = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR  = BASE_DIR / "models"

# ── Session state initialisation
if "history" not in st.session_state:
    st.session_state.history = []
if "active_page" not in st.session_state:
    st.session_state.active_page = "Clinical Screening"
if "threshold_mode" not in st.session_state:
    st.session_state.threshold_mode = "Screening"
if "custom_threshold" not in st.session_state:
    st.session_state.custom_threshold = float(engine.get_default_threshold())


# =============================================================================
#  GLOBAL CSS — Dark clinical theme with top navigation
# =============================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@300;400;500&display=swap');

:root {
    --bg:           #070c14;
    --bg-card:      #0d1625;
    --bg-card2:     #111e30;
    --border:       #1a2d45;
    --border-light: #243e5e;
    --txt:          #d9e8f5;
    --txt-2:        #b0c4de;
    --txt-3:        #94a3b8;
    --accent:       #00c2ff;
    --accent-dim:   #005a78;
    --red:          #ff4560;
    --red-bg:       #1a0610;
    --amber:        #ffb020;
    --amber-bg:     #1a1006;
    --green:        #00d68f;
    --green-bg:     #04150e;
    --font-head:    'Syne', sans-serif;
    --font-body:    'Inter', sans-serif;
    --font-mono:    'IBM Plex Mono', monospace;
    --nav-h:        52px;
}

/* ── Reset */
html, body, [class*="css"] {
    font-family: var(--font-body) !important;
    background-color: var(--bg) !important;
    color: var(--txt) !important;
}
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
section[data-testid="stSidebar"] { display: none !important; }

/* ── Main container */
.main .block-container {
    padding: calc(var(--nav-h) + 1.5rem) 2rem 4rem !important;
    max-width: 1380px !important;
}

/* ── Top Navigation Bar */
.topnav {
    position: fixed;
    top: 0; left: 0; right: 0;
    height: var(--nav-h);
    background: rgba(7, 12, 20, 0.97);
    border-bottom: 1px solid var(--border);
    backdrop-filter: blur(10px);
    z-index: 9999;
    display: flex;
    align-items: center;
    padding: 0 2rem;
    gap: 0;
}
.topnav-brand {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    margin-right: 2.5rem;
    flex-shrink: 0;
}
.topnav-brand-icon {
    font-size: 1.15rem;
    line-height: 1;
}
.topnav-brand-name {
    font-family: var(--font-head);
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--txt);
    letter-spacing: -0.01em;
}
.topnav-brand-tag {
    font-size: 0.6rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--txt-3);
    margin-left: 0.2rem;
    padding: 0.12rem 0.4rem;
    border: 1px solid var(--border);
    border-radius: 3px;
}
.topnav-links {
    display: flex;
    align-items: center;
    gap: 0.15rem;
    flex: 1;
}
.nav-btn {
    background: none;
    border: none;
    padding: 0.35rem 0.85rem;
    border-radius: 6px;
    font-family: var(--font-body);
    font-size: 0.82rem;
    font-weight: 500;
    color: var(--txt-2);
    cursor: pointer;
    transition: all 0.15s;
    white-space: nowrap;
    letter-spacing: 0.01em;
}
.nav-btn:hover { background: var(--bg-card); color: var(--txt); }
.nav-btn.active {
    background: var(--bg-card2);
    color: var(--accent);
    border: 1px solid var(--border-light);
}
.topnav-right {
    margin-left: auto;
    display: flex;
    align-items: center;
    gap: 0.6rem;
    flex-shrink: 0;
}
.nav-metric-chip {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    color: var(--green);
    background: var(--green-bg);
    border: 1px solid var(--green);
    border-radius: 4px;
    padding: 0.2rem 0.5rem;
    letter-spacing: 0.04em;
}
.nav-model-chip {
    font-size: 0.68rem;
    color: var(--txt-3);
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

/* ── Page header */
.page-hero {
    margin-bottom: 1.75rem;
    padding-bottom: 1.1rem;
    border-bottom: 1px solid var(--border);
}
.page-hero h1 {
    font-family: var(--font-head);
    font-size: 1.45rem;
    font-weight: 700;
    color: var(--txt);
    margin: 0 0 0.25rem;
    letter-spacing: -0.02em;
}
.page-hero p {
    font-size: 0.8rem;
    color: var(--txt-3);
    margin: 0;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}

/* ── Cards */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 1rem;
}
.card-sm {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.75rem;
}
.card-label {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--txt-3);
    margin-bottom: 0.55rem;
    font-family: var(--font-head);
}

/* ── Risk display */
.risk-score-wrap {
    text-align: center;
    padding: 1.5rem 1rem 1rem;
}
.risk-score-num {
    font-family: var(--font-head);
    font-size: 5.5rem;
    font-weight: 700;
    line-height: 0.9;
    letter-spacing: -0.04em;
}
.risk-score-num.red   { color: var(--red); }
.risk-score-num.amber { color: var(--amber); }
.risk-score-num.green { color: var(--green); }
.risk-score-sub {
    font-size: 0.7rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--txt-3);
    margin-top: 0.5rem;
}

/* ── Tier badges */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    border-radius: 5px;
    padding: 0.3rem 0.85rem;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    font-family: var(--font-head);
}
.badge-red   { background: var(--red-bg);   color: var(--red);   border: 1px solid var(--red); }
.badge-amber { background: var(--amber-bg); color: var(--amber); border: 1px solid var(--amber); }
.badge-green { background: var(--green-bg); color: var(--green); border: 1px solid var(--green); }

/* ── Alerts */
.alert {
    border-radius: 8px;
    padding: 0.75rem 1rem;
    font-size: 0.82rem;
    margin: 0.6rem 0;
    border-left-width: 3px;
    border-left-style: solid;
}
.alert-warn  { background: #150e02; border-color: var(--amber); color: #fcd97d; }
.alert-err   { background: var(--red-bg); border-color: var(--red); color: #ff8fa1; }
.alert-info  { background: #030e1c; border-color: var(--accent); color: #78d0f0; }
.alert-ok    { background: var(--green-bg); border-color: var(--green); color: #5aefc4; }

/* ── SHAP bars */
.shap-row { display:flex; align-items:center; gap:0.7rem; margin:0.45rem 0; }
.shap-lbl { width:185px; font-size:0.78rem; color:var(--txt-2); flex-shrink:0; }
.shap-track { flex:1; background:var(--bg-card2); border-radius:3px; height:7px; overflow:hidden; }
.shap-bar-pos { height:100%; border-radius:3px; background: var(--red); }
.shap-bar-neg { height:100%; border-radius:3px; background: var(--green); }
.shap-num { width:54px; text-align:right; font-family:var(--font-mono);
            font-size:0.7rem; flex-shrink:0; }
.shap-num.pos { color: var(--red); }
.shap-num.neg { color: var(--green); }

/* ── Metric tiles */
.metric-grid { display:grid; gap:0.75rem; }
.metric-tile {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-val {
    font-family: var(--font-head);
    font-size: 1.9rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1;
    letter-spacing: -0.02em;
}
.metric-lbl {
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--txt-3);
    margin-top: 0.35rem;
}

/* ── Gauge SVG container */
.gauge-wrap { text-align: center; padding: 0.5rem 0; }

/* ── Progress bar (inference) */
.stProgress > div > div > div > div {
    background: var(--accent) !important;
}

/* ── Upload area */
div[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 1px dashed var(--border-light) !important;
    border-radius: 10px !important;
}

/* ── Dataframe */
div[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

/* ── Inputs */
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] select {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--txt) !important;
    border-radius: 6px !important;
}

/* ── Buttons */
.stButton > button {
    background: var(--accent-dim) !important;
    color: var(--accent) !important;
    border: 1px solid var(--accent) !important;
    border-radius: 6px !important;
    font-family: var(--font-head) !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.04em !important;
    padding: 0.45rem 1.2rem !important;
    transition: all 0.15s !important;
}
.stButton > button:hover {
    background: var(--accent) !important;
    color: #000 !important;
}
.stDownloadButton > button {
    background: transparent !important;
    color: var(--txt-2) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: 6px !important;
    font-family: var(--font-body) !important;
    font-size: 0.8rem !important;
}

/* ── Slider */
div[data-testid="stSlider"] > div > div > div > div {
    background: var(--accent) !important;
}

/* ── Tabs */
button[data-baseweb="tab"] {
    font-family: var(--font-head) !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    color: var(--txt-3) !important;
    text-transform: uppercase !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--accent) !important;
}

/* ── Radio */
div[data-testid="stRadio"] label {
    font-size: 0.82rem !important;
    color: var(--txt-2) !important;
}

/* ── Expander */
div[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* ── Divider */
.sep { border: none; border-top: 1px solid var(--border); margin: 1.25rem 0; }

/* ── History table row */
.hist-row {
    display: grid;
    grid-template-columns: 90px 60px 70px 110px 1fr;
    align-items: center;
    gap: 1rem;
    padding: 0.65rem 0.9rem;
    border-bottom: 1px solid var(--border);
    font-size: 0.8rem;
}
.hist-row:hover { background: var(--bg-card2); }
.hist-head { color: var(--txt-3); font-size: 0.65rem; letter-spacing: 0.1em;
             text-transform: uppercase; font-family: var(--font-head); }

/* ── Disclaimer */
.disclaimer {
    background: #05080f;
    border: 1px solid var(--border);
    border-left: 3px solid var(--red);
    border-radius: 8px;
    padding: 0.85rem 1rem;
    font-size: 0.75rem;
    color: var(--red) !important;
    line-height: 1.65;
    margin-top: 1rem;
}

/* ── Confidence meter */
.conf-meter {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.78rem;
    margin: 0.4rem 0;
}
.conf-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}

/* ── Boundary pulse */
@keyframes pulse-border {
    0%   { border-color: var(--amber); }
    50%  { border-color: transparent; }
    100% { border-color: var(--amber); }
}
.boundary-alert {
    animation: pulse-border 1.4s ease-in-out infinite;
    border: 2px solid var(--amber);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    background: var(--amber-bg);
    color: #fcd97d;
    font-size: 0.82rem;
    margin: 0.6rem 0;
}

</style>
""", unsafe_allow_html=True)


# =============================================================================
#  TOP NAVIGATION BAR
# =============================================================================

PAGES = ["Clinical Screening", "Session History", "Batch Processing",
         "Model Transparency", "About"]

# Render nav HTML
mc_data  = engine.get_model_card()
perf     = mc_data.get("test_performance", {})
auc_str  = f"AUC {perf.get('auc_roc', 0):.4f}"

st.markdown(f"""
<div class="topnav">
<div class="topnav-brand">
<span class="topnav-brand-icon">👁</span>
<span class="topnav-brand-name">Vision K</span>
<span class="topnav-brand-tag">IDSC 2026</span>
<span style='font-family:var(--font-head); font-weight:800; color:white; font-size:0.85rem; margin-left:0.5rem;'>ARE YOU GON?</span>
</div>
<div class="topnav-right">
<span style='font-family:var(--font-head); font-weight:600; color:var(--accent); font-size:0.85rem;'>by Vital Ks</span>
</div>
</div>
""", unsafe_allow_html=True)

# ── Streamlit radio for actual routing (visually hidden via CSS)
st.markdown("""
<style>
div[data-testid="stRadio"] > label { display: none; }
div[data-testid="stRadio"] > div {
    display: flex !important;
    flex-direction: row !important;
    gap: 0.4rem !important;
    flex-wrap: wrap;
    margin-bottom: 0.75rem;
    background: rgba(7,12,20,0.95);
    padding: 0.5rem 0.75rem;
    border-radius: 8px;
    border: 1px solid var(--border);
}
div[data-testid="stRadio"] > div > label {
    display: flex !important;
    align-items: center;
    gap: 0.35rem;
    padding: 0.3rem 0.85rem !important;
    border-radius: 6px !important;
    border: 1px solid transparent !important;
    cursor: pointer;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    color: #7a9bbf !important;
    transition: all 0.15s;
    background: transparent !important;
}
div[data-testid="stRadio"] > div > label:hover {
    background: #0d1625 !important;
    color: #d9e8f5 !important;
}
div[data-testid="stRadio"] > div > label[data-selected="true"],
div[data-testid="stRadio"] > div > label:has(input:checked) {
    background: #111e30 !important;
    color: #00c2ff !important;
    border-color: #243e5e !important;
}
div[data-testid="stRadio"] input { display: none !important; }
</style>
""", unsafe_allow_html=True)

page = st.radio(
    "nav",
    ["🔬 Clinical Screening", "📋 Session History", "📦 Batch Processing",
     "📊 Model Transparency", "ℹ️ About"],
    horizontal=True,
    label_visibility="collapsed",
    index=PAGES.index(st.session_state.active_page)
         if st.session_state.active_page in PAGES else 0,
)

# Map radio label back to clean page name
page_map = {
    "🔬 Clinical Screening":       "Clinical Screening",
    "📋 Session History":    "Session History",
    "📦 Batch Processing":   "Batch Processing",
    "📊 Model Transparency": "Model Transparency",
    "ℹ️ About":               "About",
}
active_page = page_map.get(page, "Clinical Screening")
st.session_state.active_page = active_page


# =============================================================================
#  THRESHOLD CONTROLS  (global, shown as compact bar below nav)
# =============================================================================

# 1. Set the default threshold logic first
if st.session_state.threshold_mode == "Screening":
    threshold = 0.25
elif st.session_state.threshold_mode == "Diagnostic":
    threshold = 0.50
else:
    threshold = st.session_state.custom_threshold

# 2. Allow the expander to overwrite 'threshold' if user interacts with it
with st.expander("⚙ Threshold Settings", expanded=False):
    tc1, tc2, tc3 = st.columns([1, 1, 2])
    with tc1:
        mode = st.radio(
            "Mode",
            ["Screening", "Diagnostic", "Custom"],
            index=["Screening", "Diagnostic", "Custom"].index(
                st.session_state.threshold_mode
            ),
            help="Screening: high sensitivity. Diagnostic: high specificity.",
        )
        st.session_state.threshold_mode = mode

    with tc2:
        if mode == "Screening":
            threshold = 0.25
        elif mode == "Diagnostic":
            threshold = 0.50
        else:
            threshold = st.slider(
                "Custom threshold",
                min_value=0.10, max_value=0.70,
                value=st.session_state.custom_threshold,
                step=0.01, format="%.2f",
                key="thresh_slider",
            )
            st.session_state.custom_threshold = threshold

        st.markdown(
            f"<div style='font-size:0.78rem;color:#7a9bbf;margin-top:0.4rem;'>"
            f"Active: <span style='color:#00c2ff;font-family:IBM Plex Mono,monospace;"
            f"font-size:0.82rem;'>{threshold:.3f}</span></div>",
            unsafe_allow_html=True
        )

    with tc3:
        st.markdown(
            "<div style='font-size:0.75rem;color:#3a5570;line-height:1.6;'>"
            "<b style='color:#7a9bbf;'>Screening Mode (0.25)</b> — "
            "Prioritises sensitivity.<br>"
            "<b style='color:#7a9bbf;'>Diagnostic Mode (0.50)</b> — "
            "Prioritises specificity.<br>"
            "</div>",
            unsafe_allow_html=True
        )

# =============================================================================
#  HELPER FUNCTIONS
# =============================================================================

def risk_badge_html(tier: str) -> str:
    cls = {
        "High Risk":       "badge-red",
        "Glaucoma Suspect":"badge-amber",
        "Low Risk":        "badge-green",
    }.get(tier, "badge-green")
    icons = {"High Risk":"⚠", "Glaucoma Suspect":"◉", "Low Risk":"✓"}
    icon = icons.get(tier, "")
    return f'<span class="badge {cls}">{icon} {tier}</span>'


def shap_bars_html(top_features: list, max_abs: float) -> str:
    html = ""
    for f in top_features:
        pct     = min(int(abs(f["shap_val"]) / max(max_abs, 1e-6) * 100), 100)
        bar_cls = "shap-bar-pos" if f["shap_val"] > 0 else "shap-bar-neg"
        num_cls = "pos" if f["shap_val"] > 0 else "neg"
        html += (
            f'<div class="shap-row">'
            f'  <div class="shap-lbl">{f["label"]}</div>'
            f'  <div class="shap-track">'
            f'    <div class="{bar_cls}" style="width:{pct}%"></div>'
            f'  </div>'
            f'  <div class="shap-num {num_cls}">{f["shap_val"]:+.3f}</div>'
            f'</div>'
        )
    return html


def gauge_svg(score: float, color: str) -> str:
    """Renders a semicircular gauge SVG for the risk score."""
    colors_map = {"red": "#ff4560", "amber": "#ffb020", "green": "#00d68f"}
    c    = colors_map.get(color, "#00d68f")
    r    = 70
    cx, cy = 90, 90
    # Arc from 180° to 0° (left to right) — half circle
    angle = 180 - (score / 100) * 180   # degrees, 180=leftmost, 0=rightmost
    import math
    rad   = math.radians(angle)
    ex    = cx + r * math.cos(math.radians(180 - (score / 100) * 180))
    ey    = cy - r * math.sin(math.radians(score / 100 * 180))
    # Large arc flag
    large = 1 if score > 50 else 0
    return f"""
    <svg viewBox="0 0 180 110" xmlns="http://www.w3.org/2000/svg" width="180" height="110">
      <path d="M {cx-r},{cy} A {r},{r} 0 0 1 {cx+r},{cy}"
            fill="none" stroke="#1a2d45" stroke-width="10" stroke-linecap="round"/>
      <path d="M {cx-r},{cy} A {r},{r} 0 0 1 {ex:.2f},{ey:.2f}"
            fill="none" stroke="{c}" stroke-width="10" stroke-linecap="round"
            opacity="0.9"/>
      <circle cx="{ex:.2f}" cy="{ey:.2f}" r="5" fill="{c}"/>
      <text x="{cx}" y="{cy+2}" text-anchor="middle" fill="{c}"
            font-family="Syne,sans-serif" font-size="28" font-weight="700"
            letter-spacing="-1">{score:.0f}</text>
      <text x="{cx}" y="{cy+18}" text-anchor="middle" fill="#3a5570"
            font-family="Syne,sans-serif" font-size="8" letter-spacing="2">/100</text>
      <text x="{cx-r-2}" y="{cy+16}" text-anchor="middle" fill="#3a5570"
            font-family="IBM Plex Mono,monospace" font-size="8">0</text>
      <text x="{cx+r+2}" y="{cy+16}" text-anchor="middle" fill="#3a5570"
            font-family="IBM Plex Mono,monospace" font-size="8">100</text>
    </svg>"""


def generate_pdf_report(patient_info: dict, result: dict, overlay_img: np.ndarray) -> bytes:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            HRFlowable, Image as RLImage, Paragraph,
            SimpleDocTemplate, Spacer, Table, TableStyle,
        )

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
                                leftMargin=20*mm, rightMargin=20*mm,
                                topMargin=20*mm, bottomMargin=20*mm)
        styles = getSampleStyleSheet()

        title_s = ParagraphStyle("T", parent=styles["Heading1"], fontSize=15,
                                 textColor=colors.HexColor("#1a365d"), spaceAfter=4)
        sub_s   = ParagraphStyle("S", parent=styles["Normal"], fontSize=9,
                                 textColor=colors.HexColor("#4a5568"), spaceAfter=12)
        sec_s   = ParagraphStyle("H", parent=styles["Heading2"], fontSize=11,
                                 textColor=colors.HexColor("#2b6cb0"),
                                 spaceBefore=10, spaceAfter=5)
        bod_s   = ParagraphStyle("B", parent=styles["Normal"], fontSize=9,
                                 textColor=colors.HexColor("#2d3748"),
                                 spaceAfter=4, leading=14)
        dis_s   = ParagraphStyle("D", parent=styles["Normal"], fontSize=8,
                                 textColor=colors.HexColor("#718096"),
                                 spaceAfter=4, leading=12)

        risk = result["risk"]
        tier_color = {
            "High Risk":        colors.HexColor("#c53030"),
            "Glaucoma Suspect": colors.HexColor("#c05621"),
            "Low Risk":         colors.HexColor("#276749"),
        }.get(risk["risk_tier"], colors.HexColor("#276749"))

        story = []

        # Header
        story.append(Paragraph("Vision K Clinical Screening Report", title_s))
        story.append(Paragraph(
            f"EfficientNet-B3 · IDSC 2026 · Generated: "
            f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} · For specialist review only",
            sub_s))
        story.append(HRFlowable(width="100%", thickness=1,
                                color=colors.HexColor("#2b6cb0"), spaceAfter=8))

        # Patient info
        story.append(Paragraph("Patient Information", sec_s))
        p_data = [
            ["Patient ID", patient_info.get("patient_id", "—"),
             "Date",       datetime.datetime.now().strftime("%Y-%m-%d")],
            ["Age",        str(patient_info.get("age", "—")),
             "Eye",        patient_info.get("eye", "—")],
            ["Referring",  patient_info.get("doctor", "—"),
             "Notes",      str(patient_info.get("notes", ""))[:60]],
        ]
        pt = Table(p_data, colWidths=[30*mm, 60*mm, 25*mm, 55*mm])
        pt.setStyle(TableStyle([
            ("FONTSIZE",         (0, 0), (-1, -1), 8),
            ("FONTNAME",         (0, 0), (0, -1),  "Helvetica-Bold"),
            ("FONTNAME",         (2, 0), (2, -1),  "Helvetica-Bold"),
            ("ROWBACKGROUNDS",   (0, 0), (-1, -1),
             [colors.HexColor("#f7fafc"), colors.HexColor("#edf2f7")]),
            ("GRID",             (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e0")),
            ("PADDING",          (0, 0), (-1, -1), 5),
        ]))
        story.append(pt)
        story.append(Spacer(1, 6*mm))

        # AI Results
        story.append(Paragraph("AI Prediction Results", sec_s))
        r_data = [
            ["Risk Score", f"{risk['risk_score']:.1f} / 100",
             "Risk Tier",  risk["risk_tier"]],
            ["GON Detected", "YES" if risk["gon_detected"] else "NO",
             "Confidence", risk["confidence"]],
            ["Cal. Prob.", f"{risk['cal_prob']*100:.1f}%",
             "Threshold",  f"{risk['threshold']:.3f}"],
        ]
        rt = Table(r_data, colWidths=[30*mm, 60*mm, 25*mm, 55*mm])
        rt.setStyle(TableStyle([
            ("FONTSIZE",       (0, 0), (-1, -1), 9),
            ("FONTNAME",       (0, 0), (0, -1),  "Helvetica-Bold"),
            ("FONTNAME",       (2, 0), (2, -1),  "Helvetica-Bold"),
            ("TEXTCOLOR",      (1, 0), (1, 0),   tier_color),
            ("FONTNAME",       (1, 0), (1, 0),   "Helvetica-Bold"),
            ("FONTSIZE",       (1, 0), (1, 0),   11),
            ("GRID",           (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e0")),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1),
             [colors.HexColor("#fff5f5"), colors.HexColor("#f7fafc"),
              colors.HexColor("#f0fff4")]),
            ("PADDING",        (0, 0), (-1, -1), 5),
        ]))
        story.append(rt)
        story.append(Spacer(1, 4*mm))

        # Recommendation
        story.append(Paragraph("Clinical Recommendation", sec_s))
        story.append(Paragraph(
            f"<b>Urgency:</b> {risk['urgency']} — {risk['timeline']}", bod_s))
        story.append(Paragraph(f"<b>Action:</b> {risk['action']}", bod_s))
        story.append(Paragraph(f"<b>Evidence:</b> {risk['evidence']}", bod_s))
        story.append(Spacer(1, 4*mm))

        # Grad-CAM
        story.append(Paragraph("Visual Explainability — Grad-CAM Heatmap", sec_s))
        story.append(Paragraph(
            "Red/yellow = high model attention · Blue = low attention. "
            "Clinically meaningful focus should be on the optic disc region.", bod_s))
        overlay_pil = Image.fromarray(overlay_img).resize((300, 300))
        ib = io.BytesIO()
        overlay_pil.save(ib, format="PNG")
        ib.seek(0)
        story.append(RLImage(ib, width=75*mm, height=75*mm))
        story.append(Spacer(1, 3*mm))

        attn = result["gradcam"]["attention"]
        story.append(Paragraph(
            f"<b>Attention:</b> {attn['note']} "
            f"(centre: {attn['center_pct']}%, border: {attn['border_pct']}%)", bod_s))
        story.append(Spacer(1, 4*mm))

        # SHAP
        if result.get("shap"):
            story.append(Paragraph("Structural Explainability — SHAP Feature Analysis", sec_s))
            for f in result["shap"]["top_features"]:
                d = "increases" if f["shap_val"] > 0 else "decreases"
                story.append(Paragraph(
                    f"• <b>{f['label']}</b> ({d} risk, SHAP={f['shap_val']:+.3f}): "
                    f"{f['sentence']}", bod_s))
            story.append(Spacer(1, 4*mm))

        # Image quality
        qc = result["quality"]
        story.append(Paragraph("Image Quality Assessment", sec_s))
        story.append(Paragraph(
            f"Status: {'PASSED' if qc['passed'] else 'FAILED'} | "
            f"Sharpness: {qc['blur_score']} | "
            f"Brightness: {qc['brightness']} | "
            f"Contrast: {qc['contrast']}", bod_s))
        story.append(Spacer(1, 5*mm))

        # Disclaimer
        story.append(HRFlowable(width="100%", thickness=0.5,
                                color=colors.HexColor("#cbd5e0"), spaceAfter=5))
        story.append(Paragraph(
            "<b>CLINICAL DISCLAIMER:</b> This report is generated by an AI screening "
            "support tool (IDSC 2026 competition project). It does NOT constitute a "
            "clinical diagnosis and must NOT be used as the sole basis for any clinical "
            "decision. All findings require review and verification by a qualified "
            "ophthalmologist. Trained on single-site data (HYGD, Israeli hospital "
            "population) — cross-site generalisation not validated. "
            "IOP, visual fields, and OCT RNFL are not incorporated.",
            dis_s))

        doc.build(story)
        return buf.getvalue()

    except ImportError:
        return None


# =============================================================================
#  PAGE 1 — Clinical Screening
# =============================================================================

def page_scan():
    st.markdown("""
    <div class="page-hero">
        <h1>Clinical Screening</h1>
        <p>Upload a retinal fundus image for AI-powered glaucoma risk assessment</p>
    </div>""", unsafe_allow_html=True)

    # ── Patient intake form
    col_form, col_upload = st.columns([1, 1.4], gap="large")

    with col_form:
        st.markdown('<div class="card-label">Patient Information</div>',
                    unsafe_allow_html=True)
        with st.container():
            patient_id = st.text_input("Patient ID", placeholder="e.g. PAT-00142",
                                       key="pid")
            c1, c2 = st.columns(2)
            with c1:
                age = st.number_input("Age", min_value=1, max_value=120,
                                      value=55, step=1, key="page")
            with c2:
                eye = st.selectbox("Eye", ["OD (Right)", "OS (Left)", "Both"])
            doctor  = st.text_input("Referring Doctor", placeholder="Dr. Smith",
                                    key="doc")
            notes   = st.text_area("Clinical Notes", placeholder="Optional notes…",
                                   height=80, key="notes")

    with col_upload:
        st.markdown('<div class="card-label">Fundus Image Upload</div>',
                    unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Drag & drop or click to upload",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        if uploaded:
            pil_img = Image.open(uploaded).convert("RGB")
            st.image(pil_img, caption="Uploaded image", width=350)

    st.markdown("<hr class='sep'>", unsafe_allow_html=True)

    # ── Run inference
    if uploaded is None:
        st.markdown(
            '<div class="alert alert-info">Upload a retinal fundus image above to begin analysis.</div>',
            unsafe_allow_html=True)
        return

    run_btn = st.button("▶  Run AI Analysis", key="run_btn")
    if not run_btn and "last_result" not in st.session_state:
        return

    if run_btn:
        pil_img = Image.open(uploaded).convert("RGB")
        with st.spinner("Running inference pipeline…"):
            result = engine.predict(pil_img, threshold=threshold, run_shap=True)
        st.session_state.last_result   = result
        st.session_state.last_pil      = pil_img
        st.session_state.last_patient  = {
            "patient_id": patient_id or "—",
            "age": age, "eye": eye,
            "doctor": doctor or "—",
            "notes": notes or "",
        }
        # ── Save to session history
        st.session_state.history.append({
            "timestamp":  datetime.datetime.now().strftime("%H:%M:%S"),
            "patient_id": patient_id or "—",
            "eye":        eye,
            "score":      result["risk"]["risk_score"],
            "tier":       result["risk"]["risk_tier"],
            "confidence": result["risk"]["confidence"],
            "result":     result,
            "patient":    st.session_state.last_patient,
        })

    # ── Use cached result
    result     = st.session_state.get("last_result")
    pil_img    = st.session_state.get("last_pil")
    pat_info   = st.session_state.get("last_patient", {})

    if result is None:
        return

    risk     = result["risk"]
    quality  = result["quality"]
    gradcam  = result["gradcam"]
    shap_res = result["shap"]

    # ── Boundary alert
    if risk["confidence"] == "Low — Boundary Case":
        st.markdown(
            f'<div class="boundary-alert">⚠ <b>Boundary Case Alert</b> — '
            f'Risk score is {risk["boundary_distance"]:.3f} from the threshold. '
            f'Clinical review strongly recommended regardless of AI outcome.</div>',
            unsafe_allow_html=True)

    # ── Attention warning
    attn = gradcam["attention"]
    if attn.get("warning"):
        st.markdown(f'<div class="alert alert-warn">🔍 {attn["warning"]}</div>',
                    unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════
    # RESULTS LAYOUT
    # ═══════════════════════════════════════════════════
    r_left, r_mid, r_right = st.columns([1, 1.1, 1.2], gap="large")

    # ── LEFT: Risk score + tier
    with r_left:
        st.markdown('<div class="card-label">Risk Assessment</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div class="card" style="text-align:center; padding: 1.5rem 1rem;">
            <div class="gauge-wrap">{gauge_svg(risk['risk_score'], risk['color'])}</div>
            <div style="margin-top:0.75rem;">{risk_badge_html(risk['risk_tier'])}</div>
            <div style="margin-top:0.9rem; font-size:0.78rem; color:var(--txt-2);
                        line-height:1.7;">
                GON Detected: <b style="color:{'var(--red)' if risk['gon_detected'] else 'var(--green)'}"
                >{'YES' if risk['gon_detected'] else 'NO'}</b><br>
                Calibrated P(GON): <span style="font-family:'IBM Plex Mono',monospace;
                    color:var(--accent);">{risk['cal_prob']*100:.1f}%</span><br>
                Threshold: <span style="font-family:'IBM Plex Mono',monospace;
                    color:var(--txt-3);">{risk['threshold']:.3f}</span>
            </div>
        </div>""", unsafe_allow_html=True)

        # Confidence
        conf_colors = {"High": "#00d68f", "Moderate": "#ffb020",
                       "Low — Boundary Case": "#ff4560"}
        cc = conf_colors.get(risk["confidence"], "#7a9bbf")
        st.markdown(f"""
        <div class="card-sm">
            <div class="card-label">Confidence</div>
            <div class="conf-meter">
                <div class="conf-dot" style="background:{cc};"></div>
                <span style="color:{cc}; font-weight:600; font-size:0.8rem;">
                    {risk['confidence']}</span>
            </div>
            <div style="font-size:0.74rem; color:var(--txt-3); margin-top:0.3rem;">
                {risk['confidence_note']}</div>
        </div>""", unsafe_allow_html=True)

        # Recommendation
        urgency_colors = {"URGENT": "#ff4560", "MONITOR": "#ffb020", "ROUTINE": "#00d68f"}
        uc = urgency_colors.get(risk["urgency"], "#7a9bbf")
        st.markdown(f"""
        <div class="card">
            <div class="card-label">Clinical Recommendation</div>
            <div style="font-size:0.72rem; font-weight:700; letter-spacing:0.08em;
                        color:{uc}; margin-bottom:0.4rem;">{risk['urgency']}</div>
            <div style="font-size:0.78rem; color:var(--txt-2); line-height:1.65;">
                <b style="color:var(--txt);">Timeline:</b> {risk['timeline']}<br>
                <b style="color:var(--txt);">Action:</b> {risk['action']}<br>
                <b style="color:var(--txt);">Evidence:</b> <span style="color:var(--txt-3);">
                    {risk['evidence']}</span>
            </div>
        </div>""", unsafe_allow_html=True)

    # ── MIDDLE: Image triplet
    with r_mid:
        st.markdown('<div class="card-label">Image Analysis</div>',
                    unsafe_allow_html=True)
        tab_orig, tab_pre, tab_cam = st.tabs(
            ["Original", "CLAHE Enhanced", "Grad-CAM"])

        with tab_orig:
            st.image(pil_img, use_container_width=True, caption="Raw upload")

        with tab_pre:
            st.image(result["preprocessed_pil"],
                     use_container_width=True,
                     caption="CLAHE + green sharpening (model input)")

        with tab_cam:
            st.image(gradcam["overlay"],
                     use_container_width=True,
                     caption="Grad-CAM · Red = high attention")
            st.markdown(
                f'<div style="font-size:0.73rem;color:var(--txt-3);margin-top:0.3rem;">'
                f'Centre focus: {attn["center_pct"]}% · '
                f'Border focus: {attn["border_pct"]}% · '
                f'{"✓ Clinically plausible" if attn["plausible"] else "⚠ Review attention"}'
                f'</div>', unsafe_allow_html=True)

        # Image quality stats
        st.markdown(f"""
        <div class="card-sm" style="margin-top:0.75rem;">
            <div class="card-label">Image Quality</div>
            <div style="display:flex; gap:1rem; font-size:0.76rem; color:var(--txt-2);">
                <span>Sharpness: <b style="color:{'var(--green)' if quality['blur_score']>100 else 'var(--amber)'};"
                    >{quality['blur_score']:.0f}</b></span>
                <span>Brightness: <b>{quality['brightness']:.0f}</b></span>
                <span>Contrast: <b>{quality['contrast']:.0f}</b></span>
                <span style="color:{'var(--green)' if quality['passed'] else 'var(--red)'};">
                    {"✓ PASS" if quality['passed'] else "✗ FAIL"}</span>
            </div>
        </div>""", unsafe_allow_html=True)

    # ── RIGHT: SHAP
    with r_right:
        st.markdown('<div class="card-label">SHAP · Feature Attribution (WHY)</div>',
                    unsafe_allow_html=True)
        if shap_res:
            tf       = shap_res["top_features"]
            max_abs  = max(abs(f["shap_val"]) for f in tf) if tf else 1.0
            shap_html = shap_bars_html(tf, max_abs)
            st.markdown(f'<div class="card">{shap_html}</div>',
                        unsafe_allow_html=True)

            st.markdown('<div class="card-label" style="margin-top:0.75rem;">'
                        'Explanation</div>',
                        unsafe_allow_html=True)
            for f in tf[:3]:
                icon = "🔴" if f["shap_val"] > 0 else "🟢"
                st.markdown(
                    f'<div class="card-sm" style="margin-bottom:0.5rem;">'
                    f'<div style="font-size:0.75rem;font-weight:600;color:var(--txt);'
                    f'margin-bottom:0.2rem;">{icon} {f["label"]}</div>'
                    f'<div style="font-size:0.74rem;color:var(--txt-2);line-height:1.5;">'
                    f'{f["sentence"]}</div></div>',
                    unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="alert alert-warn">SHAP not available. '
                'Check lgb_shap_companion.pkl exists.</div>',
                unsafe_allow_html=True)

    # ── PDF Report
    st.markdown("<hr class='sep'>", unsafe_allow_html=True)
    st.markdown('<div class="card-label">Export</div>', unsafe_allow_html=True)
    if st.button("📄  Generate PDF Report"):
        pdf_bytes = generate_pdf_report(pat_info, result, gradcam["overlay"])
        if pdf_bytes:
            fname = f"GlaucomaAI_{pat_info.get('patient_id','PATIENT')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            st.download_button("⬇ Download PDF Report", data=pdf_bytes,
                               file_name=fname, mime="application/pdf")
        else:
            st.error("reportlab not installed. Run: pip install reportlab")


# =============================================================================
#  PAGE 2 — SESSION HISTORY
# =============================================================================

def page_history():
    st.markdown("""
    <div class="page-hero">
        <h1>Session History</h1>
        <p>All scans performed in this session</p>
    </div>""", unsafe_allow_html=True)

    hist = st.session_state.history
    if not hist:
        st.markdown(
            '<div class="alert alert-info">No scans performed yet in this session. '
            'Go to Clinical Screening to analyse an image.</div>',
            unsafe_allow_html=True)
        return

    # Summary stats
    scores = [h["score"] for h in hist]
    high_n = sum(1 for h in hist if h["tier"] == "High Risk")
    susp_n = sum(1 for h in hist if h["tier"] == "Glaucoma Suspect")
    low_n  = sum(1 for h in hist if h["tier"] == "Low Risk")

    s1, s2, s3, s4 = st.columns(4)
    for col, val, lbl in [
        (s1, len(hist),      "Total Scans"),
        (s2, high_n,         "High Risk"),
        (s3, susp_n,         "Glaucoma Suspect"),
        (s4, low_n,          "Low Risk"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-tile">
                <div class="metric-val">{val}</div>
                <div class="metric-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Table header
    st.markdown("""
    <div class="hist-row hist-head">
        <span>Time</span>
        <span>Patient</span>
        <span>Eye</span>
        <span>Score</span>
        <span>Tier</span>
    </div>""", unsafe_allow_html=True)

    for i, h in enumerate(reversed(hist)):
        tier_colors = {
            "High Risk": "var(--red)",
            "Glaucoma Suspect": "var(--amber)",
            "Low Risk": "var(--green)",
        }
        tc = tier_colors.get(h["tier"], "var(--txt-2)")
        st.markdown(f"""
        <div class="hist-row">
            <span style="font-family:'IBM Plex Mono',monospace;color:var(--txt-3);">
                {h['timestamp']}</span>
            <span style="color:var(--txt);">{h['patient_id']}</span>
            <span style="color:var(--txt-2);">{h['eye']}</span>
            <span style="font-family:'IBM Plex Mono',monospace;color:{tc};
                         font-weight:600;">{h['score']:.1f}</span>
            <span>{risk_badge_html(h['tier'])}</span>
        </div>""", unsafe_allow_html=True)

    # Export history as CSV
    st.markdown("<br>", unsafe_allow_html=True)
    df = pd.DataFrame([{
        "Timestamp":  h["timestamp"],
        "Patient ID": h["patient_id"],
        "Eye":        h["eye"],
        "Risk Score": h["score"],
        "Risk Tier":  h["tier"],
        "Confidence": h["confidence"],
    } for h in hist])
    st.download_button(
        "⬇ Export History CSV",
        data=df.to_csv(index=False),
        file_name=f"glaucoma_ai_session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )

    # ── Detailed view
    st.markdown("<hr class='sep'>", unsafe_allow_html=True)
    st.markdown('<div class="card-label">Detailed View</div>', unsafe_allow_html=True)
    idx = st.selectbox(
        "Select a scan",
        options=list(range(len(hist))),
        format_func=lambda i: f"{hist[i]['timestamp']} — {hist[i]['patient_id']} — {hist[i]['tier']}",
        label_visibility="collapsed",
    )
    h = hist[idx]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="card">
            <div class="card-label">Prediction</div>
            <div style="font-size:0.82rem;color:var(--txt-2);line-height:1.8;">
                Patient: <b style="color:var(--txt);">{h['patient_id']}</b><br>
                Eye: <b style="color:var(--txt);">{h['eye']}</b><br>
                Risk Score: <b style="font-family:'IBM Plex Mono',monospace;">{h['score']:.1f}</b><br>
                Tier: {risk_badge_html(h['tier'])}<br>
                Confidence: <b style="color:var(--txt);">{h['confidence']}</b>
            </div>
        </div>""", unsafe_allow_html=True)
    with c2:
        risk = h["result"]["risk"]
        st.markdown(f"""
        <div class="card">
            <div class="card-label">Recommendation</div>
            <div style="font-size:0.78rem;color:var(--txt-2);line-height:1.8;">
                Urgency: <b style="color:var(--txt);">{risk['urgency']}</b><br>
                Timeline: <b style="color:var(--txt);">{risk['timeline']}</b><br>
                Action: <span style="color:var(--txt-2);">{risk['action']}</span>
            </div>
        </div>""", unsafe_allow_html=True)


# =============================================================================
#  PAGE 3 — BATCH PROCESSING
# =============================================================================

def page_batch():
    st.markdown("""
    <div class="page-hero">
        <h1>Batch Processing</h1>
        <p>Upload a ZIP archive of fundus images for automated screening</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="alert alert-info">
        Upload a <b>.zip</b> file containing fundus images (.jpg, .jpeg, .png).
        Images will be processed sequentially. Results are sorted by risk score
        (highest first).
    </div>""", unsafe_allow_html=True)

    zip_file = st.file_uploader("Upload ZIP archive", type=["zip"],
                                label_visibility="collapsed")

    if zip_file is None:
        return

    if st.button("▶  Run Batch Analysis"):
        with zipfile.ZipFile(zip_file, "r") as zf:
            image_names = [n for n in zf.namelist()
                           if n.lower().endswith((".jpg", ".jpeg", ".png"))
                           and not n.startswith("__MACOSX")]

        if not image_names:
            st.error("No valid image files found in ZIP.")
            return

        st.markdown(f'<div class="alert alert-info">Found {len(image_names)} images.</div>',
                    unsafe_allow_html=True)
        progress  = st.progress(0)
        status_ph = st.empty()
        rows      = []

        with zipfile.ZipFile(zip_file, "r") as zf:
            for i, name in enumerate(image_names):
                status_ph.markdown(
                    f'<div style="font-size:0.8rem;color:var(--txt-2);">'
                    f'Processing {i+1}/{len(image_names)}: {name}</div>',
                    unsafe_allow_html=True)
                try:
                    img_bytes = zf.read(name)
                    pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    result    = engine.predict(pil_img, threshold=threshold,
                                              run_shap=False)
                    risk      = result["risk"]
                    quality   = result["quality"]
                    rows.append({
                        "Image":       name,
                        "Risk Score":  risk["risk_score"],
                        "Risk Tier":   risk["risk_tier"],
                        "GON Detected":risk["gon_detected"],
                        "Cal. P(GON)": f"{risk['cal_prob']*100:.1f}%",
                        "Confidence":  risk["confidence"],
                        "Urgency":     risk["urgency"],
                        "Timeline":    risk["timeline"],
                        "QC Passed":   quality["passed"],
                        "Sharpness":   quality["blur_score"],
                    })
                except Exception as e:
                    rows.append({
                        "Image":      name,
                        "Risk Score": None,
                        "Risk Tier":  f"ERROR: {str(e)[:40]}",
                        "GON Detected": None, "Cal. P(GON)": None,
                        "Confidence": None, "Urgency": None,
                        "Timeline": None, "QC Passed": None, "Sharpness": None,
                    })
                progress.progress((i + 1) / len(image_names))

        status_ph.empty()
        progress.empty()

        df = pd.DataFrame(rows)
        df_sorted = df.dropna(subset=["Risk Score"]).sort_values(
            "Risk Score", ascending=False)

        st.markdown("<hr class='sep'>", unsafe_allow_html=True)

        # Summary tiles
        b1, b2, b3, b4 = st.columns(4)
        total   = len(rows)
        high_n  = sum(1 for r in rows if r["Risk Tier"] == "High Risk")
        susp_n  = sum(1 for r in rows if r["Risk Tier"] == "Glaucoma Suspect")
        fail_qc = sum(1 for r in rows if r["QC Passed"] is False)

        for col, val, lbl in [
            (b1, total,   "Images Processed"),
            (b2, high_n,  "High Risk"),
            (b3, susp_n,  "Glaucoma Suspect"),
            (b4, fail_qc, "QC Failed"),
        ]:
            with col:
                st.markdown(f"""
                <div class="metric-tile">
                    <div class="metric-val">{val}</div>
                    <div class="metric-lbl">{lbl}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(df_sorted, hide_index=True, use_container_width=True)

        st.download_button(
            "⬇ Download Results CSV",
            data=df_sorted.to_csv(index=False),
            file_name=f"batch_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )


# =============================================================================
#  PAGE 4 — MODEL TRANSPARENCY
# =============================================================================

def page_transparency():
    st.markdown("""
    <div class="page-hero">
        <h1>Model Transparency</h1>
        <p>Test-set performance, calibration, explainability evidence, and known limitations</p>
    </div>""", unsafe_allow_html=True)

    mc   = engine.get_model_card()
    perf = mc.get("test_performance", {})

    # ── Performance tiles
    tiles = [
        (f"{perf.get('auc_roc', 0):.4f}", "AUC-ROC"),
        (f"{perf.get('sensitivity', 0)*100:.1f}%", "Sensitivity"),
        (f"{perf.get('specificity', 0)*100:.1f}%", "Specificity"),
        (f"{perf.get('f1_score', 0)*100:.1f}%", "F1 Score"),
        (f"{perf.get('brier_score', 0):.4f}", "Brier Score"),
        (f"{perf.get('fn_count', 0)}", "Missed GON (FN)"),
    ]
    cols = st.columns(len(tiles))
    for col, (val, lbl) in zip(cols, tiles):
        with col:
            st.markdown(f"""
            <div class="metric-tile">
                <div class="metric-val">{val}</div>
                <div class="metric-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    with col_l:
        # Risk tier PPV/NPV
        st.markdown('<div class="card-label">Risk Tier Validation (Test Set)</div>',
                    unsafe_allow_html=True)
        try:
            pred_df = pd.read_csv(RESULTS_DIR / "test_predictions.csv")
            def assign_tier(s):
                return "High Risk" if s >= 60 else "Glaucoma Suspect" if s >= 30 else "Low Risk"
            pred_df["tier"] = pred_df["risk_score"].apply(assign_tier)
            rows = []
            for tier in ["High Risk", "Glaucoma Suspect", "Low Risk"]:
                sub = pred_df[pred_df["tier"] == tier]
                n   = len(sub)
                if n == 0:
                    continue
                gon = (sub["true_label"] == 1).sum()
                rows.append({
                    "Tier":    tier,
                    "N":       n,
                    "GON":     gon,
                    "Non-GON": n - gon,
                    "PPV":     f"{gon/n*100:.0f}%",
                    "NPV":     f"{(n-gon)/n*100:.0f}%",
                })
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True,
                             use_container_width=True)
        except FileNotFoundError:
            st.info("test_predictions.csv not found.")

        st.markdown("<br>", unsafe_allow_html=True)

        # Calibration info
        cal = mc.get("calibration", {})
        st.markdown(f"""
        <div class="card">
            <div class="card-label">Calibration</div>
            <div style="font-size:0.8rem;color:var(--txt-2);line-height:1.9;">
                Method: <span style="color:var(--txt);">Temperature Scaling</span><br>
                Temperature T:
                <span style="font-family:'IBM Plex Mono',monospace;color:var(--accent);">
                    {cal.get('temperature', 0):.4f}</span>
                <span style="color:var(--txt-3);font-size:0.72rem;">
                    (model was slightly underconfident)</span><br>
                Default threshold:
                <span style="font-family:'IBM Plex Mono',monospace;color:var(--accent);">
                    {cal.get('threshold', 0):.4f}</span><br>
                Target sensitivity:
                <span style="color:var(--green);">
                    {cal.get('target_sensitivity', 0)*100:.0f}%</span>
            </div>
        </div>""", unsafe_allow_html=True)

    with col_r:
        # K-Fold
        st.markdown('<div class="card-label">K-Fold Cross-Validation (5-Fold)</div>',
                    unsafe_allow_html=True)
        try:
            kf = pd.read_csv(RESULTS_DIR / "kfold_results.csv")
            st.dataframe(kf.round(4), hide_index=True, use_container_width=True)
            for c in ["auc_roc", "sensitivity", "specificity", "f1_score"]:
                if c in kf.columns:
                    st.markdown(
                        f'<div style="font-size:0.76rem;color:var(--txt-2);margin:0.2rem 0;">'
                        f'{c}: <span style="font-family:\'IBM Plex Mono\',monospace;'
                        f'color:var(--accent);">'
                        f'{kf[c].mean():.4f} ± {kf[c].std():.4f}</span></div>',
                        unsafe_allow_html=True)
        except FileNotFoundError:
            st.markdown(
                '<div class="alert alert-info">K-Fold not yet run. '
                'Set RUN_KFOLD=True in hygd_pipeline.py then rerun Section 9.</div>',
                unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # TTA
        st.markdown('<div class="card-label">TTA vs Standard Inference</div>',
                    unsafe_allow_html=True)
        try:
            tta     = pd.read_csv(RESULTS_DIR / "tta_predictions.csv")
            changed = (tta["std_prediction"] != tta["tta_prediction"]).sum()
            total   = len(tta)
            st.markdown(f"""
            <div class="card">
                <div style="font-size:0.8rem;color:var(--txt-2);line-height:1.9;">
                    Test images: <span style="color:var(--txt);">{total}</span><br>
                    Predictions changed by TTA:
                    <span style="color:var(--amber);">{changed}</span><br>
                    Stability:
                    <span style="color:var(--green);">
                        {(1-changed/total)*100:.1f}% unchanged</span>
                </div>
            </div>""", unsafe_allow_html=True)
        except FileNotFoundError:
            st.info("tta_predictions.csv not found.")

    st.markdown("<hr class='sep'>", unsafe_allow_html=True)

    # Known limitations
    st.markdown('<div class="card-label">Known Limitations</div>',
                unsafe_allow_html=True)
    for lim in mc.get("known_limitations", []):
        st.markdown(
            f'<div style="font-size:0.8rem;color:var(--txt-2);margin:0.45rem 0;">'
            f'<span style="color:var(--txt-3);">•</span> {lim}</div>',
            unsafe_allow_html=True)

    with st.expander("Full Model Card (JSON)"):
        st.json(mc)


# =============================================================================
#  PAGE 5 — ABOUT
# =============================================================================

def page_about():
    st.markdown("""
    <div class="page-hero">
        <h1>About</h1>
        <p>IDSC 2026 · Vision K Clinical Decision Support System</p>
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown("""
        <div class="card">
            <div class="card-label">Project</div>
            <div style="font-size:0.82rem;color:var(--txt-2);line-height:1.9;">
                <b style="color:var(--txt);">Competition:</b> IDSC 2026<br>
                <b style="color:var(--txt);">Task:</b>
                    Glaucomatous Optic Neuropathy (GON) detection<br>
                <b style="color:var(--txt);">Approach:</b>
                    EfficientNet-B3 + Temperature Scaling calibration<br>
                <b style="color:var(--txt);">XAI:</b>
                    Grad-CAM (spatial) + SHAP (LightGBM companion)<br>
                <b style="color:var(--txt);">Intended use:</b>
                    Screening support tool — not a clinical diagnosis
            </div>
        </div>

        <div class="card">
            <div class="card-label">Dataset — HYGD (PhysioNet)</div>
            <div style="font-size:0.82rem;color:var(--txt-2);line-height:1.9;">
                <b style="color:var(--txt);">Source:</b>
                    Hillel Yaffe Glaucoma Dataset, PhysioNet (open access)<br>
                <b style="color:var(--txt);">Images:</b>
                    741 after quality filtering (GON+: 542, GON−: 199)<br>
                <b style="color:var(--txt);">Split:</b>
                    70/15/15 stratified, seed=42<br>
                <b style="color:var(--txt);">Quality filter:</b>
                    Ophthalmologist score ≥ 3.0
            </div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="card">
            <div class="card-label">Model Architecture</div>
            <div style="font-size:0.82rem;color:var(--txt-2);line-height:1.9;">
                <b style="color:var(--txt);">Backbone:</b>
                    EfficientNet-B3 (ImageNet pretrained)<br>
                <b style="color:var(--txt);">Head:</b>
                    Dropout(0.4) → Linear(1536→256) → BN → ReLU → Linear(256→1)<br>
                <b style="color:var(--txt);">Training:</b>
                    Phase 1 frozen · Phase 2 last 3 blocks unfrozen<br>
                <b style="color:var(--txt);">Input:</b>
                    300×300 px · CLAHE + green channel sharpening
            </div>
        </div>

        <div class="card">
            <div class="card-label">Risk Tier Evidence Basis</div>
            <div style="font-size:0.82rem;line-height:1.9;">
                <b style="color:var(--red);">High Risk (score ≥ 60):</b>
                <span style="color:var(--txt-2);">
                    Ting et al. 2017 — Nature Medicine</span><br>
                <b style="color:var(--amber);">Glaucoma Suspect (30–59):</b>
                <span style="color:var(--txt-2);">
                    EGS Guidelines 2021</span><br>
                <b style="color:var(--green);">Low Risk (&lt;30):</b>
                <span style="color:var(--txt-2);">
                    NICE Guidelines</span>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
        <b>⚠ Clinical Disclaimer</b><br>
        This system is an AI screening support tool developed for the IDSC 2026 competition.
        It does <b>NOT</b> constitute a clinical diagnosis and must <b>NOT</b> be used as the
        sole basis for any clinical decision. All findings require review and verification by
        a qualified ophthalmologist. Trained exclusively on single-site data from the HYGD
        dataset (Hillel Yaffe Medical Centre, Israel) — cross-site generalisation is not
        validated. IOP measurements, visual field tests, and OCT RNFL are not incorporated.
        External validation on ACRIMA (Spanish dataset) showed domain shift (AUC ~0.52),
        confirming this tool requires local recalibration before deployment in new populations.
    </div>""", unsafe_allow_html=True)


# =============================================================================
#  PAGE ROUTER
# =============================================================================

if   active_page == "Clinical Screening":    page_scan()
elif active_page == "Session History":    page_history()
elif active_page == "Batch Processing":   page_batch()
elif active_page == "Model Transparency": page_transparency()
elif active_page == "About":              page_about()