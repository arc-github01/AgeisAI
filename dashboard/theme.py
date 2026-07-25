"""Visual language for the AEGIS SOC console.

One palette, defined once, shared by the CSS and the Plotly template so that a
CRITICAL badge and a CRITICAL bar are the same red.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# -- Palette ------------------------------------------------------------------
BG = "#0b1017"
PANEL = "#131b26"
PANEL_ALT = "#0f1620"
BORDER = "#1f2b3a"
TEXT = "#e6edf5"
MUTED = "#8b9bb0"
ACCENT = "#38bdf8"
ACCENT_DIM = "#1e6f8f"

SEVERITY_COLORS: dict[str, str] = {
    "LOW": "#2dd4bf",
    "MEDIUM": "#fbbf24",
    "HIGH": "#fb923c",
    "CRITICAL": "#ef4444",
}

ATTACK_COLORS: dict[str, str] = {
    "BENIGN": "#334155",
    "BRUTE_FORCE": "#ef4444",
    "IMPOSSIBLE_TRAVEL": "#a78bfa",
    "CREDENTIAL_STUFFING": "#f472b6",
    "LATERAL_MOVEMENT": "#fb923c",
    "DEVICE_SPOOFING": "#38bdf8",
    "LOW_AND_SLOW_EXFILTRATION": "#fbbf24",
    "INSIDER_DRIFT": "#94a3b8",
}

PLOTLY_TEMPLATE = "aegis"


def _register_plotly_template() -> None:
    if PLOTLY_TEMPLATE in pio.templates:
        return
    template = go.layout.Template()
    template.layout = go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, family="Inter, Segoe UI, sans-serif", size=12),
        title=dict(font=dict(size=14, color=TEXT), x=0.01, xanchor="left"),
        colorway=[ACCENT, "#a78bfa", "#fbbf24", "#2dd4bf", "#fb923c", "#f472b6", "#94a3b8"],
        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER,
                   tickfont=dict(color=MUTED)),
        yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER,
                   tickfont=dict(color=MUTED)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED)),
        margin=dict(l=48, r=24, t=48, b=40),
        hoverlabel=dict(bgcolor=PANEL, bordercolor=BORDER, font=dict(color=TEXT)),
    )
    pio.templates[PLOTLY_TEMPLATE] = template


_CSS = f"""
<style>
  .stApp {{ background: {BG}; }}
  section[data-testid="stSidebar"] {{
      background: {PANEL_ALT};
      border-right: 1px solid {BORDER};
  }}
  .block-container {{ padding-top: 2.2rem; max-width: 1500px; }}

  /* --- brand --- */
  .aegis-brand {{ display:flex; align-items:baseline; gap:.55rem; margin-bottom:.15rem; }}
  .aegis-brand .mark {{
      font-size:1.55rem; font-weight:800; letter-spacing:.14em; color:{TEXT};
  }}
  .aegis-brand .mark span {{ color:{ACCENT}; }}
  .aegis-brand .ver {{ font-size:.68rem; color:{MUTED}; font-family:ui-monospace,monospace; }}
  .aegis-tagline {{
      font-size:.72rem; color:{MUTED}; letter-spacing:.05em; margin-bottom:1.1rem;
  }}

  /* --- page header --- */
  .aegis-page-title {{
      font-size:1.35rem; font-weight:650; color:{TEXT}; margin:0 0 .15rem 0;
  }}
  .aegis-page-sub {{ font-size:.82rem; color:{MUTED}; margin-bottom:1.1rem; }}
  .aegis-rule {{ border-top:1px solid {BORDER}; margin:.2rem 0 1.1rem 0; }}

  /* --- KPI cards --- */
  .aegis-kpi {{
      background:{PANEL}; border:1px solid {BORDER}; border-radius:10px;
      padding:.85rem .95rem; height:100%;
  }}
  .aegis-kpi .label {{
      font-size:.68rem; text-transform:uppercase; letter-spacing:.09em; color:{MUTED};
  }}
  .aegis-kpi .value {{
      font-size:1.6rem; font-weight:700; color:{TEXT}; line-height:1.5;
      font-variant-numeric:tabular-nums;
  }}
  .aegis-kpi .value.pending {{ color:{BORDER}; }}
  .aegis-kpi .foot {{ font-size:.7rem; color:{MUTED}; }}

  /* --- panels --- */
  .aegis-panel {{
      background:{PANEL}; border:1px solid {BORDER}; border-radius:10px; padding:1rem 1.1rem;
  }}
  .aegis-panel h4 {{
      font-size:.74rem; text-transform:uppercase; letter-spacing:.09em;
      color:{MUTED}; margin:0 0 .6rem 0; font-weight:600;
  }}
  .aegis-empty {{
      background:{PANEL_ALT}; border:1px dashed {BORDER}; border-radius:10px;
      padding:1.15rem 1.25rem; color:{MUTED}; font-size:.84rem;
  }}
  .aegis-empty .head {{
      color:{TEXT}; font-weight:600; font-size:.9rem; margin-bottom:.45rem;
  }}
  .aegis-empty code {{
      background:{BG}; border:1px solid {BORDER}; border-radius:5px;
      padding:.1rem .35rem; color:{ACCENT}; font-size:.78rem;
  }}
  .aegis-empty ul {{ margin:.4rem 0 .2rem 1.1rem; padding:0; }}

  /* --- badges --- */
  .sev {{
      display:inline-block; padding:.1rem .5rem; border-radius:999px;
      font-size:.68rem; font-weight:700; letter-spacing:.06em;
  }}
  .chip {{
      display:inline-block; padding:.12rem .5rem; margin:0 .25rem .3rem 0;
      border-radius:6px; background:{PANEL_ALT}; border:1px solid {BORDER};
      color:{MUTED}; font-size:.72rem; font-family:ui-monospace,monospace;
  }}

  /* --- pipeline strip --- */
  .aegis-stage {{
      background:{PANEL_ALT}; border:1px solid {BORDER}; border-radius:8px;
      padding:.5rem .3rem; text-align:center; font-size:.7rem; color:{MUTED};
      letter-spacing:.04em;
  }}
  .aegis-stage.on {{ border-color:{ACCENT_DIM}; color:{ACCENT}; }}

  /* --- sidebar status list --- */
  .aegis-status {{
      font-size:.74rem; color:{MUTED}; display:flex; justify-content:space-between;
      padding:.16rem 0; border-bottom:1px solid rgba(31,43,58,.55);
  }}
  .aegis-status .ok {{ color:#2dd4bf; }}
  .aegis-status .pending {{ color:#475569; }}
  .aegis-status .ph {{ font-family:ui-monospace,monospace; font-size:.68rem; }}
</style>
"""


def apply() -> None:
    """Inject the stylesheet and register the Plotly template. Idempotent."""
    _register_plotly_template()
    st.markdown(_CSS, unsafe_allow_html=True)


def severity_color(severity: str) -> str:
    return SEVERITY_COLORS.get(str(severity).upper(), MUTED)


def attack_color(attack: str) -> str:
    return ATTACK_COLORS.get(str(attack).upper(), MUTED)
