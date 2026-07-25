"""AEGIS analyst console - Streamlit entry point.

Run from the repository root:

    streamlit run app.py

The app is a thin router: it builds a :class:`DashboardContext`, renders the
sidebar, and delegates to the selected page. Navigation uses a plain sidebar
radio rather than ``st.navigation`` so the whole console is exercisable
headlessly by ``streamlit.testing.v1.AppTest`` in the test suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:  # allows `streamlit run app.py` from anywhere
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from dashboard import PAGES, PAGES_BY_LABEL, get_context  # noqa: E402
from dashboard import theme  # noqa: E402
from dashboard.components import brand, sidebar_status  # noqa: E402


def main() -> None:
    st.set_page_config(
        page_title="AEGIS | Behavioral Threat Detection",
        page_icon=":shield:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    theme.apply()

    ctx = get_context()

    with st.sidebar:
        brand()
        label = st.radio(
            "Navigation",
            [page.label for page in PAGES],
            label_visibility="collapsed",
            key="nav",
        )
        st.divider()
        sidebar_status(ctx)
        st.divider()
        profile = ctx.cfg.generator_profile()
        st.caption(
            f"dataset profile `{profile['name']}` · seed `{ctx.cfg['seed.master']}` · "
            f"alert budget `{ctx.cfg['alerting.budget_fraction']:.1%}`"
        )

    PAGES_BY_LABEL[label].render(ctx)


main()
