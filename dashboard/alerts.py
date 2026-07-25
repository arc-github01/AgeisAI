"""Alert Queue and Alert Detail: the analyst's primary workspace."""

from __future__ import annotations

import streamlit as st

from src.schema import ATTACK_CLASSES, Severity

from . import charts
from .components import (
    KPI,
    awaiting_data,
    kpi_row,
    page_header,
    panel_title,
    severity_badge,
)
from .state import DashboardContext

TITLE = "Alert Queue"
SUBTITLE = "Risk-ranked alerts within the analyst triage budget"

QUEUE_COLUMNS = [
    "timestamp",
    "entity_id",
    "entity_type",
    "attack_type",
    "risk_score",
    "severity",
    "top_reason",
]


def render(ctx: DashboardContext) -> None:
    page_header(TITLE, SUBTITLE)
    alerts = ctx.alerts()

    with st.container():
        col_sev, col_type, col_risk, col_entity = st.columns([1.1, 1.4, 1.2, 1.1])
        severities = col_sev.multiselect(
            "Severity", [s.value for s in Severity], default=[], key="flt_severity"
        )
        attack_types = col_type.multiselect(
            "Attack type", list(ATTACK_CLASSES), default=[], key="flt_attack"
        )
        min_risk = col_risk.slider("Minimum risk", 0, 100, 0, step=5, key="flt_risk")
        entity_query = col_entity.text_input("Entity contains", "", key="flt_entity")

    if awaiting_data(
        ctx,
        "alerts",
        note="Filters above are live and will apply to the alert store the moment "
        "the risk engine produces it.",
    ):
        _empty_detail_layout(ctx)
        return

    view = alerts
    if severities:
        view = view[view["severity"].isin(severities)]
    if attack_types:
        view = view[view["attack_type"].isin(attack_types)]
    if min_risk:
        view = view[view["risk_score"] >= min_risk]
    if entity_query:
        view = view[view["entity_id"].astype(str).str.contains(entity_query, case=False)]
    view = view.sort_values("risk_score", ascending=False)

    kpi_row(
        [
            KPI("Matching alerts", len(view)),
            KPI("Critical", int((view["severity"] == "CRITICAL").sum())),
            KPI("Peak risk", f"{view['risk_score'].max():.0f}" if len(view) else "--"),
            KPI("Distinct entities", int(view["entity_id"].nunique()) if len(view) else 0),
        ]
    )
    st.write("")

    columns = [c for c in QUEUE_COLUMNS if c in view.columns]
    page_size = int(ctx.cfg.get("dashboard.alerts_page_size", 50))
    st.dataframe(
        view[columns].head(page_size),
        width="stretch",
        hide_index=True,
        height=440,
    )

    st.write("")
    panel_title("Alert detail")
    if view.empty:
        st.caption("No alert matches the current filters.")
        return

    labels = [
        f"{row.risk_score:.0f} | {row.entity_id} | {row.attack_type}"
        for row in view.head(page_size).itertuples()
    ]
    choice = st.selectbox("Selected alert", labels, index=0, key="alert_choice")
    _render_detail(ctx, view.head(page_size).iloc[labels.index(choice)])


def _render_detail(ctx: DashboardContext, alert) -> None:
    left, right = st.columns([1, 1.3])
    with left:
        st.markdown(
            f"**Risk {alert.get('risk_score', 0):.0f}/100** &nbsp; "
            f"{severity_badge(alert.get('severity', 'LOW'))}",
            unsafe_allow_html=True,
        )
        st.markdown(f"**Suspected attack:** `{alert.get('attack_type', 'unknown')}`")
        st.markdown(f"**Entity:** `{alert.get('entity_id', '-')}`")
        st.markdown(f"**Observed:** {alert.get('timestamp', '-')}")
        narrative = alert.get("narrative")
        if narrative:
            st.markdown("**Threat narrative**")
            st.info(narrative)
    with right:
        panel_title("Contributing factors")
        st.plotly_chart(
            charts.contribution_bars(alert.get("contributions")), width="stretch"
        )


def _empty_detail_layout(ctx: DashboardContext) -> None:
    """Keep the final layout visible while the alert store does not exist."""
    kpi_row(
        [
            KPI("Matching alerts", None),
            KPI("Critical", None),
            KPI("Peak risk", None),
            KPI("Distinct entities", None),
        ]
    )
    st.write("")
    panel_title("Alert detail")
    left, right = st.columns([1, 1.3])
    with left:
        st.caption(
            "Selecting an alert will show its risk score, severity, suspected attack "
            "category, entity, and the deterministic evidence behind the decision."
        )
    with right:
        st.plotly_chart(charts.contribution_bars(None), width="stretch")
