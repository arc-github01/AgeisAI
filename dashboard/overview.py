"""SOC Overview: the situational-awareness landing page."""

from __future__ import annotations

import streamlit as st

from . import charts
from .components import KPI, awaiting_data, kpi_row, page_header, panel_title
from .state import DashboardContext

TITLE = "SOC Overview"
SUBTITLE = "Environment-wide behavioral risk posture"


def render(ctx: DashboardContext) -> None:
    page_header(TITLE, SUBTITLE)

    events = ctx.events()
    alerts = ctx.alerts()

    total_events = len(events) if events is not None else None
    total_entities = len(ctx.entity_ids()) or None
    total_alerts = len(alerts) if alerts is not None else None
    critical = (
        int((alerts["severity"] == "CRITICAL").sum())
        if alerts is not None and "severity" in alerts.columns
        else None
    )
    alert_rate = (
        f"{len(alerts) / len(events):.2%}"
        if alerts is not None and events is not None and len(events)
        else None
    )
    budget = ctx.cfg["alerting.budget_fraction"]

    kpi_row(
        [
            KPI("Events ingested", total_events, foot="labelled synthetic corpus"),
            KPI("Entities monitored", total_entities, foot="users / services / devices"),
            KPI("Alerts raised", total_alerts, foot=f"top {budget:.0%} analyst budget"),
            KPI("Critical", critical, foot="risk 81-100"),
            KPI("Alert rate", alert_rate, foot="alerts / events"),
        ]
    )
    st.write("")

    if awaiting_data(
        ctx,
        "events",
        "alerts",
        note="Layout, charts and KPI wiring below are live; only the data source is pending.",
    ):
        st.write("")

    left, right = st.columns([1, 1])
    with left:
        panel_title("Threat distribution")
        counts = None
        if alerts is not None and "attack_type" in alerts.columns:
            counts = alerts["attack_type"].value_counts().to_dict()
        st.plotly_chart(charts.attack_distribution(counts), width="stretch")
    with right:
        panel_title("Severity mix")
        severities = None
        if alerts is not None and "severity" in alerts.columns:
            severities = alerts["severity"].value_counts().to_dict()
        st.plotly_chart(charts.severity_donut(severities), width="stretch")

    panel_title("Risk over time")
    timeline = None
    if alerts is not None and {"timestamp", "risk_score"} <= set(alerts.columns):
        timeline = alerts[["timestamp", "risk_score"]]
    st.plotly_chart(charts.risk_timeline(timeline), width="stretch")

    panel_title("Highest-risk entities")
    if alerts is not None and {"entity_id", "risk_score"} <= set(alerts.columns):
        ranked = (
            alerts.groupby("entity_id")
            .agg(alerts=("risk_score", "size"), peak_risk=("risk_score", "max"),
                 mean_risk=("risk_score", "mean"))
            .sort_values("peak_risk", ascending=False)
            .head(int(ctx.cfg.get("dashboard.top_entities", 10)))
            .reset_index()
        )
        st.dataframe(ranked, width="stretch", hide_index=True)
    else:
        st.caption("Entity risk ranking populates once the risk engine writes alerts.")
