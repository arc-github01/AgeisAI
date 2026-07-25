"""SOC Overview — analyst situational-awareness landing page."""

from __future__ import annotations

import html

import streamlit as st

from . import charts
from .components import KPI, kpi_row, panel_title, severity_badge
from .data_provider import DashboardDataProvider
from .state import DashboardContext

TITLE = "AEGIS"
SUBTITLE = "Adaptive Behavioral Threat Detection for IT/OT Systems"
DESCRIPTION = (
    "Behavioral intelligence across users, service accounts, and edge devices."
)


def _overview_header() -> None:
    st.markdown(
        f'<div class="aegis-page-title">{html.escape(TITLE)}</div>'
        f'<div class="aegis-page-sub">{html.escape(SUBTITLE)}</div>'
        f'<div class="aegis-page-sub" style="margin-top:-.55rem;font-size:.78rem">'
        f"{html.escape(DESCRIPTION)}</div>"
        f'<div class="aegis-rule"></div>',
        unsafe_allow_html=True,
    )


def _data_source_notice(provider: DashboardDataProvider) -> None:
    if provider.is_mock:
        st.info(
            "**Development fixture active** — dashboard data is synthetic frontend "
            "sample data, not measured model output. Switch `dashboard.data_source` "
            "to `pipeline` in config once real alerts are available."
        )
    else:
        st.caption(f"Data source: {provider.source_label}")


def _render_recent_critical(alerts) -> None:
    if alerts.empty:
        st.caption("No critical alerts in the current dataset.")
        return
    for row in alerts.itertuples(index=False):
        attack = getattr(row, "attack_type", "UNKNOWN")
        st.markdown(
            f"{severity_badge(getattr(row, 'severity', 'CRITICAL'))} &nbsp; "
            f"**{getattr(row, 'entity_id', '-')}** &middot; "
            f"{attack.replace('_', ' ').title()} &middot; "
            f"Risk **{getattr(row, 'risk_score', 0):.0f}**  \n"
            f"<span style='color:#8b9bb0;font-size:.82rem'>"
            f"{html.escape(str(getattr(row, 'short_reason', '')))}</span>",
            unsafe_allow_html=True,
        )


def render(ctx: DashboardContext) -> None:
    provider = DashboardDataProvider.from_context(ctx)
    kpis = provider.get_overview_kpis()
    top_n = int(ctx.cfg.get("dashboard.top_entities", 10))
    recent_n = int(ctx.cfg.get("dashboard.recent_critical_alerts", 8))

    _overview_header()
    _data_source_notice(provider)

    if not provider.has_data and not provider.is_mock:
        st.warning(
            "Pipeline mode is active but scored alerts are not available yet. "
            "Generate events and run the detection pipeline, or set "
            "`dashboard.data_source: mock` for frontend development."
        )

    kpi_row(
        [
            KPI("Events processed", kpis.events_processed, foot="access events ingested"),
            KPI("Entities monitored", kpis.entities_monitored, foot="users / services / devices"),
            KPI("Active alerts", kpis.active_alerts, foot="within analyst triage scope"),
            KPI("Critical alerts", kpis.critical_alerts, foot="risk 81–100"),
            KPI("Alert rate", kpis.alert_rate_pct, foot="alerts / events processed"),
        ]
    )
    st.write("")

    panel_title("Threat activity over time")
    st.plotly_chart(
        charts.alert_activity_timeline(provider.get_alert_timeline()),
        width="stretch",
    )

    left, right = st.columns([1.15, 0.85])
    with left:
        panel_title("Threat distribution")
        st.plotly_chart(
            charts.attack_distribution(provider.get_threat_distribution()),
            width="stretch",
        )
    with right:
        panel_title("Severity distribution")
        st.plotly_chart(
            charts.severity_donut(provider.get_severity_distribution()),
            width="stretch",
        )

    mid_left, mid_right = st.columns([1.35, 1])
    with mid_left:
        panel_title("Top risk entities")
        top_entities = provider.get_top_risk_entities(top_n)
        if top_entities.empty:
            st.caption("No ranked entities yet.")
        else:
            display = top_entities.rename(
                columns={
                    "entity_id": "Entity",
                    "type": "Type",
                    "risk_score": "Risk",
                    "primary_signal": "Primary Signal",
                }
            )
            st.dataframe(display, width="stretch", hide_index=True, height=320)
    with mid_right:
        panel_title("Entity type distribution")
        st.plotly_chart(
            charts.entity_type_distribution(provider.get_entity_type_distribution()),
            width="stretch",
        )

    panel_title("Recent critical alerts")
    _render_recent_critical(provider.get_recent_critical_alerts(recent_n))
