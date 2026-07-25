"""Alert Queue and Alert Detail: the analyst's primary workspace."""

from __future__ import annotations

import html

import streamlit as st

from src.schema import ATTACK_CLASSES, EntityType, Severity

from .components import KPI, kpi_row, page_header, panel_title, severity_badge
from .contracts import ATTACK_DISPLAY_NAMES, ENTITY_TYPE_LABELS
from .data_provider import AlertQueueFilters, DashboardDataProvider
from .state import DashboardContext

TITLE = "Alert Queue"
SUBTITLE = "Risk-ranked alerts within the analyst triage budget"


def _data_source_notice(provider: DashboardDataProvider) -> None:
    if provider.is_mock:
        st.info(
            "**Development fixture active** — alerts shown here are synthetic sample "
            "data for UI development, not output from the detection pipeline."
        )
    elif not provider.has_data:
        st.warning(
            "Pipeline mode is active but no scored alerts are available yet. "
            "Set `dashboard.data_source: mock` for frontend development."
        )
    else:
        st.caption(f"Data source: {provider.source_label}")


def _build_filters() -> AlertQueueFilters:
    col_sev, col_type, col_etype, col_risk, col_entity = st.columns(
        [1.0, 1.2, 1.1, 1.0, 1.0]
    )
    severities = tuple(
        col_sev.multiselect(
            "Severity",
            [s.value for s in Severity],
            default=[],
            key="flt_severity",
        )
    )
    attack_types = tuple(
        col_type.multiselect(
            "Attack type",
            list(ATTACK_CLASSES),
            default=[],
            key="flt_attack",
            format_func=lambda x: ATTACK_DISPLAY_NAMES.get(x, x),
        )
    )
    entity_types = tuple(
        col_etype.multiselect(
            "Entity type",
            [t.value for t in EntityType],
            default=[],
            key="flt_entity_type",
            format_func=lambda x: ENTITY_TYPE_LABELS.get(x, x),
        )
    )
    min_risk = float(
        col_risk.slider("Minimum risk", 0, 100, 0, step=5, key="flt_risk")
    )
    entity_query = col_entity.text_input("Entity contains", "", key="flt_entity")
    return AlertQueueFilters(
        severities=severities,
        attack_types=attack_types,
        entity_types=entity_types,
        min_risk=min_risk,
        entity_query=entity_query,
    )


def _render_detail(provider: DashboardDataProvider, alert_id: str) -> None:
    alert = provider.get_alert_by_id(alert_id)
    if alert is None:
        st.caption("Selected alert is no longer available under the current filters.")
        return

    attack_label = ATTACK_DISPLAY_NAMES.get(
        str(alert.get("attack_type", "")), str(alert.get("attack_type", "unknown"))
    )
    entity_type_label = ENTITY_TYPE_LABELS.get(
        str(alert.get("entity_type", "")), str(alert.get("entity_type", "-"))
    )
    reasons = provider.parse_reasons(alert.get("reasons")) or provider.parse_reasons(
        alert.get("short_reason")
    )

    left, right = st.columns([1, 1.25])
    with left:
        st.markdown(
            f"**Risk {float(alert.get('risk_score', 0)):.0f}/100** &nbsp; "
            f"{severity_badge(str(alert.get('severity', 'LOW')))}",
            unsafe_allow_html=True,
        )
        st.markdown(f"**Alert ID:** `{alert.get('alert_id', '-')}`")
        st.markdown(f"**Suspected attack:** {attack_label}")
        st.markdown(
            f"**Entity:** `{alert.get('entity_id', '-')}` &nbsp; ({entity_type_label})"
        )
        st.markdown(f"**Observed:** {alert.get('timestamp', '-')}")

        st.markdown("**Score breakdown**")
        st.markdown(
            f"- Behavioral anomaly: `{float(alert.get('anomaly_score', 0)):.3f}`  \n"
            f"- Sequence anomaly: `{float(alert.get('sequence_score', 0)):.3f}`  \n"
            f"- Classification confidence: `{float(alert.get('attack_confidence', 0)):.3f}`"
        )

        if reasons:
            st.markdown("**Contributing factors**")
            items = "".join(f"<li>{html.escape(r)}</li>" for r in reasons)
            st.markdown(f"<ul style='margin-top:.2rem'>{items}</ul>", unsafe_allow_html=True)
        elif alert.get("short_reason"):
            st.markdown(f"**Reason:** {alert.get('short_reason')}")

    with right:
        from . import charts

        panel_title("Score contributions")
        st.plotly_chart(
            charts.contribution_bars(provider.get_score_contributions(alert)),
            width="stretch",
        )


def render(ctx: DashboardContext) -> None:
    provider = DashboardDataProvider.from_context(ctx)
    page_header(TITLE, SUBTITLE)
    _data_source_notice(provider)

    filters = _build_filters()
    summary = provider.get_alert_queue_summary(filters)
    page_size = int(ctx.cfg.get("dashboard.alerts_page_size", 50))
    filtered = provider.filter_alerts(filters)

    kpi_row(
        [
            KPI("Matching alerts", summary.matching_alerts),
            KPI("Critical", summary.critical_alerts),
            KPI(
                "Peak risk",
                f"{summary.peak_risk:.0f}" if summary.peak_risk is not None else "--",
            ),
            KPI("Distinct entities", summary.distinct_entities),
        ]
    )
    st.write("")

    if filtered.empty:
        st.caption("No alerts match the current filters.")
        panel_title("Alert detail")
        st.caption("Adjust filters or wait for the detection pipeline to produce alerts.")
        return

    queue_table = provider.get_alert_queue_table(filters, limit=page_size)
    st.dataframe(queue_table, width="stretch", hide_index=True, height=420)

    st.write("")
    panel_title("Alert detail")

    options = filtered.head(page_size)
    choice_labels = {
        str(row.alert_id): (
            f"{row.risk_score:.0f} | {row.severity} | {row.entity_id} | "
            f"{ATTACK_DISPLAY_NAMES.get(row.attack_type, row.attack_type)}"
        )
        for row in options.itertuples()
    }
    selected = st.selectbox(
        "Selected alert",
        list(choice_labels.keys()),
        format_func=lambda aid: choice_labels[aid],
        key="alert_choice",
    )
    _render_detail(provider, selected)
