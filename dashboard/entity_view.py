"""Entity Investigation: what is normal for THIS entity, and what changed."""

from __future__ import annotations

import html

import streamlit as st

from src.schema import EntityType

from . import charts
from .components import KPI, chips, kpi_row, page_header, panel_title, severity_badge
from .contracts import ATTACK_DISPLAY_NAMES, ENTITY_TYPE_LABELS, REASON_CODE_LABELS
from .data_provider import DashboardDataProvider, EntityBehaviorProfile
from .state import DashboardContext

TITLE = "Entity Investigation"
SUBTITLE = "Behavioral baseline, history and deviations for a single entity"


def _data_source_notice(provider: DashboardDataProvider) -> None:
    if provider.is_mock:
        st.info(
            "**Development fixture active** — entity profiles and histories are "
            "synthetic sample data for UI development, not learned pipeline profiles."
        )
    elif not provider.has_data:
        st.warning(
            "Pipeline mode is active but entity/event data is not available yet. "
            "Set `dashboard.data_source: mock` for frontend development."
        )
    else:
        st.caption(f"Data source: {provider.source_label}")


def _entity_selector(provider: DashboardDataProvider) -> str | None:
    col_type, col_search, col_entity = st.columns([1.0, 1.0, 1.4])
    entity_type = col_type.selectbox(
        "Entity type",
        ["All"] + [t.value for t in EntityType],
        format_func=lambda x: "All types" if x == "All" else ENTITY_TYPE_LABELS.get(x, x),
        key="entity_type_filter",
    )
    query = col_search.text_input("Search entity ID", "", key="entity_search")
    type_filter = None if entity_type == "All" else entity_type
    entity_ids = provider.list_entity_ids(entity_type=type_filter, query=query)
    if not entity_ids:
        col_entity.selectbox("Entity", ["-- no matching entities --"], disabled=True)
        return None
    # Prefer entities that already have alerts when present.
    alerted = set(provider.get_alerts()["entity_id"].astype(str)) if not provider.get_alerts().empty else set()
    preferred = [eid for eid in entity_ids if eid in alerted]
    ordered = preferred + [eid for eid in entity_ids if eid not in alerted]
    default_index = 0
    return col_entity.selectbox("Entity", ordered, index=default_index, key="entity_choice")


def _render_metadata(summary) -> None:
    type_label = ENTITY_TYPE_LABELS.get(summary.entity_type, summary.entity_type)
    st.markdown(
        f"**{summary.entity_id}** &nbsp;·&nbsp; {type_label}  \n"
        f"Role: `{summary.role or 'unknown'}` &nbsp;·&nbsp; "
        f"Department: `{summary.department or 'n/a'}`  \n"
        f"Home region: {summary.home_city or '-'}, {summary.home_country or '-'}",
        unsafe_allow_html=True,
    )


def _render_profile(profile: EntityBehaviorProfile) -> None:
    avg_session = (
        f"{profile.avg_session_seconds:,.0f}s" if profile.avg_session_seconds is not None else "-"
    )
    sections = (
        ("Typical hours", profile.typical_hours),
        ("Known locations", profile.known_locations),
        ("Known devices", profile.known_devices),
        ("Typical resources", profile.typical_resources),
        ("Authentication methods", profile.auth_methods),
    )
    for label, values in sections:
        st.markdown(
            f'<div style="font-size:.72rem;letter-spacing:.08em;color:#8b9bb0;'
            f'text-transform:uppercase;margin-top:.45rem">{html.escape(label)}</div>'
            f"{chips(values, empty='insufficient history')}",
            unsafe_allow_html=True,
        )
    st.caption(f"Average session duration: {avg_session}")


def _render_recent_alerts(provider: DashboardDataProvider, entity_id: str) -> None:
    alerts = provider.get_entity_alerts(entity_id).head(6)
    if alerts.empty:
        st.caption("No alerts recorded for this entity.")
        return
    for _, row in alerts.iterrows():
        attack = ATTACK_DISPLAY_NAMES.get(str(row.get("attack_type")), str(row.get("attack_type")))
        conf = row.get("attack_confidence")
        conf_txt = f"{float(conf):.0%}" if conf is not None and str(conf) != "nan" else "—"
        st.markdown(
            f"{severity_badge(str(row.get('severity', 'LOW')))} &nbsp; "
            f"**{float(row.get('risk_score', 0)):.0f}** · {attack} ({conf_txt})  \n"
            f"<span style='color:#8b9bb0;font-size:.82rem'>"
            f"{html.escape(str(row.get('short_reason') or ''))}</span>",
            unsafe_allow_html=True,
        )
        codes = provider.parse_reason_codes(row)
        if codes:
            labels = [
                REASON_CODE_LABELS.get(code, code.replace("_", " ").title())
                for code in codes[:4]
            ]
            st.markdown(chips(labels), unsafe_allow_html=True)


def render(ctx: DashboardContext) -> None:
    provider = DashboardDataProvider.from_context(ctx)
    page_header(TITLE, SUBTITLE)
    _data_source_notice(provider)

    entity_id = _entity_selector(provider)
    if entity_id is None:
        st.caption("Adjust the entity type or search filter to select an entity.")
        return

    summary = provider.get_entity_summary(entity_id)
    if summary is None:
        st.caption("Entity metadata is unavailable.")
        return

    _render_metadata(summary)

    confidence_pct = f"{summary.profile_confidence:.0%}"
    kpi_row(
        [
            KPI("Events observed", summary.events_observed, foot="drives profile confidence"),
            KPI(
                "First seen",
                summary.first_seen.strftime("%Y-%m-%d %H:%M") if summary.first_seen is not None else None,
                foot="cold-start reference",
            ),
            KPI(
                "Profile confidence",
                confidence_pct,
                foot=f"stage: {summary.profile_stage}",
            ),
            KPI("Open alerts", summary.open_alerts, foot="matching this entity"),
            KPI(
                "Peak risk",
                f"{summary.peak_risk:.0f}" if summary.peak_risk is not None else None,
                foot="from risk history when available",
            ),
        ]
    )
    st.write("")

    left, right = st.columns([1, 1])
    with left:
        panel_title("Normal behaviour profile")
        _render_profile(provider.get_entity_profile(entity_id))
    with right:
        panel_title("Risk evolution over time")
        st.plotly_chart(
            charts.risk_timeline(provider.get_entity_risk_timeline(entity_id)),
            width="stretch",
        )
        if summary.peak_risk is not None:
            st.caption(
                f"Peak risk {summary.peak_risk:.0f} · "
                f"Mean risk {summary.mean_risk:.0f}"
                if summary.mean_risk is not None
                else f"Peak risk {summary.peak_risk:.0f}"
            )

    panel_title("Alert explanations for this entity")
    _render_recent_alerts(provider, entity_id)

    panel_title("Chronological event history")
    history_limit = int(ctx.cfg.get("dashboard.entity_history_rows", 200))
    history = provider.get_entity_event_history(entity_id, limit=history_limit)
    if history.empty:
        st.caption("No events recorded for this entity in the current dataset.")
    else:
        st.dataframe(history, width="stretch", hide_index=True, height=340)
        st.caption(
            "Operational view — campaign / ground-truth labels are intentionally omitted."
        )
