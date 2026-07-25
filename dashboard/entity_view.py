"""Entity Investigation: what is normal for THIS entity, and what changed."""

from __future__ import annotations

import streamlit as st

from . import charts
from .components import KPI, awaiting_data, chips, kpi_row, page_header, panel_title
from .state import DashboardContext

TITLE = "Entity Investigation"
SUBTITLE = "Behavioral baseline, history and deviations for a single entity"


def render(ctx: DashboardContext) -> None:
    page_header(TITLE, SUBTITLE)

    entity_ids = ctx.entity_ids()
    selector = st.columns([1.4, 3])[0]
    entity_id = selector.selectbox(
        "Entity",
        entity_ids or ["-- no entities generated yet --"],
        disabled=not entity_ids,
        key="entity_choice",
    )

    blocked = awaiting_data(
        ctx,
        "events",
        "profiles",
        note="The baseline panels below are the core of the project's philosophy: "
        "deviation is judged against this entity's own history, not a global rule.",
    )

    events = ctx.events()
    entity_events = None
    if events is not None and entity_ids and "entity_id" in events.columns:
        entity_events = events[events["entity_id"].astype(str) == str(entity_id)]

    n_events = len(entity_events) if entity_events is not None else None
    first_seen = (
        str(entity_events["timestamp"].min()) if entity_events is not None
        and not entity_events.empty else None
    )
    kpi_row(
        [
            KPI("Events observed", n_events, foot="drives profile confidence"),
            KPI("First seen", first_seen or None, foot="cold-start reference"),
            KPI("Profile confidence", None, foot="personal vs cohort blend"),
            KPI("Open alerts", None, foot="unresolved"),
        ]
    )
    st.write("")

    left, right = st.columns([1, 1])
    with left:
        panel_title("Normal behaviour profile")
        _profile_panel(entity_events)
    with right:
        panel_title("Activity timeline")
        timeline = None
        alerts = ctx.alerts()
        if (
            alerts is not None
            and entity_ids
            and {"timestamp", "risk_score", "entity_id"} <= set(alerts.columns)
        ):
            subset = alerts[alerts["entity_id"].astype(str) == str(entity_id)]
            timeline = subset[["timestamp", "risk_score"]] if not subset.empty else None
        st.plotly_chart(charts.risk_timeline(timeline), width="stretch")

    panel_title("Event history")
    if entity_events is not None and not entity_events.empty:
        st.dataframe(
            entity_events.sort_values("timestamp", ascending=False).head(200),
            width="stretch",
            hide_index=True,
            height=340,
        )
    elif not blocked:
        st.caption("No events recorded for this entity.")


def _profile_panel(entity_events) -> None:
    """Known devices / locations / resources, read from history when available."""

    def unique(column: str) -> list[str]:
        if entity_events is None or entity_events.empty or column not in entity_events:
            return []
        return sorted(entity_events[column].dropna().astype(str).unique())[:12]

    for label, column in (
        ("Typical hours", "hour_of_day"),
        ("Known locations", "city"),
        ("Known devices", "device_id"),
        ("Typical resources", "resource_accessed"),
        ("Authentication methods", "auth_method"),
    ):
        st.markdown(
            f'<div style="font-size:.72rem;letter-spacing:.08em;color:#8b9bb0;'
            f'text-transform:uppercase;margin-top:.4rem">{label}</div>'
            f"{chips(unique(column), empty='pending phase 2')}",
            unsafe_allow_html=True,
        )
