"""Live Attack Simulator.

Injection synthesises real event(s), then delegates scoring to
``src.detection.process_injection``. Until Phase 11 implements that pipeline,
the UI shows the generated input events and an explicit not-ready message -
never a fabricated alert.
"""

from __future__ import annotations

import html

import streamlit as st

from src.schema import ATTACK_CLASSES

from .components import page_header, panel_title, pipeline_strip, severity_badge
from .contracts import ATTACK_DISPLAY_NAMES, ENTITY_TYPE_LABELS
from .data_provider import DashboardDataProvider
from .simulator_service import SimulatorService
from .state import DashboardContext

TITLE = "Live Attack Simulator"
SUBTITLE = "Inject a synthetic attack into the live detection pipeline"


def _render_prerequisites(service: SimulatorService) -> None:
    rows = "".join(
        f"<div class='aegis-status'>"
        f"<span class=\"{'ok' if item.ready else 'pending'}\">"
        f"{'&#9679;' if item.ready else '&#9675;'} {html.escape(item.key)}</span>"
        f"<span class='ph'>{'ready' if item.ready else f'P{item.phase}'}</span>"
        f"</div>"
        for item in service.prerequisites()
    )
    st.markdown(rows, unsafe_allow_html=True)


def _render_entity_context(provider: DashboardDataProvider, entity_id: str) -> None:
    summary = provider.get_entity_summary(entity_id)
    if summary is None:
        return
    type_label = ENTITY_TYPE_LABELS.get(summary.entity_type, summary.entity_type)
    st.caption(
        f"{type_label} · {summary.home_city}, {summary.home_country} · "
        f"{summary.events_observed} historical events · {summary.open_alerts} open alerts"
    )


def _render_outcome(service: SimulatorService, outcome) -> None:
    from . import charts
    from .data_provider import DashboardDataProvider

    if outcome.error:
        st.warning(outcome.error)

    st.markdown("**Generated injection events**")
    st.caption(
        "These rows are the simulator input. They are not pre-scored alerts."
    )
    display = outcome.events.copy()
    display["timestamp"] = display["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    st.dataframe(display, width="stretch", hide_index=True)

    if outcome.result and outcome.result.alert is not None:
        alert = outcome.result.alert
        st.markdown("**Pipeline alert**")
        st.markdown(
            f"{severity_badge(str(alert.get('severity', 'LOW')))} &nbsp; "
            f"Risk **{float(alert.get('risk_score', 0)):.0f}** · "
            f"{ATTACK_DISPLAY_NAMES.get(str(alert.get('attack_type', '')), alert.get('attack_type', '-'))}",
            unsafe_allow_html=True,
        )
        provider = DashboardDataProvider(mode="mock")
        st.plotly_chart(
            charts.contribution_bars(provider.get_score_contributions(alert)),
            width="stretch",
        )
    elif outcome.success and outcome.result:
        st.info(outcome.result.message or "Pipeline completed without raising an alert.")


def render(ctx: DashboardContext) -> None:
    provider = DashboardDataProvider.from_context(ctx)
    service = SimulatorService(ctx, provider)

    page_header(TITLE, SUBTITLE)

    if provider.is_mock:
        st.info(
            "**Development fixture active** — entity selection uses the mock roster. "
            "Injection remains disabled until pipeline artifacts exist on disk, even in mock mode."
        )

    entity_ids = service.list_entities()
    ready = service.is_ready()

    left, right = st.columns([1, 1.55])
    with left:
        panel_title("Injection controls")
        entity_id = st.selectbox(
            "Target entity",
            entity_ids or ["-- no entities available --"],
            disabled=not entity_ids,
            key="sim_entity",
        )
        if entity_ids and entity_id in entity_ids:
            _render_entity_context(provider, entity_id)

        attack = st.selectbox(
            "Attack scenario",
            list(ATTACK_CLASSES),
            format_func=lambda x: ATTACK_DISPLAY_NAMES.get(x, x),
            key="sim_attack",
        )
        st.caption(service.scenario_description(attack))
        intensity = st.slider(
            "Intensity",
            1,
            5,
            3,
            key="sim_intensity",
            help="Scales the number of generated events and deviation strength.",
        )
        injected = st.button(
            "INJECT ATTACK",
            type="primary",
            width="stretch",
            disabled=not ready or not entity_ids,
            key="sim_inject",
        )
        if not ready:
            st.caption(
                "Injection unlocks when all prerequisite pipeline artifacts exist "
                "(entities, events, profiles, models)."
            )

        panel_title("Prerequisites")
        _render_prerequisites(service)

    with right:
        panel_title("Detection path")
        pipeline_strip(active=service.active_pipeline_stages())
        st.caption(
            "An injected event must traverse exactly this path - the same code that "
            "scores the offline corpus. Nothing on this page is pre-computed."
        )

        if injected and entity_ids and entity_id in entity_ids:
            with st.spinner("Generating events and invoking detection pipeline..."):
                outcome = service.run(entity_id, attack, intensity)
            _render_outcome(service, outcome)
        else:
            st.caption("Configure a scenario and inject to invoke the pipeline.")
