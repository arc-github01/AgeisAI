"""Live Attack Simulator.

Injection synthesises real attack campaigns via ``src.generator.live_injection``,
then scores them with ``src.detection.process_injection``. Until pipeline
artifacts exist on disk, the UI shows prerequisites and never fabricates alerts.
Successful runs also publish alerts into the SOC session overlay.
"""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from src.schema import ATTACK_CLASSES

from . import charts, live_state
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


_SEVERITY_ORDER = ("LOW", "MEDIUM", "HIGH", "CRITICAL")


def _render_scored_events(outcome) -> None:
    """Per-event pipeline output, so a no-alert run still shows real scores."""
    results = outcome.result.results if outcome.result else []
    if not results:
        return

    frame = pd.DataFrame(
        [
            {
                "Timestamp": r.timestamp,
                "Entity": r.entity_id,
                "Risk": round(r.risk_score, 1),
                "Severity": r.severity,
                "Predicted Type": ATTACK_DISPLAY_NAMES.get(
                    r.predicted_attack_type, r.predicted_attack_type
                ),
                "Confidence": round(r.attack_confidence, 2),
                "Alerted": r.alerted,
                "Top reason": r.short_reason,
                "Latency (ms)": round(r.latency_ms, 1),
            }
            for r in results
        ]
    )

    peak = max(r.risk_score for r in results)
    severity = max(
        (r.severity for r in results),
        key=lambda s: _SEVERITY_ORDER.index(s) if s in _SEVERITY_ORDER else -1,
    )
    alerted = sum(1 for r in results if r.alerted)
    mean_latency = sum(r.latency_ms for r in results) / len(results)

    cols = st.columns(4)
    cols[0].metric("Events scored", len(results))
    cols[1].metric("Peak risk", f"{peak:.0f}")
    cols[2].metric("Peak severity", severity)
    cols[3].metric("Mean latency", f"{mean_latency:.0f} ms")

    st.dataframe(
        frame.sort_values("Risk", ascending=False),
        width="stretch",
        hide_index=True,
        height=260,
    )

    if alerted == 0:
        st.info(
            f"No alert raised: peak risk {peak:.0f} ({severity}) stayed below the "
            "alerting threshold. Patient, low-signal campaigns are expected to "
            "sit under the analyst alert budget — this is a real pipeline result, "
            "not a failure."
        )


def _render_outcome(provider: DashboardDataProvider, outcome) -> None:
    if outcome.error:
        st.warning(outcome.error)

    st.markdown("**Generated injection events**")
    st.caption(
        "These rows are synthesised by the real attack injectors — not pre-scored alerts."
    )
    if outcome.events is not None and not outcome.events.empty:
        display = outcome.events.copy()
        display["timestamp"] = display["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        st.dataframe(display, width="stretch", hide_index=True, height=240)

    panel_title("Pipeline output")
    _render_scored_events(outcome)

    if outcome.result and outcome.result.alert is not None:
        alert = (
            outcome.result.alert.to_dict()
            if hasattr(outcome.result.alert, "to_dict")
            else dict(outcome.result.alert)
        )
        st.markdown("**Pipeline alert**")
        st.markdown(
            f"{severity_badge(str(alert.get('severity', 'LOW')))} &nbsp; "
            f"Risk **{float(alert.get('risk_score', 0)):.0f}** · "
            f"{ATTACK_DISPLAY_NAMES.get(str(alert.get('attack_type', '')), alert.get('attack_type', '-'))}",
            unsafe_allow_html=True,
        )
        if outcome.alerts_posted:
            st.caption(
                f"Posted {outcome.alerts_posted} alert(s) to Overview / Alert Queue "
                f"for this session ({live_state.live_alert_count()} live overlay total)."
            )
        contributions = provider.get_score_contributions(alert)
        if not contributions.empty:
            st.plotly_chart(
                charts.contribution_bars(contributions),
                width="stretch",
            )
        reasons = alert.get("reasons")
        if reasons:
            st.markdown("**Why**")
            st.write(reasons)
    elif outcome.success and outcome.result and not outcome.result.results:
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
            help="Scales campaign size; lower intensity uses stealthier variants.",
        )
        injected = st.button(
            "INJECT ATTACK",
            type="primary",
            width="stretch",
            disabled=not ready or not entity_ids,
            key="sim_inject",
        )
        clear = st.button(
            "Clear live overlays",
            width="stretch",
            disabled=live_state.live_alert_count() == 0
            and live_state.live_events_frame().empty,
            key="sim_clear",
        )
        if clear:
            live_state.clear_live_overlays()
            st.rerun()

        if not ready:
            st.caption(
                "Injection unlocks when all prerequisite pipeline artifacts exist "
                "(entities, events, profiles, models)."
            )

        panel_title("Prerequisites")
        _render_prerequisites(service)

    with right:
        panel_title("Detection path")
        stages = service.active_pipeline_stages()
        if ready:
            stages = stages | {
                "EVENT",
                "FEATURES",
                "PROFILE",
                "ANOMALY",
                "CLASSIFIER",
                "RISK",
                "EXPLANATION",
                "ALERT",
            }
        pipeline_strip(active=stages)
        st.caption(
            "An injected event traverses the same ``process_event`` path that scores "
            "the offline corpus. Nothing on this page is pre-computed."
        )

        if injected and entity_ids and entity_id in entity_ids:
            with st.spinner("Generating campaign and invoking detection pipeline..."):
                outcome = service.run(entity_id, attack, intensity)
            _render_outcome(provider, outcome)
        else:
            st.caption("Configure a scenario and inject to invoke the pipeline.")
