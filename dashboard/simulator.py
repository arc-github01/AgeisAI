"""Live Attack Simulator.

Contract for this page, fixed now so it cannot quietly become a fake demo:
the INJECT control must synthesise real event(s) via the same generator used to
build the dataset, push them through the same feature engineering, anomaly
detector, classifier, risk engine and explainer, and display whatever the
pipeline actually returns - including a miss.

Until those components exist the control stays disabled and says so. No
pre-baked alert is ever rendered here.
"""

from __future__ import annotations

import streamlit as st

from src.schema import ATTACK_CLASSES

from .components import awaiting_data, page_header, panel_title, pipeline_strip
from .state import DashboardContext

TITLE = "Live Attack Simulator"
SUBTITLE = "Inject a synthetic attack into the live detection pipeline"

REQUIRED = ("entities", "events", "profiles", "anomaly_detector", "attack_classifier")

SCENARIOS: dict[str, str] = {
    "BRUTE_FORCE": "Burst of failed authentications from one source in a short window.",
    "IMPOSSIBLE_TRAVEL": "Second authentication from a distant city sooner than travel allows.",
    "CREDENTIAL_STUFFING": "One source IP attempting many entity IDs with high failure rate.",
    "LATERAL_MOVEMENT": "Sudden breadth of never-before-accessed internal systems.",
    "DEVICE_SPOOFING": "Known identity presenting an inconsistent device fingerprint.",
    "LOW_AND_SLOW_EXFILTRATION": "Small off-hours accesses accumulating over days.",
    "INSIDER_DRIFT": "Gradual, legitimate-looking expansion of resource footprint.",
}


def render(ctx: DashboardContext) -> None:
    page_header(TITLE, SUBTITLE)

    entity_ids = ctx.entity_ids()
    ready = ctx.has(*REQUIRED)

    left, right = st.columns([1, 1.6])
    with left:
        panel_title("Injection")
        entity_id = st.selectbox(
            "Target entity",
            entity_ids or ["-- generate the environment first --"],
            disabled=not entity_ids,
            key="sim_entity",
        )
        attack = st.selectbox(
            "Attack scenario", list(ATTACK_CLASSES), key="sim_attack"
        )
        st.caption(SCENARIOS.get(attack, ""))
        intensity = st.slider("Intensity", 1, 5, 3, key="sim_intensity",
                              help="Scales event count and how far behaviour deviates.")
        injected = st.button(
            "INJECT ATTACK",
            type="primary",
            width="stretch",
            disabled=not ready,
            key="sim_inject",
        )
        if not ready:
            st.caption("Injection unlocks when the detection pipeline exists (phase 11).")

    with right:
        panel_title("Detection path")
        pipeline_strip(active=set())
        st.caption(
            "An injected event traverses exactly the path above - the same code that "
            "scores the offline corpus. Nothing on this page is pre-computed."
        )

        if awaiting_data(ctx, *REQUIRED):
            return

        if injected:
            _run_injection(ctx, entity_id, attack, intensity)
        else:
            st.caption("Configure a scenario and inject to see the pipeline result.")


def _run_injection(ctx: DashboardContext, entity_id: str, attack: str, intensity: int) -> None:
    """Wired in phase 11 to the real detection service.

    Deliberately raises rather than rendering a placeholder alert: a simulator
    that invents its own output would invalidate the entire demonstration.
    """
    raise NotImplementedError(
        "attack injection is wired to the detection pipeline in phase 11"
    )
