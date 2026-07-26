"""Tests for the live attack simulator."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from dashboard.mock_data import generate_entities, generate_injection_events
from dashboard.simulator_service import REQUIRED_ARTIFACTS, SimulatorService
from dashboard.state import DashboardContext
from src.artifacts import artifact
from src.paths import PROJECT_ROOT

APP = str(PROJECT_ROOT / "app.py")


def test_injection_events_impossible_travel_fixture():
    entities = generate_entities()
    home = entities.loc[entities["entity_id"] == "USR_001", "home_city"].iloc[0]
    events = generate_injection_events("USR_001", "IMPOSSIBLE_TRAVEL", 3, entities)
    assert len(events) == 2
    assert events.iloc[0]["city"] == home
    assert events.iloc[1]["city"] == "London"
    assert events["event_id"].str.startswith("INJ_").all()
    from src.schema import IDENTITY_COLUMNS, OBSERVATION_COLUMNS, validate_events

    validate_events(events, require_labels=False)
    for col in IDENTITY_COLUMNS + OBSERVATION_COLUMNS:
        assert col in events.columns


@pytest.mark.parametrize(
    "attack_type,min_events",
    [
        ("BRUTE_FORCE", 8),
        ("LATERAL_MOVEMENT", 3),
        ("LOW_AND_SLOW_EXFILTRATION", 3),
    ],
)
def test_injection_event_volume_scales_with_intensity(attack_type, min_events):
    entities = generate_entities()
    low = generate_injection_events("USR_010", attack_type, 1, entities)
    high = generate_injection_events("USR_010", attack_type, 5, entities)
    assert len(high) >= len(low) >= min_events


def test_simulator_service_not_ready_without_artifacts():
    service = SimulatorService(DashboardContext.build())
    assert not service.is_ready()
    assert any(not item.ready for item in service.prerequisites())


def test_simulator_run_is_honest_when_pipeline_missing():
    for key in REQUIRED_ARTIFACTS:
        path = artifact(key).path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"placeholder")

    service = SimulatorService(DashboardContext.build())
    assert service.is_ready()
    outcome = service.run("USR_001", "IMPOSSIBLE_TRAVEL", 3)
    assert not outcome.success
    assert outcome.result is None
    assert outcome.error
    assert outcome.alerts_posted == 0
    lowered = outcome.error.lower()
    assert (
        "pipeline" in lowered
        or "failed to load" in lowered
        or "failed to run" in lowered
        or "expecting value" in lowered
        or "entities" in lowered
    )


def test_simulator_page_renders_with_controls():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value("Attack Simulator").run()
    assert not at.exception
    inject = [b for b in at.button if b.label == "INJECT ATTACK"]
    assert len(inject) == 1 and inject[0].disabled
    assert at.selectbox, "expected entity and attack selectors"
    clear = [b for b in at.button if b.label == "Clear live overlays"]
    assert len(clear) == 1


def test_simulator_uses_live_injection_module_not_legacy_generator():
    import dashboard.simulator_service as service

    text = Path(service.__file__).read_text(encoding="utf-8")
    assert "data_generator" not in text
    assert "synthesize_live_attack" in text
    assert "live_state" in text
