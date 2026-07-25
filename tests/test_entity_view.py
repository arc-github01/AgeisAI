"""Tests for Entity Investigation page and provider methods."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from dashboard.data_provider import DashboardDataProvider
from src.paths import PROJECT_ROOT

APP = str(PROJECT_ROOT / "app.py")


def test_list_entities_and_metadata():
    provider = DashboardDataProvider(mode="mock")
    ids = provider.list_entity_ids()
    assert len(ids) == 1000
    assert provider.list_entity_ids(entity_type="user")[0].startswith("USR_")

    meta = provider.get_entity_metadata("USR_001")
    assert meta is not None
    assert meta["entity_type"] == "user"


def test_entity_with_alert_has_rich_history():
    provider = DashboardDataProvider(mode="mock")
    alert_entity = provider.get_alerts().iloc[0]["entity_id"]
    events = provider.get_entity_events(alert_entity)
    assert len(events) >= 25
    profile = provider.get_entity_profile(alert_entity)
    assert profile.known_devices
    assert profile.typical_resources
    assert profile.known_locations


def test_entity_summary_and_confidence_stages():
    provider = DashboardDataProvider(mode="mock")
    alert_entity = provider.get_alerts().iloc[0]["entity_id"]
    summary = provider.get_entity_summary(alert_entity)
    assert summary is not None
    assert summary.events_observed >= 25
    assert summary.open_alerts >= 1
    assert summary.profile_stage in {"cold-start", "blending", "mature"}
    assert 0.0 < summary.profile_confidence <= 1.0
    assert summary.peak_risk is not None


def test_entity_event_and_alert_history_tables():
    provider = DashboardDataProvider(mode="mock")
    entity_id = provider.get_alerts().iloc[0]["entity_id"]
    events = provider.get_entity_event_history(entity_id, limit=10)
    assert len(events) <= 10
    assert "Resource" in events.columns

    alerts = provider.get_entity_alert_history(entity_id, limit=5)
    assert not alerts.empty
    assert "Attack Type" in alerts.columns


def test_entity_investigation_page_renders():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value("Entity Investigation").run()
    assert not at.exception
    assert at.selectbox, "expected entity selector"
    assert at.dataframe, "expected event/alert tables"
    body = " ".join(block.value for block in at.markdown)
    assert "Events observed" in body
    assert "Normal behaviour profile" in body
    assert at.info and "Development fixture active" in at.info[0].value


def test_entity_view_does_not_import_generator_modules():
    import dashboard.entity_view as page

    text = Path(page.__file__).read_text(encoding="utf-8")
    assert "data_generator" not in text
    assert "src.generator" not in text
