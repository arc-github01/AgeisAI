"""Tests for the SOC overview data layer and page wiring."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from dashboard.data_provider import DashboardDataProvider
from dashboard.mock_data import (
    EVENTS_PROCESSED_TOTAL,
    N_ALERTS,
    N_CRITICAL,
    N_ENTITIES,
    generate_alerts,
    generate_entities,
    fixture_summary,
)
from src.paths import PROJECT_ROOT
from src.schema import ATTACK_CLASSES

APP = str(PROJECT_ROOT / "app.py")


def test_mock_fixture_is_deterministic():
    first = generate_alerts(generate_entities())
    second = generate_alerts(generate_entities())
    pd.testing.assert_frame_equal(first, second)


def test_mock_fixture_sizes_match_contract():
    summary = fixture_summary()
    assert summary["events_processed"] == EVENTS_PROCESSED_TOTAL
    assert summary["entities_monitored"] == N_ENTITIES
    assert summary["active_alerts"] == N_ALERTS
    assert summary["critical_alerts"] == N_CRITICAL
    assert summary["alert_rate"] == pytest.approx(N_ALERTS / EVENTS_PROCESSED_TOTAL)


def test_provider_mock_mode_exposes_expected_kpis():
    provider = DashboardDataProvider(mode="mock")
    assert provider.is_mock
    assert provider.source_label == "development fixture"
    kpis = provider.get_overview_kpis()
    assert kpis.events_processed == EVENTS_PROCESSED_TOTAL
    assert kpis.entities_monitored == N_ENTITIES
    assert kpis.active_alerts == N_ALERTS
    assert kpis.critical_alerts == N_CRITICAL


def test_provider_aggregates_cover_all_attack_classes():
    provider = DashboardDataProvider(mode="mock")
    alerts = provider.get_alerts()
    assert len(alerts) == N_ALERTS
    assert set(alerts["attack_type"]).issubset(set(ATTACK_CLASSES))
    distribution = provider.get_threat_distribution()
    assert len(distribution) == len(ATTACK_CLASSES)
    timeline = provider.get_alert_timeline()
    assert not timeline.empty
    assert set(timeline["severity"]).issubset({"LOW", "MEDIUM", "HIGH", "CRITICAL"})


def test_provider_top_entities_and_recent_critical():
    provider = DashboardDataProvider(mode="mock")
    top = provider.get_top_risk_entities(5)
    assert len(top) == 5
    assert list(top.columns) == ["entity_id", "type", "risk_score", "primary_signal"]
    assert top["risk_score"].is_monotonic_decreasing

    recent = provider.get_recent_critical_alerts(5)
    assert len(recent) <= 5
    assert (recent["severity"] == "CRITICAL").all()


def test_provider_pipeline_mode_without_artifacts_is_empty():
    provider = DashboardDataProvider(mode="pipeline")
    assert not provider.is_mock
    assert not provider.has_data
    kpis = provider.get_overview_kpis()
    assert kpis.active_alerts == 0
    assert kpis.events_processed == 0


def test_provider_does_not_import_generator_modules():
    import dashboard.data_provider as dp
    import dashboard.mock_data as md
    import dashboard.overview as ov

    for module in (dp, md, ov):
        text = Path(module.__file__).read_text(encoding="utf-8")
        assert "data_generator" not in text
        assert "src.generator" not in text


def test_overview_renders_with_development_fixture():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value("SOC Overview").run()
    assert not at.exception
    body = " ".join(block.value for block in at.markdown)
    assert "Events processed" in body
    assert "Threat activity over time" in body
    assert "Top risk entities" in body
    assert "Predicted attack-type mix" in body or "Threat distribution" in body
    assert at.info and "Development fixture active" in at.info[0].value


def test_overview_tables_render():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value("SOC Overview").run()
    assert not at.exception
    assert at.dataframe, "expected the top-risk-entities table"
