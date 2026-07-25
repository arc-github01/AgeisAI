"""Tests for the Alert Queue page and provider filtering."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from dashboard.data_provider import AlertQueueFilters, DashboardDataProvider
from dashboard.mock_data import N_ALERTS, N_CRITICAL
from src.paths import PROJECT_ROOT
from src.schema import Severity

APP = str(PROJECT_ROOT / "app.py")


def test_filter_alerts_by_severity_and_risk():
    provider = DashboardDataProvider(mode="mock")
    filters = AlertQueueFilters(severities=(Severity.CRITICAL.value,), min_risk=80.0)
    view = provider.filter_alerts(filters)
    assert not view.empty
    assert (view["severity"] == Severity.CRITICAL.value).all()
    assert (view["risk_score"] >= 80).all()


def test_filter_alerts_by_entity_query():
    provider = DashboardDataProvider(mode="mock")
    sample_id = provider.get_alerts().iloc[0]["entity_id"]
    prefix = str(sample_id)[:4]
    view = provider.filter_alerts(AlertQueueFilters(entity_query=prefix))
    assert not view.empty
    assert view["entity_id"].astype(str).str.contains(prefix).all()


def test_alert_queue_summary_and_ranking():
    provider = DashboardDataProvider(mode="mock")
    summary = provider.get_alert_queue_summary()
    assert summary.matching_alerts == N_ALERTS
    assert summary.critical_alerts == N_CRITICAL
    assert summary.peak_risk is not None and summary.peak_risk >= 81

    ranked = provider.filter_alerts()
    assert ranked["risk_score"].is_monotonic_decreasing


def test_alert_queue_table_uses_friendly_labels():
    provider = DashboardDataProvider(mode="mock")
    table = provider.get_alert_queue_table(limit=5)
    assert "Attack Type" in table.columns
    assert "Reason" in table.columns
    assert len(table) == 5
    assert "Brute Force" in table["Attack Type"].values or "Impossible Travel" in table[
        "Attack Type"
    ].values


def test_get_alert_by_id_and_contributions():
    provider = DashboardDataProvider(mode="mock")
    first_id = provider.get_alerts().iloc[0]["alert_id"]
    alert = provider.get_alert_by_id(first_id)
    assert alert is not None
    assert alert["alert_id"] == first_id

    factors = provider.get_score_contributions(alert)
    assert len(factors) == 3
    assert pytest.approx(factors["contribution"].sum(), abs=1e-6) == 1.0

    reasons = provider.parse_reasons(alert["reasons"])
    assert len(reasons) >= 2


def test_alert_queue_page_renders_with_fixture():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value("Alert Queue").run()
    assert not at.exception
    assert at.dataframe, "expected ranked alert table"
    assert at.selectbox, "expected alert selector for detail pane"
    assert at.info and "Development fixture active" in at.info[0].value
    body = " ".join(block.value for block in at.markdown)
    assert "Matching alerts" in body
    assert "Alert detail" in body


def test_alert_queue_does_not_import_generator_modules():
    import dashboard.alerts as alerts_page

    text = Path(alerts_page.__file__).read_text(encoding="utf-8")
    assert "data_generator" not in text
    assert "src.generator" not in text


def test_pipeline_mode_alert_queue_is_empty():
    provider = DashboardDataProvider(mode="pipeline")
    summary = provider.get_alert_queue_summary()
    assert summary.matching_alerts == 0
    assert provider.get_alert_queue_table().empty
