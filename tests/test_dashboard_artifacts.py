"""Dashboard loading against realistic pipeline-shaped artifacts."""

from __future__ import annotations

import json

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from dashboard.data_provider import DashboardDataProvider
from dashboard.contracts import ATTACK_DISPLAY_NAMES
from dashboard.replay_service import StreamingReplayService
from dashboard.state import DashboardContext
from src.artifacts import artifact_path
from src.paths import PROJECT_ROOT

APP = str(PROJECT_ROOT / "app.py")


def _write_min_pipeline_artifacts() -> None:
    events = pd.DataFrame(
        [
            {
                "event_id": "EVT_1",
                "timestamp": "2025-01-10T10:00:00Z",
                "entity_id": "USR_001",
                "entity_type": "user",
                "source_ip": "10.0.0.1",
                "city": "Austin",
                "country": "US",
                "resource_accessed": "vpn-gateway",
                "auth_method": "password",
                "auth_success": True,
                "session_duration_s": 120.0,
                "device_id": "DEV_A",
            },
            {
                "event_id": "EVT_2",
                "timestamp": "2025-01-10T22:15:00Z",
                "entity_id": "USR_001",
                "entity_type": "user",
                "source_ip": "10.0.0.9",
                "city": "Berlin",
                "country": "DE",
                "resource_accessed": "finance-db",
                "auth_method": "password",
                "auth_success": True,
                "session_duration_s": 90.0,
                "device_id": "DEV_B",
            },
        ]
    )
    alerts = pd.DataFrame(
        [
            {
                "alert_id": "ALT_1",
                "timestamp": "2025-01-10T22:15:00Z",
                "entity_id": "USR_001",
                "entity_type": "user",
                "event_id": "EVT_2",
                "anomaly_score": 0.91,
                "sequence_score": 0.44,
                "attack_type": "IMPOSSIBLE_TRAVEL",
                "attack_confidence": 0.82,
                "risk_score": 88.0,
                "severity": "CRITICAL",
                "short_reason": "PERSISTENT_SUSPICIOUS_ACTIVITY",
                "reasons": (
                    "PERSISTENT_SUSPICIOUS_ACTIVITY (+36.5) + "
                    "RULE_THRESHOLD_EXCEEDED (+22.2)"
                ),
                "reason_codes": json.dumps(
                    ["PERSISTENT_SUSPICIOUS_ACTIVITY", "RULE_THRESHOLD_EXCEEDED"]
                ),
                "top_contributors": json.dumps(
                    [
                        {"code": "PERSISTENT_SUSPICIOUS_ACTIVITY", "contribution": 36.5},
                        {"code": "RULE_THRESHOLD_EXCEEDED", "contribution": 22.2},
                        {"code": "ISOLATION_FOREST_ANOMALY", "contribution": 7.1},
                    ]
                ),
            }
        ]
    )
    risk_scores = pd.DataFrame(
        [
            {
                "event_id": "EVT_1",
                "timestamp": "2025-01-10T10:00:00Z",
                "entity_id": "USR_001",
                "risk_score": 12.0,
                "risk_severity": "LOW",
                "isolation_forest_contribution": 4.0,
                "rule_contribution": 0.0,
                "context_contribution": 2.0,
                "persistence_contribution": 0.0,
            },
            {
                "event_id": "EVT_2",
                "timestamp": "2025-01-10T22:15:00Z",
                "entity_id": "USR_001",
                "risk_score": 88.0,
                "risk_severity": "CRITICAL",
                "isolation_forest_contribution": 7.1,
                "rule_contribution": 22.2,
                "context_contribution": 11.0,
                "persistence_contribution": 36.5,
            },
        ]
    )
    entities = {
        "profile": "test",
        "generated_at": "2025-01-01T00:00:00Z",
        "master_seed": 1,
        "dataset_summary": {},
        "entities": [
            {
                "entity_id": "USR_001",
                "entity_type": "user",
                "role": "analyst",
                "home_country": "US",
                "home_city": "Austin",
                "devices": ["DEV_A"],
                "resources": ["vpn-gateway", "mail"],
                "auth_methods": ["password", "mfa"],
                "preferred_login_hour": 9,
                "session_duration_mean_s": 300.0,
            }
        ],
    }

    events.to_parquet(artifact_path("events", ensure_parent=True), index=False)
    alerts.to_parquet(artifact_path("alerts", ensure_parent=True), index=False)
    risk_scores.to_parquet(artifact_path("risk_scores", ensure_parent=True), index=False)
    artifact_path("entities", ensure_parent=True).write_text(
        json.dumps(entities), encoding="utf-8"
    )


def test_provider_loads_pipeline_artifacts_with_explainability():
    _write_min_pipeline_artifacts()
    provider = DashboardDataProvider(DashboardContext.build(), mode="pipeline")

    assert not provider.is_mock
    assert provider.has_data
    assert len(provider.get_alerts()) == 1
    assert len(provider.get_entities()) == 1
    assert provider.get_entities().iloc[0]["entity_id"] == "USR_001"

    alert = provider.get_alert_by_id("ALT_1")
    assert alert is not None
    assert alert["event_id"] == "EVT_2"
    assert provider.parse_reason_codes(alert) == [
        "PERSISTENT_SUSPICIOUS_ACTIVITY",
        "RULE_THRESHOLD_EXCEEDED",
    ]

    contributions = provider.get_score_contributions(alert)
    assert not contributions.empty
    assert "Persistent suspicious activity" in contributions["factor"].tolist()
    # Real hybrid contributions are absolute points, not forced to sum to 1.
    assert contributions["contribution"].sum() == pytest.approx(65.8, abs=0.1)

    timeline = provider.get_entity_risk_timeline("USR_001")
    assert timeline is not None and len(timeline) == 2
    assert float(timeline["risk_score"].iloc[-1]) == 88.0

    event = provider.get_alert_event_context(alert)
    assert event is not None
    assert event["city"] == "Berlin"

    assert provider.severity_distribution_source() == "scored events"
    severity = provider.get_severity_distribution()
    assert severity.get("LOW") == 1
    assert severity.get("CRITICAL") == 1
    assert ATTACK_DISPLAY_NAMES["BENIGN"] == "Benign (no attack class)"


def test_provider_empty_pipeline_fails_gracefully():
    provider = DashboardDataProvider(mode="pipeline")
    assert not provider.has_data
    assert provider.get_overview_kpis().active_alerts == 0
    assert provider.get_alert_queue_table().empty
    assert provider.get_entity_risk_timeline("missing") is None
    assert provider.list_entity_ids() == []


def test_streaming_replay_page_renders_without_prerequisites():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value("Streaming Replay").run()
    assert not at.exception
    assert at.button
    labels = [b.label for b in at.button]
    assert "Replay next batch" in labels
    inject = [b for b in at.button if b.label == "Replay next batch"]
    assert inject and inject[0].disabled


def test_streaming_replay_service_reports_missing_prerequisites():
    service = StreamingReplayService(DashboardContext.build())
    assert not service.is_ready()
    outcome = service.replay_batch(
        engine=object(),  # type: ignore[arg-type]
        cursor=0,
        batch_size=5,
    )
    assert not outcome.success
    assert outcome.error and "prerequisites missing" in outcome.error.lower()


def test_alert_queue_and_overview_use_pipeline_when_artifacts_exist():
    _write_min_pipeline_artifacts()
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value("SOC Overview").run()
    assert not at.exception
    captions = " ".join(c.value for c in at.caption)
    assert "pipeline artifacts" in captions.lower()
    assert not any("Development fixture active" in i.value for i in at.info)

    at.sidebar.radio[0].set_value("Alert Queue").run()
    assert not at.exception
    assert at.dataframe
    body = " ".join(block.value for block in at.markdown)
    assert "Alert investigation" in body or "Why this looked unusual" in body
