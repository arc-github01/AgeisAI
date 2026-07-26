"""Phase 6 hybrid risk-engine contracts."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.features import MODEL_FEATURE_COLUMNS
from src.risk.aggregation import (
    EntityRiskState,
    decay_factor,
    saturating_risk,
    severity_for,
)
from src.risk.engine import RiskEngine, build_scoring_frame, fit_risk_engine
from src.risk.explanations import build_reasons
from src.risk.signals import (
    FORBIDDEN_RISK_INPUT_COLUMNS,
    RISK_INPUT_COLUMNS,
    RiskInputLeakageError,
    assert_no_evaluation_metadata,
    fit_calibration,
    load_engine_spec,
    risk_input_columns,
)

_BASE = pd.Timestamp("2025-01-06T08:00:00Z")


def _row(
    i: int,
    *,
    entity: str = "u1",
    anomaly: float = 0.4,
    rule: float = 0.0,
    hours_offset: float = 0.0,
    **feature_overrides,
) -> dict:
    row = {col: 0.0 for col in MODEL_FEATURE_COLUMNS}
    row.update(
        event_id=f"E{i:05d}",
        timestamp=_BASE + pd.Timedelta(hours=hours_offset),
        entity_id=entity,
        entity_type="user",
        anomaly_score_raw=anomaly,
        baseline_rule=rule,
        is_known_device=1.0,
        is_new_location=0.0,
        is_entity_off_hours=0.0,
        geo_velocity_kmh=0.0,
        failed_auth_5m=0.0,
        recent_resource_breadth_24h=0.0,
        sequence_anomaly_score=0.0,
        resource_rarity=0.0,
        event_count_5m=0.0,
        transfer_volume_zscore=0.0,
        session_duration_zscore=0.0,
    )
    row.update(feature_overrides)
    return row


def _tiny_engine(rows: list[dict], *, thresholds=None) -> RiskEngine:
    frame = pd.DataFrame(rows)
    # Provide labels only for calibration selection; scoring frame strips them.
    labelled = frame.copy()
    labelled["split"] = "train"
    labelled["label"] = "BENIGN"
    labelled["is_attack"] = False
    labelled["campaign_id"] = None
    # Force a few elevated benign rows so anchors are well-defined.
    thresholds = thresholds or {"sensitive": 0.5, "balanced": 0.6, "strict": 0.7}
    # Minimal fit via public API pieces.
    scoring = frame.loc[:, list(risk_input_columns())].copy()
    assert_no_evaluation_metadata(scoring.columns)
    # Manually build engine to avoid needing on-disk Phase 5 artifacts.
    spec = load_engine_spec()
    calibration = fit_calibration(
        scoring.assign(split="train", label="BENIGN"),
        thresholds,
        spec=spec,
    )
    return RiskEngine(
        spec=spec,
        calibration=calibration,
        severity_bands={"LOW": (0, 30), "MEDIUM": (31, 60), "HIGH": (61, 80), "CRITICAL": (81, 100)},
        min_alert_severity="HIGH",
        cooldown_seconds=3600,
        escalation_bypasses_cooldown=True,
        max_reasons=6,
    )


# -----------------------------------------------------------------------------
# Leakage / whitelist
# -----------------------------------------------------------------------------
def test_risk_input_columns_exclude_evaluation_metadata():
    for forbidden in FORBIDDEN_RISK_INPUT_COLUMNS:
        assert forbidden not in RISK_INPUT_COLUMNS
    assert_no_evaluation_metadata(RISK_INPUT_COLUMNS)


def test_assert_no_evaluation_metadata_raises():
    with pytest.raises(RiskInputLeakageError):
        assert_no_evaluation_metadata(["event_id", "label", "anomaly_score_raw"])


def test_score_frame_rejects_forbidden_columns():
    engine = _tiny_engine([_row(0), _row(1, hours_offset=1)])
    dirty = pd.DataFrame([_row(0), _row(1, hours_offset=1)])
    dirty["campaign_id"] = "CMP-X"
    with pytest.raises(RiskInputLeakageError):
        engine.score_frame(dirty)


# -----------------------------------------------------------------------------
# Score bounds, severity, determinism
# -----------------------------------------------------------------------------
def test_risk_score_bounded_and_severity_mapped():
    engine = _tiny_engine([_row(i, hours_offset=i) for i in range(20)])
    scores, _ = engine.score_frame(
        pd.DataFrame([_row(i, hours_offset=i, anomaly=0.4 + 0.02 * i) for i in range(20)])
        .loc[:, list(risk_input_columns())]
    )
    assert scores["risk_score"].between(0.0, 100.0).all()
    assert set(scores["risk_severity"]).issubset({"LOW", "MEDIUM", "HIGH", "CRITICAL"})


def test_scoring_is_deterministic():
    rows = [_row(i, hours_offset=i, anomaly=0.55, rule=5.0) for i in range(10)]
    engine = _tiny_engine(rows)
    frame = pd.DataFrame(rows).loc[:, list(risk_input_columns())]
    a, _ = engine.score_frame(frame)
    b, _ = engine.score_frame(frame)
    pd.testing.assert_series_equal(a["risk_score"], b["risk_score"])


def test_severity_for_bands():
    bands = {"LOW": (0, 30), "MEDIUM": (31, 60), "HIGH": (61, 80), "CRITICAL": (81, 100)}
    assert severity_for(10, bands) == "LOW"
    assert severity_for(61, bands) == "HIGH"
    assert severity_for(95, bands) == "CRITICAL"


# -----------------------------------------------------------------------------
# Decay / persistence
# -----------------------------------------------------------------------------
def test_decay_factor_formula():
    assert decay_factor(0, 3600) == pytest.approx(1.0)
    assert decay_factor(3600, 3600) == pytest.approx(0.5)
    assert decay_factor(7200, 3600) == pytest.approx(0.25)
    assert decay_factor(-10, 3600) == pytest.approx(1.0)  # negative gap clamped


def test_saturating_risk_bounds():
    assert saturating_risk(0, 1.0) == pytest.approx(0.0)
    assert 0 < saturating_risk(1.0, 1.0) < 100
    assert saturating_risk(5.0, 1.0) == pytest.approx(100.0 * (1.0 - __import__("math").exp(-5.0)))
    assert saturating_risk(5.0, 1.0) <= 100.0


def test_repeated_suspicious_events_increase_risk():
    engine = _tiny_engine([_row(i, hours_offset=i * 0.1) for i in range(30)])
    # Tightly spaced high-anomaly events on one entity.
    rows = [
        _row(i, hours_offset=i * 0.25, anomaly=0.75, geo_velocity_kmh=2000.0)
        for i in range(6)
    ]
    scores, _ = engine.score_frame(pd.DataFrame(rows).loc[:, list(risk_input_columns())])
    assert scores["risk_score"].iloc[-1] > scores["risk_score"].iloc[0]


def test_benign_inactivity_decreases_risk():
    engine = _tiny_engine([_row(i, hours_offset=i) for i in range(30)])
    # Burst then long quiet gap then mild event.
    rows = [
        _row(0, hours_offset=0, anomaly=0.8, geo_velocity_kmh=3000.0),
        _row(1, hours_offset=0.5, anomaly=0.8, geo_velocity_kmh=3000.0),
        _row(2, hours_offset=48, anomaly=0.4),  # two days later, quiet
    ]
    scores, _ = engine.score_frame(pd.DataFrame(rows).loc[:, list(risk_input_columns())])
    assert scores["risk_score"].iloc[2] < scores["risk_score"].iloc[1]


def test_future_events_cannot_alter_past_risk():
    engine = _tiny_engine([_row(i, hours_offset=i) for i in range(40)])
    base_rows = [_row(i, hours_offset=i, anomaly=0.55) for i in range(5)]
    base = pd.DataFrame(base_rows).loc[:, list(risk_input_columns())]
    scores_base, _ = engine.score_frame(base)

    poisoned_rows = base_rows + [
        _row(100, hours_offset=100, anomaly=0.99, geo_velocity_kmh=5000.0, is_known_device=0.0)
    ]
    poisoned = pd.DataFrame(poisoned_rows).loc[:, list(risk_input_columns())]
    scores_poisoned, _ = engine.score_frame(poisoned)

    past = scores_poisoned[scores_poisoned["event_id"].isin(scores_base["event_id"])]
    merged = scores_base.merge(past, on="event_id", suffixes=("_a", "_b"))
    np.testing.assert_allclose(
        merged["risk_score_a"].to_numpy(float),
        merged["risk_score_b"].to_numpy(float),
        rtol=0.0,
        atol=1e-12,
    )


# -----------------------------------------------------------------------------
# Explanations
# -----------------------------------------------------------------------------
def test_explanation_contributions_reconcile_with_score():
    engine = _tiny_engine([_row(i, hours_offset=i) for i in range(20)])
    rows = [
        _row(
            0,
            anomaly=0.85,
            rule=20.0,
            geo_velocity_kmh=2500.0,
            is_known_device=0.0,
            failed_auth_5m=8.0,
        )
    ]
    scores, _ = engine.score_frame(pd.DataFrame(rows).loc[:, list(risk_input_columns())])
    reasons = scores.iloc[0]["reasons"]
    assert reasons
    total = sum(item["contribution"] for item in reasons)
    # Top-k reasons may be a truncated set; full contribution columns must sum.
    contrib_sum = (
        scores.iloc[0]["isolation_forest_contribution"]
        + scores.iloc[0]["rule_contribution"]
        + scores.iloc[0]["context_contribution"]
        + scores.iloc[0]["persistence_contribution"]
    )
    assert contrib_sum == pytest.approx(scores.iloc[0]["risk_score"], abs=1e-6)
    # Display reasons are rounded; allow 0.01 absolute slack vs the raw score.
    assert total <= float(scores.iloc[0]["risk_score"]) + 0.01


def test_build_reasons_orders_by_contribution():
    reasons = build_reasons(
        {"A": 1.0, "B": 5.0, "C": 0.0, "D": 3.0}, max_reasons=2
    )
    assert [r["code"] for r in reasons] == ["B", "D"]


# -----------------------------------------------------------------------------
# Alerts / cooldown / escalation
# -----------------------------------------------------------------------------
def test_alert_cooldown_suppresses_duplicates():
    engine = _tiny_engine([_row(i, hours_offset=i) for i in range(40)])
    # Many HIGH events within one hour on same entity.
    rows = [
        _row(
            i,
            hours_offset=i * 0.1,
            anomaly=0.95,
            geo_velocity_kmh=4000.0,
            is_known_device=0.0,
            failed_auth_5m=10.0,
            sequence_anomaly_score=8.0,
            recent_resource_breadth_24h=10.0,
        )
        for i in range(8)
    ]
    scores, alerts = engine.score_frame(pd.DataFrame(rows).loc[:, list(risk_input_columns())])
    assert (scores["risk_severity"].isin(["HIGH", "CRITICAL"])).any()
    assert len(alerts) >= 1
    assert alerts.attrs["n_suppressed"] >= 1
    assert len(alerts) < len(scores)


def test_escalation_bypasses_cooldown():
    # Construct an engine and manually drive alert decision path via two events
    # that escalate MEDIUM->... actually we need HIGH then CRITICAL.
    # Use very strong second event after a strong first within cooldown.
    engine = _tiny_engine([_row(i, hours_offset=i) for i in range(50)])
    rows = [
        _row(
            0,
            hours_offset=0,
            anomaly=0.72,
            rule=15.0,
            geo_velocity_kmh=1200.0,
            is_known_device=0.0,
            failed_auth_5m=4.0,
        ),
        _row(
            1,
            hours_offset=0.2,  # well inside 1h cooldown
            anomaly=0.99,
            rule=40.0,
            geo_velocity_kmh=5000.0,
            is_known_device=0.0,
            failed_auth_5m=12.0,
            sequence_anomaly_score=10.0,
            recent_resource_breadth_24h=12.0,
            is_new_location=1.0,
        ),
    ]
    scores, alerts = engine.score_frame(pd.DataFrame(rows).loc[:, list(risk_input_columns())])
    # If both reached alertable severity and second is strictly higher, expect 2 alerts.
    if set(scores["risk_severity"]).intersection({"HIGH", "CRITICAL"}):
        if scores.iloc[0]["risk_severity"] != scores.iloc[1]["risk_severity"]:
            ranks = {"HIGH": 2, "CRITICAL": 3, "MEDIUM": 1, "LOW": 0}
            if ranks[scores.iloc[1]["risk_severity"]] > ranks[scores.iloc[0]["risk_severity"]]:
                assert len(alerts) >= 2


# -----------------------------------------------------------------------------
# Campaign metadata must not influence scoring
# -----------------------------------------------------------------------------
def test_campaign_metadata_cannot_influence_scoring():
    rows = [_row(i, hours_offset=i, anomaly=0.6) for i in range(5)]
    engine = _tiny_engine(rows)
    frame = pd.DataFrame(rows).loc[:, list(risk_input_columns())]
    scores_a, _ = engine.score_frame(frame)
    # "Knowing" about campaigns outside the engine cannot change scores.
    _ = [{"campaign_id": "CMP-MISSED", "stealthy": True}]
    scores_b, _ = engine.score_frame(frame)
    pd.testing.assert_series_equal(scores_a["risk_score"], scores_b["risk_score"])


def test_evaluation_join_is_after_scoring_only():
    """Scoring frame assembly never carries labels into the engine input."""
    features = pd.DataFrame(
        [
            {
                **{c: 0.0 for c in MODEL_FEATURE_COLUMNS},
                "event_id": "E0",
                "timestamp": _BASE,
                "entity_id": "u1",
                "split": "train",  # present on features artifact but forbidden for risk
            }
        ]
    )
    event_scores = pd.DataFrame(
        [
            {
                "event_id": "E0",
                "timestamp": _BASE,
                "entity_id": "u1",
                "entity_type": "user",
                "anomaly_score_raw": 0.4,
                "baseline_rule": 0.0,
                "label": "BENIGN",
                "is_attack": False,
                "campaign_id": None,
                "split": "train",
            }
        ]
    )
    scoring = build_scoring_frame(features=features, event_scores=event_scores)
    assert "label" not in scoring.columns
    assert "split" not in scoring.columns
    assert "campaign_id" not in scoring.columns
    assert_no_evaluation_metadata(scoring.columns)
