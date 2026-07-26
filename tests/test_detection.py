"""Phase 9 streaming inference contracts."""

from __future__ import annotations

import inspect

import joblib
import pandas as pd
import pytest

from src.artifacts import artifact_path
from src.detection.engine import (
    FORBIDDEN_INFERENCE_COLUMNS,
    StreamingEngine,
    StreamingOrderError,
    get_default_engine,
    process_event,
    reset_default_engine,
)
from src.detection.replay import replay_events, results_to_frame
from src.drift.store import AdaptiveProfileStore
from src.features import MODEL_FEATURE_COLUMNS, build_features, compute_event_features, empty_profile
from src.generator import generate_dataset
from src.models.anomaly_detector import build_scores
from src.models.baselines import fit_rule_baseline
from src.models.classifier import classifier_training_rows, train_attack_classifier
from src.models.dataset import load_scoring_frame, training_matrix
from src.models.model import train_anomaly_model
from src.profiling import BehaviourProfile, ProfileBundle
from src.risk.aggregation import EntityRiskState
from src.risk.engine import AlertState, RiskEngine, build_scoring_frame, fit_risk_engine
from src.risk.signals import (
    assert_no_evaluation_metadata,
    fit_calibration,
    load_engine_spec,
    risk_input_columns,
)
from src.schema import FORBIDDEN_FEATURE_COLUMNS

pytestmark = pytest.mark.no_pipeline_cleanup

_BASE = pd.Timestamp("2025-03-01T08:00:00Z")
_PROFILE = {
    "name": "stream-test",
    "n_users": 8,
    "n_service_accounts": 2,
    "n_edge_devices": 2,
    "days": 14,
    "target_events": 900,
}


def _observation_event(
    i: int,
    *,
    entity: str = "u1",
    hours: float = 0.0,
    resource: str = "VPN",
    city: str = "NYC",
    country: str = "US",
    lat: float = 40.7,
    lon: float = -74.0,
    auth_success: bool = True,
    device_id: str = "dev-a",
    role: str = "analyst",
) -> pd.Series:
    return pd.Series(
        {
            "event_id": f"E{i:05d}",
            "timestamp": _BASE + pd.Timedelta(hours=hours),
            "entity_id": entity,
            "entity_type": "user",
            "role": role,
            "source_ip": "10.0.0.1",
            "country": country,
            "city": city,
            "latitude": lat,
            "longitude": lon,
            "auth_method": "SSO",
            "auth_success": auth_success,
            "resource_accessed": resource,
            "action": "access",
            "command_sequence": "",
            "session_duration_s": 120.0,
            "bytes_transferred": 1500.0,
            "device_id": device_id,
            "device_os": "Windows",
            "device_firmware": "1.0",
            "device_protocol": "HTTPS",
            "device_mac": "aa:aa:aa:aa:aa:aa",
        }
    )


def _frozen_profile(key: str = "u1") -> BehaviourProfile:
    return BehaviourProfile(
        key=key,
        n_events=10,
        login_hour_counts={"8": 10},
        location_counts={"US|NYC": 10},
        device_counts={"dev-a|aa:aa:aa:aa:aa:aa": 10},
        resource_counts={"VPN": 10},
        auth_method_counts={"SSO": 10},
        duration_mean=100.0,
        duration_std=10.0,
        bytes_mean=1000.0,
        bytes_std=100.0,
        auth_failure_rate=0.05,
        baseline_latitude=40.7,
        baseline_longitude=-74.0,
        transitions={"<START>->VPN": 1, "VPN->VPN": 9},
        total_transitions=10,
    )


@pytest.fixture(scope="module")
def stream_bundle():
    """Train a compact offline stack once for streaming contract tests."""
    generated = generate_dataset(profile=_PROFILE)
    features, bundle = build_features(generated.events)
    # Keep artifacts for the module lifetime; function-scoped cleaner would
    # otherwise delete thresholds mid-setup of dependent helpers.
    generated.events.to_parquet(artifact_path("events", ensure_parent=True), index=False)
    features.to_parquet(artifact_path("features", ensure_parent=True), index=False)
    joblib.dump(bundle, artifact_path("profiles", ensure_parent=True))

    frame = load_scoring_frame(features, generated.events)
    train = training_matrix(frame)
    anomaly = train_anomaly_model(train)
    rule = fit_rule_baseline(train)
    clf = train_attack_classifier(classifier_training_rows(frame))
    anomaly.save()
    rule.save()
    clf.save()
    scores = build_scores(frame, anomaly, rule)
    scores.to_parquet(artifact_path("event_scores", ensure_parent=True), index=False)
    scoring = build_scoring_frame(features, scores)
    risk = fit_risk_engine(scoring, scores)
    risk.save()
    return {
        "events": generated.events,
        "features": features,
        "bundle": bundle,
        "anomaly": anomaly,
        "rule": rule,
        "clf": clf,
        "risk": risk,
    }


def _engine(stream_bundle, *, apply_drift: bool = False) -> StreamingEngine:
    bundle = stream_bundle["bundle"]
    store = AdaptiveProfileStore.from_bundle(
        bundle,
        ewma_halflife_days=14.0,
        baseline_update_max_risk=40.0,
        rolling_window_days=30.0,
    )
    return StreamingEngine(
        anomaly_model=stream_bundle["anomaly"],
        rule_baseline=stream_bundle["rule"],
        classifier=stream_bundle["clf"],
        risk_engine=stream_bundle["risk"],
        profile_bundle=bundle,
        adaptive_store=store,
        smoothing=1.0,
        std_floor=1.0,
        velocity_cap=100000.0,
        history_retention_hours=0.0,
        apply_drift_updates=apply_drift,
    )


def _tiny_risk_engine() -> RiskEngine:
    rows = []
    for i in range(30):
        row = {col: 0.0 for col in MODEL_FEATURE_COLUMNS}
        row.update(
            event_id=f"C{i:04d}",
            timestamp=_BASE + pd.Timedelta(minutes=i),
            entity_id="cal",
            entity_type="user",
            anomaly_score_raw=0.4 + 0.01 * (i % 5),
            baseline_rule=0.0,
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
        rows.append(row)
    frame = pd.DataFrame(rows)
    scoring = frame.loc[:, list(risk_input_columns())].copy()
    assert_no_evaluation_metadata(scoring.columns)
    spec = load_engine_spec()
    calibration = fit_calibration(
        scoring.assign(split="train", label="BENIGN"),
        {"sensitive": 0.5, "balanced": 0.6, "strict": 0.7},
        spec=spec,
    )
    return RiskEngine(
        spec=spec,
        calibration=calibration,
        severity_bands={
            "LOW": (0, 30),
            "MEDIUM": (31, 60),
            "HIGH": (61, 80),
            "CRITICAL": (81, 100),
        },
        min_alert_severity="HIGH",
        cooldown_seconds=3600,
        escalation_bypasses_cooldown=True,
        max_reasons=6,
    )


def test_forbidden_inference_columns_cover_labels_and_eval_metadata():
    for col in FORBIDDEN_FEATURE_COLUMNS:
        assert col in FORBIDDEN_INFERENCE_COLUMNS
    for col in ("campaign_id", "difficulty", "stealthy", "split"):
        assert col in FORBIDDEN_INFERENCE_COLUMNS


def test_process_event_strips_labels_and_still_scores(stream_bundle):
    engine = _engine(stream_bundle, apply_drift=False)
    row = stream_bundle["events"].iloc[0].copy()
    assert "label" in row.index
    result = engine.process_event(row)
    assert result.event_id == str(row.event_id)
    assert 0.0 <= result.risk_score <= 100.0
    assert result.severity in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}


def test_streaming_features_exclude_split_metadata(stream_bundle):
    engine = _engine(stream_bundle, apply_drift=False)
    row = stream_bundle["events"].iloc[0].copy()
    cleaned = engine._strip_forbidden(row)
    feature_dict = compute_event_features(
        cleaned,
        history=[],
        previous_fingerprint=None,
        profile=_frozen_profile(),
        profile_source="entity",
        profile_confidence=1.0,
        smoothing=1.0,
        std_floor=1.0,
        velocity_cap=100000.0,
        cutoff=None,
    )
    assert "split" not in feature_dict
    result = engine.process_event(row)
    assert result.event_id == str(row.event_id)


def test_label_flip_does_not_change_risk_score(stream_bundle):
    engine = _engine(stream_bundle, apply_drift=False)
    row = stream_bundle["events"].iloc[0].copy()
    a = engine.process_event(row)
    engine.reset_runtime_state()
    flipped = row.copy()
    flipped["label"] = "BRUTE_FORCE"
    flipped["is_attack"] = True
    flipped["campaign_id"] = "FAKE"
    b = engine.process_event(flipped)
    assert a.risk_score == b.risk_score
    assert a.anomaly_score_raw == b.anomaly_score_raw


def test_reset_runtime_state_rewinds_adaptive_store(stream_bundle):
    engine = _engine(stream_bundle, apply_drift=True)
    sample = (
        stream_bundle["events"]
        .sort_values(["timestamp", "event_id"], kind="stable")
        .head(8)
    )
    first, _ = replay_events(sample, engine=engine, apply_drift_updates=True)
    engine.reset_runtime_state()
    second, _ = replay_events(sample, engine=engine, apply_drift_updates=True)
    assert [r.risk_score for r in first] == [r.risk_score for r in second]
    assert [r.profile_updated for r in first] == [r.profile_updated for r in second]


def test_empty_profile_does_not_inflate_first_event_zscores():
    row = _observation_event(0, hours=0)
    profile = empty_profile(row)
    feats = compute_event_features(
        row,
        history=[],
        previous_fingerprint=None,
        profile=profile,
        profile_source="none",
        profile_confidence=0.0,
        smoothing=1.0,
        std_floor=1.0,
        velocity_cap=100000.0,
        cutoff=None,
    )
    assert feats["session_duration_zscore"] == pytest.approx(0.0)
    assert feats["transfer_volume_zscore"] == pytest.approx(0.0)


def test_streaming_rejects_out_of_order_events_per_entity(stream_bundle):
    engine = _engine(stream_bundle, apply_drift=False)
    entity = str(stream_bundle["events"].iloc[0].entity_id)
    rows = (
        stream_bundle["events"][stream_bundle["events"].entity_id == entity]
        .sort_values(["timestamp", "event_id"], kind="stable")
        .head(3)
    )
    engine.process_event(rows.iloc[0])
    engine.process_event(rows.iloc[1])
    with pytest.raises(StreamingOrderError):
        engine.process_event(rows.iloc[0])


def test_entity_state_is_isolated(stream_bundle):
    engine = _engine(stream_bundle, apply_drift=False)
    entities = stream_bundle["events"]["entity_id"].astype(str).unique()[:2]
    a = (
        stream_bundle["events"][stream_bundle["events"].entity_id == entities[0]]
        .sort_values("timestamp", kind="stable")
        .iloc[0]
    )
    b = (
        stream_bundle["events"][stream_bundle["events"].entity_id == entities[1]]
        .sort_values("timestamp", kind="stable")
        .iloc[0]
    )
    engine.process_event(a)
    assert entities[0] in engine.histories
    assert entities[1] not in engine.histories
    engine.process_event(b)
    assert len(engine.histories[entities[0]]) == 1
    assert len(engine.histories[entities[1]]) == 1


def test_features_are_causal_history_excludes_current_and_future():
    event = _observation_event(0, hours=0)
    future = _observation_event(1, hours=0.05, resource="Payroll")  # ~3 minutes later
    profile = _frozen_profile()
    feats = compute_event_features(
        event,
        history=[],
        previous_fingerprint=None,
        profile=profile,
        profile_source="entity",
        profile_confidence=1.0,
        smoothing=1.0,
        std_floor=1.0,
        velocity_cap=100000.0,
    )
    assert feats["event_count_5m"] == 0.0
    feats2 = compute_event_features(
        future,
        history=[event],
        previous_fingerprint="dev-a|aa:aa:aa:aa:aa:aa",
        profile=profile,
        profile_source="entity",
        profile_confidence=1.0,
        smoothing=1.0,
        std_floor=1.0,
        velocity_cap=100000.0,
    )
    assert feats2["event_count_5m"] == 1.0


def test_deterministic_replay(stream_bundle):
    sample = (
        stream_bundle["events"]
        .sort_values(["timestamp", "event_id"], kind="stable")
        .head(40)
        .reset_index(drop=True)
    )
    eng_a = _engine(stream_bundle, apply_drift=False)
    eng_b = _engine(stream_bundle, apply_drift=False)
    ra, _ = replay_events(sample, engine=eng_a, apply_drift_updates=False)
    rb, _ = replay_events(sample, engine=eng_b, apply_drift_updates=False)
    assert [r.risk_score for r in ra] == [r.risk_score for r in rb]
    assert [r.anomaly_score_raw for r in ra] == [r.anomaly_score_raw for r in rb]
    assert [r.alerted for r in ra] == [r.alerted for r in rb]


def test_offline_streaming_anomaly_consistency(stream_bundle):
    features = stream_bundle["features"]
    events = stream_bundle["events"]
    anomaly = stream_bundle["anomaly"]
    sample_ids = (
        features.sort_values(["timestamp", "event_id"], kind="stable")
        .head(40)["event_id"]
        .tolist()
    )
    max_ts = events.loc[events.event_id.isin(sample_ids), "timestamp"].max()
    entity_set = set(events.loc[events.event_id.isin(sample_ids), "entity_id"].astype(str))
    history = events[
        (events.entity_id.isin(entity_set)) & (events.timestamp <= max_ts)
    ].sort_values(["timestamp", "event_id"], kind="stable")

    engine = _engine(stream_bundle, apply_drift=False)
    results, _ = replay_events(history, engine=engine, apply_drift_updates=False)
    by_id = {r.event_id: r for r in results}
    offline = features.set_index("event_id")
    for eid in sample_ids:
        offline_row = offline.loc[eid]
        expected = float(anomaly.raw_scores(offline_row.to_frame().T)[0])
        assert by_id[eid].anomaly_score_raw == pytest.approx(expected, rel=1e-9, abs=1e-9)


def test_cold_start_new_entity_does_not_crash(stream_bundle):
    engine = _engine(stream_bundle, apply_drift=False)
    row = stream_bundle["events"].iloc[0].copy()
    row["event_id"] = "COLD-START-001"
    row["entity_id"] = "BRAND-NEW-ENTITY"
    result = engine.process_event(row)
    assert result.profile_source == "cohort"
    assert result.profile_confidence == 0.0
    assert 0.0 <= result.risk_score <= 100.0


def test_alert_cooldown_suppresses_second_critical_event():
    risk = _tiny_risk_engine()
    row = {col: 100.0 for col in MODEL_FEATURE_COLUMNS}
    row.update(
        event_id="ALERT-1",
        timestamp=_BASE,
        entity_id="u1",
        entity_type="user",
        anomaly_score_raw=100.0,
        baseline_rule=100.0,
        is_known_device=0.0,
    )
    first_score, first_alert, first_risk, first_alert_state, first_suppressed = (
        risk.score_event(pd.Series(row))
    )
    row["event_id"] = "ALERT-2"
    row["timestamp"] = _BASE + pd.Timedelta(minutes=1)
    second_score, second_alert, _, _, second_suppressed = risk.score_event(
        pd.Series(row),
        risk_state=first_risk,
        alert_state=first_alert_state,
    )

    assert first_score["risk_severity"] == "CRITICAL"
    assert first_alert is not None
    assert first_suppressed is False
    assert second_score["risk_severity"] == "CRITICAL"
    assert second_alert is None
    assert second_suppressed is True


def test_risk_persistence_decays_across_gap():
    risk = _tiny_risk_engine()
    state = EntityRiskState(evidence=4.0, last_timestamp=float(_BASE.timestamp()))
    row = {col: 0.0 for col in MODEL_FEATURE_COLUMNS}
    row.update(
        event_id="P1",
        timestamp=_BASE + pd.Timedelta(hours=48),
        entity_id="u1",
        entity_type="user",
        anomaly_score_raw=0.4,
        baseline_rule=0.0,
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
    score, _, new_state, _, _ = risk.score_event(
        pd.Series(row), risk_state=state, alert_state=AlertState()
    )
    assert new_state.evidence < 4.0 + score["instantaneous_evidence"]
    assert score["persistence_contribution"] >= 0.0


def test_adaptive_profile_gate_blocks_high_risk(stream_bundle):
    engine = _engine(stream_bundle, apply_drift=True)
    engine.adaptive_store.baseline_update_max_risk = 5.0
    row = stream_bundle["events"].sort_values("timestamp", kind="stable").iloc[10]
    before = engine.adaptive_store.n_blocked
    result = engine.process_event(row)
    if result.risk_score >= 5.0:
        assert result.profile_updated is False
        assert engine.adaptive_store.n_blocked == before + 1
    else:
        assert result.profile_updated is True


def test_adaptive_low_risk_updates_profile(stream_bundle):
    engine = _engine(stream_bundle, apply_drift=True)
    engine.adaptive_store.baseline_update_max_risk = 99.0
    row = stream_bundle["events"].sort_values("timestamp", kind="stable").iloc[5]
    result = engine.process_event(row)
    if result.risk_score < 99.0:
        assert result.profile_updated is True


def test_process_event_signature_has_no_label_parameter():
    params = inspect.signature(process_event).parameters
    assert "label" not in params
    assert "campaign_id" not in params
    body = inspect.getsource(StreamingEngine.process_event)
    assert "row.label" not in body


def test_module_process_event_reuses_stateful_default_engine(
    stream_bundle, monkeypatch
):
    engine = _engine(stream_bundle, apply_drift=False)
    reset_default_engine()
    monkeypatch.setattr(
        "src.detection.engine.StreamingEngine.load",
        lambda: engine,
    )
    entity = str(stream_bundle["events"].iloc[0].entity_id)
    rows = (
        stream_bundle["events"][stream_bundle["events"].entity_id == entity]
        .sort_values(["timestamp", "event_id"], kind="stable")
        .head(2)
    )

    process_event(rows.iloc[0])
    process_event(rows.iloc[1])

    assert get_default_engine() is engine
    assert len(engine.histories[entity]) == 2
    reset_default_engine()


def test_results_frame_roundtrip_columns(stream_bundle):
    engine = _engine(stream_bundle, apply_drift=False)
    sample = stream_bundle["events"].sort_values(
        ["timestamp", "event_id"], kind="stable"
    ).head(5)
    results, _ = replay_events(sample, engine=engine, apply_drift_updates=False)
    frame = results_to_frame(results)
    assert {
        "event_id",
        "risk_score",
        "alerted",
        "profile_updated",
        "latency_ms",
    }.issubset(frame.columns)
    assert len(frame) == 5
