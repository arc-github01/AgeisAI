"""Phase 8 concept-drift contracts: risk gate, causality, determinism."""

from __future__ import annotations

import inspect

import pandas as pd
import pytest

from src.drift.evaluate import evaluate_drift
from src.drift.store import AdaptiveEntityProfile, AdaptiveProfileStore
from src.drift.update import apply_event_to_profile, ewma_decay, maybe_update, replay
from src.profiling import BehaviourProfile, ProfileBundle
from src.drift import update as update_mod


_BASE = pd.Timestamp("2025-02-01T08:00:00Z")


def _frozen_profile(key: str = "u1") -> BehaviourProfile:
    return BehaviourProfile(
        key=key,
        n_events=20,
        login_hour_counts={"8": 10, "9": 10},
        location_counts={"US|NYC": 20},
        device_counts={"dev-a|aa:aa:aa:aa:aa:aa": 20},
        resource_counts={"VPN": 15, "Email": 5},
        auth_method_counts={"SSO": 20},
        duration_mean=100.0,
        duration_std=10.0,
        bytes_mean=1000.0,
        bytes_std=100.0,
        auth_failure_rate=0.05,
        baseline_latitude=40.7,
        baseline_longitude=-74.0,
        transitions={"<START>->VPN": 1, "VPN->Email": 5, "Email->VPN": 4},
        total_transitions=20,
    )


def _bundle() -> ProfileBundle:
    return ProfileBundle(
        cutoff=_BASE - pd.Timedelta(days=1),
        entity_profiles={"u1": _frozen_profile("u1")},
        cohort_profiles={},
        cohort_keys=("entity_type", "role"),
        min_events_for_personal=5,
    )


def _store(*, max_risk: float = 40.0) -> AdaptiveProfileStore:
    return AdaptiveProfileStore.from_bundle(
        _bundle(),
        ewma_halflife_days=14.0,
        baseline_update_max_risk=max_risk,
        rolling_window_days=30.0,
    )


def _event(
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
    label: str = "BENIGN",
) -> pd.Series:
    return pd.Series(
        {
            "event_id": f"E{i:05d}",
            "timestamp": _BASE + pd.Timedelta(hours=hours),
            "entity_id": entity,
            "entity_type": "user",
            "role": "analyst",
            "country": country,
            "city": city,
            "latitude": lat,
            "longitude": lon,
            "auth_method": "SSO",
            "auth_success": auth_success,
            "resource_accessed": resource,
            "session_duration_s": 120.0,
            "bytes_transferred": 1500.0,
            "device_id": "dev-a",
            "device_mac": "aa:aa:aa:aa:aa:aa",
            "label": label,
            "is_attack": label not in ("BENIGN", "INSIDER_DRIFT"),
        }
    )


# -----------------------------------------------------------------------------
# Unit: EWMA math + gate
# -----------------------------------------------------------------------------
def test_ewma_decay_half_life():
    assert ewma_decay(0.0, 14.0) == 1.0
    assert ewma_decay(14.0, 14.0) == pytest.approx(0.5)
    assert ewma_decay(28.0, 14.0) == pytest.approx(0.25)


def test_high_risk_event_does_not_mutate_profile():
    store = _store(max_risk=40.0)
    before = store.entity_profiles["u1"].copy()
    row = _event(1, resource="Payroll")
    updated = maybe_update(store, row, risk_score=85.0)
    assert updated is False
    after = store.entity_profiles["u1"]
    assert after.resource_counts == before.resource_counts
    assert after.n_events == before.n_events
    assert after.n_updates == 0
    assert store.n_blocked == 1
    assert "Payroll" not in after.resource_counts


def test_low_risk_event_adapts_counts():
    store = _store(max_risk=40.0)
    row = _event(1, resource="Payroll", city="Boston", lat=42.3, lon=-71.0)
    updated = maybe_update(store, row, risk_score=10.0)
    assert updated is True
    after = store.entity_profiles["u1"]
    assert after.n_updates == 1
    assert after.resource_counts.get("Payroll", 0) >= 1.0
    assert any("Boston" in k for k in after.location_counts)
    assert after.n_events > 20.0  # decayed prior + 1


def test_maybe_update_signature_has_no_label_parameter():
    params = inspect.signature(maybe_update).parameters
    assert "label" not in params
    assert "is_attack" not in params
    assert "campaign_id" not in params
    # Body must not branch on ground-truth columns.
    body = inspect.getsource(update_mod.maybe_update).split(":\n", 1)[1]
    assert "row.label" not in body
    assert 'row["label"]' not in body
    assert "is_attack" not in body
    assert "campaign_id" not in body


def test_labels_do_not_gate_when_risk_is_low():
    """A malicious label with low risk still updates — gate is risk-only."""
    store = _store(max_risk=40.0)
    row = _event(1, resource="DarkWeb", label="LATERAL_MOVEMENT")
    assert maybe_update(store, row, risk_score=5.0) is True
    assert "DarkWeb" in store.entity_profiles["u1"].resource_counts


def test_replay_is_deterministic():
    events = pd.DataFrame([_event(i, hours=float(i), resource=f"R{i % 3}") for i in range(8)])
    risk = pd.DataFrame(
        {"event_id": events.event_id, "risk_score": [5.0, 50.0, 12.0, 90.0, 8.0, 8.0, 8.0, 8.0]}
    )
    store_a = _store()
    store_b = _store()
    replay(store_a, events, risk)
    replay(store_b, events.sample(frac=1.0, random_state=0), risk)
    a = store_a.entity_profiles["u1"]
    b = store_b.entity_profiles["u1"]
    assert a.n_updates == b.n_updates
    assert a.n_events == pytest.approx(b.n_events)
    assert a.resource_counts == b.resource_counts
    assert store_a.n_blocked == store_b.n_blocked == 2


def test_causality_future_event_does_not_affect_earlier_snapshot():
    store = _store()
    early = _event(1, hours=0.0, resource="VPN")
    late = _event(2, hours=48.0, resource="Payroll")
    maybe_update(store, early, 10.0)
    mid = store.entity_profiles["u1"].copy()
    maybe_update(store, late, 10.0)
    # Mid snapshot must not contain the future resource mass from `late`.
    assert mid.resource_counts.get("Payroll", 0.0) == 0.0
    assert store.entity_profiles["u1"].resource_counts.get("Payroll", 0.0) >= 1.0


def test_insider_drift_low_risk_is_absorbed_in_evaluation_doc():
    store = _store(max_risk=40.0)
    rows = [
        _event(1, hours=0.0, resource="HR Portal", label="INSIDER_DRIFT"),
        _event(2, hours=2.0, resource="HR Portal", label="INSIDER_DRIFT"),
        _event(3, hours=4.0, resource="VPN", label="BENIGN"),
        _event(4, hours=6.0, resource="ExfilShare", label="LOW_AND_SLOW_EXFILTRATION"),
    ]
    events = pd.DataFrame(rows)
    risk = pd.DataFrame(
        {
            "event_id": [r.event_id for r in rows],
            "risk_score": [15.0, 18.0, 8.0, 92.0],
        }
    )
    replay(store, events, risk)
    doc = evaluate_drift(store, events, risk, max_risk=40.0)
    assert doc["labels_never_gate"] is True
    assert doc["poisoning_resistance"]["high_risk_block_rate"] == 1.0
    assert doc["adaptation"]["insider_drift_updated"] == 2
    assert doc["adaptation"]["insider_drift_blocked"] == 0
    assert any(ex["entity_id"] == "u1" for ex in doc["insider_drift_exhibits"])
    assert "ExfilShare" not in store.entity_profiles["u1"].resource_counts
    assert store.entity_profiles["u1"].resource_counts.get("HR Portal", 0) >= 1.0


def test_snapshot_roundtrips_to_behaviour_profile():
    adaptive = AdaptiveEntityProfile.from_frozen(_frozen_profile())
    apply_event_to_profile(adaptive, _event(1), half_life_days=14.0)
    snap = adaptive.snapshot()
    assert isinstance(snap, BehaviourProfile)
    assert snap.key == "u1"
    assert snap.n_events >= 1
