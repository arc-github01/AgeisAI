"""Phase 4 chronological behavioural feature pipeline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.artifacts import artifact_path
from src.config import load_config
from src.generator.entities import COHORT_ARCHETYPES
from src.profiling import BehaviourProfile, fit_profiles
from src.schema import assert_no_label_leakage, validate_events
from src.utils.geo import haversine_km, implied_velocity_kmh

MODEL_FEATURE_COLUMNS = (
    "hour", "day_of_week", "is_weekend", "inter_event_gap_s", "event_count_5m",
    "event_count_60m", "failed_auth_5m", "failed_auth_30m", "login_hour_rarity",
    "is_entity_off_hours", "location_rarity", "is_new_location",
    "geo_distance_from_baseline_km", "geo_distance_from_previous_km",
    "geo_velocity_kmh", "is_known_device", "device_rarity", "fingerprint_changed",
    "is_known_resource", "resource_rarity", "resource_sensitivity",
    "recent_resource_breadth_24h", "auth_method_rarity", "auth_failure_deviation",
    "session_duration_zscore", "transfer_volume_zscore", "transition_probability",
    "sequence_anomaly_score",
)
assert_no_label_leakage(MODEL_FEATURE_COLUMNS)

_SENSITIVITY = {"low": 1.0, "medium": 2.0, "high": 3.0, "critical": 4.0}
RESOURCE_SENSITIVITY = {
    resource.name: _SENSITIVITY[resource.sensitivity]
    for cohort in COHORT_ARCHETYPES.values()
    for resource in cohort.resources
}


def _rarity(counts: dict[str, int], value: str, total: int, smoothing: float) -> float:
    return -np.log((counts.get(value, 0) + smoothing) / (total + smoothing * (len(counts) + 1)))


def _z(value: float, mean: float, std: float, floor: float) -> float:
    return (value - mean) / max(std, floor)


def empty_profile(event: pd.Series) -> BehaviourProfile:
    """Cold-start placeholder when no personal or cohort profile exists.

    Continuous moments are seeded from the current observation so the first
    event has near-zero z-scores. Seeding from ``(0, 1)`` made ordinary byte
    and duration values look extreme and could permanently block risk-gated
    adaptive updates for brand-new entities.
    """
    duration = float(event.session_duration_s)
    nbytes = float(event.bytes_transferred)
    return BehaviourProfile(
        "empty",
        0,
        {},
        {},
        {},
        {},
        {},
        duration,
        1.0,
        nbytes,
        1.0,
        0.0,
        float(event.latitude),
        float(event.longitude),
        {},
        0,
    )


def compute_event_features(
    event: pd.Series,
    *,
    history: list[pd.Series],
    previous_fingerprint: str | None,
    profile: BehaviourProfile,
    profile_source: str,
    profile_confidence: float,
    smoothing: float,
    std_floor: float,
    velocity_cap: float,
    cutoff: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Causal features for one event given only prior history and a frozen profile view.

    Shared by the offline batch pipeline and the streaming ``process_event`` path
    so both use the same formulas. History must contain only earlier events for
    the same entity; the current event is not appended here.
    """
    ts = pd.Timestamp(event.timestamp)
    entity_id = str(event.entity_id)
    result: dict[str, Any] = {
        "event_id": event.event_id,
        "timestamp": ts,
        "entity_id": entity_id,
        "hour": float(ts.hour),
        "day_of_week": float(ts.dayofweek),
        "is_weekend": float(ts.dayofweek >= 5),
        "profile_source": profile_source,
        "profile_confidence": float(profile_confidence),
    }
    if cutoff is not None:
        # Compare in a timezone-safe way (generator timestamps may be UTC-aware).
        ts_cmp = ts.tz_localize(None) if ts.tzinfo is not None else ts
        cut_cmp = pd.Timestamp(cutoff)
        if cut_cmp.tzinfo is not None:
            cut_cmp = cut_cmp.tz_localize(None)
        result["split"] = "train" if ts_cmp <= cut_cmp else "evaluation"

    previous = history[-1] if history else None
    result["inter_event_gap_s"] = (
        (ts - previous.timestamp).total_seconds() if previous is not None else 0.0
    )
    for minutes in (5, 60):
        result[f"event_count_{minutes}m"] = float(
            sum(
                (ts - prior.timestamp).total_seconds() <= minutes * 60
                for prior in history
            )
        )
    for minutes in (5, 30):
        result[f"failed_auth_{minutes}m"] = float(
            sum(
                (ts - prior.timestamp).total_seconds() <= minutes * 60
                and not bool(prior.auth_success)
                for prior in history
            )
        )

    location = f"{event.country}|{event.city}"
    fingerprint = f"{event.device_id}|{event.device_mac}"
    result["login_hour_rarity"] = _rarity(
        profile.login_hour_counts, str(ts.hour), profile.n_events, smoothing
    )
    result["is_entity_off_hours"] = float(
        profile.login_hour_counts.get(str(ts.hour), 0) == 0
    )
    result["location_rarity"] = _rarity(
        profile.location_counts, location, profile.n_events, smoothing
    )
    result["is_new_location"] = float(location not in profile.location_counts)
    result["is_known_device"] = float(fingerprint in profile.device_counts)
    result["device_rarity"] = _rarity(
        profile.device_counts, fingerprint, profile.n_events, smoothing
    )
    result["fingerprint_changed"] = float(
        previous_fingerprint is not None and fingerprint != previous_fingerprint
    )
    result["is_known_resource"] = float(
        event.resource_accessed in profile.resource_counts
    )
    result["resource_rarity"] = _rarity(
        profile.resource_counts, event.resource_accessed, profile.n_events, smoothing
    )
    result["resource_sensitivity"] = RESOURCE_SENSITIVITY.get(
        event.resource_accessed, 1.0
    )
    result["auth_method_rarity"] = _rarity(
        profile.auth_method_counts, event.auth_method, profile.n_events, smoothing
    )
    result["auth_failure_deviation"] = float(
        (not bool(event.auth_success)) - profile.auth_failure_rate
    )
    result["session_duration_zscore"] = _z(
        float(event.session_duration_s),
        profile.duration_mean,
        profile.duration_std,
        std_floor,
    )
    result["transfer_volume_zscore"] = _z(
        float(event.bytes_transferred),
        profile.bytes_mean,
        profile.bytes_std,
        std_floor,
    )
    recent = [
        prior.resource_accessed
        for prior in history
        if (ts - prior.timestamp).total_seconds() <= 86400
    ]
    result["recent_resource_breadth_24h"] = float(len(set(recent)))
    result["geo_distance_from_baseline_km"] = haversine_km(
        profile.baseline_latitude,
        profile.baseline_longitude,
        float(event.latitude),
        float(event.longitude),
    )
    if previous is None:
        result["geo_distance_from_previous_km"] = 0.0
        result["geo_velocity_kmh"] = 0.0
        transition = "<START>->" + event.resource_accessed
    else:
        distance = haversine_km(
            previous.latitude,
            previous.longitude,
            event.latitude,
            event.longitude,
        )
        result["geo_distance_from_previous_km"] = distance
        result["geo_velocity_kmh"] = min(
            velocity_cap,
            implied_velocity_kmh(distance, result["inter_event_gap_s"]),
        )
        transition = previous.resource_accessed + "->" + event.resource_accessed
    result["transition_probability"] = (
        profile.transitions.get(transition, 0) + smoothing
    ) / (profile.total_transitions + smoothing * (len(profile.transitions) + 1))
    result["sequence_anomaly_score"] = -np.log(result["transition_probability"])
    return result


def build_features(events: pd.DataFrame) -> tuple[pd.DataFrame, Any]:
    cfg = load_config()
    validate_events(events, strict_order=True)
    frame = events.copy().sort_values(["timestamp", "event_id"], kind="stable").reset_index(drop=True)
    cutoff = frame.timestamp.iloc[max(1, int(len(frame) * float(cfg["evaluation.train_fraction"]))) - 1]
    bundle = fit_profiles(
        frame, cutoff,
        cohort_keys=list(cfg["profiling.cohort_keys"]),
        min_events_for_personal=int(cfg["profiling.min_events_for_personal"]),
    )
    smoothing = float(cfg.get("features.rarity_smoothing", 1.0))
    std_floor = float(cfg.get("features.std_floor", 1.0))
    velocity_cap = float(cfg.get("features.velocity_cap_kmh", 100000.0))
    rows: list[dict[str, Any]] = []
    for entity_id, group in frame.groupby("entity_id", sort=False):
        history: list[pd.Series] = []
        previous_fp = None
        for _, event in group.sort_values("timestamp", kind="stable").iterrows():
            profile, source, confidence = bundle.resolve(event)
            if profile is None:
                profile = empty_profile(event)
            result = compute_event_features(
                event,
                history=history,
                previous_fingerprint=previous_fp,
                profile=profile,
                profile_source=source,
                profile_confidence=confidence,
                smoothing=smoothing,
                std_floor=std_floor,
                velocity_cap=velocity_cap,
                cutoff=cutoff,
            )
            rows.append(result)
            history.append(event)
            previous_fp = f"{event.device_id}|{event.device_mac}"
    features = pd.DataFrame(rows).sort_values(["timestamp", "event_id"], kind="stable").reset_index(drop=True)
    return features, bundle


def save_features(features: pd.DataFrame, bundle: Any) -> dict[str, Path]:
    assert_no_label_leakage(MODEL_FEATURE_COLUMNS)
    if not np.isfinite(features.loc[:, MODEL_FEATURE_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("feature matrix contains NaN or infinite model values")
    out = artifact_path("features", ensure_parent=True)
    features.to_parquet(out, index=False)
    profiles = artifact_path("profiles", ensure_parent=True)
    joblib.dump(bundle, profiles)
    report = artifact_path("feature_validation", ensure_parent=True)
    report.write_text(json.dumps({"rows": len(features), "model_features": list(MODEL_FEATURE_COLUMNS)}), encoding="utf-8")
    return {"features": out, "profiles": profiles, "feature_validation": report}
