"""Phase 4 feature contracts."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import load_config
from src.features import MODEL_FEATURE_COLUMNS, build_features
from src.generator import generate_dataset
from src.schema import FORBIDDEN_FEATURE_COLUMNS


PROFILE = {"name": "feature-test", "n_users": 16, "n_service_accounts": 3,
           "n_edge_devices": 3, "days": 21, "target_events": 4000}


def test_feature_matrix_is_joinable_finite_and_leakage_safe():
    generated = generate_dataset(profile=PROFILE)
    features, bundle = build_features(generated.events)
    assert len(features) == len(generated.events)
    assert features.event_id.is_unique
    assert not set(MODEL_FEATURE_COLUMNS) & FORBIDDEN_FEATURE_COLUMNS
    assert np.isfinite(features.loc[:, MODEL_FEATURE_COLUMNS].to_numpy(float)).all()
    assert set(features.split) == {"train", "evaluation"}


def test_profiles_exclude_attack_rows_and_future_rows():
    generated = generate_dataset(profile=PROFILE)
    events = (
        generated.events
        .sort_values(["timestamp", "event_id"], kind="stable")
        .reset_index(drop=True)
    )
    _, bundle = build_features(events)
    train_benign = events[(events.timestamp <= bundle.cutoff) & (events.label == "BENIGN")]
    assert sum(profile.n_events for profile in bundle.entity_profiles.values()) == len(train_benign)
    for entity_id, group in train_benign.groupby("entity_id", sort=False):
        assert bundle.entity_profiles[str(entity_id)].n_events == len(group)
    cohort_keys = list(load_config()["profiling.cohort_keys"])
    for key, group in train_benign.groupby(cohort_keys, sort=False):
        parts = key if isinstance(key, tuple) else (key,)
        bundle_key = "|".join(str(part) for part in parts)
        assert bundle.cohort_profiles[bundle_key].n_events == len(group)


def test_new_entity_resolves_to_matching_cohort():
    generated = generate_dataset(profile=PROFILE)
    events = generated.events.sort_values(
        ["timestamp", "event_id"], kind="stable"
    ).reset_index(drop=True)
    _, bundle = build_features(events)
    event = events.iloc[0].copy()
    event["entity_id"] = "UNSEEN-ENTITY"

    profile, source, confidence = bundle.resolve(event)

    assert profile is not None
    assert source == "cohort"
    assert confidence == 0.0


def test_expected_attack_signals_are_visible():
    generated = generate_dataset(profile=PROFILE)
    features, _ = build_features(generated.events)
    joined = generated.events[["event_id", "label"]].merge(features, on="event_id")
    benign = joined[joined.label == "BENIGN"]
    brute = joined[joined.label == "BRUTE_FORCE"]
    travel = joined[joined.label == "IMPOSSIBLE_TRAVEL"]
    lateral = joined[joined.label == "LATERAL_MOVEMENT"]
    spoof = joined[joined.label == "DEVICE_SPOOFING"]
    assert brute.failed_auth_5m.median() >= benign.failed_auth_5m.median()
    assert travel.geo_velocity_kmh.max() > 900
    assert lateral.sequence_anomaly_score.median() >= benign.sequence_anomaly_score.median()
    assert spoof.is_known_device.mean() <= benign.is_known_device.mean()


def test_model_features_are_causal_for_an_event_against_future_rows():
    generated = generate_dataset(profile=PROFILE)
    events = (
        generated.events
        .sort_values(["timestamp", "event_id"], kind="stable")
        .reset_index(drop=True)
    )
    features_full, _ = build_features(events)
    eval_rows = features_full[features_full["split"] == "evaluation"]
    target = None
    for _, row in eval_rows.iterrows():
        if (
            (events["entity_id"] == row["entity_id"])
            & (events["timestamp"] > row["timestamp"])
        ).any():
            target = row
            break
    assert target is not None
    target_event_id = target["event_id"]
    target_ts = target["timestamp"]
    target_mask = events["timestamp"] > target_ts

    extreme = events.copy()
    extreme.loc[target_mask, "label"] = "IMPOSSIBLE_TRAVEL"
    extreme.loc[target_mask, "is_attack"] = True
    extreme.loc[target_mask, "latitude"] = 89.9
    extreme.loc[target_mask, "longitude"] = 179.9
    extreme.loc[target_mask, "auth_success"] = False
    extreme.loc[target_mask, "session_duration_s"] = 999999.0
    extreme.loc[target_mask, "bytes_transferred"] = 1e12
    features_extreme, _ = build_features(extreme)

    removed = events.copy()
    # Keep timeline length and ordering stable while removing future signal
    # content from the chosen event horizon.
    replacement = removed.loc[~target_mask].iloc[-1]
    for col in removed.columns:
        removed.loc[target_mask, col] = replacement[col]
    removed.loc[target_mask, "timestamp"] = events.loc[target_mask, "timestamp"].values
    removed.loc[target_mask, "event_id"] = events.loc[target_mask, "event_id"].values
    features_removed, _ = build_features(removed.sort_values(["timestamp", "event_id"], kind="stable"))

    full_vec = features_full.loc[
        features_full["event_id"] == target_event_id, list(MODEL_FEATURE_COLUMNS)
    ].iloc[0]
    extreme_vec = features_extreme.loc[
        features_extreme["event_id"] == target_event_id, list(MODEL_FEATURE_COLUMNS)
    ].iloc[0]
    removed_vec = features_removed.loc[
        features_removed["event_id"] == target_event_id, list(MODEL_FEATURE_COLUMNS)
    ].iloc[0]
    np.testing.assert_allclose(extreme_vec.to_numpy(float), full_vec.to_numpy(float), rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(removed_vec.to_numpy(float), full_vec.to_numpy(float), rtol=0.0, atol=1e-9)


def test_geo_baseline_comes_from_frozen_profile_not_full_timeline_mean():
    generated = generate_dataset(profile=PROFILE)
    events = (
        generated.events
        .sort_values(["timestamp", "event_id"], kind="stable")
        .reset_index(drop=True)
    )
    features, bundle = build_features(events)
    merged = features.merge(
        events.loc[:, ["event_id", "label"]],
        on="event_id",
        how="left",
    )
    # Old leaky implementation made this exactly equal for every entity.
    full_series_mean = merged.groupby("entity_id")["geo_distance_from_previous_km"].transform("mean")
    assert not np.allclose(
        merged["geo_distance_from_baseline_km"].to_numpy(float),
        full_series_mean.to_numpy(float),
        rtol=0.0,
        atol=1e-9,
    )
