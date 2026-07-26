"""Frozen, BENIGN-only behavioural profiles used by Phase 4."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


def _counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    return {str(k): int(v) for k, v in frame[column].value_counts().items()}


def _mean_std(frame: pd.DataFrame, column: str) -> tuple[float, float]:
    values = frame[column].astype(float)
    return float(values.mean()), float(values.std(ddof=0))


@dataclass(frozen=True)
class BehaviourProfile:
    """Learned statistics, never generator ground truth."""

    key: str
    n_events: int
    login_hour_counts: dict[str, int]
    location_counts: dict[str, int]
    device_counts: dict[str, int]
    resource_counts: dict[str, int]
    auth_method_counts: dict[str, int]
    duration_mean: float
    duration_std: float
    bytes_mean: float
    bytes_std: float
    auth_failure_rate: float
    baseline_latitude: float
    baseline_longitude: float
    transitions: dict[str, int]
    total_transitions: int


def fit_profile(key: str, events: pd.DataFrame) -> BehaviourProfile:
    """Fit one profile from already-filtered chronological BENIGN events."""
    ordered = events.sort_values("timestamp", kind="stable")
    transition_keys = (
        ordered["resource_accessed"].shift().fillna("<START>")
        + "->"
        + ordered["resource_accessed"]
    )
    transitions = {str(k): int(v) for k, v in transition_keys.value_counts().items()}
    duration_mean, duration_std = _mean_std(ordered, "session_duration_s")
    bytes_mean, bytes_std = _mean_std(ordered, "bytes_transferred")
    baseline_latitude = float(ordered["latitude"].astype(float).mean())
    baseline_longitude = float(ordered["longitude"].astype(float).mean())
    return BehaviourProfile(
        key=key,
        n_events=len(ordered),
        login_hour_counts=_counts(ordered.assign(_hour=ordered.timestamp.dt.hour), "_hour"),
        location_counts=_counts(ordered.assign(_location=ordered.country + "|" + ordered.city), "_location"),
        device_counts=_counts(ordered.assign(_device=ordered.device_id + "|" + ordered.device_mac), "_device"),
        resource_counts=_counts(ordered, "resource_accessed"),
        auth_method_counts=_counts(ordered, "auth_method"),
        duration_mean=duration_mean,
        duration_std=duration_std,
        bytes_mean=bytes_mean,
        bytes_std=bytes_std,
        auth_failure_rate=float((~ordered.auth_success).mean()),
        baseline_latitude=baseline_latitude,
        baseline_longitude=baseline_longitude,
        transitions=transitions,
        total_transitions=int(len(transition_keys)),
    )
