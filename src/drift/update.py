"""Risk-gated EWMA updates for adaptive entity profiles.

Gate on hybrid ``risk_score`` only — never on attack labels — so an adversary
cannot poison baselines by repeating high-risk behaviour, and evaluation labels
never leak into adaptation policy.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .store import AdaptiveEntityProfile, AdaptiveProfileStore

SECONDS_PER_DAY = 86400.0


def ewma_decay(dt_days: float, half_life_days: float) -> float:
    """Return ``0.5 ** (dt / half_life)`` clamped to a sensible range."""
    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive")
    if dt_days <= 0:
        return 1.0
    return float(0.5 ** (dt_days / half_life_days))


def _bump_count(counts: dict[str, float], key: str, decay: float) -> None:
    decayed = {k: float(v) * decay for k, v in counts.items() if float(v) * decay > 1e-12}
    counts.clear()
    counts.update(decayed)
    counts[key] = counts.get(key, 0.0) + 1.0


def _ewma_moment(
    mean: float, m2: float, value: float, decay: float
) -> tuple[float, float]:
    """Update EWMA mean and second-moment tracker."""
    if decay >= 1.0 and mean == 0.0 and m2 == 0.0:
        return float(value), 0.0
    mean_old = mean
    mean_new = decay * mean_old + (1.0 - decay) * value
    # Parallel to Welford: track EWMA of squared deviation around the new mean.
    residual = value - mean_old
    m2_new = decay * m2 + (1.0 - decay) * residual * (value - mean_new)
    return float(mean_new), float(max(m2_new, 0.0))


def apply_event_to_profile(
    profile: AdaptiveEntityProfile,
    row: Any,
    *,
    half_life_days: float,
) -> None:
    """Mutate ``profile`` with one accepted event (caller already passed the gate)."""
    ts = pd.Timestamp(row.timestamp)
    ts_s = float(ts.timestamp())
    if profile.last_timestamp is None:
        decay = 1.0
        dt_days = 0.0
    else:
        dt_days = max(0.0, (ts_s - float(profile.last_timestamp)) / SECONDS_PER_DAY)
        decay = ewma_decay(dt_days, half_life_days)

    hour_key = str(int(ts.hour))
    location = f"{row.country}|{row.city}"
    device = f"{row.device_id}|{row.device_mac}"
    resource = str(row.resource_accessed)
    auth = str(row.auth_method)

    _bump_count(profile.login_hour_counts, hour_key, decay)
    _bump_count(profile.location_counts, location, decay)
    _bump_count(profile.device_counts, device, decay)
    _bump_count(profile.resource_counts, resource, decay)
    _bump_count(profile.auth_method_counts, auth, decay)

    profile.n_events = decay * float(profile.n_events) + 1.0

    duration = float(row.session_duration_s)
    profile.duration_mean, profile.duration_m2 = _ewma_moment(
        profile.duration_mean, profile.duration_m2, duration, decay
    )
    nbytes = float(row.bytes_transferred)
    profile.bytes_mean, profile.bytes_m2 = _ewma_moment(
        profile.bytes_mean, profile.bytes_m2, nbytes, decay
    )

    fail = 0.0 if bool(row.auth_success) else 1.0
    profile.auth_failure_rate = decay * profile.auth_failure_rate + (1.0 - decay) * fail

    # Baseline geo drifts slowly with accepted low-risk locations.
    lat = float(row.latitude)
    lon = float(row.longitude)
    profile.baseline_latitude = (
        decay * profile.baseline_latitude + (1.0 - decay) * lat
    )
    profile.baseline_longitude = (
        decay * profile.baseline_longitude + (1.0 - decay) * lon
    )

    prev = profile.last_resource if profile.last_resource is not None else "<START>"
    tkey = f"{prev}->{resource}"
    _bump_count(profile.transitions, tkey, decay)
    profile.total_transitions = decay * float(profile.total_transitions) + 1.0
    profile.last_resource = resource
    profile.last_timestamp = ts_s
    profile.n_updates += 1


def maybe_update(
    store: AdaptiveProfileStore,
    row: Any,
    risk_score: float,
    *,
    event_id: str | None = None,
) -> bool:
    """Apply a risk-gated update. Returns True if the profile was updated.

    Labels are intentionally not accepted as arguments.
    """
    store.n_considered += 1
    eid = event_id if event_id is not None else str(getattr(row, "event_id", ""))
    if float(risk_score) >= float(store.baseline_update_max_risk):
        store.n_blocked += 1
        profile = store.entity_profiles.get(str(row.entity_id))
        if profile is not None:
            profile.n_blocked += 1
        if eid:
            store.blocked_event_ids.append(eid)
        return False

    profile = store.ensure_entity(str(row.entity_id), row)
    apply_event_to_profile(
        profile, row, half_life_days=store.ewma_halflife_days
    )
    store.n_updated += 1
    if eid:
        store.updated_event_ids.append(eid)
    return True


def replay(
    store: AdaptiveProfileStore,
    events: pd.DataFrame,
    risk_scores: pd.DataFrame,
) -> AdaptiveProfileStore:
    """Replay events in causal order, joining risk scores by ``event_id``.

    Only events at or after the frozen profile cutoff participate (the store
    was seeded from pre-cutoff BENIGN history). Callers may pass a pre-filtered
    frame; this function sorts stably and does not peek ahead.
    """
    if "event_id" not in risk_scores.columns or "risk_score" not in risk_scores.columns:
        raise ValueError("risk_scores must contain event_id and risk_score")
    risk_map = risk_scores.set_index("event_id")["risk_score"].astype(float)
    ordered = events.sort_values(["timestamp", "event_id"], kind="stable")
    for _, row in ordered.iterrows():
        eid = str(row.event_id)
        if eid not in risk_map.index:
            raise KeyError(f"missing risk_score for event_id={eid}")
        maybe_update(store, row, float(risk_map.loc[eid]), event_id=eid)
    return store


__all__ = [
    "apply_event_to_profile",
    "ewma_decay",
    "maybe_update",
    "replay",
]
