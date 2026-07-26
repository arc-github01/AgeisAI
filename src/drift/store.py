"""Mutable adaptive entity profiles for concept-drift updates.

THE UPDATE FORMULA
------------------
Profiles start as a copy of the frozen Phase-4 ``BehaviourProfile``. For an
accepted event at time ``t`` with previous update time ``t_prev``:

    dt_days = max(0, (t - t_prev) / 86400)
    decay   = 0.5 ** (dt_days / ewma_halflife_days)

Categorical counts (hour, location, device, resource, auth, transitions)::

    count[k]      <- decay * count[k]   for all k
    count[obs]    <- count[obs] + 1
    n_events      <- decay * n_events + 1

Continuous moments (duration, bytes, lat/lon)::

    mean <- decay * mean + (1 - decay) * value
    # Welford-style EWMA second moment for std:
    m2   <- decay * m2 + (1 - decay) * (value - mean_old) * (value - mean)
    std  <- sqrt(max(m2, 0))

Auth failure rate uses the same EWMA mean update on {0,1}.

Events with ``risk_score >= baseline_update_max_risk`` are rejected and leave
the profile unchanged (poisoning resistance). Labels never participate in the
gate.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from src.profiling import BehaviourProfile, ProfileBundle, cohort_key


@dataclass
class AdaptiveEntityProfile:
    """Mutable per-entity statistics (float counts so EWMA decay is exact)."""

    key: str
    n_events: float
    login_hour_counts: dict[str, float]
    location_counts: dict[str, float]
    device_counts: dict[str, float]
    resource_counts: dict[str, float]
    auth_method_counts: dict[str, float]
    duration_mean: float
    duration_m2: float
    bytes_mean: float
    bytes_m2: float
    auth_failure_rate: float
    baseline_latitude: float
    baseline_longitude: float
    transitions: dict[str, float]
    total_transitions: float
    last_timestamp: float | None = None
    last_resource: str | None = None
    n_updates: int = 0
    n_blocked: int = 0

    @classmethod
    def from_frozen(cls, profile: BehaviourProfile) -> "AdaptiveEntityProfile":
        return cls(
            key=profile.key,
            n_events=float(profile.n_events),
            login_hour_counts={k: float(v) for k, v in profile.login_hour_counts.items()},
            location_counts={k: float(v) for k, v in profile.location_counts.items()},
            device_counts={k: float(v) for k, v in profile.device_counts.items()},
            resource_counts={k: float(v) for k, v in profile.resource_counts.items()},
            auth_method_counts={k: float(v) for k, v in profile.auth_method_counts.items()},
            duration_mean=float(profile.duration_mean),
            duration_m2=float(profile.duration_std) ** 2,
            bytes_mean=float(profile.bytes_mean),
            bytes_m2=float(profile.bytes_std) ** 2,
            auth_failure_rate=float(profile.auth_failure_rate),
            baseline_latitude=float(profile.baseline_latitude),
            baseline_longitude=float(profile.baseline_longitude),
            transitions={k: float(v) for k, v in profile.transitions.items()},
            total_transitions=float(profile.total_transitions),
        )

    def snapshot(self) -> BehaviourProfile:
        """Immutable view compatible with Phase-4 feature code."""
        return BehaviourProfile(
            key=self.key,
            n_events=int(round(self.n_events)),
            login_hour_counts={k: int(round(v)) for k, v in self.login_hour_counts.items()},
            location_counts={k: int(round(v)) for k, v in self.location_counts.items()},
            device_counts={k: int(round(v)) for k, v in self.device_counts.items()},
            resource_counts={k: int(round(v)) for k, v in self.resource_counts.items()},
            auth_method_counts={k: int(round(v)) for k, v in self.auth_method_counts.items()},
            duration_mean=self.duration_mean,
            duration_std=math.sqrt(max(self.duration_m2, 0.0)),
            bytes_mean=self.bytes_mean,
            bytes_std=math.sqrt(max(self.bytes_m2, 0.0)),
            auth_failure_rate=self.auth_failure_rate,
            baseline_latitude=self.baseline_latitude,
            baseline_longitude=self.baseline_longitude,
            transitions={k: int(round(v)) for k, v in self.transitions.items()},
            total_transitions=int(round(self.total_transitions)),
        )

    def copy(self) -> "AdaptiveEntityProfile":
        return deepcopy(self)


@dataclass
class AdaptiveProfileStore:
    """Causal adaptive store seeded from a frozen ``ProfileBundle``."""

    entity_profiles: dict[str, AdaptiveEntityProfile]
    cohort_profiles: dict[str, BehaviourProfile]
    cohort_keys: tuple[str, ...]
    min_events_for_personal: int
    ewma_halflife_days: float
    baseline_update_max_risk: float
    rolling_window_days: float
    n_considered: int = 0
    n_updated: int = 0
    n_blocked: int = 0
    blocked_event_ids: list[str] = field(default_factory=list)
    updated_event_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_bundle(
        cls,
        bundle: ProfileBundle,
        *,
        ewma_halflife_days: float,
        baseline_update_max_risk: float,
        rolling_window_days: float,
    ) -> "AdaptiveProfileStore":
        return cls(
            entity_profiles={
                key: AdaptiveEntityProfile.from_frozen(profile)
                for key, profile in bundle.entity_profiles.items()
            },
            cohort_profiles=dict(bundle.cohort_profiles),
            cohort_keys=tuple(bundle.cohort_keys),
            min_events_for_personal=int(bundle.min_events_for_personal),
            ewma_halflife_days=float(ewma_halflife_days),
            baseline_update_max_risk=float(baseline_update_max_risk),
            rolling_window_days=float(rolling_window_days),
        )

    def resolve(
        self, entity_id: str, row: Any
    ) -> tuple[AdaptiveEntityProfile | BehaviourProfile | None, str, float]:
        personal = self.entity_profiles.get(str(entity_id))
        if personal and personal.n_events >= self.min_events_for_personal:
            return personal, "entity", 1.0
        ckey = cohort_key(row, list(self.cohort_keys))
        cohort = self.cohort_profiles.get(ckey)
        if cohort:
            confidence = min(
                1.0,
                (personal.n_events if personal else 0.0) / self.min_events_for_personal,
            )
            return cohort, "cohort", confidence
        return personal, "none", 0.0

    def ensure_entity(self, entity_id: str, row: Any) -> AdaptiveEntityProfile:
        """Return mutable personal profile, cloning cohort if the entity is new."""
        key = str(entity_id)
        if key in self.entity_profiles:
            return self.entity_profiles[key]
        ckey = cohort_key(row, list(self.cohort_keys))
        cohort = self.cohort_profiles.get(ckey)
        if cohort is not None:
            profile = AdaptiveEntityProfile.from_frozen(cohort)
            profile.key = key
        else:
            profile = AdaptiveEntityProfile(
                key=key,
                n_events=0.0,
                login_hour_counts={},
                location_counts={},
                device_counts={},
                resource_counts={},
                auth_method_counts={},
                duration_mean=0.0,
                duration_m2=1.0,
                bytes_mean=0.0,
                bytes_m2=1.0,
                auth_failure_rate=0.0,
                baseline_latitude=float(getattr(row, "latitude", 0.0)),
                baseline_longitude=float(getattr(row, "longitude", 0.0)),
                transitions={},
                total_transitions=0.0,
            )
        self.entity_profiles[key] = profile
        return profile


__all__ = ["AdaptiveEntityProfile", "AdaptiveProfileStore"]
