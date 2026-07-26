"""Cohort fallback profiles and frozen profile bundle."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .entity_profile import BehaviourProfile, fit_profile


def cohort_key(row: pd.Series, keys: list[str]) -> str:
    return "|".join(str(row[key]) for key in keys)


def _group_key(value: Any) -> str:
    """Encode pandas groupby keys identically to inference-time lookup."""
    parts = value if isinstance(value, tuple) else (value,)
    return "|".join(str(part) for part in parts)


@dataclass(frozen=True)
class ProfileBundle:
    cutoff: pd.Timestamp
    entity_profiles: dict[str, BehaviourProfile]
    cohort_profiles: dict[str, BehaviourProfile]
    cohort_keys: tuple[str, ...]
    min_events_for_personal: int

    def resolve(self, row: pd.Series) -> tuple[BehaviourProfile | None, str, float]:
        personal = self.entity_profiles.get(str(row.entity_id))
        if personal and personal.n_events >= self.min_events_for_personal:
            return personal, "entity", 1.0
        cohort = self.cohort_profiles.get(cohort_key(row, list(self.cohort_keys)))
        if cohort:
            confidence = min(1.0, (personal.n_events if personal else 0) / self.min_events_for_personal)
            return cohort, "cohort", confidence
        return personal, "none", 0.0


def fit_profiles(
    events: pd.DataFrame, cutoff: pd.Timestamp, *, cohort_keys: list[str], min_events_for_personal: int
) -> ProfileBundle:
    """Fit only rows up to cutoff and explicitly labelled BENIGN."""
    train = events[(events.timestamp <= cutoff) & (events.label == "BENIGN")].copy()
    entity_profiles = {
        str(entity_id): fit_profile(str(entity_id), group)
        for entity_id, group in train.groupby("entity_id", sort=False)
    }
    cohort_profiles = {
        _group_key(key): fit_profile(_group_key(key), group)
        for key, group in train.groupby(cohort_keys, sort=False)
    }
    return ProfileBundle(
        cutoff=pd.Timestamp(cutoff),
        entity_profiles=entity_profiles,
        cohort_profiles=cohort_profiles,
        cohort_keys=tuple(cohort_keys),
        min_events_for_personal=min_events_for_personal,
    )
