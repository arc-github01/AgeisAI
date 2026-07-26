"""Live attack synthesis for the Phase 11 SOC simulator.

Builds a fresh campaign with the same injectors used for the offline dataset,
anchored after the persisted event corpus so ``process_injection`` can warm
real history and score through ``StreamingEngine.process_event``.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

from src.artifacts import artifact_path
from src.config import load_config
from src.schema import (
    ATTACK_CLASSES,
    IDENTITY_COLUMNS,
    OBSERVATION_COLUMNS,
    AttackType,
    validate_events,
)
from src.utils.seeding import get_rng

from .attacks import (
    AttackCampaign,
    campaign_size_range,
    inject_brute_force,
    inject_credential_stuffing,
    inject_device_spoofing,
    inject_impossible_travel,
    inject_insider_drift,
    inject_lateral_movement,
    inject_low_and_slow_exfiltration,
)
from .entities import (
    EntityBehavioralProfile,
    load_population_records,
)

_LIVE_STUFFING_VICTIMS = (3, 8)

#: Attempts per victim in a live stuffing run. The defining shape is fan-out
#: (many identities, few sources), not depth, so a couple of tries each is
#: faithful and keeps the demo interactive.
_LIVE_STUFFING_ATTEMPTS = (1, 3)

#: Demo-scaled campaign shapes.
#:
#: The offline ranges in ``config.yaml`` are sized for a 21-90 day corpus. Used
#: unchanged for a live injection they break the demo in two ways: an operator
#: waits ~15s while 141 events are scored one at a time, and a slow-burn
#: campaign gets compressed into the few days available after the corpus, so a
#: 35-day insider-drift ramp arrives at ~13 events/day and stops looking like
#: gradual drift at all. These overrides keep every attack's defining signal
#: (a brute force is still a burst of failures; drift still ramps) at a size a
#: live audience will sit through. Injectors clamp against these same keys, so
#: ``campaign_size_range`` stays consistent with what is actually generated.
_LIVE_CAMPAIGN_SHAPES: dict[str, dict[str, Any]] = {
    "brute_force": {
        "attempts": [10, 26],
        "stealth_attempts": [6, 12],
    },
    "lateral_movement": {
        "hops": [4, 8],
        "stealth_hops": [3, 4],
    },
    "device_spoofing": {
        "events": [3, 8],
    },
    "low_and_slow_exfiltration": {
        "days": [4, 10],
        "events_per_day": [1, 2],
    },
    "insider_drift": {
        "days": [5, 10],
        "new_resources": [2, 3],
    },
}


def _load_entities_doc() -> dict[str, Any] | list[Any]:
    path = artifact_path("entities")
    if not path.exists():
        raise FileNotFoundError(
            f"entities artifact missing at {path}; run `python -m src.generator` first"
        )
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _load_events() -> pd.DataFrame:
    path = artifact_path("events")
    if not path.exists():
        raise FileNotFoundError(
            f"events artifact missing at {path}; run `python -m src.generator` first"
        )
    frame = pd.read_parquet(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame


def _intensity_level(intensity: int) -> int:
    return max(1, min(5, int(intensity)))


def _stealthy_for_intensity(intensity: int) -> bool:
    """Lower intensity → stealthier campaign (harder supporting signals)."""
    return _intensity_level(intensity) <= 2


def live_attacks_config(attacks_cfg: dict[str, Any]) -> dict[str, Any]:
    """Copy of the attack config with demo-scaled campaign shapes applied."""
    scaled = {key: dict(value) if isinstance(value, dict) else value
              for key, value in attacks_cfg.items()}
    for attack_key, overrides in _LIVE_CAMPAIGN_SHAPES.items():
        if attack_key in scaled and isinstance(scaled[attack_key], dict):
            scaled[attack_key] = {**scaled[attack_key], **overrides}
    return scaled


def _scale_between(low: int, high: int, intensity: int) -> int:
    level = _intensity_level(intensity)
    fraction = (level - 1) / 4.0
    return max(1, int(round(low + fraction * (high - low))))


def _target_events(
    attack_type: AttackType,
    intensity: int,
    attacks_cfg: dict[str, Any],
    *,
    stealthy: bool,
) -> int:
    low, high = campaign_size_range(attack_type, attacks_cfg, stealthy=stealthy)
    return _scale_between(int(low), int(high), intensity)


def _window_start(
    events: pd.DataFrame,
    entity_id: str,
    rng: np.random.Generator,
    *,
    stealthy: bool,
) -> datetime:
    """Place the campaign just after the corpus so history warming stays causal."""
    if events.empty:
        base = datetime.fromisoformat(str(load_config()["generator.start_date"]))
        return base + timedelta(days=1, hours=10)

    corpus_end = pd.Timestamp(events["timestamp"].max()).to_pydatetime()
    entity_rows = events[events["entity_id"].astype(str) == str(entity_id)]
    if not entity_rows.empty:
        entity_end = pd.Timestamp(entity_rows["timestamp"].max()).to_pydatetime()
        base = max(corpus_end, entity_end)
    else:
        base = corpus_end

    # Small gap after history so rolling windows see prior behaviour, not the attack.
    offset_hours = float(rng.uniform(1.0, 4.0))
    start = base + timedelta(hours=offset_hours)
    # Snap to a plausible hour on the *next* calendar morning/night without
    # moving earlier than ``start`` (replace() alone can rewind into history).
    day = (start + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    if stealthy:
        return day + timedelta(hours=10, minutes=int(rng.integers(0, 60)))
    return day + timedelta(
        hours=int(rng.integers(1, 5)),
        minutes=int(rng.integers(0, 60)),
    )


def _travel_anchor(
    events: pd.DataFrame,
    entity_id: str,
    window_start: datetime,
) -> dict[str, Any]:
    entity_rows = events[events["entity_id"].astype(str) == str(entity_id)].sort_values(
        "timestamp", kind="stable"
    )
    if entity_rows.empty:
        raise ValueError(
            f"impossible travel requires historical events for {entity_id}"
        )
    row = entity_rows.iloc[-1]
    anchor_ts = pd.Timestamp(row["timestamp"]).to_pydatetime()
    gap_minutes = max(
        90.0,
        (window_start - anchor_ts).total_seconds() / 60.0,
    )
    return {
        "timestamp": anchor_ts,
        "latitude": float(row["latitude"]),
        "longitude": float(row["longitude"]),
        "next_gap_minutes": gap_minutes,
    }


def _pick_stuffing_victims(
    population: list[EntityBehavioralProfile],
    primary: EntityBehavioralProfile,
    intensity: int,
    rng: np.random.Generator,
) -> list[EntityBehavioralProfile]:
    users = [p for p in population if p.entity_type.value == "user" and p.entity_id != primary.entity_id]
    low, high = _LIVE_STUFFING_VICTIMS
    wanted = max(low, min(high, low + _intensity_level(intensity) - 1))
    if not users:
        return [primary]
    count = min(wanted - 1, len(users))
    picks = rng.choice(len(users), size=count, replace=False) if count else []
    others = [users[int(i)] for i in np.atleast_1d(picks)] if count else []
    return [primary, *others]


def build_live_campaign(
    entity_id: str,
    attack_type: str,
    intensity: int,
    *,
    population: list[EntityBehavioralProfile] | None = None,
    events: pd.DataFrame | None = None,
    stealthy: bool | None = None,
) -> AttackCampaign:
    """Create one live attack campaign against ``entity_id``."""
    if attack_type not in ATTACK_CLASSES:
        raise ValueError(f"unknown attack type: {attack_type}")

    cfg = load_config()
    attacks_cfg = live_attacks_config(cfg["generator.attacks"])
    label = AttackType(attack_type)
    stealth = _stealthy_for_intensity(intensity) if stealthy is None else bool(stealthy)
    target = _target_events(label, intensity, attacks_cfg, stealthy=stealth)
    rng = get_rng(f"generator.live.{entity_id}.{attack_type}.{_intensity_level(intensity)}")

    if population is None:
        population = load_population_records(_load_entities_doc())
    by_id = {profile.entity_id: profile for profile in population}
    if entity_id not in by_id:
        raise ValueError(f"unknown entity_id: {entity_id}")
    profile = by_id[entity_id]

    history = events if events is not None else _load_events()
    window_start = _window_start(history, entity_id, rng, stealthy=stealth)
    campaign_id = (
        f"LIVE-{label.value}-{entity_id}-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-"
        f"{_intensity_level(intensity)}"
    )

    if label is AttackType.IMPOSSIBLE_TRAVEL:
        anchor = _travel_anchor(history, entity_id, window_start)
        return inject_impossible_travel(
            profile,
            campaign_id=campaign_id,
            rng=rng,
            attacks_cfg=attacks_cfg,
            anchor=anchor,
            stealthy=stealth,
            target_events=target,
            max_plausible_kmh=float(cfg["features.max_plausible_kmh"]),
        )

    if label is AttackType.CREDENTIAL_STUFFING:
        victims = _pick_stuffing_victims(population, profile, intensity, rng)
        # Fan-out is the signal, so size the campaign as victims x attempts
        # rather than from the corpus-scale per-campaign event budget.
        attempts_each = _scale_between(*_LIVE_STUFFING_ATTEMPTS, intensity)
        return inject_credential_stuffing(
            victims,
            campaign_id=campaign_id,
            rng=rng,
            attacks_cfg=attacks_cfg,
            window_start=window_start,
            stealthy=stealth,
            target_events=len(victims) * attempts_each,
        )

    if label is AttackType.BRUTE_FORCE:
        return inject_brute_force(
            profile,
            campaign_id=campaign_id,
            rng=rng,
            attacks_cfg=attacks_cfg,
            window_start=window_start,
            stealthy=stealth,
            target_events=target,
        )

    if label is AttackType.LATERAL_MOVEMENT:
        return inject_lateral_movement(
            profile,
            campaign_id=campaign_id,
            rng=rng,
            attacks_cfg=attacks_cfg,
            window_start=window_start,
            stealthy=stealth,
            target_events=target,
        )

    if label is AttackType.DEVICE_SPOOFING:
        return inject_device_spoofing(
            profile,
            campaign_id=campaign_id,
            rng=rng,
            attacks_cfg=attacks_cfg,
            window_start=window_start,
            stealthy=stealth,
            target_events=target,
        )

    if label is AttackType.LOW_AND_SLOW_EXFILTRATION:
        return inject_low_and_slow_exfiltration(
            profile,
            campaign_id=campaign_id,
            rng=rng,
            attacks_cfg=attacks_cfg,
            window_start=window_start,
            stealthy=stealth,
            target_events=target,
        )

    if label is AttackType.INSIDER_DRIFT:
        return inject_insider_drift(
            profile,
            campaign_id=campaign_id,
            rng=rng,
            attacks_cfg=attacks_cfg,
            window_start=window_start,
            stealthy=stealth,
            target_events=target,
        )

    raise ValueError(f"unsupported live attack type: {attack_type}")


def campaign_to_injection_frame(campaign: AttackCampaign) -> pd.DataFrame:
    """Convert a campaign into the observation frame expected by ``process_injection``."""
    if not campaign.events:
        raise ValueError("campaign produced no events")
    frame = pd.DataFrame(campaign.events).copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.sort_values(["timestamp", "entity_id"], kind="stable").reset_index(drop=True)
    frame["event_id"] = [f"INJ-{i:04d}" for i in range(1, len(frame) + 1)]
    required = list(IDENTITY_COLUMNS + OBSERVATION_COLUMNS)
    # Labels may be present for debugging but are stripped before scoring.
    out = frame.loc[:, [c for c in required if c in frame.columns]]
    validate_events(out, require_labels=False)
    return out


def synthesize_live_attack(
    entity_id: str,
    attack_type: str,
    intensity: int,
    *,
    population: list[EntityBehavioralProfile] | None = None,
    events: pd.DataFrame | None = None,
    stealthy: bool | None = None,
) -> pd.DataFrame:
    """Public API: synthesise live injection events for the SOC simulator."""
    campaign = build_live_campaign(
        entity_id,
        attack_type,
        intensity,
        population=population,
        events=events,
        stealthy=stealthy,
    )
    return campaign_to_injection_frame(campaign)


__all__ = [
    "build_live_campaign",
    "campaign_to_injection_frame",
    "synthesize_live_attack",
]
