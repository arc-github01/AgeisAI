"""Attack campaign injection for the synthetic environment (Phase 3).

Design principle
----------------
An injected attack is **a mutation of a real entity's own behaviour**, never a
random outlier row. Every event starts from the victim's habitual context — its
identity, role, geography, device registry and resource catalogue — and only
the fields the attack actually changes are overwritten. A detector that learns
"unusual for *this* entity" therefore has something real to find, and one that
tries to memorise a categorical giveaway (an attacker-only `auth_method`, say)
has nothing to latch onto.

Difficulty is deliberately mixed. A uniformly blatant attack set produces a
near-perfect PR-AUC that proves nothing, so ``generator.attacks.stealth_fraction``
of each type is generated as a subtle variant: the defining signal is still
present, but the supporting signals that would make it trivial are not.

Every injector builds exactly one campaign and can be called on its own, which
is what the Phase 11 live simulator will use to push a fresh attack through the
real inference pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.schema import MALICIOUS_CLASSES, AttackType, EntityType
from src.utils.geo import haversine_km, implied_velocity_kmh
from src.utils.seeding import get_rng

from .entities import (
    COHORT_ARCHETYPES,
    REMOTE_EGRESS,
    EntityBehavioralProfile,
    RegisteredDevice,
    ResourceDef,
    random_ip_in_subnet,
)

# ---------------------------------------------------------------------------
# Hostile infrastructure — deliberately disjoint from the benign geography in
# entities.py, so an "obvious" campaign lands somewhere the population never
# legitimately connects from.
# ---------------------------------------------------------------------------

HOSTILE_ORIGINS: tuple[dict[str, Any], ...] = (
    {"country": "Russia", "city": "Moscow", "lat": 55.7558, "lon": 37.6173},
    {"country": "Nigeria", "city": "Lagos", "lat": 6.5244, "lon": 3.3792},
    {"country": "China", "city": "Shenzhen", "lat": 22.5431, "lon": 114.0579},
    {"country": "Brazil", "city": "Sao Paulo", "lat": -23.5505, "lon": -46.6333},
    {"country": "Netherlands", "city": "Amsterdam", "lat": 52.3676, "lon": 4.9041},
)

_SENSITIVITY_RANK: dict[str, int] = {"low": 0, "medium": 1, "high": 2, "critical": 3}

#: An impossible journey is one login plus whatever the attacker did next.
_TRAVEL_EVENT_RANGE = (1, 3)


def campaign_size_range(
    attack_type: AttackType, attacks_cfg: dict[str, Any], *, stealthy: bool
) -> tuple[int, int]:
    """Event count one campaign of this type and difficulty can plausibly have.

    Read from the same config keys the injectors clamp against, deliberately:
    the orchestrator's budget arithmetic is only trustworthy if its idea of a
    campaign's size is the injector's idea. An "obvious" brute force cannot be
    8 events long however tight the budget is, and an allocator that assumed
    otherwise would silently overshoot the target prevalence.
    """
    cfg = attacks_cfg[attack_type.value.lower()]

    if attack_type is AttackType.BRUTE_FORCE:
        low, high = cfg["stealth_attempts"] if stealthy else cfg["attempts"]
        return int(low), int(high)
    if attack_type is AttackType.IMPOSSIBLE_TRAVEL:
        return _TRAVEL_EVENT_RANGE
    if attack_type is AttackType.CREDENTIAL_STUFFING:
        # One attempt per victim at minimum; a few each at the top end.
        low, high = cfg["victims"]
        return int(low), int(high) * 3
    if attack_type is AttackType.LATERAL_MOVEMENT:
        low, high = cfg["stealth_hops"] if stealthy else cfg["hops"]
        return int(low), int(high)
    if attack_type is AttackType.DEVICE_SPOOFING:
        low, high = cfg["events"]
        return int(low), int(high)
    if attack_type is AttackType.LOW_AND_SLOW_EXFILTRATION:
        # Expected, not best-case: the daily count is drawn per day, so a
        # campaign spanning the maximum days still only averages the mean rate.
        day_low, day_high = cfg["days"]
        per_low, per_high = cfg["events_per_day"]
        mean_per_day = max(1.0, (per_low + per_high) / 2.0)
        return int(day_low * mean_per_day), int(day_high * mean_per_day)
    if attack_type is AttackType.INSIDER_DRIFT:
        # The ramp spreads whatever it is given, so only the floor is structural.
        return 2, int(cfg["days"][1]) * 3

    raise ValueError(f"no size range defined for {attack_type}")

#: Roles each attack plausibly targets. Absent types accept any user.
_ROLE_POOLS: dict[AttackType, tuple[str, ...]] = {
    AttackType.LATERAL_MOVEMENT: ("developer", "it_admin"),
    AttackType.INSIDER_DRIFT: ("hr", "finance", "developer"),
}

#: An entity needs some history before compromising it means anything.
_MIN_HISTORY_EVENTS = 20


@dataclass(frozen=True)
class AttackCampaign:
    """One coordinated intrusion, and the events it produced."""

    campaign_id: str
    attack_type: AttackType
    entity_ids: tuple[str, ...]
    stealthy: bool
    events: list[dict[str, Any]]

    def __len__(self) -> int:
        return len(self.events)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.events)


# ---------------------------------------------------------------------------
# Event construction
# ---------------------------------------------------------------------------


def _base_event(
    profile: EntityBehavioralProfile,
    timestamp: datetime,
    label: AttackType,
    campaign_id: str,
    *,
    device: RegisteredDevice | None = None,
) -> dict[str, Any]:
    """An event pre-filled with the entity's own habitual context.

    Injectors overwrite only the fields their attack actually changes, which is
    what keeps a campaign a believable mutation rather than a random row.
    """
    endpoint = device or profile.devices[0]
    return {
        "timestamp": timestamp,
        "entity_id": profile.entity_id,
        "entity_type": profile.entity_type.value,
        "role": profile.role,
        "source_ip": random_ip_in_subnet(profile.ip_network, profile.rng),
        "country": profile.home_geo.country,
        "city": profile.home_geo.city,
        "latitude": profile.home_geo.latitude,
        "longitude": profile.home_geo.longitude,
        "auth_method": profile.primary_auth_method,
        "auth_success": True,
        "resource_accessed": profile.cohort.resources[0].name,
        "action": profile.cohort.resources[0].action,
        "command_sequence": profile.cohort.resources[0].name.replace(" ", "_").upper(),
        "session_duration_s": round(profile.cohort.session_duration_mean_s, 1),
        "bytes_transferred": float(profile.cohort.resources[0].typical_bytes),
        "device_id": endpoint.device_id,
        "device_os": endpoint.device_os,
        "device_firmware": endpoint.device_firmware,
        "device_protocol": endpoint.device_protocol,
        "device_mac": endpoint.device_mac,
        "label": label.value,
        "is_attack": label.value in MALICIOUS_CLASSES,
        "campaign_id": campaign_id,
    }


def _apply_geo(event: dict[str, Any], site: dict[str, Any]) -> dict[str, Any]:
    event["country"] = site["country"]
    event["city"] = site["city"]
    event["latitude"] = float(site["lat"])
    event["longitude"] = float(site["lon"])
    return event


def _apply_resource(event: dict[str, Any], resource: ResourceDef) -> dict[str, Any]:
    event["resource_accessed"] = resource.name
    event["action"] = resource.action
    event["bytes_transferred"] = float(resource.typical_bytes)
    return event


def _attacker_ip(networks: Sequence[str], rng: np.random.Generator) -> str:
    network = str(rng.choice(list(networks)))
    return random_ip_in_subnet(network, rng)


def _spread_timestamps(
    start: datetime, span_seconds: float, count: int, rng: np.random.Generator
) -> list[datetime]:
    """``count`` strictly increasing whole-second stamps inside a window.

    Strictness matters: the dataset guarantees one event per entity per second,
    and a burst attack is the one place where random draws would otherwise
    collide constantly.
    """
    if count <= 1:
        return [start]
    span = max(float(span_seconds), float(count))
    offsets = np.sort(rng.uniform(0.0, span, size=count))
    seconds = np.round(offsets).astype(np.int64)
    for index in range(1, count):
        if seconds[index] <= seconds[index - 1]:
            seconds[index] = seconds[index - 1] + 1
    return [start + timedelta(seconds=int(value)) for value in seconds]


def _foreign_resources(profile: EntityBehavioralProfile) -> list[ResourceDef]:
    """Resources belonging to *other* cohorts, ascending by sensitivity.

    This is the pool a compromised account pivots into: real systems that exist
    in the enterprise but have no business appearing in this entity's history.
    """
    own = {resource.name for resource in profile.cohort.resources}
    pool: dict[str, ResourceDef] = {}
    for role, cohort in COHORT_ARCHETYPES.items():
        if role == profile.role:
            continue
        for resource in cohort.resources:
            if resource.name not in own:
                pool.setdefault(resource.name, resource)
    return sorted(
        pool.values(),
        key=lambda r: (_SENSITIVITY_RANK.get(r.sensitivity, 0), r.name),
    )


def _unknown_device(
    profile: EntityBehavioralProfile, rng: np.random.Generator, *, stealthy: bool
) -> RegisteredDevice:
    """A device the entity has never presented before.

    The stealthy form keeps the registered ``device_id`` and OS and changes only
    the hardware fingerprint, which is what a cloned or re-imaged endpoint looks
    like — far harder than an outright unknown machine.
    """
    known = profile.devices[0]
    mac = ":".join(f"{octet:02x}" for octet in rng.integers(0, 256, size=6))
    if stealthy:
        return RegisteredDevice(
            device_id=known.device_id,
            device_os=known.device_os,
            device_firmware=f"FW-{rng.integers(1, 9)}.{rng.integers(0, 9)}.{rng.integers(0, 9)}",
            device_protocol=known.device_protocol,
            device_mac=mac,
            is_primary=False,
        )
    return RegisteredDevice(
        device_id=f"UNREG-{rng.integers(100000, 999999)}",
        device_os=str(rng.choice(["Windows 7", "Kali Linux", "Android 9", "Unknown"])),
        device_firmware="UNKNOWN",
        device_protocol=known.device_protocol,
        device_mac=mac,
        is_primary=False,
    )


def _off_hours(day: datetime, rng: np.random.Generator) -> datetime:
    """A timestamp in the small hours of the given day."""
    return day.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(
        seconds=int(rng.integers(1 * 3600, 5 * 3600))
    )


def _working_hours(day: datetime, rng: np.random.Generator) -> datetime:
    return day.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(
        seconds=int(rng.integers(9 * 3600, 17 * 3600))
    )


def _pick(items: Sequence[Any], rng: np.random.Generator) -> Any:
    """Index-based choice; ``rng.choice`` mangles sequences of dataclasses."""
    return items[int(rng.integers(len(items)))]


# ---------------------------------------------------------------------------
# Injectors — one campaign each
# ---------------------------------------------------------------------------


def inject_brute_force(
    profile: EntityBehavioralProfile,
    *,
    campaign_id: str,
    rng: np.random.Generator,
    attacks_cfg: dict[str, Any],
    window_start: datetime,
    stealthy: bool,
    target_events: int,
) -> AttackCampaign:
    """One account, one source, a wall of failed authentications.

    Stealth variant trades volume for patience: a handful of attempts spread
    over hours, which never trips a naive "N failures in 5 minutes" rule.
    """
    cfg = attacks_cfg["brute_force"]
    low, high = cfg["stealth_attempts"] if stealthy else cfg["attempts"]
    attempts = int(np.clip(target_events, low, high))
    span_low, span_high = (
        cfg["stealth_window_minutes"] if stealthy else cfg["window_minutes"]
    )
    span_seconds = float(rng.uniform(span_low, span_high)) * 60.0

    entry_point = profile.cohort.resources[0]
    source_ip = _attacker_ip(attacks_cfg["attacker_networks"], rng)
    origin = None if stealthy else _pick(HOSTILE_ORIGINS, rng)
    breaks_in = bool(rng.random() < float(cfg["final_success_prob"]))

    events: list[dict[str, Any]] = []
    stamps = _spread_timestamps(window_start, span_seconds, attempts, rng)
    for index, stamp in enumerate(stamps):
        event = _base_event(profile, stamp, AttackType.BRUTE_FORCE, campaign_id)
        _apply_resource(event, entry_point)
        event["source_ip"] = source_ip
        event["auth_success"] = breaks_in and index == len(stamps) - 1
        event["session_duration_s"] = round(float(rng.uniform(0.4, 3.0)), 1)
        event["bytes_transferred"] = float(rng.integers(64, 512))
        if origin is not None:
            _apply_geo(event, origin)
        events.append(event)

    return AttackCampaign(
        campaign_id=campaign_id,
        attack_type=AttackType.BRUTE_FORCE,
        entity_ids=(profile.entity_id,),
        stealthy=stealthy,
        events=events,
    )


def inject_impossible_travel(
    profile: EntityBehavioralProfile,
    *,
    campaign_id: str,
    rng: np.random.Generator,
    attacks_cfg: dict[str, Any],
    anchor: dict[str, Any],
    stealthy: bool,
    target_events: int,
    max_plausible_kmh: float,
) -> AttackCampaign:
    """A second session from somewhere the entity could not have reached.

    The anchor is a *real* benign event already in the stream, so the resulting
    velocity is a genuine property of the dataset rather than a claim. The gap
    is shortened if necessary to guarantee the journey exceeds
    ``features.max_plausible_kmh`` with margin.

    Stealth variant travels to a city the entity legitimately uses, on a known
    device — leaving velocity as the only signal.
    """
    cfg = attacks_cfg["impossible_travel"]
    minimum_km = float(cfg["stealth_min_km"] if stealthy else cfg["min_km"])
    candidates = REMOTE_EGRESS if stealthy else HOSTILE_ORIGINS

    anchor_lat = float(anchor["latitude"])
    anchor_lon = float(anchor["longitude"])
    ranked = sorted(
        candidates,
        key=lambda site: haversine_km(anchor_lat, anchor_lon, site["lat"], site["lon"]),
        reverse=True,
    )
    viable = [
        site
        for site in ranked
        if haversine_km(anchor_lat, anchor_lon, site["lat"], site["lon"]) >= minimum_km
    ]
    destination = _pick(viable, rng) if viable else ranked[0]
    distance_km = haversine_km(
        anchor_lat, anchor_lon, destination["lat"], destination["lon"]
    )

    gap_low, gap_high = cfg["gap_minutes"]
    gap_minutes = float(rng.uniform(gap_low, gap_high))
    # Whatever the draw produced, the trip must stay physically impossible...
    gap_minutes = min(gap_minutes, distance_km / (max_plausible_kmh * 1.15) * 60.0)
    # ...and must land before the entity's next genuine event, so that the
    # velocity a detector measures is the one built here.
    gap_minutes = min(gap_minutes, float(anchor["next_gap_minutes"]) * 0.8)
    gap_minutes = max(gap_minutes, 1.0)

    # Checked with the same helper the feature layer uses: a campaign labelled
    # IMPOSSIBLE_TRAVEL whose journey is merely fast is a corrupt label, and it
    # would be measured as a false negative for the rest of the project.
    velocity = implied_velocity_kmh(distance_km, gap_minutes * 60.0)
    if velocity <= max_plausible_kmh:
        raise RuntimeError(
            f"{campaign_id}: {distance_km:.0f} km in {gap_minutes:.1f} min implies "
            f"{velocity:.0f} km/h, which does not exceed the "
            f"{max_plausible_kmh:.0f} km/h plausibility threshold"
        )

    login_at = anchor["timestamp"] + timedelta(minutes=gap_minutes)
    device = profile.devices[0] if stealthy else _unknown_device(profile, rng, stealthy=False)
    source_ip = _attacker_ip(attacks_cfg["attacker_networks"], rng)

    low, high = _TRAVEL_EVENT_RANGE
    count = int(np.clip(target_events, low, high))
    remaining_minutes = max(
        1.0, float(anchor["next_gap_minutes"]) - gap_minutes
    )
    stamps = _spread_timestamps(
        login_at, min(remaining_minutes, 30.0) * 60.0, count, rng
    )

    events: list[dict[str, Any]] = []
    trail: list[str] = []
    for index, stamp in enumerate(stamps):
        resource = (
            profile.cohort.resources[0]
            if index == 0
            else _pick(profile.cohort.resources, rng)
        )
        event = _base_event(
            profile, stamp, AttackType.IMPOSSIBLE_TRAVEL, campaign_id, device=device
        )
        _apply_resource(event, resource)
        _apply_geo(event, destination)
        event["source_ip"] = source_ip
        trail.append(resource.name.replace(" ", "_").upper())
        event["command_sequence"] = "->".join(trail)
        events.append(event)

    return AttackCampaign(
        campaign_id=campaign_id,
        attack_type=AttackType.IMPOSSIBLE_TRAVEL,
        entity_ids=(profile.entity_id,),
        stealthy=stealthy,
        events=events,
    )


def inject_credential_stuffing(
    victims: Sequence[EntityBehavioralProfile],
    *,
    campaign_id: str,
    rng: np.random.Generator,
    attacks_cfg: dict[str, Any],
    window_start: datetime,
    stealthy: bool,
    target_events: int,
) -> AttackCampaign:
    """A few hostile sources sprayed across many accounts.

    The distinguishing shape is the fan-out: victim count vastly exceeds source
    count, and one tool fingerprint recurs across unrelated identities. Each
    victim uses its *own* auth method, so the campaign cannot be spotted by a
    categorical tell.
    """
    cfg = attacks_cfg["credential_stuffing"]
    ip_low, ip_high = cfg["stealth_source_ips"] if stealthy else cfg["source_ips"]
    source_count = int(rng.integers(ip_low, ip_high + 1))
    sources = [
        _attacker_ip(attacks_cfg["attacker_networks"], rng) for _ in range(source_count)
    ]

    span_low, span_high = (
        cfg["stealth_window_minutes"] if stealthy else cfg["window_minutes"]
    )
    span_seconds = float(rng.uniform(span_low, span_high)) * 60.0
    origin = _pick(REMOTE_EGRESS if stealthy else HOSTILE_ORIGINS, rng)
    success_rate = float(cfg["success_rate"])

    # One fingerprint shared by every victim: the attacker's tooling, and the
    # single strongest cross-entity link a detector can pick up on.
    tool = RegisteredDevice(
        device_id=f"STUFF-{campaign_id[-3:]}",
        device_os="Unknown",
        device_firmware="UNKNOWN",
        device_protocol="HTTPS",
        device_mac=":".join(f"{octet:02x}" for octet in rng.integers(0, 256, size=6)),
        is_primary=False,
    )
    # Integer division alone would systematically lose up to one attempt per
    # victim, which at scale pulls the whole dataset under target prevalence.
    base, remainder = divmod(max(target_events, len(victims)), max(len(victims), 1))

    events: list[dict[str, Any]] = []
    for index, victim in enumerate(victims):
        per_victim = max(1, base + (1 if index < remainder else 0))
        offset = float(rng.uniform(0.0, span_seconds))
        stamps = _spread_timestamps(
            window_start + timedelta(seconds=offset), 90.0, per_victim, rng
        )
        source_ip = sources[int(rng.integers(len(sources)))]
        for stamp in stamps:
            event = _base_event(
                victim, stamp, AttackType.CREDENTIAL_STUFFING, campaign_id, device=tool
            )
            _apply_resource(event, victim.cohort.resources[0])
            _apply_geo(event, origin)
            event["source_ip"] = source_ip
            event["auth_success"] = bool(rng.random() < success_rate)
            event["session_duration_s"] = round(float(rng.uniform(0.3, 2.0)), 1)
            event["bytes_transferred"] = float(rng.integers(64, 400))
            events.append(event)

    return AttackCampaign(
        campaign_id=campaign_id,
        attack_type=AttackType.CREDENTIAL_STUFFING,
        entity_ids=tuple(victim.entity_id for victim in victims),
        stealthy=stealthy,
        events=events,
    )


def inject_lateral_movement(
    profile: EntityBehavioralProfile,
    *,
    campaign_id: str,
    rng: np.random.Generator,
    attacks_cfg: dict[str, Any],
    window_start: datetime,
    stealthy: bool,
    target_events: int,
) -> AttackCampaign:
    """A compromised account walking into systems it has no business touching.

    Hops are drawn from other cohorts' real catalogues and ordered by ascending
    sensitivity, finishing on a critical asset. Credentials, device and network
    stay the entity's own — the attacker is already inside.
    """
    cfg = attacks_cfg["lateral_movement"]
    low, high = cfg["stealth_hops"] if stealthy else cfg["hops"]
    hops = int(np.clip(target_events, low, high))

    pool = _foreign_resources(profile)
    critical = [r for r in pool if r.sensitivity == "critical"]
    approach = [r for r in pool if r.sensitivity != "critical"]

    chain: list[ResourceDef] = []
    if approach:
        picks = rng.choice(
            len(approach), size=min(hops - 1, len(approach)), replace=False
        )
        chain = sorted(
            (approach[int(i)] for i in np.atleast_1d(picks)),
            key=lambda r: _SENSITIVITY_RANK.get(r.sensitivity, 0),
        )
    chain.append(_pick(critical, rng) if critical else _pick(pool, rng))

    # Minutes apart: a hands-on-keyboard operator, not an automated scan.
    stamps = _spread_timestamps(window_start, 60.0 * 8 * len(chain), len(chain), rng)

    events: list[dict[str, Any]] = []
    trail: list[str] = []
    for stamp, resource in zip(stamps, chain):
        event = _base_event(profile, stamp, AttackType.LATERAL_MOVEMENT, campaign_id)
        _apply_resource(event, resource)
        trail.append(resource.name.replace(" ", "_").upper())
        event["command_sequence"] = "->".join(trail)
        event["session_duration_s"] = round(float(rng.uniform(120.0, 900.0)), 1)
        event["bytes_transferred"] = float(
            max(64.0, rng.normal(resource.typical_bytes * 1.6, resource.typical_bytes * 0.3))
        )
        events.append(event)

    return AttackCampaign(
        campaign_id=campaign_id,
        attack_type=AttackType.LATERAL_MOVEMENT,
        entity_ids=(profile.entity_id,),
        stealthy=stealthy,
        events=events,
    )


def inject_device_spoofing(
    profile: EntityBehavioralProfile,
    *,
    campaign_id: str,
    rng: np.random.Generator,
    attacks_cfg: dict[str, Any],
    window_start: datetime,
    stealthy: bool,
    target_events: int,
) -> AttackCampaign:
    """Right identity, right place, right hours — wrong hardware.

    Stealth variant reuses the registered ``device_id`` and OS and changes only
    the MAC and firmware, which is the fingerprint drift a cloned endpoint
    produces rather than an obviously foreign machine.
    """
    cfg = attacks_cfg["device_spoofing"]
    low, high = cfg["events"]
    count = int(np.clip(target_events, low, high))
    device = _unknown_device(profile, rng, stealthy=stealthy)

    stamps = _spread_timestamps(window_start, 2 * 3600.0, count, rng)
    events: list[dict[str, Any]] = []
    trail: list[str] = []
    for stamp in stamps:
        resource = _pick(profile.cohort.resources, rng)
        event = _base_event(
            profile, stamp, AttackType.DEVICE_SPOOFING, campaign_id, device=device
        )
        _apply_resource(event, resource)
        trail.append(resource.name.replace(" ", "_").upper())
        event["command_sequence"] = "->".join(trail)
        events.append(event)

    return AttackCampaign(
        campaign_id=campaign_id,
        attack_type=AttackType.DEVICE_SPOOFING,
        entity_ids=(profile.entity_id,),
        stealthy=stealthy,
        events=events,
    )


def inject_low_and_slow_exfiltration(
    profile: EntityBehavioralProfile,
    *,
    campaign_id: str,
    rng: np.random.Generator,
    attacks_cfg: dict[str, Any],
    window_start: datetime,
    stealthy: bool,
    target_events: int,
    max_days: int | None = None,
) -> AttackCampaign:
    """Patient theft: small off-hours transfers that compound over weeks.

    The resource is one the entity legitimately uses, so nothing about *what*
    is touched looks wrong. Only the volume trend and the hour give it away,
    and the stealth variant grows slowly enough to stay inside the entity's
    normal range for most of the campaign.
    """
    cfg = attacks_cfg["low_and_slow_exfiltration"]
    day_low, day_high = cfg["days"]
    per_low, per_high = cfg["events_per_day"]
    growth = float(cfg["stealth_bytes_growth"] if stealthy else cfg["bytes_growth"])

    mean_per_day = max(1.0, (per_low + per_high) / 2.0)
    days = int(np.clip(round(target_events / mean_per_day), day_low, day_high))
    if max_days is not None:
        days = max(1, min(days, max_days))
    resource = max(profile.cohort.resources, key=lambda r: r.typical_bytes)

    events: list[dict[str, Any]] = []
    for day_index in range(days):
        day = window_start + timedelta(days=day_index)
        count = int(rng.integers(per_low, per_high + 1))
        stamps = _spread_timestamps(_off_hours(day, rng), 3600.0, count, rng)
        volume = resource.typical_bytes * (growth**day_index)
        for stamp in stamps:
            event = _base_event(
                profile, stamp, AttackType.LOW_AND_SLOW_EXFILTRATION, campaign_id
            )
            _apply_resource(event, resource)
            event["bytes_transferred"] = float(
                max(64.0, rng.normal(volume, volume * 0.12))
            )
            event["session_duration_s"] = round(float(rng.uniform(60.0, 400.0)), 1)
            events.append(event)

    return AttackCampaign(
        campaign_id=campaign_id,
        attack_type=AttackType.LOW_AND_SLOW_EXFILTRATION,
        entity_ids=(profile.entity_id,),
        stealthy=stealthy,
        events=events,
    )


def inject_insider_drift(
    profile: EntityBehavioralProfile,
    *,
    campaign_id: str,
    rng: np.random.Generator,
    attacks_cfg: dict[str, Any],
    window_start: datetime,
    stealthy: bool,
    target_events: int,
    max_days: int | None = None,
) -> AttackCampaign:
    """Legitimate-looking expansion of an employee's resource footprint.

    This is the deliberate false-positive trap and the concept-drift exhibit,
    not an intrusion: normal hours, known device, own network, successful
    authentication, and a footprint that widens gradually the way a genuine
    role change does. It carries ``label=INSIDER_DRIFT`` but ``is_attack`` is
    False, matching ``MALICIOUS_CLASSES`` in the schema, so it is reported
    separately instead of counting as a missed intrusion.
    """
    cfg = attacks_cfg["insider_drift"]
    day_low, day_high = cfg["days"]
    days = int(np.clip(target_events // 2, day_low, day_high))
    if max_days is not None:
        days = max(1, min(days, max_days))

    resource_low, resource_high = cfg["new_resources"]
    wanted = int(rng.integers(resource_low, resource_high + 1))
    # Lower-sensitivity systems first: a plausible remit change, not a smash-and-grab.
    pool = [r for r in _foreign_resources(profile) if r.sensitivity != "critical"]
    if not pool:
        pool = _foreign_resources(profile)
    picks = rng.choice(len(pool), size=min(wanted, len(pool)), replace=False)
    adopted = [pool[int(index)] for index in np.atleast_1d(picks)]

    # A linear ramp: rare at first, routine by the end.
    weights = np.arange(1, days + 1, dtype=float)
    weights /= weights.sum()
    counts = np.floor(weights * target_events).astype(int)
    shortfall = target_events - int(counts.sum())
    for index in range(shortfall):
        counts[-(index % days) - 1] += 1

    events: list[dict[str, Any]] = []
    for day_index, count in enumerate(counts):
        if count <= 0:
            continue
        day = window_start + timedelta(days=day_index)
        stamps = _spread_timestamps(_working_hours(day, rng), 4 * 3600.0, int(count), rng)
        for stamp in stamps:
            resource = _pick(adopted, rng)
            event = _base_event(profile, stamp, AttackType.INSIDER_DRIFT, campaign_id)
            _apply_resource(event, resource)
            event["command_sequence"] = resource.name.replace(" ", "_").upper()
            events.append(event)

    return AttackCampaign(
        campaign_id=campaign_id,
        attack_type=AttackType.INSIDER_DRIFT,
        entity_ids=(profile.entity_id,),
        stealthy=stealthy,
        events=events,
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

#: An impossible-travel anchor needs enough clear air after it that the injected
#: login really is the entity's next event, otherwise the velocity a detector
#: measures would be against some intervening benign row instead.
_MIN_ANCHOR_GAP_MINUTES = 90.0

#: Campaigns that unfold over days need room inside the horizon.
_CAMPAIGN_SPAN_KEYS: dict[AttackType, str] = {
    AttackType.LOW_AND_SLOW_EXFILTRATION: "low_and_slow_exfiltration",
    AttackType.INSIDER_DRIFT: "insider_drift",
}


@dataclass(frozen=True)
class _Allocation:
    attack_type: AttackType
    campaigns: int
    events_per_campaign: int


class AttackOrchestrator:
    """Turns an event budget into concrete campaigns against real entities."""

    def __init__(
        self,
        population: list[EntityBehavioralProfile],
        benign: pd.DataFrame,
        *,
        start_date: datetime,
        simulation_days: int,
        attacks_cfg: dict[str, Any],
        attack_mix: dict[str, float],
        prevalence: float,
        max_plausible_kmh: float,
    ) -> None:
        if not population:
            raise ValueError("cannot inject attacks into an empty population")
        if not 0.0 <= prevalence < 1.0:
            raise ValueError(f"attack prevalence must be in [0, 1); got {prevalence}")

        self.population = population
        self.benign = benign
        self.start_date = start_date
        self.simulation_days = simulation_days
        self.horizon_end = start_date + timedelta(days=simulation_days)
        self.attacks_cfg = attacks_cfg
        self.attack_mix = attack_mix
        self.prevalence = prevalence
        self.max_plausible_kmh = max_plausible_kmh
        self.stealth_fraction = float(attacks_cfg.get("stealth_fraction", 0.0))

        # Solving for the count that makes attacks exactly `prevalence` of the
        # merged total, since the benign stream was already sized to leave room.
        self.budget = int(round(len(benign) * prevalence / (1.0 - prevalence)))
        self.history: dict[str, int] = (
            benign.groupby("entity_id").size().to_dict() if not benign.empty else {}
        )
        self._anchors = self._build_anchor_index()

    # -- planning -----------------------------------------------------------
    def _build_anchor_index(self) -> pd.DataFrame:
        """Benign events usable as an impossible-travel origin."""
        if self.benign.empty:
            return pd.DataFrame()

        columns = ["entity_id", "timestamp", "latitude", "longitude"]
        frame = self.benign[columns].sort_values(["entity_id", "timestamp"])
        following = frame.groupby("entity_id")["timestamp"].shift(-1)
        gap = (following - frame["timestamp"]).dt.total_seconds() / 60.0
        # An entity's final event has all the remaining horizon after it.
        tail = (self.horizon_end - frame["timestamp"]).dt.total_seconds() / 60.0
        frame = frame.assign(next_gap_minutes=gap.fillna(tail))
        return frame[frame["next_gap_minutes"] >= _MIN_ANCHOR_GAP_MINUTES]

    def _solve_campaign_count(
        self, attack_type: AttackType, type_budget: float, configured: int
    ) -> int:
        """Campaign count whose expected event total lands nearest the budget.

        The configured count is a preference, not a constraint: prevalence and
        plausible campaign shapes are the things that have to hold. A small
        profile therefore gets fewer campaigns than configured, and a large one
        gets more when a single campaign cannot absorb its share. Ties break
        toward the configured count.
        """
        stealth_low, stealth_high = campaign_size_range(
            attack_type, self.attacks_cfg, stealthy=True
        )
        obvious_low, obvious_high = campaign_size_range(
            attack_type, self.attacks_cfg, stealthy=False
        )

        def error_for(count: int) -> float:
            target = type_budget / count
            stealth = int(round(count * self.stealth_fraction))
            expected = stealth * min(max(target, stealth_low), stealth_high) + (
                count - stealth
            ) * min(max(target, obvious_low), obvious_high)
            return abs(expected - type_budget)

        # The ceiling has to reach the count at which even maximally-sized
        # campaigns could absorb the budget, or a type whose incidents are
        # inherently tiny (impossible travel is 1-3 events) silently undershoots
        # its share and drags the whole dataset below target prevalence.
        saturating = math.ceil(type_budget / max(1, min(stealth_high, obvious_high)))
        counts = range(1, max(configured * 6, saturating + 1, 12) + 1)
        # A single campaign can absorb any budget exactly, so pure error
        # minimisation would always return 1 — useless for campaign-level
        # recall. Accept everything inside a tolerance band instead, then take
        # the count nearest the configured preference.
        tolerance = max(1.0, type_budget * 0.10)
        acceptable = [count for count in counts if error_for(count) <= tolerance]
        if acceptable:
            return min(acceptable, key=lambda count: abs(count - configured))
        return min(counts, key=error_for)

    def _allocations(self) -> list[_Allocation]:
        total_weight = sum(float(v) for v in self.attack_mix.values())
        allocations: list[_Allocation] = []
        for attack_type in AttackType:
            if attack_type is AttackType.BENIGN:
                continue
            key = attack_type.value.lower()
            weight = float(self.attack_mix.get(key, 0.0))
            if weight <= 0.0 or key not in self.attacks_cfg:
                continue

            type_budget = self.budget * weight / total_weight
            configured = max(1, int(self.attacks_cfg[key]["campaigns"]))
            campaigns = self._solve_campaign_count(attack_type, type_budget, configured)
            allocations.append(
                _Allocation(
                    attack_type=attack_type,
                    campaigns=campaigns,
                    events_per_campaign=max(1, round(type_budget / campaigns)),
                )
            )
        return allocations

    def _eligible(self, attack_type: AttackType) -> list[EntityBehavioralProfile]:
        if attack_type is AttackType.LOW_AND_SLOW_EXFILTRATION:
            wanted = (EntityType.USER, EntityType.SERVICE_ACCOUNT)
        else:
            wanted = (EntityType.USER,)
        pool = [p for p in self.population if p.entity_type in wanted]

        roles = _ROLE_POOLS.get(attack_type)
        if roles:
            pool = [p for p in pool if p.role in roles] or pool

        experienced = [
            p for p in pool if self.history.get(p.entity_id, 0) >= _MIN_HISTORY_EVENTS
        ]
        return experienced or pool

    def _campaign_start(
        self, rng: np.random.Generator, stealthy: bool, span_days: int
    ) -> datetime:
        latest = max(0, self.simulation_days - span_days)
        day = self.start_date + timedelta(days=int(rng.integers(0, latest + 1)))
        return _working_hours(day, rng) if stealthy else _off_hours(day, rng)

    def _span_days(self, attack_type: AttackType) -> int:
        """How much room to reserve after the start, capped to the horizon.

        The configured maxima (up to 35 days for insider drift) exceed the dev
        profile's 21-day horizon, so a long campaign is allowed to start in the
        first half of the run and is then truncated to fit.
        """
        key = _CAMPAIGN_SPAN_KEYS.get(attack_type)
        if key is None:
            return 1
        configured = int(self.attacks_cfg[key]["days"][1])
        return min(configured, max(1, self.simulation_days // 2))

    def _pick_anchor(
        self, entity_id: str, rng: np.random.Generator
    ) -> dict[str, Any] | None:
        if self._anchors.empty:
            return None
        options = self._anchors[self._anchors["entity_id"] == entity_id]
        if options.empty:
            return None
        row = options.iloc[int(rng.integers(len(options)))]
        return {
            "timestamp": row["timestamp"].to_pydatetime(),
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "next_gap_minutes": float(row["next_gap_minutes"]),
        }

    # -- execution ----------------------------------------------------------
    def _build_one(
        self,
        allocation: _Allocation,
        campaign_id: str,
        rng: np.random.Generator,
        eligible: list[EntityBehavioralProfile],
        stealthy: bool,
    ) -> AttackCampaign | None:
        attack_type = allocation.attack_type
        target = allocation.events_per_campaign

        if attack_type is AttackType.CREDENTIAL_STUFFING:
            low, high = self.attacks_cfg["credential_stuffing"]["victims"]
            count = min(int(rng.integers(low, high + 1)), len(eligible))
            picks = rng.choice(len(eligible), size=max(1, count), replace=False)
            victims = [eligible[int(index)] for index in np.atleast_1d(picks)]
            return inject_credential_stuffing(
                victims,
                campaign_id=campaign_id,
                rng=rng,
                attacks_cfg=self.attacks_cfg,
                window_start=self._campaign_start(rng, stealthy, 1),
                stealthy=stealthy,
                target_events=target,
            )

        profile = _pick(eligible, rng)

        if attack_type is AttackType.IMPOSSIBLE_TRAVEL:
            anchor = self._pick_anchor(profile.entity_id, rng)
            if anchor is None:
                return None
            return inject_impossible_travel(
                profile,
                campaign_id=campaign_id,
                rng=rng,
                attacks_cfg=self.attacks_cfg,
                anchor=anchor,
                stealthy=stealthy,
                target_events=target,
                max_plausible_kmh=self.max_plausible_kmh,
            )

        injectors = {
            AttackType.BRUTE_FORCE: inject_brute_force,
            AttackType.LATERAL_MOVEMENT: inject_lateral_movement,
            AttackType.DEVICE_SPOOFING: inject_device_spoofing,
            AttackType.LOW_AND_SLOW_EXFILTRATION: inject_low_and_slow_exfiltration,
            AttackType.INSIDER_DRIFT: inject_insider_drift,
        }
        window_start = self._campaign_start(rng, stealthy, self._span_days(attack_type))
        extra: dict[str, Any] = {}
        if attack_type in _CAMPAIGN_SPAN_KEYS:
            extra["max_days"] = max(1, (self.horizon_end - window_start).days)

        return injectors[attack_type](
            profile,
            campaign_id=campaign_id,
            rng=rng,
            attacks_cfg=self.attacks_cfg,
            window_start=window_start,
            stealthy=stealthy,
            target_events=target,
            **extra,
        )

    def inject(self) -> list[AttackCampaign]:
        """Build every campaign the budget calls for."""
        campaigns: list[AttackCampaign] = []
        for allocation in self._allocations():
            attack_type = allocation.attack_type
            rng = get_rng(f"generator.attacks.{attack_type.value}")
            eligible = self._eligible(attack_type)
            if not eligible:
                continue

            stealth_flags = np.zeros(allocation.campaigns, dtype=bool)
            stealth_flags[: int(round(allocation.campaigns * self.stealth_fraction))] = True
            rng.shuffle(stealth_flags)

            for index in range(allocation.campaigns):
                campaign = self._build_one(
                    allocation,
                    f"CMP-{attack_type.value}-{index + 1:03d}",
                    rng,
                    eligible,
                    bool(stealth_flags[index]),
                )
                if campaign is not None and campaign.events:
                    campaigns.append(campaign)
        return campaigns


def merge_campaigns(
    benign: pd.DataFrame, campaigns: Sequence[AttackCampaign]
) -> pd.DataFrame:
    """Fold campaigns into the benign stream and re-finalize the result."""
    from .normal_behavior import finalize_events

    rows = [event for campaign in campaigns for event in campaign.events]
    if not rows:
        return finalize_events(benign)

    combined = pd.concat(
        [benign.drop(columns=["event_id"], errors="ignore"), pd.DataFrame(rows)],
        ignore_index=True,
    )
    merged = finalize_events(combined)
    # concat discards attrs, and session length is a benign-stream property that
    # attack events (which have no sessions) should not be allowed to dilute.
    merged.attrs["median_session_events"] = benign.attrs.get("median_session_events")
    return merged


def campaign_summary(campaigns: Sequence[AttackCampaign]) -> dict[str, Any]:
    """Per-type campaign and event counts for the run summary."""
    per_type: dict[str, dict[str, int]] = {}
    for campaign in campaigns:
        entry = per_type.setdefault(
            campaign.attack_type.value, {"campaigns": 0, "events": 0, "stealth": 0}
        )
        entry["campaigns"] += 1
        entry["events"] += len(campaign.events)
        entry["stealth"] += int(campaign.stealthy)
    return {
        "campaigns": len(campaigns),
        "campaign_events": sum(len(c) for c in campaigns),
        "stealth_campaigns": sum(1 for c in campaigns if c.stealthy),
        "per_attack": per_type,
    }


__all__ = [
    "HOSTILE_ORIGINS",
    "campaign_size_range",
    "AttackCampaign",
    "AttackOrchestrator",
    "campaign_summary",
    "inject_brute_force",
    "inject_credential_stuffing",
    "inject_device_spoofing",
    "inject_impossible_travel",
    "inject_insider_drift",
    "inject_lateral_movement",
    "inject_low_and_slow_exfiltration",
    "merge_campaigns",
]
