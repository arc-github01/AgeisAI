"""Session-based normal behaviour simulation engine.

Unlike row-at-a-time generators, this module simulates *work sessions*:

    Day planner  ->  Session scheduler  ->  Markov resource walk  ->  Event rows

Each session produces a short sequence of causally linked events that share
device context, geography, and a growing ``command_sequence`` string. That
sequence is exactly what the downstream Markov anomaly scorer will consume.

Three invariants the rest of the pipeline relies on:

* **The horizon is exactly ``generator.days``.** The event budget is spread
  across those days by raising each entity's session rate, never by quietly
  extending the calendar. Weekday/weekend structure therefore stays aligned
  with real dates.
* **One entity never has two events in the same second.** Geo-velocity and
  inter-event-frequency features divide by the gap to an entity's previous
  event, so a zero gap is a division by zero waiting to happen.
* **Everything is drawn from seeded streams.** Re-running with an unchanged
  ``seed.master`` reproduces the dataset row for row, ``event_id`` included.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from src.schema import EVENT_COLUMNS, AttackType, EntityType
from src.utils.seeding import get_rng

from .entities import (
    COHORT_ARCHETYPES,
    CohortArchetype,
    EntityBehavioralProfile,
    RegisteredDevice,
    ResourceDef,
    pick_session_location,
    random_ip_in_subnet,
)

#: Hours of the day a human-operated entity spreads its sessions over. Sessions
#: run past it only when the day's budget cannot fit inside the window.
_WORKDAY_HOURS = 10.0

#: Two sessions of the same entity never abut more closely than this.
_MIN_SESSION_GAP_S = 120.0

#: A single day carries at most this multiple of an entity's mean daily volume,
#: so a backlog cannot pile up into an implausible burst at the end of the run.
_DAILY_BURST_CAP = 3.0

#: Monte-Carlo samples used to estimate the mean event yield of one session.
_SESSION_YIELD_SAMPLES = 1000

#: Timestamp collisions are nudged forward a second at a time; this bounds it.
_MAX_DECONFLICT_PASSES = 8


@dataclass(frozen=True)
class SessionPlan:
    """One scheduled, non-overlapping session window."""

    start: datetime
    n_steps: int
    duration_s: float


def _resource_lookup(cohort: CohortArchetype) -> dict[str, ResourceDef]:
    return {r.name: r for r in cohort.resources}


def _markov_step(current: str, cohort: CohortArchetype, rng: np.random.Generator) -> str:
    """Walk the cohort transition graph; return 'END' to terminate session."""
    options = cohort.transitions.get(current, {"END": 1.0})
    labels = list(options.keys())
    probs = np.array([options[k] for k in labels], dtype=float)
    probs /= probs.sum()
    return str(rng.choice(labels, p=probs))


def _walk_resources(
    cohort: CohortArchetype, n_steps: int, rng: np.random.Generator
) -> list[ResourceDef]:
    """Resources touched by one session, in order."""
    resources = _resource_lookup(cohort)
    steps: list[ResourceDef] = []
    state = "START"
    for _ in range(n_steps):
        nxt = _markov_step(state, cohort, rng)
        if nxt == "END":
            break
        if nxt in resources:
            steps.append(resources[nxt])
            state = nxt
    return steps or [cohort.resources[0]]


@lru_cache(maxsize=None)
def _expected_events_per_session(role: str) -> float:
    """Mean number of events one session of this cohort yields.

    Needed to convert a per-day *event* budget into a per-day *session* count.
    It is measured rather than assumed because the transition graph's ``END``
    probabilities usually terminate a walk well before ``session_steps_mean``.
    """
    cohort = COHORT_ARCHETYPES[role]
    rng = get_rng(f"generator.session_yield.{role}")
    total = 0
    for _ in range(_SESSION_YIELD_SAMPLES):
        n_steps = max(2, int(rng.poisson(cohort.session_steps_mean)))
        total += len(_walk_resources(cohort, n_steps, rng))
    return total / _SESSION_YIELD_SAMPLES


def _pick_device(profile: EntityBehavioralProfile) -> RegisteredDevice:
    rng = profile.rng
    if len(profile.devices) == 1:
        return profile.devices[0]
    weights = np.array([0.85 if d.is_primary else 0.15 for d in profile.devices])
    idx = int(rng.choice(len(profile.devices), p=weights / weights.sum()))
    return profile.devices[idx]


def active_days(
    profile: EntityBehavioralProfile,
    start_date: datetime,
    simulation_days: int,
) -> list[int]:
    """Day offsets on which this entity does anything at all.

    Decided up front so the event budget can be divided by the days the entity
    actually works. Dividing by raw calendar days instead would under-target
    every working day and leave the budget unmet at the end of the horizon.
    """
    cohort = profile.cohort
    rng = profile.rng
    if cohort.entity_type == EntityType.EDGE_DEVICE:
        return list(range(simulation_days))

    days: list[int] = []
    for offset in range(simulation_days):
        weekday = (start_date + timedelta(days=offset)).weekday()
        if weekday in profile.working_days or rng.random() < cohort.weekend_activity_prob:
            days.append(offset)
    return days


def plan_day(
    profile: EntityBehavioralProfile,
    day_offset: int,
    start_date: datetime,
    noise_cfg: dict[str, Any],
    event_target: int,
    *,
    not_before: datetime | None = None,
    horizon_end: datetime | None = None,
) -> list[SessionPlan]:
    """Lay out one day's sessions so that none of them overlap.

    Session starts are spread over the entity's active window in slots. A
    session that outgrows its slot simply pushes the next one later, which is
    what keeps the no-overlap guarantee intact on busy days.

    The window is allowed to run past midnight, because a night shift genuinely
    does. ``not_before`` carries the previous day's finishing time so a wrapped
    shift cannot collide with the next one.
    """
    cohort = profile.cohort
    rng = profile.rng
    day_start = start_date + timedelta(days=day_offset)
    is_edge = cohort.entity_type == EntityType.EDGE_DEVICE

    n_sessions = max(1, round(event_target / _expected_events_per_session(cohort.role)))

    if is_edge:
        # OT/IoT endpoints beacon around the clock rather than on a shift, so
        # the first beacon is jittered within its own slot instead of anchored
        # to a login hour.
        window_start_h = float(rng.random() * 24.0 / n_sessions)
        window_hours = 24.0
    else:
        jitter_std = float(noise_cfg.get("hour_jitter_std", 1.2))
        window_start_h = float(
            np.clip(rng.normal(profile.preferred_login_hour, jitter_std), 0.0, 23.0)
        )
        window_hours = _WORKDAY_HOURS

    window_start = (day_start + timedelta(hours=window_start_h)).replace(microsecond=0)
    window_end = window_start + timedelta(hours=window_hours)
    cursor = max(window_start, not_before) if not_before is not None else window_start
    slot_s = window_hours * 3600.0 / n_sessions

    plans: list[SessionPlan] = []
    for _ in range(n_sessions):
        if cursor >= window_end:
            break
        duration_s = float(
            max(
                30.0,
                rng.normal(
                    cohort.session_duration_mean_s,
                    cohort.session_duration_mean_s * 0.25,
                ),
            )
        )
        if horizon_end is not None and cursor + timedelta(seconds=duration_s) > horizon_end:
            break
        plans.append(
            SessionPlan(
                start=cursor,
                n_steps=max(2, int(rng.poisson(cohort.session_steps_mean))),
                duration_s=duration_s,
            )
        )
        idle_s = max(_MIN_SESSION_GAP_S, slot_s - duration_s)
        cursor += timedelta(
            seconds=duration_s + max(_MIN_SESSION_GAP_S, float(rng.normal(idle_s, idle_s * 0.25)))
        )
        cursor = cursor.replace(microsecond=0)
    return plans


def simulate_session(
    profile: EntityBehavioralProfile,
    plan: SessionPlan,
    noise_cfg: dict[str, Any],
    session_id: str,
) -> list[dict[str, Any]]:
    """Simulate one coherent session as multiple canonical event rows."""
    cohort = profile.cohort
    rng = profile.rng
    device = _pick_device(profile)
    geo = pick_session_location(profile, noise_cfg)
    source_ip = random_ip_in_subnet(profile.ip_network, rng)
    auth_success = rng.random() > float(
        noise_cfg.get("benign_auth_failure_rate", cohort.auth_failure_rate)
    )

    steps = _walk_resources(cohort, plan.n_steps, rng)
    step_spacing_s = plan.duration_s / len(steps)
    off_profile_prob = float(noise_cfg.get("off_profile_resource_prob", 0.04))

    events: list[dict[str, Any]] = []
    command_parts: list[str] = []

    for step_idx, resource in enumerate(steps):
        # Occasional off-profile resource: legitimate drift, not an attack.
        if step_idx > 0 and rng.random() < off_profile_prob:
            resource = cohort.resources[int(rng.integers(len(cohort.resources)))]
        command_parts.append(resource.name.replace(" ", "_").upper())

        timestamp = plan.start + timedelta(seconds=round(step_idx * step_spacing_s))
        bytes_transferred = max(
            64.0, float(rng.normal(resource.typical_bytes, resource.typical_bytes * 0.3))
        )

        events.append(
            {
                "timestamp": timestamp,
                "entity_id": profile.entity_id,
                "entity_type": profile.entity_type.value,
                "role": profile.role,
                "source_ip": source_ip,
                "country": geo.country,
                "city": geo.city,
                "latitude": geo.latitude,
                "longitude": geo.longitude,
                "auth_method": profile.primary_auth_method,
                "auth_success": auth_success if step_idx == 0 else True,
                "resource_accessed": resource.name,
                "action": resource.action,
                "command_sequence": "->".join(command_parts),
                "session_duration_s": round(plan.duration_s, 1),
                "bytes_transferred": round(bytes_transferred, 0),
                "device_id": device.device_id,
                "device_os": device.device_os,
                "device_firmware": device.device_firmware,
                "device_protocol": device.device_protocol,
                "device_mac": device.device_mac,
                "label": AttackType.BENIGN.value,
                "is_attack": False,
                "campaign_id": None,
                "_session_id": session_id,  # dropped before the frame is returned
            }
        )

    return events


def _deconflict_timestamps(frame: pd.DataFrame) -> pd.DataFrame:
    """Guarantee at most one event per (entity, second).

    Collisions are pushed forward one second at a time. Doing it as a repeated
    ``cumcount`` keeps the operation vectorised and order-stable, so the result
    is identical on every run.
    """
    for _ in range(_MAX_DECONFLICT_PASSES):
        offsets = frame.groupby(["entity_id", "timestamp"]).cumcount()
        if not offsets.any():
            return frame
        frame = frame.assign(
            timestamp=frame["timestamp"] + pd.to_timedelta(offsets, unit="s")
        )
    raise RuntimeError(
        "could not separate simultaneous events for an entity; "
        "session scheduling produced an implausible pile-up"
    )


def finalize_events(frame: pd.DataFrame) -> pd.DataFrame:
    """Put an event frame into its canonical, contract-compliant final form.

    Deconflicts simultaneous events, sorts chronologically, assigns stable
    ``event_id`` values and enforces :data:`EVENT_COLUMNS` order.

    Deliberately idempotent: it runs once over the benign stream and again
    after attack campaigns are merged in, at which point every ``event_id``
    must be reissued so that ids stay chronological across the combined
    dataset.
    """
    if frame.empty:
        return frame

    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])

    if "_session_id" in frame.columns:
        sessions = frame.groupby("_session_id").size()
        median_session_events = float(sessions.median())
        frame = frame.drop(columns=["_session_id"])
    else:
        median_session_events = frame.attrs.get("median_session_events")

    frame = _deconflict_timestamps(frame)
    frame = frame.sort_values(["timestamp", "entity_id"], kind="mergesort")
    frame = frame.reset_index(drop=True)

    # Reissued after the global sort so ids are chronological and stable.
    frame["event_id"] = [f"EVT-{i:09d}" for i in range(1, len(frame) + 1)]
    frame["campaign_id"] = frame["campaign_id"].astype("string")
    frame = frame[list(EVENT_COLUMNS)]
    frame.attrs["median_session_events"] = median_session_events
    return frame


class NormalBehaviorEngine:
    """Drives the enterprise simulation over the configured time horizon."""

    def __init__(
        self,
        profiles: list[EntityBehavioralProfile],
        *,
        start_date: datetime,
        simulation_days: int,
        target_events: int,
        noise_cfg: dict[str, Any],
    ) -> None:
        if not profiles:
            raise ValueError("cannot simulate an empty population")
        if simulation_days < 1:
            raise ValueError(f"simulation_days must be >= 1; got {simulation_days}")
        if target_events < 1:
            raise ValueError(f"target_events must be >= 1; got {target_events}")

        self.profiles = profiles
        self.start_date = start_date
        self.simulation_days = simulation_days
        self.target_events = target_events
        self.noise_cfg = noise_cfg

    def _allocate_entity_budgets(self) -> dict[str, int]:
        """Split the event budget by entity activity weight.

        Edge devices beacon far more often than a person logs in, so raw
        session rates would drown the dataset in heartbeats. These weights keep
        the human-to-machine mix representative of what an analyst reviews.
        """
        weights: list[float] = []
        for profile in self.profiles:
            if profile.entity_type == EntityType.USER:
                weights.append(1.0)
            elif profile.entity_type == EntityType.SERVICE_ACCOUNT:
                weights.append(0.55)
            else:
                weights.append(0.18)

        total = sum(weights)
        counts = [max(10, int(self.target_events * w / total)) for w in weights]
        shortfall = self.target_events - sum(counts)
        order = sorted(range(len(counts)), key=lambda i: weights[i], reverse=True)
        index = 0
        while shortfall > 0:
            counts[order[index % len(order)]] += 1
            shortfall -= 1
            index += 1
        return {p.entity_id: counts[i] for i, p in enumerate(self.profiles)}

    def _simulate_entity(
        self, profile: EntityBehavioralProfile, budget: int
    ) -> list[dict[str, Any]]:
        """Spread one entity's budget across the fixed horizon."""
        events: list[dict[str, Any]] = []
        days = active_days(profile, self.start_date, self.simulation_days)
        if not days:
            return events

        max_daily = max(1, math.ceil(budget / len(days) * _DAILY_BURST_CAP))
        horizon_end = self.start_date + timedelta(days=self.simulation_days)
        not_before: datetime | None = None

        for index, day_offset in enumerate(days):
            produced = len(events)
            if produced >= budget:
                break
            days_left = len(days) - index
            target_today = min(max_daily, max(1, math.ceil((budget - produced) / days_left)))
            plans = plan_day(
                profile,
                day_offset,
                self.start_date,
                self.noise_cfg,
                target_today,
                not_before=not_before,
                horizon_end=horizon_end,
            )
            for session_index, plan in enumerate(plans):
                events.extend(
                    simulate_session(
                        profile,
                        plan,
                        self.noise_cfg,
                        f"{profile.entity_id}-D{day_offset:04d}S{session_index:02d}",
                    )
                )
            if plans:
                last = plans[-1]
                not_before = last.start + timedelta(
                    seconds=last.duration_s + _MIN_SESSION_GAP_S
                )
        return events

    def generate(self) -> pd.DataFrame:
        budgets = self._allocate_entity_budgets()
        rows: list[dict[str, Any]] = []
        for profile in self.profiles:
            rows.extend(self._simulate_entity(profile, budgets[profile.entity_id]))

        return finalize_events(pd.DataFrame(rows))


def summarize_dataset(frame: pd.DataFrame) -> dict[str, Any]:
    """Quality summary used by the CLI, the tests and the written report."""
    if frame.empty:
        return {"total_rows": 0, "unique_entities": 0, "benign_only": True}

    def _count(entity_type: EntityType) -> int:
        return int(
            frame.loc[frame["entity_type"] == entity_type.value, "entity_id"].nunique()
        )

    first, last = frame["timestamp"].min(), frame["timestamp"].max()
    labels = frame["label"]
    attack_rows = int(labels.ne(AttackType.BENIGN.value).sum())
    return {
        "total_rows": int(len(frame)),
        "unique_entities": int(frame["entity_id"].nunique()),
        "users": _count(EntityType.USER),
        "service_accounts": _count(EntityType.SERVICE_ACCOUNT),
        "edge_devices": _count(EntityType.EDGE_DEVICE),
        "benign_only": attack_rows == 0,
        "date_min": str(first),
        "date_max": str(last),
        "span_days": int((last.normalize() - first.normalize()).days) + 1,
        "median_session_events": frame.attrs.get("median_session_events"),
        "duplicate_entity_timestamps": int(
            frame.duplicated(["entity_id", "timestamp"]).sum()
        ),
        "auth_failure_rate": round(float(1.0 - frame["auth_success"].mean()), 4),
        "distinct_resources": int(frame["resource_accessed"].nunique()),
        # Achieved, not configured: clamping campaign sizes to plausible shapes
        # means the target prevalence is approached rather than hit exactly.
        "attack_rows": attack_rows,
        "attack_prevalence": round(attack_rows / len(frame), 5),
        "malicious_rows": int(frame["is_attack"].sum()),
        "label_counts": {
            str(label): int(count) for label, count in labels.value_counts().items()
        },
        "campaigns_present": int(frame["campaign_id"].nunique(dropna=True)),
    }
