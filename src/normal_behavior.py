"""Session-based normal behaviour simulation engine.

Unlike row-at-a-time generators, this module simulates *work sessions*:

    Day planner  ->  Session scheduler  ->  Markov resource walk  ->  Event rows

Each session produces a short sequence of causally linked events that share
device context, geography, and a growing ``command_sequence`` string. That
sequence is exactly what the downstream Markov anomaly scorer will consume.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from src.schema import AttackType, EntityType
from src.utils.seeding import get_rng

from .entities import (
    CohortArchetype,
    EntityBehavioralProfile,
    GeoAnchor,
    RegisteredDevice,
    ResourceDef,
    pick_session_location,
    random_ip_in_subnet,
)


@dataclass
class _SessionPlan:
    day_offset: int
    start_hour: float
    n_steps: int


def _resource_lookup(cohort: CohortArchetype) -> dict[str, ResourceDef]:
    return {r.name: r for r in cohort.resources}


def _markov_step(current: str, cohort: CohortArchetype, rng: np.random.Generator) -> str:
    """Walk the cohort transition graph; return 'END' to terminate session."""
    options = cohort.transitions.get(current, {"END": 1.0})
    labels = list(options.keys())
    probs = np.array([options[k] for k in labels], dtype=float)
    probs /= probs.sum()
    return str(rng.choice(labels, p=probs))


def _pick_device(profile: EntityBehavioralProfile) -> RegisteredDevice:
    rng = profile.rng
    if len(profile.devices) == 1:
        return profile.devices[0]
    weights = [0.85 if d.is_primary else 0.15 for d in profile.devices]
    idx = int(rng.choice(len(profile.devices), p=np.array(weights) / sum(weights)))
    return profile.devices[idx]


def _plan_sessions_for_day(
    profile: EntityBehavioralProfile,
    day_offset: int,
    weekday: int,
    noise_cfg: dict[str, Any],
) -> list[_SessionPlan]:
    cohort = profile.cohort
    rng = profile.rng

    if cohort.entity_type != EntityType.EDGE_DEVICE:
        if weekday not in profile.working_days and rng.random() > cohort.weekend_activity_prob:
            return []
        n_sessions = max(0, int(rng.poisson(cohort.sessions_per_day_mean)))
    else:
        # OT/IoT: hourly heartbeats
        n_sessions = max(1, int(rng.poisson(cohort.sessions_per_day_mean)))

    plans: list[_SessionPlan] = []
    jitter_std = float(noise_cfg.get("hour_jitter_std", 1.2))
    for _ in range(n_sessions):
        if cohort.entity_type == EntityType.EDGE_DEVICE:
            start_hour = float(rng.integers(0, 24)) + rng.random()
        else:
            start_hour = float(
                np.clip(rng.normal(profile.preferred_login_hour, jitter_std), 0.0, 23.5)
            )
        n_steps = max(2, int(rng.poisson(cohort.session_steps_mean)))
        plans.append(_SessionPlan(day_offset=day_offset, start_hour=start_hour, n_steps=n_steps))
    plans.sort(key=lambda p: p.start_hour)
    return plans


def _timestamp(base: datetime, day_offset: int, hour_float: float, minute_offset: int = 0) -> datetime:
    day = base + timedelta(days=day_offset)
    hour = int(hour_float)
    minute = int((hour_float - hour) * 60) + minute_offset
    minute = min(max(minute, 0), 59)
    return day.replace(hour=hour, minute=minute, second=0, microsecond=0)


def simulate_session(
    profile: EntityBehavioralProfile,
    plan: _SessionPlan,
    start_date: datetime,
    noise_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Simulate one coherent session as multiple canonical event rows."""
    cohort = profile.cohort
    rng = profile.rng
    resources = _resource_lookup(cohort)
    device = _pick_device(profile)
    geo = pick_session_location(profile, noise_cfg)
    source_ip = random_ip_in_subnet(profile.ip_network, rng)

    session_id = str(uuid.uuid4())[:8]
    session_duration_s = float(
        max(30.0, rng.normal(cohort.session_duration_mean_s, cohort.session_duration_mean_s * 0.25))
    )
    auth_success = rng.random() > float(
        noise_cfg.get("benign_auth_failure_rate", cohort.auth_failure_rate)
    )

    # Build Markov walk
    steps: list[ResourceDef] = []
    state = "START"
    for _ in range(plan.n_steps):
        nxt = _markov_step(state, cohort, rng)
        if nxt == "END":
            break
        if nxt in resources:
            steps.append(resources[nxt])
            state = nxt

    if not steps:
        steps = [resources[cohort.resources[0].name]]

    events: list[dict[str, Any]] = []
    command_parts: list[str] = []
    step_spacing_min = max(1, int(session_duration_s / max(len(steps), 1) / 60))

    for step_idx, resource in enumerate(steps):
        command_parts.append(resource.name.replace(" ", "_").upper())
        command_sequence = "->".join(command_parts)

        # Occasional off-profile resource (legitimate drift/noise)
        if (
            step_idx > 0
            and rng.random() < float(noise_cfg.get("off_profile_resource_prob", 0.04))
        ):
            alt = rng.choice(cohort.resources)
            resource = alt
            command_parts[-1] = alt.name.replace(" ", "_").upper()
            command_sequence = "->".join(command_parts)

        ts = _timestamp(start_date, plan.day_offset, plan.start_hour, step_idx * step_spacing_min)
        byte_noise = float(rng.normal(resource.typical_bytes, resource.typical_bytes * 0.3))
        bytes_transferred = float(max(64.0, byte_noise))

        events.append(
            {
                "event_id": str(uuid.uuid4()),
                "timestamp": ts,
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
                "command_sequence": command_sequence,
                "session_duration_s": round(session_duration_s, 1),
                "bytes_transferred": round(bytes_transferred, 0),
                "device_id": device.device_id,
                "device_os": device.device_os,
                "device_firmware": device.device_firmware,
                "device_protocol": device.device_protocol,
                "device_mac": device.device_mac,
                "label": AttackType.BENIGN.value,
                "is_attack": False,
                "campaign_id": None,
                "_session_id": session_id,  # dropped before schema validation
            }
        )

    return events


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
        self.profiles = profiles
        self.start_date = start_date
        self.simulation_days = simulation_days
        self.target_events = target_events
        self.noise_cfg = noise_cfg
        self.rng = get_rng("generator.normal")

    def _allocate_entity_budgets(self) -> dict[str, int]:
        """Split event budget by entity activity weight."""
        weights: list[float] = []
        for p in self.profiles:
            if p.entity_type == EntityType.USER:
                weights.append(1.0)
            elif p.entity_type == EntityType.SERVICE_ACCOUNT:
                weights.append(0.55)
            else:
                weights.append(0.18)
        total = sum(weights)
        counts = [max(10, int(self.target_events * w / total)) for w in weights]
        # Fix rounding
        diff = self.target_events - sum(counts)
        order = sorted(range(len(counts)), key=lambda i: weights[i], reverse=True)
        i = 0
        while diff > 0:
            counts[order[i % len(order)]] += 1
            diff -= 1
            i += 1
        return {self.profiles[i].entity_id: counts[i] for i in range(len(self.profiles))}

    def generate(self) -> pd.DataFrame:
        budgets = self._allocate_entity_budgets()
        all_events: list[dict[str, Any]] = []

        for profile in self.profiles:
            target = budgets[profile.entity_id]
            entity_events: list[dict[str, Any]] = []

            # Walk the simulation horizon; if the calendar is exhausted before the
            # per-entity budget is met, cycle additional days (models ongoing ops).
            day_cursor = 0
            max_day_cursor = self.simulation_days * 12
            while len(entity_events) < target and day_cursor < max_day_cursor:
                sim_day = day_cursor % self.simulation_days
                weekday = (self.start_date + timedelta(days=sim_day)).weekday()
                plans = _plan_sessions_for_day(profile, day_cursor, weekday, self.noise_cfg)
                for plan in plans:
                    entity_events.extend(
                        simulate_session(profile, plan, self.start_date, self.noise_cfg)
                    )
                    if len(entity_events) >= target:
                        break
                day_cursor += 1

            if len(entity_events) > target:
                self.rng.shuffle(entity_events)
                entity_events = entity_events[:target]

            all_events.extend(entity_events)

        frame = pd.DataFrame(all_events)
        if frame.empty:
            return frame

        frame.drop(columns=["_session_id"], inplace=True, errors="ignore")
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        frame.sort_values("timestamp", inplace=True, kind="mergesort")
        frame.reset_index(drop=True, inplace=True)

        # Global trim to exact target
        if len(frame) > self.target_events:
            frame = frame.iloc[: self.target_events].copy()

        return frame


def summarize_normal_dataset(frame: pd.DataFrame) -> dict[str, Any]:
    """Quick quality summary for CLI / tests."""
    return {
        "total_rows": len(frame),
        "unique_entities": frame["entity_id"].nunique() if len(frame) else 0,
        "users": frame.loc[frame["entity_type"] == EntityType.USER.value, "entity_id"].nunique()
        if len(frame)
        else 0,
        "service_accounts": frame.loc[
            frame["entity_type"] == EntityType.SERVICE_ACCOUNT.value, "entity_id"
        ].nunique()
        if len(frame)
        else 0,
        "edge_devices": frame.loc[
            frame["entity_type"] == EntityType.EDGE_DEVICE.value, "entity_id"
        ].nunique()
        if len(frame)
        else 0,
        "benign_only": bool((frame["label"] == AttackType.BENIGN.value).all()) if len(frame) else True,
        "date_min": str(frame["timestamp"].min()) if len(frame) else None,
        "date_max": str(frame["timestamp"].max()) if len(frame) else None,
        "median_session_steps": float(
            frame.groupby(["entity_id", "session_duration_s"]).size().median()
        )
        if len(frame)
        else 0.0,
    }
