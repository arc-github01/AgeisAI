"""Phase 2 generator tests: population, scheduling, horizon, persistence.

The assertions here are the contract every later phase inherits. In particular
the timestamp invariants are not cosmetic: feature engineering divides by the
gap between an entity's consecutive events, and the temporal train/test split
assumes the dataset spans the horizon the config asked for.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pandas as pd
import pytest

from src.artifacts import artifact
from src.config import load_config
from src.generator import build_population, generate_normal_events, save
from src.generator.entities import COHORT_ARCHETYPES, population_to_records
from src.generator.normal_behavior import SessionPlan, plan_day, simulate_session
from src.schema import EVENT_COLUMNS, LABEL_COLUMNS, AttackType, EntityType, validate_events

#: Small enough to keep the suite fast, large enough to exercise every cohort.
TINY_PROFILE = {
    "name": "test",
    "n_users": 8,
    "n_service_accounts": 3,
    "n_edge_devices": 3,
    "days": 14,
    "target_events": 1200,
}


@pytest.fixture(scope="module")
def generated():
    return generate_normal_events(profile=TINY_PROFILE)


@pytest.fixture(scope="module")
def events(generated) -> pd.DataFrame:
    return generated.events


# -----------------------------------------------------------------------------
# Population
# -----------------------------------------------------------------------------
def test_population_matches_the_requested_sizing():
    population = build_population(n_users=8, n_service_accounts=3, n_edge_devices=3)
    by_type = {t: [p for p in population if p.entity_type == t] for t in EntityType}

    assert len(population) == 14
    assert len(by_type[EntityType.USER]) == 8
    assert len(by_type[EntityType.SERVICE_ACCOUNT]) == 3
    assert len(by_type[EntityType.EDGE_DEVICE]) == 3
    assert len({p.entity_id for p in population}) == 14


def test_every_entity_has_a_coherent_behavioural_definition():
    for profile in build_population(n_users=8, n_service_accounts=3, n_edge_devices=3):
        cohort = COHORT_ARCHETYPES[profile.role]
        assert cohort.entity_type == profile.entity_type
        assert profile.devices, "every entity needs at least one registered device"
        assert profile.primary_auth_method in cohort.auth_methods
        assert cohort.resources
        assert "START" in cohort.transitions


def test_population_is_reproducible_from_the_master_seed():
    first = population_to_records(build_population(n_users=8, n_service_accounts=3, n_edge_devices=3))
    second = population_to_records(build_population(n_users=8, n_service_accounts=3, n_edge_devices=3))
    assert first == second


def test_population_records_are_json_serialisable():
    records = population_to_records(build_population(n_users=4, n_service_accounts=1, n_edge_devices=1))
    restored = json.loads(json.dumps(records))
    assert len(restored) == 6
    assert all(record["entity_id"] for record in restored)


# -----------------------------------------------------------------------------
# Event contract
# -----------------------------------------------------------------------------
def test_events_satisfy_the_canonical_schema(events):
    validate_events(events, strict_order=True)
    assert list(events.columns) == list(EVENT_COLUMNS)
    assert not events.empty


def test_phase_two_emits_benign_events_only(events):
    assert (events["label"] == AttackType.BENIGN.value).all()
    assert not events["is_attack"].any()
    assert events["campaign_id"].isna().all()


def test_event_ids_are_unique_and_chronological(events):
    assert events["event_id"].is_unique
    assert list(events["event_id"]) == sorted(events["event_id"])


def test_generation_is_reproducible(events):
    again = generate_normal_events(profile=TINY_PROFILE).events
    pd.testing.assert_frame_equal(events, again)


def test_dataset_size_lands_near_the_configured_budget(events):
    prevalence = float(load_config()["generator.attack_prevalence"])
    expected = TINY_PROFILE["target_events"] * (1.0 - prevalence)
    assert 0.9 * expected <= len(events) <= 1.1 * expected


# -----------------------------------------------------------------------------
# Temporal invariants
# -----------------------------------------------------------------------------
def test_dataset_spans_exactly_the_configured_horizon(events):
    first, last = events["timestamp"].min(), events["timestamp"].max()
    span_days = (last.normalize() - first.normalize()).days + 1
    assert span_days <= TINY_PROFILE["days"]
    # A horizon that is barely used would mean the budget was crammed into a
    # handful of days, which would defeat the temporal split.
    assert span_days >= TINY_PROFILE["days"] - 1


def test_no_entity_has_two_events_in_the_same_second(events):
    assert not events.duplicated(["entity_id", "timestamp"]).any()


def test_events_use_sub_minute_resolution(events):
    # The previous scheduler clamped every event onto a whole minute, which
    # collapsed multi-step sessions onto one timestamp.
    assert events["timestamp"].dt.second.nunique() > 1


def test_activity_is_spread_across_the_horizon(events):
    per_day = events.groupby(events["timestamp"].dt.normalize()).size()
    assert len(per_day) >= TINY_PROFILE["days"] - 2
    assert per_day.max() <= 5 * per_day.median()


def test_weekday_structure_follows_the_real_calendar(events):
    users = events[events["entity_type"] == EntityType.USER.value]
    by_weekday = users.groupby(users["timestamp"].dt.weekday).size()
    weekday_mean = by_weekday.reindex(range(5)).fillna(0).mean()
    weekend_mean = by_weekday.reindex([5, 6]).fillna(0).mean()
    assert weekday_mean > weekend_mean


# -----------------------------------------------------------------------------
# Session mechanics
# -----------------------------------------------------------------------------
def test_sessions_advance_in_time_and_may_cross_an_hour_boundary():
    profile = build_population(n_users=1, n_service_accounts=0, n_edge_devices=0)[0]
    crossed_an_hour = False

    for index in range(50):
        plan = SessionPlan(
            start=datetime(2025, 1, 6, 9, 40, 0) + timedelta(days=index),
            n_steps=6,
            duration_s=3600.0,
        )
        stamps = [e["timestamp"] for e in simulate_session(profile, plan, {}, f"S{index}")]
        assert stamps == sorted(stamps)
        assert len(set(stamps)) == len(stamps)
        assert stamps[-1] - stamps[0] < timedelta(seconds=plan.duration_s)
        crossed_an_hour |= stamps[-1].hour != stamps[0].hour

    assert crossed_an_hour, "sessions must be able to run past the end of an hour"


def test_command_sequence_accumulates_within_a_session():
    profile = build_population(n_users=1, n_service_accounts=0, n_edge_devices=0)[0]
    plan = SessionPlan(start=datetime(2025, 1, 6, 9, 0, 0), n_steps=6, duration_s=1800.0)
    events = simulate_session(profile, plan, {}, "S0")

    depths = [event["command_sequence"].count("->") for event in events]
    assert depths == list(range(len(events)))
    assert events[0]["command_sequence"] == events[-1]["command_sequence"].split("->")[0]


def test_planned_sessions_never_overlap():
    profile = build_population(n_users=1, n_service_accounts=0, n_edge_devices=0)[0]
    plans = plan_day(profile, 0, datetime(2025, 1, 6), {"hour_jitter_std": 1.2}, 40)

    assert len(plans) > 1
    for earlier, later in zip(plans, plans[1:]):
        assert later.start >= earlier.start + timedelta(seconds=earlier.duration_s)


def test_edge_devices_beacon_around_the_clock(events):
    edge = events[events["entity_type"] == EntityType.EDGE_DEVICE.value]
    assert not edge.empty
    assert edge["timestamp"].dt.hour.nunique() > 12


# -----------------------------------------------------------------------------
# Persistence
# -----------------------------------------------------------------------------
def test_run_is_written_to_the_registered_artifact_paths(generated):
    paths = save(generated)

    assert paths["entities"] == artifact("entities").path
    assert paths["events"] == artifact("events").path
    assert artifact("entities").exists()
    assert artifact("events").exists()


def test_persisted_events_round_trip_through_parquet(generated):
    save(generated)
    restored = pd.read_parquet(artifact("events").path)

    assert list(restored.columns) == list(EVENT_COLUMNS)
    assert len(restored) == len(generated.events)
    validate_events(restored, strict_order=True)


def test_persisted_entities_cover_every_acting_entity(generated):
    save(generated)
    document = json.loads(artifact("entities").path.read_text(encoding="utf-8"))

    roster = {record["entity_id"] for record in document["entities"]}
    assert document["profile"] == "test"
    assert document["dataset_summary"]["total_rows"] == len(generated.events)
    assert set(generated.events["entity_id"]).issubset(roster)


def test_persisted_entities_carry_no_ground_truth_labels(generated):
    save(generated)
    document = json.loads(artifact("entities").path.read_text(encoding="utf-8"))
    for record in document["entities"]:
        assert not set(record).intersection(LABEL_COLUMNS)
