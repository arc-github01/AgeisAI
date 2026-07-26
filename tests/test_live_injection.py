"""Phase 11 live attack synthesis tests."""

from __future__ import annotations

import pandas as pd
import pytest

from src.generator import (
    build_population,
    generate_normal_events,
    record_to_profile,
    synthesize_live_attack,
)
from src.generator.entities import population_to_records
from src.generator.live_injection import build_live_campaign, campaign_to_injection_frame
from src.schema import ATTACK_CLASSES, IDENTITY_COLUMNS, OBSERVATION_COLUMNS, validate_events

TINY_PROFILE = {
    "name": "live-test",
    "n_users": 6,
    "n_service_accounts": 2,
    "n_edge_devices": 2,
    "days": 10,
    "target_events": 800,
}


@pytest.fixture(scope="module")
def live_corpus():
    result = generate_normal_events(profile=TINY_PROFILE)
    return result.population, result.events


def test_record_to_profile_round_trip():
    original = build_population(n_users=3, n_service_accounts=1, n_edge_devices=1)[0]
    restored = record_to_profile(population_to_records([original])[0])
    assert restored.entity_id == original.entity_id
    assert restored.role == original.role
    assert restored.home_geo.city == original.home_geo.city
    assert restored.devices[0].device_id == original.devices[0].device_id
    assert [r.name for r in restored.cohort.resources] == [
        r.name for r in original.cohort.resources
    ]


@pytest.mark.parametrize("attack_type", list(ATTACK_CLASSES))
def test_synthesize_live_attack_produces_valid_observations(live_corpus, attack_type):
    population, events = live_corpus
    entity_id = next(p.entity_id for p in population if p.entity_type.value == "user")
    frame = synthesize_live_attack(
        entity_id,
        attack_type,
        intensity=3,
        population=population,
        events=events,
    )
    validate_events(frame, require_labels=False)
    for col in IDENTITY_COLUMNS + OBSERVATION_COLUMNS:
        assert col in frame.columns
    assert frame["event_id"].str.startswith("INJ-").all()
    entity_end = events.loc[
        events["entity_id"].astype(str) == str(entity_id), "timestamp"
    ].max()
    # Campaigns are placed after the target entity's history (impossible travel
    # anchors on the last observed login; other types start after the corpus).
    assert pd.Timestamp(frame["timestamp"].min()) > pd.Timestamp(entity_end)


def test_live_impossible_travel_is_after_anchor(live_corpus):
    population, events = live_corpus
    entity_id = next(p.entity_id for p in population if p.entity_type.value == "user")
    campaign = build_live_campaign(
        entity_id,
        "IMPOSSIBLE_TRAVEL",
        intensity=4,
        population=population,
        events=events,
        stealthy=False,
    )
    frame = campaign_to_injection_frame(campaign)
    entity_history = events[events["entity_id"] == entity_id]
    last_lat = float(entity_history.iloc[-1]["latitude"])
    assert any(abs(float(lat) - last_lat) > 1.0 for lat in frame["latitude"])


def test_live_intensity_scales_brute_force(live_corpus):
    population, events = live_corpus
    entity_id = next(p.entity_id for p in population if p.entity_type.value == "user")
    low = synthesize_live_attack(
        entity_id, "BRUTE_FORCE", 1, population=population, events=events
    )
    high = synthesize_live_attack(
        entity_id, "BRUTE_FORCE", 5, population=population, events=events
    )
    assert len(high) >= len(low)


#: A live injection is scored one event at a time in the browser request, so an
#: unbounded campaign turns the demo into a multi-second stall.
_MAX_LIVE_EVENTS = 40


@pytest.mark.parametrize("attack_type", list(ATTACK_CLASSES))
def test_live_campaigns_stay_demo_sized(live_corpus, attack_type):
    population, events = live_corpus
    entity_id = next(p.entity_id for p in population if p.entity_type.value == "user")
    frame = synthesize_live_attack(
        entity_id,
        attack_type,
        intensity=5,
        population=population,
        events=events,
    )
    assert 0 < len(frame) <= _MAX_LIVE_EVENTS


def test_live_slow_burn_campaigns_stay_gradual(live_corpus):
    """Insider drift / exfiltration must ramp over days, not arrive in a burst."""
    population, events = live_corpus
    entity_id = next(p.entity_id for p in population if p.entity_type.value == "user")
    for attack_type in ("INSIDER_DRIFT", "LOW_AND_SLOW_EXFILTRATION"):
        frame = synthesize_live_attack(
            entity_id, attack_type, 5, population=population, events=events
        )
        span_days = max(
            1.0,
            (frame["timestamp"].max() - frame["timestamp"].min()).total_seconds() / 86400.0,
        )
        assert len(frame) / span_days <= 4.0, f"{attack_type} is too dense to be gradual"


def test_live_credential_stuffing_fans_out_across_victims(live_corpus):
    population, events = live_corpus
    entity_id = next(p.entity_id for p in population if p.entity_type.value == "user")
    frame = synthesize_live_attack(
        entity_id, "CREDENTIAL_STUFFING", 4, population=population, events=events
    )
    victims = frame["entity_id"].nunique()
    assert victims >= 2, "stuffing must span multiple identities"
    assert len(frame) / victims <= 4, "fan-out, not depth, is the defining shape"
