"""Phase 3 attack-injection tests.

Each per-injector test asserts the *defining signal* of its attack rather than a
row count, because the row count is a budget artefact while the signal is the
thing every downstream detector, feature and metric depends on. If lateral
movement stopped touching foreign resources the dataset would still look fine by
volume and be worthless.

The stealth assertions matter just as much: a dataset where every attack is
blatant produces a near-perfect PR-AUC that says nothing about the model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.generator import build_population, generate_dataset
from src.generator.attacks import (
    HOSTILE_ORIGINS,
    AttackOrchestrator,
    inject_brute_force,
    inject_credential_stuffing,
    inject_device_spoofing,
    inject_impossible_travel,
    inject_insider_drift,
    inject_lateral_movement,
    inject_low_and_slow_exfiltration,
)
from src.generator.entities import COHORT_ARCHETYPES
from src.schema import (
    ATTACK_CLASSES,
    EVENT_COLUMNS,
    MALICIOUS_CLASSES,
    AttackType,
    EntityType,
    validate_events,
)
from src.utils.geo import haversine_km, haversine_km_array, implied_velocity_kmh
from src.utils.seeding import get_rng

#: Big enough that the budget affords several campaigns of every type, small
#: enough to keep the suite quick.
INJECTED_PROFILE = {
    "name": "test-attacks",
    "n_users": 30,
    "n_service_accounts": 6,
    "n_edge_devices": 8,
    "days": 21,
    "target_events": 12000,
}

START = pd.Timestamp("2025-01-08 02:00:00").to_pydatetime()


@pytest.fixture(scope="module")
def attacks_cfg() -> dict:
    return load_config()["generator.attacks"]


@pytest.fixture(scope="module")
def population():
    return build_population(n_users=20, n_service_accounts=4, n_edge_devices=4)


@pytest.fixture(scope="module")
def generated():
    return generate_dataset(profile=INJECTED_PROFILE)


@pytest.fixture(scope="module")
def events(generated) -> pd.DataFrame:
    return generated.events


def _user(population, role: str | None = None):
    for profile in population:
        if profile.entity_type is EntityType.USER and (
            role is None or profile.role == role
        ):
            return profile
    raise AssertionError(f"no user with role {role!r} in the test population")


def _users(population) -> list:
    return [p for p in population if p.entity_type is EntityType.USER]


def _both_variants(build):
    """The same injector run obviously and stealthily, for comparison."""
    return build(stealthy=False), build(stealthy=True)


# -----------------------------------------------------------------------------
# Brute force
# -----------------------------------------------------------------------------
def test_brute_force_concentrates_failures_on_one_account_from_one_source(
    population, attacks_cfg
):
    profile = _user(population)
    campaign = inject_brute_force(
        profile,
        campaign_id="CMP-BRUTE_FORCE-001",
        rng=get_rng("test.brute"),
        attacks_cfg=attacks_cfg,
        window_start=START,
        stealthy=False,
        target_events=40,
    )
    frame = campaign.to_frame()

    assert frame["entity_id"].nunique() == 1
    assert frame["source_ip"].nunique() == 1
    # A brute force is defined by its failures; at most the final attempt lands.
    assert (~frame["auth_success"]).sum() >= len(frame) - 1
    assert (frame["session_duration_s"] < 5.0).all()
    assert frame["label"].eq(AttackType.BRUTE_FORCE.value).all()
    assert frame["timestamp"].is_monotonic_increasing


def test_brute_force_stealth_variant_is_slower_and_quieter(population, attacks_cfg):
    profile = _user(population)
    obvious, stealthy = _both_variants(
        lambda *, stealthy: inject_brute_force(
            profile,
            campaign_id="CMP-BRUTE_FORCE-002",
            rng=get_rng(f"test.brute.{stealthy}"),
            attacks_cfg=attacks_cfg,
            window_start=START,
            stealthy=stealthy,
            target_events=40,
        )
    )

    assert len(stealthy) < len(obvious)

    def rate_per_minute(campaign) -> float:
        stamps = [event["timestamp"] for event in campaign.events]
        span = max((max(stamps) - min(stamps)).total_seconds() / 60.0, 1e-9)
        return len(campaign) / span

    assert rate_per_minute(stealthy) < rate_per_minute(obvious)
    # Obvious campaigns advertise themselves geographically; stealth ones do not.
    hostile = {site["city"] for site in HOSTILE_ORIGINS}
    assert set(obvious.to_frame()["city"]) <= hostile
    assert not set(stealthy.to_frame()["city"]) & hostile


# -----------------------------------------------------------------------------
# Impossible travel
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("stealthy", [False, True])
def test_impossible_travel_exceeds_the_plausible_velocity_threshold(
    population, attacks_cfg, stealthy
):
    profile = _user(population)
    threshold = float(load_config()["features.max_plausible_kmh"])
    anchor = {
        "timestamp": START,
        "latitude": profile.home_geo.latitude,
        "longitude": profile.home_geo.longitude,
        "next_gap_minutes": 240.0,
    }
    campaign = inject_impossible_travel(
        profile,
        campaign_id="CMP-IMPOSSIBLE_TRAVEL-001",
        rng=get_rng(f"test.travel.{stealthy}"),
        attacks_cfg=attacks_cfg,
        anchor=anchor,
        stealthy=stealthy,
        target_events=2,
        max_plausible_kmh=threshold,
    )

    login = campaign.events[0]
    distance = haversine_km(
        anchor["latitude"], anchor["longitude"], login["latitude"], login["longitude"]
    )
    elapsed = (login["timestamp"] - anchor["timestamp"]).total_seconds()
    assert implied_velocity_kmh(distance, elapsed) > threshold


def test_impossible_travel_stealth_variant_keeps_a_known_device_and_city(
    population, attacks_cfg
):
    profile = _user(population)
    anchor = {
        "timestamp": START,
        "latitude": profile.home_geo.latitude,
        "longitude": profile.home_geo.longitude,
        "next_gap_minutes": 240.0,
    }
    obvious, stealthy = _both_variants(
        lambda *, stealthy: inject_impossible_travel(
            profile,
            campaign_id="CMP-IMPOSSIBLE_TRAVEL-002",
            rng=get_rng(f"test.travel2.{stealthy}"),
            attacks_cfg=attacks_cfg,
            anchor=anchor,
            stealthy=stealthy,
            target_events=2,
            max_plausible_kmh=900.0,
        )
    )

    known_devices = {device.device_id for device in profile.devices}
    hostile = {site["city"] for site in HOSTILE_ORIGINS}

    assert set(stealthy.to_frame()["device_id"]) <= known_devices
    assert not set(stealthy.to_frame()["city"]) & hostile
    assert not set(obvious.to_frame()["device_id"]) & known_devices
    assert set(obvious.to_frame()["city"]) <= hostile


# -----------------------------------------------------------------------------
# Credential stuffing
# -----------------------------------------------------------------------------
def test_credential_stuffing_fans_few_sources_across_many_victims(
    population, attacks_cfg
):
    victims = _users(population)
    campaign = inject_credential_stuffing(
        victims,
        campaign_id="CMP-CREDENTIAL_STUFFING-001",
        rng=get_rng("test.stuffing"),
        attacks_cfg=attacks_cfg,
        window_start=START,
        stealthy=False,
        target_events=len(victims),
    )
    frame = campaign.to_frame()

    assert frame["entity_id"].nunique() == len(victims)
    assert frame["source_ip"].nunique() < frame["entity_id"].nunique()
    # One shared campaign id is what lets evaluation score the spray as a unit.
    assert frame["campaign_id"].nunique() == 1
    # The same tool fingerprint recurring across unrelated identities is the tell.
    assert frame["device_mac"].nunique() == 1
    assert frame["auth_success"].mean() < 0.30
    # Each victim keeps its own auth method: no categorical giveaway to memorise.
    expected = {p.entity_id: p.primary_auth_method for p in victims}
    assert all(row.auth_method == expected[row.entity_id] for row in frame.itertuples())


def test_credential_stuffing_stealth_variant_spreads_wider(population, attacks_cfg):
    victims = _users(population)
    obvious, stealthy = _both_variants(
        lambda *, stealthy: inject_credential_stuffing(
            victims,
            campaign_id="CMP-CREDENTIAL_STUFFING-002",
            rng=get_rng(f"test.stuffing2.{stealthy}"),
            attacks_cfg=attacks_cfg,
            window_start=START,
            stealthy=stealthy,
            target_events=len(victims) * 2,
        )
    )

    def sources(campaign) -> int:
        return campaign.to_frame()["source_ip"].nunique()

    def span_minutes(campaign) -> float:
        stamps = campaign.to_frame()["timestamp"]
        return (stamps.max() - stamps.min()).total_seconds() / 60.0

    assert sources(stealthy) > sources(obvious)
    assert span_minutes(stealthy) > span_minutes(obvious)


# -----------------------------------------------------------------------------
# Lateral movement
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("stealthy", [False, True])
def test_lateral_movement_only_touches_resources_outside_its_own_cohort(
    population, attacks_cfg, stealthy
):
    profile = _user(population, "developer")
    campaign = inject_lateral_movement(
        profile,
        campaign_id="CMP-LATERAL_MOVEMENT-001",
        rng=get_rng(f"test.lateral.{stealthy}"),
        attacks_cfg=attacks_cfg,
        window_start=START,
        stealthy=stealthy,
        target_events=6,
    )
    frame = campaign.to_frame()

    own = {resource.name for resource in profile.cohort.resources}
    assert not set(frame["resource_accessed"]) & own
    # The command trail has to grow, since the sequence model reads it directly.
    lengths = [len(value.split("->")) for value in frame["command_sequence"]]
    assert lengths == sorted(lengths)
    assert lengths[-1] == len(frame)


def test_lateral_movement_escalates_and_stealth_variant_makes_fewer_hops(
    population, attacks_cfg
):
    profile = _user(population, "developer")
    obvious, stealthy = _both_variants(
        lambda *, stealthy: inject_lateral_movement(
            profile,
            campaign_id="CMP-LATERAL_MOVEMENT-002",
            rng=get_rng(f"test.lateral2.{stealthy}"),
            attacks_cfg=attacks_cfg,
            window_start=START,
            stealthy=stealthy,
            target_events=8,
        )
    )

    assert len(stealthy) < len(obvious)

    sensitivity = {
        resource.name: resource.sensitivity
        for cohort in COHORT_ARCHETYPES.values()
        for resource in cohort.resources
    }
    final = obvious.events[-1]["resource_accessed"]
    assert sensitivity[final] == "critical"


# -----------------------------------------------------------------------------
# Device spoofing
# -----------------------------------------------------------------------------
def test_device_spoofing_presents_hardware_the_entity_never_registered(
    population, attacks_cfg
):
    profile = _user(population)
    campaign = inject_device_spoofing(
        profile,
        campaign_id="CMP-DEVICE_SPOOFING-001",
        rng=get_rng("test.spoof"),
        attacks_cfg=attacks_cfg,
        window_start=START,
        stealthy=False,
        target_events=6,
    )
    frame = campaign.to_frame()

    assert not set(frame["device_id"]) & {d.device_id for d in profile.devices}
    assert not set(frame["device_mac"]) & {d.device_mac for d in profile.devices}
    # Everything else is genuinely the entity: only the hardware is wrong.
    assert set(frame["city"]) == {profile.home_geo.city}
    assert frame["auth_success"].all()


def test_device_spoofing_stealth_variant_keeps_the_registered_identifier(
    population, attacks_cfg
):
    profile = _user(population)
    campaign = inject_device_spoofing(
        profile,
        campaign_id="CMP-DEVICE_SPOOFING-002",
        rng=get_rng("test.spoof2"),
        attacks_cfg=attacks_cfg,
        window_start=START,
        stealthy=True,
        target_events=6,
    )
    frame = campaign.to_frame()

    assert set(frame["device_id"]) <= {d.device_id for d in profile.devices}
    assert set(frame["device_os"]) <= {d.device_os for d in profile.devices}
    # Only the fingerprint drifts, which is what a cloned endpoint looks like.
    assert not set(frame["device_mac"]) & {d.device_mac for d in profile.devices}


# -----------------------------------------------------------------------------
# Low-and-slow exfiltration
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("stealthy", [False, True])
def test_low_and_slow_grows_volume_across_multiple_days(
    population, attacks_cfg, stealthy
):
    profile = _user(population)
    campaign = inject_low_and_slow_exfiltration(
        profile,
        campaign_id="CMP-LOW_AND_SLOW_EXFILTRATION-001",
        rng=get_rng(f"test.exfil.{stealthy}"),
        attacks_cfg=attacks_cfg,
        window_start=START,
        stealthy=stealthy,
        target_events=20,
    )
    frame = campaign.to_frame()

    assert frame["timestamp"].dt.normalize().nunique() >= 7
    # Off-hours is half the signal; the volume trend is the other half.
    assert (frame["timestamp"].dt.hour < 6).all()

    daily = frame.groupby(frame["timestamp"].dt.normalize())["bytes_transferred"].mean()
    assert daily.iloc[-1] > daily.iloc[0]
    # The resource is one the entity legitimately uses, so nothing looks foreign.
    assert set(frame["resource_accessed"]) <= {
        resource.name for resource in profile.cohort.resources
    }


def test_low_and_slow_stealth_variant_grows_more_gently(population, attacks_cfg):
    profile = _user(population)
    obvious, stealthy = _both_variants(
        lambda *, stealthy: inject_low_and_slow_exfiltration(
            profile,
            campaign_id="CMP-LOW_AND_SLOW_EXFILTRATION-002",
            rng=get_rng(f"test.exfil2.{stealthy}"),
            attacks_cfg=attacks_cfg,
            window_start=START,
            stealthy=stealthy,
            target_events=30,
        )
    )

    def growth(campaign) -> float:
        frame = campaign.to_frame()
        daily = frame.groupby(frame["timestamp"].dt.normalize())[
            "bytes_transferred"
        ].mean()
        return float(daily.iloc[-1] / daily.iloc[0])

    assert growth(stealthy) < growth(obvious)


# -----------------------------------------------------------------------------
# Insider drift — the labelled-but-benign edge case
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("stealthy", [False, True])
def test_insider_drift_is_labelled_without_being_counted_as_an_attack(
    population, attacks_cfg, stealthy
):
    profile = _user(population, "hr")
    campaign = inject_insider_drift(
        profile,
        campaign_id="CMP-INSIDER_DRIFT-001",
        rng=get_rng(f"test.drift.{stealthy}"),
        attacks_cfg=attacks_cfg,
        window_start=pd.Timestamp("2025-01-08 10:00:00").to_pydatetime(),
        stealthy=stealthy,
        target_events=30,
    )
    frame = campaign.to_frame()

    assert frame["label"].eq(AttackType.INSIDER_DRIFT.value).all()
    # Ambiguous by design: excluded from MALICIOUS_CLASSES so a detector that
    # flags it is measured as a false positive, not credited with a catch.
    assert not frame["is_attack"].any()
    assert frame["auth_success"].all()
    assert frame["campaign_id"].nunique() == 1

    # Nothing else about it looks hostile: own devices, normal hours.
    assert set(frame["device_id"]) <= {d.device_id for d in profile.devices}
    assert frame["timestamp"].dt.hour.between(9, 21).all()

    # The footprint widens gradually rather than appearing all at once.
    daily = frame.groupby(frame["timestamp"].dt.normalize()).size()
    assert len(daily) >= 5
    assert daily.tail(3).sum() > daily.head(3).sum()


def test_insider_drift_adopts_resources_from_another_cohort(population, attacks_cfg):
    profile = _user(population, "hr")
    campaign = inject_insider_drift(
        profile,
        campaign_id="CMP-INSIDER_DRIFT-002",
        rng=get_rng("test.drift2"),
        attacks_cfg=attacks_cfg,
        window_start=pd.Timestamp("2025-01-08 10:00:00").to_pydatetime(),
        stealthy=False,
        target_events=30,
    )
    frame = campaign.to_frame()

    own = {resource.name for resource in profile.cohort.resources}
    adopted = set(frame["resource_accessed"])
    low, high = attacks_cfg["insider_drift"]["new_resources"]

    assert not adopted & own
    assert low <= len(adopted) <= high


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------
def test_orchestrator_rejects_an_impossible_prevalence(population, attacks_cfg):
    with pytest.raises(ValueError):
        AttackOrchestrator(
            population,
            pd.DataFrame(),
            start_date=START,
            simulation_days=21,
            attacks_cfg=attacks_cfg,
            attack_mix={"brute_force": 1.0},
            prevalence=1.0,
            max_plausible_kmh=900.0,
        )


def test_every_attack_class_produces_at_least_one_campaign(generated):
    produced = {campaign.attack_type.value for campaign in generated.campaigns}
    assert produced == set(ATTACK_CLASSES)


def test_the_dataset_mixes_blatant_and_stealthy_campaigns(generated, attacks_cfg):
    stealth = [c for c in generated.campaigns if c.stealthy]
    obvious = [c for c in generated.campaigns if not c.stealthy]
    assert stealth and obvious

    expected = float(attacks_cfg["stealth_fraction"])
    achieved = len(stealth) / len(generated.campaigns)
    assert abs(achieved - expected) <= 0.15

    # Per type, only where the budget affords enough campaigns to split — a type
    # that only gets two campaigns at this scale cannot show both difficulties,
    # which is why each injector's stealth variant is also tested directly.
    by_type: dict[str, list[bool]] = {}
    for campaign in generated.campaigns:
        by_type.setdefault(campaign.attack_type.value, []).append(campaign.stealthy)

    missing = {
        name: flags
        for name, flags in by_type.items()
        if len(flags) >= 4 and len(set(flags)) < 2
    }
    assert not missing, f"only one difficulty generated for: {sorted(missing)}"


def test_campaign_ids_are_unique_and_carry_a_single_label(generated):
    ids = [campaign.campaign_id for campaign in generated.campaigns]
    assert len(ids) == len(set(ids))

    labelled = generated.events.dropna(subset=["campaign_id"])
    assert (labelled.groupby("campaign_id")["label"].nunique() == 1).all()


# -----------------------------------------------------------------------------
# Dataset-level invariants
# -----------------------------------------------------------------------------
def test_injected_dataset_still_satisfies_the_event_contract(events):
    validate_events(events, strict_order=True)
    assert list(events.columns) == list(EVENT_COLUMNS)
    assert events["event_id"].is_unique
    assert events["timestamp"].is_monotonic_increasing
    assert not events.duplicated(["entity_id", "timestamp"]).any()


def test_all_seven_attack_classes_appear_in_the_dataset(events):
    assert set(ATTACK_CLASSES) <= set(events["label"].unique())


def test_achieved_prevalence_is_close_to_the_configured_target(events):
    target = float(load_config()["generator.attack_prevalence"])
    achieved = float(events["label"].ne(AttackType.BENIGN.value).mean())
    # Clamping campaigns to plausible shapes means the target is approached, not
    # hit exactly; a wide miss would mean the budget allocator has broken.
    assert abs(achieved - target) <= 0.5 * target


def test_is_attack_agrees_with_the_schema_definition_of_malicious(events):
    expected = events["label"].isin(MALICIOUS_CLASSES)
    assert events["is_attack"].equals(expected)
    # Insider drift is the case that makes this non-trivial.
    drift = events[events["label"] == AttackType.INSIDER_DRIFT.value]
    assert not drift.empty and not drift["is_attack"].any()


def test_campaign_id_is_set_exactly_on_non_benign_rows(events):
    benign = events["label"] == AttackType.BENIGN.value
    assert events.loc[benign, "campaign_id"].isna().all()
    assert events.loc[~benign, "campaign_id"].notna().all()


def test_attack_events_stay_inside_the_simulation_horizon(events):
    cfg = load_config()
    start = pd.Timestamp(str(cfg["generator.start_date"]))
    end = start + pd.Timedelta(days=INJECTED_PROFILE["days"])

    attacks = events[events["label"] != AttackType.BENIGN.value]
    assert attacks["timestamp"].min() >= start
    assert attacks["timestamp"].max() < end


def test_attacks_never_leak_ground_truth_into_observable_fields(events):
    attacks = events[events["label"] != AttackType.BENIGN.value]
    # Attack rows must reuse the same vocabulary as benign traffic, or a model
    # could win by memorising an attacker-only category instead of learning
    # behaviour.
    benign = events[events["label"] == AttackType.BENIGN.value]
    assert set(attacks["auth_method"]) <= set(benign["auth_method"])
    assert set(attacks["action"]) <= set(benign["action"])
    assert set(attacks["entity_id"]) <= set(benign["entity_id"])


def test_injection_is_reproducible_across_runs():
    first = generate_dataset(profile=INJECTED_PROFILE)
    second = generate_dataset(profile=INJECTED_PROFILE)

    pd.testing.assert_frame_equal(first.events, second.events)
    assert [c.campaign_id for c in first.campaigns] == [
        c.campaign_id for c in second.campaigns
    ]
    assert [c.stealthy for c in first.campaigns] == [
        c.stealthy for c in second.campaigns
    ]


def test_benign_only_generation_remains_available(generated):
    clean = generate_dataset(profile=INJECTED_PROFILE, inject=False)

    assert clean.events["label"].eq(AttackType.BENIGN.value).all()
    assert not clean.campaigns
    # The benign stream is a strict prefix of the work the full run does.
    assert len(clean.events) < len(generated.events)


def test_geo_helpers_behave_at_the_boundaries():
    assert haversine_km(0.0, 0.0, 0.0, 0.0) == pytest.approx(0.0)
    # Chennai to London, a known distance to roughly a percent.
    assert haversine_km(13.0827, 80.2707, 51.5074, -0.1278) == pytest.approx(
        8250.0, rel=0.02
    )
    assert implied_velocity_kmh(100.0, 3600.0) == pytest.approx(100.0)
    assert implied_velocity_kmh(0.0, 0.0) == 0.0
    assert implied_velocity_kmh(500.0, 0.0) == float("inf")

    # The vectorised form is what the Phase 4 feature pass uses, so it has to
    # agree with the scalar one exactly.
    lats = np.array([0.0, 13.0827, -33.8688])
    lons = np.array([0.0, 80.2707, 151.2093])
    assert np.allclose(
        haversine_km_array(lats, lons, np.flip(lats), np.flip(lons)),
        [haversine_km(a, b, c, d) for a, b, c, d in
         zip(lats, lons, np.flip(lats), np.flip(lons))],
    )
