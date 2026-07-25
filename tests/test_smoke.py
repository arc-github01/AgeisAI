"""Phase 1 smoke tests: environment, configuration, determinism, schema guards.

These are intentionally cheap and fast - they are the tripwire that tells us the
foundation is intact before any ML component is added on top of it.
"""

from __future__ import annotations

import pandas as pd
import pytest

import src
from src.config import Config, ConfigError, load_config
from src.paths import CONFIG_FILE, PROJECT_ROOT
from src.schema import (
    ATTACK_CLASSES,
    EVENT_COLUMNS,
    LABEL_COLUMNS,
    AttackType,
    EntityType,
    LabelLeakageError,
    SchemaError,
    Severity,
    assert_no_label_leakage,
    feature_safe_columns,
    validate_events,
)
from src.utils import derive_seed, get_rng, seed_everything


# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
def test_third_party_stack_imports():
    import joblib  # noqa: F401
    import numpy  # noqa: F401
    import plotly  # noqa: F401
    import sklearn  # noqa: F401
    import yaml  # noqa: F401

    assert src.__version__


def test_project_root_is_repository_root():
    assert (PROJECT_ROOT / "requirements.txt").exists()
    assert CONFIG_FILE.exists()


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
def test_config_loads_and_supports_dotted_access():
    cfg = load_config()
    assert isinstance(cfg, Config)
    assert cfg["project.name"] == "AEGIS"
    assert isinstance(cfg["risk.weights.ml_anomaly"], float)
    assert cfg.get("does.not.exist", "fallback") == "fallback"
    with pytest.raises(ConfigError):
        _ = cfg["does.not.exist"]


def test_config_is_immutable_from_the_outside():
    cfg = load_config()
    weights = cfg["risk.weights"]
    weights["ml_anomaly"] = 999.0
    assert load_config()["risk.weights.ml_anomaly"] != 999.0


def test_risk_weights_sum_to_one():
    weights = load_config()["risk.weights"]
    assert pytest.approx(sum(weights.values()), abs=1e-9) == 1.0


def test_severity_bands_cover_zero_to_hundred_without_gaps():
    bands = load_config()["risk.severity_bands"]
    assert set(bands) == {s.value for s in Severity}
    ordered = sorted(bands.values(), key=lambda pair: pair[0])
    assert ordered[0][0] == 0 and ordered[-1][1] == 100
    for lower, upper in zip(ordered, ordered[1:]):
        assert upper[0] == lower[1] + 1


def test_attack_mix_matches_the_taxonomy():
    mix = load_config()["generator.attack_mix"]
    assert {k.upper() for k in mix} == set(ATTACK_CLASSES)
    assert all(v > 0 for v in mix.values())


def test_attack_prevalence_stays_realistically_low():
    prevalence = load_config()["generator.attack_prevalence"]
    assert 0.005 <= prevalence <= 0.03


def test_generator_profile_resolves():
    profile = load_config().generator_profile()
    assert profile["name"] in {"dev", "full"}
    assert profile["target_events"] > 0
    assert profile["n_users"] > 0


def test_config_paths_are_created_on_demand():
    cfg = load_config()
    for key in ("data_generated", "models", "artifacts"):
        assert cfg.path(key).is_dir()


# -----------------------------------------------------------------------------
# Determinism
# -----------------------------------------------------------------------------
def test_derived_seeds_are_stable_and_component_specific():
    assert derive_seed("generator") == derive_seed("generator")
    assert derive_seed("generator") != derive_seed("attacks")


def test_rng_streams_are_reproducible_and_independent():
    first = get_rng("generator").normal(size=5)
    second = get_rng("generator").normal(size=5)
    other = get_rng("attacks").normal(size=5)
    assert (first == second).all()
    assert not (first == other).all()


def test_seed_everything_returns_the_derived_seed():
    assert seed_everything("global") == derive_seed("global")


# -----------------------------------------------------------------------------
# Schema + label-leakage guard
# -----------------------------------------------------------------------------
def test_taxonomy_is_complete():
    assert len(ATTACK_CLASSES) == 7
    assert AttackType.BENIGN.value not in ATTACK_CLASSES
    assert set(EntityType) == {"user", "service_account", "edge_device"}


def test_label_columns_are_rejected_as_features():
    with pytest.raises(LabelLeakageError):
        assert_no_label_leakage(["hour_of_day", "is_attack"])
    assert_no_label_leakage(["hour_of_day", "geo_velocity_kmh"])
    assert feature_safe_columns(["hour_of_day", *LABEL_COLUMNS]) == ["hour_of_day"]


def _minimal_event_frame() -> pd.DataFrame:
    row = {column: "x" for column in EVENT_COLUMNS}
    row.update(
        timestamp=pd.Timestamp("2025-01-06 09:00:00"),
        entity_type=EntityType.USER.value,
        auth_success=True,
        latitude=13.08,
        longitude=80.27,
        session_duration_s=612.0,
        bytes_transferred=2048.0,
        label=AttackType.BENIGN.value,
        is_attack=False,
        campaign_id=None,
    )
    return pd.DataFrame([row])


def test_validate_events_accepts_a_conforming_frame():
    frame = _minimal_event_frame()
    assert validate_events(frame, strict_order=True) is frame


def test_validate_events_rejects_missing_columns_and_bad_labels():
    with pytest.raises(SchemaError):
        validate_events(_minimal_event_frame().drop(columns=["source_ip"]))
    bad = _minimal_event_frame()
    bad["label"] = "NOT_A_REAL_LABEL"
    with pytest.raises(SchemaError):
        validate_events(bad)


def test_validate_events_allows_unlabelled_live_events():
    live = _minimal_event_frame().drop(columns=list(LABEL_COLUMNS))
    validate_events(live, require_labels=False)
