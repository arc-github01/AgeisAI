"""Risk signal definitions, activations and their benign calibration.

Every signal turns one observable quantity into a bounded activation in
``[0, 1]``. Nothing here may see ground truth: :data:`RISK_INPUT_COLUMNS` is the
whitelist of what the engine is allowed to read at scoring time, and
:func:`assert_no_evaluation_metadata` makes a violation an exception rather than
a code-review opinion.

Labels *are* used in exactly one place, and only in the same way Phase 5 uses
them: :func:`fit_calibration` selects benign training-period rows to derive
activation anchors. Selecting fitting rows is not the same as feeding a label to
a scorer, and the causality tests pin the distinction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from src.config import load_config
from src.features import MODEL_FEATURE_COLUMNS

#: Identity/context the engine may read. All are observable at inference time.
RISK_IDENTITY_COLUMNS: tuple[str, ...] = (
    "event_id",
    "timestamp",
    "entity_id",
    "entity_type",
)

#: Upstream detector outputs the engine consumes (Phase 5 artifacts).
RISK_DETECTOR_COLUMNS: tuple[str, ...] = ("anomaly_score_raw", "baseline_rule")

#: Ground-truth / bookkeeping columns that must never reach the engine.
FORBIDDEN_RISK_INPUT_COLUMNS: frozenset[str] = frozenset(
    {
        "label",
        "is_attack",
        "attack_type",
        "campaign_id",
        "stealthy",
        "difficulty",
        "split",
    }
)


class RiskInputLeakageError(RuntimeError):
    """Raised when evaluation metadata reaches the risk engine's inputs."""


@dataclass(frozen=True)
class SignalSpec:
    """One configured contextual signal."""

    name: str
    reason: str
    weight: float
    mode: str
    feature: str | None = None
    low: float | None = None
    high: float | None = None
    components: tuple[str, ...] = ()

    def features_used(self) -> tuple[str, ...]:
        if self.mode == "min_of":
            return self.components
        return (self.feature,) if self.feature else ()


@dataclass(frozen=True)
class EngineSpec:
    """The fully resolved, configuration-driven engine definition."""

    version: int
    evidence_scale: float
    low_quantile: float
    high_quantile: float
    halflife_seconds: float
    persistence_weight: float
    state_cap: float
    persistence_reason: str
    isolation_forest_weight: float
    isolation_forest_reason: str
    rule_weight: float
    rule_reason: str
    context: tuple[SignalSpec, ...]

    def context_features(self) -> tuple[str, ...]:
        seen: list[str] = []
        for spec in self.context:
            for feature in spec.features_used():
                if feature not in seen:
                    seen.append(feature)
        return tuple(seen)


def load_engine_spec(cfg: Any | None = None) -> EngineSpec:
    """Build the engine specification from ``config.yaml``."""
    cfg = cfg or load_config()
    engine = cfg["risk.engine"]
    persistence = engine["persistence"]
    signals = engine["signals"]
    context = tuple(
        SignalSpec(
            name=str(item["name"]),
            reason=str(item["reason"]),
            weight=float(item["weight"]),
            mode=str(item["mode"]),
            feature=item.get("feature"),
            low=item.get("low"),
            high=item.get("high"),
            components=tuple(item.get("components", ())),
        )
        for item in signals["context"]
    )
    spec = EngineSpec(
        version=int(engine["version"]),
        evidence_scale=float(engine["evidence_scale"]),
        low_quantile=float(engine["calibration"]["low_quantile"]),
        high_quantile=float(engine["calibration"]["high_quantile"]),
        halflife_seconds=float(persistence["halflife_seconds"]),
        persistence_weight=float(persistence["weight"]),
        state_cap=float(persistence["state_cap"]),
        persistence_reason=str(persistence["reason"]),
        isolation_forest_weight=float(signals["isolation_forest"]["weight"]),
        isolation_forest_reason=str(signals["isolation_forest"]["reason"]),
        rule_weight=float(signals["rule"]["weight"]),
        rule_reason=str(signals["rule"]["reason"]),
        context=context,
    )
    unknown = [f for f in spec.context_features() if f not in MODEL_FEATURE_COLUMNS]
    if unknown:
        raise ValueError(f"risk context signals reference unknown features: {unknown}")
    return spec


def risk_input_columns(spec: EngineSpec | None = None) -> tuple[str, ...]:
    """The exhaustive whitelist of columns the engine reads while scoring."""
    spec = spec or load_engine_spec()
    return RISK_IDENTITY_COLUMNS + RISK_DETECTOR_COLUMNS + spec.context_features()


#: Resolved from the default configuration, mirroring MODEL_FEATURE_COLUMNS.
RISK_INPUT_COLUMNS: tuple[str, ...] = risk_input_columns()


def assert_no_evaluation_metadata(columns: Iterable[str]) -> None:
    """Fail loudly if ground truth appears in a risk-input column set."""
    offenders = sorted(FORBIDDEN_RISK_INPUT_COLUMNS.intersection(columns))
    if offenders:
        raise RiskInputLeakageError(
            "evaluation metadata must never be a risk engine input: "
            + ", ".join(offenders)
        )


assert_no_evaluation_metadata(RISK_INPUT_COLUMNS)


# -----------------------------------------------------------------------------
# Calibration
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Calibration:
    """Activation anchors derived from benign training-period events only."""

    isolation_forest_floor: float
    isolation_forest_thresholds: dict[str, float]
    rule_low: float
    rule_high: float
    feature_anchors: dict[str, tuple[float, float]] = field(default_factory=dict)
    n_calibration_events: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "isolation_forest_floor": self.isolation_forest_floor,
            "isolation_forest_thresholds": self.isolation_forest_thresholds,
            "rule_low": self.rule_low,
            "rule_high": self.rule_high,
            "feature_anchors": {k: list(v) for k, v in self.feature_anchors.items()},
            "n_calibration_events": self.n_calibration_events,
        }


def _anchor(values: np.ndarray, low_q: float, high_q: float) -> tuple[float, float]:
    low = float(np.quantile(values, low_q))
    high = float(np.quantile(values, high_q))
    if high <= low:
        # Degenerate/near-constant feature: fall back to a unit ramp above low so
        # the signal can still fire rather than silently dividing by zero.
        high = low + 1.0
    return low, high


def fit_calibration(
    benign_training_rows: pd.DataFrame,
    isolation_forest_thresholds: dict[str, float],
    spec: EngineSpec | None = None,
) -> Calibration:
    """Derive activation anchors from benign training-period events.

    ``benign_training_rows`` must already be restricted to benign rows inside
    the training split; this function never inspects labels itself.
    """
    spec = spec or load_engine_spec()
    if benign_training_rows.empty:
        raise ValueError("no benign training rows to calibrate the risk engine")

    anchors: dict[str, tuple[float, float]] = {}
    for signal in spec.context:
        for feature in signal.features_used():
            if feature in anchors or signal.mode in {"boolean", "inverse_boolean"}:
                continue
            values = benign_training_rows[feature].to_numpy(dtype=float)
            if signal.mode == "abs_ramp":
                values = np.abs(values)
            if signal.mode == "fixed_ramp":
                continue
            anchors[feature] = _anchor(values, spec.low_quantile, spec.high_quantile)

    forest_scores = benign_training_rows["anomaly_score_raw"].to_numpy(dtype=float)
    rule_scores = benign_training_rows["baseline_rule"].to_numpy(dtype=float)
    rule_low, rule_high = _anchor(rule_scores, spec.low_quantile, spec.high_quantile)
    return Calibration(
        # Half of benign traffic sits below the median and earns no anomaly risk.
        isolation_forest_floor=float(np.quantile(forest_scores, 0.5)),
        isolation_forest_thresholds=dict(isolation_forest_thresholds),
        rule_low=rule_low,
        rule_high=rule_high,
        feature_anchors=anchors,
        n_calibration_events=int(len(benign_training_rows)),
    )


# -----------------------------------------------------------------------------
# Activations
# -----------------------------------------------------------------------------
def _ramp(value: float, low: float, high: float) -> float:
    if high <= low:
        return 1.0 if value > low else 0.0
    return float(min(1.0, max(0.0, (value - low) / (high - low))))


def isolation_forest_activation(score: float, calibration: Calibration) -> float:
    """Piecewise activation anchored on the Phase 5 calibrated operating points.

    Each operating point is worth a quarter of the activation range, so the
    engine keeps the operational meaning of "sensitive / balanced / strict"
    instead of inventing its own normalisation. Above ``strict`` the mapping is
    an exponential approach to 1.0, which stays strictly monotonic (no ranking
    information is lost) while remaining bounded.
    """
    thresholds = calibration.isolation_forest_thresholds
    sensitive = thresholds["sensitive"]
    balanced = thresholds["balanced"]
    strict = thresholds["strict"]
    floor = calibration.isolation_forest_floor

    if score <= floor:
        return 0.0
    if score <= sensitive:
        return 0.25 * _ramp(score, floor, sensitive)
    if score <= balanced:
        return 0.25 + 0.25 * _ramp(score, sensitive, balanced)
    if score <= strict:
        return 0.50 + 0.25 * _ramp(score, balanced, strict)
    scale = max(strict - balanced, 1e-9)
    return 0.75 + 0.25 * (1.0 - math.exp(-(score - strict) / scale))


def isolation_forest_operating_point(score: float, calibration: Calibration) -> str:
    thresholds = calibration.isolation_forest_thresholds
    if score >= thresholds["strict"]:
        return "strict"
    if score >= thresholds["balanced"]:
        return "balanced"
    if score >= thresholds["sensitive"]:
        return "sensitive"
    return "below_sensitive"


def rule_activation(score: float, calibration: Calibration) -> float:
    """Smooth saturating activation above the benign rule-score anchor."""
    span = max(calibration.rule_high - calibration.rule_low, 1e-9)
    excess = max(0.0, score - calibration.rule_low)
    return float(1.0 - math.exp(-excess / span))


def context_activation(
    signal: SignalSpec, row: dict[str, float], calibration: Calibration
) -> float:
    """Bounded activation for one contextual signal."""
    if signal.mode == "boolean":
        return float(min(1.0, max(0.0, float(row[signal.feature]))))
    if signal.mode == "inverse_boolean":
        return float(min(1.0, max(0.0, 1.0 - float(row[signal.feature]))))
    if signal.mode == "fixed_ramp":
        return _ramp(float(row[signal.feature]), float(signal.low), float(signal.high))
    if signal.mode == "ramp":
        low, high = calibration.feature_anchors[signal.feature]
        return _ramp(float(row[signal.feature]), low, high)
    if signal.mode == "abs_ramp":
        low, high = calibration.feature_anchors[signal.feature]
        return _ramp(abs(float(row[signal.feature])), low, high)
    if signal.mode == "min_of":
        parts = []
        for feature in signal.components:
            low, high = calibration.feature_anchors[feature]
            parts.append(_ramp(float(row[feature]), low, high))
        return float(min(parts)) if parts else 0.0
    raise ValueError(f"unknown risk signal mode: {signal.mode!r}")


__all__ = [
    "Calibration",
    "EngineSpec",
    "FORBIDDEN_RISK_INPUT_COLUMNS",
    "RISK_DETECTOR_COLUMNS",
    "RISK_IDENTITY_COLUMNS",
    "RISK_INPUT_COLUMNS",
    "RiskInputLeakageError",
    "SignalSpec",
    "assert_no_evaluation_metadata",
    "context_activation",
    "fit_calibration",
    "isolation_forest_activation",
    "isolation_forest_operating_point",
    "load_engine_spec",
    "risk_input_columns",
    "rule_activation",
]
