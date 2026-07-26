"""Phase 6 hybrid behavioural risk engine.

Combines IsolationForest anomaly scores, the honest rule baseline, contextual
behavioural features and causal entity-level persistence into an interpretable
``risk_score ∈ [0, 100]`` with structured explanations and HIGH/CRITICAL alerts.
"""

from .aggregation import (
    EntityRiskState,
    decay_factor,
    saturating_risk,
    severity_for,
)
from .engine import RiskEngine, build_scoring_frame, fit_risk_engine, run
from .signals import (
    FORBIDDEN_RISK_INPUT_COLUMNS,
    RISK_INPUT_COLUMNS,
    RiskInputLeakageError,
    assert_no_evaluation_metadata,
    load_engine_spec,
)

__all__ = [
    "EntityRiskState",
    "FORBIDDEN_RISK_INPUT_COLUMNS",
    "RISK_INPUT_COLUMNS",
    "RiskEngine",
    "RiskInputLeakageError",
    "assert_no_evaluation_metadata",
    "build_scoring_frame",
    "decay_factor",
    "fit_risk_engine",
    "load_engine_spec",
    "run",
    "saturating_risk",
    "severity_for",
]
