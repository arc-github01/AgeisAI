"""Live attack injection contract for the detection pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


class DetectionPipelineNotReadyError(RuntimeError):
    """Raised when injection is requested before the detector is implemented."""


@dataclass(frozen=True)
class InjectionRequest:
    entity_id: str
    attack_type: str
    intensity: int = 3


@dataclass
class InjectionResult:
    """Output of a live injection run."""

    request: InjectionRequest
    events: pd.DataFrame
    alert: pd.Series | None = None
    stages_completed: list[str] = field(default_factory=list)
    message: str = ""


def process_injection(events: pd.DataFrame, request: InjectionRequest) -> InjectionResult:
    """Score injected events through the live detection pipeline.

    Phase 11 replaces this stub with feature engineering, profiling, anomaly
    detection, classification, risk scoring and explainability.
    """
    if events.empty:
        raise ValueError("injection requires at least one event")

    raise DetectionPipelineNotReadyError(
        "Live detection pipeline is not implemented yet (Phase 11). "
        "Events were generated but not scored."
    )


__all__ = [
    "DetectionPipelineNotReadyError",
    "InjectionRequest",
    "InjectionResult",
    "process_injection",
]
