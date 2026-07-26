"""Structured result of a single streaming ``process_event`` call."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class EventResult:
    """Near-real-time inference outcome for one access event."""

    event_id: str
    entity_id: str
    entity_type: str
    timestamp: str
    anomaly_score_raw: float
    anomaly_score: float
    baseline_rule: float
    risk_score: float
    severity: str
    predicted_attack_type: str
    attack_confidence: float
    reason_codes: tuple[str, ...]
    reasons: tuple[dict[str, Any], ...]
    short_reason: str
    alerted: bool
    alert_suppressed: bool
    alert: dict[str, Any] | None
    profile_updated: bool
    profile_source: str
    profile_confidence: float
    entity_evidence_state: float
    latency_ms: float = 0.0
    stages_completed: tuple[str, ...] = field(
        default_factory=lambda: (
            "FEATURES",
            "ANOMALY",
            "CLASSIFIER",
            "RISK",
            "EXPLANATION",
            "ALERT",
            "DRIFT",
        )
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = ["EventResult"]
