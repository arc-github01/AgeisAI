"""Live attack injection contract for the detection pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.artifacts import artifact_path
from src.schema import IDENTITY_COLUMNS, OBSERVATION_COLUMNS, validate_events

from .engine import StreamingEngine
from .replay import warm_entity_histories
from .result import EventResult


class DetectionPipelineNotReadyError(RuntimeError):
    """Raised when injection is requested before required artifacts exist."""


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
    results: list[EventResult] = field(default_factory=list)


_REQUIRED_FOR_INJECTION: tuple[str, ...] = (
    "profiles",
    "anomaly_detector",
    "attack_classifier",
    "risk_calibration",
    "features",
    "events",
)


def _artifact_ready(key: str) -> bool:
    path = artifact_path(key)
    if key == "risk_calibration":
        # RiskEngine.load() requires the joblib sibling, not only the JSON snapshot.
        return path.with_suffix(".joblib").exists()
    return path.exists()


def _assert_ready() -> None:
    missing = [key for key in _REQUIRED_FOR_INJECTION if not _artifact_ready(key)]
    # Prefer the persisted rule baseline when present; StreamingEngine can fall back.
    if not artifact_path("rule_baseline").exists() and not artifact_path("features").exists():
        missing.append("rule_baseline_or_features")
    if missing:
        raise DetectionPipelineNotReadyError(
            "Live detection pipeline prerequisites missing: "
            + ", ".join(missing)
            + ". Run the offline pipeline through `python -m src.risk` (and "
            "`python -m src.drift` for adaptive profiles) before injecting."
        )


def process_injection(events: pd.DataFrame, request: InjectionRequest) -> InjectionResult:
    """Score injected events through the real streaming detection pipeline."""
    if events.empty:
        raise ValueError("injection requires at least one event")
    validate_events(events, require_labels=False)
    _assert_ready()

    try:
        engine = StreamingEngine.load(apply_drift_updates=True)
    except Exception as exc:  # noqa: BLE001 - surface as not-ready for the UI
        raise DetectionPipelineNotReadyError(
            f"Failed to load streaming detection pipeline: {exc}"
        ) from exc

    cols = [c for c in IDENTITY_COLUMNS + OBSERVATION_COLUMNS if c in events.columns]
    ordered = events.loc[:, cols].sort_values(
        ["timestamp", "event_id"], kind="stable"
    )
    inject_start = pd.Timestamp(ordered.iloc[0]["timestamp"])
    involved = sorted(ordered["entity_id"].astype(str).unique().tolist())

    # Warm history strictly before the injected stream for every involved entity
    # (credential stuffing can span multiple victims).
    history = pd.read_parquet(artifact_path("events"))
    history["timestamp"] = pd.to_datetime(history["timestamp"])
    entity_history = history[
        (history["entity_id"].astype(str).isin(involved))
        & (history["timestamp"] < inject_start)
    ]
    if not entity_history.empty:
        warm_entity_histories(engine, entity_history, score=False)

    results: list[EventResult] = []
    for _, row in ordered.iterrows():
        results.append(engine.process_event(row))

    alert_series = None
    for result in reversed(results):
        if result.alert is not None:
            alert_series = pd.Series(result.alert)
            break

    stages = list(results[-1].stages_completed) if results else []
    message = (
        f"Processed {len(results)} injected event(s); "
        f"{sum(1 for r in results if r.alerted)} alert(s) emitted."
    )
    return InjectionResult(
        request=request,
        events=events,
        alert=alert_series,
        stages_completed=stages,
        message=message,
        results=results,
    )


__all__ = [
    "DetectionPipelineNotReadyError",
    "InjectionRequest",
    "InjectionResult",
    "process_injection",
]
