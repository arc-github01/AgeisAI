"""Detection pipeline entry points — streaming inference and live injection."""

from __future__ import annotations

from .engine import (
    FORBIDDEN_INFERENCE_COLUMNS,
    INFERENCE_COLUMNS,
    StreamingEngine,
    StreamingOrderError,
    get_default_engine,
    process_event,
    reset_default_engine,
)
from .injection import (
    DetectionPipelineNotReadyError,
    InjectionRequest,
    InjectionResult,
    process_injection,
)
from .replay import replay_events, results_to_frame, run_replay, warm_entity_histories
from .result import EventResult

__all__ = [
    "DetectionPipelineNotReadyError",
    "EventResult",
    "FORBIDDEN_INFERENCE_COLUMNS",
    "INFERENCE_COLUMNS",
    "InjectionRequest",
    "InjectionResult",
    "StreamingEngine",
    "StreamingOrderError",
    "get_default_engine",
    "process_event",
    "process_injection",
    "replay_events",
    "reset_default_engine",
    "results_to_frame",
    "run_replay",
    "warm_entity_histories",
]
