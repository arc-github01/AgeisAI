"""Causal replay of historical events through ``process_event``."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.artifacts import artifact_path
from src.evaluation.report import RunManifest, write_json
from src.schema import IDENTITY_COLUMNS, OBSERVATION_COLUMNS

from .engine import StreamingEngine
from .result import EventResult


def _latency_stats(
    latencies_ms: list[float], *, wall_time_s: float | None = None
) -> dict[str, Any]:
    if not latencies_ms:
        return {
            "n": 0,
            "mean_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "throughput_events_per_sec": None,
        }
    arr = np.asarray(latencies_ms, dtype=float)
    measured_s = (
        float(wall_time_s)
        if wall_time_s is not None
        else float(arr.sum() / 1000.0)
    )
    return {
        "n": int(len(arr)),
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "wall_time_s": measured_s,
        "throughput_events_per_sec": (
            float(len(arr) / measured_s) if measured_s > 0 else None
        ),
        "note": (
            "Measured on this local process for the replayed stream. "
            "Production throughput depends on deployment topology and hardware."
        ),
    }


def replay_events(
    events: pd.DataFrame,
    *,
    engine: StreamingEngine | None = None,
    apply_drift_updates: bool | None = None,
) -> tuple[list[EventResult], StreamingEngine]:
    """Feed events one-by-one in stable chronological order through ``process_event``."""
    cols = [c for c in IDENTITY_COLUMNS + OBSERVATION_COLUMNS if c in events.columns]
    ordered = events.loc[:, cols].sort_values(
        ["timestamp", "event_id"], kind="stable"
    ).reset_index(drop=True)

    eng = engine or StreamingEngine.load(
        apply_drift_updates=True if apply_drift_updates is None else apply_drift_updates
    )
    if apply_drift_updates is not None:
        eng.apply_drift_updates = apply_drift_updates

    results: list[EventResult] = []
    for _, row in ordered.iterrows():
        results.append(eng.process_event(row))
    return results, eng


def warm_entity_histories(
    engine: StreamingEngine,
    history_events: pd.DataFrame,
    *,
    score: bool = False,
) -> None:
    """Prime per-entity histories (and optionally full state) from prior events.

    When ``score`` is False, only rolling history/fingerprints are updated so a
    live injection can see recent context without re-emitting alerts for the past.
    """
    cols = [c for c in IDENTITY_COLUMNS + OBSERVATION_COLUMNS if c in history_events.columns]
    ordered = history_events.loc[:, cols].sort_values(
        ["timestamp", "event_id"], kind="stable"
    )
    if score:
        for _, row in ordered.iterrows():
            engine.process_event(row)
        return
    for _, row in ordered.iterrows():
        cleaned = engine._strip_forbidden(row)
        entity_id = str(cleaned.entity_id)
        ts = pd.Timestamp(cleaned.timestamp)
        hist = engine.histories.setdefault(entity_id, [])
        hist.append(cleaned)
        engine.previous_fingerprint[entity_id] = (
            f"{cleaned.device_id}|{cleaned.device_mac}"
        )
        engine._trim_history(entity_id, ts)


def results_to_frame(results: Iterable[EventResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "event_id": r.event_id,
                "entity_id": r.entity_id,
                "entity_type": r.entity_type,
                "timestamp": r.timestamp,
                "anomaly_score_raw": r.anomaly_score_raw,
                "anomaly_score": r.anomaly_score,
                "baseline_rule": r.baseline_rule,
                "risk_score": r.risk_score,
                "severity": r.severity,
                "predicted_attack_type": r.predicted_attack_type,
                "attack_confidence": r.attack_confidence,
                "alerted": r.alerted,
                "alert_suppressed": r.alert_suppressed,
                "profile_updated": r.profile_updated,
                "profile_source": r.profile_source,
                "profile_confidence": r.profile_confidence,
                "entity_evidence_state": r.entity_evidence_state,
                "latency_ms": r.latency_ms,
                "short_reason": r.short_reason,
            }
        )
    return pd.DataFrame(rows)


def run_replay(
    *,
    apply_drift_updates: bool = True,
    max_events: int | None = None,
) -> dict[str, Path]:
    """Replay events through streaming inference; write scores + latency metrics."""
    from src.config import load_config

    cfg = load_config()
    if max_events is None:
        configured = cfg.get("detection.replay_max_events")
        max_events = None if configured is None else int(configured)

    events = pd.read_parquet(artifact_path("events"))
    corpus_n = int(len(events))
    events = events.sort_values(["timestamp", "event_id"], kind="stable")
    if max_events is not None:
        events = events.head(int(max_events)).reset_index(drop=True)
    engine = StreamingEngine.load(apply_drift_updates=apply_drift_updates)
    replay_started = time.perf_counter()
    results, engine = replay_events(
        events, engine=engine, apply_drift_updates=apply_drift_updates
    )
    replay_wall_time_s = time.perf_counter() - replay_started
    frame = results_to_frame(results)
    scores_path = artifact_path("streaming_scores", ensure_parent=True)
    frame.to_parquet(scores_path, index=False)

    latencies = [r.latency_ms for r in results]
    doc: dict[str, Any] = {
        "manifest": asdict(
            RunManifest.capture(notes="Phase 9 streaming replay performance")
        ),
        "n_events": len(results),
        "n_events_in_corpus": corpus_n,
        "replay_max_events": max_events,
        "n_alerts": int(sum(1 for r in results if r.alerted)),
        "n_suppressed": int(sum(1 for r in results if r.alert_suppressed)),
        "n_profile_updates": int(sum(1 for r in results if r.profile_updated)),
        "apply_drift_updates": apply_drift_updates,
        "latency": _latency_stats(latencies, wall_time_s=replay_wall_time_s),
    }
    metrics_path = artifact_path("streaming_metrics", ensure_parent=True)
    write_json(doc, metrics_path)
    return {"streaming_scores": scores_path, "streaming_metrics": metrics_path}


__all__ = [
    "replay_events",
    "results_to_frame",
    "run_replay",
    "warm_entity_histories",
]
