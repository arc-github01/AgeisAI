"""Reliable batch replay of historical events through ``process_event``.

Kept separate from the Attack Simulator so a simple "replay next N" button
cannot destabilize the main artifact-backed SOC views.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.artifacts import Artifact, artifact
from src.detection.engine import StreamingEngine
from src.detection.replay import results_to_frame
from src.schema import IDENTITY_COLUMNS, OBSERVATION_COLUMNS

from .data_provider import DashboardDataProvider
from .state import DashboardContext

REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "events",
    "profiles",
    "features",
    "anomaly_detector",
    "attack_classifier",
    "risk_calibration",
    "rule_baseline",
)


@dataclass(frozen=True)
class PrerequisiteStatus:
    key: str
    description: str
    phase: int
    produced_by: str
    ready: bool


@dataclass(frozen=True)
class ReplayBatchOutcome:
    success: bool
    processed: int
    cursor: int
    total_events: int
    results: pd.DataFrame
    alerts_raised: int
    mean_latency_ms: float | None
    error: str | None = None


class StreamingReplayService:
    """Session-friendly wrapper around Phase 9 ``StreamingEngine.process_event``."""

    def __init__(
        self,
        ctx: DashboardContext,
        provider: DashboardDataProvider | None = None,
    ) -> None:
        self._ctx = ctx
        self._provider = provider or DashboardDataProvider.from_context(ctx)

    def prerequisites(self) -> list[PrerequisiteStatus]:
        rows: list[PrerequisiteStatus] = []
        for key in REQUIRED_ARTIFACTS:
            item: Artifact = artifact(key)
            rows.append(
                PrerequisiteStatus(
                    key=key,
                    description=item.description,
                    phase=item.phase,
                    produced_by=item.produced_by,
                    ready=item.exists(),
                )
            )
        return rows

    def is_ready(self) -> bool:
        return all(item.ready for item in self.prerequisites())

    def event_count(self) -> int:
        events = self._raw_events()
        return 0 if events.empty else len(events)

    def _raw_events(self) -> pd.DataFrame:
        """Load full schema events for ``process_event`` (not the UI subset)."""
        raw = self._ctx.events()
        if raw is None or raw.empty:
            # Fallback only when the raw artifact is unavailable.
            return self._provider.get_events()
        return raw

    def _ordered_events(self) -> pd.DataFrame:
        events = self._raw_events()
        if events.empty:
            return events
        cols = [c for c in IDENTITY_COLUMNS + OBSERVATION_COLUMNS if c in events.columns]
        return events.loc[:, cols].sort_values(
            ["timestamp", "event_id"], kind="stable"
        ).reset_index(drop=True)

    def load_engine(self, *, apply_drift_updates: bool = True) -> StreamingEngine:
        return StreamingEngine.load(apply_drift_updates=apply_drift_updates)

    def replay_batch(
        self,
        *,
        engine: StreamingEngine,
        cursor: int,
        batch_size: int,
    ) -> ReplayBatchOutcome:
        if not self.is_ready():
            missing = [p.key for p in self.prerequisites() if not p.ready]
            return ReplayBatchOutcome(
                success=False,
                processed=0,
                cursor=cursor,
                total_events=self.event_count(),
                results=pd.DataFrame(),
                alerts_raised=0,
                mean_latency_ms=None,
                error=f"Streaming prerequisites missing: {', '.join(missing)}",
            )

        ordered = self._ordered_events()
        total = len(ordered)
        if total == 0:
            return ReplayBatchOutcome(
                success=False,
                processed=0,
                cursor=cursor,
                total_events=0,
                results=pd.DataFrame(),
                alerts_raised=0,
                mean_latency_ms=None,
                error="No events available to replay.",
            )

        start = max(0, int(cursor))
        if start >= total:
            return ReplayBatchOutcome(
                success=True,
                processed=0,
                cursor=total,
                total_events=total,
                results=pd.DataFrame(),
                alerts_raised=0,
                mean_latency_ms=None,
                error=None,
            )

        end = min(total, start + max(1, int(batch_size)))
        batch = ordered.iloc[start:end]
        try:
            results = [engine.process_event(row) for _, row in batch.iterrows()]
        except Exception as exc:  # noqa: BLE001 — surface to analyst UI
            return ReplayBatchOutcome(
                success=False,
                processed=0,
                cursor=start,
                total_events=total,
                results=pd.DataFrame(),
                alerts_raised=0,
                mean_latency_ms=None,
                error=str(exc),
            )

        frame = results_to_frame(results)
        alerts_raised = int(sum(1 for item in results if item.alerted))
        latencies = [float(item.latency_ms) for item in results]
        mean_latency = float(sum(latencies) / len(latencies)) if latencies else None
        return ReplayBatchOutcome(
            success=True,
            processed=len(results),
            cursor=end,
            total_events=total,
            results=frame,
            alerts_raised=alerts_raised,
            mean_latency_ms=mean_latency,
            error=None,
        )

    def offline_replay_preview(self, *, limit: int = 200) -> pd.DataFrame:
        """Show previously persisted Phase 9 streaming scores when available."""
        scores = self._provider.get_streaming_scores()
        if scores is None or scores.empty:
            return pd.DataFrame()
        view = scores.copy()
        if "timestamp" in view.columns:
            view = view.sort_values("timestamp", ascending=False)
        return view.head(limit)

    def streaming_metrics_summary(self) -> dict[str, Any] | None:
        return self._provider.get_streaming_metrics()


__all__ = [
    "PrerequisiteStatus",
    "REQUIRED_ARTIFACTS",
    "ReplayBatchOutcome",
    "StreamingReplayService",
]
