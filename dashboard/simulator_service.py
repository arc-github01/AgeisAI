"""Attack simulator orchestration for the dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from src.artifacts import Artifact, artifact
from src.detection.injection import (
    DetectionPipelineNotReadyError,
    InjectionRequest,
    InjectionResult,
    process_injection,
)
from src.generator.live_injection import synthesize_live_attack
from src.schema import IDENTITY_COLUMNS, OBSERVATION_COLUMNS

from . import live_state, mock_data
from .contracts import DASHBOARD_ALERT_COLUMNS
from .data_provider import DashboardDataProvider
from .state import DashboardContext

REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "entities",
    "events",
    "profiles",
    "features",
    "anomaly_detector",
    "attack_classifier",
    "risk_calibration",
    "rule_baseline",
)

PIPELINE_STAGE_ARTIFACTS: dict[str, str | None] = {
    "EVENT": "events",
    "FEATURES": "features",
    "PROFILE": "profiles",
    "ANOMALY": "anomaly_detector",
    "CLASSIFIER": "attack_classifier",
    "RISK": "risk_calibration",
    "EXPLANATION": "alerts",
    "ALERT": "alerts",
}

SCENARIO_DESCRIPTIONS: dict[str, str] = {
    "BRUTE_FORCE": "Burst of failed authentications from one source in a short window.",
    "IMPOSSIBLE_TRAVEL": "Second authentication from a distant city sooner than travel allows.",
    "CREDENTIAL_STUFFING": "Repeated authentication failures consistent with credential reuse.",
    "LATERAL_MOVEMENT": "Sudden breadth of never-before-accessed internal systems.",
    "DEVICE_SPOOFING": "Known identity presenting an inconsistent device fingerprint.",
    "LOW_AND_SLOW_EXFILTRATION": "Small off-hours accesses accumulating over days.",
    "INSIDER_DRIFT": "Gradual, legitimate-looking expansion of resource footprint.",
}


@dataclass(frozen=True)
class PrerequisiteStatus:
    key: str
    description: str
    phase: int
    produced_by: str
    ready: bool


@dataclass(frozen=True)
class SimulatorRunOutcome:
    """Result of an injection attempt."""

    success: bool
    request: InjectionRequest
    events: pd.DataFrame
    result: InjectionResult | None = None
    error: str | None = None
    alerts_posted: int = 0


def _alert_from_result(result_alert: Mapping[str, Any] | pd.Series) -> dict[str, Any]:
    """Normalise a pipeline alert dict/series into the dashboard alert contract."""
    if isinstance(result_alert, pd.Series):
        raw = result_alert.to_dict()
    else:
        raw = dict(result_alert)
    row: dict[str, Any] = {}
    for column in DASHBOARD_ALERT_COLUMNS:
        row[column] = raw.get(column)
    if row.get("timestamp") is not None:
        row["timestamp"] = pd.Timestamp(row["timestamp"])
    # Prefer structured contributors for the contribution chart.
    if row.get("top_contributors") is None and isinstance(raw.get("reasons"), list):
        row["top_contributors"] = raw["reasons"]
    return row


class SimulatorService:
    """Generate injection events and delegate scoring to the detection pipeline."""

    def __init__(
        self,
        ctx: DashboardContext,
        provider: DashboardDataProvider | None = None,
    ) -> None:
        self._ctx = ctx
        self._provider = provider or DashboardDataProvider.from_context(ctx)

    def list_entities(self) -> list[str]:
        return self._provider.list_entity_ids()

    def scenario_description(self, attack_type: str) -> str:
        return SCENARIO_DESCRIPTIONS.get(attack_type, "")

    def prerequisites(self) -> list[PrerequisiteStatus]:
        statuses: list[PrerequisiteStatus] = []
        for key in REQUIRED_ARTIFACTS:
            item: Artifact = artifact(key)
            statuses.append(
                PrerequisiteStatus(
                    key=key,
                    description=item.description,
                    phase=item.phase,
                    produced_by=item.produced_by,
                    ready=self._ctx.has(key),
                )
            )
        return statuses

    def is_ready(self) -> bool:
        return all(status.ready for status in self.prerequisites())

    def active_pipeline_stages(self) -> set[str]:
        active: set[str] = set()
        for stage, artifact_key in PIPELINE_STAGE_ARTIFACTS.items():
            if artifact_key is None:
                continue
            if self._ctx.has(artifact_key):
                active.add(stage)
        return active

    def synthesize_events(
        self,
        entity_id: str,
        attack_type: str,
        intensity: int,
    ) -> pd.DataFrame:
        """Create injection input events via the real Phase 3 attack injectors.

        Falls back to the development fixture only when the generator roster is
        unavailable (unit tests / fixture-only shell).
        """
        if self._ctx.has("entities") and self._ctx.has("events"):
            return synthesize_live_attack(
                entity_id=entity_id,
                attack_type=attack_type,
                intensity=intensity,
            )

        entities = self._provider.get_entities()
        if entities.empty:
            raise ValueError("entity roster is unavailable")
        return mock_data.generate_injection_events(
            entity_id=entity_id,
            attack_type=attack_type,
            intensity=intensity,
            entities=entities,
        )

    def run(
        self,
        entity_id: str,
        attack_type: str,
        intensity: int,
    ) -> SimulatorRunOutcome:
        if not self.is_ready():
            raise RuntimeError("simulator prerequisites are not satisfied")

        request = InjectionRequest(
            entity_id=entity_id,
            attack_type=attack_type,
            intensity=intensity,
        )
        events = pd.DataFrame()
        try:
            events = self.synthesize_events(entity_id, attack_type, intensity)
            result = process_injection(events, request)
            posted = self._publish_to_dashboard(events, result)
            return SimulatorRunOutcome(
                success=True,
                request=request,
                events=events,
                result=result,
                alerts_posted=posted,
            )
        except (
            DetectionPipelineNotReadyError,
            ValueError,
            FileNotFoundError,
            OSError,
            KeyError,
            TypeError,
        ) as exc:
            return SimulatorRunOutcome(
                success=False,
                request=request,
                events=events,
                error=str(exc),
            )
        except Exception as exc:  # noqa: BLE001 — never fabricate alerts on any failure
            return SimulatorRunOutcome(
                success=False,
                request=request,
                events=events,
                error=f"Failed to run live detection pipeline: {exc}",
            )

    def _publish_to_dashboard(
        self,
        events: pd.DataFrame,
        result: InjectionResult,
    ) -> int:
        """Push scored alerts + injection events into session overlays."""
        alerts: list[dict[str, Any]] = []
        for scored in result.results:
            if scored.alert is None:
                continue
            alerts.append(_alert_from_result(scored.alert))

        # Keep observation columns that entity investigation can display.
        keep = [c for c in IDENTITY_COLUMNS + OBSERVATION_COLUMNS if c in events.columns]
        live_events = events.loc[:, keep].copy() if keep else events.copy()
        live_state.append_live_injection(alerts=alerts, events=live_events)
        return len(alerts)
