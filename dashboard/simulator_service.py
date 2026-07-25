"""Attack simulator orchestration for the dashboard."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.artifacts import Artifact, artifact
from src.detection.injection import (
    DetectionPipelineNotReadyError,
    InjectionRequest,
    InjectionResult,
    process_injection,
)

from . import mock_data
from .data_provider import DashboardDataProvider
from .state import DashboardContext

REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "entities",
    "events",
    "profiles",
    "anomaly_detector",
    "attack_classifier",
)

PIPELINE_STAGE_ARTIFACTS: dict[str, str | None] = {
    "EVENT": "events",
    "FEATURES": "features",
    "PROFILE": "profiles",
    "ANOMALY": "anomaly_detector",
    "CLASSIFIER": "attack_classifier",
    "RISK": "alerts",
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
        """Create injection input events.

        Uses the development fixture until the backend generator exposes a live
        attack injection API.
        """
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
        events = self.synthesize_events(entity_id, attack_type, intensity)
        try:
            result = process_injection(events, request)
            return SimulatorRunOutcome(
                success=True,
                request=request,
                events=events,
                result=result,
            )
        except DetectionPipelineNotReadyError as exc:
            return SimulatorRunOutcome(
                success=False,
                request=request,
                events=events,
                error=str(exc),
            )
