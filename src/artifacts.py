"""Canonical artifact registry for AEGIS.

Every file the pipeline produces is declared here exactly once: the generator
writes to `artifact("events").path`, and the dashboard reads from the same
object. Nothing hardcodes a path string, so a producing phase and a consuming
phase can never drift apart.

The registry doubles as the dashboard's build checklist: each entry knows which
phase creates it and which command to run, which is what the "awaiting data"
empty states display.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .config import load_config
from .paths import resolve


@dataclass(frozen=True)
class Artifact:
    """A declared output of the pipeline."""

    key: str
    directory: str          # a `paths.*` key from config.yaml
    filename: str
    phase: int
    produced_by: str        # the command a user runs to create it
    description: str

    @property
    def path(self) -> Path:
        return load_config().path(self.directory) / self.filename

    def exists(self) -> bool:
        return self.path.exists()

    def status(self) -> "ArtifactStatus":
        path = self.path
        exists = path.exists()
        stat = path.stat() if exists else None
        return ArtifactStatus(
            artifact=self,
            exists=exists,
            modified=datetime.fromtimestamp(stat.st_mtime) if stat else None,
            size_bytes=stat.st_size if stat else None,
        )


@dataclass(frozen=True)
class ArtifactStatus:
    artifact: Artifact
    exists: bool
    modified: datetime | None
    size_bytes: int | None

    @property
    def key(self) -> str:
        return self.artifact.key

    @property
    def label(self) -> str:
        return self.artifact.description

    def human_size(self) -> str:
        if self.size_bytes is None:
            return "-"
        size = float(self.size_bytes)
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024 or unit == "GB":
                return f"{size:,.0f} {unit}" if unit == "B" else f"{size:,.1f} {unit}"
            size /= 1024
        return f"{size:,.1f} GB"


_ARTIFACTS: tuple[Artifact, ...] = (
    Artifact(
        key="entities",
        directory="data_generated",
        filename="entities.json",
        phase=2,
        produced_by="python -m src.generator",
        description="Synthetic entity population and behavioural profiles",
    ),
    Artifact(
        key="events",
        directory="data_generated",
        filename="events.parquet",
        phase=3,
        produced_by="python -m src.generator",
        description="Labelled access-event dataset (benign + injected attacks)",
    ),
    Artifact(
        key="campaign_metadata",
        directory="data_generated",
        filename="campaigns.json",
        phase=3,
        produced_by="python -m src.generator",
        description="Campaign ground truth for evaluation only",
    ),
    Artifact(
        key="features",
        directory="data_processed",
        filename="features.parquet",
        phase=4,
        produced_by="python -m src.features",
        description="Engineered behavioral feature matrix",
    ),
    Artifact(
        key="profiles",
        directory="models",
        filename="entity_profiles.joblib",
        phase=4,
        produced_by="python -m src.features",
        description="Per-entity and cohort behavioural baselines",
    ),
    Artifact(
        key="feature_validation",
        directory="artifacts",
        filename="feature_validation.json",
        phase=4,
        produced_by="python -m src.features",
        description="Feature sanity and leakage validation summary",
    ),
    Artifact(
        key="anomaly_detector",
        directory="models",
        filename="anomaly_detector.joblib",
        phase=6,
        produced_by="python -m src.models.anomaly_detector",
        description="IsolationForest detector bundle (preprocessing + model + feature order + config)",
    ),
    Artifact(
        key="anomaly_thresholds",
        directory="models",
        filename="anomaly_thresholds.json",
        phase=6,
        produced_by="python -m src.models.anomaly_detector",
        description="Calibrated operating-point thresholds and methodology",
    ),
    Artifact(
        key="rule_baseline",
        directory="models",
        filename="rule_baseline.joblib",
        phase=6,
        produced_by="python -m src.models.anomaly_detector",
        description="Honest rule-baseline standardisation fitted on benign train rows",
    ),
    Artifact(
        key="event_scores",
        directory="data_processed",
        filename="event_scores.parquet",
        phase=6,
        produced_by="python -m src.models.anomaly_detector",
        description="Per-event anomaly scores with evaluation-only labels/metadata",
    ),
    Artifact(
        key="evaluation_metrics",
        directory="artifacts",
        filename="metrics/evaluation_metrics.json",
        phase=6,
        produced_by="python -m src.models.anomaly_detector",
        description="Event-level and security-relevant anomaly detection metrics",
    ),
    Artifact(
        key="campaign_metrics",
        directory="artifacts",
        filename="metrics/campaign_metrics.json",
        phase=6,
        produced_by="python -m src.models.anomaly_detector",
        description="Campaign-level detection rate, latency and breakdowns",
    ),
    Artifact(
        key="attack_classifier",
        directory="models",
        filename="attack_classifier.joblib",
        phase=7,
        produced_by="python -m src.models.attack_classifier",
        description="Supervised RandomForest attack-type classifier bundle",
    ),
    Artifact(
        key="classifications",
        directory="data_processed",
        filename="classifications.parquet",
        phase=7,
        produced_by="python -m src.models.attack_classifier",
        description="Per-event predicted attack class and confidence",
    ),
    Artifact(
        key="classifier_metrics",
        directory="artifacts",
        filename="metrics/classifier_metrics.json",
        phase=7,
        produced_by="python -m src.models.attack_classifier",
        description="Per-class precision/recall/F1, confusion matrix, macro F1",
    ),
    Artifact(
        key="risk_calibration",
        directory="models",
        filename="risk_calibration.json",
        phase=8,
        produced_by="python -m src.risk",
        description="Risk-engine activation anchors and configuration snapshot",
    ),
    Artifact(
        key="risk_scores",
        directory="data_processed",
        filename="risk_scores.parquet",
        phase=8,
        produced_by="python -m src.risk",
        description="Per-event hybrid risk scores with contribution breakdown",
    ),
    Artifact(
        key="alerts",
        directory="artifacts",
        filename="alerts.parquet",
        phase=8,
        produced_by="python -m src.risk",
        description="HIGH/CRITICAL alert store with reasons and cooldown metadata",
    ),
    Artifact(
        key="risk_evaluation",
        directory="artifacts",
        filename="metrics/risk_evaluation.json",
        phase=8,
        produced_by="python -m src.risk",
        description="Risk-engine event-level and operational metrics",
    ),
    Artifact(
        key="risk_campaign_metrics",
        directory="artifacts",
        filename="metrics/risk_campaign_metrics.json",
        phase=8,
        produced_by="python -m src.risk",
        description="Risk-engine campaign detection and IF/rule/hybrid comparison",
    ),
    Artifact(
        key="adaptive_profiles",
        directory="models",
        filename="adaptive_profiles.joblib",
        phase=9,
        produced_by="python -m src.drift",
        description="Poisoning-resistant adaptive entity profiles after causal replay",
    ),
    Artifact(
        key="drift_evaluation",
        directory="artifacts",
        filename="metrics/drift_evaluation.json",
        phase=9,
        produced_by="python -m src.drift",
        description="Concept-drift adaptation vs poisoning-resistance evaluation",
    ),
    Artifact(
        key="streaming_scores",
        directory="data_processed",
        filename="streaming_scores.parquet",
        phase=10,
        produced_by="python -m src.detection",
        description="Per-event streaming process_event scores from causal replay",
    ),
    Artifact(
        key="streaming_metrics",
        directory="artifacts",
        filename="metrics/streaming_metrics.json",
        phase=10,
        produced_by="python -m src.detection",
        description="Streaming latency/throughput measured on local replay",
    ),
    Artifact(
        key="metrics",
        directory="artifacts",
        filename="metrics/latest.json",
        phase=12,
        produced_by="python -m src.evaluation",
        description="Evaluation metrics (PR-AUC, per-attack, alert budget)",
    ),
    Artifact(
        key="manifest",
        directory="artifacts",
        filename="run_manifest.json",
        phase=12,
        produced_by="python -m src.evaluation",
        description="Reproducibility manifest (seed, config, versions)",
    ),
)

REGISTRY: dict[str, Artifact] = {a.key: a for a in _ARTIFACTS}


def artifact(key: str) -> Artifact:
    if key not in REGISTRY:
        raise KeyError(f"unknown artifact {key!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[key]


def artifact_path(key: str, *, ensure_parent: bool = False) -> Path:
    path = artifact(key).path
    if ensure_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def pipeline_status() -> list[ArtifactStatus]:
    """Status of every declared artifact, ordered by producing phase."""
    return [a.status() for a in sorted(_ARTIFACTS, key=lambda a: (a.phase, a.key))]


def missing_artifacts(*keys: str) -> list[Artifact]:
    """Which of the requested artifacts do not exist yet."""
    return [artifact(k) for k in keys if not artifact(k).exists()]


def figures_dir() -> Path:
    path = load_config().path("artifacts") / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def metrics_dir() -> Path:
    path = load_config().path("artifacts") / "metrics"
    path.mkdir(parents=True, exist_ok=True)
    return path


__all__ = [
    "Artifact",
    "ArtifactStatus",
    "REGISTRY",
    "artifact",
    "artifact_path",
    "pipeline_status",
    "missing_artifacts",
    "figures_dir",
    "metrics_dir",
    "resolve",
]
