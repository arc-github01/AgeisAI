"""Hybrid behavioural risk engine: score events, maintain entity state, emit alerts.

Consumes validated Phase 5 scores and Phase 4 features. Does not retrain the
IsolationForest, does not invent new supervised detectors, and never reads
evaluation metadata while scoring.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.artifacts import artifact_path
from src.config import load_config
from src.evaluation.report import RunManifest, write_json
from src.features import MODEL_FEATURE_COLUMNS
from src.models.dataset import training_matrix

from .aggregation import (
    EntityRiskState,
    decay_factor,
    load_severity_bands,
    saturating_risk,
    severity_for,
    severity_rank,
)
from .explanations import build_reasons, reasons_to_text, short_reason
from .signals import (
    RISK_DETECTOR_COLUMNS,
    RISK_IDENTITY_COLUMNS,
    Calibration,
    EngineSpec,
    assert_no_evaluation_metadata,
    context_activation,
    fit_calibration,
    isolation_forest_activation,
    isolation_forest_operating_point,
    load_engine_spec,
    risk_input_columns,
    rule_activation,
)


@dataclass
class AlertState:
    """Per-entity alert bookkeeping for cooldown / escalation."""

    last_alert_ts: float | None = None
    last_severity: str | None = None


@dataclass(frozen=True)
class RiskEngine:
    """Fitted hybrid risk engine ready for causal streaming scoring."""

    spec: EngineSpec
    calibration: Calibration
    severity_bands: dict[str, tuple[float, float]]
    min_alert_severity: str
    cooldown_seconds: float
    escalation_bypasses_cooldown: bool
    max_reasons: int

    def score_event(
        self,
        row: Any,
        *,
        risk_state: EntityRiskState | None = None,
        alert_state: AlertState | None = None,
        predicted_label: str = "UNKNOWN",
        attack_confidence: float = 0.0,
        alert_sequence: int = 0,
    ) -> tuple[dict[str, Any], dict[str, Any] | None, EntityRiskState, AlertState, bool]:
        """Score one event with causal risk/alert state.

        Classifier outputs name the alert only — they never enter the risk score.
        Returns ``(score_row, alert_row_or_None, new_risk_state, new_alert_state,
        was_suppressed)``.
        """
        state = risk_state or EntityRiskState()
        astate = alert_state or AlertState()
        entity_id = str(row.entity_id)
        ts = pd.Timestamp(row.timestamp)
        ts_seconds = float(ts.value) / 1e9

        elapsed = 0.0 if state.last_timestamp is None else ts_seconds - state.last_timestamp
        decayed = state.evidence * decay_factor(elapsed, self.spec.halflife_seconds)

        feature_row = {
            feature: float(getattr(row, feature))
            for feature in self.spec.context_features()
        }
        if_score = float(row.anomaly_score_raw)
        rule_score = float(row.baseline_rule)

        if_act = isolation_forest_activation(if_score, self.calibration)
        rule_act = rule_activation(rule_score, self.calibration)
        context_acts: dict[str, float] = {}
        for signal in self.spec.context:
            context_acts[signal.name] = context_activation(
                signal, feature_row, self.calibration
            )

        if_evidence = self.spec.isolation_forest_weight * if_act
        rule_evidence = self.spec.rule_weight * rule_act
        context_evidence_by_reason: dict[str, float] = {
            signal.reason: signal.weight * context_acts[signal.name]
            for signal in self.spec.context
        }
        context_evidence = float(sum(context_evidence_by_reason.values()))
        instantaneous = if_evidence + rule_evidence + context_evidence
        persistence_evidence = self.spec.persistence_weight * decayed
        total_evidence = instantaneous + persistence_evidence
        risk_score = saturating_risk(total_evidence, self.spec.evidence_scale)
        severity = severity_for(risk_score, self.severity_bands)

        evidence_terms = {
            self.spec.isolation_forest_reason: if_evidence,
            self.spec.rule_reason: rule_evidence,
            **context_evidence_by_reason,
            self.spec.persistence_reason: persistence_evidence,
        }
        if total_evidence > 0:
            scale = risk_score / total_evidence
            contributions = {k: v * scale for k, v in evidence_terms.items()}
        else:
            contributions = {k: 0.0 for k in evidence_terms}

        reasons = build_reasons(contributions, max_reasons=self.max_reasons)
        new_evidence = min(decayed + instantaneous, self.spec.state_cap)
        score_row = {
            "event_id": row.event_id,
            "timestamp": ts,
            "entity_id": entity_id,
            "entity_type": row.entity_type,
            "risk_score": risk_score,
            "risk_severity": severity,
            "isolation_forest_contribution": contributions[
                self.spec.isolation_forest_reason
            ],
            "rule_contribution": contributions[self.spec.rule_reason],
            "context_contribution": float(
                sum(contributions[r] for r in context_evidence_by_reason)
            ),
            "persistence_contribution": contributions[self.spec.persistence_reason],
            "isolation_forest_operating_point": isolation_forest_operating_point(
                if_score, self.calibration
            ),
            "instantaneous_evidence": instantaneous,
            "total_evidence": total_evidence,
            "entity_evidence_state": new_evidence,
            "reason_codes": [item["code"] for item in reasons],
            "reasons": reasons,
            "short_reason": short_reason(reasons),
        }
        new_risk = EntityRiskState(evidence=new_evidence, last_timestamp=ts_seconds)
        new_alert = astate
        alert_row: dict[str, Any] | None = None
        was_suppressed = False

        if severity_rank(severity) >= severity_rank(self.min_alert_severity):
            emit, was_suppressed = self._should_alert(astate, severity, ts_seconds)
            if emit:
                alert_row = {
                    "alert_id": (
                        f"ALT-{entity_id}-{ts.strftime('%Y%m%dT%H%M%S')}-"
                        f"{alert_sequence:05d}"
                    ),
                    "timestamp": ts,
                    "entity_id": entity_id,
                    "entity_type": row.entity_type,
                    "event_id": row.event_id,
                    "risk_score": risk_score,
                    "severity": severity,
                    "reason_codes": [item["code"] for item in reasons],
                    "top_contributors": reasons,
                    "short_reason": short_reason(reasons),
                    "reasons": reasons_to_text(reasons),
                    "anomaly_score": if_act,
                    "sequence_score": context_acts.get(
                        "rare_sequence_transition", 0.0
                    ),
                    "attack_type": predicted_label,
                    "attack_confidence": float(attack_confidence),
                }
                new_alert = AlertState(
                    last_alert_ts=ts_seconds, last_severity=severity
                )

        return score_row, alert_row, new_risk, new_alert, was_suppressed

    def score_frame(
        self,
        frame: pd.DataFrame,
        *,
        classifications: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Score every event causally; return ``(risk_scores, alerts)``.

        ``frame`` must contain the risk-input whitelist columns and be free of
        evaluation metadata. Rows are processed in stable timestamp order so the
        same stream always produces the same state trajectory.

        Optional ``classifications`` (Phase 7) supply ``attack_type`` /
        ``attack_confidence`` on alerts only — they never enter the risk score.
        """
        required = risk_input_columns(self.spec)
        missing = [c for c in required if c not in frame.columns]
        if missing:
            raise ValueError(f"risk scoring frame missing columns: {missing}")
        assert_no_evaluation_metadata(frame.columns)

        class_lookup: dict[str, tuple[str, float]] = {}
        if classifications is not None and not classifications.empty:
            for rec in classifications.loc[
                :, ["event_id", "predicted_label", "attack_confidence"]
            ].itertuples(index=False):
                class_lookup[str(rec.event_id)] = (
                    str(rec.predicted_label),
                    float(rec.attack_confidence),
                )

        ordered = frame.sort_values(["timestamp", "event_id"], kind="stable").reset_index(
            drop=True
        )
        states: dict[str, EntityRiskState] = {}
        alert_states: dict[str, AlertState] = {}
        score_rows: list[dict[str, Any]] = []
        alert_rows: list[dict[str, Any]] = []
        n_suppressed = 0

        for row in ordered.itertuples(index=False):
            entity_id = str(row.entity_id)
            predicted = class_lookup.get(str(row.event_id), ("UNKNOWN", 0.0))
            score_row, alert_row, new_risk, new_alert, suppressed = self.score_event(
                row,
                risk_state=states.get(entity_id),
                alert_state=alert_states.get(entity_id),
                predicted_label=predicted[0],
                attack_confidence=predicted[1],
                alert_sequence=len(alert_rows),
            )
            score_rows.append(score_row)
            states[entity_id] = new_risk
            alert_states[entity_id] = new_alert
            if suppressed:
                n_suppressed += 1
            if alert_row is not None:
                alert_rows.append(alert_row)

        scores = pd.DataFrame(score_rows)
        alerts = pd.DataFrame(alert_rows)
        if not alerts.empty:
            alerts.attrs["n_suppressed"] = n_suppressed
        else:
            alerts = pd.DataFrame(
                columns=[
                    "alert_id",
                    "timestamp",
                    "entity_id",
                    "entity_type",
                    "event_id",
                    "risk_score",
                    "severity",
                    "reason_codes",
                    "top_contributors",
                    "short_reason",
                    "reasons",
                    "anomaly_score",
                    "sequence_score",
                    "attack_type",
                    "attack_confidence",
                ]
            )
            alerts.attrs["n_suppressed"] = n_suppressed
        return scores, alerts

    def _should_alert(
        self, state: AlertState, severity: str, ts_seconds: float
    ) -> tuple[bool, bool]:
        """Return ``(emit, was_suppressed)`` under cooldown / escalation rules."""
        if state.last_alert_ts is None:
            return True, False
        elapsed = ts_seconds - state.last_alert_ts
        in_cooldown = elapsed < self.cooldown_seconds
        escalated = (
            self.escalation_bypasses_cooldown
            and state.last_severity is not None
            and severity_rank(severity) > severity_rank(state.last_severity)
        )
        if in_cooldown and not escalated:
            return False, True
        return True, False

    def save(self) -> Path:
        path = artifact_path("risk_calibration", ensure_parent=True)
        payload = {
            "engine_version": self.spec.version,
            "calibration": self.calibration.to_dict(),
            "spec": {
                "evidence_scale": self.spec.evidence_scale,
                "halflife_seconds": self.spec.halflife_seconds,
                "persistence_weight": self.spec.persistence_weight,
                "state_cap": self.spec.state_cap,
                "isolation_forest_weight": self.spec.isolation_forest_weight,
                "rule_weight": self.spec.rule_weight,
                "context": [asdict(s) for s in self.spec.context],
            },
            "severity_bands": self.severity_bands,
            "alerting": {
                "min_severity": self.min_alert_severity,
                "cooldown_seconds": self.cooldown_seconds,
                "escalation_bypasses_cooldown": self.escalation_bypasses_cooldown,
                "max_reasons": self.max_reasons,
            },
            "manifest": asdict(RunManifest.capture(notes="phase6 risk engine calibration")),
        }
        write_json(payload, path)
        # Also persist a joblib snapshot so streaming reload is trivial.
        joblib.dump(self, path.with_suffix(".joblib"))
        return path

    @staticmethod
    def load() -> "RiskEngine":
        return joblib.load(artifact_path("risk_calibration").with_suffix(".joblib"))


def build_scoring_frame(
    features: pd.DataFrame | None = None,
    event_scores: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join features + Phase 5 scores and strip evaluation metadata for scoring."""
    if features is None:
        features = pd.read_parquet(artifact_path("features"))
    if event_scores is None:
        event_scores = pd.read_parquet(artifact_path("event_scores"))

    detector = event_scores.loc[:, ["event_id", *RISK_DETECTOR_COLUMNS]]
    # Features carry identity + MODEL_FEATURE_COLUMNS; scores carry detector outputs.
    # Evaluation columns that may ride on event_scores must not enter the join result
    # used for scoring — we deliberately select only the risk whitelist.
    identity_from_features = [c for c in RISK_IDENTITY_COLUMNS if c in features.columns]
    context_cols = [c for c in MODEL_FEATURE_COLUMNS if c in features.columns]
    base = features.loc[:, identity_from_features + context_cols]
    merged = base.merge(detector, on="event_id", how="inner", validate="one_to_one")

    # entity_type may only live on event_scores in some pipelines; backfill if needed.
    if "entity_type" not in merged.columns and "entity_type" in event_scores.columns:
        merged = merged.merge(
            event_scores.loc[:, ["event_id", "entity_type"]],
            on="event_id",
            how="left",
            validate="one_to_one",
        )

    required = risk_input_columns()
    missing = [c for c in required if c not in merged.columns]
    if missing:
        raise ValueError(f"unable to assemble risk scoring frame; missing {missing}")
    scoring = merged.loc[:, list(required)].copy()
    assert_no_evaluation_metadata(scoring.columns)
    return scoring.sort_values(["timestamp", "event_id"], kind="stable").reset_index(
        drop=True
    )


def fit_risk_engine(
    scoring_frame: pd.DataFrame,
    event_scores_with_labels: pd.DataFrame,
) -> RiskEngine:
    """Fit activation anchors on benign training rows; never use evaluation labels.

    Labels are read from ``event_scores_with_labels`` solely to *select* the
    benign training rows that calibrate activations — the same discipline Phase 5
    uses for IsolationForest training. Those labels never enter the scoring frame.
    """
    spec = load_engine_spec()
    cfg = load_config()
    # Join labels only for row selection, then immediately drop them.
    selection = scoring_frame.merge(
        event_scores_with_labels.loc[:, ["event_id", "split", "label"]],
        on="event_id",
        how="left",
        validate="one_to_one",
    )
    benign_train = training_matrix(selection)
    # Drop selection metadata before any further use of the frame for scoring.
    thresholds_path = artifact_path("anomaly_thresholds")
    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))[
        "thresholds_raw_score"
    ]
    calibration = fit_calibration(benign_train, thresholds, spec=spec)

    alerting = cfg["risk.engine.alerting"]
    return RiskEngine(
        spec=spec,
        calibration=calibration,
        severity_bands=load_severity_bands(cfg),
        min_alert_severity=str(alerting["min_severity"]),
        cooldown_seconds=float(alerting["cooldown_seconds"]),
        escalation_bypasses_cooldown=bool(alerting["escalation_bypasses_cooldown"]),
        max_reasons=int(alerting["max_reasons"]),
    )


def run() -> dict[str, Path]:
    """Fit, score, persist risk artifacts (evaluation is a separate step)."""
    event_scores = pd.read_parquet(artifact_path("event_scores"))
    scoring_frame = build_scoring_frame(event_scores=event_scores)
    engine = fit_risk_engine(scoring_frame, event_scores)
    classifications = None
    class_path = artifact_path("classifications")
    if class_path.exists():
        classifications = pd.read_parquet(class_path)
    risk_scores, alerts = engine.score_frame(
        scoring_frame, classifications=classifications
    )

    cal_path = engine.save()
    scores_path = artifact_path("risk_scores", ensure_parent=True)
    # Persist reasons as JSON strings for parquet compatibility.
    persist = risk_scores.copy()
    persist["reason_codes"] = persist["reason_codes"].map(json.dumps)
    persist["reasons"] = persist["reasons"].map(json.dumps)
    persist.to_parquet(scores_path, index=False)

    alerts_path = artifact_path("alerts", ensure_parent=True)
    alerts_out = alerts.copy()
    if not alerts_out.empty:
        alerts_out["reason_codes"] = alerts_out["reason_codes"].map(json.dumps)
        alerts_out["top_contributors"] = alerts_out["top_contributors"].map(json.dumps)
    alerts_out.to_parquet(alerts_path, index=False)

    # Persist operational bookkeeping that parquet attrs would lose on reload.
    ops_path = alerts_path.with_name("alert_ops.json")
    write_json(
        {"n_suppressed": int(getattr(alerts, "attrs", {}).get("n_suppressed", 0))},
        ops_path,
    )

    return {
        "risk_calibration": cal_path,
        "risk_scores": scores_path,
        "alerts": alerts_path,
    }


__all__ = [
    "AlertState",
    "RiskEngine",
    "build_scoring_frame",
    "fit_risk_engine",
    "run",
]
