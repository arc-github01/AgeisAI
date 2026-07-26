"""Streaming inference engine: ``process_event`` over persisted Phase 5–8 artifacts.

Does not retrain models or rewrite offline ``features.parquet``. Features use
the same formulas as Phase 4; IsolationForest, rule baseline, hybrid risk, and
attack classifier are the persisted Phase 5–7 objects; adaptive updates reuse
Phase 8's risk gate.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Mapping

import joblib
import pandas as pd

from src.artifacts import artifact_path
from src.config import load_config
from src.drift.store import AdaptiveProfileStore
from src.drift.update import maybe_update
from src.features import MODEL_FEATURE_COLUMNS, compute_event_features, empty_profile
from src.models.baselines import RuleBaseline, fit_rule_baseline
from src.models.classifier import AttackClassifier
from src.models.dataset import load_scoring_frame, training_matrix
from src.models.model import AnomalyModel
from src.profiling import BehaviourProfile, ProfileBundle
from src.risk.aggregation import EntityRiskState
from src.risk.engine import AlertState, RiskEngine
from src.risk.signals import FORBIDDEN_RISK_INPUT_COLUMNS
from src.schema import (
    FORBIDDEN_FEATURE_COLUMNS,
    IDENTITY_COLUMNS,
    OBSERVATION_COLUMNS,
    assert_no_label_leakage,
    validate_events,
)

from .result import EventResult

INFERENCE_COLUMNS: tuple[str, ...] = IDENTITY_COLUMNS + OBSERVATION_COLUMNS
FORBIDDEN_INFERENCE_COLUMNS: frozenset[str] = frozenset(FORBIDDEN_FEATURE_COLUMNS) | frozenset(
    FORBIDDEN_RISK_INPUT_COLUMNS
) | frozenset({"predicted_label", "attack_confidence", "stealthy", "difficulty"})


class StreamingOrderError(ValueError):
    """Raised when events arrive out of chronological order for an entity."""


@dataclass
class StreamingEngine:
    """Stateful near-real-time path reusing offline models and config."""

    anomaly_model: AnomalyModel
    rule_baseline: RuleBaseline
    classifier: AttackClassifier
    risk_engine: RiskEngine
    profile_bundle: ProfileBundle
    adaptive_store: AdaptiveProfileStore
    smoothing: float
    std_floor: float
    velocity_cap: float
    history_retention_hours: float
    apply_drift_updates: bool = True
    allow_out_of_order: bool = False
    # Per-entity causal state
    histories: dict[str, list[pd.Series]] = field(default_factory=dict)
    previous_fingerprint: dict[str, str | None] = field(default_factory=dict)
    risk_states: dict[str, EntityRiskState] = field(default_factory=dict)
    alert_states: dict[str, AlertState] = field(default_factory=dict)
    n_alerts_emitted: int = 0
    n_processed: int = 0

    @classmethod
    def load(
        cls,
        *,
        apply_drift_updates: bool | None = None,
        use_persisted_adaptive: bool = False,
    ) -> "StreamingEngine":
        """Load Phase 5–8 artifacts from disk."""
        cfg = load_config()
        if apply_drift_updates is None:
            apply_drift_updates = bool(
                cfg.get("detection.apply_drift_updates_default", True)
            )
        bundle: ProfileBundle = joblib.load(artifact_path("profiles"))
        anomaly = AnomalyModel.load()
        classifier = AttackClassifier.load()
        risk = RiskEngine.load()

        rule_path = artifact_path("rule_baseline")
        if rule_path.exists():
            rule = RuleBaseline.load()
        else:
            # Backward-compatible fallback for older Phase 5 artifact sets.
            scoring_frame = load_scoring_frame()
            rule = fit_rule_baseline(training_matrix(scoring_frame))

        drift_half = cfg.get("drift.ewma_halflife_days")
        if drift_half is None:
            drift_half = cfg["profiling.ewma_halflife_days"]
        if use_persisted_adaptive and artifact_path("adaptive_profiles").exists():
            adaptive = joblib.load(artifact_path("adaptive_profiles"))
        else:
            adaptive = AdaptiveProfileStore.from_bundle(
                bundle,
                ewma_halflife_days=float(drift_half),
                baseline_update_max_risk=float(cfg["drift.baseline_update_max_risk"]),
                rolling_window_days=float(cfg["drift.rolling_window_days"]),
            )

        return cls(
            anomaly_model=anomaly,
            rule_baseline=rule,
            classifier=classifier,
            risk_engine=risk,
            profile_bundle=bundle,
            adaptive_store=adaptive,
            smoothing=float(cfg.get("features.rarity_smoothing", 1.0)),
            std_floor=float(cfg.get("features.std_floor", 1.0)),
            velocity_cap=float(cfg.get("features.velocity_cap_kmh", 100000.0)),
            history_retention_hours=float(
                cfg.get("detection.history_retention_hours", 0.0)
            ),
            apply_drift_updates=bool(apply_drift_updates),
        )

    def reset_runtime_state(self) -> None:
        """Clear per-entity histories/risk/alert/adaptive state (keep models)."""
        self.histories.clear()
        self.previous_fingerprint.clear()
        self.risk_states.clear()
        self.alert_states.clear()
        self.n_alerts_emitted = 0
        self.n_processed = 0
        # Adaptive profile mass must also rewind; otherwise a second replay after
        # reset silently continues from poisoned/adapted counts.
        self.adaptive_store = AdaptiveProfileStore.from_bundle(
            self.profile_bundle,
            ewma_halflife_days=self.adaptive_store.ewma_halflife_days,
            baseline_update_max_risk=self.adaptive_store.baseline_update_max_risk,
            rolling_window_days=self.adaptive_store.rolling_window_days,
        )

    def _strip_forbidden(self, event: pd.Series) -> pd.Series:
        safe_cols = [c for c in INFERENCE_COLUMNS if c in event.index]
        cleaned = event.loc[safe_cols].copy()
        assert_no_label_leakage(cleaned.index)
        missing = [c for c in INFERENCE_COLUMNS if c not in cleaned.index]
        if missing:
            raise ValueError(f"event missing required inference columns: {missing}")
        return cleaned

    def _resolve_profile(
        self, event: pd.Series
    ) -> tuple[BehaviourProfile, str, float]:
        if self.apply_drift_updates:
            resolved, source, confidence = self.adaptive_store.resolve(
                str(event.entity_id), event
            )
            if resolved is None:
                return empty_profile(event), "none", 0.0
            if hasattr(resolved, "snapshot"):
                return resolved.snapshot(), source, confidence
            return resolved, source, confidence
        profile, source, confidence = self.profile_bundle.resolve(event)
        if profile is None:
            return empty_profile(event), "none", 0.0
        return profile, source, confidence

    def _trim_history(self, entity_id: str, now: pd.Timestamp) -> None:
        retention_s = self.history_retention_hours * 3600.0
        history = self.histories.get(entity_id, [])
        if not history or retention_s <= 0:
            return
        kept = [
            prior
            for prior in history
            if (now - pd.Timestamp(prior.timestamp)).total_seconds() <= retention_s
        ]
        self.histories[entity_id] = kept

    def process_event(self, event: Mapping[str, Any] | pd.Series) -> EventResult:
        """Run the full causal streaming path for one access event."""
        t0 = time.perf_counter()
        if isinstance(event, pd.Series):
            raw = event
        else:
            raw = pd.Series(dict(event))
        if not isinstance(raw.get("timestamp"), pd.Timestamp):
            raw = raw.copy()
            raw["timestamp"] = pd.Timestamp(raw["timestamp"])

        # Validate observation contract without requiring labels.
        validate_events(pd.DataFrame([raw]), require_labels=False)
        cleaned = self._strip_forbidden(raw)
        entity_id = str(cleaned.entity_id)
        ts = pd.Timestamp(cleaned.timestamp)

        history = self.histories.setdefault(entity_id, [])
        if history and not self.allow_out_of_order:
            last_ts = pd.Timestamp(history[-1].timestamp)
            if ts < last_ts or (
                ts == last_ts and str(cleaned.event_id) < str(history[-1].event_id)
            ):
                raise StreamingOrderError(
                    f"out-of-order event {cleaned.event_id} for entity {entity_id}: "
                    f"{ts} before {last_ts}"
                )

        profile, profile_source, profile_confidence = self._resolve_profile(cleaned)
        feature_dict = compute_event_features(
            cleaned,
            history=history,
            previous_fingerprint=self.previous_fingerprint.get(entity_id),
            profile=profile,
            profile_source=profile_source,
            profile_confidence=profile_confidence,
            smoothing=self.smoothing,
            std_floor=self.std_floor,
            velocity_cap=self.velocity_cap,
            # Live inference has no train/eval split; omit evaluation metadata.
            cutoff=None,
        )
        assert_no_label_leakage(feature_dict.keys())
        feature_frame = pd.DataFrame([feature_dict])
        # Attach detectors.
        anomaly_raw = float(self.anomaly_model.raw_scores(feature_frame)[0])
        score_span = self.anomaly_model.score_max - self.anomaly_model.score_min
        anomaly_disp = (
            0.0
            if score_span <= 0
            else min(
                1.0,
                max(
                    0.0,
                    (anomaly_raw - self.anomaly_model.score_min) / score_span,
                ),
            )
        )
        rule_score = float(self.rule_baseline.score(feature_frame)[0])

        pred = self.classifier.predict(feature_frame).iloc[0]
        predicted_label = str(pred["predicted_label"])
        attack_confidence = float(pred["attack_confidence"])

        risk_row = feature_frame.iloc[0].copy()
        risk_row["entity_type"] = cleaned.entity_type
        risk_row["anomaly_score_raw"] = anomaly_raw
        risk_row["baseline_rule"] = rule_score
        # Build a SimpleNamespace-like row for RiskEngine.score_event
        scoring = pd.DataFrame(
            [
                {
                    **{c: risk_row[c] for c in MODEL_FEATURE_COLUMNS if c in risk_row.index},
                    "event_id": cleaned.event_id,
                    "timestamp": ts,
                    "entity_id": entity_id,
                    "entity_type": cleaned.entity_type,
                    "anomaly_score_raw": anomaly_raw,
                    "baseline_rule": rule_score,
                }
            ]
        )
        # Ensure context features required by risk engine are present.
        for col in self.risk_engine.spec.context_features():
            if col not in scoring.columns:
                scoring[col] = float(feature_dict.get(col, 0.0))

        score_dict, alert_dict, new_risk, new_alert, suppressed = self.risk_engine.score_event(
            scoring.iloc[0],
            risk_state=self.risk_states.get(entity_id),
            alert_state=self.alert_states.get(entity_id),
            predicted_label=predicted_label,
            attack_confidence=attack_confidence,
            alert_sequence=self.n_alerts_emitted,
        )
        self.risk_states[entity_id] = new_risk
        self.alert_states[entity_id] = new_alert
        alerted = alert_dict is not None
        if alerted:
            self.n_alerts_emitted += 1

        profile_updated = False
        if self.apply_drift_updates:
            profile_updated = maybe_update(
                self.adaptive_store,
                cleaned,
                float(score_dict["risk_score"]),
                event_id=str(cleaned.event_id),
            )

        # Commit history AFTER scoring (causal).
        history.append(cleaned)
        self.previous_fingerprint[entity_id] = (
            f"{cleaned.device_id}|{cleaned.device_mac}"
        )
        self._trim_history(entity_id, ts)
        self.n_processed += 1
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return EventResult(
            event_id=str(cleaned.event_id),
            entity_id=entity_id,
            entity_type=str(cleaned.entity_type),
            timestamp=ts.isoformat(),
            anomaly_score_raw=anomaly_raw,
            anomaly_score=anomaly_disp,
            baseline_rule=rule_score,
            risk_score=float(score_dict["risk_score"]),
            severity=str(score_dict["risk_severity"]),
            predicted_attack_type=predicted_label,
            attack_confidence=attack_confidence,
            reason_codes=tuple(score_dict["reason_codes"]),
            reasons=tuple(score_dict["reasons"]),
            short_reason=str(score_dict["short_reason"]),
            alerted=alerted,
            alert_suppressed=bool(suppressed),
            alert=alert_dict,
            profile_updated=profile_updated,
            profile_source=profile_source,
            profile_confidence=float(profile_confidence),
            entity_evidence_state=float(score_dict["entity_evidence_state"]),
            latency_ms=latency_ms,
        )


def process_event(
    event: Mapping[str, Any] | pd.Series,
    *,
    engine: StreamingEngine | None = None,
) -> EventResult:
    """Public streaming entry point.

    If ``engine`` is omitted, a process-local default engine is loaded once and
    reused so repeated ``process_event(event)`` calls preserve entity state.
    Long-lived services should still own an explicit engine for lifecycle and
    persistence control.
    """
    eng = engine or get_default_engine()
    return eng.process_event(event)


@lru_cache(maxsize=1)
def get_default_engine() -> StreamingEngine:
    """Return the process-local stateful engine used by ``process_event``."""
    return StreamingEngine.load()


def reset_default_engine() -> None:
    """Discard the process-local engine, primarily for service restarts/tests."""
    get_default_engine.cache_clear()


__all__ = [
    "FORBIDDEN_INFERENCE_COLUMNS",
    "INFERENCE_COLUMNS",
    "StreamingEngine",
    "StreamingOrderError",
    "get_default_engine",
    "process_event",
    "reset_default_engine",
]
