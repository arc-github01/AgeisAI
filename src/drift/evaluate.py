"""Concept-drift evaluation: adaptation vs poisoning resistance.

Metrics join labels and campaign metadata *after* adaptive replay so the gate
itself never saw them. Headline numbers:

- share of high-risk / malicious events blocked from baseline updates
- share of low-risk benign (and INSIDER_DRIFT) events absorbed
- per-entity resource/location distribution shift for INSIDER_DRIFT exhibits
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.artifacts import artifact_path
from src.evaluation.report import RunManifest, write_json
from src.schema import MALICIOUS_CLASSES, AttackType

from .store import AdaptiveProfileStore


def _share(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def evaluate_drift(
    store: AdaptiveProfileStore,
    events: pd.DataFrame,
    risk_scores: pd.DataFrame,
    *,
    max_risk: float,
) -> dict[str, Any]:
    """Build an evaluation document from a completed causal replay."""
    risk = risk_scores.loc[:, ["event_id", "risk_score"]].copy()
    frame = events.merge(risk, on="event_id", how="inner", validate="one_to_one")
    if len(frame) != len(events):
        raise ValueError("events/risk_scores event_id mismatch during drift evaluation")

    blocked = set(store.blocked_event_ids)
    updated = set(store.updated_event_ids)

    frame = frame.copy()
    frame["was_blocked"] = frame["event_id"].astype(str).isin(blocked)
    frame["was_updated"] = frame["event_id"].astype(str).isin(updated)
    # Consistency: gate decision must match risk threshold (labels unused).
    expected_block = frame["risk_score"] >= max_risk
    if not bool((frame["was_blocked"] == expected_block).all()):
        raise AssertionError("drift gate decisions disagree with risk threshold")

    malicious = frame["label"].isin(list(MALICIOUS_CLASSES))
    insider = frame["label"] == AttackType.INSIDER_DRIFT.value
    benign = frame["label"] == AttackType.BENIGN.value
    high_risk = frame["risk_score"] >= max_risk
    low_risk = ~high_risk

    doc: dict[str, Any] = {
        "gate": {
            "baseline_update_max_risk": float(max_risk),
            "ewma_halflife_days": float(store.ewma_halflife_days),
            "rolling_window_days": float(store.rolling_window_days),
            "n_considered": int(store.n_considered),
            "n_updated": int(store.n_updated),
            "n_blocked": int(store.n_blocked),
            "update_rate": _share(store.n_updated, store.n_considered),
            "block_rate": _share(store.n_blocked, store.n_considered),
        },
        "poisoning_resistance": {
            "high_risk_events": int(high_risk.sum()),
            "high_risk_blocked": int((high_risk & frame["was_blocked"]).sum()),
            "high_risk_block_rate": _share(
                int((high_risk & frame["was_blocked"]).sum()), int(high_risk.sum())
            ),
            "malicious_events": int(malicious.sum()),
            "malicious_blocked": int((malicious & frame["was_blocked"]).sum()),
            "malicious_block_rate": _share(
                int((malicious & frame["was_blocked"]).sum()), int(malicious.sum())
            ),
            "malicious_high_risk_block_rate": _share(
                int((malicious & high_risk & frame["was_blocked"]).sum()),
                int((malicious & high_risk).sum()),
            ),
            "note": (
                "Gate uses risk_score only. Malicious events that score below "
                "baseline_update_max_risk can still update (honest stealth gap)."
            ),
        },
        "adaptation": {
            "low_risk_benign": int((benign & low_risk).sum()),
            "low_risk_benign_updated": int(
                (benign & low_risk & frame["was_updated"]).sum()
            ),
            "low_risk_benign_update_rate": _share(
                int((benign & low_risk & frame["was_updated"]).sum()),
                int((benign & low_risk).sum()),
            ),
            "insider_drift_events": int(insider.sum()),
            "insider_drift_updated": int((insider & frame["was_updated"]).sum()),
            "insider_drift_blocked": int((insider & frame["was_blocked"]).sum()),
            "insider_drift_update_rate": _share(
                int((insider & frame["was_updated"]).sum()), int(insider.sum())
            ),
            "insider_drift_low_risk_update_rate": _share(
                int((insider & low_risk & frame["was_updated"]).sum()),
                int((insider & low_risk).sum()),
            ),
        },
        "labels_never_gate": True,
        "manifest": asdict(
            RunManifest.capture(notes="Phase 8 concept-drift evaluation")
        ),
    }

    exhibits: list[dict[str, Any]] = []
    if insider.any():
        for entity_id, group in frame.loc[insider].groupby("entity_id", sort=False):
            adaptive = store.entity_profiles.get(str(entity_id))
            if adaptive is None:
                continue
            top_resources = sorted(
                adaptive.resource_counts.items(), key=lambda kv: -kv[1]
            )[:5]
            exhibits.append(
                {
                    "entity_id": str(entity_id),
                    "n_insider_events": int(len(group)),
                    "n_updated": int(group["was_updated"].sum()),
                    "n_blocked": int(group["was_blocked"].sum()),
                    "adaptive_top_resources": [
                        {"resource": k, "ewma_count": float(v)} for k, v in top_resources
                    ],
                    "mean_risk": float(group["risk_score"].mean()),
                }
            )
    doc["insider_drift_exhibits"] = exhibits
    return doc


def save_evaluation(doc: dict[str, Any]) -> Path:
    path = artifact_path("drift_evaluation", ensure_parent=True)
    write_json(doc, path)
    return path


def dump_store(store: AdaptiveProfileStore, path: Path | None = None) -> Path:
    path = path or artifact_path("adaptive_profiles", ensure_parent=True)
    # Drop per-event audit lists from the joblib (evaluation JSON has aggregates).
    slim = AdaptiveProfileStore(
        entity_profiles=store.entity_profiles,
        cohort_profiles=store.cohort_profiles,
        cohort_keys=store.cohort_keys,
        min_events_for_personal=store.min_events_for_personal,
        ewma_halflife_days=store.ewma_halflife_days,
        baseline_update_max_risk=store.baseline_update_max_risk,
        rolling_window_days=store.rolling_window_days,
        n_considered=store.n_considered,
        n_updated=store.n_updated,
        n_blocked=store.n_blocked,
        blocked_event_ids=[],
        updated_event_ids=[],
    )
    joblib.dump(slim, path)
    return path


__all__ = ["dump_store", "evaluate_drift", "save_evaluation"]
