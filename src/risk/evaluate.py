"""Independent evaluation of the hybrid risk engine.

Labels and campaign metadata are joined *after* scoring. This module never
feeds ground truth back into the engine — it only measures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.artifacts import artifact_path
from src.config import load_config
from src.evaluation.metrics import detection_metrics, ranking_metrics
from src.evaluation.report import RunManifest, write_json
from src.models.evaluate import campaign_level_metrics
from src.schema import MALICIOUS_CLASSES


def _binary_metrics(frame: pd.DataFrame, score_col: str, *, threshold: float) -> dict[str, Any]:
    y_true = frame["is_attack"].to_numpy(dtype=int)
    scores = frame[score_col].to_numpy(dtype=float)
    pr_auc, roc_auc = ranking_metrics(y_true, scores)
    point = detection_metrics(y_true, scores, threshold=threshold)
    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "precision": point.precision,
        "recall": point.recall,
        "f1": point.f1,
        "fpr": point.fpr,
        "threshold": threshold,
        "n_events": int(len(frame)),
        "n_attacks": int(y_true.sum()),
    }


def _threshold_at_benign_fpr(
    train_benign_scores: np.ndarray, target_fpr: float
) -> float:
    """Operating threshold from benign training scores only (fair comparison)."""
    return float(np.quantile(train_benign_scores, 1.0 - target_fpr))


def operational_alert_metrics(
    alerts: pd.DataFrame,
    eval_scores: pd.DataFrame,
    *,
    n_suppressed: int,
) -> dict[str, Any]:
    eval_benign = eval_scores[eval_scores["label"] == "BENIGN"]
    n_benign = max(1, len(eval_benign))
    if alerts.empty:
        return {
            "n_alerts": 0,
            "n_high": 0,
            "n_critical": 0,
            "alerts_per_1000_benign": 0.0,
            "unique_entities_alerted": 0,
            "duplicate_alerts_suppressed": int(n_suppressed),
        }
    # Restrict operational burden to the evaluation window.
    eval_entity_events = set(eval_scores["event_id"])
    eval_alerts = alerts[alerts["event_id"].isin(eval_entity_events)] if "event_id" in alerts.columns else alerts
    return {
        "n_alerts": int(len(eval_alerts)),
        "n_high": int((eval_alerts["severity"] == "HIGH").sum()),
        "n_critical": int((eval_alerts["severity"] == "CRITICAL").sum()),
        "alerts_per_1000_benign": float(1000.0 * len(eval_alerts) / n_benign),
        "unique_entities_alerted": int(eval_alerts["entity_id"].nunique()),
        "duplicate_alerts_suppressed": int(n_suppressed),
    }


def method_overlap(
    eval_frame: pd.DataFrame,
    full_timeline: pd.DataFrame,
    *,
    if_mask: np.ndarray,
    rule_mask: np.ndarray,
    hybrid_mask: np.ndarray,
    campaigns_meta: list[dict[str, Any]],
) -> dict[str, Any]:
    """Campaign-level detection overlap across IF / rule / hybrid."""
    if_cam = campaign_level_metrics(
        eval_frame, full_timeline, if_mask, campaigns_meta, operating_point="if_balanced"
    )
    rule_cam = campaign_level_metrics(
        eval_frame, full_timeline, rule_mask, campaigns_meta, operating_point="rule_balanced"
    )
    hybrid_cam = campaign_level_metrics(
        eval_frame, full_timeline, hybrid_mask, campaigns_meta, operating_point="hybrid_high"
    )

    def detected_set(metrics: dict[str, Any]) -> set[str]:
        per = metrics.get("per_campaign", [])
        return {
            row["campaign_id"]
            for row in per
            if row.get("label") in MALICIOUS_CLASSES and row.get("detected")
        }

    if_set = detected_set(if_cam)
    rule_set = detected_set(rule_cam)
    hybrid_set = detected_set(hybrid_cam)
    all_malicious = {
        row["campaign_id"]
        for row in hybrid_cam.get("per_campaign", [])
        if row.get("label") in MALICIOUS_CLASSES
    }

    return {
        "isolation_forest": if_cam["malicious"] if "malicious" in if_cam else if_cam,
        "rule": rule_cam["malicious"] if "malicious" in rule_cam else rule_cam,
        "hybrid": hybrid_cam["malicious"] if "malicious" in hybrid_cam else hybrid_cam,
        "hybrid_full": hybrid_cam,
        "overlap": {
            "if_only": sorted(if_set - rule_set - hybrid_set),
            "rule_only": sorted(rule_set - if_set - hybrid_set),
            "hybrid_only": sorted(hybrid_set - if_set - rule_set),
            "if_and_rule_not_hybrid": sorted((if_set & rule_set) - hybrid_set),
            "multiple_methods": sorted(
                (if_set & rule_set)
                | (if_set & hybrid_set)
                | (rule_set & hybrid_set)
            ),
            "none": sorted(all_malicious - if_set - rule_set - hybrid_set),
            "all_three": sorted(if_set & rule_set & hybrid_set),
        },
        "by_attack_type_hybrid": hybrid_cam.get("by_attack_type_malicious", {}),
        "obvious_malicious_detection_rate": hybrid_cam.get(
            "obvious_malicious_detection_rate"
        ),
        "stealth_malicious_detection_rate": hybrid_cam.get(
            "stealth_malicious_detection_rate"
        ),
    }


def hard_category_analysis(
    eval_frame: pd.DataFrame,
    *,
    if_mask: np.ndarray,
    rule_mask: np.ndarray,
    hybrid_mask: np.ndarray,
) -> dict[str, Any]:
    """Per-label catch rates for the categories Phase 5 struggled with."""
    focus = ("DEVICE_SPOOFING", "IMPOSSIBLE_TRAVEL", "LATERAL_MOVEMENT")
    out: dict[str, Any] = {}
    for label in focus:
        rows = eval_frame["label"] == label
        n = int(rows.sum())
        if n == 0:
            out[label] = {"n_events": 0}
            continue
        out[label] = {
            "n_events": n,
            "if_recall": float(if_mask[rows].mean()),
            "rule_recall": float(rule_mask[rows].mean()),
            "hybrid_recall": float(hybrid_mask[rows].mean()),
        }
    return out


def evaluate_risk(
    risk_scores: pd.DataFrame,
    event_scores: pd.DataFrame,
    alerts: pd.DataFrame,
    *,
    n_suppressed: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Join labels after scoring and produce event + campaign metric documents."""
    cfg = load_config()
    # Labels joined AFTER scoring — never present during engine.score_frame.
    labelled = risk_scores.merge(
        event_scores.loc[:, ["event_id", "split", "label", "is_attack", "campaign_id",
                             "anomaly_score_raw", "baseline_rule"]],
        on="event_id",
        how="left",
        validate="one_to_one",
    )
    if labelled["label"].isna().any():
        raise ValueError("risk scores failed to join evaluation labels")

    eval_frame = labelled[labelled["split"] == "evaluation"].reset_index(drop=True)
    train_benign = labelled[
        (labelled["split"] == "train") & (labelled["label"] == "BENIGN")
    ]

    # Fair comparison: all three methods use a ~1% benign-train FPR threshold,
    # matching the Phase 5 "balanced" operating-point policy. Hybrid also reports
    # its severity-band operating point (HIGH) separately as the deployed view.
    target_fpr = 1.0 - float(
        cfg["models.anomaly_detector.calibration.operating_points.balanced"]
    )
    if_thr = _threshold_at_benign_fpr(
        train_benign["anomaly_score_raw"].to_numpy(float), target_fpr
    )
    rule_thr = _threshold_at_benign_fpr(
        train_benign["baseline_rule"].to_numpy(float), target_fpr
    )
    hybrid_thr = _threshold_at_benign_fpr(
        train_benign["risk_score"].to_numpy(float), target_fpr
    )
    # Deployed hybrid alert gate follows risk.engine.alerting.min_severity.
    min_sev = str(cfg["risk.engine.alerting.min_severity"])
    deployed_threshold = float(cfg[f"risk.severity_bands.{min_sev}"][0])
    high_band_low = float(cfg["risk.severity_bands.HIGH"][0])

    event_metrics = {
        "isolation_forest": _binary_metrics(
            eval_frame, "anomaly_score_raw", threshold=if_thr
        ),
        "rule": _binary_metrics(eval_frame, "baseline_rule", threshold=rule_thr),
        "hybrid_fpr_matched": _binary_metrics(
            eval_frame, "risk_score", threshold=hybrid_thr
        ),
        "hybrid_severity_high": _binary_metrics(
            eval_frame, "risk_score", threshold=high_band_low
        ),
        "hybrid_deployed": _binary_metrics(
            eval_frame, "risk_score", threshold=deployed_threshold
        ),
        "comparison_policy": {
            "fpr_matched_target": target_fpr,
            "hybrid_deployed_threshold": deployed_threshold,
            "hybrid_deployed_gate": f"severity>={min_sev}",
            "hybrid_watch_gate": "severity>=HIGH",
        },
    }

    if_mask = eval_frame["anomaly_score_raw"].to_numpy(float) >= if_thr
    rule_mask = eval_frame["baseline_rule"].to_numpy(float) >= rule_thr
    hybrid_mask = eval_frame["risk_score"].to_numpy(float) >= deployed_threshold

    campaigns_meta = []
    camp_path = artifact_path("campaign_metadata")
    if camp_path.exists():
        campaigns_meta = json.loads(camp_path.read_text(encoding="utf-8"))

    overlap = method_overlap(
        eval_frame,
        labelled,
        if_mask=if_mask,
        rule_mask=rule_mask,
        hybrid_mask=hybrid_mask,
        campaigns_meta=campaigns_meta,
    )
    hard = hard_category_analysis(
        eval_frame, if_mask=if_mask, rule_mask=rule_mask, hybrid_mask=hybrid_mask
    )
    ops = operational_alert_metrics(alerts, eval_frame, n_suppressed=n_suppressed)

    event_doc = {
        "manifest": asdict_safe(RunManifest.capture(notes="phase6 risk evaluation")),
        "metrics": {
            "event_level": event_metrics,
            "operational": ops,
            "hard_categories": hard,
        },
    }
    campaign_doc = {
        "manifest": event_doc["manifest"],
        "metrics": overlap,
    }
    return event_doc, campaign_doc


def asdict_safe(manifest: RunManifest) -> dict[str, Any]:
    from dataclasses import asdict

    return asdict(manifest)


def save_evaluation(
    event_doc: dict[str, Any], campaign_doc: dict[str, Any]
) -> dict[str, Path]:
    event_path = write_json(
        event_doc, artifact_path("risk_evaluation", ensure_parent=True)
    )
    campaign_path = write_json(
        campaign_doc, artifact_path("risk_campaign_metrics", ensure_parent=True)
    )
    return {"risk_evaluation": event_path, "risk_campaign_metrics": campaign_path}


__all__ = [
    "evaluate_risk",
    "hard_category_analysis",
    "method_overlap",
    "operational_alert_metrics",
    "save_evaluation",
]
