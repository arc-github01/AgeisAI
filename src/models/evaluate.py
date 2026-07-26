"""Event-, security- and campaign-level evaluation for the anomaly detector.

Everything here consumes ground-truth labels for *measurement only*. Nothing in
this module influences the model, its preprocessing or its shipped thresholds.
Ranking metrics use the raw monotonic score; recall-at-FPR curves derive their
own thresholds from evaluation benign scores, which is a reporting device, not a
model-selection one.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    budget_sweep,
    campaign_detection,
    detection_metrics,
    ranking_metrics,
)
from src.schema import ATTACK_CLASSES, AttackType, MALICIOUS_CLASSES

_SCORE_QUANTILES = (0.5, 0.9, 0.95, 0.99, 1.0)


def _distribution(scores: np.ndarray) -> dict[str, float]:
    if scores.size == 0:
        return {"n": 0}
    summary: dict[str, float] = {
        "n": int(scores.size),
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "min": float(scores.min()),
    }
    for q in _SCORE_QUANTILES:
        summary[f"q{int(q * 100)}"] = float(np.quantile(scores, q))
    return summary


def recall_at_fpr(
    y_true: np.ndarray, scores: np.ndarray, fpr_target: float
) -> dict[str, float]:
    """Recall achievable while holding the benign false-positive rate at target.

    The threshold is read off benign evaluation scores. This uses labels to
    *measure*, never to pick an operating point the model ships with.
    """
    benign = scores[y_true == 0]
    attacks = scores[y_true == 1]
    if benign.size == 0 or attacks.size == 0:
        return {"fpr_target": fpr_target, "threshold": float("nan"), "recall": float("nan")}
    threshold = float(np.quantile(benign, 1.0 - fpr_target))
    return {
        "fpr_target": fpr_target,
        "threshold": threshold,
        "recall": float((attacks >= threshold).mean()),
        "achieved_fpr": float((benign >= threshold).mean()),
    }


def _operating_point_metrics(
    y_true: np.ndarray, scores: np.ndarray, thresholds: dict[str, float]
) -> dict[str, Any]:
    points: dict[str, Any] = {}
    for name, threshold in thresholds.items():
        points[name] = detection_metrics(
            y_true, scores, threshold=threshold
        ).to_dict()
    return points


def event_level_metrics(
    frame: pd.DataFrame,
    score_col: str,
    *,
    thresholds: dict[str, float],
    budget_fraction: float,
    fpr_targets: Sequence[float],
) -> dict[str, Any]:
    """PR-AUC (primary), ROC-AUC, operating points, budget + FPR sweeps."""
    y_true = frame["is_attack"].to_numpy(dtype=int)
    scores = frame[score_col].to_numpy(dtype=float)
    pr_auc, roc_auc = ranking_metrics(y_true, scores)
    return {
        "n_events": int(len(frame)),
        "n_attacks": int(y_true.sum()),
        "prevalence": float(y_true.mean()),
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "budget": detection_metrics(
            y_true, scores, budget_fraction=budget_fraction
        ).to_dict(),
        "operating_points": _operating_point_metrics(y_true, scores, thresholds),
        "recall_at_fpr": [recall_at_fpr(y_true, scores, f) for f in fpr_targets],
        "budget_sweep": budget_sweep(y_true, scores).to_dict("records"),
        "score_distribution": {
            "benign": _distribution(scores[y_true == 0]),
            "attack": _distribution(scores[y_true == 1]),
        },
    }


def per_attack_type_metrics(
    frame: pd.DataFrame, score_col: str, *, budget_fraction: float
) -> dict[str, Any]:
    """PR-AUC and recall for each attack type, benign vs that type alone.

    INSIDER_DRIFT is reported too, but as an edge case: it is not in
    ``is_attack`` and a detection of it is a false positive, so its row measures
    sensitivity, not catch rate.
    """
    benign = frame[frame["label"] == AttackType.BENIGN.value]
    result: dict[str, Any] = {}
    for attack in ATTACK_CLASSES:
        attack_rows = frame[frame["label"] == attack]
        if attack_rows.empty:
            result[attack] = {"n": 0}
            continue
        subset = pd.concat([benign, attack_rows], ignore_index=True)
        y_true = (subset["label"] == attack).to_numpy(dtype=int)
        scores = subset[score_col].to_numpy(dtype=float)
        pr_auc, roc_auc = ranking_metrics(y_true, scores)
        budget = detection_metrics(y_true, scores, budget_fraction=budget_fraction)
        result[attack] = {
            "n": int(len(attack_rows)),
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "recall_at_budget": budget.recall,
            "is_edge_case": attack == AttackType.INSIDER_DRIFT.value,
        }
    return result


def _stealth_split(frame: pd.DataFrame, stealth_by_campaign: dict[str, bool]) -> pd.Series:
    return frame["campaign_id"].map(stealth_by_campaign)


def obvious_vs_stealth(
    frame: pd.DataFrame,
    score_col: str,
    stealth_by_campaign: dict[str, bool],
    *,
    budget_fraction: float,
) -> dict[str, Any]:
    """Headline metrics on benign+obvious vs benign+stealth attack subsets."""
    benign = frame[frame["label"] == AttackType.BENIGN.value]
    stealth_flag = _stealth_split(frame, stealth_by_campaign)
    out: dict[str, Any] = {}
    for name, want_stealth in (("obvious", False), ("stealth", True)):
        attacks = frame[frame["is_attack"] & (stealth_flag == want_stealth)]
        if attacks.empty:
            out[name] = {"n": 0}
            continue
        subset = pd.concat([benign, attacks], ignore_index=True)
        y_true = subset["is_attack"].to_numpy(dtype=int)
        scores = subset[score_col].to_numpy(dtype=float)
        pr_auc, roc_auc = ranking_metrics(y_true, scores)
        budget = detection_metrics(y_true, scores, budget_fraction=budget_fraction)
        out[name] = {
            "n": int(len(attacks)),
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "recall_at_budget": budget.recall,
        }
    return out


def _breakdown(per_campaign: pd.DataFrame, by: str) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for key, group in per_campaign.groupby(by, sort=True):
        detected = group["detected"]
        groups[str(key)] = {
            "n_campaigns": int(len(group)),
            "n_detected": int(detected.sum()),
            "detection_rate": float(detected.mean()),
            "median_latency_seconds": float(
                group.loc[detected, "latency_seconds"].median()
            )
            if detected.any()
            else None,
            "median_pct_events_before_detection": float(
                group.loc[detected, "pct_events_before_detection"].median()
            )
            if detected.any()
            else None,
        }
    return groups


def _subset_detection_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "n_campaigns": 0,
            "n_detected": 0,
            "detection_rate": float("nan"),
            "missed_campaigns": [],
            "median_latency_seconds": None,
            "median_events_before_detection": None,
            "n_latency_excluded_straddling": 0,
        }
    detected = frame["detected"]
    eligible = detected & ~frame["straddles_split"]
    return {
        "n_campaigns": int(len(frame)),
        "n_detected": int(detected.sum()),
        "detection_rate": float(detected.mean()),
        "missed_campaigns": frame.loc[~detected, "campaign_id"].tolist(),
        # For straddling campaigns we cannot observe pre-cutoff alerts here, so
        # latency aggregation excludes them instead of silently biasing low.
        "median_latency_seconds": float(frame.loc[eligible, "latency_seconds"].median())
        if eligible.any()
        else None,
        "median_events_before_detection": float(
            frame.loc[eligible, "events_before_detection"].median()
        )
        if eligible.any()
        else None,
        "n_latency_excluded_straddling": int((detected & frame["straddles_split"]).sum()),
    }


def campaign_level_metrics(
    eval_frame: pd.DataFrame,
    full_timeline: pd.DataFrame,
    alert_mask: np.ndarray,
    campaigns_meta: list[dict[str, Any]],
    *,
    operating_point: str,
) -> dict[str, Any]:
    """Campaign detection rate, latency and breakdowns using campaigns.json."""
    detection = campaign_detection(eval_frame, alert_mask)
    per_campaign = detection.per_campaign.copy()
    if per_campaign.empty:
        return {"operating_point": operating_point, "n_campaigns": 0}

    metadata_by_campaign = {c["campaign_id"]: c for c in campaigns_meta}
    stealth_by_campaign = {
        campaign_id: bool(meta.get("stealthy", False))
        for campaign_id, meta in metadata_by_campaign.items()
    }

    timeline = full_timeline.dropna(subset=["campaign_id"]).copy()
    true_start_by_campaign = timeline.groupby("campaign_id")["timestamp"].min().to_dict()
    has_train_rows = (
        timeline.assign(_is_train=timeline["split"].eq("train"))
        .groupby("campaign_id")["_is_train"]
        .any()
        .to_dict()
    )
    entity_type_group_by_campaign = (
        timeline.groupby("campaign_id")["entity_type"]
        .agg(lambda s: "|".join(sorted(set(s.astype(str)))))
        .to_dict()
    )
    entity_ids_by_campaign = {
        campaign_id: list(meta.get("entity_ids", []))
        for campaign_id, meta in metadata_by_campaign.items()
    }
    eval_entity_type_by_campaign = (
        eval_frame.dropna(subset=["campaign_id"])
        .groupby("campaign_id")["entity_type"]
        .agg(lambda s: "|".join(sorted(set(s.astype(str)))))
        .to_dict()
    )
    per_campaign["stealthy"] = per_campaign["campaign_id"].map(stealth_by_campaign)
    per_campaign["difficulty"] = per_campaign["stealthy"].map(
        {True: "stealth", False: "obvious"}
    )
    per_campaign["attack_type"] = per_campaign["campaign_id"].map(
        {
            campaign_id: str(meta.get("attack_type", ""))
            for campaign_id, meta in metadata_by_campaign.items()
        }
    )
    per_campaign["entity_ids"] = per_campaign["campaign_id"].map(entity_ids_by_campaign)
    per_campaign["entity_type_group"] = per_campaign["campaign_id"].map(
        entity_type_group_by_campaign
    )
    per_campaign["entity_type_evaluation"] = per_campaign["campaign_id"].map(
        eval_entity_type_by_campaign
    )
    per_campaign["true_start"] = per_campaign["campaign_id"].map(true_start_by_campaign)
    per_campaign["straddles_split"] = per_campaign["campaign_id"].map(has_train_rows).fillna(
        False
    )
    per_campaign["latency_seconds_true_start"] = np.where(
        per_campaign["detected"] & per_campaign["true_start"].notna(),
        (
            pd.to_datetime(per_campaign["first_alert"])
            - pd.to_datetime(per_campaign["true_start"])
        ).dt.total_seconds(),
        np.nan,
    )
    per_campaign["pct_events_before_detection"] = np.where(
        per_campaign["detected"],
        per_campaign["events_before_detection"] / per_campaign["n_events"] * 100.0,
        np.nan,
    )
    malicious = per_campaign[per_campaign["label"].isin(MALICIOUS_CLASSES)].copy()
    insider = per_campaign[per_campaign["label"] == AttackType.INSIDER_DRIFT.value].copy()
    malicious_summary = _subset_detection_summary(malicious)
    insider_summary = _subset_detection_summary(insider)
    obvious = malicious[malicious["difficulty"] == "obvious"]
    stealth = malicious[malicious["difficulty"] == "stealth"]
    straddling_campaign_ids = per_campaign.loc[
        per_campaign["straddles_split"], "campaign_id"
    ].tolist()

    return {
        "operating_point": operating_point,
        "n_campaigns_total": int(detection.n_campaigns),
        "n_campaigns_malicious": int(len(malicious)),
        "n_campaigns_insider_drift": int(len(insider)),
        "malicious": malicious_summary,
        "insider_drift": insider_summary,
        "malicious_detection_rate": malicious_summary["detection_rate"],
        "obvious_malicious_detection_rate": float(obvious["detected"].mean())
        if not obvious.empty
        else float("nan"),
        "stealth_malicious_detection_rate": float(stealth["detected"].mean())
        if not stealth.empty
        else float("nan"),
        "by_attack_type_malicious": _breakdown(malicious, "label"),
        "by_difficulty_malicious": _breakdown(malicious, "difficulty"),
        "by_entity_type_malicious": _breakdown(malicious, "entity_type_group"),
        "straddling": {
            "n_campaigns": int(len(straddling_campaign_ids)),
            "campaign_ids": straddling_campaign_ids,
            "latency_policy": "excluded_from_median_latency_aggregates",
        },
        "per_campaign": per_campaign.drop(columns=["start", "first_alert"]).to_dict("records"),
    }


__all__ = [
    "campaign_level_metrics",
    "event_level_metrics",
    "obvious_vs_stealth",
    "per_attack_type_metrics",
    "recall_at_fpr",
]
