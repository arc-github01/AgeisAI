"""Imbalance-aware evaluation metrics for AEGIS.

Defined *before* any model exists, deliberately: the metrics a project reports
should not be chosen after seeing which ones look good.

Why these metrics
-----------------
At ~1% attack prevalence, accuracy is worthless (predict "normal" always -> 99%).
The quantities that actually describe a usable detector are:

* **PR-AUC** - threshold-free quality on the rare positive class.
* **Recall @ alert budget** - of the attacks that exist, how many appear in the
  top 1% of events a human can actually triage? This is the operational number.
* **Per-attack precision/recall/F1** - a detector that finds only brute force is
  not a behavioral detector.
* **False-positive rate** - the analyst-fatigue metric.
* **Campaign detection latency** - for low-and-slow behaviour, *when* an attack
  was caught matters as much as *whether*.

Conventions
-----------
* ``y_true`` is a binary integer array (1 = malicious ground truth).
* ``y_score`` is "higher means more suspicious" - anomaly score or risk score.
* Degenerate inputs (no positives, no negatives, constant scores) return ``nan``
  rather than raising, so an early-phase pipeline can still emit a report.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

NAN = float("nan")


# -----------------------------------------------------------------------------
# Input handling
# -----------------------------------------------------------------------------
def _as_binary(y_true: Iterable) -> np.ndarray:
    arr = np.asarray(list(y_true) if not isinstance(y_true, np.ndarray) else y_true)
    if arr.size == 0:
        raise ValueError("y_true is empty")
    if arr.dtype == bool:
        return arr.astype(np.int8)
    arr = arr.astype(np.int64)
    invalid = set(np.unique(arr)) - {0, 1}
    if invalid:
        raise ValueError(f"y_true must be binary 0/1; found {sorted(invalid)}")
    return arr.astype(np.int8)


def _as_score(y_score: Iterable, n: int) -> np.ndarray:
    arr = np.asarray(y_score, dtype=float)
    if arr.shape[0] != n:
        raise ValueError(f"y_score length {arr.shape[0]} != y_true length {n}")
    if not np.isfinite(arr).all():
        raise ValueError("y_score contains NaN or infinite values")
    return arr


# -----------------------------------------------------------------------------
# Alert budget
# -----------------------------------------------------------------------------
def budget_alert_mask(
    y_score: Iterable, budget_fraction: float
) -> tuple[np.ndarray, float, int]:
    """Select exactly the top ``budget_fraction`` of events as alerts.

    Returns ``(mask, threshold_score, n_alerts)``.

    Ties are broken by original event order (earliest first) via a stable sort,
    so the budget is honoured *exactly* rather than overflowing on tied scores.
    That is the operationally correct choice: an analyst team has a fixed
    capacity, not a fixed score cut-off.
    """
    scores = np.asarray(y_score, dtype=float)
    if scores.size == 0:
        raise ValueError("y_score is empty")
    if not 0 < budget_fraction <= 1:
        raise ValueError(f"budget_fraction must be in (0, 1]; got {budget_fraction}")

    n = scores.shape[0]
    k = max(1, math.ceil(n * budget_fraction))
    order = np.argsort(-scores, kind="stable")[:k]
    mask = np.zeros(n, dtype=bool)
    mask[order] = True
    return mask, float(scores[order[-1]]), int(k)


# -----------------------------------------------------------------------------
# Result containers
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ConfusionCounts:
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else NAN

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else NAN

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if not np.isfinite(p) or not np.isfinite(r) or (p + r) == 0:
            return NAN
        return 2 * p * r / (p + r)

    @property
    def fpr(self) -> float:
        denom = self.fp + self.tn
        return self.fp / denom if denom else NAN


@dataclass(frozen=True)
class DetectionMetrics:
    """Full binary-detection result at one operating point."""

    n_events: int
    n_positives: int
    prevalence: float
    threshold: float
    n_alerts: int
    alert_rate: float
    precision: float
    recall: float
    f1: float
    fpr: float
    pr_auc: float
    roc_auc: float
    counts: ConfusionCounts
    operating_point: str = "threshold"

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["counts"] = asdict(self.counts)
        return payload


def confusion_counts(y_true: Iterable, y_pred: Iterable) -> ConfusionCounts:
    truth = _as_binary(y_true)
    pred = np.asarray(y_pred).astype(bool)
    if pred.shape[0] != truth.shape[0]:
        raise ValueError("y_pred and y_true lengths differ")
    tp = int(np.sum((truth == 1) & pred))
    fp = int(np.sum((truth == 0) & pred))
    tn = int(np.sum((truth == 0) & ~pred))
    fn = int(np.sum((truth == 1) & ~pred))
    return ConfusionCounts(tp=tp, fp=fp, tn=tn, fn=fn)


def ranking_metrics(y_true: Iterable, y_score: Iterable) -> tuple[float, float]:
    """Threshold-free ``(pr_auc, roc_auc)``; ``nan`` when undefined."""
    truth = _as_binary(y_true)
    scores = _as_score(y_score, truth.shape[0])
    positives = int(truth.sum())
    if positives == 0 or positives == truth.shape[0]:
        return NAN, NAN
    return (
        float(average_precision_score(truth, scores)),
        float(roc_auc_score(truth, scores)),
    )


def detection_metrics(
    y_true: Iterable,
    y_score: Iterable,
    *,
    threshold: float | None = None,
    budget_fraction: float | None = None,
) -> DetectionMetrics:
    """Evaluate a detector at a threshold or, preferably, at an alert budget.

    Exactly one of ``threshold`` / ``budget_fraction`` must be given.
    """
    if (threshold is None) == (budget_fraction is None):
        raise ValueError("pass exactly one of threshold= or budget_fraction=")

    truth = _as_binary(y_true)
    scores = _as_score(y_score, truth.shape[0])

    if budget_fraction is not None:
        mask, threshold_value, _ = budget_alert_mask(scores, budget_fraction)
        operating_point = f"budget@{budget_fraction:.4g}"
    else:
        threshold_value = float(threshold)
        mask = scores >= threshold_value
        operating_point = f"threshold@{threshold_value:.6g}"

    counts = confusion_counts(truth, mask)
    pr_auc, roc_auc = ranking_metrics(truth, scores)
    n = int(truth.shape[0])

    return DetectionMetrics(
        n_events=n,
        n_positives=int(truth.sum()),
        prevalence=float(truth.mean()),
        threshold=threshold_value,
        n_alerts=int(mask.sum()),
        alert_rate=float(mask.mean()),
        precision=counts.precision,
        recall=counts.recall,
        f1=counts.f1,
        fpr=counts.fpr,
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        counts=counts,
        operating_point=operating_point,
    )


# -----------------------------------------------------------------------------
# Curves
# -----------------------------------------------------------------------------
def pr_curve(y_true: Iterable, y_score: Iterable) -> pd.DataFrame:
    truth = _as_binary(y_true)
    scores = _as_score(y_score, truth.shape[0])
    if truth.sum() == 0:
        return pd.DataFrame(columns=["recall", "precision", "threshold"])
    precision, recall, thresholds = precision_recall_curve(truth, scores)
    # precision_recall_curve returns one more point than thresholds
    return pd.DataFrame(
        {
            "recall": recall[:-1],
            "precision": precision[:-1],
            "threshold": thresholds,
        }
    )


def roc_curve_points(y_true: Iterable, y_score: Iterable) -> pd.DataFrame:
    truth = _as_binary(y_true)
    scores = _as_score(y_score, truth.shape[0])
    if truth.sum() == 0 or truth.sum() == truth.shape[0]:
        return pd.DataFrame(columns=["fpr", "tpr", "threshold"])
    fpr, tpr, thresholds = roc_curve(truth, scores)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})


def budget_sweep(
    y_true: Iterable,
    y_score: Iterable,
    fractions: Sequence[float] = (0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.10),
) -> pd.DataFrame:
    """Precision / recall / FPR as the analyst alert budget is varied.

    This is the chart that answers "what do we get for the capacity we have?"
    """
    truth = _as_binary(y_true)
    scores = _as_score(y_score, truth.shape[0])
    rows = []
    for fraction in fractions:
        result = detection_metrics(truth, scores, budget_fraction=fraction)
        rows.append(
            {
                "budget_fraction": fraction,
                "n_alerts": result.n_alerts,
                "precision": result.precision,
                "recall": result.recall,
                "f1": result.f1,
                "fpr": result.fpr,
                "threshold": result.threshold,
            }
        )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Multi-class attack-type classification
# -----------------------------------------------------------------------------
def per_class_metrics(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> pd.DataFrame:
    """Precision / recall / F1 / support for every attack category."""
    precision, recall, f1, support = precision_recall_fscore_support(
        list(y_true), list(y_pred), labels=list(labels), zero_division=0
    )
    frame = pd.DataFrame(
        {
            "class": list(labels),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )
    present = frame["support"] > 0
    frame.attrs["macro_f1_present_classes"] = (
        float(frame.loc[present, "f1"].mean()) if present.any() else NAN
    )
    return frame


def confusion_frame(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> pd.DataFrame:
    """Labelled confusion matrix (rows = truth, columns = prediction)."""
    matrix = confusion_matrix(list(y_true), list(y_pred), labels=list(labels))
    return pd.DataFrame(matrix, index=list(labels), columns=list(labels))


# -----------------------------------------------------------------------------
# Campaign-level detection (the metric that matters for low-and-slow)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class CampaignDetection:
    n_campaigns: int
    n_detected: int
    campaign_recall: float
    median_latency_seconds: float
    median_events_before_detection: float
    per_campaign: pd.DataFrame = field(repr=False)

    def to_dict(self) -> dict:
        return {
            "n_campaigns": self.n_campaigns,
            "n_detected": self.n_detected,
            "campaign_recall": self.campaign_recall,
            "median_latency_seconds": self.median_latency_seconds,
            "median_events_before_detection": self.median_events_before_detection,
            "per_campaign": self.per_campaign.to_dict("records"),
        }


def campaign_detection(
    events: pd.DataFrame,
    alert_mask: Iterable[bool],
    *,
    campaign_col: str = "campaign_id",
    time_col: str = "timestamp",
    label_col: str = "label",
) -> CampaignDetection:
    """How quickly is each injected attack campaign caught?

    Event-level recall understates a detector that catches one event of a
    50-event campaign - which, operationally, is a caught intrusion. This
    reports per-campaign coverage and time-to-first-alert instead.
    """
    frame = events.loc[:, [campaign_col, time_col, label_col]].copy()
    frame["alerted"] = np.asarray(alert_mask, dtype=bool)
    frame = frame[frame[campaign_col].notna()]
    if frame.empty:
        return CampaignDetection(0, 0, NAN, NAN, NAN, pd.DataFrame())

    frame = frame.sort_values(time_col)
    rows = []
    for campaign_id, group in frame.groupby(campaign_col, sort=True):
        start = group[time_col].iloc[0]
        alerted = group[group["alerted"]]
        detected = not alerted.empty
        first = alerted[time_col].iloc[0] if detected else pd.NaT
        rows.append(
            {
                "campaign_id": campaign_id,
                "label": group[label_col].iloc[0],
                "n_events": int(len(group)),
                "start": start,
                "detected": detected,
                "first_alert": first,
                "latency_seconds": (first - start).total_seconds() if detected else NAN,
                "events_before_detection": (
                    int((group[time_col] < first).sum()) if detected else NAN
                ),
            }
        )

    per_campaign = pd.DataFrame(rows)
    detected = per_campaign["detected"]
    return CampaignDetection(
        n_campaigns=int(len(per_campaign)),
        n_detected=int(detected.sum()),
        campaign_recall=float(detected.mean()),
        median_latency_seconds=float(per_campaign.loc[detected, "latency_seconds"].median())
        if detected.any()
        else NAN,
        median_events_before_detection=float(
            per_campaign.loc[detected, "events_before_detection"].median()
        )
        if detected.any()
        else NAN,
        per_campaign=per_campaign,
    )


__all__ = [
    "CampaignDetection",
    "ConfusionCounts",
    "DetectionMetrics",
    "budget_alert_mask",
    "budget_sweep",
    "campaign_detection",
    "confusion_counts",
    "confusion_frame",
    "detection_metrics",
    "per_class_metrics",
    "pr_curve",
    "ranking_metrics",
    "roc_curve_points",
]
