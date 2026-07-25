"""Evaluation: imbalance-aware metrics, reproducible reporting, report figures.

Deliberately defined before the models exist, so that the numbers the project
reports are fixed by the problem (rare-event detection under a fixed analyst
budget) rather than chosen after seeing which look flattering.
"""

from __future__ import annotations

from .metrics import (
    CampaignDetection,
    ConfusionCounts,
    DetectionMetrics,
    budget_alert_mask,
    budget_sweep,
    campaign_detection,
    confusion_counts,
    confusion_frame,
    detection_metrics,
    per_class_metrics,
    pr_curve,
    ranking_metrics,
    roc_curve_points,
)
from .report import RunManifest, load_metrics, save_metrics

__all__ = [
    "CampaignDetection",
    "ConfusionCounts",
    "DetectionMetrics",
    "RunManifest",
    "budget_alert_mask",
    "budget_sweep",
    "campaign_detection",
    "confusion_counts",
    "confusion_frame",
    "detection_metrics",
    "load_metrics",
    "per_class_metrics",
    "pr_curve",
    "ranking_metrics",
    "roc_curve_points",
    "save_metrics",
]
