"""Phase 5 entry point: train, score, evaluate the IsolationForest detector.

Run with ``python -m src.models.anomaly_detector``. Produces the registered
model bundle, calibrated thresholds, per-event scores, and the event- and
campaign-level metric documents.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.artifacts import artifact_path
from src.config import load_config
from src.evaluation.report import RunManifest, write_json

from .baselines import fit_rule_baseline, random_scores
from .dataset import evaluation_frame, load_scoring_frame, training_matrix
from .evaluate import (
    campaign_level_metrics,
    event_level_metrics,
    obvious_vs_stealth,
    per_attack_type_metrics,
)
from .model import AnomalyModel, train_anomaly_model

_METADATA_COLUMNS = (
    "event_id", "timestamp", "entity_id", "entity_type",
    "split", "label", "is_attack", "campaign_id",
)


def _load_campaign_metadata() -> list[dict[str, Any]]:
    path = artifact_path("campaign_metadata")
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def build_scores(frame: pd.DataFrame, model: AnomalyModel, rule) -> pd.DataFrame:
    """Per-event score table: identity + evaluation metadata + all detectors."""
    scores = frame.loc[:, list(_METADATA_COLUMNS)].copy()
    # anomaly_score_raw is the canonical ranking/evaluation score.
    # anomaly_score is display-only [0, 1] normalisation for UI readability.
    scores["anomaly_score"] = model.anomaly_score(frame)
    scores["anomaly_score_raw"] = model.raw_scores(frame)
    scores["baseline_random"] = random_scores(len(frame))
    scores["baseline_rule"] = rule.score(frame)
    return scores


def evaluate_all(
    eval_scores: pd.DataFrame,
    all_scores: pd.DataFrame,
    model: AnomalyModel,
    campaigns_meta: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = load_config()
    budget_fraction = float(cfg["alerting.budget_fraction"])
    fpr_targets = list(cfg["evaluation.fpr_targets"])
    default_point = model.calibration.get("default", "balanced")

    # Operating-point thresholds live in raw-score space; the model score used
    # for evaluation is therefore the raw one, which is strictly monotonic.
    stealth_by_campaign = {c["campaign_id"]: bool(c["stealthy"]) for c in campaigns_meta}

    event_metrics = {
        "model": event_level_metrics(
            eval_scores, "anomaly_score_raw",
            thresholds=model.thresholds,
            budget_fraction=budget_fraction,
            fpr_targets=fpr_targets,
        ),
        "per_attack_type": per_attack_type_metrics(
            eval_scores, "anomaly_score_raw", budget_fraction=budget_fraction
        ),
        "obvious_vs_stealth": obvious_vs_stealth(
            eval_scores, "anomaly_score_raw", stealth_by_campaign,
            budget_fraction=budget_fraction,
        ),
        "baselines": {
            "random": event_level_metrics(
                eval_scores, "baseline_random",
                thresholds={}, budget_fraction=budget_fraction, fpr_targets=fpr_targets,
            ),
            "rule": event_level_metrics(
                eval_scores, "baseline_rule",
                thresholds={}, budget_fraction=budget_fraction, fpr_targets=fpr_targets,
            ),
        },
    }

    alert_mask = (
        eval_scores["anomaly_score_raw"].to_numpy(dtype=float)
        >= model.thresholds[default_point]
    )
    campaign_metrics = campaign_level_metrics(
        eval_scores,
        all_scores,
        alert_mask,
        campaigns_meta,
        operating_point=default_point,
    )
    return event_metrics, campaign_metrics


def run() -> dict[str, Path]:
    frame = load_scoring_frame()
    training_rows = training_matrix(frame)
    model = train_anomaly_model(training_rows)
    rule = fit_rule_baseline(training_rows)

    scores = build_scores(frame, model, rule)
    eval_scores = scores[scores["split"] == "evaluation"].reset_index(drop=True)
    campaigns_meta = _load_campaign_metadata()
    event_metrics, campaign_metrics = evaluate_all(
        eval_scores,
        scores,
        model,
        campaigns_meta,
    )

    manifest = asdict(RunManifest.capture(notes="phase5 anomaly detector"))
    paths = model.save()
    rule_path = rule.save()
    scores_path = artifact_path("event_scores", ensure_parent=True)
    scores.to_parquet(scores_path, index=False)
    eval_path = write_json(
        {"manifest": manifest, "metrics": event_metrics},
        artifact_path("evaluation_metrics", ensure_parent=True),
    )
    campaign_path = write_json(
        {"manifest": manifest, "metrics": campaign_metrics},
        artifact_path("campaign_metrics", ensure_parent=True),
    )
    paths.update(
        rule_baseline=rule_path,
        event_scores=scores_path,
        evaluation_metrics=eval_path,
        campaign_metrics=campaign_path,
    )
    return paths


def main() -> None:
    paths = run()
    metrics = json.loads(artifact_path("evaluation_metrics").read_text("utf-8"))["metrics"]
    model_pr = metrics["model"]["pr_auc"]
    rule_pr = metrics["baselines"]["rule"]["pr_auc"]
    random_pr = metrics["baselines"]["random"]["pr_auc"]
    print(json.dumps(
        {
            "pr_auc": {"model": model_pr, "rule": rule_pr, "random": random_pr},
            "recall_at_budget": metrics["model"]["budget"]["recall"],
            "paths": {k: str(v) for k, v in paths.items()},
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
