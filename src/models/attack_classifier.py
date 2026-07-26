"""Phase 7 entry point: train, score, evaluate the attack-type classifier.

Run with ``python -m src.models.attack_classifier``.

Produces the classifier bundle, per-event classifications, and per-class
metrics. If ``alerts.parquet`` already exists from the risk engine, alert rows
are enriched with ``attack_type`` / ``attack_confidence`` from the classifier
(evaluation metadata still never enters model features).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.artifacts import artifact_path
from src.evaluation.metrics import confusion_frame, per_class_metrics
from src.evaluation.report import RunManifest, write_json
from src.schema import ATTACK_CLASSES, AttackType, MALICIOUS_CLASSES

from .classifier import (
    CLASSIFIER_CLASSES,
    AttackClassifier,
    classifier_training_rows,
    train_attack_classifier,
)
from .dataset import evaluation_frame, load_scoring_frame


def build_classifications(frame: pd.DataFrame, model: AttackClassifier) -> pd.DataFrame:
    """Identity + evaluation metadata + predictions (labels for eval join only)."""
    meta = frame.loc[
        :,
        ["event_id", "timestamp", "entity_id", "entity_type", "split", "label", "is_attack", "campaign_id"],
    ].copy()
    preds = model.predict(frame)
    return pd.concat([meta.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)


def evaluate_classifier(eval_classifications: pd.DataFrame) -> dict[str, Any]:
    """Per-class precision/recall/F1, confusion matrix, malicious-only view."""
    y_true = eval_classifications["label"].astype(str).tolist()
    y_pred = eval_classifications["predicted_label"].astype(str).tolist()
    labels = list(CLASSIFIER_CLASSES)
    per_class = per_class_metrics(y_true, y_pred, labels)
    confusion = confusion_frame(y_true, y_pred, labels)

    overall_accuracy = float((eval_classifications["label"] == eval_classifications["predicted_label"]).mean())
    # Among truly malicious evaluation events, how often is the predicted class exact?
    malicious = eval_classifications[eval_classifications["label"].isin(MALICIOUS_CLASSES)]
    if malicious.empty:
        malicious_top1 = float("nan")
    else:
        malicious_top1 = float((malicious["label"] == malicious["predicted_label"]).mean())

    # Attack-vs-benign binary view of the multi-class output.
    true_attack = eval_classifications["label"].isin(MALICIOUS_CLASSES).to_numpy()
    pred_attack = eval_classifications["predicted_label"].isin(MALICIOUS_CLASSES).to_numpy()
    tp = int((true_attack & pred_attack).sum())
    fp = int((~true_attack & pred_attack).sum())
    tn = int((~true_attack & ~pred_attack).sum())
    fn = int((true_attack & ~pred_attack).sum())
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if np.isfinite(precision) and np.isfinite(recall) and (precision + recall)
        else float("nan")
    )

    edge = eval_classifications[eval_classifications["label"] == AttackType.INSIDER_DRIFT.value]
    edge_recall = (
        float((edge["predicted_label"] == AttackType.INSIDER_DRIFT.value).mean())
        if not edge.empty
        else float("nan")
    )

    return {
        "n_evaluation_events": int(len(eval_classifications)),
        "overall_accuracy": overall_accuracy,
        "macro_f1_present_classes": per_class.attrs.get("macro_f1_present_classes"),
        "per_class": per_class.to_dict("records"),
        "confusion_matrix": {
            "labels": labels,
            "matrix": confusion.to_numpy().astype(int).tolist(),
        },
        "malicious_top1_accuracy": malicious_top1,
        "binary_attack_detection": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "counts": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        },
        "insider_drift_recall": edge_recall,
        "attack_classes": list(ATTACK_CLASSES),
        "malicious_classes": list(MALICIOUS_CLASSES),
    }


def enrich_alerts_with_classifications(classifications: pd.DataFrame) -> Path | None:
    """Patch existing risk alerts with classifier attack_type / confidence."""
    alerts_path = artifact_path("alerts")
    if not alerts_path.exists():
        return None
    alerts = pd.read_parquet(alerts_path)
    if alerts.empty or "event_id" not in alerts.columns:
        return alerts_path

    preds = classifications.loc[:, ["event_id", "predicted_label", "attack_confidence"]]
    merged = alerts.drop(columns=["attack_type", "attack_confidence"], errors="ignore").merge(
        preds, on="event_id", how="left", validate="many_to_one"
    )
    merged["attack_type"] = merged["predicted_label"].fillna("UNKNOWN")
    merged["attack_confidence"] = merged["attack_confidence"].fillna(0.0)
    merged = merged.drop(columns=["predicted_label"])
    merged.to_parquet(alerts_path, index=False)
    return alerts_path


def run() -> dict[str, Path]:
    frame = load_scoring_frame()
    train_rows = classifier_training_rows(frame)
    model = train_attack_classifier(train_rows)

    classifications = build_classifications(frame, model)
    eval_rows = evaluation_frame(classifications)
    metrics = evaluate_classifier(eval_rows)

    model_path = model.save()
    class_path = artifact_path("classifications", ensure_parent=True)
    classifications.to_parquet(class_path, index=False)

    manifest = asdict(RunManifest.capture(notes="phase7 attack classifier"))
    metrics_path = write_json(
        {"manifest": manifest, "metrics": metrics},
        artifact_path("classifier_metrics", ensure_parent=True),
    )
    paths = {
        "attack_classifier": model_path,
        "classifications": class_path,
        "classifier_metrics": metrics_path,
    }
    alerts_path = enrich_alerts_with_classifications(classifications)
    if alerts_path is not None:
        paths["alerts"] = alerts_path
    return paths


def main() -> None:
    paths = run()
    metrics = json.loads(artifact_path("classifier_metrics").read_text(encoding="utf-8"))[
        "metrics"
    ]
    print(
        json.dumps(
            {
                "macro_f1_present_classes": metrics["macro_f1_present_classes"],
                "malicious_top1_accuracy": metrics["malicious_top1_accuracy"],
                "binary_attack_detection": metrics["binary_attack_detection"],
                "paths": {k: str(v) for k, v in paths.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
