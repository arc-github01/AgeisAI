"""Phase 7 attack-classifier contracts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import MODEL_FEATURE_COLUMNS
from src.models.attack_classifier import (
    build_classifications,
    enrich_alerts_with_classifications,
    evaluate_classifier,
)
from src.models.classifier import (
    CLASSIFIER_CLASSES,
    classifier_training_rows,
    train_attack_classifier,
)
from src.models.dataset import EVALUATION_COLUMNS, evaluation_frame
from src.schema import FORBIDDEN_FEATURE_COLUMNS, MALICIOUS_CLASSES

_BASE = pd.Timestamp("2025-01-06T08:00:00Z")


def _make_frame(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    def add(n: int, *, split: str, label: str, shift: float):
        for _ in range(n):
            row = {col: float(rng.normal(shift, 0.4)) for col in MODEL_FEATURE_COLUMNS}
            # Give each attack type a distinctive feature bump so the forest can learn.
            if label == "BRUTE_FORCE":
                row["failed_auth_5m"] = float(rng.normal(8.0, 0.5))
            elif label == "IMPOSSIBLE_TRAVEL":
                row["geo_velocity_kmh"] = float(rng.normal(5000.0, 100.0))
            elif label == "LATERAL_MOVEMENT":
                row["sequence_anomaly_score"] = float(rng.normal(6.0, 0.3))
                row["recent_resource_breadth_24h"] = float(rng.normal(8.0, 0.5))
            elif label == "DEVICE_SPOOFING":
                row["is_known_device"] = 0.0
                row["device_rarity"] = float(rng.normal(4.0, 0.2))
            row.update(
                event_id=f"E{len(rows):05d}",
                timestamp=_BASE + pd.Timedelta(minutes=len(rows)),
                entity_id=f"user-{len(rows) % 7}",
                entity_type="user",
                split=split,
                label=label,
                is_attack=label in MALICIOUS_CLASSES,
                campaign_id=None if label == "BENIGN" else f"CMP-{label}",
            )
            rows.append(row)

    add(120, split="train", label="BENIGN", shift=0.0)
    add(20, split="train", label="BRUTE_FORCE", shift=1.0)
    add(15, split="train", label="IMPOSSIBLE_TRAVEL", shift=1.0)
    add(15, split="train", label="LATERAL_MOVEMENT", shift=1.0)
    add(12, split="train", label="DEVICE_SPOOFING", shift=1.0)
    add(40, split="evaluation", label="BENIGN", shift=0.0)
    add(8, split="evaluation", label="BRUTE_FORCE", shift=1.0)
    add(6, split="evaluation", label="IMPOSSIBLE_TRAVEL", shift=1.0)
    add(6, split="evaluation", label="LATERAL_MOVEMENT", shift=1.0)
    add(5, split="evaluation", label="DEVICE_SPOOFING", shift=1.0)
    return pd.DataFrame(rows)


def test_classifier_features_exclude_labels():
    for forbidden in FORBIDDEN_FEATURE_COLUMNS:
        assert forbidden not in MODEL_FEATURE_COLUMNS
    model = train_attack_classifier(classifier_training_rows(_make_frame()))
    assert model.feature_columns == tuple(MODEL_FEATURE_COLUMNS)
    assert set(EVALUATION_COLUMNS).isdisjoint(model.feature_columns)


def test_classifier_trains_on_training_split_only():
    frame = _make_frame()
    train = classifier_training_rows(frame)
    assert (train["split"] == "train").all()
    assert set(train["label"]).issubset(CLASSIFIER_CLASSES)
    # Evaluation rows must not be required for fitting.
    model = train_attack_classifier(train)
    eval_rows = evaluation_frame(frame)
    preds = model.predict(eval_rows)
    assert len(preds) == len(eval_rows)
    assert preds["attack_confidence"].between(0.0, 1.0).all()


def test_classifier_is_deterministic():
    train = classifier_training_rows(_make_frame())
    a = train_attack_classifier(train)
    b = train_attack_classifier(train)
    assert a.random_state == b.random_state
    eval_rows = evaluation_frame(_make_frame())
    pa = a.predict(eval_rows)
    pb = b.predict(eval_rows)
    assert list(pa["predicted_label"]) == list(pb["predicted_label"])
    np.testing.assert_allclose(
        pa["attack_confidence"].to_numpy(float),
        pb["attack_confidence"].to_numpy(float),
        rtol=0.0,
        atol=1e-12,
    )


def test_classifier_separates_obvious_attack_types():
    frame = _make_frame()
    model = train_attack_classifier(classifier_training_rows(frame))
    classified = build_classifications(frame, model)
    eval_rows = evaluation_frame(classified)
    metrics = evaluate_classifier(eval_rows)
    assert metrics["macro_f1_present_classes"] > 0.5
    # Distinctive attack types should be learnable on this synthetic frame.
    per_class = {row["class"]: row for row in metrics["per_class"]}
    assert per_class["BRUTE_FORCE"]["f1"] > 0.5
    assert per_class["IMPOSSIBLE_TRAVEL"]["f1"] > 0.5


def test_evaluate_classifier_reports_confusion_and_binary_view():
    frame = _make_frame()
    model = train_attack_classifier(classifier_training_rows(frame))
    classified = build_classifications(frame, model)
    metrics = evaluate_classifier(evaluation_frame(classified))
    assert "confusion_matrix" in metrics
    assert metrics["confusion_matrix"]["labels"] == list(CLASSIFIER_CLASSES)
    assert {"precision", "recall", "f1"} <= set(metrics["binary_attack_detection"])


def test_enrich_alerts_uses_predictions_not_ground_truth(tmp_path, monkeypatch):
    import src.models.attack_classifier as ac

    frame = _make_frame()
    model = train_attack_classifier(classifier_training_rows(frame))
    classified = build_classifications(frame, model)
    eval_attack = classified[(classified["split"] == "evaluation") & classified["is_attack"]].iloc[0]

    alerts = pd.DataFrame(
        [
            {
                "alert_id": "ALT-1",
                "event_id": eval_attack["event_id"],
                "entity_id": eval_attack["entity_id"],
                "attack_type": "UNKNOWN",
                "attack_confidence": 0.0,
                "risk_score": 90.0,
                "severity": "CRITICAL",
            }
        ]
    )
    alerts_path = tmp_path / "alerts.parquet"
    alerts.to_parquet(alerts_path, index=False)

    monkeypatch.setattr(ac, "artifact_path", lambda key, ensure_parent=False: alerts_path)
    enrich_alerts_with_classifications(classified)
    enriched = pd.read_parquet(alerts_path)
    assert enriched.iloc[0]["attack_type"] == eval_attack["predicted_label"]
    assert float(enriched.iloc[0]["attack_confidence"]) > 0.0
    # Must come from the model prediction, not by copying the label column blindly
    # in the enricher (enricher only joins predicted_label).
    assert "label" not in enriched.columns
