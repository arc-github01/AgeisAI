"""Supervised attack-type classifier (Phase 7).

Trains a RandomForest on ``MODEL_FEATURE_COLUMNS`` using training-split labels.
Labels supervise the fit but never enter the feature matrix. The IsolationForest
and risk engine remain unchanged; this module only names attack types and
returns a confidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.artifacts import artifact_path
from src.config import load_config
from src.features import MODEL_FEATURE_COLUMNS
from src.schema import ATTACK_CLASSES, AttackType, assert_no_label_leakage
from src.utils.seeding import derive_seed

#: Classes the classifier is trained to separate, including BENIGN.
CLASSIFIER_CLASSES: tuple[str, ...] = (AttackType.BENIGN.value, *ATTACK_CLASSES)


@dataclass(frozen=True)
class AttackClassifier:
    """Fitted multi-class attack-type model plus preprocessing."""

    scaler: StandardScaler
    forest: RandomForestClassifier
    feature_columns: tuple[str, ...]
    classes: tuple[str, ...]
    config: dict[str, Any]
    random_state: int

    def _matrix(self, frame: pd.DataFrame) -> np.ndarray:
        return frame.loc[:, list(self.feature_columns)].to_numpy(dtype=float)

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return ``predicted_label`` and ``attack_confidence`` per row."""
        transformed = self.scaler.transform(self._matrix(frame))
        proba = self.forest.predict_proba(transformed)
        # RandomForestClassifier.predict() calls predict_proba() internally and
        # then takes the argmax. Reusing the probabilities avoids traversing all
        # trees twice on the latency-sensitive single-event streaming path.
        labels = self.forest.classes_.take(np.argmax(proba, axis=1))
        confidence = proba.max(axis=1)
        # Probability of the predicted class among malicious classes (excl. BENIGN),
        # useful when the top class is an attack; 0 when predicted BENIGN.
        class_index = {name: i for i, name in enumerate(self.forest.classes_)}
        out = pd.DataFrame(
            {
                "predicted_label": labels.astype(str),
                "attack_confidence": confidence.astype(float),
            },
            index=frame.index,
        )
        # Per-class probabilities for evaluation / debugging.
        for name in self.classes:
            if name in class_index:
                out[f"proba_{name}"] = proba[:, class_index[name]]
            else:
                out[f"proba_{name}"] = 0.0
        return out

    def save(self) -> Path:
        path = artifact_path("attack_classifier", ensure_parent=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load() -> "AttackClassifier":
        return joblib.load(artifact_path("attack_classifier"))


def classifier_training_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """All training-split rows (benign + attacks). Labels supervise; features do not include them."""
    return frame.loc[frame["split"] == "train"].reset_index(drop=True)


def train_attack_classifier(training_rows: pd.DataFrame) -> AttackClassifier:
    """Fit scaler + RandomForest on training-split labelled events."""
    assert_no_label_leakage(MODEL_FEATURE_COLUMNS)
    if training_rows.empty:
        raise ValueError("no training rows for the attack classifier")
    if "label" not in training_rows.columns:
        raise ValueError("classifier training requires a label column for supervision")

    cfg = load_config()
    model_cfg = dict(cfg["models.attack_classifier"])
    random_state = derive_seed("models.attack_classifier")

    labels = training_rows["label"].astype(str)
    unknown = sorted(set(labels) - set(CLASSIFIER_CLASSES))
    if unknown:
        raise ValueError(f"unexpected training labels: {unknown}")

    matrix = training_rows.loc[:, list(MODEL_FEATURE_COLUMNS)].to_numpy(dtype=float)
    scaler = StandardScaler().fit(matrix)
    forest = RandomForestClassifier(
        n_estimators=int(model_cfg.get("n_estimators", 400)),
        max_depth=int(model_cfg["max_depth"]) if model_cfg.get("max_depth") is not None else None,
        min_samples_leaf=int(model_cfg.get("min_samples_leaf", 3)),
        class_weight=model_cfg.get("class_weight", "balanced_subsample"),
        n_jobs=int(model_cfg.get("n_jobs", -1)),
        random_state=random_state,
    ).fit(scaler.transform(matrix), labels.to_numpy())

    return AttackClassifier(
        scaler=scaler,
        forest=forest,
        feature_columns=tuple(MODEL_FEATURE_COLUMNS),
        classes=CLASSIFIER_CLASSES,
        config=model_cfg,
        random_state=random_state,
    )


__all__ = [
    "CLASSIFIER_CLASSES",
    "AttackClassifier",
    "classifier_training_rows",
    "train_attack_classifier",
]
