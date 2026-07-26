"""IsolationForest anomaly detector: fit, score, calibrate, persist.

Design commitments (all enforced by tests):

* The forest and its scaler are fit **only** on benign training-split rows.
* ``score_samples`` returns "higher = more normal"; we negate it so that higher
  always means more anomalous.
* Ranking metrics use the raw (unbounded, strictly monotonic) score so PR-AUC
  and ROC-AUC lose nothing to saturation. A separate min-max ``anomaly_score``
  in [0, 1] exists only for display and is derived from training scores.
* Operating-point thresholds are quantiles of benign *calibration* scores from
  the training period, never from held-out evaluation labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.artifacts import artifact_path
from src.config import load_config
from src.features import MODEL_FEATURE_COLUMNS
from src.schema import assert_no_label_leakage
from src.utils.seeding import derive_seed


@dataclass(frozen=True)
class AnomalyModel:
    """Fitted detector plus everything needed to reproduce its scores."""

    scaler: StandardScaler
    forest: IsolationForest
    feature_columns: tuple[str, ...]
    config: dict[str, Any]
    random_state: int
    score_min: float
    score_max: float
    thresholds: dict[str, float]
    calibration: dict[str, Any]

    # -- scoring ------------------------------------------------------------
    def _matrix(self, frame: pd.DataFrame) -> np.ndarray:
        return frame.loc[:, list(self.feature_columns)].to_numpy(dtype=float)

    def raw_scores(self, frame: pd.DataFrame) -> np.ndarray:
        """Unbounded anomaly score, higher = more anomalous (monotonic)."""
        transformed = self.scaler.transform(self._matrix(frame))
        return -self.forest.score_samples(transformed)

    def anomaly_score(self, frame: pd.DataFrame) -> np.ndarray:
        """Display score in [0, 1], min-max scaled against training scores."""
        raw = self.raw_scores(frame)
        span = self.score_max - self.score_min
        if span <= 0:
            return np.zeros_like(raw)
        return np.clip((raw - self.score_min) / span, 0.0, 1.0)

    def alert_mask(self, frame: pd.DataFrame, operating_point: str) -> np.ndarray:
        """Boolean alerts at a calibrated operating point (raw-score space)."""
        if operating_point not in self.thresholds:
            raise KeyError(
                f"unknown operating point {operating_point!r}; "
                f"have {sorted(self.thresholds)}"
            )
        return self.raw_scores(frame) >= self.thresholds[operating_point]

    # -- persistence --------------------------------------------------------
    def save(self) -> dict[str, Path]:
        bundle_path = artifact_path("anomaly_detector", ensure_parent=True)
        joblib.dump(self, bundle_path)
        thresholds_path = artifact_path("anomaly_thresholds", ensure_parent=True)
        import json

        thresholds_path.write_text(
            json.dumps(
                {
                    "methodology": self.calibration,
                    "thresholds_raw_score": self.thresholds,
                    "score_min": self.score_min,
                    "score_max": self.score_max,
                    "feature_columns": list(self.feature_columns),
                    "random_state": self.random_state,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return {"anomaly_detector": bundle_path, "anomaly_thresholds": thresholds_path}

    @staticmethod
    def load() -> "AnomalyModel":
        return joblib.load(artifact_path("anomaly_detector"))


def calibrate_thresholds(
    benign_calibration_raw: np.ndarray, operating_points: dict[str, float]
) -> dict[str, float]:
    """Raw-score thresholds at benign-calibration quantiles.

    Thresholding uses ``score >= threshold``. With finite samples and ties, the
    achieved benign FPR can differ slightly from exactly ``1 - quantile``.
    """
    return {
        name: float(np.quantile(benign_calibration_raw, quantile))
        for name, quantile in operating_points.items()
    }


def split_fit_and_calibration_rows(
    training_rows: pd.DataFrame,
    *,
    fit_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronologically split benign training rows into fit and calibration sets."""
    if not 0.0 < fit_fraction < 1.0:
        raise ValueError(f"calibration fit_fraction must be in (0, 1); got {fit_fraction}")
    ordered = training_rows.sort_values(["timestamp", "event_id"], kind="stable").reset_index(drop=True)
    if len(ordered) < 2:
        raise ValueError("need at least 2 benign training rows for fit/calibration split")
    fit_count = max(1, int(len(ordered) * fit_fraction))
    fit_count = min(fit_count, len(ordered) - 1)  # always leave calibration holdout
    fit_rows = ordered.iloc[:fit_count].reset_index(drop=True)
    calibration_rows = ordered.iloc[fit_count:].reset_index(drop=True)
    return fit_rows, calibration_rows


def train_anomaly_model(training_rows: pd.DataFrame) -> AnomalyModel:
    """Fit scaler + IsolationForest on benign training rows and calibrate.

    ``training_rows`` must already be filtered to benign training-split events;
    this function does not see labels and cannot filter them itself.
    """
    assert_no_label_leakage(MODEL_FEATURE_COLUMNS)
    if training_rows.empty:
        raise ValueError("no benign training rows to fit the anomaly detector")

    cfg = load_config()
    model_cfg = dict(cfg["models.anomaly_detector"])
    calibration_cfg = dict(model_cfg.get("calibration", {}))
    operating_points = dict(
        calibration_cfg.get(
            "operating_points", {"strict": 0.999, "balanced": 0.99, "sensitive": 0.95}
        )
    )
    fit_fraction = float(calibration_cfg.get("fit_fraction", 0.8))
    random_state = derive_seed("models.anomaly_detector")
    fit_rows, calibration_rows = split_fit_and_calibration_rows(
        training_rows,
        fit_fraction=fit_fraction,
    )
    fit_matrix = fit_rows.loc[:, list(MODEL_FEATURE_COLUMNS)].to_numpy(dtype=float)
    calibration_matrix = calibration_rows.loc[:, list(MODEL_FEATURE_COLUMNS)].to_numpy(dtype=float)
    scaler = StandardScaler().fit(fit_matrix)

    max_samples = model_cfg.get("max_samples", "auto")
    forest = IsolationForest(
        n_estimators=int(model_cfg.get("n_estimators", 300)),
        max_samples=max_samples if max_samples == "auto" else int(max_samples),
        # The project uses score_samples + external quantile calibration for
        # thresholds; contamination does not set the shipped operating points.
        contamination=float(model_cfg.get("contamination", 0.02)),
        n_jobs=int(model_cfg.get("n_jobs", -1)),
        random_state=random_state,
    ).fit(scaler.transform(fit_matrix))

    fit_raw = -forest.score_samples(scaler.transform(fit_matrix))
    benign_calibration_raw = -forest.score_samples(scaler.transform(calibration_matrix))
    thresholds = calibrate_thresholds(benign_calibration_raw, operating_points)

    return AnomalyModel(
        scaler=scaler,
        forest=forest,
        feature_columns=tuple(MODEL_FEATURE_COLUMNS),
        config=model_cfg,
        random_state=random_state,
        score_min=float(fit_raw.min()),
        score_max=float(fit_raw.max()),
        thresholds=thresholds,
        calibration={
            "method": calibration_cfg.get(
                "method", "benign_train_chronological_holdout_quantile"
            ),
            "default": calibration_cfg.get("default", "balanced"),
            "operating_points": operating_points,
            "fit_fraction": fit_fraction,
            "n_fit_events": int(len(fit_rows)),
            "n_calibration_events": int(len(calibration_rows)),
            "fit_window_end": fit_rows["timestamp"].iloc[-1].isoformat(),
            "calibration_window_start": calibration_rows["timestamp"].iloc[0].isoformat(),
        },
    )


__all__ = [
    "AnomalyModel",
    "calibrate_thresholds",
    "split_fit_and_calibration_rows",
    "train_anomaly_model",
]
