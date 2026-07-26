"""Honest reference detectors the IsolationForest has to beat.

Neither is crippled on purpose. The rule baseline in particular is a legitimate,
if naive, detector: standardise a handful of features that a human analyst would
intuitively call suspicious, on benign training data, and sum them. If the
learned model cannot outperform that, the learned model is not earning its keep.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.artifacts import artifact_path
from src.config import load_config
from src.utils.seeding import derive_seed


def random_scores(n: int, *, component: str = "models.baseline.random") -> np.ndarray:
    """A seeded random anomaly score: the absolute floor of usefulness."""
    return np.random.default_rng(derive_seed(component)).random(n)


@dataclass(frozen=True)
class RuleBaseline:
    """Sum of benign-standardised suspicious features (higher = worse)."""

    features: tuple[str, ...]
    means: np.ndarray
    stds: np.ndarray

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = frame.loc[:, list(self.features)].to_numpy(dtype=float)
        standardised = (matrix - self.means) / self.stds
        return standardised.sum(axis=1)

    def save(self, path: Path | None = None) -> Path:
        target = path or artifact_path("rule_baseline", ensure_parent=True)
        joblib.dump(self, target)
        return target

    @staticmethod
    def load(path: Path | None = None) -> "RuleBaseline":
        target = path or artifact_path("rule_baseline")
        return joblib.load(target)


def fit_rule_baseline(training_rows: pd.DataFrame) -> RuleBaseline:
    """Fit the rule baseline's standardisation on benign training rows only."""
    features = tuple(load_config()["models.baselines.rule_features"])
    matrix = training_rows.loc[:, list(features)].to_numpy(dtype=float)
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    stds[stds == 0] = 1.0  # a constant feature contributes nothing, not a divide-by-zero
    return RuleBaseline(features=features, means=means, stds=stds)


__all__ = ["RuleBaseline", "fit_rule_baseline", "random_scores"]