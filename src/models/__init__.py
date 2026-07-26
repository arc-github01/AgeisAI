"""AEGIS models: IsolationForest anomaly detection + attack-type classifier."""

from .baselines import RuleBaseline, fit_rule_baseline, random_scores
from .classifier import (
    CLASSIFIER_CLASSES,
    AttackClassifier,
    classifier_training_rows,
    train_attack_classifier,
)
from .dataset import evaluation_frame, load_scoring_frame, training_matrix
from .model import (
    AnomalyModel,
    calibrate_thresholds,
    split_fit_and_calibration_rows,
    train_anomaly_model,
)

__all__ = [
    "AnomalyModel",
    "AttackClassifier",
    "CLASSIFIER_CLASSES",
    "RuleBaseline",
    "calibrate_thresholds",
    "classifier_training_rows",
    "evaluation_frame",
    "fit_rule_baseline",
    "load_scoring_frame",
    "random_scores",
    "split_fit_and_calibration_rows",
    "train_anomaly_model",
    "train_attack_classifier",
    "training_matrix",
]
