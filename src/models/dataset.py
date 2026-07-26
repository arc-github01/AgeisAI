"""Assemble the scoring frame for Phase 5 (anomaly detection).

The feature matrix (`features.parquet`) deliberately carries no ground truth, so
labels, campaign identifiers and entity type are joined back from
`events.parquet` here. These evaluation-only columns ride alongside the feature
columns in memory but are never handed to the model: `MODEL_FEATURE_COLUMNS` is
the single source of truth for what the detector consumes.
"""

from __future__ import annotations

import pandas as pd

from src.artifacts import artifact_path
from src.features import MODEL_FEATURE_COLUMNS
from src.schema import assert_no_label_leakage

#: Columns joined purely for training-row selection and evaluation. Forbidden as
#: model inputs; the split guarantees they never reach the feature matrix.
EVALUATION_COLUMNS = ("entity_type", "label", "is_attack", "campaign_id")


def load_scoring_frame(
    features: pd.DataFrame | None = None,
    events: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Feature matrix joined to evaluation-only ground truth on ``event_id``."""
    if features is None:
        features = pd.read_parquet(artifact_path("features"))
    if events is None:
        events = pd.read_parquet(artifact_path("events"))

    # A wrong join would silently leak or misalign labels, so verify it is 1:1.
    assert_no_label_leakage(MODEL_FEATURE_COLUMNS)
    truth = events.loc[:, ["event_id", *EVALUATION_COLUMNS]]
    merged = features.merge(truth, on="event_id", how="left", validate="one_to_one")
    if merged["label"].isna().any():
        raise ValueError("feature rows without a matching event label; join is broken")
    return merged.sort_values(["timestamp", "event_id"], kind="stable").reset_index(drop=True)


def training_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows the IsolationForest may fit on: benign AND in the training split.

    Attack rows and every evaluation-split row are excluded here, which is the
    load-bearing guarantee behind the unsupervised, leakage-free training claim.
    """
    mask = (frame["split"] == "train") & (frame["label"] == "BENIGN")
    return frame.loc[mask].reset_index(drop=True)


def evaluation_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """The held-out evaluation split, scored but never trained or calibrated on."""
    return frame.loc[frame["split"] == "evaluation"].reset_index(drop=True)


__all__ = [
    "EVALUATION_COLUMNS",
    "evaluation_frame",
    "load_scoring_frame",
    "training_matrix",
]
