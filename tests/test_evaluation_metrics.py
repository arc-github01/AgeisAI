"""Tests for the evaluation layer.

Expected values here are hand-computed, not captured from a previous run, so
these tests can actually catch a wrong metric rather than freezing one in.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from src.artifacts import REGISTRY, artifact, missing_artifacts, pipeline_status
from src.evaluation import metrics as M
from src.evaluation.report import RunManifest, load_metrics, save_metrics


# -----------------------------------------------------------------------------
# Alert budget - the operational core
# -----------------------------------------------------------------------------
def test_budget_selects_exactly_top_k_events():
    scores = np.arange(1000, dtype=float)
    mask, threshold, k = M.budget_alert_mask(scores, 0.01)
    assert k == 10 and mask.sum() == 10
    assert threshold == 990.0
    assert mask[-10:].all() and not mask[:-10].any()


def test_budget_rounds_up_and_always_alerts_at_least_once():
    _, _, k = M.budget_alert_mask(np.arange(101, dtype=float), 0.01)
    assert k == 2  # ceil(101 * 0.01)
    _, _, k_small = M.budget_alert_mask(np.arange(10, dtype=float), 0.001)
    assert k_small == 1


def test_budget_honours_capacity_exactly_when_scores_are_tied():
    # A naive `score >= threshold` rule would alert on all 100 tied events and
    # blow the analyst budget by 10x.
    mask, _, k = M.budget_alert_mask(np.ones(100), 0.10)
    assert k == 10 and mask.sum() == 10
    assert mask[:10].all()  # deterministic tie-break: earliest events first


def test_budget_rejects_invalid_fractions():
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            M.budget_alert_mask(np.arange(10, dtype=float), bad)


def test_recall_at_budget_is_hand_verifiable():
    # 100 events, 5 malicious. Only the single highest-scoring event is
    # malicious; the other 4 attacks score below every benign event.
    y_true = np.zeros(100, dtype=int)
    y_score = np.linspace(0.0, 0.95, 100)
    y_true[99] = 1                      # top-scoring event, inside a 1% budget
    y_true[[0, 1, 2, 3]] = 1            # missed attacks, lowest scores
    result = M.detection_metrics(y_true, y_score, budget_fraction=0.01)
    assert result.n_alerts == 1
    assert result.recall == pytest.approx(0.2)      # 1 of 5 attacks
    assert result.precision == pytest.approx(1.0)   # the single alert was real
    assert result.counts.tp == 1 and result.counts.fn == 4 and result.counts.fp == 0
    assert result.operating_point == "budget@0.01"


# -----------------------------------------------------------------------------
# Ranking metrics
# -----------------------------------------------------------------------------
def test_perfect_and_chance_rankings():
    y_true = np.array([0] * 90 + [1] * 10)
    perfect = np.concatenate([np.zeros(90), np.ones(10)])
    pr_auc, roc_auc = M.ranking_metrics(y_true, perfect)
    assert pr_auc == pytest.approx(1.0) and roc_auc == pytest.approx(1.0)

    constant = np.full(100, 0.5)
    _, chance_roc = M.ranking_metrics(y_true, constant)
    assert chance_roc == pytest.approx(0.5)


def test_inverted_ranking_scores_below_chance():
    y_true = np.array([0] * 90 + [1] * 10)
    inverted = np.concatenate([np.ones(90), np.zeros(10)])
    _, roc_auc = M.ranking_metrics(y_true, inverted)
    assert roc_auc == pytest.approx(0.0)


def test_degenerate_inputs_return_nan_instead_of_raising():
    pr_auc, roc_auc = M.ranking_metrics(np.zeros(50, dtype=int), np.random.rand(50))
    assert math.isnan(pr_auc) and math.isnan(roc_auc)


def test_input_validation():
    with pytest.raises(ValueError):
        M.ranking_metrics([], [])
    with pytest.raises(ValueError):
        M.ranking_metrics([0, 1, 2], [0.1, 0.2, 0.3])          # not binary
    with pytest.raises(ValueError):
        M.ranking_metrics([0, 1], [0.1, np.nan])                # non-finite score
    with pytest.raises(ValueError):
        M.ranking_metrics([0, 1, 1], [0.1, 0.2])                # length mismatch
    with pytest.raises(ValueError):
        M.detection_metrics([0, 1], [0.1, 0.9])                 # no operating point
    with pytest.raises(ValueError):
        M.detection_metrics([0, 1], [0.1, 0.9], threshold=0.5, budget_fraction=0.1)


def test_confusion_counts_and_derived_rates():
    y_true = [1, 1, 0, 0, 0, 0]
    y_pred = [True, False, True, False, False, False]
    counts = M.confusion_counts(y_true, y_pred)
    assert (counts.tp, counts.fn, counts.fp, counts.tn) == (1, 1, 1, 3)
    assert counts.precision == pytest.approx(0.5)
    assert counts.recall == pytest.approx(0.5)
    assert counts.f1 == pytest.approx(0.5)
    assert counts.fpr == pytest.approx(0.25)


def test_accuracy_trap_is_visible_in_the_metrics():
    # The "predict normal for everything" classifier: 99% accurate, useless.
    y_true = np.array([0] * 990 + [1] * 10)
    y_score = np.zeros(1000)
    result = M.detection_metrics(y_true, y_score, threshold=0.5)
    assert result.n_alerts == 0
    assert result.recall == pytest.approx(0.0)
    assert math.isnan(result.precision)     # no alerts -> precision undefined
    assert result.prevalence == pytest.approx(0.01)


# -----------------------------------------------------------------------------
# Curves and sweeps
# -----------------------------------------------------------------------------
def test_budget_sweep_recall_is_monotonic_in_capacity():
    rng = np.random.default_rng(0)
    y_true = (rng.random(2000) < 0.02).astype(int)
    y_score = rng.random(2000) + y_true * 0.6
    sweep = M.budget_sweep(y_true, y_score)
    assert list(sweep["budget_fraction"]) == sorted(sweep["budget_fraction"])
    assert sweep["recall"].is_monotonic_increasing
    assert sweep["n_alerts"].is_monotonic_increasing


def test_curve_helpers_return_plottable_frames():
    rng = np.random.default_rng(1)
    y_true = (rng.random(500) < 0.05).astype(int)
    y_score = rng.random(500) + y_true * 0.5
    pr = M.pr_curve(y_true, y_score)
    roc = M.roc_curve_points(y_true, y_score)
    assert set(pr.columns) == {"recall", "precision", "threshold"}
    assert set(roc.columns) == {"fpr", "tpr", "threshold"}
    assert len(pr) > 0 and len(roc) > 0
    assert M.pr_curve(np.zeros(10, dtype=int), np.arange(10.0)).empty


# -----------------------------------------------------------------------------
# Multi-class attack typing
# -----------------------------------------------------------------------------
def test_per_class_metrics_reports_every_requested_class():
    labels = ["BRUTE_FORCE", "IMPOSSIBLE_TRAVEL", "LATERAL_MOVEMENT"]
    y_true = ["BRUTE_FORCE", "BRUTE_FORCE", "IMPOSSIBLE_TRAVEL"]
    y_pred = ["BRUTE_FORCE", "IMPOSSIBLE_TRAVEL", "IMPOSSIBLE_TRAVEL"]
    frame = M.per_class_metrics(y_true, y_pred, labels)
    assert list(frame["class"]) == labels
    brute = frame.set_index("class").loc["BRUTE_FORCE"]
    assert brute["precision"] == pytest.approx(1.0)
    assert brute["recall"] == pytest.approx(0.5)
    # A class with no support is still reported, at zero, rather than dropped.
    assert frame.set_index("class").loc["LATERAL_MOVEMENT", "support"] == 0


def test_confusion_frame_is_labelled_truth_by_prediction():
    labels = ["A", "B"]
    frame = M.confusion_frame(["A", "A", "B"], ["A", "B", "B"], labels)
    assert frame.loc["A", "A"] == 1 and frame.loc["A", "B"] == 1
    assert frame.loc["B", "B"] == 1 and frame.loc["B", "A"] == 0


# -----------------------------------------------------------------------------
# Campaign-level detection latency
# -----------------------------------------------------------------------------
def _campaign_frame() -> pd.DataFrame:
    start = pd.Timestamp("2025-02-01 08:00:00")
    rows = []
    for offset in range(4):  # campaign A: caught on its 3rd event (+2h)
        rows.append({"campaign_id": "A", "label": "LOW_AND_SLOW_EXFILTRATION",
                     "timestamp": start + pd.Timedelta(hours=offset)})
    for offset in range(2):  # campaign B: never caught
        rows.append({"campaign_id": "B", "label": "BRUTE_FORCE",
                     "timestamp": start + pd.Timedelta(hours=offset)})
    rows.append({"campaign_id": None, "label": "BENIGN", "timestamp": start})
    return pd.DataFrame(rows)


def test_campaign_detection_latency_and_coverage():
    frame = _campaign_frame()
    alerts = [False, False, True, False, False, False, False]
    result = M.campaign_detection(frame, alerts)
    assert result.n_campaigns == 2 and result.n_detected == 1
    assert result.campaign_recall == pytest.approx(0.5)
    assert result.median_latency_seconds == pytest.approx(2 * 3600)
    assert result.median_events_before_detection == pytest.approx(2)
    caught = result.per_campaign.set_index("campaign_id")
    assert bool(caught.loc["A", "detected"]) and not bool(caught.loc["B", "detected"])


def test_campaign_detection_handles_a_dataset_with_no_campaigns():
    frame = pd.DataFrame(
        {"campaign_id": [None, None], "label": ["BENIGN", "BENIGN"],
         "timestamp": pd.to_datetime(["2025-01-01", "2025-01-02"])}
    )
    result = M.campaign_detection(frame, [False, False])
    assert result.n_campaigns == 0 and math.isnan(result.campaign_recall)


# -----------------------------------------------------------------------------
# Artifact registry
# -----------------------------------------------------------------------------
def test_registry_declares_every_pipeline_output_once():
    keys = {"entities", "events", "features", "profiles", "anomaly_detector",
            "attack_classifier", "alerts", "metrics", "manifest"}
    assert keys <= set(REGISTRY)
    paths = [a.path for a in REGISTRY.values()]
    assert len(paths) == len(set(paths))


def test_registry_reports_missing_artifacts_without_creating_them():
    events = artifact("events")
    assert not events.path.exists()
    assert events in missing_artifacts("events")
    statuses = pipeline_status()
    assert [s.artifact.phase for s in statuses] == sorted(s.artifact.phase for s in statuses)
    assert all(s.human_size() == "-" for s in statuses if not s.exists)


# -----------------------------------------------------------------------------
# Reproducible reporting
# -----------------------------------------------------------------------------
def test_manifest_captures_seed_and_config():
    manifest = RunManifest.capture(notes="unit-test")
    assert manifest.master_seed == 20260725
    assert manifest.config_snapshot["project"]["name"] == "AEGIS"
    assert manifest.packages["scikit-learn"] != "not-installed"
    assert manifest.notes == "unit-test"


def test_metrics_roundtrip_serialises_numpy_and_nan():
    payload = {
        "detection": M.detection_metrics(
            np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]), budget_fraction=0.5
        ).to_dict(),
        "numpy_scalar": np.float64(0.25),
        "undefined": float("nan"),
        "array": np.arange(3),
    }
    path = save_metrics(payload, name="unit_test_roundtrip")
    assert path.exists()

    loaded = load_metrics("unit_test_roundtrip")
    assert loaded["metrics"]["numpy_scalar"] == 0.25
    assert loaded["metrics"]["undefined"] is None      # NaN -> null, not a crash
    assert loaded["metrics"]["array"] == [0, 1, 2]
    assert loaded["metrics"]["detection"]["recall"] == 1.0
    assert loaded["manifest"]["master_seed"] == 20260725
    assert json.dumps(loaded)  # the document is plain JSON

    assert load_metrics("a_run_that_never_happened") is None
    path.unlink()


# -----------------------------------------------------------------------------
# Report figures
# -----------------------------------------------------------------------------
def test_report_figures_render_headlessly_and_save():
    from src.evaluation import plots

    rng = np.random.default_rng(7)
    y_true = (rng.random(800) < 0.03).astype(int)
    y_score = rng.random(800) + y_true * 0.5
    sweep = M.budget_sweep(y_true, y_score)
    matrix = M.confusion_frame(["A", "B", "B"], ["A", "A", "B"], ["A", "B"])

    figures = {
        "pr": plots.pr_curve_figure(y_true, y_score),
        "roc": plots.roc_curve_figure(y_true, y_score),
        "confusion": plots.confusion_matrix_figure(matrix),
        "budget": plots.budget_sweep_figure(sweep, budget_fraction=0.01),
        "scores": plots.score_distribution_figure(y_true, y_score),
    }
    for name, figure in figures.items():
        path = plots.save_figure(figure, f"unit_test_{name}")
        assert path.exists() and path.stat().st_size > 1000, name
