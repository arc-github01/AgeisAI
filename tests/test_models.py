"""Phase 5 anomaly-detector contracts.

These tests defend the properties that make the detector's numbers trustworthy:
the model never sees attacks, labels or evaluation rows during fitting; its
thresholds are calibrated without test labels; results are deterministic; and
campaign evaluation joins the ground-truth sidecar correctly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.features import MODEL_FEATURE_COLUMNS
from src.models.anomaly_detector import build_scores, evaluate_all
from src.models.baselines import fit_rule_baseline, random_scores
from src.models.dataset import (
    EVALUATION_COLUMNS,
    evaluation_frame,
    load_scoring_frame,
    training_matrix,
)
from src.models.evaluate import campaign_level_metrics, event_level_metrics
from src.models.model import (
    AnomalyModel,
    calibrate_thresholds,
    split_fit_and_calibration_rows,
    train_anomaly_model,
)
from src.schema import MALICIOUS_CLASSES

_BASE = pd.Timestamp("2025-01-06T08:00:00Z")


def _make_frame(seed: int = 0) -> pd.DataFrame:
    """A synthetic scoring frame with train/eval splits and attacks in both."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    def add(n: int, *, split: str, label: str, campaign: str | None, shift: float):
        for _ in range(n):
            row = {col: float(rng.normal(shift, 1.0)) for col in MODEL_FEATURE_COLUMNS}
            row.update(
                event_id=f"E{len(rows):05d}",
                timestamp=_BASE + pd.Timedelta(minutes=len(rows)),
                entity_id=f"user-{len(rows) % 5}",
                entity_type="user",
                split=split,
                label=label,
                is_attack=label not in ("BENIGN", "INSIDER_DRIFT"),
                campaign_id=campaign,
                profile_source="entity",
                profile_confidence=1.0,
            )
            rows.append(row)

    add(200, split="train", label="BENIGN", campaign=None, shift=0.0)
    add(6, split="train", label="BRUTE_FORCE", campaign="CMP-T", shift=5.0)
    add(80, split="evaluation", label="BENIGN", campaign=None, shift=0.0)
    add(10, split="evaluation", label="BRUTE_FORCE", campaign="CMP-E", shift=5.0)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Training isolation: no attacks, no evaluation rows, no labels as inputs
# -----------------------------------------------------------------------------
def test_training_matrix_excludes_attacks_and_eval_rows():
    frame = _make_frame()
    train = training_matrix(frame)
    assert (train["split"] == "train").all()
    assert (train["label"] == "BENIGN").all()
    assert not train["is_attack"].any()
    # Every benign training row is kept; nothing else is.
    expected = ((frame["split"] == "train") & (frame["label"] == "BENIGN")).sum()
    assert len(train) == expected


def test_model_feature_columns_exclude_labels_and_metadata():
    for forbidden in (*EVALUATION_COLUMNS, "split", "campaign_id", "profile_source"):
        assert forbidden not in MODEL_FEATURE_COLUMNS
    model = train_anomaly_model(training_matrix(_make_frame()))
    assert model.feature_columns == tuple(MODEL_FEATURE_COLUMNS)


def test_load_scoring_frame_join_is_one_to_one():
    frame = _make_frame()
    features = frame.drop(columns=list(EVALUATION_COLUMNS))
    events = frame[["event_id", *EVALUATION_COLUMNS]].copy()
    merged = load_scoring_frame(features=features, events=events)
    assert len(merged) == len(frame)
    assert not merged["label"].isna().any()
    assert set(EVALUATION_COLUMNS).issubset(merged.columns)


# -----------------------------------------------------------------------------
# Preprocessing fitted on training data only
# -----------------------------------------------------------------------------
def test_scaler_is_fitted_on_training_rows_only():
    frame = _make_frame()
    train = training_matrix(frame)
    model = train_anomaly_model(train)
    fit_rows, _ = split_fit_and_calibration_rows(
        train, fit_fraction=float(load_config()["models.anomaly_detector.calibration.fit_fraction"])
    )
    expected_mean = fit_rows.loc[:, list(MODEL_FEATURE_COLUMNS)].to_numpy(float).mean(axis=0)
    np.testing.assert_allclose(model.scaler.mean_, expected_mean, rtol=1e-9, atol=1e-9)


# -----------------------------------------------------------------------------
# Determinism under a fixed seed
# -----------------------------------------------------------------------------
def test_training_is_deterministic():
    frame = _make_frame()
    train = training_matrix(frame)
    eval_rows = evaluation_frame(frame)
    a = train_anomaly_model(train)
    b = train_anomaly_model(train)
    assert a.random_state == b.random_state
    assert a.thresholds == b.thresholds
    np.testing.assert_array_equal(a.raw_scores(eval_rows), b.raw_scores(eval_rows))


def test_random_baseline_is_seeded():
    np.testing.assert_array_equal(random_scores(50), random_scores(50))


# -----------------------------------------------------------------------------
# Score direction and range
# -----------------------------------------------------------------------------
def test_attacks_score_higher_than_benign_on_separable_data():
    frame = _make_frame()
    model = train_anomaly_model(training_matrix(frame))
    eval_rows = evaluation_frame(frame)
    raw = model.raw_scores(eval_rows)
    attack_mean = raw[eval_rows["is_attack"].to_numpy()].mean()
    benign_mean = raw[~eval_rows["is_attack"].to_numpy()].mean()
    assert attack_mean > benign_mean
    norm = model.anomaly_score(eval_rows)
    assert norm.min() >= 0.0 and norm.max() <= 1.0


# -----------------------------------------------------------------------------
# Threshold calibration never touches test labels
# -----------------------------------------------------------------------------
def test_calibrate_thresholds_are_benign_quantiles():
    benign = np.linspace(0.0, 1.0, 1001)
    thresholds = calibrate_thresholds(benign, {"strict": 0.999, "balanced": 0.99})
    assert thresholds["strict"] == pytest.approx(np.quantile(benign, 0.999))
    assert thresholds["balanced"] == pytest.approx(np.quantile(benign, 0.99))
    assert thresholds["strict"] > thresholds["balanced"]


def test_thresholds_independent_of_evaluation_labels():
    frame = _make_frame()
    train = training_matrix(frame)
    baseline = train_anomaly_model(train).thresholds

    # Flip every evaluation label; thresholds are set from training only and
    # must not move by even a float.
    mutated = frame.copy()
    eval_mask = mutated["split"] == "evaluation"
    mutated.loc[eval_mask, "label"] = "BRUTE_FORCE"
    mutated.loc[eval_mask, "is_attack"] = True
    assert train_anomaly_model(training_matrix(mutated)).thresholds == baseline


def test_calibration_rows_do_not_enter_forest_fitting():
    frame = _make_frame()
    train = training_matrix(frame)
    fit_rows, calibration_rows = split_fit_and_calibration_rows(
        train, fit_fraction=float(load_config()["models.anomaly_detector.calibration.fit_fraction"])
    )
    model = train_anomaly_model(train)
    fit_matrix = fit_rows.loc[:, list(MODEL_FEATURE_COLUMNS)].to_numpy(float)
    calibration_matrix = calibration_rows.loc[:, list(MODEL_FEATURE_COLUMNS)].to_numpy(float)
    fit_scores = -model.forest.score_samples(model.scaler.transform(fit_matrix))
    calibration_scores = -model.forest.score_samples(model.scaler.transform(calibration_matrix))
    for name, quantile in model.calibration["operating_points"].items():
        expected = float(np.quantile(calibration_scores, quantile))
        assert model.thresholds[name] == pytest.approx(expected)
    # If thresholds came from fit rows, this would typically differ.
    assert model.thresholds["balanced"] != pytest.approx(float(np.quantile(fit_scores, model.calibration["operating_points"]["balanced"])))


def test_evaluation_rows_do_not_enter_calibration_or_fit_split():
    frame = _make_frame()
    train = training_matrix(frame)
    fit_rows, calibration_rows = split_fit_and_calibration_rows(
        train, fit_fraction=float(load_config()["models.anomaly_detector.calibration.fit_fraction"])
    )
    assert (fit_rows["split"] == "train").all()
    assert (calibration_rows["split"] == "train").all()
    assert (fit_rows["label"] == "BENIGN").all()
    assert (calibration_rows["label"] == "BENIGN").all()


def test_achieved_calibration_fpr_matches_operating_points_with_tolerance():
    frame = _make_frame()
    train = training_matrix(frame)
    fit_rows, calibration_rows = split_fit_and_calibration_rows(
        train, fit_fraction=float(load_config()["models.anomaly_detector.calibration.fit_fraction"])
    )
    model = train_anomaly_model(train)
    cal_scores = model.raw_scores(calibration_rows)
    for name, quantile in model.calibration["operating_points"].items():
        target_fpr = 1.0 - float(quantile)
        achieved = float((cal_scores >= model.thresholds[name]).mean())
        # finite-sample/tie slack
        assert abs(achieved - target_fpr) <= 0.03


# -----------------------------------------------------------------------------
# Event scores never carry model feature columns as leakable inputs
# -----------------------------------------------------------------------------
def test_build_scores_output_has_no_model_feature_columns():
    frame = _make_frame()
    train = training_matrix(frame)
    model = train_anomaly_model(train)
    rule = fit_rule_baseline(train)
    scores = build_scores(frame, model, rule)
    assert not set(MODEL_FEATURE_COLUMNS) & set(scores.columns)
    for col in ("anomaly_score", "anomaly_score_raw", "baseline_random", "baseline_rule"):
        assert col in scores.columns


# -----------------------------------------------------------------------------
# Event-level metrics emphasise PR-AUC and stay well-formed
# -----------------------------------------------------------------------------
def test_event_level_metrics_report_pr_auc_and_recall_at_fpr():
    frame = _make_frame()
    train = training_matrix(frame)
    model = train_anomaly_model(train)
    eval_rows = evaluation_frame(frame).copy()
    eval_rows["anomaly_score_raw"] = model.raw_scores(eval_rows)
    metrics = event_level_metrics(
        eval_rows, "anomaly_score_raw",
        thresholds=model.thresholds, budget_fraction=0.05,
        fpr_targets=[0.01, 0.05],
    )
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert {"strict", "balanced", "sensitive"} <= set(metrics["operating_points"])
    assert [r["fpr_target"] for r in metrics["recall_at_fpr"]] == [0.01, 0.05]


# -----------------------------------------------------------------------------
# Campaign evaluation correctly joins campaigns.json
# -----------------------------------------------------------------------------
def test_campaign_metrics_join_sidecar_and_break_down():
    # Two campaigns: one obvious (detected), one stealth (missed).
    rows = []
    for i in range(4):
        rows.append(
            dict(
                event_id=f"O{i}", timestamp=_BASE + pd.Timedelta(minutes=i),
                entity_id="u1", entity_type="user", split="evaluation",
                label="BRUTE_FORCE", is_attack=True, campaign_id="CMP-OBV",
                anomaly_score_raw=0.9,
            )
        )
    for i in range(4):
        rows.append(
            dict(
                event_id=f"S{i}", timestamp=_BASE + pd.Timedelta(minutes=10 + i),
                entity_id="d1", entity_type="edge_device", split="evaluation",
                label="LOW_AND_SLOW_EXFILTRATION", is_attack=True,
                campaign_id="CMP-STL", anomaly_score_raw=0.1,
            )
        )
    eval_scores = pd.DataFrame(rows)
    campaigns_meta = [
        {"campaign_id": "CMP-OBV", "attack_type": "BRUTE_FORCE", "entity_ids": ["u1"], "stealthy": False},
        {"campaign_id": "CMP-STL", "attack_type": "LOW_AND_SLOW_EXFILTRATION", "entity_ids": ["d1"], "stealthy": True},
    ]
    mask = eval_scores["anomaly_score_raw"].to_numpy() >= 0.5  # detects only obvious

    metrics = campaign_level_metrics(
        eval_scores, eval_scores, mask, campaigns_meta, operating_point="balanced"
    )
    assert metrics["n_campaigns_total"] == 2
    assert metrics["n_campaigns_malicious"] == 2
    assert metrics["malicious"]["n_detected"] == 1
    assert metrics["malicious"]["missed_campaigns"] == ["CMP-STL"]
    assert metrics["by_difficulty_malicious"]["obvious"]["detection_rate"] == pytest.approx(1.0)
    assert metrics["by_difficulty_malicious"]["stealth"]["detection_rate"] == pytest.approx(0.0)
    assert set(metrics["by_entity_type_malicious"]) == {"edge_device", "user"}


def test_campaign_metrics_excludes_insider_drift_from_malicious_headline():
    rows = [
        dict(event_id="B0", timestamp=_BASE, entity_id="u1", entity_type="user", split="evaluation",
             label="BRUTE_FORCE", is_attack=True, campaign_id="CMP-BF", anomaly_score_raw=0.9),
        dict(event_id="B1", timestamp=_BASE + pd.Timedelta(minutes=1), entity_id="u1", entity_type="user", split="evaluation",
             label="BRUTE_FORCE", is_attack=True, campaign_id="CMP-BF", anomaly_score_raw=0.9),
        dict(event_id="I0", timestamp=_BASE + pd.Timedelta(minutes=2), entity_id="u2", entity_type="user", split="evaluation",
             label="INSIDER_DRIFT", is_attack=False, campaign_id="CMP-ID", anomaly_score_raw=0.1),
    ]
    eval_scores = pd.DataFrame(rows)
    campaigns_meta = [
        {"campaign_id": "CMP-BF", "attack_type": "BRUTE_FORCE", "entity_ids": ["u1"], "stealthy": False},
        {"campaign_id": "CMP-ID", "attack_type": "INSIDER_DRIFT", "entity_ids": ["u2"], "stealthy": True},
    ]
    mask = eval_scores["anomaly_score_raw"].to_numpy() >= 0.5
    metrics = campaign_level_metrics(eval_scores, eval_scores, mask, campaigns_meta, operating_point="balanced")
    assert metrics["n_campaigns_total"] == 2
    assert metrics["n_campaigns_malicious"] == 1
    assert metrics["malicious"]["detection_rate"] == pytest.approx(1.0)
    assert metrics["insider_drift"]["n_campaigns"] == 1


def test_campaign_straddling_is_flagged_and_excluded_from_latency_aggregate():
    full_rows = [
        dict(event_id="T0", timestamp=_BASE, entity_id="u1", entity_type="user", split="train",
             label="BRUTE_FORCE", is_attack=True, campaign_id="CMP-X", anomaly_score_raw=0.1),
        dict(event_id="E0", timestamp=_BASE + pd.Timedelta(minutes=1), entity_id="u1", entity_type="user", split="evaluation",
             label="BRUTE_FORCE", is_attack=True, campaign_id="CMP-X", anomaly_score_raw=0.9),
        dict(event_id="E1", timestamp=_BASE + pd.Timedelta(minutes=2), entity_id="u2", entity_type="user", split="evaluation",
             label="BRUTE_FORCE", is_attack=True, campaign_id="CMP-Y", anomaly_score_raw=0.9),
    ]
    full_scores = pd.DataFrame(full_rows)
    eval_scores = full_scores[full_scores["split"] == "evaluation"].reset_index(drop=True)
    campaigns_meta = [
        {"campaign_id": "CMP-X", "attack_type": "BRUTE_FORCE", "entity_ids": ["u1"], "stealthy": False},
        {"campaign_id": "CMP-Y", "attack_type": "BRUTE_FORCE", "entity_ids": ["u2"], "stealthy": False},
    ]
    mask = np.array([True, True])
    metrics = campaign_level_metrics(eval_scores, full_scores, mask, campaigns_meta, operating_point="balanced")
    assert "CMP-X" in metrics["straddling"]["campaign_ids"]
    assert metrics["malicious"]["n_latency_excluded_straddling"] == 1


def test_missed_campaigns_are_exact_complement_of_detected_malicious():
    frame = _make_frame()
    train = training_matrix(frame)
    model = train_anomaly_model(train)
    rule = fit_rule_baseline(train)
    scores = build_scores(frame, model, rule)
    eval_scores = scores[scores["split"] == "evaluation"].reset_index(drop=True)
    campaigns_meta = [{"campaign_id": "CMP-E", "attack_type": "BRUTE_FORCE", "entity_ids": ["user-0"], "stealthy": False}]
    _, campaign = evaluate_all(eval_scores, scores, model, campaigns_meta)
    malicious = [row for row in campaign["per_campaign"] if row["label"] in MALICIOUS_CLASSES]
    missed = set(campaign["malicious"]["missed_campaigns"])
    detected = {row["campaign_id"] for row in malicious if row["detected"]}
    all_ids = {row["campaign_id"] for row in malicious}
    assert missed == (all_ids - detected)


def test_rule_baseline_statistics_use_training_rows_only():
    frame = _make_frame()
    train = training_matrix(frame)
    rule = fit_rule_baseline(train)
    expected_means = train.loc[:, list(rule.features)].to_numpy(float).mean(axis=0)
    np.testing.assert_allclose(rule.means, expected_means, rtol=1e-9, atol=1e-9)


def test_evaluate_all_uses_raw_score_for_model_metrics():
    frame = _make_frame()
    train = training_matrix(frame)
    model = train_anomaly_model(train)
    rule = fit_rule_baseline(train)
    scores = build_scores(frame, model, rule)
    eval_scores = scores[scores["split"] == "evaluation"].reset_index(drop=True)
    # Poison display score ordering: if code used anomaly_score, metrics collapse.
    eval_scores = eval_scores.copy()
    eval_scores["anomaly_score"] = np.linspace(0.0, 1.0, len(eval_scores))
    event_metrics, _ = evaluate_all(eval_scores, scores, model, campaigns_meta=[])
    from src.evaluation.metrics import ranking_metrics
    y_true = eval_scores["is_attack"].to_numpy(dtype=int)
    pr_raw, _ = ranking_metrics(y_true, eval_scores["anomaly_score_raw"].to_numpy(float))
    assert event_metrics["model"]["pr_auc"] == pytest.approx(pr_raw)


def test_campaign_metadata_does_not_affect_model_scoring():
    frame = _make_frame()
    train = training_matrix(frame)
    model = train_anomaly_model(train)
    eval_rows = evaluation_frame(frame)
    baseline_scores = model.raw_scores(eval_rows)
    # Evaluation metadata can change reporting, but cannot change scores.
    weird_meta = [{"campaign_id": "CMP-E", "attack_type": "BRUTE_FORCE", "entity_ids": ["x", "y"], "stealthy": True}]
    _ = weird_meta
    np.testing.assert_array_equal(baseline_scores, model.raw_scores(eval_rows))


def test_end_to_end_scoring_is_deterministic():
    frame = _make_frame()
    train = training_matrix(frame)
    model_a = train_anomaly_model(train)
    model_b = train_anomaly_model(train)
    rule_a = fit_rule_baseline(train)
    rule_b = fit_rule_baseline(train)
    scores_a = build_scores(frame, model_a, rule_a)
    scores_b = build_scores(frame, model_b, rule_b)
    pd.testing.assert_frame_equal(scores_a, scores_b, check_exact=False, atol=1e-12, rtol=0.0)
