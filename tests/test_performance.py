"""Tests for Model Performance page and evaluation artifact loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from dashboard.data_provider import DashboardDataProvider
from src.evaluation import metrics as M
from src.evaluation.report import save_metrics
from src.paths import PROJECT_ROOT
from src.schema import ATTACK_CLASSES

APP = str(PROJECT_ROOT / "app.py")


def _sample_metrics_payload() -> dict:
    rng = np.random.default_rng(42)
    y_true = (rng.random(2000) < 0.02).astype(int)
    y_score = rng.random(2000) + y_true * 0.55
    labels = list(ATTACK_CLASSES)[:4]
    y_class_true = rng.choice(labels, size=400)
    y_class_pred = y_class_true.copy()
    y_class_pred[:40] = rng.choice(labels, size=40)

    return {
        "detection": M.detection_metrics(y_true, y_score, budget_fraction=0.01).to_dict(),
        "pr_curve": M.pr_curve(y_true, y_score).to_dict("records"),
        "roc_curve": M.roc_curve_points(y_true, y_score).to_dict("records"),
        "budget_sweep": M.budget_sweep(y_true, y_score).to_dict("records"),
        "confusion_matrix": {
            "index": labels,
            "columns": labels,
            "values": M.confusion_frame(y_class_true, y_class_pred, labels).to_numpy().tolist(),
        },
        "per_class": M.per_class_metrics(y_class_true, y_class_pred, labels).to_dict("records"),
        "campaign_detection": {
            "n_campaigns": 12,
            "n_detected": 9,
            "campaign_recall": 0.75,
            "median_latency_seconds": 7200.0,
            "median_events_before_detection": 3.0,
        },
    }


def test_provider_without_artifact_reports_no_metrics():
    provider = DashboardDataProvider(mode="mock")
    assert not provider.has_evaluation_metrics()
    kpis = provider.get_performance_kpis()
    assert kpis.pr_auc is None
    assert provider.get_pr_curve() is None
    assert provider.get_campaign_performance() is None


def test_provider_loads_saved_evaluation_artifact():
    save_metrics(_sample_metrics_payload(), name="unit_test_performance")
    provider = DashboardDataProvider(mode="mock")
    assert provider.has_evaluation_metrics()
    kpis = provider.get_performance_kpis()
    assert kpis.pr_auc is not None and kpis.pr_auc > 0
    assert provider.get_pr_curve() is not None
    assert provider.get_confusion_matrix() is not None
    per_class = provider.get_per_class_metrics()
    assert per_class is not None and "class" in per_class.columns
    campaign = provider.get_campaign_performance()
    assert campaign is not None and campaign.n_campaigns == 12


def test_performance_page_pending_without_evaluation():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value("Model Performance").run()
    assert not at.exception
    body = " ".join(block.value for block in at.markdown)
    assert "PR-AUC" in body
    assert 'class="value pending">--' in body
    assert at.warning, "expected pending evaluation warning"
    assert "Evaluation not yet available" in at.warning[0].value
    captions = " ".join(block.value for block in at.caption)
    assert "Accuracy is intentionally absent" in captions


@pytest.mark.retain_metrics_artifact
def test_performance_page_renders_saved_metrics():
    save_metrics(_sample_metrics_payload(), name="ui_test_performance")
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value("Model Performance").run()
    assert not at.exception
    body = " ".join(block.value for block in at.markdown)
    assert "PR-AUC" in body
    assert 'class="value pending">--' not in body
    assert at.dataframe, "expected per-attack metrics table"


def test_performance_page_does_not_import_generator_modules():
    import dashboard.performance as page

    text = Path(page.__file__).read_text(encoding="utf-8")
    assert "data_generator" not in text
    assert "mock_data" not in text
