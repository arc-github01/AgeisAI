"""Model Performance: the evaluation story, shown to the analyst and the judge.

Only reads metrics written by ``src.evaluation``. If a run has not happened,
this page says so - it never computes a flattering number on the fly.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from . import charts
from .components import KPI, awaiting_data, kpi_row, page_header, panel_title
from .state import DashboardContext

TITLE = "Model Performance"
SUBTITLE = "Detection quality under extreme class imbalance and a fixed alert budget"


def render(ctx: DashboardContext) -> None:
    page_header(TITLE, SUBTITLE)

    metrics = ctx.metrics()
    manifest = ctx.manifest()
    budget = ctx.cfg["alerting.budget_fraction"]

    detection = (metrics or {}).get("detection", {})
    kpi_row(
        [
            KPI("PR-AUC", _fmt(detection.get("pr_auc")), foot="primary metric"),
            KPI("ROC-AUC", _fmt(detection.get("roc_auc"))),
            KPI(f"Recall @ {budget:.0%}", _fmt(detection.get("recall")),
                foot="within analyst budget"),
            KPI("Precision", _fmt(detection.get("precision"))),
            KPI("False positive rate", _fmt(detection.get("fpr"))),
        ]
    )
    st.write("")

    st.caption(
        "Accuracy is intentionally absent: at ~1% prevalence, always predicting "
        "'normal' scores ~99% while detecting nothing."
    )

    if awaiting_data(
        ctx,
        "metrics",
        note="Every figure on this page is read from artifacts/metrics/latest.json, "
        "which is stamped with the seed, config snapshot and package versions of the "
        "run that produced it.",
    ):
        _empty_layout(budget)
        return

    left, right = st.columns(2)
    with left:
        panel_title("Precision-recall")
        st.plotly_chart(
            charts.pr_curve(_frame(metrics, "pr_curve"), detection.get("prevalence")),
            width="stretch",
        )
    with right:
        panel_title("ROC")
        st.plotly_chart(charts.roc_curve(_frame(metrics, "roc_curve")),
                        width="stretch")

    panel_title("Detection quality vs analyst capacity")
    st.plotly_chart(charts.budget_sweep(_frame(metrics, "budget_sweep"), budget),
                    width="stretch")

    left, right = st.columns([1.2, 1])
    with left:
        panel_title("Attack-type confusion matrix")
        confusion = metrics.get("confusion_matrix")
        matrix = pd.DataFrame(confusion) if confusion else None
        st.plotly_chart(charts.confusion_heatmap(matrix), width="stretch")
    with right:
        panel_title("Per-attack metrics")
        per_class = _frame(metrics, "per_class")
        if per_class is not None:
            st.dataframe(per_class, width="stretch", hide_index=True)
        else:
            st.caption("Per-attack precision / recall / F1 appears after classification.")

    if manifest:
        with st.expander("Run provenance"):
            st.json(
                {
                    "run_id": manifest.get("run_id"),
                    "created_at": manifest.get("created_at"),
                    "master_seed": manifest.get("master_seed"),
                    "git_commit": manifest.get("git_commit"),
                    "packages": manifest.get("packages"),
                }
            )


def _empty_layout(budget: float) -> None:
    left, right = st.columns(2)
    with left:
        panel_title("Precision-recall")
        st.plotly_chart(charts.pr_curve(None), width="stretch")
    with right:
        panel_title("ROC")
        st.plotly_chart(charts.roc_curve(None), width="stretch")
    panel_title("Detection quality vs analyst capacity")
    st.plotly_chart(charts.budget_sweep(None, budget), width="stretch")
    panel_title("Attack-type confusion matrix")
    st.plotly_chart(charts.confusion_heatmap(None), width="stretch")


def _frame(metrics: dict | None, key: str) -> pd.DataFrame | None:
    records = (metrics or {}).get(key)
    if not records:
        return None
    frame = pd.DataFrame(records)
    return frame if not frame.empty else None


def _fmt(value) -> str | None:
    if value is None:
        return None
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)
