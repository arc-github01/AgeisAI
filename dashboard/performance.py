"""Model Performance: evaluation results from the real metrics artifact only."""

from __future__ import annotations

import streamlit as st

from . import charts
from .components import KPI, kpi_row, page_header, panel_title
from .data_provider import DashboardDataProvider
from .state import DashboardContext

TITLE = "Model Performance"
SUBTITLE = "Detection quality under extreme class imbalance and a fixed alert budget"


def _fmt(value: float | None) -> str | None:
    if value is None:
        return None
    return f"{value:.3f}"


def _fmt_pct(value: float | None) -> str | None:
    if value is None:
        return None
    return f"{value:.1%}"


def _pending_notice() -> None:
    st.warning(
        "**Evaluation not yet available.** Metrics on this page are loaded only from "
        "`artifacts/metrics/latest.json`, produced by `python -m src.evaluation` "
        "(Phase 12). No model performance numbers are fabricated for the dashboard."
    )


def _render_provenance(provider: DashboardDataProvider) -> None:
    manifest = provider.get_performance_manifest()
    if not manifest:
        return
    with st.expander("Run provenance"):
        st.json(
            {
                "run_id": manifest.get("run_id"),
                "created_at": manifest.get("created_at"),
                "master_seed": manifest.get("master_seed"),
                "git_commit": manifest.get("git_commit"),
                "packages": manifest.get("packages"),
                "notes": manifest.get("notes"),
            }
        )


def render(ctx: DashboardContext) -> None:
    provider = DashboardDataProvider.from_context(ctx)
    budget = float(ctx.cfg["alerting.budget_fraction"])
    kpis = provider.get_performance_kpis()

    page_header(TITLE, SUBTITLE)

    if not provider.has_evaluation_metrics():
        _pending_notice()
        kpi_row(
            [
                KPI("PR-AUC", None, foot="primary metric"),
                KPI("ROC-AUC", None),
                KPI(f"Recall @ {budget:.0%}", None, foot="within analyst budget"),
                KPI("Precision", None),
                KPI("False positive rate", None),
            ]
        )
        st.write("")
        st.caption(
            "Accuracy is intentionally absent: at ~1% prevalence, always predicting "
            "'normal' scores ~99% while detecting nothing."
        )
        _empty_layout(budget)
        return

    st.caption(f"Data source: evaluation artifact · {provider.get_performance_manifest().get('run_id', 'latest')}")
    kpi_row(
        [
            KPI("PR-AUC", _fmt(kpis.pr_auc), foot="primary metric"),
            KPI("ROC-AUC", _fmt(kpis.roc_auc)),
            KPI(
                f"Recall @ {budget:.0%}",
                _fmt(kpis.recall_at_budget),
                foot=kpis.operating_point or "within analyst budget",
            ),
            KPI("Precision", _fmt(kpis.precision)),
            KPI("False positive rate", _fmt(kpis.fpr)),
        ]
    )
    st.write("")
    st.caption(
        "Accuracy is intentionally absent: at ~1% prevalence, always predicting "
        "'normal' scores ~99% while detecting nothing."
    )

    left, right = st.columns(2)
    with left:
        panel_title("Precision-recall")
        st.plotly_chart(
            charts.pr_curve(provider.get_pr_curve(), kpis.prevalence),
            width="stretch",
        )
    with right:
        panel_title("ROC")
        st.plotly_chart(charts.roc_curve(provider.get_roc_curve()), width="stretch")

    panel_title("Detection quality vs analyst capacity")
    st.plotly_chart(
        charts.budget_sweep(provider.get_budget_sweep(), budget),
        width="stretch",
    )

    left, right = st.columns([1.2, 1])
    with left:
        panel_title("Attack-type confusion matrix")
        st.plotly_chart(
            charts.confusion_heatmap(provider.get_confusion_matrix()),
            width="stretch",
        )
    with right:
        panel_title("Per-attack metrics")
        per_class = provider.get_per_class_metrics()
        if per_class is not None and not per_class.empty:
            st.dataframe(per_class, width="stretch", hide_index=True)
        else:
            st.caption("Per-attack precision / recall / F1 not present in this run.")

    campaign = provider.get_campaign_performance()
    if campaign is not None:
        panel_title("Campaign-level detection")
        kpi_row(
            [
                KPI("Campaigns", campaign.n_campaigns),
                KPI("Detected", campaign.n_detected),
                KPI("Campaign recall", _fmt_pct(campaign.campaign_recall)),
                KPI(
                    "Median latency",
                    f"{campaign.median_latency_seconds / 3600:.1f}h"
                    if campaign.median_latency_seconds is not None
                    else None,
                    foot="time to first alert",
                ),
                KPI(
                    "Events before detection",
                    campaign.median_events_before_detection,
                    foot="median across caught campaigns",
                    fmt="{:.0f}",
                ),
            ]
        )

    _render_provenance(provider)


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
