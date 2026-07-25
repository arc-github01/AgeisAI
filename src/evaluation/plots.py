"""Static report figures (matplotlib).

Deliberately separate from ``dashboard/charts.py``: these are light-background,
print-ready PNGs for the written report, produced headlessly and saved into
``artifacts/figures/``. The dashboard renders its own interactive dark-theme
Plotly versions of the same quantities.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: figures are files, never windows

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..artifacts import figures_dir  # noqa: E402
from . import metrics as M  # noqa: E402

_STYLE = {
    "figure.figsize": (6.4, 4.4),
    "figure.dpi": 140,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 9,
}
_ACCENT = "#0f4c81"
_WARN = "#c0392b"


def save_figure(fig: plt.Figure, name: str) -> Path:
    path = figures_dir() / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def pr_curve_figure(y_true, y_score, *, title: str = "Precision-Recall") -> plt.Figure:
    curve = M.pr_curve(y_true, y_score)
    pr_auc, _ = M.ranking_metrics(y_true, y_score)
    prevalence = float(np.mean(np.asarray(y_true, dtype=float)))
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots()
        if not curve.empty:
            ax.plot(curve["recall"], curve["precision"], color=_ACCENT, lw=1.8)
        ax.axhline(
            prevalence,
            color=_WARN,
            ls="--",
            lw=1.0,
            label=f"random baseline ({prevalence:.3%})",
        )
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{title}  (PR-AUC = {pr_auc:.3f})")
        ax.legend(loc="upper right", frameon=False)
    return fig


def roc_curve_figure(y_true, y_score, *, title: str = "ROC") -> plt.Figure:
    curve = M.roc_curve_points(y_true, y_score)
    _, roc_auc = M.ranking_metrics(y_true, y_score)
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots()
        if not curve.empty:
            ax.plot(curve["fpr"], curve["tpr"], color=_ACCENT, lw=1.8)
        ax.plot([0, 1], [0, 1], color="#999999", ls="--", lw=1.0, label="chance")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{title}  (ROC-AUC = {roc_auc:.3f})")
        ax.legend(loc="lower right", frameon=False)
    return fig


def confusion_matrix_figure(
    matrix: pd.DataFrame, *, title: str = "Confusion matrix", normalize: bool = True
) -> plt.Figure:
    values = matrix.to_numpy(dtype=float)
    if normalize:
        row_sums = values.sum(axis=1, keepdims=True)
        shown = np.divide(values, row_sums, out=np.zeros_like(values), where=row_sums > 0)
    else:
        shown = values
    with plt.rc_context({**_STYLE, "axes.grid": False, "figure.figsize": (7.0, 6.0)}):
        fig, ax = plt.subplots()
        image = ax.imshow(shown, cmap="Blues", vmin=0, vmax=shown.max() or 1)
        ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(matrix.index)), matrix.index)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(title)
        threshold = (shown.max() or 1) / 2
        for i in range(shown.shape[0]):
            for j in range(shown.shape[1]):
                text = f"{shown[i, j]:.2f}" if normalize else f"{int(values[i, j])}"
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if shown[i, j] > threshold else "#222222",
                )
        fig.colorbar(image, ax=ax, shrink=0.8)
    return fig


def budget_sweep_figure(
    sweep: pd.DataFrame, *, budget_fraction: float | None = None
) -> plt.Figure:
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots()
        ax.plot(sweep["budget_fraction"], sweep["recall"], marker="o", lw=1.6,
                color=_ACCENT, label="recall")
        ax.plot(sweep["budget_fraction"], sweep["precision"], marker="s", lw=1.6,
                color=_WARN, label="precision")
        if budget_fraction is not None:
            ax.axvline(budget_fraction, color="#555555", ls=":", lw=1.2,
                       label=f"analyst budget ({budget_fraction:.1%})")
        ax.set_xscale("log")
        ax.set_xlabel("Alert budget (fraction of events triaged)")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.02)
        ax.set_title("Detection quality vs analyst capacity")
        ax.legend(frameon=False)
    return fig


def score_distribution_figure(
    y_true, y_score, *, title: str = "Risk score distribution"
) -> plt.Figure:
    truth = np.asarray(y_true).astype(bool)
    scores = np.asarray(y_score, dtype=float)
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots()
        bins = np.linspace(scores.min(), scores.max(), 60)
        ax.hist(scores[~truth], bins=bins, color="#9bb7d4", label="benign", log=True)
        ax.hist(scores[truth], bins=bins, color=_WARN, alpha=0.85, label="malicious",
                log=True)
        ax.set_xlabel("Score")
        ax.set_ylabel("Event count (log)")
        ax.set_title(title)
        ax.legend(frameon=False)
    return fig


__all__ = [
    "budget_sweep_figure",
    "confusion_matrix_figure",
    "pr_curve_figure",
    "roc_curve_figure",
    "save_figure",
    "score_distribution_figure",
]
