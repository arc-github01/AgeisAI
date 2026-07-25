"""Interactive Plotly charts for the console.

Every chart degrades to :func:`empty_figure` when its data is absent, so the
layout is stable from the first phase to the last.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from . import theme
from .contracts import SEVERITY_ORDER

_TEMPLATE = theme.PLOTLY_TEMPLATE


def _base(fig: go.Figure, *, height: int = 300, title: str | None = None) -> go.Figure:
    theme._register_plotly_template()
    fig.update_layout(template=_TEMPLATE, height=height, title=title)
    return fig


def empty_figure(message: str = "No data yet", *, height: int = 300) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        showarrow=False,
        font=dict(color=theme.MUTED, size=12),
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _base(fig, height=height)


def alert_activity_timeline(
    timeline: pd.DataFrame | None, *, height: int = 320
) -> go.Figure:
    """Stacked alert counts over time, coloured by severity."""
    if timeline is None or timeline.empty:
        return empty_figure("Alert activity timeline appears once alerts are available",
                            height=height)
    pivot = timeline.pivot_table(
        index="timestamp", columns="severity", values="count", aggfunc="sum", fill_value=0
    )
    for severity in SEVERITY_ORDER:
        if severity not in pivot.columns:
            pivot[severity] = 0
    pivot = pivot[[s for s in SEVERITY_ORDER if s in pivot.columns]]

    fig = go.Figure()
    for severity in pivot.columns:
        fig.add_trace(
            go.Bar(
                x=pivot.index,
                y=pivot[severity],
                name=severity,
                marker_color=theme.severity_color(severity),
            )
        )
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title="time")
    fig.update_yaxes(title="alerts", rangemode="tozero")
    return _base(fig, height=height, title="Threat activity over time")


def entity_type_distribution(
    counts: dict[str, int] | None, *, height: int = 280
) -> go.Figure:
    if not counts:
        return empty_figure("Entity distribution appears once entities are registered",
                            height=height)
    labels = list(counts.keys())
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=[counts[k] for k in labels],
            marker=dict(color=theme.ACCENT),
        )
    )
    fig.update_yaxes(rangemode="tozero", title="entities")
    return _base(fig, height=height)


def severity_donut(counts: dict[str, int] | None, *, height: int = 300) -> go.Figure:
    if not counts:
        return empty_figure("Severity mix appears once alerts are scored", height=height)
    labels = list(counts)
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=[counts[k] for k in labels],
            hole=0.62,
            marker=dict(colors=[theme.severity_color(k) for k in labels]),
            textinfo="label+value",
        )
    )
    return _base(fig, height=height)


def attack_distribution(counts: dict[str, int] | None, *, height: int = 300) -> go.Figure:
    if not counts:
        return empty_figure("Threat distribution appears once events are classified",
                            height=height)
    ordered = sorted(counts.items(), key=lambda kv: kv[1])
    fig = go.Figure(
        go.Bar(
            x=[v for _, v in ordered],
            y=[k for k, _ in ordered],
            orientation="h",
            marker=dict(color=[theme.attack_color(k) for k, _ in ordered]),
        )
    )
    return _base(fig, height=height)


def risk_timeline(frame: pd.DataFrame | None, *, height: int = 300) -> go.Figure:
    """Expects columns ``timestamp`` and ``risk_score``."""
    if frame is None or frame.empty:
        return empty_figure("Risk over time appears once the risk engine runs",
                            height=height)
    fig = go.Figure(
        go.Scattergl(
            x=frame["timestamp"],
            y=frame["risk_score"],
            mode="markers",
            marker=dict(size=5, color=frame["risk_score"], colorscale="Turbo",
                        cmin=0, cmax=100, showscale=False),
        )
    )
    fig.update_yaxes(range=[0, 100], title="risk")
    return _base(fig, height=height)


def pr_curve(curve: pd.DataFrame | None, prevalence: float | None = None,
             *, height: int = 320) -> go.Figure:
    if curve is None or curve.empty:
        return empty_figure("Precision-recall curve appears after evaluation", height=height)
    fig = go.Figure(
        go.Scatter(x=curve["recall"], y=curve["precision"], mode="lines",
                   line=dict(color=theme.ACCENT, width=2), name="model")
    )
    if prevalence is not None:
        fig.add_hline(y=prevalence, line=dict(color="#ef4444", dash="dash", width=1),
                      annotation_text=f"random ({prevalence:.2%})",
                      annotation_font=dict(color=theme.MUTED, size=10))
    fig.update_xaxes(title="recall", range=[0, 1])
    fig.update_yaxes(title="precision", range=[0, 1.02])
    return _base(fig, height=height)


def roc_curve(curve: pd.DataFrame | None, *, height: int = 320) -> go.Figure:
    if curve is None or curve.empty:
        return empty_figure("ROC curve appears after evaluation", height=height)
    fig = go.Figure(
        go.Scatter(x=curve["fpr"], y=curve["tpr"], mode="lines",
                   line=dict(color=theme.ACCENT, width=2), name="model")
    )
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                  line=dict(color=theme.MUTED, dash="dash", width=1))
    fig.update_xaxes(title="false positive rate", range=[0, 1])
    fig.update_yaxes(title="true positive rate", range=[0, 1.02])
    return _base(fig, height=height)


def confusion_heatmap(matrix: pd.DataFrame | None, *, height: int = 420) -> go.Figure:
    if matrix is None or matrix.empty:
        return empty_figure("Confusion matrix appears after the classifier is trained",
                            height=height)
    fig = go.Figure(
        go.Heatmap(
            z=matrix.to_numpy(),
            x=[str(c) for c in matrix.columns],
            y=[str(i) for i in matrix.index],
            colorscale="Blues",
            showscale=True,
        )
    )
    fig.update_xaxes(title="predicted")
    fig.update_yaxes(title="actual", autorange="reversed")
    return _base(fig, height=height)


def budget_sweep(sweep: pd.DataFrame | None, budget: float | None = None,
                 *, height: int = 320) -> go.Figure:
    if sweep is None or sweep.empty:
        return empty_figure("Alert-budget sweep appears after evaluation", height=height)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sweep["budget_fraction"], y=sweep["recall"], mode="lines+markers",
                             name="recall", line=dict(color=theme.ACCENT, width=2)))
    fig.add_trace(go.Scatter(x=sweep["budget_fraction"], y=sweep["precision"],
                             mode="lines+markers", name="precision",
                             line=dict(color="#fbbf24", width=2)))
    if budget is not None:
        fig.add_vline(x=budget, line=dict(color=theme.MUTED, dash="dot", width=1),
                      annotation_text=f"budget {budget:.1%}",
                      annotation_font=dict(color=theme.MUTED, size=10))
    fig.update_xaxes(title="alert budget (fraction of events triaged)", type="log")
    fig.update_yaxes(title="score", range=[0, 1.02])
    return _base(fig, height=height)


def contribution_bars(factors: pd.DataFrame | None, *, height: int = 280) -> go.Figure:
    """Explainability: per-factor contribution to the risk score."""
    if factors is None or factors.empty:
        return empty_figure("Contributing factors appear once an alert is selected",
                            height=height)
    ordered = factors.sort_values("contribution")
    fig = go.Figure(
        go.Bar(x=ordered["contribution"], y=ordered["factor"], orientation="h",
               marker=dict(color=theme.ACCENT))
    )
    fig.update_xaxes(title="contribution to risk")
    return _base(fig, height=height)
