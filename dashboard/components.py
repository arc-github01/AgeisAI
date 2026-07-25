"""Reusable UI atoms for the AEGIS console.

The most important one is :func:`awaiting_data`. Rather than inventing numbers
for a component that does not exist yet, a page states exactly which artifact is
missing, which phase produces it and which command creates it. The shell is
therefore both a demo of the final layout and an accurate build status board.
"""

from __future__ import annotations

import html
from dataclasses import dataclass

import streamlit as st

from src.artifacts import Artifact
from src.schema import Severity

from . import theme
from .state import DashboardContext


@dataclass(frozen=True)
class KPI:
    label: str
    value: str | int | float | None = None
    foot: str = ""
    fmt: str = "{:,}"

    def rendered(self) -> tuple[str, bool]:
        if self.value is None:
            return "--", True
        if isinstance(self.value, (int, float)):
            return self.fmt.format(self.value), False
        return str(self.value), False


def page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="aegis-page-title">{html.escape(title)}</div>'
        f'<div class="aegis-page-sub">{html.escape(subtitle)}</div>'
        f'<div class="aegis-rule"></div>',
        unsafe_allow_html=True,
    )


def kpi_row(items: list[KPI]) -> None:
    for column, item in zip(st.columns(len(items)), items):
        value, pending = item.rendered()
        with column:
            st.markdown(
                f'<div class="aegis-kpi">'
                f'<div class="label">{html.escape(item.label)}</div>'
                f'<div class="value{" pending" if pending else ""}">{html.escape(value)}</div>'
                f'<div class="foot">{html.escape(item.foot)}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )


def panel_title(title: str) -> None:
    st.markdown(
        f'<div class="aegis-panel" style="padding:.55rem .9rem;margin-bottom:.5rem">'
        f"<h4>{html.escape(title)}</h4></div>",
        unsafe_allow_html=True,
    )


def severity_badge(severity: str | Severity) -> str:
    name = str(severity).upper()
    color = theme.severity_color(name)
    return (
        f'<span class="sev" style="background:{color}22;color:{color};'
        f'border:1px solid {color}55">{html.escape(name)}</span>'
    )


def chips(values: list[str], *, empty: str = "no data yet") -> str:
    if not values:
        return f'<span class="chip">{html.escape(empty)}</span>'
    return "".join(f'<span class="chip">{html.escape(str(v))}</span>' for v in values)


def awaiting_data(ctx: DashboardContext, *keys: str, note: str = "") -> bool:
    """Render an honest empty state. Returns ``True`` if the page is blocked."""
    missing: list[Artifact] = ctx.missing(*keys)
    if not missing:
        return False

    phases = sorted({item.phase for item in missing})
    items = "".join(
        f"<li><b>{html.escape(item.description)}</b> &mdash; phase {item.phase}, "
        f"produced by <code>{html.escape(item.produced_by)}</code></li>"
        for item in missing
    )
    st.markdown(
        f'<div class="aegis-empty">'
        f'<div class="head">Awaiting phase {", ".join(str(p) for p in phases)}</div>'
        f"<div>This view is fully wired and will populate as soon as its inputs exist. "
        f"Missing inputs:</div><ul>{items}</ul>"
        f"{f'<div>{html.escape(note)}</div>' if note else ''}"
        f"</div>",
        unsafe_allow_html=True,
    )
    return True


def pipeline_strip(active: set[str] | None = None) -> None:
    """The detection path an event travels. Highlights the implemented stages."""
    stages = [
        "EVENT",
        "FEATURES",
        "PROFILE",
        "ANOMALY",
        "CLASSIFIER",
        "RISK",
        "EXPLANATION",
        "ALERT",
    ]
    active = active or set()
    for column, stage in zip(st.columns(len(stages)), stages):
        state = "on" if stage in active else ""
        column.markdown(
            f'<div class="aegis-stage {state}">{stage}</div>', unsafe_allow_html=True
        )


def sidebar_status(ctx: DashboardContext) -> None:
    ready, total = ctx.readiness()
    st.markdown(
        f'<div class="aegis-status" style="border:none;margin-bottom:.2rem">'
        f"<span><b>PIPELINE</b></span><span>{ready}/{total} ready</span></div>",
        unsafe_allow_html=True,
    )
    st.progress(ready / total if total else 0.0)
    rows = "".join(
        f'<div class="aegis-status">'
        f'<span class="{"ok" if s.exists else "pending"}">&#9679; {html.escape(s.key)}</span>'
        f'<span class="ph">{"ready" if s.exists else f"P{s.artifact.phase}"}</span>'
        f"</div>"
        for s in ctx.ordered_statuses()
    )
    st.markdown(rows, unsafe_allow_html=True)


def brand() -> None:
    from src import __version__

    st.markdown(
        f'<div class="aegis-brand"><span class="mark">AE<span>G</span>IS</span>'
        f'<span class="ver">v{__version__}</span></div>'
        f'<div class="aegis-tagline">ADAPTIVE BEHAVIORAL THREAT DETECTION &middot; IT / OT</div>',
        unsafe_allow_html=True,
    )
