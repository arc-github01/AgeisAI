"""AEGIS analyst console (Streamlit).

Page modules expose ``TITLE``, ``SUBTITLE`` and ``render(ctx)`` and hold no
state of their own; all data access goes through
:class:`dashboard.state.DashboardContext`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from . import alerts, entity_view, overview, performance, simulator
from .state import DashboardContext, get_context


@dataclass(frozen=True)
class Page:
    key: str
    label: str
    render: Callable[[DashboardContext], None]


PAGES: tuple[Page, ...] = (
    Page("overview", "SOC Overview", overview.render),
    Page("alerts", "Alert Queue", alerts.render),
    Page("entity", "Entity Investigation", entity_view.render),
    Page("simulator", "Attack Simulator", simulator.render),
    Page("performance", "Model Performance", performance.render),
)

PAGES_BY_LABEL: dict[str, Page] = {page.label: page for page in PAGES}

__all__ = [
    "DashboardContext",
    "PAGES",
    "PAGES_BY_LABEL",
    "Page",
    "get_context",
    "alerts",
    "entity_view",
    "overview",
    "performance",
    "simulator",
]
