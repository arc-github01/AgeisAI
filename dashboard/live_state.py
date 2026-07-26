"""Session-scoped live overlays for simulator-injected alerts and events.

Persisted pipeline artifacts remain the source of truth on disk. Injected
results are kept in Streamlit session state so Overview / Alert Queue /
Entity Investigation can show them immediately without rewriting parquet.
"""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd
import streamlit as st

from .contracts import DASHBOARD_ALERT_COLUMNS, DASHBOARD_EVENT_COLUMNS

_LIVE_ALERTS_KEY = "aegis_live_alerts"
_LIVE_EVENTS_KEY = "aegis_live_events"

# Fallback store for unit tests / non-Streamlit callers.
_FALLBACK: dict[str, list[Any]] = {
    _LIVE_ALERTS_KEY: [],
    _LIVE_EVENTS_KEY: [],
}


def _store() -> Any:
    """Prefer Streamlit session state; fall back when no ScriptRunContext exists."""
    try:
        return st.session_state
    except Exception:  # noqa: BLE001
        return _FALLBACK


def _ensure_lists() -> None:
    store = _store()
    store.setdefault(_LIVE_ALERTS_KEY, [])
    store.setdefault(_LIVE_EVENTS_KEY, [])


def clear_live_overlays() -> None:
    """Drop any simulator overlays for this browser session."""
    store = _store()
    store[_LIVE_ALERTS_KEY] = []
    store[_LIVE_EVENTS_KEY] = []


def append_live_injection(
    *,
    alerts: list[Mapping[str, Any]],
    events: pd.DataFrame | None = None,
) -> None:
    """Append freshly scored injection results to the session overlay."""
    _ensure_lists()
    store = _store()
    if alerts:
        existing = list(store[_LIVE_ALERTS_KEY])
        for alert in alerts:
            existing.append(dict(alert))
        store[_LIVE_ALERTS_KEY] = existing[-200:]

    if events is not None and not events.empty:
        rows = events.to_dict(orient="records")
        prior = list(store[_LIVE_EVENTS_KEY])
        prior.extend(rows)
        store[_LIVE_EVENTS_KEY] = prior[-500:]


def live_alerts_frame() -> pd.DataFrame:
    _ensure_lists()
    rows = list(_store().get(_LIVE_ALERTS_KEY, []))
    if not rows:
        return pd.DataFrame(columns=list(DASHBOARD_ALERT_COLUMNS))
    frame = pd.DataFrame(rows)
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame


def live_events_frame() -> pd.DataFrame:
    _ensure_lists()
    rows = list(_store().get(_LIVE_EVENTS_KEY, []))
    if not rows:
        return pd.DataFrame(columns=list(DASHBOARD_EVENT_COLUMNS))
    frame = pd.DataFrame(rows)
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    if "session_duration_s" in frame.columns and "session_duration" not in frame.columns:
        frame["session_duration"] = frame["session_duration_s"]
    return frame


def live_alert_count() -> int:
    _ensure_lists()
    return len(_store().get(_LIVE_ALERTS_KEY, []))


__all__ = [
    "append_live_injection",
    "clear_live_overlays",
    "live_alert_count",
    "live_alerts_frame",
    "live_events_frame",
]
