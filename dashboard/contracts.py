"""Dashboard data contracts.

These are the column sets the SOC UI reads. They are a deliberate subset of the
full pipeline schema in ``src/schema.py`` so the dashboard stays decoupled from
generator internals.

When the real pipeline writes alerts, it must conform to
:const:`ALERT_COLUMNS`. Events may carry additional generator fields; the UI
ignores anything not listed here.
"""

from __future__ import annotations

from src.schema import ATTACK_CLASSES, EntityType, Severity

# Columns the overview and downstream pages consume from an event table.
DASHBOARD_EVENT_COLUMNS: tuple[str, ...] = (
    "event_id",
    "timestamp",
    "entity_id",
    "entity_type",
    "source_ip",
    "city",
    "country",
    "resource_accessed",
    "auth_method",
    "auth_success",
    "session_duration",
    "device_id",
)

# Columns the alert queue and overview consume from a scored alert table.
DASHBOARD_ALERT_COLUMNS: tuple[str, ...] = (
    "alert_id",
    "timestamp",
    "entity_id",
    "entity_type",
    "anomaly_score",
    "sequence_score",
    "attack_type",
    "attack_confidence",
    "risk_score",
    "severity",
    "short_reason",
    "reasons",
)

# Entity roster (one row per monitored entity).
DASHBOARD_ENTITY_COLUMNS: tuple[str, ...] = (
    "entity_id",
    "entity_type",
    "role",
    "department",
    "home_city",
    "home_country",
)

ENTITY_TYPE_LABELS: dict[str, str] = {
    EntityType.USER.value: "Users",
    EntityType.SERVICE_ACCOUNT.value: "Service Accounts",
    EntityType.EDGE_DEVICE.value: "Edge Devices",
}

ATTACK_DISPLAY_NAMES: dict[str, str] = {
    "BRUTE_FORCE": "Brute Force",
    "IMPOSSIBLE_TRAVEL": "Impossible Travel",
    "CREDENTIAL_STUFFING": "Credential Stuffing",
    "LATERAL_MOVEMENT": "Lateral Movement",
    "DEVICE_SPOOFING": "Device Spoofing",
    "LOW_AND_SLOW_EXFILTRATION": "Low-and-Slow Exfiltration",
    "INSIDER_DRIFT": "Insider Drift",
}

SEVERITY_ORDER: tuple[str, ...] = tuple(s.value for s in Severity)

# Columns shown in the ranked alert queue table.
QUEUE_DISPLAY_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "entity_id",
    "entity_type",
    "attack_type",
    "risk_score",
    "severity",
    "short_reason",
)

# Friendly column headers for the queue table.
QUEUE_COLUMN_LABELS: dict[str, str] = {
    "timestamp": "Timestamp",
    "entity_id": "Entity",
    "entity_type": "Type",
    "attack_type": "Attack Type",
    "risk_score": "Risk",
    "severity": "Severity",
    "short_reason": "Reason",
}

# Columns shown in the entity event history table.
EVENT_HISTORY_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "resource_accessed",
    "city",
    "country",
    "auth_method",
    "auth_success",
    "device_id",
    "session_duration",
)

EVENT_HISTORY_LABELS: dict[str, str] = {
    "timestamp": "Timestamp",
    "resource_accessed": "Resource",
    "city": "City",
    "country": "Country",
    "auth_method": "Auth",
    "auth_success": "Success",
    "device_id": "Device",
    "session_duration": "Session (s)",
}

# Sections expected inside ``artifacts/metrics/latest.json -> metrics``.
# Written by ``python -m src.evaluation`` in Phase 12; never fabricated here.
PERFORMANCE_METRIC_SECTIONS: tuple[str, ...] = (
    "detection",
    "pr_curve",
    "roc_curve",
    "budget_sweep",
    "confusion_matrix",
    "per_class",
    "campaign_detection",
)

__all__ = [
    "ATTACK_CLASSES",
    "ATTACK_DISPLAY_NAMES",
    "DASHBOARD_ALERT_COLUMNS",
    "DASHBOARD_ENTITY_COLUMNS",
    "DASHBOARD_EVENT_COLUMNS",
    "ENTITY_TYPE_LABELS",
    "EVENT_HISTORY_COLUMNS",
    "EVENT_HISTORY_LABELS",
    "PERFORMANCE_METRIC_SECTIONS",
    "QUEUE_COLUMN_LABELS",
    "QUEUE_DISPLAY_COLUMNS",
    "SEVERITY_ORDER",
]
