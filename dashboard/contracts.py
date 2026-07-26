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
    "event_id",
    "anomaly_score",
    "sequence_score",
    "attack_type",
    "attack_confidence",
    "risk_score",
    "severity",
    "short_reason",
    "reasons",
    "reason_codes",
    "top_contributors",
)

# Extra observation fields useful for investigation (kept when present).
DASHBOARD_EVENT_EXTRA_COLUMNS: tuple[str, ...] = (
    "latitude",
    "longitude",
    "bytes_transferred",
    "device_mac",
    "role",
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
    "BENIGN": "Benign (no attack class)",
    "BRUTE_FORCE": "Brute Force",
    "IMPOSSIBLE_TRAVEL": "Impossible Travel",
    "CREDENTIAL_STUFFING": "Credential Stuffing",
    "LATERAL_MOVEMENT": "Lateral Movement",
    "DEVICE_SPOOFING": "Device Spoofing",
    "LOW_AND_SLOW_EXFILTRATION": "Low-and-Slow Exfiltration",
    "INSIDER_DRIFT": "Insider Drift",
}

# Analyst-facing labels for hybrid-risk reason codes (from Phase 6 artifacts).
REASON_CODE_LABELS: dict[str, str] = {
    "ISOLATION_FOREST_ANOMALY": "Isolation Forest anomaly",
    "RULE_THRESHOLD_EXCEEDED": "Rule threshold exceeded",
    "PERSISTENT_SUSPICIOUS_ACTIVITY": "Persistent suspicious activity",
    "NEW_DEVICE": "New / unfamiliar device",
    "OFF_HOURS_ACTIVITY": "Off-hours activity",
    "UNUSUAL_LOCATION": "Unusual location for this entity",
    "UNUSUAL_SESSION_DURATION": "Unusual session duration",
    "RARE_SEQUENCE_TRANSITION": "Rare resource-sequence transition",
    "IMPOSSIBLE_VELOCITY": "Travel between locations is physically implausible",
    "RARE_RESOURCE": "Rare / never-before-seen resource",
    "RESOURCE_BREADTH_SPIKE": "Sudden expansion of accessed resources",
    "UNUSUAL_TRANSFER_VOLUME": "Unusual transfer volume",
    "AUTH_FAILURE_BURST": "Authentication failure burst",
    "ACTIVITY_BURST": "Unusual burst of activity",
    "RESOURCE_TRAVERSAL_PATTERN": "Suspicious resource-traversal pattern",
    "GEOGRAPHIC_ANOMALY": "Geographic anomaly",
    "SEQUENCE_ANOMALY": "Resource-sequence anomaly",
    "VOLUME_ANOMALY": "Transfer volume anomaly",
    "IMPOSSIBLE_TRAVEL": "Impossible travel pattern",
    "LATERAL_EXPANSION": "Lateral resource expansion",
    "COLD_START_UNCERTAINTY": "Cold-start uncertainty",
}

SEVERITY_ORDER: tuple[str, ...] = tuple(s.value for s in Severity)

# Columns shown in the ranked alert queue table.
QUEUE_DISPLAY_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "entity_id",
    "entity_type",
    "attack_type",
    "attack_confidence",
    "risk_score",
    "severity",
    "short_reason",
)

# Friendly column headers for the queue table.
QUEUE_COLUMN_LABELS: dict[str, str] = {
    "timestamp": "Timestamp",
    "entity_id": "Entity",
    "entity_type": "Type",
    "attack_type": "Predicted Type",
    "attack_confidence": "Confidence",
    "risk_score": "Risk",
    "severity": "Severity",
    "short_reason": "Top Reason",
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
    "DASHBOARD_EVENT_EXTRA_COLUMNS",
    "ENTITY_TYPE_LABELS",
    "EVENT_HISTORY_COLUMNS",
    "EVENT_HISTORY_LABELS",
    "PERFORMANCE_METRIC_SECTIONS",
    "QUEUE_COLUMN_LABELS",
    "QUEUE_DISPLAY_COLUMNS",
    "REASON_CODE_LABELS",
    "SEVERITY_ORDER",
]
