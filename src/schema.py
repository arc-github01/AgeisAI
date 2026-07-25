"""Canonical event schema and threat taxonomy for AEGIS.

Why this module exists
----------------------
Every later phase (generator, feature engineering, profiling, models,
dashboard, simulator) must agree on exactly one event contract. Defining it
once, up front, is what makes the attack simulator able to push a freshly
minted event through the *same* pipeline as the offline dataset.

It also enforces the single hardest ML rule in this project:

    **Ground-truth labels must never reach an inference feature.**

:func:`assert_no_label_leakage` is called by the feature layer, so leakage is a
test failure rather than a code-review opinion.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Iterable, Sequence

import pandas as pd


class EntityType(StrEnum):
    USER = "user"
    SERVICE_ACCOUNT = "service_account"
    EDGE_DEVICE = "edge_device"


class AttackType(StrEnum):
    """The taxonomy fixed by the problem statement.

    ``BENIGN`` is the normal baseline. ``INSIDER_DRIFT`` is deliberately an
    *edge case* rather than a clean attack: it is legitimate-looking privilege
    expansion, used for false-positive tuning and concept-drift testing.
    """

    BENIGN = "BENIGN"
    BRUTE_FORCE = "BRUTE_FORCE"
    IMPOSSIBLE_TRAVEL = "IMPOSSIBLE_TRAVEL"
    CREDENTIAL_STUFFING = "CREDENTIAL_STUFFING"
    LATERAL_MOVEMENT = "LATERAL_MOVEMENT"
    DEVICE_SPOOFING = "DEVICE_SPOOFING"
    LOW_AND_SLOW_EXFILTRATION = "LOW_AND_SLOW_EXFILTRATION"
    INSIDER_DRIFT = "INSIDER_DRIFT"


#: Attack categories the supervised classifier is expected to separate.
ATTACK_CLASSES: tuple[str, ...] = tuple(
    a.value for a in AttackType if a is not AttackType.BENIGN
)

#: Categories treated as true positives for detection metrics. INSIDER_DRIFT is
#: excluded by default because it is an ambiguous edge case, not a confirmed
#: intrusion; evaluation reports it separately.
MALICIOUS_CLASSES: tuple[str, ...] = tuple(
    a for a in ATTACK_CLASSES if a != AttackType.INSIDER_DRIFT.value
)


class Severity(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# -----------------------------------------------------------------------------
# Event contract
# -----------------------------------------------------------------------------

#: Identity of the event and the acting entity.
IDENTITY_COLUMNS: tuple[str, ...] = (
    "event_id",
    "timestamp",
    "entity_id",
    "entity_type",
    "role",
)

#: Observable telemetry. This is all an inference-time detector may see.
OBSERVATION_COLUMNS: tuple[str, ...] = (
    # network / geography (structured, not a flat string)
    "source_ip",
    "country",
    "city",
    "latitude",
    "longitude",
    # authentication
    "auth_method",
    "auth_success",
    # activity
    "resource_accessed",
    "action",
    "command_sequence",
    "session_duration_s",
    "bytes_transferred",
    # device fingerprint (decomposed)
    "device_id",
    "device_os",
    "device_firmware",
    "device_protocol",
    "device_mac",
)

#: Ground truth. Training/evaluation only - NEVER an input to a feature.
LABEL_COLUMNS: tuple[str, ...] = (
    "label",        # AttackType value
    "is_attack",    # bool, derived from label
    "campaign_id",  # groups events belonging to one injected campaign
)

EVENT_COLUMNS: tuple[str, ...] = IDENTITY_COLUMNS + OBSERVATION_COLUMNS + LABEL_COLUMNS

#: Columns a model or feature transformer is forbidden from consuming.
FORBIDDEN_FEATURE_COLUMNS: frozenset[str] = frozenset(LABEL_COLUMNS)


class LabelLeakageError(RuntimeError):
    """Raised when ground-truth information reaches the feature space."""


class SchemaError(ValueError):
    """Raised when an event frame violates the canonical contract."""


def assert_no_label_leakage(columns: Iterable[str]) -> None:
    """Fail loudly if any label column is present in a feature column set."""
    offenders = sorted(FORBIDDEN_FEATURE_COLUMNS.intersection(columns))
    if offenders:
        raise LabelLeakageError(
            "label columns must never be used as inference features: "
            + ", ".join(offenders)
        )


def feature_safe_columns(columns: Iterable[str]) -> list[str]:
    """Drop label columns from an arbitrary column list."""
    return [c for c in columns if c not in FORBIDDEN_FEATURE_COLUMNS]


def validate_events(
    frame: pd.DataFrame,
    *,
    require_labels: bool = True,
    strict_order: bool = False,
) -> pd.DataFrame:
    """Validate an event frame against the canonical contract.

    Parameters
    ----------
    require_labels:
        ``False`` for live/streaming events (the simulator, production
        inference), where ground truth does not exist.
    strict_order:
        Also require the frame to be sorted by timestamp, which the temporal
        feature and split logic depends on.
    """
    expected: Sequence[str] = EVENT_COLUMNS if require_labels else (
        IDENTITY_COLUMNS + OBSERVATION_COLUMNS
    )
    missing = [c for c in expected if c not in frame.columns]
    if missing:
        raise SchemaError(f"event frame is missing required columns: {missing}")

    if not pd.api.types.is_datetime64_any_dtype(frame["timestamp"]):
        raise SchemaError("'timestamp' must be a datetime64 column")

    if strict_order and not frame["timestamp"].is_monotonic_increasing:
        raise SchemaError("event frame must be sorted by ascending timestamp")

    if require_labels:
        unknown = set(frame["label"].unique()) - {a.value for a in AttackType}
        if unknown:
            raise SchemaError(f"unknown label values: {sorted(unknown)}")

    return frame
