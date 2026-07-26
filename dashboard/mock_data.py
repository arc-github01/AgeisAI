"""DEVELOPMENT FIXTURE — REPLACE WITH REAL PIPELINE OUTPUT

Generates deterministic sample datasets so the SOC dashboard can be built and
demonstrated before the detection pipeline exists.

These values are NOT measured model outputs. They exist solely for frontend
development and must never be presented as evaluation results.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import hashlib

import numpy as np
import pandas as pd

from src.schema import ATTACK_CLASSES, IDENTITY_COLUMNS, OBSERVATION_COLUMNS, EntityType, Severity

from .contracts import DASHBOARD_ALERT_COLUMNS, DASHBOARD_ENTITY_COLUMNS, DASHBOARD_EVENT_COLUMNS

# ---------------------------------------------------------------------------
# Fixture sizing (tuned to produce realistic overview KPIs without huge frames)
# ---------------------------------------------------------------------------
FIXTURE_SEED = 20260725

# Total corpus size represented by the simulated environment (KPI denominator).
EVENTS_PROCESSED_TOTAL = 204_382

N_USERS = 700
N_SERVICE_ACCOUNTS = 200
N_EDGE_DEVICES = 100
N_ENTITIES = N_USERS + N_SERVICE_ACCOUNTS + N_EDGE_DEVICES  # 1_000

N_ALERTS = 412
N_CRITICAL = 19

SIMULATION_START = datetime(2025, 1, 6, 0, 0, 0)
SIMULATION_DAYS = 90

CITIES = [
    ("Chennai", "IN"),
    ("Bengaluru", "IN"),
    ("London", "GB"),
    ("Singapore", "SG"),
    ("Dallas", "US"),
    ("Frankfurt", "DE"),
]

# Approximate coordinates so injection events satisfy the full observation schema.
CITY_COORDS: dict[tuple[str, str], tuple[float, float]] = {
    ("Chennai", "IN"): (13.0827, 80.2707),
    ("Bengaluru", "IN"): (12.9716, 77.5946),
    ("London", "GB"): (51.5074, -0.1278),
    ("Singapore", "SG"): (1.3521, 103.8198),
    ("Dallas", "US"): (32.7767, -96.7970),
    ("Frankfurt", "DE"): (50.1109, 8.6821),
}

RESOURCES = [
    "email",
    "git",
    "dev_server",
    "admin_console",
    "hr_portal",
    "scada_hmi",
    "plc_gateway",
    "backup_server",
]

AUTH_METHODS = ["password", "sso", "mfa", "certificate", "api_key"]

REASON_TEMPLATES: dict[str, list[str]] = {
    "BRUTE_FORCE": [
        "Repeated failed authentication burst",
        "High failure rate in 5-minute window",
    ],
    "IMPOSSIBLE_TRAVEL": [
        "Impossible geo velocity + unseen device",
        "Login from new country within implausible travel time",
    ],
    "CREDENTIAL_STUFFING": [
        "Source IP targeting many entity IDs with failures",
        "Credential misuse pattern across accounts",
    ],
    "LATERAL_MOVEMENT": [
        "Unusual internal resource breadth",
        "Never-before-accessed admin systems",
    ],
    "DEVICE_SPOOFING": [
        "Device fingerprint inconsistent with entity history",
        "Unexpected OS and protocol combination",
    ],
    "LOW_AND_SLOW_EXFILTRATION": [
        "Gradual off-hours resource access accumulation",
        "Small repeated transfers to unusual resource",
    ],
    "INSIDER_DRIFT": [
        "Gradual expansion of privileged resources",
        "Slow privilege footprint growth",
    ],
}

# Multi-factor explanations for the alert detail pane (fixture only).
REASON_FACTORS: dict[str, list[str]] = {
    "BRUTE_FORCE": [
        "Failed attempts spike in 5m window",
        "Source IP failure rate above entity baseline",
        "Authentication burst outside normal hours",
    ],
    "IMPOSSIBLE_TRAVEL": [
        "Geographic velocity exceeds plausible travel",
        "Device fingerprint never observed for entity",
        "Login outside entity time distribution",
    ],
    "CREDENTIAL_STUFFING": [
        "Single source targeting many entity IDs",
        "High cross-account failure rate",
        "Unusual auth method for entity",
    ],
    "LATERAL_MOVEMENT": [
        "Resource never previously accessed",
        "Abnormal internal system breadth",
        "Low transition probability in sequence",
    ],
    "DEVICE_SPOOFING": [
        "OS mismatch vs entity history",
        "Unexpected protocol for entity",
        "New device identifier observed",
    ],
    "LOW_AND_SLOW_EXFILTRATION": [
        "Off-hours resource access pattern",
        "Gradual volume accumulation",
        "Unusual resource for entity profile",
    ],
    "INSIDER_DRIFT": [
        "Privilege footprint expanding over time",
        "New sensitive resources accessed",
        "Behavior diverging from cohort baseline",
    ],
}


def _rng() -> np.random.Generator:
    return np.random.default_rng(FIXTURE_SEED)


def _entity_id(entity_type: str, index: int) -> str:
    prefix = {"user": "USR", "service_account": "SVC", "edge_device": "DEV"}[entity_type]
    return f"{prefix}_{index:03d}"


def generate_entities() -> pd.DataFrame:
    """Return the monitored-entity roster (1 000 entities)."""
    rng = _rng()
    rows: list[dict] = []
    counters = {"user": 0, "service_account": 0, "edge_device": 0}
    quotas = {
        EntityType.USER.value: N_USERS,
        EntityType.SERVICE_ACCOUNT.value: N_SERVICE_ACCOUNTS,
        EntityType.EDGE_DEVICE.value: N_EDGE_DEVICES,
    }
    roles = {
        EntityType.USER.value: ["engineer", "analyst", "operator", "manager"],
        EntityType.SERVICE_ACCOUNT.value: ["ci_runner", "backup_agent", "monitoring"],
        EntityType.EDGE_DEVICE.value: ["plc", "hmi", "sensor_gateway"],
    }
    for entity_type, count in quotas.items():
        for _ in range(count):
            counters[entity_type] += 1
            city, country = CITIES[int(rng.integers(0, len(CITIES)))]
            rows.append(
                {
                    "entity_id": _entity_id(entity_type, counters[entity_type]),
                    "entity_type": entity_type,
                    "role": str(rng.choice(roles[entity_type])),
                    "department": str(rng.choice(["engineering", "operations", "security", "it"])),
                    "home_city": city,
                    "home_country": country,
                }
            )
    frame = pd.DataFrame(rows, columns=list(DASHBOARD_ENTITY_COLUMNS))
    return frame.sort_values("entity_id").reset_index(drop=True)


def generate_alerts(entities: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return a scored alert table (412 alerts, 19 critical)."""
    if entities is None:
        entities = generate_entities()
    rng = _rng()
    entity_rows = entities.to_dict("records")

    # Weight attack types for a realistic mix.
    attack_weights = np.array([0.22, 0.14, 0.18, 0.16, 0.12, 0.10, 0.08])
    attack_types = list(ATTACK_CLASSES)

    rows: list[dict] = []
    span_seconds = SIMULATION_DAYS * 24 * 3600

    for i in range(N_ALERTS):
        entity = entity_rows[int(rng.integers(0, len(entity_rows)))]
        attack = attack_types[int(rng.choice(len(attack_types), p=attack_weights))]
        offset = float(rng.uniform(0, span_seconds))
        ts = SIMULATION_START + timedelta(seconds=offset)

        # Risk scores: reserve top band for critical alerts.
        if i < N_CRITICAL:
            risk = float(rng.uniform(81, 99))
            severity = Severity.CRITICAL.value
        else:
            risk = float(rng.uniform(22, 80))
            if risk >= 61:
                severity = Severity.HIGH.value
            elif risk >= 31:
                severity = Severity.MEDIUM.value
            else:
                severity = Severity.LOW.value

        anomaly = float(np.clip(rng.normal(0.55, 0.18), 0.05, 1.0))
        sequence = float(np.clip(rng.normal(0.45, 0.2), 0.05, 1.0))
        confidence = float(np.clip(rng.normal(0.72, 0.12), 0.35, 0.99))
        reason = str(rng.choice(REASON_TEMPLATES[attack]))
        factors = REASON_FACTORS[attack]
        pick = int(rng.integers(2, min(4, len(factors) + 1)))
        chosen = list(rng.choice(factors, size=pick, replace=False))
        reasons_text = " + ".join(chosen)

        rows.append(
            {
                "alert_id": f"ALT_{i + 1:05d}",
                "timestamp": ts,
                "entity_id": entity["entity_id"],
                "entity_type": entity["entity_type"],
                "anomaly_score": round(anomaly, 4),
                "sequence_score": round(sequence, 4),
                "attack_type": attack,
                "attack_confidence": round(confidence, 4),
                "risk_score": round(risk, 1),
                "severity": severity,
                "short_reason": reason,
                "reasons": reasons_text,
            }
        )

    frame = pd.DataFrame(rows, columns=list(DASHBOARD_ALERT_COLUMNS))
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame.sort_values("timestamp").reset_index(drop=True)


def _stable_device(entity_id: str) -> str:
    """Deterministic primary device for an entity (fixture convenience)."""
    try:
        suffix = int(entity_id.rsplit("_", 1)[-1])
    except ValueError:
        suffix = abs(hash(entity_id)) % 999
    return f"DEV_{suffix:03d}"


def _stable_mac(device_id: str) -> str:
    digest = hashlib.md5(device_id.encode("utf-8")).hexdigest()
    return ":".join(digest[i : i + 2] for i in range(0, 12, 2))


def _city_coords(city: str, country: str) -> tuple[float, float]:
    return CITY_COORDS.get((city, country), (0.0, 0.0))


def _event_row(
    rng: np.random.Generator,
    *,
    counter: int,
    entity: dict,
    ts: datetime,
    city: str,
    country: str,
) -> dict:
    lat, lon = _city_coords(city, country)
    device_id = _stable_device(entity["entity_id"])
    return {
        "event_id": f"EVT_{counter:06d}",
        "timestamp": ts,
        "entity_id": entity["entity_id"],
        "entity_type": entity["entity_type"],
        "role": str(entity.get("role") or "unknown"),
        "source_ip": f"10.{rng.integers(0, 255)}.{rng.integers(0, 255)}.{rng.integers(1, 254)}",
        "city": city,
        "country": country,
        "latitude": lat,
        "longitude": lon,
        "resource_accessed": str(rng.choice(RESOURCES)),
        "action": "access",
        "command_sequence": "",
        "auth_method": str(rng.choice(AUTH_METHODS)),
        "auth_success": bool(rng.random() > 0.04),
        "session_duration": int(rng.integers(60, 7200)),
        "session_duration_s": float(rng.integers(60, 7200)),
        "bytes_transferred": float(rng.integers(500, 50_000)),
        "device_id": device_id,
        "device_os": "Windows",
        "device_firmware": "1.0.0",
        "device_protocol": "HTTPS",
        "device_mac": _stable_mac(device_id),
    }


def generate_events_sample(*, n_background: int = 3_500) -> pd.DataFrame:
    """Event sample with richer per-entity history for investigation views."""
    rng = _rng()
    entities = generate_entities()
    alerts = generate_alerts(entities)
    entity_lookup = {row["entity_id"]: row for row in entities.to_dict("records")}
    span_seconds = SIMULATION_DAYS * 24 * 3600
    rows: list[dict] = []
    counter = 0

    # Richer histories for entities that appear in the alert fixture.
    for entity_id in alerts["entity_id"].unique():
        entity = entity_lookup[str(entity_id)]
        home_city = entity.get("home_city", CITIES[0][0])
        home_country = entity.get("home_country", CITIES[0][1])
        for _ in range(int(rng.integers(25, 50))):
            counter += 1
            ts = SIMULATION_START + timedelta(seconds=float(rng.uniform(0, span_seconds)))
            if rng.random() < 0.82:
                city, country = home_city, home_country
            else:
                city, country = CITIES[int(rng.integers(0, len(CITIES)))]
            rows.append(_event_row(rng, counter=counter, entity=entity, ts=ts, city=city, country=country))

    # Background activity across the fleet.
    entity_rows = entities.to_dict("records")
    for _ in range(n_background):
        counter += 1
        entity = entity_rows[int(rng.integers(0, len(entity_rows)))]
        ts = SIMULATION_START + timedelta(seconds=float(rng.uniform(0, span_seconds)))
        city, country = CITIES[int(rng.integers(0, len(CITIES)))]
        rows.append(_event_row(rng, counter=counter, entity=entity, ts=ts, city=city, country=country))

    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame.sort_values("timestamp").reset_index(drop=True)


def _injection_rng(entity_id: str, attack_type: str) -> np.random.Generator:
    digest = hashlib.blake2b(
        f"{FIXTURE_SEED}:{entity_id}:{attack_type}:injection".encode(),
        digest_size=8,
    ).digest()
    seed = int.from_bytes(digest, "big") % (2**32)
    return np.random.default_rng(seed)


def _intensity_scale(intensity: int, base: int) -> int:
    level = max(1, min(5, int(intensity)))
    return base + (level - 1) * 2


def generate_injection_events(
    entity_id: str,
    attack_type: str,
    intensity: int,
    entities: pd.DataFrame,
) -> pd.DataFrame:
    """DEVELOPMENT FIXTURE — synthesise attack events for simulator input only.

    The live simulator prefers generator-owned campaign synthesis when pipeline
    artifacts exist. This helper remains for unit tests and fixture-only shells.
    """
    if attack_type not in ATTACK_CLASSES:
        raise ValueError(f"unknown attack type: {attack_type}")

    match = entities[entities["entity_id"].astype(str) == str(entity_id)]
    if match.empty:
        raise ValueError(f"unknown entity_id: {entity_id}")
    entity = match.iloc[0].to_dict()

    rng = _injection_rng(entity_id, attack_type)
    home_city = str(entity.get("home_city") or CITIES[0][0])
    home_country = str(entity.get("home_country") or CITIES[0][1])
    base_time = SIMULATION_START + timedelta(days=max(SIMULATION_DAYS - 3, 0), hours=9)
    rows: list[dict] = []
    counter = 0
    source_ip = f"203.{rng.integers(0, 255)}.{rng.integers(0, 255)}.{rng.integers(1, 254)}"
    primary_device = _stable_device(entity_id)

    def add_event(
        ts: datetime,
        *,
        city: str,
        country: str,
        resource: str,
        auth_success: bool,
        device_id: str | None = None,
        auth_method: str | None = None,
        bytes_transferred: float | None = None,
    ) -> None:
        nonlocal counter
        counter += 1
        row = _event_row(rng, counter=counter, entity=entity, ts=ts, city=city, country=country)
        row["event_id"] = f"INJ_{counter:04d}"
        row["source_ip"] = source_ip
        row["resource_accessed"] = resource
        row["auth_success"] = auth_success
        chosen_device = device_id or primary_device
        row["device_id"] = chosen_device
        row["device_mac"] = _stable_mac(chosen_device)
        if auth_method:
            row["auth_method"] = auth_method
        if bytes_transferred is not None:
            row["bytes_transferred"] = float(bytes_transferred)
        rows.append(row)

    if attack_type == "BRUTE_FORCE":
        n_failures = _intensity_scale(intensity, 8)
        for i in range(n_failures):
            add_event(
                base_time + timedelta(minutes=i),
                city=home_city,
                country=home_country,
                resource="auth_gateway",
                auth_success=False,
                auth_method="password",
            )
    elif attack_type == "IMPOSSIBLE_TRAVEL":
        add_event(
            base_time,
            city=home_city,
            country=home_country,
            resource="email",
            auth_success=True,
        )
        add_event(
            base_time + timedelta(minutes=30),
            city="London",
            country="GB",
            resource="vpn_portal",
            auth_success=True,
            device_id=f"DEV_{999:03d}",
        )
    elif attack_type == "CREDENTIAL_STUFFING":
        n_failures = _intensity_scale(intensity, 6)
        for i in range(n_failures):
            add_event(
                base_time + timedelta(seconds=i * 20),
                city=home_city,
                country=home_country,
                resource="auth_gateway",
                auth_success=False,
                auth_method="password",
            )
    elif attack_type == "LATERAL_MOVEMENT":
        resources = ["admin_console", "backup_server", "plc_gateway", "hr_portal", "dev_server"]
        n_steps = min(_intensity_scale(intensity, 3), len(resources))
        for i in range(n_steps):
            add_event(
                base_time + timedelta(minutes=i * 4),
                city=home_city,
                country=home_country,
                resource=resources[i],
                auth_success=True,
            )
    elif attack_type == "DEVICE_SPOOFING":
        add_event(
            base_time,
            city=home_city,
            country=home_country,
            resource="email",
            auth_success=True,
        )
        add_event(
            base_time + timedelta(minutes=5),
            city=home_city,
            country=home_country,
            resource="git",
            auth_success=True,
            device_id=f"DEV_{888:03d}",
            auth_method="certificate",
        )
    elif attack_type == "LOW_AND_SLOW_EXFILTRATION":
        n_steps = _intensity_scale(intensity, 3)
        for i in range(n_steps):
            add_event(
                base_time + timedelta(days=i, hours=2),
                city=home_city,
                country=home_country,
                resource="backup_server",
                auth_success=True,
            )
    elif attack_type == "INSIDER_DRIFT":
        resources = ["email", "git", "dev_server", "admin_console", "payroll_portal"]
        n_steps = min(_intensity_scale(intensity, 2), len(resources))
        for i in range(n_steps):
            add_event(
                base_time + timedelta(days=i * 7),
                city=home_city,
                country=home_country,
                resource=resources[i],
                auth_success=True,
            )
    else:  # pragma: no cover
        raise ValueError(f"unsupported attack type: {attack_type}")

    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    # Injection events must satisfy the full observation contract for process_event.
    required = list(IDENTITY_COLUMNS + OBSERVATION_COLUMNS)
    return frame.loc[:, required].sort_values("timestamp").reset_index(drop=True)


def fixture_summary() -> dict[str, int | float]:
    """Aggregate KPI values derived from the fixture constants and alert frame."""
    alerts = generate_alerts()
    critical = int((alerts["severity"] == Severity.CRITICAL.value).sum())
    return {
        "events_processed": EVENTS_PROCESSED_TOTAL,
        "entities_monitored": N_ENTITIES,
        "active_alerts": len(alerts),
        "critical_alerts": critical,
        "alert_rate": len(alerts) / EVENTS_PROCESSED_TOTAL,
    }
