"""Persistent entity behavioural profiles — the generator's 'digital twins'.

Design principle (ADR-13)
-------------------------
Normal behaviour is NOT sampled as independent random rows. Each entity is a
*behavioural model* with:

    - circadian activity rhythm (when they work)
    - geographic anchor (where they usually connect from)
    - device registry (known endpoints)
    - resource catalogue (what they normally touch)
    - Markov transition graph (how they move between resources within a session)

The normal-behaviour engine *simulates sessions* through these models. That is
what makes downstream sequence features, profiling, and the live attack
simulator meaningful — they all operate on the same causal structure.
"""

from __future__ import annotations

import ipaddress
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from src.schema import EntityType
from src.utils.seeding import get_rng

# ---------------------------------------------------------------------------
# Shared geography — enterprise office / plant anchors
# ---------------------------------------------------------------------------

ENTERPRISE_SITES: tuple[dict[str, Any], ...] = (
    {"country": "India", "city": "Chennai", "lat": 13.0827, "lon": 80.2707},
    {"country": "India", "city": "Bangalore", "lat": 12.9716, "lon": 77.5946},
    {"country": "United States", "city": "Atlanta", "lat": 33.7490, "lon": -84.3880},
    {"country": "United States", "city": "Charlotte", "lat": 35.2271, "lon": -80.8431},
    {"country": "United Kingdom", "city": "London", "lat": 51.5074, "lon": -0.1278},
    {"country": "Germany", "city": "Frankfurt", "lat": 50.1109, "lon": 8.6821},
)

# VPN / travel egress used for benign remote-work variation
REMOTE_EGRESS: tuple[dict[str, Any], ...] = (
    {"country": "United Kingdom", "city": "London", "lat": 51.5074, "lon": -0.1278},
    {"country": "Singapore", "city": "Singapore", "lat": 1.3521, "lon": 103.8198},
    {"country": "United States", "city": "Seattle", "lat": 47.6062, "lon": -122.3321},
)


@dataclass(frozen=True)
class GeoAnchor:
    country: str
    city: str
    latitude: float
    longitude: float


@dataclass(frozen=True)
class RegisteredDevice:
    """One endpoint an entity is known to use."""

    device_id: str
    device_os: str
    device_firmware: str
    device_protocol: str
    device_mac: str
    is_primary: bool = True


@dataclass
class ResourceDef:
    name: str
    action: str
    sensitivity: str  # low | medium | high | critical
    typical_bytes: int


@dataclass
class CohortArchetype:
    """Role-level template shared by a cohort (cold-start fallback later)."""

    role: str
    entity_type: EntityType
    resources: list[ResourceDef]
    transitions: dict[str, dict[str, float]]  # Markov: current -> {next: prob}
    login_hour_mean: float
    login_hour_std: float
    sessions_per_day_mean: float
    session_steps_mean: int
    session_duration_mean_s: float
    auth_methods: list[str]
    auth_failure_rate: float
    weekend_activity_prob: float
    vpn_prob: float
    secondary_device_prob: float


@dataclass
class EntityBehavioralProfile:
    """Complete behavioural model for one persistent actor."""

    entity_id: str
    entity_type: EntityType
    role: str
    cohort: CohortArchetype
    home_geo: GeoAnchor
    ip_network: str
    working_days: set[int]  # 0=Mon .. 6=Sun
    preferred_login_hour: float
    devices: list[RegisteredDevice]
    primary_auth_method: str
    rng: np.random.Generator = field(repr=False)


# ---------------------------------------------------------------------------
# Cohort archetypes — role behaviour, NOT per-row randomness
# ---------------------------------------------------------------------------

def _arc(name: str, action: str, sensitivity: str = "medium", typical_bytes: int = 4096) -> ResourceDef:
    return ResourceDef(name=name, action=action, sensitivity=sensitivity, typical_bytes=typical_bytes)


COHORT_ARCHETYPES: dict[str, CohortArchetype] = {
    "developer": CohortArchetype(
        role="developer",
        entity_type=EntityType.USER,
        resources=[
            _arc("Corporate SSO", "LOGIN", "low", 512),
            _arc("Email", "ACCESS", "medium", 8192),
            _arc("GitHub", "ACCESS", "high", 65536),
            _arc("Jira", "ACCESS", "medium", 16384),
            _arc("Jenkins", "ACCESS", "high", 32768),
            _arc("Dev Server", "COMMAND", "critical", 131072),
        ],
        transitions={
            "START": {"Corporate SSO": 1.0},
            "Corporate SSO": {"Email": 0.35, "GitHub": 0.40, "Jira": 0.25},
            "Email": {"GitHub": 0.45, "Jira": 0.35, "Dev Server": 0.10, "END": 0.10},
            "GitHub": {"Jenkins": 0.35, "Dev Server": 0.40, "Jira": 0.15, "END": 0.10},
            "Jira": {"GitHub": 0.30, "Email": 0.20, "END": 0.50},
            "Jenkins": {"Dev Server": 0.50, "GitHub": 0.20, "END": 0.30},
            "Dev Server": {"GitHub": 0.25, "Jenkins": 0.15, "END": 0.60},
        },
        login_hour_mean=9.0,
        login_hour_std=0.8,
        sessions_per_day_mean=1.8,
        session_steps_mean=5,
        session_duration_mean_s=2700,
        auth_methods=["SSO", "MFA", "SSH_KEY"],
        auth_failure_rate=0.03,
        weekend_activity_prob=0.08,
        vpn_prob=0.12,
        secondary_device_prob=0.10,
    ),
    "hr": CohortArchetype(
        role="hr",
        entity_type=EntityType.USER,
        resources=[
            _arc("Corporate SSO", "LOGIN", "low", 512),
            _arc("Workday", "ACCESS", "high", 16384),
            _arc("Payroll Portal", "ACCESS", "critical", 8192),
            _arc("Benefits Admin", "ACCESS", "medium", 4096),
        ],
        transitions={
            "START": {"Corporate SSO": 1.0},
            "Corporate SSO": {"Workday": 0.55, "Payroll Portal": 0.25, "Benefits Admin": 0.20},
            "Workday": {"Payroll Portal": 0.30, "Benefits Admin": 0.25, "END": 0.45},
            "Payroll Portal": {"Workday": 0.20, "END": 0.80},
            "Benefits Admin": {"Workday": 0.35, "END": 0.65},
        },
        login_hour_mean=8.5,
        login_hour_std=0.6,
        sessions_per_day_mean=1.4,
        session_steps_mean=4,
        session_duration_mean_s=2100,
        auth_methods=["SSO", "MFA"],
        auth_failure_rate=0.02,
        weekend_activity_prob=0.03,
        vpn_prob=0.05,
        secondary_device_prob=0.06,
    ),
    "finance": CohortArchetype(
        role="finance",
        entity_type=EntityType.USER,
        resources=[
            _arc("Corporate SSO", "LOGIN", "low", 512),
            _arc("SAP", "ACCESS", "critical", 32768),
            _arc("Oracle Financials", "ACCESS", "critical", 32768),
            _arc("Treasury Portal", "ACCESS", "critical", 16384),
        ],
        transitions={
            "START": {"Corporate SSO": 1.0},
            "Corporate SSO": {"SAP": 0.50, "Oracle Financials": 0.35, "Treasury Portal": 0.15},
            "SAP": {"Oracle Financials": 0.25, "Treasury Portal": 0.20, "END": 0.55},
            "Oracle Financials": {"SAP": 0.20, "END": 0.80},
            "Treasury Portal": {"SAP": 0.30, "END": 0.70},
        },
        login_hour_mean=8.0,
        login_hour_std=0.5,
        sessions_per_day_mean=1.5,
        session_steps_mean=4,
        session_duration_mean_s=3000,
        auth_methods=["SSO", "MFA", "SMARTCARD"],
        auth_failure_rate=0.02,
        weekend_activity_prob=0.04,
        vpn_prob=0.08,
        secondary_device_prob=0.05,
    ),
    "night_shift_operator": CohortArchetype(
        role="night_shift_operator",
        entity_type=EntityType.USER,
        resources=[
            _arc("OT Gateway", "LOGIN", "high", 256),
            _arc("SCADA HMI", "ACCESS", "critical", 8192),
            _arc("Plant Historian", "ACCESS", "high", 16384),
            _arc("Alarm Console", "ACCESS", "critical", 4096),
        ],
        transitions={
            "START": {"OT Gateway": 1.0},
            "OT Gateway": {"SCADA HMI": 0.45, "Alarm Console": 0.35, "Plant Historian": 0.20},
            "SCADA HMI": {"Alarm Console": 0.30, "Plant Historian": 0.25, "END": 0.45},
            "Alarm Console": {"SCADA HMI": 0.40, "END": 0.60},
            "Plant Historian": {"SCADA HMI": 0.30, "END": 0.70},
        },
        login_hour_mean=22.0,
        login_hour_std=1.0,
        sessions_per_day_mean=1.2,
        session_steps_mean=4,
        session_duration_mean_s=3600,
        auth_methods=["MFA", "SMARTCARD"],
        auth_failure_rate=0.01,
        weekend_activity_prob=0.40,
        vpn_prob=0.0,
        secondary_device_prob=0.03,
    ),
    "it_admin": CohortArchetype(
        role="it_admin",
        entity_type=EntityType.USER,
        resources=[
            _arc("Corporate SSO", "LOGIN", "low", 512),
            _arc("Active Directory", "ACCESS", "critical", 16384),
            _arc("ServiceNow", "ACCESS", "medium", 8192),
            _arc("Splunk", "ACCESS", "high", 65536),
            _arc("AWS Console", "ACCESS", "critical", 32768),
        ],
        transitions={
            "START": {"Corporate SSO": 1.0},
            "Corporate SSO": {"ServiceNow": 0.35, "Active Directory": 0.30, "Splunk": 0.20, "AWS Console": 0.15},
            "ServiceNow": {"Active Directory": 0.25, "Splunk": 0.20, "END": 0.55},
            "Active Directory": {"AWS Console": 0.25, "Splunk": 0.20, "END": 0.55},
            "Splunk": {"ServiceNow": 0.20, "END": 0.80},
            "AWS Console": {"Active Directory": 0.20, "END": 0.80},
        },
        login_hour_mean=8.5,
        login_hour_std=1.0,
        sessions_per_day_mean=2.0,
        session_steps_mean=5,
        session_duration_mean_s=3300,
        auth_methods=["MFA", "SSH_KEY", "SSO"],
        auth_failure_rate=0.02,
        weekend_activity_prob=0.15,
        vpn_prob=0.18,
        secondary_device_prob=0.12,
    ),
    "service_batch": CohortArchetype(
        role="service_batch",
        entity_type=EntityType.SERVICE_ACCOUNT,
        resources=[
            _arc("API Gateway", "API_CALL", "medium", 2048),
            _arc("PostgreSQL", "QUERY", "critical", 131072),
            _arc("Kafka", "PUBLISH", "high", 65536),
            _arc("S3 Archive", "ACCESS", "high", 262144),
        ],
        transitions={
            "START": {"API Gateway": 1.0},
            "API Gateway": {"PostgreSQL": 0.45, "Kafka": 0.35, "S3 Archive": 0.20},
            "PostgreSQL": {"Kafka": 0.30, "S3 Archive": 0.25, "END": 0.45},
            "Kafka": {"PostgreSQL": 0.20, "END": 0.80},
            "S3 Archive": {"END": 1.0},
        },
        login_hour_mean=2.5,
        login_hour_std=1.5,
        sessions_per_day_mean=6.0,
        session_steps_mean=3,
        session_duration_mean_s=120,
        auth_methods=["API_KEY", "CERTIFICATE", "OAUTH2"],
        auth_failure_rate=0.005,
        weekend_activity_prob=0.50,
        vpn_prob=0.0,
        secondary_device_prob=0.0,
    ),
    "service_integration": CohortArchetype(
        role="service_integration",
        entity_type=EntityType.SERVICE_ACCOUNT,
        resources=[
            _arc("LDAP Bind", "LOGIN", "high", 512),
            _arc("REST Integration", "API_CALL", "medium", 4096),
            _arc("MongoDB", "QUERY", "critical", 98304),
        ],
        transitions={
            "START": {"LDAP Bind": 1.0},
            "LDAP Bind": {"REST Integration": 0.55, "MongoDB": 0.45},
            "REST Integration": {"MongoDB": 0.40, "END": 0.60},
            "MongoDB": {"REST Integration": 0.25, "END": 0.75},
        },
        login_hour_mean=4.0,
        login_hour_std=2.0,
        sessions_per_day_mean=8.0,
        session_steps_mean=3,
        session_duration_mean_s=90,
        auth_methods=["CERTIFICATE", "OAUTH2"],
        auth_failure_rate=0.005,
        weekend_activity_prob=0.45,
        vpn_prob=0.0,
        secondary_device_prob=0.0,
    ),
    "iot_sensor": CohortArchetype(
        role="iot_sensor",
        entity_type=EntityType.EDGE_DEVICE,
        resources=[
            _arc("MQTT Broker", "HEARTBEAT", "low", 256),
            _arc("Telemetry Stream", "TELEMETRY", "medium", 1024),
        ],
        transitions={
            "START": {"MQTT Broker": 1.0},
            "MQTT Broker": {"Telemetry Stream": 0.70, "END": 0.30},
            "Telemetry Stream": {"MQTT Broker": 0.60, "END": 0.40},
        },
        login_hour_mean=0.0,
        login_hour_std=0.0,
        sessions_per_day_mean=24.0,
        session_steps_mean=2,
        session_duration_mean_s=30,
        auth_methods=["CERTIFICATE", "PRE_SHARED_KEY"],
        auth_failure_rate=0.001,
        weekend_activity_prob=1.0,
        vpn_prob=0.0,
        secondary_device_prob=0.0,
    ),
    "scada_plc": CohortArchetype(
        role="scada_plc",
        entity_type=EntityType.EDGE_DEVICE,
        resources=[
            _arc("Modbus Gateway", "HEARTBEAT", "high", 128),
            _arc("SCADA Register", "TELEMETRY", "critical", 512),
            _arc("Firmware Channel", "ACCESS", "critical", 2048),
        ],
        transitions={
            "START": {"Modbus Gateway": 1.0},
            "Modbus Gateway": {"SCADA Register": 0.80, "END": 0.20},
            "SCADA Register": {"Modbus Gateway": 0.50, "Firmware Channel": 0.10, "END": 0.40},
            "Firmware Channel": {"Modbus Gateway": 0.30, "END": 0.70},
        },
        login_hour_mean=0.0,
        login_hour_std=0.0,
        sessions_per_day_mean=24.0,
        session_steps_mean=2,
        session_duration_mean_s=15,
        auth_methods=["CERTIFICATE"],
        auth_failure_rate=0.001,
        weekend_activity_prob=1.0,
        vpn_prob=0.0,
        secondary_device_prob=0.0,
    ),
}

# How many users per role when building a population (fractions, normalised at build time)
USER_ROLE_MIX: dict[str, float] = {
    "developer": 0.40,
    "hr": 0.12,
    "finance": 0.12,
    "it_admin": 0.10,
    "night_shift_operator": 0.08,
    # remainder -> sales-like via developer cohort variation handled by index
}

SERVICE_ROLE_MIX: dict[str, float] = {
    "service_batch": 0.55,
    "service_integration": 0.45,
}

EDGE_ROLE_MIX: dict[str, float] = {
    "iot_sensor": 0.65,
    "scada_plc": 0.35,
}


def _pick_geo(rng: np.random.Generator) -> GeoAnchor:
    site = rng.choice(ENTERPRISE_SITES)
    return GeoAnchor(
        country=site["country"],
        city=site["city"],
        latitude=round(float(site["lat"]) + rng.normal(0, 0.03), 4),
        longitude=round(float(site["lon"]) + rng.normal(0, 0.03), 4),
    )


def _ip_network(rng: np.random.Generator) -> str:
    return f"10.{rng.integers(1, 254)}.{rng.integers(0, 255)}.0/24"


def _mac(rng: np.random.Generator) -> str:
    octets = rng.integers(0, 256, size=6)
    return ":".join(f"{o:02x}" for o in octets)


def _build_devices(entity_id: str, cohort: CohortArchetype, rng: np.random.Generator) -> list[RegisteredDevice]:
    if cohort.entity_type == EntityType.EDGE_DEVICE:
        os_name = "Embedded Linux 5.4" if cohort.role == "iot_sensor" else "RTOS v3.2"
        firmware = "FW-2.1.0" if cohort.role == "iot_sensor" else "FW-4.0.2"
        protocol = "MQTT" if cohort.role == "iot_sensor" else "Modbus/TCP"
    elif cohort.entity_type == EntityType.SERVICE_ACCOUNT:
        os_name = "Linux Container"
        firmware = "N/A"
        protocol = "HTTPS"
    else:
        os_name = str(rng.choice(["Windows 11", "macOS 14", "Ubuntu 22.04"]))
        firmware = "N/A"
        protocol = str(rng.choice(["HTTPS", "HTTPS", "SSH"]))

    primary = RegisteredDevice(
        device_id=f"{entity_id}-D1",
        device_os=os_name,
        device_firmware=firmware,
        device_protocol=protocol,
        device_mac=_mac(rng),
        is_primary=True,
    )
    devices = [primary]
    if cohort.entity_type == EntityType.USER and rng.random() < cohort.secondary_device_prob:
        devices.append(
            RegisteredDevice(
                device_id=f"{entity_id}-D2",
                device_os=str(rng.choice(["iOS 17", "Android 14", "Windows 11"])),
                device_firmware="N/A",
                device_protocol="HTTPS",
                device_mac=_mac(rng),
                is_primary=False,
            )
        )
    return devices


def _working_days(rng: np.random.Generator, cohort: CohortArchetype) -> set[int]:
    if cohort.entity_type == EntityType.EDGE_DEVICE:
        return set(range(7))
    days = {0, 1, 2, 3, 4}
    if rng.random() < 0.20:
        days.add(5)
    return days


def _assign_roles(count: int, mix: dict[str, float], rng: np.random.Generator) -> list[str]:
    roles = list(mix.keys())
    weights = np.array([mix[r] for r in roles], dtype=float)
    weights /= weights.sum()
    return list(rng.choice(roles, size=count, p=weights))


def build_population(
    *,
    n_users: int,
    n_service_accounts: int,
    n_edge_devices: int,
) -> list[EntityBehavioralProfile]:
    """Materialise the enterprise population as behavioural digital twins."""
    rng = get_rng("generator.entities")
    profiles: list[EntityBehavioralProfile] = []

    user_roles = _assign_roles(n_users, USER_ROLE_MIX, rng)
    for idx, role in enumerate(user_roles[:n_users], start=1):
        cohort = COHORT_ARCHETYPES[role]
        eid = f"USR-{idx:04d}"
        profiles.append(
            EntityBehavioralProfile(
                entity_id=eid,
                entity_type=EntityType.USER,
                role=role,
                cohort=cohort,
                home_geo=_pick_geo(rng),
                ip_network=_ip_network(rng),
                working_days=_working_days(rng, cohort),
                preferred_login_hour=float(rng.normal(cohort.login_hour_mean, cohort.login_hour_std)),
                devices=_build_devices(eid, cohort, rng),
                primary_auth_method=str(rng.choice(cohort.auth_methods)),
                rng=get_rng(f"generator.entity.{eid}"),
            )
        )

    svc_roles = _assign_roles(n_service_accounts, SERVICE_ROLE_MIX, rng)
    for idx, role in enumerate(svc_roles[:n_service_accounts], start=1):
        cohort = COHORT_ARCHETYPES[role]
        eid = f"SVC-{idx:03d}"
        profiles.append(
            EntityBehavioralProfile(
                entity_id=eid,
                entity_type=EntityType.SERVICE_ACCOUNT,
                role=role,
                cohort=cohort,
                home_geo=_pick_geo(rng),
                ip_network=_ip_network(rng),
                working_days=_working_days(rng, cohort),
                preferred_login_hour=float(rng.normal(cohort.login_hour_mean, cohort.login_hour_std)),
                devices=_build_devices(eid, cohort, rng),
                primary_auth_method=str(rng.choice(cohort.auth_methods)),
                rng=get_rng(f"generator.entity.{eid}"),
            )
        )

    edge_roles = _assign_roles(n_edge_devices, EDGE_ROLE_MIX, rng)
    for idx, role in enumerate(edge_roles[:n_edge_devices], start=1):
        cohort = COHORT_ARCHETYPES[role]
        eid = f"EDG-{idx:03d}"
        profiles.append(
            EntityBehavioralProfile(
                entity_id=eid,
                entity_type=EntityType.EDGE_DEVICE,
                role=role,
                cohort=cohort,
                home_geo=_pick_geo(rng),
                ip_network=_ip_network(rng),
                working_days=_working_days(rng, cohort),
                preferred_login_hour=0.0,
                devices=_build_devices(eid, cohort, rng),
                primary_auth_method=str(rng.choice(cohort.auth_methods)),
                rng=get_rng(f"generator.entity.{eid}"),
            )
        )

    return profiles


def profile_to_record(profile: EntityBehavioralProfile) -> dict[str, Any]:
    """JSON-serialisable description of one entity's ground-truth behaviour.

    This is the *generator's* definition of the entity, not the *learned*
    profile that Phase 5 infers from observed events. Keeping both lets the
    evaluation ask how closely a learned baseline recovers the truth.
    """
    cohort = profile.cohort
    return {
        "entity_id": profile.entity_id,
        "entity_type": profile.entity_type.value,
        "role": profile.role,
        "home_country": profile.home_geo.country,
        "home_city": profile.home_geo.city,
        "home_latitude": profile.home_geo.latitude,
        "home_longitude": profile.home_geo.longitude,
        "ip_network": profile.ip_network,
        "working_days": sorted(profile.working_days),
        "preferred_login_hour": round(profile.preferred_login_hour, 4),
        "primary_auth_method": profile.primary_auth_method,
        "auth_methods": list(cohort.auth_methods),
        "auth_failure_rate": cohort.auth_failure_rate,
        "devices": [asdict(device) for device in profile.devices],
        "resources": [asdict(resource) for resource in cohort.resources],
        "resource_transitions": {
            state: dict(options) for state, options in cohort.transitions.items()
        },
        "sessions_per_day_mean": cohort.sessions_per_day_mean,
        "session_duration_mean_s": cohort.session_duration_mean_s,
    }


def record_to_profile(record: dict[str, Any]) -> EntityBehavioralProfile:
    """Rebuild a behavioural profile from an ``entities.json`` roster row.

    Prefers the role's cohort archetype (same digital-twin template used at
    generation time) and restores personal geography, devices and auth from the
    persisted record so live attack injection mutates the real entity.
    """
    entity_id = str(record["entity_id"])
    role = str(record["role"])
    entity_type = EntityType(str(record["entity_type"]))

    if role in COHORT_ARCHETYPES:
        cohort = COHORT_ARCHETYPES[role]
    else:
        resources = [
            ResourceDef(
                name=str(item["name"]),
                action=str(item.get("action", "ACCESS")),
                sensitivity=str(item.get("sensitivity", "medium")),
                typical_bytes=int(item.get("typical_bytes", 4096)),
            )
            for item in record.get("resources", [])
        ]
        if not resources:
            resources = [_arc("Corporate SSO", "LOGIN", "low", 512)]
        transitions = {
            str(state): {str(dst): float(prob) for dst, prob in options.items()}
            for state, options in (record.get("resource_transitions") or {}).items()
        }
        cohort = CohortArchetype(
            role=role,
            entity_type=entity_type,
            resources=resources,
            transitions=transitions or {"START": {resources[0].name: 1.0}},
            login_hour_mean=float(record.get("preferred_login_hour", 9.0)),
            login_hour_std=0.8,
            sessions_per_day_mean=float(record.get("sessions_per_day_mean", 1.5)),
            session_steps_mean=4,
            session_duration_mean_s=float(record.get("session_duration_mean_s", 1800.0)),
            auth_methods=list(record.get("auth_methods") or [record.get("primary_auth_method", "SSO")]),
            auth_failure_rate=float(record.get("auth_failure_rate", 0.03)),
            weekend_activity_prob=0.05,
            vpn_prob=0.08,
            secondary_device_prob=0.08,
        )

    devices_raw = record.get("devices") or []
    devices = [
        RegisteredDevice(
            device_id=str(item["device_id"]),
            device_os=str(item.get("device_os", "Unknown")),
            device_firmware=str(item.get("device_firmware", "UNKNOWN")),
            device_protocol=str(item.get("device_protocol", "HTTPS")),
            device_mac=str(item.get("device_mac", "00:00:00:00:00:00")),
            is_primary=bool(item.get("is_primary", True)),
        )
        for item in devices_raw
    ]
    if not devices:
        devices = _build_devices(entity_id, cohort, get_rng(f"generator.entity.{entity_id}"))

    working = record.get("working_days")
    working_days = set(int(day) for day in working) if working else set(range(5))

    return EntityBehavioralProfile(
        entity_id=entity_id,
        entity_type=entity_type,
        role=role,
        cohort=cohort,
        home_geo=GeoAnchor(
            country=str(record.get("home_country", "India")),
            city=str(record.get("home_city", "Chennai")),
            latitude=float(record.get("home_latitude", 13.0827)),
            longitude=float(record.get("home_longitude", 80.2707)),
        ),
        ip_network=str(record.get("ip_network", "10.0.0.0/24")),
        working_days=working_days,
        preferred_login_hour=float(record.get("preferred_login_hour", cohort.login_hour_mean)),
        devices=devices,
        primary_auth_method=str(
            record.get("primary_auth_method") or cohort.auth_methods[0]
        ),
        rng=get_rng(f"generator.entity.{entity_id}"),
    )


def load_population_records(entities_doc: dict[str, Any] | list[Any]) -> list[EntityBehavioralProfile]:
    """Load every entity profile from an ``entities`` artifact document."""
    if isinstance(entities_doc, dict) and isinstance(entities_doc.get("entities"), list):
        records = entities_doc["entities"]
    elif isinstance(entities_doc, list):
        records = entities_doc
    else:
        raise ValueError("entities artifact must be a list or {entities: [...]}")
    return [record_to_profile(record) for record in records]


def population_to_records(
    profiles: list[EntityBehavioralProfile],
) -> list[dict[str, Any]]:
    """Serialise a whole population for the ``entities`` artifact."""
    return [profile_to_record(profile) for profile in profiles]


def random_ip_in_subnet(network_cidr: str, rng: np.random.Generator) -> str:
    network = ipaddress.IPv4Network(network_cidr, strict=False)
    hosts = list(network.hosts())
    return str(rng.choice(hosts)) if hosts else str(network.network_address)


def pick_session_location(
    profile: EntityBehavioralProfile,
    noise_cfg: dict[str, Any],
) -> GeoAnchor:
    """Home office by default; occasional benign VPN egress."""
    rng = profile.rng
    vpn_prob = float(noise_cfg.get("off_profile_location_prob", profile.cohort.vpn_prob))
    if rng.random() < vpn_prob:
        remote = rng.choice(REMOTE_EGRESS)
        return GeoAnchor(
            country=remote["country"],
            city=remote["city"],
            latitude=remote["lat"],
            longitude=remote["lon"],
        )
    return profile.home_geo
