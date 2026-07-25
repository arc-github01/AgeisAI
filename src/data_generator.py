"""Synthetic enterprise behavioral access log generator for autoencoder training.

Generates chronologically ordered events where each entity (user, service
account, or device) maintains a consistent behavioral profile. Normal activity
dominates (~97-98%); injected attack campaigns modify otherwise realistic
profiles rather than producing random outliers.

Usage:
    python -m src.data_generator
"""

from __future__ import annotations

import ipaddress
import math
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from faker import Faker

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TARGET_EVENTS = 100_000
N_USERS = 100
N_SERVICE_ACCOUNTS = 30
N_DEVICES = 70
ATTACK_FRACTION = 0.025  # ~2.5% anomalous rows
START_DATE = datetime(2025, 1, 6, 0, 0, 0)  # Monday
SIMULATION_DAYS = 90
MASTER_SEED = 20260725
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "behavioral_logs.csv"

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Role templates: resources, typical hours, auth, protocols
ROLE_PROFILES: dict[str, dict[str, Any]] = {
    "developer": {
        "resources": [
            ("GitHub", "development"),
            ("Jira", "project_management"),
            ("Jenkins", "ci_cd"),
            ("Confluence", "documentation"),
            ("Docker Registry", "development"),
        ],
        "login_hour_mean": 9.0,
        "login_hour_std": 0.8,
        "session_duration_mean": 45,
        "session_duration_std": 15,
        "command_count_mean": 85,
        "command_count_std": 25,
        "auth_methods": ["SSO", "MFA", "SSH_KEY"],
        "protocols": ["HTTPS", "SSH"],
        "os_versions": ["Windows 11", "macOS 14", "Ubuntu 22.04"],
        "weekend_prob": 0.08,
        "vpn_prob": 0.12,
    },
    "hr": {
        "resources": [
            ("Workday", "hr"),
            ("Payroll Portal", "hr"),
            ("Benefits Admin", "hr"),
            ("ADP", "hr"),
        ],
        "login_hour_mean": 8.5,
        "login_hour_std": 0.6,
        "session_duration_mean": 35,
        "session_duration_std": 10,
        "command_count_mean": 25,
        "command_count_std": 8,
        "auth_methods": ["SSO", "MFA"],
        "protocols": ["HTTPS"],
        "os_versions": ["Windows 11", "Windows 10"],
        "weekend_prob": 0.03,
        "vpn_prob": 0.05,
    },
    "finance": {
        "resources": [
            ("SAP", "erp"),
            ("Oracle Financials", "erp"),
            ("Excel Online", "productivity"),
            ("Treasury Portal", "finance"),
        ],
        "login_hour_mean": 8.0,
        "login_hour_std": 0.5,
        "session_duration_mean": 50,
        "session_duration_std": 12,
        "command_count_mean": 40,
        "command_count_std": 12,
        "auth_methods": ["SSO", "MFA", "SMARTCARD"],
        "protocols": ["HTTPS"],
        "os_versions": ["Windows 11", "Windows 10"],
        "weekend_prob": 0.04,
        "vpn_prob": 0.08,
    },
    "it_admin": {
        "resources": [
            ("Active Directory", "infrastructure"),
            ("ServiceNow", "it_ops"),
            ("VMware vCenter", "infrastructure"),
            ("Splunk", "security"),
            ("AWS Console", "cloud"),
        ],
        "login_hour_mean": 8.5,
        "login_hour_std": 1.0,
        "session_duration_mean": 55,
        "session_duration_std": 18,
        "command_count_mean": 120,
        "command_count_std": 35,
        "auth_methods": ["MFA", "SSH_KEY", "SSO"],
        "protocols": ["HTTPS", "SSH", "RDP"],
        "os_versions": ["Windows Server 2022", "Ubuntu 22.04", "macOS 14"],
        "weekend_prob": 0.15,
        "vpn_prob": 0.20,
    },
    "sales": {
        "resources": [
            ("Salesforce", "crm"),
            ("HubSpot", "crm"),
            ("LinkedIn Sales", "sales"),
            ("Zoom", "communication"),
        ],
        "login_hour_mean": 9.5,
        "login_hour_std": 1.2,
        "session_duration_mean": 40,
        "session_duration_std": 14,
        "command_count_mean": 30,
        "command_count_std": 10,
        "auth_methods": ["SSO", "MFA"],
        "protocols": ["HTTPS"],
        "os_versions": ["Windows 11", "macOS 14", "iOS 17"],
        "weekend_prob": 0.10,
        "vpn_prob": 0.15,
    },
}

SERVICE_ACCOUNT_PROFILE = {
    "resources": [
        ("REST API Gateway", "api"),
        ("PostgreSQL", "database"),
        ("MongoDB", "database"),
        ("Kafka", "messaging"),
        ("S3 Bucket", "storage"),
        ("LDAP", "directory"),
    ],
    "login_hour_mean": 3.0,  # off-peak batch jobs
    "login_hour_std": 2.0,
    "session_duration_mean": 8,
    "session_duration_std": 4,
    "command_count_mean": 200,
    "command_count_std": 80,
    "auth_methods": ["API_KEY", "CERTIFICATE", "OAUTH2"],
    "protocols": ["HTTPS", "TCP", "AMQP"],
    "os_versions": ["Linux Container", "Windows Server 2019"],
    "weekend_prob": 0.35,
    "vpn_prob": 0.0,
}

DEVICE_PROFILE = {
    "resources": [
        ("MQTT Broker", "iot"),
        ("SCADA Gateway", "ot"),
        ("Telemetry Endpoint", "iot"),
        ("Firmware Update Server", "ot"),
        ("HTTPS Heartbeat", "iot"),
    ],
    "login_hour_mean": 0.0,  # 24/7 heartbeat
    "login_hour_std": 0.0,
    "session_duration_mean": 2,
    "session_duration_std": 1,
    "command_count_mean": 5,
    "command_count_std": 2,
    "auth_methods": ["CERTIFICATE", "PRE_SHARED_KEY"],
    "protocols": ["MQTT", "HTTPS", "Modbus/TCP"],
    "os_versions": ["Embedded Linux 5.4", "RTOS v3.2", "Firmware 2.1"],
    "weekend_prob": 1.0,  # always on
    "vpn_prob": 0.0,
}

REMOTE_LOCATIONS = [
    {"country": "United Kingdom", "city": "London", "lat": 51.5074, "lon": -0.1278},
    {"country": "Germany", "city": "Berlin", "lat": 52.5200, "lon": 13.4050},
    {"country": "Japan", "city": "Tokyo", "lat": 35.6762, "lon": 139.6503},
    {"country": "Australia", "city": "Sydney", "lat": -33.8688, "lon": 151.2093},
    {"country": "Brazil", "city": "Sao Paulo", "lat": -23.5505, "lon": -46.6333},
    {"country": "Singapore", "city": "Singapore", "lat": 1.3521, "lon": 103.8198},
    {"country": "India", "city": "Mumbai", "lat": 19.0760, "lon": 72.8777},
]

# Primary office locations — employees cluster near hubs with slight GPS jitter
ENTERPRISE_HUBS = [
    {"country": "United States", "city": "Chicago", "lat": 41.8781, "lon": -87.6298},
    {"country": "United States", "city": "Atlanta", "lat": 33.7490, "lon": -84.3880},
    {"country": "United States", "city": "Dallas", "lat": 32.7767, "lon": -96.7970},
    {"country": "United Kingdom", "city": "London", "lat": 51.5074, "lon": -0.1278},
    {"country": "Germany", "city": "Frankfurt", "lat": 50.1109, "lon": 8.6821},
    {"country": "India", "city": "Bangalore", "lat": 12.9716, "lon": 77.5946},
    {"country": "India", "city": "Pune", "lat": 18.5204, "lon": 73.8567},
]

ATTACK_TYPES = [
    "BRUTE_FORCE",
    "CREDENTIAL_STUFFING",
    "IMPOSSIBLE_TRAVEL",
    "DEVICE_SPOOFING",
    "LATERAL_MOVEMENT",
    "LOW_AND_SLOW_EXFILTRATION",
    "INSIDER_DRIFT",
]

# Relative campaign sizes (normalized at injection time)
ATTACK_MIX = {
    "BRUTE_FORCE": 0.22,
    "CREDENTIAL_STUFFING": 0.18,
    "IMPOSSIBLE_TRAVEL": 0.14,
    "DEVICE_SPOOFING": 0.12,
    "LATERAL_MOVEMENT": 0.16,
    "LOW_AND_SLOW_EXFILTRATION": 0.10,
    "INSIDER_DRIFT": 0.08,
}

LATERAL_MOVEMENT_RESOURCES = [
    ("Domain Controller", "infrastructure"),
    ("Backup Server", "infrastructure"),
    ("HR Database", "database"),
    ("Finance Share", "finance"),
    ("Admin Panel", "infrastructure"),
    ("Internal Wiki", "documentation"),
]

EXFIL_RESOURCES = [
    ("File Share Export", "storage"),
    ("S3 Bulk Download", "cloud"),
    ("Database Dump Tool", "database"),
    ("Email Archive", "communication"),
]

INSIDER_ESCALATION = [
    ("Privileged Access Manager", "security"),
    ("Audit Logs", "security"),
    ("User Provisioning", "infrastructure"),
    ("Firewall Rules", "security"),
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class GeoLocation:
    country: str
    city: str
    latitude: float
    longitude: float


@dataclass
class EntityProfile:
    """Persistent behavioral baseline for one actor in the enterprise."""

    entity_id: str
    entity_type: str  # user | service_account | device
    role: str
    home_location: GeoLocation
    ip_network: str  # CIDR, e.g. 10.42.15.0/24
    preferred_login_hour: float
    working_days: set[int]  # 0=Monday .. 6=Sunday
    resources: list[tuple[str, str]]
    auth_method: str
    avg_session_duration: float
    avg_command_count: float
    device_fingerprint: str
    os_version: str
    protocol: str
    profile_template: dict[str, Any]
    rng: np.random.Generator = field(repr=False)


@dataclass
class EntityState:
    """Runtime memory of the most recent observation per entity."""

    last_timestamp: datetime | None = None
    last_location: GeoLocation | None = None
    last_fingerprint: str | None = None
    known_fingerprints: set[str] = field(default_factory=set)
    known_locations: set[tuple[str, str]] = field(default_factory=set)


@dataclass
class RawEvent:
    """Pre-derivation event; temporal fields filled in a second pass."""

    event_id: str
    entity_id: str
    entity_type: str
    timestamp: datetime
    source_ip: str
    country: str
    city: str
    latitude: float
    longitude: float
    resource_accessed: str
    resource_category: str
    auth_method: str
    login_success: bool
    failed_attempts: int
    session_duration: float
    command_count: int
    device_fingerprint: str
    os_version: str
    protocol: str
    label: str
    attack_type: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two WGS-84 points."""
    radius = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def random_ip_in_subnet(network_cidr: str, rng: np.random.Generator) -> str:
    network = ipaddress.IPv4Network(network_cidr, strict=False)
    # Avoid network/broadcast addresses on small subnets
    hosts = list(network.hosts())
    if not hosts:
        return str(network.network_address)
    return str(rng.choice(hosts))


def pick_working_days(rng: np.random.Generator) -> set[int]:
    """Most entities work Mon-Fri; some include Saturday."""
    days = {0, 1, 2, 3, 4}
    if rng.random() < 0.25:
        days.add(5)
    return days


def jitter_hour(base_hour: float, std: float, rng: np.random.Generator) -> float:
    """Login hour with gaussian jitter, clamped to [0, 23.99]."""
    return float(np.clip(rng.normal(base_hour, std), 0.0, 23.99))


def make_timestamp(
    day_offset: int,
    login_hour: float,
    rng: np.random.Generator,
) -> datetime:
    base = START_DATE + timedelta(days=day_offset)
    hour = int(login_hour)
    minute = int((login_hour - hour) * 60) + int(rng.integers(0, 15))
    minute = min(minute, 59)
    second = int(rng.integers(0, 59))
    return base.replace(hour=hour, minute=minute, second=second, microsecond=0)


def new_event_id() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Profile factory
# ---------------------------------------------------------------------------


class ProfileFactory:
    """Builds the full entity population with unique baselines."""

    def __init__(self, faker: Faker, master_seed: int) -> None:
        self.faker = faker
        self.master_seed = master_seed

    def _home_location(self, rng: np.random.Generator) -> GeoLocation:
        """Assign a realistic office geo anchor with small coordinate jitter."""
        hub = rng.choice(ENTERPRISE_HUBS)
        lat = hub["lat"] + float(rng.normal(0, 0.04))
        lon = hub["lon"] + float(rng.normal(0, 0.04))
        return GeoLocation(
            country=hub["country"],
            city=hub["city"],
            latitude=round(lat, 4),
            longitude=round(lon, 4),
        )

    def _ip_network(self, rng: np.random.Generator) -> str:
        second = rng.integers(1, 254)
        third = rng.integers(0, 255)
        return f"10.{second}.{third}.0/24"

    def _device_fingerprint(self, entity_id: str, os_version: str, rng: np.random.Generator) -> str:
        suffix = rng.integers(100000, 999999)
        return f"FP-{entity_id[-6:]}-{os_version[:3].upper()}-{suffix}"

    def _build_entity(
        self,
        entity_id: str,
        entity_type: str,
        role: str,
        template: dict[str, Any],
        component: str,
    ) -> EntityProfile:
        rng = np.random.default_rng(self.master_seed + hash(component) % 10_000_000)
        home = self._home_location(rng)
        resources = template["resources"].copy()
        rng.shuffle(resources)
        auth = str(rng.choice(template["auth_methods"]))
        os_version = str(rng.choice(template["os_versions"]))
        protocol = str(rng.choice(template["protocols"]))
        return EntityProfile(
            entity_id=entity_id,
            entity_type=entity_type,
            role=role,
            home_location=home,
            ip_network=self._ip_network(rng),
            preferred_login_hour=float(
                rng.normal(template["login_hour_mean"], template["login_hour_std"])
            ),
            working_days=pick_working_days(rng),
            resources=resources,
            auth_method=auth,
            avg_session_duration=float(
                max(1.0, rng.normal(template["session_duration_mean"], template["session_duration_std"]))
            ),
            avg_command_count=float(
                max(1.0, rng.normal(template["command_count_mean"], template["command_count_std"]))
            ),
            device_fingerprint=self._device_fingerprint(entity_id, os_version, rng),
            os_version=os_version,
            protocol=protocol,
            profile_template=template,
            rng=rng,
        )

    def create_population(self) -> list[EntityProfile]:
        profiles: list[EntityProfile] = []

        role_counts = {
            "developer": 40,
            "hr": 15,
            "finance": 15,
            "it_admin": 10,
            "sales": 20,
        }
        user_idx = 0
        for role, count in role_counts.items():
            for _ in range(count):
                user_idx += 1
                eid = f"USR-{user_idx:04d}"
                profiles.append(
                    self._build_entity(eid, "user", role, ROLE_PROFILES[role], f"user-{eid}")
                )

        for i in range(1, N_SERVICE_ACCOUNTS + 1):
            eid = f"SVC-{i:03d}"
            profiles.append(
                self._build_entity(
                    eid, "service_account", "service", SERVICE_ACCOUNT_PROFILE, f"svc-{eid}"
                )
            )

        for i in range(1, N_DEVICES + 1):
            eid = f"DEV-{i:03d}"
            profiles.append(
                self._build_entity(eid, "device", "iot", DEVICE_PROFILE, f"dev-{eid}")
            )

        return profiles


# ---------------------------------------------------------------------------
# Normal behaviour generator
# ---------------------------------------------------------------------------


class NormalEventGenerator:
    """Samples habitual sessions from entity profiles with natural variation."""

    def __init__(self, profiles: list[EntityProfile], target_normal: int) -> None:
        self.profiles = profiles
        self.target_normal = target_normal
        self._session_counts = self._allocate_sessions(profiles, target_normal)

    @staticmethod
    def _entity_weight(profile: EntityProfile) -> float:
        if profile.entity_type == "user":
            return 1.0
        if profile.entity_type == "service_account":
            return 0.45
        return 0.12

    def _allocate_sessions(
        self, profiles: list[EntityProfile], target: int
    ) -> dict[str, int]:
        """Split the normal-event budget proportionally across entities."""
        weights = [self._entity_weight(p) for p in profiles]
        total_weight = sum(weights)
        counts = [max(5, int(target * w / total_weight)) for w in weights]
        assigned = sum(counts)
        # Distribute rounding remainder to the most active user entities
        remainder = target - assigned
        order = sorted(range(len(profiles)), key=lambda i: weights[i], reverse=True)
        idx = 0
        while remainder > 0 and order:
            counts[order[idx % len(order)]] += 1
            remainder -= 1
            idx += 1
        return {profiles[i].entity_id: counts[i] for i in range(len(profiles))}

    def _session_count_for_entity(self, profile: EntityProfile) -> int:
        return self._session_counts[profile.entity_id]

    def _pick_day(self, profile: EntityProfile) -> int:
        rng = profile.rng
        for _ in range(20):
            day = int(rng.integers(0, SIMULATION_DAYS))
            weekday = (START_DATE + timedelta(days=day)).weekday()
            if weekday in profile.working_days:
                return day
            if rng.random() < profile.profile_template.get("weekend_prob", 0.05):
                return day
        return int(rng.integers(0, SIMULATION_DAYS))

    def _maybe_vpn_location(
        self, profile: EntityProfile, use_vpn: bool
    ) -> tuple[GeoLocation, str]:
        rng = profile.rng
        if use_vpn:
            remote = rng.choice(REMOTE_LOCATIONS)
            loc = GeoLocation(
                country=remote["country"],
                city=remote["city"],
                latitude=remote["lat"],
                longitude=remote["lon"],
            )
            # VPN egress still uses corporate subnet occasionally
            ip = random_ip_in_subnet(profile.ip_network, rng)
            return loc, ip
        return profile.home_location, random_ip_in_subnet(profile.ip_network, rng)

    def generate(self) -> list[RawEvent]:
        events: list[RawEvent] = []
        for profile in self.profiles:
            n_sessions = self._session_count_for_entity(profile)
            template = profile.profile_template
            for _ in range(n_sessions):
                day = self._pick_day(profile)
                hour = jitter_hour(
                    profile.preferred_login_hour,
                    template.get("login_hour_std", 1.0) * 0.5,
                    profile.rng,
                )
                # ±30 minute natural variation via jitter_hour + make_timestamp noise
                ts = make_timestamp(day, hour, profile.rng)

                use_vpn = profile.rng.random() < template.get("vpn_prob", 0.0)
                location, source_ip = self._maybe_vpn_location(profile, use_vpn)

                resource, category = profile.resources[
                    profile.rng.integers(0, len(profile.resources))
                ]

                session_duration = float(
                    max(1.0, profile.rng.normal(profile.avg_session_duration, profile.avg_session_duration * 0.25))
                )
                command_count = int(
                    max(1, profile.rng.normal(profile.avg_command_count, profile.avg_command_count * 0.2))
                )

                # Benign fat-finger failures (~3%)
                login_success = profile.rng.random() > 0.03
                failed_attempts = int(profile.rng.integers(1, 3)) if not login_success else 0

                # Secondary device occasionally
                fingerprint = profile.device_fingerprint
                os_version = profile.os_version
                if profile.rng.random() < template.get("secondary_device_prob", 0.08):
                    os_version = str(profile.rng.choice(template["os_versions"]))
                    fingerprint = f"FP-{profile.entity_id[-6:]}-{os_version[:3].upper()}-{profile.rng.integers(100000, 999999)}"

                events.append(
                    RawEvent(
                        event_id=new_event_id(),
                        entity_id=profile.entity_id,
                        entity_type=profile.entity_type,
                        timestamp=ts,
                        source_ip=source_ip,
                        country=location.country,
                        city=location.city,
                        latitude=location.latitude,
                        longitude=location.longitude,
                        resource_accessed=resource,
                        resource_category=category,
                        auth_method=profile.auth_method,
                        login_success=login_success,
                        failed_attempts=failed_attempts,
                        session_duration=session_duration,
                        command_count=command_count,
                        device_fingerprint=fingerprint,
                        os_version=os_version,
                        protocol=profile.protocol,
                        label="normal",
                        attack_type="none",
                    )
                )

        # Trim or pad toward target by random subsample / duplicate days
        if len(events) > self.target_normal:
            random.shuffle(events)
            events = events[: self.target_normal]
        return events


# ---------------------------------------------------------------------------
# Attack injectors — each campaign mutates a real profile
# ---------------------------------------------------------------------------


class AttackInjector:
    """Injects coordinated anomaly campaigns into the event stream."""

    def __init__(self, profiles: list[EntityProfile], target_anomaly: int) -> None:
        self.profiles = {p.entity_id: p for p in profiles}
        self.users = [p for p in profiles if p.entity_type == "user"]
        self.target_anomaly = target_anomaly
        self.rng = np.random.default_rng(MASTER_SEED + 999)

    def _alloc_campaign_sizes(self) -> dict[str, int]:
        total_weight = sum(ATTACK_MIX.values())
        sizes = {
            attack: max(1, int(self.target_anomaly * weight / total_weight))
            for attack, weight in ATTACK_MIX.items()
        }
        # Adjust rounding drift
        diff = self.target_anomaly - sum(sizes.values())
        if diff > 0:
            sizes["BRUTE_FORCE"] += diff
        return sizes

    def inject_brute_force(self, n_events: int) -> list[RawEvent]:
        events: list[RawEvent] = []
        profile = self.rng.choice(self.users)
        day = int(self.rng.integers(10, SIMULATION_DAYS - 5))
        base_ts = START_DATE + timedelta(days=day, hours=2, minutes=15)
        attacker_ip = random_ip_in_subnet("203.0.113.0/24", self.rng)
        resource, category = profile.resources[0]

        per_burst = max(5, n_events // 4)
        ts = base_ts
        for _ in range(n_events):
            ts += timedelta(seconds=int(self.rng.integers(5, 45)))
            events.append(
                RawEvent(
                    event_id=new_event_id(),
                    entity_id=profile.entity_id,
                    entity_type="user",
                    timestamp=ts,
                    source_ip=attacker_ip,
                    country=profile.home_location.country,
                    city=profile.home_location.city,
                    latitude=profile.home_location.latitude,
                    longitude=profile.home_location.longitude,
                    resource_accessed=resource,
                    resource_category=category,
                    auth_method=profile.auth_method,
                    login_success=False,
                    failed_attempts=int(self.rng.integers(3, 12)),
                    session_duration=float(self.rng.uniform(0.1, 0.5)),
                    command_count=0,
                    device_fingerprint=profile.device_fingerprint,
                    os_version=profile.os_version,
                    protocol=profile.protocol,
                    label="anomaly",
                    attack_type="BRUTE_FORCE",
                )
            )
            if len(events) % per_burst == 0:
                ts += timedelta(minutes=int(self.rng.integers(20, 90)))
        return events

    def inject_credential_stuffing(self, n_events: int) -> list[RawEvent]:
        events: list[RawEvent] = []
        day = int(self.rng.integers(15, SIMULATION_DAYS - 5))
        base_ts = START_DATE + timedelta(days=day, hours=4, minutes=0)
        attacker_ip = random_ip_in_subnet("198.51.100.0/24", self.rng)
        victims = self.rng.choice(self.users, size=min(25, len(self.users)), replace=False)
        ts = base_ts
        for i in range(n_events):
            victim = victims[i % len(victims)]
            resource, category = victim.resources[0]
            ts += timedelta(seconds=int(self.rng.integers(3, 20)))
            events.append(
                RawEvent(
                    event_id=new_event_id(),
                    entity_id=victim.entity_id,
                    entity_type="user",
                    timestamp=ts,
                    source_ip=attacker_ip,
                    country="United States",
                    city="Unknown",
                    latitude=37.7749,
                    longitude=-122.4194,
                    resource_accessed=resource,
                    resource_category=category,
                    auth_method="PASSWORD",
                    login_success=self.rng.random() < 0.05,
                    failed_attempts=int(self.rng.integers(5, 20)),
                    session_duration=float(self.rng.uniform(0.1, 0.3)),
                    command_count=0,
                    device_fingerprint=f"FP-STUFF-{self.rng.integers(100000, 999999)}",
                    os_version="Unknown",
                    protocol="HTTPS",
                    label="anomaly",
                    attack_type="CREDENTIAL_STUFFING",
                )
            )
        return events

    def inject_impossible_travel(self, n_events: int) -> list[RawEvent]:
        events: list[RawEvent] = []
        profile = self.rng.choice(self.users)
        day = int(self.rng.integers(20, SIMULATION_DAYS - 10))
        home = profile.home_location
        remote_a = REMOTE_LOCATIONS[0]
        remote_b = REMOTE_LOCATIONS[2]

        # Legitimate login at home
        ts_home = START_DATE + timedelta(days=day, hours=9, minutes=10)
        resource, category = profile.resources[0]
        events.append(
            RawEvent(
                event_id=new_event_id(),
                entity_id=profile.entity_id,
                entity_type="user",
                timestamp=ts_home,
                source_ip=random_ip_in_subnet(profile.ip_network, profile.rng),
                country=home.country,
                city=home.city,
                latitude=home.latitude,
                longitude=home.longitude,
                resource_accessed=resource,
                resource_category=category,
                auth_method=profile.auth_method,
                login_success=True,
                failed_attempts=0,
                session_duration=profile.avg_session_duration,
                command_count=int(profile.avg_command_count),
                device_fingerprint=profile.device_fingerprint,
                os_version=profile.os_version,
                protocol=profile.protocol,
                label="normal",
                attack_type="none",
            )
        )

        # Impossible login 25 minutes later from distant country
        ts_remote = ts_home + timedelta(minutes=25)
        events.append(
            RawEvent(
                event_id=new_event_id(),
                entity_id=profile.entity_id,
                entity_type="user",
                timestamp=ts_remote,
                source_ip=random_ip_in_subnet("203.0.113.0/24", self.rng),
                country=remote_a["country"],
                city=remote_a["city"],
                latitude=remote_a["lat"],
                longitude=remote_a["lon"],
                resource_accessed=resource,
                resource_category=category,
                auth_method=profile.auth_method,
                login_success=True,
                failed_attempts=0,
                session_duration=profile.avg_session_duration * 0.5,
                command_count=int(profile.avg_command_count * 0.3),
                device_fingerprint=profile.device_fingerprint,
                os_version=profile.os_version,
                protocol=profile.protocol,
                label="anomaly",
                attack_type="IMPOSSIBLE_TRAVEL",
            )
        )

        # Additional scattered impossible-travel pairs to reach n_events
        while len(events) < n_events:
            profile = self.rng.choice(self.users)
            home = profile.home_location
            remote = self.rng.choice(REMOTE_LOCATIONS)
            d = int(self.rng.integers(25, SIMULATION_DAYS - 5))
            ts1 = START_DATE + timedelta(days=d, hours=14)
            ts2 = ts1 + timedelta(minutes=int(self.rng.integers(15, 40)))
            resource, category = profile.resources[0]
            events.extend(
                [
                    RawEvent(
                        event_id=new_event_id(),
                        entity_id=profile.entity_id,
                        entity_type="user",
                        timestamp=ts1,
                        source_ip=random_ip_in_subnet(profile.ip_network, profile.rng),
                        country=home.country,
                        city=home.city,
                        latitude=home.latitude,
                        longitude=home.longitude,
                        resource_accessed=resource,
                        resource_category=category,
                        auth_method=profile.auth_method,
                        login_success=True,
                        failed_attempts=0,
                        session_duration=profile.avg_session_duration,
                        command_count=int(profile.avg_command_count),
                        device_fingerprint=profile.device_fingerprint,
                        os_version=profile.os_version,
                        protocol=profile.protocol,
                        label="normal",
                        attack_type="none",
                    ),
                    RawEvent(
                        event_id=new_event_id(),
                        entity_id=profile.entity_id,
                        entity_type="user",
                        timestamp=ts2,
                        source_ip=random_ip_in_subnet("203.0.113.0/24", self.rng),
                        country=remote["country"],
                        city=remote["city"],
                        latitude=remote["lat"],
                        longitude=remote["lon"],
                        resource_accessed=resource,
                        resource_category=category,
                        auth_method=profile.auth_method,
                        login_success=True,
                        failed_attempts=0,
                        session_duration=profile.avg_session_duration,
                        command_count=int(profile.avg_command_count),
                        device_fingerprint=profile.device_fingerprint,
                        os_version=profile.os_version,
                        protocol=profile.protocol,
                        label="anomaly",
                        attack_type="IMPOSSIBLE_TRAVEL",
                    ),
                ]
            )
        return events[:n_events]

    def inject_device_spoofing(self, n_events: int) -> list[RawEvent]:
        events: list[RawEvent] = []
        for _ in range(n_events):
            profile = self.rng.choice(self.users)
            day = int(self.rng.integers(5, SIMULATION_DAYS - 5))
            ts = START_DATE + timedelta(days=day, hours=int(self.rng.integers(8, 18)))
            resource, category = profile.resources[0]
            spoof_fp = f"FP-SPOOF-{self.rng.integers(100000, 999999)}"
            events.append(
                RawEvent(
                    event_id=new_event_id(),
                    entity_id=profile.entity_id,
                    entity_type="user",
                    timestamp=ts,
                    source_ip=random_ip_in_subnet(profile.ip_network, profile.rng),
                    country=profile.home_location.country,
                    city=profile.home_location.city,
                    latitude=profile.home_location.latitude,
                    longitude=profile.home_location.longitude,
                    resource_accessed=resource,
                    resource_category=category,
                    auth_method=profile.auth_method,
                    login_success=True,
                    failed_attempts=0,
                    session_duration=profile.avg_session_duration,
                    command_count=int(profile.avg_command_count),
                    device_fingerprint=spoof_fp,
                    os_version=str(self.rng.choice(["Android 13", "Unknown Linux", "Windows 7"])),
                    protocol=profile.protocol,
                    label="anomaly",
                    attack_type="DEVICE_SPOOFING",
                )
            )
        return events

    def inject_lateral_movement(self, n_events: int) -> list[RawEvent]:
        events: list[RawEvent] = []
        profile = self.rng.choice([p for p in self.users if p.role in ("developer", "it_admin")])
        day = int(self.rng.integers(30, SIMULATION_DAYS - 5))
        ts = START_DATE + timedelta(days=day, hours=22, minutes=30)
        for i in range(n_events):
            resource, category = LATERAL_MOVEMENT_RESOURCES[i % len(LATERAL_MOVEMENT_RESOURCES)]
            ts += timedelta(minutes=int(self.rng.integers(2, 8)))
            events.append(
                RawEvent(
                    event_id=new_event_id(),
                    entity_id=profile.entity_id,
                    entity_type="user",
                    timestamp=ts,
                    source_ip=random_ip_in_subnet(profile.ip_network, profile.rng),
                    country=profile.home_location.country,
                    city=profile.home_location.city,
                    latitude=profile.home_location.latitude,
                    longitude=profile.home_location.longitude,
                    resource_accessed=resource,
                    resource_category=category,
                    auth_method=profile.auth_method,
                    login_success=True,
                    failed_attempts=0,
                    session_duration=float(self.rng.uniform(5, 15)),
                    command_count=int(self.rng.integers(40, 120)),
                    device_fingerprint=profile.device_fingerprint,
                    os_version=profile.os_version,
                    protocol=profile.protocol,
                    label="anomaly",
                    attack_type="LATERAL_MOVEMENT",
                )
            )
        return events

    def inject_low_and_slow_exfil(self, n_events: int) -> list[RawEvent]:
        events: list[RawEvent] = []
        profile = self.rng.choice(self.users)
        start_day = int(self.rng.integers(10, SIMULATION_DAYS - 30))
        duration_scale = 1.0
        for i in range(n_events):
            day = min(start_day + (i // 3), SIMULATION_DAYS - 1)
            hour = 2 + (i % 3)  # off-hours 2-4 AM
            ts = START_DATE + timedelta(days=day, hours=hour, minutes=int(self.rng.integers(0, 45)))
            resource, category = EXFIL_RESOURCES[i % len(EXFIL_RESOURCES)]
            duration_scale = min(4.0, duration_scale * 1.08)
            events.append(
                RawEvent(
                    event_id=new_event_id(),
                    entity_id=profile.entity_id,
                    entity_type="user",
                    timestamp=ts,
                    source_ip=random_ip_in_subnet(profile.ip_network, profile.rng),
                    country=profile.home_location.country,
                    city=profile.home_location.city,
                    latitude=profile.home_location.latitude,
                    longitude=profile.home_location.longitude,
                    resource_accessed=resource,
                    resource_category=category,
                    auth_method=profile.auth_method,
                    login_success=True,
                    failed_attempts=0,
                    session_duration=profile.avg_session_duration * duration_scale,
                    command_count=int(profile.avg_command_count * duration_scale),
                    device_fingerprint=profile.device_fingerprint,
                    os_version=profile.os_version,
                    protocol=profile.protocol,
                    label="anomaly",
                    attack_type="LOW_AND_SLOW_EXFILTRATION",
                )
            )
        return events

    def inject_insider_drift(self, n_events: int) -> list[RawEvent]:
        events: list[RawEvent] = []
        profile = self.rng.choice([p for p in self.users if p.role in ("hr", "finance", "sales")])
        start_day = int(self.rng.integers(20, SIMULATION_DAYS - 25))
        for i in range(n_events):
            day = min(start_day + i, SIMULATION_DAYS - 1)
            ts = START_DATE + timedelta(
                days=day,
                hours=int(profile.preferred_login_hour),
                minutes=int(self.rng.integers(0, 30)),
            )
            if i < n_events // 2:
                resource, category = profile.resources[0]
                label = "normal"
                attack = "none"
            else:
                resource, category = INSIDER_ESCALATION[i % len(INSIDER_ESCALATION)]
                label = "anomaly"
                attack = "INSIDER_DRIFT"
            events.append(
                RawEvent(
                    event_id=new_event_id(),
                    entity_id=profile.entity_id,
                    entity_type="user",
                    timestamp=ts,
                    source_ip=random_ip_in_subnet(profile.ip_network, profile.rng),
                    country=profile.home_location.country,
                    city=profile.home_location.city,
                    latitude=profile.home_location.latitude,
                    longitude=profile.home_location.longitude,
                    resource_accessed=resource,
                    resource_category=category,
                    auth_method=profile.auth_method,
                    login_success=True,
                    failed_attempts=0,
                    session_duration=profile.avg_session_duration * (1 + 0.05 * i),
                    command_count=int(profile.avg_command_count * (1 + 0.03 * i)),
                    device_fingerprint=profile.device_fingerprint,
                    os_version=profile.os_version,
                    protocol=profile.protocol,
                    label=label,
                    attack_type=attack,
                )
            )
        return events

    def inject_all(self) -> list[RawEvent]:
        sizes = self._alloc_campaign_sizes()
        campaigns: list[RawEvent] = []
        campaigns.extend(self.inject_brute_force(sizes["BRUTE_FORCE"]))
        campaigns.extend(self.inject_credential_stuffing(sizes["CREDENTIAL_STUFFING"]))
        campaigns.extend(self.inject_impossible_travel(sizes["IMPOSSIBLE_TRAVEL"]))
        campaigns.extend(self.inject_device_spoofing(sizes["DEVICE_SPOOFING"]))
        campaigns.extend(self.inject_lateral_movement(sizes["LATERAL_MOVEMENT"]))
        campaigns.extend(self.inject_low_and_slow_exfil(sizes["LOW_AND_SLOW_EXFILTRATION"]))
        campaigns.extend(self.inject_insider_drift(sizes["INSIDER_DRIFT"]))
        return campaigns


# ---------------------------------------------------------------------------
# Temporal enrichment
# ---------------------------------------------------------------------------


def enrich_temporal_fields(events: list[RawEvent]) -> pd.DataFrame:
    """Sort globally, then compute per-entity temporal/geo deviation features."""
    events_sorted = sorted(events, key=lambda e: (e.timestamp, e.entity_id))
    states: dict[str, EntityState] = {}
    rows: list[dict[str, Any]] = []

    for ev in events_sorted:
        state = states.setdefault(ev.entity_id, EntityState())

        if state.last_timestamp is None:
            time_since = np.nan
            geo_dist = np.nan
        else:
            delta = ev.timestamp - state.last_timestamp
            time_since = round(delta.total_seconds() / 60.0, 2)
            if state.last_location is not None:
                geo_dist = round(
                    haversine_km(
                        state.last_location.latitude,
                        state.last_location.longitude,
                        ev.latitude,
                        ev.longitude,
                    ),
                    2,
                )
            else:
                geo_dist = np.nan

        loc_key = (ev.country, ev.city)
        is_new_location = loc_key not in state.known_locations
        is_new_device = ev.device_fingerprint not in state.known_fingerprints

        state.known_locations.add(loc_key)
        state.known_fingerprints.add(ev.device_fingerprint)
        state.last_timestamp = ev.timestamp
        state.last_location = GeoLocation(ev.country, ev.city, ev.latitude, ev.longitude)
        state.last_fingerprint = ev.device_fingerprint

        rows.append(
            {
                "event_id": ev.event_id,
                "entity_id": ev.entity_id,
                "entity_type": ev.entity_type,
                "timestamp": ev.timestamp.isoformat(sep=" "),
                "day_of_week": DAY_NAMES[ev.timestamp.weekday()],
                "login_hour": ev.timestamp.hour + ev.timestamp.minute / 60.0,
                "source_ip": ev.source_ip,
                "country": ev.country,
                "city": ev.city,
                "resource_accessed": ev.resource_accessed,
                "resource_category": ev.resource_category,
                "auth_method": ev.auth_method,
                "login_success": ev.login_success,
                "failed_attempts": ev.failed_attempts,
                "session_duration": round(ev.session_duration, 2),
                "command_count": ev.command_count,
                "device_fingerprint": ev.device_fingerprint,
                "os_version": ev.os_version,
                "protocol": ev.protocol,
                "is_new_device": is_new_device,
                "is_new_location": is_new_location,
                "geo_distance_from_previous": geo_dist,
                "time_since_last_login_minutes": time_since,
                "label": ev.label,
                "attack_type": ev.attack_type,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def generate_dataset() -> pd.DataFrame:
    """Build the full behavioral log dataset."""
    random.seed(MASTER_SEED)
    np.random.seed(MASTER_SEED)
    faker = Faker()
    Faker.seed(MASTER_SEED)

    target_anomaly = int(TARGET_EVENTS * ATTACK_FRACTION)
    target_normal = TARGET_EVENTS - target_anomaly

    profiles = ProfileFactory(faker, MASTER_SEED).create_population()
    normal_events = NormalEventGenerator(profiles, target_normal).generate()
    anomaly_events = AttackInjector(profiles, target_anomaly).inject_all()

    all_events = normal_events + anomaly_events

    # Trim if we overshot the global target (keep all anomaly rows)
    if len(all_events) > TARGET_EVENTS:
        excess = len(all_events) - TARGET_EVENTS
        normal_only = [e for e in all_events if e.label == "normal"]
        anomaly_only = [e for e in all_events if e.label == "anomaly"]
        random.shuffle(normal_only)
        all_events = normal_only[: len(normal_only) - excess] + anomaly_only

    return enrich_temporal_fields(all_events)


def print_summary(df: pd.DataFrame) -> None:
    """Print dataset quality summary to stdout."""
    print("\n=== Behavioral Logs Dataset Summary ===")
    print(f"Total rows:              {len(df):,}")
    print(f"Normal rows:             {(df['label'] == 'normal').sum():,}")
    print(f"Anomaly rows:            {(df['label'] == 'anomaly').sum():,}")
    print(f"Anomaly rate:            {(df['label'] == 'anomaly').mean():.2%}")
    print("\nRows per attack type:")
    attack_counts = df.loc[df["attack_type"] != "none", "attack_type"].value_counts()
    for attack in ATTACK_TYPES:
        print(f"  {attack:30s} {attack_counts.get(attack, 0):,}")
    print(f"\nUnique users:            {df.loc[df['entity_type'] == 'user', 'entity_id'].nunique()}")
    print(
        f"Unique service accounts: {df.loc[df['entity_type'] == 'service_account', 'entity_id'].nunique()}"
    )
    print(f"Unique devices:          {df.loc[df['entity_type'] == 'device', 'entity_id'].nunique()}")
    print(f"Date range:              {df['timestamp'].min()} -> {df['timestamp'].max()}")


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate_dataset()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved dataset to {OUTPUT_PATH}")
    print_summary(df)


if __name__ == "__main__":
    main()
