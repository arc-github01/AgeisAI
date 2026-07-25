"""Dashboard data provider — the single boundary between UI and data sources.

    UI pages  →  DashboardDataProvider  →  mock fixture (now)
                                       →  pipeline artifacts (later)

Switch ``config.yaml`` ``dashboard.data_source`` to ``pipeline`` once real
alert and event files exist. The overview page never reads mock data directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

import pandas as pd

from src.config import load_config

from . import mock_data
from .contracts import (
    ATTACK_DISPLAY_NAMES,
    DASHBOARD_ALERT_COLUMNS,
    DASHBOARD_ENTITY_COLUMNS,
    DASHBOARD_EVENT_COLUMNS,
    ENTITY_TYPE_LABELS,
    EVENT_HISTORY_COLUMNS,
    EVENT_HISTORY_LABELS,
    QUEUE_COLUMN_LABELS,
    QUEUE_DISPLAY_COLUMNS,
    SEVERITY_ORDER,
)
from .state import DashboardContext

DataSourceMode = Literal["auto", "mock", "pipeline"]


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if pd.notna(number) else None


@dataclass(frozen=True)
class AlertQueueFilters:
    """Filter criteria for the analyst alert queue."""

    severities: tuple[str, ...] = ()
    attack_types: tuple[str, ...] = ()
    entity_types: tuple[str, ...] = ()
    min_risk: float = 0.0
    entity_query: str = ""


@dataclass(frozen=True)
class AlertQueueSummary:
    matching_alerts: int
    critical_alerts: int
    peak_risk: float | None
    distinct_entities: int


@dataclass(frozen=True)
class EntityInvestigationSummary:
    entity_id: str
    entity_type: str
    role: str | None
    department: str | None
    home_city: str | None
    home_country: str | None
    events_observed: int
    first_seen: pd.Timestamp | None
    last_seen: pd.Timestamp | None
    profile_confidence: float
    profile_stage: str
    open_alerts: int
    peak_risk: float | None
    mean_risk: float | None


@dataclass(frozen=True)
class EntityBehaviorProfile:
    typical_hours: list[str]
    known_locations: list[str]
    known_devices: list[str]
    typical_resources: list[str]
    auth_methods: list[str]
    avg_session_seconds: float | None


@dataclass(frozen=True)
class PerformanceKPIs:
    pr_auc: float | None
    roc_auc: float | None
    recall_at_budget: float | None
    precision: float | None
    fpr: float | None
    prevalence: float | None
    operating_point: str | None


@dataclass(frozen=True)
class CampaignPerformanceSummary:
    n_campaigns: int
    n_detected: int
    campaign_recall: float | None
    median_latency_seconds: float | None
    median_events_before_detection: float | None


@dataclass(frozen=True)
class OverviewKPIs:
    events_processed: int
    entities_monitored: int
    active_alerts: int
    critical_alerts: int
    alert_rate: float

    @property
    def alert_rate_pct(self) -> str:
        return f"{self.alert_rate:.2%}"


@dataclass(frozen=True)
class ProviderSnapshot:
    """Resolved datasets plus metadata about where they came from."""

    events: pd.DataFrame
    alerts: pd.DataFrame
    entities: pd.DataFrame
    source: str
    is_mock: bool


def _empty_frame(columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def _normalize_alerts(frame: pd.DataFrame) -> pd.DataFrame:
    """Map pipeline column names to the dashboard contract when needed."""
    if frame.empty:
        return _empty_frame(DASHBOARD_ALERT_COLUMNS)
    out = frame.copy()
    if "top_reason" in out.columns and "short_reason" not in out.columns:
        out["short_reason"] = out["top_reason"]
    missing = [c for c in DASHBOARD_ALERT_COLUMNS if c not in out.columns]
    for col in missing:
        out[col] = None
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    return out[list(DASHBOARD_ALERT_COLUMNS)]


def _normalize_events(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return _empty_frame(DASHBOARD_EVENT_COLUMNS)
    out = frame.copy()
    if "session_duration_s" in out.columns and "session_duration" not in out.columns:
        out["session_duration"] = out["session_duration_s"]
    missing = [c for c in DASHBOARD_EVENT_COLUMNS if c not in out.columns]
    for col in missing:
        out[col] = None
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    return out[list(DASHBOARD_EVENT_COLUMNS)]


def _entities_from_events(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return _empty_frame(DASHBOARD_ENTITY_COLUMNS)
    grouped = (
        events.groupby(["entity_id", "entity_type"], as_index=False)
        .agg(timestamp=("timestamp", "min"))
        .drop(columns=["timestamp"])
    )
    for col in DASHBOARD_ENTITY_COLUMNS:
        if col not in grouped.columns:
            grouped[col] = None
    return grouped[list(DASHBOARD_ENTITY_COLUMNS)]


class DashboardDataProvider:
    """Read-only access to dashboard datasets and derived overview aggregates."""

    def __init__(
        self,
        ctx: DashboardContext | None = None,
        *,
        mode: DataSourceMode | None = None,
    ) -> None:
        self._ctx = ctx or DashboardContext.build()
        cfg_mode = load_config().get("dashboard.data_source", "auto")
        self._mode: DataSourceMode = mode or cfg_mode  # type: ignore[assignment]
        self._snapshot = self._resolve()

    @classmethod
    def from_context(cls, ctx: DashboardContext) -> "DashboardDataProvider":
        return cls(ctx)

    @property
    def source_label(self) -> str:
        return self._snapshot.source

    @property
    def is_mock(self) -> bool:
        return self._snapshot.is_mock

    @property
    def has_data(self) -> bool:
        return not self._snapshot.alerts.empty or not self._snapshot.events.empty

    # -- Raw tables ---------------------------------------------------------
    def get_events(self) -> pd.DataFrame:
        return self._snapshot.events

    def get_alerts(self) -> pd.DataFrame:
        return self._snapshot.alerts

    def get_entities(self) -> pd.DataFrame:
        return self._snapshot.entities

    # -- Overview aggregates ------------------------------------------------
    def get_overview_kpis(self) -> OverviewKPIs:
        alerts = self.get_alerts()
        entities = self.get_entities()
        events = self.get_events()

        if self.is_mock:
            summary = mock_data.fixture_summary()
            return OverviewKPIs(
                events_processed=int(summary["events_processed"]),
                entities_monitored=int(summary["entities_monitored"]),
                active_alerts=int(summary["active_alerts"]),
                critical_alerts=int(summary["critical_alerts"]),
                alert_rate=float(summary["alert_rate"]),
            )

        events_processed = len(events) if not events.empty else 0
        active = len(alerts)
        critical = int((alerts["severity"] == "CRITICAL").sum()) if not alerts.empty else 0
        rate = (active / events_processed) if events_processed else 0.0
        return OverviewKPIs(
            events_processed=events_processed,
            entities_monitored=len(entities) if not entities.empty else 0,
            active_alerts=active,
            critical_alerts=critical,
            alert_rate=rate,
        )

    def get_alert_timeline(self, *, freq: str = "6h") -> pd.DataFrame:
        """Alert counts over time, broken out by severity."""
        alerts = self.get_alerts()
        if alerts.empty:
            return pd.DataFrame(columns=["timestamp", "severity", "count"])
        bucketed = alerts.copy()
        bucketed["timestamp"] = bucketed["timestamp"].dt.floor(freq)
        grouped = (
            bucketed.groupby(["timestamp", "severity"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
        )
        return grouped.sort_values("timestamp")

    def get_threat_distribution(self) -> dict[str, int]:
        alerts = self.get_alerts()
        if alerts.empty:
            return {}
        counts = alerts["attack_type"].value_counts().to_dict()
        return {ATTACK_DISPLAY_NAMES.get(k, k): int(v) for k, v in counts.items()}

    def get_severity_distribution(self) -> dict[str, int]:
        alerts = self.get_alerts()
        if alerts.empty:
            return {}
        counts = alerts["severity"].value_counts().to_dict()
        return {k: int(counts.get(k, 0)) for k in SEVERITY_ORDER if counts.get(k, 0)}

    def get_entity_type_distribution(self) -> dict[str, int]:
        entities = self.get_entities()
        if entities.empty:
            alerts = self.get_alerts()
            if alerts.empty:
                return {}
            entities = alerts[["entity_id", "entity_type"]].drop_duplicates()
        counts = entities["entity_type"].value_counts().to_dict()
        return {ENTITY_TYPE_LABELS.get(k, k): int(v) for k, v in counts.items()}

    def get_top_risk_entities(self, n: int = 10) -> pd.DataFrame:
        alerts = self.get_alerts()
        if alerts.empty:
            return pd.DataFrame(
                columns=["entity_id", "entity_type", "risk_score", "primary_signal"]
            )
        ranked = (
            alerts.sort_values("risk_score", ascending=False)
            .groupby(["entity_id", "entity_type"], as_index=False)
            .first()
        )
        ranked["primary_signal"] = ranked["attack_type"].map(
            lambda x: ATTACK_DISPLAY_NAMES.get(x, str(x))
        )
        return (
            ranked.rename(columns={"entity_type": "type"})
            .loc[:, ["entity_id", "type", "risk_score", "primary_signal"]]
            .sort_values("risk_score", ascending=False)
            .head(n)
            .reset_index(drop=True)
        )

    def get_recent_critical_alerts(self, n: int = 8) -> pd.DataFrame:
        alerts = self.get_alerts()
        if alerts.empty:
            return pd.DataFrame(columns=list(DASHBOARD_ALERT_COLUMNS))
        critical = alerts[alerts["severity"] == "CRITICAL"].copy()
        if critical.empty:
            critical = alerts.nlargest(n, "risk_score")
        return critical.sort_values(["timestamp", "risk_score"], ascending=[False, False]).head(n)

    # -- Alert queue --------------------------------------------------------
    def filter_alerts(self, filters: AlertQueueFilters | None = None) -> pd.DataFrame:
        """Return alerts matching the filter criteria, ranked for triage."""
        filters = filters or AlertQueueFilters()
        alerts = self.get_alerts()
        if alerts.empty:
            return alerts

        view = alerts.copy()
        if filters.severities:
            view = view[view["severity"].isin(filters.severities)]
        if filters.attack_types:
            view = view[view["attack_type"].isin(filters.attack_types)]
        if filters.entity_types:
            view = view[view["entity_type"].isin(filters.entity_types)]
        if filters.min_risk > 0:
            view = view[view["risk_score"] >= filters.min_risk]
        if filters.entity_query.strip():
            query = filters.entity_query.strip()
            view = view[view["entity_id"].astype(str).str.contains(query, case=False)]

        return view.sort_values(
            ["risk_score", "timestamp"], ascending=[False, False]
        ).reset_index(drop=True)

    def get_alert_queue_summary(self, filters: AlertQueueFilters | None = None) -> AlertQueueSummary:
        view = self.filter_alerts(filters)
        if view.empty:
            return AlertQueueSummary(0, 0, None, 0)
        return AlertQueueSummary(
            matching_alerts=len(view),
            critical_alerts=int((view["severity"] == "CRITICAL").sum()),
            peak_risk=float(view["risk_score"].max()),
            distinct_entities=int(view["entity_id"].nunique()),
        )

    def get_alert_queue_table(
        self, filters: AlertQueueFilters | None = None, *, limit: int = 50
    ) -> pd.DataFrame:
        """Display-ready queue table with friendly labels."""
        view = self.filter_alerts(filters).head(limit)
        if view.empty:
            return pd.DataFrame(columns=[QUEUE_COLUMN_LABELS[c] for c in QUEUE_DISPLAY_COLUMNS])

        display = view.loc[:, list(QUEUE_DISPLAY_COLUMNS)].copy()
        display["attack_type"] = display["attack_type"].map(
            lambda x: ATTACK_DISPLAY_NAMES.get(x, str(x))
        )
        display["entity_type"] = display["entity_type"].map(
            lambda x: ENTITY_TYPE_LABELS.get(x, str(x).replace("_", " ").title())
        )
        display["timestamp"] = display["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
        return display.rename(columns=QUEUE_COLUMN_LABELS)

    def get_alert_by_id(self, alert_id: str) -> pd.Series | None:
        alerts = self.get_alerts()
        if alerts.empty or not alert_id:
            return None
        match = alerts[alerts["alert_id"] == alert_id]
        if match.empty:
            return None
        return match.iloc[0]

    @staticmethod
    def parse_reasons(reasons: Any) -> list[str]:
        """Split a reasons field into individual contributing statements."""
        if reasons is None or (isinstance(reasons, float) and pd.isna(reasons)):
            return []
        if isinstance(reasons, list):
            return [str(item).strip() for item in reasons if str(item).strip()]
        text = str(reasons).strip()
        if not text:
            return []
        parts = [part.strip() for part in text.replace("\n", "+").split("+")]
        return [part for part in parts if part]

    def get_score_contributions(self, alert: Mapping[str, Any]) -> pd.DataFrame:
        """Map persisted alert scores to a contribution chart (no inference)."""
        labels = {
            "anomaly_score": "Behavioral anomaly score",
            "sequence_score": "Sequence anomaly score",
            "attack_confidence": "Attack classification confidence",
        }
        rows = []
        for key, label in labels.items():
            value = alert.get(key)
            if value is None or (isinstance(value, float) and pd.isna(value)):
                continue
            rows.append({"factor": label, "contribution": float(value)})
        if not rows:
            return pd.DataFrame(columns=["factor", "contribution"])
        frame = pd.DataFrame(rows)
        total = frame["contribution"].sum()
        if total > 0:
            frame["contribution"] = frame["contribution"] / total
        return frame.sort_values("contribution")

    # -- Entity investigation -----------------------------------------------
    def list_entity_ids(
        self,
        *,
        entity_type: str | None = None,
        query: str = "",
    ) -> list[str]:
        entities = self.get_entities()
        if entities.empty:
            return []
        view = entities
        if entity_type:
            view = view[view["entity_type"] == entity_type]
        if query.strip():
            view = view[view["entity_id"].astype(str).str.contains(query.strip(), case=False)]
        return sorted(view["entity_id"].astype(str).tolist())

    def get_entity_metadata(self, entity_id: str) -> pd.Series | None:
        entities = self.get_entities()
        if entities.empty:
            return None
        match = entities[entities["entity_id"].astype(str) == str(entity_id)]
        if match.empty:
            return None
        return match.iloc[0]

    def get_entity_events(self, entity_id: str) -> pd.DataFrame:
        events = self.get_events()
        if events.empty:
            return events
        return events[events["entity_id"].astype(str) == str(entity_id)].sort_values("timestamp")

    def get_entity_alerts(self, entity_id: str) -> pd.DataFrame:
        alerts = self.get_alerts()
        if alerts.empty:
            return alerts
        return alerts[alerts["entity_id"].astype(str) == str(entity_id)].sort_values(
            ["risk_score", "timestamp"], ascending=[False, False]
        )

    def _profile_confidence(self, event_count: int) -> tuple[float, str]:
        cfg = load_config()
        maturity = int(cfg["profiling.maturity_events"])
        minimum = int(cfg["profiling.min_events_for_personal"])
        if event_count < minimum:
            return min(0.3, event_count / max(minimum, 1) * 0.3), "cold-start"
        if event_count < maturity:
            span = max(maturity - minimum, 1)
            blend = 0.3 + 0.7 * ((event_count - minimum) / span)
            return min(1.0, blend), "blending"
        return 1.0, "mature"

    def get_entity_summary(self, entity_id: str) -> EntityInvestigationSummary | None:
        meta = self.get_entity_metadata(entity_id)
        if meta is None:
            return None
        events = self.get_entity_events(entity_id)
        alerts = self.get_entity_alerts(entity_id)
        confidence, stage = self._profile_confidence(len(events))
        return EntityInvestigationSummary(
            entity_id=str(entity_id),
            entity_type=str(meta.get("entity_type", "")),
            role=meta.get("role"),
            department=meta.get("department"),
            home_city=meta.get("home_city"),
            home_country=meta.get("home_country"),
            events_observed=len(events),
            first_seen=events["timestamp"].min() if not events.empty else None,
            last_seen=events["timestamp"].max() if not events.empty else None,
            profile_confidence=confidence,
            profile_stage=stage,
            open_alerts=len(alerts),
            peak_risk=float(alerts["risk_score"].max()) if not alerts.empty else None,
            mean_risk=float(alerts["risk_score"].mean()) if not alerts.empty else None,
        )

    def get_entity_profile(self, entity_id: str) -> EntityBehaviorProfile:
        events = self.get_entity_events(entity_id)
        meta = self.get_entity_metadata(entity_id)

        if events.empty:
            locations = []
            if meta is not None and meta.get("home_city"):
                locations.append(f"{meta.get('home_city')}, {meta.get('home_country', '')}".strip(", "))
            return EntityBehaviorProfile(
                typical_hours=[],
                known_locations=locations,
                known_devices=[],
                typical_resources=[],
                auth_methods=[],
                avg_session_seconds=None,
            )

        hours = events["timestamp"].dt.hour.value_counts().head(4).index.tolist()
        hour_labels = [f"{h:02d}:00" for h in sorted(hours)]
        locations = sorted(
            events.apply(lambda r: f"{r['city']}, {r['country']}", axis=1).dropna().unique().astype(str)
        )[:8]
        devices = sorted(events["device_id"].dropna().astype(str).unique())[:8]
        resources = events["resource_accessed"].value_counts().head(8).index.astype(str).tolist()
        auth = sorted(events["auth_method"].dropna().astype(str).unique())[:6]
        avg_session = float(events["session_duration"].mean()) if "session_duration" in events else None

        return EntityBehaviorProfile(
            typical_hours=hour_labels,
            known_locations=locations,
            known_devices=devices,
            typical_resources=resources,
            auth_methods=auth,
            avg_session_seconds=avg_session,
        )

    def get_entity_event_history(
        self, entity_id: str, *, limit: int = 200
    ) -> pd.DataFrame:
        events = self.get_entity_events(entity_id).tail(limit).sort_values(
            "timestamp", ascending=False
        )
        if events.empty:
            return pd.DataFrame(columns=[EVENT_HISTORY_LABELS[c] for c in EVENT_HISTORY_COLUMNS])
        display = events.loc[:, list(EVENT_HISTORY_COLUMNS)].copy()
        display["timestamp"] = display["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
        display["auth_success"] = display["auth_success"].map({True: "yes", False: "no"})
        return display.rename(columns=EVENT_HISTORY_LABELS)

    def get_entity_alert_history(
        self, entity_id: str, *, limit: int = 8
    ) -> pd.DataFrame:
        alerts = self.get_entity_alerts(entity_id).head(limit)
        if alerts.empty:
            return pd.DataFrame(
                columns=["Timestamp", "Severity", "Attack Type", "Risk", "Reason"]
            )
        display = alerts.copy()
        display["attack_type"] = display["attack_type"].map(
            lambda x: ATTACK_DISPLAY_NAMES.get(x, str(x))
        )
        display["timestamp"] = display["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
        return (
            display.rename(
                columns={
                    "timestamp": "Timestamp",
                    "severity": "Severity",
                    "attack_type": "Attack Type",
                    "risk_score": "Risk",
                    "short_reason": "Reason",
                }
            )
            .loc[:, ["Timestamp", "Severity", "Attack Type", "Risk", "Reason"]]
        )

    def get_entity_risk_timeline(self, entity_id: str) -> pd.DataFrame | None:
        alerts = self.get_entity_alerts(entity_id)
        if alerts.empty:
            return None
        return alerts[["timestamp", "risk_score"]]

    # -- Model performance (Phase 12 artifact only — never mocked) ------------
    def has_evaluation_metrics(self) -> bool:
        document = self.get_metrics_document()
        return bool(document and isinstance(document.get("metrics"), dict))

    def get_metrics_document(self) -> dict[str, Any] | None:
        if not self._ctx.has("metrics"):
            return None
        from src.evaluation.report import load_metrics

        document = load_metrics("latest")
        return document if isinstance(document, dict) else None

    def get_performance_manifest(self) -> dict[str, Any] | None:
        document = self.get_metrics_document()
        if not document:
            return None
        manifest = document.get("manifest")
        return manifest if isinstance(manifest, dict) else None

    def _metrics_payload(self) -> dict[str, Any]:
        document = self.get_metrics_document()
        if not document:
            return {}
        payload = document.get("metrics")
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _records_frame(payload: dict[str, Any], key: str) -> pd.DataFrame | None:
        records = payload.get(key)
        if not records:
            return None
        frame = pd.DataFrame(records)
        return frame if not frame.empty else None

    @staticmethod
    def _confusion_matrix_frame(raw: Any) -> pd.DataFrame | None:
        if raw is None:
            return None
        if isinstance(raw, pd.DataFrame):
            return raw if not raw.empty else None
        if isinstance(raw, dict):
            if {"index", "columns", "values"} <= set(raw):
                return pd.DataFrame(raw["values"], index=raw["index"], columns=raw["columns"])
            frame = pd.DataFrame(raw)
            return frame if not frame.empty else None
        if isinstance(raw, list):
            frame = pd.DataFrame(raw)
            return frame if not frame.empty else None
        return None

    def get_performance_kpis(self) -> PerformanceKPIs:
        detection = self._metrics_payload().get("detection", {})
        if not isinstance(detection, dict):
            detection = {}
        return PerformanceKPIs(
            pr_auc=_optional_float(detection.get("pr_auc")),
            roc_auc=_optional_float(detection.get("roc_auc")),
            recall_at_budget=_optional_float(detection.get("recall")),
            precision=_optional_float(detection.get("precision")),
            fpr=_optional_float(detection.get("fpr")),
            prevalence=_optional_float(detection.get("prevalence")),
            operating_point=detection.get("operating_point"),
        )

    def get_pr_curve(self) -> pd.DataFrame | None:
        return self._records_frame(self._metrics_payload(), "pr_curve")

    def get_roc_curve(self) -> pd.DataFrame | None:
        return self._records_frame(self._metrics_payload(), "roc_curve")

    def get_budget_sweep(self) -> pd.DataFrame | None:
        return self._records_frame(self._metrics_payload(), "budget_sweep")

    def get_confusion_matrix(self) -> pd.DataFrame | None:
        return self._confusion_matrix_frame(self._metrics_payload().get("confusion_matrix"))

    def get_per_class_metrics(self) -> pd.DataFrame | None:
        frame = self._records_frame(self._metrics_payload(), "per_class")
        if frame is None or frame.empty or "class" not in frame.columns:
            return frame
        display = frame.copy()
        display["class"] = display["class"].map(
            lambda x: ATTACK_DISPLAY_NAMES.get(str(x), str(x))
        )
        return display

    def get_campaign_performance(self) -> CampaignPerformanceSummary | None:
        raw = self._metrics_payload().get("campaign_detection")
        if not isinstance(raw, dict) or not raw:
            return None
        return CampaignPerformanceSummary(
            n_campaigns=int(raw.get("n_campaigns", 0)),
            n_detected=int(raw.get("n_detected", 0)),
            campaign_recall=_optional_float(raw.get("campaign_recall")),
            median_latency_seconds=_optional_float(raw.get("median_latency_seconds")),
            median_events_before_detection=_optional_float(
                raw.get("median_events_before_detection")
            ),
        )

    # -- Internal -----------------------------------------------------------
    def _resolve(self) -> ProviderSnapshot:
        mode = self._mode
        pipeline_ready = self._ctx.has("events") and self._ctx.has("alerts")

        use_pipeline = mode == "pipeline" or (mode == "auto" and pipeline_ready)

        if use_pipeline:
            if not pipeline_ready:
                return ProviderSnapshot(
                    events=_empty_frame(DASHBOARD_EVENT_COLUMNS),
                    alerts=_empty_frame(DASHBOARD_ALERT_COLUMNS),
                    entities=_empty_frame(DASHBOARD_ENTITY_COLUMNS),
                    source="pipeline artifacts (not yet available)",
                    is_mock=False,
                )
            events_raw = self._ctx.events()
            alerts_raw = self._ctx.alerts()
            entities_raw = self._ctx.entities()
            events = _normalize_events(
                events_raw if events_raw is not None else _empty_frame(DASHBOARD_EVENT_COLUMNS)
            )
            alerts = _normalize_alerts(
                alerts_raw if alerts_raw is not None else _empty_frame(DASHBOARD_ALERT_COLUMNS)
            )
            if isinstance(entities_raw, list) and entities_raw:
                entities = pd.DataFrame(entities_raw)
            elif isinstance(entities_raw, dict) and entities_raw:
                entities = pd.DataFrame(list(entities_raw.values()))
            else:
                entities = _entities_from_events(events)
            for col in DASHBOARD_ENTITY_COLUMNS:
                if col not in entities.columns:
                    entities[col] = None
            entities = (
                entities[list(DASHBOARD_ENTITY_COLUMNS)]
                if not entities.empty
                else _empty_frame(DASHBOARD_ENTITY_COLUMNS)
            )
            return ProviderSnapshot(
                events=events,
                alerts=alerts,
                entities=entities,
                source="pipeline artifacts",
                is_mock=False,
            )

        return ProviderSnapshot(
            events=mock_data.generate_events_sample(),
            alerts=mock_data.generate_alerts(),
            entities=mock_data.generate_entities(),
            source="development fixture",
            is_mock=True,
        )


def get_provider(mode: DataSourceMode | None = None) -> DashboardDataProvider:
    """Convenience constructor for tests and scripts."""
    return DashboardDataProvider(mode=mode)
