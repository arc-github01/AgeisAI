"""Data access layer for the dashboard.

Pages are pure functions of a :class:`DashboardContext`. They never touch the
filesystem directly and never assume data exists - which is what lets the shell
run end-to-end before the generator and models are built, and lets the live
simulator later swap in freshly scored events without a page rewrite.

Loading is cached on ``(path, mtime)`` so regenerating a dataset invalidates the
cache automatically.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.artifacts import Artifact, ArtifactStatus, artifact, pipeline_status
from src.config import Config, load_config


def _signature(path: Path) -> tuple[str, float, int] | None:
    if not path.exists():
        return None
    stat = path.stat()
    return (str(path), stat.st_mtime, stat.st_size)


@st.cache_data(show_spinner=False)
def _read_parquet(signature: tuple[str, float, int]) -> pd.DataFrame:
    return pd.read_parquet(signature[0])


@st.cache_data(show_spinner=False)
def _read_json(signature: tuple[str, float, int]) -> Any:
    with open(signature[0], "r", encoding="utf-8") as handle:
        return json.load(handle)


@dataclass
class DashboardContext:
    """Everything a page needs, with honest "not yet" answers."""

    cfg: Config
    statuses: dict[str, ArtifactStatus]

    @classmethod
    def build(cls) -> "DashboardContext":
        return cls(
            cfg=load_config(),
            statuses={status.key: status for status in pipeline_status()},
        )

    # -- availability -------------------------------------------------------
    def has(self, *keys: str) -> bool:
        return all(self.statuses[key].exists for key in keys)

    def missing(self, *keys: str) -> list[Artifact]:
        return [self.statuses[key].artifact for key in keys if not self.statuses[key].exists]

    def ordered_statuses(self) -> list[ArtifactStatus]:
        return sorted(self.statuses.values(), key=lambda s: (s.artifact.phase, s.key))

    def readiness(self) -> tuple[int, int]:
        ready = sum(1 for s in self.statuses.values() if s.exists)
        return ready, len(self.statuses)

    # -- loaders ------------------------------------------------------------
    def _frame(self, key: str) -> pd.DataFrame | None:
        signature = _signature(artifact(key).path)
        return None if signature is None else _read_parquet(signature)

    def _json(self, key: str) -> Any | None:
        signature = _signature(artifact(key).path)
        return None if signature is None else _read_json(signature)

    def events(self) -> pd.DataFrame | None:
        return self._frame("events")

    def alerts(self) -> pd.DataFrame | None:
        return self._frame("alerts")

    def features(self) -> pd.DataFrame | None:
        return self._frame("features")

    def risk_scores(self) -> pd.DataFrame | None:
        return self._frame("risk_scores")

    def classifications(self) -> pd.DataFrame | None:
        return self._frame("classifications")

    def streaming_scores(self) -> pd.DataFrame | None:
        return self._frame("streaming_scores")

    def entities(self) -> Any | None:
        return self._json("entities")

    def risk_evaluation(self) -> dict | None:
        return self._json("risk_evaluation")

    def drift_evaluation(self) -> dict | None:
        return self._json("drift_evaluation")

    def streaming_metrics(self) -> dict | None:
        return self._json("streaming_metrics")

    def metrics(self) -> dict | None:
        document = self._json("metrics")
        return document.get("metrics") if isinstance(document, dict) else None

    def manifest(self) -> dict | None:
        document = self._json("metrics")
        return document.get("manifest") if isinstance(document, dict) else None

    def entity_ids(self) -> list[str]:
        """Selectable entities, from the entity roster or the event log."""
        roster = self.entities()
        if isinstance(roster, dict) and isinstance(roster.get("entities"), list):
            return sorted(
                str(item.get("entity_id"))
                for item in roster["entities"]
                if item.get("entity_id") is not None
            )
        if isinstance(roster, list) and roster:
            return sorted(str(item.get("entity_id")) for item in roster)
        if isinstance(roster, dict) and roster:
            # Legacy flat map of entity_id -> profile.
            if all(isinstance(v, dict) for v in roster.values()):
                return sorted(str(key) for key in roster)
        events = self.events()
        if events is not None and "entity_id" in events.columns:
            return sorted(events["entity_id"].astype(str).unique().tolist())
        return []


def get_context() -> DashboardContext:
    """Fresh context per rerun: artifact existence must never be cached."""
    return DashboardContext.build()
