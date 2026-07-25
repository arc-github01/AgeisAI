"""Metrics persistence and reproducibility manifests.

Any number that appears in the report or the dashboard must be traceable to a
run: which seed, which config, which package versions, which git commit. This
module is the only sanctioned way to write metrics to disk, so a metric without
a manifest is a metric that never happened.
"""

from __future__ import annotations

import json
import math
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..artifacts import artifact_path, metrics_dir
from ..config import load_config
from ..paths import PROJECT_ROOT

_TRACKED_PACKAGES = ("numpy", "pandas", "scikit-learn", "scipy", "streamlit", "plotly")


class AegisJSONEncoder(json.JSONEncoder):
    """Fallback for object types that survive :func:`sanitize_for_json`."""

    def default(self, o: Any) -> Any:
        if isinstance(o, (pd.Timestamp, datetime)):
            return o.isoformat()
        if isinstance(o, np.ndarray):
            return sanitize_for_json(o.tolist())
        return str(o)


def sanitize_for_json(value: Any) -> Any:
    """Convert a payload into strictly valid JSON types.

    Non-finite floats become ``null``. This must happen *before* ``json.dump``
    rather than inside an encoder hook: ``JSONEncoder.default`` is never called
    for native ``float``, so ``json`` would happily emit the bare tokens ``NaN``
    and ``Infinity``, which are not valid JSON and break every non-Python
    consumer of our metrics files. ``nan`` is a legitimate metric value here
    (undefined precision when a detector raises no alerts), so it has to be
    represented rather than rejected.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [sanitize_for_json(item) for item in value.tolist()]
    if value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, pd.DataFrame):
        return [sanitize_for_json(row) for row in value.to_dict("records")]
    if isinstance(value, pd.Series):
        return [sanitize_for_json(item) for item in value.to_list()]
    if isinstance(value, dict):
        return {str(key): sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [sanitize_for_json(item) for item in value]
    return value


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return None
    output = result.stdout.strip()
    return output or None


def _package_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    versions: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:  # pragma: no cover
            versions[name] = "not-installed"
    return versions


@dataclass(frozen=True)
class RunManifest:
    """Everything needed to reproduce a set of reported numbers."""

    run_id: str
    created_at: str
    aegis_version: str
    python_version: str
    platform: str
    master_seed: int
    git_commit: str | None
    packages: dict[str, str]
    config_snapshot: dict[str, Any] = field(repr=False)
    notes: str | None = None

    @classmethod
    def capture(cls, notes: str | None = None) -> "RunManifest":
        from .. import __version__

        cfg = load_config()
        now = datetime.now(timezone.utc)
        return cls(
            run_id=now.strftime("%Y%m%dT%H%M%SZ"),
            created_at=now.isoformat(),
            aegis_version=__version__,
            python_version=platform.python_version(),
            platform=platform.platform(),
            master_seed=int(cfg["seed.master"]),
            git_commit=_git_commit(),
            packages=_package_versions(),
            config_snapshot=cfg.as_dict(),
            notes=notes,
        )

    def save(self) -> Path:
        path = artifact_path("manifest", ensure_parent=True)
        write_json(asdict(self), path)
        return path


def write_json(payload: Any, path: str | Path) -> Path:
    """Write strictly valid JSON (``allow_nan=False`` makes violations loud)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(
            sanitize_for_json(payload),
            handle,
            indent=2,
            cls=AegisJSONEncoder,
            allow_nan=False,
            sort_keys=False,
        )
    return target


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_metrics(
    payload: dict[str, Any],
    *,
    name: str = "latest",
    notes: str | None = None,
    write_manifest: bool = True,
) -> Path:
    """Persist a metrics payload, stamped with its reproducibility manifest.

    Writes ``artifacts/metrics/<name>.json`` and, unless ``name`` already is
    ``latest``, also refreshes ``artifacts/metrics/latest.json`` so the
    dashboard always has a single well-known entry point.
    """
    manifest = RunManifest.capture(notes=notes)
    document = {
        "manifest": asdict(manifest),
        "metrics": payload,
    }
    path = write_json(document, metrics_dir() / f"{name}.json")
    if name != "latest":
        write_json(document, metrics_dir() / "latest.json")
    if write_manifest:
        manifest.save()
    return path


def load_metrics(name: str = "latest") -> dict[str, Any] | None:
    """Load a saved metrics document, or ``None`` if the run has not happened."""
    path = metrics_dir() / f"{name}.json"
    if not path.exists():
        return None
    return read_json(path)


__all__ = [
    "AegisJSONEncoder",
    "RunManifest",
    "load_metrics",
    "read_json",
    "save_metrics",
    "write_json",
]
