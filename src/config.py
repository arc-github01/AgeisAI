"""Configuration loading for AEGIS.

A single YAML file (``config/config.yaml``) is the source of truth for every
tunable parameter. Modules never hardcode behavioural constants; they read them
from the :class:`Config` object returned by :func:`load_config`.

Dotted lookup is supported so call sites read like the YAML they mirror::

    cfg = load_config()
    cfg["risk.weights.ml_anomaly"]
    cfg.get("models.anomaly_detector.n_estimators", 100)
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml

from .paths import CONFIG_FILE, ensure_dirs, resolve

_MISSING = object()


class ConfigError(KeyError):
    """Raised when a required configuration key is absent."""


class Config(Mapping[str, Any]):
    """Immutable read-only view over the parsed configuration tree."""

    def __init__(self, data: Mapping[str, Any], source: Path | None = None) -> None:
        self._data: dict[str, Any] = copy.deepcopy(dict(data))
        self.source = source

    # -- Mapping protocol ---------------------------------------------------
    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Config(source={self.source}, keys={sorted(self._data)})"

    def __getitem__(self, key: str) -> Any:
        value = self.get(key, _MISSING)
        if value is _MISSING:
            raise ConfigError(f"missing configuration key: {key!r}")
        return value

    # -- Dotted access ------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in key.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return copy.deepcopy(node) if isinstance(node, (dict, list)) else node

    def section(self, key: str) -> "Config":
        """Return a sub-tree as its own :class:`Config`."""
        value = self[key]
        if not isinstance(value, Mapping):
            raise ConfigError(f"configuration key {key!r} is not a section")
        return Config(value, source=self.source)

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    # -- Convenience --------------------------------------------------------
    def path(self, key: str) -> Path:
        """Resolve a ``paths.*`` entry to an absolute path (created if needed)."""
        return ensure_dirs(self[f"paths.{key}"])[0]

    def generator_profile(self) -> dict[str, Any]:
        """Return the active synthetic-environment sizing profile."""
        name = self["generator.profile"]
        profiles = self["generator.profiles"]
        if name not in profiles:
            raise ConfigError(
                f"generator.profile={name!r} not found in generator.profiles "
                f"(available: {sorted(profiles)})"
            )
        return {"name": name, **profiles[name]}


_cache: dict[Path, Config] = {}


def load_config(path: str | Path | None = None, *, reload: bool = False) -> Config:
    """Load and cache the configuration file."""
    config_path = resolve(path) if path is not None else CONFIG_FILE
    if not config_path.exists():
        raise FileNotFoundError(f"configuration file not found: {config_path}")
    if reload or config_path not in _cache:
        with config_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise ConfigError(f"{config_path} must contain a YAML mapping at the top level")
        _cache[config_path] = Config(data, source=config_path)
    return _cache[config_path]
