"""Filesystem locations for AEGIS.

Everything is resolved relative to the project root (the directory containing
``config/config.yaml``) so the package behaves identically whether it is
launched from the repo root, from ``streamlit run``, or from pytest.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

CONFIG_DIR: Path = PROJECT_ROOT / "config"
CONFIG_FILE: Path = CONFIG_DIR / "config.yaml"


def resolve(relative: str | Path) -> Path:
    """Resolve a config-declared relative path against the project root."""
    path = Path(relative)
    return path if path.is_absolute() else PROJECT_ROOT / path


def ensure_dirs(*paths: str | Path) -> list[Path]:
    """Create directories if missing and return the resolved paths."""
    created: list[Path] = []
    for raw in paths:
        path = resolve(raw)
        path.mkdir(parents=True, exist_ok=True)
        created.append(path)
    return created
