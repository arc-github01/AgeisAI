"""Test isolation.

Tests must never write into the real workspace. Without this, a test that saves
a metrics document leaves ``artifacts/metrics/latest.json`` behind and the
dashboard cheerfully renders unit-test numbers as if they were a real run -
exactly the class of "fake results" this project must not ship.

Every output directory is therefore redirected to a temporary root for the whole
session. Config is loaded once and cached, so patching the cached instance is
enough for library code, Streamlit pages and AppTest runs alike.
"""

from __future__ import annotations

import pytest

from src.config import load_config

_REDIRECTED = ("data_raw", "data_processed", "data_generated", "models", "artifacts")


@pytest.fixture(scope="session", autouse=True)
def isolated_workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp("aegis_workspace")
    cfg = load_config()
    original = {key: cfg._data["paths"][key] for key in _REDIRECTED}
    for key in _REDIRECTED:
        target = root / key
        target.mkdir(parents=True, exist_ok=True)
        cfg._data["paths"][key] = str(target)
    yield root
    cfg._data["paths"].update(original)
