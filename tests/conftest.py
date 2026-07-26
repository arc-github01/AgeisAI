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
from src.evaluation.report import metrics_dir

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


def clear_metrics_artifacts() -> None:
    """Remove evaluation JSON artifacts from the isolated workspace."""
    folder = metrics_dir()
    if folder.exists():
        for path in folder.glob("*.json"):
            path.unlink(missing_ok=True)


def clear_pipeline_artifacts() -> None:
    """Remove on-disk pipeline artifacts created during tests."""
    from src.artifacts import REGISTRY

    for item in REGISTRY.values():
        path = item.path
        if path.exists():
            path.unlink(missing_ok=True)


@pytest.fixture(autouse=True)
def _clean_metrics_between_tests(request: pytest.FixtureRequest):
    """Drop metrics artifacts unless the test explicitly retains them."""
    retain = "retain_metrics_artifact" in request.keywords
    # Streaming module tests hold models in memory but still need on-disk
    # thresholds during the module-scoped bootstrap; skip clearing for them.
    if request.node.get_closest_marker("no_pipeline_cleanup"):
        if not retain:
            clear_metrics_artifacts()
        yield
        if not retain:
            clear_metrics_artifacts()
        return
    clear_pipeline_artifacts()
    if not retain:
        clear_metrics_artifacts()
    yield
    clear_pipeline_artifacts()
    if not retain:
        clear_metrics_artifacts()


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "retain_metrics_artifact: keep evaluation metrics JSON for this test",
    )
    config.addinivalue_line(
        "markers",
        "no_pipeline_cleanup: preserve module-scoped pipeline fixtures",
    )
