"""Headless tests for the SOC console.

`streamlit.testing.v1.AppTest` executes the real app script, so these are not
import smoke tests: every page is actually rendered and any exception in layout,
theming, chart construction or the empty-state logic fails the suite.

The console must be robust in its *hardest* state - no data at all - because
that is the state it is in for most of the build.
"""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from dashboard import PAGES
from dashboard.state import DashboardContext
from src.paths import PROJECT_ROOT
from src.schema import ATTACK_CLASSES

APP = str(PROJECT_ROOT / "app.py")


@pytest.fixture(scope="module")
def app() -> AppTest:
    instance = AppTest.from_file(APP, default_timeout=60)
    instance.run()
    return instance


def test_app_starts_with_no_data_present(app: AppTest):
    assert not app.exception
    assert app.sidebar.radio[0].options == [page.label for page in PAGES]


@pytest.mark.parametrize("label", [page.label for page in PAGES])
def test_every_page_renders_without_data(label: str):
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value(label).run()
    assert not at.exception, f"page {label!r} raised: {at.exception}"
    assert at.markdown, f"page {label!r} rendered nothing"


def test_pages_expose_the_expected_contract():
    for page in PAGES:
        module = page.render.__module__
        assert page.render.__name__ == "render", module
        assert page.label and page.key


def test_context_reports_missing_artifacts_honestly():
    ctx = DashboardContext.build()
    ready, total = ctx.readiness()
    assert total == len(ctx.statuses) and ready >= 0
    assert ctx.events() is None
    assert ctx.alerts() is None
    assert ctx.metrics() is None
    assert ctx.entity_ids() == []
    assert {a.key for a in ctx.missing("events", "alerts")} == {"events", "alerts"}


def test_overview_renders_with_development_fixture():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value("SOC Overview").run()
    assert not at.exception
    body = " ".join(block.value for block in at.markdown)
    assert "Events processed" in body
    assert at.info and "Development fixture active" in at.info[0].value


def test_simulator_disables_injection_until_prerequisites_exist():
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    at.sidebar.radio[0].set_value("Attack Simulator").run()
    assert not at.exception

    inject = [b for b in at.button if b.label == "INJECT ATTACK"]
    assert len(inject) == 1 and inject[0].disabled

    scenarios = [s for s in at.selectbox if s.key == "sim_attack"]
    assert scenarios and "Impossible Travel" in scenarios[0].options


def test_alert_filters_are_present_and_queue_populates():
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    at.sidebar.radio[0].set_value("Alert Queue").run()
    assert not at.exception
    assert {"flt_severity", "flt_attack", "flt_entity_type"} <= {w.key for w in at.multiselect}
    assert "flt_risk" in {w.key for w in at.slider}
    assert at.dataframe, "expected alert queue table with fixture data"
    assert at.selectbox, "expected alert detail selector"


def test_performance_page_refuses_to_invent_metrics():
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    at.sidebar.radio[0].set_value("Model Performance").run()
    assert not at.exception
    body = " ".join(block.value for block in at.markdown)
    assert "PR-AUC" in body
    assert 'class="value pending">--' in body
    assert at.warning and "Evaluation not yet available" in at.warning[0].value
    assert at.info and any("Evaluation / debug view" in block.value for block in at.info)
    captions = " ".join(block.value for block in at.caption)
    assert "Accuracy is intentionally absent" in captions
