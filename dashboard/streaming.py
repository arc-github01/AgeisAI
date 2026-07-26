"""Streaming Replay Demo — batch ``process_event`` walkthrough for evaluators."""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from . import charts
from .components import KPI, kpi_row, page_header, panel_title, pipeline_strip
from .contracts import ATTACK_DISPLAY_NAMES
from .data_provider import DashboardDataProvider
from .replay_service import StreamingReplayService
from .state import DashboardContext

TITLE = "Streaming Replay"
SUBTITLE = "Walk historical events through process_event() in reliable batches"


def _render_prerequisites(service: StreamingReplayService) -> None:
    rows = "".join(
        f"<div class='aegis-status'>"
        f"<span class=\"{'ok' if item.ready else 'pending'}\">"
        f"{'&#9679;' if item.ready else '&#9675;'} {html.escape(item.key)}</span>"
        f"<span class='ph'>{'ready' if item.ready else f'P{item.phase}'}</span>"
        f"</div>"
        for item in service.prerequisites()
    )
    st.markdown(rows, unsafe_allow_html=True)


def _session_defaults() -> None:
    st.session_state.setdefault("stream_cursor", 0)
    st.session_state.setdefault("stream_results", [])
    st.session_state.setdefault("stream_engine", None)
    st.session_state.setdefault("stream_error", None)


def _results_frame(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _display_live_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    display = frame.copy()
    if "predicted_attack_type" in display.columns:
        display["predicted_attack_type"] = display["predicted_attack_type"].map(
            lambda x: ATTACK_DISPLAY_NAMES.get(str(x), str(x))
        )
    keep = [
        c
        for c in (
            "timestamp",
            "entity_id",
            "risk_score",
            "severity",
            "predicted_attack_type",
            "attack_confidence",
            "alerted",
            "short_reason",
            "latency_ms",
        )
        if c in display.columns
    ]
    return display[keep].rename(
        columns={
            "timestamp": "Timestamp",
            "entity_id": "Entity",
            "risk_score": "Risk",
            "severity": "Severity",
            "predicted_attack_type": "Predicted Type",
            "attack_confidence": "Confidence",
            "alerted": "Alerted",
            "short_reason": "Reason",
            "latency_ms": "Latency (ms)",
        }
    )


def render(ctx: DashboardContext) -> None:
    provider = DashboardDataProvider.from_context(ctx)
    service = StreamingReplayService(ctx, provider)
    _session_defaults()

    page_header(TITLE, SUBTITLE)
    st.caption(
        "Demo only — batches call the real Phase 9 streaming engine. "
        "Operational Overview / Alert Queue continue to read persisted artifacts."
    )

    if provider.is_mock:
        st.info(
            "**Development fixture active** — live replay stays disabled until "
            "pipeline models and events exist on disk."
        )

    ready = service.is_ready()
    left, right = st.columns([1, 1.4])
    with left:
        panel_title("Replay controls")
        batch_size = st.slider("Events per batch", 1, 50, 10, key="stream_batch")
        apply_drift = st.checkbox(
            "Apply risk-gated adaptive updates",
            value=True,
            key="stream_drift",
        )
        col_a, col_b, col_c = st.columns(3)
        run_clicked = col_a.button(
            "Replay next batch",
            type="primary",
            disabled=not ready,
            key="stream_run",
        )
        reset_clicked = col_b.button("Reset replay", key="stream_reset")
        col_c.caption(f"{service.event_count():,} events available")

        st.write("")
        panel_title("Pipeline readiness")
        _render_prerequisites(service)
        pipeline_strip(
            {
                "EVENT",
                "FEATURES",
                "PROFILE",
                "ANOMALY",
                "CLASSIFIER",
                "RISK",
                "EXPLANATION",
                "ALERT",
            }
            if ready
            else {"EVENT"}
        )

    if reset_clicked:
        st.session_state["stream_cursor"] = 0
        st.session_state["stream_results"] = []
        st.session_state["stream_engine"] = None
        st.session_state["stream_error"] = None

    if run_clicked:
        try:
            engine = st.session_state.get("stream_engine")
            if engine is None:
                with st.spinner("Loading streaming engine…"):
                    engine = service.load_engine(apply_drift_updates=apply_drift)
                    st.session_state["stream_engine"] = engine
            engine.apply_drift_updates = apply_drift
            outcome = service.replay_batch(
                engine=engine,
                cursor=int(st.session_state["stream_cursor"]),
                batch_size=int(batch_size),
            )
            if outcome.error and not outcome.success:
                st.session_state["stream_error"] = outcome.error
            else:
                st.session_state["stream_error"] = None
                st.session_state["stream_cursor"] = outcome.cursor
                if not outcome.results.empty:
                    existing = list(st.session_state["stream_results"])
                    existing.extend(outcome.results.to_dict(orient="records"))
                    st.session_state["stream_results"] = existing[-500:]
                if outcome.processed == 0 and outcome.cursor >= outcome.total_events:
                    st.success("Reached the end of the available event stream.")
        except Exception as exc:  # noqa: BLE001
            st.session_state["stream_error"] = str(exc)

    live = _results_frame(st.session_state["stream_results"])
    cursor = int(st.session_state["stream_cursor"])
    total = service.event_count()
    alerts_live = int(live["alerted"].sum()) if not live.empty and "alerted" in live.columns else 0
    mean_latency = (
        float(live["latency_ms"].mean())
        if not live.empty and "latency_ms" in live.columns
        else None
    )

    with right:
        panel_title("Live batch results")
        kpi_row(
            [
                KPI("Cursor", f"{cursor:,} / {total:,}"),
                KPI("Scored this session", len(live)),
                KPI("Alerts raised", alerts_live),
                KPI(
                    "Mean latency",
                    f"{mean_latency:.0f} ms" if mean_latency is not None else None,
                    foot="process_event wall time",
                ),
            ]
        )
        if st.session_state.get("stream_error"):
            st.warning(st.session_state["stream_error"])
        display = _display_live_table(live.tail(100).iloc[::-1])
        if display.empty:
            st.caption(
                "Press **Replay next batch** to score events through "
                "`StreamingEngine.process_event`."
            )
        else:
            st.dataframe(display, width="stretch", hide_index=True, height=360)
            if "risk_score" in live.columns and "timestamp" in live.columns:
                st.plotly_chart(
                    charts.risk_timeline(live[["timestamp", "risk_score"]]),
                    width="stretch",
                )

    st.write("")
    panel_title("Offline Phase 9 replay artifact")
    metrics = service.streaming_metrics_summary()
    if metrics:
        latency = metrics.get("latency") if isinstance(metrics.get("latency"), dict) else {}
        st.caption(
            "Persisted streaming evaluation (not live session). "
            "Ground-truth campaign labels are not shown here."
        )
        cols = st.columns(4)
        cols[0].metric("Events replayed", metrics.get("n_events", latency.get("n", "—")))
        cols[1].metric(
            "Mean latency",
            f"{latency['mean_ms']:.0f} ms" if latency.get("mean_ms") is not None else "—",
        )
        cols[2].metric(
            "P95 latency",
            f"{latency['p95_ms']:.0f} ms" if latency.get("p95_ms") is not None else "—",
        )
        cols[3].metric(
            "Throughput",
            f"{latency['throughput_events_per_sec']:.1f}/s"
            if latency.get("throughput_events_per_sec") is not None
            else "—",
        )
    preview = service.offline_replay_preview(limit=50)
    if preview.empty:
        st.caption(
            "No `streaming_scores.parquet` yet — run `python -m src.detection` after "
            "Phases 5–8 to persist an offline replay artifact."
        )
    else:
        st.dataframe(preview, width="stretch", hide_index=True, height=280)
