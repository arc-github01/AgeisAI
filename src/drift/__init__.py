"""Phase 8 concept-drift runner: risk-gated adaptive profile replay.

Frozen Phase-4 ``ProfileBundle`` remains the offline feature source. This
module seeds a separate ``AdaptiveProfileStore``, replays post-cutoff events
in time order, and updates only when ``risk_score < baseline_update_max_risk``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.artifacts import artifact_path
from src.config import load_config
from src.profiling import ProfileBundle

from .evaluate import dump_store, evaluate_drift, save_evaluation
from .store import AdaptiveProfileStore
from .update import replay


def load_drift_params(cfg: Any | None = None) -> dict[str, float | bool]:
    cfg = cfg or load_config()
    half = cfg.get("drift.ewma_halflife_days")
    if half is None:
        half = cfg["profiling.ewma_halflife_days"]
    return {
        "enabled": bool(cfg.get("drift.enabled", True)),
        "baseline_update_max_risk": float(cfg["drift.baseline_update_max_risk"]),
        "rolling_window_days": float(cfg["drift.rolling_window_days"]),
        "ewma_halflife_days": float(half),
    }


def run(*, cutoff: pd.Timestamp | None = None) -> dict[str, Path]:
    """Seed adaptive store, replay post-cutoff events, write artifacts."""
    cfg = load_config()
    params = load_drift_params(cfg)
    if not params["enabled"]:
        raise RuntimeError("drift.enabled is false; refusing to run Phase 8")

    bundle: ProfileBundle = joblib.load(artifact_path("profiles"))
    events = pd.read_parquet(artifact_path("events"))
    risk_scores = pd.read_parquet(artifact_path("risk_scores"))

    cutoff_ts = pd.Timestamp(cutoff) if cutoff is not None else pd.Timestamp(bundle.cutoff)
    # Strictly after frozen cutoff — training-window history already lives in the seed.
    post = events[events["timestamp"] > cutoff_ts].copy()

    store = AdaptiveProfileStore.from_bundle(
        bundle,
        ewma_halflife_days=float(params["ewma_halflife_days"]),
        baseline_update_max_risk=float(params["baseline_update_max_risk"]),
        rolling_window_days=float(params["rolling_window_days"]),
    )
    replay(store, post, risk_scores)

    profiles_path = dump_store(store)
    doc = evaluate_drift(
        store,
        post,
        risk_scores,
        max_risk=float(params["baseline_update_max_risk"]),
    )
    doc["cutoff"] = cutoff_ts.isoformat()
    doc["n_post_cutoff_events"] = int(len(post))
    eval_path = save_evaluation(doc)
    return {"adaptive_profiles": profiles_path, "drift_evaluation": eval_path}


__all__ = ["load_drift_params", "run"]
