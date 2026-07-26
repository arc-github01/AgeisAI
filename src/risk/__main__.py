"""CLI entry point: ``python -m src.risk``."""

from __future__ import annotations

import json

import pandas as pd

from src.artifacts import artifact_path

from .engine import run as run_engine
from .evaluate import evaluate_risk, save_evaluation


def main() -> None:
    paths = run_engine()
    risk_scores = pd.read_parquet(paths["risk_scores"])
    # Reconstruct list columns for evaluation convenience.
    risk_scores["reason_codes"] = risk_scores["reason_codes"].map(json.loads)
    risk_scores["reasons"] = risk_scores["reasons"].map(json.loads)

    alerts = pd.read_parquet(paths["alerts"])
    if not alerts.empty and "reason_codes" in alerts.columns:
        # reason_codes may already be strings from parquet
        sample = alerts["reason_codes"].iloc[0]
        if isinstance(sample, str):
            alerts["reason_codes"] = alerts["reason_codes"].map(json.loads)
            alerts["top_contributors"] = alerts["top_contributors"].map(json.loads)

    event_scores = pd.read_parquet(artifact_path("event_scores"))
    ops_path = paths["alerts"].with_name("alert_ops.json")
    n_suppressed = 0
    if ops_path.exists():
        n_suppressed = int(json.loads(ops_path.read_text(encoding="utf-8")).get("n_suppressed", 0))
    event_doc, campaign_doc = evaluate_risk(
        risk_scores, event_scores, alerts, n_suppressed=n_suppressed
    )
    eval_paths = save_evaluation(event_doc, campaign_doc)
    paths.update(eval_paths)

    metrics = event_doc["metrics"]["event_level"]
    campaign = campaign_doc["metrics"]
    print(
        json.dumps(
            {
                "pr_auc": {
                    "isolation_forest": metrics["isolation_forest"]["pr_auc"],
                    "rule": metrics["rule"]["pr_auc"],
                    "hybrid_fpr_matched": metrics["hybrid_fpr_matched"]["pr_auc"],
                    "hybrid_deployed": metrics["hybrid_deployed"]["pr_auc"],
                },
                "malicious_campaign_detection": {
                    "isolation_forest": campaign["isolation_forest"].get(
                        "detection_rate"
                    ),
                    "rule": campaign["rule"].get("detection_rate"),
                    "hybrid": campaign["hybrid"].get("detection_rate"),
                },
                "operational": event_doc["metrics"]["operational"],
                "paths": {k: str(v) for k, v in paths.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
