"""CLI entry point: ``python -m src.drift``."""

from __future__ import annotations

import json

from . import run


def main() -> None:
    paths = run()
    # Re-read evaluation for the console summary.
    eval_path = paths["drift_evaluation"]
    doc = json.loads(eval_path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "cutoff": doc.get("cutoff"),
                "n_post_cutoff_events": doc.get("n_post_cutoff_events"),
                "gate": doc.get("gate"),
                "poisoning_resistance": {
                    k: doc["poisoning_resistance"][k]
                    for k in (
                        "high_risk_block_rate",
                        "malicious_block_rate",
                        "malicious_high_risk_block_rate",
                    )
                    if k in doc.get("poisoning_resistance", {})
                },
                "adaptation": {
                    k: doc["adaptation"][k]
                    for k in (
                        "low_risk_benign_update_rate",
                        "insider_drift_update_rate",
                        "insider_drift_low_risk_update_rate",
                    )
                    if k in doc.get("adaptation", {})
                },
                "n_insider_exhibits": len(doc.get("insider_drift_exhibits", [])),
                "paths": {k: str(v) for k, v in paths.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
