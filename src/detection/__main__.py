"""CLI entry point: ``python -m src.detection`` — streaming replay + latency."""

from __future__ import annotations

import json

from .replay import run_replay


def main() -> None:
    paths = run_replay(apply_drift_updates=True)
    metrics = json.loads(paths["streaming_metrics"].read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "n_events": metrics["n_events"],
                "n_alerts": metrics["n_alerts"],
                "n_profile_updates": metrics["n_profile_updates"],
                "latency": metrics["latency"],
                "paths": {k: str(v) for k, v in paths.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
