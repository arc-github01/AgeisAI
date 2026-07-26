"""Build the registered Phase 4 artifacts."""
from __future__ import annotations
import json
import pandas as pd
from src.artifacts import artifact_path
from .pipeline import build_features, save_features

def main() -> None:
    events = pd.read_parquet(artifact_path("events"))
    features, bundle = build_features(events)
    paths = save_features(features, bundle)
    print(json.dumps({"rows": len(features), "paths": {k: str(v) for k, v in paths.items()}}, indent=2))

if __name__ == "__main__":
    main()
