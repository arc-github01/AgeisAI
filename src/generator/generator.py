"""CLI entry point for synthetic normal event generation (Phase 2).

Usage::

    python -m src.generator.generator
    python -m src.generator.generator --profile full
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.paths import ensure_dirs
from src.schema import validate_events

from .entities import build_population
from .normal_behavior import NormalBehaviorEngine, summarize_normal_dataset


def generate_normal_events(profile_name: str | None = None) -> pd.DataFrame:
    cfg = load_config()
    if profile_name:
        # Temporarily override profile via shallow dict copy is not supported;
        # read profile block directly.
        profiles_cfg = cfg["generator.profiles"]
        if profile_name not in profiles_cfg:
            raise ValueError(f"unknown generator profile: {profile_name!r}")
        profile = {"name": profile_name, **profiles_cfg[profile_name]}
    else:
        profile = cfg.generator_profile()

    start_date = datetime.fromisoformat(str(cfg["generator.start_date"]))
    noise_cfg = cfg["generator.noise"]
    attack_prevalence = float(cfg["generator.attack_prevalence"])

    # Phase 2: generate benign events only; reserve headroom for Phase 3 attacks.
    benign_target = int(profile["target_events"] * (1.0 - attack_prevalence))

    population = build_population(
        n_users=int(profile["n_users"]),
        n_service_accounts=int(profile["n_service_accounts"]),
        n_edge_devices=int(profile["n_edge_devices"]),
        noise_cfg=noise_cfg,
    )

    engine = NormalBehaviorEngine(
        population,
        start_date=start_date,
        simulation_days=int(profile["days"]),
        target_events=benign_target,
        noise_cfg=noise_cfg,
    )
    frame = engine.generate()
    return validate_events(frame, strict_order=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="AEGIS synthetic normal event generator")
    parser.add_argument("--profile", choices=["dev", "full"], default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override output path (default: data/generated/normal_events.csv)",
    )
    args = parser.parse_args(argv)

    cfg = load_config()
    out_dir = ensure_dirs(cfg["paths.data_generated"])[0]
    output = args.output or (out_dir / "normal_events.csv")

    frame = generate_normal_events(args.profile)
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.suffix.lower() == ".parquet":
        frame.to_parquet(output, index=False)
    else:
        frame.to_csv(output, index=False)

    meta = {
        "profile": args.profile or cfg["generator.profile"],
        **summarize_normal_dataset(frame),
    }
    meta_path = output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved {len(frame):,} benign events -> {output}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
