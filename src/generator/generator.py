"""CLI entry point for synthetic environment generation (Phases 2-3).

Produces the two artifacts the rest of the pipeline consumes, both declared in
:mod:`src.artifacts` so that no path string is written down twice:

    data/generated/entities.json     entity roster + ground-truth behaviour
    data/generated/events.parquet    labelled access-event dataset

Usage::

    python -m src.generator
    python -m src.generator --profile full
    python -m src.generator --benign-only
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.artifacts import artifact_path
from src.config import load_config
from src.schema import validate_events

from .attacks import (
    AttackCampaign,
    AttackOrchestrator,
    campaign_summary,
    merge_campaigns,
)
from .entities import EntityBehavioralProfile, build_population, population_to_records
from .normal_behavior import NormalBehaviorEngine, summarize_dataset


@dataclass(frozen=True)
class GenerationResult:
    """Everything one generator run produced, before it is written to disk."""

    profile_name: str
    population: list[EntityBehavioralProfile]
    events: pd.DataFrame
    campaigns: list[AttackCampaign] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "profile": self.profile_name,
            "entities": len(self.population),
            **summarize_dataset(self.events),
            "attacks": campaign_summary(self.campaigns),
        }


def resolve_profile(name: str | None = None) -> dict[str, Any]:
    """Look up a sizing profile from ``generator.profiles``."""
    cfg = load_config()
    if name is None:
        return cfg.generator_profile()
    profiles = cfg["generator.profiles"]
    if name not in profiles:
        raise ValueError(
            f"unknown generator profile {name!r}; available: {sorted(profiles)}"
        )
    return {"name": name, **profiles[name]}


def generate_normal_events(
    profile_name: str | None = None,
    *,
    profile: dict[str, Any] | None = None,
) -> GenerationResult:
    """Build the entity population and simulate its benign activity.

    ``profile`` accepts an explicit sizing block, which keeps tests fast without
    having to mutate the shared configuration.
    """
    cfg = load_config()
    active = dict(profile) if profile is not None else resolve_profile(profile_name)

    start_date = datetime.fromisoformat(str(cfg["generator.start_date"]))
    noise_cfg = cfg["generator.noise"]

    # Phase 3 injects attacks into the headroom this leaves behind.
    attack_prevalence = float(cfg["generator.attack_prevalence"])
    benign_target = max(1, int(active["target_events"] * (1.0 - attack_prevalence)))

    population = build_population(
        n_users=int(active["n_users"]),
        n_service_accounts=int(active["n_service_accounts"]),
        n_edge_devices=int(active["n_edge_devices"]),
    )
    engine = NormalBehaviorEngine(
        population,
        start_date=start_date,
        simulation_days=int(active["days"]),
        target_events=benign_target,
        noise_cfg=noise_cfg,
    )
    events = validate_events(engine.generate(), strict_order=True)
    return GenerationResult(
        profile_name=str(active.get("name", "custom")),
        population=population,
        events=events,
    )


def generate_dataset(
    profile_name: str | None = None,
    *,
    profile: dict[str, Any] | None = None,
    inject: bool = True,
) -> GenerationResult:
    """The full dataset: benign behaviour with attack campaigns folded in.

    The benign pass is reused unchanged, so the attacks land in the headroom
    ``generate_normal_events`` already reserved. ``inject=False`` gives the
    benign-only dataset back, which is what the cold-start and drift work needs
    as a clean reference.
    """
    benign_run = generate_normal_events(profile_name, profile=profile)
    if not inject:
        return benign_run

    cfg = load_config()
    active = dict(profile) if profile is not None else resolve_profile(profile_name)

    orchestrator = AttackOrchestrator(
        benign_run.population,
        benign_run.events,
        start_date=datetime.fromisoformat(str(cfg["generator.start_date"])),
        simulation_days=int(active["days"]),
        attacks_cfg=cfg["generator.attacks"],
        attack_mix=cfg["generator.attack_mix"],
        prevalence=float(cfg["generator.attack_prevalence"]),
        max_plausible_kmh=float(cfg["features.max_plausible_kmh"]),
    )
    campaigns = orchestrator.inject()
    events = validate_events(
        merge_campaigns(benign_run.events, campaigns), strict_order=True
    )
    return GenerationResult(
        profile_name=benign_run.profile_name,
        population=benign_run.population,
        events=events,
        campaigns=campaigns,
    )


def save(result: GenerationResult) -> dict[str, Path]:
    """Write the run to its registered artifact paths."""
    entities_path = artifact_path("entities", ensure_parent=True)
    entities_path.write_text(
        json.dumps(
            {
                "profile": result.profile_name,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "master_seed": int(load_config()["seed.master"]),
                "dataset_summary": result.summary(),
                "entities": population_to_records(result.population),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    events_path = artifact_path("events", ensure_parent=True)
    result.events.to_parquet(events_path, index=False)
    campaigns_path = artifact_path("campaign_metadata", ensure_parent=True)
    campaigns_path.write_text(
        json.dumps(
            [
                {
                    "campaign_id": campaign.campaign_id,
                    "attack_type": campaign.attack_type.value,
                    "entity_ids": list(campaign.entity_ids),
                    "stealthy": campaign.stealthy,
                }
                for campaign in result.campaigns
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"entities": entities_path, "events": events_path, "campaign_metadata": campaigns_path}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="AEGIS synthetic environment and access-event generator"
    )
    parser.add_argument("--profile", choices=["dev", "full"], default=None)
    parser.add_argument(
        "--benign-only",
        action="store_true",
        help="skip attack injection and emit a clean reference dataset",
    )
    args = parser.parse_args(argv)

    result = generate_dataset(args.profile, inject=not args.benign_only)
    paths = save(result)

    print(json.dumps(result.summary(), indent=2))
    for key, path in paths.items():
        print(f"{key:>9}: {path}")


if __name__ == "__main__":
    main()
