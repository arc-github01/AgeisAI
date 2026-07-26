"""Synthetic enterprise environment generator for AEGIS.

Persistent entity profiles and session-based normal behaviour (Phase 2), plus
the attack campaigns injected into that stream (Phase 3).
"""

from __future__ import annotations

from .attacks import (
    HOSTILE_ORIGINS,
    AttackCampaign,
    AttackOrchestrator,
    campaign_summary,
    merge_campaigns,
)
from .entities import (
    EntityBehavioralProfile,
    build_population,
    load_population_records,
    population_to_records,
    profile_to_record,
    record_to_profile,
)
from .live_injection import (
    build_live_campaign,
    campaign_to_injection_frame,
    synthesize_live_attack,
)
from .generator import (
    GenerationResult,
    generate_dataset,
    generate_normal_events,
    main,
    resolve_profile,
    save,
)
from .normal_behavior import (
    NormalBehaviorEngine,
    finalize_events,
    summarize_dataset,
)

__all__ = [
    "HOSTILE_ORIGINS",
    "AttackCampaign",
    "AttackOrchestrator",
    "EntityBehavioralProfile",
    "GenerationResult",
    "NormalBehaviorEngine",
    "build_live_campaign",
    "build_population",
    "campaign_summary",
    "campaign_to_injection_frame",
    "finalize_events",
    "generate_dataset",
    "generate_normal_events",
    "load_population_records",
    "main",
    "merge_campaigns",
    "population_to_records",
    "profile_to_record",
    "record_to_profile",
    "resolve_profile",
    "save",
    "summarize_dataset",
    "synthesize_live_attack",
]
