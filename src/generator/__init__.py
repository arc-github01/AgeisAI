"""Synthetic enterprise environment generator for AEGIS.

Phase 2 scope: persistent entity profiles + session-based normal behaviour.
Attack injection lives in ``attacks.py`` (Phase 3).
"""

from __future__ import annotations

from .entities import EntityBehavioralProfile, build_population
from .generator import generate_normal_events, main
from .normal_behavior import NormalBehaviorEngine

__all__ = [
    "EntityBehavioralProfile",
    "build_population",
    "NormalBehaviorEngine",
    "generate_normal_events",
    "main",
]
