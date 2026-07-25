"""AEGIS - Adaptive Behavioral Threat Detection for IT/OT Systems.

Package layout (each sub-package is introduced by the phase that needs it):

    src.config          configuration loading (YAML single source of truth)
    src.paths           project-root-relative filesystem locations
    src.schema          canonical event contract + threat taxonomy
    src.utils           seeding, logging
    src.generator       synthetic entities, normal behaviour, attack injection
    src.features        behavioral / geographic / sequence feature engineering
    src.profiling       per-entity + cohort behavioural baselines
    src.models          IsolationForest anomaly detector, attack classifier
    src.detection       detection orchestration + risk engine
    src.explainability  deterministic reason attribution for every alert
    src.drift           adaptive, poisoning-resistant baseline updates
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
