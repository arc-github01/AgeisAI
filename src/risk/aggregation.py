"""Risk aggregation: time decay, saturating combination, severity mapping.

THE FORMULA
-----------
For entity ``e`` with events at ``t_1 < t_2 < ...`` and accumulated evidence
state ``S`` (``S_0 = 0``):

    dt          = (t_k - t_{k-1}) in seconds                      # 0 for k = 1
    decayed     = S_{k-1} * 0.5 ** (dt / halflife_seconds)
    E_k         = sum_i weight_i * activation_i(event_k)          # instantaneous
    P_k         = persistence_weight * decayed                    # history
    T_k         = E_k + P_k                                       # total evidence
    risk_k      = 100 * (1 - exp(-T_k / evidence_scale))
    S_k         = min(decayed + E_k, state_cap)

Why this shape:

* **Saturating, not additive.** ``1 - exp(-T)`` is a soft OR. Many weak signals
  can accumulate into a high score (the stealth-campaign requirement) while no
  finite amount of evidence can leave ``[0, 100]``, so the score never needs an
  arbitrary clip that would destroy ranking information.
* **Decay is driven by wall-clock time, never by row counts.** An entity that
  goes quiet for two half-lives keeps a quarter of its evidence regardless of
  how many events other entities produced meanwhile.
* **State accumulates ``E`` only, never ``P``.** Feeding the persistence term
  back into the state would compound history against itself and let a single
  burst ratchet an entity to CRITICAL forever.
* **Contributions reconcile exactly.** Because ``risk`` is a monotone function
  of a *sum*, each term's share of ``T`` is a defensible attribution, and
  rescaling those shares by ``risk`` makes the reported contributions sum to the
  reported score exactly. No post-hoc explanation model is involved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.config import load_config
from src.schema import Severity


@dataclass
class EntityRiskState:
    """Causal per-entity state. This is the entire streaming checkpoint."""

    evidence: float = 0.0
    last_timestamp: float | None = None

    def copy(self) -> "EntityRiskState":
        return EntityRiskState(self.evidence, self.last_timestamp)


def decay_factor(elapsed_seconds: float, halflife_seconds: float) -> float:
    """``0.5 ** (dt / halflife)``, clamped so negative gaps cannot amplify risk."""
    if halflife_seconds <= 0:
        return 0.0
    elapsed = max(0.0, float(elapsed_seconds))
    return float(0.5 ** (elapsed / halflife_seconds))


def saturating_risk(total_evidence: float, evidence_scale: float) -> float:
    """Map unbounded non-negative evidence onto ``[0, 100]`` (asymptotic)."""
    if evidence_scale <= 0:
        raise ValueError("evidence_scale must be positive")
    total = max(0.0, float(total_evidence))
    return float(100.0 * (1.0 - math.exp(-total / evidence_scale)))


def evidence_for_risk(risk_score: float, evidence_scale: float) -> float:
    """Inverse of :func:`saturating_risk`, used to express bands as evidence."""
    ratio = min(max(float(risk_score), 0.0), 99.999999) / 100.0
    return float(-evidence_scale * math.log(1.0 - ratio))


def load_severity_bands(cfg=None) -> dict[str, tuple[float, float]]:
    cfg = cfg or load_config()
    bands = cfg["risk.severity_bands"]
    return {name: (float(low), float(high)) for name, (low, high) in bands.items()}


def severity_for(risk_score: float, bands: dict[str, tuple[float, float]]) -> str:
    """Map a score to its severity band.

    Bands are declared as inclusive integer ranges with unit gaps
    (``[0, 30]``, ``[31, 60]``, ...), so a continuous score is assigned to the
    lowest band whose upper bound it does not exceed.
    """
    ordered = sorted(bands.items(), key=lambda item: item[1][0])
    for name, (_low, high) in ordered:
        if risk_score <= high:
            return name
    return ordered[-1][0]


#: Ascending severity order, used for escalation comparisons.
SEVERITY_RANK: dict[str, int] = {s.value: i for i, s in enumerate(Severity)}


def severity_rank(name: str) -> int:
    return SEVERITY_RANK[name]


__all__ = [
    "EntityRiskState",
    "SEVERITY_RANK",
    "decay_factor",
    "evidence_for_risk",
    "load_severity_bands",
    "saturating_risk",
    "severity_for",
    "severity_rank",
]
