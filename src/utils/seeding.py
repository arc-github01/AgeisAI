"""Deterministic randomness for AEGIS.

Reproducibility is an explicit evaluation requirement: a reviewer must be able
to regenerate the dataset and the metrics exactly.

Design: one ``seed.master`` value in config. Every component derives its own
*independent* substream from ``(master, component_name)`` via a stable hash.
Consequences that matter in practice:

* Adding a new component later does not shift the random draws of existing
  components (which would silently invalidate previously reported metrics).
* Two components never accidentally share a stream.
* The derivation uses ``blake2b`` rather than ``hash()`` because Python's
  builtin string hashing is salted per-process and therefore not reproducible.
"""

from __future__ import annotations

import hashlib
import random

import numpy as np

from ..config import load_config

_SEED_SPACE = 2**32


def master_seed() -> int:
    """The single configured root seed."""
    return int(load_config()["seed.master"])


def derive_seed(component: str, master: int | None = None) -> int:
    """Derive a stable 32-bit seed for a named component."""
    root = master_seed() if master is None else int(master)
    digest = hashlib.blake2b(
        f"{root}:{component}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") % _SEED_SPACE


def get_rng(component: str, master: int | None = None) -> np.random.Generator:
    """Return a fresh, independent numpy Generator for a component."""
    return np.random.default_rng(derive_seed(component, master))


def seed_everything(component: str = "global", master: int | None = None) -> int:
    """Seed the global interpreter-level RNGs.

    Prefer :func:`get_rng` inside library code; this exists for third-party
    libraries (scikit-learn, Faker) that read from global state.
    """
    seed = derive_seed(component, master)
    random.seed(seed)
    np.random.seed(seed)
    try:  # Faker is optional at import time; only the generator needs it.
        from faker import Faker

        Faker.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    return seed
