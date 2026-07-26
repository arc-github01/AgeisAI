"""Great-circle geography helpers.

Two consumers, which is why this lives in ``utils`` rather than inside either
of them:

* the attack generator, which must *construct* a physically impossible journey
  and prove it is impossible before emitting it;
* the Phase 4 feature layer, which must *measure* the same quantity on every
  event pair to produce ``geo_distance_from_baseline`` and ``geo_velocity``.

Both sides using one implementation is what makes the injected ground truth and
the detected signal directly comparable.
"""

from __future__ import annotations

import numpy as np

#: Volumetric mean radius of the Earth (km). Good to ~0.3% for our purposes.
EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two WGS-84 points, in kilometres."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    inner = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    )
    return float(2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(inner, 0.0, 1.0))))


def haversine_km_array(
    lat1: np.ndarray | float,
    lon1: np.ndarray | float,
    lat2: np.ndarray | float,
    lon2: np.ndarray | float,
) -> np.ndarray:
    """Vectorised :func:`haversine_km` for whole-column feature passes."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(np.asarray(lat2) - np.asarray(lat1))
    dlambda = np.radians(np.asarray(lon2) - np.asarray(lon1))
    inner = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(inner, 0.0, 1.0)))


def implied_velocity_kmh(distance_km: float, elapsed_seconds: float) -> float:
    """Speed a journey would have required.

    A non-zero distance covered in no time is genuinely infinite velocity, not
    an error: two simultaneous logins from different cities is exactly the
    signal impossible-travel detection exists to catch. Returning ``inf`` keeps
    that comparable against a threshold instead of raising.
    """
    if elapsed_seconds <= 0:
        return 0.0 if distance_km == 0 else float("inf")
    return float(distance_km / (elapsed_seconds / 3600.0))


__all__ = [
    "EARTH_RADIUS_KM",
    "haversine_km",
    "haversine_km_array",
    "implied_velocity_kmh",
]
