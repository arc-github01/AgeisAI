"""Structured reason codes for HIGH/CRITICAL risk events.

Explanations are attribution of the already-computed evidence sum — never a
second model and never a look at ground truth. Each reason's contribution is
proportional to its share of total evidence, rescaled so the reported
contributions sum exactly to the reported risk score.
"""

from __future__ import annotations

from typing import Any


def build_reasons(
    contributions: dict[str, float],
    *,
    max_reasons: int,
) -> list[dict[str, Any]]:
    """Top contributing reason codes, sorted by contribution descending.

    ``contributions`` maps reason code -> contribution to the final risk score.
    Zero/near-zero contributions are dropped so the explanation stays sparse.
    """
    ordered = sorted(
        ((code, float(value)) for code, value in contributions.items() if value > 1e-9),
        key=lambda item: item[1],
        reverse=True,
    )
    return [
        {"code": code, "contribution": round(value, 3)}
        for code, value in ordered[:max_reasons]
    ]


def reasons_to_text(reasons: list[dict[str, Any]]) -> str:
    """Dashboard-friendly flat string (``CODE (+contrib) + ...``)."""
    if not reasons:
        return ""
    return " + ".join(f"{item['code']} (+{item['contribution']:.1f})" for item in reasons)


def short_reason(reasons: list[dict[str, Any]]) -> str:
    """Single highest-contribution reason, or empty when none fire."""
    return str(reasons[0]["code"]) if reasons else ""


__all__ = ["build_reasons", "reasons_to_text", "short_reason"]
