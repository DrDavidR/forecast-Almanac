"""
Canonical structural attractor registry for CANONICAL_CAUSAL_BASELINE_0.

12 structural specs — ML specs (LGBM_MODEL, PROPHET_MODEL) exist in V102 but
are excluded from the canonical baseline. Source: phase4ubr decision lock.
"""
from __future__ import annotations

from typing import List

from s4dr_canonical.modelo_v102 import PeriodSpecFactory, _ModelSpec

STRUCTURAL_ATTRACTOR_NAMES: tuple[str, ...] = (
    "T7",
    "T7_30",
    "T14",
    "T28",
    "T56",
    "T84",
    "DOM",
    "BIMONTH_M_M1",
    "ROLLING3M",
    "PAY_CYCLE",
    "LY_SAME_BUCKET",
    "LY_DOM",
)

STRUCTURAL_ATTRACTOR_COUNT: int = len(STRUCTURAL_ATTRACTOR_NAMES)


def canonical_structural_specs() -> List[_ModelSpec]:
    """Return the 12 structural specs."""
    return PeriodSpecFactory.default_specs()
