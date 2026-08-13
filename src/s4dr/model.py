"""
Canonical S4DR model — structural-only, paper-aligned.

Scientific invariants (always active, not configurable):
  STRUCTURAL_ONLY: exactly 12 structural candidates; LGBM and CatBoost are
    NOT part of the canonical candidate library and must not appear in any
    forecast, selector weight, or metric computation.
  F2: per-pseudo-origin eval matrix reconstruction (eliminates selector lookahead)
  F3: no ML training or inference at any point

Causal improvements incorporated from CANONICAL_CAUSAL_BASELINE_1:
  eval_weeks          = 10  (matches production effective default)
  selector_policy     = FROZEN_DURING_EVALUATION  (Hedge disabled, frozen priors)
  autoencoder_enabled = False

F1 (as-of cleaning) is injected via CausalPreprocessConfig; always True in
canonical runs but is a per-run protocol choice, not a model invariant.
"""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from s4dr.attractors import STRUCTURAL_ATTRACTOR_NAMES
from s4dr_causal_experimental.model import CausalPreprocessConfig, CausalS4DRModelV102
from s4dr_canonical.modelo_v102 import S4DRModelV102

_BASELINE_PREPROCESS: CausalPreprocessConfig = CausalPreprocessConfig(
    aplicar_homologacion_ceros=True,
    aplicar_outlier_clipping=True,
)

_STRUCTURAL_CODES: frozenset[str] = frozenset(STRUCTURAL_ATTRACTOR_NAMES)
_ML_CODES: frozenset[str] = frozenset({"LGBM_MODEL", "CATBOOST_MODEL", "PROPHET_MODEL"})

# Column names that would appear in a forecast DataFrame if ML candidates leaked.
_ML_COLUMN_NAMES: frozenset[str] = frozenset({
    "yhat_LGBM_MODEL", "yhat_CATBOOST_MODEL", "yhat_PROPHET_MODEL"
})

# All valid kind values dispatched by _Attractors.predict_non_ml in modelo_v102.py.
# Any kind outside this set must raise — silent 0.0 fallback is forbidden.
_KNOWN_ATTRACTOR_KINDS: frozenset[str] = frozenset({
    "weekly_phase", "dom", "bimonth", "rolling3", "paycycle", "ly_bucket", "ly_dom"
})


def assert_no_ml_columns(df: pd.DataFrame) -> None:
    """Integrity gate: raise if any prohibited ML prediction column is present.

    Column PRESENCE alone is contamination — zeros are not exempt.
    Call this on any forecast panel before computing metrics.
    """
    present = [c for c in df.columns if c in _ML_COLUMN_NAMES]
    if present:
        raise ValueError(
            f"ML contamination gate FAIL — prohibited columns present: {present}. "
            f"Column presence alone is contamination; zeros are not exempt."
        )


class S4DRModel(CausalS4DRModelV102):
    """
    Canonical structural S4DR for the paper.

    Enforces exactly 12 structural candidates. LGBM_MODEL and CATBOOST_MODEL
    are excluded from self.models at construction — they never receive
    predictions, selector weights, or appear in any metric.
    """

    CANONICAL_CANDIDATE_COUNT: int = 12
    CANONICAL_CANDIDATES: tuple[str, ...] = tuple(STRUCTURAL_ATTRACTOR_NAMES)

    def __init__(
        self,
        id_unico: str,
        *args: Any,
        eval_weeks: int = 10,
        causal_preprocess: Optional[CausalPreprocessConfig] = None,
        **kwargs: Any,
    ) -> None:
        if causal_preprocess is None:
            causal_preprocess = _BASELINE_PREPROCESS
        # Merge any caller-supplied disabled_specs with the ML exclusion list
        caller_disabled: set[str] = set(kwargs.pop("disabled_specs", None) or set())
        kwargs["disabled_specs"] = caller_disabled | _ML_CODES
        super().__init__(
            id_unico,
            *args,
            eval_weeks=eval_weeks,
            causal_preprocess=causal_preprocess,
            **kwargs,
        )
        # Defensive: strip any ML spec that might survive the filter
        self.models = [m for m in self.models if m.code not in _ML_CODES]
        self.specs_included = [m.code for m in self.models]
        assert len(self.models) == self.CANONICAL_CANDIDATE_COUNT, (
            f"Expected {self.CANONICAL_CANDIDATE_COUNT} structural candidates, "
            f"got {len(self.models)}: {self.specs_included}"
        )

    def _prepare_ml_models(self) -> None:
        """No-op: structural-only model never trains or loads ML models."""

    def _predict_model(self, spec: Any, d: Any, hist: Any) -> float:
        """Override: raise explicitly for any unknown attractor kind.

        predict_non_ml in the immutable v102 has a silent return 0.0 fallback
        for unrecognised kinds.  This override eliminates that silent path so
        any unknown spec surfaces immediately as a hard error.
        """
        if spec.kind not in _KNOWN_ATTRACTOR_KINDS:
            raise ValueError(
                f"Unknown spec kind {spec.kind!r} for candidate {spec.code!r}. "
                f"Known kinds: {sorted(_KNOWN_ATTRACTOR_KINDS)}. "
                f"No silent fallback in canonical model."
            )
        return super()._predict_model(spec, d, hist)

    def _new_scratch_model(self, history: pd.DataFrame) -> S4DRModelV102:
        scratch_kw = {**self._scratch_kwargs}
        existing_disabled: set[str] = set(scratch_kw.pop("disabled_specs", None) or set())
        scratch_kw["disabled_specs"] = existing_disabled | _ML_CODES
        scratch = S4DRModelV102(
            self.id,
            *self._scratch_args,
            **scratch_kw,
        )
        scratch._prepare_ml_models = lambda: None  # type: ignore[method-assign]
        scratch.actualizar_modelo(history)
        return scratch
