"""Causal experimental v102 variant.

The implementation delegates candidate formulas and selector mechanics to the frozen
v102-compatible package. Its only behavioral changes are the causal eligibility and
per-pseudo-origin preprocessing boundaries required by Phase 3.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Optional

import numpy as np
import pandas as pd

from s4dr_canonical.data_cleaned import (
    aplicar_limpieza_s4dr_hasta_cutoff,
    construir_append_real_clean_no_leak,
)
from s4dr_canonical.modelo_v102 import S4DRModelV102


@dataclass(frozen=True)
class CausalPreprocessConfig:
    aplicar_homologacion_ceros: bool = True
    aplicar_outlier_clipping: bool = True


def _frame_hash(frame: pd.DataFrame) -> str:
    if frame is None or frame.empty:
        return sha256(b"").hexdigest()
    normalized = frame.copy().sort_index(axis=1)
    normalized = normalized.sort_values(list(normalized.columns)).reset_index(drop=True)
    return sha256(normalized.to_csv(index=False).encode("utf-8")).hexdigest()


class CausalS4DRModelV102(S4DRModelV102):
    """v102 formulas with causal cleaning and ML fitting at every selector pseudo-origin."""

    variant_name = "CAUSAL_EXPERIMENTAL_CANDIDATE"
    baseline_name = "BASELINE_V102_COMPAT"

    def __init__(
        self,
        id_unico: str,
        *args: Any,
        causal_preprocess: Optional[CausalPreprocessConfig] = None,
        **kwargs: Any,
    ) -> None:
        self._scratch_args = args
        self._scratch_kwargs = dict(kwargs)
        self.causal_preprocess = causal_preprocess or CausalPreprocessConfig()
        self._raw_source = pd.DataFrame(columns=["fecha", "valor", "id"])
        self._pseudo_origin_audit: list[dict[str, Any]] = []
        super().__init__(id_unico, *args, **kwargs)

    @staticmethod
    def _standardize_raw(frame: pd.DataFrame) -> pd.DataFrame:
        required = {"fecha", "valor", "id"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Causal model requires standardized columns: {sorted(missing)}")
        result = frame[["fecha", "valor", "id"]].copy()
        result["fecha"] = pd.to_datetime(result["fecha"], errors="coerce").dt.normalize()
        result["valor"] = pd.to_numeric(result["valor"], errors="coerce")
        result["id"] = result["id"].astype(str).str.strip()
        return result.dropna(subset=["fecha", "valor", "id"]).sort_values("fecha").drop_duplicates("fecha", keep="last").reset_index(drop=True)

    def _clean_raw_until(self, cutoff: pd.Timestamp) -> pd.DataFrame:
        raw = self._raw_source.loc[self._raw_source["fecha"] <= pd.Timestamp(cutoff).normalize()].copy()
        if raw.empty:
            return raw
        external = raw.rename(columns={"fecha": "Fecha", "valor": "Cantidad", "id": "CajeroId"})
        cleaned, _ = aplicar_limpieza_s4dr_hasta_cutoff(
            external,
            cutoff_date=pd.Timestamp(cutoff).normalize(),
            col_fecha="Fecha",
            col_cajero="CajeroId",
            col_y="Cantidad",
            aplicar_homologacion_ceros=self.causal_preprocess.aplicar_homologacion_ceros,
            aplicar_outlier_clipping=self.causal_preprocess.aplicar_outlier_clipping,
        )
        if "Cantidad_clean" not in cleaned.columns:
            cleaned["Cantidad_clean"] = pd.to_numeric(cleaned["Cantidad"], errors="coerce")
        return cleaned[["Fecha", "Cantidad_clean", "CajeroId"]].rename(
            columns={"Fecha": "fecha", "Cantidad_clean": "valor", "CajeroId": "id"}
        ).dropna(subset=["fecha", "valor", "id"]).sort_values("fecha").reset_index(drop=True)

    def actualizar_modelo(self, df_nuevos: pd.DataFrame) -> None:
        self._raw_source = self._standardize_raw(df_nuevos)
        if self._raw_source.empty:
            return super().actualizar_modelo(self._raw_source)
        clean_current = self._clean_raw_until(self._raw_source["fecha"].max())
        super().actualizar_modelo(clean_current)

    def _new_scratch_model(self, history: pd.DataFrame) -> S4DRModelV102:
        scratch = S4DRModelV102(self.id, *self._scratch_args, **self._scratch_kwargs)
        scratch.actualizar_modelo(history)
        return scratch

    def _clean_target_value(self, target: pd.Timestamp) -> float:
        raw = self._raw_source.rename(columns={"fecha": "Fecha", "valor": "Cantidad", "id": "CajeroId"})
        clean_target = construir_append_real_clean_no_leak(
            raw,
            fechas_target=pd.DatetimeIndex([target]),
            id_mapeado=self.id,
            col_fecha="Fecha",
            col_cajero="CajeroId",
            col_y="Cantidad",
            aplicar_homologacion_ceros=self.causal_preprocess.aplicar_homologacion_ceros,
            aplicar_outlier_clipping=self.causal_preprocess.aplicar_outlier_clipping,
        )
        if clean_target.empty:
            return np.nan
        return float(clean_target["valor"].iloc[0])

    def _build_eval_matrix(self, anchor_date: pd.Timestamp) -> pd.DataFrame:
        """Build selector evidence by reconstructing preprocessing for each pseudo-origin."""
        if self._raw_source.empty:
            return pd.DataFrame()
        anchor = self._to_ts(anchor_date)
        eval_days = int(min(30, max(7, self.eval_weeks * 7)))
        end = anchor - pd.Timedelta(days=1)
        start = end - pd.Timedelta(days=eval_days - 1)
        candidate_dates = pd.date_range(start=start, end=end, freq="D").normalize()
        rows: list[dict[str, Any]] = []
        self._pseudo_origin_audit = []

        for target in candidate_dates:
            history_cutoff = target - pd.Timedelta(days=1)
            history = self._clean_raw_until(history_cutoff)
            if history.empty:
                continue
            scratch = self._new_scratch_model(history)
            row: dict[str, Any] = {
                "fecha": target,
                "y_true": self._clean_target_value(target),
                "weekday": int(target.weekday()),
                "week_of_month": int(self._week_of_month(target)),
            }
            for spec in scratch.models:
                hist_t = scratch._hist_for_spec(spec, target)
                row[f"yhat_{spec.code}"] = float(scratch._predict_model(spec, target, hist_t))
            rows.append(row)
            self._pseudo_origin_audit.append(
                {
                    "pseudo_origin": str(target.date()),
                    "history_cutoff": str(history_cutoff.date()),
                    "training_rows": int(len(history)),
                    "training_hash": _frame_hash(history),
                    "target_value": float(row["y_true"]) if np.isfinite(row["y_true"]) else None,
                    "candidate_hash": sha256(
                        "|".join(f"{key}:{row[key]:.12g}" for key in sorted(row) if key.startswith("yhat_")).encode("utf-8")
                    ).hexdigest(),
                }
            )
        return pd.DataFrame(rows)

    def pseudo_origin_audit(self) -> pd.DataFrame:
        return pd.DataFrame(self._pseudo_origin_audit).copy()
