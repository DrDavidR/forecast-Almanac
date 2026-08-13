from dataclasses import dataclass
from typing import Dict, Literal, Optional, List

import numpy as np
import pandas as pd

from .limpieza import (
    BenchmarkCleanConfig,
    aplicar_limpieza_s4dr,
    clasificar_ceros_3clases,
    homologar_ceros,
    infer_perfil_cajero,
    limpiar_outliers_por_cliente_fast,
    preprocess_like_benchmark_from_df,
)


class S4DRDataCleaner:
    """Limpieza S4DR con capa opcional de autoencoder."""

    def __init__(
        self,
        *,
        col_fecha: str = "Fecha",
        col_cajero: str = "CajeroId",
        col_y: str = "Cantidad",
        aplicar_homologacion_ceros: bool = True,
        aplicar_outlier_clipping: bool = True,
        cfg_benchmark=None,
        audit_level: Literal["none", "light", "full"] = "light",
    ) -> None:
        self.col_fecha = col_fecha
        self.col_cajero = col_cajero
        self.col_y = col_y
        self.aplicar_homologacion_ceros = aplicar_homologacion_ceros
        self.aplicar_outlier_clipping = aplicar_outlier_clipping
        self.cfg_benchmark = cfg_benchmark
        self.audit_level = audit_level
        self.reportes_: Dict[str, pd.DataFrame] = {}

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        self._validate_input(df)
        df_clean, reportes = self._run_rule_based_cleaning(df.copy())
        if "Cantidad_clean" not in df_clean.columns:
            df_clean["Cantidad_clean"] = pd.to_numeric(df_clean[self.col_y], errors="coerce")
        self.reportes_ = reportes
        return df_clean, reportes

    def fit_transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        return self.transform(df)

    def _validate_input(self, df: pd.DataFrame) -> None:
        required = [self.col_cajero, self.col_fecha, self.col_y]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Faltan columnas requeridas: {missing}")

    def _run_rule_based_cleaning(self, df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        cfg_benchmark = self.cfg_benchmark or BenchmarkCleanConfig(
            col_fecha=self.col_fecha,
            col_cajero=self.col_cajero,
            col_y=self.col_y,
        )

        df_b, audit_b = preprocess_like_benchmark_from_df(df, cfg_benchmark)
        reportes: Dict[str, pd.DataFrame] = {"audit_benchmark": audit_b}

        df_work = df_b.copy()
        df_work["_Client"] = df_work[self.col_cajero].astype(str)
        df_work["_Date"] = pd.to_datetime(df_work[self.col_fecha], errors="coerce")
        df_work["_Consumption"] = pd.to_numeric(df_work["TotalClean"], errors="coerce").astype(float)

        zeros = clasificar_ceros_3clases(
            df_work[["_Client", "_Date", "_Consumption"]].rename(
                columns={"_Client": "Client", "_Date": "Date", "_Consumption": "Consumption"}
            ),
            col_client="Client",
            col_date="Date",
            col_value="Consumption",
            ventana=9,
            eps_frac_mediana=0.005,
            min_run_zero=3,
            usar_calendar=True,
        ).rename(columns={"Client": "_Client", "Date": "_Date"})

        df_work = df_work.merge(zeros, on=["_Client", "_Date"], how="left")
        df_work["is_zero_legit"] = df_work["is_zero_legit"].fillna(False).astype(bool)

        if self.audit_level == "full":
            reportes["zeros_3clases"] = df_work[
                ["_Client", "_Date", "zero_class", "is_zero_legit", "is_zero_outage", "is_zero_censored", "is_zero_unknown"]
            ].copy()

        perfiles = infer_perfil_cajero(
            df_work[["_Client", "_Date", "_Consumption"]],
            col_client="_Client",
            col_date="_Date",
            col_value="_Consumption",
        )
        reportes["perfiles"] = perfiles.copy() if self.audit_level in ("light", "full") else perfiles.iloc[0:0].copy()

        perfil_map = dict(zip(perfiles["_Client"].astype(str), perfiles["perfil"].astype(str)))
        df_work["_perfil"] = df_work["_Client"].map(perfil_map).fillna("normal")

        params = {
            "stable": dict(ventana=7, eps=0.005, ratio_min=0.20, min_run_low=2, k_mad=2.8, protect_top_k=4),
            "volatile": dict(ventana=11, eps=0.006, ratio_min=0.22, min_run_low=4, k_mad=3.6, protect_top_k=6),
            "payday_heavy": dict(ventana=11, eps=0.004, ratio_min=0.20, min_run_low=3, k_mad=3.5, protect_top_k=6),
            "weekend_heavy": dict(ventana=9, eps=0.005, ratio_min=0.20, min_run_low=3, k_mad=3.2, protect_top_k=5),
            "weekday_heavy": dict(ventana=9, eps=0.005, ratio_min=0.20, min_run_low=3, k_mad=3.2, protect_top_k=5),
            "normal": dict(ventana=9, eps=0.005, ratio_min=0.20, min_run_low=3, k_mad=3.0, protect_top_k=4),
        }

        df_work["was_imputed"] = False
        df_work["Cantidad_clean"] = df_work["_Consumption"].astype(float)

        if self.aplicar_homologacion_ceros:
            hom_frames = []
            for perfil, g in df_work.groupby("_perfil", sort=False):
                p = params.get(str(perfil), params["normal"])
                hom = homologar_ceros(
                    g[["_Client", "_Date", "_Consumption", "is_zero_legit"]],
                    col_client="_Client",
                    col_date="_Date",
                    col_value="_Consumption",
                    ventana=int(p["ventana"]),
                    eps_frac_mediana=float(p["eps"]),
                    ratio_min=float(p["ratio_min"]),
                    min_run_low=int(p["min_run_low"]),
                    usar_calendar=True,
                    protect_col="is_zero_legit",
                )
                hom_frames.append(hom)

            hom_all = pd.concat(hom_frames, ignore_index=True) if hom_frames else pd.DataFrame(columns=["_Client", "_Date", "_Consumption"])
            if self.audit_level == "full":
                reportes["flags_homologacion"] = hom_all.copy()

            df_work = df_work.merge(hom_all, on=["_Client", "_Date"], how="left", suffixes=("", "_hom"))
            if "_Consumption_hom" in df_work.columns:
                df_work["Cantidad_clean"] = pd.to_numeric(df_work["_Consumption_hom"], errors="coerce").astype(float)
            if "was_imputed" in df_work.columns:
                df_work["was_imputed"] = df_work["was_imputed"].fillna(False).astype(bool)

        df_work["was_clipped"] = False
        df_work["is_protected_payday"] = False
        df_work["is_protected_topk"] = False

        if self.aplicar_outlier_clipping:
            clip_frames = []
            for perfil, g in df_work.groupby("_perfil", sort=False):
                p = params.get(str(perfil), params["normal"])
                clip_in = g[["_Client", "_Date", "Cantidad_clean"]].rename(columns={"Cantidad_clean": "_Consumption2"})
                clip = limpiar_outliers_por_cliente_fast(
                    clip_in,
                    col_client="_Client",
                    col_date="_Date",
                    col_value="_Consumption2",
                    protect_top_k_mes=int(p["protect_top_k"]),
                    pay_days="auto",
                    pay_top_m=5,
                    pay_min_uplift=1.30,
                    pay_min_count=2,
                    k_mad=float(p["k_mad"]),
                    floor=0.0,
                    return_bands=False,
                )
                clip_frames.append(clip)

            clip_all = pd.concat(clip_frames, ignore_index=True) if clip_frames else pd.DataFrame(columns=["_Client", "_Date", "_Consumption2"])
            if self.audit_level == "full":
                reportes["flags_outliers"] = clip_all.copy()

            df_work = df_work.merge(
                clip_all.rename(columns={"_Consumption2": "Cantidad_clean_clipped"}),
                on=["_Client", "_Date"],
                how="left",
                suffixes=("", "_clip"),
            )
            if "Cantidad_clean_clipped" in df_work.columns:
                df_work["Cantidad_clean"] = pd.to_numeric(df_work["Cantidad_clean_clipped"], errors="coerce").astype(float)
            for c in ["was_clipped", "is_protected_payday", "is_protected_topk"]:
                if c in df_work.columns:
                    df_work[c] = df_work[c].fillna(False).astype(bool)

        if self.audit_level in ("light", "full"):
            g = df_work.groupby("_Client", sort=False)
            audit_por_cajero = g.agg(min_fecha=("_Date", "min"), max_fecha=("_Date", "max"), n_dias=("_Date", "size")).reset_index()
            audit_por_cajero["perfil"] = audit_por_cajero["_Client"].map(perfil_map).fillna("normal")
            for out_col, src_col in [
                ("pct_missing_hard", "mask_missing_hard"),
                ("pct_zero", "mask_zero"),
                ("pct_zero_legit", "is_zero_legit"),
                ("pct_zero_outage", "is_zero_outage"),
                ("pct_zero_censored", "is_zero_censored"),
                ("pct_imputed", "was_imputed"),
                ("pct_clipped", "was_clipped"),
            ]:
                audit_por_cajero[out_col] = g[src_col].mean().to_numpy(dtype=float) if src_col in df_work.columns else np.nan
            reportes["audit_por_cajero"] = audit_por_cajero

        df_final = df_work.copy()
        df_final[self.col_y] = pd.to_numeric(df_final["Cantidad_clean"], errors="coerce")
        reportes["audit_final"] = pd.DataFrame(
            [
                {
                    "n_rows": int(len(df_final)),
                    "pct_missing_mask": float(df_final["mask_missing"].mean()) if "mask_missing" in df_final.columns and len(df_final) else 0.0,
                    "pct_imputed": float(df_final["was_imputed"].mean()) if "was_imputed" in df_final.columns and len(df_final) else 0.0,
                    "pct_clipped": float(df_final["was_clipped"].mean()) if "was_clipped" in df_final.columns and len(df_final) else 0.0,
                    "audit_level": self.audit_level,
                    "aplicar_homologacion_ceros": bool(self.aplicar_homologacion_ceros),
                    "aplicar_outlier_clipping": bool(self.aplicar_outlier_clipping),
                }
            ]
        )
        return df_final, reportes


def get_cutoff_date_no_leak(
    df_c_raw: pd.DataFrame,
    *,
    col_fecha: str,
    horizonte_total: int,
) -> Optional[pd.Timestamp]:
    if df_c_raw is None or df_c_raw.empty:
        return None
    fechas = pd.to_datetime(df_c_raw[col_fecha], errors="coerce").dt.normalize().dropna()
    if fechas.empty:
        return None
    uniq = pd.Index(fechas.unique()).sort_values()
    h = int(horizonte_total)
    if h > 0 and len(uniq) > h:
        return pd.Timestamp(uniq[-h - 1])
    return pd.Timestamp(uniq.max())


def _build_cleaner(
    *,
    col_fecha: str,
    col_cajero: str,
    col_y: str,
    aplicar_homologacion_ceros: bool,
    aplicar_outlier_clipping: bool,
) -> S4DRDataCleaner:
    return S4DRDataCleaner(
        col_fecha=col_fecha,
        col_cajero=col_cajero,
        col_y=col_y,
        aplicar_homologacion_ceros=aplicar_homologacion_ceros,
        aplicar_outlier_clipping=aplicar_outlier_clipping,
        audit_level="light",
    )


def aplicar_limpieza_s4dr_hasta_cutoff(
    df_c_raw: pd.DataFrame,
    *,
    cutoff_date: pd.Timestamp,
    col_fecha: str = "Fecha",
    col_cajero: str = "CajeroId",
    col_y: str = "Cantidad",
    aplicar_homologacion_ceros: bool = True,
    aplicar_outlier_clipping: bool = True,
    lookback_days: Optional[int] = None,
) -> tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    if df_c_raw is None or df_c_raw.empty:
        return df_c_raw.copy(), {"audit_final": pd.DataFrame([{"n_rows": 0}])}

    d = df_c_raw.copy()
    d[col_fecha] = pd.to_datetime(d[col_fecha], errors="coerce")
    d = d.dropna(subset=[col_fecha]).sort_values(col_fecha)
    cutoff_date = pd.to_datetime(cutoff_date).normalize()
    d = d[d[col_fecha].dt.normalize() <= cutoff_date].copy()
    if lookback_days is not None:
        start = cutoff_date - pd.Timedelta(days=int(lookback_days) - 1)
        d = d[d[col_fecha].dt.normalize() >= start].copy()

    cleaner = _build_cleaner(
        col_fecha=col_fecha,
        col_cajero=col_cajero,
        col_y=col_y,
        aplicar_homologacion_ceros=aplicar_homologacion_ceros,
        aplicar_outlier_clipping=aplicar_outlier_clipping,
    )
    return cleaner.fit_transform(d)


def construir_append_real_clean_no_leak(
    df_c_raw: pd.DataFrame,
    *,
    fechas_target: pd.DatetimeIndex,
    id_mapeado: str,
    col_fecha: str = "Fecha",
    col_cajero: str = "CajeroId",
    col_y: str = "Cantidad",
    aplicar_homologacion_ceros: bool = True,
    aplicar_outlier_clipping: bool = True,
    window_days: int = 180,
) -> pd.DataFrame:
    fechas_target = pd.DatetimeIndex(fechas_target)
    if df_c_raw is None or df_c_raw.empty or len(fechas_target) == 0:
        return pd.DataFrame(columns=["Client", "fecha", "valor"])

    d = df_c_raw.copy()
    d[col_fecha] = pd.to_datetime(d[col_fecha], errors="coerce")
    d = d.dropna(subset=[col_fecha]).sort_values(col_fecha)
    max_target = pd.to_datetime(fechas_target.max()).normalize()
    start = max_target - pd.Timedelta(days=max(1, int(window_days)) - 1)
    d = d[(d[col_fecha].dt.normalize() >= start) & (d[col_fecha].dt.normalize() <= max_target)].copy()
    if d.empty:
        return pd.DataFrame(columns=["Client", "fecha", "valor"])

    cleaner = _build_cleaner(
        col_fecha=col_fecha,
        col_cajero=col_cajero,
        col_y=col_y,
        aplicar_homologacion_ceros=aplicar_homologacion_ceros,
        aplicar_outlier_clipping=aplicar_outlier_clipping,
    )
    df_win_clean, _ = cleaner.fit_transform(d)
    df_win_clean[col_fecha] = pd.to_datetime(df_win_clean[col_fecha], errors="coerce").dt.normalize()
    if "Cantidad_clean" not in df_win_clean.columns:
        df_win_clean["Cantidad_clean"] = pd.to_numeric(df_win_clean[col_y], errors="coerce")
    bloque = df_win_clean[df_win_clean[col_fecha].isin(fechas_target)][[col_fecha, "Cantidad_clean"]].copy()
    if bloque.empty:
        return pd.DataFrame(columns=["Client", "fecha", "valor"])

    out = bloque.rename(columns={col_fecha: "fecha", "Cantidad_clean": "valor"})
    out["Client"] = id_mapeado
    return out[["Client", "fecha", "valor"]]
