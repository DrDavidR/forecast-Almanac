
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Dict, Literal, Tuple
from dataclasses import dataclass
from typing import Literal, Optional, Dict, Any, Iterable
from typing import Literal

try:
    import pandera as pa
    from pandera import Column, DataFrameSchema, Check
except ImportError:
    pa = None


# Constantes de calendario
DOW = ["lun", "mar", "mie", "jue", "vie", "sab", "dom"]  
PAY_DAYS = {1, 14, 15, 30, 31}


@dataclass
class CleanConfig:
    col_fecha: str = "fecha"
    col_cajero: str = "cajero"
    col_y: str = "dispensado"

    dedup_agg: Literal["sum", "max", "mean"] = "sum"
    reindex_daily: bool = True
    outlier_method: Literal["none", "winsor", "mad"] = "winsor"
    bucket: Literal["global", "weekday", "paybucket"] = "weekday"

    winsor_p_low: float = 0.01
    winsor_p_high: float = 0.99
    min_points_per_bucket: int = 12
    mad_k: float = 3.5  


@dataclass
class BenchmarkCleanConfig:

    col_fecha: str = "Fecha"
    col_cajero: str = "CajeroId"
    col_y: str = "Cantidad"
    dedup_agg: Literal["sum", "max", "mean"] = "sum"
    reindex_daily: bool = True
    zero_or_negative_is_missing: bool = True
    totalclean_from_hard_missing: bool = True
    fill_method: Literal["ffill", "bfill", "none"] = "ffill"
    fill_limit_days: Optional[int] = 30   # None => sin límite
    drop_invalid_ids: bool = True
    add_calendar: bool = True



def get_cutoff_date_no_leak(
    df_c_raw: pd.DataFrame,
    *,
    col_fecha: str,
    horizonte_total: int,
) -> Optional[pd.Timestamp]:

    if df_c_raw is None or df_c_raw.empty:
        return None

    fechas = pd.to_datetime(df_c_raw[col_fecha], errors="coerce").dt.normalize()
    fechas = fechas.dropna()
    if fechas.empty:
        return None

    uniq = pd.Index(fechas.unique()).sort_values()
    h = int(horizonte_total)
    if h > 0 and len(uniq) > h:
        return pd.Timestamp(uniq[-h - 1])
    return pd.Timestamp(uniq.max())


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

    df_clean, rep = aplicar_limpieza_s4dr(
        d,
        col_fecha=col_fecha,
        col_cajero=col_cajero,
        col_y=col_y,
        aplicar_homologacion_ceros=aplicar_homologacion_ceros,
        aplicar_outlier_clipping=aplicar_outlier_clipping,
    )
    return df_clean, rep


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

    df_win_clean, _ = aplicar_limpieza_s4dr(
        d,
        col_fecha=col_fecha,
        col_cajero=col_cajero,
        col_y=col_y,
        aplicar_homologacion_ceros=aplicar_homologacion_ceros,
        aplicar_outlier_clipping=aplicar_outlier_clipping,
    )

    df_win_clean[col_fecha] = pd.to_datetime(df_win_clean[col_fecha], errors="coerce").dt.normalize()
    bloque = df_win_clean[df_win_clean[col_fecha].isin(fechas_target)][[col_fecha, "Cantidad_clean"]].copy()
    if bloque.empty:
        return pd.DataFrame(columns=["Client", "fecha", "valor"])

    out = bloque.rename(columns={col_fecha: "fecha", "Cantidad_clean": "valor"})
    out["Client"] = id_mapeado
    return out[["Client", "fecha", "valor"]]


PROFILE = Literal["stable", "volatile", "payday_heavy", "weekend_heavy", "weekday_heavy", "normal"]
ZEROCLASS = Literal["legit", "outage", "censored", "unknown"]

def _baseline_past(
    d: pd.DataFrame,
    *,
    col_client: str,
    col_date: str,
    col_value: str,
    ventana: int = 9,
    usar_calendar: bool = True,
) -> pd.Series:

    x = d.copy()
    x[col_date] = pd.to_datetime(x[col_date], errors="coerce")
    x = x.dropna(subset=[col_date]).sort_values([col_client, col_date]).reset_index(drop=True)

    dt = x[col_date]
    x["_weekday"] = dt.dt.weekday.astype("int64")
    if usar_calendar:
        x["_month"] = dt.dt.month.astype("int64")
        dom = dt.dt.day
        x["_week_of_month"] = ((dom - 1) // 7 + 1).clip(1, 5).astype("int64")

    y = x[col_value].astype(float)
    x["_y_hist"] = y.where(y > 0, np.nan) 

    def _rolling_median_past(keys):
        rolled = (x.groupby(keys)["_y_hist"]
                    .rolling(window=ventana, min_periods=2)
                    .median())
        lvl = list(range(len(keys)))
        rolled = rolled.groupby(level=lvl).shift(1)
        return rolled.reset_index(level=keys, drop=True)

    if usar_calendar:
        b4 = _rolling_median_past([col_client, "_month", "_weekday", "_week_of_month"])
        b3 = _rolling_median_past([col_client, "_month", "_weekday"])
    else:
        b4 = pd.Series(np.nan, index=x.index)
        b3 = pd.Series(np.nan, index=x.index)

    b2 = _rolling_median_past([col_client, "_weekday"])
    b1 = _rolling_median_past([col_client])

    baseline = b4.fillna(b3).fillna(b2).fillna(b1)
    baseline = baseline.reindex(x.index)

    baseline.index = x.index
    return baseline


def clasificar_ceros_3clases(
    d: pd.DataFrame,
    *,
    col_client: str,
    col_date: str,
    col_value: str,
    ventana: int = 9,
    eps_frac_mediana: float = 0.005,
    min_run_zero: int = 3,
    prev_high_mult_med: float = 2.5,
    prev_high_mult_base: float = 2.0,
    usar_calendar: bool = True,
) -> pd.DataFrame:

    x = d.copy()
    x[col_date] = pd.to_datetime(x[col_date], errors="coerce")
    x = x.dropna(subset=[col_date]).sort_values([col_client, col_date]).reset_index(drop=True)

    y = x[col_value].astype(float)
    is_zero = y.notna() & (y == 0)

    pos = x.loc[y > 0, [col_client, col_value]].copy()
    pos[col_value] = pos[col_value].astype(float)
    med_pos = pos.groupby(col_client)[col_value].median() if len(pos) else pd.Series(dtype=float)
    global_med = float(pos[col_value].median()) if len(pos) else 0.0

    med_map = x[col_client].map(med_pos.to_dict()).fillna(global_med)
    eps = (med_map * eps_frac_mediana).clip(lower=1.0)

    baseline = _baseline_past(
        x, col_client=col_client, col_date=col_date, col_value=col_value,
        ventana=ventana, usar_calendar=usar_calendar
    )

    prev0 = is_zero.groupby(x[col_client]).shift(1)
    change0 = (is_zero != prev0).fillna(True).astype("int64")
    run_id0 = change0.groupby(x[col_client]).cumsum()
    run_len0 = is_zero.groupby([x[col_client], run_id0]).transform("size")

    min_legit = np.maximum(2.0 * eps, 0.02 * med_map).astype(float)
    min_outage = np.maximum(5.0 * eps, 0.10 * med_map).astype(float)

    y_prev = y.groupby(x[col_client]).shift(1)
    base_prev = baseline.groupby(x[col_client]).shift(1)

    censored = (
        is_zero
        & baseline.notna()
        & (baseline >= min_outage)
        & y_prev.notna()
        & (
            y_prev >= np.maximum(prev_high_mult_med * med_map, prev_high_mult_base * base_prev.fillna(0.0))
        )
    )

    outage = (
        is_zero
        & baseline.notna()
        & (baseline >= min_outage)
        & (run_len0 >= min_run_zero)
        & (~censored)
    )

    legit = (
        is_zero
        & baseline.notna()
        & (baseline <= min_legit)
        & (~outage)
        & (~censored)
    )

    unknown = is_zero & (~legit) & (~outage) & (~censored)

    zero_class = np.where(legit, "legit",
                  np.where(censored, "censored",
                  np.where(outage, "outage", "unknown")))

    out = x[[col_client, col_date]].copy()
    out["zero_class"] = zero_class
    out["is_zero_legit"] = legit.values
    out["is_zero_outage"] = outage.values
    out["is_zero_censored"] = censored.values
    out["is_zero_unknown"] = unknown.values
    return out


def infer_perfil_cajero(
    d: pd.DataFrame,
    *,
    col_client: str,
    col_date: str,
    col_value: str,
) -> pd.DataFrame:

    x = d.copy()
    x[col_date] = pd.to_datetime(x[col_date], errors="coerce")
    x = x.dropna(subset=[col_date]).sort_values([col_client, col_date]).reset_index(drop=True)
    y = x[col_value].astype(float)

    dt = x[col_date]
    is_pay = dt.dt.day.isin(list(PAY_DAYS))
    is_weekend = dt.dt.weekday.isin([5, 6])

    def _safe_mean(s):
        s = s[np.isfinite(s)]
        return float(s.mean()) if s.size else np.nan

    def _safe_std(s):
        s = s[np.isfinite(s)]
        return float(s.std(ddof=0)) if s.size else np.nan

    rows = []
    for c, g in x.groupby(col_client, sort=False):
        yy = g[col_value].astype(float).to_numpy()
        yy_pos = yy[(yy > 0) & np.isfinite(yy)]
        if yy_pos.size == 0:
            rows.append((str(c), "normal", np.nan, np.nan, np.nan, np.nan, np.nan))
            continue

        med = float(np.median(yy_pos))
        mu = float(np.mean(yy_pos))
        sd = _safe_std(yy_pos)
        cv = (sd / mu) if (mu and np.isfinite(sd)) else np.nan

        gdt = pd.to_datetime(g[col_date], errors="coerce")
        gpay = gdt.dt.day.isin(list(PAY_DAYS)).to_numpy()
        gwe = gdt.dt.weekday.isin([5, 6]).to_numpy()

        mean_pay = _safe_mean(yy_pos[gpay[(yy > 0) & np.isfinite(yy)]]) if np.any(gpay) else np.nan
        mean_nopay = _safe_mean(yy_pos[~gpay[(yy > 0) & np.isfinite(yy)]]) if np.any(~gpay) else np.nan
        payday_ratio = (mean_pay / mean_nopay) if (np.isfinite(mean_pay) and np.isfinite(mean_nopay) and mean_nopay > 0) else np.nan

        mean_we = _safe_mean(yy_pos[gwe[(yy > 0) & np.isfinite(yy)]]) if np.any(gwe) else np.nan
        mean_wd = _safe_mean(yy_pos[~gwe[(yy > 0) & np.isfinite(yy)]]) if np.any(~gwe) else np.nan
        weekend_ratio = (mean_we / mean_wd) if (np.isfinite(mean_we) and np.isfinite(mean_wd) and mean_wd > 0) else np.nan

        perfil: PROFILE = "normal"
        if np.isfinite(payday_ratio) and payday_ratio >= 1.6:
            perfil = "payday_heavy"
        elif np.isfinite(weekend_ratio) and weekend_ratio >= 1.3:
            perfil = "weekend_heavy"
        elif np.isfinite(weekend_ratio) and weekend_ratio <= (1 / 1.3):
            perfil = "weekday_heavy"
        elif np.isfinite(cv) and cv <= 0.6:
            perfil = "stable"
        elif np.isfinite(cv) and cv >= 1.2:
            perfil = "volatile"

        rows.append((str(c), perfil, med, cv, payday_ratio, weekend_ratio, float((g[col_value] == 0).mean())))

    return pd.DataFrame(rows, columns=[
        "_Client", "perfil", "med_pos", "cv_pos", "payday_ratio", "weekend_ratio", "zero_rate"
    ])

def _normalize_date(s: pd.Series) -> pd.Series:
    # force datetime, keep only date (normalize)
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _pay_bucket(date_series: pd.Series) -> pd.Series:
    d = pd.to_datetime(date_series, errors="coerce")
    return np.where(d.dt.day.isin(list(PAY_DAYS)), "pay", "nopay")


def _schema_validate(df: pd.DataFrame, cfg: CleanConfig) -> None:
    """Validación dura (si pandera está instalado) para limpieza legacy."""
    if pa is None:
        return

    schema = DataFrameSchema(
        {
            cfg.col_fecha: Column(pa.DateTime, nullable=False),
            cfg.col_cajero: Column(str, nullable=False),
            cfg.col_y: Column(float, nullable=True, checks=[
                Check.ge(0, ignore_na=True)
            ])
        },
        strict=False
    )
    schema.validate(df, lazy=True)


def _aggregate_duplicates(df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    agg_map = {"sum": "sum", "max": "max", "mean": "mean"}[cfg.dedup_agg]
    out = (
        df.groupby([cfg.col_cajero, cfg.col_fecha], as_index=False)[cfg.col_y]
          .agg(agg_map)
    )
    return out


def _apply_winsor_by_group(df: pd.DataFrame, cfg: CleanConfig, group_col: Optional[str]) -> Dict[str, Any]:
    y = cfg.col_y
    stats: Dict[str, Any] = {"method": "winsor", "group_col": group_col, "capped_count": 0, "groups": {}}

    if group_col is None:
        series = df[y]
        if series.notna().sum() >= cfg.min_points_per_bucket:
            lo = series.quantile(cfg.winsor_p_low)
            hi = series.quantile(cfg.winsor_p_high)
            capped = series.clip(lower=lo, upper=hi)
            stats["capped_count"] = int((capped != series).sum(skipna=True))
            df[y] = capped
            stats["groups"]["global"] = {"lo": float(lo), "hi": float(hi)}
        return stats

    for g, idx in df.groupby(group_col).groups.items():
        series = df.loc[idx, y]
        n = int(series.notna().sum())
        if n < cfg.min_points_per_bucket:
            continue
        lo = series.quantile(cfg.winsor_p_low)
        hi = series.quantile(cfg.winsor_p_high)
        capped = series.clip(lower=lo, upper=hi)
        stats["capped_count"] += int((capped != series).sum(skipna=True))
        df.loc[idx, y] = capped
        stats["groups"][str(g)] = {"n": n, "lo": float(lo), "hi": float(hi)}

    return stats


def _apply_mad_by_group(df: pd.DataFrame, cfg: CleanConfig, group_col: Optional[str]) -> Dict[str, Any]:
    y = cfg.col_y
    stats: Dict[str, Any] = {"method": "mad", "group_col": group_col, "capped_count": 0, "groups": {}}

    def _cap_with_mad(series: pd.Series):
        s = series.dropna()
        if len(s) < cfg.min_points_per_bucket:
            return series, None
        med = s.median()
        mad = np.median(np.abs(s - med))
        robust_sigma = 1.4826 * mad
        if robust_sigma == 0 or np.isnan(robust_sigma):
            return series, None
        lo = med - cfg.mad_k * robust_sigma
        hi = med + cfg.mad_k * robust_sigma
        capped = series.clip(lower=lo, upper=hi)
        meta = {"med": float(med), "mad": float(mad), "lo": float(lo), "hi": float(hi), "n": int(len(s))}
        return capped, meta

    if group_col is None:
        capped, meta = _cap_with_mad(df[y])
        if meta:
            stats["capped_count"] = int((capped != df[y]).sum(skipna=True))
            df[y] = capped
            stats["groups"]["global"] = meta
        return stats

    for g, idx in df.groupby(group_col).groups.items():
        series = df.loc[idx, y]
        capped, meta = _cap_with_mad(series)
        if meta:
            stats["capped_count"] += int((capped != series).sum(skipna=True))
            df.loc[idx, y] = capped
            stats["groups"][str(g)] = meta
    return stats


def clean_dispensed_history(df: pd.DataFrame, cfg: CleanConfig) -> tuple[pd.DataFrame, pd.DataFrame]:

    df = df.copy()

    df[cfg.col_fecha] = _normalize_date(df[cfg.col_fecha])

    df[cfg.col_y] = pd.to_numeric(df[cfg.col_y], errors="coerce")

    before = len(df)
    df = df[df[cfg.col_fecha].notna() & df[cfg.col_cajero].notna()].copy()
    dropped_null_keys = before - len(df)

    neg_count = int((df[cfg.col_y] < 0).sum(skipna=True))
    df.loc[df[cfg.col_y] < 0, cfg.col_y] = np.nan

    _schema_validate(df, cfg)

    dup_rows = int(df.duplicated([cfg.col_cajero, cfg.col_fecha]).sum())
    df = _aggregate_duplicates(df, cfg)

    gap_inserted = 0
    if cfg.reindex_daily:
        out_frames = []
        for caj, g in df.groupby(cfg.col_cajero):
            g = g.sort_values(cfg.col_fecha).set_index(cfg.col_fecha)
            full_idx = pd.date_range(g.index.min(), g.index.max(), freq="D")
            gap_inserted += int(len(full_idx) - len(g))
            g = g.reindex(full_idx)
            g[cfg.col_cajero] = caj
            g.index.name = cfg.col_fecha
            out_frames.append(g.reset_index())
        df = pd.concat(out_frames, ignore_index=True)

    group_col = None
    if cfg.bucket == "weekday":
        df["_bucket"] = pd.to_datetime(df[cfg.col_fecha]).dt.weekday.map(lambda i: DOW[i])
        group_col = "_bucket"
    elif cfg.bucket == "paybucket":
        df["_bucket"] = _pay_bucket(df[cfg.col_fecha])
        group_col = "_bucket"

    outlier_stats: Dict[str, Any] = {"method": "none"}
    if cfg.outlier_method == "winsor":
        outlier_stats = _apply_winsor_by_group(df, cfg, group_col)
    elif cfg.outlier_method == "mad":
        outlier_stats = _apply_mad_by_group(df, cfg, group_col)

    if "_bucket" in df.columns:
        df.drop(columns=["_bucket"], inplace=True)

    audit = {
        "rows_input": before,
        "rows_after_drop_null_keys": before - dropped_null_keys,
        "dropped_null_keys": dropped_null_keys,
        "negatives_to_nan": neg_count,
        "duplicate_rows_detected": dup_rows,
        "gaps_inserted_as_nan": gap_inserted,
        "outlier_method": outlier_stats.get("method", "none"),
        "outlier_group_col": outlier_stats.get("group_col", None),
        "outlier_capped_count": outlier_stats.get("capped_count", 0),
    }
    audit_df = pd.DataFrame([audit])
    return df, audit_df


def homologar_ceros(
    df: pd.DataFrame,
    col_client: str = "Client",
    col_date: str = "Date",
    col_value: str = "Consumption",
    *,
    ventana: int = 9,
    eps_frac_mediana: float = 0.005,
    ratio_min: float = 0.20,
    min_run_low: int = 3,
    usar_calendar: bool = True,
    protect_col: str | None = None,          
) -> pd.DataFrame:

    d = df.copy()
    d[col_date] = pd.to_datetime(d[col_date], errors="coerce")
    d = d.dropna(subset=[col_date]).sort_values([col_client, col_date]).reset_index(drop=True)

    protect = pd.Series(False, index=d.index)
    if protect_col is not None and protect_col in d.columns:
        protect = d[protect_col].fillna(False).astype(bool)

    d["_weekday"] = d[col_date].dt.weekday.astype("int64")
    if usar_calendar:
        d["_month"] = d[col_date].dt.month.astype("int64")
        dom = d[col_date].dt.day
        d["_week_of_month"] = ((dom - 1) // 7 + 1).clip(lower=1, upper=5).astype("int64")

    y = d[col_value].astype(float)

    pos = d.loc[y > 0, [col_client, col_value]].copy()
    pos[col_value] = pos[col_value].astype(float)
    if len(pos):
        med_pos = pos.groupby(col_client)[col_value].median()
        global_med = float(pos[col_value].median())
    else:
        med_pos = pd.Series(dtype=float)
        global_med = 0.0

    eps_map = (med_pos * eps_frac_mediana).to_dict()
    d["_eps"] = d[col_client].map(eps_map).fillna(global_med * eps_frac_mediana).clip(lower=1.0)

    is_low = y.notna() & (y <= d["_eps"])
    is_low = is_low & (~protect) 

    prev = is_low.groupby(d[col_client]).shift(1)
    change = (is_low != prev).fillna(True).astype("int64")
    run_id = change.groupby(d[col_client]).cumsum()
    run_len = is_low.groupby([d[col_client], run_id]).transform("size")
    is_run_low = is_low & (run_len >= min_run_low)
    is_run_low = is_run_low & (~protect)
    d["_y_hist"] = y.mask(is_low, np.nan)

    def _rolling_median_past(group_keys):
        rolled = (d.groupby(group_keys)["_y_hist"]
                    .rolling(window=ventana, min_periods=2)
                    .median())
        lvl = list(range(len(group_keys)))
        rolled = rolled.groupby(level=lvl).shift(1)
        return rolled.reset_index(level=group_keys, drop=True)

    if usar_calendar:
        k4 = [col_client, "_month", "_weekday", "_week_of_month"]
        b4 = _rolling_median_past(k4)
        k3 = [col_client, "_month", "_weekday"]
        b3 = _rolling_median_past(k3)
    else:
        b4 = pd.Series(np.nan, index=d.index)
        b3 = pd.Series(np.nan, index=d.index)

    k2 = [col_client, "_weekday"]
    b2 = _rolling_median_past(k2)

    k1 = [col_client]
    b1 = _rolling_median_past(k1)

    baseline = (
        b4.reindex(d.index)
          .fillna(b3.reindex(d.index))
          .fillna(b2.reindex(d.index))
          .fillna(b1.reindex(d.index))
    )

    is_drop = y.notna() & (~is_low) & baseline.notna() & (y < ratio_min * baseline)
    is_drop = is_drop & (~protect) 
    to_impute = is_low | is_run_low | is_drop
    to_impute = (is_low | is_run_low | is_drop) & (~protect)
    y_new = y.where(~to_impute, baseline)
    y_new = y_new.where(y_new.notna(), y).clip(lower=0.0)

    out = d[[col_client, col_date]].copy()
    out[col_value] = y_new
    out["is_low_like"] = is_low.values
    out["is_run_low"] = is_run_low.values
    out["is_drop_vs_baseline"] = is_drop.values
    out["was_imputed"] = to_impute.values
    return out


def limpiar_outliers_por_cliente_fast(
    df: pd.DataFrame,
    *,
    col_client: str = "Client",
    col_date: str = "Date",
    col_value: str = "Consumption",
    protect_top_k_mes: int = 4,
    pay_days="auto",
    pay_top_m: int = 5,
    pay_min_uplift: float = 1.30,
    pay_min_count: int = 2,
    k_mad: float = 3.0,
    floor: float = 0.0,
    return_bands: bool = False,
) -> pd.DataFrame:

    d = df.copy()
    d[col_date] = pd.to_datetime(d[col_date], errors="coerce")
    d = d.dropna(subset=[col_date]).sort_values([col_client, col_date]).reset_index(drop=True)
    y = d[col_value].astype(float)

    d["_month"] = d[col_date].dt.month.astype("int64")
    d["_weekday"] = d[col_date].dt.weekday.astype("int64")
    d["_dom"] = d[col_date].dt.day.astype("int64")
    d["_week_of_month"] = ((d["_dom"] - 1) // 7 + 1).clip(lower=1, upper=5).astype("int64")
    d["_ym"] = d[col_date].dt.to_period("M")

    # Top-K por (Client, ym)
    rank_desc = y.groupby([d[col_client], d["_ym"]]).rank(method="first", ascending=False)
    mask_topk = rank_desc <= protect_top_k_mes

    # Pay days auto por cliente
    if isinstance(pay_days, str) and pay_days.lower() == "auto":
        tmp = d[[col_client, "_dom"]].copy()
        tmp["y"] = y

        tmp_pos = tmp[tmp["y"] > 0].copy()  # <- clave
        if len(tmp_pos):
            stats_dom = (tmp_pos.groupby([col_client, "_dom"])["y"]
                           .agg(med="median", cnt="count")
                           .reset_index())
            med_cli = tmp_pos.groupby(col_client)["y"].median().rename("med_cli").reset_index()
            stats_dom = stats_dom.merge(med_cli, on=col_client, how="left")
            stats_dom["uplift"] = stats_dom["med"] / stats_dom["med_cli"].replace(0, np.nan)

            cand = stats_dom[(stats_dom["cnt"] >= pay_min_count) & (stats_dom["uplift"] >= pay_min_uplift)].copy()
            cand["rk"] = cand.groupby(col_client)["med"].rank(method="first", ascending=False)
            cand = cand[cand["rk"] <= pay_top_m]

            pay_lookup = set(zip(cand[col_client].astype(str), cand["_dom"].astype(int)))
            mask_pay = pd.Series(
                [(str(c), int(dd)) in pay_lookup for c, dd in zip(d[col_client], d["_dom"])],
                index=d.index
            )
        else:
            mask_pay = pd.Series(False, index=d.index)
    else:
        pay_set = set(int(x) for x in pay_days)
        mask_pay = d["_dom"].isin(pay_set)

    mask_protegido = mask_topk | mask_pay

    keys = [col_client, "_month", "_weekday", "_week_of_month"]
    base = d.loc[~mask_protegido, keys].copy()
    base["y"] = y.loc[~mask_protegido].values

    def _mad(a: np.ndarray) -> float:
        a = a[np.isfinite(a)]
        if a.size == 0:
            return np.nan
        med = np.median(a)
        return np.median(np.abs(a - med))

    def _nanpct(a: np.ndarray, q: float) -> float:
        a = a[np.isfinite(a)]
        if a.size == 0:
            return np.nan
        return float(np.percentile(a, q))

    agg = base.groupby(keys)["y"].agg(
        med="median",
        mad=lambda s: _mad(s.to_numpy()),
        q25=lambda s: _nanpct(s.to_numpy(), 25),
        q75=lambda s: _nanpct(s.to_numpy(), 75),
    ).reset_index()

    agg["sigma"] = 1.4826 * agg["mad"]
    iqr = agg["q75"] - agg["q25"]
    agg["sigma_fallback"] = (iqr / 1.349).replace(0, np.nan)  # aprox sigma normal
    agg["sigma_eff"] = agg["sigma"].where(agg["sigma"] > 0, agg["sigma_fallback"])

    agg["lo"] = (agg["med"] - k_mad * agg["sigma_eff"]).clip(lower=floor)
    agg["hi"] = (agg["med"] + k_mad * agg["sigma_eff"]).clip(lower=floor)

    d[col_client] = d[col_client].astype(str)
    agg[col_client] = agg[col_client].astype(str)

    d = d.merge(
        agg[[col_client, "_month", "_weekday", "_week_of_month", "lo", "hi"]],
        on=keys,
        how="left"
    )

    lo = d["lo"]
    hi = d["hi"]
    valid = lo.notna() & hi.notna() & (~mask_protegido)

    y_clip = y.copy()
    y_clip.loc[valid] = np.minimum(np.maximum(y.loc[valid], lo.loc[valid]), hi.loc[valid])
    y_clip = y_clip.clip(lower=floor)

    out = d[[col_client, col_date]].copy()
    out[col_value] = y_clip
    out["was_clipped"] = ((y_clip != y).values & valid.values)
    out["is_protected_payday"] = mask_pay.values
    out["is_protected_topk"] = mask_topk.values

    if return_bands:
        out["clip_lo"] = lo.values
        out["clip_hi"] = hi.values

    return out



def dedup_por_cliente_fecha(
    df: pd.DataFrame,
    col_cliente: str = "Client",
    col_fecha: str = "Date",
    col_orden_reciente: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:

    if col_cliente not in df.columns or col_fecha not in df.columns:
        raise ValueError(f"Faltan columnas requeridas: '{col_cliente}' y '{col_fecha}'")

    d = df.copy()
    d[col_fecha] = pd.to_datetime(d[col_fecha], errors="coerce")

    base = d.groupby(col_cliente, dropna=False).size().rename("total")

    if col_orden_reciente and col_orden_reciente in d.columns:
        d = d.sort_values([col_cliente, col_fecha, col_orden_reciente],
                          ascending=[True, True, False])
        keep = "first"
    else:
        keep = "last"

    d_clean = d.drop_duplicates(subset=[col_cliente, col_fecha], keep=keep)

    after = d_clean.groupby(col_cliente, dropna=False).size().rename("restantes")
    rep = (pd.concat([base, after], axis=1)
           .assign(duplicados_eliminados=lambda x: x["total"] - x["restantes"])
           .reset_index())

    d_clean = d_clean.sort_values([col_cliente, col_fecha]).reset_index(drop=True)
    return d_clean, rep

def _add_calendar_features(d: pd.DataFrame, fecha_col: str) -> pd.DataFrame:

    dt = pd.to_datetime(d[fecha_col], errors="coerce")

    d["cal_dow"] = dt.dt.weekday.astype("int16")                       # 0..6
    d["cal_dom"] = dt.dt.day.astype("int16")                           # 1..31
    d["cal_month"] = dt.dt.month.astype("int16")                       # 1..12
    d["cal_wom"] = ((dt.dt.day - 1) // 7 + 1).clip(1, 5).astype("int16")
    d["cal_is_payday"] = dt.dt.day.isin(list(PAY_DAYS)).astype("int8")  # 0/1

    base = dt.min()
    d["cal_ordinal"] = (dt - base).dt.days.astype("float64")

    return d


def preprocess_like_benchmark_from_df(
    df: pd.DataFrame,
    cfg: BenchmarkCleanConfig | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:

    cfg = cfg or BenchmarkCleanConfig()
    d = df.copy()

    # 1) Validación básica
    before = len(d)
    for c in (cfg.col_fecha, cfg.col_cajero, cfg.col_y):
        if c not in d.columns:
            raise KeyError(f"Columna requerida faltante: {c}")

    d[cfg.col_fecha] = _normalize_date(d[cfg.col_fecha])

    mask_keys = d[cfg.col_fecha].notna() & d[cfg.col_cajero].notna()
    dropped_null_keys = int((~mask_keys).sum())
    d = d.loc[mask_keys].copy()

    d[cfg.col_cajero] = d[cfg.col_cajero].astype(str).str.strip()
    if cfg.drop_invalid_ids:
        bad = d[cfg.col_cajero].str.lower().isin(["", "nan", "none", "null"])
        d = d.loc[~bad].copy()

    d[cfg.col_y] = pd.to_numeric(d[cfg.col_y], errors="coerce")

    dup_rows = int(d.duplicated([cfg.col_cajero, cfg.col_fecha]).sum())
    ccfg = CleanConfig(
        col_fecha=cfg.col_fecha,
        col_cajero=cfg.col_cajero,
        col_y=cfg.col_y,
        dedup_agg=cfg.dedup_agg,
        reindex_daily=False,
        outlier_method="none",
        bucket="global",
    )
    d = _aggregate_duplicates(d, ccfg)

    gaps_inserted = 0
    if cfg.reindex_daily:
        out_frames = []
        for caj, g in d.groupby(cfg.col_cajero, sort=False):
            g = g.sort_values(cfg.col_fecha).set_index(cfg.col_fecha)
            if g.index.size == 0:
                continue
            full_idx = pd.date_range(g.index.min(), g.index.max(), freq="D")
            gaps_inserted += int(len(full_idx) - len(g))
            g = g.reindex(full_idx)
            g[cfg.col_cajero] = caj
            g.index.name = cfg.col_fecha
            out_frames.append(g.reset_index())
        d = pd.concat(out_frames, ignore_index=True) if out_frames else d

    y = d[cfg.col_y].astype(float)

    mask_missing_hard = y.isna() | (y < 0)
    mask_zero = y.eq(0)

    mask_missing = mask_missing_hard.copy()
    if cfg.zero_or_negative_is_missing:
        mask_missing = mask_missing | (y <= 0)

    d["mask_missing_hard"] = mask_missing_hard.astype(bool)
    d["mask_zero"] = mask_zero.astype(bool)
    d["mask_missing"] = mask_missing.astype(bool)

    m_totalclean = d["mask_missing_hard"] if cfg.totalclean_from_hard_missing else d["mask_missing"]
    d["TotalClean"] = y.where(~m_totalclean, np.nan).astype(float)

    if cfg.fill_method == "none":
        d["TotalFilled"] = d["TotalClean"]
    else:
        d = d.sort_values([cfg.col_cajero, cfg.col_fecha]).reset_index(drop=True)
        limit = cfg.fill_limit_days
        gb = d.groupby(cfg.col_cajero)["TotalClean"]
        if cfg.fill_method == "ffill":
            filled = gb.ffill(limit=limit) if limit is not None else gb.ffill()
        else:
            filled = gb.bfill(limit=limit) if limit is not None else gb.bfill()
        d["TotalFilled"] = filled

    if cfg.add_calendar:
        d = _add_calendar_features(d, cfg.col_fecha)

    audit = {
        "rows_input": int(before),
        "rows_after_drop_null_keys": int(len(d)),
        "dropped_null_keys": int(dropped_null_keys),
        "duplicate_rows_detected": int(dup_rows),
        "gaps_inserted_as_nan": int(gaps_inserted),
        "missing_hard_count": int(d["mask_missing_hard"].sum()),
        "zero_count": int(d["mask_zero"].sum()),
        "missing_count_mask": int(d["mask_missing"].sum()),
        "missing_ratio_mask": float(d["mask_missing"].mean()) if len(d) else 0.0,
    }
    audit_df = pd.DataFrame([audit])
    return d, audit_df


def aplicar_limpieza_s4dr(
    df: pd.DataFrame,
    *,
    col_fecha: str = "Fecha",
    col_cajero: str = "CajeroId",
    col_y: str = "Cantidad",
    aplicar_homologacion_ceros: bool = True,
    aplicar_outlier_clipping: bool = True,
    cfg_benchmark: BenchmarkCleanConfig | None = None,
    audit_level: Literal["none", "light", "full"] = "light",
) -> tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:

    cfg_benchmark = cfg_benchmark or BenchmarkCleanConfig(
        col_fecha=col_fecha,
        col_cajero=col_cajero,
        col_y=col_y
    )

    df_b, audit_b = preprocess_like_benchmark_from_df(df, cfg_benchmark)
    reportes: Dict[str, pd.DataFrame] = {"audit_benchmark": audit_b}

    df_work = df_b.copy()
    df_work["_Client"] = df_work[col_cajero].astype(str)
    df_work["_Date"] = pd.to_datetime(df_work[col_fecha], errors="coerce")
    df_work["_Consumption"] = df_work["TotalClean"].astype(float)

    zeros = clasificar_ceros_3clases(
        df_work[["_Client", "_Date", "_Consumption"]].rename(columns={
            "_Client": "Client", "_Date": "Date", "_Consumption": "Consumption"
        }),
        col_client="Client", col_date="Date", col_value="Consumption",
        ventana=9, eps_frac_mediana=0.005, min_run_zero=3, usar_calendar=True
    ).rename(columns={"Client": "_Client", "Date": "_Date"})

    df_work = df_work.merge(zeros, on=["_Client", "_Date"], how="left")
    df_work["is_zero_legit"] = df_work["is_zero_legit"].fillna(False).astype(bool)

    if audit_level == "full":
        reportes["zeros_3clases"] = df_work[["_Client", "_Date", "zero_class",
                                            "is_zero_legit", "is_zero_outage", "is_zero_censored", "is_zero_unknown"]].copy()

    perfiles = infer_perfil_cajero(
        df_work[["_Client", "_Date", "_Consumption"]],
        col_client="_Client", col_date="_Date", col_value="_Consumption"
    )
    reportes["perfiles"] = perfiles.copy() if audit_level in ("light", "full") else perfiles.iloc[0:0].copy()

    perfil_map = dict(zip(perfiles["_Client"].astype(str), perfiles["perfil"].astype(str)))
    df_work["_perfil"] = df_work["_Client"].map(perfil_map).fillna("normal")

    PARAMS = {
        "stable":       dict(ventana=7,  eps=0.005, ratio_min=0.20, min_run_low=2, k_mad=2.8, protect_top_k=4),
        "volatile":     dict(ventana=11, eps=0.006, ratio_min=0.22, min_run_low=4, k_mad=3.6, protect_top_k=6),
        "payday_heavy": dict(ventana=11, eps=0.004, ratio_min=0.20, min_run_low=3, k_mad=3.5, protect_top_k=6),
        "weekend_heavy":dict(ventana=9,  eps=0.005, ratio_min=0.20, min_run_low=3, k_mad=3.2, protect_top_k=5),
        "weekday_heavy":dict(ventana=9,  eps=0.005, ratio_min=0.20, min_run_low=3, k_mad=3.2, protect_top_k=5),
        "normal":       dict(ventana=9,  eps=0.005, ratio_min=0.20, min_run_low=3, k_mad=3.0, protect_top_k=4),
    }

    df_work["was_imputed"] = False
    df_work["Cantidad_clean"] = df_work["_Consumption"].astype(float)

    if aplicar_homologacion_ceros:
        hom_frames = []
        for perfil, g in df_work.groupby("_perfil", sort=False):
            p = PARAMS.get(str(perfil), PARAMS["normal"])
            hom = homologar_ceros(
                g[["_Client", "_Date", "_Consumption", "is_zero_legit"]],
                col_client="_Client", col_date="_Date", col_value="_Consumption",
                ventana=int(p["ventana"]),
                eps_frac_mediana=float(p["eps"]),
                ratio_min=float(p["ratio_min"]),
                min_run_low=int(p["min_run_low"]),
                usar_calendar=True,
                protect_col="is_zero_legit",   
            )
            hom_frames.append(hom)

        hom_all = pd.concat(hom_frames, ignore_index=True) if hom_frames else pd.DataFrame(columns=["_Client", "_Date", "_Consumption"])

        if audit_level == "full":
            reportes["flags_homologacion"] = hom_all.copy()

        df_work = df_work.merge(
            hom_all,
            on=["_Client", "_Date"],
            how="left",
            suffixes=("", "_hom"),
        )
        if "_Consumption_hom" in df_work.columns:
            df_work["Cantidad_clean"] = df_work["_Consumption_hom"].astype(float)
        if "was_imputed" in df_work.columns:
            df_work["was_imputed"] = df_work["was_imputed"].fillna(False).astype(bool)

    df_work["was_clipped"] = False
    df_work["is_protected_payday"] = False
    df_work["is_protected_topk"] = False

    if aplicar_outlier_clipping:
        clip_frames = []
        for perfil, g in df_work.groupby("_perfil", sort=False):
            p = PARAMS.get(str(perfil), PARAMS["normal"])
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

        if audit_level == "full":
            reportes["flags_outliers"] = clip_all.copy()

        df_work = df_work.merge(
            clip_all.rename(columns={"_Consumption2": "Cantidad_clean_clipped"}),
            on=["_Client", "_Date"],
            how="left",
            suffixes=("", "_clip"),
        )
        if "Cantidad_clean_clipped" in df_work.columns:
            df_work["Cantidad_clean"] = df_work["Cantidad_clean_clipped"].astype(float)

        for c in ["was_clipped", "is_protected_payday", "is_protected_topk"]:
            if c in df_work.columns:
                df_work[c] = df_work[c].fillna(False).astype(bool)

    if audit_level in ("light", "full"):
        g = df_work.groupby("_Client", sort=False)
        audit_por_cajero = g.agg(
            min_fecha=("_Date", "min"),
            max_fecha=("_Date", "max"),
            n_dias=("_Date", "size"),
        ).reset_index()

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

    if audit_level == "none":
        drop_flags = [
            "was_imputed", "was_clipped", "is_protected_payday", "is_protected_topk",
            "is_low_like", "is_run_low", "is_drop_vs_baseline",
            "zero_class", "is_zero_legit", "is_zero_outage", "is_zero_censored", "is_zero_unknown",
            "_perfil",
        ]
        df_final.drop(columns=[c for c in drop_flags if c in df_final.columns], inplace=True, errors="ignore")

    cols_drop = [c for c in df_final.columns if c.startswith("_")]
    df_final.drop(columns=cols_drop, inplace=True, errors="ignore")

    reportes["audit_final"] = pd.DataFrame([{
        "n_rows": int(len(df_final)),
        "pct_missing_mask": float(df_final["mask_missing"].mean()) if "mask_missing" in df_final.columns and len(df_final) else 0.0,
        "pct_imputed": float(df_final["was_imputed"].mean()) if "was_imputed" in df_final.columns and len(df_final) else 0.0,
        "pct_clipped": float(df_final["was_clipped"].mean()) if "was_clipped" in df_final.columns and len(df_final) else 0.0,
        "audit_level": audit_level,
        "aplicar_homologacion_ceros": bool(aplicar_homologacion_ceros),
        "aplicar_outlier_clipping": bool(aplicar_outlier_clipping),
    }])

    return df_final, reportes

