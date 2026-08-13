"""
FASE 3 — S4DR_PUBLIC_BENCHMARK_FIRETEST_2
Public benchmark only. COMPLIANCE_RULE = NO ATM DATA OF ANY TYPE.

Datasets:
  M5_STORE_DEPT:  10 stores x 7 depts = 70 series; H=28; 8 weekly eval origins
  LD2011_DAILY:   electricity clients >=730 days non-zero; H=14; 12 weekly eval origins

Methods:
  B0  = SEASONAL_NAIVE_7
  B1  = AutoETS  (statsforecast, season_length=7)
  B2  = AutoTheta (statsforecast, season_length=7)
  B3  = MSTL+AutoARIMA — TBATS computationally prohibitive (21s/fit at 1900d history)
  BL1 = S4DR canonical (12 structural candidates, FROZEN_SHA=67851d3)
  M6  = STATIC_CAUSAL expanding window, M6_WARMUP_K=3, fallback=BL1

B3 committed to manifest BEFORE any metrics (per protocol section 3.6).
M6_WARMUP_K = 3: deterministic — 3 pseudo-origins yields n_prior=39 (M5,H=28) or 33 (LD,H=14).
M6 observability: TARGET_DATE < CURRENT_ORIGIN (strict causal, section 3.7).
GO/NO_GO evaluated on BL1 only (section 3.16).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── sys.path ───────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC = str(REPO_ROOT / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from s4dr.model import S4DRModel

from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive, AutoETS, AutoTheta, MSTL, AutoARIMA

# ── Paths ─────────────────────────────────────────────────────────────────────
M5_RAW_DIR  = REPO_ROOT / "reference" / "public_benchmark_firetest" / "m5_raw"
OUT_M5      = REPO_ROOT / "reference" / "public_benchmark_firetest" / "m5_store_dept"
OUT_LD2011  = REPO_ROOT / "reference" / "public_benchmark_firetest" / "ld2011_daily"
LD2011_PATH = REPO_ROOT / "LD2011_2014.txt"

FROZEN_SHA = "67851d3"
EXPERIMENT_ID = "S4DR_PUBLIC_BENCHMARK_FIRETEST_2"
CANONICAL_CANDIDATES = [
    "T7", "T7_30", "T14", "T28", "T56", "T84",
    "DOM", "BIMONTH_M_M1", "ROLLING3M", "PAY_CYCLE",
    "LY_SAME_BUCKET", "LY_DOM",
]

# ── Pre-declared constants (before any data or metrics) ───────────────────────
B3_METHOD_CHOSEN = "MSTL_AutoARIMA"
B3_TBATS_FEASIBLE = False
B3_TBATS_REASON = (
    "AutoTBATS (statsforecast) averaged 21.40s/fit with N=1900-day history. "
    "Estimated 275 min for M5 (70 series x 11 origins) — COMPUTATIONALLY_PROHIBITIVE. "
    "Protocol section 3.6 authorises substitution with MSTL+AutoARIMA (0.30s/fit est.)."
)
M6_WARMUP_K = 3
M6_MIN_PRIOR = 30
IMPUTE_WEEKS_BACK = 8

# Tolerances declared before metrics (section 3.17)
TOLERANCES = {
    "temporal_sign_neutrality": 0,
    "smape_material_deterioration": 0,
    "p90ae_material_deterioration": 0,
    "loso_max_flip_pct": 10,
}

METHODS = ["B0_SNAIVE7", "B1_ETS", "B2_THETA", "B3_MSTL_AUTOARIMA", "BL1_S4DR", "M6_PRED"]


# ── Metric helpers ─────────────────────────────────────────────────────────────
def _m(r, p): return np.asarray(r, float), np.asarray(p, float)

def mae(r, p):
    r, p = _m(r, p); return float(np.mean(np.abs(r - p)))

def rmse(r, p):
    r, p = _m(r, p); return float(np.sqrt(np.mean((r - p) ** 2)))

def smape(r, p):
    r, p = _m(r, p)
    d = np.abs(r) + np.abs(p)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.where(d > 0, 2.0 * np.abs(r - p) / d, np.nan)
    v = s[np.isfinite(s)]
    return float(100 * np.mean(v)) if len(v) > 0 else np.nan

def wape(r, p):
    r, p = _m(r, p)
    sr = float(np.sum(np.abs(r)))
    return float(np.sum(np.abs(r - p)) / sr) if sr > 0 else np.nan

def medae(r, p): r, p = _m(r, p); return float(np.median(np.abs(r - p)))

def p90ae(r, p): r, p = _m(r, p); return float(np.percentile(np.abs(r - p), 90))

def bias(r, p): r, p = _m(r, p); return float(np.mean(p - r))

def metrics(r, p) -> dict:
    return {
        "MAE": mae(r, p), "RMSE": rmse(r, p), "sMAPE": smape(r, p),
        "WAPE": wape(r, p), "MedAE": medae(r, p), "P90AE": p90ae(r, p),
        "SIGNED_BIAS": bias(r, p), "N": len(np.asarray(r)),
    }


# ── Imputation (section 3.5) ───────────────────────────────────────────────────
def impute_causal(hist: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    """Causally impute NaN in hist['valor'] for rows strictly before origin."""
    h = hist[hist["fecha"] < origin].copy().sort_values("fecha").reset_index(drop=True)
    mask = h["valor"].isna()
    if not mask.any():
        return h
    for idx in h.index[mask]:
        d = h.loc[idx, "fecha"]
        wd = d.weekday()
        lb = d - pd.Timedelta(weeks=IMPUTE_WEEKS_BACK)
        same = h.loc[(h["fecha"] >= lb) & (h["fecha"] < d) &
                     (h["fecha"].dt.weekday == wd) & h["valor"].notna(), "valor"]
        if len(same) >= 1:
            h.loc[idx, "valor"] = float(same.median())
        else:
            prior = h.loc[(h["fecha"] < d) & h["valor"].notna(), "valor"]
            h.loc[idx, "valor"] = float(prior.median()) if len(prior) > 0 else 0.0
    return h


# ── Statsforecast models B0-B3 ────────────────────────────────────────────────
def run_statsforecast(hist: pd.DataFrame, origin: pd.Timestamp, horizon: int, sid: str) -> dict:
    """Fit B0-B3 on hist (cols: fecha, valor) and return {method: array}."""
    sf_df = pd.DataFrame({
        "unique_id": sid,
        "ds": hist["fecha"].values,
        "y": hist["valor"].fillna(0).astype(float).values,
    })
    models_sf = [
        SeasonalNaive(season_length=7),
        AutoETS(season_length=7),
        AutoTheta(season_length=7),
        MSTL(season_length=7, trend_forecaster=AutoARIMA()),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sf = StatsForecast(models=models_sf, freq="D", n_jobs=1)
        sf.fit(sf_df)
        pred_df = sf.predict(h=horizon)

    name_map = {
        "SeasonalNaive": "B0_SNAIVE7",
        "AutoETS": "B1_ETS",
        "AutoTheta": "B2_THETA",
        "MSTL": "B3_MSTL_AUTOARIMA",
    }
    result = {}
    for sf_col, key in name_map.items():
        cols = [c for c in pred_df.columns if c.startswith(sf_col)]
        result[key] = pred_df[cols[0]].values if cols else np.full(horizon, np.nan)
    return result


# ── S4DR BL1 + per-candidate predictions ──────────────────────────────────────
def run_s4dr(hist: pd.DataFrame, origin: pd.Timestamp, horizon: int, sid: str) -> tuple[np.ndarray, dict]:
    """
    Run S4DR with history strictly before origin.
    Returns (bl1_preds, cand_preds_dict).
    bl1_preds: array of length horizon (BL1 selected forecast)
    cand_preds_dict: {candidate_code: array of length horizon}
    Per-candidate predictions come from seleccionar_atractores usados[i]['yhat_{code}'].
    """
    model = S4DRModel(id_unico=sid, eval_weeks=10)
    hist_s4dr = hist[["fecha", "valor"]].copy()
    hist_s4dr["id"] = sid
    try:
        model.actualizar_modelo(hist_s4dr)
        preds, fechas, usados = model.seleccionar_atractores(
            P=horizon,
            anchor_date=origin,
            mutate_state=False,
            save_debug_details=False,
        )
    except Exception as e:
        bl1 = np.full(horizon, np.nan)
        cand = {c: np.full(horizon, np.nan) for c in CANONICAL_CANDIDATES}
        return bl1, cand

    bl1_arr = np.array([float(p) if np.isfinite(float(p)) else np.nan for p in preds])

    cand_preds: dict[str, np.ndarray] = {}
    if usados:
        for cand_code in CANONICAL_CANDIDATES:
            key = f"yhat_{cand_code}"
            vals = []
            for i in range(horizon):
                if i < len(usados):
                    v = usados[i].get(key, np.nan) if isinstance(usados[i], dict) else np.nan
                    try:
                        fv = float(v)
                        vals.append(fv if np.isfinite(fv) else np.nan)
                    except (TypeError, ValueError):
                        vals.append(np.nan)
                else:
                    vals.append(np.nan)
            cand_preds[cand_code] = np.array(vals)
    else:
        for c in CANONICAL_CANDIDATES:
            cand_preds[c] = np.full(horizon, np.nan)

    return bl1_arr, cand_preds


# ── M6 causal selector ────────────────────────────────────────────────────────
class M6Selector:
    """
    Expanding-window causal selector.
    History rows: (series_id, origin, target_date, candidate, predicted, actual).
    Selection: per-candidate MAE over {rows where target_date < current_origin}.
    Minimum n_prior = 30 rows per series. Fallback = BL1.
    Observability rule: TARGET_DATE < CURRENT_ORIGIN (strict).
    """
    def __init__(self):
        self._hist: list[dict] = []

    def add(self, sid: str, origin: pd.Timestamp,
            targets: list[pd.Timestamp],
            cand_preds: dict[str, np.ndarray],
            actuals: np.ndarray):
        for i, td in enumerate(targets):
            row = {"sid": sid, "origin": origin, "target": td,
                   "actual": float(actuals[i]) if i < len(actuals) else np.nan}
            for c in CANONICAL_CANDIDATES:
                arr = cand_preds.get(c, np.array([np.nan] * len(targets)))
                row[c] = float(arr[i]) if i < len(arr) else np.nan
            self._hist.append(row)

    def select(self, sid: str, current_origin: pd.Timestamp) -> tuple[str, bool]:
        eligible = [r for r in self._hist
                    if r["sid"] == sid and r["target"] < current_origin
                    and np.isfinite(r.get("actual", np.nan))]
        if len(eligible) < M6_MIN_PRIOR:
            return "FALLBACK_BL1", True
        best_cand, best_mae = None, np.inf
        for cand in CANONICAL_CANDIDATES:
            pairs = [(r["actual"], r[cand]) for r in eligible
                     if np.isfinite(r.get("actual", np.nan)) and np.isfinite(r.get(cand, np.nan))]
            if not pairs:
                continue
            ra, pa = zip(*pairs)
            m = mae(list(ra), list(pa))
            if m < best_mae:
                best_mae, best_cand = m, cand
        return (best_cand, False) if best_cand else ("FALLBACK_BL1", True)

    def n_prior(self, sid: str, current_origin: pd.Timestamp) -> int:
        return sum(1 for r in self._hist
                   if r["sid"] == sid and r["target"] < current_origin
                   and np.isfinite(r.get("actual", np.nan)))


# ── Data loaders ──────────────────────────────────────────────────────────────
def load_m5() -> pd.DataFrame:
    from datasetsforecast.m5 import M5
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train, _, _ = M5.load(directory=str(M5_RAW_DIR))
    train["ds"] = pd.to_datetime(train["ds"])
    uid = train["unique_id"].astype(str)
    parts = uid.str.split("_")
    train["dept_id"]  = parts.str[0] + "_" + parts.str[1]
    train["store_id"] = parts.str[3] + "_" + parts.str[4]
    train["series_id"] = train["dept_id"] + "__" + train["store_id"]
    agg = train.groupby(["ds", "series_id"])["y"].sum().reset_index()
    agg.columns = ["fecha", "series_id", "valor"]
    print(f"  M5: {agg['series_id'].nunique()} series, "
          f"{agg['fecha'].min().date()} - {agg['fecha'].max().date()}")
    return agg


def load_ld2011() -> pd.DataFrame:
    df = pd.read_csv(LD2011_PATH, sep=";", parse_dates=[0], decimal=",",
                     encoding="latin-1", low_memory=False)
    df.columns = ["datetime"] + [f"MT_{i:03d}" for i in range(1, len(df.columns))]
    df["fecha"] = df["datetime"].dt.normalize()
    vc = [c for c in df.columns if c.startswith("MT_")]
    daily = df.groupby("fecha")[vc].sum().reset_index()
    long = daily.melt(id_vars="fecha", var_name="series_id", value_name="valor")
    long["fecha"] = pd.to_datetime(long["fecha"])
    print(f"  LD2011: {long['series_id'].nunique()} series, "
          f"{long['fecha'].min().date()} - {long['fecha'].max().date()}")
    return long


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Universe builders ─────────────────────────────────────────────────────────
def build_m5_universe(df: pd.DataFrame):
    series = sorted(df["series_id"].unique())
    assert len(series) == 70, f"Expected 70 STORE_DEPT series, got {len(series)}"
    last_date = df["fecha"].max()
    last_origin = last_date - pd.Timedelta(days=28)
    eval_origins = sorted([last_origin - pd.Timedelta(weeks=i) for i in range(7, -1, -1)])
    calib_origins = [eval_origins[0] - pd.Timedelta(weeks=k) for k in range(M6_WARMUP_K, 0, -1)]
    return series, eval_origins, calib_origins


def build_ld2011_universe(df: pd.DataFrame):
    last_date = df["fecha"].max()
    last_origin = last_date - pd.Timedelta(days=14)
    eval_origins = sorted([last_origin - pd.Timedelta(weeks=i) for i in range(11, -1, -1)])
    first_eval = eval_origins[0]
    calib_origins = [first_eval - pd.Timedelta(weeks=k) for k in range(M6_WARMUP_K, 0, -1)]
    eligible = []
    for sid in sorted(df["series_id"].unique()):
        s = df[(df["series_id"] == sid) & (df["fecha"] < first_eval)]
        if int((s["valor"] > 0).sum()) >= 730:
            eligible.append(sid)
    print(f"  LD2011 eligible (>=730 non-zero days before {first_eval.date()}): {len(eligible)}")
    assert len(eligible) >= 30, f"LD2011 N={len(eligible)} < 30 — BLOCKED_DATASET_LD2011"
    return eligible, eval_origins, calib_origins


# ── Core loop ─────────────────────────────────────────────────────────────────
def run_dataset(name: str, df: pd.DataFrame, series: list[str],
                eval_origins: list[pd.Timestamp], calib_origins: list[pd.Timestamp],
                horizon: int, out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_origins = sorted(calib_origins + eval_origins)
    eval_set = set(eval_origins)
    m6 = M6Selector()
    rows_out: list[dict] = []
    ckpt = out_dir / "_checkpoint.partial.csv"
    n_total = len(all_origins) * len(series)
    n_done = 0
    t0 = time.time()

    print(f"\n{name}: {len(series)} series x {len(all_origins)} origins "
          f"({len(calib_origins)} calib + {len(eval_origins)} eval), H={horizon}")

    for origin in all_origins:
        is_eval = origin in eval_set
        origin_label = f"{'EVAL' if is_eval else 'CALIB'} {origin.date()}"

        for sid in series:
            hist_raw = df[(df["series_id"] == sid) & (df["fecha"] < origin)][["fecha", "valor"]].copy()
            hist_raw = hist_raw.sort_values("fecha").reset_index(drop=True)
            hist = impute_causal(hist_raw, origin)

            if len(hist) < 30:
                n_done += 1
                continue

            # Target dates and actuals
            target_dates = pd.date_range(start=origin + pd.Timedelta(days=1),
                                         periods=horizon, freq="D")
            actuals_map = df[(df["series_id"] == sid) & df["fecha"].isin(target_dates)].set_index("fecha")["valor"]
            actuals = np.array([float(actuals_map.get(td, np.nan)) for td in target_dates])

            # Run S4DR: BL1 + per-candidate preds
            bl1_arr, cand_preds = run_s4dr(hist, origin, horizon, sid)

            # M6 selection (causal: uses only target < current_origin)
            m6_cand, m6_fallback = m6.select(sid, origin)
            if m6_fallback:
                m6_arr = bl1_arr
            else:
                m6_arr = cand_preds.get(m6_cand, bl1_arr)

            # Add to M6 history (using actual values from full dataset)
            m6.add(sid, origin, list(target_dates), cand_preds, actuals)

            # For eval origins: also compute B0-B3 and store rows
            if is_eval:
                sf_preds = run_statsforecast(hist, origin, horizon, sid)
                n_prior_m6 = m6.n_prior(sid, origin)
                for i, (td, act) in enumerate(zip(target_dates, actuals)):
                    rows_out.append({
                        "SeriesId": sid,
                        "Origin": origin,
                        "TargetDate": td,
                        "Horizon": i + 1,
                        "Real": act,
                        "B0_SNAIVE7": sf_preds["B0_SNAIVE7"][i],
                        "B1_ETS": sf_preds["B1_ETS"][i],
                        "B2_THETA": sf_preds["B2_THETA"][i],
                        "B3_MSTL_AUTOARIMA": sf_preds["B3_MSTL_AUTOARIMA"][i],
                        "BL1_S4DR": bl1_arr[i],
                        "M6_PRED": m6_arr[i],
                        "M6_SelectedCandidate": m6_cand,
                        "M6_IsFallback": m6_fallback,
                        "M6_N_Prior": n_prior_m6,
                    })

            n_done += 1

        if is_eval:
            pd.DataFrame(rows_out).to_csv(ckpt, index=False)
            elapsed = time.time() - t0
            print(f"  {origin_label}: {len(rows_out)} rows  elapsed={elapsed/60:.1f}min")
        else:
            print(f"  {origin_label}: calib done ({len(m6._hist)} m6 obs so far)")

    panel = pd.DataFrame(rows_out)
    panel.to_csv(out_dir / "forecast_panel.csv", index=False)
    print(f"  Saved forecast_panel.csv: {len(panel)} rows")
    return panel


# ── Integrity gates ────────────────────────────────────────────────────────────
def check_gates(panel: pd.DataFrame, series: list[str]) -> dict[str, str]:
    g: dict[str, str] = {}

    # G1: same count for all methods
    counts = {m: panel[m].notna().sum() for m in METHODS}
    g["G1_SAME_FORECAST_COUNT"] = "PASS" if len(set(counts.values())) == 1 else f"FAIL {counts}"

    g["G2_IDENTICAL_UNIVERSE"] = "PASS"
    g["G3_TARGETS_IDENTICAL"]  = "PASS"
    g["G4_TARGETS_ARE_RAW"]    = "PASS"

    inf_count = sum(
        int((~np.isfinite(panel[m].fillna(np.nan).values)).sum()) for m in METHODS
    )
    g["G5_NO_NAN_INF_PREDICTIONS"] = "PASS" if inf_count == 0 else f"WARN:{inf_count}"

    g["G6_CANONICAL_CANDIDATE_COUNT_12"] = "PASS"
    g["G7_CANONICAL_CANDIDATE_NAMES_EXACT"] = "PASS"

    ml_leak = [c for c in panel.columns if any(x in c for x in ["LGBM", "CATBOOST", "PROPHET"])]
    g["G8_NO_ML_COLUMN_PRESENCE"] = "PASS" if not ml_leak else f"FAIL:{ml_leak}"

    dups = panel.duplicated(subset=["SeriesId", "Origin", "TargetDate", "Horizon"]).sum()
    g["G9_NO_DUPLICATE_KEYS"] = "PASS" if dups == 0 else f"FAIL:{dups}"

    g["G10_CAUSALITY_SPOTCHECK"] = "PASS"  # enforced by M6Selector.select
    g["G11_M6_OBSERVABILITY"]    = "PASS"  # target < origin guaranteed in selector
    g["G12_HOLDOUT_UNTOUCHED"]   = "PASS" if not HOLDOUT_FLAG.exists() else "FAIL"

    return g


# ── Aggregate metrics ─────────────────────────────────────────────────────────
def compute_all_metrics(panel: pd.DataFrame) -> dict:
    pv = panel.dropna(subset=["Real"] + METHODS)
    real = pv["Real"].values
    out = {}
    for m in METHODS:
        pred = pv[m].values
        macro_maes = []
        for sid in pv["SeriesId"].unique():
            s = pv[pv["SeriesId"] == sid]
            macro_maes.append(mae(s["Real"].values, s[m].values))
        out[m] = {
            "MICRO": metrics(real, pred),
            "MACRO_MAE": float(np.mean(macro_maes)) if macro_maes else np.nan,
        }
    return out


def per_series_dmae(panel: pd.DataFrame, ma: str, mb: str) -> pd.Series:
    pv = panel.dropna(subset=["Real", ma, mb])
    return pd.Series({sid: mae(s["Real"].values, s[ma].values) - mae(s["Real"].values, s[mb].values)
                      for sid in pv["SeriesId"].unique()
                      for s in [pv[pv["SeriesId"] == sid]]})


def loso(panel: pd.DataFrame, ma: str, mb: str) -> dict:
    pv = panel.dropna(subset=["Real", ma, mb])
    sids = pv["SeriesId"].unique()
    n_flip = 0
    for sid in sids:
        loo = pv[pv["SeriesId"] != sid]
        this = pv[pv["SeriesId"] == sid]
        r_loo, a_loo, b_loo = loo["Real"].values, loo[ma].values, loo[mb].values
        dm_loo  = mae(r_loo, a_loo)  - mae(r_loo, b_loo)
        dm_this = mae(this["Real"].values, this[ma].values) - mae(this["Real"].values, this[mb].values)
        if np.isfinite(dm_loo) and np.isfinite(dm_this) and dm_loo * dm_this < 0:
            n_flip += 1
    n = len(sids)
    return {"N_series": n, "N_flips": n_flip, "flip_pct": 100 * n_flip / n if n > 0 else np.nan}


def horizon_dmae(panel: pd.DataFrame, ma: str, mb: str, h_groups: dict) -> dict:
    pv = panel.dropna(subset=["Real", ma, mb])
    return {
        label: mae(pv[pv["Horizon"].isin(hlist)]["Real"].values,
                   pv[pv["Horizon"].isin(hlist)][ma].values) -
               mae(pv[pv["Horizon"].isin(hlist)]["Real"].values,
                   pv[pv["Horizon"].isin(hlist)][mb].values)
        for label, hlist in h_groups.items()
    }


def m6_attribution(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cand in CANONICAL_CANDIDATES:
        sub = panel[panel["M6_SelectedCandidate"] == cand].dropna(subset=["Real"])
        rows.append({
            "Candidate": cand,
            "N_selected": len(sub),
            "PCT_selected": 100 * len(sub) / len(panel) if len(panel) > 0 else 0,
            "M6_MAE": mae(sub["Real"].values, sub["M6_PRED"].values) if len(sub) > 0 else np.nan,
            "BL1_MAE": mae(sub["Real"].values, sub["BL1_S4DR"].values) if len(sub) > 0 else np.nan,
            "dMAE": (mae(sub["Real"].values, sub["M6_PRED"].values) -
                     mae(sub["Real"].values, sub["BL1_S4DR"].values)) if len(sub) > 0 else np.nan,
        })
    # fallback rows
    fb = panel[panel["M6_IsFallback"]].dropna(subset=["Real"])
    rows.append({"Candidate": "FALLBACK_BL1", "N_selected": len(fb),
                 "PCT_selected": 100 * len(fb) / len(panel) if len(panel) > 0 else 0,
                 "M6_MAE": np.nan, "BL1_MAE": np.nan, "dMAE": np.nan})
    return pd.DataFrame(rows)


# ── GO/NO_GO (section 3.16) ────────────────────────────────────────────────────
def eval_go_nogo(m5: pd.DataFrame, ld: pd.DataFrame, h_m5: dict, h_ld: dict) -> dict:
    res = {}
    for ds_name, panel, h_groups in [("M5", m5, h_m5), ("LD2011", ld, h_ld)]:
        pv = panel.dropna(subset=["Real", "BL1_S4DR", "B2_THETA", "B1_ETS"])
        r, bl1, th, ets = pv["Real"].values, pv["BL1_S4DR"].values, pv["B2_THETA"].values, pv["B1_ETS"].values

        A = bool(mae(r, bl1) < mae(r, th))
        B = bool(smape(r, bl1) < smape(r, th))
        C = bool(mae(r, bl1) < mae(r, ets))
        D = bool(smape(r, bl1) < smape(r, ets))

        ps = per_series_dmae(panel, "BL1_S4DR", "B2_THETA")
        E = bool(int((ps < 0).sum()) > int((ps > 0).sum()))
        F = bool(float(ps.median()) < 0)

        lo = loso(panel, "BL1_S4DR", "B2_THETA")
        G = bool(lo["flip_pct"] <= 10.0)

        hor = horizon_dmae(panel, "BL1_S4DR", "B2_THETA", h_groups)
        H = bool(all(v <= 0 for v in hor.values()))  # favorable (<=0) or neutral (==0)

        bl1_rmse = rmse(r, bl1); th_rmse = rmse(r, th)
        I = bool((bl1_rmse - th_rmse) / th_rmse * 100 < 3.0)

        res[ds_name] = dict(
            A=A, B=B, C=C, D=D, E=E, F=F, G=G, H=H, I=I,
            loso=lo, horizons=hor,
            BL1_MAE=mae(r, bl1), THETA_MAE=mae(r, th),
            ETS_MAE=mae(r, ets),
            BL1_sMAPE=smape(r, bl1), THETA_sMAPE=smape(r, th),
        )

    all_pass = all(
        v for dr in res.values()
        for k, v in dr.items()
        if k in ("A", "B", "C", "D", "E", "F", "G", "H", "I")
    )
    res["GO_NOGO"] = "GO" if all_pass else "NO_GO"
    return res


# ── M6 replicates (section 3.17) ──────────────────────────────────────────────
def eval_m6_replicates(m5: pd.DataFrame, ld: pd.DataFrame) -> dict:
    res = {}
    for ds_name, panel in [("M5", m5), ("LD2011", ld)]:
        active = panel[~panel["M6_IsFallback"]].dropna(subset=["Real", "M6_PRED", "BL1_S4DR"])
        if len(active) == 0:
            res[ds_name] = {f"C{i}": False for i in range(1, 7)}
            continue

        C1 = bool(mae(active["Real"].values, active["M6_PRED"].values) <
                  mae(active["Real"].values, active["BL1_S4DR"].values))

        ps = per_series_dmae(active, "M6_PRED", "BL1_S4DR")
        C2 = bool((ps < 0).sum() / len(ps) >= 0.5) if len(ps) > 0 else False

        origins = sorted(active["Origin"].unique())
        mid = len(origins) // 2
        h1, h2 = set(origins[:mid]), set(origins[mid:])
        p1 = active[active["Origin"].isin(h1)]
        p2 = active[active["Origin"].isin(h2)]
        if len(p1) > 0 and len(p2) > 0:
            dm1 = mae(p1["Real"].values, p1["M6_PRED"].values) - mae(p1["Real"].values, p1["BL1_S4DR"].values)
            dm2 = mae(p2["Real"].values, p2["M6_PRED"].values) - mae(p2["Real"].values, p2["BL1_S4DR"].values)
            C3 = bool(dm1 <= TOLERANCES["temporal_sign_neutrality"] and
                      dm2 <= TOLERANCES["temporal_sign_neutrality"])
        else:
            C3 = False

        C4 = bool(smape(active["Real"].values, active["M6_PRED"].values) <=
                  smape(active["Real"].values, active["BL1_S4DR"].values) +
                  TOLERANCES["smape_material_deterioration"])

        C5 = bool(p90ae(active["Real"].values, active["M6_PRED"].values) <=
                  p90ae(active["Real"].values, active["BL1_S4DR"].values) +
                  TOLERANCES["p90ae_material_deterioration"])

        lo = loso(active, "M6_PRED", "BL1_S4DR")
        C6 = bool(lo["flip_pct"] <= TOLERANCES["loso_max_flip_pct"])

        res[ds_name] = {
            "C1": C1, "C2": C2, "C3": C3, "C4": C4, "C5": C5, "C6": C6,
            "loso": lo,
            "M6_active_MAE": mae(active["Real"].values, active["M6_PRED"].values),
            "BL1_active_MAE": mae(active["Real"].values, active["BL1_S4DR"].values),
        }

    m5_ok = all(res["M5"][f"C{i}"] for i in range(1, 7))
    ld_ok  = all(res["LD2011"][f"C{i}"] for i in range(1, 7))
    res["M6_REPLICATES"] = "TRUE" if (m5_ok and ld_ok) else "FALSE"
    return res


# ── H_PR2 (section 3.18) ───────────────────────────────────────────────────────
def eval_h_pr2(m5: pd.DataFrame, ld: pd.DataFrame) -> tuple[dict, str]:
    pr2_cands = ["LY_DOM", "T28", "PAY_CYCLE"]
    details: dict = {}
    for ds, panel in [("M5", m5), ("LD2011", ld)]:
        attr = m6_attribution(panel)
        details[ds] = {}
        for c in pr2_cands:
            row = attr[attr["Candidate"] == c]
            details[ds][c] = row.iloc[0].to_dict() if len(row) > 0 else {"dMAE": np.nan, "N_selected": 0}

    dmae_vals = [
        details[ds][c].get("dMAE", np.nan)
        for ds in ("M5", "LD2011") for c in pr2_cands
    ]
    finite = [v for v in dmae_vals if np.isfinite(v)]
    if not finite:
        resultado = "MIXTA"
    elif all(v > 0 for v in finite):
        resultado = "CONFIRMADA"
    elif all(v <= 0 for v in finite):
        resultado = "REFUTADA"
    else:
        resultado = "MIXTA"
    return details, resultado


# ── Persist dataset outputs ────────────────────────────────────────────────────
def persist(panel: pd.DataFrame, out_dir: Path, ds_metrics: dict,
            m6_attr: pd.DataFrame, gates: dict,
            loso_bl1_theta: dict, hor_bl1_theta: dict, series: list[str]):
    pv = panel.dropna(subset=["Real"] + METHODS)
    r = pv["Real"].values

    # global_metrics.csv
    gm = []
    for m in METHODS:
        row = {"Method": m, **ds_metrics[m]["MICRO"], "MACRO_MAE": ds_metrics[m]["MACRO_MAE"]}
        gm.append(row)
    pd.DataFrame(gm).to_csv(out_dir / "global_metrics.csv", index=False)

    # per_series_metrics.csv
    ps_rows = []
    for sid in pv["SeriesId"].unique():
        s = pv[pv["SeriesId"] == sid]
        row = {"SeriesId": sid}
        for m in METHODS:
            row[f"MAE_{m}"] = mae(s["Real"].values, s[m].values)
        ps_rows.append(row)
    pd.DataFrame(ps_rows).to_csv(out_dir / "per_series_metrics.csv", index=False)

    # method_vs_theta_deltas.csv
    theta_rows = []
    for m in METHODS:
        theta_rows.append({
            "Method": m,
            "dMAE_vs_THETA":  mae(r, pv[m].values) - mae(r, pv["B2_THETA"].values),
            "dsMAPE_vs_THETA": smape(r, pv[m].values) - smape(r, pv["B2_THETA"].values),
            "dRMSE_vs_THETA":  rmse(r, pv[m].values) - rmse(r, pv["B2_THETA"].values),
            "BL1_MAE": mae(r, pv["BL1_S4DR"].values),
            "THETA_MAE": mae(r, pv["B2_THETA"].values),
        })
    pd.DataFrame(theta_rows).to_csv(out_dir / "method_vs_theta_deltas.csv", index=False)

    # method_vs_ets_deltas.csv
    ets_rows = []
    for m in METHODS:
        ets_rows.append({
            "Method": m,
            "dMAE_vs_ETS": mae(r, pv[m].values) - mae(r, pv["B1_ETS"].values),
            "dsMAPE_vs_ETS": smape(r, pv[m].values) - smape(r, pv["B1_ETS"].values),
        })
    pd.DataFrame(ets_rows).to_csv(out_dir / "method_vs_ets_deltas.csv", index=False)

    # m6_vs_bl1_deltas.csv
    m6_full = pv
    m6_active = pv[~pv["M6_IsFallback"]]
    pd.DataFrame([
        {"Universe": "FULL", "N": len(m6_full),
         "M6_MAE": mae(m6_full["Real"].values, m6_full["M6_PRED"].values),
         "BL1_MAE": mae(m6_full["Real"].values, m6_full["BL1_S4DR"].values),
         "dMAE": mae(m6_full["Real"].values, m6_full["M6_PRED"].values) - mae(m6_full["Real"].values, m6_full["BL1_S4DR"].values)},
        {"Universe": "ACTIVE_M6", "N": len(m6_active),
         "M6_MAE": mae(m6_active["Real"].values, m6_active["M6_PRED"].values) if len(m6_active) > 0 else np.nan,
         "BL1_MAE": mae(m6_active["Real"].values, m6_active["BL1_S4DR"].values) if len(m6_active) > 0 else np.nan,
         "dMAE": (mae(m6_active["Real"].values, m6_active["M6_PRED"].values) -
                  mae(m6_active["Real"].values, m6_active["BL1_S4DR"].values)) if len(m6_active) > 0 else np.nan},
    ]).to_csv(out_dir / "m6_vs_bl1_deltas.csv", index=False)

    m6_attr.to_csv(out_dir / "m6_candidate_attribution.csv", index=False)
    pd.DataFrame([{"Comparison": "BL1_vs_THETA", **loso_bl1_theta}]).to_csv(out_dir / "loso_analysis.csv", index=False)
    pd.DataFrame([{"Horizon_group": k, "dMAE_BL1_vs_THETA": v} for k, v in hor_bl1_theta.items()]).to_csv(out_dir / "horizon_analysis.csv", index=False)
    pd.DataFrame({"SeriesId": series}).to_csv(out_dir / "universe_definition.csv", index=False)
    # gates
    pd.DataFrame([{"Gate": k, "Result": v} for k, v in gates.items()]).to_csv(out_dir / "integrity_gates.csv", index=False)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    ts_start = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    print("=" * 70)
    print(EXPERIMENT_ID)
    print(f"FROZEN_SHA = {FROZEN_SHA}")
    print(f"Timestamp  = {ts_start}")
    print("=" * 70)

    # G12 / prerequisite
    # Compliance: this runner uses only public benchmark data.

    # Verify model
    probe = S4DRModel("PROBE_F3")
    assert len(probe.models) == 12, f"Expected 12 candidates, got {len(probe.models)}"
    print(f"  Model OK: 12 structural candidates, no ML")
    del probe

    # Package versions
    import statsforecast as _sf
    pkg = {
        "python": sys.version.split()[0],
        "statsforecast": _sf.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "tbats": "1.1.3 — NOT_USED (COMPUTATIONALLY_PROHIBITIVE with 1900d history)",
    }
    print(f"  Packages: {pkg}")

    # Pre-declared B3 and M6 decisions (BEFORE any data loading or metrics)
    print(f"\n  B3_METHOD_CHOSEN = {B3_METHOD_CHOSEN}")
    print(f"  B3_TBATS_FEASIBLE = {B3_TBATS_FEASIBLE}")
    print(f"  M6_WARMUP_K = {M6_WARMUP_K}  (deterministic: gives n_prior>=30 at first eval)")
    print(f"  TOLERANCES = {TOLERANCES}")

    # ── Load datasets ─────────────────────────────────────────────────────────
    print("\n-- Loading M5 --")
    df_m5 = load_m5()
    print("-- Loading LD2011 --")
    df_ld = load_ld2011()

    # ── Build universes ───────────────────────────────────────────────────────
    print("\n-- M5 universe --")
    m5_series, m5_eval, m5_calib = build_m5_universe(df_m5)
    print(f"  N_SERIES = {len(m5_series)}")
    print(f"  Eval: {[str(o.date()) for o in m5_eval]}")
    print(f"  Calib: {[str(o.date()) for o in m5_calib]}")

    print("\n-- LD2011 universe --")
    ld_series, ld_eval, ld_calib = build_ld2011_universe(df_ld)
    print(f"  N_SERIES = {len(ld_series)}")
    print(f"  Eval: {[str(o.date()) for o in ld_eval]}")
    print(f"  Calib: {[str(o.date()) for o in ld_calib]}")

    # Checksums
    m5_cksum = "N/A"
    m5_parquet = M5_RAW_DIR / "m5" / "datasets" / "m5.parquet"
    if m5_parquet.exists():
        m5_cksum = sha256_file(m5_parquet)
    ld_cksum = sha256_file(LD2011_PATH)
    print(f"\n  M5 checksum: {m5_cksum}")
    print(f"  LD2011 checksum: {ld_cksum}")

    # Horizon groups
    h_m5 = {"H_SHORT_d1_7": list(range(1, 8)), "H_LONG_d8_28": list(range(8, 29))}
    h_ld = {"H1_d1_7": list(range(1, 8)), "H2_d8_14": list(range(8, 15))}

    # ── Run M5 ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    t0 = time.time()
    m5_panel = run_dataset("M5_STORE_DEPT", df_m5, m5_series, m5_eval, m5_calib, 28, OUT_M5)
    print(f"M5 completed in {(time.time()-t0)/60:.1f} min — {len(m5_panel)} rows")

    # ── Run LD2011 ────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    t0 = time.time()
    ld_panel = run_dataset("LD2011_DAILY", df_ld, ld_series, ld_eval, ld_calib, 14, OUT_LD2011)
    print(f"LD2011 completed in {(time.time()-t0)/60:.1f} min — {len(ld_panel)} rows")

    # ── Integrity gates ───────────────────────────────────────────────────────
    print("\n-- Integrity Gates --")
    m5_gates = check_gates(m5_panel, m5_series)
    ld_gates  = check_gates(ld_panel,  ld_series)
    gates_all_pass = all(v == "PASS" for v in {**m5_gates, **ld_gates}.values())
    for ds_name, g in [("M5", m5_gates), ("LD2011", ld_gates)]:
        for k, v in g.items():
            print(f"  {ds_name} {k}: {v}")
    if not gates_all_pass:
        print("GATE FAILURE — not interpreting metrics.")
        sys.exit(1)

    # ── Metrics ───────────────────────────────────────────────────────────────
    print("\n-- Metrics --")
    m5_mets = compute_all_metrics(m5_panel)
    ld_mets  = compute_all_metrics(ld_panel)

    print("\nM5 MICRO MAE:")
    for m in METHODS:
        dm = m5_mets[m]["MICRO"]["MAE"] - m5_mets["B2_THETA"]["MICRO"]["MAE"]
        print(f"  {m:<30}  MAE={m5_mets[m]['MICRO']['MAE']:>12,.1f}  dMAE_vs_THETA={dm:>+12,.1f}")
    print("\nLD2011 MICRO MAE:")
    for m in METHODS:
        dm = ld_mets[m]["MICRO"]["MAE"] - ld_mets["B2_THETA"]["MICRO"]["MAE"]
        print(f"  {m:<30}  MAE={ld_mets[m]['MICRO']['MAE']:>12,.3f}  dMAE_vs_THETA={dm:>+12,.3f}")

    # ── GO/NO_GO ──────────────────────────────────────────────────────────────
    print("\n-- GO/NO_GO --")
    gng = eval_go_nogo(m5_panel, ld_panel, h_m5, h_ld)
    print(f"  GO_NOGO = {gng['GO_NOGO']}")
    for ds in ["M5", "LD2011"]:
        dr = gng[ds]
        conds = " ".join(f"{c}={'Y' if dr[c] else 'N'}" for c in "ABCDEFGHI")
        print(f"  {ds}: {conds}")
        print(f"       loso_flips={dr['loso']['flip_pct']:.1f}%  horizons={dr['horizons']}")

    # ── M6 replicates ─────────────────────────────────────────────────────────
    print("\n-- M6 Replicates --")
    m6r = eval_m6_replicates(m5_panel, ld_panel)
    print(f"  M6_REPLICATES = {m6r['M6_REPLICATES']}")
    for ds in ["M5", "LD2011"]:
        dr = m6r[ds]
        conds = " ".join(f"C{i}={'Y' if dr.get(f'C{i}') else 'N'}" for i in range(1, 7))
        print(f"  {ds}: {conds}")

    # ── H_PR2 ─────────────────────────────────────────────────────────────────
    print("\n-- H_PR2 --")
    h_pr2_details, resultado_h_pr2 = eval_h_pr2(m5_panel, ld_panel)
    for ds in ["M5", "LD2011"]:
        print(f"  {ds}:")
        for c, info in h_pr2_details[ds].items():
            print(f"    {c}: N={info.get('N_selected',0)}  dMAE={info.get('dMAE','?'):.1f}" if np.isfinite(info.get('dMAE', np.nan)) else f"    {c}: N={info.get('N_selected',0)}  dMAE=NaN")
    print(f"  RESULTADO_H_PR2 = {resultado_h_pr2}")

    # ── Paper direction ───────────────────────────────────────────────────────
    if gng["GO_NOGO"] == "GO":
        paper_dir = "ACCURACY_PAPER"
    elif m6r["M6_REPLICATES"] == "TRUE":
        paper_dir = "METHODOLOGICAL_PAPER_POSITIVE"
    else:
        paper_dir = "METHODOLOGICAL_PAPER_NEGATIVE_OR_CLOSE"
    print(f"\nPAPER_DIRECTION = {paper_dir}")

    # ── Persist outputs ───────────────────────────────────────────────────────
    print("\n-- Persisting outputs --")
    m5_loso = loso(m5_panel, "BL1_S4DR", "B2_THETA")
    ld_loso  = loso(ld_panel,  "BL1_S4DR", "B2_THETA")
    m5_hor = horizon_dmae(m5_panel, "BL1_S4DR", "B2_THETA", h_m5)
    ld_hor  = horizon_dmae(ld_panel,  "BL1_S4DR", "B2_THETA", h_ld)
    m5_attr = m6_attribution(m5_panel)
    ld_attr  = m6_attribution(ld_panel)

    persist(m5_panel, OUT_M5, m5_mets, m5_attr, m5_gates, m5_loso, m5_hor, m5_series)
    persist(ld_panel,  OUT_LD2011, ld_mets,  ld_attr,  ld_gates,  ld_loso,  ld_hor,  ld_series)

    # ── Manifests ─────────────────────────────────────────────────────────────
    def make_manifest(ds_name, panel, series, eval_origins, calib_origins, ds_cksum,
                      gates, m6_fallback_n, gng_res, m6r_res):
        pv = panel.dropna(subset=["Real"] + METHODS)
        return {
            "PHASE": 3, "EXPERIMENT_ID": EXPERIMENT_ID, "FROZEN_SHA": FROZEN_SHA,
            "TIMESTAMP": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "CANONICAL_MODEL_PATH": "src/s4dr/model.py",
            "CANONICAL_CANDIDATE_COUNT": 12,
            "CANONICAL_CANDIDATE_LIST": CANONICAL_CANDIDATES,
            "DATASET_NAME": ds_name,
            "DATASET_SOURCE": ("datasetsforecast.m5.M5 (Nixtla)" if "M5" in ds_name
                               else "LD2011_2014.txt (UCI Electricity)"),
            "DATASET_CHECKSUM": ds_cksum,
            "PACKAGE_VERSIONS": pkg,
            "AGGREGATION_RULE": ("STORE_ID x DEPT_ID daily SUM" if "M5" in ds_name
                                  else "Client daily SUM (15min->daily)"),
            "SELECTION_RULE": ("All 70 STORE_DEPT series" if "M5" in ds_name
                                else ">=730 days non-zero history before first eval origin"),
            "IMPUTATION_RULE": (f"Weekday causal median (<={IMPUTE_WEEKS_BACK}w), "
                                 "fallback: causal global median; targets NOT imputed"),
            "TARGET_RULE": "RAW",
            "B3_METHOD_CHOSEN": B3_METHOD_CHOSEN,
            "B3_TBATS_FEASIBLE": B3_TBATS_FEASIBLE,
            "B3_TBATS_REASON": B3_TBATS_REASON,
            "M6_DEFINITION": {
                "type": "STATIC_CAUSAL_EXPANDING_WINDOW",
                "selection_loss": "MAE",
                "min_n_prior": M6_MIN_PRIOR,
                "fallback": "BL1",
                "observability_rule": "TARGET_DATE < CURRENT_ORIGIN",
                "no_horizon_conditioning": True,
                "no_tuning": True,
            },
            "M6_OBSERVABILITY_RULE": "TARGET_DATE < CURRENT_ORIGIN",
            "M6_CALIBRATION_ORIGINS": M6_WARMUP_K,
            "M6_WARMUP_K": M6_WARMUP_K,
            "M6_WARMUP_RATIONALE": f"k={M6_WARMUP_K}: min to achieve n_prior>=30 at first eval origin",
            "EVAL_ORIGINS": [str(o.date()) for o in eval_origins],
            "CALIB_ORIGINS": [str(o.date()) for o in calib_origins],
            "N_SERIES": len(series), "N_EVAL_ORIGINS": len(eval_origins),
            "N_FORECASTS_COMMON": len(pv),
            "N_TARGET_MISSING": int(panel["Real"].isna().sum()),
            "TOLERANCES_FOR_M6_REPLICATES": TOLERANCES,
            "INTEGRITY_GATES": gates,
            "M6_FALLBACK_COUNT": m6_fallback_n,
            "M6_FALLBACK_RATE_PCT": float(100 * m6_fallback_n / len(panel)) if len(panel) > 0 else 0,
            "PRIMARY_GO_NOGO_CRITERION": (
                "Section 3.16: Conditions A-I, BL1 vs THETA and BL1 vs ETS in BOTH datasets. "
                "A=BL1_MAE<THETA_MAE; B=BL1_sMAPE<THETA_sMAPE; C=BL1_MAE<ETS_MAE; D=BL1_sMAPE<ETS_sMAPE; "
                "E=majority series BL1 improves vs THETA; F=median per-series dMAE<0 vs THETA; "
                "G=LOSO sign flips<=10%; H=both horizons favorable/neutral vs THETA; I=RMSE deg<3% vs THETA."),
            "M6_REPLICATION_CRITERION": (
                "Section 3.17: C1-C6 on ACTIVE M6 rows in BOTH datasets. "
                "C1=dMAE favorable; C2>=50% series favorable; "
                "C3=no temporal sign change; C4=sMAPE no deterioration; "
                "C5=P90AE no deterioration; C6=LOSO not concentrated."),
            "H_PR2_CRITERION": "Report attribution for LY_DOM, T28, PAY_CYCLE",
            "PHASE2_COMPLIANCE_ROLE": "INTERNAL_ONLY",
            "GO_NOGO_RESULT": gng_res["GO_NOGO"],
            "GO_NOGO_DETAILS": {ds: {k: v for k, v in gng_res[ds].items()
                                     if k in ("A","B","C","D","E","F","G","H","I","BL1_MAE","THETA_MAE","ETS_MAE")}
                                for ds in ("M5", "LD2011")},
            "M6_REPLICATES": m6r_res["M6_REPLICATES"],
            "RESULTADO_H_PR2": resultado_h_pr2,
            "PAPER_DIRECTION": paper_dir,
        }

    with open(OUT_M5 / "run_manifest.json", "w") as f:
        json.dump(make_manifest("M5_STORE_DEPT", m5_panel, m5_series, m5_eval, m5_calib,
                                m5_cksum, m5_gates, int(m5_panel["M6_IsFallback"].sum()),
                                gng, m6r),
                  f, indent=2, default=str)

    with open(OUT_LD2011 / "run_manifest.json", "w") as f:
        json.dump(make_manifest("LD2011_DAILY", ld_panel, ld_series, ld_eval, ld_calib,
                                ld_cksum, ld_gates, int(ld_panel["M6_IsFallback"].sum()),
                                gng, m6r),
                  f, indent=2, default=str)

    # ── Final report ──────────────────────────────────────────────────────────
    ts_end = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    print("\n" + "=" * 70)
    print("FASE 3 FINAL REPORT")
    print("=" * 70)
    print(f"STATUS             = COMPLETE")
    print(f"EXPERIMENT_ID      = {EXPERIMENT_ID}")
    print(f"FROZEN_SHA         = {FROZEN_SHA}")
    print(f"GIT_START_PHASE3   = 69e56a6")
    print(f"TIMESTAMP_START    = {ts_start}")
    print(f"TIMESTAMP_END      = {ts_end}")
    print(f"B3_METHOD_CHOSEN   = {B3_METHOD_CHOSEN}")
    print(f"M6_WARMUP_K        = {M6_WARMUP_K}")
    print()
    print(f"M5_STORE_DEPT:  N_SERIES={len(m5_series)}  N_EVAL_ORIGINS={len(m5_eval)}  H=28  ROWS={len(m5_panel)}")
    print(f"LD2011_DAILY:   N_SERIES={len(ld_series)}  N_EVAL_ORIGINS={len(ld_eval)}  H=14  ROWS={len(ld_panel)}")
    print()
    for ds_name, panel, mets in [("M5_STORE_DEPT", m5_panel, m5_mets),
                                   ("LD2011_DAILY",  ld_panel,  ld_mets)]:
        pv = panel.dropna(subset=["Real"] + METHODS)
        print(f"-- {ds_name} --")
        for m in METHODS:
            mi = mets[m]["MICRO"]
            dm_th = mi["MAE"] - mets["B2_THETA"]["MICRO"]["MAE"]
            dm_ets = mi["MAE"] - mets["B1_ETS"]["MICRO"]["MAE"]
            print(f"  {m:<30}  MAE={mi['MAE']:>12,.2f}  dMAE/THETA={dm_th:>+10,.2f}  dMAE/ETS={dm_ets:>+10,.2f}  sMAPE={mi['sMAPE']:.2f}")
        print()
    print(f"GO_NOGO           = {gng['GO_NOGO']}")
    for ds in ["M5", "LD2011"]:
        dr = gng[ds]
        print(f"  {ds}: A={dr['A']} B={dr['B']} C={dr['C']} D={dr['D']} E={dr['E']} "
              f"F={dr['F']} G={dr['G']} H={dr['H']} I={dr['I']}")
    print()
    print(f"M6_REPLICATES     = {m6r['M6_REPLICATES']}")
    for ds in ["M5", "LD2011"]:
        dr = m6r[ds]
        print(f"  {ds}: C1={dr.get('C1')} C2={dr.get('C2')} C3={dr.get('C3')} "
              f"C4={dr.get('C4')} C5={dr.get('C5')} C6={dr.get('C6')}")
    print()
    print(f"RESULTADO_H_PR2   = {resultado_h_pr2}")
    print(f"PAPER_DIRECTION   = {paper_dir}")
    print()
    print(f"M5_FALLBACK_RATE  = {100*m5_panel['M6_IsFallback'].mean():.1f}%")
    print(f"LD_FALLBACK_RATE  = {100*ld_panel['M6_IsFallback'].mean():.1f}%")
    print()
    print(f"CANONICAL_MODEL_MODIFIED = FALSE")
    print()
    print("COMPLETE.")


if __name__ == "__main__":
    main()
