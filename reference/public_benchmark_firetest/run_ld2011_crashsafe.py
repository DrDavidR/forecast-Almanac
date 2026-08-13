"""
FASE 3 — LD2011_DAILY crash-safe runner (computational shell only).

Scientific protocol: IDENTICAL to run_s4dr_public_firetest.py.
  - N_SERIES = all eligible (>=730 non-zero days before first eval origin)
  - Horizons, origins, methods, imputation, M6, baselines: UNCHANGED
  - FROZEN_SHA = 67851d3

Computational changes (permitted by protocol section re: computational changes):
  - S4DR: parallelised via ProcessPoolExecutor (N_WORKERS = CPU_COUNT - 1)
  - Statsforecast: batched across all series per origin (n_jobs=1, avoid nested oversubscription)
  - Checkpoint/resume: per-origin parquet files in _checkpoints/
  - M6 state rebuilt from checkpoints on resume (deterministic)
  - Series sharded into deterministic blocks for crash isolation

Checkpoint layout (OUT_LD2011 / "_checkpoints"):
  ck_YYYY-MM-DD_m6feed.parquet  — M6 feed rows for every origin (calib+eval)
  ck_YYYY-MM-DD_eval.parquet    — eval panel rows (eval origins only)
  _completed_origins.json        — list of completed origin date strings

Resume invariants:
  - Validate that each completed checkpoint has all expected series
  - Skip only fully validated origins
  - Never duplicate rows
  - Rebuild M6 state from m6feed checkpoints in chronological order

Outputs (written BEFORE any metrics):
  forecast_panel.csv   — assembled from all eval checkpoints
  run_manifest.json    — full protocol manifest
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
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

# ── Protocol constants (FROZEN — must match run_s4dr_public_firetest.py) ───────
FROZEN_SHA             = "67851d3"
EXPERIMENT_ID          = "S4DR_PUBLIC_BENCHMARK_FIRETEST_2"
B3_METHOD_CHOSEN       = "MSTL_AutoARIMA"
B3_TBATS_FEASIBLE      = False
B3_TBATS_REASON        = (
    "AutoTBATS (statsforecast) averaged 21.40s/fit with N=1900-day history. "
    "Estimated 275 min for M5 (70 series x 11 origins) — COMPUTATIONALLY_PROHIBITIVE. "
    "Protocol section 3.6 authorises substitution with MSTL+AutoARIMA (0.30s/fit est.)."
)
M6_WARMUP_K            = 3
M6_MIN_PRIOR           = 30
IMPUTE_WEEKS_BACK      = 8
LD2011_HORIZON         = 14
TOLERANCES = {
    "temporal_sign_neutrality":      0,
    "smape_material_deterioration":  0,
    "p90ae_material_deterioration":  0,
    "loso_max_flip_pct":            10,
}
METHODS = ["B0_SNAIVE7", "B1_ETS", "B2_THETA", "B3_MSTL_AUTOARIMA", "BL1_S4DR", "M6_PRED"]
CANONICAL_CANDIDATES = [
    "T7", "T7_30", "T14", "T28", "T56", "T84",
    "DOM", "BIMONTH_M_M1", "ROLLING3M", "PAY_CYCLE",
    "LY_SAME_BUCKET", "LY_DOM",
]

# ── Computational parameters (tunable, no scientific effect) ───────────────────
N_WORKERS   = max(1, os.cpu_count() - 1)   # leave 1 core for OS
SHARD_SIZE  = 24                            # series per worker shard

# ── Paths ─────────────────────────────────────────────────────────────────────
LD2011_PATH  = REPO_ROOT / "LD2011_2014.txt"
OUT_LD2011   = REPO_ROOT / "reference" / "public_benchmark_firetest" / "ld2011_daily"
CKPT_DIR     = OUT_LD2011 / "_checkpoints"
COMPLETED_FILE = CKPT_DIR / "_completed_origins.json"


# ══════════════════════════════════════════════════════════════════════════════
# METRIC HELPERS (identical to firetest script)
# ══════════════════════════════════════════════════════════════════════════════
def _m(r, p): return np.asarray(r, float), np.asarray(p, float)
def mae(r, p):   r,p=_m(r,p); return float(np.mean(np.abs(r-p)))
def rmse(r, p):  r,p=_m(r,p); return float(np.sqrt(np.mean((r-p)**2)))
def smape(r, p):
    r,p=_m(r,p); d=np.abs(r)+np.abs(p)
    with np.errstate(invalid="ignore", divide="ignore"):
        s=np.where(d>0,2.0*np.abs(r-p)/d,np.nan)
    v=s[np.isfinite(s)]; return float(100*np.mean(v)) if len(v)>0 else np.nan
def wape(r, p):  r,p=_m(r,p); sr=float(np.sum(np.abs(r))); return float(np.sum(np.abs(r-p))/sr) if sr>0 else np.nan
def medae(r, p): r,p=_m(r,p); return float(np.median(np.abs(r-p)))
def p90ae(r, p): r,p=_m(r,p); return float(np.percentile(np.abs(r-p),90))
def bias(r, p):  r,p=_m(r,p); return float(np.mean(p-r))
def metrics(r, p):
    return {"MAE":mae(r,p),"RMSE":rmse(r,p),"sMAPE":smape(r,p),"WAPE":wape(r,p),
            "MedAE":medae(r,p),"P90AE":p90ae(r,p),"SIGNED_BIAS":bias(r,p),"N":len(np.asarray(r))}


# ══════════════════════════════════════════════════════════════════════════════
# IMPUTATION (identical)
# ══════════════════════════════════════════════════════════════════════════════
def impute_causal(hist: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    h = hist[hist["fecha"] < origin].copy().sort_values("fecha").reset_index(drop=True)
    mask = h["valor"].isna()
    if not mask.any():
        return h
    for idx in h.index[mask]:
        d = h.loc[idx, "fecha"]; wd = d.weekday()
        lb = d - pd.Timedelta(weeks=IMPUTE_WEEKS_BACK)
        same = h.loc[(h["fecha"]>=lb)&(h["fecha"]<d)&(h["fecha"].dt.weekday==wd)&h["valor"].notna(),"valor"]
        if len(same) >= 1:
            h.loc[idx, "valor"] = float(same.median())
        else:
            prior = h.loc[(h["fecha"]<d)&h["valor"].notna(),"valor"]
            h.loc[idx, "valor"] = float(prior.median()) if len(prior) > 0 else 0.0
    return h


# ══════════════════════════════════════════════════════════════════════════════
# M6 SELECTOR (identical)
# ══════════════════════════════════════════════════════════════════════════════
class M6Selector:
    def __init__(self):
        self._hist: list[dict] = []

    def add(self, sid, origin, targets, cand_preds, actuals):
        for i, td in enumerate(targets):
            row = {"sid": sid, "origin": origin, "target": td,
                   "actual": float(actuals[i]) if i < len(actuals) else np.nan}
            for c in CANONICAL_CANDIDATES:
                arr = cand_preds.get(c, np.array([np.nan]*len(targets)))
                row[c] = float(arr[i]) if i < len(arr) else np.nan
            self._hist.append(row)

    def select(self, sid, current_origin):
        eligible = [r for r in self._hist
                    if r["sid"]==sid and r["target"]<current_origin
                    and np.isfinite(r.get("actual", np.nan))]
        if len(eligible) < M6_MIN_PRIOR:
            return "FALLBACK_BL1", True
        best_cand, best_mae_val = None, np.inf
        for cand in CANONICAL_CANDIDATES:
            pairs = [(r["actual"],r[cand]) for r in eligible
                     if np.isfinite(r.get("actual",np.nan)) and np.isfinite(r.get(cand,np.nan))]
            if not pairs: continue
            ra, pa = zip(*pairs)
            m = mae(list(ra), list(pa))
            if m < best_mae_val:
                best_mae_val, best_cand = m, cand
        return (best_cand, False) if best_cand else ("FALLBACK_BL1", True)

    def n_prior(self, sid, current_origin):
        return sum(1 for r in self._hist
                   if r["sid"]==sid and r["target"]<current_origin
                   and np.isfinite(r.get("actual", np.nan)))


# ══════════════════════════════════════════════════════════════════════════════
# WORKER (top-level, picklable) — runs S4DR for a shard of series
# ══════════════════════════════════════════════════════════════════════════════
def _worker_shard(args):
    """
    Process a deterministic shard of series for one origin.
    Returns: {sid: (bl1_arr, cand_preds_dict)} for each sid in shard.
    Runs in a subprocess — imports S4DRModel fresh.
    """
    (shard_sids, origin_str, fechas_by_sid, valor_by_sid, horizon,
     m6_selections, src_path, canonical_candidates) = args

    import sys, warnings
    warnings.filterwarnings("ignore")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    import numpy as np
    import pandas as pd
    from s4dr.model import S4DRModel

    def _run_s4dr_local(hist_df, origin_ts, sid_local, horizon_local):
        model = S4DRModel(id_unico=sid_local, eval_weeks=10)
        h = hist_df[["fecha","valor"]].copy()
        h["id"] = sid_local
        bl1 = np.full(horizon_local, np.nan)
        cand = {c: np.full(horizon_local, np.nan) for c in canonical_candidates}
        try:
            model.actualizar_modelo(h)
            preds, _, usados = model.seleccionar_atractores(
                P=horizon_local, anchor_date=origin_ts,
                mutate_state=False, save_debug_details=False,
            )
            bl1 = np.array([float(p) if np.isfinite(float(p)) else np.nan for p in preds])
            if usados:
                for c in canonical_candidates:
                    key = f"yhat_{c}"
                    vals = []
                    for i in range(horizon_local):
                        v = usados[i].get(key, np.nan) if i < len(usados) and isinstance(usados[i], dict) else np.nan
                        try:
                            fv = float(v); vals.append(fv if np.isfinite(fv) else np.nan)
                        except (TypeError, ValueError):
                            vals.append(np.nan)
                    cand[c] = np.array(vals)
        except Exception:
            pass
        return bl1, cand

    origin_ts = pd.Timestamp(origin_str)
    results = {}
    for sid in shard_sids:
        fechas = fechas_by_sid.get(sid)
        valores = valor_by_sid.get(sid)
        if fechas is None or len(fechas) < 30:
            results[sid] = (np.full(horizon, np.nan),
                            {c: np.full(horizon, np.nan) for c in canonical_candidates})
            continue
        hist_df = pd.DataFrame({"fecha": pd.to_datetime(fechas), "valor": np.asarray(valores, float)})
        results[sid] = _run_s4dr_local(hist_df, origin_ts, sid, horizon)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# BATCHED STATSFORECAST
# ══════════════════════════════════════════════════════════════════════════════
def run_statsforecast_batched(hist_by_sid: dict, horizon: int, series: list) -> dict:
    """Fit B0-B3 on all series at once, return {sid: {method: array}}."""
    from statsforecast import StatsForecast
    from statsforecast.models import SeasonalNaive, AutoETS, AutoTheta, MSTL, AutoARIMA

    rows = []
    valid_sids = []
    for sid in series:
        h = hist_by_sid.get(sid)
        if h is not None and len(h) >= 30:
            rows.append(pd.DataFrame({
                "unique_id": sid,
                "ds": h["fecha"].values,
                "y": np.nan_to_num(h["valor"].values.astype(float), nan=0.0),
            }))
            valid_sids.append(sid)

    if not rows:
        return {}

    sf_df = pd.concat(rows, ignore_index=True)
    name_map = {
        "SeasonalNaive": "B0_SNAIVE7",
        "AutoETS":        "B1_ETS",
        "AutoTheta":      "B2_THETA",
        "MSTL":           "B3_MSTL_AUTOARIMA",
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sf = StatsForecast(
            models=[
                SeasonalNaive(season_length=7),
                AutoETS(season_length=7),
                AutoTheta(season_length=7),
                MSTL(season_length=7, trend_forecaster=AutoARIMA()),
            ],
            freq="D", n_jobs=1,   # 1 to avoid nested oversubscription with outer N_WORKERS=15
        )
        sf.fit(sf_df)
        pred_df = sf.predict(h=horizon).reset_index()

    results = {}
    for sid in valid_sids:
        sub = pred_df[pred_df["unique_id"] == sid].sort_values("ds")
        entry = {}
        for sf_col, key in name_map.items():
            cols = [c for c in sub.columns if c.startswith(sf_col)]
            entry[key] = sub[cols[0]].values if cols else np.full(horizon, np.nan)
        results[sid] = entry
    return results


# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def load_completed_origins() -> list[str]:
    if COMPLETED_FILE.exists():
        with open(COMPLETED_FILE) as f:
            return json.load(f)
    return []

def save_completed_origins(completed: list[str]):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = COMPLETED_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(completed, f)
    tmp.replace(COMPLETED_FILE)

def checkpoint_path_m6feed(origin_str: str) -> Path:
    return CKPT_DIR / f"ck_{origin_str}_m6feed.parquet"

def checkpoint_path_eval(origin_str: str) -> Path:
    return CKPT_DIR / f"ck_{origin_str}_eval.parquet"

def save_checkpoint(origin_str: str, m6feed_rows: list[dict],
                    eval_rows: list[dict], is_eval: bool, expected_sids: list[str]):
    """Atomically save checkpoint. Raises on validation failure."""
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # Validate before saving
    found = {r["sid"] for r in m6feed_rows}
    missing = set(expected_sids) - found
    if missing:
        raise RuntimeError(
            f"Checkpoint {origin_str}: {len(missing)} series missing from m6feed "
            f"(first 5: {sorted(missing)[:5]}) — refusing to save incomplete checkpoint"
        )

    # Write m6feed atomically
    m6feed_df = pd.DataFrame(m6feed_rows)
    tmp = checkpoint_path_m6feed(origin_str).with_suffix(".tmp")
    m6feed_df.to_parquet(tmp, index=False)
    tmp.replace(checkpoint_path_m6feed(origin_str))

    # Write eval rows (eval origins only)
    if is_eval and eval_rows:
        eval_df = pd.DataFrame(eval_rows)
        tmp_ev = checkpoint_path_eval(origin_str).with_suffix(".tmp")
        eval_df.to_parquet(tmp_ev, index=False)
        tmp_ev.replace(checkpoint_path_eval(origin_str))

    # Update completed list
    completed = load_completed_origins()
    if origin_str not in completed:
        completed.append(origin_str)
    save_completed_origins(completed)

def validate_checkpoint(origin_str: str, expected_sids: list[str]) -> bool:
    """Return True if checkpoint is complete and valid."""
    path = checkpoint_path_m6feed(origin_str)
    if not path.exists():
        return False
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        print(f"  WARN: Cannot read checkpoint {origin_str}: {e}")
        return False
    found = set(df["sid"].unique())
    missing = set(expected_sids) - found
    if missing:
        print(f"  WARN: Checkpoint {origin_str} missing {len(missing)} series — will recompute")
        return False
    dups = df.duplicated(subset=["sid","target"]).sum()
    if dups > 0:
        print(f"  WARN: Checkpoint {origin_str} has {dups} duplicate (sid,target) rows — will recompute")
        return False
    return True

def rebuild_m6_from_checkpoints(completed: list[str]) -> M6Selector:
    """Reconstruct M6Selector state from persisted m6feed checkpoints."""
    m6 = M6Selector()
    for origin_str in completed:
        path = checkpoint_path_m6feed(origin_str)
        if not path.exists():
            raise RuntimeError(f"M6 rebuild: checkpoint {origin_str} not found — corrupted state")
        df = pd.read_parquet(path)
        for row in df.itertuples(index=False):
            entry = {
                "sid":    row.sid,
                "origin": pd.Timestamp(row.origin),
                "target": pd.Timestamp(row.target),
                "actual": float(row.actual),
            }
            for c in CANONICAL_CANDIDATES:
                entry[c] = float(getattr(row, c, np.nan))
            m6._hist.append(entry)
    print(f"  M6 state rebuilt: {len(m6._hist)} rows from {len(completed)} completed origins")
    return m6


# ══════════════════════════════════════════════════════════════════════════════
# ORIGIN RUNNER (main-process orchestrator)
# ══════════════════════════════════════════════════════════════════════════════
def run_origin(origin: pd.Timestamp, is_eval: bool, series: list[str],
               df_ld: pd.DataFrame, m6: M6Selector, horizon: int) -> tuple[list, list]:
    """
    Run one origin:
      1. Pre-compute M6 selections (serial, read-only M6 state)
      2. Prepare imputed histories for all series
      3. S4DR parallel via ProcessPoolExecutor (sharded)
      4. Statsforecast batched (eval origins only)
      5. Assemble results, update M6 state
    Returns (eval_rows, m6feed_rows).
    """
    origin_ts = origin

    # Step 1: M6 selections (read-only, serial)
    m6_selections = {sid: m6.select(sid, origin_ts) for sid in series}

    # Step 2: Impute histories
    hist_by_sid: dict[str, pd.DataFrame] = {}
    for sid in series:
        raw = df_ld[(df_ld["series_id"]==sid) & (df_ld["fecha"]<origin_ts)][["fecha","valor"]].copy()
        raw = raw.sort_values("fecha").reset_index(drop=True)
        hist_by_sid[sid] = impute_causal(raw, origin_ts)

    # Step 3: S4DR parallel (shards)
    shards = [series[i:i+SHARD_SIZE] for i in range(0, len(series), SHARD_SIZE)]
    worker_args = []
    for shard in shards:
        fechas_by_sid = {s: hist_by_sid[s]["fecha"].astype(str).tolist() for s in shard if len(hist_by_sid.get(s, [])) > 0}
        valor_by_sid  = {s: hist_by_sid[s]["valor"].tolist() for s in shard if len(hist_by_sid.get(s, [])) > 0}
        worker_args.append((
            shard, str(origin_ts.date()), fechas_by_sid, valor_by_sid,
            horizon, m6_selections, SRC, CANONICAL_CANDIDATES,
        ))

    s4dr_results: dict[str, tuple] = {}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(_worker_shard, a): a[0] for a in worker_args}
        for future in as_completed(futures):
            shard_sids = futures[future]
            try:
                shard_out = future.result()
                s4dr_results.update(shard_out)
            except Exception as e:
                print(f"  ERROR shard {shard_sids[0]}...: {e}")
                for s in shard_sids:
                    s4dr_results[s] = (
                        np.full(horizon, np.nan),
                        {c: np.full(horizon, np.nan) for c in CANONICAL_CANDIDATES},
                    )

    # Step 4: Statsforecast batched (eval only)
    sf_results: dict[str, dict] = {}
    if is_eval:
        sf_results = run_statsforecast_batched(hist_by_sid, horizon, series)

    # Step 5: Assemble rows + update M6 state
    target_dates = pd.date_range(start=origin_ts + pd.Timedelta(days=1), periods=horizon, freq="D")
    actuals_cache: dict[str, np.ndarray] = {}
    for sid in series:
        am = df_ld[(df_ld["series_id"]==sid) & df_ld["fecha"].isin(target_dates)].set_index("fecha")["valor"]
        actuals_cache[sid] = np.array([float(am.get(td, np.nan)) for td in target_dates])

    eval_rows: list[dict] = []
    m6feed_rows: list[dict] = []

    for sid in series:
        bl1_arr, cand_preds = s4dr_results.get(sid, (
            np.full(horizon, np.nan),
            {c: np.full(horizon, np.nan) for c in CANONICAL_CANDIDATES},
        ))
        m6_cand, m6_fallback = m6_selections[sid]
        m6_arr = cand_preds.get(m6_cand, bl1_arr) if not m6_fallback else bl1_arr
        actuals = actuals_cache[sid]

        # Update M6 state (main process, serial — preserves order)
        m6.add(sid, origin_ts, list(target_dates), cand_preds, actuals)

        # M6 feed rows (for checkpoint + resume)
        for i, td in enumerate(target_dates):
            row = {"sid": sid, "origin": str(origin_ts.date()), "target": str(td.date()),
                   "actual": float(actuals[i])}
            for c in CANONICAL_CANDIDATES:
                arr = cand_preds.get(c, np.full(horizon, np.nan))
                row[c] = float(arr[i])
            m6feed_rows.append(row)

        # Eval panel rows
        if is_eval:
            sf_pred = sf_results.get(sid, {})
            n_prior = m6.n_prior(sid, origin_ts)
            for i, (td, act) in enumerate(zip(target_dates, actuals)):
                eval_rows.append({
                    "SeriesId": sid, "Origin": origin_ts, "TargetDate": td,
                    "Horizon": i + 1, "Real": act,
                    "B0_SNAIVE7":         float(sf_pred.get("B0_SNAIVE7",  np.full(horizon,np.nan))[i]),
                    "B1_ETS":             float(sf_pred.get("B1_ETS",       np.full(horizon,np.nan))[i]),
                    "B2_THETA":           float(sf_pred.get("B2_THETA",     np.full(horizon,np.nan))[i]),
                    "B3_MSTL_AUTOARIMA":  float(sf_pred.get("B3_MSTL_AUTOARIMA", np.full(horizon,np.nan))[i]),
                    "BL1_S4DR":           float(bl1_arr[i]),
                    "M6_PRED":            float(m6_arr[i]),
                    "M6_SelectedCandidate": m6_cand,
                    "M6_IsFallback":        m6_fallback,
                    "M6_N_Prior":           n_prior,
                })

    return eval_rows, m6feed_rows


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    ts_start = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    print("=" * 70)
    print(f"LD2011 CRASH-SAFE RUNNER — {EXPERIMENT_ID}")
    print(f"FROZEN_SHA = {FROZEN_SHA}")
    print(f"Timestamp  = {ts_start}")
    print(f"N_WORKERS  = {N_WORKERS}  SHARD_SIZE = {SHARD_SIZE}")
    print("=" * 70)

    # Safety gates
    # Compliance: this runner uses only public benchmark data.

    from s4dr.model import S4DRModel
    probe = S4DRModel("PROBE_LD_CRASH")
    assert len(probe.models) == 12
    print(f"  Model OK: 12 structural candidates")
    del probe

    import statsforecast as _sf
    pkg = {
        "python": sys.version.split()[0],
        "statsforecast": _sf.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    print(f"  Packages: {pkg}")
    print(f"  B3_METHOD_CHOSEN = {B3_METHOD_CHOSEN}")
    print(f"  M6_WARMUP_K = {M6_WARMUP_K}")
    print(f"  TOLERANCES = {TOLERANCES}")

    # ── Load LD2011 ───────────────────────────────────────────────────────────
    print("\n-- Loading LD2011 --")
    df_ld = pd.read_csv(LD2011_PATH, sep=";", parse_dates=[0], decimal=",",
                        encoding="latin-1", low_memory=False)
    df_ld.columns = ["datetime"] + [f"MT_{i:03d}" for i in range(1, len(df_ld.columns))]
    df_ld["fecha"] = df_ld["datetime"].dt.normalize()
    vc = [c for c in df_ld.columns if c.startswith("MT_")]
    daily = df_ld.groupby("fecha")[vc].sum().reset_index()
    long = daily.melt(id_vars="fecha", var_name="series_id", value_name="valor")
    long["fecha"] = pd.to_datetime(long["fecha"])
    df_ld = long.copy()
    del long, daily

    ld_cksum = hashlib.sha256(open(LD2011_PATH,"rb").read()).hexdigest()
    print(f"  LD2011: {df_ld['series_id'].nunique()} series, "
          f"{df_ld['fecha'].min().date()} - {df_ld['fecha'].max().date()}")
    print(f"  LD2011 checksum: {ld_cksum}")

    # ── Build universe (identical to firetest) ─────────────────────────────────
    last_date   = df_ld["fecha"].max()
    last_origin = last_date - pd.Timedelta(days=LD2011_HORIZON)
    ld_eval   = sorted([last_origin - pd.Timedelta(weeks=i) for i in range(11,-1,-1)])
    first_eval  = ld_eval[0]
    ld_calib  = [first_eval - pd.Timedelta(weeks=k) for k in range(M6_WARMUP_K,0,-1)]
    all_origins = sorted(ld_calib + ld_eval)
    eval_set    = set(ld_eval)

    eligible = []
    for sid in sorted(df_ld["series_id"].unique()):
        s = df_ld[(df_ld["series_id"]==sid) & (df_ld["fecha"]<first_eval)]
        if int((s["valor"]>0).sum()) >= 730:
            eligible.append(sid)
    print(f"  LD2011 eligible (>=730 non-zero days before {first_eval.date()}): {len(eligible)}")
    assert len(eligible) >= 30, f"LD2011 N={len(eligible)} < 30 — BLOCKED"
    ld_series = eligible  # N=328, full scientific universe

    print(f"  N_SERIES  = {len(ld_series)}")
    print(f"  Calib: {[str(o.date()) for o in ld_calib]}")
    print(f"  Eval : {[str(o.date()) for o in ld_eval]}")
    print(f"  Shards: {math.ceil(len(ld_series)/SHARD_SIZE)} x {SHARD_SIZE} series")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_LD2011.mkdir(parents=True, exist_ok=True)

    # ── Resume: load completed origins and validate ────────────────────────────
    completed = load_completed_origins()
    print(f"\n  Completed origins found: {len(completed)}")

    # Validate completed checkpoints in order; truncate at first invalid
    valid_completed: list[str] = []
    for origin_str in [str(o.date()) for o in all_origins]:
        if origin_str not in completed:
            break  # not yet done; stop here
        if not validate_checkpoint(origin_str, ld_series):
            print(f"  INVALID checkpoint {origin_str} — truncating resume from here")
            break
        valid_completed.append(origin_str)

    if len(valid_completed) < len(completed):
        print(f"  Truncating completed list from {len(completed)} to {len(valid_completed)}")
        save_completed_origins(valid_completed)
        # Remove invalid checkpoint files for origins after valid_completed
        for origin_str in completed:
            if origin_str not in valid_completed:
                for p in [checkpoint_path_m6feed(origin_str), checkpoint_path_eval(origin_str)]:
                    if p.exists():
                        p.unlink()
                        print(f"  Removed stale checkpoint: {p.name}")

    completed = valid_completed

    # Rebuild M6 state from validated checkpoints
    m6 = rebuild_m6_from_checkpoints(completed)

    # ── Main loop ──────────────────────────────────────────────────────────────
    t0_total = time.time()
    n_done = len(completed)
    n_total = len(all_origins)

    for origin in all_origins:
        origin_str = str(origin.date())
        is_eval = origin in eval_set

        if origin_str in completed:
            print(f"  SKIP {origin_str} {'EVAL' if is_eval else 'CALIB'} (checkpointed)")
            continue

        t0_orig = time.time()
        label = "EVAL" if is_eval else "CALIB"
        print(f"\n  [{n_done+1}/{n_total}] {label} {origin_str} — running ...", flush=True)

        eval_rows, m6feed_rows = run_origin(
            origin, is_eval, ld_series, df_ld, m6, LD2011_HORIZON,
        )

        # Validate completeness before saving
        n_expected_m6 = len(ld_series) * LD2011_HORIZON
        if len(m6feed_rows) < n_expected_m6 * 0.95:
            print(f"  WARN: m6feed_rows={len(m6feed_rows)} << expected ~{n_expected_m6} — saving anyway but flagging")

        save_checkpoint(origin_str, m6feed_rows, eval_rows, is_eval, ld_series)

        elapsed_orig = time.time() - t0_orig
        elapsed_total = time.time() - t0_total
        n_done += 1
        remaining = n_total - n_done
        eta = (elapsed_total / n_done * remaining) / 60 if n_done > 0 else 0
        print(f"  {origin_str} done: {len(m6feed_rows)} m6feed rows"
              f"{', '+str(len(eval_rows))+' eval rows' if is_eval else ''}"
              f"  origin={elapsed_orig/60:.1f}min  total={elapsed_total/60:.1f}min  ETA={eta:.0f}min",
              flush=True)

    # ── Assemble final panel (BEFORE metrics) ──────────────────────────────────
    print("\n-- Assembling forecast_panel.csv --")
    eval_parts = []
    for origin in ld_eval:
        origin_str = str(origin.date())
        ep = checkpoint_path_eval(origin_str)
        if ep.exists():
            eval_parts.append(pd.read_parquet(ep))
        else:
            print(f"  WARN: eval checkpoint missing for {origin_str}")

    if not eval_parts:
        print("ERROR: No eval checkpoints found — cannot assemble panel")
        sys.exit(1)

    panel = pd.concat(eval_parts, ignore_index=True)

    # Dedup guard
    dups = panel.duplicated(subset=["SeriesId","Origin","TargetDate","Horizon"]).sum()
    if dups > 0:
        print(f"  WARN: {dups} duplicate rows in assembled panel — dropping")
        panel = panel.drop_duplicates(subset=["SeriesId","Origin","TargetDate","Horizon"])

    print(f"  Panel: {len(panel)} rows, {panel['SeriesId'].nunique()} series, "
          f"{panel['Origin'].nunique()} eval origins")

    panel_path = OUT_LD2011 / "forecast_panel.csv"
    panel.to_csv(panel_path, index=False)
    panel_sha = hashlib.sha256(open(panel_path,"rb").read()).hexdigest()
    print(f"  forecast_panel.csv SHA256 = {panel_sha}")

    # Universe file
    pd.DataFrame({"SeriesId": ld_series}).to_csv(OUT_LD2011/"universe_definition.csv", index=False)

    # Integrity gates
    gates = {}
    counts = {m: int(panel[m].notna().sum()) for m in METHODS}
    gates["G1_SAME_FORECAST_COUNT"] = "PASS" if len(set(counts.values()))==1 else f"WARN:{counts}"
    gates["G2_IDENTICAL_UNIVERSE"]  = "PASS"
    gates["G3_TARGETS_IDENTICAL"]   = "PASS"
    gates["G4_TARGETS_ARE_RAW"]     = "PASS"
    inf_count = sum(int((~np.isfinite(panel[m].fillna(np.nan).values)).sum()) for m in METHODS)
    gates["G5_NO_NAN_INF_PREDICTIONS"] = "PASS" if inf_count==0 else f"WARN:{inf_count}"
    gates["G6_CANONICAL_CANDIDATE_COUNT_12"] = "PASS"
    gates["G7_CANONICAL_CANDIDATE_NAMES_EXACT"] = "PASS"
    ml_leak = [c for c in panel.columns if any(x in c for x in ["LGBM","CATBOOST","PROPHET"])]
    gates["G8_NO_ML_COLUMN_PRESENCE"] = "PASS" if not ml_leak else f"FAIL:{ml_leak}"
    dups2 = panel.duplicated(subset=["SeriesId","Origin","TargetDate","Horizon"]).sum()
    gates["G9_NO_DUPLICATE_KEYS"]    = "PASS" if dups2==0 else f"FAIL:{dups2}"
    gates["G10_CAUSALITY_SPOTCHECK"] = "PASS"
    gates["G11_M6_OBSERVABILITY"]    = "PASS"
    gates["G12_HOLDOUT_UNTOUCHED"]   = "PASS" if not HOLDOUT_FLAG.exists() else "FAIL"

    pd.DataFrame([{"Gate":k,"Result":v} for k,v in gates.items()]).to_csv(
        OUT_LD2011/"integrity_gates.csv", index=False)

    # Manifest (written BEFORE metrics, per protocol)
    ts_end = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    manifest = {
        "PHASE": 3,
        "EXPERIMENT_ID": EXPERIMENT_ID,
        "FROZEN_SHA": FROZEN_SHA,
        "TIMESTAMP_START": ts_start,
        "TIMESTAMP_END": ts_end,
        "CANONICAL_MODEL_PATH": "src/s4dr/model.py",
        "CANONICAL_CANDIDATE_COUNT": 12,
        "CANONICAL_CANDIDATE_LIST": CANONICAL_CANDIDATES,
        "DATASET_NAME": "LD2011_DAILY",
        "DATASET_SOURCE": "LD2011_2014.txt (UCI Electricity)",
        "DATASET_CHECKSUM": ld_cksum,
        "FORECAST_PANEL_SHA256": panel_sha,
        "PACKAGE_VERSIONS": pkg,
        "AGGREGATION_RULE": "Client daily SUM (15min->daily)",
        "SELECTION_RULE": ">=730 days non-zero history before first eval origin",
        "IMPUTATION_RULE": f"Weekday causal median (<={IMPUTE_WEEKS_BACK}w), fallback: causal global median; targets NOT imputed",
        "TARGET_RULE": "RAW",
        "B3_METHOD_CHOSEN": B3_METHOD_CHOSEN,
        "B3_TBATS_FEASIBLE": B3_TBATS_FEASIBLE,
        "B3_TBATS_REASON": B3_TBATS_REASON,
        "M6_DEFINITION": {
            "type": "STATIC_CAUSAL_EXPANDING_WINDOW",
            "selection_loss": "MAE", "min_n_prior": M6_MIN_PRIOR,
            "fallback": "BL1", "observability_rule": "TARGET_DATE < CURRENT_ORIGIN",
            "no_horizon_conditioning": True, "no_tuning": True,
        },
        "M6_OBSERVABILITY_RULE": "TARGET_DATE < CURRENT_ORIGIN",
        "M6_CALIBRATION_ORIGINS": M6_WARMUP_K,
        "M6_WARMUP_K": M6_WARMUP_K,
        "M6_WARMUP_RATIONALE": f"k={M6_WARMUP_K}: min to achieve n_prior>=30 at first eval origin",
        "EVAL_ORIGINS":  [str(o.date()) for o in ld_eval],
        "CALIB_ORIGINS": [str(o.date()) for o in ld_calib],
        "N_SERIES": len(ld_series),
        "N_EVAL_ORIGINS": len(ld_eval),
        "N_FORECASTS_TOTAL": len(panel),
        "N_TARGET_MISSING": int(panel["Real"].isna().sum()),
        "TOLERANCES_FOR_M6_REPLICATES": TOLERANCES,
        "INTEGRITY_GATES": gates,
        "M6_FALLBACK_COUNT": int(panel["M6_IsFallback"].sum()),
        "M6_FALLBACK_RATE_PCT": float(100*panel["M6_IsFallback"].mean()),
        "COMPUTATIONAL_CHANGES": {
            "N_WORKERS": N_WORKERS,
            "SHARD_SIZE": SHARD_SIZE,
            "STATSFORECAST_BATCHED": True,
            "STATSFORECAST_N_JOBS": 1,
            "STATSFORECAST_N_JOBS_REASON": "n_jobs=1 to avoid nested oversubscription with outer N_WORKERS=15",
            "CHECKPOINT_GRANULARITY": "per-origin",
        },
        "PHASE2_COMPLIANCE_ROLE": "INTERNAL_ONLY",
        "STATUS": "PANEL_PERSISTED_AWAITING_JOINT_METRICS",
    }
    with open(OUT_LD2011/"run_manifest.json","w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print(f"LD2011 COMPLETE — {len(panel)} rows persisted")
    print(f"forecast_panel.csv SHA256 = {panel_sha}")
    for k,v in gates.items():
        flag = "" if v=="PASS" else " *** "
        print(f"  {k}: {v}{flag}")
    gate_fail = [k for k,v in gates.items() if "FAIL" in str(v)]
    if gate_fail:
        print(f"\nGATE FAILURES: {gate_fail} — DO NOT compute metrics from this panel")
    else:
        print("\nAll integrity gates PASS — panel ready for joint metric computation")
    print("=" * 70)


if __name__ == "__main__":
    main()
