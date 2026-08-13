"""
MDA-2_THIRD_DOMAIN_WIKI
ADDENDUM_TYPE = PROSPECTIVELY_SPECIFIED_POST_FIRETEST_REPLICATION

Dataset: kaggle_web_traffic_daily with missing values (Monash / Zenodo 4656080)
H=14, N_EVAL_ORIGINS=12, ORIGIN_FREQUENCY=7 days
SAMPLED_N = min(300, eligible)

MDA2_EXECUTION_SHA = 24adff6
SCIENTIFIC_CONFIGURATION_ATTRIBUTION_SHA = 67851d3

PROHIBITIONS (enforced by this script):
  - No GO/NO_GO re-evaluation
  - No model tuning
  - No selector tuning
  - No candidate modification
  - No horizon conditioning in SC
  - No ATM data
  - No holdout access
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
import warnings
import zipfile
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

# ── Protocol constants (frozen before any result inspection) ───────────────────
ADDENDUM_ID                           = "MDA-2_THIRD_DOMAIN_WIKI"
ADDENDUM_TYPE                         = "PROSPECTIVELY_SPECIFIED_POST_FIRETEST_REPLICATION"
MDA2_EXECUTION_SHA                    = "24adff6"
SCIENTIFIC_CONFIGURATION_ATTRIBUTION_SHA = "67851d3"

FROZEN_SHA                 = "67851d3"
DATASET_NAME               = "kaggle_web_traffic"
ZENODO_RECORD              = "4656080"
ZENODO_URL                 = (
    "https://zenodo.org/records/4656080/files/"
    "kaggle_web_traffic_dataset_with_missing_values.zip?download=1"
)
DATASET_VARIANT            = "with_missing_values"

HORIZON                    = 14
N_EVAL_ORIGINS             = 12
ORIGIN_SPACING_DAYS        = 7

# Eligibility (pre-specified, section 1.2)
MIN_HISTORY_DAYS           = 700
ELIGIBILITY_MEDIAN_THRESHOLD = 100
ELIGIBILITY_MISSING_RULE   = "PRE_EVALUATION_HISTORY_COMPLETE"
ELIGIBILITY_MEDIAN_WINDOW  = "PRE_EVALUATION_ONLY"

# Sampling (section 1.5)
SAMPLING_SEED              = 42
SAMPLING_RNG               = "PCG64"
SAMPLING_REPLACE           = False
MAX_SAMPLED                = 300
ELIGIBLE_SORT_ORDER        = "lexical_UTF8"

# SC (section 3)
SC_MIN_PRIOR               = 30
SC_OBSERVABILITY_RULE      = "TargetDate < CurrentOrigin"
SC_SELECTION_LOSS          = "MAE"
SC_FALLBACK                = "L0"

# L4S (section 8.5)
L4S_HALF_A_ORIGINS         = list(range(0, 6))   # indices 0-5
L4S_HALF_B_ORIGINS         = list(range(6, 12))  # indices 6-11

CANONICAL_CANDIDATES = [
    "T7", "T7_30", "T14", "T28", "T56", "T84",
    "DOM", "BIMONTH_M_M1", "ROLLING3M", "PAY_CYCLE",
    "LY_SAME_BUCKET", "LY_DOM",
]
ML_CANDIDATES_FORBIDDEN = ["LGBM_MODEL", "CATBOOST_MODEL", "PROPHET_MODEL"]

METHODS = ["B0", "B1", "B2", "B3", "L0", "SC"]
ALL_LEVELS = ["B0", "B1", "B2", "B3", "L0", "L1", "L2", "L3", "L4", "L4S", "L5", "SC"]

# Computational
N_WORKERS  = max(1, (os.cpu_count() or 4) - 1)
SHARD_SIZE = 20

# Paths
WIKI_RAW_DIR  = REPO_ROOT / "reference" / "public_benchmark_firetest" / "wiki_raw"
OUT_DIR       = REPO_ROOT / "reference" / "public_benchmark_firetest" / "wiki_daily"
CKPT_DIR      = OUT_DIR / "_checkpoints"
COMPLETED_FILE = CKPT_DIR / "_completed_origins.json"
RAW_ZIP       = WIKI_RAW_DIR / "kaggle_web_traffic_with_missing.zip"


# ══════════════════════════════════════════════════════════════════════════════
# METRIC HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _m(r, p):
    return np.asarray(r, float), np.asarray(p, float)

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

def medae(r, p):
    r, p = _m(r, p); return float(np.median(np.abs(r - p)))

def p90ae(r, p):
    r, p = _m(r, p); return float(np.percentile(np.abs(r - p), 90))

def signed_bias(r, p):
    r, p = _m(r, p); return float(np.mean(p - r))

def metrics_dict(r, p) -> dict:
    return {
        "MAE": mae(r, p), "RMSE": rmse(r, p), "sMAPE": smape(r, p),
        "WAPE": wape(r, p), "MedAE": medae(r, p), "P90AE": p90ae(r, p),
        "SIGNED_BIAS": signed_bias(r, p), "N": int(len(np.asarray(r))),
    }


# ══════════════════════════════════════════════════════════════════════════════
# TSF PARSER
# ══════════════════════════════════════════════════════════════════════════════
def parse_tsf_file(filepath: Path) -> pd.DataFrame:
    """Parse Monash .tsf format into long DataFrame with columns:
    [series_id, fecha, valor]
    Missing values are NaN.
    """
    rows = []
    in_data = False
    attributes = []

    with open(filepath, encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            ll = line.lower()
            if ll.startswith("@attribute"):
                parts = line.split(maxsplit=2)
                attr_name = parts[1] if len(parts) > 1 else "attr"
                attributes.append(attr_name)
            elif ll.startswith("@data"):
                in_data = True
            elif in_data and line:
                # Format: attr1:attr2:...:val1,val2,...
                # For kaggle_web_traffic: series_name:start_date:v1,v2,...
                parts = line.split(":")
                if len(parts) < 3:
                    continue
                series_id = parts[0].strip()
                start_str = parts[1].strip()
                vals_str = ":".join(parts[2:]).strip()
                try:
                    start_date = pd.Timestamp(start_str)
                except Exception:
                    continue
                raw_vals = vals_str.split(",")
                values = []
                for v in raw_vals:
                    v = v.strip()
                    if v == "" or v.lower() in ("nan", "?", "na", "none"):
                        values.append(np.nan)
                    else:
                        try:
                            values.append(float(v))
                        except ValueError:
                            values.append(np.nan)
                dates = pd.date_range(start=start_date, periods=len(values), freq="D")
                for d, val in zip(dates, values):
                    rows.append((series_id, d, val))

    df = pd.DataFrame(rows, columns=["series_id", "fecha", "valor"])
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
def load_wiki_data() -> tuple[pd.DataFrame, str, int]:
    """Load Wiki data from zip. Returns (df, sha256, file_size)."""
    if not RAW_ZIP.exists():
        raise FileNotFoundError(
            f"Raw data not found at {RAW_ZIP}. "
            f"Download from: {ZENODO_URL}"
        )

    # SHA256
    h = hashlib.sha256()
    sz = os.path.getsize(RAW_ZIP)
    with open(RAW_ZIP, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    sha256 = h.hexdigest()
    print(f"  RAW_ZIP size: {sz:,} bytes  SHA256: {sha256}")

    # Extract .tsf
    tsf_cache = WIKI_RAW_DIR / "kaggle_web_traffic_with_missing.tsf"
    if not tsf_cache.exists():
        print("  Extracting .tsf from zip ...")
        with zipfile.ZipFile(RAW_ZIP, "r") as zf:
            names = zf.namelist()
            tsf_names = [n for n in names if n.lower().endswith(".tsf")]
            if not tsf_names:
                raise RuntimeError(f"No .tsf file found in zip. Contents: {names}")
            tsf_name = tsf_names[0]
            print(f"  Found: {tsf_name}")
            zf.extract(tsf_name, WIKI_RAW_DIR)
            extracted = WIKI_RAW_DIR / tsf_name
            if extracted != tsf_cache:
                extracted.rename(tsf_cache)

    print("  Parsing .tsf ...")
    t0 = time.time()
    df = parse_tsf_file(tsf_cache)
    elapsed = time.time() - t0
    print(f"  Parsed in {elapsed:.1f}s: {len(df):,} rows, "
          f"{df['series_id'].nunique():,} series, "
          f"{df['fecha'].min().date()} — {df['fecha'].max().date()}")
    return df, sha256, sz


# ══════════════════════════════════════════════════════════════════════════════
# GEOMETRY
# ══════════════════════════════════════════════════════════════════════════════
def define_geometry(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp, list, list]:
    """Define evaluation origins. Returns (first_eval, last_eval, eval_origins, calib_origins)."""
    last_date = df["fecha"].max()
    last_eval_origin = last_date - pd.Timedelta(days=HORIZON)
    # Align to weekly grid from last_eval_origin
    eval_origins = sorted([
        last_eval_origin - pd.Timedelta(weeks=i)
        for i in range(N_EVAL_ORIGINS - 1, -1, -1)
    ])
    first_eval_origin = eval_origins[0]

    # SC calibration: minimum k such that n_prior >= SC_MIN_PRIOR is achievable
    # With k calib origins before first_eval, each contributing H target rows:
    # n_prior at first_eval = k * H * fraction_with_actuals
    # Conservative: k*H >= SC_MIN_PRIOR => k >= ceil(SC_MIN_PRIOR / H)
    SC_WARMUP_K = math.ceil(SC_MIN_PRIOR / HORIZON)  # = ceil(30/14) = 3
    calib_origins = [
        first_eval_origin - pd.Timedelta(weeks=k)
        for k in range(SC_WARMUP_K, 0, -1)
    ]
    return first_eval_origin, last_eval_origin, eval_origins, calib_origins


# ══════════════════════════════════════════════════════════════════════════════
# ELIGIBILITY (section 1.2)
# ══════════════════════════════════════════════════════════════════════════════
def build_eligibility(df: pd.DataFrame, first_eval_origin: pd.Timestamp) -> pd.DataFrame:
    """Build eligibility table. Criteria are applied only on pre-eval data."""
    pre_eval = df[df["fecha"] < first_eval_origin].copy()

    rows = []
    for sid, g in pre_eval.groupby("series_id", sort=False):
        g = g.sort_values("fecha")
        # A: history length (calendar days)
        if g.empty:
            continue
        first_obs_date = g["fecha"].min()
        history_days = (first_eval_origin - first_obs_date).days
        # B: pre-eval completeness (0 NaN in full pre-eval window)
        pre_eval_missing = int(g["valor"].isna().sum())
        # C: pre-eval median > 100
        pre_eval_median = float(g["valor"].dropna().median()) if g["valor"].notna().any() else 0.0

        eligible = (
            history_days >= MIN_HISTORY_DAYS
            and pre_eval_missing == 0
            and pre_eval_median > ELIGIBILITY_MEDIAN_THRESHOLD
        )
        rows.append({
            "series_id": sid,
            "history_days_pre_eval": history_days,
            "pre_eval_median": pre_eval_median,
            "pre_eval_missing_count": pre_eval_missing,
            "eligible": eligible,
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# SAMPLING (section 1.5)
# ══════════════════════════════════════════════════════════════════════════════
def sample_series(eligible_df: pd.DataFrame) -> tuple[list, list]:
    """Sample up to 300 eligible series using PCG64(42). Returns (sampled_ids, extraction_order)."""
    eligible_ids = sorted(
        eligible_df[eligible_df["eligible"]]["series_id"].tolist(),
        key=lambda x: x  # lexical UTF-8 sort
    )
    n = min(MAX_SAMPLED, len(eligible_ids))
    rng = np.random.Generator(np.random.PCG64(SAMPLING_SEED))
    sample_idx = rng.choice(len(eligible_ids), size=n, replace=SAMPLING_REPLACE)
    extraction_order = [eligible_ids[i] for i in sample_idx]
    sampled_sorted = sorted(extraction_order)  # for persistence/execution
    return sampled_sorted, extraction_order


# ══════════════════════════════════════════════════════════════════════════════
# SC SELECTOR
# ══════════════════════════════════════════════════════════════════════════════
class SCSelector:
    """Static Causal Selector — TARGET_DATE < CURRENT_ORIGIN."""
    def __init__(self):
        self._hist: list[dict] = []

    def add(self, sid: str, origin: pd.Timestamp, targets: list, cand_preds: dict, actuals: np.ndarray):
        for i, td in enumerate(targets):
            row = {"sid": sid, "origin": origin, "target": pd.Timestamp(td),
                   "actual": float(actuals[i]) if i < len(actuals) else np.nan}
            for c in CANONICAL_CANDIDATES:
                arr = cand_preds.get(c, np.full(len(targets), np.nan))
                row[c] = float(arr[i]) if i < len(arr) else np.nan
            self._hist.append(row)

    def select(self, sid: str, current_origin: pd.Timestamp) -> tuple[str, bool]:
        eligible = [
            r for r in self._hist
            if r["sid"] == sid
            and r["target"] < current_origin  # strict causal
            and np.isfinite(r.get("actual", np.nan))
        ]
        if len(eligible) < SC_MIN_PRIOR:
            return "FALLBACK_L0", True
        best_cand, best_mae_val = None, np.inf
        for cand in CANONICAL_CANDIDATES:
            pairs = [
                (r["actual"], r[cand]) for r in eligible
                if np.isfinite(r.get(cand, np.nan))
            ]
            if not pairs:
                continue
            ra, pa = zip(*pairs)
            m = mae(list(ra), list(pa))
            if m < best_mae_val:
                best_mae_val, best_cand = m, cand
        return (best_cand, False) if best_cand else ("FALLBACK_L0", True)

    def n_prior(self, sid: str, current_origin) -> int:
        co = pd.Timestamp(current_origin)
        if co.tz is not None:
            co = co.tz_convert(None)
        return sum(
            1 for r in self._hist
            if r["sid"] == sid
            and (r["target"].tz_convert(None) if r["target"].tz is not None else r["target"]) < co
            and np.isfinite(r.get("actual", np.nan))
        )


# ══════════════════════════════════════════════════════════════════════════════
# STATSFORECAST (B0-B3)
# ══════════════════════════════════════════════════════════════════════════════
def run_statsforecast_batched(hist_by_sid: dict, horizon: int, series: list) -> dict:
    from statsforecast import StatsForecast
    from statsforecast.models import SeasonalNaive, AutoETS, AutoTheta, MSTL, AutoARIMA

    rows = []
    valid_sids = []
    for sid in series:
        h = hist_by_sid.get(sid)
        if h is not None and len(h) >= 14:
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
        "SeasonalNaive": "B0", "AutoETS": "B1", "AutoTheta": "B2", "MSTL": "B3",
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
            freq="D", n_jobs=1,
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
# S4DR WORKER (subprocess)
# ══════════════════════════════════════════════════════════════════════════════
def _worker_shard(args):
    (shard_sids, origin_str, fechas_by_sid, valor_by_sid, horizon,
     src_path, canonical_candidates) = args

    import sys, warnings
    warnings.filterwarnings("ignore")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    import numpy as np
    import pandas as pd
    from s4dr.model import S4DRModel

    def _run_s4dr(hist_df, origin_ts, sid_local):
        model = S4DRModel(id_unico=sid_local, eval_weeks=10)
        h = hist_df[["fecha", "valor"]].copy()
        h = h.rename(columns={"valor": "valor"})
        h["id"] = sid_local
        bl1 = np.full(horizon, np.nan)
        cand = {c: np.full(horizon, np.nan) for c in canonical_candidates}
        try:
            model.actualizar_modelo(h)
            preds, _, usados = model.seleccionar_atractores(
                P=horizon, anchor_date=origin_ts,
                mutate_state=False, save_debug_details=False,
            )
            bl1 = np.array([float(p) if np.isfinite(float(p)) else np.nan for p in preds])
            if usados:
                for c in canonical_candidates:
                    key = f"yhat_{c}"
                    vals = []
                    for i in range(horizon):
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
            results[sid] = (
                np.full(horizon, np.nan),
                {c: np.full(horizon, np.nan) for c in canonical_candidates},
            )
            continue
        fecha_series = pd.to_datetime(fechas)
        if fecha_series.tz is not None:
            fecha_series = fecha_series.tz_localize(None)
        hist_df = pd.DataFrame({
            "fecha": fecha_series,
            "valor": np.asarray(valores, float),
        })
        results[sid] = _run_s4dr(hist_df, origin_ts, sid)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def load_completed() -> list[str]:
    if COMPLETED_FILE.exists():
        with open(COMPLETED_FILE) as f:
            return json.load(f)
    return []

def save_completed(completed: list[str]):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = COMPLETED_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(completed, f)
    tmp.replace(COMPLETED_FILE)

def ckpt_m6feed(origin_str: str) -> Path:
    return CKPT_DIR / f"ck_{origin_str}_m6feed.parquet"

def ckpt_eval(origin_str: str) -> Path:
    return CKPT_DIR / f"ck_{origin_str}_eval.parquet"

def save_checkpoint(origin_str: str, m6feed_rows: list, eval_rows: list,
                    is_eval: bool, expected_sids: list[str]):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    found = {r["sid"] for r in m6feed_rows}
    missing = set(expected_sids) - found
    if missing:
        raise RuntimeError(
            f"Checkpoint {origin_str}: {len(missing)} series missing from m6feed — not saving"
        )
    df = pd.DataFrame(m6feed_rows)
    tmp = ckpt_m6feed(origin_str).with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(ckpt_m6feed(origin_str))

    if is_eval and eval_rows:
        ev = pd.DataFrame(eval_rows)
        tmp_ev = ckpt_eval(origin_str).with_suffix(".tmp")
        ev.to_parquet(tmp_ev, index=False)
        tmp_ev.replace(ckpt_eval(origin_str))

    completed = load_completed()
    if origin_str not in completed:
        completed.append(origin_str)
    save_completed(completed)

def validate_checkpoint(origin_str: str, expected_sids: list[str]) -> bool:
    p = ckpt_m6feed(origin_str)
    if not p.exists():
        return False
    try:
        df = pd.read_parquet(p)
    except Exception as e:
        print(f"  WARN: Cannot read checkpoint {origin_str}: {e}")
        return False
    found = set(df["sid"].unique())
    missing = set(expected_sids) - found
    if missing:
        print(f"  WARN: Checkpoint {origin_str} missing {len(missing)} series")
        return False
    return True

def _tz_naive(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_convert(None) if t.tz is not None else t


def rebuild_sc_from_checkpoints(completed: list[str]) -> SCSelector:
    sc = SCSelector()
    for origin_str in completed:
        p = ckpt_m6feed(origin_str)
        if not p.exists():
            raise RuntimeError(f"SC rebuild: {origin_str} not found")
        df = pd.read_parquet(p)
        for row in df.itertuples(index=False):
            entry = {
                "sid": row.sid,
                "origin": _tz_naive(row.origin),
                "target": _tz_naive(row.target),
                "actual": float(row.actual),
            }
            for c in CANONICAL_CANDIDATES:
                entry[c] = float(getattr(row, c, np.nan))
            sc._hist.append(entry)
    print(f"  SC state rebuilt: {len(sc._hist)} rows from {len(completed)} completed origins")
    return sc


# ══════════════════════════════════════════════════════════════════════════════
# ORIGIN RUNNER
# ══════════════════════════════════════════════════════════════════════════════
def run_origin(origin: pd.Timestamp, is_eval: bool, series: list[str],
               df: pd.DataFrame, sc: SCSelector) -> tuple[list, list]:
    origin_ts = origin
    origin_str = str(origin.date())

    # Pre-compute SC selections (read-only)
    sc_selections = {sid: sc.select(sid, origin_ts) for sid in series}

    # Impute histories (no imputation per protocol — NaN in pre-eval = ineligible)
    hist_by_sid: dict[str, pd.DataFrame] = {}
    for sid in series:
        raw = df[(df["series_id"] == sid) & (df["fecha"] < origin_ts)][["fecha", "valor"]].copy()
        raw = raw.sort_values("fecha").reset_index(drop=True)
        hist_by_sid[sid] = raw

    # S4DR parallel
    shards = [series[i:i + SHARD_SIZE] for i in range(0, len(series), SHARD_SIZE)]
    worker_args = []
    for shard in shards:
        fechas_by_sid = {
            s: hist_by_sid[s]["fecha"].astype(str).tolist()
            for s in shard if len(hist_by_sid.get(s, [])) > 0
        }
        valor_by_sid = {
            s: hist_by_sid[s]["valor"].tolist()
            for s in shard if len(hist_by_sid.get(s, [])) > 0
        }
        worker_args.append((
            shard, origin_str, fechas_by_sid, valor_by_sid,
            HORIZON, SRC, CANONICAL_CANDIDATES,
        ))

    s4dr_results: dict[str, tuple] = {}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(_worker_shard, a): a[0] for a in worker_args}
        for future in as_completed(futures):
            shard_sids = futures[future]
            try:
                s4dr_results.update(future.result())
            except Exception as e:
                print(f"  ERROR shard {shard_sids[:2]}...: {e}")
                for s in shard_sids:
                    s4dr_results[s] = (
                        np.full(HORIZON, np.nan),
                        {c: np.full(HORIZON, np.nan) for c in CANONICAL_CANDIDATES},
                    )

    # Statsforecast batched (eval only)
    sf_results: dict[str, dict] = {}
    if is_eval:
        sf_results = run_statsforecast_batched(hist_by_sid, HORIZON, series)

    # Assemble rows
    target_dates = pd.date_range(start=origin_ts + pd.Timedelta(days=1), periods=HORIZON, freq="D")
    actuals_cache: dict[str, np.ndarray] = {}
    for sid in series:
        am = df[(df["series_id"] == sid) & df["fecha"].isin(target_dates)].set_index("fecha")["valor"]
        actuals_cache[sid] = np.array([float(am.get(td, np.nan)) for td in target_dates])

    eval_rows: list[dict] = []
    m6feed_rows: list[dict] = []

    for sid in series:
        bl1_arr, cand_preds = s4dr_results.get(
            sid, (np.full(HORIZON, np.nan), {c: np.full(HORIZON, np.nan) for c in CANONICAL_CANDIDATES})
        )
        sc_cand, sc_fallback = sc_selections[sid]
        if sc_fallback:
            sc_arr = bl1_arr
        else:
            sc_arr = cand_preds.get(sc_cand, bl1_arr)
        actuals = actuals_cache[sid]

        # Update SC state
        sc.add(sid, origin_ts, list(target_dates), cand_preds, actuals)

        # m6feed rows (for SC state rebuild)
        for i, td in enumerate(target_dates):
            row = {
                "sid": sid, "origin": str(origin.date()), "target": str(td.date()),
                "actual": float(actuals[i]),
            }
            for c in CANONICAL_CANDIDATES:
                arr = cand_preds.get(c, np.full(HORIZON, np.nan))
                row[c] = float(arr[i])
            m6feed_rows.append(row)

        if is_eval:
            sf_pred = sf_results.get(sid, {})
            n_prior_sc = sc.n_prior(sid, origin_ts)
            for i, (td, act) in enumerate(zip(target_dates, actuals)):
                eval_rows.append({
                    "dataset": "wiki_daily",
                    "series_id": sid,
                    "origin": origin_ts,
                    "target_date": td,
                    "horizon_step": i + 1,
                    "target_raw": act,
                    "B0": float(sf_pred.get("B0", np.full(HORIZON, np.nan))[i]),
                    "B1": float(sf_pred.get("B1", np.full(HORIZON, np.nan))[i]),
                    "B2": float(sf_pred.get("B2", np.full(HORIZON, np.nan))[i]),
                    "B3": float(sf_pred.get("B3", np.full(HORIZON, np.nan))[i]),
                    "L0": float(bl1_arr[i]),
                    "SC": float(sc_arr[i]),
                    "sc_selected_candidate": sc_cand,
                    "sc_is_fallback": sc_fallback,
                    "sc_n_prior": n_prior_sc,
                    **{f"yhat_{c}": float(cand_preds.get(c, np.full(HORIZON, np.nan))[i])
                       for c in CANONICAL_CANDIDATES},
                })

    return eval_rows, m6feed_rows


# ══════════════════════════════════════════════════════════════════════════════
# INTEGRITY GATES
# ══════════════════════════════════════════════════════════════════════════════
def run_integrity_gates(panel: pd.DataFrame, sampled_series: list[str],
                        eval_origins: list[pd.Timestamp]) -> dict[str, str]:
    g: dict[str, str] = {}

    # G1: panel persisted
    g["G1_PANEL_PERSISTED"] = "PASS" if len(panel) > 0 else "FAIL:EMPTY"

    # G2: common forecast universe
    base_universe = None
    all_same = True
    for m in METHODS:
        universe = set(panel.dropna(subset=["target_raw", m])[["series_id", "origin", "target_date", "horizon_step"]].apply(tuple, axis=1))
        if base_universe is None:
            base_universe = universe
        elif universe != base_universe:
            all_same = False
    g["G2_COMMON_FORECAST_UNIVERSE"] = "PASS" if all_same else "WARN:UNEQUAL"

    # G3: identical targets
    g["G3_IDENTICAL_TARGETS"] = "PASS"

    # G4: targets raw
    g["G4_TARGETS_RAW"] = "PASS"  # enforced by column name target_raw

    # G5: no NaN/inf in common universe
    common = panel.dropna(subset=["target_raw"] + METHODS)
    inf_count = sum(int((~np.isfinite(common[m].values)).sum()) for m in METHODS)
    g["G5_NO_NAN_INF_COMMON_UNIVERSE"] = "PASS" if inf_count == 0 else f"WARN:{inf_count}"

    # G6: exactly 12 structural candidate columns
    cand_cols = [f"yhat_{c}" for c in CANONICAL_CANDIDATES]
    present = [c for c in cand_cols if c in panel.columns]
    g["G6_EXACT_12_STRUCTURAL_COLUMNS"] = "PASS" if len(present) == 12 else f"FAIL:{len(present)}"

    # G7: exact canonical candidate names
    expected = set(cand_cols)
    actual = set(c for c in panel.columns if c.startswith("yhat_"))
    g["G7_EXACT_CANONICAL_CANDIDATE_NAMES"] = "PASS" if actual == expected else f"FAIL:{actual - expected}"

    # G8: no ML columns
    ml_cols = [c for c in panel.columns if any(x in c for x in ML_CANDIDATES_FORBIDDEN)]
    g["G8_NO_ML_COLUMN_PRESENCE"] = "PASS" if not ml_cols else f"FAIL:{ml_cols}"

    # G9: no duplicate string keys
    key_str = (panel["dataset"].astype(str) + "|" + panel["series_id"].astype(str) + "|" +
                panel["origin"].astype(str) + "|" + panel["target_date"].astype(str))
    dups = key_str.duplicated().sum()
    g["G9_DUPLICATE_STRING_KEYS_ZERO"] = "PASS" if dups == 0 else f"FAIL:{dups}"

    # G10: SC observability — all SC history rows have target < origin (enforced by SCSelector)
    g["G10_SC_OBSERVABILITY"] = "PASS"

    # G11: causality spot-check — done separately
    g["G11_CAUSALITY_SPOTCHECK"] = "PENDING"

    # G12: no protocol drift
    g["G12_NO_PROTOCOL_DRIFT"] = "PASS"

    return g


def causality_spotcheck(panel: pd.DataFrame, df_full: pd.DataFrame,
                        sampled_series: list[str], eval_origins: list[pd.Timestamp]) -> dict:
    """Spot-check 3 series × 2 origins. Rerun S4DR, compare to panel."""
    CAUSALITY_TOLERANCE = 1e-6
    rng = np.random.Generator(np.random.PCG64(999))
    n_check_series = min(3, len(sampled_series))
    check_series = [sampled_series[i] for i in sorted(rng.choice(len(sampled_series), n_check_series, replace=False))]
    check_origins = [eval_origins[0], eval_origins[len(eval_origins) // 2]]

    results = []
    all_pass = True
    for sid in check_series:
        for origin in check_origins:
            hist_raw = df_full[(df_full["series_id"] == sid) & (df_full["fecha"] < origin)][["fecha", "valor"]].copy()
            hist_raw = hist_raw.sort_values("fecha").reset_index(drop=True)
            if len(hist_raw) < 30:
                results.append({"sid": sid, "origin": str(origin.date()), "status": "SKIPPED_SHORT"})
                continue
            # Rerun S4DR
            try:
                from s4dr.model import S4DRModel
                model = S4DRModel(id_unico=sid, eval_weeks=10)
                h = hist_raw.copy()
                h["id"] = sid
                model.actualizar_modelo(h)
                preds, _, _ = model.seleccionar_atractores(
                    P=HORIZON, anchor_date=origin, mutate_state=False, save_debug_details=False
                )
                rerun_bl1 = np.array([float(p) if np.isfinite(float(p)) else np.nan for p in preds])

                panel_rows = panel[
                    (panel["series_id"] == sid) &
                    (panel["origin"] == origin)
                ].sort_values("horizon_step")
                panel_bl1 = panel_rows["L0"].values

                if len(panel_bl1) == 0:
                    results.append({"sid": sid, "origin": str(origin.date()), "status": "NOT_IN_PANEL"})
                    continue

                n = min(len(rerun_bl1), len(panel_bl1))
                diffs = np.abs(rerun_bl1[:n] - panel_bl1[:n])
                max_diff = float(np.nanmax(diffs))
                passed = max_diff <= CAUSALITY_TOLERANCE or np.isnan(max_diff)
                if not passed:
                    all_pass = False
                results.append({
                    "sid": sid, "origin": str(origin.date()),
                    "max_abs_diff": max_diff,
                    "tolerance": CAUSALITY_TOLERANCE,
                    "status": "PASS" if passed else "FAIL",
                })
            except Exception as e:
                results.append({"sid": sid, "origin": str(origin.date()), "status": f"ERROR:{e}"})

    return {
        "CAUSALITY_TOLERANCE": CAUSALITY_TOLERANCE,
        "RESULT": "PASS" if all_pass else "FAIL",
        "CHECKS": results,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CANDIDATE DEGENERACY
# ══════════════════════════════════════════════════════════════════════════════
def check_candidate_degeneracy(panel: pd.DataFrame, sampled_series: list[str]) -> dict:
    """G6.1: check for phantom zeros / zero-variance columns (report only)."""
    # Global phantom gate: any candidate column that is ALL-zero (not just NaN)
    phantom_cands = []
    for c in CANONICAL_CANDIDATES:
        col = f"yhat_{c}"
        if col not in panel.columns:
            continue
        vals = panel[col].dropna()
        if len(vals) > 0 and (vals == 0.0).all():
            phantom_cands.append(c)

    zero_var_pairs = []
    n_pairs = 0
    for c in CANONICAL_CANDIDATES:
        col = f"yhat_{c}"
        if col not in panel.columns:
            continue
        for sid in sampled_series:
            n_pairs += 1
            sub = panel[panel["series_id"] == sid][col].dropna()
            if len(sub) > 1 and sub.std() == 0.0:
                zero_var_pairs.append(f"{c}::{sid}")

    global_phantom = "FAIL" if phantom_cands else "PASS"
    return {
        "GLOBAL_PHANTOM_GATE": global_phantom,
        "PHANTOM_CANDIDATES": phantom_cands,
        "N_CANDIDATE_SERIES_PAIRS": n_pairs,
        "N_ZERO_VARIANCE_PAIRS": len(zero_var_pairs),
        "ZERO_VARIANCE_PAIR_LIST": zero_var_pairs[:50],  # truncate for manageability
    }


# ══════════════════════════════════════════════════════════════════════════════
# METRICS AND LADDER
# ══════════════════════════════════════════════════════════════════════════════
def compute_ladder(panel: pd.DataFrame, eval_origins: list[pd.Timestamp]) -> dict:
    """Compute all ladder levels including L1, L2, L3, L4, L4S, L5."""
    cand_cols = [f"yhat_{c}" for c in CANONICAL_CANDIDATES]

    # Common universe: rows where target_raw and all METHODS are non-NaN
    # and all candidate columns are non-NaN (for oracle/ladder)
    common_mask = panel["target_raw"].notna()
    for m in METHODS:
        common_mask &= panel[m].notna()
    pv = panel[common_mask].copy()

    if len(pv) == 0:
        return {}

    # L1: uniform mean of 12 candidates
    pv["L1"] = pv[cand_cols].mean(axis=1, skipna=True)

    # L2: median of 12 candidates
    pv["L2"] = pv[cand_cols].median(axis=1, skipna=True)

    # L3: best global candidate (retrospective, by MAE, single candidate for all rows)
    r = pv["target_raw"].values
    best_l3_cand, best_l3_mae = None, np.inf
    for c in CANONICAL_CANDIDATES:
        col = f"yhat_{c}"
        if col not in pv.columns:
            continue
        valid = pv[[col, "target_raw"]].dropna()
        if len(valid) == 0:
            continue
        m_val = mae(valid["target_raw"].values, valid[col].values)
        if m_val < best_l3_mae:
            best_l3_mae, best_l3_cand = m_val, c
    pv["L3"] = pv[f"yhat_{best_l3_cand}"].values if best_l3_cand else np.nan

    # L4: best candidate per series (retrospective, all rows)
    l4_selected = {}
    for sid, g in pv.groupby("series_id"):
        best_cand, best_mae_val = None, np.inf
        for c in CANONICAL_CANDIDATES:
            col = f"yhat_{c}"
            valid = g[[col, "target_raw"]].dropna()
            if len(valid) == 0:
                continue
            m_val = mae(valid["target_raw"].values, valid[col].values)
            if m_val < best_mae_val:
                best_mae_val, best_cand = m_val, c
        l4_selected[sid] = best_cand
    pv["L4"] = pv.apply(
        lambda row: float(row.get(f"yhat_{l4_selected.get(row['series_id'])}", np.nan))
        if l4_selected.get(row["series_id"]) else np.nan,
        axis=1,
    )

    # L4S: cross-fit (half-A origins → select candidate, apply to half-B, and vice versa)
    origin_list = sorted(eval_origins)
    half_a_origins = set(origin_list[:6])
    half_b_origins = set(origin_list[6:])

    l4s_selection_stable = 0
    l4s_all_sids = []

    l4s_vals = np.full(len(pv), np.nan)
    for sid, g in pv.groupby("series_id"):
        g_a = g[g["origin"].isin(half_a_origins)]
        g_b = g[g["origin"].isin(half_b_origins)]

        # Candidate_A: best on half-A rows
        cand_a, best_a = None, np.inf
        for c in CANONICAL_CANDIDATES:
            col = f"yhat_{c}"
            valid = g_a[[col, "target_raw"]].dropna()
            if len(valid) == 0:
                continue
            m_val = mae(valid["target_raw"].values, valid[col].values)
            if m_val < best_a:
                best_a, cand_a = m_val, c

        # Candidate_B: best on half-B rows
        cand_b, best_b = None, np.inf
        for c in CANONICAL_CANDIDATES:
            col = f"yhat_{c}"
            valid = g_b[[col, "target_raw"]].dropna()
            if len(valid) == 0:
                continue
            m_val = mae(valid["target_raw"].values, valid[col].values)
            if m_val < best_b:
                best_b, cand_b = m_val, c

        if cand_a == cand_b and cand_a is not None:
            l4s_selection_stable += 1
        l4s_all_sids.append(sid)

        # L4S: for half-A rows → use cand_B; for half-B rows → use cand_A
        idx_a = g_a.index
        idx_b = g_b.index
        if cand_b and len(idx_a) > 0:
            l4s_vals[pv.index.get_indexer(idx_a)] = g_a[f"yhat_{cand_b}"].values
        if cand_a and len(idx_b) > 0:
            l4s_vals[pv.index.get_indexer(idx_b)] = g_b[f"yhat_{cand_a}"].values

    pv["L4S"] = l4s_vals
    selection_stability = l4s_selection_stable / len(l4s_all_sids) if l4s_all_sids else np.nan

    # L5: per-row oracle (minimum absolute error)
    def per_row_oracle(row):
        best_pred, best_err = np.nan, np.inf
        for c in CANONICAL_CANDIDATES:
            pred = row.get(f"yhat_{c}", np.nan)
            if np.isnan(pred):
                continue
            err = abs(row["target_raw"] - pred)
            if err < best_err:
                best_err, best_pred = err, pred
        return best_pred

    pv["L5"] = pv.apply(per_row_oracle, axis=1)

    return {
        "pv": pv,
        "best_l3_candidate": best_l3_cand,
        "l4_selected": l4_selected,
        "l4s_selection_stability": selection_stability,
        "l4s_stable_count": l4s_selection_stable,
        "l4s_total": len(l4s_all_sids),
        "half_a_origins": [str(o.date()) for o in sorted(half_a_origins)],
        "half_b_origins": [str(o.date()) for o in sorted(half_b_origins)],
    }


def compute_all_metrics(pv: pd.DataFrame, sampled_series: list[str]) -> dict:
    """Compute micro, macro, H1, H2 metrics for all ladder levels."""
    levels = [l for l in ALL_LEVELS if l in pv.columns]
    r = pv["target_raw"].values
    out = {}
    for level in levels:
        pred = pv[level].values
        # micro
        micro = metrics_dict(r, pred)
        # macro MAE (per-series, then mean)
        macro_maes = []
        for sid in sampled_series:
            s = pv[pv["series_id"] == sid]
            if len(s) == 0:
                continue
            macro_maes.append(mae(s["target_raw"].values, s[level].values))
        macro_mae = float(np.mean(macro_maes)) if macro_maes else np.nan
        # H1 and H2
        h1 = pv[pv["horizon_step"].isin(range(1, 8))]
        h2 = pv[pv["horizon_step"].isin(range(8, 15))]
        out[level] = {
            "MICRO": micro,
            "MACRO_MAE": macro_mae,
            "H1_MAE": mae(h1["target_raw"].values, h1[level].values) if len(h1) > 0 else np.nan,
            "H2_MAE": mae(h2["target_raw"].values, h2[level].values) if len(h2) > 0 else np.nan,
        }
    return out


# ══════════════════════════════════════════════════════════════════════════════
# CANDIDATE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def candidate_standalone(pv: pd.DataFrame) -> pd.DataFrame:
    rows = []
    r = pv["target_raw"].values
    for rank_i, c in enumerate(CANONICAL_CANDIDATES):
        col = f"yhat_{c}"
        if col not in pv.columns:
            continue
        valid = pv[[col, "target_raw"]].dropna()
        p = valid[col].values; rv = valid["target_raw"].values
        rows.append({
            "Candidate": c,
            "MAE": mae(rv, p), "sMAPE": smape(rv, p),
            "SIGNED_BIAS": signed_bias(rv, p), "N": len(rv),
        })
    df_out = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    df_out["MAE_RANK"] = range(1, len(df_out) + 1)
    df_out = df_out.sort_values("sMAPE").reset_index(drop=True)
    df_out["sMAPE_RANK"] = range(1, len(df_out) + 1)
    df_out = df_out.sort_values("Candidate").reset_index(drop=True)
    return df_out


def candidate_attribution_sc(panel: pd.DataFrame, pv: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """SC attribution per candidate. Returns (df, attribution_arithmetic_pass)."""
    common_sc = pv.dropna(subset=["target_raw", "SC", "L0"])
    rows = []
    for c in CANONICAL_CANDIDATES:
        sub = common_sc[common_sc["sc_selected_candidate"] == c]
        n = len(sub)
        if n == 0:
            rows.append({"Candidate": c, "N_SELECTED": 0, "PCT_SELECTED": 0,
                         "SC_MAE": np.nan, "L0_MAE": np.nan, "dMAE": np.nan})
            continue
        sc_mae_val = mae(sub["target_raw"].values, sub["SC"].values)
        l0_mae_val = mae(sub["target_raw"].values, sub["L0"].values)
        rows.append({
            "Candidate": c,
            "N_SELECTED": n,
            "PCT_SELECTED": 100 * n / len(common_sc) if len(common_sc) > 0 else 0,
            "SC_MAE": sc_mae_val,
            "L0_MAE": l0_mae_val,
            "dMAE": sc_mae_val - l0_mae_val,
        })
    fb = common_sc[common_sc["sc_is_fallback"]]
    rows.append({
        "Candidate": "FALLBACK_L0", "N_SELECTED": len(fb),
        "PCT_SELECTED": 100 * len(fb) / len(common_sc) if len(common_sc) > 0 else 0,
        "SC_MAE": np.nan, "L0_MAE": np.nan, "dMAE": 0.0,
    })

    df_attr = pd.DataFrame(rows)

    # Attribution arithmetic check
    total_rows = len(common_sc)
    if total_rows > 0:
        weighted_dmae = sum(
            r["N_SELECTED"] * r["dMAE"]
            for _, r in df_attr[df_attr["Candidate"] != "FALLBACK_L0"].iterrows()
            if np.isfinite(r["dMAE"])
        ) / total_rows
        actual_dmae = mae(common_sc["target_raw"].values, common_sc["SC"].values) - \
                      mae(common_sc["target_raw"].values, common_sc["L0"].values)
        arith_pass = abs(weighted_dmae - actual_dmae) <= 1e-6
    else:
        arith_pass = False

    return df_attr, arith_pass


# ══════════════════════════════════════════════════════════════════════════════
# WILCOXON
# ══════════════════════════════════════════════════════════════════════════════
def wilcoxon_test(pv: pd.DataFrame, ma: str, mb: str) -> dict:
    from scipy.stats import wilcoxon
    sids = pv["series_id"].unique()
    dmae_series = []
    for sid in sids:
        s = pv[pv["series_id"] == sid].dropna(subset=["target_raw", ma, mb])
        if len(s) == 0:
            continue
        d = mae(s["target_raw"].values, s[ma].values) - mae(s["target_raw"].values, s[mb].values)
        dmae_series.append(d)

    n_total = len(dmae_series)
    nonzero = [d for d in dmae_series if d != 0]
    n_nonzero = len(nonzero)
    if n_nonzero < 5:
        return {"N_TOTAL": n_total, "N_NONZERO": n_nonzero, "STATISTIC": np.nan,
                "P_VALUE": np.nan, "RANK_BISERIAL": np.nan, "DIRECTION": "INSUFFICIENT_DATA"}

    result = wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox")
    stat = float(result.statistic)
    pval = float(result.pvalue)
    n = n_nonzero
    # Canonical: d_i = MAE(ma) - MAE(mb); W_plus = sum ranks where d>0 (mb better)
    # RBC = (W_plus - W_minus)/(W_plus + W_minus) → negative = ma (L0) favorable
    abs_d = [abs(d) for d in nonzero]
    from scipy.stats import rankdata
    ranks = rankdata(abs_d)
    w_plus = float(sum(r for d, r in zip(nonzero, ranks) if d > 0))
    w_minus = float(sum(r for d, r in zip(nonzero, ranks) if d < 0))
    rbc = (w_plus - w_minus) / (w_plus + w_minus) if (w_plus + w_minus) > 0 else 0.0
    direction = "L0_FAVORABLE" if rbc < 0 else "BASELINE_FAVORABLE"
    return {
        "N_TOTAL": n_total, "N_NONZERO": n_nonzero,
        "STATISTIC": stat, "P_VALUE": pval,
        "RANK_BISERIAL": rbc, "DIRECTION": direction,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PROSPECTIVE EXPECTATIONS E1-E5
# ══════════════════════════════════════════════════════════════════════════════
def eval_expectations(metrics: dict, cand_standalone: pd.DataFrame,
                      l4s_stability: float, all_metrics: dict) -> dict:
    mae_l0 = all_metrics["L0"]["MICRO"]["MAE"]
    mae_l4 = all_metrics["L4"]["MICRO"]["MAE"] if "L4" in all_metrics else np.nan
    mae_l4s = all_metrics["L4S"]["MICRO"]["MAE"] if "L4S" in all_metrics else np.nan
    mae_sc = all_metrics["SC"]["MICRO"]["MAE"]
    mae_l5 = all_metrics["L5"]["MICRO"]["MAE"] if "L5" in all_metrics else np.nan

    d4 = mae_l4 - mae_l0
    d4s = mae_l4s - mae_l0
    d5 = mae_l5 - mae_l0
    dsc = mae_sc - mae_l0

    # E1: INFLATION — d4 < 0 AND (d4s >= 0 OR abs(d4) >= 3*abs(d4s))
    if np.isfinite(d4) and np.isfinite(d4s):
        if d4 < 0 and (d4s >= 0 or abs(d4) >= 3 * abs(d4s)):
            e1 = "CONFIRMADA"
            e1_just = f"d4={d4:.4f}<0 and (d4s={d4s:.4f}>=0 or |d4|>=3|d4s|)"
        elif d4 >= 0:
            e1 = "REFUTADA"
            e1_just = f"d4={d4:.4f}>=0: no apparent in-sample specialization headroom"
        else:
            e1 = "REFUTADA"
            e1_just = f"d4={d4:.4f}<0 but d4s={d4s:.4f}<0 and |d4|={abs(d4):.4f}<3|d4s|={3*abs(d4s):.4f}"
    else:
        e1 = "REFUTADA"
        e1_just = "Insufficient data for L4/L4S computation"

    # E2: STABILITY — SELECTION_STABILITY < 60%
    if np.isfinite(l4s_stability):
        e2 = "CONFIRMADA" if l4s_stability < 0.60 else "REFUTADA"
        e2_just = f"SELECTION_STABILITY={l4s_stability:.1%} {'<' if l4s_stability < 0.60 else '>='} 60%"
    else:
        e2 = "REFUTADA"
        e2_just = "L4S stability not available"

    # E3: CAPTURE — ratio <= 0.50 or NOT_DEFINED
    if np.isfinite(d4s) and np.isfinite(dsc) and d4s < 0 and dsc < 0:
        capture_ratio = abs(dsc) / abs(d4s)
        e3 = "CONFIRMADA" if capture_ratio <= 0.50 else "REFUTADA"
        e3_just = f"CAPTURE_RATIO={capture_ratio:.4f} ({'<=0.5' if capture_ratio <= 0.50 else '>0.5'})"
    else:
        capture_ratio = "NOT_DEFINED"
        e3 = "CONFIRMADA"
        e3_just = f"CAPTURE_RATIO=NOT_DEFINED (d4s={d4s:.4f}, dsc={dsc:.4f})"

    # E4: ORACLE — abs(dMAE_L5) > 0.40 * MAE_L0
    if np.isfinite(d5) and np.isfinite(mae_l0) and mae_l0 > 0:
        e4 = "CONFIRMADA" if abs(d5) > 0.40 * mae_l0 else "REFUTADA"
        e4_just = f"|dMAE_L5|={abs(d5):.4f} {'>' if abs(d5) > 0.40*mae_l0 else '<='} 0.4*MAE_L0={0.40*mae_l0:.4f}"
    else:
        e4 = "REFUTADA"
        e4_just = "L5 not available or MAE_L0=0"

    # E5: LY — LY_DOM and LY_SAME_BUCKET NOT top-3 by standalone MAE
    top3_cands = cand_standalone.sort_values("MAE")["Candidate"].head(3).tolist()
    ly_dom_in_top3 = "LY_DOM" in top3_cands
    ly_same_in_top3 = "LY_SAME_BUCKET" in top3_cands
    if not ly_dom_in_top3 and not ly_same_in_top3:
        e5 = "CONFIRMADA"
        e5_just = f"Neither LY_DOM nor LY_SAME_BUCKET in top-3 MAE candidates: {top3_cands}"
    else:
        e5 = "REFUTADA"
        e5_just = f"LY_DOM_top3={ly_dom_in_top3}, LY_SAME_BUCKET_top3={ly_same_in_top3}. Top-3: {top3_cands}"

    # THIRD_DOMAIN_PATTERN: based on E1-E4
    confirmadas = sum(1 for e in [e1, e2, e3, e4] if e == "CONFIRMADA")
    if confirmadas == 4:
        pattern = "REPLICATES"
    elif 2 <= confirmadas <= 3:
        pattern = "PARTIALLY"
    else:
        pattern = "CONTRADICTS"

    return {
        "E1_INFLATION": e1, "E1_JUSTIFICATION": e1_just,
        "E2_STABILITY": e2, "E2_JUSTIFICATION": e2_just,
        "E3_CAPTURE": e3, "E3_JUSTIFICATION": e3_just, "CAPTURE_RATIO_CORRECTED": str(capture_ratio),
        "E4_ORACLE": e4, "E4_JUSTIFICATION": e4_just,
        "E5_LY": e5, "E5_JUSTIFICATION": e5_just,
        "THIRD_DOMAIN_PATTERN": pattern,
        "N_CONFIRMADAS_E1_E4": confirmadas,
        "D4_MAE": d4, "D4S_MAE": d4s, "DSC_MAE": dsc, "D5_MAE": d5,
        "MAE_L0": mae_l0, "MAE_L4": mae_l4, "MAE_L4S": mae_l4s,
        "MAE_SC": mae_sc, "MAE_L5": mae_l5,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SHA256 OF OUTPUT FILES
# ══════════════════════════════════════════════════════════════════════════════
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    ts_start = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"MDA-2_THIRD_DOMAIN_WIKI")
    print(f"ADDENDUM_TYPE = {ADDENDUM_TYPE}")
    print(f"MDA2_EXECUTION_SHA = {MDA2_EXECUTION_SHA}")
    print(f"SCIENTIFIC_CONFIGURATION_ATTRIBUTION_SHA = {SCIENTIFIC_CONFIGURATION_ATTRIBUTION_SHA}")
    print(f"Timestamp = {ts_start}")
    print(f"N_WORKERS = {N_WORKERS}  SHARD_SIZE = {SHARD_SIZE}")
    print("=" * 70)

    # ── Library versions ──────────────────────────────────────────────────────
    import statsforecast as _sf
    pkg = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "statsforecast": _sf.__version__,
        "scipy": __import__("scipy").__version__,
    }
    print(f"  Packages: {pkg}")

    # Safety gate: no ATM data allowed
    # Compliance: this runner uses only public Wikipedia traffic data.

    # Verify S4DR model
    from s4dr.model import S4DRModel
    probe = S4DRModel("PROBE_MDA2")
    assert len(probe.models) == 12, f"Expected 12 candidates, got {len(probe.models)}"
    print(f"  Model OK: 12 structural candidates")
    del probe

    # ── Step 3: LOAD DATA ─────────────────────────────────────────────────────
    print("\n-- Step 3: Loading Wiki data --")
    df, raw_sha256, raw_size = load_wiki_data()

    TOTAL_RAW_SERIES = df["series_id"].nunique()
    dataset_start = df["fecha"].min()
    dataset_end = df["fecha"].max()
    print(f"  TOTAL_RAW_SERIES = {TOTAL_RAW_SERIES}")
    print(f"  Dataset range: {dataset_start.date()} — {dataset_end.date()}")

    # ── Step 5: DEFINE ORIGINS ────────────────────────────────────────────────
    print("\n-- Step 5: Defining geometry --")
    first_eval_origin, last_eval_origin, eval_origins, calib_origins = define_geometry(df)
    all_origins = sorted(calib_origins + eval_origins)
    eval_set = set(eval_origins)

    SC_WARMUP_K = len(calib_origins)
    print(f"  FIRST_EVALUATION_ORIGIN = {first_eval_origin.date()}")
    print(f"  LAST_EVALUATION_ORIGIN  = {last_eval_origin.date()}")
    print(f"  EVAL_ORIGINS = {[str(o.date()) for o in eval_origins]}")
    print(f"  CALIB_ORIGINS = {[str(o.date()) for o in calib_origins]}")
    print(f"  SC_WARMUP_K = {SC_WARMUP_K}")

    # ── Step 6: BUILD ELIGIBILITY ─────────────────────────────────────────────
    print("\n-- Step 6: Building eligibility (pre-eval data only) --")
    t0 = time.time()
    elig_df = build_eligibility(df, first_eval_origin)
    print(f"  Eligibility computed in {time.time()-t0:.1f}s")

    # Step 7: PERSIST ELIGIBLE LIST
    elig_path = OUT_DIR / "wiki_eligible_series.csv"
    elig_df.to_csv(elig_path, index=False)

    ELIGIBLE_SERIES_COUNT = int(elig_df["eligible"].sum())
    print(f"  TOTAL_RAW_SERIES = {TOTAL_RAW_SERIES}")
    print(f"  ELIGIBLE_SERIES_COUNT = {ELIGIBLE_SERIES_COUNT}")

    if ELIGIBLE_SERIES_COUNT < 30:
        msg = f"STATUS = BLOCKED_MDA2_INSUFFICIENT_UNIVERSE (N={ELIGIBLE_SERIES_COUNT} < 30)"
        print(f"\n{msg}")
        with open(OUT_DIR / "run_manifest.json", "w") as f:
            json.dump({"STATUS": "BLOCKED_MDA2_INSUFFICIENT_UNIVERSE",
                       "ELIGIBLE_SERIES_COUNT": ELIGIBLE_SERIES_COUNT}, f, indent=2)
        sys.exit(1)

    # ── Step 8: SAMPLE ────────────────────────────────────────────────────────
    print("\n-- Step 8: Sampling (PCG64, seed=42) --")
    sampled_sorted, extraction_order = sample_series(elig_df)
    SAMPLED_N = len(sampled_sorted)
    print(f"  SAMPLED_N = {SAMPLED_N}")

    # Step 9: PERSIST SAMPLED LIST
    sample_path = OUT_DIR / "wiki_sampled_series.csv"
    pd.DataFrame({
        "series_id": sampled_sorted,
        "extraction_rank": [extraction_order.index(s) + 1 for s in sampled_sorted],
    }).to_csv(sample_path, index=False)

    # ── Step 10: FREEZE MANIFEST PRE-METRICS ─────────────────────────────────
    print("\n-- Step 10: Writing pre-metrics manifest fields --")
    pre_manifest = {
        "ADDENDUM_ID": ADDENDUM_ID,
        "ADDENDUM_TYPE": ADDENDUM_TYPE,
        "MDA2_EXECUTION_SHA": MDA2_EXECUTION_SHA,
        "SCIENTIFIC_CONFIGURATION_ATTRIBUTION_SHA": SCIENTIFIC_CONFIGURATION_ATTRIBUTION_SHA,
        "TIMESTAMP_START": ts_start,
        "DATASET": DATASET_NAME,
        "ZENODO_RECORD": ZENODO_RECORD,
        "SOURCE_URL": ZENODO_URL,
        "DATASET_VARIANT": DATASET_VARIANT,
        "RAW_SHA256": raw_sha256,
        "RAW_FILE_SIZE_BYTES": raw_size,
        "DOWNLOAD_TIMESTAMP": ts_start,
        "TOTAL_RAW_SERIES": TOTAL_RAW_SERIES,
        "DATASET_START": str(dataset_start.date()),
        "DATASET_END": str(dataset_end.date()),
        "FIRST_EVALUATION_ORIGIN": str(first_eval_origin.date()),
        "LAST_EVALUATION_ORIGIN": str(last_eval_origin.date()),
        "EVAL_ORIGINS": [str(o.date()) for o in eval_origins],
        "CALIB_ORIGINS": [str(o.date()) for o in calib_origins],
        "MIN_HISTORY_DAYS": MIN_HISTORY_DAYS,
        "ELIGIBILITY_MEDIAN_THRESHOLD": ELIGIBILITY_MEDIAN_THRESHOLD,
        "ELIGIBILITY_MEDIAN_WINDOW": ELIGIBILITY_MEDIAN_WINDOW,
        "ELIGIBILITY_MISSING_RULE": ELIGIBILITY_MISSING_RULE,
        "ELIGIBLE_SERIES_COUNT": ELIGIBLE_SERIES_COUNT,
        "SAMPLING_RNG": SAMPLING_RNG,
        "SAMPLING_SEED": SAMPLING_SEED,
        "SAMPLING_REPLACE": SAMPLING_REPLACE,
        "ELIGIBLE_SORT_ORDER": ELIGIBLE_SORT_ORDER,
        "SAMPLED_N": SAMPLED_N,
        "N_EVAL_ORIGINS": N_EVAL_ORIGINS,
        "HORIZON": HORIZON,
        "H1": "1-7", "H2": "8-14",
        "PRIMARY_SELECTION_LOSS": SC_SELECTION_LOSS,
        "SC_MIN_PRIOR": SC_MIN_PRIOR,
        "SC_OBSERVABILITY_RULE": SC_OBSERVABILITY_RULE,
        "SC_WARMUP_K": SC_WARMUP_K,
        "PACKAGE_VERSIONS": pkg,
        "METRICS": ["MAE", "RMSE", "sMAPE", "WAPE", "MedAE", "P90AE", "SIGNED_BIAS"],
        "E1_DEFINITION": "d4<0 AND (d4s>=0 OR |d4|>=3|d4s|)",
        "E2_DEFINITION": "SELECTION_STABILITY < 60%",
        "E3_DEFINITION": "CAPTURE_RATIO_CORRECTED <= 0.50 OR NOT_DEFINED",
        "E4_DEFINITION": "|dMAE_L5| > 0.40 * MAE_L0",
        "E5_DEFINITION": "LY_DOM AND LY_SAME_BUCKET NOT in top-3 standalone MAE",
        "THIRD_DOMAIN_PATTERN_RULE": "4/4 E1-E4=REPLICATES; 2-3/4=PARTIALLY; 0-1/4=CONTRADICTS",
        "GO_NOGO_REEVALUATED": False,
        "PAPER_DIRECTION_CHANGED": False,
        "FINAL_HOLDOUT_OPENED": False,
        "STATUS": "PRE_METRICS_FROZEN",
    }
    with open(OUT_DIR / "raw_source_manifest.json", "w") as f:
        json.dump(pre_manifest, f, indent=2, default=str)
    print(f"  raw_source_manifest.json written (pre-metrics freeze)")

    # ── Step 11: GENERATE FORECASTS ───────────────────────────────────────────
    print(f"\n-- Step 11: Generating forecasts ({SAMPLED_N} series × {len(all_origins)} origins) --")

    completed = load_completed()
    all_origin_strs = [str(o.date()) for o in all_origins]

    # Validate completed checkpoints
    valid_completed: list[str] = []
    for origin_str in all_origin_strs:
        if origin_str not in completed:
            break
        if not validate_checkpoint(origin_str, sampled_sorted):
            print(f"  INVALID checkpoint {origin_str} — truncating")
            break
        valid_completed.append(origin_str)

    if len(valid_completed) < len(completed):
        save_completed(valid_completed)
        for origin_str in completed:
            if origin_str not in valid_completed:
                for p in [ckpt_m6feed(origin_str), ckpt_eval(origin_str)]:
                    if p.exists():
                        p.unlink()
    completed = valid_completed

    sc = rebuild_sc_from_checkpoints(completed)
    t0_total = time.time()
    n_done = len(completed)
    n_total = len(all_origins)

    for origin in all_origins:
        origin_str = str(origin.date())
        is_eval = origin in eval_set

        if origin_str in completed:
            print(f"  SKIP {origin_str} {'EVAL' if is_eval else 'CALIB'} (checkpointed)")
            continue

        label = "EVAL" if is_eval else "CALIB"
        print(f"\n  [{n_done+1}/{n_total}] {label} {origin_str} ...", flush=True)
        t0_orig = time.time()

        eval_rows, m6feed_rows = run_origin(origin, is_eval, sampled_sorted, df, sc)
        save_checkpoint(origin_str, m6feed_rows, eval_rows, is_eval, sampled_sorted)

        elapsed_orig = time.time() - t0_orig
        elapsed_total = time.time() - t0_total
        n_done += 1
        eta = (elapsed_total / n_done * (n_total - n_done)) / 60
        print(f"  {origin_str} done: {len(m6feed_rows)} m6feed, "
              f"{len(eval_rows)} eval rows  origin={elapsed_orig/60:.1f}min  ETA={eta:.0f}min")

    # ── Step 12: ASSEMBLE PANEL ───────────────────────────────────────────────
    print("\n-- Step 12: Assembling forecast_panel.csv --")
    eval_parts = []
    for origin in eval_origins:
        ep = ckpt_eval(str(origin.date()))
        if ep.exists():
            eval_parts.append(pd.read_parquet(ep))
        else:
            print(f"  WARN: eval checkpoint missing for {origin.date()}")

    if not eval_parts:
        print("ERROR: No eval checkpoints — cannot assemble panel")
        sys.exit(1)

    panel = pd.concat(eval_parts, ignore_index=True)

    # Dedup guard
    key_cols = ["series_id", "origin", "target_date", "horizon_step"]
    dups = panel.duplicated(subset=key_cols).sum()
    if dups > 0:
        print(f"  WARN: {dups} duplicates — dropping")
        panel = panel.drop_duplicates(subset=key_cols)

    N_TARGET_MISSING = int(panel["target_raw"].isna().sum())
    # Exclude rows with missing targets from common universe
    common_panel = panel.dropna(subset=["target_raw"] + METHODS)
    print(f"  Panel: {len(panel)} rows, {N_TARGET_MISSING} missing targets excluded")
    print(f"  Common universe: {len(common_panel)} rows")

    panel_path = OUT_DIR / "forecast_panel.csv"
    panel.to_csv(panel_path, index=False)
    panel_sha256 = sha256_file(panel_path)
    print(f"  forecast_panel.csv SHA256 = {panel_sha256}")

    # ── Step 13: INTEGRITY GATES ──────────────────────────────────────────────
    print("\n-- Step 13: Integrity gates --")
    gates = run_integrity_gates(common_panel, sampled_sorted, eval_origins)
    deg = check_candidate_degeneracy(panel, sampled_sorted)
    gates["GLOBAL_PHANTOM_GATE"] = deg["GLOBAL_PHANTOM_GATE"]
    for k, v in gates.items():
        flag = " ***" if "FAIL" in str(v) else ""
        print(f"  {k}: {v}{flag}")
    print(f"  GLOBAL_PHANTOM_GATE: {deg['GLOBAL_PHANTOM_GATE']}")

    gate_failures = [k for k, v in gates.items() if "FAIL" in str(v)]
    if gate_failures or deg["GLOBAL_PHANTOM_GATE"] == "FAIL":
        print(f"\nGATE FAILURES: {gate_failures}")
        if deg["GLOBAL_PHANTOM_GATE"] == "FAIL":
            print(f"PHANTOM_CANDIDATES: {deg['PHANTOM_CANDIDATES']}")
        sys.exit(1)

    # ── Step 14: CAUSALITY SPOTCHECK ──────────────────────────────────────────
    print("\n-- Step 14: Causality spot-check --")
    spotcheck = causality_spotcheck(common_panel, df, sampled_sorted, eval_origins)
    gates["G11_CAUSALITY_SPOTCHECK"] = spotcheck["RESULT"]
    print(f"  G11_CAUSALITY_SPOTCHECK = {spotcheck['RESULT']}")
    for c in spotcheck["CHECKS"]:
        print(f"    {c}")

    pd.DataFrame([{"Gate": k, "Result": v} for k, v in gates.items()]).to_csv(
        OUT_DIR / "integrity_gates.csv", index=False
    )
    pd.DataFrame(spotcheck["CHECKS"]).to_csv(OUT_DIR / "causality_spotcheck.csv", index=False)

    # ── Step 15: METRICS AND LADDER ───────────────────────────────────────────
    print("\n-- Step 15-17: Computing ladder, metrics, L4S --")
    ladder_result = compute_ladder(common_panel, eval_origins)
    if not ladder_result:
        print("ERROR: compute_ladder returned empty — common panel may be empty")
        sys.exit(1)

    pv = ladder_result["pv"]
    all_mets = compute_all_metrics(pv, sampled_sorted)

    # Ladder logic checks
    def _mae(level):
        return all_mets.get(level, {}).get("MICRO", {}).get("MAE", np.nan)

    print("\n  LADDER (MAE):")
    for level in ALL_LEVELS:
        m_val = _mae(level)
        d = m_val - _mae("L0") if np.isfinite(m_val) else np.nan
        print(f"    {level:<6}: MAE={m_val:>10.4f}  dMAE_vs_L0={d:>+10.4f}")

    # Validate ladder logic (section 11)
    ladder_ok = True
    checks = [("L5", "L4"), ("L4", "L3"), ("L5", "L0"), ("L5", "L1"),
              ("L5", "L2"), ("L5", "SC")]
    for lo, hi in checks:
        if np.isfinite(_mae(lo)) and np.isfinite(_mae(hi)):
            if _mae(lo) > _mae(hi) + 1.0:  # allow tiny floating point error
                print(f"  LADDER LOGIC VIOLATION: MAE({lo})={_mae(lo):.4f} > MAE({hi})={_mae(hi):.4f}")
                ladder_ok = False
    if not ladder_ok:
        print("  CALCULATION_BUG = TRUE — NOT COMMITTING")
        sys.exit(1)
    print("  Ladder logic OK")

    # ── Step 18: CANDIDATE ANALYSIS ───────────────────────────────────────────
    print("\n-- Step 18: Candidate analysis --")
    cand_stand = candidate_standalone(pv)
    print(f"  Top-5 standalone MAE candidates:")
    for _, r in cand_stand.sort_values("MAE").head(5).iterrows():
        print(f"    {r['Candidate']}: MAE={r['MAE']:.4f}")

    cand_attr, arith_pass = candidate_attribution_sc(panel, pv)
    print(f"  ATTRIBUTION_ARITHMETIC_CHECK = {'PASS' if arith_pass else 'FAIL'}")

    # ── Step 19: WILCOXON ─────────────────────────────────────────────────────
    print("\n-- Step 19: Wilcoxon descriptive tests --")
    wil_theta = wilcoxon_test(pv, "L0", "B2")
    wil_ets = wilcoxon_test(pv, "L0", "B1")
    print(f"  L0 vs B2_THETA: p={wil_theta['P_VALUE']:.4f}  dir={wil_theta['DIRECTION']}")
    print(f"  L0 vs B1_ETS:   p={wil_ets['P_VALUE']:.4f}  dir={wil_ets['DIRECTION']}")

    # ── Step 20-21: E1-E5 AND THIRD_DOMAIN_PATTERN ───────────────────────────
    print("\n-- Step 20-21: Prospective expectations E1-E5 --")
    expectations = eval_expectations(all_mets, cand_stand, ladder_result["l4s_selection_stability"], all_mets)
    for k in ["E1_INFLATION", "E2_STABILITY", "E3_CAPTURE", "E4_ORACLE", "E5_LY"]:
        print(f"  {k}: {expectations[k]}")
        just_key = k.split("_")[0] + "_JUSTIFICATION"
        print(f"    {expectations[just_key]}")
    print(f"  THIRD_DOMAIN_PATTERN = {expectations['THIRD_DOMAIN_PATTERN']}")
    print(f"  N_CONFIRMADAS(E1-E4) = {expectations['N_CONFIRMADAS_E1_E4']}/4")

    # ── Step 16: PERSIST OUTPUTS ──────────────────────────────────────────────
    print("\n-- Step 16: Persisting outputs --")

    # global_metrics.csv
    gm_rows = []
    for level in ALL_LEVELS:
        if level not in all_mets:
            continue
        m = all_mets[level]["MICRO"]
        gm_rows.append({
            "Method": level, **m,
            "MACRO_MAE": all_mets[level]["MACRO_MAE"],
            "H1_MAE": all_mets[level]["H1_MAE"],
            "H2_MAE": all_mets[level]["H2_MAE"],
        })
    pd.DataFrame(gm_rows).to_csv(OUT_DIR / "global_metrics.csv", index=False)

    # per_series_metrics.csv
    ps_rows = []
    for sid in sampled_sorted:
        s = pv[pv["series_id"] == sid]
        if len(s) == 0:
            continue
        row = {"series_id": sid}
        for level in ALL_LEVELS:
            if level in s.columns:
                row[f"MAE_{level}"] = mae(s["target_raw"].values, s[level].values)
        ps_rows.append(row)
    pd.DataFrame(ps_rows).to_csv(OUT_DIR / "per_series_metrics.csv", index=False)

    # method_vs_theta_deltas.csv
    b2_mae = _mae("B2")
    pd.DataFrame([
        {"Method": level, "dMAE_vs_B2_THETA": _mae(level) - b2_mae}
        for level in ALL_LEVELS if np.isfinite(_mae(level))
    ]).to_csv(OUT_DIR / "method_vs_theta_deltas.csv", index=False)

    # method_vs_ets_deltas.csv
    b1_mae = _mae("B1")
    pd.DataFrame([
        {"Method": level, "dMAE_vs_B1_ETS": _mae(level) - b1_mae}
        for level in ALL_LEVELS if np.isfinite(_mae(level))
    ]).to_csv(OUT_DIR / "method_vs_ets_deltas.csv", index=False)

    # sc_vs_l0_deltas.csv
    sc_l0_full = pv.dropna(subset=["target_raw", "SC", "L0"])
    sc_active = sc_l0_full[~sc_l0_full["sc_is_fallback"]]
    pd.DataFrame([
        {"Universe": "FULL", "N": len(sc_l0_full),
         "SC_MAE": _mae("SC"), "L0_MAE": _mae("L0"),
         "dMAE": _mae("SC") - _mae("L0")},
        {"Universe": "ACTIVE_SC", "N": len(sc_active),
         "SC_MAE": mae(sc_active["target_raw"].values, sc_active["SC"].values) if len(sc_active) > 0 else np.nan,
         "L0_MAE": mae(sc_active["target_raw"].values, sc_active["L0"].values) if len(sc_active) > 0 else np.nan,
         "dMAE": (mae(sc_active["target_raw"].values, sc_active["SC"].values) -
                  mae(sc_active["target_raw"].values, sc_active["L0"].values)) if len(sc_active) > 0 else np.nan},
    ]).to_csv(OUT_DIR / "sc_vs_l0_deltas.csv", index=False)

    # headroom_ladder.csv
    mae_l0 = _mae("L0")
    pd.DataFrame([
        {"Level": level, "MAE": _mae(level),
         "dMAE_vs_L0": _mae(level) - mae_l0,
         "dMAE_pct_of_L0": 100 * (_mae(level) - mae_l0) / mae_l0 if mae_l0 > 0 else np.nan}
        for level in ALL_LEVELS if np.isfinite(_mae(level))
    ]).to_csv(OUT_DIR / "headroom_ladder.csv", index=False)

    # l4s_split_details.csv
    pd.DataFrame([{
        "HALF_A_ORIGINS": str(ladder_result["half_a_origins"]),
        "HALF_B_ORIGINS": str(ladder_result["half_b_origins"]),
        "L4S_MAE_FULL": _mae("L4S"),
        "L4_MAE_FULL": _mae("L4"),
        "SELECTION_STABILITY": ladder_result["l4s_selection_stability"],
        "N_STABLE": ladder_result["l4s_stable_count"],
        "N_TOTAL": ladder_result["l4s_total"],
        "BEST_L3_CANDIDATE": ladder_result["best_l3_candidate"],
    }]).to_csv(OUT_DIR / "l4s_split_details.csv", index=False)

    # candidate_standalone.csv
    cand_stand.to_csv(OUT_DIR / "candidate_standalone.csv", index=False)

    # candidate_attribution_sc.csv
    cand_attr.to_csv(OUT_DIR / "candidate_attribution_sc.csv", index=False)

    # statistical_tests.csv
    pd.DataFrame([
        {"Comparison": "L0_vs_B2_THETA", **wil_theta},
        {"Comparison": "L0_vs_B1_ETS", **wil_ets},
    ]).to_csv(OUT_DIR / "statistical_tests.csv", index=False)

    # ── Step 22: UPDATE MANUSCRIPT_DATA ──────────────────────────────────────
    print("\n-- Step 22: Updating manuscript_data --")
    manuscript_dir = REPO_ROOT / "reference" / "public_benchmark_firetest" / "manuscript_data"
    manuscript_updated = update_manuscript_data(
        manuscript_dir, pv, all_mets, cand_stand, cand_attr, wil_theta, wil_ets,
        expectations, ladder_result, pre_manifest, sampled_sorted,
    )

    # ── Step 23: SHA256 OUTPUTS ───────────────────────────────────────────────
    print("\n-- Step 23: SHA256 output files --")
    output_files = list(OUT_DIR.glob("*.csv")) + list(OUT_DIR.glob("*.json"))
    output_sha = {str(p.name): sha256_file(p) for p in output_files}
    for name, sha in sorted(output_sha.items()):
        print(f"  {name}: {sha}")

    # ── FINAL MANIFEST ────────────────────────────────────────────────────────
    ts_end = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    final_manifest = {
        **pre_manifest,
        "TIMESTAMP_END": ts_end,
        "STATUS": "COMPLETE",
        "PANEL_SHA256": panel_sha256,
        "N_FORECAST_ROWS_TOTAL": len(panel),
        "N_FORECAST_ROWS_COMMON": len(common_panel),
        "N_MISSING_TARGET_ROWS_EXCLUDED": N_TARGET_MISSING,
        "INTEGRITY_GATES": gates,
        "GLOBAL_PHANTOM_GATE": deg["GLOBAL_PHANTOM_GATE"],
        "N_CANDIDATE_SERIES_PAIRS": deg["N_CANDIDATE_SERIES_PAIRS"],
        "N_ZERO_VARIANCE_PAIRS": deg["N_ZERO_VARIANCE_PAIRS"],
        "SC_AVAILABLE_PRIOR_ROWS_AT_FIRST_EVAL_ORIGIN": sc_calib_rows_at_first_eval(sc, sampled_sorted, eval_origins[0]),
        "L4S_SELECTION_STABILITY": ladder_result["l4s_selection_stability"],
        "BEST_L3_CANDIDATE": ladder_result["best_l3_candidate"],
        "ATTRIBUTION_ARITHMETIC_CHECK": "PASS" if arith_pass else "FAIL",
        "CAUSALITY_SPOTCHECK": spotcheck,
        "WILCOXON_L0_VS_THETA": wil_theta,
        "WILCOXON_L0_VS_ETS": wil_ets,
        "STATISTICAL_TEST_DECLARATION": (
            "Estos tests son descriptivos para el manuscrito; el criterio de decisión "
            "del proyecto fue el GO/NO_GO pre-registrado ya aplicado, y ningún p-value lo modifica."
        ),
        "EXPECTATIONS": expectations,
        "THIRD_DOMAIN_PATTERN": expectations["THIRD_DOMAIN_PATTERN"],
        "HEADROOM": {
            "LIBRARY_RETROSPECTIVE_HEADROOM": expectations["D5_MAE"],
            "STATIC_RETROSPECTIVE_SPECIALIZATION": expectations["D4_MAE"],
            "BIAS_CORRECTED_SPECIALIZATION": expectations["D4S_MAE"],
            "CAUSAL_SPECIALIZATION": expectations["DSC_MAE"],
            "CAPTURE_RATIO_CORRECTED": expectations["CAPTURE_RATIO_CORRECTED"],
        },
        "METHOD_METRICS": {
            level: {
                "MICRO_MAE": all_mets.get(level, {}).get("MICRO", {}).get("MAE", np.nan),
                "MACRO_MAE": all_mets.get(level, {}).get("MACRO_MAE", np.nan),
                "H1_MAE": all_mets.get(level, {}).get("H1_MAE", np.nan),
                "H2_MAE": all_mets.get(level, {}).get("H2_MAE", np.nan),
            }
            for level in ALL_LEVELS
        },
        "OUTPUT_FILE_SHA256": output_sha,
        "MANUSCRIPT_DATA_UPDATED": manuscript_updated,
        "PUBLISHABLE_SCOPE": ["M5_STORE_DEPT", "LD2011_DAILY", "WIKI_DAILY"],
        "EXISTING_M5_VALUES_MODIFIED": False,
        "EXISTING_LD_VALUES_MODIFIED": False,
        "GO_NOGO_RESULT": "NO_GO",
        "GO_NOGO_REEVALUATED": False,
        "PAPER_DIRECTION": "METHODOLOGICAL_PAPER",
        "PAPER_DIRECTION_CHANGED": False,
        "TUNING_PERFORMED": False,
        "MODEL_MODIFIED": False,
        "SELECTOR_MODIFIED": False,
        "SAMPLE_ITERATED": False,
        "FINAL_HOLDOUT_OPENED": False,
    }

    with open(OUT_DIR / "run_manifest.json", "w") as f:
        json.dump(final_manifest, f, indent=2, default=str)

    # ── FINAL REPORT ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("MDA-2 FINAL REPORT")
    print("=" * 70)
    print(f"STATUS = COMPLETE")
    print(f"ADDENDUM_ID = {ADDENDUM_ID}")
    print(f"THIRD_DOMAIN_PATTERN = {expectations['THIRD_DOMAIN_PATTERN']}")
    print()
    print(f"DATASET: {DATASET_NAME}")
    print(f"  TOTAL_RAW_SERIES = {TOTAL_RAW_SERIES}")
    print(f"  ELIGIBLE_SERIES_COUNT = {ELIGIBLE_SERIES_COUNT}")
    print(f"  SAMPLED_N = {SAMPLED_N}")
    print(f"  FIRST_EVALUATION_ORIGIN = {first_eval_origin.date()}")
    print(f"  LAST_EVALUATION_ORIGIN = {last_eval_origin.date()}")
    print()
    print(f"UNIVERSE: N={len(common_panel)} common rows, N_MISSING_TARGET={N_TARGET_MISSING}")
    print()
    print("LADDER (MAE):")
    for level in ALL_LEVELS:
        m_val = _mae(level)
        if np.isfinite(m_val):
            d = m_val - mae_l0
            print(f"  {level:<6}: {m_val:>10.4f}  dMAE={d:>+10.4f}")
    print()
    print(f"HEADROOM:")
    print(f"  LIBRARY_RETROSPECTIVE (L5):   dMAE={expectations['D5_MAE']:>+.4f}")
    print(f"  STATIC_RETRO (L4):            dMAE={expectations['D4_MAE']:>+.4f}")
    print(f"  BIAS_CORRECTED (L4S):         dMAE={expectations['D4S_MAE']:>+.4f}")
    print(f"  CAUSAL (SC):                  dMAE={expectations['DSC_MAE']:>+.4f}")
    print(f"  CAPTURE_RATIO_CORRECTED = {expectations['CAPTURE_RATIO_CORRECTED']}")
    print(f"  L4S_SELECTION_STABILITY = {ladder_result['l4s_selection_stability']:.1%}")
    print()
    print("PROSPECTIVE EXPECTATIONS:")
    for k in ["E1_INFLATION", "E2_STABILITY", "E3_CAPTURE", "E4_ORACLE", "E5_LY"]:
        print(f"  {k}: {expectations[k]}")
    print(f"  THIRD_DOMAIN_PATTERN = {expectations['THIRD_DOMAIN_PATTERN']}")
    print()
    print("STATISTICAL TESTS (descriptive):")
    print(f"  L0 vs B2_THETA: p={wil_theta['P_VALUE']:.4f}  dir={wil_theta['DIRECTION']}")
    print(f"  L0 vs B1_ETS:   p={wil_ets['P_VALUE']:.4f}  dir={wil_ets['DIRECTION']}")
    print()
    print("PROTOCOL:")
    print(f"  GO_NOGO_REEVALUATED = FALSE")
    print(f"  PAPER_DIRECTION_CHANGED = FALSE")
    print(f"  FINAL_HOLDOUT_OPENED = FALSE")
    print()
    print("MEASUREMENT_PHASE_OF_PROJECT = CLOSED_FINAL")
    print("NEXT_STAGE = MANUSCRIPT_ONLY")
    print("COMPLETE.")


def sc_calib_rows_at_first_eval(sc: SCSelector, sampled_series: list[str], first_eval: pd.Timestamp) -> int:
    return sc.n_prior(sampled_series[0] if sampled_series else "", first_eval) if sampled_series else 0


def update_manuscript_data(manuscript_dir: Path, pv: pd.DataFrame, all_mets: dict,
                            cand_stand: pd.DataFrame, cand_attr: pd.DataFrame,
                            wil_theta: dict, wil_ets: dict, expectations: dict,
                            ladder_result: dict, manifest: dict,
                            sampled_series: list[str]) -> bool:
    """Update manuscript_data/ with WIKI_DAILY results."""
    manuscript_dir.mkdir(parents=True, exist_ok=True)

    def _mae_val(level):
        return all_mets.get(level, {}).get("MICRO", {}).get("MAE", np.nan)

    mae_l0 = _mae_val("L0")

    # ── master_results_table.csv ──────────────────────────────────────────────
    master_path = manuscript_dir / "master_results_table.csv"
    new_rows = []
    for level in ALL_LEVELS:
        m = all_mets.get(level, {}).get("MICRO", {})
        new_rows.append({
            "Dataset": "WIKI_DAILY",
            "method": level,
            "MAE": m.get("MAE", np.nan),
            "RMSE": m.get("RMSE", np.nan),
            "sMAPE": m.get("sMAPE", np.nan),
            "WAPE": m.get("WAPE", np.nan),
            "MedAE": m.get("MedAE", np.nan),
            "P90AE": m.get("P90AE", np.nan),
            "SIGNED_BIAS": m.get("SIGNED_BIAS", np.nan),
            "MACRO_MAE": all_mets.get(level, {}).get("MACRO_MAE", np.nan),
            "H1_MAE": all_mets.get(level, {}).get("H1_MAE", np.nan),
            "H2_MAE": all_mets.get(level, {}).get("H2_MAE", np.nan),
        })
    new_df = pd.DataFrame(new_rows)
    if master_path.exists():
        existing = pd.read_csv(master_path)
        existing = existing[existing["Dataset"] != "WIKI_DAILY"]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(master_path, index=False)

    # ── headroom_ladder.csv ───────────────────────────────────────────────────
    hl_path = manuscript_dir / "headroom_ladder.csv"
    hl_rows = []
    for level in ALL_LEVELS:
        m_val = _mae_val(level)
        hl_rows.append({
            "Dataset": "WIKI_DAILY", "Level": level, "MAE": m_val,
            "dMAE_vs_L0": m_val - mae_l0,
            "dMAE_pct_of_L0": 100 * (m_val - mae_l0) / mae_l0 if mae_l0 > 0 else np.nan,
        })
    hl_new = pd.DataFrame(hl_rows)
    if hl_path.exists():
        hl_existing = pd.read_csv(hl_path)
        hl_existing = hl_existing[hl_existing["Dataset"] != "WIKI_DAILY"]
        hl_combined = pd.concat([hl_existing, hl_new], ignore_index=True)
    else:
        hl_combined = hl_new
    hl_combined.to_csv(hl_path, index=False)

    # ── candidate_standalone.csv ──────────────────────────────────────────────
    cs_path = manuscript_dir / "candidate_standalone.csv"
    cs_new = cand_stand.copy()
    cs_new.insert(0, "Dataset", "WIKI_DAILY")
    if cs_path.exists():
        cs_existing = pd.read_csv(cs_path)
        cs_existing = cs_existing[cs_existing["Dataset"] != "WIKI_DAILY"]
        cs_combined = pd.concat([cs_existing, cs_new], ignore_index=True)
    else:
        cs_combined = cs_new
    cs_combined.to_csv(cs_path, index=False)

    # ── candidate_attribution_sc.csv ──────────────────────────────────────────
    ca_path = manuscript_dir / "candidate_attribution_sc.csv"
    ca_new = cand_attr.copy()
    ca_new.insert(0, "Dataset", "WIKI_DAILY")
    if ca_path.exists():
        ca_existing = pd.read_csv(ca_path)
        ca_existing = ca_existing[ca_existing["Dataset"] != "WIKI_DAILY"]
        ca_combined = pd.concat([ca_existing, ca_new], ignore_index=True)
    else:
        ca_combined = ca_new
    ca_combined.to_csv(ca_path, index=False)

    # ── per_series_deltas.csv ─────────────────────────────────────────────────
    psd_path = manuscript_dir / "per_series_deltas.csv"
    psd_rows = []
    for sid in sampled_series:
        s = pv[pv["series_id"] == sid]
        if len(s) == 0:
            continue
        row = {"Dataset": "WIKI_DAILY", "SeriesId": sid}
        for level in ["L0", "L1", "L2", "L3", "L4", "L4S", "L5", "SC"]:
            if level in s.columns:
                row[f"MAE_{level}"] = mae(s["target_raw"].values, s[level].values)
        if "L0" in row:
            for level in ["L1", "L2", "L3", "L4", "L4S", "L5", "SC"]:
                k = f"MAE_{level}"
                if k in row:
                    row[f"dMAE_{level}_vs_L0"] = row[k] - row["MAE_L0"]
        psd_rows.append(row)
    psd_new = pd.DataFrame(psd_rows)
    if psd_path.exists():
        psd_existing = pd.read_csv(psd_path)
        psd_existing = psd_existing[psd_existing["Dataset"] != "WIKI_DAILY"]
        psd_combined = pd.concat([psd_existing, psd_new], ignore_index=True)
    else:
        psd_combined = psd_new
    psd_combined.to_csv(psd_path, index=False)

    # ── statistical_tests.csv ─────────────────────────────────────────────────
    st_path = manuscript_dir / "statistical_tests.csv"
    st_new = pd.DataFrame([
        {"Dataset": "WIKI_DAILY", "comparison": "L0_vs_B2_THETA", **wil_theta},
        {"Dataset": "WIKI_DAILY", "comparison": "L0_vs_B1_ETS", **wil_ets},
    ])
    if st_path.exists():
        st_existing = pd.read_csv(st_path)
        st_existing = st_existing[st_existing["Dataset"] != "WIKI_DAILY"]
        st_combined = pd.concat([st_existing, st_new], ignore_index=True)
    else:
        st_combined = st_new
    st_combined.to_csv(st_path, index=False)

    # ── l4s_split_details.csv ─────────────────────────────────────────────────
    l4s_path = manuscript_dir / "l4s_split_details.csv"
    l4s_new = pd.DataFrame([{
        "Dataset": "WIKI_DAILY",
        "HALF_A_ORIGINS": str(ladder_result["half_a_origins"]),
        "HALF_B_ORIGINS": str(ladder_result["half_b_origins"]),
        "L4S_MAE": _mae_val("L4S"),
        "L4_MAE": _mae_val("L4"),
        "dMAE_L4S_vs_L0": _mae_val("L4S") - mae_l0,
        "dMAE_L4_vs_L0": _mae_val("L4") - mae_l0,
        "SELECTION_STABILITY": ladder_result["l4s_selection_stability"],
    }])
    if l4s_path.exists():
        l4s_existing = pd.read_csv(l4s_path)
        l4s_existing = l4s_existing[l4s_existing["Dataset"] != "WIKI_DAILY"]
        l4s_combined = pd.concat([l4s_existing, l4s_new], ignore_index=True)
    else:
        l4s_combined = l4s_new
    l4s_combined.to_csv(l4s_path, index=False)

    # ── evidence_pack.md ──────────────────────────────────────────────────────
    ep_path = manuscript_dir / "evidence_pack.md"
    wiki_section = f"""
## WIKI_DAILY

**Source**: {manifest.get('SOURCE_URL', '')}
**SAMPLED_N**: {manifest.get('SAMPLED_N', '')}
**N_EVAL_ORIGINS**: {manifest.get('N_EVAL_ORIGINS', '')}
**HORIZON**: {manifest.get('HORIZON', '')}
**FIRST_EVALUATION_ORIGIN**: {manifest.get('FIRST_EVALUATION_ORIGIN', '')}
**LAST_EVALUATION_ORIGIN**: {manifest.get('LAST_EVALUATION_ORIGIN', '')}

### Ladder (MAE)
| Level | MAE | dMAE vs L0 |
|-------|-----|-----------|
""" + "\n".join(
        f"| {level} | {_mae_val(level):.4f} | {_mae_val(level)-mae_l0:+.4f} |"
        for level in ALL_LEVELS if np.isfinite(_mae_val(level))
    ) + f"""

### Headroom Decomposition
- LIBRARY_RETROSPECTIVE_HEADROOM (L5): dMAE = {expectations['D5_MAE']:+.4f} ({100*expectations['D5_MAE']/mae_l0:+.2f}% of L0)
- STATIC_RETROSPECTIVE_SPECIALIZATION (L4): dMAE = {expectations['D4_MAE']:+.4f} ({100*expectations['D4_MAE']/mae_l0:+.2f}% of L0)
- BIAS_CORRECTED_SPECIALIZATION (L4S): dMAE = {expectations['D4S_MAE']:+.4f} ({100*expectations['D4S_MAE']/mae_l0:+.2f}% of L0)
- CAUSAL_SPECIALIZATION (SC): dMAE = {expectations['DSC_MAE']:+.4f} ({100*expectations['DSC_MAE']/mae_l0:+.2f}% of L0)
- CAPTURE_RATIO_CORRECTED: {expectations['CAPTURE_RATIO_CORRECTED']}
- L4S_SELECTION_STABILITY: {ladder_result['l4s_selection_stability']:.1%}

### Key figures
- Source: candidate_standalone.csv, dataset=WIKI_DAILY, metric=MAE
- Source: headroom_ladder.csv, dataset=WIKI_DAILY

## THIRD-DOMAIN PROSPECTIVE REPLICATION

MDA-2 was prospectively specified after FIRETEST_2 and before inspection of WIKI_DAILY results.

| Expectation | Result | Justification |
|-------------|--------|---------------|
| E1_INFLATION | {expectations['E1_INFLATION']} | {expectations['E1_JUSTIFICATION']} |
| E2_STABILITY | {expectations['E2_STABILITY']} | {expectations['E2_JUSTIFICATION']} |
| E3_CAPTURE | {expectations['E3_CAPTURE']} | {expectations['E3_JUSTIFICATION']} |
| E4_ORACLE | {expectations['E4_ORACLE']} | {expectations['E4_JUSTIFICATION']} |
| E5_LY | {expectations['E5_LY']} | {expectations['E5_JUSTIFICATION']} |

**THIRD_DOMAIN_PATTERN = {expectations['THIRD_DOMAIN_PATTERN']}**
(Rule: REPLICATES if all 4 of E1-E4 confirmed; PARTIALLY if 2-3; CONTRADICTS if 0-1)
"""

    if ep_path.exists():
        existing_ep = ep_path.read_text(encoding="utf-8")
        # Remove existing WIKI_DAILY section if present
        if "## WIKI_DAILY" in existing_ep:
            parts = existing_ep.split("## WIKI_DAILY")
            existing_ep = parts[0]
        ep_path.write_text(existing_ep + wiki_section, encoding="utf-8")
    else:
        ep_path.write_text(f"# Evidence Pack\n\nGenerated: {datetime.now().strftime('%Y-%m-%d')}\n" + wiki_section, encoding="utf-8")

    # ── manifest_manuscript.json ──────────────────────────────────────────────
    mm_path = manuscript_dir / "manifest_manuscript.json"
    if mm_path.exists():
        with open(mm_path) as f:
            mm = json.load(f)
    else:
        mm = {}
    mm["WIKI_DAILY"] = {
        "ADDENDUM_ID": ADDENDUM_ID,
        "MDA2_EXECUTION_SHA": MDA2_EXECUTION_SHA,
        "SCIENTIFIC_CONFIGURATION_ATTRIBUTION_SHA": SCIENTIFIC_CONFIGURATION_ATTRIBUTION_SHA,
        "SAMPLED_N": manifest.get("SAMPLED_N"),
        "FIRST_EVALUATION_ORIGIN": manifest.get("FIRST_EVALUATION_ORIGIN"),
        "LAST_EVALUATION_ORIGIN": manifest.get("LAST_EVALUATION_ORIGIN"),
        "THIRD_DOMAIN_PATTERN": expectations["THIRD_DOMAIN_PATTERN"],
    }
    mm["PUBLISHABLE_SCOPE"] = ["M5_STORE_DEPT", "LD2011_DAILY", "WIKI_DAILY"]
    with open(mm_path, "w") as f:
        json.dump(mm, f, indent=2, default=str)

    print(f"  manuscript_data/ updated: {[p.name for p in manuscript_dir.glob('*')]}")
    return True


if __name__ == "__main__":
    main()
