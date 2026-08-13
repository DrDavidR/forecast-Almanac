"""
S4DR_MANUSCRIPT_MEASUREMENT_CLOSURE
EXPERIMENT_ID  = S4DR_MANUSCRIPT_MEASUREMENT_CLOSURE
FROZEN_SHA     = 67851d3
SOURCE         = reference/public_benchmark_firetest/ (persisted files only)

PROHIBITED:
  - S4DR execution, new forecasts, model modification
  - Rerunning baselines or M6/SC
  - Using non-public operational data
  - Evaluating future experimental configurations
  - Improvising missing numbers; report as MANUSCRIPT_DATA_ADDITION_REQUEST

NOMENCLATURE:
  L0  = BL1_S4DR           (CAUSAL, DEPLOYABLE)
  L1  = UNIFORM_MEAN_12    (RETROSPECTIVE, NOT_DEPLOYABLE)
  L2  = MEDIAN_12          (RETROSPECTIVE, NOT_DEPLOYABLE)
  L3  = BEST_GLOBAL_RETRO  (RETROSPECTIVE, NOT_DEPLOYABLE)
  L4  = BEST_STATIC_RETRO  (RETROSPECTIVE, NOT_DEPLOYABLE)
  L4S = BEST_STATIC_SPLIT  (RETROSPECTIVE cross-fit, NOT_DEPLOYABLE)
  L5  = PER_ORIGIN_ORACLE  (RETROSPECTIVE per-row oracle, NOT_DEPLOYABLE)
  SC  = M6_PRED            (CAUSAL, NON_CANONICAL — already in panel)
  B0  = SNAIVE7 / B1=ETS / B2=THETA / B3=MSTL_AutoARIMA
"""
from __future__ import annotations
import json
import warnings
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent
MANUSCRIPT_DIR = BASE_DIR / "manuscript_data"
OUT_M5         = BASE_DIR / "m5_store_dept"
OUT_LD         = BASE_DIR / "ld2011_daily"
CK_LD          = OUT_LD / "_checkpoints"

MANUSCRIPT_DIR.mkdir(exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
FROZEN_SHA = "67851d3"

M5_PANEL_SHA256 = "46a7e1938bb2fda15ade5898a42c54600ae9906f1e0551828dc2a4baf41c3eaa"
LD_PANEL_SHA256 = "77e528c66c3b3ac7b16e268bd31fb3187017e0dc4ad4008d970f402de5944aac"

CANONICAL_CANDIDATES = [
    "T7", "T7_30", "T14", "T28", "T56", "T84",
    "DOM", "BIMONTH_M_M1", "ROLLING3M", "PAY_CYCLE", "LY_SAME_BUCKET", "LY_DOM",
]
BASELINES = ["B0_SNAIVE7", "B1_ETS", "B2_THETA", "B3_MSTL_AUTOARIMA"]
SC_COL    = "M6_PRED"
L0_COL    = "BL1_S4DR"

METRIC_NAMES = ["MAE", "RMSE", "sMAPE", "WAPE", "MedAE", "P90AE", "SIGNED_BIAS"]

LD_EVAL_ORIGINS = [
    "2014-10-02", "2014-10-09", "2014-10-16", "2014-10-23",
    "2014-10-30", "2014-11-06", "2014-11-13", "2014-11-20",
    "2014-11-27", "2014-12-04", "2014-12-11", "2014-12-18",
]
LD_HALF_A = LD_EVAL_ORIGINS[:6]   # origins 1–6
LD_HALF_B = LD_EVAL_ORIGINS[6:]   # origins 7–12

M5_EVAL_ORIGINS = [
    "2016-04-03", "2016-04-10", "2016-04-17", "2016-04-24",
    "2016-05-01", "2016-05-08", "2016-05-15", "2016-05-22",
]
M5_HALF_A = M5_EVAL_ORIGINS[:4]
M5_HALF_B = M5_EVAL_ORIGINS[4:]

H_PR2_CANDIDATES = ["LY_DOM", "T28", "PAY_CYCLE"]

SEP = "=" * 72


# ── Metric functions ───────────────────────────────────────────────────────────
def _a(r, p):
    return np.asarray(r, float), np.asarray(p, float)

def mae(r, p):
    r, p = _a(r, p); return float(np.mean(np.abs(r - p)))

def rmse(r, p):
    r, p = _a(r, p); return float(np.sqrt(np.mean((r - p) ** 2)))

def smape(r, p):
    r, p = _a(r, p); d = np.abs(r) + np.abs(p)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.where(d > 0, 2.0 * np.abs(r - p) / d, np.nan)
    v = s[np.isfinite(s)]
    return float(100 * np.mean(v)) if len(v) > 0 else np.nan

def wape(r, p):
    r, p = _a(r, p); sr = float(np.sum(np.abs(r)))
    return float(np.sum(np.abs(r - p)) / sr) if sr > 0 else np.nan

def medae(r, p):
    r, p = _a(r, p); return float(np.median(np.abs(r - p)))

def p90ae(r, p):
    r, p = _a(r, p); return float(np.percentile(np.abs(r - p), 90))

def bias(r, p):
    r, p = _a(r, p); return float(np.mean(p - r))

def met_all(r, p):
    return {
        "MAE":         mae(r, p),
        "RMSE":        rmse(r, p),
        "sMAPE":       smape(r, p),
        "WAPE":        wape(r, p),
        "MedAE":       medae(r, p),
        "P90AE":       p90ae(r, p),
        "SIGNED_BIAS": bias(r, p),
    }

def macro_mae_val(panel: pd.DataFrame, pred_col: str) -> float:
    vals = [mae(s["Real"].values, s[pred_col].values)
            for sid in panel["SeriesId"].unique()
            for s in [panel[panel["SeriesId"] == sid]]]
    return float(np.mean(vals))

def per_series_mae(panel: pd.DataFrame, pred_col: str) -> pd.Series:
    return pd.Series(
        {sid: mae(s["Real"].values, s[pred_col].values)
         for sid in panel["SeriesId"].unique()
         for s in [panel[panel["SeriesId"] == sid]]}
    )


# ── Wilcoxon signed-rank ───────────────────────────────────────────────────────
def wilcoxon_test(d: np.ndarray, label_a: str, label_b: str) -> dict:
    """
    d_i = MAE(label_a, series_i) - MAE(label_b, series_i)
    d_i < 0  →  label_a is better for series i
    rank_biserial convention: negative = favorable to label_a
    """
    d_clean = d[np.isfinite(d)]
    n_series = int(len(d_clean))
    n_nonzero = int(np.sum(d_clean != 0))
    base = {"comparison": f"{label_a}_vs_{label_b}", "n_series": n_series,
            "n_nonzero": n_nonzero}
    if n_series < 5:
        return {**base, "statistic": None, "pvalue": None, "rank_biserial": None,
                "note": "INSUFFICIENT_N"}
    result = stats.wilcoxon(d_clean, zero_method="wilcox", alternative="two-sided")
    d_nz = d_clean[d_clean != 0]
    abs_d = np.abs(d_nz)
    ranks = stats.rankdata(abs_d)
    pos_ranks = float(ranks[d_nz > 0].sum())   # label_a worse
    neg_ranks = float(ranks[d_nz < 0].sum())   # label_a better
    total_r   = pos_ranks + neg_ranks
    r = float((pos_ranks - neg_ranks) / total_r) if total_r > 0 else None
    # negative r = label_a better = favorable to label_a
    return {**base,
            "statistic":      float(result.statistic),
            "pvalue":         float(result.pvalue),
            "rank_biserial":  r,
            "note":           ("REJECT_H0_p<0.05" if result.pvalue < 0.05
                               else "FAIL_TO_REJECT_H0")}


# ── SHA256 ────────────────────────────────────────────────────────────────────
def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Pre-gates 2.1–2.7 ─────────────────────────────────────────────────────────
def run_pre_gates() -> list[dict]:
    gates = []

    # 2.1 — FROZEN_SHA: model must be UNCHANGED since 67851d3 (HEAD may be later)
    import subprocess
    try:
        head_sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        # Check that src/s4dr/model.py has no diff since FROZEN_SHA
        diff_out = subprocess.check_output(
            ["git", "diff", FROZEN_SHA, "HEAD", "--", "src/s4dr/model.py"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        model_unchanged = len(diff_out) == 0
        g21 = model_unchanged
        note21 = (f"HEAD={head_sha} (later than FROZEN_SHA but model.py UNCHANGED since {FROZEN_SHA})"
                  if model_unchanged else
                  f"HEAD={head_sha} model.py CHANGED since {FROZEN_SHA}")
    except Exception as e:
        head_sha = "UNKNOWN"
        g21 = False
        note21 = str(e)
    gates.append({"gate": "2.1_FROZEN_SHA", "pass": g21,
                  "expected": f"model.py_unchanged_since_{FROZEN_SHA}",
                  "observed": head_sha, "note": note21})

    # 2.2 — M5 panel SHA256
    m5_sha = sha256_file(OUT_M5 / "forecast_panel.csv")
    g22 = m5_sha == M5_PANEL_SHA256
    gates.append({"gate": "2.2_M5_SHA256", "pass": g22,
                  "expected": M5_PANEL_SHA256, "observed": m5_sha,
                  "note": "M5_CLOSED" if g22 else "SHA_MISMATCH"})

    # 2.3 — Column audit (per-candidate columns)
    m5 = pd.read_csv(OUT_M5 / "forecast_panel.csv", nrows=5)
    ld = pd.read_csv(OUT_LD  / "forecast_panel.csv", nrows=5)
    m5_has_cands = all(c in m5.columns for c in CANONICAL_CANDIDATES)
    ld_has_cands = all(c in ld.columns for c in CANONICAL_CANDIDATES)
    ck_2 = CK_LD / "ck_2014-10-02_m6feed.parquet"
    ck_df = pd.read_parquet(ck_2)
    ck_has_cands = all(c in ck_df.columns for c in CANONICAL_CANDIDATES)
    gate23_status = (
        "FULLY_AVAILABLE"     if (m5_has_cands and ld_has_cands) else
        "PARTIALLY_BLOCKED"   if (ck_has_cands and not m5_has_cands) else
        "BLOCKED_MISSING_CANDIDATE_FORECASTS"
    )
    gates.append({
        "gate": "2.3_CANDIDATE_COLUMNS",
        "pass": gate23_status != "BLOCKED_MISSING_CANDIDATE_FORECASTS",
        "status": gate23_status,
        "m5_panel_has_candidates": m5_has_cands,
        "ld_panel_has_candidates": ld_has_cands,
        "ld_m6feed_ck_has_candidates": ck_has_cands,
        "note": (
            "M5 L1-L5: BLOCKED (per-candidate data not persisted, S4DR execution prohibited). "
            "LD2011 L1-L5: RECOVERABLE from m6feed checkpoints (no S4DR needed). "
            "Flag: MANUSCRIPT_DATA_ADDITION_REQUEST for M5 L1-L5."
        ),
    })

    # 2.4 — Row counts
    m5_rows = sum(1 for _ in open(OUT_M5 / "forecast_panel.csv")) - 1
    ld_rows = sum(1 for _ in open(OUT_LD  / "forecast_panel.csv")) - 1
    g24 = m5_rows == 15680 and ld_rows == 55104
    gates.append({"gate": "2.4_ROW_COUNTS", "pass": g24,
                  "m5_rows": m5_rows, "m5_expected": 15680,
                  "ld_rows": ld_rows, "ld_expected": 55104})

    # 2.5 — No NaN in Real
    m5f = pd.read_csv(OUT_M5 / "forecast_panel.csv")
    ldf = pd.read_csv(OUT_LD  / "forecast_panel.csv")
    m5_nan_real = int(m5f["Real"].isna().sum())
    ld_nan_real = int(ldf["Real"].isna().sum())
    g25 = m5_nan_real == 0 and ld_nan_real == 0
    gates.append({"gate": "2.5_NO_NAN_REAL", "pass": g25,
                  "m5_nan_real": m5_nan_real, "ld_nan_real": ld_nan_real})

    # 2.6 — m6feed checkpoints present for all 12 eval origins
    ld_sha = sha256_file(OUT_LD / "forecast_panel.csv")
    g26_ld_sha = ld_sha == LD_PANEL_SHA256
    missing_ck = [o for o in LD_EVAL_ORIGINS
                  if not (CK_LD / f"ck_{o}_m6feed.parquet").exists()]
    g26 = g26_ld_sha and len(missing_ck) == 0
    gates.append({"gate": "2.6_LD2011_M6FEED_CHECKPOINTS", "pass": g26,
                  "ld_sha256_ok": g26_ld_sha, "missing_checkpoints": missing_ck,
                  "expected_origins": 12, "observed_origins": 12 - len(missing_ck)})

    # 2.7 — m6feed rows per checkpoint = 4592
    bad_ck_counts = {}
    for o in LD_EVAL_ORIGINS:
        ck = CK_LD / f"ck_{o}_m6feed.parquet"
        if ck.exists():
            n = len(pd.read_parquet(ck))
            if n != 4592:
                bad_ck_counts[o] = n
    g27 = len(bad_ck_counts) == 0
    gates.append({"gate": "2.7_M6FEED_ROW_COUNTS", "pass": g27,
                  "expected_per_origin": 4592,
                  "bad_counts": bad_ck_counts if bad_ck_counts else None})

    return gates


# ── Load LD2011 with per-candidate columns ─────────────────────────────────────
def load_ld2011_with_candidates() -> pd.DataFrame:
    print("  Loading LD2011 forecast panel...")
    panel = pd.read_csv(OUT_LD / "forecast_panel.csv",
                        parse_dates=["Origin", "TargetDate"])

    print("  Loading LD2011 m6feed checkpoints (eval only)...")
    feed_parts = []
    for o_str in LD_EVAL_ORIGINS:
        ck = CK_LD / f"ck_{o_str}_m6feed.parquet"
        df = pd.read_parquet(ck)
        df = df.rename(columns={"sid": "SeriesId", "origin": "Origin_str",
                                 "target": "TargetDate_str", "actual": "Real_ck"})
        df["Origin"]     = pd.to_datetime(df["Origin_str"])
        df["TargetDate"] = pd.to_datetime(df["TargetDate_str"])
        feed_parts.append(df[["SeriesId", "Origin", "TargetDate"] + CANONICAL_CANDIDATES])

    feed = pd.concat(feed_parts, ignore_index=True)
    print(f"  m6feed loaded: {len(feed)} rows")

    merged = panel.merge(
        feed, on=["SeriesId", "Origin", "TargetDate"], how="left"
    )

    # Sanity: merged rows should equal original
    assert len(merged) == len(panel), f"Merge length mismatch: {len(merged)} != {len(panel)}"
    n_missing_cands = merged[CANONICAL_CANDIDATES[0]].isna().sum()
    if n_missing_cands > 0:
        raise ValueError(f"Join left {n_missing_cands} rows without candidate data")

    print(f"  Join OK: {len(merged)} rows, {n_missing_cands} unmatched")
    return merged


# ── Compute L-levels ───────────────────────────────────────────────────────────
def compute_l_levels(df: pd.DataFrame, half_a: list[str], half_b: list[str]) -> tuple:
    """
    Adds L1–L5 columns to df (in-place copy).
    Returns (df_with_L, best_global_cand, l4s_cand_a_dict, l4s_cand_b_dict,
             per_series_l4_cand, per_series_l4s_stability)
    """
    df = df.copy()
    cands_np = df[CANONICAL_CANDIDATES].values.astype(float)  # (N, 12)
    real_np  = df["Real"].values.astype(float)

    # L1 — uniform mean
    df["L1"] = np.nanmean(cands_np, axis=1)

    # L2 — median
    df["L2"] = np.nanmedian(cands_np, axis=1)

    # L3 — best single global candidate (retrospective)
    global_maes = [float(np.mean(np.abs(real_np - cands_np[:, i])))
                   for i in range(len(CANONICAL_CANDIDATES))]
    best_global_idx  = int(np.argmin(global_maes))
    best_global_cand = CANONICAL_CANDIDATES[best_global_idx]
    df["L3"] = cands_np[:, best_global_idx]

    # L4 — best per-series candidate (retrospective, in-sample)
    l4_preds     = np.full(len(df), np.nan)
    l4_cand_map  = {}
    idx_arr      = np.arange(len(df))
    for sid in df["SeriesId"].unique():
        mask   = (df["SeriesId"] == sid).values
        r_s    = real_np[mask]
        c_s    = cands_np[mask]
        per_c  = np.mean(np.abs(c_s - r_s[:, None]), axis=0)  # (12,)
        bi     = int(np.argmin(per_c))
        l4_preds[mask] = c_s[:, bi]
        l4_cand_map[sid] = CANONICAL_CANDIDATES[bi]
    df["L4"] = l4_preds

    # L4S — cross-fit (half_A rows ← best cand from half_B; half_B rows ← best cand from half_A)
    df["_Origin_str"] = df["Origin"].dt.strftime("%Y-%m-%d")
    mask_ha = df["_Origin_str"].isin(half_a).values
    mask_hb = df["_Origin_str"].isin(half_b).values

    l4s_preds    = np.full(len(df), np.nan)
    l4s_cand_a   = {}   # selected from half_B, applied to half_A rows
    l4s_cand_b   = {}   # selected from half_A, applied to half_B rows

    for sid in df["SeriesId"].unique():
        sid_mask = (df["SeriesId"] == sid).values

        # Select on half_B → apply to half_A
        mb = sid_mask & mask_hb
        if mb.any():
            r_b  = real_np[mb]; c_b = cands_np[mb]
            bi   = int(np.argmin(np.mean(np.abs(c_b - r_b[:, None]), axis=0)))
            l4s_cand_a[sid] = CANONICAL_CANDIDATES[bi]
        else:
            l4s_cand_a[sid] = CANONICAL_CANDIDATES[0]

        # Select on half_A → apply to half_B
        ma = sid_mask & mask_ha
        if ma.any():
            r_a  = real_np[ma]; c_a = cands_np[ma]
            bi   = int(np.argmin(np.mean(np.abs(c_a - r_a[:, None]), axis=0)))
            l4s_cand_b[sid] = CANONICAL_CANDIDATES[bi]
        else:
            l4s_cand_b[sid] = CANONICAL_CANDIDATES[0]

        # Apply
        m_sid_ha = sid_mask & mask_ha
        m_sid_hb = sid_mask & mask_hb
        ci_a = CANONICAL_CANDIDATES.index(l4s_cand_a[sid])
        ci_b = CANONICAL_CANDIDATES.index(l4s_cand_b[sid])
        if m_sid_ha.any():
            l4s_preds[m_sid_ha] = cands_np[m_sid_ha, ci_a]
        if m_sid_hb.any():
            l4s_preds[m_sid_hb] = cands_np[m_sid_hb, ci_b]

    df["L4S"] = l4s_preds

    # L5 — per-row oracle (min abs error, canonical-order tie-break via argmin)
    abs_errs = np.abs(cands_np - real_np[:, None])  # (N, 12)
    best_idx  = np.argmin(abs_errs, axis=1)          # first minimum = canonical tie-break
    df["L5"]  = cands_np[np.arange(len(df)), best_idx]

    df = df.drop(columns=["_Origin_str"])

    # Selection stability for L4S
    stability = pd.DataFrame([
        {"SeriesId":       sid,
         "L4_InSample":   l4_cand_map[sid],
         "L4S_FromHalfB": l4s_cand_a[sid],   # applied to half_A
         "L4S_FromHalfA": l4s_cand_b[sid],   # applied to half_B
         "Consistent_AB":  l4s_cand_a[sid] == l4s_cand_b[sid]}
        for sid in df["SeriesId"].unique()
    ])

    return df, best_global_cand, l4s_cand_a, l4s_cand_b, l4_cand_map, stability


# ── Build full headroom row ────────────────────────────────────────────────────
L_LEVEL_COLS = {
    "B0": "B0_SNAIVE7",
    "B1": "B1_ETS",
    "B2": "B2_THETA",
    "B3": "B3_MSTL_AUTOARIMA",
    "L0": "BL1_S4DR",
    "SC": "M6_PRED",
}
L_COMPUTED   = ["L1", "L2", "L3", "L4", "L4S", "L5"]

DEPLOYABILITY = {
    "B0": "DEPLOYABLE",
    "B1": "DEPLOYABLE",
    "B2": "DEPLOYABLE",
    "B3": "DEPLOYABLE",
    "L0": "CAUSAL_DEPLOYABLE",
    "SC": "CAUSAL_NON_CANONICAL",
    "L1": "RETROSPECTIVE_NOT_DEPLOYABLE",
    "L2": "RETROSPECTIVE_NOT_DEPLOYABLE",
    "L3": "RETROSPECTIVE_NOT_DEPLOYABLE",
    "L4": "RETROSPECTIVE_NOT_DEPLOYABLE",
    "L4S": "RETROSPECTIVE_CROSS_FIT_NOT_DEPLOYABLE",
    "L5": "ORACLE_NOT_DEPLOYABLE",
}


def build_headroom_rows(ds_name: str, panel: pd.DataFrame,
                        computed_cols: dict | None) -> list[dict]:
    """
    computed_cols: {level_name: np.ndarray} for L1–L5 (None if blocked)
    Returns list of dicts with all metrics for all method levels.
    """
    rows = []
    real = panel["Real"].values

    def add_row(level, pred, pred_arr):
        if pred_arr is None:
            row = {"Dataset": ds_name, "Level": level,
                   "Deployability": DEPLOYABILITY.get(level, "UNKNOWN"),
                   "N_rows": len(panel), "BLOCKED": True}
            for m in METRIC_NAMES:
                row[f"MICRO_{m}"] = None
            row["MACRO_MAE"] = None
            rows.append(row)
            return
        r_clean = real[np.isfinite(pred_arr) & np.isfinite(real)]
        p_clean = pred_arr[np.isfinite(pred_arr) & np.isfinite(real)]
        met = met_all(r_clean, p_clean)
        # macro MAE: add pred column temporarily
        tmp = panel.copy()
        tmp["__pred"] = pred_arr
        tmp = tmp.dropna(subset=["Real", "__pred"])
        mm = float(np.mean([mae(s["Real"].values, s["__pred"].values)
                            for sid in tmp["SeriesId"].unique()
                            for s in [tmp[tmp["SeriesId"] == sid]]]))
        row = {"Dataset": ds_name, "Level": level,
               "Deployability": DEPLOYABILITY.get(level, "UNKNOWN"),
               "N_rows": int(len(p_clean)), "BLOCKED": False}
        for m in METRIC_NAMES:
            row[f"MICRO_{m}"] = met[m]
        row["MACRO_MAE"] = mm
        rows.append(row)

    # Fixed-column levels
    for level, col in L_LEVEL_COLS.items():
        if col in panel.columns:
            add_row(level, col, panel[col].values)
        else:
            add_row(level, col, None)

    # Computed L-levels
    for lname in L_COMPUTED:
        if computed_cols and lname in computed_cols:
            add_row(lname, lname, computed_cols[lname])
        else:
            add_row(lname, lname, None)  # BLOCKED

    return rows


# ── Decomposition ─────────────────────────────────────────────────────────────
def build_decomposition(ds_name: str, ht: pd.DataFrame) -> list[dict]:
    """
    ht: subset of headroom_table for one dataset (wide form).
    Returns decomposition component rows.
    """
    def get_mae(level):
        row = ht[(ht["Dataset"] == ds_name) & (ht["Level"] == level)]
        if len(row) == 0 or row.iloc[0]["BLOCKED"]:
            return None
        return row.iloc[0]["MICRO_MAE"]

    l0 = get_mae("L0")
    l1 = get_mae("L1")
    l2 = get_mae("L2")
    l3 = get_mae("L3")
    l4 = get_mae("L4")
    l4s = get_mae("L4S")
    l5  = get_mae("L5")
    sc  = get_mae("SC")

    def diff(a, b):
        return float(a - b) if (a is not None and b is not None) else None

    return [
        {"Dataset": ds_name,
         "Component": "LIBRARY_RETROSPECTIVE_HEADROOM",
         "Formula":   "L0_MAE - L3_MAE",
         "Value":     diff(l0, l3),
         "Note":      "Headroom if best global candidate were chosen retrospectively"},
        {"Dataset": ds_name,
         "Component": "STATIC_RETROSPECTIVE_SPECIALIZATION",
         "Formula":   "L3_MAE - L4_MAE",
         "Value":     diff(l3, l4),
         "Note":      "Additional benefit of per-series vs global selection (in-sample)"},
        {"Dataset": ds_name,
         "Component": "BIAS_CORRECTED_SPECIALIZATION",
         "Formula":   "L3_MAE - L4S_MAE",
         "Value":     diff(l3, l4s),
         "Note":      "Cross-fit (out-of-sample) estimate of per-series specialization"},
        {"Dataset": ds_name,
         "Component": "CAUSAL_SPECIALIZATION",
         "Formula":   "L0_MAE - SC_MAE",
         "Value":     diff(l0, sc),
         "Note":      "Benefit of causal M6 selector over BL1 (actual deployed diff)"},
        {"Dataset": ds_name,
         "Component": "ORACLE_HEADROOM",
         "Formula":   "L0_MAE - L5_MAE",
         "Value":     diff(l0, l5),
         "Note":      "Maximum possible MAE improvement via per-row oracle selection"},
    ]


# ── Cross-checks 5.1–5.6 ──────────────────────────────────────────────────────
def run_cross_checks(ds_name: str, ht_ds: pd.DataFrame,
                     panel: pd.DataFrame, computed_cols) -> list[dict]:
    checks = []

    def get_mae(level):
        row = ht_ds[ht_ds["Level"] == level]
        if len(row) == 0 or row.iloc[0]["BLOCKED"]:
            return None
        return float(row.iloc[0]["MICRO_MAE"])

    l3, l4, l4s, l5 = get_mae("L3"), get_mae("L4"), get_mae("L4S"), get_mae("L5")
    l0, sc = get_mae("L0"), get_mae("SC")

    # 5.1 — L5 ≤ all candidate MAEs per-row (oracle by construction)
    if computed_cols and "L5" in computed_cols and computed_cols["L5"] is not None:
        real = panel["Real"].values
        l5_arr = computed_cols["L5"]
        cands_np = np.column_stack([panel[c].values for c in CANONICAL_CANDIDATES])
        # Check that L5 abs error ≤ best candidate abs error per row
        l5_ae   = np.abs(real - l5_arr)
        best_ae = np.min(np.abs(cands_np - real[:, None]), axis=1)
        fail_rows = int(np.sum(l5_ae > best_ae + 1e-9))
        checks.append({"Dataset": ds_name, "Check": "5.1_L5_IS_ORACLE",
                        "pass": fail_rows == 0,
                        "detail": f"{fail_rows} rows where L5 > min_candidate"})
    else:
        checks.append({"Dataset": ds_name, "Check": "5.1_L5_IS_ORACLE",
                        "pass": None, "detail": "BLOCKED_NO_CANDIDATE_DATA"})

    # 5.2 — L4_MAE ≤ L3_MAE (per-series can only improve over global selection)
    if l3 is not None and l4 is not None:
        checks.append({"Dataset": ds_name, "Check": "5.2_L4_LE_L3",
                        "pass": l4 <= l3 + 1e-9,
                        "detail": f"L4={l4:.4f} L3={l3:.4f} diff={l4-l3:+.4f}"})
    else:
        checks.append({"Dataset": ds_name, "Check": "5.2_L4_LE_L3",
                        "pass": None, "detail": "BLOCKED"})

    # 5.3 — L5_MAE ≤ L4_MAE
    if l5 is not None and l4 is not None:
        checks.append({"Dataset": ds_name, "Check": "5.3_L5_LE_L4",
                        "pass": l5 <= l4 + 1e-9,
                        "detail": f"L5={l5:.4f} L4={l4:.4f} diff={l5-l4:+.4f}"})
    else:
        checks.append({"Dataset": ds_name, "Check": "5.3_L5_LE_L4",
                        "pass": None, "detail": "BLOCKED"})

    # 5.4 — L4S_MAE ≤ L3_MAE (cross-fit per-series still better than global)
    if l4s is not None and l3 is not None:
        checks.append({"Dataset": ds_name, "Check": "5.4_L4S_LE_L3",
                        "pass": l4s <= l3 + 1e-9,
                        "detail": f"L4S={l4s:.4f} L3={l3:.4f} diff={l4s-l3:+.4f}"})
    else:
        checks.append({"Dataset": ds_name, "Check": "5.4_L4S_LE_L3",
                        "pass": None, "detail": "BLOCKED"})

    # 5.5 — L0 MAE matches global_metrics.csv (round-trip consistency)
    gm_path = (OUT_M5 if ds_name.startswith("M5") else OUT_LD) / "global_metrics.csv"
    if gm_path.exists():
        gm = pd.read_csv(gm_path)
        row_bl1 = gm[gm["Method"] == "BL1_S4DR"]
        if len(row_bl1) > 0:
            ref_mae = float(row_bl1.iloc[0]["MAE"])
            diff55  = abs(l0 - ref_mae) if l0 is not None else None
            checks.append({"Dataset": ds_name, "Check": "5.5_L0_MAE_ROUNDTRIP",
                            "pass": diff55 is not None and diff55 < 1e-6,
                            "detail": f"L0={l0:.6f} ref={ref_mae:.6f} diff={diff55:.2e}" if diff55 is not None else "BLOCKED"})
        else:
            checks.append({"Dataset": ds_name, "Check": "5.5_L0_MAE_ROUNDTRIP",
                            "pass": None, "detail": "BL1_S4DR row not found in global_metrics"})
    else:
        checks.append({"Dataset": ds_name, "Check": "5.5_L0_MAE_ROUNDTRIP",
                        "pass": None, "detail": "global_metrics.csv not found"})

    # 5.6 — Row counts correct
    if ds_name.startswith("M5"):
        expected_rows = 15680
    else:
        expected_rows = 55104
    actual_rows = len(panel)
    checks.append({"Dataset": ds_name, "Check": "5.6_ROW_COUNT",
                    "pass": actual_rows == expected_rows,
                    "detail": f"actual={actual_rows} expected={expected_rows}"})

    return checks


# ── H_PR2 table ───────────────────────────────────────────────────────────────
def build_h_pr2_table(ds_name: str, panel: pd.DataFrame) -> list[dict]:
    rows = []
    for cand in H_PR2_CANDIDATES:
        sub = panel[panel["M6_SelectedCandidate"] == cand].dropna(subset=["Real", SC_COL, L0_COL])
        n   = len(sub)
        pct = 100 * n / len(panel) if len(panel) > 0 else 0.0
        rows.append({
            "Dataset":       ds_name,
            "Candidate":     cand,
            "N_selected":    n,
            "PCT_selected":  pct,
            "M6_MAE":        mae(sub["Real"].values, sub[SC_COL].values) if n > 0 else None,
            "BL1_MAE":       mae(sub["Real"].values, sub[L0_COL].values) if n > 0 else None,
            "dMAE_M6_vs_BL1":(mae(sub["Real"].values, sub[SC_COL].values) -
                               mae(sub["Real"].values, sub[L0_COL].values)) if n > 0 else None,
        })
    return rows


# ── Candidate global ranking (LD2011 only) ────────────────────────────────────
def build_candidate_ranking(ds_name: str, panel: pd.DataFrame,
                             computed_cols: dict) -> list[dict]:
    real = panel["Real"].values
    rows = []
    for i, cand in enumerate(CANONICAL_CANDIDATES):
        pred = computed_cols.get(cand)
        if pred is None:
            rows.append({"Dataset": ds_name, "Candidate": cand,
                         "Global_MAE": None, "BLOCKED": True})
            continue
        rows.append({
            "Dataset": ds_name, "Candidate": cand,
            "Global_MAE": mae(real, pred),
            "sMAPE":      smape(real, pred),
            "WAPE":       wape(real, pred),
            "BLOCKED":    False,
        })
    rows.sort(key=lambda x: x["Global_MAE"] if x["Global_MAE"] is not None else np.inf)
    for rank, r in enumerate(rows, start=1):
        r["Global_Rank"] = rank
    return rows


# ── Wilcoxon table ────────────────────────────────────────────────────────────
def build_wilcoxon_table(ds_name: str, panel: pd.DataFrame,
                          computed_cols: dict | None) -> list[dict]:
    rows = []
    ps_l0 = per_series_mae(panel, L0_COL)

    comparisons = [
        ("L0", "B0", L0_COL, "B0_SNAIVE7"),
        ("L0", "B1", L0_COL, "B1_ETS"),
        ("L0", "B2", L0_COL, "B2_THETA"),
        ("L0", "B3", L0_COL, "B3_MSTL_AUTOARIMA"),
        ("L0", "SC", L0_COL, SC_COL),
    ]

    if computed_cols:
        for lname in ["L1", "L2", "L3", "L4", "L4S", "L5"]:
            if lname in computed_cols and computed_cols[lname] is not None:
                comparisons.append(("L0", lname, L0_COL, lname))

    for la_name, lb_name, la_col, lb_col in comparisons:
        # Per-series MAE for both methods
        if lb_col in panel.columns:
            ps_b = per_series_mae(panel, lb_col)
        elif computed_cols and lb_col in computed_cols and computed_cols[lb_col] is not None:
            # Add as temp column
            tmp = panel.copy()
            tmp["__lb"] = computed_cols[lb_col]
            tmp = tmp.dropna(subset=["Real", "__lb"])
            ps_b = per_series_mae(tmp, "__lb")
        else:
            rows.append({"Dataset": ds_name,
                         "comparison": f"{la_name}_vs_{lb_name}",
                         "note": "BLOCKED_NO_DATA"})
            continue

        # Align series
        common = ps_l0.index.intersection(ps_b.index)
        d = (ps_l0[common] - ps_b[common]).values  # d_i < 0 → L0 better
        result = wilcoxon_test(d, la_name, lb_name)
        result["Dataset"] = ds_name
        rows.append(result)

    return rows


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(SEP)
    print("S4DR_MANUSCRIPT_MEASUREMENT_CLOSURE")
    print(f"FROZEN_SHA = {FROZEN_SHA}")
    print(SEP)

    # ── Pre-gates ─────────────────────────────────────────────────────────────
    print("\n[STEP 1] Running pre-gates 2.1–2.7...")
    gates = run_pre_gates()
    gates_df = pd.DataFrame([
        {k: v for k, v in g.items() if k not in ("missing_checkpoints", "bad_counts")}
        for g in gates
    ])
    gates_df.to_csv(MANUSCRIPT_DIR / "pre_gates_audit.csv", index=False)

    all_gates_pass = all(g["pass"] for g in gates if g["pass"] is not None)
    for g in gates:
        status = "PASS" if g["pass"] else ("PARTIAL" if g["pass"] is None else "FAIL")
        print(f"  {g['gate']:45s}  {status}")
    print(f"  ALL_GATES: {'PASS' if all_gates_pass else 'PARTIAL/FAIL (see gate 2.3)'}")

    # Identify gate 2.3 status
    gate23 = next(g for g in gates if "2.3" in g["gate"])
    gate23_status = gate23["status"]
    print(f"\n  Gate 2.3 status: {gate23_status}")
    if gate23_status == "PARTIALLY_BLOCKED":
        print("  LD2011 L1-L5: RECOVERABLE from m6feed checkpoints")
        print("  M5 L1-L5: BLOCKED — MANUSCRIPT_DATA_ADDITION_REQUEST")

    # ── Load panels ───────────────────────────────────────────────────────────
    print("\n[STEP 2] Loading panels...")
    m5_panel = pd.read_csv(OUT_M5 / "forecast_panel.csv",
                            parse_dates=["Origin", "TargetDate"])
    ld_panel_raw = pd.read_csv(OUT_LD / "forecast_panel.csv",
                                parse_dates=["Origin", "TargetDate"])
    print(f"  M5 panel: {len(m5_panel)} rows")
    print(f"  LD panel: {len(ld_panel_raw)} rows")

    # ── LD2011: join per-candidate columns ────────────────────────────────────
    print("\n[STEP 3] Joining LD2011 m6feed per-candidate columns...")
    ld_panel = load_ld2011_with_candidates()

    # ── Compute L-levels for LD2011 ───────────────────────────────────────────
    print("\n[STEP 4] Computing L1–L5 for LD2011...")
    (ld_with_L, best_global_cand_ld, l4s_cand_a_ld, l4s_cand_b_ld,
     l4_cand_map_ld, stability_ld) = compute_l_levels(ld_panel, LD_HALF_A, LD_HALF_B)

    ld_computed = {lname: ld_with_L[lname].values for lname in L_COMPUTED}
    for cand in CANONICAL_CANDIDATES:
        ld_computed[cand] = ld_panel[cand].values

    print(f"  L3 best global candidate for LD2011: {best_global_cand_ld}")
    print(f"  L4S stability: {stability_ld['Consistent_AB'].mean():.1%} consistent across halves")

    # M5: L1–L5 BLOCKED
    m5_computed = None  # blocked — no per-candidate data
    print("  M5 L1-L5: BLOCKED (per-candidate data not persisted)")

    # ── Headroom table ────────────────────────────────────────────────────────
    print("\n[STEP 5] Building headroom table...")
    ht_rows  = build_headroom_rows("M5_STORE_DEPT", m5_panel, m5_computed)
    ht_rows += build_headroom_rows("LD2011_DAILY", ld_with_L, ld_computed)
    ht = pd.DataFrame(ht_rows)
    ht.to_csv(MANUSCRIPT_DIR / "headroom_table.csv", index=False)

    print("  Headroom table (L0 / L3 / L4S / L5 MICRO MAE):")
    for ds in ["M5_STORE_DEPT", "LD2011_DAILY"]:
        sub = ht[ht["Dataset"] == ds]
        for lv in ["B2", "L0", "L3", "L4S", "L5", "SC"]:
            r = sub[sub["Level"] == lv]
            if len(r) == 0:
                continue
            r0 = r.iloc[0]
            mae_v = f"{r0['MICRO_MAE']:.4f}" if not r0["BLOCKED"] else "BLOCKED"
            print(f"    {ds} {lv:<6}: MICRO_MAE={mae_v}")

    # ── Decomposition ─────────────────────────────────────────────────────────
    print("\n[STEP 6] Building decomposition table...")
    dec_rows  = build_decomposition("M5_STORE_DEPT", ht)
    dec_rows += build_decomposition("LD2011_DAILY",  ht)
    dec = pd.DataFrame(dec_rows)
    dec.to_csv(MANUSCRIPT_DIR / "decomposition_table.csv", index=False)

    # ── Wilcoxon tests ────────────────────────────────────────────────────────
    print("\n[STEP 7] Running Wilcoxon signed-rank tests...")
    wil_rows  = build_wilcoxon_table("M5_STORE_DEPT", m5_panel, m5_computed)
    wil_rows += build_wilcoxon_table("LD2011_DAILY",  ld_with_L, ld_computed)
    wil = pd.DataFrame(wil_rows)
    wil.to_csv(MANUSCRIPT_DIR / "wilcoxon_tests.csv", index=False)

    # ── L4S selection stability ───────────────────────────────────────────────
    print("\n[STEP 8] Saving L4S selection stability (LD2011 only)...")
    stability_ld.to_csv(MANUSCRIPT_DIR / "selection_stability_l4s.csv", index=False)
    pct_consistent = stability_ld["Consistent_AB"].mean()
    print(f"  L4S consistent_AB: {pct_consistent:.1%} of {len(stability_ld)} series")

    # ── Candidate global ranking (LD2011) ─────────────────────────────────────
    print("\n[STEP 9] Building candidate ranking for LD2011...")
    cr_rows = build_candidate_ranking("LD2011_DAILY", ld_with_L, ld_computed)
    cr = pd.DataFrame(cr_rows)
    cr.to_csv(MANUSCRIPT_DIR / "candidate_ranking_ld2011.csv", index=False)

    # ── H_PR2 table ───────────────────────────────────────────────────────────
    print("\n[STEP 10] Building H_PR2 table...")
    h2_rows  = build_h_pr2_table("M5_STORE_DEPT", m5_panel)
    h2_rows += build_h_pr2_table("LD2011_DAILY",  ld_panel_raw)
    h2 = pd.DataFrame(h2_rows)
    h2.to_csv(MANUSCRIPT_DIR / "h_pr2_table.csv", index=False)

    # ── Per-series MAE (LD2011) ───────────────────────────────────────────────
    print("\n[STEP 11] Building per-series MAE for LD2011...")
    ps_rows = []
    all_ps_cols = {
        **{lv: col for lv, col in L_LEVEL_COLS.items()},
    }
    for sid in ld_with_L["SeriesId"].unique():
        s = ld_with_L[ld_with_L["SeriesId"] == sid]
        row = {"SeriesId": sid}
        for lv, col in L_LEVEL_COLS.items():
            row[f"MAE_{lv}"] = mae(s["Real"].values, s[col].values)
        for lname in L_COMPUTED:
            row[f"MAE_{lname}"] = mae(s["Real"].values, s[lname].values) if lname in s.columns else None
        ps_rows.append(row)
    ps_df = pd.DataFrame(ps_rows)
    ps_df.to_csv(MANUSCRIPT_DIR / "per_series_mae_ld2011.csv", index=False)

    # ── Cross-checks 5.1–5.6 ─────────────────────────────────────────────────
    print("\n[STEP 12] Running cross-checks 5.1–5.6...")
    ht_m5 = ht[ht["Dataset"] == "M5_STORE_DEPT"].copy()
    ht_ld = ht[ht["Dataset"] == "LD2011_DAILY"].copy()
    cc_rows  = run_cross_checks("M5_STORE_DEPT", ht_m5, m5_panel, m5_computed)
    cc_rows += run_cross_checks("LD2011_DAILY",  ht_ld, ld_with_L, ld_computed)
    cc = pd.DataFrame(cc_rows)
    cc.to_csv(MANUSCRIPT_DIR / "cross_checks.csv", index=False)
    for _, crow in cc.iterrows():
        status = "PASS" if crow["pass"] == True else ("BLOCKED" if crow["pass"] is None else "FAIL")
        print(f"  {crow['Dataset']:<20} {crow['Check']:<30}  {status}  {crow['detail']}")

    # ── Final report (mandatory format) ───────────────────────────────────────
    print(f"\n[STEP 13] Writing manuscript report...")

    def fv(v, fmt=".4f"):
        return f"{v:{fmt}}" if v is not None and not (isinstance(v, float) and np.isnan(v)) else "BLOCKED"

    def ht_val(ds, level, col):
        r = ht[(ht["Dataset"] == ds) & (ht["Level"] == level)]
        if len(r) == 0 or r.iloc[0]["BLOCKED"]:
            return None
        return r.iloc[0][col]

    lines = [
        SEP,
        "S4DR_MANUSCRIPT_MEASUREMENT_CLOSURE — FINAL REPORT",
        SEP,
        f"EXPERIMENT_ID      = S4DR_MANUSCRIPT_MEASUREMENT_CLOSURE",
        f"FROZEN_SHA         = {FROZEN_SHA}",
        f"GATE_2_3_STATUS    = {gate23_status}",
        f"M5_L1_L5           = BLOCKED (MANUSCRIPT_DATA_ADDITION_REQUEST)",
        f"LD2011_L1_L5       = AVAILABLE (recovered from m6feed checkpoints)",
        "",
        "-- PRE-GATES 2.1–2.7 --",
    ]
    for g in gates:
        s = "PASS" if g["pass"] else ("PARTIAL" if g["pass"] is None else "FAIL")
        lines.append(f"  {g['gate']:<45s}  {s}")
    lines.append(f"  ALL_GATES_PASS = {'YES' if all_gates_pass else 'PARTIAL'}")

    lines += [
        "",
        "-- HEADROOM LADDER (MICRO MAE) --",
        "  M5_STORE_DEPT:",
    ]
    for lv in ["B0", "B1", "B2", "B3", "L0", "SC", "L1", "L2", "L3", "L4", "L4S", "L5"]:
        v = ht_val("M5_STORE_DEPT", lv, "MICRO_MAE")
        dpl = DEPLOYABILITY.get(lv, "UNKNOWN")
        lines.append(f"    {lv:<6}  MAE={fv(v)}  [{dpl}]")

    lines.append("  LD2011_DAILY:")
    for lv in ["B0", "B1", "B2", "B3", "L0", "SC", "L1", "L2", "L3", "L4", "L4S", "L5"]:
        v = ht_val("LD2011_DAILY", lv, "MICRO_MAE")
        dpl = DEPLOYABILITY.get(lv, "UNKNOWN")
        lines.append(f"    {lv:<6}  MAE={fv(v)}  [{dpl}]")

    lines += [
        "",
        "-- HEADROOM LADDER (MACRO MAE) --",
        "  M5_STORE_DEPT:",
    ]
    for lv in ["B0", "B1", "B2", "B3", "L0", "SC", "L1", "L2", "L3", "L4", "L4S", "L5"]:
        v = ht_val("M5_STORE_DEPT", lv, "MACRO_MAE")
        lines.append(f"    {lv:<6}  MACRO_MAE={fv(v)}")

    lines.append("  LD2011_DAILY:")
    for lv in ["B0", "B1", "B2", "B3", "L0", "SC", "L1", "L2", "L3", "L4", "L4S", "L5"]:
        v = ht_val("LD2011_DAILY", lv, "MACRO_MAE")
        lines.append(f"    {lv:<6}  MACRO_MAE={fv(v)}")

    lines += [
        "",
        f"  L3_BEST_GLOBAL_CANDIDATE (LD2011): {best_global_cand_ld}",
        "",
        "-- DECOMPOSITION TABLE --",
    ]
    for _, dr in dec.iterrows():
        v_str = fv(dr["Value"]) if dr["Value"] is not None else "BLOCKED"
        lines.append(f"  {dr['Dataset']:<20} {dr['Component']:<45} = {v_str}")

    lines += [
        "",
        "-- L4S SELECTION STABILITY (LD2011) --",
        f"  N_series = {len(stability_ld)}",
        f"  Consistent_AB = {stability_ld['Consistent_AB'].sum()} / {len(stability_ld)}"
        f"  ({pct_consistent:.1%})",
        "",
        "-- CANDIDATE RANKING (LD2011 global MAE) --",
    ]
    for _, rr in cr[cr["BLOCKED"] == False].iterrows():
        lines.append(f"  #{int(rr['Global_Rank'])} {rr['Candidate']:<16} MAE={rr['Global_MAE']:.4f}")

    lines += [
        "",
        "-- H_PR2 TABLE (LY_DOM, T28, PAY_CYCLE) --",
    ]
    for _, rr in h2.iterrows():
        dm = fv(rr["dMAE_M6_vs_BL1"])
        lines.append(f"  {rr['Dataset']:<20} {rr['Candidate']:<16} N={rr['N_selected']:>6}"
                     f"  ({rr['PCT_selected']:>5.1f}%)  dMAE_M6_vs_BL1={dm}")

    lines += ["", "-- WILCOXON TESTS (zero_method=wilcox, two-sided) --",
              "  Convention: rank_biserial < 0 = favorable to L0 --"]
    for _, rr in wil.iterrows():
        pv = f"{rr['pvalue']:.4f}" if rr.get("pvalue") is not None else "BLOCKED"
        rb = f"{rr['rank_biserial']:.4f}" if rr.get("rank_biserial") is not None else "BLOCKED"
        note = rr.get("note", "")
        lines.append(f"  {rr['Dataset']:<20} {rr['comparison']:<25}  "
                     f"p={pv}  r_biserial={rb}  {note}")

    lines += [
        "",
        "-- CROSS-CHECKS 5.1–5.6 --",
    ]
    for _, rr in cc.iterrows():
        s = "PASS" if rr["pass"] == True else ("BLOCKED" if rr["pass"] is None else "FAIL")
        lines.append(f"  {rr['Dataset']:<20} {rr['Check']:<30} {s}  {rr['detail']}")

    lines += [
        "",
        "-- DATA AVAILABILITY FLAGS --",
        "  M5 L1-L5: MANUSCRIPT_DATA_ADDITION_REQUEST",
        "    Cause: per-candidate predictions were not persisted in M5 runner",
        "    Action required: await explicit instruction before computing",
        "  EXP-006: FUTURE_WORK_DECLARED (not executed)",
        "  H_PR1: FUTURE_WORK_DECLARED (not executed)",
        "",
        "-- OUTPUT FILES (manuscript_data/) --",
        "  1. pre_gates_audit.csv",
        "  2. headroom_table.csv",
        "  3. decomposition_table.csv",
        "  4. wilcoxon_tests.csv",
        "  5. selection_stability_l4s.csv",
        "  6. candidate_ranking_ld2011.csv",
        "  7. h_pr2_table.csv",
        "  8. per_series_mae_ld2011.csv",
        "  9. cross_checks.csv",
        "  10. manuscript_report.txt",
        "",
        "CANONICAL_MODEL_MODIFIED  = FALSE",
        "S4DR_EXECUTED             = FALSE",
        "STATUS                    = COMPLETE",
        SEP,
    ]

    report_text = "\n".join(lines)
    print(report_text)
    (MANUSCRIPT_DIR / "manuscript_report.txt").write_text(report_text, encoding="utf-8")
    print(f"\nAll 10 files written to {MANUSCRIPT_DIR}")


if __name__ == "__main__":
    main()
