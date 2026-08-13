"""
FASE 3 — Joint metrics computation.
Reads both persisted forecast panels (M5 and LD2011) and computes
GO/NO_GO, M6_REPLICATES, H_PR2, PAPER_DIRECTION per protocol.
Persists all metric CSVs BEFORE printing the final report.
No scientific protocol changes.
"""
from __future__ import annotations
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

OUT_M5  = Path(__file__).parent / "m5_store_dept"
OUT_LD  = Path(__file__).parent / "ld2011_daily"
METHODS = ["B0_SNAIVE7","B1_ETS","B2_THETA","B3_MSTL_AUTOARIMA","BL1_S4DR","M6_PRED"]
CANONICAL_CANDIDATES = [
    "T7","T7_30","T14","T28","T56","T84",
    "DOM","BIMONTH_M_M1","ROLLING3M","PAY_CYCLE","LY_SAME_BUCKET","LY_DOM",
]
TOLERANCES = {
    "temporal_sign_neutrality": 0,
    "smape_material_deterioration": 0,
    "p90ae_material_deterioration": 0,
    "loso_max_flip_pct": 10,
}

# ── Metrics ────────────────────────────────────────────────────────────────────
def _m(r, p): return np.asarray(r, float), np.asarray(p, float)
def mae(r, p):
    r, p = _m(r, p); return float(np.mean(np.abs(r - p)))
def rmse(r, p):
    r, p = _m(r, p); return float(np.sqrt(np.mean((r - p) ** 2)))
def smape(r, p):
    r, p = _m(r, p); d = np.abs(r) + np.abs(p)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.where(d > 0, 2.0 * np.abs(r - p) / d, np.nan)
    v = s[np.isfinite(s)]
    return float(100 * np.mean(v)) if len(v) > 0 else np.nan
def wape(r, p):
    r, p = _m(r, p); sr = float(np.sum(np.abs(r)))
    return float(np.sum(np.abs(r - p)) / sr) if sr > 0 else np.nan
def medae(r, p): r, p = _m(r, p); return float(np.median(np.abs(r - p)))
def p90ae(r, p): r, p = _m(r, p); return float(np.percentile(np.abs(r - p), 90))
def bias(r, p):  r, p = _m(r, p); return float(np.mean(p - r))
def met(r, p):
    return {"MAE": mae(r,p), "RMSE": rmse(r,p), "sMAPE": smape(r,p),
            "WAPE": wape(r,p), "MedAE": medae(r,p), "P90AE": p90ae(r,p),
            "SIGNED_BIAS": bias(r,p), "N": int(len(np.asarray(r)))}

def compute_metrics(panel: pd.DataFrame) -> dict:
    pv = panel.dropna(subset=["Real"] + METHODS)
    real = pv["Real"].values
    out = {}
    for m in METHODS:
        pred = pv[m].values
        macro_maes = []
        for sid in pv["SeriesId"].unique():
            s = pv[pv["SeriesId"] == sid]
            macro_maes.append(mae(s["Real"].values, s[m].values))
        out[m] = {"MICRO": met(real, pred), "MACRO_MAE": float(np.mean(macro_maes))}
    return out

def per_series_dmae(panel: pd.DataFrame, ma: str, mb: str) -> pd.Series:
    pv = panel.dropna(subset=["Real", ma, mb])
    return pd.Series({
        sid: mae(s["Real"].values, s[ma].values) - mae(s["Real"].values, s[mb].values)
        for sid in pv["SeriesId"].unique()
        for s in [pv[pv["SeriesId"] == sid]]
    })

def loso(panel: pd.DataFrame, ma: str, mb: str) -> dict:
    pv = panel.dropna(subset=["Real", ma, mb])
    sids = pv["SeriesId"].unique(); n_flip = 0
    for sid in sids:
        loo  = pv[pv["SeriesId"] != sid]
        this = pv[pv["SeriesId"] == sid]
        dm_loo  = mae(loo["Real"].values,  loo[ma].values)  - mae(loo["Real"].values,  loo[mb].values)
        dm_this = mae(this["Real"].values, this[ma].values) - mae(this["Real"].values, this[mb].values)
        if np.isfinite(dm_loo) and np.isfinite(dm_this) and dm_loo * dm_this < 0:
            n_flip += 1
    n = len(sids)
    return {"N_series": n, "N_flips": n_flip,
            "flip_pct": 100 * n_flip / n if n > 0 else np.nan}

def horizon_dmae(panel: pd.DataFrame, ma: str, mb: str, h_groups: dict) -> dict:
    pv = panel.dropna(subset=["Real", ma, mb])
    return {
        label: mae(pv[pv["Horizon"].isin(hl)]["Real"].values,
                   pv[pv["Horizon"].isin(hl)][ma].values) -
               mae(pv[pv["Horizon"].isin(hl)]["Real"].values,
                   pv[pv["Horizon"].isin(hl)][mb].values)
        for label, hl in h_groups.items()
    }

def m6_attribution(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cand in CANONICAL_CANDIDATES:
        sub = panel[panel["M6_SelectedCandidate"] == cand].dropna(subset=["Real"])
        rows.append({
            "Candidate": cand,
            "N_selected": len(sub),
            "PCT_selected": 100 * len(sub) / len(panel) if len(panel) > 0 else 0,
            "M6_MAE":  mae(sub["Real"].values, sub["M6_PRED"].values)  if len(sub) > 0 else np.nan,
            "BL1_MAE": mae(sub["Real"].values, sub["BL1_S4DR"].values) if len(sub) > 0 else np.nan,
            "dMAE": (mae(sub["Real"].values, sub["M6_PRED"].values) -
                     mae(sub["Real"].values, sub["BL1_S4DR"].values))  if len(sub) > 0 else np.nan,
        })
    fb = panel[panel["M6_IsFallback"]].dropna(subset=["Real"])
    rows.append({"Candidate": "FALLBACK_BL1", "N_selected": len(fb),
                 "PCT_selected": 100 * len(fb) / len(panel) if len(panel) > 0 else 0,
                 "M6_MAE": np.nan, "BL1_MAE": np.nan, "dMAE": np.nan})
    return pd.DataFrame(rows)


def main():
    SEP = "=" * 70

    # ── Load panels ────────────────────────────────────────────────────────────
    print("Loading panels...")
    m5 = pd.read_csv(OUT_M5 / "forecast_panel.csv", parse_dates=["Origin", "TargetDate"])
    ld = pd.read_csv(OUT_LD  / "forecast_panel.csv", parse_dates=["Origin", "TargetDate"])
    print(f"  M5:    {len(m5):>6} rows  {m5['SeriesId'].nunique()} series  {m5['Origin'].nunique()} origins")
    print(f"  LD2011:{len(ld):>6} rows  {ld['SeriesId'].nunique()} series  {ld['Origin'].nunique()} origins")

    h_m5 = {"H_SHORT_d1_7": list(range(1, 8)),  "H_LONG_d8_28": list(range(8, 29))}
    h_ld  = {"H1_d1_7":     list(range(1, 8)),  "H2_d8_14":     list(range(8, 15))}

    # ── Global metrics ─────────────────────────────────────────────────────────
    print("Computing global metrics...")
    m5_mets = compute_metrics(m5)
    ld_mets = compute_metrics(ld)

    # ── GO/NO_GO ───────────────────────────────────────────────────────────────
    print("Evaluating GO/NO_GO...")
    gng = {}
    for ds_name, panel, hg in [("M5", m5, h_m5), ("LD2011", ld, h_ld)]:
        pv = panel.dropna(subset=["Real", "BL1_S4DR", "B2_THETA", "B1_ETS"])
        r, bl1, th, ets = (pv["Real"].values, pv["BL1_S4DR"].values,
                           pv["B2_THETA"].values, pv["B1_ETS"].values)
        A = bool(mae(r, bl1) < mae(r, th))
        B = bool(smape(r, bl1) < smape(r, th))
        C = bool(mae(r, bl1) < mae(r, ets))
        D = bool(smape(r, bl1) < smape(r, ets))
        ps = per_series_dmae(panel, "BL1_S4DR", "B2_THETA")
        E = bool(int((ps < 0).sum()) > int((ps > 0).sum()))
        F = bool(float(ps.median()) < 0)
        lo = loso(panel, "BL1_S4DR", "B2_THETA")
        G = bool(lo["flip_pct"] <= 10.0)
        hor = horizon_dmae(panel, "BL1_S4DR", "B2_THETA", hg)
        H = bool(all(v <= 0 for v in hor.values()))
        bl1r = rmse(r, bl1); thr = rmse(r, th)
        I = bool((bl1r - thr) / thr * 100 < 3.0)
        gng[ds_name] = dict(
            A=A, B=B, C=C, D=D, E=E, F=F, G=G, H=H, I=I,
            loso=lo, horizons=hor,
            BL1_MAE=mae(r, bl1), THETA_MAE=mae(r, th), ETS_MAE=mae(r, ets),
            BL1_sMAPE=smape(r, bl1), THETA_sMAPE=smape(r, th),
            BL1_RMSE=bl1r, THETA_RMSE=thr,
        )

    all_pass = all(
        v for dr in [gng["M5"], gng["LD2011"]]
        for k, v in dr.items() if k in ("A","B","C","D","E","F","G","H","I")
    )
    gng["GO_NOGO"] = "GO" if all_pass else "NO_GO"

    # ── M6 replicates ──────────────────────────────────────────────────────────
    print("Evaluating M6 replicates...")
    m6r = {}
    for ds_name, panel in [("M5", m5), ("LD2011", ld)]:
        active = panel[~panel["M6_IsFallback"]].dropna(subset=["Real", "M6_PRED", "BL1_S4DR"])
        if len(active) == 0:
            m6r[ds_name] = {f"C{i}": False for i in range(1, 7)}
            continue
        C1 = bool(mae(active["Real"].values, active["M6_PRED"].values) <
                  mae(active["Real"].values, active["BL1_S4DR"].values))
        ps2 = per_series_dmae(active, "M6_PRED", "BL1_S4DR")
        C2  = bool((ps2 < 0).sum() / len(ps2) >= 0.5) if len(ps2) > 0 else False
        origins = sorted(active["Origin"].unique()); mid = len(origins) // 2
        p1 = active[active["Origin"].isin(set(origins[:mid]))]
        p2 = active[active["Origin"].isin(set(origins[mid:]))]
        if len(p1) > 0 and len(p2) > 0:
            dm1 = mae(p1["Real"].values,p1["M6_PRED"].values) - mae(p1["Real"].values,p1["BL1_S4DR"].values)
            dm2 = mae(p2["Real"].values,p2["M6_PRED"].values) - mae(p2["Real"].values,p2["BL1_S4DR"].values)
            C3  = bool(dm1 <= TOLERANCES["temporal_sign_neutrality"] and
                       dm2 <= TOLERANCES["temporal_sign_neutrality"])
        else:
            C3 = False
        C4 = bool(smape(active["Real"].values, active["M6_PRED"].values) <=
                  smape(active["Real"].values, active["BL1_S4DR"].values) +
                  TOLERANCES["smape_material_deterioration"])
        C5 = bool(p90ae(active["Real"].values, active["M6_PRED"].values) <=
                  p90ae(active["Real"].values, active["BL1_S4DR"].values) +
                  TOLERANCES["p90ae_material_deterioration"])
        lo2 = loso(active, "M6_PRED", "BL1_S4DR")
        C6  = bool(lo2["flip_pct"] <= TOLERANCES["loso_max_flip_pct"])
        m6r[ds_name] = dict(
            C1=C1, C2=C2, C3=C3, C4=C4, C5=C5, C6=C6, loso=lo2,
            M6_active_MAE  = mae(active["Real"].values, active["M6_PRED"].values),
            BL1_active_MAE = mae(active["Real"].values, active["BL1_S4DR"].values),
        )

    m5_ok = all(m6r["M5"].get(f"C{i}") for i in range(1, 7))
    ld_ok  = all(m6r["LD2011"].get(f"C{i}") for i in range(1, 7))
    m6r["M6_REPLICATES"] = "TRUE" if (m5_ok and ld_ok) else "FALSE"

    # ── H_PR2 ──────────────────────────────────────────────────────────────────
    pr2_cands = ["LY_DOM", "T28", "PAY_CYCLE"]
    h_pr2_details: dict = {}
    for ds, panel in [("M5", m5), ("LD2011", ld)]:
        attr = m6_attribution(panel); h_pr2_details[ds] = {}
        for c in pr2_cands:
            row = attr[attr["Candidate"] == c]
            h_pr2_details[ds][c] = row.iloc[0].to_dict() if len(row) > 0 else {"dMAE": np.nan, "N_selected": 0}

    dmae_vals = [h_pr2_details[ds][c].get("dMAE", np.nan)
                 for ds in ("M5", "LD2011") for c in pr2_cands]
    finite = [v for v in dmae_vals if np.isfinite(v)]
    if not finite:          resultado_h_pr2 = "MIXTA"
    elif all(v > 0  for v in finite): resultado_h_pr2 = "CONFIRMADA"
    elif all(v <= 0 for v in finite): resultado_h_pr2 = "REFUTADA"
    else:                   resultado_h_pr2 = "MIXTA"

    # ── Paper direction ────────────────────────────────────────────────────────
    if gng["GO_NOGO"] == "GO":
        paper_dir = "ACCURACY_PAPER"
    elif m6r["M6_REPLICATES"] == "TRUE":
        paper_dir = "METHODOLOGICAL_PAPER_POSITIVE"
    else:
        paper_dir = "METHODOLOGICAL_PAPER_NEGATIVE_OR_CLOSE"

    # ── Persist outputs (before printing final report) ─────────────────────────
    print("Persisting metric outputs...")
    for ds_name, panel, mets, out_dir, hg in [
        ("M5_STORE_DEPT", m5, m5_mets, OUT_M5, h_m5),
        ("LD2011_DAILY",  ld, ld_mets,  OUT_LD,  h_ld),
    ]:
        pv = panel.dropna(subset=["Real"] + METHODS)
        r  = pv["Real"].values

        pd.DataFrame([{"Method": m, **mets[m]["MICRO"], "MACRO_MAE": mets[m]["MACRO_MAE"]}
                      for m in METHODS]).to_csv(out_dir / "global_metrics.csv", index=False)

        pd.DataFrame([{"SeriesId": sid,
                       **{f"MAE_{m}": mae(s["Real"].values, s[m].values) for m in METHODS}}
                      for sid in pv["SeriesId"].unique()
                      for s in [pv[pv["SeriesId"] == sid]]
                      ]).to_csv(out_dir / "per_series_metrics.csv", index=False)

        pd.DataFrame([{"Method": m,
                       "dMAE_vs_THETA":   mae(r, pv[m].values)   - mae(r, pv["B2_THETA"].values),
                       "dsMAPE_vs_THETA": smape(r, pv[m].values) - smape(r, pv["B2_THETA"].values),
                       "dRMSE_vs_THETA":  rmse(r, pv[m].values)  - rmse(r, pv["B2_THETA"].values),
                       "BL1_MAE": mae(r, pv["BL1_S4DR"].values),
                       "THETA_MAE": mae(r, pv["B2_THETA"].values)}
                      for m in METHODS]).to_csv(out_dir / "method_vs_theta_deltas.csv", index=False)

        pd.DataFrame([{"Method": m,
                       "dMAE_vs_ETS":   mae(r, pv[m].values)   - mae(r, pv["B1_ETS"].values),
                       "dsMAPE_vs_ETS": smape(r, pv[m].values) - smape(r, pv["B1_ETS"].values)}
                      for m in METHODS]).to_csv(out_dir / "method_vs_ets_deltas.csv", index=False)

        m6full = pv; m6act = pv[~pv["M6_IsFallback"]]
        pd.DataFrame([
            {"Universe": "FULL", "N": len(m6full),
             "M6_MAE":  mae(m6full["Real"].values, m6full["M6_PRED"].values),
             "BL1_MAE": mae(m6full["Real"].values, m6full["BL1_S4DR"].values),
             "dMAE":    mae(m6full["Real"].values, m6full["M6_PRED"].values) -
                        mae(m6full["Real"].values, m6full["BL1_S4DR"].values)},
            {"Universe": "ACTIVE_M6", "N": len(m6act),
             "M6_MAE":  mae(m6act["Real"].values, m6act["M6_PRED"].values)  if len(m6act) > 0 else np.nan,
             "BL1_MAE": mae(m6act["Real"].values, m6act["BL1_S4DR"].values) if len(m6act) > 0 else np.nan,
             "dMAE":   (mae(m6act["Real"].values, m6act["M6_PRED"].values) -
                        mae(m6act["Real"].values, m6act["BL1_S4DR"].values)) if len(m6act) > 0 else np.nan},
        ]).to_csv(out_dir / "m6_vs_bl1_deltas.csv", index=False)

        m6_attribution(panel).to_csv(out_dir / "m6_candidate_attribution.csv", index=False)

        lo_bl1 = loso(panel, "BL1_S4DR", "B2_THETA")
        pd.DataFrame([{"Comparison": "BL1_vs_THETA", **lo_bl1}
                      ]).to_csv(out_dir / "loso_analysis.csv", index=False)

        hor_bl1 = horizon_dmae(panel, "BL1_S4DR", "B2_THETA", hg)
        pd.DataFrame([{"Horizon_group": k, "dMAE_BL1_vs_THETA": v}
                      for k, v in hor_bl1.items()
                      ]).to_csv(out_dir / "horizon_analysis.csv", index=False)

    print("All metric outputs persisted.")
    print()

    # ── FINAL REPORT OBLIGATORIO ───────────────────────────────────────────────
    print(SEP)
    print("FASE 3 FINAL REPORT — S4DR_PUBLIC_BENCHMARK_FIRETEST_2")
    print(SEP)
    print("STATUS             = COMPLETE")
    print("EXPERIMENT_ID      = S4DR_PUBLIC_BENCHMARK_FIRETEST_2")
    print("FROZEN_SHA         = 67851d3")
    print("GIT_START_PHASE3   = 69e56a6")
    print("B3_METHOD_CHOSEN   = MSTL_AutoARIMA  (TBATS COMPUTATIONALLY_PROHIBITIVE)")
    print("M6_WARMUP_K        = 3")
    print()
    print(f"M5_STORE_DEPT  N_SERIES=70   N_EVAL_ORIGINS=8   H=28  ROWS={len(m5)}")
    print(f"               SHA256=46a7e1938bb2fda15ade5898a42c54600ae9906f1e0551828dc2a4baf41c3eaa")
    print(f"LD2011_DAILY   N_SERIES=328  N_EVAL_ORIGINS=12  H=14  ROWS={len(ld)}")
    print(f"               SHA256=77e528c66c3b3ac7b16e268bd31fb3187017e0dc4ad4008d970f402de5944aac")
    print()

    for ds_name, panel, mets in [("M5_STORE_DEPT", m5, m5_mets), ("LD2011_DAILY", ld, ld_mets)]:
        print(f"-- {ds_name} --")
        theta_mae = mets["B2_THETA"]["MICRO"]["MAE"]
        ets_mae   = mets["B1_ETS"]["MICRO"]["MAE"]
        for m in METHODS:
            mi = mets[m]["MICRO"]
            dm_th  = mi["MAE"] - theta_mae
            dm_ets = mi["MAE"] - ets_mae
            print(f"  {m:<30}  MAE={mi['MAE']:>14,.4f}  "
                  f"dMAE/THETA={dm_th:>+12,.4f}  "
                  f"dMAE/ETS={dm_ets:>+12,.4f}  "
                  f"sMAPE={mi['sMAPE']:>7.3f}%  "
                  f"MACRO_MAE={mets[m]['MACRO_MAE']:>14,.4f}")
        print()

    print(f"GO_NOGO = {gng['GO_NOGO']}")
    for ds in ["M5", "LD2011"]:
        dr = gng[ds]
        conds = "  ".join(f"{c}={'Y' if dr[c] else 'N'}" for c in "ABCDEFGHI")
        print(f"  {ds}: {conds}")
        print(f"       BL1_MAE={dr['BL1_MAE']:.4f}  THETA_MAE={dr['THETA_MAE']:.4f}  ETS_MAE={dr['ETS_MAE']:.4f}")
        print(f"       BL1_sMAPE={dr['BL1_sMAPE']:.3f}%  THETA_sMAPE={dr['THETA_sMAPE']:.3f}%")
        print(f"       LOSO_flip%={dr['loso']['flip_pct']:.1f}%  ({dr['loso']['N_flips']}/{dr['loso']['N_series']} series flip)")
        print(f"       horizons={dr['horizons']}")
        bl1r_pct = (dr["BL1_RMSE"] - dr["THETA_RMSE"]) / dr["THETA_RMSE"] * 100
        print(f"       RMSE_deg_vs_THETA={bl1r_pct:+.2f}%  (I threshold: <3.0%)")
    print()

    print(f"M6_REPLICATES = {m6r['M6_REPLICATES']}")
    for ds in ["M5", "LD2011"]:
        dr = m6r[ds]
        conds = "  ".join(f"C{i}={'Y' if dr.get(f'C{i}') else 'N'}" for i in range(1, 7))
        print(f"  {ds}: {conds}")
        if "M6_active_MAE" in dr:
            dm_active = dr["M6_active_MAE"] - dr["BL1_active_MAE"]
            print(f"       M6_active_MAE={dr['M6_active_MAE']:.4f}  "
                  f"BL1_active_MAE={dr['BL1_active_MAE']:.4f}  "
                  f"dMAE={dm_active:+.4f}")
            print(f"       LOSO_M6_vs_BL1 flip%={dr['loso']['flip_pct']:.1f}%")
    print()

    print("H_PR2 attribution (M6 candidate selection):")
    for ds in ["M5", "LD2011"]:
        print(f"  {ds}:")
        for c in pr2_cands:
            info = h_pr2_details[ds][c]
            dm   = info.get("dMAE", np.nan)
            n    = info.get("N_selected", 0)
            pct  = info.get("PCT_selected", 0)
            dm_s = f"{dm:+.4f}" if np.isfinite(dm) else "NaN"
            print(f"    {c:<16}: N={n:>5}  ({pct:>5.1f}%)  dMAE={dm_s}")
    print(f"RESULTADO_H_PR2   = {resultado_h_pr2}")
    print()
    print(f"PAPER_DIRECTION   = {paper_dir}")
    print()
    print(f"M5_FALLBACK_RATE  = {100*m5['M6_IsFallback'].mean():.1f}%")
    print(f"LD_FALLBACK_RATE  = {100*ld['M6_IsFallback'].mean():.1f}%")
    print()
    print("CANONICAL_MODEL_MODIFIED = FALSE")
    print("PHASE2_COMPLIANCE_ROLE   = INTERNAL_ONLY")
    print()
    print("COMPLETE.")
    print(SEP)


if __name__ == "__main__":
    main()
