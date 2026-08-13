"""
Statistical sign convention tests.

Canonical convention (frozen for the entire paper):
  d_i = MAE(L0, series_i) - MAE(comparator, series_i)
  RBC = (W_plus - W_minus) / (W_plus + W_minus)
    where W_plus  = sum of ranks where d_i > 0 (comparator better)
          W_minus = sum of ranks where d_i < 0 (L0 better)
  RBC < 0  →  L0 favorable
  RBC > 0  →  comparator favorable
  RBC = 0  →  neutral
"""

import math
import csv
import json
from pathlib import Path

REPO = Path(__file__).parent.parent
MDATA = REPO / "reference" / "public_benchmark_firetest" / "manuscript_data"
WIKI  = REPO / "reference" / "public_benchmark_firetest" / "wiki_daily"


# ── helper ────────────────────────────────────────────────────────────────────
def _rbc_from_stat_n(stat: float, n: int) -> float:
    """
    scipy two-sided Wilcoxon returns stat = min(W+, W-).
    Signed canonical RBC cannot be derived from stat alone without sign context.
    This helper just returns |RBC| = |W+ - W-| / (W+ + W-).
    """
    total = n * (n + 1) / 2
    other = total - stat
    w_plus_m_w_minus = abs(other - stat)
    return w_plus_m_w_minus / total


def _rbc_synthetic(d_array):
    """Compute canonical RBC from raw difference vector."""
    from scipy.stats import rankdata
    nz = [x for x in d_array if x != 0]
    if not nz:
        return 0.0
    ranks = rankdata([abs(x) for x in nz])
    w_plus  = sum(r for x, r in zip(nz, ranks) if x > 0)
    w_minus = sum(r for x, r in zip(nz, ranks) if x < 0)
    return (w_plus - w_minus) / (w_plus + w_minus)


# ── T_RBC_ALL_NEGATIVE ────────────────────────────────────────────────────────
def test_rbc_all_negative_means_l0_uniformly_better():
    """When L0 beats comparator on every series, canonical RBC = -1.0."""
    n = 20
    d = [-float(i) for i in range(1, n + 1)]  # all d < 0 → L0 always better
    rbc = _rbc_synthetic(d)
    assert math.isclose(rbc, -1.0, abs_tol=1e-10), f"Expected -1.0, got {rbc}"


# ── T_RBC_ALL_POSITIVE ────────────────────────────────────────────────────────
def test_rbc_all_positive_means_comparator_uniformly_better():
    """When comparator beats L0 on every series, canonical RBC = +1.0."""
    n = 20
    d = [float(i) for i in range(1, n + 1)]   # all d > 0 → comparator always better
    rbc = _rbc_synthetic(d)
    assert math.isclose(rbc, 1.0, abs_tol=1e-10), f"Expected +1.0, got {rbc}"


# ── T_RBC_SIGN_MATCHES_DMAE ───────────────────────────────────────────────────
def test_rbc_sign_matches_direction_of_differences():
    """Sign of RBC must match predominant direction of d_i differences."""
    import random
    rng = random.Random(42)
    # L0 mostly better: 15 out of 20 series
    d_l0_wins = [-rng.uniform(0.1, 5.0) for _ in range(15)] + \
                [rng.uniform(0.1, 5.0) for _ in range(5)]
    rbc = _rbc_synthetic(d_l0_wins)
    assert rbc < 0, f"L0 mostly wins → expect negative RBC, got {rbc}"

    # Comparator mostly better: 15 out of 20 series
    d_comp_wins = [rng.uniform(0.1, 5.0) for _ in range(15)] + \
                  [-rng.uniform(0.1, 5.0) for _ in range(5)]
    rbc2 = _rbc_synthetic(d_comp_wins)
    assert rbc2 > 0, f"Comparator mostly wins → expect positive RBC, got {rbc2}"


# ── T_WIKI_THETA_DIRECTION ────────────────────────────────────────────────────
def test_wiki_theta_rbc_negative_and_l0_favorable():
    """Wiki L0 vs Theta: RBC must be negative (L0_FAVORABLE)."""
    rows = list(csv.DictReader((WIKI / "statistical_tests.csv").read_text().splitlines()))
    row = next(r for r in rows if "THETA" in r["Comparison"])
    rbc = float(row["RANK_BISERIAL"])
    direction = row["DIRECTION"]
    assert rbc < 0, f"Wiki Theta RBC must be negative, got {rbc}"
    assert direction == "L0_FAVORABLE"
    # Verify |RBC| matches expectation from W statistic and n
    stat = float(row["STATISTIC"])
    n    = int(row["N_NONZERO"])
    expected_abs = _rbc_from_stat_n(stat, n)
    assert math.isclose(abs(rbc), expected_abs, rel_tol=1e-6), \
        f"|RBC|={abs(rbc)} vs expected {expected_abs}"


# ── T_WIKI_ETS_DIRECTION ─────────────────────────────────────────────────────
def test_wiki_ets_rbc_negative_and_l0_favorable():
    """Wiki L0 vs ETS: RBC must be negative (L0_FAVORABLE)."""
    rows = list(csv.DictReader((WIKI / "statistical_tests.csv").read_text().splitlines()))
    row = next(r for r in rows if "ETS" in r["Comparison"])
    rbc = float(row["RANK_BISERIAL"])
    direction = row["DIRECTION"]
    assert rbc < 0, f"Wiki ETS RBC must be negative, got {rbc}"
    assert direction == "L0_FAVORABLE"
    stat = float(row["STATISTIC"])
    n    = int(row["N_NONZERO"])
    expected_abs = _rbc_from_stat_n(stat, n)
    assert math.isclose(abs(rbc), expected_abs, rel_tol=1e-6), \
        f"|RBC|={abs(rbc)} vs expected {expected_abs}"


# ── T_M5_ALL_COMPARISONS ──────────────────────────────────────────────────────
def test_m5_rbc_sign_consistent_with_known_mae_direction():
    """
    M5: L0 MAE=74.44 is lower than all baselines (B0=116.4,B1=103.8,B2=100.3,B3=104.1).
    L0 beats all baselines globally → Wilcoxon should show RBC < 0 for B0-B3.
    L0 vs SC: SC=76.46 > L0=74.44 → L0 favorable → RBC < 0.
    """
    rows = list(csv.DictReader((MDATA / "wilcoxon_tests.csv").read_text().splitlines()))
    m5 = [r for r in rows if r["Dataset"] == "M5_STORE_DEPT"]
    for r in m5:
        rbc = float(r["rank_biserial"])
        assert rbc < 0, (
            f"M5 {r['comparison']}: expected RBC<0 (L0 favorable globally), got {rbc}"
        )


# ── T_LD_ALL_COMPARISONS ─────────────────────────────────────────────────────
def test_ld2011_rbc_sign_consistent_with_known_mae_direction():
    """
    LD2011: L0 MAE=5103.74
    B0=5578.6 > L0 → L0 wins → RBC < 0  ✓
    B1=5069.2 < L0 → B1 wins → RBC > 0  (ETS beats L0 on LD2011)
    B2=5016.7 < L0 → B2 wins → RBC > 0
    B3=4874.6 < L0 → B3 wins → RBC > 0
    SC=5079.0 < L0 → SC wins → RBC > 0   BUT wait: 5079 < 5103 → SC marginally better
    Actually SC MAE is slightly lower than L0 but Wilcoxon is per-series…
    The per-series result is what matters. We just check sign consistency.
    """
    rows = list(csv.DictReader((MDATA / "wilcoxon_tests.csv").read_text().splitlines()))
    ld = {r["comparison"]: float(r["rank_biserial"])
          for r in rows if r["Dataset"] == "LD2011_DAILY"}

    # B0: global B0 MAE=5578 > L0 MAE=5103 → L0 favorable → RBC < 0
    assert ld["L0_vs_B0"] < 0, f"LD2011 L0_vs_B0 RBC should be <0, got {ld['L0_vs_B0']}"

    # B1, B2, B3: global comparator MAE < L0 MAE → comparator favorable → RBC > 0
    for comp in ["L0_vs_B1", "L0_vs_B2", "L0_vs_B3"]:
        assert ld[comp] > 0, f"LD2011 {comp} RBC should be >0 (baseline wins globally), got {ld[comp]}"

    # L1, L2: global L1/L2 MAE >> L0 → L0 favorable → RBC < 0
    for comp in ["L0_vs_L1", "L0_vs_L2"]:
        assert ld[comp] < 0, f"LD2011 {comp} RBC should be <0 (L0 better), got {ld[comp]}"

    # L4: globally L4 MAE=4325 < L0 MAE=5103 → L4 favorable → RBC > 0
    assert ld["L0_vs_L4"] > 0, f"LD2011 L0_vs_L4 RBC should be >0 (L4 better), got {ld['L0_vs_L4']}"

    # L5: oracle always better → RBC ~ +1
    assert ld["L0_vs_L5"] > 0.99, f"LD2011 L0_vs_L5 RBC should be ~+1, got {ld['L0_vs_L5']}"


# ── T_WIKI_MANIFEST_CONSISTENCY ──────────────────────────────────────────────
def test_wiki_manifest_rbc_matches_statistical_tests_csv():
    """run_manifest.json RANK_BISERIAL must match wiki statistical_tests.csv."""
    manifest = json.loads((WIKI / "run_manifest.json").read_text())
    rows = list(csv.DictReader((WIKI / "statistical_tests.csv").read_text().splitlines()))

    row_theta = next(r for r in rows if "THETA" in r["Comparison"])
    row_ets   = next(r for r in rows if "ETS" in r["Comparison"])

    assert math.isclose(
        manifest["WILCOXON_L0_VS_THETA"]["RANK_BISERIAL"],
        float(row_theta["RANK_BISERIAL"]), rel_tol=1e-9
    ), "Theta RBC mismatch between manifest and statistical_tests.csv"

    assert math.isclose(
        manifest["WILCOXON_L0_VS_ETS"]["RANK_BISERIAL"],
        float(row_ets["RANK_BISERIAL"]), rel_tol=1e-9
    ), "ETS RBC mismatch between manifest and statistical_tests.csv"


# ── T_WILCOXON_CSV_COMPLETE ───────────────────────────────────────────────────
def test_wilcoxon_tests_csv_has_all_three_domains():
    """wilcoxon_tests.csv must contain rows for M5, LD2011, and WIKI_DAILY."""
    rows = list(csv.DictReader((MDATA / "wilcoxon_tests.csv").read_text().splitlines()))
    datasets = {r["Dataset"] for r in rows}
    assert "M5_STORE_DEPT" in datasets
    assert "LD2011_DAILY" in datasets
    assert "WIKI_DAILY" in datasets
