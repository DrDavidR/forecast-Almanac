# S4DR: Structural Candidate Library for Daily Forecasting

Public reproducibility repository for:

> "Specialization Headroom in Structural Candidate Libraries:
>  A Three-Domain Empirical Analysis of Selector Limitations"

**Repository:** https://github.com/DrDavidR/forecast-Almanac

**Preprint:** arXiv (forthcoming)

---

## What this is

This repository provides:

1. **The S4DR model** (`src/s4dr/`) — a structural 12-candidate ensemble for daily time series forecasting. All candidates are interpretable structural formulas; no ML models.

2. **Benchmark runners** (`reference/public_benchmark_firetest/`) — scripts that reproduce the three-domain evaluation on M5_STORE_DEPT, LD2011_DAILY, and Wikipedia web traffic.

3. **Pre-computed manuscript data** (`reference/public_benchmark_firetest/manuscript_data/`) — the exact CSV files used to generate paper tables.

4. **Tests** (`tests/`) — structural gates that enforce causal integrity, 12-candidate count, no-ML constraint, and statistical sign convention.

---

## The 12 structural candidates

| Code | Description |
|---|---|
| T7 | 7-day trailing mean |
| T7_30 | 7-day mean, 30-day lookback |
| T14 | 14-day trailing mean |
| T28 | 28-day trailing mean |
| T56 | 56-day trailing mean |
| T84 | 84-day trailing mean |
| DOM | Day-of-month historical mean |
| BIMONTH_M_M1 | Bi-monthly pattern (current + prior month) |
| ROLLING3M | 3-month rolling mean |
| PAY_CYCLE | Pay-cycle aligned pattern |
| LY_SAME_BUCKET | Same bucket last year |
| LY_DOM | Same day-of-month last year |

---

## Datasets (not included — download separately)

| Dataset | Source |
|---|---|
| M5_STORE_DEPT | `datasetsforecast` pip package (`from datasetsforecast.m5 import M5`) |
| LD2011_DAILY | UCI ML Repository: "ElectricityLoadDiagrams20112014" |
| Wikipedia daily | Wikimedia Analytics (see runner for download instructions) |

---

## Headroom decomposition ladder

| Level | Label | Definition |
|---|---|---|
| L0 | BL1_S4DR | Naive baseline: equal-weight mean of all 12 candidates |
| L1 | UNIFORM_MEAN_12 | Same as L0 (alias) |
| L2 | MEDIAN_12 | Median of 12 candidates |
| L3 | BEST_GLOBAL_RETRO | Best single candidate globally (retrospective) |
| L4 | BEST_STATIC_RETRO | Best static candidate per series (retrospective) |
| L4S | BEST_STATIC_SPLIT | L4 estimated on training split (cross-fitted) |
| L5 | PER_ORIGIN_ORACLE | Per-origin oracle — retrospective upper bound only |
| SC | STATIC_CAUSAL_SELECTOR | Causal selector trained on expanding window |

L5 is a **diagnostic upper bound**, not a deployable target.

---

## Key findings (three-domain summary)

| Domain | L5 oracle gap | L4S inflation | SC vs L0 |
|---|---|---|---|
| M5_STORE_DEPT | −55.8% | +0.6% (inflated) | +2.7% (worse) |
| LD2011_DAILY | −61.6% | −1.4% (favorable) | −0.5% (capture ratio 35%) |
| WIKI_DAILY | −48.7% | +17.2% (inflated) | +8.9% (worse) |

---

## Quick start

```bash
pip install -r requirements.txt

# Run M5 + LD2011 firetest (requires data download — see runner header)
python reference/public_benchmark_firetest/run_s4dr_public_firetest.py

# Run Wikipedia replication
python reference/public_benchmark_firetest/run_mda2_wiki_daily.py

# Run tests
pytest tests/ -v
```

---

## Reproducing manuscript tables

After running the benchmark runners:

```bash
python reference/public_benchmark_firetest/compute_manuscript_measurements.py
python reference/public_benchmark_firetest/compute_joint_metrics.py
```

Pre-computed results are already in `reference/public_benchmark_firetest/manuscript_data/`.

---

## Statistical convention

All Wilcoxon signed-rank statistics use:

```
d_i = MAE(L0, series_i) - MAE(comparator, series_i)
RBC = (W+ - W-) / (W+ + W-)
  where W+ = sum of ranks where d_i > 0 (comparator better)
        W- = sum of ranks where d_i < 0 (L0 better)
RBC < 0  →  L0 favorable
RBC > 0  →  comparator favorable
```

---

## Citation

See `CITATION.cff` (forthcoming on arXiv submission).

---

## License

Apache 2.0 — see `LICENSE`.
