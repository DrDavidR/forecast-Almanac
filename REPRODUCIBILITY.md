## REPRODUCIBILITY

### Frozen configuration

```
FROZEN_SHA              = 67851d3
SCIENTIFIC_ATTRIBUTION  = 67851d3 (ML-column-presence gate + unknown-kind raises + expanded tests)
CANONICAL_CANDIDATES    = 12 structural (no ML)
B3_METHOD               = MSTL_AutoARIMA (TBATS computationally prohibitive)
M6_WARMUP_K             = 3
WILCOXON_CONVENTION     = canonical signed RBC (see tests/test_statistical_sign_convention.py)
```

### Measurement closure

```
MDA1_FIRETEST_STATUS    = CLOSED (M5_STORE_DEPT + LD2011_DAILY)
GO_NOGO_RESULT          = NO_GO
  Criterion: SC beats L0 on BOTH primary domains (per-series majority)
  Result: SC worse than L0 on M5; SC marginally better on LD2011 per-series

MDA2_WIKI_STATUS        = CLOSED (Wikipedia daily, prospective replication)
  ADDENDUM_TYPE         = PROSPECTIVELY_SPECIFIED_POST_FIRETEST_REPLICATION
  THIRD_DOMAIN_PATTERN  = REPLICATES (E1-E5 all CONFIRMADA)
```

### Wilcoxon sign correction (2026-08-13)

A sign inconsistency was found in the Wiki Wilcoxon computation. The runner used
`rbc = 1 - 4*stat/(n*(n+1))` which gives `|RBC|` (always positive). The canonical
formula requires `(W+ - W-) / (W+ + W-)`.

**Nature of correction:** Sign only. No W statistics, p-values, magnitudes, or
directions were changed. Both Wiki comparisons (L0 vs ETS, L0 vs Theta) are
L0_FAVORABLE — the sign correction makes RBC negative, consistent with the
convention.

**Files corrected:**
- `reference/public_benchmark_firetest/wiki_daily/statistical_tests.csv`
- `reference/public_benchmark_firetest/wiki_daily/run_manifest.json`
- `reference/public_benchmark_firetest/manuscript_data/wilcoxon_tests.csv`
- `reference/public_benchmark_firetest/run_mda2_wiki_daily.py`

**Classification:** SIGN_CONVENTION_DOCUMENTARY_CORRECTION (Case A: sign only).
No scientific conclusions changed.

### Preregistration chronology

1. Pre-registered evaluation: **M5_STORE_DEPT + LD2011_DAILY** (FIRETEST_2)
2. GO/NO_GO evaluated on these two domains only: **NO_GO**
3. After measurement closure: prospectively specified **WIKI_DAILY** replication (MDA-2)
4. Wiki results confirmed the pattern (E1-E5 CONFIRMADA)

**FORBIDDEN CLAIM:** "Three datasets were preregistered." Wiki was NOT preregistered.

### Reproducing from scratch

```bash
# Step 1: Download datasets (see runner headers for instructions)
# M5: pip install datasetsforecast; python -c "from datasetsforecast.m5 import M5; M5.load('.')"
# LD2011: download from UCI ML Repository
# Wiki: see run_mda2_wiki_daily.py header

# Step 2: Run M5 + LD2011 firetest
python reference/public_benchmark_firetest/run_s4dr_public_firetest.py

# Step 3: Run Wiki replication
python reference/public_benchmark_firetest/run_mda2_wiki_daily.py

# Step 4: Compute joint manuscript metrics
python reference/public_benchmark_firetest/compute_manuscript_measurements.py
python reference/public_benchmark_firetest/compute_joint_metrics.py

# Step 5: Verify results against pre-computed manuscript data
# Tables in reference/public_benchmark_firetest/manuscript_data/ should match
```

### Expected key values

| Domain | Level | MAE |
|---|---|---|
| M5_STORE_DEPT | L0 | 74.4431 |
| M5_STORE_DEPT | L5 | 32.9153 |
| LD2011_DAILY | L0 | 5103.7392 |
| LD2011_DAILY | L5 | 1956.8317 |
| WIKI_DAILY | L0 | 458.2723 |
| WIKI_DAILY | L5 | 235.1407 |

All values are MICRO_MAE (micro-averaged across all series in the domain).
