# Evidence Pack — S4DR Manuscript Measurement Closure + MDA-1
ADDENDUM_ID = MDA-1_M5_CANDIDATE_REGENERATION
FROZEN_SHA  = 67851d3

## GO/NO_GO
GO_NOGO_RESULT = NO_GO
M6_REPLICATES  = FALSE
PAPER_DIRECTION = METHODOLOGICAL_PAPER

## M5_STORE_DEPT Key Results (N=70 series, 8 eval origins, H=28)
- L0 (BL1_S4DR) MICRO_MAE = 74.4431
- L4S (cross-fit) MICRO_MAE = 74.8836
- L5 (oracle) MICRO_MAE = 32.9153
- SC (M6_PRED) MICRO_MAE = 76.4560
- L3 best global candidate: ROLLING3M
- L4S selection stability: 40.0%

## LD2011_DAILY Key Results (N=328 series, 12 eval origins, H=14)
- L0 (BL1_S4DR) MICRO_MAE = 5103.7392
- L4S (cross-fit) MICRO_MAE = 5033.0362
- L5 (oracle) MICRO_MAE = 1956.8317
- SC (M6_PRED) MICRO_MAE = 5078.9792

## Decomposition Convention
dMAE = method_MAE - L0_MAE (negative = method favorable vs L0)
LIBRARY_RETROSPECTIVE_HEADROOM = L5_MAE - L0_MAE
STATIC_RETROSPECTIVE_SPECIALIZATION = L4_MAE - L0_MAE
BIAS_CORRECTED_SPECIALIZATION = L4S_MAE - L0_MAE
CAUSAL_SPECIALIZATION = SC_MAE - L0_MAE
CAPTURE_RATIO_CORRECTED = |CAUSAL| / |BIAS_CORRECTED| (if both < 0)

## Prohibited Inferences
- L3/L4/L4S/L5 are NOT "achievable improvement"
- L5 is a retrospective diagnostic order statistic
- Results do NOT reopen GO_NOGO
- EXP-006 and H_PR1 remain FUTURE_WORK_DECLARED

## Protocol Compliance
CANONICAL_MODEL_MODIFIED = FALSE
FINAL_HOLDOUT_ATM_OPENED = FALSE
S4DR_EXECUTED_FOR_LD2011 = FALSE
ATM_DATA_USED = FALSE
TUNING_PERFORMED = FALSE
MEASUREMENT_PHASE_OF_PROJECT = CLOSED



## WIKI_DAILY

**Source**: https://zenodo.org/records/4656080/files/kaggle_web_traffic_dataset_with_missing_values.zip?download=1
**SAMPLED_N**: 300
**N_EVAL_ORIGINS**: 12
**HORIZON**: 14
**FIRST_EVALUATION_ORIGIN**: 2017-06-11
**LAST_EVALUATION_ORIGIN**: 2017-08-27

### Ladder (MAE)
| Level | MAE | dMAE vs L0 |
|-------|-----|-----------|
| B0 | 564.2645 | +105.9922 |
| B1 | 702.2182 | +243.9459 |
| B2 | 526.0678 | +67.7955 |
| B3 | 683.5350 | +225.2627 |
| L0 | 458.2723 | +0.0000 |
| L1 | 575.5787 | +117.3064 |
| L2 | 539.1095 | +80.8371 |
| L3 | 515.0034 | +56.7311 |
| L4 | 411.3247 | -46.9476 |
| L4S | 537.1134 | +78.8410 |
| L5 | 235.1407 | -223.1316 |
| SC | 498.8312 | +40.5589 |

### Headroom Decomposition
- LIBRARY_RETROSPECTIVE_HEADROOM (L5): dMAE = -223.1316 (-48.69% of L0)
- STATIC_RETROSPECTIVE_SPECIALIZATION (L4): dMAE = -46.9476 (-10.24% of L0)
- BIAS_CORRECTED_SPECIALIZATION (L4S): dMAE = +78.8410 (+17.20% of L0)
- CAUSAL_SPECIALIZATION (SC): dMAE = +40.5589 (+8.85% of L0)
- CAPTURE_RATIO_CORRECTED: NOT_DEFINED
- L4S_SELECTION_STABILITY: 23.3%

### Key figures
- Source: candidate_standalone.csv, dataset=WIKI_DAILY, metric=MAE
- Source: headroom_ladder.csv, dataset=WIKI_DAILY

## THIRD-DOMAIN PROSPECTIVE REPLICATION

MDA-2 was prospectively specified after FIRETEST_2 and before inspection of WIKI_DAILY results.

| Expectation | Result | Justification |
|-------------|--------|---------------|
| E1_INFLATION | CONFIRMADA | d4=-46.9476<0 and (d4s=78.8410>=0 or |d4|>=3|d4s|) |
| E2_STABILITY | CONFIRMADA | SELECTION_STABILITY=23.3% < 60% |
| E3_CAPTURE | CONFIRMADA | CAPTURE_RATIO=NOT_DEFINED (d4s=78.8410, dsc=40.5589) |
| E4_ORACLE | CONFIRMADA | |dMAE_L5|=223.1316 > 0.4*MAE_L0=183.3089 |
| E5_LY | CONFIRMADA | Neither LY_DOM nor LY_SAME_BUCKET in top-3 MAE candidates: ['T7', 'ROLLING3M', 'T14'] |

**THIRD_DOMAIN_PATTERN = REPLICATES**
(Rule: REPLICATES if all 4 of E1-E4 confirmed; PARTIALLY if 2-3; CONTRADICTS if 0-1)
