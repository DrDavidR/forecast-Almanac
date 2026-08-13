## PROVENANCE

```
PUBLIC_REPO_NAME    = s4dr-public-repro
PRIVATE_HISTORY_IMPORTED = FALSE
REPO_INIT_METHOD    = git-init-fresh-history
```

### Source archive

| Component | Origin |
|---|---|
| `src/s4dr/` | Extracted from private research repo (commit `67851d3`, SCIENTIFIC_CONFIGURATION_ATTRIBUTION_SHA) |
| `reference/public_benchmark_firetest/` | Output of M5/LD2011 firetest + MDA-2 Wiki replication |
| `tests/` | Structural gates and statistical sign convention tests |

### Sanitization applied

- Removed internal holdout compliance gates referencing private research directories
- Replaced internal runner references with equivalent public benchmark runner checks
- Synthetic series IDs (`E99999`) renamed to `SERIES_001` for clarity
- No forecast values, metrics, or statistical results were modified

### What is NOT included

This repository does not contain:

- Any operational or banking data of any type
- Operational machine identifiers or location data of any type
- Internal module `src/s4dr/clases.py` (contains operational identifiers, not used by benchmark runners)
- Private experimental results or holdout datasets
- Git history from the private research repository

### Experiment chronology

| Phase | Description | Date |
|---|---|---|
| BL1 frozen | Canonical 12-candidate model fixed | 2026-07-31 |
| FIRETEST_2 | Pre-registered evaluation on M5_STORE_DEPT + LD2011_DAILY | 2026-08 |
| GO_NOGO | Result: NO_GO (SC does not meet criterion on M5) | 2026-08-12 |
| MDA-2 | Prospective third-domain replication on Wikipedia daily traffic | 2026-08-13 |
| Public release prep | Statistical sign correction + consistency audit | 2026-08-13 |

MDA-2 (Wiki) was **prospectively specified** after FIRETEST_2 measurement closure and before inspection of Wiki results. It is NOT part of the pre-registered evaluation.
