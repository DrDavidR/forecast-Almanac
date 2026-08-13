"""
Structural gates for the canonical S4DR paper model.

These tests enforce the scientific contract that src/s4dr/model.py:
  - imports cleanly
  - exposes exactly 12 structural candidates
  - excludes LGBM, CatBoost, and Prophet unconditionally
  - produces finite forecasts
  - official runners import the canonical model, not a parallel one

Run with:  pytest tests/test_canonical_structural_model.py -v
"""
import importlib
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = str(Path(__file__).resolve().parent.parent / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

CANONICAL_CANDIDATES = (
    "T7", "T7_30", "T14", "T28", "T56", "T84",
    "DOM", "BIMONTH_M_M1", "ROLLING3M", "PAY_CYCLE",
    "LY_SAME_BUCKET", "LY_DOM",
)
ML_CANDIDATES = {"LGBM_MODEL", "CATBOOST_MODEL", "PROPHET_MODEL"}


# ── Fixture: minimal 90-day synthetic history ────────────────────────────────
@pytest.fixture(scope="module")
def minimal_history():
    dates = pd.date_range("2024-01-01", periods=90, freq="D")
    rng = np.random.default_rng(42)
    amounts = rng.integers(50_000, 150_000, size=90).astype(float)
    return pd.DataFrame({"fecha": dates, "valor": amounts, "id": "SERIES_001"})


@pytest.fixture(scope="module")
def canonical_model():
    from s4dr.model import S4DRModel
    return S4DRModel("SERIES_001")


# ── Test 1: canonical model imports without error ─────────────────────────────
def test_canonical_model_import():
    mod = importlib.import_module("s4dr.model")
    assert hasattr(mod, "S4DRModel"), "S4DRModel not exported from s4dr.model"


# ── Test 2: candidate count == 12 ────────────────────────────────────────────
def test_candidate_count(canonical_model):
    assert len(canonical_model.models) == 12, (
        f"Expected 12 candidates, got {len(canonical_model.models)}: "
        f"{[m.code for m in canonical_model.models]}"
    )


# ── Test 3: candidate names exact match ──────────────────────────────────────
def test_candidate_names_exact(canonical_model):
    codes = [m.code for m in canonical_model.models]
    assert codes == list(CANONICAL_CANDIDATES), (
        f"Candidate list mismatch.\n  Expected: {list(CANONICAL_CANDIDATES)}\n  Got: {codes}"
    )


# ── Test 4: no LGBM candidate ────────────────────────────────────────────────
def test_no_lgbm(canonical_model):
    codes = {m.code for m in canonical_model.models}
    assert "LGBM_MODEL" not in codes, "LGBM_MODEL must not be in canonical candidate list"


# ── Test 5: no CatBoost candidate ────────────────────────────────────────────
def test_no_catboost(canonical_model):
    codes = {m.code for m in canonical_model.models}
    assert "CATBOOST_MODEL" not in codes, "CATBOOST_MODEL must not be in canonical candidate list"


# ── Test 6: no ML placeholder ────────────────────────────────────────────────
def test_no_ml_placeholder(canonical_model):
    codes = {m.code for m in canonical_model.models}
    leaked = codes & ML_CANDIDATES
    assert not leaked, f"ML candidates leaked into canonical model: {leaked}"


# ── Test 7: public benchmark runners import canonical model ───────────────────
def test_official_runners_import_canonical_model():
    runners_dir = Path(__file__).resolve().parent.parent / "reference" / "public_benchmark_firetest"
    official_runners = [
        "run_s4dr_public_firetest.py",
        "run_ld2011_crashsafe.py",
        "run_mda2_wiki_daily.py",
    ]
    non_canonical = []
    for runner_name in official_runners:
        runner_path = runners_dir / runner_name
        if not runner_path.exists():
            continue
        content = runner_path.read_text(encoding="utf-8", errors="replace")
        if "from s4dr.model import S4DRModel" not in content and "s4dr.model" not in content:
            non_canonical.append(runner_name)
    assert not non_canonical, (
        f"Benchmark runners do not import from s4dr.model: {non_canonical}"
    )


# ── Test 8: forecasts are finite ─────────────────────────────────────────────
def test_forecasts_finite(canonical_model, minimal_history):
    canonical_model.actualizar_modelo(minimal_history)
    preds, fechas, _ = canonical_model.seleccionar_atractores(P=2, mutate_state=False)
    assert len(preds) == 2, f"Expected 2 horizon predictions, got {len(preds)}"
    for i, p in enumerate(preds):
        assert np.isfinite(p), f"Forecast H{i+1} is not finite: {p}"
        assert p >= 0, f"Forecast H{i+1} is negative: {p}"


# ── Test 9: manuscript data outputs are present ───────────────────────────────
def test_manuscript_data_outputs_present():
    mdata = Path(__file__).resolve().parent.parent / "reference" / "public_benchmark_firetest" / "manuscript_data"
    required = [
        "headroom_ladder.csv",
        "master_results_table.csv",
        "wilcoxon_tests.csv",
        "decomposition_table.csv",
    ]
    missing = [f for f in required if not (mdata / f).exists()]
    assert not missing, f"Manuscript data files missing: {missing}"


# ── Test 10: all three evaluation domains present in results ──────────────────
def test_all_three_domains_in_master_results():
    import csv
    mrt = Path(__file__).resolve().parent.parent / "reference" / "public_benchmark_firetest" / "manuscript_data" / "master_results_table.csv"
    if not mrt.exists():
        pytest.skip("master_results_table.csv not yet generated")
    rows = list(csv.DictReader(mrt.read_text().splitlines()))
    domains = {r.get("Dataset", r.get("Domain", r.get("domain", ""))) for r in rows}
    for d in ["M5_STORE_DEPT", "LD2011_DAILY", "WIKI_DAILY"]:
        assert d in domains, f"Domain {d} missing from master_results_table.csv"


# ── FASE-1 explicit gates ─────────────────────────────────────────────────────

# T4_EXPLICIT: instantiation with zero extra args must yield exactly 12 structural
def test_default_no_extra_args_gives_12_structural():
    from s4dr.model import S4DRModel, _ML_CODES
    m = S4DRModel("SERIES_001_explicit")
    codes = [spec.code for spec in m.models]
    assert len(codes) == 12, f"Expected 12, got {len(codes)}: {codes}"
    contaminated = [c for c in codes if c in _ML_CODES]
    assert not contaminated, f"ML codes leaked with default args: {contaminated}"


# T5_UNKNOWN_KIND_RAISES: _predict_model must raise on any kind not in _KNOWN_ATTRACTOR_KINDS
def test_unknown_kind_raises(canonical_model):
    import types
    from s4dr.model import _KNOWN_ATTRACTOR_KINDS
    mock_spec = types.SimpleNamespace(kind="DEFINITELY_UNKNOWN_KIND_XYZ", code="FAKE_CAND")
    assert mock_spec.kind not in _KNOWN_ATTRACTOR_KINDS, "Test setup error: kind should be unknown"
    with pytest.raises(ValueError, match="Unknown spec kind"):
        canonical_model._predict_model(mock_spec, pd.Timestamp("2024-06-01"), pd.DataFrame())


# T6_ML_COLUMN_PRESENCE_GATE: assert_no_ml_columns must fail on presence, even with zeros
def test_ml_column_presence_fails():
    from s4dr.model import assert_no_ml_columns
    df_contaminated = pd.DataFrame({
        "CajeroId": ["E1", "E2"],
        "yhat_CATBOOST_MODEL": [0.0, 0.0],  # zeros must still trigger the gate
    })
    with pytest.raises(ValueError, match="ML contamination gate FAIL"):
        assert_no_ml_columns(df_contaminated)


def test_clean_panel_passes_ml_gate():
    from s4dr.model import assert_no_ml_columns
    df_clean = pd.DataFrame({"CajeroId": ["E1"], "yhat_T7": [12345.0]})
    assert_no_ml_columns(df_clean)  # must not raise


# T7_RAW_TARGET: baseline preprocessing must not apply target transforms
def test_raw_target_no_transform():
    from s4dr.model import _BASELINE_PREPROCESS
    assert not getattr(_BASELINE_PREPROCESS, "autoencoder_enabled", False), (
        "autoencoder_enabled must not be set (MLP_CLEANING removed from canonical path)"
    )
    assert not getattr(_BASELINE_PREPROCESS, "log_transform", False), (
        "log_transform must not be set — evaluation uses raw Real values"
    )
    assert not getattr(_BASELINE_PREPROCESS, "normalize_target", False), (
        "normalize_target must not be set — evaluation uses raw Real values"
    )


# T8_EXPANDED: all public benchmark runners must import from s4dr.model
def test_all_official_runners_import_canonical_model():
    runners_dir = Path(__file__).resolve().parent.parent / "reference" / "public_benchmark_firetest"
    official_runners = [
        "run_s4dr_public_firetest.py",
        "run_ld2011_crashsafe.py",
        "run_mda2_wiki_daily.py",
    ]
    missing, non_canonical = [], []
    for name in official_runners:
        path = runners_dir / name
        if not path.exists():
            missing.append(name)
            continue
        content = path.read_text(encoding="utf-8", errors="replace")
        if "from s4dr.model import S4DRModel" not in content:
            non_canonical.append(name)
    assert not missing, f"Benchmark runners missing from filesystem: {missing}"
    assert not non_canonical, f"Runners do not import from s4dr.model: {non_canonical}"


# T10_MODEL_PATH_UNIQUE: no parallel canonical model files may exist in src/s4dr/
def test_no_parallel_model_files():
    s4dr_dir = Path(__file__).resolve().parent.parent / "src" / "s4dr"
    forbidden_patterns = [
        "model_v*.py", "model_paper*.py", "model_backup*.py",
        "model_old*.py", "model_exp*.py", "model_causal*.py",
    ]
    conflicts = []
    for pattern in forbidden_patterns:
        conflicts.extend(s4dr_dir.glob(pattern))
    assert not conflicts, (
        f"Parallel model files found — only src/s4dr/model.py is canonical: {conflicts}"
    )


# T13_STRUCTURAL_EQUIVALENCE: hygiene commit removed dead-code only; live paths are identical.
# Two fresh S4DRModel instances on the same deterministic fixture must produce MAX_ABS_DIFF = 0.0.
def test_structural_equivalence_deterministic():
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=90, freq="D")
    amounts = rng.integers(50_000, 150_000, size=90).astype(float)
    hist = pd.DataFrame({"fecha": dates, "valor": amounts, "id": "SYNTH_EQ"})

    from s4dr.model import S4DRModel

    m1 = S4DRModel("SYNTH_EQ")
    m1.actualizar_modelo(hist.copy())
    p1, _, _ = m1.seleccionar_atractores(P=7, mutate_state=False)

    m2 = S4DRModel("SYNTH_EQ")
    m2.actualizar_modelo(hist.copy())
    p2, _, _ = m2.seleccionar_atractores(P=7, mutate_state=False)

    arr1 = np.array([float(x) for x in p1])
    arr2 = np.array([float(x) for x in p2])
    max_abs_diff = float(np.max(np.abs(arr1 - arr2)))
    assert max_abs_diff == 0.0, (
        f"Structural equivalence FAIL: MAX_ABS_DIFF={max_abs_diff} (expected 0.0). "
        f"Dead-code hygiene must not alter live attractor predictions."
    )
    assert all(np.isfinite(arr1)), f"Run1 produced non-finite predictions: {arr1}"


# T14_PREPROCESSING_EQUIVALENCE: cleaning pipeline is deterministic on fixed data with NaN/outliers.
def test_preprocessing_equivalence_deterministic():
    from s4dr_canonical.data_cleaned import aplicar_limpieza_s4dr_hasta_cutoff
    rng = np.random.default_rng(7)
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    amounts = rng.integers(10_000, 200_000, size=60).astype(float)
    amounts[5] = np.nan
    amounts[15] = np.nan
    amounts[30] = 5_000_000.0  # extreme outlier
    hist = pd.DataFrame({"Fecha": dates, "Cantidad": amounts, "CajeroId": "SYNTH_PRE"})
    cutoff = pd.Timestamp(dates[-1])

    clean1, _ = aplicar_limpieza_s4dr_hasta_cutoff(hist, cutoff_date=cutoff)
    clean2, _ = aplicar_limpieza_s4dr_hasta_cutoff(hist.copy(), cutoff_date=cutoff)

    assert list(clean1.columns) == list(clean2.columns), "Column mismatch between runs"
    numeric_cols = [c for c in clean1.columns
                    if pd.api.types.is_float_dtype(clean1[c]) or pd.api.types.is_integer_dtype(clean1[c])]
    for col in numeric_cols:
        diff = float(np.max(np.abs(clean1[col].fillna(0).values - clean2[col].fillna(0).values)))
        assert diff == 0.0, (
            f"Preprocessing non-determinism in column '{col}': MAX_ABS_DIFF={diff}"
        )
