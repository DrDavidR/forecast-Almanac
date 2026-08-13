
# ============================================================
# Es la misma version 100, pero desacoplando clases. 13-02-2026
# Mantiene misma lógica/cálculos; solo desacopla arquitectura.
# ============================================================
import os
import json
from dataclasses import dataclass
from datetime import time, timedelta
from typing import Optional, Dict, List, Tuple, Any, Set, Callable, Sequence
from collections import Counter
import numpy as np
import pandas as pd
import inspect
import logging
# Limpieza y selector
try:
    from .selector_reliable_v3 import (
        init_state,
        select_candidates_for_day,
        update_weights,
    )
except Exception as e:
    raise ImportError(f"No se pudo importar selector_v3: {e}")

# =============================
# 1) S4DR (Separado por en clases) 13-02-2025
# =============================

@dataclass(frozen=True)
class _ModelSpec:
    name: str
    code: str                     
    kind: str                 
    period_days: Optional[int] = None
    lookback_days: Optional[int] = None   

#  política de historial / atractores

class PeriodSpecFactory:
    """Única fuente de verdad para construir la lista de specs."""
    @staticmethod
    def weekly_phase(period_days: int, *, code: Optional[str] = None,
                     name: Optional[str] = None, lookback_days: Optional[int] = None) -> _ModelSpec:
        p = int(period_days)
        c = str(code or f"T{p}")
        n = str(name or c)
        return _ModelSpec(name=n, code=c, kind="weekly_phase", period_days=p, lookback_days=lookback_days)

    @staticmethod
    def dom(*, code="DOM", name="DOM") -> _ModelSpec:
        return _ModelSpec(name=name, code=code, kind="dom")

    @staticmethod
    def bimonth(*, code="BIMONTH_M_M1", name="BIMONTH_M_M1") -> _ModelSpec:
        return _ModelSpec(name=name, code=code, kind="bimonth")

    @staticmethod
    def rolling3(*, code="ROLLING3M", name="ROLLING3M") -> _ModelSpec:
        return _ModelSpec(name=name, code=code, kind="rolling3")

    @staticmethod
    def paycycle(*, code="PAY_CYCLE", name="PAY_CYCLE") -> _ModelSpec:
        return _ModelSpec(name=name, code=code, kind="paycycle")

    @staticmethod
    def ly_bucket(*, code="LY_SAME_BUCKET", name="LY_SAME_BUCKET") -> _ModelSpec:
        return _ModelSpec(name=name, code=code, kind="ly_bucket")

    @staticmethod
    def ly_dom(*, code="LY_DOM", name="LY_DOM") -> _ModelSpec:
        return _ModelSpec(name=name, code=code, kind="ly_dom")

    @classmethod
    def default_specs(cls) -> List[_ModelSpec]:
        return [
            cls.weekly_phase(7,  code="T7"),
            cls.weekly_phase(7,  code="T7_30", lookback_days=30),
            cls.weekly_phase(14, code="T14"),
            cls.weekly_phase(28, code="T28"),
            cls.weekly_phase(56, code="T56"),
            cls.weekly_phase(84, code="T84"),
            cls.dom(),
            cls.bimonth(),
            cls.rolling3(),
            cls.paycycle(),
            cls.ly_bucket(),
            cls.ly_dom(),
        ]

    @staticmethod
    def apply_filters(
        specs: Sequence[_ModelSpec],
        *,
        allowed_specs: Optional[Set[str]] = None,
        enabled_specs: Optional[Set[str]] = None,
        disabled_specs: Optional[Set[str]] = None,
    ) -> List[_ModelSpec]:
        out = list(specs)
        if allowed_specs is not None:
            out = [s for s in out if s.code in allowed_specs]
        elif enabled_specs is not None:
            out = [s for s in out if s.code in enabled_specs]
        if disabled_specs:
            out = [s for s in out if s.code not in disabled_specs]
        return out


@dataclass
class HistoryWindowPolicy:
    """Política única para ventanas históricas + trazabilidad."""
    short_lookback_days: int
    long_lookback_days: int

    def effective_lookback(self, spec: _ModelSpec) -> int:
        if spec.lookback_days is not None:
            return int(spec.lookback_days)

        if spec.kind == "weekly_phase":
            if spec.period_days is not None and int(spec.period_days) < 56:
                lookback = int(self.short_lookback_days)
                if int(spec.period_days) == 7:
                    lookback = min(lookback, 84)
                elif int(spec.period_days) == 14:
                    lookback = min(lookback, 92)
                elif int(spec.period_days) == 28:
                    lookback = min(lookback, 120)
                return int(lookback)
            return int(self.long_lookback_days)

        if spec.kind in {"bimonth", "ly_bucket", "ly_dom", "paycycle"}:
            return int(self.long_lookback_days)

        if spec.kind == "rolling3":
            return int(self.short_lookback_days)

        if spec.kind == "dom":
            return int(self.long_lookback_days)

        # fallback
        return int(self.short_lookback_days)

    def hist_window(self, df_raw: pd.DataFrame, end_date: pd.Timestamp, lookback_days: int) -> pd.DataFrame:
        if df_raw is None or df_raw.empty:
            return pd.DataFrame(columns=["fecha", "valor", "id"])
        end_date = pd.to_datetime(end_date, errors="coerce").normalize()
        start = end_date - pd.Timedelta(days=int(lookback_days))
        return df_raw[(df_raw["fecha"] >= start) & (df_raw["fecha"] < end_date)].copy()

    def hist_for_spec(self, df_raw: pd.DataFrame, spec: _ModelSpec, d: pd.Timestamp) -> pd.DataFrame:
        lb = self.effective_lookback(spec)
        return self.hist_window(df_raw, d, lb)

    def trace_for_spec(self, spec: _ModelSpec, hist: pd.DataFrame) -> Dict[str, Any]:
        lb = self.effective_lookback(spec)
        hstart = hend = None
        hn = 0
        if hist is not None and not hist.empty and "fecha" in hist.columns:
            hn = int(len(hist))
            a = pd.to_datetime(hist["fecha"].min(), errors="coerce")
            b = pd.to_datetime(hist["fecha"].max(), errors="coerce")
            hstart = str(a.date()) if pd.notna(a) else None
            hend = str(b.date()) if pd.notna(b) else None
        return {
            "kind": spec.kind,
            "period_days": int(spec.period_days) if spec.period_days is not None else None,
            "lookback_days_effective": int(lb),
            "hist_n": hn,
            "hist_start": hstart,
            "hist_end": hend,
        }


# Atractores

class WeeklyPhaseAttractor:
    def __init__(self, week_index_fn: Callable[[pd.Timestamp], int], agg_fn: Callable[[pd.Series], float]):
        self.week_index_fn = week_index_fn
        self.agg_fn = agg_fn

    def predict(self, *, period_days: int, d: pd.Timestamp, hist: pd.DataFrame) -> float:
        d = pd.to_datetime(d, errors="coerce").normalize()
        if hist is None or hist.empty:
            return 0.0

        wd = int(d.weekday())
        T = max(7, int(period_days))

        if T == 7:
            s = hist.loc[hist["fecha"].dt.weekday == wd, "valor"]
            if s.dropna().empty:
                s = hist["valor"]
            return float(self.agg_fn(s))

        mod = max(2, T // 7)
        phase = int(self.week_index_fn(d)) % mod

        hw = hist.copy()
        hw["wd"] = hw["fecha"].dt.weekday.astype(int)
        hw["widx"] = hw["fecha"].apply(lambda x: int(self.week_index_fn(pd.to_datetime(x).normalize())))
        hw["phase"] = (hw["widx"] % mod).astype(int)

        s = hw.loc[(hw["wd"] == wd) & (hw["phase"] == phase), "valor"]
        if s.dropna().empty:
            s = hw.loc[hw["wd"] == wd, "valor"]
        if s.dropna().empty:
            s = hist["valor"]
        return float(self.agg_fn(s))


class DomAttractor:
    def __init__(self, agg_fn: Callable[[pd.Series], float]):
        self.agg_fn = agg_fn

    def predict(self, d: pd.Timestamp, hist: pd.DataFrame) -> float:
        if hist is None or hist.empty:
            return 0.0
        dom = int(pd.to_datetime(d).day)
        s = hist.loc[hist["fecha"].dt.day == dom, "valor"]
        if s.dropna().empty:
            s = hist["valor"]
        return float(self.agg_fn(s))


class BimonthAttractor:
    def __init__(self, agg_fn: Callable[[pd.Series], float]):
        self.agg_fn = agg_fn

    def predict(self, d: pd.Timestamp, hist: pd.DataFrame) -> float:
        if hist is None or hist.empty:
            return 0.0
        d = pd.to_datetime(d).normalize()
        mm = int(d.month)
        dom = int(d.day)
        prev2 = mm - 2
        if prev2 <= 0:
            prev2 += 12
        months = [mm, prev2]
        s = hist.loc[(hist["fecha"].dt.month.isin(months)) & (hist["fecha"].dt.day == dom), "valor"]
        if s.dropna().empty:
            s = hist.loc[hist["fecha"].dt.month.isin(months), "valor"]
        if s.dropna().empty:
            s = hist["valor"]
        return float(self.agg_fn(s))


class Rolling3Attractor:
    def __init__(self, agg_fn: Callable[[pd.Series], float]):
        self.agg_fn = agg_fn

    def predict(self, d: pd.Timestamp, hist: pd.DataFrame) -> float:
        if hist is None or hist.empty:
            return 0.0
        d = pd.to_datetime(d).normalize()
        wd = int(d.weekday())
        s = hist.loc[hist["fecha"].dt.weekday == wd, "valor"]
        if s.dropna().empty:
            s = hist["valor"]
        return float(self.agg_fn(s))


class PayCycleAttractor:
    def __init__(self, agg_fn: Callable[[pd.Series], float]):
        self.agg_fn = agg_fn

    def predict(self, d: pd.Timestamp, hist: pd.DataFrame) -> float:
        if hist is None or hist.empty:
            return 0.0
        d = pd.to_datetime(d).normalize()
        half = 1 if int(d.day) <= 15 else 2
        hd = hist.copy()
        hd["half"] = np.where(hd["fecha"].dt.day <= 15, 1, 2).astype(int)
        s = hd.loc[hd["half"] == half, "valor"]
        if s.dropna().empty:
            s = hist["valor"]
        return float(self.agg_fn(s))


class LyBucketAttractor:
    def __init__(self, week_of_month_fn: Callable[[pd.Timestamp], int], agg_fn: Callable[[pd.Series], float]):
        self.week_of_month_fn = week_of_month_fn
        self.agg_fn = agg_fn

    def predict(self, d: pd.Timestamp, hist: pd.DataFrame) -> float:
        if hist is None or hist.empty:
            return 0.0
        d = pd.to_datetime(d).normalize()
        target_year = int(d.year) - 1
        wd = int(d.weekday())
        wom = int(self.week_of_month_fn(d))

        hy = hist.copy()
        hy["weekday"] = hy["fecha"].dt.weekday.astype(int)
        hy["week_of_month"] = hy["fecha"].apply(lambda x: int(self.week_of_month_fn(pd.to_datetime(x).normalize()))).astype(int)

        s = hy.loc[(hy["fecha"].dt.year == target_year) & (hy["weekday"] == wd) & (hy["week_of_month"] == wom), "valor"]
        if s.dropna().empty:
            s = hy.loc[hy["fecha"].dt.year == target_year, "valor"]
        if s.dropna().empty:
            s = hist["valor"]
        return float(self.agg_fn(s))


class LyDomAttractor:
    def __init__(self, agg_fn: Callable[[pd.Series], float]):
        self.agg_fn = agg_fn

    def predict(self, d: pd.Timestamp, hist: pd.DataFrame) -> float:
        if hist is None or hist.empty:
            return 0.0
        d = pd.to_datetime(d).normalize()
        target_year = int(d.year) - 1
        mm = int(d.month)
        dom = int(d.day)

        s = hist.loc[
            (hist["fecha"].dt.year == target_year) &
            (hist["fecha"].dt.month == mm) &
            (hist["fecha"].dt.day == dom),
            "valor"
        ]
        if s.dropna().empty:
            s = hist.loc[(hist["fecha"].dt.year == target_year) & (hist["fecha"].dt.day == dom), "valor"]
        if s.dropna().empty:
            s = hist.loc[hist["fecha"].dt.year == target_year, "valor"]
        if s.dropna().empty:
            s = hist["valor"]
        return float(self.agg_fn(s))


class AttractorRegistry:
    """Dispatch (ML se mantiene en S4DRModel depende de self._predict_ml_value)."""
    def __init__(
        self,
        weekly_phase: WeeklyPhaseAttractor,
        dom: DomAttractor,
        bimonth: BimonthAttractor,
        rolling3: Rolling3Attractor,
        paycycle: PayCycleAttractor,
        ly_bucket: LyBucketAttractor,
        ly_dom: LyDomAttractor,
    ):
        self.weekly_phase = weekly_phase
        self.dom = dom
        self.bimonth = bimonth
        self.rolling3 = rolling3
        self.paycycle = paycycle
        self.ly_bucket = ly_bucket
        self.ly_dom = ly_dom

    def predict_non_ml(self, spec: _ModelSpec, d: pd.Timestamp, hist: pd.DataFrame) -> float:
        if hist is None or hist.empty:
            return 0.0

        if spec.kind == "weekly_phase":
            T = int(spec.period_days) if spec.period_days is not None else 7
            return float(self.weekly_phase.predict(period_days=T, d=d, hist=hist))

        if spec.kind == "dom":
            return float(self.dom.predict(d, hist))

        if spec.kind == "bimonth":
            return float(self.bimonth.predict(d, hist))

        if spec.kind == "rolling3":
            return float(self.rolling3.predict(d, hist))

        if spec.kind == "paycycle":
            return float(self.paycycle.predict(d, hist))

        if spec.kind == "ly_bucket":
            return float(self.ly_bucket.predict(d, hist))

        if spec.kind == "ly_dom":
            return float(self.ly_dom.predict(d, hist))

        # fallback final
        return 0.0


class PayPatternDetector:
    """Desacople de _detect_pay_pattern + reglas auxiliares."""
    def __init__(self, eps: float):
        self.eps = float(eps)

    def detect(self, hist: pd.DataFrame, *, min_uplift: float = 1.50, min_count: int = 4, top_k: int = 3) -> Dict[str, Any]:
        out = {"pay_mode": "NONE", "pay_dom_set": [], "pay_wd_set": []}
        if hist is None or hist.empty:
            return out

        h = hist.copy()
        h["dom"] = h["fecha"].dt.day.astype(int)
        h["wd"] = h["fecha"].dt.weekday.astype(int)
        h["valor"] = pd.to_numeric(h["valor"], errors="coerce")
        h = h.dropna(subset=["valor"])
        if h.empty:
            return out

        overall = float(h["valor"].mean())
        if not np.isfinite(overall) or overall <= 0:
            overall = float(np.nanmedian(np.abs(h["valor"].to_numpy(dtype=float)))) or 1.0

        dom_stats = h.groupby("dom")["valor"].agg(["mean", "count"]).reset_index()
        dom_stats["uplift"] = dom_stats["mean"] / max(overall, self.eps)
        dom_cand = dom_stats[(dom_stats["count"] >= int(min_count)) & (dom_stats["uplift"] >= float(min_uplift))]
        dom_cand = dom_cand.sort_values("uplift", ascending=False).head(int(top_k))
        if not dom_cand.empty:
            out["pay_mode"] = "DOM"
            out["pay_dom_set"] = sorted(int(x) for x in dom_cand["dom"].tolist())
            return out

        wd_stats = h.groupby("wd")["valor"].agg(["mean", "count"]).reset_index()
        wd_stats["uplift"] = wd_stats["mean"] / max(overall, self.eps)
        wd_cand = wd_stats[(wd_stats["count"] >= int(min_count)) & (wd_stats["uplift"] >= float(min_uplift))]
        wd_cand = wd_cand.sort_values("uplift", ascending=False).head(int(top_k))
        if not wd_cand.empty:
            out["pay_mode"] = "WD"
            out["pay_wd_set"] = sorted(int(x) for x in wd_cand["wd"].tolist())
            return out

        return out

    @staticmethod
    def expand_pay_dom_set(pay_dom_set: List[int]) -> List[int]:
        base = set(int(x) for x in (pay_dom_set or []))
        if 15 in base:
            base.update([14, 16])
        if 30 in base or 31 in base:
            base.update([29, 30, 31])
        return sorted(base)

    @staticmethod
    def assign_pay_bucket(
        d: pd.Timestamp,
        pay_mode: str,
        pay_dom_set: List[int],
        pay_wd_set: List[int],
        tol_days: int = 1,
    ) -> Optional[str]:
        d = pd.to_datetime(d, errors="coerce").normalize()

        if pay_mode in {"PAYDAY", "FIXED"} and pay_dom_set:
            dom = int(d.day)
            if dom in set(int(x) for x in pay_dom_set):
                return f"DOM{dom:02d}"
            return None

        if pay_mode == "DOM" and pay_dom_set:
            dom = int(d.day)
            best = None
            for cand in pay_dom_set:
                if abs(dom - int(cand)) <= int(tol_days):
                    if best is None or abs(dom - int(cand)) < abs(dom - int(best)):
                        best = int(cand)
            if best is not None:
                return f"DOM{best:02d}"
            return None

        if pay_mode == "WD" and pay_wd_set:
            wd = int(d.weekday())
            if wd in set(int(x) for x in pay_wd_set):
                return f"PAY_WD{wd}"
            return None

        return None


# Wrapper del selector (misma firma, desacoplado)

@dataclass
class SelectorResult:
    top1: Optional[str]
    top2: Optional[str]
    w1: float
    w2: float
    pred_final: float
    bucket_type: str
    bucket_id: Any
    bucket_hist_n: float


class OnlineBucketSelector:
    """ Encapsula selector_reliable_v3: - select() NO usa y_true - update() (anti-leak) """
    def __init__(
        self,
        *,
        candidates: List[str],
        priors: Dict[str, float],
        bucket_levels: List[str],
        cfg_overconfirm: Optional[Dict[str, Any]] = None,
    ):
        self.candidates = list(candidates)
        self.priors = dict(priors or {})
        self.state = init_state(candidates=self.candidates, priors=self.priors)

        cfg = dict(cfg_overconfirm or {})
        cfg["state"] = self.state
        cfg["bucket_levels"] = list(bucket_levels)
        self.cfg = cfg

    def select(self, *, day_ctx: Dict[str, Any], preds: Dict[str, float]) -> SelectorResult:
        sel = select_candidates_for_day(day_ctx=day_ctx, preds=preds, cfg=self.cfg)

        pred_final = float(sel.pred_final) if np.isfinite(sel.pred_final) else np.nan
        top1, top2 = sel.top1, sel.top2

        if not np.isfinite(pred_final):
            p1 = preds.get(top1, np.nan) if top1 is not None else np.nan
            p2 = preds.get(top2, np.nan) if top2 is not None else np.nan
            if np.isfinite(p1):
                pred_final = float(p1)
            elif np.isfinite(p2):
                pred_final = float(p2)
            else:
                pred_final = 0.0

        return SelectorResult(
            top1=top1,
            top2=top2,
            w1=float(getattr(sel, "w1", 1.0)),
            w2=float(getattr(sel, "w2", 0.0)),
            pred_final=float(pred_final),
            bucket_type=str(getattr(sel, "bucket_type", None) or "GLOBAL_FALLBACK"),
            bucket_id=getattr(sel, "bucket_id", None) if getattr(sel, "bucket_id", None) is not None else ("GLOBAL",),
            bucket_hist_n=float(getattr(sel, "bucket_hist_n", 0.0) or 0.0),
        )

    def update(self, *, day_ctx: Dict[str, Any], preds: Dict[str, float], y_true: float) -> None:
        if np.isfinite(y_true):
            update_weights(
                state=self.cfg["state"],
                day_ctx=day_ctx,
                preds=preds,
                y_true=float(y_true),
                cfg=self.cfg,
            )


class S4DRModel:
    """
    S4DR base model — 12 structural attractors:
      T7, T7_30, T14, T28, T56, T84 (weekly_phase),
      DOM, BIMONTH_M_M1, ROLLING3M, PAY_CYCLE,
      LY_SAME_BUCKET, LY_DOM.
    """

    def __init__(
        self,
        id_unico: str,
        modo: str = "dinamico",
        json_file: Optional[str] = None,
        *,
        columna_fecha: str = "fecha",
        columna_valor: str = "valor",
        columna_id: str = "id",
        short_lookback_days: int = 92,
        long_lookback_days: int = 1825,
        eval_weeks: int = 18,
        min_eval_points_bucket: int = 1,
        agg: str = "median",
        eps_wape: float = 1e-9,
        # selector tuning
        weight_alpha: float = 1.6,
        delta_wape_rel: float = 0.12,
        weight_tau_abs: float = 0.01,
        weight_tau_rel: float = 0.02,
        min_points_wom: int = 8,
        min_points_wd: int = 6,
        min_points_pay: int = 8,
        pay_min_hist: int = 6,
        pay_min_gain_abs: float = 0.02,
        pay_min_gain_rel: float = 0.05,
        debug: bool = False,
        enabled_specs: Optional[List[str]] = None,
        disabled_specs: Optional[List[str]] = None,
        allowed_specs: Optional[Set[str]] = None,
        paydays_dom_set: Optional[List[int]] = None,
        model_store_dir: Optional[str] = None,
    ) -> None:
        self.id = str(id_unico)
        self.modo = str(modo)
        self.json_file = json_file or f"./S4DR_{self.id}.json"

        self.columna_fecha = columna_fecha
        self.columna_valor = columna_valor
        self.columna_id = columna_id

        self.short_lookback_days = int(short_lookback_days)
        self.long_lookback_days = int(long_lookback_days)

        self.eval_weeks = int(eval_weeks)
        self.min_eval_points_bucket = int(min_eval_points_bucket)
        self.agg = str(agg).lower().strip()
        self.eps_wape = float(eps_wape)
        self.debug = bool(debug)

        self.weight_alpha = float(weight_alpha)
        self.delta_wape_rel = float(delta_wape_rel)
        self.weight_tau_abs = float(weight_tau_abs)
        self.weight_tau_rel = float(weight_tau_rel)

        self.min_points_wom = int(min_points_wom)
        self.min_points_wd = int(min_points_wd)
        self.min_points_pay = int(min_points_pay)

        self.pay_min_hist = int(pay_min_hist)
        self.pay_min_gain_abs = float(pay_min_gain_abs)
        self.pay_min_gain_rel = float(pay_min_gain_rel)

        base_paydays = set(int(x) for x in (paydays_dom_set or [1, 2, 14, 15, 16, 29, 30, 31]))
        base_paydays.update({16})
        self.paydays_dom_set = sorted(base_paydays)
        self._paydays_dom_set = set(self.paydays_dom_set)

        self.model_store_dir = str(model_store_dir or os.path.dirname(self.json_file) or "static")

        allowed_set = set(str(x) for x in allowed_specs) if allowed_specs else None
        enabled_set = set(str(x) for x in enabled_specs) if enabled_specs else None
        disabled_set = set(str(x) for x in disabled_specs) if disabled_specs else set()

        self.df_raw = pd.DataFrame(columns=["fecha", "valor", "id"])
        self._series = None
        self._week_anchor = None

        # -------------------------------
        # Modelos/períodos (desacoplado)
        # -------------------------------
        specs = PeriodSpecFactory.default_specs()
        self.models = PeriodSpecFactory.apply_filters(
            specs,
            allowed_specs=allowed_set,
            enabled_specs=enabled_set,
            disabled_specs=disabled_set,
        )
        self.specs_included = [spec.code for spec in self.models]

        # -------------------------------
        # Política / registry / pay-detector
        # -------------------------------
        self._hist_policy = HistoryWindowPolicy(
            short_lookback_days=int(self.short_lookback_days),
            long_lookback_days=int(self.long_lookback_days),
        )

        self._attractors = AttractorRegistry(
            weekly_phase=WeeklyPhaseAttractor(self._week_index, self._agg_value),
            dom=DomAttractor(self._agg_value),
            bimonth=BimonthAttractor(self._agg_value),
            rolling3=Rolling3Attractor(self._agg_value),
            paycycle=PayCycleAttractor(self._agg_value),
            ly_bucket=LyBucketAttractor(self._week_of_month, self._agg_value),
            ly_dom=LyDomAttractor(self._agg_value),
        )

        self._pay_detector = PayPatternDetector(eps=float(self.eps_wape))

        self.detalles_por_dia = {}
        self._load_state()

    # -----------------------------
    # Persistencia mínima
    # -----------------------------
    def _load_state(self) -> None:
        path = self.json_file
        if not path or not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
            self.detalles_por_dia = data.get("detalles_por_dia", {}) or {}
        except Exception:
            self.detalles_por_dia = {}

    def _save_state(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.json_file) or ".", exist_ok=True)
        except Exception:
            pass
        data = {"detalles_por_dia": self.detalles_por_dia}
        try:
            with open(self.json_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        except Exception:
            pass

    def prune_specs_from_analysis(self, pruned_list: List[str]) -> None:
        if not pruned_list:
            return
        drop_set = set(str(x) for x in pruned_list)
        self.models = [spec for spec in self.models if spec.code not in drop_set]
        self.specs_included = [spec.code for spec in self.models]


    @staticmethod
    def _to_ts(x) -> pd.Timestamp:
        return pd.to_datetime(x, errors="coerce").normalize()

    def _week_index(self, d: pd.Timestamp) -> int:
        """Índice semanal estable usando ancla fija (min fecha histórica)."""
        d = self._to_ts(d)
        if self._week_anchor is None:
            return int(d.to_period("W").start_time.toordinal())
        delta_days = int((d.normalize() - self._week_anchor).days)
        return delta_days // 7

    @staticmethod
    def _week_of_month(d: pd.Timestamp) -> int:
        """Semana del mes 1..5 (simple por día del mes)."""
        d = pd.to_datetime(d, errors="coerce").normalize()
        dom = int(d.day)
        wom = (dom - 1) // 7 + 1
        return int(min(5, max(1, wom)))

    def _is_payday(self, d: pd.Timestamp) -> bool:
        d = self._to_ts(d)
        return int(d.day) in self._paydays_dom_set

    def _hist_window(self, end_date: pd.Timestamp, *, lookback_days: int) -> pd.DataFrame:
        """Histórico anterior a end_date con ventana lookback_days."""
        return self._hist_policy.hist_window(self.df_raw, self._to_ts(end_date), int(lookback_days))

    def _hist_for_spec(self, spec: _ModelSpec, d: pd.Timestamp) -> pd.DataFrame:
        """Desacoplado: delega a HistoryWindowPolicy."""
        return self._hist_policy.hist_for_spec(self.df_raw, spec, self._to_ts(d))

    def _agg_value(self, s: pd.Series) -> float:
        """Regla: si vacío, usar 0.0 (no NaN)."""
        s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            return 0.0
        return float(s.mean()) if self.agg == "mean" else float(s.median())

    @staticmethod
    def _wape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if not np.any(mask):
            return float("inf")
        yt = y_true[mask]
        yp = y_pred[mask]
        denom = float(np.sum(np.abs(yt))) + float(eps)
        return float(np.sum(np.abs(yt - yp)) / denom)


    # Peso top2 desde WAPE
    def _weight_from_wapes(
        self,
        w1: float,
        w2: float,
        *,
        weight_alpha: Optional[float] = None,
        delta_wape_abs: float = 0.20,
        delta_wape_rel: float = 0.0,
        weight_tau_abs: Optional[float] = None,
        weight_tau_rel: Optional[float] = None,
    ) -> float:
        if not np.isfinite(w1):
            return 1.0
        if not np.isfinite(w2):
            return 1.0

        alpha = float(weight_alpha) if weight_alpha is not None else 1.0
        if alpha <= 0:
            alpha = 1.0

        inv1 = (1.0 / (float(w1) + self.eps_wape)) ** alpha
        inv2 = (1.0 / (float(w2) + self.eps_wape)) ** alpha
        denom = inv1 + inv2
        weight1 = float(inv1 / denom) if denom > 0 else 1.0

        diff = float(w2) - float(w1)
        rel = diff / max(float(w1), self.eps_wape)

        if diff >= float(delta_wape_abs):
            return 1.0
        if float(delta_wape_rel) > 0 and rel >= float(delta_wape_rel):
            return 1.0

        tau_abs = self.weight_tau_abs if weight_tau_abs is None else float(weight_tau_abs)
        tau_rel = self.weight_tau_rel if weight_tau_rel is None else float(weight_tau_rel)
        if tau_abs > 0 and abs(diff) <= float(tau_abs):
            return 0.5
        if tau_rel > 0 and abs(diff) <= float(tau_rel) * max(float(w1), self.eps_wape):
            return 0.5

        return float(weight1)

    # Entrenamiento/actualización
    def actualizar_modelo(self, df_nuevos: pd.DataFrame) -> None:
        """  Carga histórico y fija ancla semanal. Espera columnas configuradas por columna_fecha/valor/id."""
        if df_nuevos is None or df_nuevos.empty:
            self.df_raw = pd.DataFrame(columns=["fecha", "valor", "id"])
            self._week_anchor = None
            self._series = None
            return

        df = df_nuevos.copy()

        if self.columna_fecha != "fecha":
            df = df.rename(columns={self.columna_fecha: "fecha"})
        if self.columna_valor != "valor":
            df = df.rename(columns={self.columna_valor: "valor"})
        if self.columna_id != "id":
            df = df.rename(columns={self.columna_id: "id"})

        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce").dt.normalize()
        df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
        df["id"] = df["id"].astype(str)

        df = df.dropna(subset=["fecha", "valor", "id"])
        df = df[df["id"] == self.id].copy()
        df = df.sort_values("fecha").drop_duplicates(subset=["fecha"], keep="last").reset_index(drop=True)

        self.df_raw = df[["fecha", "valor", "id"]].copy()
        if not self.df_raw.empty:
            self._week_anchor = self.df_raw["fecha"].min().normalize()
            self._series = pd.Series(self.df_raw["valor"].to_numpy(dtype=float), index=self.df_raw["fecha"])
        else:
            self._week_anchor = None
            self._series = None

    # Atractor por periodo
    def _predict_model(self, spec: _ModelSpec, d: pd.Timestamp, hist: pd.DataFrame) -> float:
        d = self._to_ts(d)
        return float(self._attractors.predict_non_ml(spec, d, hist))

    # Matriz de evaluación
    def _build_eval_matrix(self, anchor_date: pd.Timestamp) -> pd.DataFrame:
        if self.df_raw.empty:
            return pd.DataFrame()

        anchor_date = self._to_ts(anchor_date)
        eval_days = int(min(30, max(7, self.eval_weeks * 7)))
        end = anchor_date - pd.Timedelta(days=1)
        start = end - pd.Timedelta(days=eval_days - 1)

        df_eval = self.df_raw[(self.df_raw["fecha"] >= start) & (self.df_raw["fecha"] <= end)].copy()
        df_eval = df_eval.sort_values("fecha").reset_index(drop=True)
        if df_eval.empty:
            return pd.DataFrame()

        rows = []
        for t, y in zip(df_eval["fecha"].tolist(), df_eval["valor"].tolist()):
            t = self._to_ts(t)
            row = {
                "fecha": t,
                "y_true": float(y) if np.isfinite(float(y)) else 0.0,
                "weekday": int(t.weekday()),
                "week_of_month": int(self._week_of_month(t)),
            }
            for spec in self.models:
                hist_t = self._hist_for_spec(spec, t)
                row[f"yhat_{spec.code}"] = float(self._predict_model(spec, t, hist_t))
            rows.append(row)

        return pd.DataFrame(rows)

    # Detección de patrón de pago
    def _detect_pay_pattern(
        self,
        end_date: pd.Timestamp,
        *,
        lookback_days: Optional[int] = None,
        min_uplift: float = 1.50,
        min_count: int = 4,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        end_date = self._to_ts(end_date)
        lb = int(lookback_days) if lookback_days is not None else int(self.long_lookback_days)
        hist = self._hist_window(end_date, lookback_days=lb)
        if hist is None or hist.empty:
            hist = self.df_raw.copy()
        return self._pay_detector.detect(hist, min_uplift=min_uplift, min_count=min_count, top_k=top_k)

    @staticmethod
    def _expand_pay_dom_set(pay_dom_set: List[int]) -> List[int]:
        return PayPatternDetector.expand_pay_dom_set(pay_dom_set)

    @staticmethod
    def _pay_calibration_key(d: pd.Timestamp, pay_bucket: Optional[str]) -> Optional[str]:
        dom = int(pd.to_datetime(d).normalize().day)
        if dom in {14, 15, 16}:
            return "MID_14_16"
        if dom == 29:
            return "EOM_29"
        if dom in {30, 31}:
            return "EOM_30_31"
        if pay_bucket is None or str(pay_bucket).strip() in {"", "None", "nan"}:
            return None
        return str(pay_bucket)

    @staticmethod
    def _assign_pay_bucket(
        d: pd.Timestamp,
        pay_mode: str,
        pay_dom_set: List[int],
        pay_wd_set: List[int],
        tol_days: int = 1,
    ) -> Optional[str]:
        return PayPatternDetector.assign_pay_bucket(d, pay_mode, pay_dom_set, pay_wd_set, tol_days=tol_days)


    @staticmethod
    def forecast_from_history(
        df_historico: pd.DataFrame,
        *,
        horizon: int = 15,
        base_dir: str = "static/",
        model_store_dir: Optional[str] = None,
        eval_weeks: int = 10,
        save_debug_details: bool = False,
        mutate_state: bool = False,
    ) -> pd.DataFrame:
        """
        - Limpieza: No contiene
        - Tratamiendo de ruido: No contiene
        - Input: histórico de UN solo cajero (Fecha, Cantidad, CajeroId) (o Client-> CajeroId).
        - Se especifica horizon que es el lapso de tiempo que se quiere obtener.
        - Retorna DataFrame con columnas: Fecha, MoPronosDispenAtractor, CajeroId.
        - Calibración: DESACTIVADA
        """

        def _call(obj, method_name: str, **kwargs):
            fn = getattr(obj, method_name)
            sig = inspect.signature(fn)
            filt = {k: v for k, v in kwargs.items() if k in sig.parameters}
            return fn(**filt)


        df = df_historico.copy()

        if "CajeroId" in df.columns and "id" not in df.columns:
            df = df.rename(columns={"CajeroId": "id"})

        if "Fecha" not in df.columns or "Cantidad" not in df.columns or "id" not in df.columns:
            raise ValueError("df_historico debe contener columnas: Fecha, Cantidad, CajeroId.")

        df["fecha"] = pd.to_datetime(df["Fecha"], errors="coerce").dt.normalize()
        df["valor"] = pd.to_numeric(df["Cantidad"], errors="coerce")
        df["id"] = df["id"].astype(str).str.strip()

        df = df.dropna(subset=["fecha", "valor", "id"]).copy()

        if df.empty:
            return pd.DataFrame(columns=["Fecha", "MoPronosDispenAtractor", "CajeroId"])

        os.makedirs(base_dir, exist_ok=True)
        store_dir = str(model_store_dir or base_dir)


        out = []

        df = df.sort_values(["id", "fecha"])

        for cliente, g in df.groupby("id", sort=False):
            g = (
                g.drop_duplicates(subset=["fecha"], keep="last")
                .sort_values("fecha")
                .reset_index(drop=True)
            )

            if g.empty:
                continue

            json_path = os.path.join(base_dir, f"S4DR_{cliente}.json")

            modelo = S4DRModel(
                id_unico=cliente,
                json_file=json_path,
                model_store_dir=store_dir,
            )

            try:
                modelo.eval_weeks = int(eval_weeks)
            except Exception:
                pass

            modelo.actualizar_modelo(df_nuevos=g[["fecha", "valor", "id"]])

            last_hist_date = pd.to_datetime(g["fecha"].max()).normalize()

            preds_raw, fechas_modelo, _usados = _call(
                modelo,
                "seleccionar_atractores",
                P=int(horizon),
                anchor_date=last_hist_date,
                save_debug_details=bool(save_debug_details),
                mutate_state=bool(mutate_state),
                trace_base_dir=str(base_dir),
            )

            preds_arr = np.asarray(preds_raw, dtype=float)

            if fechas_modelo is not None and len(fechas_modelo) == int(horizon):
                fechas_pred = pd.DatetimeIndex(fechas_modelo).normalize()
            else:
                fechas_pred = pd.date_range(
                    start=last_hist_date + timedelta(days=1),
                    periods=int(horizon),
                    freq="D",
                ).normalize()

            out.append(
                pd.DataFrame(
                    {
                        "Fecha": fechas_pred,
                        "MoPronosDispenAtractor": preds_arr,
                        "CajeroId": cliente,
                    }
                )
            )

        if not out:
            return pd.DataFrame(columns=["Fecha", "MoPronosDispenAtractor", "CajeroId"])

        return (
            pd.concat(out, ignore_index=True)
            .sort_values(["CajeroId", "Fecha"])
            .reset_index(drop=True)
        )

    # Guardado de matriz
 
    def _save_matrix_trace(self, df_trace: pd.DataFrame, trace_base_dir: str, last_hist_date: pd.Timestamp) -> str:
        os.makedirs(trace_base_dir or "static", exist_ok=True)
        last_hist_date = self._to_ts(last_hist_date)
        fname = f"{self.id}_{last_hist_date.strftime('%Y%m%d')}_S4DR_matrix.csv"
        path = os.path.join(trace_base_dir, fname)
        df_trace.to_csv(path, index=False)
        return path

    def seleccionar_atractores(
        self,
        P: int = 15,
        ventana_reciente: int = 30,               # compat (no se usa aqui)
        df_historial: Optional[pd.DataFrame] = None,
        *,
        anchor_date: Optional[pd.Timestamp] = None,
        mutate_state: bool = False,               # compat
        save_debug_details: bool = False,
        debug_keep_days: int = 180,
        trace_base_dir: Optional[str] = "static",
    ) -> Tuple[List[float], List[pd.Timestamp], List[Any]]:

        if self.df_raw.empty:
            raise ValueError("Modelo sin historico. Ejecuta actualizar_modelo(df) primero.")

        verbose = bool(self.debug)
        horizonte = int(P)
        start_date = self._to_ts(anchor_date) if anchor_date is not None else self._to_ts(self.df_raw["fecha"].max())

        # 1) Matriz eval
        M_eval = self._build_eval_matrix(anchor_date=start_date)
        if M_eval.empty:
            M_eval = pd.DataFrame(columns=["fecha", "y_true", "weekday", "week_of_month"])

        # 1a) Pay pattern + pay_bucket en eval
        pay_info = self._detect_pay_pattern(start_date)
        pay_mode = str(pay_info.get("pay_mode", "NONE"))
        pay_dom_set = list(pay_info.get("pay_dom_set") or [])
        pay_dom_set = self._expand_pay_dom_set(pay_dom_set)
        pay_wd_set = list(pay_info.get("pay_wd_set") or [])

        if not M_eval.empty:
            M_eval["pay_bucket"] = [
                self._assign_pay_bucket(self._to_ts(d), pay_mode, pay_dom_set, pay_wd_set)
                for d in M_eval["fecha"].tolist()
            ]
            M_eval["payday_flag"] = [
                1 if self._is_payday(self._to_ts(d)) else 0
                for d in M_eval["fecha"].tolist()
            ]

        candidate_specs = [spec.code for spec in self.models]
        if len(self.df_raw) < 56:
            candidate_specs = [s for s in candidate_specs if s != "T56"]
        specs_included = [spec.code for spec in self.models if spec.code in candidate_specs]
        if not specs_included:
            specs_included = [spec.code for spec in self.models]

        CANDIDATES = list(specs_included)

        bucket_levels_sel = [
            "weekday>week_of_month>pay_bucket",
            "weekday>pay_bucket",
            "weekday",
            "GLOBAL_FALLBACK",
        ]

        global_scores: Dict[str, float] = {}
        if not M_eval.empty:
            y_all = M_eval["y_true"].to_numpy(dtype=float)
            for spec_code in specs_included:
                col = f"yhat_{spec_code}"
                if col not in M_eval.columns:
                    global_scores[spec_code] = float("inf")
                    continue
                yp_all = M_eval[col].to_numpy(dtype=float)
                global_scores[spec_code] = self._wape(y_all, yp_all, eps=self.eps_wape)
        priors = dict(global_scores) if global_scores else {c: 1.0 for c in CANDIDATES}

        selector = OnlineBucketSelector(
            candidates=CANDIDATES,
            priors=priors,
            bucket_levels=bucket_levels_sel,
            cfg_overconfirm={
                "K": 2,
                "eta": 2.0,
                "alpha_share": 0.02,
                "loss_clip": 3.0,
                "gamma_parent": 12.0,
                "weight_floor": 1e-6,
                "ml_candidates": set(["LGBM_MODEL", "PROPHET_MODEL"]),
                "max_ml_in_topk": 2,
                "diversity_delta": 0.03,
                "diversity_min_n": 6.0,
                "w_top1": 0.7,
            },
        )

        PAYDAY_DOMS = {14, 15, 16, 29, 30, 31}
        PAY_UPLIFT_MIN = 0.90
        PAY_UPLIFT_MAX = 1.40
        PAY_UPLIFT_MIN_SAMPLES = 3

        pay_uplifts: Dict[str, float] = {}
        pay_uplift_baseline: Dict[str, str] = {}

        if not M_eval.empty:
            ratios_by_bucket: Dict[str, List[float]] = {}
            baseline_by_bucket: Dict[str, List[str]] = {}

            eval_sorted = M_eval.sort_values("fecha")

            for r in eval_sorted.itertuples(index=False):
                d_eval = self._to_ts(getattr(r, "fecha"))
                dom = int(d_eval.day)
                y_true = getattr(r, "y_true", np.nan)

                pay_bucket_eval = self._assign_pay_bucket(d_eval, pay_mode, pay_dom_set, pay_wd_set)

                pay_bucket_sel = "pay" if int(d_eval.day) in self._paydays_dom_set else "nopay"
                day_eval = {
                    "weekday": int(d_eval.weekday()),
                    "week_of_month": int(self._week_of_month(d_eval)),
                    "pay_bucket": pay_bucket_sel,
                }

                preds_eval: Dict[str, float] = {}
                for spec_code in specs_included:
                    v = getattr(r, f"yhat_{spec_code}", np.nan)
                    preds_eval[spec_code] = float(v) if np.isfinite(v) else np.nan

                if verbose:
                    preds_valid = {k: v for k, v in preds_eval.items() if np.isfinite(v)}
                    if not preds_valid:
                        raise ValueError("preds_row vacío: no hay candidatos para selector.")
                    if "weekday" not in day_eval or "pay_bucket" not in day_eval:
                        raise ValueError("day_features incompleto: requiere weekday y pay_bucket.")

                sel_res = selector.select(day_ctx=day_eval, preds=preds_eval)
                pred_final_raw = float(sel_res.pred_final)

                if np.isfinite(y_true):
                    selector.update(day_ctx=day_eval, preds=preds_eval, y_true=float(y_true))

                if dom not in PAYDAY_DOMS:
                    continue
                if not np.isfinite(y_true):
                    continue
                if pay_bucket_eval is None:
                    continue

                pred_base = pred_final_raw
                baseline_used = "pred_final"
                if not np.isfinite(pred_base):
                    vdom = getattr(r, "yhat_DOM", np.nan) if hasattr(r, "yhat_DOM") else np.nan
                    if np.isfinite(vdom):
                        pred_base = float(vdom)
                        baseline_used = "DOM"
                if not np.isfinite(pred_base) or float(pred_base) <= 0:
                    continue

                ratio = float(y_true) / (float(pred_base) + self.eps_wape)
                if not np.isfinite(ratio):
                    continue
                pb_key = self._pay_calibration_key(d_eval, pay_bucket_eval)
                if pb_key is None:
                    continue

                ratios_by_bucket.setdefault(pb_key, []).append(ratio)
                baseline_by_bucket.setdefault(pb_key, []).append(baseline_used)

            for pb, ratios in ratios_by_bucket.items():
                if len(ratios) < int(PAY_UPLIFT_MIN_SAMPLES):
                    continue
                if pb == "EOM_30_31":
                    uplift = float(np.quantile(np.asarray(ratios, dtype=float), 0.60))
                    uplift = min(max(uplift, 1.00), 1.35)
                elif pb == "EOM_29":
                    uplift = float(np.median(ratios))
                    uplift = min(max(uplift, 0.90), 1.20)
                else:
                    uplift = float(np.median(ratios))
                    uplift = min(max(uplift, float(PAY_UPLIFT_MIN)), float(PAY_UPLIFT_MAX))

                pay_uplifts[str(pb)] = uplift
                if pb in baseline_by_bucket and baseline_by_bucket[pb]:
                    pay_uplift_baseline[pb] = Counter(baseline_by_bucket[pb]).most_common(1)[0][0]

        preds: List[float] = []
        fechas: List[pd.Timestamp] = []
        usados: List[Any] = []
        selector_audit_rows: List[Dict[str, Any]] = []
        top1_counter = Counter()
        bucket_type_counter = Counter()
        fallback_counter = Counter()

        for step in range(1, int(horizonte) + 1):
            d = self._to_ts(start_date + pd.Timedelta(days=step))
            wd = int(d.weekday())
            wom = int(self._week_of_month(d))
            payday_flag = 1 if self._is_payday(d) else 0
            pay_bucket = self._assign_pay_bucket(d, pay_mode, pay_dom_set, pay_wd_set)

            yhat_cols: Dict[str, float] = {}
            spec_trace: Dict[str, Any] = {}

            for spec in self.models:
                hist_d = self._hist_for_spec(spec, d)
                yhat_cols[f"yhat_{spec.code}"] = float(self._predict_model(spec, d, hist_d))
                spec_trace[spec.code] = self._hist_policy.trace_for_spec(spec, hist_d)

            pay_bucket_sel = "pay" if int(d.day) in self._paydays_dom_set else "nopay"
            day_features = {
                "weekday": wd,
                "week_of_month": wom,
                "pay_bucket": pay_bucket_sel,
            }
            preds_selector = {spec_code: float(yhat_cols.get(f"yhat_{spec_code}", np.nan)) for spec_code in specs_included}

            if verbose:
                preds_valid = {k: v for k, v in preds_selector.items() if np.isfinite(v)}
                if not preds_valid:
                    raise ValueError("preds_row vacío: no hay candidatos para selector.")
                if "weekday" not in day_features or "pay_bucket" not in day_features:
                    raise ValueError("day_features incompleto: requiere weekday y pay_bucket.")

            sel_res = selector.select(day_ctx=day_features, preds=preds_selector)

            pred_final = float(sel_res.pred_final)
            top1 = sel_res.top1
            top2 = sel_res.top2

            weights_used = {top1: float(sel_res.w1)} if top1 else {}
            if top2:
                weights_used[top2] = float(sel_res.w2)

            losses_used = {top1: float(priors.get(top1, float("inf")))} if top1 else {}
            if top2:
                losses_used[top2] = float(priors.get(top2, float("inf")))

            candidates_used = [c for c, _ in sorted(weights_used.items(), key=lambda kv: kv[1], reverse=True)]
            pred1 = preds_selector.get(top1, np.nan) if top1 is not None else np.nan
            pred2 = preds_selector.get(top2, np.nan) if top2 is not None else np.nan
            wape1 = float(losses_used.get(top1, float("inf"))) if top1 is not None else float("inf")
            wape2 = float(losses_used.get(top2, float("inf"))) if top2 is not None else float("inf")

            bucket_type = str(sel_res.bucket_type or "GLOBAL_FALLBACK")
            bucket_id = sel_res.bucket_id if sel_res.bucket_id is not None else ("GLOBAL",)
            bucket_hist_n = int(round(float(sel_res.bucket_hist_n or 0.0)))

            top1_counter[str(top1)] += 1
            bucket_type_counter[str(bucket_type)] += 1
            if str(bucket_type).startswith("GLOBAL"):
                fallback_counter["GLOBAL"] += 1
            else:
                fallback_counter["BUCKET"] += 1

            # Ajuste payday (post)
            pay_cal_applied = 0
            uplift_value = 1.0
            uplift_baseline = "pred_final"
            cal_key = self._pay_calibration_key(d, pay_bucket)
            if payday_flag == 1 and int(d.day) in PAYDAY_DOMS and cal_key is not None:
                if str(cal_key) in pay_uplifts and np.isfinite(pred_final):
                    uplift_value = float(pay_uplifts.get(str(cal_key), 1.0))
                    uplift_baseline = str(pay_uplift_baseline.get(str(cal_key), "pred_final"))
                    pred_final = float(pred_final) * float(uplift_value)
                    pay_cal_applied = 1

            inv_w1 = 1.0 / (float(wape1) + self.eps_wape) if np.isfinite(float(wape1)) else 0.0
            inv_w2 = 1.0 / (float(wape2) + self.eps_wape) if np.isfinite(float(wape2)) else 0.0

            info = {
                "date": str(d.date()),
                "weekday": wd,
                "week_of_month": wom,
                "top1": top1,
                "top2": top2,
                "wape_top1": wape1,
                "wape_top2": wape2 if top2 is not None else None,
                "inv_wape_top1": float(inv_w1),
                "inv_wape_top2": float(inv_w2) if top2 is not None else None,
                "weight_top1": float(sel_res.w1),
                "pred_top1": float(pred1) if np.isfinite(pred1) else 0.0,
                "pred_top2": float(pred2) if top2 is not None and np.isfinite(pred2) else None,
                "pred_final": float(pred_final),

                "bucket_type": bucket_type,
                "bucket_id": bucket_id,
                "bucket_hist_n": int(bucket_hist_n),
                "pay_bucket": pay_bucket,
                "payday_flag": int(payday_flag),
                "pay_mode": pay_mode,
                "pay_dom_set": list(pay_dom_set),
                "pay_wd_set": list(pay_wd_set),
                "pay_calibration_applied": int(pay_cal_applied),
                "pay_calibration_uplift": float(uplift_value),
                "pay_calibration_baseline": str(uplift_baseline),

                "selector_weights_json": json.dumps(weights_used, ensure_ascii=True, sort_keys=True),
                "selector_losses_shrunk_json": json.dumps(losses_used, ensure_ascii=True, sort_keys=True),
                "selector_n_per_candidate_json": json.dumps({}, ensure_ascii=True, sort_keys=True),

                "spec_trace": spec_trace,
            }
            info.update({k: float(v) for k, v in yhat_cols.items()})

            preds.append(float(pred_final))
            fechas.append(d)
            usados.append(info)

            selector_audit_rows.append(
                {
                    "fecha": str(d.date()),
                    "split": "future",
                    "bucket_level_used": bucket_type,
                    "bucket_key": str(bucket_id),
                    "n_bucket": float(sel_res.bucket_hist_n or 0.0),
                    "candidates_used": "|".join(candidates_used),
                    "weights_json": json.dumps(weights_used, ensure_ascii=True, sort_keys=True),
                    "losses_shrunk_json": json.dumps(losses_used, ensure_ascii=True, sort_keys=True),
                    "n_per_candidate_json": json.dumps({}, ensure_ascii=True, sort_keys=True),
                    "yhat_final": float(pred_final),
                }
            )

            if save_debug_details:
                self.detalles_por_dia[str(d.date())] = info

        if save_debug_details:
            try:
                if len(self.detalles_por_dia) > int(debug_keep_days):
                    keys = []
                    for k in self.detalles_por_dia.keys():
                        try:
                            keys.append((pd.to_datetime(k), k))
                        except Exception:
                            continue
                    keys.sort(key=lambda x: x[0])
                    keep = set([k for _, k in keys[-int(debug_keep_days):]])
                    for k in list(self.detalles_por_dia.keys()):
                        if k not in keep:
                            self.detalles_por_dia.pop(k, None)
            except Exception:
                pass
            self._save_state()

        # Guardar matriz trazable (eval + future)
        #if trace_base_dir:
        #    try:
        #        future_rows = []
        #        for d_info in usados:
        #            d2 = pd.to_datetime(d_info["date"]).normalize()
        #            trace_obj = {
        #                "bucket_type": d_info.get("bucket_type"),
        #                "bucket_id": d_info.get("bucket_id"),
        #                "bucket_hist_n": d_info.get("bucket_hist_n"),
        #                "bucket_level_used": d_info.get("bucket_type"),
        #                "pay_bucket": d_info.get("pay_bucket"),
        #                "payday_flag": d_info.get("payday_flag"),
        #                "pay_mode": d_info.get("pay_mode"),
        #                "pay_dom_set": d_info.get("pay_dom_set"),
        #                "pay_wd_set": d_info.get("pay_wd_set"),
        #                "candidates_used": d_info.get("top1"),
        #                "weights_json": d_info.get("selector_weights_json"),
        #                "losses_shrunk_json": d_info.get("selector_losses_shrunk_json"),
        #                "n_per_candidate_json": d_info.get("selector_n_per_candidate_json"),
        #                "pay_calibration_applied": d_info.get("pay_calibration_applied"),
        #                "pay_calibration_uplift": d_info.get("pay_calibration_uplift"),
        #                "pay_calibration_baseline": d_info.get("pay_calibration_baseline"),
        #            }
        #            row = {
        #                "fecha": d2,
        #                "split": "future",
        #                "y_true": np.nan,
        #                "weekday": int(d_info["weekday"]),
        #                "week_of_month": int(d_info["week_of_month"]),
        #                "top1": d_info["top1"],
        #                "top2": d_info["top2"],
        #                "weight1": d_info["weight_top1"],
        #                "pred_final": d_info["pred_final"],
        #                "wape_top1": d_info["wape_top1"],
        #                "wape_top2": d_info["wape_top2"],
        #                "inv_wape_top1": d_info.get("inv_wape_top1"),
        #                "inv_wape_top2": d_info.get("inv_wape_top2"),
        #                "bucket_type": d_info.get("bucket_type"),
        #                "bucket_id": d_info.get("bucket_id"),
        #                "bucket_hist_n": d_info.get("bucket_hist_n"),
        #                "pay_bucket": d_info.get("pay_bucket"),
        #                "payday_flag": d_info.get("payday_flag"),
        #                "pay_mode": d_info.get("pay_mode"),
        #                "pay_dom_set": d_info.get("pay_dom_set"),
        #                "pay_wd_set": d_info.get("pay_wd_set"),
        #                "spec_trace_json": str(trace_obj),
        #            }
        #            for spec in self.models:
        #                row[f"yhat_{spec.code}"] = d_info.get(f"yhat_{spec.code}", 0.0)
        #            future_rows.append(row)
        #        M_future = pd.DataFrame(future_rows)

        #        M = M_eval.copy()
        #        M = M.sort_values("fecha")
        #        if not M.empty:
        #            M["split"] = "eval"

        #        df_trace = pd.concat([M, M_future], ignore_index=True, sort=False)

                # quitar prefijo yhat
        #        df_trace = df_trace.rename(columns={c: c.replace("yhat_", "") for c in df_trace.columns if c.startswith("yhat_")})

        #        last_hist_date = self._to_ts(self.df_raw["fecha"].max())
                # matriz de analisis para auditoria o investigación del modelo
                #self._save_matrix_trace(df_trace, trace_base_dir=str(trace_base_dir), last_hist_date=last_hist_date)

                #if selector_audit_rows:
                #    df_audit = pd.DataFrame(selector_audit_rows)
                #   audit_base = os.path.join(str(trace_base_dir), f"{self.id}_{last_hist_date.strftime('%Y%m%d')}_selector_audit")
                #    df_audit.to_csv(f"{audit_base}.csv", index=False)
                #    try:
                #        df_audit.to_parquet(f"{audit_base}.parquet", index=False)
                #    except Exception:
                #        pass

            #except Exception as e:
            #    if self.debug or save_debug_details:
            #        print(f"[S4DR] WARN: no se pudo guardar matriz en '{trace_base_dir}': {e}")

        if self.debug:
            try:
                logging.info("[WARN] [S4DR] top1 dist:", dict(top1_counter))
                logging.info("[WARN] [S4DR] bucket_type dist:", dict(bucket_type_counter))
                logging.info("[WARN] [S4DR] fallback tiers:", dict(fallback_counter))
            except Exception:
                pass

        return preds, fechas, usados



class S4DRModelV102(S4DRModel):
    """V102 structural variant — same 12 structural candidates as base."""

    def __init__(
        self,
        id_unico: str,
        modo: str = "dinamico",
        json_file: Optional[str] = None,
        *,
        columna_fecha: str = "fecha",
        columna_valor: str = "valor",
        columna_id: str = "id",
        short_lookback_days: int = 92,
        long_lookback_days: int = 1825,
        eval_weeks: int = 18,
        min_eval_points_bucket: int = 1,
        agg: str = "median",
        eps_wape: float = 1e-9,
        weight_alpha: float = 1.6,
        delta_wape_rel: float = 0.12,
        weight_tau_abs: float = 0.01,
        weight_tau_rel: float = 0.02,
        min_points_wom: int = 8,
        min_points_wd: int = 6,
        min_points_pay: int = 8,
        pay_min_hist: int = 6,
        pay_min_gain_abs: float = 0.02,
        pay_min_gain_rel: float = 0.05,
        debug: bool = False,
        enabled_specs: Optional[List[str]] = None,
        disabled_specs: Optional[List[str]] = None,
        allowed_specs: Optional[Set[str]] = None,
        paydays_dom_set: Optional[List[int]] = None,
        model_store_dir: Optional[str] = None,
    ) -> None:
        super().__init__(
            id_unico=id_unico,
            modo=modo,
            json_file=json_file,
            columna_fecha=columna_fecha,
            columna_valor=columna_valor,
            columna_id=columna_id,
            short_lookback_days=short_lookback_days,
            long_lookback_days=long_lookback_days,
            eval_weeks=eval_weeks,
            min_eval_points_bucket=min_eval_points_bucket,
            agg=agg,
            eps_wape=eps_wape,
            weight_alpha=weight_alpha,
            delta_wape_rel=delta_wape_rel,
            weight_tau_abs=weight_tau_abs,
            weight_tau_rel=weight_tau_rel,
            min_points_wom=min_points_wom,
            min_points_wd=min_points_wd,
            min_points_pay=min_points_pay,
            pay_min_hist=pay_min_hist,
            pay_min_gain_abs=pay_min_gain_abs,
            pay_min_gain_rel=pay_min_gain_rel,
            debug=debug,
            enabled_specs=enabled_specs,
            disabled_specs=disabled_specs,
            allowed_specs=allowed_specs,
            paydays_dom_set=paydays_dom_set,
            model_store_dir=model_store_dir,
        )

    def seleccionar_atractores(
        self,
        P: int = 15,
        ventana_reciente: int = 30,
        df_historial: Optional[pd.DataFrame] = None,
        *,
        anchor_date: Optional[pd.Timestamp] = None,
        mutate_state: bool = False,
        save_debug_details: bool = False,
        debug_keep_days: int = 180,
        trace_base_dir: Optional[str] = "static",
    ) -> Tuple[List[float], List[pd.Timestamp], List[Any]]:
        return super().seleccionar_atractores(
            P=P,
            ventana_reciente=ventana_reciente,
            df_historial=df_historial,
            anchor_date=anchor_date,
            mutate_state=mutate_state,
            save_debug_details=save_debug_details,
            debug_keep_days=debug_keep_days,
            trace_base_dir=trace_base_dir,
        )

    @staticmethod
    def forecast_from_history(
        df_historico: pd.DataFrame,
        *,
        horizon: int = 15,
        base_dir: str = "static/",
        model_store_dir: Optional[str] = None,
        eval_weeks: int = 10,
        save_debug_details: bool = False,
        mutate_state: bool = False,
    ) -> pd.DataFrame:
        def _call(obj, method_name: str, **kwargs):
            fn = getattr(obj, method_name)
            sig = inspect.signature(fn)
            filt = {k: v for k, v in kwargs.items() if k in sig.parameters}
            return fn(**filt)

        df = df_historico.copy()

        if "CajeroId" in df.columns and "id" not in df.columns:
            df = df.rename(columns={"CajeroId": "id"})

        if "Fecha" not in df.columns or "Cantidad" not in df.columns or "id" not in df.columns:
            raise ValueError("df_historico debe contener columnas: Fecha, Cantidad, CajeroId.")

        df["fecha"] = pd.to_datetime(df["Fecha"], errors="coerce").dt.normalize()
        df["valor"] = pd.to_numeric(df["Cantidad"], errors="coerce")
        df["id"] = df["id"].astype(str).str.strip()
        df = df.dropna(subset=["fecha", "valor", "id"]).copy()

        if df.empty:
            return pd.DataFrame(columns=["Fecha", "MoPronosticoDispensado", "CajeroId"])

        os.makedirs(base_dir, exist_ok=True)
        store_dir = str(model_store_dir or base_dir)
        out = []

        for cliente, g in df.sort_values(["id", "fecha"]).groupby("id", sort=False):
            g = g.drop_duplicates(subset=["fecha"], keep="last").sort_values("fecha").reset_index(drop=True)
            if g.empty:
                continue

            json_path = os.path.join(base_dir, f"S4DR_{cliente}.json")
            modelo = S4DRModelV102(
                id_unico=cliente,
                json_file=json_path,
                model_store_dir=store_dir,
            )
            modelo.eval_weeks = int(eval_weeks)
            modelo.actualizar_modelo(df_nuevos=g[["fecha", "valor", "id"]])

            last_hist_date = pd.to_datetime(g["fecha"].max()).normalize()
            preds_raw, fechas_modelo, _usados = _call(
                modelo,
                "seleccionar_atractores",
                P=int(horizon),
                anchor_date=last_hist_date,
                save_debug_details=bool(save_debug_details),
                mutate_state=bool(mutate_state),
                trace_base_dir=str(base_dir),
            )

            preds_arr = np.asarray(preds_raw, dtype=float)
            if fechas_modelo is not None and len(fechas_modelo) == int(horizon):
                fechas_pred = pd.DatetimeIndex(fechas_modelo).normalize()
            else:
                fechas_pred = pd.date_range(
                    start=last_hist_date + timedelta(days=1),
                    periods=int(horizon),
                    freq="D",
                ).normalize()

            out.append(
                pd.DataFrame(
                    {
                        "Fecha": fechas_pred,
                        "MoPronosticoDispensado": preds_arr,
                        "CajeroId": cliente,
                    }
                )
            )

        if not out:
            return pd.DataFrame(columns=["Fecha", "MoPronosticoDispensado", "CajeroId"])

        return pd.concat(out, ignore_index=True).sort_values(["CajeroId", "Fecha"]).reset_index(drop=True)
