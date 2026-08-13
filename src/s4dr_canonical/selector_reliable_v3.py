
# selector_reliable_v3.py
# ============================================================
# Reliable selector (v2) — online "expert advice" + hierarchical buckets
# ============================================================
# Objetivo:
#   Seleccionar (Top-1, Top-2) candidatos de pronóstico (periodos/atractores,
#   LGBM, CatBoost, etc.) de forma *estrictamente causal* (sin data leakage),
#   con actualización online (Hedge / Fixed-Share) por bucket de calendario.
#
# Por qué v2:
#   1) Evita que el selector se "pegue" indefinidamente a 1–2 modelos por priors.
#   2) Usa un algoritmo estándar (Weighted Majority / Hedge) con exploración (Fixed-Share).
#   3) Soporta jerarquía de buckets (weekday>week_of_month>pay_bucket -> ... -> GLOBAL).
#   4) Incluye controles para que ML no monopolice Top-1/Top-2 si hay un
#      candidato no-ML competitivo y con evidencia suficiente.
#
# Reglas anti-leakage (importante):
#   - select_candidates_for_day(...) solo usa:
#       (a) preds del día a seleccionar (estimados con info <= t-1)
#       (b) estado acumulado construido *solo* con días anteriores
#   - update_weights(...) debe llamarse únicamente cuando y_true del día ya es conocido.
#
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Any, Mapping
import json
import math
import numpy as np


BucketKey = Tuple[Any, ...]
StoreKey = Tuple[str, BucketKey]


# -----------------------------
# Helpers
# -----------------------------
def _safe_float(x, default=np.nan) -> float:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
        return default
    except Exception:
        return default


def _normalize_weights(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    w = np.clip(w, 0.0, np.inf)
    s = float(np.sum(w))
    if not math.isfinite(s) or s <= 0.0:
        # uniform fallback
        n = w.size
        if n == 0:
            return w
        return np.ones(n, dtype=float) / n
    return w / s


def _ape(y_true: float, y_hat: float, eps: float = 1e-9) -> float:
    y = float(y_true)
    yh = float(y_hat)
    if not math.isfinite(y) or not math.isfinite(yh):
        return np.nan
    denom = abs(y) + eps
    return abs(y - yh) / denom


def _clip_loss(loss: float, clip: float) -> float:
    if not math.isfinite(loss):
        return np.nan
    if clip is None or clip <= 0:
        return loss
    return float(min(loss, clip))


def build_bucket_key(level: str, ctx: Dict[str, Any]) -> BucketKey:
    """
    ctx debe incluir (si aplica): weekday, week_of_month, pay_bucket
    """
    if level == "weekday>week_of_month>pay_bucket":
        return (ctx.get("weekday"), ctx.get("week_of_month"), ctx.get("pay_bucket"))
    if level == "weekday>pay_bucket":
        return (ctx.get("weekday"), ctx.get("pay_bucket"))
    if level == "weekday":
        return (ctx.get("weekday"),)
    if level == "GLOBAL_FALLBACK":
        return ("GLOBAL",)
    raise ValueError(f"Nivel de bucket no soportado: {level}")


def _parent_level(level: str) -> Optional[str]:
    if level == "weekday>week_of_month>pay_bucket":
        return "weekday>pay_bucket"
    if level == "weekday>pay_bucket":
        return "weekday"
    if level == "weekday":
        return "GLOBAL_FALLBACK"
    if level == "GLOBAL_FALLBACK":
        return None
    return None



# =========================
# Config
# =========================

@dataclass(frozen=True)
class SelectorConfig:
    # --- Buckets / jerarquía ---
    bucket_levels: Tuple[Tuple[str, ...], ...] = (
        ("weekday", "pay_bucket"),
        ("weekday",),
        ("pay_bucket",),
        ("GLOBAL",),
    )
    # Umbrales informativos (no son backoff duro; se usan para "dominant bucket" / auditoría)
    n_min_fino: int = 6
    n_min_grueso: int = 10

    # --- Métrica / pérdidas ---
    eps_y: float = 1.0  # denom = max(|y|, eps_y)
    # tau por nivel controla "cuánto confiar" en el bucket con n pequeño (alpha = n/(n+tau))
    tau_default: float = 10.0
    tau_by_level: Optional[Dict[str, float]] = None
    prior_weight_base: float = 1.0  # asegura que el prior siempre tenga algo de peso

    # --- Ensamble ---
    eta: float = 2.0              # softmax(-eta*loss); mayor => más selectivo
    top_k: int = 2                # candidatos usados para pred_final
    use_top_k: bool = True

    # --- ML gating (suave y condicional) ---
    ml_candidates: Tuple[str, ...] = ("LGBM_MODEL", "PROPHET_MODEL")
    ml_floor: float = 0.15        # peso mínimo total a ML cuando aplica
    ml_delta: float = 0.02        # solo forzar ML si su loss <= best_loss + ml_delta
    support_low_thr: float = 0.35 # evidencia baja si sum(alpha_non_global) < support_low_thr

    # --- Histeresis opcional (anti flip-flop) ---
    use_hysteresis: bool = False
    switch_margin: float = 0.05   # 5% margen para cambiar ganador por bucket

# -----------------------------
# State / result
# -----------------------------
@dataclass
class SelectorResult:
    top1: str
    top2: Optional[str]
    w1: float
    w2: float
    pred_final: float
    pred_top1: float
    pred_top2: float
    bucket_type: str
    bucket_id: BucketKey
    bucket_hist_n: float
    debug: Dict[str, Any]


@dataclass
class SelectorState:
    candidates: List[str]                         # orden fijo
    priors: Dict[str, float]                      # prior loss por candidato (bajo=mejor)
    weights_store: Dict[StoreKey, List[float]]    # pesos por bucket
    n_eff_store: Dict[StoreKey, float]            # "exposición" efectiva por bucket


# -----------------------------
# Init / export / load
# -----------------------------
def init_state(
    candidates: Iterable[str],
    *,
    priors: Optional[Dict[str, float]] = None,
) -> SelectorState:
    """
    Inicializa el estado.
    - priors: pérdidas iniciales (ej. error medio en ventana de evaluación).
      Si no se dan, se inicia uniforme.
    """
    cand = list(candidates)
    if not cand:
        raise ValueError("candidates vacío.")
    if priors is None:
        priors = {c: 1.0 for c in cand}
    # sanea priors faltantes
    p = {}
    for c in cand:
        v = _safe_float(priors.get(c, np.nan), default=np.nan)
        p[c] = (v if math.isfinite(v) else 1.0)

    return SelectorState(
        candidates=cand,
        priors=p,
        weights_store={},
        n_eff_store={},
    )


def export_state(state: SelectorState, path: str) -> None:
    payload = {
        "candidates": state.candidates,
        "priors": state.priors,
        "weights_store": {
            f"{lvl}|{json.dumps(list(key))}": w
            for (lvl, key), w in state.weights_store.items()
        },
        "n_eff_store": {
            f"{lvl}|{json.dumps(list(key))}": float(n)
            for (lvl, key), n in state.n_eff_store.items()
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_state(path: str) -> SelectorState:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    cand = list(payload["candidates"])
    priors = dict(payload.get("priors", {}))

    weights_store: Dict[StoreKey, List[float]] = {}
    for k, w in payload.get("weights_store", {}).items():
        lvl, key_json = k.split("|", 1)
        key = tuple(json.loads(key_json))
        weights_store[(lvl, key)] = list(map(float, w))

    n_eff_store: Dict[StoreKey, float] = {}
    for k, n in payload.get("n_eff_store", {}).items():
        lvl, key_json = k.split("|", 1)
        key = tuple(json.loads(key_json))
        n_eff_store[(lvl, key)] = float(n)

    return SelectorState(
        candidates=cand,
        priors={c: float(priors.get(c, 1.0)) for c in cand},
        weights_store=weights_store,
        n_eff_store=n_eff_store,
    )


# =========================
# Public API: update (no leakage)
# =========================

def update_stats_for_day(
    day_features: Mapping[str, Any],
    y_true: Any,
    preds_row: Mapping[str, Any],
    *,
    state: Dict[str, Any],
    cfg: Optional[SelectorConfig] = None,
) -> Dict[str, Any]:
    """
    Actualiza stats_store con el real y todas las predicciones del día.
    Debe llamarse SOLO después de predecir y observar y_true.

    Full-feedback:
      - Actualiza para TODOS los candidatos con yhat finito.
    Full-level:
      - Actualiza para TODOS los niveles jerárquicos (fino→grueso→GLOBAL).

    Retorna audit_update (por nivel), útil para debug.
    """
    if state is None:
        raise ValueError("state es requerido para update (modo online).")

    cfg = cfg or SelectorConfig()
    try:
        cfg = SelectorConfig(**state.get("cfg", asdict(cfg)))
    except Exception:
        pass

    stats_store = state.setdefault("stats_store", {})
    candidates = list(state.get("candidates", []))

    if not _is_finite(y_true):
        return {"skipped": True, "reason": "y_true_not_finite"}

    y = float(y_true)
    denom = max(abs(y), float(cfg.eps_y))

    # preds válidos
    preds_valid = {c: float(preds_row[c]) for c in candidates if c in preds_row and _is_finite(preds_row[c])}
    if not preds_valid:
        return {"skipped": True, "reason": "no_valid_predictions"}

    audit_levels = []
    for level in cfg.bucket_levels:
        lid = _level_id(level)
        bkey = _bucket_key_for(level, day_features)
        bucket_map = _ensure_bucket_stats(stats_store, lid, bkey, candidates)

        for c, yhat in preds_valid.items():
            err = abs(y - float(yhat))
            st = bucket_map.get(c)
            if st is None:
                st = {"sum_abs_err": 0.0, "sum_abs_y": 0.0, "n": 0}
                bucket_map[c] = st
            st["sum_abs_err"] = float(st["sum_abs_err"]) + float(err)
            st["sum_abs_y"] = float(st["sum_abs_y"]) + float(denom)
            st["n"] = int(st["n"]) + 1

        # auditoría por nivel
        n_eff = _n_eff_for_bucket(stats_store, lid, bkey)
        audit_levels.append({"level_id": lid, "bucket_key": bkey, "n_eff_after": int(n_eff)})

    return {"skipped": False, "updated_levels": audit_levels}
# -----------------------------
# Core: get weights for a context
# -----------------------------
def _ensure_bucket_weights(
    state: SelectorState,
    store_key: StoreKey,
    *,
    init_from_priors: bool = True,
    weight_floor: float = 1e-6,
) -> np.ndarray:
    """
    Asegura que existe vector de pesos para store_key; si no, lo crea.
    """
    n = len(state.candidates)
    if store_key not in state.weights_store:
        if init_from_priors:
            # priors bajos => más peso
            prior_losses = np.array([state.priors.get(c, 1.0) for c in state.candidates], dtype=float)
            prior_losses = np.where(np.isfinite(prior_losses), prior_losses, 1.0)
            inv = 1.0 / np.clip(prior_losses, 1e-6, np.inf)
            w0 = _normalize_weights(inv)
        else:
            w0 = np.ones(n, dtype=float) / n

        # floor
        w0 = np.maximum(w0, weight_floor)
        w0 = _normalize_weights(w0)
        state.weights_store[store_key] = w0.tolist()
        state.n_eff_store[store_key] = 0.0

    w = np.array(state.weights_store[store_key], dtype=float)
    if w.size != n:
        # si cambió el set de candidatos, re-inicializa (mejor seguro que silencioso)
        w = np.ones(n, dtype=float) / n
    w = np.maximum(w, weight_floor)
    return _normalize_weights(w)


def _blend_with_parent(
    state: SelectorState,
    lvl: str,
    key: BucketKey,
    w: np.ndarray,
    n_eff: float,
    *,
    gamma_parent: float,
    weight_floor: float,
) -> np.ndarray:
    """
    Mezcla con parent cuando hay poca evidencia en el bucket actual.
      w_blend = (n/(n+gamma))*w_child + (gamma/(n+gamma))*w_parent
    """
    parent = _parent_level(lvl)
    if parent is None:
        return w

    parent_key = build_bucket_key(parent, {
        "weekday": key[0] if len(key) >= 1 else None,
        "week_of_month": key[1] if len(key) >= 2 else None,
        "pay_bucket": key[2] if len(key) >= 3 else (key[1] if len(key) == 2 else None),
    })

    sk_parent = (parent, parent_key)
    w_parent = _ensure_bucket_weights(state, sk_parent, init_from_priors=True, weight_floor=weight_floor)
    n_parent = float(state.n_eff_store.get(sk_parent, 0.0))

    # usa la evidencia del hijo; si es 0, casi todo al parent
    a = float(n_eff) / (float(n_eff) + float(gamma_parent))
    a = 0.0 if not math.isfinite(a) else max(0.0, min(1.0, a))

    w_blend = a * w + (1.0 - a) * w_parent
    return _normalize_weights(np.maximum(w_blend, weight_floor))


# -----------------------------
# Selection
# -----------------------------
def select_candidates_for_day(
    *,
    day_ctx: Dict[str, Any],
    preds: Dict[str, float],
    cfg: Dict[str, Any],
) -> SelectorResult:
    """
    Selección causal para un día futuro.

    Parámetros:
      day_ctx: {"weekday": int, "week_of_month": int, "pay_bucket": str, ...}
      preds:   {candidate_code: yhat}
      cfg:     dict con al menos:
        - "state": SelectorState
        - "bucket_levels": list[str]
        - "K": int (normalmente 2)
        - "eta": float (Hedge learning rate)
        - "alpha_share": float (Fixed-Share, 0.0..0.2)
        - "loss_clip": float (ej. 3.0)
        - "gamma_parent": float (mezcla con parent, ej. 12.0)
        - "weight_floor": float (ej. 1e-6)
        - "ml_candidates": set[str]
        - "max_ml_in_topk": int (0..K, ej. 2)
        - "diversity_delta": float (ej. 0.03)  # si non-ML está dentro de +3% del ML, lo metemos
        - "diversity_min_n": float (ej. 6.0)
        - "w_top1": float (ej. 0.7)
    """
    state: SelectorState = cfg["state"]
    bucket_levels: List[str] = list(cfg.get("bucket_levels", [
        "weekday>week_of_month>pay_bucket",
        "weekday>pay_bucket",
        "weekday",
        "GLOBAL_FALLBACK",
    ]))
    K = int(cfg.get("K", 2))
    if K < 1:
        K = 1

    weight_floor = float(cfg.get("weight_floor", 1e-6))
    gamma_parent = float(cfg.get("gamma_parent", 12.0))
    w_top1 = float(cfg.get("w_top1", 0.7))
    w_top1 = max(0.0, min(1.0, w_top1))
    w_top2 = 1.0 - w_top1

    ml_candidates = set(cfg.get("ml_candidates", set()))
    max_ml_in_topk = int(cfg.get("max_ml_in_topk", K))
    max_ml_in_topk = max(0, min(K, max_ml_in_topk))

    diversity_delta = float(cfg.get("diversity_delta", 0.03))
    diversity_min_n = float(cfg.get("diversity_min_n", 6.0))

    # Filtra preds finitos
    preds_f = {k: _safe_float(v, default=np.nan) for k, v in preds.items()}
    preds_f = {k: v for k, v in preds_f.items() if math.isfinite(v)}
    if not preds_f:
        # sin preds -> devuelve dummy
        return SelectorResult(
            top1=state.candidates[0],
            top2=None,
            w1=1.0, w2=0.0,
            pred_final=float("nan"),
            pred_top1=float("nan"),
            pred_top2=float("nan"),
            bucket_type="GLOBAL_FALLBACK",
            bucket_id=("GLOBAL",),
            bucket_hist_n=0.0,
            debug={"reason": "no_finite_preds"},
        )

    # Construye pesos para el bucket más específico disponible
    chosen_lvl = "GLOBAL_FALLBACK"
    chosen_key = ("GLOBAL",)
    chosen_store_key: StoreKey = ("GLOBAL_FALLBACK", ("GLOBAL",))

    # Selecciona el bucket más específico que exista o que podamos inicializar
    for lvl in bucket_levels:
        key = build_bucket_key(lvl, day_ctx)
        sk = (lvl, key)
        # inicializa si no existe (pero sin inflar evidencia)
        _ensure_bucket_weights(state, sk, init_from_priors=True, weight_floor=weight_floor)
        chosen_lvl, chosen_key, chosen_store_key = lvl, key, sk
        # usamos el primero por orden (más específico)
        break

    w_raw = _ensure_bucket_weights(state, chosen_store_key, init_from_priors=True, weight_floor=weight_floor)
    n_eff = float(state.n_eff_store.get(chosen_store_key, 0.0))

    # Mezcla con parent si evidencia baja
    w = _blend_with_parent(
        state, chosen_lvl, chosen_key, w_raw, n_eff,
        gamma_parent=gamma_parent, weight_floor=weight_floor
    )

    # Restringe a candidatos que sí tienen predicción hoy
    cand = state.candidates
    mask = np.array([1.0 if c in preds_f else 0.0 for c in cand], dtype=float)
    if mask.sum() <= 0:
        # fallback: usa el mejor pred disponible
        k0 = next(iter(preds_f.keys()))
        return SelectorResult(
            top1=k0, top2=None, w1=1.0, w2=0.0,
            pred_final=float(preds_f[k0]), pred_top1=float(preds_f[k0]), pred_top2=float("nan"),
            bucket_type=chosen_lvl, bucket_id=chosen_key, bucket_hist_n=n_eff,
            debug={"reason": "no_candidate_mask"},
        )

    w = w * mask
    w = _normalize_weights(np.maximum(w, weight_floor) * mask)

    # Ranking por peso
    order = np.argsort(-w)  # descendente
    ranked = [cand[i] for i in order if cand[i] in preds_f]

    # Enforce: no más de max_ml_in_topk ML dentro del topK (si es posible)
    if max_ml_in_topk < K and ranked:
        new_ranked = []
        ml_count = 0
        for c in ranked:
            is_ml = c in ml_candidates
            if is_ml:
                if ml_count < max_ml_in_topk:
                    new_ranked.append(c); ml_count += 1
            else:
                new_ranked.append(c)
            if len(new_ranked) >= K:
                break

        # completa si faltan (aunque sean ML)
        if len(new_ranked) < K:
            for c in ranked:
                if c not in new_ranked:
                    new_ranked.append(c)
                if len(new_ranked) >= K:
                    break

        ranked = new_ranked

    top1 = ranked[0]
    top2 = ranked[1] if (K >= 2 and len(ranked) >= 2) else None

    # Diversidad suave: si top1 y top2 son ML, intenta meter el mejor no-ML cercano
    if K >= 2 and top2 is not None and (top1 in ml_candidates) and (top2 in ml_candidates):
        # encuentra mejor no-ML por peso
        best_nonml = None
        for c in ranked[2:]:
            if c not in ml_candidates:
                best_nonml = c
                break
        if best_nonml is None:
            # busca en todo
            for c in cand:
                if c in preds_f and (c not in ml_candidates):
                    best_nonml = c
                    break

        if best_nonml is not None:
            # condición: evidencia suficiente en bucket actual (o su parent blend)
            if n_eff >= diversity_min_n:
                # compara pesos (proxy de competitividad)
                if w[cand.index(best_nonml)] >= w[cand.index(top2)] * (1.0 - diversity_delta):
                    top2 = best_nonml

    pred1 = float(preds_f[top1])
    if top2 is None:
        pred2 = float("nan")
        pred_final = pred1
        w1, w2 = 1.0, 0.0
    else:
        pred2 = float(preds_f[top2])
        pred_final = w_top1 * pred1 + w_top2 * pred2
        w1, w2 = w_top1, w_top2

    debug = {
        "weights_top": {top1: float(w[cand.index(top1)]), (top2 or "None"): float(w[cand.index(top2)]) if top2 else None},
        "n_eff": n_eff,
        "mask_sum": float(mask.sum()),
    }

    return SelectorResult(
        top1=top1,
        top2=top2,
        w1=w1,
        w2=w2,
        pred_final=float(pred_final),
        pred_top1=pred1,
        pred_top2=pred2,
        bucket_type=chosen_lvl,
        bucket_id=chosen_key,
        bucket_hist_n=float(n_eff),
        debug=debug,
    )


# -----------------------------
# Online update (Hedge / Fixed-Share)
# -----------------------------
def update_weights(
    *,
    state: SelectorState,
    day_ctx: Dict[str, Any],
    preds: Dict[str, float],
    y_true: float,
    cfg: Dict[str, Any],
) -> None:
    """
    Actualiza el estado con el resultado observado del día (SOLO cuando y_true es conocido).

    cfg:
      - "eta": float
      - "alpha_share": float
      - "loss_clip": float
      - "bucket_levels": list[str]
      - "weight_floor": float
    """
    eta = float(cfg.get("eta", 0.8))
    alpha_share = float(cfg.get("alpha_share", 0.02))
    loss_clip = cfg.get("loss_clip", 3.0)
    weight_floor = float(cfg.get("weight_floor", 1e-6))

    # pred finitos
    preds_f = {k: _safe_float(v, default=np.nan) for k, v in preds.items()}
    preds_f = {k: v for k, v in preds_f.items() if math.isfinite(v)}
    if not preds_f or not math.isfinite(_safe_float(y_true)):
        return

    y = float(y_true)
    bucket_levels: List[str] = list(cfg.get("bucket_levels", [
        "weekday>week_of_month>pay_bucket",
        "weekday>pay_bucket",
        "weekday",
        "GLOBAL_FALLBACK",
    ]))

    cand = state.candidates
    nC = len(cand)

    # computa pérdidas por candidato (APE)
    loss_vec = np.full(nC, np.nan, dtype=float)
    for i, c in enumerate(cand):
        if c in preds_f:
            loss_vec[i] = _clip_loss(_ape(y, preds_f[c]), float(loss_clip) if loss_clip is not None else None)

    # actualiza todos los niveles (esto genera info para jerarquía)
    for lvl in bucket_levels:
        key = build_bucket_key(lvl, day_ctx)
        sk: StoreKey = (lvl, key)

        w = _ensure_bucket_weights(state, sk, init_from_priors=True, weight_floor=weight_floor)

        # actualización Hedge: w_i <- w_i * exp(-eta * loss_i)
        # si loss_i es nan, no castigamos (equivalente a no-update para ese experto)
        adj = np.ones_like(w)
        m = np.isfinite(loss_vec)
        adj[m] = np.exp(-eta * loss_vec[m])
        w_new = w * adj

        # Fixed-share: mezcla con uniforme para evitar colapso
        if alpha_share > 0:
            w_new = (1.0 - alpha_share) * w_new + alpha_share * (np.ones(nC, dtype=float) / nC)

        # floor + normalize
        w_new = np.maximum(w_new, weight_floor)
        w_new = _normalize_weights(w_new)

        state.weights_store[sk] = w_new.tolist()
        state.n_eff_store[sk] = float(state.n_eff_store.get(sk, 0.0) + 1.0)


# -----------------------------
# Convenience: selector_step
# -----------------------------
def selector_step(
    *,
    day_ctx: Dict[str, Any],
    preds: Dict[str, float],
    cfg: Dict[str, Any],
    y_true: Optional[float] = None,
    update: bool = False,
) -> SelectorResult:
    """
    Unifica select + (opcional) update en un solo paso.
    Para backtest: pasar y_true y update=True.
    Para producción: update=False.
    """
    res = select_candidates_for_day(day_ctx=day_ctx, preds=preds, cfg=cfg)
    if update and (y_true is not None):
        update_weights(state=cfg["state"], day_ctx=day_ctx, preds=preds, y_true=float(y_true), cfg=cfg)
    return res
