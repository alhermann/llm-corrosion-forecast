"""
Autoregressive SDF forecasting with deterministic stabilisers
and Monte-Carlo rollouts for uncertainty quantification.
"""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import config as cfg
from .data_loading import (
    binarize_0_255,
    load_slice_across_time,
    measure_area_material,
    to_material_bool,
)
from .knn import knn_predict_delta
from .llm_interface import call_llm_delta
from .metrics import boundary_f1, dice, iou
from .pca import _field_from_img, project_field, reconstruct_field, set_seed
from .sdf_utils import material_to_sdf, postprocess_material, sdf_to_material, smooth_sdf


# ── Training-derived caps ────────────────────────────────────

def compute_training_caps(
    train_ids: List[int],
    mean_field: np.ndarray,
    Vt: np.ndarray,
    z_mean: np.ndarray,
    z_std: np.ndarray,
    k_llm: int,
    max_slices: int = 250,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """
    Compute per-component and magnitude caps from GT training deltas.

    These caps are purely derived from training statistics and contain
    no test-set information.
    """
    set_seed(seed)
    ids = train_ids.copy()
    random.shuffle(ids)
    ids = ids[: min(max_slices, len(ids))]

    deltas = []
    used = 0

    for idx in ids:
        seq = load_slice_across_time(idx, use_cache=False)
        if seq is None:
            continue

        fields = [_field_from_img(im) for im in seq]
        Z_phys = np.array(
            [project_field(f, mean_field, Vt) for f in fields], dtype=np.float32
        )
        Z_norm = (Z_phys - z_mean) / z_std
        Z_top = Z_norm[:, :k_llm]

        for t in range(cfg.NUM_TIMESTEPS - 1):
            deltas.append(Z_top[t + 1] - Z_top[t])

        used += 1
        if used % 50 == 0:
            print(f"[CAPS] processed {used}/{len(ids)} slices …", flush=True)

    if not deltas:
        raise RuntimeError("No deltas collected for training caps.")

    Y = np.stack(deltas).astype(np.float32)
    mags = np.linalg.norm(Y, axis=1)

    caps = {
        "delta_mag_cap": float(np.percentile(mags, 95)) * cfg.DELTA_MAG_P95_MULT,
        "comp_abs_cap": np.percentile(np.abs(Y), 99, axis=0).astype(np.float32)
        * cfg.DELTA_COMP_P99_MULT,
    }
    print(
        f"[CAPS] built from {Y.shape[0]} deltas (slices used={used}); "
        f"delta_mag_cap={caps['delta_mag_cap']:.3f}"
    )
    return caps


# ── Deterministic delta constraints ──────────────────────────

def clip_delta_to_training(
    delta: np.ndarray, caps: Optional[Dict], horizon: int = 1
) -> np.ndarray:
    """Apply training-derived magnitude and per-component caps, plus horizon damping."""
    d = delta.astype(np.float32).copy()

    if caps is not None and "comp_abs_cap" in caps:
        d = np.clip(d, -caps["comp_abs_cap"], caps["comp_abs_cap"])

    if caps is not None and "delta_mag_cap" in caps:
        mag = float(np.linalg.norm(d) + 1e-9)
        capm = float(caps["delta_mag_cap"])
        if mag > capm:
            d *= capm / mag

    if cfg.APPLY_HORIZON_DAMP and cfg.HORIZON_DAMP < 1.0:
        d *= float(cfg.HORIZON_DAMP ** max(0, horizon - 1))

    return d


def apply_velocity_relative_cap(
    delta: np.ndarray,
    vel: np.ndarray,
    mult: float = 6.0,
    bias: float = 0.2,
    min_cap: Optional[float] = None,
) -> np.ndarray:
    """Cap delta magnitude relative to observed latent velocity."""
    d = delta.astype(np.float32).copy()
    vmag = float(np.linalg.norm(vel) + 1e-9)
    cap = float(mult * vmag + bias)
    if min_cap is not None:
        cap = max(cap, float(min_cap))
    mag = float(np.linalg.norm(d) + 1e-9)
    if mag > cap:
        d *= cap / mag
    return d


# ── Single-step forecast ─────────────────────────────────────

def forecast_next_sdf(
    history_sdfs: List[np.ndarray],
    history_times_h: np.ndarray,
    mean_field: np.ndarray,
    Vt: np.ndarray,
    z_mean: np.ndarray,
    z_std: np.ndarray,
    k_llm: int,
    meta_df=None,
    horizon: int = 1,
    rollout_nonce: Optional[float] = None,
    caps: Optional[Dict] = None,
    knn_lib: Optional[Dict] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Predict the SDF at the next time-step given history.

    Pipeline:
        1. Project history into normalised PCA space
        2. Compute kNN prior delta (if enabled)
        3. Ask LLM for a residual correction
        4. Apply deterministic stabilisers (caps, damping, velocity cap)
        5. Reconstruct the predicted SDF
        6. Optionally smooth and enforce monotonic shrinkage
    """
    T_obs = len(history_sdfs)
    H, W = history_sdfs[0].shape

    Z_phys = np.array(
        [project_field(s, mean_field, Vt) for s in history_sdfs], dtype=np.float32
    )
    Z_norm = (Z_phys - z_mean) / z_std
    Z_top = Z_norm[:, :k_llm]

    dt_h = float(history_times_h[-1] - history_times_h[-2]) if T_obs >= 2 else 1.0

    # Metadata context
    meta_row = None
    if meta_df is not None:
        t_idx = min(int(T_obs - 1), len(meta_df) - 1)
        meta_row = {
            c: float(meta_df.loc[t_idx, c])
            for c in meta_df.columns
            if np.issubdtype(type(meta_df.loc[t_idx, c]), np.number)
            or isinstance(meta_df.loc[t_idx, c], (int, float, np.integer, np.floating))
        }

    # kNN prior
    delta_prior = np.zeros(k_llm, dtype=np.float32)
    prior_std = np.ones(k_llm, dtype=np.float32)
    if cfg.USE_KNN_GUIDE and knn_lib is not None:
        z_last = Z_top[-1].astype(np.float32)
        vel = (
            (Z_top[-1] - Z_top[-2]).astype(np.float32)
            if T_obs >= 2
            else np.zeros_like(z_last)
        )
        delta_prior, prior_std = knn_predict_delta(
            knn_lib, z_last, vel, float(history_times_h[-1]), dt_h, k=cfg.KNN_K
        )

    # Residual caps
    train_mag_cap = (
        float(caps["delta_mag_cap"]) if caps and "delta_mag_cap" in caps else 10.0
    )
    residual_norm_cap = float(cfg.RESIDUAL_NORM_FRAC * train_mag_cap)
    residual_comp_cap = float(max(0.25, 3.0 * float(np.median(prior_std))))

    # LLM call
    delta_llm, raw_text, status, llm_info = call_llm_delta(
        Z_top,
        dt_h,
        rollout_nonce=rollout_nonce,
        meta_row=meta_row,
        horizon=horizon,
        delta_prior=delta_prior if (cfg.USE_KNN_GUIDE and cfg.LLM_PREDICTS_RESIDUAL) else None,
        prior_std=prior_std if (cfg.USE_KNN_GUIDE and cfg.LLM_PREDICTS_RESIDUAL) else None,
        residual_norm_cap=residual_norm_cap if (cfg.USE_KNN_GUIDE and cfg.LLM_PREDICTS_RESIDUAL) else None,
        residual_comp_cap=residual_comp_cap if (cfg.USE_KNN_GUIDE and cfg.LLM_PREDICTS_RESIDUAL) else None,
    )
    delta_llm = delta_llm.astype(np.float32)

    # If LLM output was zero-padded, fill tail with prior
    if cfg.USE_KNN_GUIDE and llm_info.get("length_repaired", False):
        rlen = llm_info.get("returned_len")
        if rlen is not None and rlen < k_llm:
            delta_llm[rlen:] = delta_prior[rlen:]

    # Combine LLM residual with kNN prior
    if cfg.USE_KNN_GUIDE and cfg.LLM_PREDICTS_RESIDUAL:
        residual = np.clip(delta_llm, -residual_comp_cap, residual_comp_cap)
        rmag = float(np.linalg.norm(residual) + 1e-9)
        if rmag > residual_norm_cap:
            residual *= residual_norm_cap / rmag
        delta = delta_prior + float(cfg.RESIDUAL_SCALE) * residual
    else:
        residual = delta_llm - delta_prior
        residual = np.clip(residual, -residual_comp_cap, residual_comp_cap)
        rmag = float(np.linalg.norm(residual) + 1e-9)
        if rmag > residual_norm_cap:
            residual *= residual_norm_cap / rmag
        delta = delta_prior + residual

    # Deterministic stabilisers
    if cfg.APPLY_TRAINING_CAPS:
        delta = clip_delta_to_training(delta, caps=caps, horizon=horizon)

    if cfg.APPLY_VEL_REL_CAP and T_obs >= 2 and not (
        cfg.USE_KNN_GUIDE and cfg.DISABLE_VEL_CAP_WHEN_KNN
    ):
        vel = (Z_top[-1] - Z_top[-2]).astype(np.float32)
        min_cap = (
            float(cfg.VEL_CAP_MIN_FRAC_OF_TRAIN * caps["delta_mag_cap"])
            if caps and "delta_mag_cap" in caps
            else None
        )
        delta = apply_velocity_relative_cap(
            delta, vel, mult=cfg.VEL_REL_MULT, bias=cfg.VEL_REL_BIAS, min_cap=min_cap
        )

    # Update latent and reconstruct
    z_last_norm_top = Z_norm[-1, :k_llm]
    z_next_norm_top = z_last_norm_top + delta

    z_next_phys = Z_phys[-1].copy()
    z_next_phys[:k_llm] = z_next_norm_top * z_std[:k_llm] + z_mean[:k_llm]

    sdf_pred = reconstruct_field(z_next_phys, mean_field, Vt, H, W).astype(np.float32)

    sdf_pred = smooth_sdf(sdf_pred)

    if cfg.APPLY_MONO_SDF_SHRINK:
        sdf_pred = np.maximum(sdf_pred, history_sdfs[-1])

    dbg = {
        "delta_mag": float(np.linalg.norm(delta)),
        "delta_llm_mag": float(np.linalg.norm(delta_llm)),
        "delta_prior_mag": float(np.linalg.norm(delta_prior)),
        "llm_http_status": int(status),
        "llm_returned_len": int(llm_info.get("returned_len", -1)),
        "llm_retries_used": int(llm_info.get("retries_used", 0)),
        "llm_length_repaired": bool(llm_info.get("length_repaired", False)),
        "llm_salvaged_from_text": bool(llm_info.get("salvaged_from_text", False)),
    }
    return sdf_pred, dbg


# ── Monte-Carlo rollouts ─────────────────────────────────────

def rollout_mc(
    full_gt_imgs: List[np.ndarray],
    times_h_all: np.ndarray,
    start_t: int,
    mean_field: np.ndarray,
    Vt: np.ndarray,
    z_mean: np.ndarray,
    z_std: np.ndarray,
    k_llm: int,
    caps: Optional[Dict] = None,
    meta_df=None,
    knn_lib: Optional[Dict] = None,
    n_rollouts: int = 8,
    verbose: bool = True,
) -> Tuple[List[Dict], List[np.ndarray]]:
    """
    Run *n_rollouts* autoregressive forecasts from *start_t* to the end.

    Returns ``(records, final_preds)`` where *records* is a list of
    per-step metric dicts and *final_preds* is the list of predicted
    material masks at the final time-step.
    """
    gt_bin = [binarize_0_255(im) for im in full_gt_imgs]
    gt_mat = [to_material_bool(b) for b in gt_bin]
    gt_sdf = [material_to_sdf(m).astype(np.float32) for m in gt_mat]

    records: List[Dict] = []
    final_preds: List[np.ndarray] = []

    for r in range(n_rollouts):
        history_sdfs = [gt_sdf[i].copy() for i in range(start_t)]
        history_times = np.array(times_h_all[:start_t], dtype=float)

        if verbose and r == 0:
            print(f"\n  [diag] rollout {r + 1}/{n_rollouts}")

        for t in range(start_t, cfg.NUM_TIMESTEPS):
            horizon = int(t - start_t + 1)

            sdf_next, dbg = forecast_next_sdf(
                history_sdfs,
                history_times,
                mean_field,
                Vt,
                z_mean,
                z_std,
                k_llm=k_llm,
                meta_df=meta_df,
                horizon=horizon,
                rollout_nonce=1000.0 * r + horizon,
                caps=caps,
                knn_lib=knn_lib,
            )

            pred_mat = sdf_to_material(sdf_next)
            if cfg.APPLY_MASK_POSTPROC:
                prev_mat = sdf_to_material(history_sdfs[-1])
                pred_mat = postprocess_material(pred_mat, prev_material=prev_mat)

            gt_t = gt_mat[t]
            row = {
                "rollout": r,
                "t": t,
                "horizon": horizon,
                "iou": iou(gt_t, pred_mat),
                "dice": dice(gt_t, pred_mat),
                "bf1": boundary_f1(gt_t, pred_mat, tol=1),
                "area_err": abs(
                    measure_area_material(pred_mat) - measure_area_material(gt_t)
                ),
                "delta_mag": dbg["delta_mag"],
                "delta_llm_mag": dbg["delta_llm_mag"],
                "llm_http_status": dbg["llm_http_status"],
                "llm_retries_used": dbg["llm_retries_used"],
                "llm_length_repaired": dbg["llm_length_repaired"],
                "llm_salvaged_from_text": dbg["llm_salvaged_from_text"],
            }
            records.append(row)

            if verbose and r == 0:
                print(
                    f"    t={t} h={horizon}: IoU={row['iou']:.3f} "
                    f"Dice={row['dice']:.3f} AreaErr={row['area_err']:.0f}"
                )

            history_sdfs.append(sdf_next)
            history_times = np.append(history_times, times_h_all[t])

        final_preds.append(sdf_to_material(history_sdfs[-1]))

    return records, final_preds
