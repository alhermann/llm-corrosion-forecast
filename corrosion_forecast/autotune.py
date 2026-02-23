"""
Automatic PCA hyper-parameter selection (K_LLM, K_RECON, NUM_TRAIN_SLICES).

Uses a downsampled, oracle-reconstruction grid search over:
    - number of training slices
    - number of principal components (K)

Selection criteria: IoU, Dice, Boundary-F1 thresholds plus
subspace stability between independent training splits.
"""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.utils.extmath import randomized_svd

    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False

from . import config as cfg
from .data_loading import binarize_0_255, load_frame_uint8, to_material_bool
from .metrics import boundary_f1, dice, iou
from .pca import _field_from_img, set_seed
from .sdf_utils import postprocess_material, sdf_to_material


# ── Caches ───────────────────────────────────────────────────
_AUTOTUNE_FIELD_CACHE: Dict[Tuple[int, int, int, bool], np.ndarray] = {}


def _downsample2d(arr2d: np.ndarray, factor: int = 2) -> np.ndarray:
    if factor is None or factor <= 1:
        return arr2d
    return arr2d[::factor, ::factor]


def get_autotune_field(
    slice_idx: int, t: int, downsample_factor: int, use_cache: bool = True
) -> Optional[np.ndarray]:
    """Load, convert, and optionally downsample a field for auto-tuning."""
    key = (slice_idx, t, downsample_factor, bool(cfg.USE_SDF_REPRESENTATION))
    if use_cache and key in _AUTOTUNE_FIELD_CACHE:
        return _AUTOTUNE_FIELD_CACHE[key]

    img = load_frame_uint8(slice_idx, t, use_cache=True)
    if img is None:
        return None
    f = _field_from_img(img)
    f = _downsample2d(f, downsample_factor).astype(cfg.AUTOTUNE_CACHE_DTYPE, copy=False)

    if use_cache:
        _AUTOTUNE_FIELD_CACHE[key] = f
    return f


def preload_autotune_fields(
    slice_ids: List[int], timesteps: List[int], downsample_factor: int = 2
) -> None:
    """Eagerly load fields into the cache for faster grid search."""
    total = len(slice_ids) * len(timesteps)
    done = 0
    print(
        f"[AUTO-TUNE] Preloading {total} fields (downsample={downsample_factor}) …",
        flush=True,
    )
    for i, idx in enumerate(slice_ids):
        for t in timesteps:
            get_autotune_field(idx, t, downsample_factor, use_cache=True)
            done += 1
        if (i + 1) % 50 == 0:
            print(f"  preloaded {i + 1}/{len(slice_ids)} slices", flush=True)
    print("[AUTO-TUNE] Preload done.", flush=True)


def fit_pca_on_indices(
    indices: List[int],
    k_fit_max: int,
    timesteps: List[int],
    downsample_factor: int = 2,
    seed: int = 0,
    use_cache: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """Fit a PCA basis on downsampled fields from the given slice indices."""
    set_seed(seed)
    rows = []
    H = W = None

    for idx in indices:
        for t in timesteps:
            f = get_autotune_field(idx, t, downsample_factor, use_cache=use_cache)
            if f is None:
                continue
            H, W = f.shape
            rows.append(f.flatten())

    if not rows:
        raise RuntimeError("No training rows for PCA fit.")

    X = np.array(rows, dtype=np.float32)
    mean_field = X.mean(axis=0)
    Xc = X - mean_field

    if _SKLEARN_OK:
        _, _, Vt = randomized_svd(Xc, n_components=k_fit_max, random_state=seed)
    else:
        _, _, Vt_full = np.linalg.svd(Xc, full_matrices=False)
        Vt = Vt_full[:k_fit_max, :]

    return mean_field.astype(np.float32), Vt.astype(np.float32), (H, W)


def _oracle_reconstruct_mask(
    z_full: np.ndarray,
    mean_field: np.ndarray,
    Vt: np.ndarray,
    K: int,
    H: int,
    W: int,
) -> np.ndarray:
    """Reconstruct a material mask from oracle projection with K components."""
    VtK = Vt[:K, :]
    zK = z_full[:K]
    field_rec = (mean_field + zK.dot(VtK)).reshape(H, W)

    if cfg.USE_SDF_REPRESENTATION:
        pred_mat = sdf_to_material(field_rec)
    else:
        pred_mat = to_material_bool(binarize_0_255(field_rec.astype(np.uint8)))

    return postprocess_material(pred_mat)


def oracle_score_curve(
    val_indices: List[int],
    mean_field: np.ndarray,
    Vt: np.ndarray,
    K_grid: List[int],
    timesteps: List[int],
    downsample_factor: int = 2,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Evaluate oracle reconstruction quality for each K on validation slices."""
    iou_scores = {K: [] for K in K_grid}
    dice_scores = {K: [] for K in K_grid}
    bf1_scores = {K: [] for K in K_grid}

    for idx in val_indices:
        for t in timesteps:
            img = load_frame_uint8(idx, t, use_cache=True)
            if img is None:
                continue

            gt_mat = to_material_bool(binarize_0_255(img))
            field_true = get_autotune_field(idx, t, downsample_factor, use_cache=use_cache)
            if field_true is None:
                continue

            gt_mat_ds = _downsample2d(gt_mat.astype(np.uint8), downsample_factor).astype(bool)
            H, W = field_true.shape
            x_centered = field_true.flatten().astype(np.float32) - mean_field
            z_full = x_centered.dot(Vt.T)

            for K in K_grid:
                pred_mat = _oracle_reconstruct_mask(z_full, mean_field, Vt, K, H, W)
                iou_scores[K].append(iou(gt_mat_ds, pred_mat))
                dice_scores[K].append(dice(gt_mat_ds, pred_mat))
                bf1_scores[K].append(boundary_f1(gt_mat_ds, pred_mat, tol=1))

    rows = []
    for K in K_grid:
        rows.append(
            {
                "K": int(K),
                "iou_mean": float(np.mean(iou_scores[K])) if iou_scores[K] else np.nan,
                "dice_mean": float(np.mean(dice_scores[K])) if dice_scores[K] else np.nan,
                "bf1_mean": float(np.mean(bf1_scores[K])) if bf1_scores[K] else np.nan,
            }
        )
    return pd.DataFrame(rows)


def subspace_distance_deg(VtA: np.ndarray, VtB: np.ndarray, K: int) -> Tuple[float, float]:
    """Mean and max principal angle (degrees) between two K-dim subspaces."""
    A = VtA[:K, :]
    B = VtB[:K, :]
    M = A @ B.T
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    angles = np.degrees(np.arccos(s))
    return float(np.mean(angles)), float(np.max(angles))


def auto_tune_pca(train_max_index: int, seed: int = 0) -> Tuple[int, int, int]:
    """
    Grid-search for optimal (NUM_TRAIN_SLICES, K_LLM, K_PCS_MAX).

    Returns (tuned_n_train, tuned_k_llm, tuned_k_recon).
    """
    set_seed(seed)
    all_train_ids = list(range(train_max_index))
    random.shuffle(all_train_ids)

    val_ids = all_train_ids[: cfg.TUNE_VAL_SLICES]
    pool_ids = all_train_ids[cfg.TUNE_VAL_SLICES :]

    if cfg.AUTOTUNE_PRELOAD:
        need_n = max(cfg.TUNE_TRAIN_GRID)
        need_pool = need_n * cfg.AUTOTUNE_PRELOAD_POOL_MULT
        prefetch_ids = sorted(set(val_ids + pool_ids[: min(need_pool, len(pool_ids))]))
        preload_autotune_fields(prefetch_ids, cfg.TUNE_TIMESTEPS, cfg.TUNE_DOWNSAMPLE)

    results = []
    K_grid = [K for K in cfg.TUNE_K_GRID if K <= cfg.TUNE_K_FIT_MAX]

    for n_train in cfg.TUNE_TRAIN_GRID:
        train_ids = pool_ids[:n_train]
        print(f"\n[AUTO-TUNE] PCA with n_train={n_train}, k_fit_max={cfg.TUNE_K_FIT_MAX}")

        mean_field, Vt, _ = fit_pca_on_indices(
            train_ids,
            k_fit_max=cfg.TUNE_K_FIT_MAX,
            timesteps=cfg.TUNE_TIMESTEPS,
            downsample_factor=cfg.TUNE_DOWNSAMPLE,
            seed=seed,
        )

        df_curve = oracle_score_curve(
            val_ids, mean_field, Vt, K_grid, cfg.TUNE_TIMESTEPS, cfg.TUNE_DOWNSAMPLE
        )

        # Select smallest K meeting quality thresholds
        chosen_K_llm = None
        for _, row in df_curve.sort_values("K").iterrows():
            ok = row["iou_mean"] >= cfg.TUNE_TARGET_IOU and row["dice_mean"] >= cfg.TUNE_TARGET_DICE
            if cfg.TUNE_USE_BF1:
                ok = ok and row["bf1_mean"] >= cfg.TUNE_TARGET_BF1
            if ok:
                chosen_K_llm = int(row["K"])
                break

        # Select K_RECON via diminishing-returns
        df_sorted = df_curve.sort_values("K").reset_index(drop=True)
        chosen_K_recon = None
        for i in range(1, len(df_sorted)):
            K = int(df_sorted.loc[i, "K"])
            if K < cfg.TUNE_RECON_MIN:
                continue
            diou = float(df_sorted.loc[i, "iou_mean"] - df_sorted.loc[i - 1, "iou_mean"])
            ddice = float(df_sorted.loc[i, "dice_mean"] - df_sorted.loc[i - 1, "dice_mean"])
            if diou < 5e-4 and ddice < 5e-4:
                chosen_K_recon = K
                break
        if chosen_K_recon is None:
            chosen_K_recon = int(min(cfg.TUNE_RECON_MAX, max(cfg.TUNE_RECON_MIN, K_grid[-1])))

        # Subspace stability check
        stability_ok = True
        mean_angle = max_angle = np.nan
        if cfg.TUNE_USE_STABILITY:
            alt_ids = pool_ids[n_train : n_train * 2]
            _, Vt2, _ = fit_pca_on_indices(
                alt_ids,
                k_fit_max=cfg.TUNE_K_FIT_MAX,
                timesteps=cfg.TUNE_TIMESTEPS,
                downsample_factor=cfg.TUNE_DOWNSAMPLE,
                seed=seed + 1,
            )
            mean_angle, max_angle = subspace_distance_deg(
                Vt, Vt2, K=min(cfg.TUNE_STABILITY_K, cfg.TUNE_K_FIT_MAX)
            )
            stability_ok = mean_angle <= cfg.TUNE_MAX_MEAN_ANGLE_DEG

        print(df_curve[["K", "iou_mean", "dice_mean", "bf1_mean"]])
        print(
            f"[AUTO-TUNE] K_LLM={chosen_K_llm} | K_RECON={chosen_K_recon} | "
            f"stable={stability_ok} | angle={mean_angle:.2f} deg"
        )

        results.append(
            {
                "n_train": n_train,
                "K_LLM": chosen_K_llm,
                "K_RECON": chosen_K_recon,
                "stable": stability_ok,
                "mean_angle_deg": float(mean_angle) if np.isfinite(mean_angle) else np.nan,
            }
        )

    df_res = pd.DataFrame(results)
    candidates = df_res.dropna(subset=["K_LLM"]).copy()
    if cfg.TUNE_USE_STABILITY:
        candidates = candidates[candidates["stable"] == True]

    if len(candidates) == 0:
        print("[AUTO-TUNE] No candidate met thresholds → using defaults.")
        return cfg.NUM_TRAIN_SLICES, cfg.K_LLM, cfg.K_PCS_MAX

    best = candidates.sort_values(["n_train", "K_LLM"]).iloc[0]
    print(f"[AUTO-TUNE] Selected: {dict(best)}")
    return int(best["n_train"]), int(best["K_LLM"]), int(best["K_RECON"])
