"""
k-Nearest-Neighbour (kNN) delta predictor in PCA latent space.

A library of (feature, delta) pairs is built from training sequences,
where the feature vector includes the current latent state, velocity,
and normalised time scalars.  At inference the kNN prior provides an
informed starting point for the LLM residual prediction.
"""

import random
from typing import Dict, Tuple

import numpy as np

from . import config as cfg
from .data_loading import load_slice_across_time
from .pca import _field_from_img, project_field, set_seed


def _feat_from_state(
    z_last: np.ndarray,
    vel: np.ndarray,
    t_hours: float,
    dt_hours: float,
    tmax: float,
) -> np.ndarray:
    """Build the feature vector for kNN lookup."""
    t_norm = float(t_hours / (tmax + 1e-9))
    dt_norm = float(dt_hours / (tmax + 1e-9))
    return np.concatenate(
        [z_last, vel, np.array([t_norm, dt_norm], dtype=np.float32)]
    )


def build_knn_library(
    train_ids: list,
    mean_field: np.ndarray,
    Vt: np.ndarray,
    z_mean: np.ndarray,
    z_std: np.ndarray,
    k_llm: int,
    times_h_all: np.ndarray,
    max_slices: int = 300,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """
    Construct the kNN reference library from training slice sequences.

    Returns a dict with keys ``X`` (features), ``Y`` (deltas), ``tmax``.
    """
    set_seed(seed)
    ids = train_ids.copy()
    random.shuffle(ids)
    ids = ids[: min(max_slices, len(ids))]

    X_list, Y_list = [], []
    tmax = float(times_h_all[-1])
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
            z_last = Z_top[t].copy()
            vel = (Z_top[t] - Z_top[t - 1]) if t >= 1 else np.zeros_like(z_last)
            dt_h = float(times_h_all[t + 1] - times_h_all[t])
            feat = _feat_from_state(z_last, vel, times_h_all[t], dt_h, tmax)
            delta = (Z_top[t + 1] - Z_top[t]).astype(np.float32)

            X_list.append(feat.astype(np.float32))
            Y_list.append(delta)

        used += 1
        if used % 50 == 0:
            print(f"[kNN] processed {used}/{len(ids)} slices …", flush=True)

    X = np.stack(X_list)
    Y = np.stack(Y_list)
    print(f"[kNN] library built: X={X.shape}, Y={Y.shape}")
    return {"X": X, "Y": Y, "tmax": tmax}


def knn_predict_delta(
    knn_lib: Dict[str, np.ndarray],
    z_last: np.ndarray,
    vel: np.ndarray,
    t_hours: float,
    dt_hours: float,
    k: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict the next-step delta via inverse-distance weighted kNN.

    Returns (mean_delta, std_delta).
    """
    X = knn_lib["X"]
    Y = knn_lib["Y"]
    tmax = knn_lib["tmax"]

    q = _feat_from_state(z_last, vel, t_hours, dt_hours, tmax).astype(np.float32)

    d2 = np.sum((X - q[None, :]) ** 2, axis=1)
    nn = np.argpartition(d2, kth=min(k, len(d2) - 1))[:k]
    d = np.sqrt(d2[nn] + 1e-9)

    w = 1.0 / (d + 1e-6)
    w = w / (np.sum(w) + 1e-9)

    mu = (Y[nn] * w[:, None]).sum(axis=0)
    var = (w[:, None] * (Y[nn] - mu[None, :]) ** 2).sum(axis=0)
    std = np.sqrt(np.maximum(var, 1e-9))
    return mu.astype(np.float32), std.astype(np.float32)
