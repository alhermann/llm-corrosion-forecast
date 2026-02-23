"""
PCA basis fitting via SVD, projection, and reconstruction.

The corrosion mask fields (either raw binary images or SDF
representations) are flattened and arranged into a data matrix.
A global PCA basis is extracted so that each mask can be represented
as a low-dimensional latent vector z.
"""

import glob
import os
import random
from typing import Tuple

import numpy as np

try:
    from sklearn.utils.extmath import randomized_svd

    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False

from . import config as cfg
from .data_loading import binarize_0_255, load_frame_uint8, load_slice_across_time, to_material_bool
from .sdf_utils import material_to_sdf


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _field_from_img(img_uint8: np.ndarray) -> np.ndarray:
    """Convert a uint8 mask image to the chosen field representation."""
    img_bin = binarize_0_255(img_uint8)
    mat = to_material_bool(img_bin)
    if cfg.USE_SDF_REPRESENTATION:
        return material_to_sdf(mat)
    return img_bin.astype(np.float32)


def fit_global_pca(
    num_slices: int,
    max_index: int,
    k_max: int = 60,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit a global PCA basis from *num_slices* training slices.

    Parameters
    ----------
    num_slices : int
        Number of slice IDs to sample for training.
    max_index : int
        Upper bound on slice IDs to consider (exclusive).
    k_max : int
        Number of principal components to retain.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    mean_field : (N_pixels,) float32
    Vt : (k_max, N_pixels) float32  — principal component row-vectors
    z_mean : (k_max,) float32       — mean of projection coefficients
    z_std : (k_max,) float32        — std of projection coefficients
    """
    set_seed(seed)
    print(f"Building Global PCA Basis from {num_slices} training slices (IDs 0–{max_index - 1}) …")

    folder0 = os.path.join(cfg.BASE_DIR, f"{cfg.FOLDER_PREFIX}0")
    ids = [
        int(os.path.splitext(os.path.basename(f))[0])
        for f in glob.glob(os.path.join(folder0, "*.tif"))
        if os.path.splitext(os.path.basename(f))[0].isdigit()
        and int(os.path.splitext(os.path.basename(f))[0]) < max_index
    ]
    if not ids:
        raise RuntimeError("No training slice IDs found.")

    random.shuffle(ids)
    ids = ids[: min(num_slices, len(ids))]

    training_rows = []
    valid = 0
    for idx in ids:
        seq = load_slice_across_time(idx, use_cache=False)
        if seq is None:
            continue
        valid += 1
        for img in seq:
            training_rows.append(_field_from_img(img).flatten())
        if valid % 50 == 0:
            print(f"  loaded {valid}/{len(ids)} slices …", flush=True)

    if not training_rows:
        raise RuntimeError("Could not load any training data for PCA!")

    X = np.array(training_rows, dtype=np.float32)
    mean_field = X.mean(axis=0)
    Xc = X - mean_field

    if _SKLEARN_OK:
        _, _, Vt = randomized_svd(Xc, n_components=k_max, random_state=seed)
    else:
        _, _, Vt_full = np.linalg.svd(Xc, full_matrices=False)
        Vt = Vt_full[:k_max, :]

    Z = Xc.dot(Vt.T)
    z_mean = Z.mean(axis=0).astype(np.float32)
    z_std = (Z.std(axis=0) + 1e-6).astype(np.float32)

    print("Global PCA Basis built.")
    return mean_field.astype(np.float32), Vt.astype(np.float32), z_mean, z_std


def project_field(field_2d: np.ndarray, mean_field: np.ndarray, Vt: np.ndarray) -> np.ndarray:
    """Project a 2-D field onto the PCA basis → latent vector z."""
    x = field_2d.astype(np.float32).flatten()
    return (x - mean_field).dot(Vt.T)


def reconstruct_field(
    z: np.ndarray, mean_field: np.ndarray, Vt: np.ndarray, H: int, W: int
) -> np.ndarray:
    """Reconstruct a 2-D field from a latent vector z."""
    field_flat = mean_field + z.dot(Vt)
    return field_flat.reshape(H, W)
