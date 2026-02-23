"""
Signed Distance Field (SDF) utilities and mask post-processing.

An SDF assigns each pixel a signed distance to the material boundary:
    - negative inside the material
    - positive outside
The zero level-set is the boundary itself.
"""

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    binary_fill_holes,
    distance_transform_edt,
    gaussian_filter,
    label,
)

from . import config as cfg


# ── SDF ↔ material ───────────────────────────────────────────

def material_to_sdf(material_bool: np.ndarray) -> np.ndarray:
    """
    Convert a boolean material mask to an SDF (float32).

    Convention: SDF < 0 inside material, SDF > 0 outside.
    """
    inside = distance_transform_edt(material_bool)
    outside = distance_transform_edt(~material_bool)
    sdf = outside - inside
    return sdf.astype(np.float32)


def sdf_to_material(sdf: np.ndarray) -> np.ndarray:
    """Recover a boolean material mask from an SDF (zero level-set)."""
    return sdf <= 0


# ── Morphological post-processing ────────────────────────────

def keep_largest_component(mask_bool: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component."""
    lab, n = label(mask_bool)
    if n <= 1:
        return mask_bool
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    keep = np.argmax(counts)
    return lab == keep


def remove_small_components(
    mask_bool: np.ndarray, min_pixels: int = 200
) -> np.ndarray:
    """Remove connected components smaller than *min_pixels*."""
    lab, n = label(mask_bool)
    if n == 0:
        return mask_bool
    counts = np.bincount(lab.ravel())
    out = np.zeros_like(mask_bool, dtype=bool)
    for i in range(1, n + 1):
        if counts[i] >= min_pixels:
            out |= lab == i
    return out


def postprocess_material(
    pred_material: np.ndarray,
    prev_material: np.ndarray = None,
) -> np.ndarray:
    """
    Apply a cascade of morphological clean-up operations.

    Steps (each gated by config flags):
        1. Fill interior holes
        2. Remove tiny spurious components
        3. Keep only the largest component
        4. Enforce monotonic material shrinkage (corrosion can only remove material)
    """
    m = pred_material.copy()

    if cfg.FILL_HOLES:
        m = binary_fill_holes(m)
    if cfg.MIN_COMPONENT_PIXELS and cfg.MIN_COMPONENT_PIXELS > 0:
        m = remove_small_components(m, cfg.MIN_COMPONENT_PIXELS)
    if cfg.ENFORCE_SINGLE_COMPONENT:
        m = keep_largest_component(m)
    if cfg.ENFORCE_MONOTONIC_SHRINK and prev_material is not None:
        m = m & prev_material

    return m


def smooth_sdf(sdf: np.ndarray) -> np.ndarray:
    """Optional Gaussian smoothing of an SDF field."""
    if cfg.APPLY_SDF_SMOOTH and cfg.SDF_SMOOTH_SIGMA > 0:
        return gaussian_filter(sdf, sigma=float(cfg.SDF_SMOOTH_SIGMA))
    return sdf
