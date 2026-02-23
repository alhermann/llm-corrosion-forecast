"""
Dataset I/O: loading TIFF corrosion mask slices and CSV metadata.
"""

import os
import glob
from typing import Dict, List, Optional, Tuple

import imageio.v2 as imageio
import numpy as np
import pandas as pd

from . import config as cfg

# ── Frame cache ──────────────────────────────────────────────
_FRAME_CACHE: Dict[Tuple[int, int], np.ndarray] = {}


def clear_frame_cache() -> None:
    """Free all cached frames."""
    _FRAME_CACHE.clear()


def img_path(slice_idx: int, t: int) -> str:
    """Return the absolute path to a single TIFF slice."""
    folder = f"{cfg.FOLDER_PREFIX}{t}"
    return os.path.join(cfg.BASE_DIR, folder, f"{slice_idx}.tif")


def load_frame_uint8(
    slice_idx: int, t: int, use_cache: bool = True
) -> Optional[np.ndarray]:
    """
    Load a single grayscale frame as uint8.

    Returns None if the file does not exist.
    """
    key = (slice_idx, t)
    if use_cache and key in _FRAME_CACHE:
        return _FRAME_CACHE[key]

    p = img_path(slice_idx, t)
    if not os.path.isfile(p):
        return None

    img = imageio.imread(p)
    if img.ndim == 3:
        img = img[..., 0]
    img = img.astype(np.uint8)

    if use_cache:
        _FRAME_CACHE[key] = img
    return img


def load_slice_across_time(
    slice_idx: int,
    use_cache: bool = False,
) -> Optional[List[np.ndarray]]:
    """
    Load one slice index across all NUM_TIMESTEPS time-steps.

    Returns a list of (H, W) uint8 images, or None if any frame is missing.
    """
    images: List[np.ndarray] = []
    for t in range(cfg.NUM_TIMESTEPS):
        img = load_frame_uint8(slice_idx, t, use_cache=use_cache)
        if img is None:
            return None
        images.append(img)
    return images


def get_total_slice_count() -> int:
    """Count the number of .tif files in the first time-step folder."""
    folder0 = os.path.join(cfg.BASE_DIR, f"{cfg.FOLDER_PREFIX}0")
    files = glob.glob(os.path.join(folder0, "*.tif"))
    count = len(files)
    print(f"Found {count} total slices in dataset.")
    return count


def load_times_from_metadata() -> np.ndarray:
    """Return the degradation times (hours) from metadata.csv."""
    df = pd.read_csv(cfg.METADATA_CSV)
    return df["Degradation Time (h)"].values[: cfg.NUM_TIMESTEPS]


def load_global_metadata() -> pd.DataFrame:
    """Return the full metadata DataFrame."""
    return pd.read_csv(cfg.METADATA_CSV)


# ── Binarisation helpers ─────────────────────────────────────

def binarize_0_255(img: np.ndarray, thr: int = 128) -> np.ndarray:
    """Threshold to {0, 255}."""
    return np.where(img <= thr, 0, 255).astype(np.uint8)


def to_material_bool(img_0_255: np.ndarray) -> np.ndarray:
    """Convert a {0,255} mask to a boolean material mask (True = material)."""
    return img_0_255 == 0


def from_material_bool(material_bool: np.ndarray) -> np.ndarray:
    """Inverse of to_material_bool."""
    return np.where(material_bool, 0, 255).astype(np.uint8)


def measure_area_material(material_bool: np.ndarray) -> int:
    """Count material pixels."""
    return int(np.sum(material_bool))
