"""
Evaluation metrics for binary mask prediction.

Includes spatial overlap (IoU, Dice), boundary accuracy (Boundary-F1),
and probabilistic calibration metrics (Brier score, NLL, ECE).
"""

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion


# ── Spatial overlap metrics ──────────────────────────────────

def iou(gt: np.ndarray, pr: np.ndarray) -> float:
    """Intersection over Union for boolean masks."""
    inter = np.logical_and(gt, pr).sum()
    union = np.logical_or(gt, pr).sum()
    return float(inter / (union + 1e-9))


def dice(gt: np.ndarray, pr: np.ndarray) -> float:
    """Sorensen–Dice coefficient for boolean masks."""
    inter = np.logical_and(gt, pr).sum()
    s = gt.sum() + pr.sum()
    return float(2 * inter / (s + 1e-9))


def boundary_f1(gt: np.ndarray, pr: np.ndarray, tol: int = 1) -> float:
    """
    Boundary F1-score (BF1).

    Computes precision and recall of the predicted boundary pixels
    with respect to ground truth, using a tolerance of *tol* pixels
    (morphological dilation).
    """
    gt_e = np.logical_xor(gt, binary_erosion(gt))
    pr_e = np.logical_xor(pr, binary_erosion(pr))

    if gt_e.sum() == 0 and pr_e.sum() == 0:
        return 1.0
    if gt_e.sum() == 0 or pr_e.sum() == 0:
        return 0.0

    gt_d = binary_dilation(gt_e, iterations=tol)
    pr_d = binary_dilation(pr_e, iterations=tol)

    tp_p = np.logical_and(pr_e, gt_d).sum()
    tp_g = np.logical_and(gt_e, pr_d).sum()
    prec = tp_p / (pr_e.sum() + 1e-9)
    rec = tp_g / (gt_e.sum() + 1e-9)
    return float(2 * prec * rec / (prec + rec + 1e-9))


# ── Probabilistic calibration metrics ───────────────────────

def brier_score(probs: np.ndarray, y: np.ndarray) -> float:
    """Mean squared error between predicted probabilities and binary labels."""
    return float(np.mean((probs - y) ** 2))


def binary_nll(probs: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> float:
    """Negative log-likelihood for binary predictions."""
    return float(-np.mean(y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps)))


def expected_calibration_error(
    probs: np.ndarray, y: np.ndarray, n_bins: int = 15
) -> float:
    """
    Expected Calibration Error (ECE).

    Bins predictions by confidence and measures the gap between
    mean confidence and empirical accuracy in each bin.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i < n_bins - 1:
            m = (probs >= lo) & (probs < hi)
        else:
            m = (probs >= lo) & (probs <= hi)
        if m.sum() == 0:
            continue
        conf = probs[m].mean()
        acc = y[m].mean()
        ece += m.mean() * abs(acc - conf)
    return float(ece)
