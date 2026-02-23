"""
Visualisation routines for the corrosion forecasting pipeline.

All figures use matplotlib only (no seaborn dependency).
Consistent styling is applied for publication readiness.
"""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── Style defaults ───────────────────────────────────────────
_RC = {
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
}


def _apply_style():
    plt.rcParams.update(_RC)


# ── 1. Corrosion mask time-series ────────────────────────────

def plot_mask_timeseries(
    images: List[np.ndarray],
    times_h: np.ndarray,
    slice_idx: int,
) -> None:
    """Show the raw mask sequence across all time-steps."""
    _apply_style()
    T = len(images)
    fig, axs = plt.subplots(1, T, figsize=(2.2 * T, 2.8))
    for t in range(T):
        axs[t].imshow(images[t], cmap="gray", vmin=0, vmax=255)
        axs[t].axis("off")
        axs[t].set_title(f"t={t}\n{times_h[t]:.2f} h", fontsize=9)
    fig.suptitle(f"Corrosion masks over time (slice {slice_idx})")
    plt.tight_layout()
    plt.show()


# ── 2. Corrosion area vs time ────────────────────────────────

def plot_area_vs_time(
    images: List[np.ndarray],
    times_h: np.ndarray,
    slice_idx: int,
) -> None:
    """Plot the corroded (black-pixel) area as a function of degradation time."""
    _apply_style()
    areas = [int(np.sum(img == 0)) for img in images]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(times_h, areas, marker="o", linewidth=1.5)
    ax.set_xlabel("Degradation time (h)")
    ax.set_ylabel("Material pixel count")
    ax.set_title(f"Material area vs time (slice {slice_idx})")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ── 3. PCA explained variance ───────────────────────────────

def plot_explained_variance(cum_var_ratio: np.ndarray) -> None:
    """Cumulative explained variance vs number of PCs."""
    _apply_style()
    K = len(cum_var_ratio)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(np.arange(1, K + 1), cum_var_ratio, marker="o", linewidth=1.5)
    ax.set_xlabel("Number of principal components (K)")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title("PCA explained variance")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ── 4. PCA reconstruction gallery ───────────────────────────

def plot_pca_reconstructions(
    true_img: np.ndarray,
    recons: List[np.ndarray],
    K_list: List[int],
    mse_list: List[float],
    times_h: np.ndarray,
    t_target: int,
    slice_idx: int,
) -> None:
    """Original frame vs PCA reconstructions at different K."""
    _apply_style()
    n = len(K_list) + 1
    fig, axs = plt.subplots(1, n, figsize=(2.8 * n, 3))
    axs[0].imshow(true_img, cmap="gray", vmin=0, vmax=255)
    axs[0].set_title(f"Ground truth\n({times_h[t_target]:.2f} h)")
    axs[0].axis("off")
    for i, (K, rec, mse) in enumerate(zip(K_list, recons, mse_list), start=1):
        axs[i].imshow(rec, cmap="gray", vmin=0, vmax=255)
        axs[i].set_title(f"K={K}\nMSE={mse:.1f}")
        axs[i].axis("off")
    fig.suptitle(f"PCA reconstruction — slice {slice_idx}, t={t_target}")
    plt.tight_layout()
    plt.show()


# ── 5. Eigen-images ──────────────────────────────────────────

def plot_eigenimages(Vt: np.ndarray, H: int, W: int, n_show: int = 6) -> None:
    """Visualise the first few principal component images."""
    _apply_style()
    n_show = min(n_show, Vt.shape[0])
    fig, axs = plt.subplots(1, n_show, figsize=(2.8 * n_show, 3))
    for i in range(n_show):
        eig = Vt[i].reshape(H, W)
        e_min, e_max = eig.min(), eig.max()
        vis = (eig - e_min) / (e_max - e_min + 1e-9) * 255
        axs[i].imshow(vis.astype(np.uint8), cmap="gray")
        axs[i].set_title(f"PC {i + 1}")
        axs[i].axis("off")
    fig.suptitle("Principal component images (corrosion modes)")
    plt.tight_layout()
    plt.show()


# ── 6. Forecast quality vs horizon ───────────────────────────

def plot_metric_vs_horizon(
    agg: pd.DataFrame,
    metric: str = "iou",
    ylabel: str = "IoU",
    title: str = "Forecast quality vs horizon",
) -> None:
    """Plot a metric (mean +/- std) against forecast horizon."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    x = agg["horizon"].values
    y = agg[f"{metric}_mean"].values
    s = agg[f"{metric}_std"].values
    ax.plot(x, y, marker="o", linewidth=1.5, label=f"{ylabel} mean")
    ax.fill_between(x, y - s, y + s, alpha=0.2, label=r"$\pm 1\sigma$")
    ax.set_xlabel("Forecast horizon (steps)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


# ── 7. Ablation comparison ──────────────────────────────────

def plot_ablation_comparison(
    agg_ab: pd.DataFrame,
    metric: str = "iou",
    ylabel: str = "IoU",
    title: str = "Ablation: IoU vs horizon",
) -> None:
    """Compare multiple ablation variants on the same axes."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(9, 5))
    for variant in agg_ab["variant"].unique():
        sub = agg_ab[agg_ab["variant"] == variant].sort_values("horizon")
        x = sub["horizon"].values
        y = sub[f"{metric}_mean"].values
        s = sub[f"{metric}_std"].values
        ax.plot(x, y, marker="o", label=variant)
        ax.fill_between(x, y - s, y + s, alpha=0.12)
    ax.set_xlabel("Forecast horizon (steps)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


# ── 8. Delta-IoU over kNN baseline ──────────────────────────

def plot_delta_over_baseline(
    agg_ab: pd.DataFrame, baseline_name: str = "kNN-only (no LLM)"
) -> None:
    """Show the added IoU value of each variant relative to kNN-only."""
    _apply_style()
    base = agg_ab[agg_ab["variant"] == baseline_name][["horizon", "iou_mean"]].copy()
    base = base.rename(columns={"iou_mean": "iou_base"})
    merged = agg_ab.merge(base, on="horizon", how="left")
    merged["delta_iou"] = merged["iou_mean"] - merged["iou_base"]

    fig, ax = plt.subplots(figsize=(8, 4))
    for v in merged["variant"].unique():
        if v == baseline_name:
            continue
        sub = merged[merged["variant"] == v].sort_values("horizon")
        ax.plot(sub["horizon"], sub["delta_iou"], marker="o", label=v)
    ax.axhline(0.0, color="k", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel(r"$\Delta$IoU vs kNN-only")
    ax.set_title("Added value over kNN-only baseline")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


# ── 9. Gallery: GT vs prediction + uncertainty ───────────────

def plot_gallery(gallery: List[Dict], max_show: int = 6) -> None:
    """Side-by-side gallery of GT, prediction, XOR diff, and uncertainty map."""
    _apply_style()
    show = gallery[: min(max_show, len(gallery))]
    fig, axs = plt.subplots(len(show), 4, figsize=(10, 2.4 * len(show)))
    if len(show) == 1:
        axs = np.expand_dims(axs, axis=0)

    col_titles = ["GT (final)", "Mean prediction", "XOR difference", "P(material)"]
    for j, t in enumerate(col_titles):
        axs[0, j].set_title(t, fontsize=10)

    for i, g in enumerate(show):
        gt = g["gt_final"]
        pr = g["mean_pred_final"]
        diff = np.logical_xor(gt, pr)
        prob = g["prob_map"]

        axs[i, 0].imshow(gt, cmap="gray")
        axs[i, 1].imshow(pr, cmap="gray")
        axs[i, 2].imshow(diff, cmap="gray")
        im = axs[i, 3].imshow(prob, cmap="viridis", vmin=0, vmax=1)

        axs[i, 0].set_ylabel(f"Slice {g['slice']}\nstart t={g['start_t']}", fontsize=9)
        for j in range(4):
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])

    fig.suptitle("Forecast results: GT vs mean prediction + uncertainty", y=1.01)
    plt.tight_layout()
    plt.show()


# ── 10. Reliability diagram ─────────────────────────────────

def plot_reliability_diagram(
    probs: np.ndarray, y: np.ndarray, ece: float, n_bins: int = 15
) -> None:
    """Calibration / reliability curve."""
    _apply_style()
    bins = np.linspace(0, 1, n_bins + 1)
    accs, confs = [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo) & (probs <= hi)
        if m.sum() == 0:
            continue
        confs.append(float(probs[m].mean()))
        accs.append(float(y[m].mean()))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfect calibration")
    ax.plot(confs, accs, marker="o", label="Model")
    ax.set_xlabel("Mean predicted P(material)")
    ax.set_ylabel("Empirical fraction material")
    ax.set_title(f"Reliability diagram (ECE = {ece:.4f})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


# ── 11. Risk–coverage curve ──────────────────────────────────

def plot_risk_coverage(probs: np.ndarray, y: np.ndarray) -> None:
    """Does uncertainty track prediction error?"""
    _apply_style()
    eps = 1e-6
    ent = -(probs * np.log(probs + eps) + (1 - probs) * np.log(1 - probs + eps))
    hard = (probs >= 0.5).astype(np.float32)
    err = np.abs(hard - y)

    order = np.argsort(ent)
    err_sorted = err[order]

    cover_fracs = np.linspace(0.1, 1.0, 10)
    risks = []
    for cf in cover_fracs:
        k = max(1, int(cf * len(err_sorted)))
        risks.append(float(err_sorted[:k].mean()))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(cover_fracs, risks, marker="o", linewidth=1.5)
    ax.set_xlabel("Coverage (fraction of lowest-uncertainty pixels)")
    ax.set_ylabel("Mean pixel error")
    ax.set_title("Risk vs coverage (uncertainty utility)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
