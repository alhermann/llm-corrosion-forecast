#!/usr/bin/env python3
"""
Generate all results and figures for the technical report.

Usage:
    export OPENROUTER_API_KEY="sk-or-v1-..."
    python run_all.py
"""

import os
import sys
import random
import json
import time
import glob
import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import imageio.v2 as imageio
from scipy.ndimage import (
    distance_transform_edt, binary_fill_holes, binary_erosion,
    binary_dilation, label, gaussian_filter,
)
from sklearn.utils.extmath import randomized_svd
import requests
from dotenv import load_dotenv

# Load .env file from the project root
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

FIGS = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGS, exist_ok=True)

BASE_DIR = os.path.join(os.path.dirname(__file__), "Corrosion_Masks")
METADATA_CSV = os.path.join(BASE_DIR, "metadata.csv")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
NUM_TIMESTEPS = 10
FOLDER_PREFIX = "Processed_"

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
    "figure.dpi": 200,
})

# ══════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════

def load_frame(slice_idx, t):
    p = os.path.join(BASE_DIR, f"{FOLDER_PREFIX}{t}", f"{slice_idx}.tif")
    if not os.path.isfile(p):
        return None
    img = imageio.imread(p)
    if img.ndim == 3:
        img = img[..., 0]
    return img.astype(np.uint8)

def load_slice_seq(slice_idx):
    imgs = []
    for t in range(NUM_TIMESTEPS):
        im = load_frame(slice_idx, t)
        if im is None:
            return None
        imgs.append(im)
    return imgs

def get_total_slices():
    folder0 = os.path.join(BASE_DIR, f"{FOLDER_PREFIX}0")
    return len(glob.glob(os.path.join(folder0, "*.tif")))

def load_times():
    df = pd.read_csv(METADATA_CSV)
    return df["Degradation Time (h)"].values[:NUM_TIMESTEPS]

def binarize(img, thr=128):
    return np.where(img <= thr, 0, 255).astype(np.uint8)

def to_mat(img_bin):
    return img_bin == 0

def mat_to_sdf(mat):
    inside = distance_transform_edt(mat)
    outside = distance_transform_edt(~mat)
    return (outside - inside).astype(np.float32)

def sdf_to_mat(sdf):
    return sdf <= 0

def field_from_img(img):
    return mat_to_sdf(to_mat(binarize(img)))

# ── metrics ──
def iou(gt, pr):
    i = np.logical_and(gt, pr).sum()
    u = np.logical_or(gt, pr).sum()
    return float(i / (u + 1e-9))

def dice_score(gt, pr):
    i = np.logical_and(gt, pr).sum()
    return float(2 * i / (gt.sum() + pr.sum() + 1e-9))

def bf1(gt, pr, tol=1):
    ge = np.logical_xor(gt, binary_erosion(gt))
    pe = np.logical_xor(pr, binary_erosion(pr))
    if ge.sum() == 0 and pe.sum() == 0: return 1.0
    if ge.sum() == 0 or pe.sum() == 0: return 0.0
    gd = binary_dilation(ge, iterations=tol)
    pd_ = binary_dilation(pe, iterations=tol)
    prec = np.logical_and(pe, gd).sum() / (pe.sum() + 1e-9)
    rec = np.logical_and(ge, pd_).sum() / (ge.sum() + 1e-9)
    return float(2 * prec * rec / (prec + rec + 1e-9))

# ── postprocess ──
def postprocess(pred_mat, prev_mat=None):
    m = binary_fill_holes(pred_mat)
    lab, n = label(m)
    if n > 0:
        counts = np.bincount(lab.ravel()); counts[0] = 0
        m = lab == np.argmax(counts)
    if prev_mat is not None:
        m = m & prev_mat
    return m

print("=" * 60)
print("CORROSION FORECASTING — FULL RESULTS GENERATION")
print("=" * 60)

times_h = load_times()
TOTAL = get_total_slices()
TRAIN_IDX = int(TOTAL * 0.8)
meta_df = pd.read_csv(METADATA_CSV)

print(f"Total slices: {TOTAL}, Train: 0-{TRAIN_IDX-1}, Test: {TRAIN_IDX}-{TOTAL-1}")
print(f"Times (h): {times_h}")

# ══════════════════════════════════════════════════════════════
# FIGURE 1: Corrosion mask time-series (3 representative slices)
# ══════════════════════════════════════════════════════════════
print("\n[FIG 1] Mask time-series ...")
demo_slices = [100, 500, 900]
fig, axs = plt.subplots(3, NUM_TIMESTEPS, figsize=(2.2 * NUM_TIMESTEPS, 7))
for row, si in enumerate(demo_slices):
    seq = load_slice_seq(si)
    for t in range(NUM_TIMESTEPS):
        axs[row, t].imshow(seq[t], cmap="gray", vmin=0, vmax=255)
        axs[row, t].axis("off")
        if row == 0:
            axs[row, t].set_title(f"{times_h[t]:.2f} h", fontsize=9)
    axs[row, 0].set_ylabel(f"Slice {si}", fontsize=10)
fig.suptitle("Corrosion mask evolution across time-steps", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig01_mask_timeseries.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(FIGS, "fig01_mask_timeseries.png"), bbox_inches="tight")
plt.close()
print("  -> fig01_mask_timeseries.pdf")

# ══════════════════════════════════════════════════════════════
# FIGURE 2: Material area vs time (multiple slices)
# ══════════════════════════════════════════════════════════════
print("[FIG 2] Area vs time ...")
fig, ax = plt.subplots(figsize=(6, 3.5))
sample_slices = [50, 200, 400, 600, 800, 1000, 1200]
for si in sample_slices:
    seq = load_slice_seq(si)
    if seq is None: continue
    areas = [np.sum(to_mat(binarize(im))) for im in seq]
    areas_norm = [a / areas[0] * 100 for a in areas]
    ax.plot(times_h, areas_norm, marker=".", linewidth=1, label=f"Slice {si}", alpha=0.7)
ax.set_xlabel("Degradation time (h)")
ax.set_ylabel("Remaining material (%)")
ax.set_title("Material area vs degradation time")
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig02_area_vs_time.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(FIGS, "fig02_area_vs_time.png"), bbox_inches="tight")
plt.close()
print("  -> fig02_area_vs_time.pdf")

# ══════════════════════════════════════════════════════════════
# FIGURE 3: SDF visualisation (binary mask vs SDF)
# ══════════════════════════════════════════════════════════════
print("[FIG 3] SDF visualisation ...")
demo_img = load_frame(500, 0)
demo_bin = binarize(demo_img)
demo_sdf = field_from_img(demo_img)

fig, axs = plt.subplots(1, 3, figsize=(11, 3.5))
axs[0].imshow(demo_img, cmap="gray", vmin=0, vmax=255)
axs[0].set_title("Raw mask (t=0)")
axs[1].imshow(to_mat(demo_bin), cmap="gray")
axs[1].set_title("Binary material mask")
im = axs[2].imshow(demo_sdf, cmap="RdBu_r", vmin=-20, vmax=20)
axs[2].set_title("Signed Distance Field")
plt.colorbar(im, ax=axs[2], shrink=0.8, label="SDF value")
for a in axs: a.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig03_sdf_visualisation.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(FIGS, "fig03_sdf_visualisation.png"), bbox_inches="tight")
plt.close()
print("  -> fig03_sdf_visualisation.pdf")

# ══════════════════════════════════════════════════════════════
# PCA: fit global basis
# ══════════════════════════════════════════════════════════════
print("\n[PCA] Fitting global SDF-PCA basis on 400 training slices ...")
np.random.seed(0); random.seed(0)
K_MAX = 80

train_ids = list(range(TRAIN_IDX))
random.shuffle(train_ids)
train_ids = train_ids[:400]

rows = []
for idx in train_ids:
    seq = load_slice_seq(idx)
    if seq is None: continue
    for im in seq:
        rows.append(field_from_img(im).flatten())
    if len(rows) % 500 == 0:
        print(f"  loaded {len(rows)} fields ...", flush=True)

X = np.array(rows, dtype=np.float32)
mean_field = X.mean(axis=0)
Xc = X - mean_field
print(f"  Data matrix: {X.shape}")
_, S_full, Vt = randomized_svd(Xc, n_components=K_MAX, random_state=0)
Z = Xc.dot(Vt.T)
z_mean = Z.mean(axis=0).astype(np.float32)
z_std = (Z.std(axis=0) + 1e-6).astype(np.float32)
print(f"  PCA basis: Vt={Vt.shape}, singular values range [{S_full[0]:.1f}, {S_full[-1]:.1f}]")

H, W = load_frame(0, 0).shape

# Explained variance
var = S_full ** 2
var_ratio = var / var.sum()
cum_var = np.cumsum(var_ratio)

# ══════════════════════════════════════════════════════════════
# FIGURE 4: Explained variance
# ══════════════════════════════════════════════════════════════
print("[FIG 4] Explained variance ...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
ax1.bar(np.arange(1, K_MAX+1), var_ratio, color="steelblue", alpha=0.7)
ax1.set_xlabel("Principal component")
ax1.set_ylabel("Explained variance ratio")
ax1.set_title("Individual explained variance")
ax1.set_xlim(0.5, K_MAX+0.5)
ax1.grid(True, alpha=0.3)

ax2.plot(np.arange(1, K_MAX+1), cum_var, marker=".", linewidth=1.5, color="steelblue")
ax2.axhline(0.99, color="red", linestyle="--", linewidth=0.8, label="99%")
ax2.axhline(0.999, color="orange", linestyle="--", linewidth=0.8, label="99.9%")
k99 = np.searchsorted(cum_var, 0.99) + 1
k999 = np.searchsorted(cum_var, 0.999) + 1
ax2.axvline(k99, color="red", linestyle=":", linewidth=0.8)
ax2.axvline(k999, color="orange", linestyle=":", linewidth=0.8)
ax2.set_xlabel("Number of principal components (K)")
ax2.set_ylabel("Cumulative explained variance")
ax2.set_title(f"Cumulative variance (99% at K={k99}, 99.9% at K={k999})")
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig04_explained_variance.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(FIGS, "fig04_explained_variance.png"), bbox_inches="tight")
plt.close()
print(f"  -> fig04_explained_variance.pdf  (99% at K={k99}, 99.9% at K={k999})")

# ══════════════════════════════════════════════════════════════
# FIGURE 5: Eigen-images (principal component images)
# ══════════════════════════════════════════════════════════════
print("[FIG 5] Eigen-images ...")
n_eig = 8
fig, axs = plt.subplots(2, 4, figsize=(12, 6))
for i in range(n_eig):
    ax = axs[i // 4, i % 4]
    eig = Vt[i].reshape(H, W)
    im = ax.imshow(eig, cmap="RdBu_r")
    ax.set_title(f"PC {i+1} ({var_ratio[i]*100:.1f}%)", fontsize=10)
    ax.axis("off")
    plt.colorbar(im, ax=ax, shrink=0.7)
fig.suptitle("First 8 principal component images (SDF corrosion modes)", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig05_eigenimages.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(FIGS, "fig05_eigenimages.png"), bbox_inches="tight")
plt.close()
print("  -> fig05_eigenimages.pdf")

# ══════════════════════════════════════════════════════════════
# FIGURE 6: Oracle PCA reconstruction quality vs K
# ══════════════════════════════════════════════════════════════
print("[FIG 6] Oracle reconstruction quality ...")
K_grid = [5, 10, 20, 40, 60, 80]

# Evaluate on 50 validation slices
val_ids = list(range(TRAIN_IDX))
random.shuffle(val_ids)
val_ids = val_ids[:50]

oracle_iou = {K: [] for K in K_grid}
oracle_dice = {K: [] for K in K_grid}
oracle_bf1 = {K: [] for K in K_grid}

for idx in val_ids:
    for t in [0, 3, 6, 9]:
        im = load_frame(idx, t)
        if im is None: continue
        gt_mat = to_mat(binarize(im))
        f = field_from_img(im).flatten().astype(np.float32)
        z_full = (f - mean_field).dot(Vt.T)

        for K in K_grid:
            z_k = z_full[:K]
            rec_field = (mean_field + z_k.dot(Vt[:K, :])).reshape(H, W)
            rec_mat = sdf_to_mat(rec_field)
            rec_mat = postprocess(rec_mat)
            oracle_iou[K].append(iou(gt_mat, rec_mat))
            oracle_dice[K].append(dice_score(gt_mat, rec_mat))
            oracle_bf1[K].append(bf1(gt_mat, rec_mat))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 4))
for ax, data, lbl in [
    (ax1, oracle_iou, "IoU"), (ax2, oracle_dice, "Dice"), (ax3, oracle_bf1, "BF1")
]:
    means = [np.mean(data[K]) for K in K_grid]
    stds = [np.std(data[K]) for K in K_grid]
    ax.errorbar(K_grid, means, yerr=stds, marker="o", capsize=3, linewidth=1.5)
    ax.set_xlabel("Number of PCs (K)")
    ax.set_ylabel(lbl)
    ax.set_title(f"Oracle reconstruction: {lbl}")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0.9)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig06_oracle_reconstruction.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(FIGS, "fig06_oracle_reconstruction.png"), bbox_inches="tight")
plt.close()
print("  -> fig06_oracle_reconstruction.pdf")
for K in K_grid:
    print(f"    K={K:3d}: IoU={np.mean(oracle_iou[K]):.4f} Dice={np.mean(oracle_dice[K]):.4f} BF1={np.mean(oracle_bf1[K]):.4f}")

# ══════════════════════════════════════════════════════════════
# FIGURE 7: Visual reconstruction gallery
# ══════════════════════════════════════════════════════════════
print("[FIG 7] Reconstruction gallery ...")
gallery_slice = 700
gallery_t = 5
gimg = load_frame(gallery_slice, gallery_t)
gt_mat = to_mat(binarize(gimg))
gf = field_from_img(gimg).flatten().astype(np.float32)
gz = (gf - mean_field).dot(Vt.T)

Ks_show = [5, 10, 20, 40, 80]
fig, axs = plt.subplots(1, len(Ks_show) + 1, figsize=(3 * (len(Ks_show) + 1), 3.2))
axs[0].imshow(gt_mat, cmap="gray")
axs[0].set_title("Ground truth", fontsize=10)
axs[0].axis("off")
for i, K in enumerate(Ks_show):
    rec = (mean_field + gz[:K].dot(Vt[:K, :])).reshape(H, W)
    rec_mat = postprocess(sdf_to_mat(rec))
    sc = iou(gt_mat, rec_mat)
    axs[i+1].imshow(rec_mat, cmap="gray")
    axs[i+1].set_title(f"K={K}\nIoU={sc:.4f}", fontsize=10)
    axs[i+1].axis("off")
fig.suptitle(f"Oracle PCA reconstruction (slice {gallery_slice}, t={gallery_t})", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig07_reconstruction_gallery.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(FIGS, "fig07_reconstruction_gallery.png"), bbox_inches="tight")
plt.close()
print("  -> fig07_reconstruction_gallery.pdf")

# ══════════════════════════════════════════════════════════════
# kNN LIBRARY
# ══════════════════════════════════════════════════════════════
print("\n[kNN] Building kNN library ...")
K_LLM = 40

knn_train_ids = list(range(TRAIN_IDX))
random.shuffle(knn_train_ids)
knn_train_ids = knn_train_ids[:100]

tmax = float(times_h[-1])
knn_X, knn_Y = [], []
used = 0
for idx in knn_train_ids:
    seq = load_slice_seq(idx)
    if seq is None: continue
    fields = [field_from_img(im) for im in seq]
    Zp = np.array([(f.flatten().astype(np.float32) - mean_field).dot(Vt.T) for f in fields], dtype=np.float32)
    Zn = (Zp - z_mean) / z_std
    Zt = Zn[:, :K_LLM]
    for t in range(NUM_TIMESTEPS - 1):
        zl = Zt[t].copy()
        vel = (Zt[t] - Zt[t-1]) if t >= 1 else np.zeros_like(zl)
        dt_h = float(times_h[t+1] - times_h[t])
        feat = np.concatenate([zl, vel, np.array([times_h[t]/(tmax+1e-9), dt_h/(tmax+1e-9)], dtype=np.float32)])
        delta = (Zt[t+1] - Zt[t]).astype(np.float32)
        knn_X.append(feat)
        knn_Y.append(delta)
    used += 1
    if used % 20 == 0:
        print(f"  processed {used} slices ...", flush=True)

knn_X = np.stack(knn_X)
knn_Y = np.stack(knn_Y)
print(f"  kNN library: X={knn_X.shape}, Y={knn_Y.shape}", flush=True)

def knn_predict(z_last, vel, t_h, dt_h, k=16):
    q = np.concatenate([z_last, vel, np.array([t_h/(tmax+1e-9), dt_h/(tmax+1e-9)], dtype=np.float32)])
    d2 = np.sum((knn_X - q[None, :])**2, axis=1)
    nn = np.argpartition(d2, kth=min(k, len(d2)-1))[:k]
    d = np.sqrt(d2[nn] + 1e-9)
    w = 1.0 / (d + 1e-6); w /= (w.sum() + 1e-9)
    mu = (knn_Y[nn] * w[:, None]).sum(axis=0)
    var = (w[:, None] * (knn_Y[nn] - mu[None, :])**2).sum(axis=0)
    return mu.astype(np.float32), np.sqrt(np.maximum(var, 1e-9)).astype(np.float32)

# ══════════════════════════════════════════════════════════════
# TRAINING CAPS
# ══════════════════════════════════════════════════════════════
print("[CAPS] Computing training delta caps ...")
cap_ids = list(range(TRAIN_IDX))
random.shuffle(cap_ids)
cap_ids = cap_ids[:80]

cap_deltas = []
for idx in cap_ids:
    seq = load_slice_seq(idx)
    if seq is None: continue
    fields = [field_from_img(im) for im in seq]
    Zp = np.array([(f.flatten().astype(np.float32) - mean_field).dot(Vt.T) for f in fields], dtype=np.float32)
    Zn = (Zp - z_mean) / z_std
    Zt = Zn[:, :K_LLM]
    for t in range(NUM_TIMESTEPS - 1):
        cap_deltas.append(Zt[t+1] - Zt[t])

cap_deltas = np.stack(cap_deltas).astype(np.float32)
mags = np.linalg.norm(cap_deltas, axis=1)
delta_mag_cap = float(np.percentile(mags, 95)) * 1.2
comp_abs_cap = np.percentile(np.abs(cap_deltas), 99, axis=0).astype(np.float32) * 1.2
print(f"  delta_mag_cap={delta_mag_cap:.3f}", flush=True)

# ══════════════════════════════════════════════════════════════
# LLM INTERFACE
# ══════════════════════════════════════════════════════════════

CHAT_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODEL = "openai/gpt-4o-mini"
HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://example.com",
    "X-Title": "Corrosion-Research",
}

_FLOAT_RE = re.compile(r'[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?')

def _json_safe(x):
    if isinstance(x, (np.floating, np.integer)): return x.item()
    if isinstance(x, np.ndarray): return x.astype(float).tolist()
    if isinstance(x, (list, tuple)): return [_json_safe(v) for v in x]
    if isinstance(x, dict): return {k: _json_safe(v) for k, v in x.items()}
    return x

def call_llm(Z_top_hist, dt_h, nonce=None, horizon=1, delta_prior=None, prior_std=None,
             res_norm_cap=None, res_comp_cap=None, temperature=0.6):
    T_obs, D = Z_top_hist.shape
    last = Z_top_hist[-1].astype(np.float32)
    v = (last - Z_top_hist[-2]) if T_obs >= 2 else np.zeros(D, dtype=np.float32)
    a = np.zeros(D, dtype=np.float32)
    if T_obs >= 3:
        a = v - (Z_top_hist[-2] - Z_top_hist[-3])

    obj = _json_safe({
        "D": int(D), "dt_hours": float(dt_h),
        "last_latent": last, "velocity": v, "acceleration": a,
        "rollout_nonce": float(nonce) if nonce else 0.0,
    })
    if delta_prior is not None:
        obj["delta_prior"] = _json_safe(np.asarray(delta_prior, dtype=np.float32))
        obj["prior_std"] = _json_safe(np.asarray(prior_std, dtype=np.float32))
        obj["residual_norm_cap"] = float(res_norm_cap)
        obj["residual_comp_cap"] = float(res_comp_cap)

    prompt = (
        f"Output ONLY JSON with exactly one key predicted_delta.\n"
        f"predicted_delta must be a list of exactly D={D} floats.\n"
        f"Interpret predicted_delta as a RESIDUAL to add to delta_prior.\n"
        f"Keep it small: L2(residual) <= residual_norm_cap and abs(component) <= residual_comp_cap.\n"
        f"No extra keys, text, or markdown.\n"
        f"INPUT={json.dumps(obj, separators=(',', ':'))}"
    )

    temp_eff = temperature * (0.98 ** max(0, horizon - 1))
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "Return ONLY JSON. No prose. No markdown."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temp_eff,
        "max_tokens": 600,
        "response_format": {"type": "json_object"},
    }

    for attempt in range(3):
        try:
            t0 = time.time()
            print(f"      [LLM] call attempt {attempt+1} ...", end="", flush=True)
            resp = requests.post(CHAT_ENDPOINT, headers=HEADERS, json=payload, timeout=30)
            elapsed = time.time() - t0
            if resp.status_code != 200:
                print(f" HTTP {resp.status_code} ({elapsed:.1f}s), retrying", flush=True)
                time.sleep(2)
                continue
            text = resp.json()["choices"][0]["message"]["content"]
            print(f" OK ({elapsed:.1f}s)", flush=True)

            # Parse
            try:
                obj_r = json.loads(text)
            except:
                m = re.search(r'\{.*\}', text.replace('\n', ''), re.DOTALL)
                obj_r = json.loads(m.group()) if m else None

            if obj_r and "predicted_delta" in obj_r:
                vec = np.array(obj_r["predicted_delta"], dtype=np.float32).ravel()
                if vec.size > D: vec = vec[:D]
                elif vec.size < D: vec = np.pad(vec, (0, D - vec.size))
                if np.all(np.isfinite(vec)):
                    return vec
            print(f"      [LLM] parse failed, retrying", flush=True)
            time.sleep(1)
        except Exception as e:
            print(f" Error: {e}, retry {attempt+1}", flush=True)
            time.sleep(2)

    # fallback: return zeros (kNN prior will dominate)
    print("      [LLM] All retries failed, returning zero residual", flush=True)
    return np.zeros(D, dtype=np.float32)

# ══════════════════════════════════════════════════════════════
# FORECASTING
# ══════════════════════════════════════════════════════════════

RESIDUAL_SCALE = 0.7
HORIZON_DAMP = 0.95

def forecast_step(history_sdfs, history_times, use_llm=True, nonce=None, horizon=1, temperature=0.6):
    T_obs = len(history_sdfs)
    Zp = np.array([(s.flatten().astype(np.float32) - mean_field).dot(Vt.T) for s in history_sdfs[-min(T_obs,5):]], dtype=np.float32)
    Zn = (Zp - z_mean) / z_std
    Zt = Zn[:, :K_LLM]

    dt_h = float(history_times[-1] - history_times[-2]) if T_obs >= 2 else 1.0

    # kNN prior
    z_last = Zt[-1]
    vel = (Zt[-1] - Zt[-2]) if len(Zt) >= 2 else np.zeros_like(z_last)
    dp, ps = knn_predict(z_last, vel, float(history_times[-1]), dt_h)

    res_norm_cap = 0.6 * delta_mag_cap
    res_comp_cap = max(0.25, 3.0 * float(np.median(ps)))

    if use_llm:
        residual = call_llm(Zt, dt_h, nonce=nonce, horizon=horizon,
                           delta_prior=dp, prior_std=ps,
                           res_norm_cap=res_norm_cap, res_comp_cap=res_comp_cap,
                           temperature=temperature)
        residual = np.clip(residual, -res_comp_cap, res_comp_cap)
        rmag = np.linalg.norm(residual) + 1e-9
        if rmag > res_norm_cap: residual *= res_norm_cap / rmag
        delta = dp + RESIDUAL_SCALE * residual
    else:
        delta = dp.copy()

    # caps
    delta = np.clip(delta, -comp_abs_cap, comp_abs_cap)
    mag = np.linalg.norm(delta) + 1e-9
    if mag > delta_mag_cap: delta *= delta_mag_cap / mag
    delta *= HORIZON_DAMP ** max(0, horizon - 1)

    # reconstruct
    Zp_last = (Zn[-1] * z_std + z_mean)
    z_next_norm_top = Zn[-1, :K_LLM] + delta
    Zp_next = Zp_last.copy()
    Zp_next[:K_LLM] = z_next_norm_top * z_std[:K_LLM] + z_mean[:K_LLM]

    sdf_pred = (mean_field + Zp_next.dot(Vt)).reshape(H, W).astype(np.float32)
    sdf_pred = gaussian_filter(sdf_pred, sigma=0.8)
    sdf_pred = np.maximum(sdf_pred, history_sdfs[-1])

    return sdf_pred

def run_rollouts(gt_imgs, start_t, n_rollouts=8, use_llm=True, temperature=0.6, verbose=True):
    gt_mat = [to_mat(binarize(im)) for im in gt_imgs]
    gt_sdf = [mat_to_sdf(m).astype(np.float32) for m in gt_mat]

    records, final_preds = [], []
    for r in range(n_rollouts):
        hist_sdf = [gt_sdf[i].copy() for i in range(start_t)]
        hist_t = np.array(times_h[:start_t], dtype=float)

        for t in range(start_t, NUM_TIMESTEPS):
            h = t - start_t + 1
            sdf_next = forecast_step(hist_sdf, hist_t, use_llm=use_llm,
                                     nonce=1000*r+h, horizon=h, temperature=temperature)
            pred_mat = postprocess(sdf_to_mat(sdf_next), prev_mat=sdf_to_mat(hist_sdf[-1]))

            records.append({
                "rollout": r, "t": t, "horizon": h,
                "iou": iou(gt_mat[t], pred_mat),
                "dice": dice_score(gt_mat[t], pred_mat),
                "bf1": bf1(gt_mat[t], pred_mat),
                "area_err": abs(int(pred_mat.sum()) - int(gt_mat[t].sum())),
            })

            if verbose and r == 0:
                rec = records[-1]
                print(f"    t={t} h={h}: IoU={rec['iou']:.4f} Dice={rec['dice']:.4f} BF1={rec['bf1']:.4f}")

            hist_sdf.append(sdf_next)
            hist_t = np.append(hist_t, times_h[t])

        final_preds.append(sdf_to_mat(hist_sdf[-1]))
    return records, final_preds

# ══════════════════════════════════════════════════════════════
# SELECT TEST SLICES
# ══════════════════════════════════════════════════════════════
random.seed(42)
test_slices = []
attempts = 0
while len(test_slices) < 4 and attempts < 500:
    idx = random.randint(TRAIN_IDX, TOTAL - 1)
    if idx not in test_slices:
        seq = load_slice_seq(idx)
        if seq is not None:
            test_slices.append(idx)
    attempts += 1
print(f"\nTest slices: {test_slices}")

START_TIMES = [3, 5]

# ══════════════════════════════════════════════════════════════
# RUN LLM FORECASTING
# ══════════════════════════════════════════════════════════════
if not OPENROUTER_API_KEY:
    print("\n*** OPENROUTER_API_KEY not set — skipping LLM forecasting ***")
    sys.exit(1)

print("\n" + "=" * 60)
print("RUNNING LLM+kNN FORECASTING (MC rollouts)")
print("=" * 60)

N_ROLLOUTS = 4
all_rows_mc = []
gallery = []

for si in test_slices:
    seq = load_slice_seq(si)
    if seq is None: continue
    for st in START_TIMES:
        print(f"\n  Slice {si}, start_t={st}, {N_ROLLOUTS} MC rollouts ...")
        recs, fpreds = run_rollouts(seq, st, n_rollouts=N_ROLLOUTS, use_llm=True, temperature=0.6)
        for r in recs:
            r["slice"] = si; r["start_t"] = st; r["variant"] = "MC LLM+kNN"
        all_rows_mc.extend(recs)

        gt_final = to_mat(binarize(seq[-1]))
        prob = np.stack([p.astype(np.float32) for p in fpreds]).mean(axis=0)
        gallery.append({
            "slice": si, "start_t": st,
            "gt_final": gt_final,
            "mean_pred": prob >= 0.5,
            "prob_map": prob,
        })

print("\n" + "=" * 60)
print("RUNNING kNN-ONLY FORECASTING (no LLM calls)")
print("=" * 60)

all_rows_knn = []
for si in test_slices:
    seq = load_slice_seq(si)
    if seq is None: continue
    for st in START_TIMES:
        print(f"\n  Slice {si}, start_t={st} ...")
        recs, _ = run_rollouts(seq, st, n_rollouts=1, use_llm=False)
        for r in recs:
            r["slice"] = si; r["start_t"] = st; r["variant"] = "kNN-only"
        all_rows_knn.extend(recs)

print("\n" + "=" * 60)
print("RUNNING DETERMINISTIC LLM (T=0, single-shot)")
print("=" * 60)

all_rows_det = []
for si in test_slices:
    seq = load_slice_seq(si)
    if seq is None: continue
    for st in START_TIMES:
        print(f"\n  Slice {si}, start_t={st} ...")
        recs, _ = run_rollouts(seq, st, n_rollouts=1, use_llm=True, temperature=0.0)
        for r in recs:
            r["slice"] = si; r["start_t"] = st; r["variant"] = "Deterministic (T=0)"
        all_rows_det.extend(recs)

# Combine all
df_all = pd.DataFrame(all_rows_mc + all_rows_knn + all_rows_det)
df_all.to_csv(os.path.join(FIGS, "all_results.csv"), index=False)
print(f"\nSaved {len(df_all)} result rows to all_results.csv")

# ══════════════════════════════════════════════════════════════
# FIGURE 8: IoU vs horizon (MC LLM+kNN)
# ══════════════════════════════════════════════════════════════
print("\n[FIG 8] IoU vs horizon ...")
df_mc = df_all[df_all["variant"] == "MC LLM+kNN"]
agg_mc = df_mc.groupby("horizon").agg(
    iou_mean=("iou", "mean"), iou_std=("iou", "std"),
    dice_mean=("dice", "mean"), dice_std=("dice", "std"),
    bf1_mean=("bf1", "mean"), bf1_std=("bf1", "std"),
    area_mean=("area_err", "mean"), area_std=("area_err", "std"),
).reset_index()

fig, ax = plt.subplots(figsize=(7, 4))
x = agg_mc["horizon"].values
for metric, lbl, c in [("iou", "IoU", "steelblue"), ("dice", "Dice", "darkorange"), ("bf1", "BF1", "green")]:
    y = agg_mc[f"{metric}_mean"].values
    s = agg_mc[f"{metric}_std"].values
    ax.plot(x, y, marker="o", label=f"{lbl} (mean)", color=c, linewidth=1.5)
    ax.fill_between(x, y-s, y+s, alpha=0.15, color=c)
ax.set_xlabel("Forecast horizon (steps)")
ax.set_ylabel("Score")
ax.set_title("LLM+kNN forecast quality vs horizon (MC rollouts)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig08_iou_vs_horizon.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(FIGS, "fig08_iou_vs_horizon.png"), bbox_inches="tight")
plt.close()
print("  -> fig08_iou_vs_horizon.pdf")

# ══════════════════════════════════════════════════════════════
# FIGURE 9: Area error vs horizon
# ══════════════════════════════════════════════════════════════
print("[FIG 9] Area error vs horizon ...")
fig, ax = plt.subplots(figsize=(7, 4))
y = agg_mc["area_mean"].values
s = agg_mc["area_std"].values
ax.plot(x, y, marker="o", color="steelblue", linewidth=1.5)
ax.fill_between(x, np.maximum(0, y-s), y+s, alpha=0.2, color="steelblue")
ax.set_xlabel("Forecast horizon (steps)")
ax.set_ylabel("Absolute area error (pixels)")
ax.set_title("Area error vs forecast horizon")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig09_area_error.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(FIGS, "fig09_area_error.png"), bbox_inches="tight")
plt.close()
print("  -> fig09_area_error.pdf")

# ══════════════════════════════════════════════════════════════
# FIGURE 10: Ablation comparison (IoU)
# ══════════════════════════════════════════════════════════════
print("[FIG 10] Ablation comparison ...")
agg_ab = df_all.groupby(["variant", "horizon"]).agg(
    iou_mean=("iou", "mean"), iou_std=("iou", "std"),
    dice_mean=("dice", "mean"), dice_std=("dice", "std"),
).reset_index()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
colors = {"MC LLM+kNN": "steelblue", "kNN-only": "darkorange", "Deterministic (T=0)": "green"}
for v in agg_ab["variant"].unique():
    sub = agg_ab[agg_ab["variant"] == v].sort_values("horizon")
    c = colors.get(v, "gray")
    for ax, m, lbl in [(ax1, "iou", "IoU"), (ax2, "dice", "Dice")]:
        y = sub[f"{m}_mean"].values; s = sub[f"{m}_std"].values
        ax.plot(sub["horizon"], y, marker="o", label=v, color=c, linewidth=1.5)
        ax.fill_between(sub["horizon"], y-s, y+s, alpha=0.12, color=c)

for ax, lbl in [(ax1, "IoU"), (ax2, "Dice")]:
    ax.set_xlabel("Forecast horizon (steps)")
    ax.set_ylabel(lbl)
    ax.set_title(f"Ablation: {lbl} vs horizon")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig10_ablation_comparison.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(FIGS, "fig10_ablation_comparison.png"), bbox_inches="tight")
plt.close()
print("  -> fig10_ablation_comparison.pdf")

# ══════════════════════════════════════════════════════════════
# FIGURE 11: Delta IoU over kNN-only
# ══════════════════════════════════════════════════════════════
print("[FIG 11] Delta IoU over kNN-only ...")
base = agg_ab[agg_ab["variant"] == "kNN-only"][["horizon", "iou_mean"]].rename(columns={"iou_mean": "base"})
merged = agg_ab.merge(base, on="horizon", how="left")
merged["delta_iou"] = merged["iou_mean"] - merged["base"]

fig, ax = plt.subplots(figsize=(7, 4))
for v in merged["variant"].unique():
    if v == "kNN-only": continue
    sub = merged[merged["variant"] == v].sort_values("horizon")
    c = colors.get(v, "gray")
    ax.plot(sub["horizon"], sub["delta_iou"], marker="o", label=v, color=c, linewidth=1.5)
ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
ax.set_xlabel("Forecast horizon (steps)")
ax.set_ylabel(r"$\Delta$IoU vs kNN-only")
ax.set_title("Added value of LLM over kNN-only baseline")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig11_delta_iou.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(FIGS, "fig11_delta_iou.png"), bbox_inches="tight")
plt.close()
print("  -> fig11_delta_iou.pdf")

# ══════════════════════════════════════════════════════════════
# FIGURE 12: GT vs prediction gallery
# ══════════════════════════════════════════════════════════════
print("[FIG 12] Gallery ...")
show = gallery[:min(6, len(gallery))]
fig, axs = plt.subplots(len(show), 4, figsize=(11, 2.6 * len(show)))
if len(show) == 1: axs = np.expand_dims(axs, axis=0)

col_titles = ["GT (final)", "Mean prediction", "XOR difference", "P(material)"]
for j, t in enumerate(col_titles):
    axs[0, j].set_title(t, fontsize=10)

for i, g in enumerate(show):
    gt, pr, prob = g["gt_final"], g["mean_pred"], g["prob_map"]
    diff = np.logical_xor(gt, pr)
    axs[i, 0].imshow(gt, cmap="gray"); axs[i, 0].axis("off")
    axs[i, 1].imshow(pr, cmap="gray"); axs[i, 1].axis("off")
    axs[i, 2].imshow(diff, cmap="gray"); axs[i, 2].axis("off")
    im = axs[i, 3].imshow(prob, cmap="viridis", vmin=0, vmax=1); axs[i, 3].axis("off")
    sc = iou(gt, pr)
    axs[i, 0].set_ylabel(f"S{g['slice']}, t0={g['start_t']}\nIoU={sc:.3f}", fontsize=8)

fig.suptitle("Forecast results: GT vs mean prediction + uncertainty", y=1.01, fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig12_gallery.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(FIGS, "fig12_gallery.png"), bbox_inches="tight")
plt.close()
print("  -> fig12_gallery.pdf")

# ══════════════════════════════════════════════════════════════
# FIGURE 13: Reliability diagram
# ══════════════════════════════════════════════════════════════
print("[FIG 13] Reliability diagram ...")
P = np.concatenate([g["prob_map"].ravel() for g in gallery])
Y = np.concatenate([g["gt_final"].astype(np.float32).ravel() for g in gallery])

eps = 1e-6
brier = float(np.mean((P - Y) ** 2))
nll = float(-np.mean(Y * np.log(P + eps) + (1 - Y) * np.log(1 - P + eps)))

n_bins = 15
bins = np.linspace(0, 1, n_bins + 1)
ece = 0.0
accs, confs, bin_counts = [], [], []
for i in range(n_bins):
    lo, hi = bins[i], bins[i+1]
    m = (P >= lo) & (P < hi) if i < n_bins - 1 else (P >= lo) & (P <= hi)
    if m.sum() == 0: continue
    conf = P[m].mean(); acc = Y[m].mean()
    ece += m.mean() * abs(acc - conf)
    confs.append(conf); accs.append(acc); bin_counts.append(m.sum())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
ax1.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfect calibration")
ax1.plot(confs, accs, marker="o", color="steelblue", label="Model")
ax1.set_xlabel("Mean predicted P(material)")
ax1.set_ylabel("Empirical fraction material")
ax1.set_title(f"Reliability diagram\nBrier={brier:.5f}  NLL={nll:.5f}  ECE={ece:.5f}")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Risk-coverage
ent = -(P * np.log(P + eps) + (1 - P) * np.log(1 - P + eps))
hard = (P >= 0.5).astype(np.float32)
err = np.abs(hard - Y)
order = np.argsort(ent)
err_sorted = err[order]
cover_fracs = np.linspace(0.05, 1.0, 20)
risks = [float(err_sorted[:max(1, int(cf * len(err_sorted)))].mean()) for cf in cover_fracs]

ax2.plot(cover_fracs, risks, marker="o", color="steelblue", linewidth=1.5)
ax2.set_xlabel("Coverage (fraction of lowest-uncertainty pixels)")
ax2.set_ylabel("Mean pixel error")
ax2.set_title("Risk vs coverage (uncertainty utility)")
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "fig13_calibration.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(FIGS, "fig13_calibration.png"), bbox_inches="tight")
plt.close()
print(f"  -> fig13_calibration.pdf  (Brier={brier:.5f}, NLL={nll:.5f}, ECE={ece:.5f})")

# ══════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)

summary = df_all.groupby("variant").agg(
    iou_mean=("iou", "mean"), iou_std=("iou", "std"),
    dice_mean=("dice", "mean"), dice_std=("dice", "std"),
    bf1_mean=("bf1", "mean"), bf1_std=("bf1", "std"),
    area_mean=("area_err", "mean"),
).reset_index()
print(summary.to_string(index=False))
summary.to_csv(os.path.join(FIGS, "summary_table.csv"), index=False)

print("\n[DONE] All figures saved to:", FIGS)
print("Figures produced:")
for f in sorted(glob.glob(os.path.join(FIGS, "fig*.pdf"))):
    print(f"  {os.path.basename(f)}")


# Script runs at module level (no main() wrapper needed)
