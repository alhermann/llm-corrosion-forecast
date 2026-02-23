#!/usr/bin/env python3
"""
Main entry point for the corrosion forecasting pipeline.

Usage:
    export OPENROUTER_API_KEY="sk-or-v1-..."
    python -m corrosion_forecast.main
"""

import random

import numpy as np
import pandas as pd

from . import config as cfg
from .ablation import run_all_ablations
from .autotune import auto_tune_pca
from .data_loading import (
    binarize_0_255,
    get_total_slice_count,
    img_path,
    load_global_metadata,
    load_slice_across_time,
    load_times_from_metadata,
    to_material_bool,
)
from .forecasting import compute_training_caps, rollout_mc
from .knn import build_knn_library
from .metrics import binary_nll, brier_score, expected_calibration_error
from .pca import fit_global_pca, set_seed
from .plotting import (
    plot_ablation_comparison,
    plot_delta_over_baseline,
    plot_gallery,
    plot_metric_vs_horizon,
    plot_reliability_diagram,
    plot_risk_coverage,
)
from .sdf_utils import sdf_to_material


def main() -> None:
    set_seed(0)

    # ── Dataset discovery ────────────────────────────────────
    TOTAL_SLICES = get_total_slice_count()
    if TOTAL_SLICES == 0:
        raise RuntimeError("No slices found.")

    TRAIN_SPLIT_INDEX = int(TOTAL_SLICES * 0.8)
    print(f"Train: 0–{TRAIN_SPLIT_INDEX - 1}  |  Test: {TRAIN_SPLIT_INDEX}–{TOTAL_SLICES - 1}")

    meta_df = load_global_metadata()

    # ── Auto-tune PCA hyper-parameters ───────────────────────
    if cfg.AUTO_TUNE_PCA:
        n_train, k_llm, k_recon = auto_tune_pca(TRAIN_SPLIT_INDEX, seed=cfg.TUNE_SEED)
        cfg.NUM_TRAIN_SLICES = n_train
        cfg.K_LLM = k_llm
        cfg.K_PCS_MAX = int(max(k_recon, k_llm, cfg.TUNE_RECON_MIN))
        cfg.K_PCS_MAX = int(min(cfg.K_PCS_MAX, cfg.TUNE_RECON_MAX))
        print(f"[FINAL] N_TRAIN={cfg.NUM_TRAIN_SLICES}, K_LLM={cfg.K_LLM}, K_PCS_MAX={cfg.K_PCS_MAX}")

    # ── Fit global PCA ───────────────────────────────────────
    mean_field, Vt, z_mean, z_std = fit_global_pca(
        num_slices=cfg.NUM_TRAIN_SLICES,
        max_index=TRAIN_SPLIT_INDEX,
        k_max=cfg.K_PCS_MAX,
        seed=0,
    )

    times_h_all = load_times_from_metadata()

    # ── Training caps ────────────────────────────────────────
    train_ids = list(range(TRAIN_SPLIT_INDEX))
    caps = None
    if cfg.APPLY_TRAINING_CAPS:
        caps = compute_training_caps(
            train_ids, mean_field, Vt, z_mean, z_std,
            k_llm=cfg.K_LLM, max_slices=cfg.CAPS_MAX_SLICES,
        )

    # ── kNN library ──────────────────────────────────────────
    knn_lib = None
    if cfg.USE_KNN_GUIDE:
        knn_lib = build_knn_library(
            train_ids, mean_field, Vt, z_mean, z_std,
            k_llm=cfg.K_LLM, times_h_all=times_h_all,
            max_slices=cfg.KNN_MAX_SLICES,
        )

    # ── Select test slices ───────────────────────────────────
    valid_slices = []
    attempts = 0
    while len(valid_slices) < cfg.NUM_TEST_SLICES and attempts < 300:
        idx = random.randint(TRAIN_SPLIT_INDEX, TOTAL_SLICES - 1)
        if idx not in valid_slices and load_slice_across_time(idx) is not None:
            valid_slices.append(idx)
        attempts += 1
    print(f"Test slices: {valid_slices}")

    # ── Run forecasts ────────────────────────────────────────
    all_rows = []
    gallery = []

    for slice_idx in valid_slices:
        full_seq = load_slice_across_time(slice_idx, use_cache=False)
        if full_seq is None:
            continue

        for start_t in cfg.TEST_START_TIMES:
            print(f"\nSlice {slice_idx}, start_t={start_t}: {cfg.NUM_ROLLOUTS} rollouts …")

            recs, final_preds = rollout_mc(
                full_seq, times_h_all, start_t,
                mean_field, Vt, z_mean, z_std,
                k_llm=cfg.K_LLM, caps=caps,
                meta_df=meta_df, knn_lib=knn_lib,
                n_rollouts=cfg.NUM_ROLLOUTS,
            )

            for row in recs:
                row["slice"] = slice_idx
                row["start_t"] = start_t
            all_rows.extend(recs)

            gt_final = to_material_bool(binarize_0_255(full_seq[-1]))
            pred_stack = np.stack([p.astype(np.float32) for p in final_preds])
            prob_map = pred_stack.mean(axis=0)

            gallery.append({
                "slice": slice_idx,
                "start_t": start_t,
                "gt_final": gt_final,
                "mean_pred_final": prob_map >= 0.5,
                "prob_map": prob_map,
            })

    # ── Plots ────────────────────────────────────────────────
    if all_rows:
        df = pd.DataFrame(all_rows)
        agg = df.groupby("horizon").agg(
            iou_mean=("iou", "mean"), iou_std=("iou", "std"),
            dice_mean=("dice", "mean"), dice_std=("dice", "std"),
            area_mean=("area_err", "mean"), area_std=("area_err", "std"),
        ).reset_index()

        plot_metric_vs_horizon(agg, "iou", "IoU", "IoU vs forecast horizon")
        plot_metric_vs_horizon(agg, "dice", "Dice", "Dice vs forecast horizon")
        plot_metric_vs_horizon(agg, "area", "Area error (px)", "Area error vs horizon")

    if gallery:
        plot_gallery(gallery)

    # ── Ablation study ───────────────────────────────────────
    df_ab = run_all_ablations(
        valid_slices, times_h_all,
        mean_field, Vt, z_mean, z_std,
        k_llm=cfg.K_LLM, caps=caps,
        meta_df=meta_df, knn_lib=knn_lib,
    )

    if len(df_ab) > 0:
        agg_ab = df_ab.groupby(["variant", "horizon"]).agg(
            iou_mean=("iou", "mean"), iou_std=("iou", "std"),
            dice_mean=("dice", "mean"), dice_std=("dice", "std"),
            area_mean=("area_err", "mean"), area_std=("area_err", "std"),
        ).reset_index()

        plot_ablation_comparison(agg_ab, "iou", "IoU", "Ablation: IoU vs horizon")
        plot_ablation_comparison(agg_ab, "dice", "Dice", "Ablation: Dice vs horizon")
        plot_delta_over_baseline(agg_ab)

    # ── Uncertainty diagnostics ──────────────────────────────
    if gallery:
        P = np.concatenate([g["prob_map"].ravel() for g in gallery])
        Y = np.concatenate([g["gt_final"].astype(np.float32).ravel() for g in gallery])

        brier = brier_score(P, Y)
        nll = binary_nll(P, Y)
        ece = expected_calibration_error(P, Y)

        print(f"\n[Uncertainty]  Brier={brier:.6f}  NLL={nll:.6f}  ECE={ece:.6f}")
        plot_reliability_diagram(P, Y, ece)
        plot_risk_coverage(P, Y)


if __name__ == "__main__":
    main()
