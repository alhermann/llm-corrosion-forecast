"""
Ablation study runner.

Compares pipeline variants (MC ensemble, single-shot, deterministic,
kNN-only) by temporarily overriding global configuration parameters,
running the evaluation loop, and collecting per-step metrics.

Supports checkpoint/resume via pickle so that long runs can survive
interruptions.
"""

import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from requests.exceptions import ConnectionError, ReadTimeout

from . import config as cfg
from .data_loading import load_slice_across_time
from .forecasting import rollout_mc


def run_ablation_variant(
    name: str,
    valid_slices: List[int],
    times_h_all: np.ndarray,
    mean_field: np.ndarray,
    Vt: np.ndarray,
    z_mean: np.ndarray,
    z_std: np.ndarray,
    k_llm: int,
    caps: Optional[Dict],
    meta_df,
    knn_lib: Optional[Dict],
    n_rollouts: int,
    llm_temperature: Optional[float] = None,
    residual_scale: Optional[float] = None,
    use_knn_guide: Optional[bool] = None,
) -> Tuple[List[Dict], float]:
    """
    Run the evaluation loop for one ablation variant.

    Temporarily overrides cfg.LLM_TEMPERATURE, cfg.RESIDUAL_SCALE,
    and cfg.USE_KNN_GUIDE, then restores original values.

    Returns (rows, runtime_seconds).
    """
    orig_temp = cfg.LLM_TEMPERATURE
    orig_rs = cfg.RESIDUAL_SCALE
    orig_knn = cfg.USE_KNN_GUIDE

    if llm_temperature is not None:
        cfg.LLM_TEMPERATURE = float(llm_temperature)
    if residual_scale is not None:
        cfg.RESIDUAL_SCALE = float(residual_scale)
    if use_knn_guide is not None:
        cfg.USE_KNN_GUIDE = bool(use_knn_guide)

    t0 = time.time()
    rows: List[Dict] = []

    try:
        for slice_idx in valid_slices:
            full_seq = load_slice_across_time(slice_idx, use_cache=False)
            if full_seq is None:
                continue

            for start_t in cfg.TEST_START_TIMES:
                recs, _ = rollout_mc(
                    full_seq,
                    times_h_all,
                    start_t,
                    mean_field,
                    Vt,
                    z_mean,
                    z_std,
                    k_llm=k_llm,
                    caps=caps,
                    meta_df=meta_df,
                    knn_lib=knn_lib,
                    n_rollouts=n_rollouts,
                    verbose=False,
                )
                for r in recs:
                    r["variant"] = name
                    r["slice"] = slice_idx
                    r["start_t"] = start_t
                rows.extend(recs)
    finally:
        cfg.LLM_TEMPERATURE = orig_temp
        cfg.RESIDUAL_SCALE = orig_rs
        cfg.USE_KNN_GUIDE = orig_knn

    return rows, time.time() - t0


def run_all_ablations(
    valid_slices: List[int],
    times_h_all: np.ndarray,
    mean_field: np.ndarray,
    Vt: np.ndarray,
    z_mean: np.ndarray,
    z_std: np.ndarray,
    k_llm: int,
    caps: Optional[Dict],
    meta_df,
    knn_lib: Optional[Dict],
) -> pd.DataFrame:
    """
    Run the full ablation suite with checkpoint/resume.

    Variants:
        1. MC ensemble (baseline)
        2. Single-shot (1 rollout)
        3. Deterministic (temperature=0, 1 rollout)
        4. kNN-only (residual_scale=0, no LLM calls)
    """
    base_T = cfg.LLM_TEMPERATURE
    base_RS = cfg.RESIDUAL_SCALE
    base_knn = cfg.USE_KNN_GUIDE

    variants = [
        dict(
            name=f"MC (N={cfg.NUM_ROLLOUTS}, T={base_T:.2f})",
            n_rollouts=cfg.NUM_ROLLOUTS,
            llm_temperature=base_T,
            residual_scale=base_RS,
            use_knn_guide=base_knn,
        ),
        dict(
            name=f"Single-shot (T={base_T:.2f})",
            n_rollouts=1,
            llm_temperature=base_T,
            residual_scale=base_RS,
            use_knn_guide=base_knn,
        ),
        dict(
            name="Deterministic (T=0.00)",
            n_rollouts=1,
            llm_temperature=0.0,
            residual_scale=base_RS,
            use_knn_guide=base_knn,
        ),
        dict(
            name="kNN-only (no LLM)",
            n_rollouts=1,
            llm_temperature=base_T,
            residual_scale=0.0,
            use_knn_guide=True,
        ),
    ]

    # Resume from checkpoint
    if os.path.exists(cfg.ABLATION_SAVE_PATH):
        df_partial = pd.read_pickle(cfg.ABLATION_SAVE_PATH)
        done = set(df_partial["variant"].unique()) if len(df_partial) else set()
        print(f"[ABLATIONS] Resuming: {len(df_partial)} rows. Done: {sorted(done)}")
    else:
        df_partial = pd.DataFrame()
        done = set()

    all_rows = df_partial.to_dict("records")
    runtimes: Dict[str, float] = {}

    for v in variants:
        if v["name"] in done:
            print(f"  (skip) {v['name']} already done")
            continue

        print(f"  Running: {v['name']}")
        try:
            rows_v, rt = run_ablation_variant(
                name=v["name"],
                valid_slices=valid_slices,
                times_h_all=times_h_all,
                mean_field=mean_field,
                Vt=Vt,
                z_mean=z_mean,
                z_std=z_std,
                k_llm=k_llm,
                caps=caps,
                meta_df=meta_df,
                knn_lib=knn_lib,
                n_rollouts=v["n_rollouts"],
                llm_temperature=v.get("llm_temperature"),
                residual_scale=v.get("residual_scale"),
                use_knn_guide=v.get("use_knn_guide"),
            )
        except RuntimeError as e:
            print(f"[ABLATIONS] {v['name']} failed: {e}")
            break

        runtimes[v["name"]] = rt
        all_rows.extend(rows_v)

        pd.DataFrame(all_rows).to_pickle(cfg.ABLATION_SAVE_PATH)
        done.add(v["name"])
        print(f"[ABLATIONS] Saved checkpoint ({len(all_rows)} rows)")

    if not all_rows:
        print("[ABLATIONS] No rows produced.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    print("\n[ABLATIONS] Runtime summary:")
    for k, v in runtimes.items():
        print(f"  {k}: {v:.1f} s")

    return df
