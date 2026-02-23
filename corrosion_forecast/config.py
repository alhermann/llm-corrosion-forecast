"""
Global configuration for the corrosion forecasting pipeline.

All tuneable hyperparameters live here so that experiments are
reproducible and easy to modify from a single location.
"""

import os
import numpy as np

# ── Paths ────────────────────────────────────────────────────
BASE_DIR = os.environ.get(
    "CORROSION_BASE_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Corrosion_Masks")),
)
METADATA_CSV = os.path.join(BASE_DIR, "metadata.csv")
FOLDER_PREFIX = "Processed_"
NUM_TIMESTEPS = 10

# ── LLM / OpenRouter ────────────────────────────────────────
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
CHAT_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODEL = "openai/gpt-4o-mini"
LLM_TIMEOUT = 120
LLM_MAX_TOKENS = 600
LLM_TEMPERATURE = 0.6
LLM_MAX_RETRIES = 2
LLM_RETRY_ON_PARSE_FAILURE = True
LLM_RETRY_ON_LENGTH_MISMATCH = True
LLM_ALLOW_LENGTH_REPAIR = True
LLM_ALLOW_TEXT_SALVAGE = True
USE_RESPONSE_FORMAT_JSON = True
HORIZON_TEMP_DECAY = 0.98

# ── SDF representation ───────────────────────────────────────
USE_SDF_REPRESENTATION = True

# ── PCA defaults (may be overridden by auto-tune) ───────────
K_LLM = 40
K_PCS_MAX = 80
NUM_TRAIN_SLICES = 400

# ── Experiment ───────────────────────────────────────────────
NUM_TEST_SLICES = 2
TEST_START_TIMES = [3, 5]
NUM_ROLLOUTS = 8

# ── kNN guidance ─────────────────────────────────────────────
USE_KNN_GUIDE = True
KNN_K = 16
KNN_MAX_SLICES = 300
LLM_PREDICTS_RESIDUAL = True
RESIDUAL_SCALE = 0.7
RESIDUAL_NORM_FRAC = 0.6
DISABLE_VEL_CAP_WHEN_KNN = True
VEL_CAP_MIN_FRAC_OF_TRAIN = 0.25

# ── Stabilisers ──────────────────────────────────────────────
APPLY_TRAINING_CAPS = True
APPLY_HORIZON_DAMP = True
APPLY_VEL_REL_CAP = True
APPLY_SDF_SMOOTH = True
APPLY_MONO_SDF_SHRINK = True
APPLY_MASK_POSTPROC = True
SDF_SMOOTH_SIGMA = 0.8
DELTA_MAG_P95_MULT = 1.2
DELTA_COMP_P99_MULT = 1.2
HORIZON_DAMP = 0.95
VEL_REL_MULT = 6.0
VEL_REL_BIAS = 0.20
CAPS_MAX_SLICES = 250

# ── Post-processing ──────────────────────────────────────────
ENFORCE_SINGLE_COMPONENT = True
FILL_HOLES = True
ENFORCE_MONOTONIC_SHRINK = True
MIN_COMPONENT_PIXELS = 200

# ── Auto-tune PCA ────────────────────────────────────────────
AUTO_TUNE_PCA = True
TUNE_SEED = 0
TUNE_DOWNSAMPLE = 2
TUNE_TIMESTEPS = [0, 3, 6, 9]
TUNE_VAL_SLICES = 25
TUNE_K_FIT_MAX = 160
TUNE_TRAIN_GRID = [400, 800]
TUNE_K_GRID = [40, 60, 80, 120, 160]
TUNE_TARGET_IOU = 0.985
TUNE_TARGET_DICE = 0.99
TUNE_USE_BF1 = True
TUNE_TARGET_BF1 = 0.93
TUNE_USE_STABILITY = True
TUNE_STABILITY_K = 20
TUNE_MAX_MEAN_ANGLE_DEG = 8.0
TUNE_RECON_MIN = 40
TUNE_RECON_MAX = 160
AUTOTUNE_PRELOAD = True
AUTOTUNE_CACHE_DTYPE = np.float32
AUTOTUNE_PRELOAD_POOL_MULT = 2

# ── HTTP robustness (ablation) ───────────────────────────────
HTTP_MAX_RETRIES = 4
HTTP_BACKOFF_BASE = 1.5

# ── Ablation checkpoint ──────────────────────────────────────
ABLATION_SAVE_PATH = "ablation_partial.pkl"
