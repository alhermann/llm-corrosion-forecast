# LLM-Guided Corrosion Forecasting

**LLM-guided PCA-latent forecasting of biodegradation in Mg-4Ag alloy from in-situ synchrotron nano-CT data.**

This repository implements a hybrid machine-learning pipeline that predicts the temporal evolution of corrosion masks in biodegradable magnesium alloy (Mg-4Ag) wire samples. Binary segmentation masks from synchrotron radiation nano-computed tomography (SRnanoCT) are transformed into Signed Distance Fields, compressed via PCA into a low-dimensional latent space, and forecasted autoregressively using a k-Nearest-Neighbour prior combined with an LLM-based residual correction. Monte Carlo rollouts with stochastic LLM sampling provide calibrated uncertainty estimates.

---

## Pipeline Overview

```
TIFF slices ─→ Binarise ─→ SDF ─→ PCA basis ─→ Latent z
                                                    │
                              ┌─────────────────────┤
                              │                     ▼
                         kNN prior            LLM residual
                              │                     │
                              └──────► δ = δ_prior + α · residual
                                            │
                                    Stabilisers (caps, damping)
                                            │
                                      z_next = z + δ
                                            │
                                Reconstruct SDF ─→ Threshold ─→ Post-process
                                            │
                                    Predicted mask
```

## Key Features

- **SDF representation**: Smooth, continuous encoding of binary masks enables high-quality PCA compression
- **Auto-tuned PCA**: Grid search over training size and component count with oracle reconstruction validation
- **kNN + LLM hybrid**: Physics-informed kNN baseline + LLM residual correction for non-linear dynamics
- **Deterministic stabilisers**: Training-derived magnitude/component caps, horizon damping, velocity-relative cap, monotonic material shrinkage
- **Monte Carlo uncertainty**: Multiple stochastic rollouts yield pixel-wise probability maps
- **Ablation framework**: Systematic comparison of MC ensemble, single-shot, deterministic, and kNN-only variants
- **Calibration diagnostics**: Brier score, NLL, ECE, reliability diagrams, risk-coverage curves

## Plots

The pipeline produces these publication-ready visualisations:

| # | Plot | Purpose |
|---|------|---------|
| 1 | Mask time-series | Raw corrosion mask evolution across all time-steps |
| 2 | Material area vs time | Quantitative corrosion progression curve |
| 3 | PCA explained variance | Cumulative variance captured vs number of components |
| 4 | Eigen-images | First principal component images (corrosion modes) |
| 5 | PCA reconstruction gallery | Ground truth vs reconstructions at different K |
| 6 | IoU vs horizon | Forecast quality degradation with increasing horizon |
| 7 | Dice vs horizon | Complementary overlap metric over forecast steps |
| 8 | Area error vs horizon | Absolute pixel-count error over time |
| 9 | Ablation comparison | Side-by-side IoU/Dice of all pipeline variants |
| 10 | Delta-IoU over kNN-only | Added value of LLM residual over pure kNN |
| 11 | GT vs prediction gallery | Side-by-side: ground truth, mean prediction, XOR diff, uncertainty map |
| 12 | Reliability diagram | Calibration curve with ECE score |
| 13 | Risk-coverage curve | Does uncertainty correlate with prediction error? |

## Repository Structure

```
├── corrosion_forecast/          # Main Python package
│   ├── __init__.py
│   ├── config.py                # All hyperparameters and paths
│   ├── data_loading.py          # TIFF I/O, metadata, binarisation
│   ├── sdf_utils.py             # SDF ↔ material conversion, post-processing
│   ├── pca.py                   # PCA fitting (randomised SVD), projection
│   ├── metrics.py               # IoU, Dice, BF1, Brier, NLL, ECE
│   ├── knn.py                   # kNN library and weighted prediction
│   ├── llm_interface.py         # LLM prompt construction, API calls, parsing
│   ├── forecasting.py           # Single-step forecast + MC rollouts
│   ├── autotune.py              # PCA hyper-parameter grid search
│   ├── ablation.py              # Ablation study runner with checkpointing
│   ├── plotting.py              # All visualisation routines
│   ├── main.py                  # Pipeline orchestrator
│   └── requirements.txt         # Python dependencies
├── report/
│   └── technical_report.tex     # Full LaTeX technical report (26 pages)
│   └── technical_report.pdf     # Compiled PDF
├── .gitignore
└── README.md
```

## Setup

### 1. Install dependencies

```bash
pip install -r corrosion_forecast/requirements.txt
```

### 2. Prepare the dataset

Place the `Corrosion_Masks/` folder (containing `Processed_0/` through `Processed_9/` and `metadata.csv`) in the repository root, or set the path via environment variable:

```bash
export CORROSION_BASE_DIR="/path/to/Corrosion_Masks"
```

### 3. Set the API key

```bash
export OPENROUTER_API_KEY="sk-or-v1-your-key-here"
```

### 4. Run

```bash
python -m corrosion_forecast.main
```

## Experimental Context

The corrosion masks originate from in-situ SRnanoCT experiments on Mg-4Ag biodegradable alloy wires (80 μm diameter) degraded in simulated body fluid at 37 °C. The experimental setup and imaging are described in:

> Reimers, J., Trinh, H.C., et al. "Development of a Bioreactor-Coupled Flow-Cell Setup for 3D In Situ Nanotomography of Mg Alloy Biodegradation." *ACS Applied Materials & Interfaces* **15**, 35600–35610 (2023). [DOI: 10.1021/acsami.3c04054](https://doi.org/10.1021/acsami.3c04054)

The dataset consists of 1372 2D slices across 10 time-steps, with metadata including degradation rate, volume loss, and pitting factors.

## Technical Report

A 26-page LaTeX report is included at `report/technical_report.tex` covering:

- **Part I — Mathematical Theory**: PCA via SVD, SDF representation, kNN delta prediction, LLM-in-the-loop forecasting, deterministic stabilisers, Monte Carlo uncertainty quantification, evaluation metrics, auto-tuning
- **Part II — Code Architecture**: Module-by-module documentation, function signatures, data flow diagram, configuration guide

## License

MIT

## Acknowledgements

This work uses data from the Helmholtz-Zentrum Hereon and experiments at the PETRA III synchrotron (DESY, Hamburg).
