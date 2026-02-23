"""
corrosion_forecast — LLM-guided PCA-latent corrosion mask forecasting.

Modules:
    config          Global configuration and hyperparameters
    data_loading    Dataset I/O (TIFF slices, metadata CSV)
    sdf_utils       Signed Distance Field ↔ binary mask conversion + postprocessing
    pca             PCA basis fitting, projection, reconstruction
    metrics         IoU, Dice, Boundary-F1, calibration metrics
    knn             k-Nearest-Neighbour delta predictor in latent space
    llm_interface   OpenRouter / LLM API interaction and JSON parsing
    forecasting     Autoregressive SDF forecasting with MC rollouts
    autotune        Automatic PCA hyper-parameter selection
    ablation        Ablation study runner (MC vs single-shot vs kNN-only …)
    plotting        All visualisation routines
"""
