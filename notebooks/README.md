# Kaggle experiments

## Temporal Uncertainty E1/E2

Notebook: `kaggle_temporal_uncertainty_e1_e2.ipynb`

Purpose: validate the first research gate for the AIMO Interpretability project — whether temporal uncertainty dynamics improve robustness prediction over the official 14D aggregate uncertainty baseline under leakage-safe grouped cross-validation.

### Recommended order

1. Run with `MODE="smoke"` (8 samples, 128 generated tokens) to validate environment, model loading, feature extraction, Parquet output, and unit tests.
2. Run with `MODE="pilot"` (64 samples, 256 generated tokens) for the first meaningful E1/E2 comparison. If the subset does not contain enough groups from both classes, increase the pilot sample count to 96 or 128.
3. Only after a positive pilot result, run the full public feature extraction or move the full replication to official compute.

### Decision rule

Do not interpret smoke-mode CV scores scientifically. For the pilot/full run, compare `E2D_full_temporal_v1` against `E1_official_14d` using balanced accuracy first and plain accuracy as a secondary check.

A useful initial GO criterion is approximately `+0.03` grouped-CV balanced accuracy without a material accuracy drop. A smaller positive gain is considered promising but should be replicated on the full public set before starting Representation Dynamics.

### Kaggle runtime

Prefer 2xT4, L4, or A100. Internet access is needed for GitHub and Hugging Face unless the repository/model/data are attached as Kaggle inputs. The notebook deliberately keeps Kaggle's CUDA-enabled PyTorch installation and installs the remaining competition-compatible dependencies.

### Outputs

Save a Kaggle version so `/kaggle/working/aimo_outputs/` is preserved. Important files:

- `temporal_v1_<mode>.parquet` — official + temporal features.
- `temporal_ablation_<mode>.json` — E0/E1/E2 grouped-CV metrics.

Later 27B/120B, hidden-state dynamics, and counterfactual stability experiments should not rely on Kaggle as the primary compute environment; those are intended for the official compute application after the temporal signal is validated.
