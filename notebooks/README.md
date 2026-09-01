# Kaggle experiments

## Phase I — Temporal Uncertainty E1/E2

Notebook: `kaggle_temporal_uncertainty_e1_e2.ipynb`

Purpose: validate the first research gate for the AIMO Interpretability project — whether temporal uncertainty dynamics improve robustness prediction over the official 14D aggregate uncertainty baseline under leakage-safe grouped cross-validation.

The first 64-row / 256-token pilot produced a strong signal: the **E2A trend** feature family outperformed the official E1 baseline and also outperformed the full 66D temporal feature set. That changes the next step from "add more temporal features" to "verify the trend signal rigorously."

## Phase E2.5 — Trend Validation

Runbook: `../solutions/uncertainty-profiling/E2_5_TREND_VALIDATION.md`

The E2.5 validation stage contains four checks:

1. **Full candidate-grid replication** on the existing pilot Parquet; no new GPU inference required.
2. **Multi-seed trend micro-ablation** to isolate log-probability / entropy / top-1 / margin and slope / R² / normalized-slope contributions.
3. **Nested grouped CV** so threshold selection never sees outer validation labels.
4. **Single-pass early-exit extraction** for 128 / 256 / 512 / 1024 tokens, followed by multi-seed E1-vs-E2A evaluation.

### Existing-pilot CPU experiments

The existing `temporal_v1_pilot.parquet` can be reused directly for full-grid, multi-seed, and nested micro-ablation runs. These steps do not require rerunning the 8B model.

### Early-exit GPU experiment

The updated `compute_temporal_features.py` accepts:

```bash
--prefix-budgets 128 256 512 1024
```

It generates once to the largest requested budget and derives every prefix from the same deterministic trace, rather than performing four separate model generations.

### Outputs to preserve

Recommended E2.5 outputs under `/kaggle/working/aimo_outputs/`:

- `temporal_full_grid.json`
- `trend_micro_official.json` / `.csv`
- `trend_micro_nested.json` / `.csv`
- `early_exit_64.parquet`
- `early_exit_official.json` / `.csv`
- `early_exit_nested.json` / `.csv`

### Phase-II decision gate

Move to Representation Dynamics when the E2A trend advantage remains substantial across split seeds and preferably nested grouped CV, and when 256–512 token prefixes retain a meaningful share of the gain. At that point the preliminary result is strong enough to support an official compute proposal for hidden-state, cross-model, and counterfactual-stability experiments.

Later 27B/120B, hidden-state dynamics, and counterfactual stability experiments should not rely on Kaggle as the primary compute environment; those are intended for the official compute application after E2.5 validation.
