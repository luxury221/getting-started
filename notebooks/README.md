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

## Phase E2.6 — Early Dynamics Validation

Notebook: `AIMO_E2_6_Early_Dynamics_Kaggle.ipynb`

Runbook: `../solutions/uncertainty-profiling/E2_6_EARLY_DYNAMICS.md`

E2.6 tests how early the robustness signal emerges and whether four directional slopes preserve most of the full temporal-trend gain.

The notebook compares three predictors at four generated-token budgets:

- **E1 Official 14D** — aggregate uncertainty baseline.
- **E2A Full 26D** — E1 + all 12 temporal-trend features.
- **E2A Slope 18D** — E1 + four raw trend slopes.

Budgets: `128 / 256 / 512 / 1024` generated tokens.

The feature extractor performs one deterministic generation to the largest budget and derives all prefix features from the same trace. Validation includes 10-seed competition-style grouped CV and 5-seed nested grouped CV.

Automatic gates:

- **Gate A — Early signal:** nested BA gain >= 0.03 at a budget <= 512 with at least 4/5 positive nested seeds.
- **Gate B — Compact slope:** slope BA within 0.01 of full trend, or at least 80% gain recovery.
- **Gate C — Early plateau:** an early budget <= 512 is within 0.02 BA of the 1024-token full-trend result.

Recommended first run: `MAX_SAMPLES = 64`. If Gate A passes, repeat with `MAX_SAMPLES = 128` before moving to Representation Dynamics.

### E2.6 outputs to preserve

The notebook writes `/kaggle/working/aimo_e2_6_results.zip`. Important contents:

- `early_dynamics_64.parquet`
- `early_dynamics_official.json` / `.csv`
- `early_dynamics_nested.json` / `.csv`
- `e2_6_decision.json`

Later 27B/120B, hidden-state dynamics, and counterfactual stability experiments should use official compute after the early-dynamics signal has been replicated.
