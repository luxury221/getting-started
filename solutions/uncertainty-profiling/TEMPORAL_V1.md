# Temporal Uncertainty v1

Research extension of the official AIMO Interpretability Challenge uncertainty
baseline.

The upstream baseline compresses token-level generation confidence into 14
global statistics. Temporal Uncertainty v1 keeps those 14 features and adds 52
features that describe *when* uncertainty changes during reasoning.

## Research question

> Does the temporal structure of token-level uncertainty predict mathematical
> reasoning robustness better than aggregate uncertainty alone?

The implementation is deliberately additive: official baseline files are left
unchanged so the fork can continue to sync with `aimo-interp/getting-started`.

## Feature schema

Total dimensionality: **66D = 14 official + 52 temporal**.

| Group | Added dims | Description |
|---|---:|---|
| Trend | 12 | slope, linearity R², normalized slope for log-probability, entropy, top-1 probability, and top-2 margin |
| Segment | 16 | mean value in four equal-count reasoning quarters for each signal |
| Early/late contrast | 4 | late-quarter mean minus early-quarter mean |
| Volatility | 12 | mean absolute step, max absolute step, and step standard deviation |
| Transition / location | 8 | uncertainty peak/trough locations, robust spike/drop counts, and high↔low confidence transitions |

Reasoning position is normalized to `[0, 1]`, which makes trend and location
features comparable across generations with different token lengths.

## Files

- `uncertainty_profile/temporal_metrics.py` — pure NumPy 52D feature computation.
- `uncertainty_profile/temporal_extraction.py` — online collector that reuses the
  official generation pipeline and appends temporal features.
- `scripts/compute_temporal_features.py` — wrapper around the official resumable
  feature-cache script.
- `scripts/run_temporal_ablation.py` — grouped-CV E0/E1/E2 experiment runner.
- `scripts/train_temporal_regressor.py` — full 66D model-selection and artifact
  training wrapper.
- `tests/test_temporal_metrics.py` — dependency-light unit tests for the feature
  schema and trend semantics.

## 1. Run unit tests

From the repository root:

```bash
python -m unittest discover \
  solutions/uncertainty-profiling/tests \
  -p "test_temporal_metrics.py"
```

## 2. Smoke-test feature extraction

Use a locally cached model if available:

```bash
python solutions/uncertainty-profiling/scripts/compute_temporal_features.py \
  --max-samples 8 \
  --max-new-tokens 512 \
  --batch-size 2 \
  --local-files-only
```

The wrapper reuses the official dataset revision, source validation, prompt
deduplication, sharding, resumable partial Parquet cache, and provenance logic.
Its default output filename includes `_temporal_v1` so it does not overwrite the
official 14D cache.

## 3. Full feature extraction

```bash
python solutions/uncertainty-profiling/scripts/compute_temporal_features.py \
  --batch-size 8
```

For multiple GPUs/processes, use the official deterministic sharding arguments:

```bash
--num-shards N --shard-index K
```

## 4. E0/E1/E2 grouped-CV ablation

Fast smoke test:

```bash
python solutions/uncertainty-profiling/scripts/run_temporal_ablation.py \
  --feature-data-path data/uncertainty-profiling/<temporal-cache>.parquet \
  --quick
```

Full model-selection grid:

```bash
python solutions/uncertainty-profiling/scripts/run_temporal_ablation.py \
  --feature-data-path data/uncertainty-profiling/<temporal-cache>.parquet
```

The experiment runner preserves the official `StratifiedGroupKFold` protocol
grouped by `original_problem`.

Experiments:

- **E0** — majority-label sanity baseline.
- **E1** — official 14D uncertainty features.
- **E2A** — official + temporal trend.
- **E2B** — official + segment/early-late contrast.
- **E2C** — official + volatility/transition features.
- **E2D** — full 66D Temporal Uncertainty v1.

Primary selection metric remains balanced accuracy; plain accuracy and regression
MAE remain tie-breakers through the official trainer.

## 5. Train the full 66D artifact

```bash
python solutions/uncertainty-profiling/scripts/train_temporal_regressor.py \
  --feature-data-path data/uncertainty-profiling/<temporal-cache>.parquet
```

The default artifact filename receives `_temporal_v1`, and the default CV report
is `data/uncertainty-profiling/cv-results-temporal-v1.json`, avoiding overwrite
of the official baseline artifact/report.

## Interpretation targets

The first decision gate is intentionally simple:

1. Reproduce E1 near the official baseline under grouped CV.
2. Require E2D to improve balanced accuracy across the same grouped folds.
3. Inspect whether gains are stable across component ablations rather than
   caused by one brittle feature family.
4. Only after E2 is validated, proceed to hidden-state dynamics and
   counterfactual representation stability.

A useful first success criterion is a **+3 to +5 point grouped-CV balanced
accuracy improvement** over E1 without sacrificing plain accuracy materially.
