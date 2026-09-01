# E2.5 — Temporal Trend Validation

This phase validates the strongest pilot finding before moving to hidden-state dynamics.

## Why E2.5 exists

Pilot grouped-CV results showed:

- E1 official 14D: balanced accuracy ≈ 0.6488
- E2A trend 26D: balanced accuracy ≈ 0.7738
- E2D full temporal 66D: balanced accuracy ≈ 0.7024

The next question is therefore not "can we add more temporal features?" but:

> Which trend signal is responsible, is the gain stable across grouped-CV seeds, and how early in the reasoning trace does it appear?

## Research questions

1. **Micro-ablation** — log-probability, entropy, top-1 probability, or top-2 margin?
2. **Statistic ablation** — slope, linearity R², or normalized slope?
3. **Sparsity** — can 4D or 12D trend-only features preserve the gain?
4. **Multi-seed stability** — does the gain persist across grouped-CV splits?
5. **Nested validation** — does the gain survive threshold selection that never sees outer validation labels?
6. **Early detection** — what happens at 128 / 256 / 512 / 1024 generated tokens?

---

## A. Reuse the existing pilot Parquet: no GPU required

Run the fixed-estimator core micro-ablation over 10 split seeds:

```bash
python solutions/uncertainty-profiling/scripts/run_trend_micro_ablation.py \
  --feature-data-path /path/to/temporal_v1_pilot.parquet \
  --protocol official \
  --scope core \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --results-path /kaggle/working/aimo_outputs/trend_micro_official.json \
  --summary-csv /kaggle/working/aimo_outputs/trend_micro_official.csv
```

For individual-feature attribution, change `--scope core` to `--scope full`.

### Honest nested grouped CV

Use fewer seeds first because nested CV performs more fits:

```bash
python solutions/uncertainty-profiling/scripts/run_trend_micro_ablation.py \
  --feature-data-path /path/to/temporal_v1_pilot.parquet \
  --protocol nested \
  --scope core \
  --seeds 0 1 2 3 4 \
  --results-path /kaggle/working/aimo_outputs/trend_micro_nested.json \
  --summary-csv /kaggle/working/aimo_outputs/trend_micro_nested.csv
```

The estimator random state is fixed at 42 by default while CV split seeds vary. This isolates split stability rather than mixing split and estimator randomness.

---

## B. Full candidate-grid check on the existing pilot Parquet

The original ablation runner already supports the full model-selection grid. Omit `--quick`:

```bash
python solutions/uncertainty-profiling/scripts/run_temporal_ablation.py \
  --feature-data-path /path/to/temporal_v1_pilot.parquet \
  --results-path /kaggle/working/aimo_outputs/temporal_full_grid.json
```

This checks whether E2A remains strong when model family and hyperparameters are not restricted to the pilot quick grid.

---

## C. Single-pass early-exit extraction: one GPU generation, four budgets

Instead of generating separately for 128, 256, 512, and 1024 tokens, the extractor now generates once to the largest budget and computes prefix features from the same deterministic trace.

```bash
python solutions/uncertainty-profiling/scripts/compute_temporal_features.py \
  --prefix-budgets 128 256 512 1024 \
  --max-samples 64 \
  --batch-size 2 \
  --output /kaggle/working/aimo_outputs/early_exit_64.parquet
```

`--max-new-tokens` is automatically set to the largest prefix budget unless you provide a larger value explicitly.

The cache contains, for every budget:

- prefix official 14D uncertainty
- prefix temporal trend 12D
- effective prefix token count

No full vocabulary logits are stored.

---

## D. Early-exit matrix: multi-seed E1 vs E2A

```bash
python solutions/uncertainty-profiling/scripts/run_early_exit_validation.py \
  --feature-data-path /kaggle/working/aimo_outputs/early_exit_64.parquet \
  --budgets 128 256 512 1024 \
  --protocol official \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --results-path /kaggle/working/aimo_outputs/early_exit_official.json \
  --summary-csv /kaggle/working/aimo_outputs/early_exit_official.csv
```

Then run nested validation:

```bash
python solutions/uncertainty-profiling/scripts/run_early_exit_validation.py \
  --feature-data-path /kaggle/working/aimo_outputs/early_exit_64.parquet \
  --budgets 128 256 512 1024 \
  --protocol nested \
  --seeds 0 1 2 3 4 \
  --results-path /kaggle/working/aimo_outputs/early_exit_nested.json \
  --summary-csv /kaggle/working/aimo_outputs/early_exit_nested.csv
```

## Decision gate for Phase II

Proceed to Representation Dynamics if the following survive multi-seed and preferably nested validation:

1. E2A mean balanced accuracy > E1 by at least **+0.05** on the pilot/full public set.
2. The improvement is not caused by one pathological split.
3. At least one interpretable trend family remains consistently useful.
4. 256–512 token prefixes retain a substantial fraction of the full-prefix gain.

If these conditions hold, the compute proposal can cite preliminary evidence for **early reasoning-dynamics signals** and request resources for cross-model hidden-state experiments.
