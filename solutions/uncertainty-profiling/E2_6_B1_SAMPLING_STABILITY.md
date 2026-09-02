# E2.6-B.1 — Sampling Stability Forensics

## 0. Why this stage exists

E2.6 Stage A (64 rows) and Stage B (128 rows) agree that temporal dynamics contain robustness information, but they do **not** agree on the exact token budget where the signal becomes strongest.

The main discrepancy is at 512 tokens:

- Stage A nested CV: Full Trend improved strongly over E1.
- Stage B nested CV: the Full-Trend gain at 512 became small, while 1024 remained positive and the compact Slope representation stayed competitive.

Stage B is also a **sample-size extension**, not a fully independent replication: both stages use the same pinned source revision and `max_samples=N`, so the first 64 rows of the 128-row cache should correspond to Stage A's source slice. Before spending more GPU time, we therefore need to determine whether the Stage A→B change is driven by sample ordering, label/composition shift, or genuine heterogeneity in temporal dynamics.

This stage is CPU-only and reuses `early_dynamics_128.parquet`.

---

## 1. Main research question

> **Is the apparent early-dynamics signal stable to sample composition, or is the optimal token budget heterogeneous across problem subsets?**

We explicitly separate three sources of uncertainty:

1. **CV split variability** — already studied with repeated grouped CV.
2. **Sample-composition variability** — the focus of E2.6-B.1.
3. **Clustered evaluation uncertainty** — addressed with group bootstrap over `original_problem`.

---

## 2. Experiment A — First 64 vs second 64

Use `source_row_index` when available and split the 128-row cache into:

- `first64 = rows[0:64]`
- `second64 = rows[64:128]`

For each half, run nested grouped CV over:

- E1 Official 14D
- E2A Full 26D
- E2A Slope 18D

at budgets:

`128 / 256 / 512 / 1024`.

Primary diagnostic:

`Delta_512 = BA_Full(512) - BA_E1(512)`.

A strong order/composition shift is flagged when:

- the first-64 and second-64 deltas have opposite signs, or
- `|Delta_first64 - Delta_second64| >= 0.05`.

This directly tests whether the strong Stage-A 512 result is concentrated in the original prefix slice.

---

## 3. Experiment B — Composition summary

For `all128`, `first64`, and `second64`, record:

- row count
- unique `original_problem` groups
- robust-group count
- spurious-group count
- robust-group fraction
- mean / std / median of `absolute_accuracy_decay`

The purpose is not to explain causality from labels alone, but to establish whether the two halves differ materially in target/class composition.

Future metadata-conditioned analysis should only be added when the source dataset actually exposes reliable fields for difficulty/problem type; do not invent categories from problem text in this stage.

---

## 4. Experiment C — Repeated stratified group subsampling

Repeatedly draw 64 unique `original_problem` groups from the 128-row cache and retrain/evaluate the fixed ExtraTrees predictor.

Two sampling modes are required.

### C1. Balanced

Sample approximately 32 robust + 32 spurious groups.

Purpose: remove the large class-prevalence difference as a confounder.

### C2. Prevalence-preserving

Sample 64 groups while preserving the observed robust/spurious ratio as closely as possible.

Purpose: estimate the variability expected under the current empirical population.

For each subset, evaluate the primary budgets:

- 512
- 1024

and predictors:

- E1 14D
- Full 26D
- Slope 18D

Use a **fixed CV split seed** for every subsample so the distribution primarily reflects sample composition rather than a mixture of composition and fold randomness.

Recommended first run:

- 50 repeats per sampling mode
- 64 groups per repeat

Final paper-grade diagnostic:

- 100–200 repeats per mode if CPU time is acceptable.

Report:

- median BA
- median `Delta BA`
- 2.5th / 97.5th percentiles
- fraction of subsamples with positive delta

A budget is called **composition-stable** only when both balanced and prevalence-preserving sampling satisfy:

- median Full BA >= 0.52
- median Full-vs-E1 delta >= +0.02
- positive-delta rate >= 0.80

This is deliberately stricter than merely observing a positive mean.

---

## 5. Experiment D — Nested OOF prediction export

The previous E2.6 scripts only preserved aggregate metrics. E2.6-B.1 additionally saves row-level nested-CV outputs for every:

- budget
- predictor
- nested seed

Fields include:

- `source_row_index` when available
- `original_problem`
- true robustness label
- regression target
- OOF score
- predicted robustness class
- outer fold
- fold-specific threshold

This artifact is required for paired statistical analysis and future error inspection.

Output:

`nested_oof_predictions.csv`.

---

## 6. Experiment E — Group bootstrap confidence intervals

For every budget and nested split seed, bootstrap **original_problem groups**, not individual rows.

Each bootstrap sample draws the same number of groups with replacement and applies the same sampled groups to E1, Full, and Slope predictions.

Compute paired distributions for:

- `BA_Full - BA_E1`
- `BA_Slope - BA_E1`
- `BA_Slope - BA_Full`

Default:

- 2,000 group-bootstrap repetitions per budget / nested seed.

Important interpretation:

These intervals are conditional on the corresponding nested-CV split seed. Do not pool all split seeds as if they were independent observations. Instead, inspect whether the bootstrap conclusion is directionally consistent across split seeds.

---

## 7. Decision logic

### Route A — `INDEPENDENT_STRATIFIED_REPLICATION_128`

Choose this route if 512-token Full Trend is composition-stable under both balanced and prevalence-preserving subsampling.

Next GPU experiment:

- draw a randomized / stratified 128-group sample rather than `first N rows`
- keep the source revision pinned
- preserve group isolation
- rerun 512 / 1024, with 256 retained as a secondary slope budget

The scientific claim becomes:

> robustness-related temporal dynamics are reproducibly detectable around the middle reasoning prefix.

### Route B — `REVISE_TO_BY_1024_AND_EXTEND_2048_EOS`

Choose this route if 512 is unstable but 1024 is composition-stable.

Next GPU experiment:

- stop claiming a fixed 512-token emergence point
- extend budgets to `512 / 1024 / 2048 / EOS-or-4096`
- determine whether the signal stabilizes only at later reasoning stages

The scientific claim becomes:

> temporal dynamics become reliably predictive by approximately 1024 generated tokens, while earlier detectability is sample-dependent.

### Route C — `HETEROGENEOUS_DYNAMICS_DIAGNOSIS`

Choose this route if neither 512 nor 1024 is composition-stable.

Do **not** move directly to expensive hidden-state experiments.

Next analysis:

- inspect OOF failures by target decay / reasoning length / available dataset metadata
- test whether optimal budget varies across groups
- formulate conditional or adaptive temporal-stability hypotheses before adding representation features

---

## 8. Recommended command

Use the Stage-B feature cache directly:

```bash
python solutions/uncertainty-profiling/scripts/run_sampling_stability_forensics.py \
  --feature-data-path /path/to/early_dynamics_128.parquet \
  --budgets 128 256 512 1024 \
  --primary-budgets 512 1024 \
  --half-seeds 0 1 2 3 4 \
  --subsample-groups 64 \
  --subsample-repeats 50 \
  --bootstrap-repeats 2000 \
  --output-dir /kaggle/working/e2_6_b1_outputs
```

No GPU is needed.

---

## 9. Expected outputs

- `composition_summary.csv`
- `half_nested_metrics.csv`
- `half_nested_deltas.csv`
- `subsampling_distribution.csv`
- `subsampling_summary.csv`
- `nested_oof_predictions.csv`
- `bootstrap_budget_<B>_seed_<S>.csv`
- `bootstrap_summary.csv`
- `b1_decision.json`

The entire output directory should be preserved as one Kaggle output artifact.

---

## 10. What E2.6-B.1 can and cannot establish

This stage **can** determine whether the Stage-A/Stage-B discrepancy is consistent with sample-composition instability and whether 512 or 1024 survives repeated group resampling.

It **cannot** prove a mechanistic cause for the temporal signal, and it cannot establish independence from the underlying source dataset because all rows still come from the same pinned corpus. A truly independent replication requires a new randomized held-out group sample or another compatible data source.

Only after this diagnostic should the project decide whether to:

1. run an independent stratified replication,
2. extend temporal budgets toward completion, or
3. move to Representation Dynamics.
