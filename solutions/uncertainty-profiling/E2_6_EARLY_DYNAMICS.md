# E2.6 — Early Dynamics Validation

## 0. Motivation

E2.5 suggests that **global temporal trend** is a reproducible robustness signal rather than an artifact of one grouped-CV split.

Current preliminary evidence from the 64-row pilot:

| Protocol | E1 Official 14D BA | E2A Full Trend 26D BA | Delta |
|---|---:|---:|---:|
| Full grid, seed 42 | 0.6766 | 0.7738 | +0.0972 |
| 10-seed official grouped CV | 0.6577 ± 0.0181 | 0.7355 ± 0.0308 | +0.0778 |
| 5-seed nested grouped CV | 0.6123 ± 0.0313 | 0.6571 ± 0.0487 | +0.0448 |

The E2.5 micro-ablation also found that **E1 + four raw trend slopes (18D)** is nearly as strong as the full 26D trend representation under nested CV.

E2.6 therefore asks a narrower and more scientifically useful question:

> **How early in a reasoning trace does the robustness signal emerge, and can four directional slopes recover most of the full temporal-trend gain?**

This phase should be completed before moving to hidden-state dynamics or requesting large H200 allocations.

---

## 1. Core research questions

### RQ1 — Early detectability

Can robustness be predicted from only the first `B` generated reasoning tokens, where

`B ∈ {128, 256, 512, 1024}`?

Formally, compare

`P(Robust | U_1:B)`

across increasing prefix budgets.

### RQ2 — Value of temporal direction

At each budget, does temporal trend improve over aggregate uncertainty?

Compare:

1. **E1 Official 14D** — aggregate uncertainty only.
2. **E2A Full 26D** — E1 + all 12 temporal trend features.
3. **E2A Slope 18D** — E1 + only four raw temporal slopes.

### RQ3 — Compactness

Can E2A-Slope recover most of the E2A-Full gain?

Define

`Recovery(B) = [BA_slope(B) - BA_E1(B)] / [BA_full(B) - BA_E1(B)]`.

### RQ4 — Early plateau

Does performance saturate before the model completes long-form reasoning?

A strong result would be:

`BA_E2A(256 or 512) ≈ BA_E2A(1024)`.

### RQ5 — Stability under honest validation

Do the above conclusions survive:

- multi-seed grouped CV, and
- nested grouped CV where threshold selection never sees outer validation labels?

---

## 2. Hypotheses

### H1 — Temporal trend is already useful early

For at least one budget `B ≤ 512`:

`BA_full(B) > BA_E1(B)`.

### H2 — Direction is the dominant compact signal

For at least one budget `B ≤ 512`:

`BA_slope(B)` is within `0.01` BA of `BA_full(B)`,

or slope recovers at least `80%` of the full trend gain.

### H3 — Brittleness appears before reasoning completion

At least one budget `B ≤ 512` is within `0.02` BA of the 1024-token E2A-Full result.

### H4 — The effect is split-stable

For the selected early budget, E2A-Full improves over E1 on at least:

- `8/10` official grouped-CV seeds, and
- `4/5` nested grouped-CV seeds.

### H5 — The result scales beyond the 64-row pilot

The direction of the gain remains positive when the sample count is increased to 128 or the largest feasible public subset.

---

## 3. Experimental variables

### 3.1 Token budgets

Primary budgets:

- 128
- 256
- 512
- 1024

Do not add 64 tokens in the first E2.6 run. A 64-token secondary experiment is only useful after the main curve is stable because very short traces make slope estimation noisy.

### 3.2 Feature sets

#### E1 — Official aggregate uncertainty, 14D

The official uncertainty baseline features computed on the prefix only.

#### E2A-Full — 26D

`14D official + 12D trend`

The 12 trend features are, for each of:

- log probability
- entropy
- top-1 probability
- top-2 margin

three statistics:

- slope
- linearity R²
- normalized slope

#### E2A-Slope — 18D

`14D official + 4 raw slopes`

This is the compact directional model motivated by E2.5.

### 3.3 Estimator

Keep the estimator fixed to isolate feature and prefix effects:

- family: ExtraTrees regressor
- `n_estimators = 200`
- `max_depth = None`
- `min_samples_leaf = 4`
- `max_features = 1.0`
- estimator random seed = 42

Do **not** tune a different model family separately for every token budget in the primary E2.6 analysis. That would confound temporal information with model-selection variance.

A full candidate-grid check may be run only after a winning budget/feature set has been selected.

---

## 4. Data and sample staging

### Stage A — 64-row paired pilot

Use the same scale as E2.5 to obtain the first early-dynamics curve.

Purpose:

- verify extraction correctness,
- identify promising budgets,
- estimate effect size,
- detect obvious regressions before spending more GPU time.

### Stage B — 128-row replication

Run only if Stage A shows a coherent signal.

Purpose:

- reduce split variance,
- test whether the selected early budget remains useful,
- improve the credibility of the compute proposal.

### Stage C — largest feasible public replication

Run before making a paper-level claim or before Phase II if compute permits.

Always group by `original_problem`.

Never use ordinary random row splits.

---

## 5. Single-pass prefix extraction

### Principle

Generate once to the maximum budget:

`max_new_tokens = 1024`

and compute all smaller prefix features from the same deterministic greedy trace:

- tokens 1:128
- tokens 1:256
- tokens 1:512
- tokens 1:1024

This is preferred to four separate generation runs because:

1. it reduces GPU cost,
2. all budgets are paired on the exact same reasoning trajectory,
3. greedy decoding guarantees that the shorter prefix is the same prefix that would be observed in an otherwise identical longer generation.

### Invariant

For budget `B`, feature computation must use **only token statistics from positions ≤ B**.

No later-token information may enter the prefix metrics.

### Short generations

If a generation ends before budget `B`, use the complete available trace for that row and record:

- mean effective token count,
- median effective token count,
- fraction of rows that actually reached `B`.

Do **not** drop short generations in the primary analysis because doing so can induce selection bias.

A secondary sensitivity analysis may use the common subset that reaches 1024 tokens, but it must be labeled as secondary.

---

## 6. Validation protocol

### 6.1 Competition-style grouped CV

Run 10 split seeds:

`0 1 2 3 4 5 6 7 8 9`

Use the existing competition-style OOF threshold selection.

Purpose:

- compare with the E2.5 protocol,
- estimate split stability,
- measure paired seed deltas.

### 6.2 Nested grouped CV

Run 5 outer split seeds:

`0 1 2 3 4`

Suggested:

- outer folds: up to 5, limited by available groups per class,
- inner folds: 4 where feasible.

Threshold selection occurs only in inner CV on the outer training split.

This is the primary research-quality estimate.

---

## 7. Metrics

### Primary

- Balanced Accuracy

### Secondary

- Accuracy
- MAE against `absolute_accuracy_decay`
- split-seed standard deviation
- per-seed paired BA delta
- fraction of seeds with positive E2A-vs-E1 gain
- effective generated-token statistics

### Compactness diagnostic

For each budget:

- `full_delta_ba = BA_full - BA_E1`
- `slope_delta_ba = BA_slope - BA_E1`
- `slope_minus_full_ba = BA_slope - BA_full`
- `slope_gain_recovery_ratio`

---

## 8. Primary plots

Produce at least these figures after the run.

### Figure A — Early robustness curve

X-axis: token budget.

Y-axis: mean balanced accuracy.

Curves:

- E1 Official 14D
- E2A Full 26D
- E2A Slope 18D

Error bars: standard deviation across split seeds.

### Figure B — Temporal gain curve

X-axis: token budget.

Y-axis:

`BA_E2A - BA_E1`

Plot both Full and Slope.

### Figure C — Compactness recovery

X-axis: token budget.

Y-axis: `slope_gain_recovery_ratio`.

### Figure D — Effective length diagnostics

For each budget report:

- mean effective tokens,
- fraction reaching the requested budget.

This is needed to rule out the possibility that an apparent plateau is simply caused by many responses ending early.

---

## 9. Decision gates

### Gate A — Early temporal signal

**PASS** if, for at least one `B ≤ 512` under nested CV:

- `mean ΔBA_full ≥ +0.03`, and
- E2A-Full beats E1 on at least `4/5` nested seeds.

### Gate B — Compact slope model

**PASS** if, at the selected early budget, either:

- `BA_slope ≥ BA_full - 0.01`, or
- `slope_gain_recovery_ratio ≥ 0.80`.

### Gate C — Early plateau

**PASS** if the best budget `B ≤ 512` satisfies:

`BA_full(B) ≥ BA_full(1024) - 0.02`.

### Gate D — Replication

**PASS** if the selected trend-vs-E1 gain remains positive at 128 samples or the largest feasible public replication.

---

## 10. Interpretation matrix

### Outcome 1 — A + B + C pass

Strongest result.

Interpretation:

> Robustness-related dynamics emerge early, and a compact directional slope representation captures most of the useful temporal signal.

Next:

- replicate at larger N,
- prepare compute proposal,
- proceed to Representation Dynamics.

### Outcome 2 — A passes, B fails

Early signal exists but requires richer trajectory-shape information.

Next:

- retain E2A-Full 26D,
- inspect which R² / normalized-slope terms are necessary,
- do not force the compact 18D model.

### Outcome 3 — A passes, C fails

Temporal dynamics help, but only after longer reasoning.

Interpretation:

- robustness signal is late-emerging rather than early-emerging.

Next:

- focus Phase II representation analysis around later reasoning positions.

### Outcome 4 — only 1024 is positive

Do not claim early detection.

The temporal-trend hypothesis can still survive, but the paper story changes from early diagnosis to trajectory-level diagnosis.

### Outcome 5 — nested gains disappear

Stop before expensive hidden-state work.

Investigate:

- sample size,
- threshold sensitivity,
- class/group composition,
- feature instability,
- possible overfitting in the pilot.

---

## 11. Commands

### 11.1 Stage A: one-pass 64-row extraction

```bash
python solutions/uncertainty-profiling/scripts/compute_temporal_features.py \
  --feature-model-id deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
  --prefix-budgets 128 256 512 1024 \
  --max-samples 64 \
  --batch-size 1 \
  --output /kaggle/working/aimo_outputs/e2_6_early_64.parquet \
  --overwrite
```

The extractor automatically ensures generation reaches the largest requested prefix budget.

### 11.2 10-seed official grouped CV

```bash
python solutions/uncertainty-profiling/scripts/run_early_exit_validation.py \
  --feature-data-path /kaggle/working/aimo_outputs/e2_6_early_64.parquet \
  --budgets 128 256 512 1024 \
  --protocol official \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --results-path /kaggle/working/aimo_outputs/e2_6_official.json \
  --summary-csv /kaggle/working/aimo_outputs/e2_6_official.csv
```

### 11.3 5-seed nested grouped CV

```bash
python solutions/uncertainty-profiling/scripts/run_early_exit_validation.py \
  --feature-data-path /kaggle/working/aimo_outputs/e2_6_early_64.parquet \
  --budgets 128 256 512 1024 \
  --protocol nested \
  --seeds 0 1 2 3 4 \
  --results-path /kaggle/working/aimo_outputs/e2_6_nested.json \
  --summary-csv /kaggle/working/aimo_outputs/e2_6_nested.csv
```

### 11.4 Stage B replication

If Stage A passes Gate A, repeat extraction with:

```bash
--max-samples 128
```

Do not change the feature definitions or estimator between Stage A and Stage B.

---

## 12. Required outputs

Keep these files:

- `e2_6_early_64.parquet`
- `e2_6_official.json`
- `e2_6_official.csv`
- `e2_6_nested.json`
- `e2_6_nested.csv`

For Stage B, use the same naming convention with `_128`.

Record additionally:

- repository commit SHA,
- model ID,
- dataset/source revision,
- sample count,
- unique group count,
- CUDA/GPU type,
- dependency versions.

---

## 13. Compute strategy

E2.6 remains a Kaggle-scale experiment because:

- only one 8B model is required,
- generation is performed once to 1024 tokens,
- all smaller budgets reuse the same trace,
- CV is CPU-side after the feature cache is produced.

Do not request H200 primarily for E2.6.

The official compute request becomes much stronger after E2.6 if we can state both:

1. temporal trend provides a stable robustness gain, and
2. the gain appears within the first 256–512 reasoning tokens.

---

## 14. Phase-II trigger

Proceed to **Representation Dynamics** only after either:

### Strong trigger

- Gate A passes,
- Gate B passes,
- Gate C passes,
- and the direction replicates at larger N.

### Minimum trigger

- nested E2A gain remains clearly positive at larger N,
- even if the signal emerges only at 1024 tokens.

At that point the next mechanistic question becomes:

> **Does the same early output-level convergence signal correspond to stable latent representation dynamics across layers and model families?**

That is the bridge from E2.6 to Phase II.
