# AIMO Interpretability Challenge

This guide explains how to develop and submit a robustness method. It uses the
repository's **trained representation probe** as a complete, working example.

## Contents

- [What your method does](#what-your-method-does)
- [Prerequisites](#prerequisites)
- [Quick start](#quick-start)
- [Which data is available?](#which-data-is-available)
- [Example method: trained representation probe](#example-method-trained-representation-probe)
- [Runtime and available libraries](#runtime-and-available-libraries)
- [What to submit](#what-to-submit)
- [Check the submission locally](#check-the-submission-locally)
- [How to submit](#how-to-submit)
- [Repository layout](#repository-layout)

## What your method does

For each model and batch of mathematical problems, your method predicts whether
the model's behavior is robust:

```python
are_robust(model_id: str, problems: list[Problem]) -> list[bool]
```

The returned list must:

- contain exactly one prediction per input problem;
- preserve the input order;
- contain real Python `bool` values, not integers or NumPy boolean scalars;
- use `True` for robust and `False` for not robust.

Your method can load the specified language model (`model_id`) through Hugging
Face, run inference on the supplied `problems`, and predict whether the model is
robust on each problem. You may use or extend any available Hugging Face
functionality and combine signals from the forward pass to produce the
robustness predictions.

Model weights are available through the evaluation worker's offline Hugging
Face cache. You do not have access to other models.

## Prerequisites

You need:

- Python 3.9+ and [`uv`](https://docs.astral.sh/uv/) for the local tooling;
- `git` to clone the repository;
- Docker, if you want to reproduce the offline evaluation container locally
  (recommended before submitting).

Clone the repository and install dependencies:

```bash
git clone git@github.com:aimo-interp/codabench.git
cd codabench
uv sync
```

## Quick start

The fastest path from a fresh clone to a scored local run:

```bash
# 1. Import the public validation sample into data/val-sample/.
uv run scripts/import_hf_dataset.py

# 2. Score the example method on that sample.
uv run scripts/run_local.py solutions/trained-probe \
  --input-dir data/val-sample/input \
  --reference-dir data/val-sample/reference
```

Without a trained probe artifact, the example predicts `False` for every case
(a valid but trivial baseline). The rest of this guide explains how to plug in a
real artifact, package a submission, and validate it. To write your own method,
copy `solutions/trained-probe/` and replace the implementation behind
`are_robust`.

## Which data is available?

### Public development data

The public validation sample is
[`aimo-interp/val-sample`](https://huggingface.co/datasets/aimo-interp/val-sample).
It contains examples with:

- `model_id`: the model whose robustness is being evaluated;
- `original_problem`: the mathematical problem text;
- `permutation_type`: the relevant perturbation categories;
- `model_is_robust`: the public development label.

Import it into this repository with:

```bash
uv run scripts/import_hf_dataset.py
```

This creates:

```text
data/val-sample/
├── input/
│   └── cases.jsonl
├── reference/
│   └── labels.jsonl
└── metadata.json
```

The public labels are for development only. The final evaluation uses private
problems and labels.

### Data available during evaluation

For each call, the trained probe receives a `model_id` and a list of `Problem`
objects:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Problem:
    original_problem: str
    permutation_type: list[str]
```

Your submission does not receive case IDs, reference labels, private dataset
files, or network access.

## Example method: trained representation probe

This baseline demonstrates how to compose a solution.

Start with this directory:

```text
trained-probe/
├── solution.py
├── probe_inference.py
└── probe_artifacts/
    └── probe_artifact.pkl
```

The `solution.py` file defines only the Codabench interface:

```python
from dataclasses import dataclass

from probe_inference import predict_robustness


@dataclass(frozen=True)
class Problem:
    original_problem: str
    permutation_type: list[str]


def are_robust(model_id: str, problems: list[Problem]) -> list[bool]:
    problem_texts = [problem.original_problem for problem in problems]
    return predict_robustness(model_id, problem_texts)
```

`probe_inference.py` contains the implementation in three sections:

1. **Artifact loading:** find and validate `probe_artifact.pkl`.
2. **Hidden-state extraction:** load the offline model, format the problem,
   perform a forward pass, and select the final prompt-token representation
   from the model layer.
3. **Probe scoring:** apply a collection of linear probes and average their
   margins.

For one probe, the decision is:

```python
margin = weights @ hidden_state + bias - threshold
prediction = margin >= 0
```

The ensemble averages these margins and returns `True` when the mean is
non-negative. The artifact stores weights, biases, thresholds, the selected
layer, the Hugging Face model ID, and the system prompt. It does not contain
precomputed hidden states.

## Runtime and available libraries

The evaluation container is offline. It does not run `pip install`, `uv sync`,
or a submitted `requirements.txt`.

**You may rely only on the Python standard library and packages provided by
[`Dockerfile.competition`](Dockerfile.competition).** The current runtime
provides:

- PyTorch `2.12.1` with CUDA `12.6`;
- Accelerate `1.13.0`;
- Hugging Face Hub `1.22.0`;
- joblib `1.5.3`;
- pandas `3.0.3`;
- safetensors `0.8.0`;
- scikit-learn `1.8.0`;
- tokenizers `0.23.0`;
- Transformers `5.13.0`.

Important consequences:

- A package installed only on your laptop or cluster login environment will not
  exist during evaluation.
- Adding `requirements.txt` to the ZIP does not install anything.
- Artifacts serialized with scikit-learn or joblib should be trained with the
  runtime versions above.
- Model and tokenizer loading must work offline, so rely only on the model ID
  passed to `are_robust`. Do not call external APIs or download from Hugging
  Face during evaluation.
- You may include your own Python source files and pretrained data artifacts in
  the ZIP, such as pretrained probes, provided they comply with the competition
  rules.

## What to submit

Submit one ZIP archive containing:

- `solution.py` at the archive root;
- any helper modules or Python packages imported by `solution.py`;
- all pretrained artifacts needed, such as the trained probes;
- optional configuration files.

Do not submit:

- training datasets or private labels;
- model weights already supplied by the evaluation worker;
- virtual environments, caches, `__pycache__`, or installed packages;
- credentials, API keys, or code that requires network access.

Correct ZIP layout, for the probing example:

```text
trained-probe.zip
├── solution.py
├── probe_inference.py
└── probe_artifacts/
    └── probe_artifact.pkl
```

Incorrect ZIP layout:

```text
trained-probe.zip
└── trained-probe/
    └── solution.py
```

Build the archive from inside the method directory:

```bash
cd trained-probe
zip -r ../trained-probe.zip solution.py probe_inference.py probe_artifacts
cd ..
```

Verify that `solution.py` is at the root:

```bash
unzip -l trained-probe.zip
```

## Check the submission locally

The repository already includes the method under `solutions/trained-probe/`.
Copy the exported artifact into it:

```text
solutions/
└── trained-probe/
    ├── solution.py
    ├── probe_inference.py
    └── probe_artifacts/
        └── probe_artifact.pkl
```

```bash
mkdir -p solutions/trained-probe/probe_artifacts
cp /path/to/probe_artifact.pkl \
  solutions/trained-probe/probe_artifacts/probe_artifact.pkl
```

Run the public validation sample:

```bash
uv run scripts/run_local.py solutions/trained-probe \
  --input-dir data/val-sample/input \
  --reference-dir data/val-sample/reference
```

Expected output has this shape:

```json
{
  "accuracy": 0.7,
  "coverage": 1.0,
  "invalid_predictions": 0
}
```

The accuracy shown above is only an example. Before submission, require:

- `coverage` equal to `1.0`;
- `invalid_predictions` equal to `0`;
- deterministic predictions across repeated runs;
- successful execution in the competition Docker image, not only in a local
  virtual environment.

Build all repository artifacts with:

```bash
uv run scripts/build.py all \
  --input-dir data/val-sample/input \
  --reference-dir data/val-sample/reference
```

This creates a solution ZIP. For the probing example, the output is
`dist/solution-trained-probe.zip`.

## How to submit

1. Sign in to the AIMO competition on Codabench.
2. Open the active submission phase.
3. Upload `trained-probe.zip` or the generated
   `dist/solution-trained-probe.zip`.
4. Wait for ingestion and scoring to finish.
5. Check `coverage` and `invalid_predictions` before interpreting accuracy.

An invalid prediction usually means the method raised an exception, returned a
non-boolean value, or returned the wrong number of predictions. Detailed
participant exceptions may be hidden by the evaluator, so reproduce failures
locally with the same Docker image.


### Repository layout

```text
components/             standalone Codabench ingestion and scoring programs
competition/            competition manifest, pages, and source assets
config/                 local-to-Codabench resource mapping template
data/val-sample/        generated public validation import (ignored)
docs/                   contract and security decisions
scripts/                deterministic archive builders and local runner
solutions/              auto-discovered baseline submissions
tests/                  end-to-end tests
```
