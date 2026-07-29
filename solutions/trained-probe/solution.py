"""Codabench entry point for the trained representation probe.

This is the *only* file the evaluator calls directly. It must expose:

* a ``Problem`` dataclass matching the fields the harness passes in, and
* an ``are_robust(model_id, problems)`` function returning one bool per problem.

All heavy lifting (loading the probe artifact, running the model, scoring the
probes) lives in :mod:`probe_inference` so this file stays a thin adapter
between the Codabench contract and the implementation.
"""

from dataclasses import dataclass

from probe_inference import predict_robustness


@dataclass(frozen=True)
class Problem:
    """A single evaluation case handed to the method by the harness.

    ``original_problem`` is the (possibly perturbed) math problem text.
    ``permutation_type`` lists the perturbation categories applied to it. This
    baseline only uses the problem text, but the field is part of the contract.
    """

    original_problem: str
    permutation_type: list[str]


def are_robust(model_id: str, problems: list[Problem]) -> list[bool]:
    """Predict robustness for a batch while preserving problem order.

    Returns a list of native Python ``bool`` values, one per input problem, in
    the same order as ``problems``. ``True`` means the model is predicted to be
    robust on that problem, ``False`` means not robust.
    """
    # The probe only consumes the problem text; drop the rest of each Problem.
    problem_texts = [problem.original_problem for problem in problems]
    return predict_robustness(model_id, problem_texts)
