"""Baseline that classifies every model/problem pair as robust."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Problem:
    original_problem: str
    permutation_type: list[str]


def are_robust(model_id: str, problems: list[Problem]) -> list[bool]:
    del model_id
    return [True for _ in problems]
