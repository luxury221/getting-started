#!/usr/bin/env python3
"""Generate Temporal Uncertainty v1 or single-pass early-exit prefix features."""

from __future__ import annotations

import argparse
import functools
import sys
from pathlib import Path

SOLUTION_ROOT = Path(__file__).resolve().parents[1]
if str(SOLUTION_ROOT) not in sys.path:
    sys.path.insert(0, str(SOLUTION_ROOT))

import compute_uncertainty_features as official  # noqa: E402

from uncertainty_profile.early_exit import (  # noqa: E402
    early_exit_cache_feature_names,
    normalize_budgets,
)
from uncertainty_profile.early_exit_extraction import (  # noqa: E402
    iter_early_exit_generation_feature_batches,
)
from uncertainty_profile.temporal_extraction import (  # noqa: E402
    iter_temporal_generation_feature_batches,
)
from uncertainty_profile.temporal_metrics import ALL_FEATURE_NAMES  # noqa: E402


_original_safe_model_id = official.safe_model_id


def _temporal_safe_model_id(model_id: str) -> str:
    return f"{_original_safe_model_id(model_id)}_temporal_v1"


def _early_exit_safe_model_id(model_id: str, budgets: tuple[int, ...]) -> str:
    budget_tag = "-".join(str(value) for value in budgets)
    return f"{_original_safe_model_id(model_id)}_early_exit_{budget_tag}"


def _extract_prefix_args(argv: list[str]) -> tuple[tuple[int, ...] | None, list[str]]:
    """Parse our additive CLI flag and leave official arguments untouched."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--prefix-budgets",
        nargs="+",
        type=int,
        help=(
            "Compute E1/E2A features for these generated-token prefixes in one "
            "generation, e.g. --prefix-budgets 128 256 512 1024."
        ),
    )
    known, remaining = parser.parse_known_args(argv)
    if known.prefix_budgets is None:
        return None, remaining
    return normalize_budgets(known.prefix_budgets), remaining


def _max_new_tokens_from_args(argv: list[str]) -> int | None:
    for index, value in enumerate(argv):
        if value == "--max-new-tokens" and index + 1 < len(argv):
            return int(argv[index + 1])
        if value.startswith("--max-new-tokens="):
            return int(value.split("=", 1)[1])
    return None


def main() -> None:
    budgets, remaining = _extract_prefix_args(sys.argv[1:])

    if budgets is None:
        official.FEATURE_NAMES = ALL_FEATURE_NAMES
        official.iter_generation_feature_batches = iter_temporal_generation_feature_batches
        official.safe_model_id = _temporal_safe_model_id
        sys.argv = [sys.argv[0], *remaining]
        official.main()
        return

    max_budget = max(budgets)
    requested_max = _max_new_tokens_from_args(remaining)
    if requested_max is None:
        remaining.extend(["--max-new-tokens", str(max_budget)])
    elif requested_max < max_budget:
        raise ValueError(
            "--max-new-tokens must be at least the largest --prefix-budgets "
            f"value; got {requested_max} < {max_budget}"
        )

    official.FEATURE_NAMES = early_exit_cache_feature_names(budgets)
    official.iter_generation_feature_batches = functools.partial(
        iter_early_exit_generation_feature_batches,
        budgets=budgets,
    )
    official.safe_model_id = functools.partial(
        _early_exit_safe_model_id,
        budgets=budgets,
    )
    sys.argv = [sys.argv[0], *remaining]
    official.main()


if __name__ == "__main__":
    main()
