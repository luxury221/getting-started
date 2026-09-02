"""Prefix-budget feature schema for early robustness detection experiments."""

from __future__ import annotations

from collections.abc import Sequence

from uncertainty_profile.config import FEATURE_NAMES as OFFICIAL_FEATURE_NAMES
from uncertainty_profile.temporal_metrics import SIGNAL_NAMES, TREND_FEATURE_NAMES


def normalize_budgets(budgets: Sequence[int]) -> tuple[int, ...]:
    """Return sorted unique positive token budgets."""

    normalized = tuple(sorted({int(value) for value in budgets}))
    if not normalized:
        raise ValueError("at least one early-exit budget is required")
    if normalized[0] < 2:
        raise ValueError("early-exit budgets must be at least 2 tokens")
    return normalized


def prefix_name(budget: int, feature_name: str) -> str:
    """Namespace one feature by its generated-token budget."""

    return f"prefix_{int(budget)}__{feature_name}"


def budget_official_feature_names(budget: int) -> tuple[str, ...]:
    """Return the official 14 uncertainty features for one prefix budget."""

    return tuple(prefix_name(budget, name) for name in OFFICIAL_FEATURE_NAMES)


def budget_trend_feature_names(budget: int) -> tuple[str, ...]:
    """Return the 12 temporal-trend features for one prefix budget."""

    return tuple(prefix_name(budget, name) for name in TREND_FEATURE_NAMES)


def budget_slope_feature_names(budget: int) -> tuple[str, ...]:
    """Return the 4 raw temporal slopes for one prefix budget."""

    return tuple(
        prefix_name(budget, f"temporal_{signal}_slope")
        for signal in SIGNAL_NAMES
    )


def budget_e2a_feature_names(budget: int) -> tuple[str, ...]:
    """Return E2A-Full = official 14D + temporal trend 12D for one budget."""

    return (
        *budget_official_feature_names(budget),
        *budget_trend_feature_names(budget),
    )


def budget_e2a_slope_feature_names(budget: int) -> tuple[str, ...]:
    """Return E2A-Slope = official 14D + 4 raw temporal slopes."""

    return (
        *budget_official_feature_names(budget),
        *budget_slope_feature_names(budget),
    )


def early_exit_cache_feature_names(budgets: Sequence[int]) -> tuple[str, ...]:
    """Return every prefix feature persisted by the early-exit extractor."""

    names: list[str] = []
    for budget in normalize_budgets(budgets):
        names.append(prefix_name(budget, "num_tokens"))
        # E2A-Full already contains all official and trend features needed by
        # E1 and E2A-Slope, so no additional cache columns are required.
        names.extend(budget_e2a_feature_names(budget))
    return tuple(names)
