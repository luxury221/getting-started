"""Tests for early-exit prefix feature naming and E2.6 schema."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SOLUTION_ROOT = Path(__file__).resolve().parents[1]
if str(SOLUTION_ROOT) not in sys.path:
    sys.path.insert(0, str(SOLUTION_ROOT))

from uncertainty_profile.early_exit import (  # noqa: E402
    budget_e2a_feature_names,
    budget_e2a_slope_feature_names,
    budget_official_feature_names,
    budget_slope_feature_names,
    budget_trend_feature_names,
    early_exit_cache_feature_names,
    normalize_budgets,
    prefix_name,
)


class EarlyExitSchemaTest(unittest.TestCase):
    def test_budget_normalization(self) -> None:
        self.assertEqual(normalize_budgets([512, 128, 256, 128]), (128, 256, 512))
        with self.assertRaises(ValueError):
            normalize_budgets([])
        with self.assertRaises(ValueError):
            normalize_budgets([1, 128])

    def test_feature_dimensions(self) -> None:
        budget = 256
        self.assertEqual(len(budget_official_feature_names(budget)), 14)
        self.assertEqual(len(budget_trend_feature_names(budget)), 12)
        self.assertEqual(len(budget_slope_feature_names(budget)), 4)
        self.assertEqual(len(budget_e2a_slope_feature_names(budget)), 18)
        self.assertEqual(len(budget_e2a_feature_names(budget)), 26)

    def test_slope_model_is_strict_subset_of_full_e2a(self) -> None:
        budget = 512
        slope = set(budget_e2a_slope_feature_names(budget))
        full = set(budget_e2a_feature_names(budget))
        self.assertTrue(slope < full)

    def test_cache_schema_contains_all_e2_6_models(self) -> None:
        budgets = (128, 256, 512, 1024)
        names = early_exit_cache_feature_names(budgets)
        cache = set(names)
        self.assertEqual(len(names), 4 * 27)
        self.assertEqual(len(names), len(cache))

        for budget in budgets:
            self.assertIn(prefix_name(budget, "num_tokens"), cache)
            self.assertTrue(set(budget_official_feature_names(budget)) <= cache)
            self.assertTrue(set(budget_e2a_slope_feature_names(budget)) <= cache)
            self.assertTrue(set(budget_e2a_feature_names(budget)) <= cache)


if __name__ == "__main__":
    unittest.main()
