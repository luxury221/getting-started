"""Tests for early-exit prefix feature naming and budget normalization."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SOLUTION_ROOT = Path(__file__).resolve().parents[1]
if str(SOLUTION_ROOT) not in sys.path:
    sys.path.insert(0, str(SOLUTION_ROOT))

from uncertainty_profile.early_exit import (  # noqa: E402
    budget_e2a_feature_names,
    budget_official_feature_names,
    budget_trend_feature_names,
    early_exit_cache_feature_names,
    normalize_budgets,
)


class EarlyExitSchemaTest(unittest.TestCase):
    def test_budget_normalization(self) -> None:
        self.assertEqual(normalize_budgets([512, 128, 256, 128]), (128, 256, 512))
        with self.assertRaises(ValueError):
            normalize_budgets([])
        with self.assertRaises(ValueError):
            normalize_budgets([1, 128])

    def test_feature_dimensions(self) -> None:
        self.assertEqual(len(budget_official_feature_names(256)), 14)
        self.assertEqual(len(budget_trend_feature_names(256)), 12)
        self.assertEqual(len(budget_e2a_feature_names(256)), 26)

    def test_cache_schema_is_unique(self) -> None:
        budgets = (128, 256, 512, 1024)
        names = early_exit_cache_feature_names(budgets)
        self.assertEqual(len(names), 4 * 27)
        self.assertEqual(len(names), len(set(names)))
        for budget in budgets:
            self.assertIn(f"prefix_{budget}__num_tokens", names)
            self.assertIn(f"prefix_{budget}__temporal_entropy_slope", names)


if __name__ == "__main__":
    unittest.main()
