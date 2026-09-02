"""Unit tests for E2.6-B.1 sampling-stability helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SOLUTION_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = SOLUTION_ROOT / "scripts"
for path in (SOLUTION_ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_uncertainty_regressor as official_train  # noqa: E402
from run_sampling_stability_forensics import (  # noqa: E402
    determine_route,
    sample_groups,
    sanitize_json_numbers,
)


class SamplingStabilityForensicsTest(unittest.TestCase):
    def make_groups(self) -> pd.DataFrame:
        names = [f"g{i}" for i in range(10)]
        labels = [True] * 4 + [False] * 6
        return pd.DataFrame(
            {
                official_train.GROUP_COLUMN: names,
                official_train.LABEL_COLUMN: labels,
            }
        )

    def test_balanced_group_sampling(self) -> None:
        groups = self.make_groups()
        chosen = sample_groups(
            groups,
            n_groups=6,
            mode="balanced",
            rng=np.random.default_rng(7),
        )
        subset = groups[groups[official_train.GROUP_COLUMN].isin(chosen)]
        self.assertEqual(len(chosen), 6)
        self.assertEqual(int(subset[official_train.LABEL_COLUMN].sum()), 3)
        self.assertEqual(int((~subset[official_train.LABEL_COLUMN]).sum()), 3)

    def test_prevalence_group_sampling(self) -> None:
        groups = self.make_groups()
        chosen = sample_groups(
            groups,
            n_groups=5,
            mode="prevalence",
            rng=np.random.default_rng(9),
        )
        subset = groups[groups[official_train.GROUP_COLUMN].isin(chosen)]
        self.assertEqual(len(chosen), 5)
        self.assertGreaterEqual(int(subset[official_train.LABEL_COLUMN].sum()), 2)
        self.assertGreaterEqual(int((~subset[official_train.LABEL_COLUMN]).sum()), 2)

    def test_route_prefers_stable_512(self) -> None:
        half = pd.DataFrame(
            [
                {"subset": "first64", "budget": 512, "full_delta_ba": 0.07},
                {"subset": "second64", "budget": 512, "full_delta_ba": 0.03},
            ]
        )
        summary = pd.DataFrame(
            [
                {
                    "mode": "balanced",
                    "budget": 512,
                    "median_full_delta_ba": 0.03,
                    "full_positive_rate": 0.85,
                    "median_full_ba": 0.56,
                },
                {
                    "mode": "prevalence",
                    "budget": 512,
                    "median_full_delta_ba": 0.025,
                    "full_positive_rate": 0.82,
                    "median_full_ba": 0.55,
                },
                {
                    "mode": "balanced",
                    "budget": 1024,
                    "median_full_delta_ba": 0.03,
                    "full_positive_rate": 0.85,
                    "median_full_ba": 0.57,
                },
                {
                    "mode": "prevalence",
                    "budget": 1024,
                    "median_full_delta_ba": 0.03,
                    "full_positive_rate": 0.85,
                    "median_full_ba": 0.57,
                },
            ]
        )
        decision = determine_route(half, summary)
        self.assertEqual(
            decision["route"],
            "INDEPENDENT_STRATIFIED_REPLICATION_128",
        )

    def test_nonfinite_json_values_become_null(self) -> None:
        payload = sanitize_json_numbers({"x": float("nan"), "y": np.float64(np.inf)})
        self.assertIsNone(payload["x"])
        self.assertIsNone(payload["y"])


if __name__ == "__main__":
    unittest.main()
