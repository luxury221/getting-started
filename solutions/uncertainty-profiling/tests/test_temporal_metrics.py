"""Unit tests for Temporal Uncertainty v1 feature computation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

SOLUTION_ROOT = Path(__file__).resolve().parents[1]
if str(SOLUTION_ROOT) not in sys.path:
    sys.path.insert(0, str(SOLUTION_ROOT))

from uncertainty_profile.temporal_metrics import (  # noqa: E402
    TEMPORAL_FEATURE_NAMES,
    compute_temporal_uncertainty_metrics,
)


class TemporalMetricsTest(unittest.TestCase):
    def test_schema_has_52_features(self) -> None:
        probabilities = np.array([0.95, 0.90, 0.75, 0.40], dtype=float)
        metrics = compute_temporal_uncertainty_metrics(
            log_probs=np.log(probabilities),
            probs=probabilities,
            entropy=np.array([0.1, 0.2, 0.4, 0.9], dtype=float),
            top1_probs=np.array([0.97, 0.93, 0.80, 0.50], dtype=float),
            top2_margins=np.array([0.80, 0.70, 0.45, 0.15], dtype=float),
            high_conf_threshold=0.90,
            low_conf_threshold=0.10,
        )
        self.assertEqual(len(TEMPORAL_FEATURE_NAMES), 52)
        self.assertEqual(set(metrics), set(TEMPORAL_FEATURE_NAMES))

    def test_degrading_trace_has_expected_direction(self) -> None:
        probabilities = np.array([0.98, 0.93, 0.80, 0.55, 0.20], dtype=float)
        metrics = compute_temporal_uncertainty_metrics(
            log_probs=np.log(probabilities),
            probs=probabilities,
            entropy=np.array([0.10, 0.16, 0.30, 0.65, 1.20], dtype=float),
            top1_probs=np.array([0.99, 0.95, 0.86, 0.65, 0.30], dtype=float),
            top2_margins=np.array([0.90, 0.82, 0.60, 0.30, 0.08], dtype=float),
            high_conf_threshold=0.90,
            low_conf_threshold=0.10,
        )
        self.assertGreater(float(metrics["temporal_entropy_slope"]), 0.0)
        self.assertLess(float(metrics["temporal_logprob_slope"]), 0.0)
        self.assertGreater(
            float(metrics["temporal_entropy_late_minus_early"]),
            0.0,
        )
        self.assertGreaterEqual(
            float(metrics["temporal_entropy_peak_position"]),
            0.75,
        )

    def test_short_trace_is_safe(self) -> None:
        metrics = compute_temporal_uncertainty_metrics(
            log_probs=np.array([-0.2], dtype=float),
            probs=np.array([0.82], dtype=float),
            entropy=np.array([0.5], dtype=float),
            top1_probs=np.array([0.82], dtype=float),
            top2_margins=np.array([0.50], dtype=float),
            high_conf_threshold=0.90,
            low_conf_threshold=0.10,
        )
        self.assertTrue(np.isnan(float(metrics["temporal_entropy_slope"])))
        self.assertEqual(int(metrics["temporal_entropy_spike_count"]), 0)
        self.assertEqual(int(metrics["temporal_prob_high_to_low_count"]), 0)


if __name__ == "__main__":
    unittest.main()
