"""Temporal uncertainty features for reasoning-robustness prediction.

This module extends the official AIMO uncertainty baseline without changing its
14 scalar features. The added features preserve coarse temporal structure from
the token-level confidence traces already collected during greedy generation.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np

from uncertainty_profile.config import FEATURE_NAMES
from uncertainty_profile.metrics import MetricValue


TemporalMetrics: TypeAlias = dict[str, MetricValue]

SIGNAL_NAMES: tuple[str, ...] = (
    "logprob",
    "entropy",
    "top1_prob",
    "top2_margin",
)

TREND_FEATURE_NAMES: tuple[str, ...] = tuple(
    feature
    for signal in SIGNAL_NAMES
    for feature in (
        f"temporal_{signal}_slope",
        f"temporal_{signal}_linearity_r2",
        f"temporal_{signal}_normalized_slope",
    )
)

SEGMENT_FEATURE_NAMES: tuple[str, ...] = tuple(
    f"temporal_{signal}_q{quarter}_mean"
    for signal in SIGNAL_NAMES
    for quarter in range(1, 5)
)

CONTRAST_FEATURE_NAMES: tuple[str, ...] = tuple(
    f"temporal_{signal}_late_minus_early"
    for signal in SIGNAL_NAMES
)

VOLATILITY_FEATURE_NAMES: tuple[str, ...] = tuple(
    feature
    for signal in SIGNAL_NAMES
    for feature in (
        f"temporal_{signal}_mean_abs_step",
        f"temporal_{signal}_max_abs_step",
        f"temporal_{signal}_step_std",
    )
)

TRANSITION_FEATURE_NAMES: tuple[str, ...] = (
    "temporal_entropy_peak_position",
    "temporal_logprob_trough_position",
    "temporal_top1_trough_position",
    "temporal_margin_trough_position",
    "temporal_entropy_spike_count",
    "temporal_logprob_drop_count",
    "temporal_prob_high_to_low_count",
    "temporal_prob_low_to_high_count",
)

TEMPORAL_FEATURE_NAMES: tuple[str, ...] = (
    *TREND_FEATURE_NAMES,
    *SEGMENT_FEATURE_NAMES,
    *CONTRAST_FEATURE_NAMES,
    *VOLATILITY_FEATURE_NAMES,
    *TRANSITION_FEATURE_NAMES,
)

ALL_FEATURE_NAMES: tuple[str, ...] = (*FEATURE_NAMES, *TEMPORAL_FEATURE_NAMES)


def _finite(values: np.ndarray) -> np.ndarray:
    """Return a flat float array containing only finite values."""

    array = np.asarray(values, dtype=float).reshape(-1)
    return array[np.isfinite(array)]


def _linear_trend(values: np.ndarray) -> tuple[float, float, float]:
    """Fit a linear trend against normalized reasoning position in [0, 1]."""

    array = _finite(values)
    if array.size < 2:
        nan = float("nan")
        return nan, nan, nan

    position = np.linspace(0.0, 1.0, array.size, dtype=float)
    centered_position = position - float(np.mean(position))
    centered_values = array - float(np.mean(array))
    denominator = float(np.dot(centered_position, centered_position))
    if denominator <= 0.0:
        nan = float("nan")
        return nan, nan, nan

    slope = float(np.dot(centered_position, centered_values) / denominator)
    fitted = float(np.mean(array)) + slope * centered_position
    residual_sum = float(np.sum((array - fitted) ** 2))
    total_sum = float(np.sum(centered_values ** 2))
    r2 = 1.0 if total_sum <= 1e-12 else 1.0 - residual_sum / total_sum
    normalized_slope = slope / (float(np.std(array)) + 1e-8)
    return slope, float(r2), float(normalized_slope)


def _quartile_means(values: np.ndarray) -> tuple[float, float, float, float]:
    """Return means over four equal-count reasoning segments."""

    array = _finite(values)
    if array.size == 0:
        nan = float("nan")
        return nan, nan, nan, nan

    segments = np.array_split(array, 4)
    return tuple(
        float(np.mean(segment)) if segment.size else float("nan")
        for segment in segments
    )  # type: ignore[return-value]


def _volatility(values: np.ndarray) -> tuple[float, float, float]:
    """Summarize first-difference magnitude and variability."""

    array = _finite(values)
    if array.size < 2:
        nan = float("nan")
        return nan, nan, nan

    differences = np.diff(array)
    absolute = np.abs(differences)
    return (
        float(np.mean(absolute)),
        float(np.max(absolute)),
        float(np.std(differences)),
    )


def _normalized_extreme_position(values: np.ndarray, *, mode: str) -> float:
    """Return the normalized position of a finite maximum or minimum."""

    array = np.asarray(values, dtype=float).reshape(-1)
    finite_mask = np.isfinite(array)
    if not finite_mask.any():
        return float("nan")

    finite_indices = np.flatnonzero(finite_mask)
    finite_values = array[finite_mask]
    if mode == "max":
        local_index = int(np.argmax(finite_values))
    elif mode == "min":
        local_index = int(np.argmin(finite_values))
    else:
        raise ValueError(f"unsupported mode: {mode}")

    absolute_index = int(finite_indices[local_index])
    return float(absolute_index / max(1, array.size - 1))


def _robust_extreme_count(differences: np.ndarray, *, direction: str) -> int:
    """Count unusually large signed steps using a median/MAD threshold."""

    array = _finite(differences)
    if array.size == 0:
        return 0

    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    scale = 1.4826 * mad
    if scale <= 1e-8:
        scale = float(np.std(array))
    if scale <= 1e-8:
        return 0

    if direction == "high":
        return int(np.sum(array > median + 2.0 * scale))
    if direction == "low":
        return int(np.sum(array < median - 2.0 * scale))
    raise ValueError(f"unsupported direction: {direction}")


def compute_temporal_uncertainty_metrics(
    *,
    log_probs: np.ndarray,
    probs: np.ndarray,
    entropy: np.ndarray,
    top1_probs: np.ndarray,
    top2_margins: np.ndarray,
    high_conf_threshold: float,
    low_conf_threshold: float,
) -> TemporalMetrics:
    """Compute 52 temporal features from token-level uncertainty traces.

    The trace position is normalized to [0, 1], making slopes and extreme
    locations comparable across responses with different reasoning lengths.
    """

    if not 0.0 <= low_conf_threshold <= 1.0:
        raise ValueError("low_conf_threshold must be in [0, 1]")
    if not 0.0 <= high_conf_threshold <= 1.0:
        raise ValueError("high_conf_threshold must be in [0, 1]")

    signals = {
        "logprob": _finite(log_probs),
        "entropy": _finite(entropy),
        "top1_prob": _finite(top1_probs),
        "top2_margin": _finite(top2_margins),
    }

    metrics: TemporalMetrics = {}

    for signal_name in SIGNAL_NAMES:
        slope, r2, normalized_slope = _linear_trend(signals[signal_name])
        metrics[f"temporal_{signal_name}_slope"] = slope
        metrics[f"temporal_{signal_name}_linearity_r2"] = r2
        metrics[f"temporal_{signal_name}_normalized_slope"] = normalized_slope

    segment_values: dict[str, tuple[float, float, float, float]] = {}
    for signal_name in SIGNAL_NAMES:
        quartiles = _quartile_means(signals[signal_name])
        segment_values[signal_name] = quartiles
        for quarter, value in enumerate(quartiles, start=1):
            metrics[f"temporal_{signal_name}_q{quarter}_mean"] = value

    for signal_name in SIGNAL_NAMES:
        early = segment_values[signal_name][0]
        late = segment_values[signal_name][3]
        metrics[f"temporal_{signal_name}_late_minus_early"] = (
            float(late - early)
            if np.isfinite(early) and np.isfinite(late)
            else float("nan")
        )

    for signal_name in SIGNAL_NAMES:
        mean_abs_step, max_abs_step, step_std = _volatility(signals[signal_name])
        metrics[f"temporal_{signal_name}_mean_abs_step"] = mean_abs_step
        metrics[f"temporal_{signal_name}_max_abs_step"] = max_abs_step
        metrics[f"temporal_{signal_name}_step_std"] = step_std

    metrics["temporal_entropy_peak_position"] = _normalized_extreme_position(
        np.asarray(entropy, dtype=float), mode="max"
    )
    metrics["temporal_logprob_trough_position"] = _normalized_extreme_position(
        np.asarray(log_probs, dtype=float), mode="min"
    )
    metrics["temporal_top1_trough_position"] = _normalized_extreme_position(
        np.asarray(top1_probs, dtype=float), mode="min"
    )
    metrics["temporal_margin_trough_position"] = _normalized_extreme_position(
        np.asarray(top2_margins, dtype=float), mode="min"
    )
    metrics["temporal_entropy_spike_count"] = _robust_extreme_count(
        np.diff(signals["entropy"]), direction="high"
    )
    metrics["temporal_logprob_drop_count"] = _robust_extreme_count(
        np.diff(signals["logprob"]), direction="low"
    )

    probability_trace = _finite(probs)
    if probability_trace.size < 2:
        metrics["temporal_prob_high_to_low_count"] = 0
        metrics["temporal_prob_low_to_high_count"] = 0
    else:
        previous = probability_trace[:-1]
        current = probability_trace[1:]
        metrics["temporal_prob_high_to_low_count"] = int(
            np.sum(
                (previous >= high_conf_threshold)
                & (current <= low_conf_threshold)
            )
        )
        metrics["temporal_prob_low_to_high_count"] = int(
            np.sum(
                (previous <= low_conf_threshold)
                & (current >= high_conf_threshold)
            )
        )

    if set(metrics) != set(TEMPORAL_FEATURE_NAMES):
        missing = sorted(set(TEMPORAL_FEATURE_NAMES) - set(metrics))
        extra = sorted(set(metrics) - set(TEMPORAL_FEATURE_NAMES))
        raise RuntimeError(
            f"temporal feature schema mismatch; missing={missing}, extra={extra}"
        )
    return metrics
