#!/usr/bin/env python3
"""Multi-seed micro-ablation for Temporal Uncertainty trend features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

SOLUTION_ROOT = Path(__file__).resolve().parents[1]
ROOT = SOLUTION_ROOT.parents[1]
if str(SOLUTION_ROOT) not in sys.path:
    sys.path.insert(0, str(SOLUTION_ROOT))

import train_uncertainty_regressor as official_train  # noqa: E402

from uncertainty_profile.config import FEATURE_NAMES as OFFICIAL_FEATURE_NAMES  # noqa: E402
from uncertainty_profile.temporal_metrics import (  # noqa: E402
    SIGNAL_NAMES,
    TREND_FEATURE_NAMES,
)


DEFAULT_SEEDS = tuple(range(10))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run fixed-estimator multi-seed micro-ablations over temporal trend "
            "features, optionally with nested grouped CV for honest thresholding."
        )
    )
    parser.add_argument("--feature-data-path", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=4)
    parser.add_argument(
        "--protocol",
        choices=("official", "nested"),
        default="official",
        help=(
            "official reproduces the competition OOF-threshold protocol; nested "
            "selects thresholds only inside each outer training fold."
        ),
    )
    parser.add_argument(
        "--scope",
        choices=("core", "full"),
        default="core",
        help="full additionally tests every individual trend feature.",
    )
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument(
        "--estimator-seed",
        type=int,
        default=42,
        help="Keep estimator randomness fixed while CV split seeds vary.",
    )
    parser.add_argument("--min-samples-leaf", type=int, default=4)
    parser.add_argument(
        "--results-path",
        type=Path,
        default=ROOT / "data" / "uncertainty-profiling" / "trend-micro-ablation.json",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=ROOT / "data" / "uncertainty-profiling" / "trend-micro-ablation.csv",
    )
    return parser.parse_args()


def extra_trees_spec(args: argparse.Namespace) -> official_train.CandidateSpec:
    """Use the pilot winner as a fixed estimator to isolate feature effects."""

    return official_train.CandidateSpec(
        family="extra_trees",
        params={
            "n_estimators": int(args.n_estimators),
            "max_depth": None,
            "min_samples_leaf": int(args.min_samples_leaf),
            "max_features": 1.0,
        },
    )


def trend_features_for_signal(signal: str) -> tuple[str, ...]:
    return (
        f"temporal_{signal}_slope",
        f"temporal_{signal}_linearity_r2",
        f"temporal_{signal}_normalized_slope",
    )


def trend_features_for_statistic(statistic: str) -> tuple[str, ...]:
    if statistic not in {"slope", "linearity_r2", "normalized_slope"}:
        raise ValueError(f"unsupported trend statistic: {statistic}")
    return tuple(f"temporal_{signal}_{statistic}" for signal in SIGNAL_NAMES)


def build_experiments(scope: str) -> dict[str, tuple[str, ...]]:
    official = tuple(OFFICIAL_FEATURE_NAMES)
    experiments: dict[str, tuple[str, ...]] = {
        "E1_official_14d": official,
        "E2A_trend_only_12d": tuple(TREND_FEATURE_NAMES),
        "E2A_all_trend_26d": (*official, *TREND_FEATURE_NAMES),
    }
    for signal in SIGNAL_NAMES:
        experiments[f"E2A_signal_{signal}"] = (
            *official,
            *trend_features_for_signal(signal),
        )
    for statistic in ("slope", "linearity_r2", "normalized_slope"):
        experiments[f"E2A_stat_{statistic}"] = (
            *official,
            *trend_features_for_statistic(statistic),
        )

    experiments["E2A_sparse_slope_4d"] = trend_features_for_statistic("slope")
    experiments["E2A_sparse_r2_4d"] = trend_features_for_statistic("linearity_r2")
    experiments["E2A_sparse_normalized_slope_4d"] = trend_features_for_statistic(
        "normalized_slope"
    )

    if scope == "full":
        for feature_name in TREND_FEATURE_NAMES:
            short_name = feature_name.removeprefix("temporal_")
            experiments[f"E2A_single_{short_name}"] = (*official, feature_name)
    return experiments


def _validate_seed_list(seeds: Iterable[int]) -> tuple[int, ...]:
    values = tuple(dict.fromkeys(int(seed) for seed in seeds))
    if not values:
        raise ValueError("at least one seed is required")
    return values


def evaluate_official(
    frame: pd.DataFrame,
    *,
    feature_names: tuple[str, ...],
    spec: official_train.CandidateSpec,
    n_splits: int,
    seed: int,
    estimator_seed: int,
) -> dict[str, float]:
    """Evaluate one fixed estimator using the competition-style OOF protocol."""

    official_train.FEATURE_NAMES = feature_names
    official_train.validate_feature_data(frame, n_splits)
    splits = official_train.make_grouped_splits(frame, n_splits=n_splits, seed=seed)
    predictions = official_train.compute_oof_predictions(
        official_train.build_pipeline(spec, estimator_seed),
        frame,
        splits,
    )
    result = official_train.evaluate_predictions(
        frame,
        predictions,
        identifier=f"fixed:{spec.identifier}",
    )
    return {
        "balanced_accuracy": float(result["balanced_accuracy"]),
        "accuracy": float(result["accuracy"]),
        "mae": float(result["mae"]),
        "mse": float(result["mse"]),
        "r2": float(result["r2"]),
        "decision_threshold": float(result["decision_threshold"]),
    }


def _inner_split_count(frame: pd.DataFrame, *, requested: int) -> int:
    unique_groups = frame[
        [official_train.GROUP_COLUMN, official_train.LABEL_COLUMN]
    ].drop_duplicates(official_train.GROUP_COLUMN)
    counts = unique_groups[official_train.LABEL_COLUMN].astype(bool).value_counts()
    if len(counts) != 2:
        raise ValueError("inner training fold does not contain both robustness classes")
    available = int(counts.min())
    n_splits = min(int(requested), available)
    if n_splits < 2:
        raise ValueError("inner grouped CV requires at least two groups per class")
    return n_splits


def evaluate_nested(
    frame: pd.DataFrame,
    *,
    feature_names: tuple[str, ...],
    spec: official_train.CandidateSpec,
    n_splits: int,
    inner_splits: int,
    seed: int,
    estimator_seed: int,
) -> dict[str, float | list[float]]:
    """Nested grouped CV: threshold selection never sees outer validation labels."""

    official_train.FEATURE_NAMES = feature_names
    official_train.validate_feature_data(frame, n_splits)
    outer_splits = official_train.make_grouped_splits(
        frame, n_splits=n_splits, seed=seed
    )

    scores = np.full(len(frame), np.nan, dtype=float)
    labels = frame[official_train.LABEL_COLUMN].astype(bool).to_numpy()
    thresholds: list[float] = []

    for fold_index, (train_indices, validation_indices) in enumerate(outer_splits):
        train = frame.iloc[train_indices].reset_index(drop=True)
        validation = frame.iloc[validation_indices]
        inner_n = _inner_split_count(train, requested=inner_splits)
        inner_seed = int(seed * 1009 + fold_index + 17)
        inner = official_train.make_grouped_splits(
            train,
            n_splits=inner_n,
            seed=inner_seed,
        )

        inner_predictions = official_train.compute_oof_predictions(
            official_train.build_pipeline(spec, estimator_seed),
            train,
            inner,
        )
        threshold, _, _ = official_train.select_threshold(
            train[official_train.LABEL_COLUMN].astype(bool).to_numpy(),
            inner_predictions,
        )
        thresholds.append(float(threshold))

        pipeline = official_train.build_pipeline(spec, estimator_seed)
        pipeline.fit(
            train[list(feature_names)],
            train[official_train.TARGET_COLUMN].astype(float),
        )
        scores[validation_indices] = pipeline.predict(
            validation[list(feature_names)]
        )

    if np.isnan(scores).any() or not np.isfinite(scores).all():
        raise RuntimeError("nested CV did not produce finite outer predictions")

    binary = np.zeros(len(frame), dtype=bool)
    for threshold, (_, validation_indices) in zip(thresholds, outer_splits):
        binary[validation_indices] = scores[validation_indices] < threshold

    target = frame[official_train.TARGET_COLUMN].astype(float).to_numpy()
    return {
        "balanced_accuracy": float(balanced_accuracy_score(labels, binary)),
        "accuracy": float(accuracy_score(labels, binary)),
        "mae": float(mean_absolute_error(target, scores)),
        "mse": float(mean_squared_error(target, scores)),
        "r2": float(r2_score(target, scores)),
        "decision_threshold_mean": float(np.mean(thresholds)),
        "decision_threshold_std": float(np.std(thresholds)),
        "outer_thresholds": thresholds,
    }


def aggregate_seed_results(seed_results: list[dict[str, object]]) -> dict[str, float]:
    aggregate: dict[str, float] = {}
    for metric in ("balanced_accuracy", "accuracy", "mae", "mse", "r2"):
        values = np.asarray([float(result[metric]) for result in seed_results])
        aggregate[f"{metric}_mean"] = float(np.mean(values))
        aggregate[f"{metric}_std"] = float(np.std(values))
        aggregate[f"{metric}_min"] = float(np.min(values))
        aggregate[f"{metric}_max"] = float(np.max(values))
    return aggregate


def main() -> None:
    args = parse_args()
    seeds = _validate_seed_list(args.seeds)
    if args.n_splits < 2 or args.inner_splits < 2:
        raise ValueError("CV split counts must be at least 2")

    frame = official_train.load_feature_data(args.feature_data_path)
    experiments = build_experiments(args.scope)
    spec = extra_trees_spec(args)

    payload: dict[str, object] = {
        "protocol": args.protocol,
        "n_splits": int(args.n_splits),
        "inner_splits": int(args.inner_splits),
        "seeds": list(seeds),
        "group_column": official_train.GROUP_COLUMN,
        "estimator": {
            "family": spec.family,
            "random_state": int(args.estimator_seed),
            **spec.params,
        },
        "row_count": int(len(frame)),
        "group_count": int(frame[official_train.GROUP_COLUMN].nunique()),
        "scope": args.scope,
        "experiments": {},
    }

    experiment_payload: dict[str, object] = {}
    for experiment_id, feature_names in experiments.items():
        missing = sorted(set(feature_names) - set(frame.columns))
        if missing:
            raise ValueError(f"{experiment_id} missing feature columns: {missing}")

        print(f"\n=== {experiment_id}: {len(feature_names)}D ===", flush=True)
        seed_results: list[dict[str, object]] = []
        for seed in seeds:
            if args.protocol == "official":
                result = evaluate_official(
                    frame,
                    feature_names=feature_names,
                    spec=spec,
                    n_splits=args.n_splits,
                    seed=seed,
                    estimator_seed=args.estimator_seed,
                )
            else:
                result = evaluate_nested(
                    frame,
                    feature_names=feature_names,
                    spec=spec,
                    n_splits=args.n_splits,
                    inner_splits=args.inner_splits,
                    seed=seed,
                    estimator_seed=args.estimator_seed,
                )
            result = {"seed": int(seed), **result}
            seed_results.append(result)
            print(
                f"seed={seed:>3} BA={float(result['balanced_accuracy']):.4f} "
                f"Acc={float(result['accuracy']):.4f} "
                f"MAE={float(result['mae']):.4f}",
                flush=True,
            )

        aggregate = aggregate_seed_results(seed_results)
        experiment_payload[experiment_id] = {
            "feature_count": len(feature_names),
            "feature_names": list(feature_names),
            "aggregate": aggregate,
            "per_seed": seed_results,
        }

    payload["experiments"] = experiment_payload
    baseline_aggregate = experiment_payload["E1_official_14d"]["aggregate"]  # type: ignore[index]
    baseline_ba = float(baseline_aggregate["balanced_accuracy_mean"])  # type: ignore[index]

    summary_rows: list[dict[str, object]] = []
    for experiment_id, result in experiment_payload.items():
        aggregate = result["aggregate"]  # type: ignore[index]
        mean_ba = float(aggregate["balanced_accuracy_mean"])
        summary_rows.append(
            {
                "experiment": experiment_id,
                "features": int(result["feature_count"]),  # type: ignore[index]
                "balanced_accuracy_mean": mean_ba,
                "balanced_accuracy_std": float(aggregate["balanced_accuracy_std"]),
                "delta_ba_vs_e1": mean_ba - baseline_ba,
                "accuracy_mean": float(aggregate["accuracy_mean"]),
                "accuracy_std": float(aggregate["accuracy_std"]),
                "mae_mean": float(aggregate["mae_mean"]),
                "mae_std": float(aggregate["mae_std"]),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["balanced_accuracy_mean", "accuracy_mean"],
        ascending=False,
        kind="stable",
    )
    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    official_train.write_json(args.results_path, payload)
    summary.to_csv(args.summary_csv, index=False)

    print("\n=== Aggregate ranking ===", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"\nSaved JSON: {args.results_path}")
    print(f"Saved CSV:  {args.summary_csv}")


if __name__ == "__main__":
    main()
