#!/usr/bin/env python3
"""Run grouped-CV E0/E1/E2 ablations for Temporal Uncertainty v1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score

SOLUTION_ROOT = Path(__file__).resolve().parents[1]
ROOT = SOLUTION_ROOT.parents[1]
if str(SOLUTION_ROOT) not in sys.path:
    sys.path.insert(0, str(SOLUTION_ROOT))

import train_uncertainty_regressor as official_train  # noqa: E402

from uncertainty_profile.config import FEATURE_NAMES as OFFICIAL_FEATURE_NAMES  # noqa: E402
from uncertainty_profile.temporal_metrics import (  # noqa: E402
    ALL_FEATURE_NAMES,
    CONTRAST_FEATURE_NAMES,
    SEGMENT_FEATURE_NAMES,
    TRANSITION_FEATURE_NAMES,
    TREND_FEATURE_NAMES,
    VOLATILITY_FEATURE_NAMES,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare official and temporal uncertainty feature groups."
    )
    parser.add_argument(
        "--feature-data-path",
        type=Path,
        required=True,
        help="Temporal feature Parquet file or directory.",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use one candidate per estimator family for a fast smoke test.",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=ROOT
        / "data"
        / "uncertainty-profiling"
        / "temporal-v1-ablation.json",
    )
    return parser.parse_args()


def quick_specs() -> list[official_train.CandidateSpec]:
    return [
        official_train.CandidateSpec(
            family="random_forest",
            params={
                "n_estimators": 200,
                "max_depth": None,
                "min_samples_leaf": 4,
                "max_features": 1.0,
            },
        ),
        official_train.CandidateSpec(
            family="extra_trees",
            params={
                "n_estimators": 200,
                "max_depth": None,
                "min_samples_leaf": 4,
                "max_features": 1.0,
            },
        ),
        official_train.CandidateSpec(
            family="hist_gradient_boosting",
            params={
                "max_iter": 200,
                "learning_rate": 0.05,
                "max_leaf_nodes": 7,
                "l2_regularization": 1.0,
            },
        ),
    ]


def main() -> None:
    args = parse_args()
    if args.n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    frame = official_train.load_feature_data(args.feature_data_path)
    labels = frame[official_train.LABEL_COLUMN].astype(bool).to_numpy()

    majority_label = bool(float(np.mean(labels.astype(float))) >= 0.5)
    majority_predictions = np.full(labels.shape, majority_label, dtype=bool)
    e0 = {
        "majority_label": majority_label,
        "accuracy": float(accuracy_score(labels, majority_predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, majority_predictions)
        ),
    }

    experiments: dict[str, tuple[str, ...]] = {
        "E1_official_14d": tuple(OFFICIAL_FEATURE_NAMES),
        "E2A_trend": (*OFFICIAL_FEATURE_NAMES, *TREND_FEATURE_NAMES),
        "E2B_segment_contrast": (
            *OFFICIAL_FEATURE_NAMES,
            *SEGMENT_FEATURE_NAMES,
            *CONTRAST_FEATURE_NAMES,
        ),
        "E2C_volatility_transition": (
            *OFFICIAL_FEATURE_NAMES,
            *VOLATILITY_FEATURE_NAMES,
            *TRANSITION_FEATURE_NAMES,
        ),
        "E2D_full_temporal_v1": tuple(ALL_FEATURE_NAMES),
    }

    specs = quick_specs() if args.quick else official_train.build_candidate_specs()
    results: dict[str, object] = {
        "protocol": {
            "group_column": official_train.GROUP_COLUMN,
            "selection_metric": "balanced_accuracy",
            "n_splits": args.n_splits,
            "seed": args.seed,
            "candidate_count": len(specs),
            "quick": bool(args.quick),
        },
        "E0_majority": e0,
        "experiments": {},
    }

    experiment_results: dict[str, object] = {}
    for experiment_id, feature_names in experiments.items():
        print(
            f"\n=== {experiment_id}: {len(feature_names)} features ===",
            flush=True,
        )
        official_train.FEATURE_NAMES = feature_names
        official_train.validate_feature_data(frame, args.n_splits)
        best_spec, best_result, _, dummy_result = official_train.run_model_selection(
            frame,
            specs=specs,
            n_splits=args.n_splits,
            seed=args.seed,
        )
        experiment_results[experiment_id] = {
            "feature_count": len(feature_names),
            "feature_names": list(feature_names),
            "best_family": best_spec.family,
            "best_params": best_spec.params,
            "best_cv_metrics": best_result,
            "mean_regressor_baseline": dummy_result,
        }

    results["experiments"] = experiment_results
    official_train.write_json(args.results_path, results)
    print(json.dumps(results, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
