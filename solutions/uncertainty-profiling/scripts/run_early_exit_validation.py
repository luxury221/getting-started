#!/usr/bin/env python3
"""Evaluate E1 vs E2A across generated-token prefix budgets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SOLUTION_ROOT = Path(__file__).resolve().parents[1]
ROOT = SOLUTION_ROOT.parents[1]
if str(SOLUTION_ROOT) not in sys.path:
    sys.path.insert(0, str(SOLUTION_ROOT))

import train_uncertainty_regressor as official_train  # noqa: E402
from run_trend_micro_ablation import (  # noqa: E402
    aggregate_seed_results,
    evaluate_nested,
    evaluate_official,
)

from uncertainty_profile.early_exit import (  # noqa: E402
    budget_e2a_feature_names,
    budget_official_feature_names,
    normalize_budgets,
    prefix_name,
)


DEFAULT_SEEDS = tuple(range(10))
DEFAULT_BUDGETS = (128, 256, 512, 1024)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare official uncertainty and temporal trend at early token budgets."
    )
    parser.add_argument("--feature-data-path", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=4)
    parser.add_argument(
        "--protocol",
        choices=("official", "nested"),
        default="official",
    )
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--estimator-seed", type=int, default=42)
    parser.add_argument("--min-samples-leaf", type=int, default=4)
    parser.add_argument(
        "--results-path",
        type=Path,
        default=ROOT / "data" / "uncertainty-profiling" / "early-exit-validation.json",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=ROOT / "data" / "uncertainty-profiling" / "early-exit-validation.csv",
    )
    return parser.parse_args()


def fixed_spec(args: argparse.Namespace) -> official_train.CandidateSpec:
    return official_train.CandidateSpec(
        family="extra_trees",
        params={
            "n_estimators": int(args.n_estimators),
            "max_depth": None,
            "min_samples_leaf": int(args.min_samples_leaf),
            "max_features": 1.0,
        },
    )


def main() -> None:
    args = parse_args()
    budgets = normalize_budgets(args.budgets)
    seeds = tuple(dict.fromkeys(int(seed) for seed in args.seeds))
    if not seeds:
        raise ValueError("at least one seed is required")

    frame = official_train.load_feature_data(args.feature_data_path)
    spec = fixed_spec(args)

    payload: dict[str, object] = {
        "protocol": args.protocol,
        "budgets": list(budgets),
        "seeds": list(seeds),
        "n_splits": int(args.n_splits),
        "inner_splits": int(args.inner_splits),
        "group_column": official_train.GROUP_COLUMN,
        "estimator": {
            "family": spec.family,
            "random_state": int(args.estimator_seed),
            **spec.params,
        },
        "row_count": int(len(frame)),
        "group_count": int(frame[official_train.GROUP_COLUMN].nunique()),
        "results": {},
    }

    budget_payload: dict[str, object] = {}
    summary_rows: list[dict[str, object]] = []

    for budget in budgets:
        count_column = prefix_name(budget, "num_tokens")
        if count_column not in frame.columns:
            raise ValueError(
                f"missing {count_column}; regenerate features with "
                f"--prefix-budgets including {budget}"
            )

        counts = pd.to_numeric(frame[count_column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(counts).all():
            raise ValueError(f"{count_column} contains non-finite values")
        effective = {
            "mean_tokens": float(np.mean(counts)),
            "median_tokens": float(np.median(counts)),
            "fraction_reached_budget": float(np.mean(counts >= budget)),
        }

        experiments = {
            "E1_official_14d": budget_official_feature_names(budget),
            "E2A_trend_26d": budget_e2a_feature_names(budget),
        }
        one_budget: dict[str, object] = {"effective_tokens": effective}

        print(f"\n##### Budget {budget} tokens #####", flush=True)
        for experiment_id, feature_names in experiments.items():
            missing = sorted(set(feature_names) - set(frame.columns))
            if missing:
                raise ValueError(f"budget {budget} missing columns: {missing}")

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
                seed_results.append({"seed": int(seed), **result})

            aggregate = aggregate_seed_results(seed_results)
            one_budget[experiment_id] = {
                "feature_count": len(feature_names),
                "aggregate": aggregate,
                "per_seed": seed_results,
            }
            print(
                f"{experiment_id:<18} "
                f"BA={aggregate['balanced_accuracy_mean']:.4f}"
                f"±{aggregate['balanced_accuracy_std']:.4f} "
                f"Acc={aggregate['accuracy_mean']:.4f}"
                f"±{aggregate['accuracy_std']:.4f}",
                flush=True,
            )

        e1 = one_budget["E1_official_14d"]["aggregate"]  # type: ignore[index]
        e2 = one_budget["E2A_trend_26d"]["aggregate"]  # type: ignore[index]
        delta_ba = float(e2["balanced_accuracy_mean"]) - float(
            e1["balanced_accuracy_mean"]
        )
        delta_accuracy = float(e2["accuracy_mean"]) - float(e1["accuracy_mean"])
        one_budget["delta"] = {
            "balanced_accuracy": delta_ba,
            "accuracy": delta_accuracy,
        }
        budget_payload[str(budget)] = one_budget

        summary_rows.append(
            {
                "budget": int(budget),
                "mean_effective_tokens": effective["mean_tokens"],
                "fraction_reached_budget": effective["fraction_reached_budget"],
                "e1_ba_mean": float(e1["balanced_accuracy_mean"]),
                "e1_ba_std": float(e1["balanced_accuracy_std"]),
                "e2a_ba_mean": float(e2["balanced_accuracy_mean"]),
                "e2a_ba_std": float(e2["balanced_accuracy_std"]),
                "delta_ba": delta_ba,
                "e1_accuracy_mean": float(e1["accuracy_mean"]),
                "e2a_accuracy_mean": float(e2["accuracy_mean"]),
                "delta_accuracy": delta_accuracy,
            }
        )

    payload["results"] = budget_payload
    summary = pd.DataFrame(summary_rows).sort_values("budget")
    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    official_train.write_json(args.results_path, payload)
    summary.to_csv(args.summary_csv, index=False)

    print("\n=== Early-exit matrix ===", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"\nSaved JSON: {args.results_path}")
    print(f"Saved CSV:  {args.summary_csv}")


if __name__ == "__main__":
    main()
