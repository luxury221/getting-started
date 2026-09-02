#!/usr/bin/env python3
"""Evaluate E1, E2A-Full, and E2A-Slope across token-prefix budgets."""

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
    budget_e2a_slope_feature_names,
    budget_official_feature_names,
    normalize_budgets,
    prefix_name,
)


DEFAULT_SEEDS = tuple(range(10))
DEFAULT_BUDGETS = (128, 256, 512, 1024)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare aggregate uncertainty, full temporal trend, and compact "
            "slope dynamics at early generated-token budgets."
        )
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
    """Use the fixed ExtraTrees estimator from E2.5 to isolate feature effects."""

    return official_train.CandidateSpec(
        family="extra_trees",
        params={
            "n_estimators": int(args.n_estimators),
            "max_depth": None,
            "min_samples_leaf": int(args.min_samples_leaf),
            "max_features": 1.0,
        },
    )


def evaluate_one(
    frame: pd.DataFrame,
    *,
    feature_names: tuple[str, ...],
    spec: official_train.CandidateSpec,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, object]:
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
    return {"seed": int(seed), **result}


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
        "experiments": [
            "E1_official_14d",
            "E2A_full_26d",
            "E2A_slope_18d",
        ],
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
            "E2A_full_26d": budget_e2a_feature_names(budget),
            "E2A_slope_18d": budget_e2a_slope_feature_names(budget),
        }
        one_budget: dict[str, object] = {"effective_tokens": effective}

        print(f"\n##### Budget {budget} tokens #####", flush=True)
        for experiment_id, feature_names in experiments.items():
            missing = sorted(set(feature_names) - set(frame.columns))
            if missing:
                raise ValueError(f"budget {budget} missing columns: {missing}")

            seed_results = [
                evaluate_one(
                    frame,
                    feature_names=feature_names,
                    spec=spec,
                    args=args,
                    seed=seed,
                )
                for seed in seeds
            ]
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
        full = one_budget["E2A_full_26d"]["aggregate"]  # type: ignore[index]
        slope = one_budget["E2A_slope_18d"]["aggregate"]  # type: ignore[index]

        e1_ba = float(e1["balanced_accuracy_mean"])
        full_ba = float(full["balanced_accuracy_mean"])
        slope_ba = float(slope["balanced_accuracy_mean"])
        full_gain = full_ba - e1_ba
        slope_gain = slope_ba - e1_ba
        recovery = float(slope_gain / full_gain) if full_gain > 1e-12 else float("nan")

        deltas = {
            "full_vs_e1_ba": full_gain,
            "slope_vs_e1_ba": slope_gain,
            "slope_vs_full_ba": slope_ba - full_ba,
            "slope_gain_recovery_ratio": recovery,
            "full_vs_e1_accuracy": float(full["accuracy_mean"])
            - float(e1["accuracy_mean"]),
            "slope_vs_e1_accuracy": float(slope["accuracy_mean"])
            - float(e1["accuracy_mean"]),
        }
        one_budget["delta"] = deltas
        budget_payload[str(budget)] = one_budget

        summary_rows.append(
            {
                "budget": int(budget),
                "mean_effective_tokens": effective["mean_tokens"],
                "median_effective_tokens": effective["median_tokens"],
                "fraction_reached_budget": effective["fraction_reached_budget"],
                "e1_ba_mean": e1_ba,
                "e1_ba_std": float(e1["balanced_accuracy_std"]),
                "full_ba_mean": full_ba,
                "full_ba_std": float(full["balanced_accuracy_std"]),
                "slope_ba_mean": slope_ba,
                "slope_ba_std": float(slope["balanced_accuracy_std"]),
                "full_delta_ba": full_gain,
                "slope_delta_ba": slope_gain,
                "slope_minus_full_ba": slope_ba - full_ba,
                "slope_gain_recovery_ratio": recovery,
                "e1_accuracy_mean": float(e1["accuracy_mean"]),
                "full_accuracy_mean": float(full["accuracy_mean"]),
                "slope_accuracy_mean": float(slope["accuracy_mean"]),
            }
        )

    payload["results"] = budget_payload
    summary = pd.DataFrame(summary_rows).sort_values("budget")
    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    official_train.write_json(args.results_path, payload)
    summary.to_csv(args.summary_csv, index=False)

    print("\n=== E2.6 Early Dynamics matrix ===", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"\nSaved JSON: {args.results_path}")
    print(f"Saved CSV:  {args.summary_csv}")


if __name__ == "__main__":
    main()
