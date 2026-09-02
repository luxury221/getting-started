#!/usr/bin/env python3
"""Diagnose sampling/composition instability in E2.6 Early Dynamics.

This stage reuses an existing early_dynamics_128.parquet cache and performs CPU-only
forensics. It does not regenerate model responses.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

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
)


DEFAULT_BUDGETS = (128, 256, 512, 1024)
DEFAULT_PRIMARY_BUDGETS = (512, 1024)
DEFAULT_HALF_SEEDS = (0, 1, 2, 3, 4)
EXPERIMENTS = ("E1_official_14d", "E2A_full_26d", "E2A_slope_18d")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "CPU-only E2.6-B.1 sampling stability forensics using an existing "
            "early-dynamics feature cache."
        )
    )
    parser.add_argument("--feature-data-path", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS))
    parser.add_argument(
        "--primary-budgets",
        type=int,
        nargs="+",
        default=list(DEFAULT_PRIMARY_BUDGETS),
        help="Budgets used for repeated group-subsampling diagnostics.",
    )
    parser.add_argument(
        "--half-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_HALF_SEEDS),
        help="Nested-CV seeds used for first-half/second-half comparisons.",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=4)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--estimator-seed", type=int, default=42)
    parser.add_argument("--min-samples-leaf", type=int, default=4)
    parser.add_argument(
        "--subsample-groups",
        type=int,
        default=64,
        help="Number of unique original_problem groups per repeated subsample.",
    )
    parser.add_argument("--subsample-repeats", type=int, default=50)
    parser.add_argument(
        "--subsample-cv-seed",
        type=int,
        default=42,
        help="Fixed grouped-CV seed so subsampling variability is isolated.",
    )
    parser.add_argument("--subsample-seed", type=int, default=20260902)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260903)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "uncertainty-profiling" / "e2_6_b1_sampling_stability",
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


def feature_map(budget: int) -> dict[str, tuple[str, ...]]:
    return {
        "E1_official_14d": budget_official_feature_names(budget),
        "E2A_full_26d": budget_e2a_feature_names(budget),
        "E2A_slope_18d": budget_e2a_slope_feature_names(budget),
    }


def safe_split_count(frame: pd.DataFrame, requested: int) -> int:
    groups = frame[
        [official_train.GROUP_COLUMN, official_train.LABEL_COLUMN]
    ].drop_duplicates(official_train.GROUP_COLUMN)
    counts = groups[official_train.LABEL_COLUMN].astype(bool).value_counts()
    if len(counts) != 2:
        raise ValueError("subset must contain both robustness classes")
    n_splits = min(int(requested), int(counts.min()))
    if n_splits < 2:
        raise ValueError("subset needs at least two groups in each class")
    return n_splits


def summarize_composition(frame: pd.DataFrame, name: str) -> dict[str, object]:
    groups = frame[
        [official_train.GROUP_COLUMN, official_train.LABEL_COLUMN]
    ].drop_duplicates(official_train.GROUP_COLUMN)
    labels = groups[official_train.LABEL_COLUMN].astype(bool)
    target = pd.to_numeric(frame[official_train.TARGET_COLUMN], errors="coerce")
    return {
        "slice": name,
        "rows": int(len(frame)),
        "groups": int(frame[official_train.GROUP_COLUMN].nunique()),
        "robust_groups": int(labels.sum()),
        "spurious_groups": int((~labels).sum()),
        "robust_group_fraction": float(labels.mean()),
        "target_mean": float(target.mean()),
        "target_std": float(target.std(ddof=0)),
        "target_median": float(target.median()),
    }


def split_stage_halves(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "source_row_index" in frame.columns:
        ordered = frame.sort_values("source_row_index", kind="stable").reset_index(drop=True)
    else:
        ordered = frame.reset_index(drop=True)
    if len(ordered) < 128:
        raise ValueError(
            f"first/second-64 diagnostics require at least 128 rows; got {len(ordered)}"
        )
    return ordered.iloc[:64].copy(), ordered.iloc[64:128].copy()


def evaluate_subset_nested(
    frame: pd.DataFrame,
    *,
    budgets: tuple[int, ...],
    seeds: tuple[int, ...],
    args: argparse.Namespace,
    spec: official_train.CandidateSpec,
    subset_name: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    n_splits = safe_split_count(frame, args.n_splits)
    for budget in budgets:
        for experiment_id, features in feature_map(budget).items():
            missing = sorted(set(features) - set(frame.columns))
            if missing:
                raise ValueError(f"{subset_name} missing feature columns: {missing}")
            seed_results: list[dict[str, object]] = []
            for seed in seeds:
                result = evaluate_nested(
                    frame.reset_index(drop=True),
                    feature_names=features,
                    spec=spec,
                    n_splits=n_splits,
                    inner_splits=args.inner_splits,
                    seed=seed,
                    estimator_seed=args.estimator_seed,
                )
                seed_results.append({"seed": int(seed), **result})
            aggregate = aggregate_seed_results(seed_results)
            rows.append(
                {
                    "subset": subset_name,
                    "budget": int(budget),
                    "experiment": experiment_id,
                    "balanced_accuracy_mean": aggregate["balanced_accuracy_mean"],
                    "balanced_accuracy_std": aggregate["balanced_accuracy_std"],
                    "accuracy_mean": aggregate["accuracy_mean"],
                    "mae_mean": aggregate["mae_mean"],
                    "n_splits": int(n_splits),
                    "seeds": ",".join(str(seed) for seed in seeds),
                }
            )
    return rows


def attach_deltas(table: pd.DataFrame) -> pd.DataFrame:
    output: list[dict[str, object]] = []
    for (subset, budget), block in table.groupby(["subset", "budget"], sort=False):
        by_exp = block.set_index("experiment")
        if not set(EXPERIMENTS).issubset(by_exp.index):
            continue
        e1 = float(by_exp.loc["E1_official_14d", "balanced_accuracy_mean"])
        full = float(by_exp.loc["E2A_full_26d", "balanced_accuracy_mean"])
        slope = float(by_exp.loc["E2A_slope_18d", "balanced_accuracy_mean"])
        output.append(
            {
                "subset": subset,
                "budget": int(budget),
                "e1_ba": e1,
                "full_ba": full,
                "slope_ba": slope,
                "full_delta_ba": full - e1,
                "slope_delta_ba": slope - e1,
                "slope_minus_full_ba": slope - full,
            }
        )
    return pd.DataFrame(output).sort_values(["subset", "budget"])


def group_table(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [official_train.GROUP_COLUMN, official_train.LABEL_COLUMN]
    if "source_row_index" in frame.columns:
        columns.append("source_row_index")
    groups = frame[columns].copy()
    if "source_row_index" in groups.columns:
        groups = groups.sort_values("source_row_index", kind="stable")
    groups = groups.drop_duplicates(official_train.GROUP_COLUMN, keep="first")
    return groups.reset_index(drop=True)


def sample_groups(
    groups: pd.DataFrame,
    *,
    n_groups: int,
    mode: str,
    rng: np.random.Generator,
) -> list[str]:
    if n_groups < 4:
        raise ValueError("subsample_groups must be at least 4")
    labels = groups[official_train.LABEL_COLUMN].astype(bool)
    positive = groups.loc[labels, official_train.GROUP_COLUMN].astype(str).to_numpy()
    negative = groups.loc[~labels, official_train.GROUP_COLUMN].astype(str).to_numpy()
    if mode == "balanced":
        n_pos = n_groups // 2
        n_neg = n_groups - n_pos
    elif mode == "prevalence":
        prevalence = len(positive) / len(groups)
        n_pos = max(2, min(len(positive), int(round(n_groups * prevalence))))
        n_neg = n_groups - n_pos
        if n_neg < 2:
            n_neg = 2
            n_pos = n_groups - n_neg
    else:
        raise ValueError(f"unsupported subsample mode: {mode}")
    if n_pos > len(positive) or n_neg > len(negative):
        raise ValueError(
            f"cannot sample mode={mode}: need pos={n_pos}, neg={n_neg}, "
            f"available pos={len(positive)}, neg={len(negative)}"
        )
    chosen_pos = rng.choice(positive, size=n_pos, replace=False)
    chosen_neg = rng.choice(negative, size=n_neg, replace=False)
    chosen = np.concatenate([chosen_pos, chosen_neg])
    rng.shuffle(chosen)
    return [str(value) for value in chosen]


def evaluate_subsample_official(
    frame: pd.DataFrame,
    *,
    features: tuple[str, ...],
    spec: official_train.CandidateSpec,
    args: argparse.Namespace,
) -> dict[str, float]:
    n_splits = safe_split_count(frame, args.n_splits)
    return evaluate_official(
        frame.reset_index(drop=True),
        feature_names=features,
        spec=spec,
        n_splits=n_splits,
        seed=args.subsample_cv_seed,
        estimator_seed=args.estimator_seed,
    )


def run_subsampling(
    frame: pd.DataFrame,
    *,
    budgets: tuple[int, ...],
    args: argparse.Namespace,
    spec: official_train.CandidateSpec,
) -> pd.DataFrame:
    groups = group_table(frame)
    rng = np.random.default_rng(args.subsample_seed)
    rows: list[dict[str, object]] = []
    for mode in ("balanced", "prevalence"):
        for repeat in range(args.subsample_repeats):
            chosen_groups = sample_groups(
                groups,
                n_groups=args.subsample_groups,
                mode=mode,
                rng=rng,
            )
            subset = frame[
                frame[official_train.GROUP_COLUMN].astype(str).isin(chosen_groups)
            ].copy()
            for budget in budgets:
                metrics: dict[str, dict[str, float]] = {}
                for experiment_id, features in feature_map(budget).items():
                    metrics[experiment_id] = evaluate_subsample_official(
                        subset,
                        features=features,
                        spec=spec,
                        args=args,
                    )
                e1 = metrics["E1_official_14d"]["balanced_accuracy"]
                full = metrics["E2A_full_26d"]["balanced_accuracy"]
                slope = metrics["E2A_slope_18d"]["balanced_accuracy"]
                rows.append(
                    {
                        "mode": mode,
                        "repeat": int(repeat),
                        "budget": int(budget),
                        "groups": int(len(chosen_groups)),
                        "rows": int(len(subset)),
                        "robust_group_fraction": float(
                            group_table(subset)[official_train.LABEL_COLUMN]
                            .astype(bool)
                            .mean()
                        ),
                        "e1_ba": float(e1),
                        "full_ba": float(full),
                        "slope_ba": float(slope),
                        "full_delta_ba": float(full - e1),
                        "slope_delta_ba": float(slope - e1),
                        "slope_minus_full_ba": float(slope - full),
                    }
                )
    return pd.DataFrame(rows)


def percentile_interval(values: pd.Series, alpha: float = 0.05) -> tuple[float, float]:
    array = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if not len(array):
        return float("nan"), float("nan")
    return (
        float(np.quantile(array, alpha / 2.0)),
        float(np.quantile(array, 1.0 - alpha / 2.0)),
    )


def summarize_subsampling(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (mode, budget), block in table.groupby(["mode", "budget"], sort=True):
        full_lo, full_hi = percentile_interval(block["full_delta_ba"])
        slope_lo, slope_hi = percentile_interval(block["slope_delta_ba"])
        rows.append(
            {
                "mode": mode,
                "budget": int(budget),
                "repeats": int(len(block)),
                "median_e1_ba": float(block["e1_ba"].median()),
                "median_full_ba": float(block["full_ba"].median()),
                "median_slope_ba": float(block["slope_ba"].median()),
                "median_full_delta_ba": float(block["full_delta_ba"].median()),
                "full_delta_p025": full_lo,
                "full_delta_p975": full_hi,
                "full_positive_rate": float((block["full_delta_ba"] > 0).mean()),
                "median_slope_delta_ba": float(block["slope_delta_ba"].median()),
                "slope_delta_p025": slope_lo,
                "slope_delta_p975": slope_hi,
                "slope_positive_rate": float((block["slope_delta_ba"] > 0).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["mode", "budget"])


def inner_split_count(frame: pd.DataFrame, requested: int) -> int:
    groups = frame[
        [official_train.GROUP_COLUMN, official_train.LABEL_COLUMN]
    ].drop_duplicates(official_train.GROUP_COLUMN)
    counts = groups[official_train.LABEL_COLUMN].astype(bool).value_counts()
    if len(counts) != 2:
        raise ValueError("inner fold lacks both robustness classes")
    n_splits = min(int(requested), int(counts.min()))
    if n_splits < 2:
        raise ValueError("inner grouped CV requires at least two groups per class")
    return n_splits


def nested_oof(
    frame: pd.DataFrame,
    *,
    features: tuple[str, ...],
    spec: official_train.CandidateSpec,
    args: argparse.Namespace,
    seed: int,
) -> pd.DataFrame:
    local = frame.reset_index(drop=True).copy()
    official_train.FEATURE_NAMES = features
    n_splits = safe_split_count(local, args.n_splits)
    official_train.validate_feature_data(local, n_splits)
    outer = official_train.make_grouped_splits(local, n_splits=n_splits, seed=seed)
    scores = np.full(len(local), np.nan, dtype=float)
    binary = np.zeros(len(local), dtype=bool)
    fold_ids = np.full(len(local), -1, dtype=int)
    thresholds = np.full(len(local), np.nan, dtype=float)
    for fold_id, (train_idx, valid_idx) in enumerate(outer):
        train = local.iloc[train_idx].reset_index(drop=True)
        valid = local.iloc[valid_idx]
        inner_n = inner_split_count(train, args.inner_splits)
        inner = official_train.make_grouped_splits(
            train,
            n_splits=inner_n,
            seed=int(seed * 1009 + fold_id + 17),
        )
        inner_pred = official_train.compute_oof_predictions(
            official_train.build_pipeline(spec, args.estimator_seed),
            train,
            inner,
        )
        threshold, _, _ = official_train.select_threshold(
            train[official_train.LABEL_COLUMN].astype(bool).to_numpy(),
            inner_pred,
        )
        pipeline = official_train.build_pipeline(spec, args.estimator_seed)
        pipeline.fit(train[list(features)], train[official_train.TARGET_COLUMN].astype(float))
        fold_scores = pipeline.predict(valid[list(features)])
        scores[valid_idx] = fold_scores
        binary[valid_idx] = fold_scores < threshold
        fold_ids[valid_idx] = fold_id
        thresholds[valid_idx] = threshold
    if np.isnan(scores).any() or (fold_ids < 0).any() or np.isnan(thresholds).any():
        raise RuntimeError("nested OOF did not cover every row")
    result = pd.DataFrame(
        {
            "row_index": np.arange(len(local), dtype=int),
            "original_problem": local[official_train.GROUP_COLUMN].astype(str).to_numpy(),
            "label": local[official_train.LABEL_COLUMN].astype(bool).to_numpy(),
            "target": local[official_train.TARGET_COLUMN].astype(float).to_numpy(),
            "score": scores,
            "predicted_robust": binary,
            "outer_fold": fold_ids,
            "threshold": thresholds,
            "seed": int(seed),
        }
    )
    if "source_row_index" in local.columns:
        result.insert(
            1,
            "source_row_index",
            pd.to_numeric(local["source_row_index"], errors="coerce").to_numpy(),
        )
    return result


def bootstrap_group_deltas(
    predictions: dict[str, pd.DataFrame],
    *,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    reference = predictions["E1_official_14d"]
    groups = reference["original_problem"].drop_duplicates().to_numpy(dtype=str)
    raw_groups = reference["original_problem"].to_numpy(dtype=str)
    indices_by_group = {
        group: np.flatnonzero(raw_groups == group)
        for group in groups
    }
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for repeat in range(repeats):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        row_indices = np.concatenate([indices_by_group[str(group)] for group in sampled])
        labels = reference.iloc[row_indices]["label"].astype(bool).to_numpy()
        metrics: dict[str, float] = {}
        for experiment_id, pred in predictions.items():
            binary = pred.iloc[row_indices]["predicted_robust"].astype(bool).to_numpy()
            metrics[experiment_id] = float(balanced_accuracy_score(labels, binary))
        rows.append(
            {
                "bootstrap": int(repeat),
                "e1_ba": metrics["E1_official_14d"],
                "full_ba": metrics["E2A_full_26d"],
                "slope_ba": metrics["E2A_slope_18d"],
                "full_delta_ba": metrics["E2A_full_26d"] - metrics["E1_official_14d"],
                "slope_delta_ba": metrics["E2A_slope_18d"] - metrics["E1_official_14d"],
                "slope_minus_full_ba": metrics["E2A_slope_18d"] - metrics["E2A_full_26d"],
            }
        )
    return pd.DataFrame(rows)


def summarize_bootstrap(table: pd.DataFrame, budget: int, seed: int) -> dict[str, object]:
    full_lo, full_hi = percentile_interval(table["full_delta_ba"])
    slope_lo, slope_hi = percentile_interval(table["slope_delta_ba"])
    return {
        "budget": int(budget),
        "nested_seed": int(seed),
        "bootstrap_repeats": int(len(table)),
        "full_delta_mean": float(table["full_delta_ba"].mean()),
        "full_delta_p025": full_lo,
        "full_delta_p975": full_hi,
        "full_positive_rate": float((table["full_delta_ba"] > 0).mean()),
        "slope_delta_mean": float(table["slope_delta_ba"].mean()),
        "slope_delta_p025": slope_lo,
        "slope_delta_p975": slope_hi,
        "slope_positive_rate": float((table["slope_delta_ba"] > 0).mean()),
    }


def determine_route(
    half_deltas: pd.DataFrame,
    subsampling_summary: pd.DataFrame,
) -> dict[str, object]:
    first512 = half_deltas[
        (half_deltas["subset"] == "first64") & (half_deltas["budget"] == 512)
    ]
    second512 = half_deltas[
        (half_deltas["subset"] == "second64") & (half_deltas["budget"] == 512)
    ]
    if len(first512) and len(second512):
        first_delta = float(first512.iloc[0]["full_delta_ba"])
        second_delta = float(second512.iloc[0]["full_delta_ba"])
        composition_shift = (
            np.sign(first_delta) != np.sign(second_delta)
            or abs(first_delta - second_delta) >= 0.05
        )
    else:
        first_delta = float("nan")
        second_delta = float("nan")
        composition_shift = False

    def stable(budget: int) -> bool:
        rows = subsampling_summary[subsampling_summary["budget"] == budget]
        if len(rows) < 2:
            return False
        return bool(
            (
                (rows["median_full_delta_ba"] >= 0.02)
                & (rows["full_positive_rate"] >= 0.80)
                & (rows["median_full_ba"] >= 0.52)
            ).all()
        )

    stable512 = stable(512)
    stable1024 = stable(1024)
    if stable512:
        route = "INDEPENDENT_STRATIFIED_REPLICATION_128"
    elif stable1024:
        route = "REVISE_TO_BY_1024_AND_EXTEND_2048_EOS"
    else:
        route = "HETEROGENEOUS_DYNAMICS_DIAGNOSIS"

    return {
        "route": route,
        "composition_shift_512": bool(composition_shift),
        "first64_full_delta_512": first_delta,
        "second64_full_delta_512": second_delta,
        "subsampling_stable_512": bool(stable512),
        "subsampling_stable_1024": bool(stable1024),
    }


def sanitize_json_numbers(payload: object) -> object:
    if isinstance(payload, dict):
        return {key: sanitize_json_numbers(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [sanitize_json_numbers(value) for value in payload]
    if isinstance(payload, tuple):
        return [sanitize_json_numbers(value) for value in payload]
    if isinstance(payload, (float, np.floating)) and not math.isfinite(float(payload)):
        return None
    if isinstance(payload, np.integer):
        return int(payload)
    if isinstance(payload, np.bool_):
        return bool(payload)
    return payload


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sanitize_json_numbers(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    budgets = normalize_budgets(args.budgets)
    primary_budgets = tuple(
        budget for budget in normalize_budgets(args.primary_budgets) if budget in budgets
    )
    if not primary_budgets:
        raise ValueError("primary_budgets must overlap budgets")
    seeds = tuple(dict.fromkeys(int(seed) for seed in args.half_seeds))
    if not seeds:
        raise ValueError("at least one half seed is required")
    if args.subsample_repeats < 1 or args.bootstrap_repeats < 1:
        raise ValueError("repeat counts must be positive")

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    frame = official_train.load_feature_data(args.feature_data_path)
    spec = fixed_spec(args)

    first64, second64 = split_stage_halves(frame)
    composition = pd.DataFrame(
        [
            summarize_composition(frame, "all128"),
            summarize_composition(first64, "first64"),
            summarize_composition(second64, "second64"),
        ]
    )
    composition.to_csv(out / "composition_summary.csv", index=False)

    half_rows = []
    half_rows.extend(
        evaluate_subset_nested(
            first64,
            budgets=budgets,
            seeds=seeds,
            args=args,
            spec=spec,
            subset_name="first64",
        )
    )
    half_rows.extend(
        evaluate_subset_nested(
            second64,
            budgets=budgets,
            seeds=seeds,
            args=args,
            spec=spec,
            subset_name="second64",
        )
    )
    half_nested = pd.DataFrame(half_rows)
    half_nested.to_csv(out / "half_nested_metrics.csv", index=False)
    half_deltas = attach_deltas(half_nested)
    half_deltas.to_csv(out / "half_nested_deltas.csv", index=False)

    subsampling = run_subsampling(
        frame,
        budgets=primary_budgets,
        args=args,
        spec=spec,
    )
    subsampling.to_csv(out / "subsampling_distribution.csv", index=False)
    subsampling_summary = summarize_subsampling(subsampling)
    subsampling_summary.to_csv(out / "subsampling_summary.csv", index=False)

    oof_tables: list[pd.DataFrame] = []
    bootstrap_summaries: list[dict[str, object]] = []
    for nested_seed in seeds:
        for budget in budgets:
            predictions: dict[str, pd.DataFrame] = {}
            for experiment_id, features in feature_map(budget).items():
                pred = nested_oof(
                    frame,
                    features=features,
                    spec=spec,
                    args=args,
                    seed=nested_seed,
                )
                pred["budget"] = int(budget)
                pred["experiment"] = experiment_id
                predictions[experiment_id] = pred
                oof_tables.append(pred)
            boot = bootstrap_group_deltas(
                predictions,
                repeats=args.bootstrap_repeats,
                seed=int(args.bootstrap_seed + 10007 * nested_seed + budget),
            )
            boot["budget"] = int(budget)
            boot["nested_seed"] = int(nested_seed)
            boot.to_csv(
                out / f"bootstrap_budget_{budget}_seed_{nested_seed}.csv",
                index=False,
            )
            bootstrap_summaries.append(
                summarize_bootstrap(boot, budget=budget, seed=nested_seed)
            )

    pd.concat(oof_tables, ignore_index=True).to_csv(
        out / "nested_oof_predictions.csv", index=False
    )
    bootstrap_summary = pd.DataFrame(bootstrap_summaries).sort_values(
        ["budget", "nested_seed"]
    )
    bootstrap_summary.to_csv(out / "bootstrap_summary.csv", index=False)

    route = determine_route(half_deltas, subsampling_summary)
    write_json(
        out / "b1_decision.json",
        {
            "experiment": "E2.6-B.1 Sampling Stability Forensics",
            "feature_data_path": str(args.feature_data_path),
            "row_count": int(len(frame)),
            "group_count": int(frame[official_train.GROUP_COLUMN].nunique()),
            "budgets": list(budgets),
            "primary_budgets": list(primary_budgets),
            "half_seeds": list(seeds),
            "subsample_groups": int(args.subsample_groups),
            "subsample_repeats": int(args.subsample_repeats),
            "bootstrap_repeats": int(args.bootstrap_repeats),
            "decision": route,
        },
    )

    print("\n=== Composition ===")
    print(composition.to_string(index=False))
    print("\n=== First64 vs Second64 nested deltas ===")
    print(half_deltas.to_string(index=False))
    print("\n=== Repeated subsampling summary ===")
    print(subsampling_summary.to_string(index=False))
    print("\n=== Group-bootstrap summary ===")
    print(bootstrap_summary.to_string(index=False))
    print("\n=== E2.6-B.1 route ===")
    print(json.dumps(sanitize_json_numbers(route), indent=2, sort_keys=True))
    print(f"\nSaved outputs under: {out}")


if __name__ == "__main__":
    main()
