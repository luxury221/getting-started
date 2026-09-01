#!/usr/bin/env python3
"""Train and export a full Temporal Uncertainty v1 regressor artifact."""

from __future__ import annotations

import sys
from pathlib import Path

SOLUTION_ROOT = Path(__file__).resolve().parents[1]
ROOT = SOLUTION_ROOT.parents[1]
if str(SOLUTION_ROOT) not in sys.path:
    sys.path.insert(0, str(SOLUTION_ROOT))

import train_uncertainty_regressor as official_train  # noqa: E402
import uncertainty_profile.artifact as artifact_module  # noqa: E402
from uncertainty_profile.temporal_metrics import ALL_FEATURE_NAMES  # noqa: E402


_original_default_artifact_path = official_train.default_artifact_path


def _temporal_default_artifact_path(model_id: str) -> Path:
    path = _original_default_artifact_path(model_id)
    return path.with_name(f"{path.stem}_temporal_v1{path.suffix}")


# The official training code intentionally fixes its schema to FEATURE_NAMES.
# Patch the research process in one place rather than modifying upstream files.
official_train.FEATURE_NAMES = ALL_FEATURE_NAMES
artifact_module.FEATURE_NAMES = ALL_FEATURE_NAMES
official_train.default_artifact_path = _temporal_default_artifact_path

if "--results-path" not in sys.argv:
    sys.argv.extend(
        [
            "--results-path",
            str(
                ROOT
                / "data"
                / "uncertainty-profiling"
                / "cv-results-temporal-v1.json"
            ),
        ]
    )


if __name__ == "__main__":
    official_train.main()
