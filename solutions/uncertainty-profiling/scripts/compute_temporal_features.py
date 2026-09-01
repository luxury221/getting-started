#!/usr/bin/env python3
"""Generate the official 14 uncertainty features plus Temporal Uncertainty v1."""

from __future__ import annotations

import sys
from pathlib import Path

SOLUTION_ROOT = Path(__file__).resolve().parents[1]
if str(SOLUTION_ROOT) not in sys.path:
    sys.path.insert(0, str(SOLUTION_ROOT))

import compute_uncertainty_features as official  # noqa: E402

from uncertainty_profile.temporal_extraction import (  # noqa: E402
    iter_temporal_generation_feature_batches,
)
from uncertainty_profile.temporal_metrics import ALL_FEATURE_NAMES  # noqa: E402


_original_safe_model_id = official.safe_model_id


def _temporal_safe_model_id(model_id: str) -> str:
    return f"{_original_safe_model_id(model_id)}_temporal_v1"


# Reuse the official dataset validation, resumable cache, sharding, provenance,
# and CLI. Only the feature schema and collector are replaced.
official.FEATURE_NAMES = ALL_FEATURE_NAMES
official.iter_generation_feature_batches = iter_temporal_generation_feature_batches
official.safe_model_id = _temporal_safe_model_id


if __name__ == "__main__":
    official.main()
