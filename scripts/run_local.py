#!/usr/bin/env python3
"""Run the ingestion and scoring pipeline locally."""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "data" / "val-sample" / "input",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=ROOT / "data" / "val-sample" / "reference",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    submission = args.submission.resolve()
    if not (submission / "solution.py").is_file():
        raise SystemExit("submission must be a directory containing solution.py")

    with tempfile.TemporaryDirectory(prefix="aimo-codabench-") as temporary:
        root = Path(temporary)
        ingestion_output = root / "input" / "res"
        scoring_input = root / "input"
        scoring_output = root / "output"
        reference_link = scoring_input / "ref"
        reference_link.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(args.reference_dir.resolve(), reference_link)

        subprocess.run(
            [
                sys.executable,
                str(ROOT / "components" / "ingestion_program" / "ingestion.py"),
                str(args.input_dir.resolve()),
                str(ingestion_output),
                str(submission),
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "components" / "scoring_program" / "scoring.py"),
                str(scoring_input),
                str(scoring_output),
            ],
            check=True,
        )
        scores = json.loads((scoring_output / "scores.json").read_text(encoding="utf-8"))
        print(json.dumps(scores, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
