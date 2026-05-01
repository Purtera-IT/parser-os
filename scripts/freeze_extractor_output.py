from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.experiments.freeze import freeze_experiment_output


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze approved experimental extraction output into deterministic artifacts.")
    parser.add_argument("--experiment", type=Path, required=True, help="Experiment report JSON file")
    parser.add_argument("--approve", action="store_true", help="Required to write frozen artifacts")
    parser.add_argument("--out-dir", type=Path, default=None, help="Optional freeze output directory")
    args = parser.parse_args()

    result = freeze_experiment_output(
        experiment_path=args.experiment,
        approve=args.approve,
        out_dir=args.out_dir,
    )
    print(result.model_dump_json(indent=2))
    if not args.approve and result.status != "applied":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
