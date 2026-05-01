from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.schemas import CompileResult
from app.learning.calibration import train_calibrator


def main() -> None:
    parser = argparse.ArgumentParser(description="Train confidence calibrator from labels + compile results.")
    parser.add_argument("--labels", type=Path, required=True, help="Path to labels JSON.")
    parser.add_argument("--model-out", type=Path, required=True, help="Output path for joblib model artifact.")
    parser.add_argument("compile_results", nargs="+", type=Path, help="Compile result JSON files.")
    args = parser.parse_args()

    compile_results = [
        CompileResult.model_validate_json(path.read_text(encoding="utf-8"))
        for path in args.compile_results
    ]
    report = train_calibrator(args.labels, compile_results, args.model_out)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
