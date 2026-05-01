from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.eval.benchmark import run_packetizer_benchmark, threshold_failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Run packetizer benchmark against gold scenarios")
    parser.add_argument("--fixtures", type=Path, required=True, help="Gold scenarios directory")
    parser.add_argument("--out", type=Path, required=True, help="Output benchmark JSON report")
    parser.add_argument("--allow-fail", action="store_true", help="Always exit zero even when thresholds fail")
    args = parser.parse_args()

    report = run_packetizer_benchmark(args.fixtures)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    failures = threshold_failures(report)
    print(
        json.dumps(
            {
                "scenario_count": report.scenario_count,
                "compile_success_rate": report.aggregate_metrics.get("compile_success_rate", 0.0),
                "threshold_failures": failures,
            }
        )
    )
    if failures and not args.allow_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
