from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.eval.coverage import build_coverage_report, load_compile_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate evidence coverage diagnostics from a compile_result.json file.")
    parser.add_argument("--compile-result", type=Path, required=True, help="Path to compile result JSON.")
    parser.add_argument("--out", type=Path, required=True, help="Output coverage report JSON path.")
    args = parser.parse_args()

    payload = load_compile_payload(args.compile_result)
    report = build_coverage_report(payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "project_id": report.project_id,
                "artifact_count": len(report.artifact_reports),
                "overall_coverage_rate": report.overall_coverage_rate,
            }
        )
    )


if __name__ == "__main__":
    main()
