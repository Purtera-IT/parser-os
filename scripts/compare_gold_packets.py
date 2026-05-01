"""Compare labels/gold_packets.json to outputs/compile_result.json for any real-data case directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.eval.gold_compare import compare_case_directory, write_comparison_outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare labels/gold_packets.json vs outputs/compile_result.json under a case directory.",
    )
    parser.add_argument(
        "--case-dir",
        type=Path,
        required=True,
        help="Path to real_data_cases/<CASE_ID> (must contain labels/ and outputs/)",
    )
    args = parser.parse_args()
    case_dir = args.case_dir.resolve()
    result = compare_case_directory(case_dir)
    json_path, md_path = write_comparison_outputs(case_dir, result)
    print(json.dumps({"gold_comparison_json": str(json_path), "gold_comparison_md": str(md_path), **result.to_dict()}, indent=2))


if __name__ == "__main__":
    main()
