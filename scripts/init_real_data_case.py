from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.eval.real_data import init_case


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize a local real-data validation case scaffold.")
    parser.add_argument("--case-id", required=True, help="Case identifier (directory name)")
    parser.add_argument("--root", type=Path, default=Path("real_data_cases"), help="Root cases directory")
    parser.add_argument("--notes", default="", help="Case notes")
    parser.add_argument("--expected-risk", action="append", default=[], help="Expected risk tag (repeatable)")
    parser.add_argument(
        "--redaction-status",
        choices=["synthetic", "redacted", "production_unredacted_do_not_share"],
        default="redacted",
    )
    parser.add_argument("--allowed-for-tests", action="store_true", help="Mark case as safe for automated tests")
    args = parser.parse_args()

    created = init_case(
        root_dir=args.root,
        case_id=args.case_id,
        notes=args.notes,
        expected_risks=list(args.expected_risk),
        redaction_status=args.redaction_status,
        allowed_for_tests=args.allowed_for_tests,
    )
    print(json.dumps({"case_dir": str(created)}))


if __name__ == "__main__":
    main()
