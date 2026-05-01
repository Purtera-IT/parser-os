from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.eval.real_data import compile_case


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile a local real-data validation case.")
    parser.add_argument("--case-id", required=True, help="Case identifier")
    parser.add_argument("--root", type=Path, default=Path("real_data_cases"), help="Root cases directory")
    parser.add_argument(
        "--domain-pack",
        type=str,
        default=None,
        help="Override domain pack from case_manifest (e.g. copper_cabling)",
    )
    args = parser.parse_args()
    dp = args.domain_pack
    if dp is not None and not str(dp).strip():
        dp = None
    summary = compile_case(root_dir=args.root, case_id=args.case_id, domain_pack=dp)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
