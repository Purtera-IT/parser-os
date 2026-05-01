from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.eval.domain_certification import certify_domain_pack


def main() -> None:
    parser = argparse.ArgumentParser(description="Run domain pack certification checks and benchmarks.")
    parser.add_argument("--domain-pack", type=Path, required=True, help="Domain pack YAML path")
    parser.add_argument(
        "--fixtures",
        type=Path,
        required=True,
        help="Fixture directory (pack-specific project fixtures and/or gold scenarios)",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output certification report path")
    parser.add_argument("--allow-fail", action="store_true", help="Exit zero even when certification fails")
    args = parser.parse_args()

    report = certify_domain_pack(domain_pack_path=args.domain_pack, fixtures_dir=args.fixtures)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "pack_id": report.pack_id,
                "pack_version": report.pack_version,
                "passed": report.passed,
                "check_count": len(report.checks),
            }
        )
    )
    if not report.passed and not args.allow_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
