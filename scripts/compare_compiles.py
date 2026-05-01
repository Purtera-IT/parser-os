from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.diffing import diff_compile_results
from app.core.schemas import CompileResult


def _read_compile(path: Path) -> CompileResult:
    return CompileResult.model_validate_json(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two compile outputs and simulate packet invalidation.")
    parser.add_argument("--before", type=Path, required=True, help="Path to older compile JSON")
    parser.add_argument("--after", type=Path, required=True, help="Path to newer compile JSON")
    parser.add_argument("--out", type=Path, required=True, help="Path for compile diff JSON output")
    args = parser.parse_args()

    before = _read_compile(args.before)
    after = _read_compile(args.after)
    diff = diff_compile_results(before, after)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(diff.model_dump_json(indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "before_compile_id": diff.before_compile_id,
                "after_compile_id": diff.after_compile_id,
                "invalidated_packet_ids": len(diff.invalidated_packet_ids),
                "blast_radius_summary": diff.blast_radius_summary,
            }
        )
    )


if __name__ == "__main__":
    main()
