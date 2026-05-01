from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.eval.failure_taxonomy import FailureRecord, summarize_failure_records


def _extract_failure_records(payload: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict) and "category" in item and "message" in item:
                rows.append(item)
        return rows
    if not isinstance(payload, dict):
        return rows

    direct = payload.get("failure_records")
    if isinstance(direct, list):
        for item in direct:
            if isinstance(item, dict):
                rows.append(item)

    for key in ("scenario_results", "scenarios"):
        children = payload.get(key)
        if not isinstance(children, list):
            continue
        for child in children:
            if not isinstance(child, dict):
                continue
            nested = child.get("failure_records")
            if isinstance(nested, list):
                for item in nested:
                    if isinstance(item, dict):
                        rows.append(item)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize failure taxonomy records across report files.")
    parser.add_argument("inputs", nargs="+", type=Path, help="JSON files containing failure_records")
    parser.add_argument("--out", type=Path, default=None, help="Optional summary output path")
    args = parser.parse_args()

    records: list[FailureRecord] = []
    for path in args.inputs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in _extract_failure_records(payload):
            records.append(FailureRecord.model_validate(row))

    summary = summarize_failure_records(records)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
