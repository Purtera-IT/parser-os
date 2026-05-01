from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.learning.promotion import apply_approved_suggestion


def _load_suggestion(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("suggestions"), list):
        suggestions = [row for row in payload["suggestions"] if isinstance(row, dict)]
        if not suggestions:
            raise ValueError("Suggestion bundle did not contain any suggestion objects.")
        return suggestions[0]
    if isinstance(payload, dict):
        return payload
    raise ValueError("Suggestion payload must be a JSON object.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply an approved rule suggestion (explicit approval required).")
    parser.add_argument("--suggestion", type=Path, required=True, help="Path to suggestion JSON object/bundle")
    parser.add_argument("--approve", action="store_true", help="Required to apply filesystem changes")
    args = parser.parse_args()

    suggestion = _load_suggestion(args.suggestion)
    artifact = apply_approved_suggestion(suggestion_payload=suggestion, approve=args.approve)
    print(artifact.model_dump_json(indent=2))
    if not args.approve and artifact.status != "rejected":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
