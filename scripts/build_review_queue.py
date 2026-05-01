from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.learning.active_learning import build_active_learning_queue
from app.review.schemas import ReviewQueueFile


def main() -> None:
    parser = argparse.ArgumentParser(description="Build active learning review queue from compile_result.json")
    parser.add_argument("--compile-result", type=Path, required=True, help="Path to compile result JSON")
    parser.add_argument("--out", type=Path, required=True, help="Output path for review queue JSON")
    parser.add_argument("--max-items", type=int, default=100, help="Maximum number of queue items")
    args = parser.parse_args()

    payload = json.loads(args.compile_result.read_text(encoding="utf-8"))
    items = build_active_learning_queue(payload, max_items=args.max_items)
    queue_file = ReviewQueueFile(
        items=items,
        metadata={
            "source_compile_result": str(args.compile_result),
            "item_count": len(items),
            "max_items": args.max_items,
        },
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(queue_file.model_dump_json(indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "item_count": len(items)}))


if __name__ == "__main__":
    main()
