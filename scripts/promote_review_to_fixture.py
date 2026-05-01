from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.learning.promotion import promote_review_to_fixture


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote reviewed packet/candidate signals into a regression fixture proposal.")
    parser.add_argument("--review-labels", type=Path, required=True, help="Path to review labels JSON")
    parser.add_argument("--compile-result", type=Path, required=True, help="Path to compile result JSON")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for generated fixture proposal")
    args = parser.parse_args()

    artifact = promote_review_to_fixture(
        review_labels_path=args.review_labels,
        compile_result_path=args.compile_result,
        out_dir=args.out_dir,
    )
    print(
        json.dumps(
            {
                "promotion_id": artifact.promotion_id,
                "status": artifact.status,
                "proposed_files": artifact.proposed_files,
            }
        )
    )


if __name__ == "__main__":
    main()
