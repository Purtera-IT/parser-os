from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.eval.real_data import summarize_cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize labeled real-data validation cases.")
    parser.add_argument("--root", type=Path, default=Path("real_data_cases"), help="Root cases directory")
    parser.add_argument("--out", type=Path, default=None, help="Optional output JSON file")
    args = parser.parse_args()
    summary = summarize_cases(root_dir=args.root)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
