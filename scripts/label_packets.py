from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.eval.real_data import write_packet_label_skeleton


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate packet label skeleton for a compiled real-data case.")
    parser.add_argument("--case-id", required=True, help="Case identifier")
    parser.add_argument("--root", type=Path, default=Path("real_data_cases"), help="Root cases directory")
    args = parser.parse_args()
    path = write_packet_label_skeleton(root_dir=args.root, case_id=args.case_id)
    print(json.dumps({"packet_labels_path": str(path)}))


if __name__ == "__main__":
    main()
