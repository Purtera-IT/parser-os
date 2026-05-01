"""Compare COPPER_001 compile output to optional labels/gold_packets.json and built-in checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.eval.gold import copper_001_material_gold_checks, load_gold


def main() -> None:
    parser = argparse.ArgumentParser(description="COPPER_001 gold-style checks on compile_result.json")
    parser.add_argument(
        "--compile-result",
        type=Path,
        required=True,
        help="Path to outputs/compile_result.json",
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=None,
        help="Optional labels/gold_packets.json (GoldScenario)",
    )
    args = parser.parse_args()
    payload = json.loads(args.compile_result.read_text(encoding="utf-8"))
    report: dict = {"checks": copper_001_material_gold_checks(payload)}
    if args.gold and args.gold.is_file():
        gold = load_gold(args.gold)
        report["gold_scenario_id"] = gold.scenario_id
        report["gold_expected_packet_count"] = len(gold.expected_packets)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
