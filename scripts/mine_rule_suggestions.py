from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.domain.suggestions import RuleSuggestionFile
from app.learning.rule_miner import collect_mining_inputs, mine_rule_suggestions


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine weak-supervision rule suggestions from labels and compile outputs.")
    parser.add_argument("--labels", type=Path, required=True, help="Root directory containing labels/outputs JSON files")
    parser.add_argument("--out", type=Path, required=True, help="Output JSON path for rule suggestions")
    parser.add_argument("--min-evidence", type=int, default=2, help="Minimum positive examples before suggesting")
    args = parser.parse_args()

    compile_results, packet_labels, candidate_labels, failure_records = collect_mining_inputs(args.labels)
    suggestions = mine_rule_suggestions(
        compile_results=compile_results,
        packet_labels=packet_labels,
        candidate_labels=candidate_labels,
        failure_records=failure_records,
        min_evidence=max(1, args.min_evidence),
    )
    payload = RuleSuggestionFile(
        suggestions=suggestions,
        metadata={
            "labels_root": str(args.labels),
            "compile_result_count": len(compile_results),
            "packet_label_count": len(packet_labels),
            "candidate_label_count": len(candidate_labels),
            "failure_record_count": len(failure_records),
            "suggestion_count": len(suggestions),
            "auto_applied": False,
        },
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(payload.model_dump_json(indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "suggestion_count": len(suggestions)}))


if __name__ == "__main__":
    main()
