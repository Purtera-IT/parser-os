from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.schemas import CompileResult
from app.learning.calibration import apply_calibration


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate calibrator on compile result JSON.")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to calibrator model artifact.")
    parser.add_argument("--abstain-threshold", type=float, default=0.70, help="Abstention threshold.")
    parser.add_argument("compile_result", type=Path, help="Compile result JSON path.")
    args = parser.parse_args()

    result = CompileResult.model_validate_json(args.compile_result.read_text(encoding="utf-8"))
    calibrated = apply_calibration(result, args.model_path, abstain_threshold=args.abstain_threshold)
    packet_probs = [packet.calibrated_confidence for packet in calibrated.packets if packet.calibrated_confidence is not None]
    report = {
        "packet_count": len(calibrated.packets),
        "calibrated_packet_count": len(packet_probs),
        "abstained_packets": len([p for p in calibrated.packets if "calibration_abstain" in p.review_flags]),
        "mean_calibrated_confidence": round(sum(packet_probs) / len(packet_probs), 6) if packet_probs else None,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
