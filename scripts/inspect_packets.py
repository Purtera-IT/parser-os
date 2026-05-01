from __future__ import annotations

import argparse
import json
from pathlib import Path


def _fmt(value: object, width: int) -> str:
    text = str(value)
    if len(text) > width:
        return text[: width - 3] + "..."
    return text.ljust(width)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect packet summary from compiled JSON")
    parser.add_argument("json_path", type=Path, help="Path to compiled result JSON")
    args = parser.parse_args()

    payload = json.loads(args.json_path.read_text(encoding="utf-8"))
    packets = payload.get("packets", [])
    packets = sorted(
        packets,
        key=lambda packet: (
            (packet.get("risk") or {}).get("review_priority", 5),
            -float((packet.get("risk") or {}).get("risk_score", 0.0)),
            packet.get("family", ""),
            packet.get("anchor_key", ""),
            packet.get("id", ""),
        ),
    )

    headers = [
        ("family", 18),
        ("anchor", 24),
        ("status", 12),
        ("severity", 10),
        ("risk_score", 10),
        ("estimated_cost_exposure", 24),
        ("reason", 56),
    ]

    print(" ".join(_fmt(name, width) for name, width in headers))
    print(" ".join("-" * width for _, width in headers))

    for packet in packets:
        risk = packet.get("risk") or {}
        row = [
            packet.get("family", ""),
            packet.get("anchor_key", ""),
            packet.get("status", ""),
            risk.get("severity", ""),
            risk.get("risk_score", ""),
            risk.get("estimated_cost_exposure", ""),
            packet.get("reason", ""),
        ]
        print(" ".join(_fmt(value, width) for value, (_, width) in zip(row, headers)))


if __name__ == "__main__":
    main()
