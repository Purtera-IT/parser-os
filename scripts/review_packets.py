from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.review.schemas import PacketReview
from app.review.store import load_reviews, save_reviews, upsert_reviews


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _bool_prompt(prompt: str) -> bool | None:
    while True:
        value = input(f"{prompt} [y/n/skip]: ").strip().lower()
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        if value in {"", "s", "skip"}:
            return None
        print("Enter y, n, or skip.")


def _snippet(text: str, limit: int = 160) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _receipt_summary(atom: dict) -> str:
    receipts = atom.get("receipts") or []
    if not receipts:
        return "none"
    counts: dict[str, int] = {}
    for receipt in receipts:
        status = str(receipt.get("replay_status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return ", ".join(f"{k}:{counts[k]}" for k in sorted(counts))


def _show_packet(packet: dict, atom_by_id: dict[str, dict]) -> None:
    risk = packet.get("risk") or {}
    cert = packet.get("certificate") or {}
    print("\n" + "=" * 80)
    print(f"packet_id: {packet.get('id')}")
    print(f"family: {packet.get('family')}  anchor: {packet.get('anchor_key')}")
    print(
        f"status: {packet.get('status')}  severity: {risk.get('severity')}  "
        f"risk_score: {risk.get('risk_score')}"
    )
    print(f"reason: {_snippet(str(packet.get('reason', '')))}")
    print(f"certificate.existence_reason: {_snippet(str(cert.get('existence_reason', '')))}")
    print(f"certificate.governing_rationale: {_snippet(str(cert.get('governing_rationale', '')))}")

    print("minimal_sufficient_atoms:")
    for atom_id in cert.get("minimal_sufficient_atom_ids") or []:
        atom = atom_by_id.get(atom_id, {})
        print(
            f"  - {atom_id} | {_snippet(str(atom.get('raw_text', 'missing atom')))} | receipts={_receipt_summary(atom)}"
        )

    print("contradicting_atoms:")
    for atom_id in packet.get("contradicting_atom_ids") or []:
        atom = atom_by_id.get(atom_id, {})
        print(
            f"  - {atom_id} | {_snippet(str(atom.get('raw_text', 'missing atom')))} | receipts={_receipt_summary(atom)}"
        )


def _filter_packets(
    packets: list[dict],
    *,
    family: str | None,
    severity: str | None,
    needs_review_only: bool,
    limit: int | None,
) -> list[dict]:
    rows = list(packets)
    if family:
        rows = [packet for packet in rows if packet.get("family") == family]
    if severity:
        rows = [packet for packet in rows if (packet.get("risk") or {}).get("severity") == severity]
    if needs_review_only:
        rows = [packet for packet in rows if packet.get("status") == "needs_review"]
    rows = sorted(rows, key=lambda packet: (packet.get("family", ""), packet.get("anchor_key", ""), packet.get("id", "")))
    if limit is not None:
        rows = rows[:limit]
    return rows


def _build_review(packet: dict, *, non_interactive: bool) -> PacketReview:
    if non_interactive:
        return PacketReview(
            packet_id=str(packet.get("id")),
            family=str(packet.get("family")),
            anchor_key=str(packet.get("anchor_key")),
            correct_packet=None,
            correct_governing_atom=None,
            correct_severity=None,
            should_be_status=None,
            missing_evidence=None,
            false_positive_reason=None,
            reviewer_notes="",
            reviewed_at=_now_iso(),
        )

    correct_packet = _bool_prompt("correct packet?")
    correct_governing = _bool_prompt("correct governing atom?")
    correct_severity = _bool_prompt("correct severity?")
    should_be_status = None
    missing_evidence = None
    false_positive_reason = None
    if correct_packet is False:
        false_positive_reason = input("false positive reason (optional): ").strip() or None
        should_be_status = input("should be status (optional): ").strip() or None
        missing_evidence = input("missing evidence (optional): ").strip() or None
    notes = input("notes: ").strip()
    return PacketReview(
        packet_id=str(packet.get("id")),
        family=str(packet.get("family")),
        anchor_key=str(packet.get("anchor_key")),
        correct_packet=correct_packet,
        correct_governing_atom=correct_governing,
        correct_severity=correct_severity,
        should_be_status=should_be_status,
        missing_evidence=missing_evidence,
        false_positive_reason=false_positive_reason,
        reviewer_notes=notes,
        reviewed_at=_now_iso(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Terminal packet review console for compile_result.json")
    parser.add_argument("compile_result", type=Path, help="Path to compile_result JSON")
    parser.add_argument("--out", type=Path, required=True, help="Path to packet review labels JSON")
    parser.add_argument("--family", default=None, help="Filter by packet family")
    parser.add_argument("--severity", default=None, choices=["low", "medium", "high", "critical"], help="Filter by severity")
    parser.add_argument("--needs-review-only", action="store_true", help="Only show needs_review packets")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of packets")
    parser.add_argument("--non-interactive", action="store_true", help="Write skeleton labels without prompts")
    args = parser.parse_args()

    payload = json.loads(args.compile_result.read_text(encoding="utf-8"))
    packets = payload.get("packets", [])
    atoms = payload.get("atoms", [])
    atom_by_id = {str(atom.get("id")): atom for atom in atoms}
    selected = _filter_packets(
        packets,
        family=args.family,
        severity=args.severity,
        needs_review_only=args.needs_review_only,
        limit=args.limit,
    )

    if not selected:
        print("No packets match the selected filters.")
        save_reviews(args.out, [], metadata={"source_compile_result": str(args.compile_result), "selected_count": 0})
        return

    reviews: list[PacketReview] = []
    for packet in selected:
        _show_packet(packet, atom_by_id)
        reviews.append(_build_review(packet, non_interactive=args.non_interactive))

    existing = load_reviews(args.out)
    merged = upsert_reviews(existing, reviews)
    save_reviews(
        args.out,
        merged,
        metadata={
            "source_compile_result": str(args.compile_result),
            "selected_count": len(selected),
            "non_interactive": bool(args.non_interactive),
        },
    )
    print(f"Saved {len(reviews)} review rows to {args.out}")


if __name__ == "__main__":
    main()
