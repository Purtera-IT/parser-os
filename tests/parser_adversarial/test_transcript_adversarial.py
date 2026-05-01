from __future__ import annotations

from pathlib import Path

from app.core.compiler import compile_project
from app.parsers.transcript_parser import TranscriptParser


def test_transcript_adversarial_cases(tmp_path: Path) -> None:
    path = tmp_path / "kickoff_transcript.txt"
    path.write_text(
        "\n".join(
            [
                "[00:00:01] Jane Customer: Please remove West Wing from scope.",
                "[00:00:42] Unknown: maybe add 5 cameras at Main Campus.",
                "Decisions:",
                "- Proceed with main campus baseline.",
                "Action Items:",
                "- Customer to provide access details.",
                "Open Questions:",
                "- MDF badge access?",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    atoms = TranscriptParser().parse_artifact("proj", "art", path)
    assert atoms
    assert any(atom.atom_type.value == "customer_instruction" for atom in atoms)
    assert any(atom.atom_type.value == "open_question" for atom in atoms)
    assert any(atom.review_status.value == "needs_review" for atom in atoms if "maybe add 5" in atom.normalized_text)

    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "kickoff_transcript.txt").write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    result = compile_project(project_dir, project_id="adv_transcript", allow_unverified_receipts=True)
    assert any(packet.family.value == "missing_info" for packet in result.packets)
