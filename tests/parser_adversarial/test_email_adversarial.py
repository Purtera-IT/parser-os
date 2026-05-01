from __future__ import annotations

from pathlib import Path

from app.parsers.email_parser import EmailParser


def test_email_adversarial_cases(tmp_path: Path) -> None:
    text = (
        "From: jane.customer@example.com\n"
        "Sent: 2026-02-10 10:00\n"
        "Subject: FW: Scope update\n"
        "\n"
        "Do not proceed at West Wing. Please remove from scope.\n"
        "\n"
        "On 2026-02-01, Jane Customer wrote:\n"
        "> Please include West Wing in rollout.\n"
        "On 2026-01-20, PM wrote:\n"
        ">> Hold off pending quote.\n"
        "\n"
        "-----Original Message-----\n"
        "From: pm.internal@purtera.com\n"
        "Sent: 2026-01-10 08:00\n"
        "Subject: internal notes\n"
        "Internal only note.\n"
    )
    path = tmp_path / "thread.txt"
    path.write_text(text, encoding="utf-8")
    atoms = EmailParser().parse_artifact("proj", "art", path)
    assert atoms
    assert any(atom.atom_type.value in {"customer_instruction", "exclusion"} for atom in atoms)
    assert any(atom.authority_class.value == "quoted_old_email" for atom in atoms if atom.source_refs[0].locator.get("quoted"))
    assert any(
        atom.authority_class.value == "customer_current_authored"
        for atom in atoms
        if atom.source_refs[0].locator.get("quoted") is False
    )
    assert all(atom.source_refs for atom in atoms)
