from __future__ import annotations

from app.core.schemas import AtomType, AuthorityClass
from app.parsers.email_parser import EmailParser


def test_email_thread_parser_authority_and_locators(tmp_path) -> None:
    file_path = tmp_path / "customer_email.txt"
    file_path.write_text(
        (
            "From: client@acme.com\n"
            "Sent: Tue, 12 Mar 2026 17:30\n"
            "Subject: Scope updates\n"
            "\n"
            "Please remove West Wing from scope. Also, Main Campus requires escort access after 5pm.\n"
            "\n"
            "-----Original Message-----\n"
            "From: client@acme.com\n"
            "Sent: Mon, 11 Mar 2026 09:15\n"
            "Subject: Initial request\n"
            "> Please include West Wing in the camera rollout.\n"
        ),
        encoding="utf-8",
    )
    atoms = EmailParser().parse_artifact(
        project_id="proj_1",
        artifact_id="art_1",
        path=file_path,
    )

    assert atoms
    assert all(atom.source_refs for atom in atoms)

    exclusion_atoms = [a for a in atoms if a.atom_type == AtomType.exclusion]
    assert any("west wing" in atom.normalized_text for atom in exclusion_atoms)

    quoted_include_atoms = [
        a for a in atoms if "include west wing" in a.normalized_text and a.source_refs[0].locator.get("quoted") is True
    ]
    assert quoted_include_atoms
    assert all(a.authority_class == AuthorityClass.quoted_old_email for a in quoted_include_atoms)

    current_exclusions = [
        a
        for a in exclusion_atoms
        if "west wing" in a.normalized_text and a.source_refs[0].locator.get("quoted") is False
    ]
    assert current_exclusions
    assert all(a.authority_class == AuthorityClass.customer_current_authored for a in current_exclusions)

    constraint_atoms = [a for a in atoms if a.atom_type == AtomType.constraint]
    assert any("escort" in a.normalized_text or "after 5pm" in a.normalized_text for a in constraint_atoms)

    quoted_atoms = [a for a in atoms if a.source_refs[0].locator.get("quoted") is True]
    assert quoted_atoms
    assert all(a.authority_class != AuthorityClass.customer_current_authored for a in quoted_atoms)
