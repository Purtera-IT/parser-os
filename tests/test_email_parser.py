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


def _eml(html_body: str) -> bytes:
    return (
        "From: a@x.com\r\nTo: b@y.com\r\nSubject: t\r\nMIME-Version: 1.0\r\n"
        'Content-Type: multipart/alternative; boundary="B"\r\n\r\n'
        "--B\r\nContent-Type: text/plain; charset=utf-8\r\n\r\nplain fallback\r\n\r\n"
        "--B\r\nContent-Type: text/html; charset=utf-8\r\n\r\n"
        f"<html><body>{html_body}</body></html>\r\n--B--\r\n"
    ).encode()


def _rows(tmp_path, html_body):
    from app.parsers.email_body import _extract_email_text
    p = tmp_path / "m.eml"
    p.write_bytes(_eml(html_body))
    return [l for l in _extract_email_text(p).splitlines() if " | " in l]


def test_data_table_keeps_its_rows(tmp_path):
    rows = _rows(tmp_path, "<table><tr><td>Arkansas</td><td>327</td></tr>"
                           "<tr><td>Ohio</td><td>2,516</td></tr></table>")
    assert rows == ["Arkansas | 327", "Ohio | 2,516"]


def test_signature_table_does_not_become_rows(tmp_path):
    # Mail clients lay contact blocks out in <table>; reading tables faithfully
    # would otherwise assert a signature as structured data, and it repeats on
    # every message in the thread.
    rows = _rows(tmp_path, "<table><tr><td>Jane Doe</td><td>Account Director</td></tr>"
                           "<tr><td>Mobile: +1 774 292 9763</td><td>jane@x.com</td></tr></table>")
    assert rows == []


def test_disclaimer_table_does_not_become_rows(tmp_path):
    rows = _rows(tmp_path, "<table><tr><td>This e-mail and any files transmitted with it "
                           "are confidential and intended solely for the addressee.</td></tr></table>")
    assert rows == []


def test_a_single_number_is_enough_to_keep_a_contact_shaped_table(tmp_path):
    # A rate card carries contact-ish words too; the numeric column is what
    # separates it from a signature.
    rows = _rows(tmp_path, "<table><tr><td>Onsite Engineer</td><td>82.55</td></tr></table>")
    assert rows == ["Onsite Engineer | 82.55"]


def test_both_parsers_share_one_reader():
    from app.parsers.email_parser import _extract_email_text as a
    from app.parsers.segmenters import _extract_email_text as b
    assert a is b
