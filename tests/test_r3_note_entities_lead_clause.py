"""Live 000132, HubSpot note 110373542233: a one-line " - " bullet list whose
body was HTML-escaped twice ("server &amp;amp; virtualization").

Before the fix a single ``html.unescape`` left "&amp;" in every atom, the lead
clause before the first " - " was located on the (truncated) title line, and
no atom carried a ``char_end`` -- so the viewer could not highlight the lead
clause or any bullet on the decoded line.
"""

from __future__ import annotations

from pathlib import Path

from app.core.textio import decode_html_entities
from app.parsers.email_parser import EmailParser
from app.parsers.hubspot_note_parser import HubspotNoteParser

LEAD_RAW = (
    "Maintenance and support of the technical IT infrastructure in the areas of "
    "network, server &amp;amp; virtualization"
)
LEAD = (
    "Maintenance and support of the technical IT infrastructure in the areas of "
    "network, server & virtualization"
)
BULLETS = [
    "Support for physical network components (LAN, WLAN)",
    "Support for physical server systems",
    "Support for virtualization platforms",
    "Troubleshooting and incident handling",
    "24/7 on-call availability",
]
BODY_RAW = LEAD_RAW + "".join(f" - {b}" for b in BULLETS)
BODY = LEAD + "".join(f" - {b}" for b in BULLETS)
# The deal-uploads shape: the note's title (HubSpot truncates it) over the body.
TITLE_RAW = LEAD_RAW + "..."


def _write_note(tmp_path: Path, *, with_header: bool) -> Path:
    if with_header:
        text = (
            f"HubSpot Note: {TITLE_RAW}\n"
            "HubSpot Note ID: 110373542233\n"
            "Date: 2025-03-01\n"
            "Author: Jane Doe\n"
            "\n"
            f"{BODY_RAW}\n"
        )
    else:
        text = f"{TITLE_RAW}\n\n{BODY_RAW}\n"
    p = tmp_path / "hubspot_note_110373542233.txt"
    p.write_text(text, encoding="utf-8")
    return p


def _body_atoms(path: Path):
    atoms = HubspotNoteParser().parse_artifact("p", "a", path)
    return [a for a in atoms if (a.value or {}).get("kind") == "hubspot_note_body"]


def _by_text(atoms):
    out: dict[str, object] = {}
    for a in atoms:
        out.setdefault(" ".join(a.raw_text.split()), a)
    return out


def test_double_escaped_entity_is_decoded_in_every_note_atom(tmp_path):
    for with_header in (True, False):
        atoms = _body_atoms(_write_note(tmp_path, with_header=with_header))
        texts = _by_text(atoms)
        assert LEAD in texts, sorted(texts)
        for b in BULLETS:
            assert b in texts, (b, sorted(texts))
        for t in texts:
            assert "&amp;" not in t and "amp;" not in t, t


def test_lead_clause_is_its_own_atom_on_the_body_line_not_the_title(tmp_path):
    path = _write_note(tmp_path, with_header=False)
    lines = path.read_text(encoding="utf-8").splitlines()
    body_line = lines.index(BODY_RAW) + 1
    lead = _by_text(_body_atoms(path))[LEAD]
    loc = lead.source_refs[0].locator
    # The title line also starts with the lead clause; the atom is the body's.
    assert loc["line_start"] == body_line, loc


def test_char_offsets_slice_the_decoded_body_to_the_atom_text(tmp_path):
    for with_header in (True, False):
        path = _write_note(tmp_path, with_header=with_header)
        decoded_lines = decode_html_entities(path.read_text(encoding="utf-8")).splitlines()
        atoms = _body_atoms(path)
        assert len(_by_text(atoms)) == 1 + len(BULLETS)
        for a in atoms:
            loc = a.source_refs[0].locator
            line = decoded_lines[loc["line_start"] - 1]
            assert line == BODY
            assert "char_end" in loc, (a.raw_text, loc)
            assert line[loc["char_start"]:loc["char_end"]] == a.raw_text, (a.raw_text, loc)


def test_email_body_entities_are_decoded_before_atoms(tmp_path):
    p = tmp_path / "scope.txt"
    p.write_text(
        "From: a@example.com\n"
        "Sent: Monday, March 3, 2025 9:00 AM\n"
        "To: c@example.com\n"
        "Subject: Scope\n"
        "\n"
        "Please quote network, server &amp;amp; virtualization support for the 3 sites.\n",
        encoding="utf-8",
    )
    texts = [a.raw_text for a in EmailParser().parse_artifact("p", "a", p)]
    assert any("server & virtualization support" in t for t in texts), texts
    assert not any("&amp;" in t for t in texts), texts


def test_decode_html_entities_is_repeated_and_strict():
    assert decode_html_entities("a &amp;amp;amp; b") == "a & b"
    assert decode_html_entities("x &#38; y &#x26; z") == "x & y & z"
    # Unterminated legacy names stay: a URL's "&region=" is not "(R)ion=".
    assert decode_html_entities("https://h/?a=1&region=us AT&T") == "https://h/?a=1&region=us AT&T"
    # A line-break reference would add a line; it stays as written.
    assert decode_html_entities("a&#10;b") == "a&#10;b"
