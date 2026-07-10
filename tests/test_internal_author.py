from __future__ import annotations

from pathlib import Path

from app.core.internal_author import (
    apply_internal_author_elevation,
    classify_author_affiliation,
    is_internal_author,
)
from app.core.schemas import AtomType
from app.parsers.hubspot_note_parser import HubspotNoteParser, parse_hubspot_note_text


def test_is_internal_author_domain_based() -> None:
    assert is_internal_author(author_email="patrick@purtera-it.com")
    assert is_internal_author("Chase Smith <chase@purtera-it.com>")
    assert is_internal_author(author_email="max@optbotai.com")
    assert not is_internal_author("Patrick Kelly")  # name alone is not enough
    assert not is_internal_author(author_email="jon@cdw.com")


def test_classify_author_affiliation() -> None:
    assert classify_author_affiliation(author_email="patrick@purtera-it.com") == "internal"
    assert classify_author_affiliation(author_email="jon@cdw.com") == "external"
    assert classify_author_affiliation("Patrick Kelly") == "unknown"


def test_apply_internal_author_elevation_boosts_confidence() -> None:
    conf, flags, val = apply_internal_author_elevation(confidence=0.84, value={"text": "x"})
    assert conf >= 0.9
    assert "internal_author" in flags
    assert "trusted_internal_source" in flags
    assert val["author_affiliation"] == "internal"


def test_parse_hubspot_note_text_reads_author_email() -> None:
    parsed = parse_hubspot_note_text(
        "\n".join(
            [
                "HubSpot Note: Ubiquiti",
                "HubSpot Note ID: 111635087763",
                "Date: 2026-06-24T14:03:32.895Z",
                "Author: Patrick Kelly",
                "Author-Email: patrick@purtera-it.com",
                "",
                "Do you have resources for switches?",
            ]
        )
    )
    assert parsed["author"] == "Patrick Kelly"
    assert parsed["author_email"] == "patrick@purtera-it.com"
    assert "switches" in parsed["body"]


def test_hubspot_note_parser_stamps_internal_affiliation(tmp_path: Path) -> None:
    p = tmp_path / "010058-hs-note-111635087763-Ubiquiti.txt"
    p.write_text(
        "\n".join(
            [
                "HubSpot Note: Ubiquiti install",
                "HubSpot Note ID: 111635087763",
                "Date: 2026-06-24T14:03:32.895Z",
                "Author: Patrick Kelly",
                "Author-Email: patrick@purtera-it.com",
                "",
                "Do you have resources that can do a Ubiquiti install for some switches?",
            ]
        ),
        encoding="utf-8",
    )
    atoms = HubspotNoteParser().parse_artifact("deal-1", "art_pk", p)
    assert atoms
    for atom in atoms:
        assert atom.value.get("author_affiliation") == "internal"
        assert "internal_author" in atom.review_flags
        assert "trusted_internal_source" in atom.review_flags
        assert atom.confidence >= 0.9
    meta = next(a for a in atoms if a.atom_type == AtomType.deal_metadata)
    assert "author_affiliation=internal" in meta.raw_text
    assert meta.value.get("author_email") == "patrick@purtera-it.com"


def test_hubspot_note_parser_external_author_not_elevated(tmp_path: Path) -> None:
    p = tmp_path / "010058-hs-note-99-customer.txt"
    p.write_text(
        "\n".join(
            [
                "HubSpot Note: Customer note",
                "HubSpot Note ID: 99",
                "Author: Jon Partner",
                "Author-Email: jon@cdw.com",
                "",
                "Please install cameras onsite.",
            ]
        ),
        encoding="utf-8",
    )
    atoms = HubspotNoteParser().parse_artifact("deal-1", "art_ext", p)
    assert atoms
    for atom in atoms:
        assert atom.value.get("author_affiliation") == "external"
        assert "internal_author" not in atom.review_flags
