from __future__ import annotations

from app.core.schemas import AtomType
from app.parsers.email_parser import EmailParser


def test_hubspot_note_txt_emits_physical_site_for_compact_state_zip(tmp_path):
    p = tmp_path / "010058-hs-note-111645120815-GECKO ROBOTICS.txt"
    p.write_text(
        "\n".join(
            [
                "HubSpot Note: GECKO ROBOTICS",
                "Date: 2026-06-24T12:16:07.000Z",
                "Author: Patrick Kelly",
                "",
                "GECKO ROBOTICS 100 S COMMONS STE 145 PITTSBURGH, PA15212-5359",
            ]
        ),
        encoding="utf-8",
    )
    atoms = EmailParser().parse_artifact("p", "art_note", p)
    sites = [a for a in atoms if a.atom_type == AtomType.physical_site]
    assert len(sites) == 1
    site = sites[0]
    assert site.value["street_address"] == "100 S COMMONS STE 145"
    assert site.value["city"] == "PITTSBURGH"
    assert site.value["state"] == "PA"
    assert site.value["zip"] == "15212"
    assert site.value["aliases"] == ["GECKO ROBOTICS"]
    assert "site:pittsburgh_pa_15212" in site.entity_keys


# ── Email body-hygiene: greeting / signature / section-label / bullets ──

_PLAINTEXT_EMAIL = "\n".join(
    [
        "From: patrick@purtera-it.com",
        "To: etroci@nmcms.com",
        "Subject: 010058 - Ubiquiti Configuration Gecko Robotics",
        "Date: 2026-06-24T18:18:29.931Z",
        "MIME-Version: 1.0",
        "Content-Type: text/plain; charset=utf-8",
        "",
        "Eddie,",
        "",
        "By the end of the meeting customer clarified:",
        "Include:",
        "",
        "  *   Badge/access control setup",
        "  *   Okta integration",
        "",
        "Exclude:",
        "",
        "  *   Network buildout",
        "  *   General firewall/network configuration",
        "",
        "[cid:07131976-d75d-4133-b5d2-52a8919274ba]",
        "",
        "Thanks,",
        "",
        "Patrick Kelly",
        "",
        "Account Executive",
        "",
        "patrick@purtera-it.com",
        "",
        "770.769.7311",
    ]
)


def _write_eml(tmp_path, name="010058-hs-email-111652731176.eml"):
    p = tmp_path / name
    p.write_text(_PLAINTEXT_EMAIL, encoding="utf-8")
    return p


def _texts(atoms):
    return [a.raw_text.strip() for a in atoms]


def test_email_greeting_not_emitted_as_atom(tmp_path):
    """The salutation ("Eddie,") is envelope chrome, not deal content."""
    atoms = EmailParser().parse_artifact("p", "art_email", _write_eml(tmp_path))
    assert "Eddie," not in _texts(atoms)


def test_email_signature_block_stripped(tmp_path):
    """Everything after the sign-off ("Thanks,") is signature chrome — the
    sender name / title / phone must not become body atoms."""
    atoms = EmailParser().parse_artifact("p", "art_email", _write_eml(tmp_path))
    texts = _texts(atoms)
    for chrome in ("Thanks,", "Patrick Kelly", "Account Executive", "770.769.7311"):
        assert chrome not in texts, f"signature chrome leaked as atom: {chrome!r}"


def test_email_section_labels_not_atoms_and_items_typed(tmp_path):
    """"Include:" / "Exclude:" are headers; the ITEMS beneath them are the
    atoms, and exclude-items are typed as exclusions."""
    atoms = EmailParser().parse_artifact("p", "art_email", _write_eml(tmp_path))
    texts = _texts(atoms)
    assert "Include:" not in texts
    assert "Exclude:" not in texts

    exclusions = {a.raw_text.strip() for a in atoms if a.atom_type == AtomType.exclusion}
    assert "Network buildout" in exclusions
    assert "General firewall/network configuration" in exclusions
    # the exclusion is the ITEM, never the label
    assert "Exclude:" not in exclusions

    scope = {a.raw_text.strip() for a in atoms if a.atom_type == AtomType.scope_item}
    assert "Badge/access control setup" in scope
    assert "Okta integration" in scope


def test_email_bullet_chrome_stripped(tmp_path):
    """The atom is the item text, not the "*   " bullet marker."""
    atoms = EmailParser().parse_artifact("p", "art_email", _write_eml(tmp_path))
    for a in atoms:
        assert not a.raw_text.strip().startswith("*"), a.raw_text


def test_email_cid_marker_line_not_scope_atom(tmp_path):
    """A bare "[cid:…]" inline-attachment marker is MIME chrome, not scope."""
    atoms = EmailParser().parse_artifact("p", "art_email", _write_eml(tmp_path))
    for a in atoms:
        assert not a.raw_text.strip().startswith("[cid:"), a.raw_text
