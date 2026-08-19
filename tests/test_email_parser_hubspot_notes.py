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


def test_email_greeting_is_addressee_tag_not_atom(tmp_path):
    """Body greeting is metadata on sibling atoms — never a standalone atom."""
    atoms = EmailParser().parse_artifact("p", "art_email", _write_eml(tmp_path))
    assert not any(a.raw_text.strip() == "Eddie," for a in atoms)
    assert not any(
        (a.value or {}).get("kind") == "email_addressee" for a in atoms
    )
    stamped = [a for a in atoms if (a.value or {}).get("addressee") == "Eddie"]
    assert stamped, "expected addressee tag on email-sourced atoms"
    assert all((a.value or {}).get("to_greeting") == "Eddie" for a in stamped)
    assert not any(
        a.atom_type == AtomType.scope_item and a.raw_text.strip() == "Eddie,"
        for a in atoms
    )


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


def test_email_list_items_carry_section_context_and_per_line_locators(tmp_path):
    """Include/Exclude bullets carry ``list_section`` polarity and per-line locators."""
    atoms = EmailParser().parse_artifact("p", "art_email", _write_eml(tmp_path))

    okta = next(a for a in atoms if a.raw_text.strip() == "Okta integration")
    assert okta.atom_type == AtomType.scope_item
    assert okta.value.get("list_section") == "include"
    assert okta.value.get("section_header") == "Include"
    assert okta.value.get("kind") == "email_body_line"
    assert okta.value.get("intro") == "By the end of the meeting customer clarified:"
    assert okta.value.get("lead_in") == ["By the end of the meeting customer clarified:"]
    loc = okta.source_refs[0].locator
    assert loc.get("section_path") == [
        "By the end of the meeting customer clarified",
        "Include",
    ]
    assert loc.get("lead_in") == ["By the end of the meeting customer clarified:"]
    assert loc["line_start"] == loc["line_end"]
    assert isinstance(loc["line_start"], int)
    # Framing lead-in is connective tissue — not its own atom.
    assert "By the end of the meeting customer clarified:" not in _texts(atoms)

    buildout = next(a for a in atoms if a.raw_text.strip() == "Network buildout")
    assert buildout.atom_type == AtomType.exclusion
    assert buildout.value.get("list_section") == "exclude"
    assert buildout.value.get("section_header") == "Exclude"
    assert buildout.value.get("kind") == "email_body_line"
    assert buildout.value.get("intro") == "By the end of the meeting customer clarified:"
    ex_loc = buildout.source_refs[0].locator
    assert ex_loc.get("section_path") == [
        "By the end of the meeting customer clarified",
        "Exclude",
    ]
    assert ex_loc.get("lead_in") == ["By the end of the meeting customer clarified:"]
    assert ex_loc["line_start"] == ex_loc["line_end"]

    # Source order: Include items precede Exclude items in the email body.
    include_lines = [
        a.source_refs[0].locator["line_start"]
        for a in atoms
        if a.value.get("list_section") == "include"
    ]
    exclude_lines = [
        a.source_refs[0].locator["line_start"]
        for a in atoms
        if a.value.get("list_section") == "exclude"
    ]
    assert include_lines and exclude_lines
    assert max(include_lines) < min(exclude_lines)

    # Include siblings share one type — no requirement/task mix.
    include_atoms = [a for a in atoms if a.value.get("list_section") == "include"]
    assert include_atoms
    assert all(a.atom_type == AtomType.scope_item for a in include_atoms)


def test_email_courtesy_prose_is_body_context_not_scope(tmp_path):
    """Intro/courtesy prose is email_body_context — never fail-open scope_item."""
    body = "\n".join(
        [
            "From: a@example.com",
            "To: b@example.com",
            "Subject: Scope",
            "",
            "Appreciate you hopping on in such short notice. Attached is a summary of the call.",
            "Include:",
            "  *   Okta integration",
            "Thanks,",
        ]
    )
    p = tmp_path / "courtesy.eml"
    p.write_text(body, encoding="utf-8")
    atoms = EmailParser().parse_artifact("p", "art_c", p)
    intro = [
        a
        for a in atoms
        if a.raw_text.strip().startswith("Appreciate you hopping")
    ]
    assert len(intro) == 1
    assert intro[0].atom_type == AtomType.deal_metadata
    assert intro[0].value.get("kind") == "email_body_context"
    assert intro[0].value.get("role") == "intro"
    assert not any(
        a.atom_type == AtomType.scope_item
        and a.raw_text.strip().startswith("Appreciate you hopping")
        for a in atoms
    )
    assert "Okta integration" in _texts(atoms)


def test_email_cid_equipment_sorts_after_body_include_exclude(tmp_path, monkeypatch):
    """CID equipment rows inherit reading-order line_start after body list items."""
    from app.parsers import email_parser as ep

    cid = "equip-shot-1"
    monkeypatch.setattr(ep, "_ocr_cid_part", lambda part: "Access Point E7 Enterprise × 2\nSwitch Pro Max 48 PoE × 1\n")
    mixed = "=_Mixed_ord"
    related = "=_Related_ord"
    png = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )
    eml = "\r\n".join(
        [
            "From: a@example.com",
            "To: b@example.com",
            "Subject: Equipment list",
            "MIME-Version: 1.0",
            f'Content-Type: multipart/mixed; boundary="{mixed}"',
            "",
            f"--{mixed}",
            f'Content-Type: multipart/related; boundary="{related}"',
            "",
            f"--{related}",
            "Content-Type: text/plain; charset=utf-8",
            "",
            "Below is the full equipment list.",
            "Include:",
            "  *   Okta integration",
            "Exclude:",
            "  *   Network buildout",
            f"[cid:{cid}]",
            f"--{related}",
            "Content-Type: image/png",
            "Content-Transfer-Encoding: base64",
            f"Content-ID: <{cid}>",
            "",
            png,
            f"--{related}--",
            f"--{mixed}--",
            "",
        ]
    )
    p = tmp_path / "order.eml"
    p.write_bytes(eml.encode("utf-8"))
    atoms = EmailParser().parse_artifact("p", "art_ord", p)
    include = next(a for a in atoms if a.raw_text.strip() == "Okta integration")
    exclude = next(a for a in atoms if a.raw_text.strip() == "Network buildout")
    equipment = [a for a in atoms if a.value.get("kind") == "email_cid_equipment_line"]
    assert equipment
    include_line = include.source_refs[0].locator["line_start"]
    exclude_line = exclude.source_refs[0].locator["line_start"]
    equip_lines = [a.source_refs[0].locator["line_start"] for a in equipment]
    assert include_line < exclude_line
    assert exclude_line < min(equip_lines)
    assert all(a.source_refs[0].locator.get("kind") == "email_cid_inline" for a in equipment)
    # All CID rows share the anchor line; order is row_index only.
    assert len({a.source_refs[0].locator["line_start"] for a in equipment}) == 1
    # Connective tissue: equipment-list intro prefixes section_path when present.
    for a in equipment:
        path = a.source_refs[0].locator.get("section_path") or []
        assert path[-1] == "Equipment list"
        assert any("equipment list" in str(x).lower() for x in path)


def test_post_cid_body_note_sorts_after_equipment_rows(tmp_path, monkeypatch):
    """Post-image body notes must sort after CID equipment in document order.

    Universal: OCR rows pin to the CID anchor ``line_start`` and use
    ``row_index`` as the within-image tiebreaker. Expanding line_start by
    row would collide with later body lines (spare-AP notes, etc.).
    """
    from app.parsers import email_parser as ep

    cid = "equip-shot-post"
    monkeypatch.setattr(
        ep,
        "_ocr_cid_part",
        lambda part: (
            "Access Point E7 Enterprise × 2\n"
            "Switch Pro Max 48 PoE × 1\n"
            "Camera G6 Pro Turret × 9\n"
            "Dream Machine Beast × 2\n"
        ),
    )
    mixed = "=_Mixed_post"
    related = "=_Related_post"
    png = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )
    note = (
        '"It is worth noting that we aren\'t using all of the G6 Pro Turret '
        'cameras and we have 1 spare AP as well."'
    )
    eml = "\r\n".join(
        [
            "From: a@example.com",
            "To: b@example.com",
            "Subject: Equipment list",
            "MIME-Version: 1.0",
            f'Content-Type: multipart/mixed; boundary="{mixed}"',
            "",
            f"--{mixed}",
            f'Content-Type: multipart/related; boundary="{related}"',
            "",
            f"--{related}",
            "Content-Type: text/plain; charset=utf-8",
            "",
            "Eddie,",
            "Below is the full equipment list.",
            f"[cid:{cid}]",
            note,
            "Thanks,",
            f"--{related}",
            "Content-Type: image/png",
            "Content-Transfer-Encoding: base64",
            f"Content-ID: <{cid}>",
            "",
            png,
            f"--{related}--",
            f"--{mixed}--",
            "",
        ]
    )
    p = tmp_path / "post_cid.eml"
    p.write_bytes(eml.encode("utf-8"))
    atoms = EmailParser().parse_artifact("p", "art_post", p)

    assert not any(a.raw_text.strip() == "Eddie," for a in atoms)
    equipment = [a for a in atoms if a.value.get("kind") == "email_cid_equipment_line"]
    assert len(equipment) >= 3
    note_atom = next(a for a in atoms if "spare AP" in a.raw_text)
    equip_line = equipment[0].source_refs[0].locator["line_start"]
    note_line = note_atom.source_refs[0].locator["line_start"]
    assert note_line > equip_line
    assert all(a.source_refs[0].locator["line_start"] == equip_line for a in equipment)
    # Document-order key used by Atom Quality: (line_start, row_index).
    equip_keys = [
        (
            a.source_refs[0].locator["line_start"],
            a.source_refs[0].locator.get("row_index", 0),
        )
        for a in equipment
    ]
    note_key = (note_line, -1)
    assert max(equip_keys) < note_key or note_key[0] > max(k[0] for k in equip_keys)
    assert all(a.value.get("addressee") == "Eddie" for a in equipment)
    assert note_atom.value.get("addressee") == "Eddie"
