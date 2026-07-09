"""Universal Include/Exclude connective tissue + verbatim evidence rules."""
from __future__ import annotations

from pathlib import Path

from app.core.schemas import AtomType
from app.core.task_atom_backfill import backfill_quote_task_atoms
from app.parsers.email_parser import (
    EmailParser,
    _hardware_atoms_from_equipment_text,
    _is_ocr_junk_equipment_line,
)

_GECKO_BODY = "\n".join(
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
        "Appreciate you hopping on in such short notice. Attached is a summary of the information "
        "needed to provide a quote as well as the transcript of our conversation with the customer. "
        "This is one that if we're fast on, definitely feel confident in winning. Let us know if "
        "there is anything on our side that we can do to help or speed up the process.",
        "",
        "Below is the full equipment list. One hard requirement for him is Otka integration.",
        "By the end of the meeting customer clarified:",
        "Include:",
        "",
        "  *   Badge/access control setup",
        "  *   UID Enterprise setup",
        "  *   Okta integration",
        "  *   Camera configuration",
        "  *   Knowledge transfer / walking him through the setup",
        "",
        "Exclude:",
        "",
        "  *   Network buildout",
        "  *   General firewall/network configuration",
        "",
        "Customer specifically said:",
        '"Network build out does not need to be built into this. I can do that. '
        "It's really just the badging and integrating with Okta and all that that I need. And cameras.\"",
        "",
        "Thanks,",
        "Patrick Kelly",
    ]
)


def _write_eml(tmp_path: Path) -> Path:
    p = tmp_path / "gecko-include.eml"
    p.write_text(_GECKO_BODY, encoding="utf-8")
    return p


def test_include_bullets_carry_lead_in_and_are_scope_item(tmp_path: Path) -> None:
    atoms = EmailParser().parse_artifact("p", "art_email", _write_eml(tmp_path))
    includes = [
        a
        for a in atoms
        if isinstance(a.value, dict) and a.value.get("list_section") == "include"
    ]
    assert len(includes) == 5
    for a in includes:
        assert a.atom_type == AtomType.scope_item
        lead = a.value.get("lead_in") or []
        assert any("customer clarified" in str(x).lower() for x in lead)
        assert a.value.get("intro") and "customer clarified" in a.value["intro"].lower()
        path = (a.source_refs[0].locator or {}).get("section_path") or []
        assert path[-1] == "Include"
        assert any("clarified" in str(x).lower() for x in path)
        loc_lead = (a.source_refs[0].locator or {}).get("lead_in") or []
        assert any("clarified" in str(x).lower() for x in loc_lead)
        assert "guided handoff" not in a.raw_text.lower()
        assert "ubiquiti configuration" not in a.raw_text.lower()

    knowledge = next(a for a in includes if "Knowledge transfer" in a.raw_text)
    assert knowledge.raw_text == "Knowledge transfer / walking him through the setup"


def test_exclude_bullets_are_exclusions_with_same_lead_in(tmp_path: Path) -> None:
    atoms = EmailParser().parse_artifact("p", "art_email", _write_eml(tmp_path))
    excludes = [
        a
        for a in atoms
        if isinstance(a.value, dict) and a.value.get("list_section") == "exclude"
    ]
    assert len(excludes) == 2
    for a in excludes:
        assert a.atom_type == AtomType.exclusion
        lead = a.value.get("lead_in") or []
        assert any("customer clarified" in str(x).lower() for x in lead)


def test_customer_quote_is_customer_instruction_not_requirement(tmp_path: Path) -> None:
    atoms = EmailParser().parse_artifact("p", "art_email", _write_eml(tmp_path))
    quotes = [
        a
        for a in atoms
        if a.raw_text.strip().startswith('"') and "Network build out" in a.raw_text
    ]
    assert quotes
    assert all(a.atom_type == AtomType.customer_instruction for a in quotes)
    assert not any(a.atom_type == AtomType.requirement for a in quotes)


def test_backfill_does_not_hallucinate_include_umbrellas(tmp_path: Path) -> None:
    atoms = EmailParser().parse_artifact("p", "art_email", _write_eml(tmp_path))
    out, count = backfill_quote_task_atoms(atoms, project_id="gecko")
    texts = [a.raw_text for a in out]
    assert "Knowledge transfer / guided handoff" not in texts
    assert "Ubiquiti configuration / install support" not in texts
    # Include-list items themselves must not become tasks.
    include_tasks = [
        a
        for a in out
        if a.atom_type == AtomType.task
        and isinstance(a.value, dict)
        and a.value.get("list_section") == "include"
    ]
    assert include_tasks == []
    assert count == 0 or all(
        "guided handoff" not in a.raw_text and "Ubiquiti configuration" not in a.raw_text
        for a in out
        if a.atom_type == AtomType.task
    )


def test_ocr_junk_equipment_lines_repaired_not_dropped() -> None:
    from app.parsers.email_parser import _repair_ocr_equipment_line

    # Truncated / OCR-noise rows repair into recoverable product+qty lines.
    assert not _is_ocr_junk_equipment_line("I Access Reader Pro Juncti ... 5")
    assert not _is_ocr_junk_equipment_line("Camera Al Multi Sensor 4 1")
    assert _repair_ocr_equipment_line("I Access Reader Pro Juncti ... 5") == "Access Reader Pro Juncti 5"
    assert _repair_ocr_equipment_line("Camera Al Multi Sensor 4 1") == "Camera AI Multi Sensor 4 1"
    assert not _is_ocr_junk_equipment_line("Access G3 Reader 4")
    assert not _is_ocr_junk_equipment_line("Protect All-In-One Sensor 2")
    # Real HubSpot rows must not be junked for missing family vocabulary.
    assert not _is_ocr_junk_equipment_line("Power Distribution Pro 2")
    assert not _is_ocr_junk_equipment_line("Access Rescue KeySwitch 2")
    assert not _is_ocr_junk_equipment_line("Camera AI Multi Sensor 4 1")

    atoms = _hardware_atoms_from_equipment_text(
        project_id="deal-gecko",
        artifact_id="art1",
        filename="e.eml",
        text="\n".join(
            [
                "I Access Reader Pro Juncti ... 5",
                "Camera Al Multi Sensor 4 1",
                "Access G3 Reader 4",
                "Protect All-In-One Sensor 2",
            ]
        ),
        content_id="cid-order",
        parser_version="test",
    )
    texts = [a.raw_text for a in atoms]
    assert any("Access Reader Pro Juncti" in t for t in texts)
    assert any("Camera AI Multi Sensor" in t for t in texts)
    assert any("Access G3 Reader" in t for t in texts)
    assert any("Protect All-In-One Sensor" in t for t in texts)
    by_item = {a.value.get("item"): a.value.get("quantity") for a in atoms}
    assert by_item.get("Access Reader Pro Juncti") == 5
    assert by_item.get("Camera AI Multi Sensor 4") == 1


def test_equipment_list_intro_is_connective_not_orphan_scope(tmp_path: Path) -> None:
    from app.parsers.email_parser import _is_equipment_list_intro_line

    intro = "Below is the full equipment list. One hard requirement for him is Otka integration."
    assert _is_equipment_list_intro_line(intro)
    atoms = EmailParser().parse_artifact("p", "art_email", _write_eml(tmp_path))
    orphan = [
        a
        for a in atoms
        if a.atom_type == AtomType.scope_item
        and "full equipment list" in a.raw_text.lower()
        and a.value.get("kind") == "email_body_line"
    ]
    assert orphan == []
    # Equipment intro must also not become email_body_context (CID lead_in only).
    assert not any(
        a.value.get("kind") == "email_body_context"
        and "full equipment list" in a.raw_text.lower()
        for a in atoms
    )


def test_greeting_and_intro_precede_include_exclude(tmp_path: Path) -> None:
    """Addressee + intro body context sort before Include/Exclude in reading order."""
    atoms = EmailParser().parse_artifact("p", "art_email", _write_eml(tmp_path))

    addressee = next(a for a in atoms if a.value.get("kind") == "email_addressee")
    assert addressee.raw_text.strip() == "Eddie,"
    assert addressee.atom_type == AtomType.deal_metadata

    intro = next(a for a in atoms if a.value.get("kind") == "email_body_context")
    assert "Appreciate you hopping on" in intro.raw_text
    assert intro.atom_type == AtomType.deal_metadata
    assert intro.atom_type != AtomType.scope_item

    include_line = min(
        a.source_refs[0].locator["line_start"]
        for a in atoms
        if a.value.get("list_section") == "include"
    )
    addr_line = addressee.source_refs[0].locator["line_start"]
    intro_line = intro.source_refs[0].locator["line_start"]
    assert addr_line < intro_line < include_line


def test_cid_equipment_carries_equipment_list_lead_in(tmp_path: Path, monkeypatch) -> None:
    import base64

    from app.parsers.email_parser import EmailParser as EP

    cid = "f41c1a3b-2993-42e3-a181-e2441b3942d0"
    png = base64.b64encode(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
        )
    ).decode("ascii")
    mixed = "=_M"
    related = "=_R"
    body = (
        "Below is the full equipment list. One hard requirement for him is Otka integration.\n"
        f"[cid:{cid}]\n"
    )
    lines = [
        "From: p@example.com",
        "Subject: eq",
        f'Content-Type: multipart/mixed; boundary="{mixed}"',
        "",
        f"--{mixed}",
        f'Content-Type: multipart/related; boundary="{related}"',
        "",
        f"--{related}",
        "Content-Type: text/plain; charset=utf-8",
        "",
        body,
        f"--{related}",
        "Content-Type: image/png",
        "Content-Transfer-Encoding: base64",
        f"Content-ID: <{cid}@hubspot-ingest>",
        "",
        png,
        f"--{related}--",
        f"--{mixed}--",
        "",
    ]
    eml = tmp_path / "eq.eml"
    eml.write_bytes("\r\n".join(lines).encode("utf-8"))

    monkeypatch.setattr(
        "app.parsers.email_parser._ocr_cid_part",
        lambda part: "Order Details\nAccess Point E7 × 6\nSwitch Pro Max 48 PoE × 2\n",
    )
    atoms = EP().parse_artifact("p", "art", eml)
    equipment = [a for a in atoms if a.value.get("kind") == "email_cid_equipment_line"]
    unresolved = [a for a in atoms if a.value.get("kind") == "email_cid_unresolved"]
    assert equipment
    assert unresolved == []
    lead = equipment[0].value.get("lead_in") or []
    assert any("equipment list" in str(x).lower() for x in lead)
    path = (equipment[0].source_refs[0].locator or {}).get("section_path") or []
    assert "Equipment list" in path
    assert any("equipment list" in str(x).lower() for x in path)
