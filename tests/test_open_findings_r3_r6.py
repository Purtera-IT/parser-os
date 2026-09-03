"""The audit's remaining open items, each pinned to the mechanism that fixed it."""
from __future__ import annotations
from app.core.schemas import AtomType
from app.parsers.contact_property_block import people_from_multiline_cells
from app.core.entity_extraction import _emit_stakeholders
from app.core.orbitbrief_envelope import _retype_produced_material_scope
from app.parsers.email_parser import _is_identity_only_line

def test_r5_and_r3_marker_and_bare_identity_lines_never_reach_the_llm(monkeypatch):
    import app.core.typed_atom_classifier as tac
    from tests.test_typed_classifier_contact_deflect import _patched, _Atom
    seen = _patched(monkeypatch, tac)
    marker = _Atom(AtomType.scope_item, {"kind": "image_marker", "region_ref": "media/image1.png"})
    marker.raw_text = "[Image awaiting OCR / vision / OLE extraction] media/image1.png in Vendor Install.docx"
    name = _Atom(AtomType.scope_item, {}); name.raw_text = "Nick Robateau"
    email = _Atom(AtomType.scope_item, {}); email.raw_text = "Nick.Robateau@CDW.com"
    frag = _Atom(AtomType.scope_item, {}); frag.raw_text = "; Nick Robateau <"
    real = _Atom(AtomType.scope_item, {}); real.raw_text = "Please exclude the West Wing from scope"
    tac.classify_atoms([marker, name, email, frag, real])
    reached = [a.raw_text for b in seen for a in b]
    assert reached == ["Please exclude the West Wing from scope"], reached

def test_site_contact_label_is_not_a_person():
    keys = _emit_stakeholders("On Site Contact Name | Belisha Crouch | OSC email | Belisha.Crouch@Sodexo.com Owner: Belisha Crouch")
    assert "stakeholder:site_contact" not in keys
    assert "stakeholder:on_site_contact_name" not in keys

def test_tech_contact_multiline_cell_reads_name_and_phone():
    out = people_from_multiline_cells({"0": "TECH Contact Information (Main)", "1": "Bernie Donnelly\n404-918-0783", "2": "Escalation Contact", "3": ""})
    assert out == [{"role": "TECH Contact Information (Main)", "kind": "person", "name": "Bernie Donnelly", "phone": "404-918-0783"}]
    assert people_from_multiline_cells({"0": "Notes", "1": "line one\nline two"}) == []

def test_bernie_and_bernard_fold_in_one_document():
    from app.core.schemas import AuthorityClass, EvidenceAtom, ReviewStatus
    from app.core.semantic_dedup import dedupe_stakeholder_atoms
    def mk(aid, name, **v):
        return EvidenceAtom(id=f"a_{aid}_{name}", project_id="p", artifact_id=aid, atom_type=AtomType.stakeholder, raw_text=name, normalized_text=name.lower(),
            value={"kind": "person", "name": name, **v}, authority_class=AuthorityClass.contractual_scope, confidence=0.9, review_status=ReviewStatus.auto_accepted, entity_keys=[], parser_version="t")
    atoms = [mk("d1", "Bernard Donnelly", email="bernie.donnelly@sodexo.com", phone="404-918-0783"), mk("d1", "Bernie Donnelly")]
    out = dedupe_stakeholder_atoms(atoms)
    assert [a.value["name"] for a in out] == ["Bernard Donnelly"]
    # live 010215 shape: the sparser record carries the SAME phone (from a
    # multi-line TECH-contact cell) -- shared identity token, still one person
    out_phone = dedupe_stakeholder_atoms([
        mk("d1", "Bernard Donnelly", email="bernie.donnelly@sodexo.com", phone="404-918-0783"),
        mk("d1", "Bernie Donnelly", phone="(404) 918-0783"),
    ])
    assert [a.value["name"] for a in out_phone] == ["Bernard Donnelly"], [a.value for a in out_phone]
    # a CONFLICTING phone is evidence of two people; never fold
    out_conf = dedupe_stakeholder_atoms([
        mk("d1", "Bernard Donnelly", email="bernie.donnelly@sodexo.com", phone="404-918-0783"),
        mk("d1", "Bernie Donnelly", phone="404-555-0100"),
    ])
    assert len(out_conf) == 2
    # different documents, or two real people sharing a surname, stay apart
    out2 = dedupe_stakeholder_atoms([mk("d1", "Bernard Donnelly", email="b@x.com"), mk("d2", "Bernie Donnelly")])
    assert len(out2) == 2
    out3 = dedupe_stakeholder_atoms([mk("d1", "Bernard Donnelly", email="b@x.com"), mk("d1", "Kate Donnelly")])
    assert len(out3) == 2

def test_r6_atlas_document_rows_become_implementation_notes():
    env = {"atoms": [
        {"artifact_id": "kronos", "atom_type": "scope_item", "text": "Route the power supply cable through the clamps"},
        {"artifact_id": "kronos", "atom_type": "task", "text": "Press Button 4"},
        {"artifact_id": "sow", "atom_type": "scope_item", "text": "Install Time Clock at designated Location"},
    ]}
    docs = [{"artifact_id": "kronos", "deal_stage": {"admissible_for": "atlas"}}, {"artifact_id": "sow", "deal_stage": {"admissible_for": "evidence"}}]
    assert _retype_produced_material_scope(env, docs) == 1
    assert [a["atom_type"] for a in env["atoms"]] == ["site_implementation_note", "task", "scope_item"]
    assert env["atoms"][0]["decision_provenance"]["source"] == "produced_material"


def test_r6_inbound_install_instructions_are_still_procedure():
    """Live 010215: the Kronos instructions arrived inbound BEFORE quoting, so
    the stage rule filed them as `evidence` (what we quote from). The taxonomy
    verdict -- type INSTALL_INSTRUCTIONS, stage DELIVERY, scope global -- is the
    signal that survives, and it must be enough on its own."""
    env = {"atoms": [
        {"artifact_id": "kronos", "atom_type": "scope_item", "text": "Route the power supply cable through the clamps"},
        {"artifact_id": "sow", "atom_type": "scope_item", "text": "Install Time Clock at designated Location"},
    ]}
    docs = [
        {"artifact_id": "kronos", "deal_stage": {"admissible_for": "evidence"},
         "lifecycle": {"type": "INSTALL_INSTRUCTIONS", "stage": "DELIVERY", "admissible_for": "evidence"},
         "scope": {"scope": "global"}},
        {"artifact_id": "sow", "deal_stage": {"admissible_for": "evidence"},
         "lifecycle": {"type": "SOW", "stage": "QUOTING", "admissible_for": "evidence"}},
    ]
    assert _retype_produced_material_scope(env, docs) == 1
    assert [a["atom_type"] for a in env["atoms"]] == ["site_implementation_note", "scope_item"]
    assert "INSTALL_INSTRUCTIONS" in env["atoms"][0]["decision_provenance"]["rationale"]

def test_trailing_and_quoted_signature_lines_are_never_typed(tmp_path):
    """Live 010215 (R3, after the first fix): the leading-lines rule let a
    TRAILING authored signature and a QUOTED fragment through, and the
    vocabulary typer stamped them `exclusion`. Shape decides, not position."""
    from pathlib import Path
    from app.parsers.email_parser import EmailParser
    eml = (
        "From: Nick Robateau <nick.robateau@cdw.com>\n"
        "To: patrick@purtera-it.com\n"
        "Date: Thu, 13 Aug 2026 14:27:14 +0000\n"
        "Subject: Re: Time Clock Installs\n"
        "Message-ID: <a@x>\n\n"
        "Quinton - please exclude the West Wing from scope.\n\n"
        "Nick Robateau\n"
        "Nick.Robateau@CDW.com\n\n"
        "From: Quinton James <quinton.james@cdw.com>\n"
        "Sent: Thursday, August 13, 2026 12:56 AM\n"
        "Subject: Re: Time Clock Installs\n\n"
        "Best,\n; Nick Robateau <\n"
    )
    p = tmp_path / "t.eml"; p.write_text(eml, encoding="utf-8")
    atoms = EmailParser().parse(Path(p))
    texts = [str(getattr(a, "raw_text", "")).strip() for a in atoms if str(getattr(a.atom_type, "value", a.atom_type)) == "exclusion"]
    assert any("West Wing" in t for t in texts), texts
    assert not any(t in ("Nick Robateau", "Nick.Robateau@CDW.com", "; Nick Robateau <") for t in texts), texts


def test_identity_only_line_shape():
    assert _is_identity_only_line("Nick Robateau")
    assert _is_identity_only_line("Nick.Robateau@CDW.com")
    assert _is_identity_only_line("; Nick Robateau <")
    assert not _is_identity_only_line("Please exclude the West Wing from scope")
    assert not _is_identity_only_line("Marion County School District needs 10 clocks")
