"""A document gets a copy of a line only when its own text holds that line.

Shapes from live 010353 / 010003 / 010087 (synthetic names and numbers).
010353 (after #292/#299): the SOW PDF's CUSTOMER CONTACTS row (two people, one
"Phone number not provided") showed three times -- once under the PDF and once
under each intake file (a JSON and a .md), which list the same people their own
way and never hold the row. A refused stakeholder fold had left the intake atoms' refs on the PDF row,
and the own-copy sweep cloned the row into every document a ref named. The same
deal's HubSpot note got a stakeholder copy of a vendor contact minted from its
"Author: <name>" header; its body never names her. 010003: a reseller quote's
"<name> | <phone> | <email>" line was copied into a SOW, the draft SOW and four
emails, and 010087 a vendor contact line into five files, each copy still saying ``document_kind: vendor_quote_bom``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.cross_doc_copies import (
    COPY_FLAG,
    cited_not_held_in,
    drop_unheld_copies,
    ensure_own_copies,
    is_cross_doc_copy,
    source_lines_reader,
)
from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    EvidenceAtom,
    ReviewStatus,
    SourceRef,
)
from app.core.semantic_dedup import dedupe_stakeholder_atoms


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    monkeypatch.setenv("SOWSMITH_DISABLE_LLM", "1")


_N = [0]

PDF, JSON_DOC, MD_DOC, NOTE = "art_sow_pdf", "art_intake_json", "art_intake_md", "art_note"

ROW = ("Jon Rivera-Lopez / Dale Burke | Support Manager / Onsite Contact | "
       "Jon Rivera-Lopez: (555) 010-4229 / Dale Burke: Phone number not provided")


def _ref(artifact, atype, locator):
    _N[0] += 1
    return SourceRef(id=f"src_{artifact}_{_N[0]}", artifact_id=artifact, artifact_type=atype,
                     filename=f"{artifact}.{atype.value}", locator=dict(locator),
                     extraction_method="test", parser_version="t")


def _atom(text, artifact, *, atype=AtomType.stakeholder, value=None, ref_type=ArtifactType.txt,
          locator=None, flags=None):
    _N[0] += 1
    return EvidenceAtom(
        id=f"atm_{artifact}_{_N[0]}", project_id="p", artifact_id=artifact,
        atom_type=atype, raw_text=text, normalized_text=text.lower(), value=dict(value or {}),
        entity_keys=[], source_refs=[_ref(artifact, ref_type, locator or {"line": _N[0]})],
        authority_class=AuthorityClass.machine_extractor, confidence=0.7,
        review_flags=list(flags or []), review_status=ReviewStatus.auto_accepted, parser_version="test",
    )


def _files(tmp_path: Path) -> dict[str, Path]:
    md = tmp_path / "INTAKE_REQUEST.md"
    md.write_text(
        "# Site A\n## Job site\n100 Example Rd, Springfield, OH 45000\n## Contacts\n"
        "- **requester:** lena doe · lena@customer.example\n"
        "- **csm:** Jon Rivera-Lopez · (555) 010-4229\n"
        "- **onsite:** Dale Burke\n", encoding="utf-8")
    js = tmp_path / "INTAKE_REQUEST_X-0001.json"
    js.write_text(json.dumps({"contacts": [
        {"role": "csm", "name": "Jon Rivera-Lopez", "phone": "(555) 010-4229"},
        {"role": "onsite", "name": "Dale Burke"},
    ]}, indent=2), encoding="utf-8")
    note = tmp_path / "note.txt"
    note.write_text(
        "HubSpot Note: Waiting on feedback. parts are delayed\nHubSpot Note ID: 100000000001\n"
        "Date: 2026-10-01T21:27:47.823Z\nAuthor: Mara Bell\nAuthor-Email: mara@vendor.example\n\n"
        "Waiting on feedback. parts are delayed", encoding="utf-8")
    import fitz

    pdf = tmp_path / "sow.pdf"
    doc = fitz.open()
    page = doc.new_page()
    y = 72
    for line in ("CUSTOMER CONTACTS", "FULL NAME", "JOB TITLE", "EMAIL ADDRESS",
                 "Jon Rivera-Lopez / Dale", "Burke", "Support Manager / Onsite", "Contact",
                 "Jon Rivera-Lopez: (555) 010-4229 / Dale", "Burke: Phone number not provided",
                 "Mara Bell", "Account Executive", "mara@vendor.example"):
        page.insert_text((72, y), line, fontsize=9)
        y += 14
    doc.save(str(pdf))
    return {PDF: pdf, JSON_DOC: js, MD_DOC: md, NOTE: note}


def _shapes():
    pdf_row = _atom(ROW, PDF, ref_type=ArtifactType.pdf, locator={"page": 1, "block_kind": "table", "row_index": 0},
                    value={"kind": "table_row", "name": "Jon Rivera-Lopez", "role": "Support Manager",
                           "phone": "(555) 010-4229", "document_kind": "sow"})
    js = _atom("contacts[0].name: Jon Rivera-Lopez", JSON_DOC, ref_type=ArtifactType.json,
               locator={"kind": "json_value", "line": 4},
               value={"kind": "person", "name": "Jon Rivera-Lopez"})
    md = _atom("**csm:** Jon Rivera-Lopez · (555) 010-4229", MD_DOC,
               locator={"line_start": 6, "line_end": 6},
               value={"kind": "person", "name": "Jon Rivera-Lopez", "phone": "(555) 010-4229"})
    return pdf_row, js, md


def _row_copies(atoms):
    return [a for a in atoms if "phone number not provided" in (a.raw_text or "").lower()]


def test_contacts_row_has_one_atom_under_the_pdf_and_the_intakes_keep_their_own_lines(tmp_path):
    paths = _files(tmp_path)
    pdf_row, js, md = _shapes()
    # The real fold: the intake files' John records fold into the fuller PDF
    # row, which then cites both intake files.
    before = [js, md, pdf_row]
    kept = dedupe_stakeholder_atoms(list(before))
    dropped = [a for a in before if a not in kept]
    assert {r.artifact_id for r in pdf_row.source_refs} >= {PDF}
    for other in (js, md):  # whichever way the fold went, the row cites them
        for r in other.source_refs:
            if r not in pdf_row.source_refs:
                pdf_row.source_refs.append(r)
    copies = ensure_own_copies(kept, [], doc_lines=source_lines_reader(paths), dropped=dropped)
    final = kept + copies
    # Exactly one contacts-row atom, under the PDF.
    rows = _row_copies(final)
    assert [a.artifact_id for a in rows] == [PDF]
    # Each intake keeps its own line, in its own words -- never the row's.
    by_doc: dict[str, list[str]] = {}
    for a in final:
        by_doc.setdefault(a.artifact_id, []).append(a.raw_text)
    assert by_doc[JSON_DOC] == ["contacts[0].name: Jon Rivera-Lopez"]
    assert by_doc[MD_DOC] == ["**csm:** Jon Rivera-Lopez · (555) 010-4229"]
    for a in final:
        if a.artifact_id != PDF:
            assert "dale" not in a.raw_text.lower()
    # No document listing credits the PDF row to an intake file.
    assert cited_not_held_in(pdf_row) | set(pdf_row.value.get("also_in_documents") or []) == {JSON_DOC, MD_DOC}
    for c in copies:
        assert is_cross_doc_copy(c) and c.value["duplicate_of"]["atom_id"] == pdf_row.id


def test_contacts_row_is_not_cloned_into_intakes_when_their_atoms_stand(tmp_path):
    paths = _files(tmp_path)
    pdf_row, js, md = _shapes()
    for other in (js, md):
        pdf_row.source_refs.extend(other.source_refs)
    copies = ensure_own_copies([pdf_row, js, md], [], doc_lines=source_lines_reader(paths))
    assert copies == []
    assert cited_not_held_in(pdf_row) == {JSON_DOC, MD_DOC}
    assert "also_in_documents" not in pdf_row.value


def test_a_refused_stakeholder_fold_leaves_no_ref_on_the_survivor():
    # Same full name across documents; the intake's record alone states an
    # email, so it must stand -- and then the row must not cite it.
    pdf_row = _atom("Jon Rivera-Lopez", PDF, ref_type=ArtifactType.pdf,
                    value={"kind": "person", "name": "Jon Rivera-Lopez", "phone": "(555) 010-4229",
                           "title": "Support Manager"})
    js = _atom("Jon Rivera-Lopez · Onsite Contact", JSON_DOC, ref_type=ArtifactType.json,
               value={"kind": "person", "name": "Jon Rivera-Lopez"})
    kept = dedupe_stakeholder_atoms([pdf_row, js])
    if js in kept:
        assert {r.artifact_id for r in pdf_row.source_refs} == {PDF}


def test_a_document_that_holds_the_line_still_gets_its_copy(tmp_path):
    paths = _files(tmp_path)
    md_line = _atom("**csm:** Jon Rivera-Lopez · (555) 010-4229", MD_DOC,
                    value={"kind": "person", "name": "Jon Rivera-Lopez"})
    other = _atom("Install one camera", MD_DOC, atype=AtomType.scope_item)
    # A JSON doc whose text DOES hold the md line's words is not the case here;
    # the PDF holds "Jon Rivera-Lopez" only inside the row, not this line.
    pdf_ref = _ref(PDF, ArtifactType.pdf, {"page": 1})
    md_line.source_refs.append(pdf_ref)
    out = ensure_own_copies([md_line, other], [], doc_lines=source_lines_reader(paths))
    assert out == []
    # Same survivor citing a document whose source holds it: one copy.
    (tmp_path / "pasted.md").write_text("Notes\n- **csm:** Jon Rivera-Lopez · (555) 010-4229\n", encoding="utf-8")
    paths["art_pasted"] = tmp_path / "pasted.md"
    md_line.source_refs.append(_ref("art_pasted", ArtifactType.txt, {"line": 2}))
    out = ensure_own_copies([md_line, other], [], doc_lines=source_lines_reader(paths))
    assert [c.artifact_id for c in out] == ["art_pasted"]
    assert is_cross_doc_copy(out[0])


def test_note_author_metadata_is_not_a_copy_of_the_sow_person(tmp_path):
    paths = _files(tmp_path)
    sow_person = _atom("Mara Bell | Account Executive | mara@vendor.example", PDF,
                      ref_type=ArtifactType.pdf, value={"kind": "table_row", "name": "Mara Bell"})
    # What split_copies held after stakeholder_dedup (live atm_4fbe81db0423957d).
    note_copy = _atom("Mara Bell", NOTE, locator={"kind": "hubspot_note_body", "line": 0},
                      value={"kind": "person", "name": "Mara Bell",
                             "context": "note_id=100000000001 | author=Mara Bell | author_email=mara@vendor.example",
                             "duplicate_of": {"atom_id": sow_person.id, "artifact_id": PDF,
                                              "stage": "stakeholder_dedup"}},
                      flags=["unearned_contract_authority_demoted", COPY_FLAG])
    sow_person.value["also_in_documents"] = [NOTE]
    sow_person.source_refs.extend(note_copy.source_refs)
    header = _atom("note_id=100000000001 | author=Mara Bell | author_email=mara@vendor.example", NOTE,
                   atype=AtomType.deal_metadata, value={"kind": "hubspot_note_meta"})
    body = _atom("Waiting on feedback.", NOTE, atype=AtomType.status_update if hasattr(AtomType, "status_update") else AtomType.scope_item)
    final = [sow_person, header, body]
    doc_lines = source_lines_reader(paths)
    kept, refused = drop_unheld_copies([note_copy], final, doc_lines)
    assert kept == [] and refused == [note_copy]
    assert not is_cross_doc_copy(note_copy)
    assert "also_in_documents" not in sow_person.value
    # And the sweep does not mint one either.
    assert ensure_own_copies(final, kept, doc_lines=doc_lines) == []
    assert NOTE in cited_not_held_in(sow_person)


def test_vendor_contact_line_is_not_copied_into_documents_that_name_the_person_differently(tmp_path):
    bom_line = "Sally Hart | (800) 555-0142 | sally.hart@reseller.example"
    quote = _atom(bom_line, "art_reseller_quote", ref_type=ArtifactType.pdf,
                  value={"kind": "person", "name": "Sally Hart", "document_kind": "vendor_quote_bom",
                         "page": 1})
    mail = tmp_path / "reply.eml"
    mail.write_text(
        "From: Sally Hart <sally.hart@reseller.example>\nTo: pm@vendor.example\nSubject: RE: quote\n\n"
        "Attached is the revised quote.\n\nSally Hart\nAccount Manager | Metro Financial | Reseller\n"
        "1 Example Plaza\nDirect: 555.010.0736 | Toll Free Number: 800.555.0142\n",
        encoding="utf-8")
    holds = tmp_path / "pasted.txt"
    holds.write_text("From the quote:\nSally Hart | (800) 555-0142 | sally.hart@reseller.example\n", encoding="utf-8")
    paths = {"art_mail": mail, "art_pasted": holds}
    sig = _atom("Sally Hart | Account Manager | Metro Financial | Reseller", "art_mail",
                value={"kind": "person", "name": "Sally Hart"})
    for aid in ("art_mail", "art_pasted"):
        quote.source_refs.append(_ref(aid, ArtifactType.txt, {"line": 2}))
    out = ensure_own_copies([quote, sig], [], doc_lines=source_lines_reader(paths))
    assert [c.artifact_id for c in out] == ["art_pasted"]
    copy = out[0]
    # A copy never inherits the original's document-specific fields.
    assert "document_kind" not in copy.value and "page" not in copy.value
    assert copy.value["duplicate_of"]["artifact_id"] == "art_reseller_quote"
    assert cited_not_held_in(quote) == {"art_mail"}
    assert quote.value["also_in_documents"] == ["art_pasted"]


def test_vendor_staff_contact_line_is_not_copied_into_files_that_list_him_their_own_way(tmp_path):
    line = "Toby Tran | VP Sales | toby@vendor.example | 555.010.3490"
    sig = _atom(line, "art_email_1", value={"kind": "person", "name": "Toby Tran",
                                            "document_kind": "email"})
    sow = tmp_path / "sow.txt"
    sow.write_text("VENDOR CONTACTS\nFULL NAME JOB TITLE EMAIL ADDRESS\n"
                   "Toby Tran VP Sales toby@vendor.example\n", encoding="utf-8")
    transcript = tmp_path / "fireflies.json"
    transcript.write_text(json.dumps({"speakers": [{"name": "Toby Tran"}],
                                      "sentences": [{"speaker_name": "Toby Tran", "text": "Thanks all."}]}),
                          encoding="utf-8")
    email2 = tmp_path / "email2.eml"
    email2.write_text("From: Toby Tran <toby@vendor.example>\nSubject: RE\n\nSounds good.\n\n" + line + "\n",
                      encoding="utf-8")
    paths = {"art_sow": sow, "art_ff": transcript, "art_email_2": email2}
    for aid in paths:
        sig.source_refs.append(_ref(aid, ArtifactType.txt, {"line": 1}))
    out = ensure_own_copies([sig], [], doc_lines=source_lines_reader(paths))
    assert [c.artifact_id for c in out] == ["art_email_2"]
    assert "document_kind" not in out[0].value
    assert cited_not_held_in(sig) == {"art_sow", "art_ff"}
