"""A stage that removes an atom on purpose records a DROP; nothing restores it.

Shape of live 010353 after #338 (synthetic names and text):

* A CRM note's "Author: <name>" header line was read as a person, held as the
  note's copy of the SOW's record of that person, and refused by the own-copy
  gate (the note's body never names her). The refusal was recorded as a FOLD
  with no survivor, so the end-of-compile ledger check put it back.
* The SOW's bare onsite-contact person, lifted out of a PROJECT OVERVIEW
  sentence, was folded into the SOW step that names the same person (its
  recorded survivor, standing). The own-copy sweep then took it back as the
  SOW's copy of the intake's contact line, so it was emitted live with a
  standing survivor.
"""
from __future__ import annotations



from app.core.cross_doc_copies import (
    COPY_FLAG,
    drop_unheld_copies,
    ensure_own_copies,
    is_cross_doc_copy,
    source_lines_reader,
)
from app.core.schemas import ArtifactType, AtomType, AuthorityClass, EvidenceAtom, ReviewStatus, SourceRef
from app.core.suppression_ledger import (
    SURVIVOR_KEY,
    capture_suppressed,
    settle_ledger,
    suppression_kind,
)

_N = [0]
SOW, MD, NOTE = "art_sow", "art_intake_md", "art_note"


def _atom(text, art, *, atype=AtomType.stakeholder, value=None, locator=None, flags=None,
          ref_type=ArtifactType.txt):
    _N[0] += 1
    n = _N[0]
    ref = SourceRef(id=f"src_{art}_{n}", artifact_id=art, artifact_type=ref_type, filename=f"{art}.{ref_type.value}",
                    locator=dict(locator or {"line": n}), extraction_method="test", parser_version="t")
    return EvidenceAtom(
        id=f"atm_{art}_{n}", project_id="p", artifact_id=art, atom_type=atype, raw_text=text,
        normalized_text=text.lower(), value=dict(value or {}), entity_keys=[], source_refs=[ref],
        authority_class=AuthorityClass.machine_extractor, confidence=0.7, review_flags=list(flags or []),
        review_status=ReviewStatus.auto_accepted, parser_version="test",
    )


def _note(tmp_path):
    p = tmp_path / "note.txt"
    p.write_text(
        "HubSpot Note: Waiting on feedback. parts are delayed\nHubSpot Note ID: 100000000002\n"
        "Date: 2026-10-01T21:27:47.823Z\nAuthor: Mara Bell\nAuthor-Email: mara@vendor.example\n\n"
        "Waiting on feedback. parts are delayed", encoding="utf-8")
    return p


def test_a_copy_read_off_the_note_author_header_is_a_drop_and_stays_dropped(tmp_path):
    sow_row = _atom("Mara Bell | Account Executive | mara@vendor.example", SOW, ref_type=ArtifactType.pdf,
                    value={"kind": "table_row", "name": "Mara Bell"})
    note_copy = _atom("Mara Bell", NOTE, locator={"kind": "hubspot_note_body", "line": 0},
                      value={"kind": "person", "name": "Mara Bell",
                             "context": "note_id=100000000002 | author=Mara Bell | author_email=mara@vendor.example",
                             "duplicate_of": {"atom_id": sow_row.id, "artifact_id": SOW, "stage": "stakeholder_dedup"}},
                      flags=[COPY_FLAG])
    body = _atom("Waiting on feedback.", NOTE, atype=AtomType.scope_item)
    final = [sow_row, body]
    kept, refused = drop_unheld_copies([note_copy], final, source_lines_reader({NOTE: _note(tmp_path)}))
    assert refused == [note_copy]
    # What the compiler records for a refused copy.
    supp = capture_suppressed(refused, [], stage="own_copy_gate", reason="r")
    assert suppression_kind(note_copy) == "drop"
    assert note_copy.value["_suppression"]["kind"] == "drop"
    assert "header or author metadata" in note_copy.value["_suppression"]["drop_reason"]
    atoms, supp_after, counts = settle_ledger(final, supp)
    assert counts["restored"] == 0
    assert note_copy not in atoms and supp_after == [note_copy]


def test_a_fold_whose_survivor_stands_is_not_taken_back_as_another_documents_copy(tmp_path):
    md_path = tmp_path / "INTAKE_REQUEST.md"
    md_path.write_text("# Site\n## Contacts\n- **onsite:** Dale Burke\n", encoding="utf-8")
    sow_path = tmp_path / "sow.txt"
    sow_path.write_text(
        "PROJECT OVERVIEW\nThe technician(s) will coordinate with onsite contact Dale Burke and the support "
        "manager.\nSCOPE OF WORK\nEnsure onsite contact Dale Burke is available upon arrival.\n", encoding="utf-8")
    step = _atom("Ensure onsite contact Dale Burke is available upon arrival.", SOW, atype=AtomType.task,
                 value={"kind": "instruction", "contact_name": "Dale Burke"})
    bare = _atom("Dale Burke", SOW, value={"kind": "person", "name": "Dale Burke",
                                          "context": "The technician(s) will coordinate with onsite contact Dale Burke"},
                 locator={"page": 2, "block_kind": "paragraph", "sentence_index": 1, "sentence_count": 4})
    md_line = _atom("**onsite:** Dale Burke", MD, value={"kind": "person", "name": "Dale Burke"})
    # semantic_dedup folded the bare person into the SOW step (same document);
    # a refused stakeholder fold left the step's refs -- the bare person's
    # among them -- on the intake's contact line.
    capture_suppressed([bare], [], stage="semantic_dedup", reason="r")
    bare.value[SURVIVOR_KEY] = {"atom_id": step.id, "artifact_id": SOW, "stage": "semantic_dedup"}
    md_line.source_refs.extend(bare.source_refs)
    kept = [step, md_line]
    out = ensure_own_copies(kept, [], doc_lines=source_lines_reader({SOW: sow_path, MD: md_path}), dropped=[bare])
    assert bare not in out
    assert not is_cross_doc_copy(bare)
    assert bare.value["_suppression"]["stage"] == "semantic_dedup"
    atoms, supp, counts = settle_ledger(kept + out, [bare])
    assert counts["restored"] == 0 and bare not in atoms
    assert bare.value[SURVIVOR_KEY]["atom_id"] == step.id


def test_the_folded_atom_still_comes_back_when_it_is_the_documents_only_copy(tmp_path):
    """Unchanged: a document's own line folded INTO the canonical atom comes
    back as that document's copy (its survivor is the canonical itself)."""
    md_path = tmp_path / "INTAKE_REQUEST.md"
    md_path.write_text("# Site\n## Contacts\n- **csm:** Jon Rivera-Lopez · (555) 010-4229\n", encoding="utf-8")
    pdf_path = tmp_path / "sow.txt"
    pdf_path.write_text("Jon Rivera-Lopez | Support Manager | (555) 010-4229 | jon@customer.example\n", encoding="utf-8")
    row = _atom("Jon Rivera-Lopez | Support Manager | (555) 010-4229 | jon@customer.example", SOW,
                value={"kind": "table_row", "name": "Jon Rivera-Lopez"})
    md_line = _atom("**csm:** Jon Rivera-Lopez · (555) 010-4229", MD, value={"kind": "person", "name": "Jon Rivera-Lopez"})
    capture_suppressed([md_line], [], stage="stakeholder_dedup", reason="r")
    md_line.value[SURVIVOR_KEY] = {"atom_id": row.id, "artifact_id": SOW, "stage": "stakeholder_dedup"}
    row.source_refs.extend(md_line.source_refs)
    out = ensure_own_copies([row], [], doc_lines=source_lines_reader({SOW: pdf_path, MD: md_path}), dropped=[md_line])
    assert out == [md_line] and is_cross_doc_copy(md_line)


def test_the_envelope_ledger_row_carries_fold_or_drop(monkeypatch):
    from types import SimpleNamespace

    from app.core.orbitbrief_envelope import _suppressed_for_review

    monkeypatch.setenv("SOWSMITH_SUPPRESSED_IN_ENVELOPE", "1")
    kept = _atom("Waiting on feedback.", NOTE, atype=AtomType.scope_item)
    folded = _atom("Waiting on feedback", NOTE, atype=AtomType.scope_item)
    capture_suppressed([folded], [], stage="semantic_dedup", reason="dup")
    folded.value[SURVIVOR_KEY] = {"atom_id": kept.id, "artifact_id": NOTE, "stage": "semantic_dedup"}
    gate = _atom("coordination.walks_tech: yes", MD, atype=AtomType.scope_item)
    capture_suppressed([gate], [], stage="substance_gate", reason="no substance")
    rows = _suppressed_for_review(SimpleNamespace(suppressed_atoms=[folded, gate], project_id="p"), [kept])
    by_id = {r["id"]: r for r in rows}
    assert by_id[folded.id]["kind"] == "fold" and by_id[folded.id]["survivor"]["id"] == kept.id
    assert by_id[gate.id]["kind"] == "drop"


def test_a_fold_whose_survivor_is_a_cross_document_copy_names_it_in_the_envelope(monkeypatch):
    from types import SimpleNamespace

    from app.core.orbitbrief_envelope import _suppressed_for_review

    monkeypatch.setenv("SOWSMITH_SUPPRESSED_IN_ENVELOPE", "1")
    v1_row = _atom("Ann Lee | PM | ann@customer.example", "art_sow_v1", value={"kind": "table_row"})
    v2_copy = _atom("Ann Lee | PM | ann@customer.example", "art_sow_v2",
                    value={"kind": "table_row", "duplicate_of": {"atom_id": v1_row.id, "artifact_id": "art_sow_v1",
                                                                 "stage": "semantic_dedup"}}, flags=[COPY_FLAG])
    folded = _atom("Ann Lee | Project Manager | ann@customer.example", "art_sow_v2", value={"kind": "table_row"})
    capture_suppressed([folded], [], stage="semantic_dedup", reason="dup")
    folded.value[SURVIVOR_KEY] = {"atom_id": v2_copy.id, "artifact_id": "art_sow_v2", "stage": "semantic_dedup"}
    result = SimpleNamespace(suppressed_atoms=[folded], atoms=[v1_row, v2_copy], project_id="p")
    # The envelope's kept list leaves the copy out (it is held under its document).
    (row,) = _suppressed_for_review(result, [v1_row])
    assert row["survivor"]["id"] == v2_copy.id
    assert row["survivor"]["copy_of"] == v1_row.id
    assert row["kind"] == "fold"
