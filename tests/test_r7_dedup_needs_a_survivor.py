"""semantic_dedup never suppresses a table row that no survivor restates.

Shape of a live signed SOW (synthetic names): a revision table's rows typed
``deal_metadata`` share the identity field the key is built from (a column
label), so "v.1 | <name> | First" was folded into "v.2 | <name> | Second" -- or
into another document's row -- and its document showed one revision of two,
the suppressed row naming no survivor.
"""
from __future__ import annotations

from app.core.cross_doc_copies import COPY_FLAG, split_copies
from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    EvidenceAtom,
    ReviewStatus,
    SourceRef,
)
from app.core.semantic_dedup import semantic_dedup_atoms

_N = [0]
COLS = ["SOW VERSION", "QUOTED BY", "DATE", "REVISION HISTORY"]


def _row(artifact, cells, *, row, conf=0.9, atype=AtomType.deal_metadata, ref_type=ArtifactType.pdf,
         field="SOW VERSION", pdf=True):
    _N[0] += 1
    text = " | ".join(f"{c}: {v}" for c, v in zip(COLS, cells)) if pdf else " | ".join(cells)
    loc = {"page": 1, "block_kind": "table", "block_id": "blk_t", "row_index": row} if pdf else {
        "table_index": 0, "row": row}
    ref = SourceRef(id=f"src_{artifact}_{_N[0]}", artifact_id=artifact, artifact_type=ref_type,
                    filename=f"{artifact}.{ref_type.value}", locator=loc,
                    extraction_method="test", parser_version="t")
    return EvidenceAtom(
        id=f"atm_{artifact}_{_N[0]}", project_id="p", artifact_id=artifact, atom_type=atype,
        raw_text=text, normalized_text=text.lower(),
        value={"kind": "table_row", "columns": COLS, "cells": dict(zip(COLS, cells)),
               "field_name": field, "value": cells[0]},
        entity_keys=[], source_refs=[ref], authority_class=AuthorityClass.machine_extractor,
        confidence=conf, review_flags=[], review_status=ReviewStatus.auto_accepted, parser_version="test",
    )


V1 = ["v.1", "Oskar Lindqvist", "07/09/2026", "First"]
V2 = ["v.2", "Dana Whitfield", "07/16/2026", "Second"]


def test_two_rows_of_one_table_both_stand():
    v1 = _row("art_signed", V1, row=0, conf=0.8)
    v2 = _row("art_signed", V2, row=1, conf=0.9)
    out = semantic_dedup_atoms([v1, v2])
    assert v1 in out and v2 in out


def test_a_row_is_not_folded_into_another_documents_different_row():
    order = {"art_docx": (0, 1.0, 0), "art_signed": (0, 2.0, 1)}
    docx = _row("art_docx", V1, row=1, ref_type=ArtifactType.docx, pdf=False)
    broken = _row("art_signed", ["v.1", "O", "skar Lindqvist", "First"], row=0)
    out = semantic_dedup_atoms([docx, broken], doc_order=order)
    assert docx in out and broken in out


def test_the_same_row_still_folds_and_its_document_keeps_a_copy():
    order = {"art_v1": (0, 1.0, 0), "art_signed": (0, 2.0, 1)}
    first = _row("art_v1", V1, row=0)
    later = _row("art_signed", V1, row=0)
    before = [first, later]
    out = semantic_dedup_atoms(list(before), doc_order=order)
    assert out == [first]
    copies = split_copies(before, out, stage="semantic_dedup")
    assert copies == [later] and COPY_FLAG in later.review_flags


def test_prose_metadata_still_collapses_on_its_key():
    a = _row("art_a", V1, row=0)
    b = _row("art_a", V2, row=1)
    for x in (a, b):
        x.value = {"field_name": "SOW VERSION", "value": "v.1"}
        x.source_refs[0].locator = {"page": 1, "line": 3}
    assert len(semantic_dedup_atoms([a, b])) == 1


# ---------------------------------------------------------------------------
# Every folded atom names a survivor that stands, or is kept.
#
# Shape of live 010353 (synthetic names): the SOW's site sentence and the
# intake JSON's site.name were typed physical_site with no site id, so the site
# merge dropped them as "covered" by the one located site -- the dispatch-brief
# fragment in a THIRD document -- without folding them into anything. They were
# suppressed with survivor null and both documents lost their site line.
# ---------------------------------------------------------------------------

from app.core.cross_doc_copies import SURVIVOR_KEY, settle_folds  # noqa: E402
from app.core.semantic_dedup import take_folds  # noqa: E402


def _site(text, artifact, value, ref_type=ArtifactType.txt, loc=None):
    _N[0] += 1
    ref = SourceRef(id=f"src_{artifact}_{_N[0]}", artifact_id=artifact, artifact_type=ref_type,
                    filename=f"{artifact}.{ref_type.value}", locator=dict(loc or {"line": _N[0]}),
                    extraction_method="test", parser_version="t")
    return EvidenceAtom(
        id=f"atm_{artifact}_{_N[0]}", project_id="p", artifact_id=artifact, atom_type=AtomType.physical_site,
        raw_text=text, normalized_text=text.lower(), value=dict(value), entity_keys=[],
        source_refs=[ref], authority_class=AuthorityClass.machine_extractor, confidence=0.8,
        review_flags=[], review_status=ReviewStatus.auto_accepted, parser_version="test",
    )


def _dedup_and_settle(atoms, order):
    take_folds()
    out = semantic_dedup_atoms(list(atoms), doc_order=order)
    copies = split_copies(atoms, out, stage="semantic_dedup")
    out, more, restored = settle_folds(atoms, out, copies, take_folds(), stage="semantic_dedup")
    return out, copies + more, restored


def test_site_lines_of_other_documents_are_kept_as_their_copies():
    brief = ("1 x Camera system(s) - Dome 4MP Camera; 1 x Mount / mast - 20 foot pole "
             "at Ridgeview Yard located at 4410 SR-31, Kenton, OH")
    sow = _site("This SOW supports the Ridgeview Yard site at 4410 SR-31, Kenton, OH on behalf of Acme.",
                "art_sow", {"name": "Ridgeview Yard", "address": "4410 SR-31, Kenton, OH"}, ArtifactType.pdf)
    jsn = _site("site.name: Ridgeview Yard", "art_json", {"name": "Ridgeview Yard"},
                loc={"kind": "json_value", "json_pointer": "/site/name"})
    disp = _site(brief, "art_md", {"site_id": "RIDGEVIEW-YARD-4410-SR-31-KENTON-OH", "name": "Kenton",
                                   "address": "4410 SR-31, Kenton, OH"})
    order = {"art_sow": (0, 1.0, 0), "art_md": (0, 2.0, 1), "art_json": (0, 3.0, 2)}
    out, copies, _restored = _dedup_and_settle([sow, jsn, disp], order)
    assert disp in out
    for a in (sow, jsn):
        assert a in copies and COPY_FLAG in a.review_flags
        assert a.value["duplicate_of"]["atom_id"] == disp.id


def test_a_fold_into_nothing_is_put_back():
    a = _site("Ridgeview Yard", "art_md", {"name": "Ridgeview Yard"})
    b = _site("Ridgeview Yard gate", "art_md", {"name": "Ridgeview Yard"})
    # A stage that dropped b without folding it anywhere.
    out, copies, restored = settle_folds([a, b], [a], [], {}, stage="semantic_dedup")
    assert out == [a, b] and restored == [b] and copies == []


def test_a_same_document_fold_records_its_survivor():
    first = _row("art_signed", V1, row=0)
    again = _row("art_signed", V1, row=0)
    take_folds()
    out = semantic_dedup_atoms([first, again])
    out, more, restored = settle_folds([first, again], out, [], take_folds(), stage="semantic_dedup")
    assert out == [first] and not more and not restored
    assert again.value[SURVIVOR_KEY]["atom_id"] == first.id
