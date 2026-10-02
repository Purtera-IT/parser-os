"""A collapse/rollup gate removes a line only when a kept atom says it.

On two-column PDF pages, duplicate_atom_collapse and table_rollup dropped a
spec panel and numbered steps 6-15: each step was a >0.92-similar "duplicate"
of its neighbour, and the table rows became "N table rows (rolled up)". They
survived only in the suppression ledger, where nobody can see or label them.
"""
from __future__ import annotations

from app.core.entity_resolution import collapse_duplicate_atoms
from app.core.schemas import ArtifactType, AtomType, AuthorityClass, EvidenceAtom, ReviewStatus, SourceRef
from app.core.suppression_ledger import keep_unsurvived_lines
from app.core.table_rollup import roll_up_table_rows


def _atom(text, *, artifact="art_pdf", atype=AtomType.scope_item, value=None, conf=0.7,
          art_type=ArtifactType.pdf, filename="spec.pdf"):
    return EvidenceAtom(
        id=f"atm_{abs(hash((artifact, text, atype.value))) % 10**9}", project_id="p", artifact_id=artifact,
        atom_type=atype, raw_text=text, normalized_text=text.lower(), value=dict(value or {}),
        entity_keys=[],
        source_refs=[SourceRef(id=f"src_{artifact}", artifact_id=artifact, artifact_type=art_type,
                               filename=filename, locator={}, extraction_method="test", parser_version="t")],
        authority_class=AuthorityClass.machine_extractor, confidence=conf, review_flags=[],
        review_status=ReviewStatus.auto_accepted, parser_version="test",
    )


#: Steps whose text differs by a word: character edit distance calls them
#: >92% alike, and the collapse kept one of five.
STEPS = [
    f"Step: Install the ceiling-mounted wireless access point in the {room} using the provided bracket kit"
    for room in ("east lobby", "west lobby", "north lobby", "south lobby", "main lobby")
]


def test_numbered_steps_are_not_duplicates_of_each_other():
    before = [_atom(t) for t in STEPS]
    collapsed = collapse_duplicate_atoms(list(before))
    assert len(collapsed) < len(before), "precondition: the fuzzy collapse merges the steps"
    kept, restored = keep_unsurvived_lines(before, collapsed, stage="duplicate_atom_collapse")
    assert {a.raw_text for a in kept} == set(STEPS)
    assert restored and all(a.value["_survivor_guard"]["stage"] == "duplicate_atom_collapse" for a in restored)
    # reading order is the document's
    assert [a.raw_text for a in kept] == STEPS


def test_a_true_duplicate_stays_collapsed():
    line = "Contractor shall label every cable at both ends with the outlet ID."
    before = [_atom(line, conf=0.9), _atom(line + " ", conf=0.5)]
    collapsed = collapse_duplicate_atoms(list(before))
    kept, restored = keep_unsurvived_lines(before, collapsed, stage="duplicate_atom_collapse")
    assert restored == [] and len(kept) == 1


def _row(i, *, artifact, art_type, filename):
    return _atom(
        f"Step {i} | Mount rack rail {i} | $1{i:02d}.00", artifact=artifact, atype=AtomType.raw_table_row,
        art_type=art_type, filename=filename,
        value={"_columns": ["Step", "Task", "Price"], "_row": [str(i), f"Mount rack rail {i}", 100.0 + i],
               "_sheet": "page 3"},
    )


def test_table_rollup_keeps_a_pdfs_rows():
    from app.core.compiler import _keep_unsurvived_document_rows

    before = [_row(i, artifact="art_pdf", art_type=ArtifactType.pdf, filename="spec.pdf") for i in range(60)]
    rolled, _ = roll_up_table_rows(list(before))
    assert len(rolled) == 1, "precondition: the rollup folds the page's rows"
    warnings: list[str] = []
    kept = _keep_unsurvived_document_rows(before, rolled, warnings)
    assert [a.raw_text for a in kept] == [a.raw_text for a in before]
    assert not any((a.value or {}).get("_rolled_up") for a in kept), "nothing is folded, so no count line"


def test_table_rollup_still_folds_a_spreadsheet():
    from app.core.compiler import _keep_unsurvived_document_rows

    before = [_row(i, artifact="art_xlsx", art_type=ArtifactType.xlsx, filename="rates.xlsx") for i in range(60)]
    rolled, _ = roll_up_table_rows(list(before))
    kept = _keep_unsurvived_document_rows(before, rolled, [])
    assert len(kept) == 1 and kept[0].value.get("_rolled_up")
