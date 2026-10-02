"""Stakeholder dedup never drops the only copy of a multi-person contact row.

Deal 010353: the SOW "CUSTOMER CONTACTS" row named John Ozuna-Diaz AND Danny,
with "Danny's phone not provided". It keyed on John (also listed elsewhere),
folded into John's record, and Danny, his role and the note left the compile.
"""
from __future__ import annotations

from app.core import fold_invariants as fold
from app.core.schemas import ArtifactType, AtomType, AuthorityClass, EvidenceAtom, ReviewStatus, SourceRef
from app.core.semantic_dedup import dedupe_stakeholder_atoms, semantic_dedup_atoms


def _atom(aid, text, value=None, *, art="a", atype=AtomType.stakeholder):
    return EvidenceAtom(
        id=aid, project_id="p", artifact_id=art, atom_type=atype, raw_text=text,
        normalized_text=text.lower(), value=value or {}, entity_keys=[],
        source_refs=[SourceRef(id=f"s{aid}", artifact_id=art, artifact_type=ArtifactType.docx,
                               filename=f"{art}.docx", locator={"table_index": 1, "row": len(aid)},
                               extraction_method="t", parser_version="t")],
        authority_class=AuthorityClass.contractual_scope, confidence=0.8,
        review_status=ReviewStatus.needs_review, review_flags=[], parser_version="t",
    )


ROW = ("CUSTOMER CONTACTS | John Ozuna-Diaz | Client Support Manager | john@ox.com | "
       "Danny Reyes | Site Lead | Danny's phone not provided")


def _texts(atoms):
    return " ".join(f"{a.raw_text} {a.value}" for a in atoms)


def test_multi_person_row_survives_when_one_person_is_listed_elsewhere() -> None:
    john = _atom("j1", "John Ozuna-Diaz | Client Support Manager | john@ox.com",
                 {"name": "John Ozuna-Diaz", "email": "john@ox.com", "role": "Client Support Manager"},
                 art="intake")
    row = _atom("j2", ROW, {"name": "John Ozuna-Diaz", "email": "john@ox.com"}, art="sow")
    for order in ([john, row], [row, john]):
        out = dedupe_stakeholder_atoms(list(order))
        text = _texts(out)
        assert "Danny Reyes" in text and "phone not provided" in text, [a.raw_text for a in out]
        assert "Site Lead" in text


def test_same_document_multi_person_row_survives() -> None:
    john = _atom("k1", "John Ozuna-Diaz", {"name": "John Ozuna-Diaz"}, art="sow")
    row = _atom("k2", ROW, {"name": "John Ozuna-Diaz"}, art="sow")
    out = dedupe_stakeholder_atoms([john, row])
    assert "Danny's phone not provided" in _texts(out)
    out = semantic_dedup_atoms([_atom("k3", "John Ozuna-Diaz", {"name": "John Ozuna-Diaz"}, art="sow"),
                                _atom("k4", ROW, {"name": "John Ozuna-Diaz"}, art="sow")])
    assert "Danny's phone not provided" in _texts(out)


def test_absence_note_alone_is_detail() -> None:
    w = _atom("n1", "Danny Reyes | Site Lead", {"name": "Danny Reyes"})
    l = _atom("n2", "Danny Reyes | Site Lead | phone not provided", {"name": "Danny Reyes"})
    assert "note:not provided" in fold.person_detail_only_the_loser_states(w, l)
    assert not fold.person_detail_only_the_loser_states(l, w)
    out = dedupe_stakeholder_atoms([w, l])
    assert "phone not provided" in _texts(out)


def test_extra_name_and_role_are_detail_labels_are_not() -> None:
    w = _atom("x1", "John Ozuna-Diaz", {"name": "John Ozuna-Diaz"})
    assert fold.person_detail_only_the_loser_states(
        w, _atom("x2", "John Ozuna-Diaz | Danny Reyes", {"name": "John Ozuna-Diaz"})) >= {"word:danny", "word:reyes"}
    assert "word:lead" in fold.person_detail_only_the_loser_states(
        w, _atom("x3", "John Ozuna-Diaz | Site Lead", {"name": "John Ozuna-Diaz"}))
    for text in ("Name: John Ozuna-Diaz | Phone: | Email:", "John Ozuna-Diaz - Contact"):
        assert not fold.person_detail_only_the_loser_states(w, _atom("x4", text)), text


def test_plain_duplicates_still_fold() -> None:
    a = _atom("d1", "John Ozuna-Diaz | Client Support Manager", {"name": "John Ozuna-Diaz"}, art="x")
    b = _atom("d2", "Name: John Ozuna-Diaz, Client Support Manager", {"name": "John Ozuna-Diaz"}, art="y")
    assert len(dedupe_stakeholder_atoms([a, b])) == 1


# ── the table-row blob of a contact row the classifier read as one person ──


def _cell(atom, row=7):
    atom.source_refs[0].locator = {"table_index": 4, "row": row}
    return atom


def test_contact_row_blob_naming_a_second_person_is_not_a_double() -> None:
    from app.core.semantic_dedup import _suppress_table_row_blob_doubles

    john = _cell(_atom("t1", "Name: John Ozuna-Diaz | Title: Client Support Manager | Email: john@ox.com",
                       {"name": "John Ozuna-Diaz", "email": "john@ox.com"}, art="sow"))
    blob = _cell(_atom("t2", ROW, {"kind": "table_row"}, art="sow", atype=AtomType.scope_item))
    out = _suppress_table_row_blob_doubles([john, blob])
    assert "Danny's phone not provided" in _texts(out)


def test_contact_row_blob_saying_nothing_more_still_folds() -> None:
    from app.core.semantic_dedup import _suppress_table_row_blob_doubles

    john = _cell(_atom("u1", "Name: John Ozuna-Diaz | Title: Client Support Manager | Email: john@ox.com",
                       {"name": "John Ozuna-Diaz", "email": "john@ox.com"}, art="sow"))
    blob = _cell(_atom("u2", "John Ozuna-Diaz | Client Support Manager | john@ox.com",
                       {"kind": "table_row"}, art="sow", atype=AtomType.scope_item))
    assert _suppress_table_row_blob_doubles([john, blob]) == [john]


# ── merge_signature_rows: a contact row on the signature page ──


def test_contact_row_on_signature_page_naming_a_second_person_is_kept() -> None:
    from app.core.atom_type_sanity import merge_signature_rows

    def _pg(a):
        a.source_refs[0].locator = {"table_index": 9, "row": len(a.id), "page": 3}
        return a

    rows = [
        _pg(_atom("s1", "Ox: Name: John Ozuna-Diaz | Customer: Name: Pat Doe", {"name": "John Ozuna-Diaz"},
                  atype=AtomType.signatory)),
        _pg(_atom("s22", "Ox: Title: Client Support Manager | Customer: Title: Director", atype=AtomType.signatory)),
        _pg(_atom("s333", "Name: John Ozuna-Diaz | Name: Danny Reyes | Danny's phone not provided",
                  {"name": "John Ozuna-Diaz"})),
    ]
    merge_signature_rows(rows)
    assert "Danny's phone not provided" in _texts(rows)


# ── 010087: Trent's SOW contact row vs a bare-name mention ──


def test_contact_row_with_email_survives_a_bare_name_mention() -> None:
    order = {"mail": (0,), "sow": (1,)}
    row_values = (
        {"name": "Trent Smith", "email": "trent@acme.com", "role": "Project Manager"},
        {"name": "Trent Smith", "email": "trent@acme.com", "role": "Project Manager", "phone": "555-201-3344"},
    )
    for vals in row_values:
        row_text = "Trent Smith | Project Manager | trent@acme.com" + (" | 555-201-3344" if "phone" in vals else "")
        for bare_text, bare_val in (("Trent Smith", {"name": "Trent Smith"}),
                                    ("Trent Smith <trent@acme.com>", {"name": "Trent Smith", "email": "trent@acme.com"})):
            for row_first in (True, False):
                bare = _atom("tb", bare_text, dict(bare_val), art="mail")
                bare.confidence = 0.95
                row = _atom("tr", row_text, dict(vals), art="sow")
                row.confidence = 0.5
                atoms = [row, bare] if row_first else [bare, row]
                out = dedupe_stakeholder_atoms(semantic_dedup_atoms(atoms, doc_order=order), doc_order=order)
                assert any(a.raw_text == row_text for a in out), (bare_text, vals, [a.raw_text for a in out])
