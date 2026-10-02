"""Who owns a line two documents share, and what the other document keeps.

Deal 000132's labeling audit (compile 5181b802):

* HubSpot note 110373542233 was written in May; a June email repeated its
  eight scope bullets and its city list. The dedup kept the later, larger
  email as the survivor, so the note showed 5 atoms and its "Locations" lines
  had none (items 2 and 4).
* The v2 SOW repeats v1's clauses; a clause folded across documents simply
  disappeared from the document that lost (item 5).

The earliest document -- by its own date, else by reading order -- owns the
canonical atom. Every other document keeps its own copy of the line, flagged
``cross_doc_copy`` with ``duplicate_of`` naming the canonical atom, held out
of every count.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.core.cross_doc_copies import (
    COPY_FLAG,
    document_order,
    is_cross_doc_copy,
    split_copies,
)
from app.core.pasted_note_dedup import collapse_pasted_note_duplicates
from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    EvidenceAtom,
    ReviewStatus,
    SourceRef,
)
from app.core.semantic_dedup import cross_type_dedup_atoms, semantic_dedup_atoms


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    monkeypatch.setenv("SOWSMITH_DISABLE_LLM", "1")


_N = [0]


def _atom(text, artifact, value=None, *, atype=AtomType.scope_item, conf=0.7, flags=None, locator=None):
    _N[0] += 1
    return EvidenceAtom(
        id=f"atm_{artifact}_{_N[0]}", project_id="p", artifact_id=artifact,
        atom_type=atype, raw_text=text, normalized_text=text.lower(), value=dict(value or {}),
        entity_keys=[],
        source_refs=[SourceRef(id=f"src_{artifact}_{_N[0]}", artifact_id=artifact, artifact_type=ArtifactType.txt,
                               filename=f"{artifact}.txt", locator=dict(locator or {"line": _N[0]}),
                               extraction_method="test", parser_version="t")],
        authority_class=AuthorityClass.machine_extractor,
        confidence=conf, review_flags=list(flags or []), review_status=ReviewStatus.auto_accepted,
        parser_version="test",
    )


BULLETS = [
    "Replace the core switch stack in each office server room",
    "Install two new wireless access points per floor",
    "Rack and dress all patch panels in the MDF",
    "Label every drop at both ends per the customer standard",
    "Configure VLANs for voice, data and guest traffic",
    "Migrate the existing firewall rules to the new appliance",
    "Provide as-built drawings after the cutover is complete",
    "Remove and dispose of all decommissioned equipment",
]
CITIES = ["Delphos, OH", "Hudson, WI", "Plymouth, MI", "Troy, MI", "Tupelo, MS", "Wilmington, DE"]


# ---------------------------------------------------------------------------
# Document order
# ---------------------------------------------------------------------------


def test_document_order_reads_each_documents_own_date():
    mail = [_atom("x", "mail", {"kind": "email_body_line", "message_index": 0,
                                "authored_at": "Mon, 1 Jun 2026 10:00:00 -0400"})]
    # A quoted line's date is the date of the message it quotes, not the file's.
    mail.append(_atom("y", "mail", {"kind": "email_body_line", "quoted": True,
                                    "authored_at": "Fri, 1 May 2026 10:00:00 -0400"}))
    note = [_atom("note_id=1", "note", {"kind": "hubspot_note_meta", "date": "2026-05-29T18:58:14Z"})]
    sow = [_atom("clause", "sow", {})]
    order = document_order(mail + note + sow)
    assert order["note"] < order["mail"], "the May note precedes the June email"
    assert order["mail"] < order["sow"], "an undated document comes after the dated ones"

    order = document_order(mail + note + sow, provenance={"sow.txt": {"authored_at": "2026-01-02T00:00:00Z"}})
    assert order["sow"] < order["note"], "the manifest's authored time dates a file"


# ---------------------------------------------------------------------------
# Item 2 / 4: the earliest document is the survivor
# ---------------------------------------------------------------------------


def _note(lines, artifact="note"):
    return [_atom(t, artifact, {"kind": "hubspot_note_body"}, flags=["hubspot_note_parser"]) for t in lines]


def _mail(lines, artifact="mail", **extra):
    return [_atom(t, artifact, {"kind": "email_body_line", "message_index": 0, **extra}) for t in lines]


def test_pasted_note_dedup_gives_the_lines_to_the_earlier_note():
    # The lines themselves carry no dates -- the DOCUMENTS do. 000132: the
    # note's bullets and city list went to the later email.
    note = _note(BULLETS + ["Locations"] + CITIES)
    mail = _mail(BULLETS + ["Locations"] + CITIES)
    order = {"note": (0, 1_780_081_094.0, 1), "mail": (0, 1_780_322_400.0, 0)}
    kept, dropped = collapse_pasted_note_duplicates(mail + note, doc_order=order)
    assert all(a in kept for a in note), "the earlier note keeps every line"
    assert dropped and {a.artifact_id for a in dropped} == {"mail"}


def test_pasted_note_dedup_still_folds_a_note_pasted_from_an_earlier_mail():
    note = _note(BULLETS)
    mail = _mail(BULLETS)
    order = {"mail": (0, 1_000.0, 0), "note": (0, 2_000.0, 1)}
    kept, dropped = collapse_pasted_note_duplicates(mail + note, doc_order=order)
    assert {a.artifact_id for a in dropped} == {"note"}


def test_semantic_dedup_survivor_is_the_earliest_document_not_the_best_copy():
    text = "This quote does not include any permits or permit fees."
    v1 = _atom(text, "sow_v1", {"statement": text, "domain": "commercial"},
               atype=AtomType.pricing_assumption, conf=0.6)
    v2 = _atom(text, "sow_v2", {"statement": text, "domain": "commercial"},
               atype=AtomType.pricing_assumption, conf=0.95)
    kept = semantic_dedup_atoms([v2, v1], doc_order={"sow_v1": (1, 0.0, 0), "sow_v2": (1, 0.0, 1)})
    assert [a.artifact_id for a in kept if a.atom_type == AtomType.pricing_assumption] == ["sow_v1"]
    # Without an order nothing changes: the better copy wins, as before.
    v1b = _atom(text, "sow_v1", {"statement": text, "domain": "commercial"},
                atype=AtomType.pricing_assumption, conf=0.6)
    v2b = _atom(text, "sow_v2", {"statement": text, "domain": "commercial"},
                atype=AtomType.pricing_assumption, conf=0.95)
    assert [a.artifact_id for a in semantic_dedup_atoms([v2b, v1b])] == ["sow_v2"]


def test_cross_type_dedup_keeps_the_earliest_documents_line_and_each_documents_own_folds():
    line = "Label every drop at both ends per the customer standard"
    note = _atom(line, "note", {"kind": "hubspot_note_body"}, atype=AtomType.scope_item)
    mail_meta = _atom(line, "mail", {}, atype=AtomType.deal_metadata)
    mail_scope = _atom(line, "mail", {}, atype=AtomType.scope_item)
    order = {"note": (0, 1.0, 1), "mail": (0, 2.0, 0)}
    kept = cross_type_dedup_atoms([mail_meta, mail_scope, note], doc_order=order)
    assert note in kept, "the earlier note owns the line"
    assert mail_meta not in kept and mail_scope not in kept, "the mail's retypings fold onto it"


# ---------------------------------------------------------------------------
# Item 5: a cross-document fold leaves a copy under its own document
# ---------------------------------------------------------------------------


def test_split_copies_marks_cross_document_folds_and_leaves_in_document_folds_alone():
    text = "Customer will provide unescorted access to all MDF and IDF rooms."
    v1 = _atom(text, "sow_v1", {"statement": text}, atype=AtomType.pricing_assumption)
    v2 = _atom(text, "sow_v2", {"statement": text}, atype=AtomType.pricing_assumption)
    v2_twice = _atom(text, "sow_v2", {"statement": text}, atype=AtomType.pricing_assumption)
    before = [v1, v2, v2_twice]
    after = semantic_dedup_atoms(list(before), doc_order={"sow_v1": (1, 0.0, 0), "sow_v2": (1, 0.0, 1)})
    assert after == [v1]
    copies = split_copies(before, after, stage="semantic_dedup")
    assert copies, "v2 keeps a copy of the clause it shares with v1"
    assert all(c.artifact_id == "sow_v2" for c in copies)
    for c in copies:
        assert COPY_FLAG in c.review_flags and is_cross_doc_copy(c)
        assert c.value["duplicate_of"]["atom_id"] == v1.id
        assert c.value["duplicate_of"]["artifact_id"] == "sow_v1"
    assert v1.value["also_in_documents"] == ["sow_v2"]

    # Two copies of a line in ONE document are still one atom.
    a = _atom("Install two new wireless access points per floor", "sow_v1")
    b = _atom("Install two new wireless access points per floor", "sow_v1")
    assert split_copies([a, b], [a], stage="x") == []


def test_a_quoted_echo_is_not_a_copy():
    note = _note(["Maintenance and support of the technical IT infrastructure across all offices."])
    mail = _mail(["Maintenance and support of the technical IT infrastructure across all offices."], quoted=True)
    kept, dropped = collapse_pasted_note_duplicates(mail + note)
    assert dropped == mail
    assert split_copies(mail + note, kept, stage="pasted_note_dedup") == []


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

NOTE_FILE = """HubSpot Note: Scope for IT support
HubSpot Note ID: 110373542233
Date: 2026-05-29T18:58:14.007Z
Author: Trent Torrence
Author-Email: t@purtera-it.com

@Christopher Picchetti here is the scope we discussed

{bullets}
Locations
{cities}
"""

LATER_EMAIL = """From: Christopher Picchetti <cp@customer.com>
To: Trent Torrence <t@purtera-it.com>
Subject: IT support scope - pricing
Date: Mon, 1 Jun 2026 10:00:00 -0400
Message-ID: <later@customer.com>
Content-Type: text/plain; charset=utf-8

Trent, here is the scope we discussed for the pricing team.

{bullets}
Locations
{cities}
"""


def _write_note_and_email(tmp_path: Path) -> None:
    fill = {"bullets": "\n".join(f"- {b}" for b in BULLETS), "cities": "\n".join(CITIES)}
    (tmp_path / "000132-hs-note-110373542233.txt").write_text(NOTE_FILE.format(**fill), encoding="utf-8")
    (tmp_path / "000132-hs-email-later.eml").write_text(LATER_EMAIL.format(**fill), encoding="utf-8")


def _norm(t: str) -> str:
    return t.lstrip("- ").strip().rstrip(".")


def test_compile_the_earlier_note_owns_its_bullets_and_cities_and_the_email_keeps_copies(tmp_path: Path):
    from app.core.compiler import compile_project

    _write_note_and_email(tmp_path)
    r = compile_project(tmp_path, project_id="p", allow_errors=True, use_cache=False)

    note_id = next(a.artifact_id for a in r.atoms if (a.value or {}).get("hubspot_note_id"))
    mail_id = next(a.artifact_id for a in r.atoms if a.artifact_id != note_id)
    note_own = {_norm(a.raw_text) for a in r.atoms if a.artifact_id == note_id and not is_cross_doc_copy(a)}
    for line in BULLETS + CITIES:
        assert line in note_own, f"{line!r} must be the note's own atom"

    by_id = {a.id: a for a in r.atoms}
    mail_copies = [a for a in r.atoms if a.artifact_id == mail_id and is_cross_doc_copy(a)]
    mail_lines = {_norm(a.raw_text) for a in r.atoms if a.artifact_id == mail_id}
    for line in BULLETS + CITIES:
        assert line in mail_lines, f"the email still shows its own {line!r}"
    assert len(mail_copies) >= len(BULLETS) + len(CITIES)
    for c in mail_copies:
        canon = by_id.get(c.value["duplicate_of"]["atom_id"])
        assert canon is not None, "a copy points at an atom that is in the result"
        assert canon.artifact_id == note_id and not is_cross_doc_copy(canon)
        assert c.source_refs and c.source_refs[0].filename.endswith(".eml"), "a copy keeps its own locator"

    # A copy is not a suppression and is not counted.
    sup_ids = {a.id for a in r.suppressed_atoms}
    assert not any(c.id in sup_ids for c in mail_copies)
    from app.core.quality_metrics import compute_quality

    q = compute_quality(r)
    assert q.atom_count == sum(1 for a in r.atoms if not is_cross_doc_copy(a)
                               and "admission_regex" not in (a.review_flags or []))


def test_envelope_lists_copies_under_their_document_and_counts_them_nowhere(tmp_path: Path):
    from app.core.compiler import compile_project
    from app.core.orbitbrief_envelope import build_orbitbrief_envelope

    _write_note_and_email(tmp_path)
    r = compile_project(tmp_path, project_id="p", allow_errors=True, use_cache=False)
    env = build_orbitbrief_envelope(project_dir=tmp_path, compile_result=r)

    copies = [a for a in env["atoms"] if a.get("duplicate_of")]
    assert copies, "the copies reach the walk"
    docs = {d["artifact_id"]: d for d in env["documents"]}
    for c in copies:
        doc = docs[c["artifact_id"]]
        assert c["id"] in doc.get("copy_atom_ids", [])
        assert c["id"] not in doc["atom_ids"], "atom_ids never counts a copy"
        assert "cross_doc_copy" in c["review_flags"]


def _sow(path: Path, title: str, extra: list[str]) -> None:
    docx = pytest.importorskip("docx")
    d = docx.Document()
    d.add_heading(title, 1)
    d.add_heading("Scope of Work", 2)
    for t in ["PurTera will install two new wireless access points per floor."] + extra:
        d.add_paragraph(t, style="List Bullet")
    d.save(str(path))


def test_compile_v2_sow_keeps_its_copy_of_a_v1_clause(tmp_path: Path):
    from app.core.compiler import compile_project

    _sow(tmp_path / "000132 SOW v1.docx", "Statement of Work v1", [])
    _sow(tmp_path / "000132 SOW v2.docx", "Statement of Work v2",
         ["PurTera will provide as-built drawings after cutover."])
    r = compile_project(tmp_path, project_id="p", allow_errors=True, use_cache=False)
    name = {a.artifact_id: a.source_refs[0].filename for a in r.atoms if a.source_refs}
    clause = "PurTera will install two new wireless access points per floor."
    holders = {name[a.artifact_id] for a in r.atoms if a.raw_text == clause}
    assert holders == {"000132 SOW v1.docx", "000132 SOW v2.docx"}, "each SOW shows its own copy"
    canon = [a for a in r.atoms if a.raw_text == clause and not is_cross_doc_copy(a)]
    assert canon and {name[a.artifact_id] for a in canon} == {"000132 SOW v1.docx"}, "v1 owns it"
