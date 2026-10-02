"""Every document keeps its own atom for every line it holds.

Live 000132 (after the earliest-owner rule): the v2 SOW still listed 35 atoms
owned by the v1 SOW, the v2 Deal Kit 18 owned by v1, a HubSpot note 6 owned by
an email whose bullets it quoted, and the v1 SOW's "Supported Locations" cities
were credited to other documents. Each was a fold that merged the document's
source ref onto another document's survivor by a path ``split_copies`` never
saw (a type-specific pass, a site merge, a list-split piece, a quoted line), so
the document had no atom of its own for the line.

Live 010003: lines of earlier emails sat under later replies filed in another
thread, and a signed SOW listed the draft's atoms as its own.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.core.cross_doc_copies import (
    COPY_FLAG,
    ensure_own_copies,
    is_cross_doc_copy,
    quoted_in,
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


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    monkeypatch.setenv("SOWSMITH_DISABLE_LLM", "1")


_N = [0]


def _ref(artifact, atype=ArtifactType.txt, locator=None):
    _N[0] += 1
    return SourceRef(id=f"src_{artifact}_{_N[0]}", artifact_id=artifact, artifact_type=atype,
                     filename=f"{artifact}.{atype.value}", locator=dict(locator or {"line": _N[0]}),
                     extraction_method="test", parser_version="t")


def _atom(text, artifact, *, atype=AtomType.scope_item, value=None, refs=None, flags=None,
          ref_type=ArtifactType.txt, locator=None):
    _N[0] += 1
    return EvidenceAtom(
        id=f"atm_{artifact}_{_N[0]}", project_id="p", artifact_id=artifact,
        atom_type=atype, raw_text=text, normalized_text=text.lower(), value=dict(value or {}),
        entity_keys=[], source_refs=list(refs or [_ref(artifact, ref_type, locator)]),
        authority_class=AuthorityClass.machine_extractor, confidence=0.7,
        review_flags=list(flags or []), review_status=ReviewStatus.auto_accepted, parser_version="test",
    )


def _fold(winner, loser):
    """What every dedup pass does to a survivor: carry the loser's refs."""
    for r in loser.source_refs:
        if r not in winner.source_refs:
            winner.source_refs.append(r)


# ---------------------------------------------------------------------------
# The sweep: one copy per document a survivor cites
# ---------------------------------------------------------------------------

def test_docx_table_row_folded_onto_v1_gets_a_copy_in_v2_with_v2s_locator():
    row = "Install wireless access point | 4 | Main office"
    v1 = _atom(row, "sow_v1", atype=AtomType.raw_table_row, ref_type=ArtifactType.docx,
               locator={"table": 0, "row": 1})
    v2 = _atom(row, "sow_v2", atype=AtomType.raw_table_row, ref_type=ArtifactType.docx,
               locator={"table": 0, "row": 3})
    _fold(v1, v2)
    copies = ensure_own_copies([v1], [])
    assert len(copies) == 1
    c = copies[0]
    assert c.artifact_id == "sow_v2" and is_cross_doc_copy(c)
    assert c.atom_type == AtomType.raw_table_row and c.raw_text == row
    assert [r.locator for r in c.source_refs] == [{"table": 0, "row": 3}], "the copy cites v2's own row"
    assert c.value["duplicate_of"]["atom_id"] == v1.id
    assert "sow_v2" in v1.value["also_in_documents"]
    assert not is_cross_doc_copy(v1)
    # Idempotent: a second sweep sees the copy.
    assert ensure_own_copies([v1], copies) == []


def test_xlsx_row_typed_twice_gets_a_copy_of_the_type_that_folded():
    """A Deal Kit row is a raw_table_row AND a bom_line. v2's bom_line folded
    onto v1's while v2's raw_table_row stood: v2 still needs its bom_line."""
    row = "C9300-48P | Catalyst 9300 48-port | 2 | 5200"
    v1_bom = _atom(row, "kit_v1", atype=AtomType.bom_line, ref_type=ArtifactType.xlsx,
                   locator={"sheet": "BOM", "row": 2})
    v2_raw = _atom(row, "kit_v2", atype=AtomType.raw_table_row, ref_type=ArtifactType.xlsx,
                   locator={"sheet": "BOM", "row": 2})
    v2_bom = _atom(row, "kit_v2", atype=AtomType.bom_line, ref_type=ArtifactType.xlsx,
                   locator={"sheet": "BOM", "row": 2})
    kept = [v1_bom, v2_raw]
    # split_copies cannot see this fold: the same words survive in v2.
    assert split_copies([v1_bom, v2_raw, v2_bom], kept, stage="semantic_dedup") == []
    _fold(v1_bom, v2_bom)
    copies = ensure_own_copies(kept, [])
    assert [(c.artifact_id, c.atom_type) for c in copies] == [("kit_v2", AtomType.bom_line)]


def test_a_list_split_city_credited_to_another_document_comes_back_to_the_sow():
    """The SOW's list line CONTAINS "Plymouth, MI"; that is not the SOW's
    atom for the city. Its split piece folded onto the note's site."""
    parent = _atom("Supported Locations: Delphos, OH, Hudson, WI, Plymouth, MI, Troy, MI, Wilmington, DE",
                   "sow_v1", ref_type=ArtifactType.docx, locator={"paragraph": 4})
    kept = [parent]
    for city in ("Plymouth, MI", "Troy, MI", "Wilmington, DE"):
        site = _atom(city, "note", atype=AtomType.physical_site)
        piece = _atom(city, "sow_v1", atype=AtomType.physical_site, ref_type=ArtifactType.docx,
                      locator={"paragraph": 4, "list_item": city})
        _fold(site, piece)
        kept.append(site)
    copies = ensure_own_copies(kept, [])
    assert sorted((c.artifact_id, c.raw_text) for c in copies) == [
        ("sow_v1", "Plymouth, MI"), ("sow_v1", "Troy, MI"), ("sow_v1", "Wilmington, DE"),
    ]
    assert all(c.atom_type == AtomType.physical_site for c in copies)


def test_a_document_that_already_holds_the_line_gets_no_second_copy():
    v1 = _atom("Provide as-built drawings after cutover.", "sow_v1")
    v2 = _atom("Provide as-built drawings after cutover.", "sow_v2")
    v2_dup = _atom("Provide as-built drawings after cutover.", "sow_v2")
    _fold(v1, v2_dup)
    assert ensure_own_copies([v1, v2], []) == []


def test_atoms_built_from_several_documents_are_not_copied():
    conflict = _atom("Site address differs between the SOW and the kit", "sow_v1",
                     atype=AtomType.open_question, value={"kind": "cross_document_conflict"},
                     flags=["cross_document_conflict"])
    conflict.source_refs.append(_ref("kit_v1"))
    assert ensure_own_copies([conflict], []) == []


# ---------------------------------------------------------------------------
# Quoted lines: a note's quote is its own line; a reply's quote is not
# ---------------------------------------------------------------------------

BULLETS = [
    "Replace the core switch stack in each office server room",
    "Install two new wireless access points per floor",
    "Rack and dress all patch panels in the MDF",
]


def test_a_note_quoting_an_emails_bullets_keeps_its_own_copies():
    mail = [_atom(b, "mail", value={"kind": "email_body_line", "message_index": 0}, ref_type=ArtifactType.email)
            for b in BULLETS]
    note = [_atom(f"> - {b}", "note", value={"kind": "hubspot_note_body", "quoted": True},
                  flags=["hubspot_note_parser"]) for b in BULLETS]
    before = mail + note
    kept = [a for a in before if a.artifact_id == "mail"]
    for m, n in zip(mail, note):
        _fold(m, n)
    copies = split_copies(before, kept, stage="pasted_note_dedup")
    assert sorted(c.id for c in copies) == sorted(n.id for n in note), "each quoted note bullet is a copy"
    assert all(c.artifact_id == "note" and COPY_FLAG in c.review_flags for c in copies)
    assert ensure_own_copies(kept, copies) == [], "nothing left for the sweep"


def test_a_reply_quoting_an_earlier_email_is_not_credited_the_line():
    original = _atom(BULLETS[0], "mail_a", value={"kind": "email_body_line", "message_index": 0},
                     ref_type=ArtifactType.email)
    echo = _atom(BULLETS[0], "mail_b", value={"kind": "email_body_line", "message_index": 1, "quoted": True},
                 ref_type=ArtifactType.email)
    _fold(original, echo)
    assert split_copies([original, echo], [original], stage="semantic_dedup") == []
    assert quoted_in(original) == {"mail_b"}
    assert ensure_own_copies([original], []) == [], "the reply's quote is the original's line"


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

def _compile(tmp_path: Path):
    from app.core.compiler import compile_project
    from app.core.orbitbrief_envelope import build_orbitbrief_envelope

    r = compile_project(tmp_path, project_id="p", allow_errors=True, use_cache=False)
    env = build_orbitbrief_envelope(project_dir=tmp_path, compile_result=r)
    return r, env


def _txt(a: dict) -> str:
    return str(a.get("raw_text") or a.get("text") or "")


def _foreign_in_atom_ids(env) -> dict[str, list[str]]:
    by_id = {a["id"]: a for a in env["atoms"]}
    out: dict[str, list[str]] = {}
    for d in env["documents"]:
        for i in d["atom_ids"]:
            a = by_id.get(i)
            if a and a["artifact_id"] != d["artifact_id"]:
                out.setdefault(d["filename"], []).append(_txt(a))
    return out


def test_envelope_lists_a_sows_own_copy_not_the_earlier_sows_atom(tmp_path: Path):
    docx = pytest.importorskip("docx")
    for name, extra in (("000132 SOW v1.docx", []), ("000132 SOW v2.docx", ["PurTera will provide as-built drawings."])):
        d = docx.Document()
        d.add_heading("Statement of Work", 1)
        for t in ["PurTera will install two new wireless access points per floor."] + extra:
            d.add_paragraph(t, style="List Bullet")
        d.save(str(tmp_path / name))
    r, env = _compile(tmp_path)
    assert _foreign_in_atom_ids(env) == {}, "no SOW lists the other SOW's atoms as its own"
    docs = {d["filename"]: d for d in env["documents"]}
    by_id = {a["id"]: a for a in env["atoms"]}
    v2_lines = [_txt(by_id[i]) for i in docs["000132 SOW v2.docx"].get("copy_atom_ids", [])]
    assert "PurTera will install two new wireless access points per floor." in v2_lines


def test_signed_pdf_sow_keeps_its_own_clauses(tmp_path: Path):
    fitz = pytest.importorskip("fitz")
    lines = [
        "Services Fees hereunder are FIXED FEES and will be invoiced upon completion. Total: $1,622.00",
        "Technicians submit a timesheet every Monday for the prior week of services.",
        "PurTera will install twelve wireless access points at the Dallas warehouse.",
    ]
    for name, extra in (("010003 SOW draft.pdf", []), ("010003 SOW signed.pdf", ["DocuSign Envelope ID: 1234-ABCD"])):
        doc = fitz.open()
        page = doc.new_page()
        for i, line in enumerate(lines + extra):
            page.insert_text((72, 72 + 40 * i), line, fontsize=9)
        doc.save(str(tmp_path / name))
    r, env = _compile(tmp_path)
    assert _foreign_in_atom_ids(env) == {}
    by_id = {a["id"]: a for a in env["atoms"]}
    signed = next(d for d in env["documents"] if d["filename"] == "010003 SOW signed.pdf")
    held = " ".join(_txt(by_id[i]) for i in signed["atom_ids"] + signed.get("copy_atom_ids", []))
    for line in lines:
        assert line[:60] in held, f"the signed SOW shows its own {line[:40]!r}"


EARLIER = """From: Saga Ops <saga@customer.com>
To: Victor Lee <victor@purtera-it.com>
Subject: Site list for the rollout
Date: Mon, 10 Aug 2026 09:00:00 -0400
Message-ID: <a1@customer.com>
Content-Type: text/plain; charset=utf-8

Victor,

We need twelve access points installed at the Dallas warehouse.
Please schedule the cabling crew for the second week of September.
The loading dock entrance is on the north side of the building.
"""

LATER_OTHER_THREAD = """From: Victor Lee <victor@purtera-it.com>
To: Saga Ops <saga@customer.com>
Subject: Crew dates
Date: Tue, 11 Aug 2026 16:00:00 -0400
Message-ID: <b2@purtera-it.com>
Content-Type: text/plain; charset=utf-8

Thanks, we will confirm the crew dates by Friday.

On Mon, Aug 10, 2026 at 9:00 AM Saga Ops <saga@customer.com> wrote:
> Victor,
>
> We need twelve access points installed at the Dallas warehouse.
> Please schedule the cabling crew for the second week of September.
> The loading dock entrance is on the north side of the building.
"""


def test_a_reply_in_another_thread_does_not_hold_the_earlier_emails_lines(tmp_path: Path):
    (tmp_path / "010003-hs-email-111.eml").write_text(EARLIER, encoding="utf-8")
    (tmp_path / "010003-hs-email-222.eml").write_text(LATER_OTHER_THREAD, encoding="utf-8")
    r, env = _compile(tmp_path)
    by_id = {a["id"]: a for a in env["atoms"]}
    docs = {d["filename"]: d for d in env["documents"]}
    later = docs["010003-hs-email-222.eml"]
    later_text = [_txt(by_id[i]) for i in later["atom_ids"] + later.get("copy_atom_ids", [])]
    earlier = docs["010003-hs-email-111.eml"]
    earlier_text = [_txt(by_id[i]) for i in earlier["atom_ids"]]
    for line in ("We need twelve access points installed at the Dallas warehouse.",
                 "Please schedule the cabling crew for the second week of September.",
                 "The loading dock entrance is on the north side of the building."):
        assert line in earlier_text, "the earlier email keeps its own line"
        assert line not in later_text, "the reply only quoted it"
    assert any("crew dates by Friday" in t for t in later_text)
    assert not any(t.startswith("From: Saga Ops") for t in later_text), "nor the quoted header of a held message"
    assert _foreign_in_atom_ids(env) == {}
