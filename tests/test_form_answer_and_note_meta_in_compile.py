"""Two compile stages that tore a deal's own records into "fragment" rows.

Live 000037 after #484:
* prose_list_split cut a PDF form answer ("PACKING MATERIALS (check all that
  apply): a; b; c; d; e") into five label-less children;
* the person pass read the note's own "note_id=... | author=Trent Torrence"
  line as prose and minted the note's author as a bare-name stakeholder row.
"""
from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from app.core.compiler import compile_project
from app.core.schemas import AtomType, AuthorityClass
from app.parsers.pdf._shared import _make_atom

ANSWER = ("PACKING MATERIALS (check all that apply): Wooden pallets; Shrink wrap rolls; "
          "Moving blankets; Cardboard boxes #___; OTHER (Please describe): foam corner guards")
NOTE = (
    "HubSpot Note: Note\nHubSpot Note ID: 108134987187\nDate: 2026-04-16T17:08:06.021Z\n"
    "Author: Trent Torrence\nAuthor-Email: t@example.com\n\n"
    "Hey Guys -\nThink a couple small little decoms and pack/ships.\n"
    "2x locations with 7x nodes at each location\nThanks!\nDan\n"
)


@pytest.fixture(scope="module")
def compiled(tmp_path_factory):
    d: Path = tmp_path_factory.mktemp("deal")
    (d / "000037-hs-note-108134987187-Note.txt").write_text(NOTE)
    doc = fitz.open()
    doc.new_page().insert_text((50, 60), "FREIGHT FORM", fontsize=12)
    doc.save(str(d / "freight.pdf"))

    import app.parsers.pdf.form_answers as fa

    def _answers(*, pdf_path, project_id, artifact_id, filename, parser_version):
        return {0: [_make_atom(
            text=ANSWER, project_id=project_id, artifact_id=artifact_id, filename=filename,
            parser_version=parser_version, atom_type=AtomType.scope_item,
            authority_class=AuthorityClass.customer_current_authored, confidence=0.9,
            locator={"page": 0, "block_id": "p0_form_answers", "row_index": 0,
                     "extraction": fa.FORM_ANSWER_EXTRACTION},
            value={"kind": "form_answer", "question": "PACKING MATERIALS (check all that apply)",
                   "answers": ANSWER.split(": ", 1)[1].split("; "), "sources": ["widget"]},
        )]}

    mp = pytest.MonkeyPatch()
    mp.setattr(fa, "form_answer_atoms", _answers)
    try:
        return compile_project(d, project_id="t", use_cache=False, allow_errors=True)
    finally:
        mp.undo()


def _value(a):
    return a.value if isinstance(a.value, dict) else {}


def test_form_answer_is_not_split_into_its_options(compiled):
    answers = [a for a in compiled.atoms if _value(a).get("kind") == "form_answer"]
    assert [a.raw_text for a in answers] == [ANSWER]
    parent = answers[0].id
    assert not [a for a in compiled.atoms if _value(a).get("_parent_atom_id") == parent]


def test_note_author_is_not_a_stakeholder_row(compiled):
    people = [a for a in compiled.atoms if a.atom_type.value == "stakeholder"
              and "Trent Torrence" in (a.raw_text or "")]
    assert people == []
    assert any(_value(a).get("kind") == "hubspot_note_meta" for a in compiled.atoms)
