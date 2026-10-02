"""A vision transcription of text the page already reads is a suppression.

Deal 010246's install guide (re-run on #268): the step page is both a text
layer and a picture of the steps. The vision pass transcribed the picture
and every step arrived twice, the copy with its number swapped ("Step 3:
Loosen the turnbuckles..." beside the text layer's "5 Loosen the
turnbuckles..."). When the same page's text layer already has an atom
carrying the vision line's words, the vision line is recorded as a
suppression (stage vision_copy_of_text_layer, naming the covering atom)
instead of emitted.
"""
from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from app.core import pdf_image_vision
from app.core.schemas import ArtifactType, AtomType, AuthorityClass, EvidenceAtom, ReviewStatus, SourceRef

STEP = "Loosen the turnbuckles on both guy wires until the mast can be lowered safely by hand."


def _text_atom(text: str, page: int) -> EvidenceAtom:
    return EvidenceAtom(
        id=f"t{page}{len(text)}", project_id="p", artifact_id="a", atom_type=AtomType.scope_item,
        raw_text=text, normalized_text=text.lower(), value={}, entity_keys=[],
        source_refs=[SourceRef(id="s", artifact_id="a", artifact_type=ArtifactType.pdf, filename="g.pdf",
                               locator={"page": page}, extraction_method="pdf_text", parser_version="t")],
        authority_class=AuthorityClass.contractual_scope, confidence=0.8,
        review_status=ReviewStatus.needs_review, review_flags=[], parser_version="t",
    )


def _vision(text: str, page: int):
    marker = _text_atom("[image]", page)
    return pdf_image_vision._emit_atom(
        marker=marker, pdf_name="g.pdf", region_ref=f"page{page}/image7", page_index=page,
        text=text, image_kind="instructions", fact_kind="image_instruction_step", confidence=0.6,
    )


def test_vision_copy_of_a_text_step_is_split_off() -> None:
    text = [_text_atom(f"5 {STEP}", 4), _text_atom("6 Raise the mast back to vertical.", 4)]
    copy = _vision(f"Step 3: {STEP}", 4)
    new = _vision("Step 9: Photograph the finished install from the north side of the pole.", 4)
    other_page = _vision(f"Step 3: {STEP}", 7)
    kept, copies = pdf_image_vision.split_text_layer_copies([copy, new, other_page], text)
    assert copies == [copy]
    assert copy.value["covered_by_text_atom"] == text[0].id
    assert new in kept and other_page in kept


def test_compile_records_vision_copies_as_suppressions(tmp_path: Path, monkeypatch) -> None:
    from app.core.compiler import compile_project

    monkeypatch.setenv("SOWSMITH_DISABLE_LLM", "1")
    deal = tmp_path / "deal"
    deal.mkdir()
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 72), f"5 {STEP}", fontsize=10, fontname="helv")
    doc.save(deal / "Install Guide.pdf")

    def fake(atoms):
        pdf = [a for a in atoms if str(a.source_refs[0].filename).endswith(".pdf")]
        marker = pdf[0]
        v = pdf_image_vision._emit_atom(
            marker=marker, pdf_name="Install Guide.pdf", region_ref="page0/image3", page_index=0,
            text=f"Step 3: {STEP}", image_kind="instructions",
            fact_kind="image_instruction_step", confidence=0.6,
        )
        return [v]

    monkeypatch.setattr(pdf_image_vision, "enabled", lambda: True)
    monkeypatch.setattr(pdf_image_vision, "process_image_markers", fake)
    r = compile_project(deal, project_id="p", allow_errors=True, use_cache=False)
    assert not any(a.raw_text.startswith("Step 3:") for a in r.atoms), [a.raw_text for a in r.atoms]
    sup = [s for s in r.suppressed_atoms if (s.raw_text or "").startswith("Step 3:")]
    assert sup and "suppressed:vision_copy_of_text_layer" in sup[0].review_flags
