"""Coverage for the whole-document text fallback (Gap D).

A large share of DOCX/text atoms used to come back ``unsupported`` —
their locator lacked a precise paragraph/table index or line range, so
the verifier couldn't resolve them even though the text was sitting
right there in the file. The fallback confirms the atom's text against
the whole document body, upgrading those to a real (if lower-precision)
``verified`` receipt — while still refusing to verify text that genuinely
isn't present (no false positives).
"""

from __future__ import annotations

from pathlib import Path

from docx import Document

from app.core.source_replay import replay_source_ref
from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    EvidenceAtom,
    ReviewStatus,
    SourceRef,
)


def _atom(text: str, *, artifact_id: str, artifact_type: ArtifactType, filename: str, locator: dict) -> EvidenceAtom:
    return EvidenceAtom(
        id=f"atm_{abs(hash((text, filename))) % (10**12):012x}",
        project_id="p",
        artifact_id=artifact_id,
        atom_type=AtomType.requirement,
        raw_text=text,
        normalized_text=text.lower(),
        value={},
        entity_keys=[],
        source_refs=[
            SourceRef(
                id="src_1",
                artifact_id=artifact_id,
                artifact_type=artifact_type,
                filename=filename,
                locator=locator,
                extraction_method="test",
                parser_version="t",
            )
        ],
        receipts=[],
        authority_class=AuthorityClass.machine_extractor,
        confidence=0.8,
        confidence_raw=0.8,
        calibrated_confidence=0.8,
        review_status=ReviewStatus.needs_review,
        review_flags=[],
        parser_version="t",
    )


def _write_docx(path: Path, paragraphs: list[str]) -> None:
    doc = Document()
    for p in paragraphs:
        doc.add_paragraph(p)
    doc.save(str(path))


# ───────────────────────── DOCX ─────────────────────────


def test_docx_unmodeled_locator_verifies_via_fallback(tmp_path: Path) -> None:
    """An atom whose locator has no paragraph_index/table_index still
    verifies when its text is present in the document body."""
    docx_path = tmp_path / "scope.docx"
    _write_docx(docx_path, [
        "Project kickoff overview.",
        "Replace approximately 110 displays across the resort property.",
        "Manual TV configuration, roughly 15 minutes per unit.",
    ])
    atom = _atom(
        "Replace approximately 110 displays across the resort property.",
        artifact_id="a1",
        artifact_type=ArtifactType.docx,
        filename="scope.docx",
        locator={"section": "scope", "run_offset": 4},  # no paragraph_index/table_index
    )
    receipt = replay_source_ref(atom, atom.source_refs[0], {"a1": docx_path})
    assert receipt.replay_status == "verified"
    assert "fallback" in receipt.reason.lower()
    assert receipt.extracted_snippet and "110 displays" in receipt.extracted_snippet


def test_docx_unmodeled_locator_absent_text_stays_unsupported(tmp_path: Path) -> None:
    """No false positives: text that isn't in the document stays unsupported."""
    docx_path = tmp_path / "scope.docx"
    _write_docx(docx_path, ["Project kickoff overview.", "Manual TV configuration."])
    atom = _atom(
        "Install fiber backbone across twelve buildings with redundant uplinks.",
        artifact_id="a1",
        artifact_type=ArtifactType.docx,
        filename="scope.docx",
        locator={"section": "scope"},
    )
    receipt = replay_source_ref(atom, atom.source_refs[0], {"a1": docx_path})
    assert receipt.replay_status == "unsupported"


def test_docx_precise_paragraph_locator_still_used(tmp_path: Path) -> None:
    """The fallback must not shadow the precise verifier: a valid
    paragraph_index still verifies through the normal path."""
    docx_path = tmp_path / "scope.docx"
    _write_docx(docx_path, ["First line.", "Replace 110 displays at the resort."])
    atom = _atom(
        "Replace 110 displays at the resort.",
        artifact_id="a1",
        artifact_type=ArtifactType.docx,
        filename="scope.docx",
        locator={"paragraph_index": 1},
    )
    receipt = replay_source_ref(atom, atom.source_refs[0], {"a1": docx_path})
    assert receipt.replay_status == "verified"
    assert "fallback" not in receipt.reason.lower()


# ───────────────────────── text files ─────────────────────────


def test_text_missing_line_range_verifies_via_fallback(tmp_path: Path) -> None:
    txt_path = tmp_path / "notes.txt"
    txt_path.write_text(
        "Project Notes\nReplace approximately 110 displays at the resort.\n",
        encoding="utf-8",
    )
    atom = _atom(
        "Replace approximately 110 displays at the resort.",
        artifact_id="a1",
        artifact_type=ArtifactType.txt,
        filename="notes.txt",
        locator={},  # no line_start/line_end
    )
    receipt = replay_source_ref(atom, atom.source_refs[0], {"a1": txt_path})
    assert receipt.replay_status == "verified"
    assert "fallback" in receipt.reason.lower()


def test_text_missing_line_range_absent_text_stays_unsupported(tmp_path: Path) -> None:
    txt_path = tmp_path / "notes.txt"
    txt_path.write_text("Project Notes\nManual TV configuration.\n", encoding="utf-8")
    atom = _atom(
        "Install fiber backbone across twelve buildings.",
        artifact_id="a1",
        artifact_type=ArtifactType.txt,
        filename="notes.txt",
        locator={},
    )
    receipt = replay_source_ref(atom, atom.source_refs[0], {"a1": txt_path})
    assert receipt.replay_status == "unsupported"
