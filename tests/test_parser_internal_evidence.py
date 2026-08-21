"""Inside the parsers, the same rule as at the router: structure over strings.

The routing audit established the evidence order -- container, then content,
then extension, then filename as a prior that may raise a claim but never
create one. These are the places INSIDE the two big content parsers where the
same rule was being broken:

  * ``DocxParser._heading_level`` matched the English substring "heading", so
    section detection depended on the LOCALE of the Word install that wrote
    the file, and on whether anyone had used a corporate template.
  * ``detect_hybrid_summary_transcript`` accepted a filename as sufficient
    evidence that a PDF was a transcript, which changed the resulting atom
    TYPE.

Both are pinned here because both are invisible failures: nothing raises, the
document simply comes out flat or comes out mistyped.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

from app.parsers.docx_parser import DocxParser

_SECTIONS = [
    ("Scope of Work", [
        "Contractor shall install forty IP cameras at the Atlanta facility.",
        "All drops terminate in IDF 3 and are certified to TIA-568.",
    ]),
    ("Site Access", [
        "Escort is required at all times inside the yard.",
        "Work occurs after hours on weekends only.",
    ]),
    ("Exclusions", [
        "Mid-turn jumpers are excluded from this bill of materials.",
        "Conduit above ten feet is by others.",
    ]),
]


def _build_docx(target: Path, heading_style_name: str | None) -> Path:
    """``None`` uses built-in 'Heading 1' -- the shape the whole corpus has."""
    doc = Document()
    if heading_style_name is None:
        use = "Heading 1"
    else:
        style = doc.styles.add_style(heading_style_name, WD_STYLE_TYPE.PARAGRAPH)
        style.base_style = doc.styles["Heading 1"]
        # Word writes outlineLvl into the style itself; set it explicitly so
        # this is the real shape and not a python-docx default.
        ppr = style.element.get_or_add_pPr()
        lvl = OxmlElement("w:outlineLvl")
        lvl.set(qn("w:val"), "0")
        ppr.append(lvl)
        use = heading_style_name
    for title, paragraphs in _SECTIONS:
        doc.add_paragraph(title, style=use)
        for text in paragraphs:
            doc.add_paragraph(text)
    doc.save(target)
    return target


def _section_paths(path: Path) -> set[str]:
    output = DocxParser().parse_artifact(project_id="p", artifact_id="a", path=path)
    atoms = output.atoms if hasattr(output, "atoms") else output
    found: set[str] = set()
    for atom in atoms:
        locator = (atom.source_refs[0].locator if atom.source_refs else {}) or {}
        section = locator.get("section_path") or locator.get("section")
        if section:
            found.add("/".join(section) if isinstance(section, list) else str(section))
    return found


@pytest.mark.parametrize(
    "style_name, label",
    [
        (None, "built-in 'Heading 1'"),
        ("Uberschrift 1", "German Word"),
        ("Titre 1", "French Word"),
        ("Encabezado 1", "Spanish Word"),
        ("PurTera Section Head", "corporate template based on Heading 1"),
    ],
)
def test_headings_are_found_by_structure_not_by_an_english_string(
    tmp_path: Path, style_name: str | None, label: str
) -> None:
    """The same document, three sections deep, under five heading styles.

    ``_heading_level`` answered from ``style.name.startswith("heading")``.
    Every style above carries ``w:outlineLvl`` -- Word writes it into heading
    styles and a style based on Heading N inherits it -- but only the first
    contains the substring. Measured before the fix: 3 sections for the
    built-in style, ZERO for the other four, with every atom losing its
    section_path.

    This file had already learned the lesson for lists:
    ``_paragraph_is_list_item`` reads ``w:numPr`` off the XML because "Word
    frequently leaves list paragraphs on the Normal style while carrying real
    numbering". Headings never got the same treatment.
    """
    target = _build_docx(tmp_path / "sow.docx", style_name)
    paths = _section_paths(target)
    for expected, _body in _SECTIONS:
        assert any(expected in p for p in paths), (
            f"{label}: section {expected!r} lost; found {sorted(paths)}"
        )


def test_outline_fallback_never_overrules_the_style_name(tmp_path: Path) -> None:
    """The fallback is additive, and that is what makes it safe to ship.

    It runs only after the style name has declined to answer, so it can add a
    heading the name could not see and can never change one the name already
    found. Across all 13 real .docx available the two signals already agree
    exactly, and parser output is byte-identical before and after.
    """
    built_in = _section_paths(_build_docx(tmp_path / "a.docx", None))
    localised = _section_paths(_build_docx(tmp_path / "b.docx", "Uberschrift 1"))
    assert built_in == localised


# ── the PDF half: a filename must not decide the atom TYPE ───────────────


def _pdf(target: Path, lines: list[str]) -> Path:
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    page = doc.new_page()
    y = 72
    for line in lines:
        page.insert_text((72, y), line, fontsize=11)
        y += 18
    doc.save(str(target))
    doc.close()
    return target


_SOW_LINES = [
    "Statement of Work",
    "",
    "Contractor shall install forty IP cameras at the Atlanta facility.",
    "All drops terminate in IDF 3 and are certified to TIA-568.",
    "Escort is required at all times inside the yard.",
    "Mid-turn jumpers are excluded from this bill of materials.",
    "Conduit above ten feet is by others.",
]

_TRANSCRIPT_LINES = [
    "Kickoff Call",
    "",
    "Cliff Creech [00:12] we need forty sites by Q3.",
    "Dana Whitfield [00:31] escort is required at the Atlanta dock.",
    "Cliff Creech [01:04] mid-turn jumpers are excluded.",
    "Dana Whitfield [01:22] confirm the badge list by Friday.",
]


def _plan_kind(path: Path) -> str | None:
    fitz = pytest.importorskip("fitz")
    from app.core.hybrid_summary_transcript import detect_hybrid_summary_transcript

    with fitz.open(str(path)) as doc:
        pages = [(page.get_text() or "") for page in doc]
    plan = detect_hybrid_summary_transcript(filename=path.name, title=None, page_texts=pages)
    return plan.kind if plan else None


def test_a_scope_pdf_named_transcript_is_not_re_atomised_as_conversation(
    tmp_path: Path,
) -> None:
    """``title_or_filename_transcript`` alone satisfied the gate.

    So a PDF whose NAME contained "transcript", with no speaker stamps and no
    Full Transcript marker, fell through to kind="transcript_only" and had its
    prose rebuilt as conversation turns. On one document rendered twice from
    identical text, the filename changed the atom TYPE:

        scope_of_work.pdf              -> no plan          exclusion
        kickoff_meeting_transcript.pdf -> transcript_only  action_item

    exclusion is scope-governing; action_item is a follow-up note.
    """
    assert _plan_kind(_pdf(tmp_path / "scope_of_work.pdf", _SOW_LINES)) is None
    assert _plan_kind(_pdf(tmp_path / "kickoff_meeting_transcript.pdf", _SOW_LINES)) is None


def test_a_real_transcript_pdf_is_still_detected_without_a_helpful_name(
    tmp_path: Path,
) -> None:
    """The other half: removing the shortcut must not cost the capability.

    Speaker-timestamp density is content and qualifies on its own, so a real
    transcript is found under a neutral name.
    """
    assert _plan_kind(_pdf(tmp_path / "attachment_c.pdf", _TRANSCRIPT_LINES)) == "transcript_only"
    assert _plan_kind(_pdf(tmp_path / "kickoff_transcript.pdf", _TRANSCRIPT_LINES)) == "transcript_only"
