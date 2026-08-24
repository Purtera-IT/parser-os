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


# ── xlsx: a tab NAME must not delete the sheet's contents ────────────────


_SCOPE_ROWS = [
    ["Site", "Building", "Device", "Qty", "Notes"],
    ["ATL-01", "Building C", "IP Camera", 24, "escort required"],
    ["ATL-02", "Building C", "IP Camera", 18, "after hours"],
    ["ATL-03", "Building D", "Access Point", 12, ""],
]
_LOOKUP_ROWS = [
    ["PS-L1-ENG-LABOR", "L1", "Hourly"],
    ["PS-L2-ENG-LABOR", "L2", "Hourly"],
    ["PS-L3-ENG-LABOR", "L3", "Hourly"],
    ["PS-L4-ENG-LABOR", "L4", "Fixed"],
    ["PS-L0-TECH-LABOR", "L0", "Hourly"],
]


@pytest.mark.parametrize(
    "tab", ["Lookup", "Helper", "DO NOT EDIT", "Dropdown", "Validation"]
)
def test_a_real_scope_table_survives_a_backing_data_tab_name(tab: str) -> None:
    """``suppress=True`` drops every atom on the sheet, and rule 1 decided it
    on the tab name -- the comment said "regardless of content".

    A tab name is as forgeable as a filename and gets renamed far more often:
    a template's "Lookup" tab that now holds the real BOM lost the whole
    table silently. The guard is the one this file already applies to the
    FINANCIAL name rule, whose comment says "the data-header guard already
    protects any genuine scope table that happens to live under such a name".
    """
    from app.parsers.sheet_classifier import classify_sheet

    assert not classify_sheet(tab, _SCOPE_ROWS).suppress, (
        f"a Site/Device/Qty table was dropped because the tab is called {tab!r}"
    )


@pytest.mark.parametrize(
    "tab", ["Lookup", "Helper", "DO NOT EDIT", "SELL RATES", "PriceList"]
)
def test_genuine_backing_data_is_still_suppressed(tab: str) -> None:
    """The other half: the rule exists for a reason and must keep working.

    ``_looks_like_data_header`` wants two or more tokens from a scope/BOM
    vocabulary (site, device, qty, part number, ...). A rate card's
    ``Code | Skill | Billing`` carries none of them, so rate cards and price
    books keep collapsing exactly as before -- which is the correct behaviour
    for boilerplate that appears on most deals.
    """
    from app.parsers.sheet_classifier import classify_sheet

    assert classify_sheet(tab, _LOOKUP_ROWS).suppress


# ── pptx: a slide title comes from a title placeholder, not a substring ──


def test_a_subtitle_is_not_a_title(tmp_path: Path) -> None:
    """``"title" in str(placeholder_format.type).lower()`` matches SUBTITLE.

    The scan broke on the first near-miss, so a subtitle sitting earlier in
    shape order became the slide title -- and a slide title becomes the
    section heading for every atom on that slide. Found on a real deck: slide
    1 titled itself "April 12, 2025" while its CENTER_TITLE, later in the
    shape tree, held the actual title.
    """
    pptx = pytest.importorskip("pptx")
    from app.parsers.pptx_parser import PptxParser

    presentation = pptx.Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[0])
    slide.shapes.title.text = "Structured Cabling Rollout"
    slide.placeholders[1].text = "April 12, 2025"

    # Reproduce the real deck's ordering: subtitle first in the shape tree.
    tree = slide.shapes._spTree
    element = slide.shapes.title._element
    tree.remove(element)
    tree.append(element)

    target = tmp_path / "deck.pptx"
    presentation.save(target)

    output = PptxParser().parse_artifact(project_id="p", artifact_id="a", path=target)
    atoms = output.atoms if hasattr(output, "atoms") else output
    titles = set()
    for atom in atoms:
        locator = (atom.source_refs[0].locator if atom.source_refs else {}) or {}
        found = locator.get("slide_title") or locator.get("section")
        if found:
            titles.add(found)
    assert "April 12, 2025" not in titles, "a subtitle was used as the slide title"
    assert "Structured Cabling Rollout" in titles


# ── html: a CSS class name must not silence the document ────────────────


_SOW_HTML = """<html><body>
<h1>Statement of Work</h1>
<h2>Scope</h2>
<p>Contractor shall install forty IP cameras at the Atlanta facility.</p>
<p>All drops terminate in IDF 3 and are certified to TIA-568.</p>
<h2>Exclusions</h2>
<ul><li>Mid-turn jumpers are excluded from this bill of materials.</li>
<li>Conduit above ten feet is by others.</li></ul>
<table><tr><th>Site</th><th>Qty</th></tr><tr><td>ATL-01</td><td>24</td></tr></table>
{extra}
</body></html>"""

# A stock Bulma notice. "message-body" is a standard component class in Bulma
# and several CMS themes -- nothing to do with chat.
_BULMA_NOTICE = (
    '<article class="message is-warning">'
    '<div class="message-body">Escort is required at all times.</div></article>'
)


def _html_atoms(target: Path, extra: str) -> int:
    from app.parsers.universal_parsers import HtmlParser

    target.write_text(_SOW_HTML.format(extra=extra), encoding="utf-8")
    output = HtmlParser().parse_artifact_full(
        project_id="p", artifact_id="a", path=target
    )
    return len(output.atoms)


def test_one_css_class_cannot_delete_the_whole_page(tmp_path: Path) -> None:
    """The chat detectors matched CSS class names and then RETURNED.

    So when a detector was wrong, the generic heading / paragraph / table walk
    never ran and the page produced nothing. Measured: an ordinary SOW page
    yields 11 atoms; adding a single ``<div class="message-body">`` -- one
    Bulma notice -- produced ZERO. Silent, total loss of the document.
    """
    plain = _html_atoms(tmp_path / "plain.html", "")
    with_notice = _html_atoms(tmp_path / "notice.html", _BULMA_NOTICE)
    assert plain == 11
    assert with_notice == plain, (
        f"a Bulma notice cost {plain - with_notice} atoms"
    )


def test_a_real_chat_export_still_takes_the_chat_path(tmp_path: Path) -> None:
    """The chat path has to keep winning when it actually has messages.

    It now has to PRODUCE something to win: the detector firing is no longer
    enough, because the detector reads class names and the extractor reads
    ``data-tid`` -- they could disagree, and when they did the document was
    lost rather than merely mis-scanned.
    """
    from app.parsers.universal_parsers import HtmlParser

    messages = "".join(
        '<div><div data-tid="messageHeader">Cliff Creech</div>'
        f'<div data-tid="message-body">we need forty sites by Q3, point {i}.</div></div>'
        for i in range(5)
    )
    target = tmp_path / "teams_export.html"
    target.write_text(f"<html><body><h1>Chat</h1>{messages}</body></html>", encoding="utf-8")
    output = HtmlParser().parse_artifact_full(project_id="p", artifact_id="a", path=target)
    kinds = {(atom.value or {}).get("kind") for atom in output.atoms}
    assert "teams_message" in kinds
    assert any("Cliff Creech" in atom.raw_text for atom in output.atoms)


# ── json: stepping aside is not the same as leaving nobody holding it ────


def test_a_json_deferral_does_not_drop_the_file(tmp_path: Path) -> None:
    """JsonParser returned 0.00 to hand the file to TranscriptParser.

    TranscriptParser's .json branch needs ``json.loads`` to SUCCEED, and the
    router hands it a 4000-character sample, so a large .json raises there and
    scores 0.00 too. Both parsers stood down and the artifact routed to NONE.

    Measured on the local corpus: 2062 of 2500 artifacts landed on NONE -- all
    valid .json that parse fine in full. They carry the key ``"segments"``,
    which in those files means document segments, and which is an ordinary
    business word in any case (network, customer, cable).
    """
    from app.parsers.registry import choose_parser

    payload = '{\n  "segments": [\n' + ",\n".join(
        f'    {{"id": {i}, "label": "cable segment {i}", "sites": ["ATL-{i:02d}"]}}'
        for i in range(400)
    ) + "\n  ]\n}\n"
    target = tmp_path / "artifact_cache.json"
    target.write_text(payload, encoding="utf-8")
    assert len(payload) > 4000, "the point of the case is that the sample truncates"

    parser, match, _all = choose_parser(target)
    assert parser is not None, "valid JSON routed to NONE"
    assert match.confidence >= 0.5


def test_a_real_transcript_json_still_reaches_the_transcript_parser(
    tmp_path: Path,
) -> None:
    """The deferral must still defer. 0.55 sits below TranscriptParser's 0.8."""
    from app.parsers.registry import choose_parser

    payload = (
        '{"utterances": ['
        + ",".join(
            f'{{"speaker": "Cliff", "text": "point {i}", "start": {i}}}'
            for i in range(5)
        )
        + "]}"
    )
    target = tmp_path / "meeting.json"
    target.write_text(payload, encoding="utf-8")
    parser, _match, _all = choose_parser(target)
    assert type(parser).__name__ == "TranscriptParser"
