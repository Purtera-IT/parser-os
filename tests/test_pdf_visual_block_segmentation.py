"""PDF text is chunked on visual blocks, read column by column; SOW paragraphs
are split per clause in both the PDF and the DOCX path.

Deal 010003, CDW "QUOTE CONFIRMATION": ``page.get_text("text")`` has no blank
line between blocks, and the prose splitter only ends a paragraph at a blank
line or a heading. So the greeting ("MATTHEW BRUNTON,"), the eProcurement
log-in boilerplate, the "click below" call to action and the red "Convert
Quote to Order" button were ONE atom. The same page read its shipping /
remit-to / sales-contact boxes across rows into one atom, and an OxBlue
install guide interleaved its two step columns.
"""

from __future__ import annotations

from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

BOILERPLATE = [
    "Thank you for considering CDW for your technology needs. The details of your quote are below.",
    "If you are an eProcurement or single sign on customer, please log into your system to access the",
    "CDW site. You can search for your quote to retrieve and transfer back into your system for processing.",
]
CTA = "For all other customers, click below to convert your quote to an order."
BUTTON = "Convert Quote to Order"


def _cdw_quote(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((140, 40), "Hardware   Software   Services   IT Solutions   Brands   Research Hub", fontsize=9)
    page.insert_text((36, 90), "QUOTE CONFIRMATION", fontsize=18, fontname="hebo")
    y = 125.0
    page.insert_text((36, y), "MATTHEW BRUNTON,", fontsize=10)
    y += 24
    top = y
    for ln in BOILERPLATE:
        page.insert_text((36, y), ln, fontsize=10)
        y += 13
    # the middle sentence is a link
    page.insert_link({"kind": fitz.LINK_URI, "from": fitz.Rect(36, top + 3, 576, top + 26),
                      "uri": "https://example.com/login"})
    y += 11
    page.insert_text((36, y), CTA, fontsize=10)
    y += 14
    page.draw_rect(fitz.Rect(36, y, 196, y + 24), color=(0.8, 0, 0), fill=(0.8, 0, 0))
    page.insert_text((52, y + 16), BUTTON, fontsize=11, fontname="hebo", color=(1, 1, 1))
    y += 50
    xs = [36, 140, 240, 380, 480]
    for x, h in zip(xs, ["QUOTE #", "QUOTE DATE", "QUOTE REFERENCE", "CUSTOMER #", "GRAND TOTAL"]):
        page.insert_text((x, y), h, fontsize=9, fontname="hebo")
    y += 14
    for x, v in zip(xs, ["PSNV676", "1/14/2026", "SAMSUNG", "15018865", "$4,691.64"]):
        page.insert_text((x, y), v, fontsize=10)
    doc.save(str(path))
    doc.close()


def _blocks(pdf: Path) -> list[str]:
    from app.parsers.orbitbrief_pdf import build_structured_document

    out: list[str] = []

    def walk(secs):
        for s in secs:
            for b in s.get("blocks") or []:
                if b.get("kind") == "bullet_list":
                    out.extend(i.get("text", "") for i in b.get("items") or [])
                elif b.get("text"):
                    out.append(b["text"])
            walk(s.get("subsections") or [])

    for page in build_structured_document(pdf)["pages"]:
        walk(page.get("sections") or [])
    return out


def _atom_texts(pdf: Path) -> list[str]:
    from app.parsers.orbitbrief_pdf import OrbitBriefPdfParser

    out = OrbitBriefPdfParser().parse(pdf)
    return [a.raw_text for a in getattr(out, "atoms", out)]


def _holding(blocks: list[str], needle: str) -> list[str]:
    return [b for b in blocks if needle in b]


def test_greeting_boilerplate_cta_and_button_are_separate_blocks(tmp_path):
    pdf = tmp_path / "quote.pdf"
    _cdw_quote(pdf)
    blocks = _blocks(pdf)
    greeting = _holding(blocks, "MATTHEW BRUNTON")
    assert greeting and all("Thank you" not in b for b in greeting), blocks
    boiler = _holding(blocks, "Thank you for considering")
    assert boiler, blocks
    assert all("click below" not in b and BUTTON not in b for b in boiler), blocks
    cta = _holding(blocks, "click below")
    assert cta and all(BUTTON not in b for b in cta), blocks
    # the quote's own facts never share a block with the portal boilerplate
    facts = _holding(blocks, "PSNV676")
    assert facts and all("Thank you" not in b and "click below" not in b for b in facts), blocks
    assert any("$4,691.64" in b for b in facts), "grand total lost from the quote header"


def test_a_wrapped_paragraph_stays_one_block(tmp_path):
    pdf = tmp_path / "quote.pdf"
    _cdw_quote(pdf)
    blocks = _blocks(pdf)
    # the three wrapped lines (and the sentence wrapped across lines 2-3) are
    # one paragraph, not three
    assert any("access the CDW site." in b and "for processing." in b for b in blocks), blocks


def test_cdw_atoms_are_small(tmp_path):
    pdf = tmp_path / "quote.pdf"
    _cdw_quote(pdf)
    atoms = _atom_texts(pdf)
    assert atoms
    assert not any("MATTHEW BRUNTON" in a and "Thank you" in a for a in atoms), atoms
    assert not any("Thank you" in a and "click below" in a for a in atoms), atoms
    assert max(len(a) for a in atoms) <= 400, atoms


def test_never_merges_across_a_heading(tmp_path):
    pdf = tmp_path / "heading.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    y = 60.0
    for ln in ["The installer will mount all access points on the ceiling grid.",
               "Cable will be dressed and labelled at both ends."]:
        page.insert_text((36, y), ln, fontsize=10)
        y += 13
    page.insert_text((36, y + 2), "Customer Responsibilities", fontsize=14, fontname="hebo")
    y += 18
    for ln in ["The customer will provide escorted access to every closet.",
               "Lifts and ladders are supplied by the customer."]:
        page.insert_text((36, y), ln, fontsize=10)
        y += 13
    doc.save(str(pdf))
    doc.close()
    blocks = _blocks(pdf)
    assert not any("labelled at both ends" in b and "escorted access" in b for b in blocks), blocks
    assert not any("Customer Responsibilities" in b and "ceiling grid" in b for b in blocks), blocks


def _three_boxes(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((36, 60), "QUOTE DETAILS", fontsize=14, fontname="hebo")
    boxes = {
        36: ["Shipping Method", "UPS Ground (2-3 Day)", "Ship to: Acme Corp", "100 Main Street",
             "Springfield, IL 62701"],
        230: ["Please remit payments to:", "CDW Government", "75 Remittance Drive", "Suite 1515",
              "Chicago, IL 60675-1515"],
        424: ["Sales Contact Info", "Sarah Halpern", "(866) 339-4118", "sarahal@cdw.com"],
    }
    for x, lines in boxes.items():
        y = 100.0
        for i, ln in enumerate(lines):
            page.insert_text((x, y), ln, fontsize=10, fontname="hebo" if i == 0 else "helv")
            y += 13
    doc.save(str(path))
    doc.close()


def test_side_by_side_boxes_are_read_box_by_box(tmp_path):
    pdf = tmp_path / "boxes.pdf"
    _three_boxes(pdf)
    blocks = _blocks(pdf) + _atom_texts(pdf)
    markers = {"ship": "UPS Ground", "remit": "Remittance Drive", "contact": "Sarah Halpern"}
    for text in blocks:
        hits = [k for k, m in markers.items() if m in text]
        assert len(hits) <= 1, f"boxes fused: {text!r}"
    for m in markers.values():
        assert any(m in b for b in blocks), f"{m} lost"
    # each box stays whole
    assert any("UPS Ground" in b and "Springfield, IL 62701" in b for b in blocks), blocks
    assert any("Remittance Drive" in b and "Chicago, IL 60675-1515" in b for b in blocks), blocks
    assert any("Sarah Halpern" in b and "sarahal@cdw.com" in b for b in blocks), blocks


LEFT_STEPS = ["8. Mount the camera bracket to the", "pole using the supplied stainless", "steel bands.",
              "9. Route the power cable down the", "pole and secure it every 12 inches.",
              "10. Connect the antenna to the port", "marked LTE on the enclosure."]
RIGHT_STEPS = ["11. Power on the unit and wait for", "the status light to turn solid green.",
               "12. Verify the live view in the", "OxBlue portal before leaving site.",
               "13. Photograph the completed install", "from two angles and upload them."]


def test_two_column_steps_never_interleave(tmp_path):
    pdf = tmp_path / "steps.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((36, 50), "INSTALLATION GUIDE", fontsize=16, fontname="hebo")
    for x, lines in ((36, LEFT_STEPS), (316, RIGHT_STEPS)):
        y = 90.0
        for ln in lines:
            page.insert_text((x, y), ln, fontsize=10)
            y += 13
    doc.save(str(pdf))
    doc.close()
    atoms = _atom_texts(pdf)
    expected = [
        "Mount the camera bracket to the pole using the supplied stainless steel bands.",
        "Route the power cable down the pole and secure it every 12 inches.",
        "Connect the antenna to the port marked LTE on the enclosure.",
        "Power on the unit and wait for the status light to turn solid green.",
        "Verify the live view in the OxBlue portal before leaving site.",
        "Photograph the completed install from two angles and upload them.",
    ]
    for step in expected:
        assert any(step in a for a in atoms), (step, atoms)
    # one step per atom: no atom carries two steps
    for a in atoms:
        assert sum(1 for s in expected if s[:20] in a) <= 1, a


# ── clause split (shared by PDF and DOCX) ──────────────────────────────────

ACCEPTANCE = (
    "Acceptance of the Services occurs when the Provider delivers the completion report. "
    "Customer has five (5) business days to review the report and identify any deficiency in writing. "
    "If Customer does not respond within that period the Services are deemed accepted. "
    "Provider will correct any reported deficiency at no additional charge."
)
LEAD_IN_PARA = (
    "Provide onsite installation services for the new display walls in each conference room. "
    "Provider will perform the following:"
)


def test_split_clauses_units():
    from app.parsers.clause_split import split_clauses

    parts = split_clauses(ACCEPTANCE)
    assert len(parts) == 4
    assert parts[0].startswith("Acceptance of the Services")
    assert parts[-1].startswith("Provider will correct")
    # a trailing list lead-in comes off even on a short paragraph
    assert split_clauses(LEAD_IN_PARA) == [
        "Provide onsite installation services for the new display walls in each conference room.",
        "Provider will perform the following:",
    ]
    # a sentence leaning on the one before stays with it; a short paragraph
    # without a lead-in stays whole; a decimal never splits
    assert split_clauses("Rates are $93,583.25 total. It is fixed for six months.") == []
    rate = split_clauses(
        "Business Hours are as follows:\nStandard: Mon-Fri 8am-5pm at the quoted rate\n"
        "After hours: billed at 150% of the quoted rate\nWeekends and holidays: billed at 200%"
    )
    assert rate[0] == "Business Hours are as follows:" and len(rate) == 4


def test_pdf_trailing_lead_in_is_its_own_atom(tmp_path):
    pdf = tmp_path / "sow.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((36, 50), "SCOPE OF WORK", fontsize=14, fontname="hebo")
    y = 80.0
    for ln in ["Provide onsite installation services for the new display walls in each",
               "conference room. Provider will perform the following:"]:
        page.insert_text((36, y), ln, fontsize=10)
        y += 13
    for ln in ["• Mount the displays on the supplied wall brackets", "• Terminate and test each HDMI run",
               "• Remove packaging from site"]:
        page.insert_text((48, y), ln, fontsize=10)
        y += 13
    doc.save(str(pdf))
    doc.close()
    atoms = _atom_texts(pdf)
    assert not any("display walls" in a and "following:" in a for a in atoms), atoms
    assert any(a.startswith("Provide onsite installation") for a in atoms), atoms


def _docx(path: Path, paragraphs: list[str]) -> None:
    docx = pytest.importorskip("docx")
    d = docx.Document()
    d.add_heading("Statement of Work", level=1)
    for p in paragraphs:
        d.add_paragraph(p)
    d.save(str(path))


def _docx_atoms(path: Path):
    from app.parsers.docx_parser import DocxParser

    out = DocxParser().parse_artifact(project_id="p", artifact_id="a", path=path)
    return list(getattr(out, "atoms", out))


def test_docx_paragraph_splits_like_the_signed_pdf(tmp_path):
    from app.parsers.clause_split import split_clauses

    p = tmp_path / "sow_draft.docx"
    _docx(p, [ACCEPTANCE])
    texts = [a.raw_text for a in _docx_atoms(p)]
    assert texts == split_clauses(ACCEPTANCE), texts


def test_docx_one_atom_per_sentence_not_per_type(tmp_path):
    p = tmp_path / "sow.docx"
    sent = "Install the cabling at Site 4; ceiling access panels are out of scope."
    _docx(p, [sent, "Unrelated filler paragraph describing the project overview in detail."])
    atoms = [a for a in _docx_atoms(p) if a.raw_text == sent]
    assert len(atoms) == 1, [(a.atom_type, a.raw_text) for a in atoms]
    assert atoms[0].atom_type.value == "exclusion"
    assert "scope_item" in atoms[0].value.get("alt_atom_types", [])


def test_docx_same_text_from_two_paths_emits_once():
    from types import SimpleNamespace

    from app.core.schemas import AtomType
    from app.parsers.docx_parser import _dedupe_repeated_text

    a = SimpleNamespace(normalized_text="site 4 install", atom_type=AtomType.scope_item, value={})
    b = SimpleNamespace(normalized_text="site 4 install", atom_type=AtomType.scope_item, value={})
    row = SimpleNamespace(normalized_text="site 4 install", atom_type=AtomType.raw_table_row, value={})
    assert _dedupe_repeated_text([a, b, row]) == [a, row]


def test_pdf_rate_rows_under_a_lead_in_are_one_atom_each(tmp_path):
    pdf = tmp_path / "rates.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((36, 50), "PRICING", fontsize=14, fontname="hebo")
    y = 80.0
    for ln in ["All Services are performed during normal Business Hours, which are as follows:",
               "Standard: Monday-Friday 8:00am-5:00pm at the quoted rate",
               "After Hours: billed at 150% of the quoted rate",
               "Weekends: billed at 150% of the quoted rate",
               "Holidays: billed at 200% of the quoted rate"]:
        page.insert_text((36, y), ln, fontsize=10)
        y += 13
    doc.save(str(pdf))
    doc.close()
    atoms = _atom_texts(pdf)
    assert not any("Business Hours" in a and "150%" in a for a in atoms), atoms
    assert sum(1 for a in atoms if "% of the quoted rate" in a) == 3, atoms


def test_wrapped_prose_after_a_colon_is_not_split_per_line():
    from app.parsers.clause_split import split_clauses

    text = (
        "The Contractor shall comply with the following:\n"
        "All work is performed to the latest TIA-568 standard and the Owner's\n"
        "Facilities Guide, including labelling, testing, and documentation of\n"
        "Every drop."
    )
    parts = split_clauses(text)
    assert not any(p.startswith("Facilities Guide") for p in parts), parts
