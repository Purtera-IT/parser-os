"""Round-2 fixes from re-parsing signed SOWs and install guides (010087, 010246).

1. A two-column numbered instruction page set each step number apart from its
   text, so the number column read as figures and the page was read as a
   TABLE across both columns ("1 Flip the two breakers... | 2 | Loosen the
   turnbuckles...").
2. One return step transcribed twice from an instruction image ("Step 3:" and
   "Step 5:") was two atoms.
3. The free-standing step number "6" reached the atomizer on its own.
4. Bullets under an "Out of Scope" heading were typed scope (and the shortest
   hidden as chatter); the compile then demoted every exclusion that had no
   negation word of its own back to scope_item.
5. Signature badges (company, "DocuSigned by:", signer, title, date) were
   typed scope.
6. The "Docusign Envelope ID: ..." page stamp became the page title (every
   atom's section_path root) and was glued onto the next line.
7. The substance gate dropped the fee row "Engineer (est 3 hrs per site) ...
   99 ... $9,504" as OCR debris on "est" and "hrs".
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

fitz = pytest.importorskip("fitz")

ENVELOPE = "Docusign Envelope ID: 3F2A9C1E-1B2C-4D5E-9F00-ABCDEF123456"


def _pdf_atoms(pdf: Path):
    from app.parsers.orbitbrief_pdf import OrbitBriefPdfParser

    out = OrbitBriefPdfParser().parse(pdf)
    return list(getattr(out, "atoms", out))


def _type(a) -> str:
    t = a.atom_type
    return t.value if hasattr(t, "value") else str(t)


def _section_path(a) -> list[str]:
    loc = a.source_refs[0].locator if a.source_refs else {}
    return list(loc.get("section_path") or [])


# ── 1 / 3: two-column numbered steps ──────────────────────────────────────

LEFT = [("1", ["Flip the two breakers on the panel to", "the OFF position before starting."]),
        ("3", ["Remove the cover plate and set the", "screws aside for reuse."]),
        ("5", ["Return the unit to the shipping box", "and seal it with the supplied tape."])]
RIGHT = [("2", ["Loosen the turnbuckles on both", "sides of the mast."]),
         ("4", ["Disconnect the antenna lead from", "the port marked LTE."]),
         ("6", ["Attach the prepaid return label to", "the top of the box."])]


def _numbered_two_column(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((36, 50), "REMOVAL INSTRUCTIONS", fontsize=16, fontname="hebo")
    for x, steps in ((36, LEFT), (316, RIGHT)):
        y = 100.0
        for n, lines in steps:
            # the number in a large bold face, set apart from its sentence
            page.insert_text((x, y), n, fontsize=14, fontname="hebo")
            for ln in lines:
                page.insert_text((x + 28, y), ln, fontsize=10)
                y += 13
            y += 20
    doc.save(str(path))
    doc.close()


def test_two_column_numbered_steps_are_read_per_column_one_step_per_atom(tmp_path):
    pdf = tmp_path / "steps.pdf"
    _numbered_two_column(pdf)
    texts = [a.raw_text for a in _pdf_atoms(pdf)]
    steps = {n: " ".join(lines) for n, lines in LEFT + RIGHT}
    for n, sentence in steps.items():
        assert f"{n} {sentence}" in texts, (n, texts)
    for t in texts:
        assert " | " not in t, t
        assert sum(1 for s in steps.values() if s[:20] in t) <= 1, t
    # left column top to bottom, then the right column
    order = [next(i for i, t in enumerate(texts) if steps[n][:20] in t) for n in "135246"]
    assert order == sorted(order), texts


def test_a_lone_step_number_is_never_its_own_atom(tmp_path):
    pdf = tmp_path / "steps.pdf"
    _numbered_two_column(pdf)
    for a in _pdf_atoms(pdf):
        assert not a.raw_text.strip().rstrip(".)").isdigit(), a.raw_text


def test_a_column_of_figures_is_still_a_table():
    """The enumerator glue must not touch a real grid of numbers."""
    from app.parsers.pdf.layout_text import _attach_enumerators, _Seg

    segs = [_Seg(36, 100, 120, 110, "Cat6 drops", 10, False, 0),
            _Seg(300, 100, 310, 110, "6", 10, False, 0),
            _Seg(330, 100, 360, 110, "$450", 10, False, 0)]
    out = _attach_enumerators(segs)
    assert [s.text for s in out] == ["Cat6 drops", "6", "$450"]


# ── 2: one instruction step per page, however many crops carried it ─────

def _step_atom(text: str, region: str):
    from app.core.pdf_image_vision import _emit_atom

    marker = SimpleNamespace(project_id="p", artifact_id="a", parser_version="t", id="m")
    return _emit_atom(marker=marker, pdf_name="guide.pdf", region_ref=region, page_index=4,
                      text=text, image_kind="instructions", fact_kind="image_instruction_step",
                      confidence=0.7)


def test_the_same_step_from_two_crops_is_one_atom_with_the_repeat_recorded():
    from app.core.pdf_image_vision import _drop_repeated_steps

    seen: dict = {}
    first = _drop_repeated_steps(
        [_step_atom("Step 3: Return the unit in the original box", "r1"),
         _step_atom("Step 4: Attach the prepaid label", "r1")], seen, "guide.pdf", 4)
    second = _drop_repeated_steps(
        [_step_atom("Step 5: Return the unit in the original box", "r2"),
         _step_atom("Step 6: Seal the box with tape", "r2")], seen, "guide.pdf", 4)
    texts = [a.raw_text for a in first + second]
    assert texts == ["Step 3: Return the unit in the original box", "Step 4: Attach the prepaid label",
                     "Step 6: Seal the box with tape"]
    assert first[0].value["duplicate_step_numbers"] == ["5"]
    assert first[0].value["duplicate_regions"] == ["r2"]
    # a different page is a different step
    other = _drop_repeated_steps(
        [_step_atom("Step 3: Return the unit in the original box", "r9")], seen, "guide.pdf", 7)
    assert len(other) == 1


def test_one_transcription_repeating_a_step_emits_it_once(monkeypatch):
    import app.core.pdf_image_vision as piv

    ocr = "1 Remove the cover 2 Return the unit in the original box 3 Return the unit in the original box"
    monkeypatch.setattr(piv, "_ocr_crop", lambda *a, **k: ocr)
    monkeypatch.setattr(piv, "_vlm", lambda *a, **k: json.dumps({"steps": [
        {"n": 1, "action": "Remove the cover"},
        {"n": 3, "action": "Return the unit in the original box"},
        {"n": 5, "action": "Return the unit in the original box"},
    ]}))
    marker = SimpleNamespace(project_id="p", artifact_id="a", parser_version="t", id="m")
    atoms = piv._transcribe(marker=marker, pdf_name="g.pdf", page_index=4, region_ref="r1",
                            saved_path="x.png", crop=b"x", envelope="", image_kind="instructions")
    texts = [a.raw_text for a in atoms]
    assert texts == ["Step 1: Remove the cover", "Step 3: Return the unit in the original box"]
    assert atoms[1].value["duplicate_step_numbers"] == [5]


# ── 4-7: the signed SOW ───────────────────────────────────────────────────

OOS = ["Long-term warehousing of customer equipment.",
       "Chromebook imaging.",
       "Electrical work, conduit or core drilling.",
       "Removal or disposal of legacy hardware.",
       "Permits and inspection fees.",
       "Install of cabling beyond 10 drops per site.",
       "Repair of pre-existing building damage.",
       "Configuration of customer firewall policies."]
FEES = [("Engineer (est 3 hrs per site)", "99", "$9,504"),
        ("Materials (cabling, connectors, labels)", "99", "$1,485")]


def _signed_sow(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((36, 20), ENVELOPE, fontsize=7)
    page.insert_text((36, 60), "STATEMENT OF WORK", fontsize=16, fontname="hebo")
    y = 95
    page.insert_text((36, y), "Scope of Services", fontsize=12, fontname="hebo")
    y += 20
    for ln in ("Install and configure 2 wireless access points per site.",
               "Mount the network rack and dress all patch cables."):
        page.insert_text((44, y), "•", fontsize=10)
        page.insert_text((56, y), ln, fontsize=10)
        y += 14
    y += 12
    page.insert_text((36, y), "Out of Scope", fontsize=12, fontname="hebo")
    y += 20
    for ln in OOS:
        page.insert_text((44, y), "•", fontsize=10)
        page.insert_text((56, y), ln, fontsize=10)
        y += 14
    y += 12
    page.insert_text((36, y), "Fees", fontsize=12, fontname="hebo")
    y += 20
    for row in (("Description", "Qty", "Amount"),) + tuple(FEES):
        for x, c in zip((36, 330, 450), row):
            page.insert_text((x, y), c, fontsize=10)
        y += 14
    page = doc.new_page(width=612, height=792)
    page.insert_text((36, 20), ENVELOPE, fontsize=7)
    page.insert_text((36, 60), "Signatures", fontsize=12, fontname="hebo")
    for x, (co, nm, ti, dt) in zip((36, 330), (("AMTIVO", "Jane Doe", "VP Operations", "3/5/2025"),
                                               ("ACME IT SERVICES LLC", "John Smith", "Director of Sales", "3/6/2025"))):
        yy = 100
        for t in (co, "DocuSigned by:", nm, f"Name: {nm}", f"Title: {ti}", f"Date: {dt}"):
            page.insert_text((x, yy), t, fontsize=10)
            yy += 14
    doc.save(str(path))
    doc.close()


@pytest.fixture(scope="module")
def sow_atoms(tmp_path_factory):
    pdf = tmp_path_factory.mktemp("sow") / "signed_sow.pdf"
    _signed_sow(pdf)
    return _pdf_atoms(pdf)


def test_out_of_scope_bullets_are_exclusions_never_chatter(sow_atoms):
    by_text = {a.raw_text: a for a in sow_atoms}
    for line in OOS:
        a = by_text.get(line)
        assert a is not None, (line, list(by_text))
        assert _type(a) == "exclusion", (line, _type(a))
        assert "chatter" not in (a.review_flags or []), line


def test_the_fee_table_after_the_exclusions_is_not_excluded(sow_atoms):
    for a in sow_atoms:
        if "$9,504" in a.raw_text or "$1,485" in a.raw_text:
            assert _type(a) != "exclusion", (a.raw_text, _section_path(a))


def test_signature_badges_are_never_scope(sow_atoms):
    sig = [a for a in sow_atoms if any(s in a.raw_text for s in (
        "Jane Doe", "John Smith", "VP Operations", "Director of Sales", "DocuSigned", "3/5/2025"))]
    assert len(sig) >= 8, [a.raw_text for a in sow_atoms]
    for a in sig:
        assert _type(a) in {"signatory", "deal_metadata"}, (a.raw_text, _type(a))
    assert any(_type(a) == "signatory" and a.value.get("name") == "Jane Doe" for a in sig)
    badge = [a for a in sig if a.raw_text == "DocuSigned by:"]
    assert badge and all("chatter" in a.review_flags for a in badge)


def test_docusign_envelope_stamp_is_a_chatter_atom_not_a_title(sow_atoms):
    stamps = [a for a in sow_atoms if "Envelope ID" in a.raw_text]
    assert stamps, "the stamp must stay an atom"
    for a in stamps:
        assert a.raw_text == ENVELOPE
        assert _type(a) == "deal_metadata"
        assert "chatter" in a.review_flags
        assert a.value.get("rejected_by") == "doc_stamp"
    for a in sow_atoms:
        assert not any("Envelope ID" in s for s in _section_path(a)), (a.raw_text, _section_path(a))


def test_stamp_only_atoms_from_any_page_builder_are_flagged():
    from app.parsers.orbitbrief_pdf import _flag_doc_stamps

    a = SimpleNamespace(raw_text=ENVELOPE, value={}, atom_type=None, review_flags=[],
                        review_status=None, confidence=0.9)
    b = SimpleNamespace(raw_text="Install the rack.", value={}, atom_type="scope_item",
                        review_flags=[], review_status=None, confidence=0.9)
    _flag_doc_stamps([a, b])
    assert "chatter" in a.review_flags and a.value["rejected_by"] == "doc_stamp"
    assert b.review_flags == [] and b.atom_type == "scope_item"


def test_section_shapes():
    from app.parsers.sow_sections import (
        is_exclusion_heading,
        is_signature_heading,
        split_doc_stamp,
        under_exclusion_heading,
    )

    for h in ("OUT OF SCOPE", "Out of Scope", "Out of Scope:", "7. Exclusions", "Scope Exclusions",
              "Services Not Included", "Out-of-Scope Items"):
        assert is_exclusion_heading(h), h
    for h in ("Inclusions and Exclusions", "The following is out of scope.", "Exclusions apply to all work"):
        assert not is_exclusion_heading(h), h
    assert is_signature_heading("Signatures") and not is_signature_heading("Signature required on delivery")
    assert split_doc_stamp(ENVELOPE + " Signatures") == (ENVELOPE, "Signatures")
    assert under_exclusion_heading(["SOW", "Out of Scope"])
    assert not under_exclusion_heading(["SOW", "Out of Scope", "Fees"])
    assert not under_exclusion_heading(["SOW", "Scope of Services"])


# ── 4, compile side: the heading is the negation ─────────────────────────

def _atom(text: str, atom_type: str, section_path: list[str]):
    from app.core.schemas import (
        ArtifactType, AtomType, AuthorityClass, EvidenceAtom, ReviewStatus, SourceRef,
    )

    return EvidenceAtom(
        id=f"atm_{abs(hash(text))}", project_id="p", artifact_id="a",
        atom_type=AtomType(atom_type), raw_text=text, normalized_text=text.lower(),
        value={}, entity_keys=[],
        source_refs=[SourceRef(id=f"src_{abs(hash(text))}", artifact_id="a",
                               artifact_type=ArtifactType.pdf, filename="sow.pdf",
                               locator={"page": 0, "section_path": section_path},
                               extraction_method="t", parser_version="t")],
        receipts=[], authority_class=AuthorityClass.contractual_scope, confidence=0.8,
        review_status=ReviewStatus.auto_accepted, review_flags=[], parser_version="t",
    )


def test_an_exclusion_under_an_exclusions_heading_needs_no_negation_word():
    from app.core.atom_type_sanity import demote_exclusions_without_negation

    under = _atom("Long-term warehousing of customer equipment.", "exclusion", ["SOW", "OUT OF SCOPE"])
    loose = _atom("There is a site assessment survey after each install", "exclusion", ["SOW", "Scope"])
    demote_exclusions_without_negation([under, loose])
    assert _type(under) == "exclusion"
    assert _type(loose) == "scope_item"


def test_chatter_prediction_skips_lines_under_an_exclusions_heading():
    from app.core.deal_chatter import mark_chatter

    a = _atom("Looking forward to the opportunity", "scope_item", ["SOW", "Exclusions"])
    b = _atom("Looking forward to the opportunity", "scope_item", ["SOW", "Overview"])
    mark_chatter([a, b])
    assert "chatter" not in a.review_flags
    assert "chatter" in b.review_flags


def _docx_sow(path: Path) -> None:
    docx = pytest.importorskip("docx")
    d = docx.Document()
    d.add_heading("Statement of Work", 1)
    d.add_paragraph().add_run("Scope of Services").bold = True
    d.add_paragraph("Install and configure two wireless access points at each site.", style="List Bullet")
    d.add_paragraph().add_run("Out of Scope").bold = True
    for t in OOS[:2] + ["Install of low voltage cabling beyond 10 drops per site."]:
        d.add_paragraph(t, style="List Bullet")
    d.add_paragraph("Site survey of extra sites.")
    d.add_paragraph().add_run("Fees").bold = True
    d.add_paragraph("Engineer onsite labor is billed per site as listed in the fee table.")
    d.save(str(path))


def test_docx_out_of_scope_lines_are_exclusions_never_chatter(tmp_path):
    from app.parsers.docx_parser import DocxParser

    p = tmp_path / "sow.docx"
    _docx_sow(p)
    out = DocxParser().parse_artifact(project_id="p", artifact_id="a", path=p)
    by_text = {a.raw_text: a for a in getattr(out, "atoms", out)}
    for line in OOS[:2] + ["Install of low voltage cabling beyond 10 drops per site.", "Site survey of extra sites."]:
        a = by_text[line]
        assert _type(a) == "exclusion", (line, _type(a))
        assert "chatter" not in a.review_flags, line
    # the lexical guess rides along for the type head
    assert "scope_item" in by_text["Install of low voltage cabling beyond 10 drops per site."].value["alt_atom_types"]
    # the section after the exclusions is not excluded
    assert _type(by_text["Engineer onsite labor is billed per site as listed in the fee table."]) != "exclusion"
    assert _type(by_text["Install and configure two wireless access points at each site."]) != "exclusion"


# ── 7: priced rows survive the substance gate ─────────────────────────────

@pytest.mark.parametrize("row", [
    "Engineer (est 3 hrs per site) | 99 | $9,504",
    "Engineer (est 3 hrs per site): 99 $9,504",
    "Matls & misc consumables (est) 99 $1,485",
])
def test_priced_rows_are_not_ocr_debris(row):
    from app.core.atom_substance_gate import drop_unreadable_text
    from app.core.text_quality import is_unreadable

    assert not is_unreadable(row)
    kept, dropped = drop_unreadable_text([_atom(row, "scope_item", ["SOW", "Fees"])])
    assert len(kept) == 1 and not dropped


@pytest.mark.parametrize("debris", [
    "44 Marware and materats ae ot nclded",
    "Tes aks wilenur tht projetcompen mee egutements $40",
])
def test_ocr_debris_is_still_debris_with_a_price(debris):
    from app.core.text_quality import is_unreadable

    assert is_unreadable(debris)


def test_a_section_after_the_signatures_is_not_a_signature_block():
    from app.parsers.orbitbrief_pdf import _mark_signature_blocks

    sections = [
        {"heading": "Signatures", "blocks": [
            {"kind": "paragraph", "text": "DocuSigned by:", "lines": ["DocuSigned by:"]}]},
        {"heading": "AMTIVO", "blocks": [
            {"kind": "paragraph", "text": "Jane Doe", "lines": ["Jane Doe", "Title: VP Operations"]}]},
        {"heading": "EXHIBIT A", "blocks": [
            {"kind": "paragraph", "text": "Site List", "lines": ["Site List"]}]},
    ]
    _mark_signature_blocks(sections)
    assert sections[0]["blocks"][0].get("signature_block")
    assert sections[1]["blocks"][0].get("signature_block")
    assert not sections[2]["blocks"][0].get("signature_block")
