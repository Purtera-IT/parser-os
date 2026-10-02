"""PDF contact / form grids read one record per row (010353, 010087).

010353's SOW (page 2) came out as "STATEMENT OF WORK: FULL NAME: ..." rows and
one atom per contact field ("JOB TITLE: Director of Op...", "EMAIL ADDRESS:
John", "Ozuna-Diaz: (470) 567-4..."), and 010087's signed SOW glued a
revision table onto the contacts table beside it ("QUOTED BY: Octavian Mitroi
| REVISION HISTORY: First | JOB TITLE: ..."). Now:

* a title band merged across a grid's first row is a caption, never a column
  name prefixed to every row;
* a grid whose cells label themselves ("FULL NAME: Chase Smith") has no
  header row: each row is one atom with all its fields, values that wrapped
  inside a cell joined back;
* a ruled cell never "restores" a clipped prefix from the cell beside it;
* label-over-value contact records with no ruling are one atom per person,
  and a field label ("EMAIL ADDRESS:") never becomes a section heading;
* two framed tables side by side stay two tables;
* a fee row stays one vendor line item.
"""

from __future__ import annotations

from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")


def _atoms(pdf: Path):
    from app.parsers.orbitbrief_pdf import OrbitBriefPdfParser

    out = OrbitBriefPdfParser().parse_artifact("p", "art", pdf)
    return list(out if isinstance(out, list) else out.atoms)


def _sp(a) -> list[str]:
    return list((a.source_refs[0].locator or {}).get("section_path") or [])


def _grid(page, xs, y0, h, rows, *, ruled=True, wrap=False, bold_head=False):
    for r, row in enumerate(rows):
        for c, t in enumerate(row):
            rect = fitz.Rect(xs[c], y0 + r * h, xs[c + 1], y0 + (r + 1) * h)
            if ruled:
                page.draw_rect(rect, color=(0, 0, 0), width=0.7)
            if not t:
                continue
            font = "hebo" if (bold_head and r == 0) else "helv"
            if wrap:
                page.insert_textbox(fitz.Rect(rect.x0 + 3, rect.y0 + 3, rect.x1 - 3, rect.y1 - 3),
                                    t, fontsize=9, fontname=font)
            else:
                page.insert_text((rect.x0 + 3, rect.y0 + 14), t, fontsize=8, fontname=font)


def test_a_title_band_over_a_grid_is_never_a_row_prefix(tmp_path):
    pdf = tmp_path / "sow.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.draw_rect(fitz.Rect(36, 60, 576, 90), color=(0, 0, 0), width=0.7)
    page.insert_text((240, 80), "STATEMENT OF WORK", fontsize=14, fontname="hebo")
    _grid(page, [36, 196, 376, 576], 90, 30, [
        ["FULL NAME: Chase Smith", "JOB TITLE: Director of Operations", "EMAIL ADDRESS: chase.smith@example.com"],
        ["FULL NAME: Megan Blevins", "JOB TITLE: Global Strategy Lead", "EMAIL ADDRESS: megan.b@example.com"],
    ])
    doc.save(str(pdf))
    texts = [a.raw_text for a in _atoms(pdf)]
    assert "FULL NAME: Chase Smith | JOB TITLE: Director of Operations | EMAIL ADDRESS: chase.smith@example.com" in texts, texts
    assert "FULL NAME: Megan Blevins | JOB TITLE: Global Strategy Lead | EMAIL ADDRESS: megan.b@example.com" in texts, texts
    for t in texts:
        assert "STATEMENT OF WORK:" not in t and "col_" not in t and not t.startswith("|"), t


def test_a_self_labelled_contact_grid_is_one_atom_per_row_with_wrapped_values(tmp_path):
    pdf = tmp_path / "sow.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((200, 60), "STATEMENT OF WORK", fontsize=18, fontname="hebo")
    _grid(page, [40, 160, 290, 430, 572], 100, 40, [
        ["QUOTED BY: Tanner Norris", "FULL NAME: Chase Smith", "JOB TITLE: Director of Operations", "DATE: September 17, 2025"],
        ["PREPARED FOR: Ox", "FULL NAME: Megan Blevins", "JOB TITLE: Global Strategy Lead", "PHONE: (404) 555-0100"],
        ["", "FULL NAME: John Ozuna-Diaz", "EMAIL ADDRESS: john.ozuna@example.com", "PHONE: (470) 567-4000"],
    ], wrap=True)
    page.insert_text((230, 770), "WWW.PURTERA-IT.COM", fontsize=8)
    doc.save(str(pdf))
    atoms = _atoms(pdf)
    texts = [a.raw_text for a in atoms]
    assert ("QUOTED BY: Tanner Norris | FULL NAME: Chase Smith | JOB TITLE: Director of Operations"
            " | DATE: September 17, 2025") in texts, texts
    assert ("PREPARED FOR: Ox | FULL NAME: Megan Blevins | JOB TITLE: Global Strategy Lead"
            " | PHONE: (404) 555-0100") in texts, texts
    # the empty first cell leaves no "| ", the wrapped name and email are whole
    assert "FULL NAME: John Ozuna-Diaz | EMAIL ADDRESS: john.ozuna@example.com | PHONE: (470) 567-4000" in texts, texts
    for t in texts:
        assert not t.startswith("|") and "Norris FULL NAME" not in t and "::" not in t, t
    footer = [a for a in atoms if a.raw_text == "WWW.PURTERA-IT.COM"]
    assert all("chatter" in a.review_flags for a in footer)


def test_label_over_value_contacts_are_one_atom_per_person(tmp_path):
    pdf = tmp_path / "sow.pdf"
    people = [("Chase Smith", "Director of Operations", "chase.smith@example.com", "(404) 555-0100"),
              ("Megan Blevins", "Global Strategy Lead", "megan.blevins@example.com", "(404) 555-0111"),
              ("John Ozuna-Diaz", "IT Manager", "john.ozuna-diaz@example.com", "(470) 567-4000")]
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((220, 50), "STATEMENT OF WORK", fontsize=16, fontname="hebo")
    page.insert_text((36, 90), "CUSTOMER CONTACTS", fontsize=11, fontname="hebo")
    y = 115
    for person in people:
        for x, lab, val in zip((36, 170, 310, 470), ("FULL NAME:", "JOB TITLE:", "EMAIL ADDRESS:", "PHONE:"), person):
            page.insert_text((x, y), lab, fontsize=8, fontname="hebo")
            page.insert_text((x, y + 11), val, fontsize=8)
        y += 34
    page.insert_text((36, y + 20), "Purtera will dispatch a technician to each site listed above.", fontsize=10)
    doc.save(str(pdf))
    atoms = _atoms(pdf)
    texts = [a.raw_text for a in atoms]
    for name, title, email, phone in people:
        want = f"FULL NAME: {name} | JOB TITLE: {title} | EMAIL ADDRESS: {email} | PHONE: {phone}"
        assert want in texts, texts
    for a in atoms:
        assert not any(s.rstrip().endswith(":") for s in _sp(a)), (a.raw_text, _sp(a))
    assert "Purtera will dispatch a technician to each site listed above." in texts


def test_two_framed_tables_side_by_side_are_never_one_row(tmp_path):
    pdf = tmp_path / "sow.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((220, 50), "STATEMENT OF WORK", fontsize=16, fontname="hebo")
    _grid(page, [36, 140, 290], 80, 22, [["QUOTED BY:", "JOB TITLE:"], ["Octavian Mitroi", "Executive VP Sales"]],
          ruled=False, bold_head=True)
    _grid(page, [330, 440, 576], 80, 22, [["REVISION HISTORY:", "DATE:"], ["First", "03/01/2025"]],
          ruled=False, bold_head=True)
    page.draw_rect(fitz.Rect(36, 80, 290, 124), color=(0, 0, 0), width=0.7)
    page.draw_rect(fitz.Rect(330, 80, 576, 124), color=(0, 0, 0), width=0.7)
    page.insert_text((36, 200), "The technician will connect the VC links in each room.", fontsize=10)
    doc.save(str(pdf))
    texts = [a.raw_text for a in _atoms(pdf)]
    assert "QUOTED BY: Octavian Mitroi | JOB TITLE: Executive VP Sales" in texts, texts
    assert "REVISION HISTORY: First | DATE: 03/01/2025" in texts, texts
    for t in texts:
        assert not ("Octavian" in t and "REVISION" in t), t


@pytest.mark.parametrize("ruled", [True, False])
def test_a_fee_row_stays_one_line_item(tmp_path, ruled):
    pdf = tmp_path / "sow.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((200, 60), "STATEMENT OF WORK", fontsize=18, fontname="hebo")
    page.insert_text((36, 100), "FEES", fontsize=12, fontname="hebo")
    _grid(page, [36, 250, 340, 420, 480, 576], 115, 22, [
        ["Service Description", "Hourly Rate", "Unit", "Quantity", "Total Price"],
        ["Engineer (est 3 hrs per site)", "$96.00", "Hourly", "99", "$9,504.00"],
    ], ruled=ruled)
    page.insert_text((36, 220), "Invoices are due net 30 from the invoice date.", fontsize=10)
    doc.save(str(pdf))
    rows = [a for a in _atoms(pdf) if "$9,504.00" in a.raw_text or "$96.00" in a.raw_text]
    assert len(rows) == 1, [a.raw_text for a in rows]
    a = rows[0]
    assert a.raw_text == ("Service Description: Engineer (est 3 hrs per site) | Hourly Rate: $96.00"
                          " | Unit: Hourly | Quantity: 99 | Total Price: $9,504.00")
    assert getattr(a.atom_type, "value", a.atom_type) == "vendor_line_item"


def test_shapes():
    from app.parsers.orbitbrief_pdf import _looks_like_section_heading, _row_to_text
    from app.parsers.pdf._shared import _drop_title_band, _grid_is_self_labelled

    for t in ("EMAIL ADDRESS:", "JOB TITLE:", "FULL NAME:", "PHONE:", "QUOTED BY:"):
        assert not _looks_like_section_heading(t), t
    assert _looks_like_section_heading("CUSTOMER CONTACTS")
    assert _looks_like_section_heading("SCOPE OF WORK:")
    band = [["STATEMENT OF WORK", None, None], ["A: 1", "B: 2", "C: 3"]]
    assert _drop_title_band(band) == band[1:]
    # an empty header cell of its own (not a merged span) keeps the header
    assert _drop_title_band([["Item", ""], ["Cat6", "5"]]) == [["Item", ""], ["Cat6", "5"]]
    assert _grid_is_self_labelled([["FULL NAME: A", "JOB TITLE: B"], ["FULL NAME: C", "PHONE: 1"]])
    assert not _grid_is_self_labelled([["Description", "Qty"], ["Cat6 drop", "4"]])
    assert _row_to_text({"QUOTED BY:": "Octavian Mitroi", "Qty": "4"}) == "QUOTED BY: Octavian Mitroi | Qty: 4"
