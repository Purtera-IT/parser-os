"""PDF forms drawn as ruled grids: row / field separation and marked answers.

Every fixture is a synthetic PDF built here (made-up labels and values): a
Word-style form grid whose borders are thin filled rectangles, whose label
cells are bold on a shaded band with an inset band per text line (which is
what makes ``find_tables`` split one cell into sub-rows), and whose option
boxes are Wingdings glyphs. Widgets and annotations are added with PyMuPDF.
"""
from __future__ import annotations

from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

from app.parsers.orbitbrief_pdf import OrbitBriefPdfParser  # noqa: E402
from app.parsers.orbitbrief_pdf import _looks_like_section_heading  # noqa: E402
from app.parsers.orbitbrief_pdf import _merge_table_extractions  # noqa: E402
from app.parsers.orbitbrief_pdf import _row_to_text  # noqa: E402
from app.parsers.pdf import form_grid as fg  # noqa: E402
from app.parsers.pdf.form_answers import read_form_answers  # noqa: E402

H = 792.0  # page height: fixtures are laid out top-down, PDF draws bottom-up


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    monkeypatch.setenv("SOWSMITH_DISABLE_LLM", "1")


# ── a tiny PDF writer: base-14 text, a Wingdings box glyph, filled rects ──

def _esc(t: str) -> str:
    return t.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


class _Canvas:
    def __init__(self) -> None:
        self.ops: list[str] = []

    def fill(self, x0: float, y0: float, x1: float, y1: float, gray: float) -> None:
        self.ops.append(f"{gray} g {x0} {H - y1} {x1 - x0} {y1 - y0} re f")

    def hrule(self, x0: float, x1: float, y: float) -> None:
        self.fill(x0, y - 0.25, x1, y + 0.25, 0.6)

    def vrule(self, x: float, y0: float, y1: float) -> None:
        self.fill(x - 0.25, y0, x + 0.25, y1, 0.6)

    def text(self, x: float, baseline: float, s: str, *, bold: bool = False, size: float = 8) -> None:
        font = "F2" if bold else "F1"
        self.ops.append(f"0 g BT /{font} {size} Tf {x} {H - baseline} Td ({_esc(s)}) Tj ET")

    def box(self, x: float, baseline: float, size: float = 8) -> None:
        """A Wingdings empty box (code 0xA8), the way Word sets a form box."""
        self.ops.append(f"0 g BT /F3 {size} Tf {x} {H - baseline} Td (\\250) Tj ET")

    def pdf(self) -> bytes:
        content = "\n".join(self.ops).encode("latin-1")
        objs: list[bytes] = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R /F2 5 0 R /F3 6 0 R >> >> /Contents 7 0 R >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
            b"<< /Type /Font /Subtype /TrueType /BaseFont /Wingdings-Regular "
            b"/Encoding << /Type /Encoding /Differences [168 /uniF0A8] >> >>",
            f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream",
        ]
        out = b"%PDF-1.4\n"
        offs = []
        for i, o in enumerate(objs, 1):
            offs.append(len(out))
            out += f"{i} 0 obj\n".encode() + o + b"\nendobj\n"
        xref = len(out)
        out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
        for o in offs:
            out += f"{o:010d} 00000 n \n".encode()
        out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
        return out


L0, L1, R = 30.0, 110.0, 580.0  # label column and table edges
SHADE = 0.9


def _label(c: _Canvas, y0: float, y1: float, lines: list[tuple[str, bool]], x1: float = L1) -> None:
    """A shaded label cell with one inset band per text line (Word's layout)."""
    c.fill(L0, y0, x1, y1, SHADE)
    step = (y1 - y0 - 2) / len(lines)
    for n, (t, bold) in enumerate(lines):
        top = y0 + 1 + n * step
        c.fill(L0 + 4, top, x1 - 4, top + step, SHADE)
        c.text(L0 + 6, top + step - 2.5, t, bold=bold)


def _form_pdf() -> _Canvas:
    c = _Canvas()
    tops = [100, 116, 140, 164, 190, 214, 238, 262]
    # Row 0: one field.
    _label(c, 100, 116, [("CUSTOMER", True)])
    c.text(118, 111, "Acme Testing Co", size=10)
    # Row 1: a label wrapped over two lines, and a second field on the row.
    _label(c, 116, 140, [("BACKUP", True), ("COORDINATOR", True)])
    c.text(118, 131, "Jordan Example", size=10)
    c.fill(250, 116, 300, 140, SHADE)
    c.text(254, 131, "PHONE:", bold=True)
    c.text(305, 131, "555-0100", size=10)
    # Row 2: a label with its "(...)" caption under it; four fields on the row.
    _label(c, 140, 164, [("DOCK ADDRESS", True), ("(Pickup Point)", False)])
    c.text(118, 155, "12 Sample Rd", size=10)
    for x, lab, val in ((250, "CITY:", "Faketown"), (380, "STATE:", "ZZ"), (470, "ZIP:", "00000")):
        c.fill(x, 140, x + 40, 164, SHADE)
        c.text(x + 4, 155, lab, bold=True)
        c.text(x + 45, 155, val, size=10)
    # Row 3: a question wrapped over two lines, options beside it.
    _label(c, 164, 190, [("Is a loading", True), ("ramp available?", True)], x1=245)
    c.box(255, 180)
    c.text(265, 180, "YES")
    c.box(300, 180)
    c.text(310, 180, "NO")
    # Row 4: another question.
    _label(c, 190, 214, [("Is a forklift on site?", True)], x1=245)
    c.box(255, 205)
    c.text(265, 205, "YES")
    c.box(300, 205)
    c.text(310, 205, "NO")
    # Row 5: a field filled in a widget (added later), row 6: a typed note.
    _label(c, 214, 238, [("Pallet count", True)])
    _label(c, 238, 262, [("Special notes:", True)])
    c.text(118, 253, "Call ahead", size=10)
    # Borders: every row ruled apart; the cell walls of each row.
    for y in tops:
        c.hrule(L0, R, y)
    for y0, y1 in zip(tops, tops[1:]):
        c.vrule(L0, y0, y1)
        c.vrule(R, y0, y1)
    for y0, y1 in zip(tops, tops[1:]):
        if y0 in (164, 190):
            c.vrule(245, y0, y1)
        else:
            c.vrule(L1, y0, y1)
    c.vrule(250, 116, 140)
    c.vrule(300, 116, 140)
    for x in (250, 290, 380, 420, 470, 510):
        c.vrule(x, 140, 164)
    return c


def _save(tmp_path: Path, name: str, data: bytes) -> Path:
    p = tmp_path / name
    p.write_bytes(data)
    return p


def _table_rows(pdf: Path) -> list[str]:
    atoms = getattr(OrbitBriefPdfParser().parse(pdf), "atoms", [])
    return [a.raw_text.strip() for a in atoms
            if isinstance(a.value, dict) and a.value.get("kind") == "table_row"]


def _answers(pdf: Path) -> list[str]:
    atoms = getattr(OrbitBriefPdfParser().parse(pdf), "atoms", [])
    return [a.raw_text.strip() for a in atoms
            if isinstance(a.value, dict) and a.value.get("kind") == "form_answer"]


# ── 1 + 2: continuation merge and form-grid reading, end to end ──────────

def test_form_grid_rows_read_as_label_value_pairs(tmp_path):
    rows = _table_rows(_save(tmp_path, "form.pdf", _form_pdf().pdf()))
    assert "CUSTOMER: Acme Testing Co" in rows
    # A label wrapped over two lines is one label; a second field on the row
    # keeps its own label.
    assert "BACKUP COORDINATOR: Jordan Example | PHONE: 555-0100" in rows
    # The caption under a bold label belongs to it; four fields on one row.
    assert "DOCK ADDRESS (Pickup Point): 12 Sample Rd | CITY: Faketown | STATE: ZZ | ZIP: 00000" in rows
    assert "Special notes: Call ahead" in rows
    ramp = [r for r in rows if r.startswith("Is a loading ramp available?:")]
    assert ramp and "☐" in ramp[0] and "YES" in ramp[0] and "NO" in ramp[0]
    # Nothing split off as its own row, nothing keyed by the first field,
    # no placeholder column names.
    for orphan in ("COORDINATOR", "(Pickup Point)", "ramp available?", "STATE:"):
        assert orphan not in rows
    assert not any(r.startswith("CUSTOMER:") and "Acme" not in r for r in rows)
    assert not any("col_" in r for r in rows)


def test_rows_ruled_apart_are_never_merged(tmp_path):
    """Two one-line labels in rows a rule separates stay two rows."""
    rows = _table_rows(_save(tmp_path, "form.pdf", _form_pdf().pdf()))
    assert any(r.startswith("Is a forklift on site?") for r in rows)
    assert not any("ramp available?" in r and "forklift" in r for r in rows)


def test_merge_continuation_rows_folds_sub_rows(tmp_path):
    pdf = _save(tmp_path, "form.pdf", _form_pdf().pdf())
    with fitz.open(str(pdf)) as doc:
        page = doc[0]
        table = page.find_tables(strategy="lines").tables[0]
        raw = table.extract()
        rows, view = fg.merge_continuation_rows(page, table, raw)
    assert len(rows) < len(raw)
    assert len(view.rows) == len(rows)
    flat = [" ".join(fg._text(c) for c in r if fg._text(c)) for r in rows]
    assert any("BACKUP COORDINATOR" in t for t in flat)
    assert any("Is a loading ramp available?" in t for t in flat)


# ── 2: the form-grid gate ────────────────────────────────────────────────

def _cell(text: str, kind: str, x0: float, x1: float = 0.0) -> fg.FormCell:
    return fg.FormCell(text, (x0, 0.0, x1 or x0 + 50.0, 10.0), kind)


def test_header_over_data_rows_is_not_a_form():
    cells = [
        [_cell("Item", "label", 0), _cell("Qty", "label", 60), _cell("Price", "label", 120)],
        [_cell("Widget", "value", 0), _cell("2", "value", 60), _cell("$4", "value", 120)],
        [_cell("Gadget", "value", 0), _cell("1", "value", 60), _cell("$9", "value", 120)],
    ]
    assert not fg.is_form_grid(cells)


def test_bold_row_names_under_a_header_are_not_a_form():
    cells = [
        [_cell("Site", "label", 0), _cell("Start", "label", 60)],
        [_cell("North", "label", 0), _cell("May 2", "value", 60)],
        [_cell("South", "label", 0), _cell("May 9", "value", 60)],
    ]
    assert not fg.is_form_grid(cells)


def test_read_form_row_pairs_each_label_with_its_values():
    row = [_cell("ADDRESS", "label", 0), _cell("1 Test Way", "value", 60),
           _cell("CITY:", "label", 120), _cell("Faketown", "value", 180),
           _cell("ZIP:", "label", 240)]
    text, fields = fg.read_form_row(row)
    assert text == "ADDRESS: 1 Test Way | CITY: Faketown | ZIP:"
    assert fields == [("ADDRESS", "1 Test Way"), ("CITY", "Faketown"), ("ZIP", "")]


def test_paired_questions_take_one_line_of_options_each():
    q = fg.FormCell("Is there a crane? If so, may we use it?", (0, 0, 100, 20), "label")
    o = fg.FormCell("☐YES ☐NO ☑YES ☐NO", (100, 0, 200, 20), "options", ["☐YES ☐NO", "☑YES ☐NO"])
    text, fields = fg.read_form_row([q, o])
    assert fields == [("Is there a crane?", "☐YES ☐NO"), ("If so, may we use it?", "☑YES ☐NO")]


# ── glyphs ───────────────────────────────────────────────────────────────

def test_wingdings_box_maps_to_ballot_box(tmp_path):
    pdf = _save(tmp_path, "form.pdf", _form_pdf().pdf())
    with fitz.open(str(pdf)) as doc:
        gmap = fg.page_glyph_map(doc[0])
    assert gmap.get("") == "☐"
    assert fg.apply_glyph_map("YES NO", gmap) == "☐YES ☐NO"


def test_a_tick_read_beside_its_box_is_one_checked_box():
    assert fg.apply_glyph_map("\x14 Work  Home", {"": "☐", "\x14": "☑"}) == "☑ Work ☐ Home"


def test_glyph_kind_needs_the_symbol_font():
    assert fg.glyph_kind("", "Wingdings-Regular") == "box"
    assert fg.glyph_kind("o", "Wingdings-Regular") == "box"
    assert fg.glyph_kind("o", "Helvetica") is None


# ── 3: no placeholder names in atom text ─────────────────────────────────

def test_placeholder_column_names_never_reach_the_text():
    assert _row_to_text({"col_0": "Price", "Qty": "2", "col_3": "50%"}) == "Price | Qty: 2 | 50%"
    assert _row_to_text({"QUOTED BY:": "Pat"}) == "QUOTED BY: Pat"


# ── 4: header-grid replacement guard ─────────────────────────────────────

def test_a_small_header_grid_does_not_replace_a_big_ruled_table():
    ruled = [{"kind": "table", "columns": ["A", "col_1", "col_2"], "rows": [{"A": "x"}]}]
    big = fitz.Rect(0, 0, 500, 700)
    small = fitz.Rect(0, 0, 500, 300)   # 43% of the ruled table
    grid = [{"kind": "table", "columns": ["H1", "H2"], "rows": [{"H1": "y"}]}]
    blocks, _ = _merge_table_extractions(ruled, [big], grid, [small], min_cover=0.8)
    assert blocks[0] is ruled[0]
    blocks, _ = _merge_table_extractions(ruled, [big], grid, [fitz.Rect(0, 0, 500, 650)], min_cover=0.8)
    assert blocks == grid


# ── 5: "#" headings and option lines ─────────────────────────────────────

def test_hash_is_a_heading_marker_only_before_a_word():
    assert _looks_like_section_heading("# Scope of Work")
    assert not _looks_like_section_heading("# of pallets")
    assert not _looks_like_section_heading("#12 bracket")
    assert not _looks_like_section_heading("☐YES ☐NO")
    assert not _looks_like_section_heading(" BOXES  PALLETS")


# ── 6: form answers from widgets and markups ─────────────────────────────

def _with_marks(tmp_path: Path) -> Path:
    pdf = _save(tmp_path, "blank.pdf", _form_pdf().pdf())
    doc = fitz.open(str(pdf))
    page = doc[0]
    # A ticked checkbox widget over the ramp question's YES box.
    cb = fitz.Widget()
    cb.field_type = fitz.PDF_WIDGET_TYPE_CHECKBOX
    cb.field_name = "ramp_yes"
    cb.rect = fitz.Rect(254, 173, 262, 181)
    cb.field_value = True
    page.add_widget(cb)
    # An unticked one over its NO box: not an answer.
    cb2 = fitz.Widget()
    cb2.field_type = fitz.PDF_WIDGET_TYPE_CHECKBOX
    cb2.field_name = "ramp_no"
    cb2.rect = fitz.Rect(299, 173, 307, 181)
    cb2.field_value = False
    page.add_widget(cb2)
    # A typed value in a text widget.
    tw = fitz.Widget()
    tw.field_type = fitz.PDF_WIDGET_TYPE_TEXT
    tw.field_name = "pallets"
    tw.rect = fitz.Rect(115, 218, 300, 234)
    tw.field_value = "3"
    page.add_widget(tw)
    # A highlight over the forklift question's NO box: an answer.
    page.add_highlight_annot(fitz.Rect(299, 198, 318, 207))
    # A highlight over plain text: emphasis, not an answer.
    page.add_highlight_annot(fitz.Rect(117, 103, 200, 113))
    # A caret note on the notes row.
    caret = page.add_caret_annot(fitz.Point(200, 250))
    caret.set_info(content="after 2pm")
    caret.update()
    out = tmp_path / "filled.pdf"
    doc.save(str(out))
    doc.close()
    return out


def test_form_answers_read_widgets_highlights_and_notes(tmp_path):
    pdf = _with_marks(tmp_path)
    found = read_form_answers(pdf)
    texts = {a.text: a.sources for page in found.values() for a in page}
    assert texts.get("Is a loading ramp available?: YES") == ["widget"]
    assert texts.get("Is a forklift on site?: NO") == ["highlight"]
    assert texts.get("Pallet count: 3") == ["widget"]
    assert texts.get("Special notes: after 2pm") == ["caret"]
    # The unticked box and the highlight over plain text are not answers.
    assert not any("Acme" in t for t in texts)
    assert not any(t.startswith("Is a loading ramp available?: NO") for t in texts)


def test_form_answers_become_atoms_after_their_page(tmp_path):
    answers = _answers(_with_marks(tmp_path))
    assert "Is a forklift on site?: NO" in answers
    assert "Pallet count: 3" in answers


def test_a_pdf_with_no_widgets_or_markups_has_no_answers(tmp_path):
    assert read_form_answers(_save(tmp_path, "blank.pdf", _form_pdf().pdf())) == {}
