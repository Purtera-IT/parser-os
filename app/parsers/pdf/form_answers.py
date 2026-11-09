"""Form answers: what the person filling in a PDF form actually marked.

A filled questionnaire carries its answers outside the page text: typed
values in text widgets, ticked checkbox / radio widgets, highlight or square
markups drawn over an empty box, a caret or free-text note beside a field.
The text parse reads none of these, so a form came back as its blank
template -- every option listed, none chosen -- and two sites' forms that
answered differently read the same.

This pass is additive. For every page it reads the AcroForm widget values and
the markup annotations, maps each mark to the option label printed right of
the box it covers, and to the question on that form-grid row, and returns
one "Question: Answer" record per question with where the answer came from.
A highlight counts only when it covers a box glyph or a widget: a highlight
over plain text is emphasis, not an answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

from app.parsers.pdf import form_grid as fg

#: Markup annotations that can mark an answer.
_COVER_TYPES = {"Highlight", "Square", "Underline", "Squiggly", "StrikeOut"}
_NOTE_TYPES = {"Caret", "FreeText"}
_COUNT_AS_ANSWER = {"Highlight", "Square"}

#: A label that only tells the reader how to answer ("Check all that apply:").
_INSTRUCTION_RE = re.compile(
    r"^\(?\s*(?:check|select|choose|tick|mark|circle|please\s+(?:check|select|choose|indicate))\b", re.I)

Rect = tuple[float, float, float, float]


@dataclass
class FormAnswer:
    question: str
    answers: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    bbox: Rect | None = None

    @property
    def text(self) -> str:
        q = self.question.rstrip().rstrip(":").rstrip()
        a = "; ".join(self.answers)
        return f"{q}: {a}" if q else a


def _rect(r: Any) -> Rect:
    return (float(r[0]), float(r[1]), float(r[2]), float(r[3]))


def _overlaps(a: Rect, b: Rect, pad: float = 0.0) -> bool:
    return (min(a[2], b[2]) + pad > max(a[0], b[0])
            and min(a[3], b[3]) + pad > max(a[1], b[1]))


def _inside(x: float, y: float, r: Rect, pad: float = 1.0) -> bool:
    return r[0] - pad <= x <= r[2] + pad and r[1] - pad <= y <= r[3] + pad


class _Page:
    """One page's characters, glyphs and form-grid cells, read once."""

    def __init__(self, page: Any):
        self.page = page
        self.chars = fg._rawdict_chars(page)
        self.glyphs = [g for g in fg.page_glyphs(page) if g.kind in ("box", "checked")]
        self.ticks = [g for g in fg.page_glyphs(page) if g.kind == "tick"]
        self.grid_rows: list[list[fg.FormCell]] = []
        self._read_grids()

    def _read_grids(self) -> None:
        try:
            tables = list(self.page.find_tables(strategy="lines").tables or [])
        except Exception:
            return
        nested = fg.nested_table_indexes(tables) if len(tables) > 1 else set()
        gmap = fg.page_glyph_map(self.page)
        geo = fg.page_geometry(self.page)
        spans = fg.page_spans(self.page)
        for i, t in enumerate(tables):
            if i in nested:
                continue
            try:
                rows = [[fg.apply_glyph_map(c, gmap) for c in r] for r in t.extract()]
                rows, view = fg.merge_continuation_rows(
                    self.page, t, rows, geometry=geo, spans=spans, gmap=gmap)
                cells = fg.classify_cells(self.page, view, rows, geometry=geo, spans=spans)
            except Exception:
                continue
            for row in cells or []:
                if row:
                    self.grid_rows.append(row)

    # ── option labels ─────────────────────────────────────────────────
    def label_right_of(self, box: Rect) -> str:
        """The option text printed right of a box: the characters on the
        box's line from its right edge up to the next box or a wide gap."""
        cy = (box[1] + box[3]) / 2.0
        h = max(box[3] - box[1], 4.0)
        line = [c for c in self.chars
                if abs((c[1][1] + c[1][3]) / 2.0 - cy) <= 0.6 * h and c[1][0] >= box[2] - 0.5]
        line.sort(key=lambda c: c[1][0])
        out: list[str] = []
        last_x = box[2]
        for c, bbox, font, _flags, size in line:
            if fg.glyph_kind(c, font) or fg._is_tick_char(c, font):
                if out and "".join(out).strip():
                    break
                last_x = bbox[2]
                continue
            if bbox[0] - last_x > max(2.5 * (size or 8.0), 12.0) and "".join(out).strip():
                break
            out.append(c)
            last_x = bbox[2]
        return " ".join("".join(out).split()).strip(" :")

    def option_left_of(self, x: float, y: float) -> str:
        """The option a point inside an options cell belongs to: the label of
        the nearest box to its left on the same line."""
        cands = [g for g in self.glyphs
                 if g.bbox[2] <= x + 1 and abs((g.bbox[1] + g.bbox[3]) / 2.0 - y) <= 6.0]
        if not cands:
            return ""
        g = max(cands, key=lambda g: g.bbox[2])
        return self.label_right_of(g.bbox)

    # ── questions ─────────────────────────────────────────────────────
    def locate(self, x: float, y: float, rect: Rect | None = None) -> tuple[list[fg.FormCell], int] | None:
        """The form-grid cell a mark falls in: the cell its rectangle overlaps
        most (a widget drawn across a cell edge belongs to the cell holding
        most of it), else the cell holding its centre."""
        best: tuple[float, list[fg.FormCell], int] | None = None
        if rect is not None:
            for row in self.grid_rows:
                for n, c in enumerate(row):
                    w = min(rect[2], c.bbox[2]) - max(rect[0], c.bbox[0])
                    h = min(rect[3], c.bbox[3]) - max(rect[1], c.bbox[1])
                    if w > 0 and h > 0 and (best is None or w * h > best[0]):
                        best = (w * h, row, n)
            if best is not None:
                return best[1], best[2]
        for row in self.grid_rows:
            for n, c in enumerate(row):
                if _inside(x, y, c.bbox, pad=0.0):
                    return row, n
        return None

    def question_at(self, x: float, y: float, rect: Rect | None = None) -> tuple[str, str]:
        """(question, cell kind) for a mark at ``(x, y)``: the nearest label
        left of it on its form-grid row; an instruction label ("Check all
        that apply:") is read with the row's own question before it. A row
        with no label takes the question row right above it that has no
        answer cell of its own ("Maximum truck size? (choose below)")."""
        hit = self.locate(x, y, rect)
        if hit is not None:
            row, n = hit
            kind = row[n].kind
            labels = [c for c in row[:n] if c.kind == "label"]
            if row[n].kind == "label" and not labels:
                labels = [row[n]]
            if labels:
                near = labels[-1].text
                first = labels[0].text
                if near is not first and _INSTRUCTION_RE.match(near):
                    return f"{first.rstrip(': ')} - {near}", kind
                return near, kind
            # A row named by its first cell ("Server - Large | 7 | ...").
            if n > 0:
                return row[0].text, kind
            # A row of options only, under the question it answers.
            if all(c.kind == "options" for c in row):
                idx = self.grid_rows.index(row)
                if idx > 0:
                    prev = self.grid_rows[idx - 1]
                    if prev and prev[-1].kind == "label" and prev[-1].bbox[3] <= row[n].bbox[1] + 2:
                        return prev[-1].text, kind
            return "", kind
        return self._label_line_left(x, y), ""

    def _label_line_left(self, x: float, y: float) -> str:
        """Off the grid: the bold text left of the point on its line, else
        the nearest bold line above it."""
        spans = [s for s in fg._text_spans(fg.page_spans(self.page)) if s.bold]
        same = [s for s in spans if s.bbox[2] <= x + 1 and s.bbox[1] - 2 <= y <= s.bbox[3] + 2]
        if same:
            return " ".join(s.text for s in sorted(same, key=lambda s: s.bbox[0])).strip()
        above = [s for s in spans if s.bbox[3] <= y + 1 and y - s.bbox[3] <= 30 and s.bbox[0] <= x]
        if above:
            best = max(above, key=lambda s: s.bbox[3])
            return best.text.strip()
        return ""


def _annot_rects(annot: Any) -> list[Rect]:
    """The areas an annotation marks: a highlight's quads, else its rect."""
    try:
        verts = list(annot.vertices or [])
    except Exception:
        verts = []
    out: list[Rect] = []
    if verts and len(verts) >= 4:
        for i in range(0, len(verts) - 3, 4):
            q = verts[i:i + 4]
            xs = [float(p[0]) for p in q]
            ys = [float(p[1]) for p in q]
            out.append((min(xs), min(ys), max(xs), max(ys)))
    if not out:
        try:
            out.append(_rect(annot.rect))
        except Exception:
            pass
    return out


def _widget_on(w: Any) -> bool:
    v = getattr(w, "field_value", None)
    if v is None or v is False:
        return False
    s = str(v).strip()
    return bool(s) and s.lower() not in ("off", "false", "0", "no")


def read_page_answers(page: Any) -> list[FormAnswer]:
    """Every answer marked on one page, one record per question, in reading
    order (top to bottom, left to right)."""
    try:
        widgets = list(page.widgets() or [])
    except Exception:
        widgets = []
    try:
        annots = [a for a in (page.annots() or []) if a.type[1] in (_COVER_TYPES | _NOTE_TYPES)]
    except Exception:
        annots = []
    if not widgets and not annots:
        return []
    ctx = _Page(page)
    marks: list[tuple[float, float, str, str, str, Rect]] = []  # (y, x, question, answer, source, rect)
    seen: set[tuple[str, str]] = set()

    def add(x: float, y: float, answer: str, source: str, rect: Rect,
            question: str | None = None) -> None:
        answer = " ".join(str(answer or "").split())
        if not answer:
            return
        q = question if question is not None else ctx.question_at(x, y, rect)[0]
        key = (q, answer.casefold())
        if key in seen:
            return
        seen.add(key)
        marks.append((y, x, q, answer, source, rect))

    def checked_box(rect: Rect, source: str) -> bool:
        hit = [g for g in ctx.glyphs if _overlaps(g.bbox, rect, 0.5)]
        for g in hit:
            cx, cy = (g.bbox[0] + g.bbox[2]) / 2.0, (g.bbox[1] + g.bbox[3]) / 2.0
            add(cx, cy, ctx.label_right_of(g.bbox) or "(checked)", source, g.bbox)
        return bool(hit)

    # Widgets: typed text, ticked boxes / radios.
    for w in widgets:
        try:
            r = _rect(w.rect)
            kind = str(getattr(w, "field_type_string", "") or "")
        except Exception:
            continue
        cx, cy = (r[0] + r[2]) / 2.0, (r[1] + r[3]) / 2.0
        if kind in ("CheckBox", "RadioButton"):
            if not _widget_on(w):
                continue
            if not checked_box(r, "widget"):
                add(cx, cy, ctx.label_right_of(r) or "(checked)", "widget", r)
        elif kind in ("Text", "ComboBox", "ListBox"):
            val = getattr(w, "field_value", None)
            if isinstance(val, (list, tuple)):
                val = ", ".join(str(v) for v in val)
            val = " ".join(str(val or "").split())
            if not val:
                continue
            q, cell_kind = ctx.question_at(cx, cy, r)
            if cell_kind == "options":
                opt = ctx.option_left_of(r[0], cy)
                if opt:
                    val = f"{opt} {val}"
            add(cx, cy, val, "widget", r, question=q)

    # Ticks drawn over boxes with no widget behind them (a flattened form).
    for t in ctx.ticks:
        if any(_overlaps(t.bbox, _rect(w.rect), 1.0) for w in widgets):
            continue
        checked_box(t.bbox, "tick")

    # Markups.
    for a in annots:
        typ = a.type[1]
        rects = _annot_rects(a)
        if typ in _COVER_TYPES:
            if typ not in _COUNT_AS_ANSWER:
                continue
            covered = False
            for r in rects:
                covered = checked_box(r, typ.lower()) or covered
            if not covered:
                for w in widgets:
                    try:
                        wr = _rect(w.rect)
                    except Exception:
                        continue
                    if any(_overlaps(wr, r, 0.5) for r in rects):
                        cx, cy = (wr[0] + wr[2]) / 2.0, (wr[1] + wr[3]) / 2.0
                        add(cx, cy, ctx.label_right_of(wr) or "(marked)", typ.lower(), wr)
        else:
            try:
                content = str((a.info or {}).get("content") or "").strip()
            except Exception:
                content = ""
            if not content or not rects:
                continue
            r = rects[0]
            cx, cy = (r[0] + r[2]) / 2.0, (r[1] + r[3]) / 2.0
            q, cell_kind = ctx.question_at(cx, cy, r)
            if cell_kind == "options":
                opt = ctx.option_left_of(r[0], cy)
                if opt:
                    content = f"{opt} {content}"
            add(cx, cy, content, typ.lower(), r, question=q)

    # One record per question, in reading order.
    marks.sort(key=lambda m: (round(m[0] / 3.0), m[1]))
    by_q: dict[str, FormAnswer] = {}
    order: list[str] = []
    for y, x, q, ans, src, rect in marks:
        rec = by_q.get(q)
        if rec is None:
            rec = by_q[q] = FormAnswer(question=q, bbox=rect)
            order.append(q)
        if ans not in rec.answers:
            rec.answers.append(ans)
        if src not in rec.sources:
            rec.sources.append(src)
        b = rec.bbox or rect
        rec.bbox = (min(b[0], rect[0]), min(b[1], rect[1]), max(b[2], rect[2]), max(b[3], rect[3]))
    return [by_q[q] for q in order]


def read_form_answers(pdf_path: Path) -> dict[int, list[FormAnswer]]:
    """``{page_index: [answers]}`` for every page that marks one."""
    out: dict[int, list[FormAnswer]] = {}
    try:
        import fitz  # type: ignore[import-not-found]
    except Exception:
        return out
    try:
        with fitz.open(str(pdf_path)) as doc:
            has_annots = any(True for p in doc for _ in (p.annots() or []))
            if not doc.is_form_pdf and not has_annots:
                return out
            for i, page in enumerate(doc):
                try:
                    found = read_page_answers(page)
                except Exception:
                    continue
                if found:
                    out[i] = found
    except Exception:
        return {}
    return out


def form_answer_atoms(
    *, pdf_path: Path, project_id: str, artifact_id: str, filename: str, parser_version: str,
) -> dict[int, list[Any]]:
    """``{page_index: [atom]}``: one "Question: Answer" atom per answered
    question, its value carrying the question, the answers and where each
    came from (widget / highlight / square / caret / freetext / tick)."""
    from app.core.schemas import AtomType
    from app.core.schemas import AuthorityClass
    from app.parsers.pdf._shared import _make_atom

    out: dict[int, list[Any]] = {}
    for page, answers in read_form_answers(pdf_path).items():
        for n, ans in enumerate(answers):
            text = ans.text
            if not text:
                continue
            out.setdefault(page, []).append(_make_atom(
                text=text,
                project_id=project_id,
                artifact_id=artifact_id,
                filename=filename,
                parser_version=parser_version,
                atom_type=AtomType.scope_item,
                authority_class=AuthorityClass.customer_current_authored,
                confidence=0.9,
                locator={
                    "page": page,
                    "block_id": f"p{page}_form_answers",
                    "row_index": n,
                    "bbox": [round(v, 1) for v in ans.bbox] if ans.bbox else None,
                    "extraction": FORM_ANSWER_EXTRACTION,
                },
                value={
                    "kind": "form_answer",
                    "question": ans.question.rstrip().rstrip(":").rstrip(),
                    "answers": list(ans.answers),
                    "sources": list(ans.sources),
                },
            ))
    return out


#: The locator ``extraction`` a form-answer atom carries.
FORM_ANSWER_EXTRACTION = "pdf_form_answers_v1"
