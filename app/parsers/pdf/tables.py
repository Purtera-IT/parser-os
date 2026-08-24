"""Table recovery from a PDF, by ruling, by column geometry, and by text.

A PDF has no table structure -- only ink. These are the three ways of inferring one: trust the ruled cells fitz reports, infer columns from x-positions when there are no rules, or read a vertically-stacked label/value run that is a table in intent only.
"""

from __future__ import annotations

from app.core.ids import stable_id
from app.core.normalizers import normalize_text
from app.core.schemas import ArtifactType
from app.core.schemas import AtomType
from app.core.schemas import AuthorityClass
from app.core.schemas import EvidenceAtom
from app.core.schemas import ReviewStatus
from app.core.schemas import SourceRef
from app.parsers.pdf._shared import TABLE_ROW_CONFIDENCE
from app.parsers.pdf._shared import _NEW_TABLE_HEADER_RE
from app.parsers.pdf._shared import _classify_table
from app.parsers.pdf._shared import _looks_like_form_field
from app.parsers.pdf._shared import _looks_like_page_footer
from app.parsers.pdf._shared import _make_atom
from app.parsers.pdf._shared import _table_rows_repaired
from pathlib import Path
from typing import Any
import re


_REVISION_ROW_RE = re.compile(
    r"^rev(?:ision)?\s*[:#]?\s*\d+(?:\.\d+)?", re.IGNORECASE
)
_TOOL_ROW_RE = re.compile(r"^item\s*:\s*\d+\s*\|\s*description\s*:", re.IGNORECASE)

def _looks_like_document_control_row(row_text: str, columns: list[str] | None = None) -> bool:
    """True for a table row that describes the DOCUMENT, not the work.

    Two kinds show up in every vendor install spec and both were landing as
    ``scope_item`` -- i.e. as contractual scope a quote/scope head reads:

      * revision history ("Rev 1.4 | 12/03/2025 | Add roll door installation")
        -- a changelog of the PDF, not work anyone is buying.
      * the tools list ("Item: 1 | Description: Cordless Drill") -- what the
        installer brings in their van. A cordless drill is not a deliverable.

    On the Xtra Lease spec that was 20 of 123 scope items (17%). Both stay as
    atoms (the census still needs the region covered, and the revision date is
    genuinely useful provenance) -- they are just typed as metadata instead of
    scope.
    """
    t = " ".join(str(row_text or "").split())
    if not t:
        return False
    cols = {str(c or "").strip().lower() for c in (columns or [])}
    if _TOOL_ROW_RE.match(t):
        return True
    if _REVISION_ROW_RE.match(t.split("|")[0].strip()):
        return True
    if {"revision", "date"} <= cols or {"item", "description"} <= cols:
        return True
    return False

def _structured_doc_has_tables(structured_doc: dict[str, Any]) -> bool:
    """True iff any section/subsection contains a block of kind='table'."""
    for page in structured_doc.get("pages") or []:
        for section in page.get("sections") or []:
            stack: list[dict[str, Any]] = [section]
            while stack:
                cur = stack.pop()
                for b in cur.get("blocks") or []:
                    if isinstance(b, dict) and b.get("kind") == "table":
                        return True
                for sub in cur.get("subsections") or []:
                    stack.append(sub)
    return False

def _fitz_generic_table_fallback(
    *,
    pdf_path: Path,
    project_id: str,
    artifact_id: str,
    parser_version: str,
    structured_doc: dict[str, Any],
) -> list[EvidenceAtom]:
    """Recover ANY tabular content fitz.find_tables sees that the
    structured pipeline didn't surface.

    The structured extractor's heuristic table detector misses
    reportlab-generated tables and some scanned/CSV-converted PDFs.
    fitz's vector-based table finder catches those. We emit one
    table_row-shaped atom per row so enrich_entities can pull
    part_numbers / quantities / money out of the cells.

    Skipped entirely when the structured pipeline already exposed
    at least one table — that path is more accurate and we don't
    want to double-emit.
    """
    if _structured_doc_has_tables(structured_doc):
        return []
    try:
        import fitz  # type: ignore[import-not-found]
    except Exception:
        return []
    out: list[EvidenceAtom] = []
    try:
        doc = fitz.open(str(pdf_path))
    except Exception:
        return []
    try:
        for page_index, page in enumerate(doc):
            try:
                tables_finder = page.find_tables()
            except Exception:
                continue
            tables = list(getattr(tables_finder, "tables", []) or [])
            if not tables:
                continue
            for table_index, table in enumerate(tables):
                try:
                    extracted = _table_rows_repaired(page, table)
                except Exception:
                    continue
                if not extracted or len(extracted) < 2:
                    continue
                header = [(c or "").strip() for c in extracted[0]]
                body = extracted[1:]
                # Build columns list (use col_N for blank headers)
                columns = [
                    header[i] if i < len(header) and header[i] else f"col_{i}"
                    for i in range(len(header) if header else (len(body[0]) if body else 0))
                ]
                # Classify the table once (pricing vs scope)
                sample_cells: list[str] = []
                for r in body[:5]:
                    for c in r or ():
                        if c is None:
                            continue
                        s = " ".join(str(c).split()).strip()
                        if s:
                            sample_cells.append(s)
                try:
                    atom_type, authority = _classify_table(
                        section_path=[],
                        columns=columns,
                        sample_cells=sample_cells,
                    )
                except Exception:
                    atom_type, authority = AtomType.scope_item, AuthorityClass.contractual_scope
                # Bbox for locator
                try:
                    bbox = table.bbox
                    locator_base = {
                        "page": int(page_index),
                        "block_kind": "table",
                        "bbox": list(bbox),
                        "extraction": "fitz_generic_table_fallback_v1",
                        "table_index": table_index,
                    }
                except Exception:
                    locator_base = {
                        "page": int(page_index),
                        "block_kind": "table",
                        "extraction": "fitz_generic_table_fallback_v1",
                        "table_index": table_index,
                    }
                for row_index, row in enumerate(body):
                    if not row:
                        continue
                    cells: dict[str, str] = {}
                    cell_strs: list[str] = []
                    for i, c in enumerate(row):
                        col_name = columns[i] if i < len(columns) else f"col_{i}"
                        val = " ".join(str(c or "").split()).strip()
                        if val:
                            cells[col_name] = val
                            cell_strs.append(f"{col_name}: {val}")
                    if not cells:
                        continue
                    row_text = " | ".join(cell_strs)
                    # Skip rows whose text is a form-field / page-footer
                    if _looks_like_form_field(row_text) or _looks_like_page_footer(row_text):
                        continue
                    row_atom_type = atom_type
                    if _looks_like_document_control_row(row_text, columns):
                        row_atom_type = AtomType.deal_metadata
                    out.append(
                        _make_atom(
                            text=row_text,
                            project_id=project_id,
                            artifact_id=artifact_id,
                            filename=pdf_path.name,
                            parser_version=parser_version,
                            atom_type=row_atom_type,
                            authority_class=authority,
                            confidence=TABLE_ROW_CONFIDENCE,
                            locator={**locator_base, "row_index": row_index},
                            value={
                                "kind": "table_row",
                                "columns": columns,
                                "cells": cells,
                                "fallback": "fitz_generic_table",
                            },
                        )
                    )
    finally:
        try:
            doc.close()
        except Exception:
            pass
    return out

def _extract_column_tables(pdf_path: Path, page_index: int) -> tuple[list[dict[str, Any]], list[Any]]:
    """Recover UNRULED column tables on a text-rich page from word geometry.

    Many real tables have NO ruling lines — a ``date | event`` timeline, a
    ``Factor | Weight`` criteria grid, a two-column ``key | value`` block. PyMuPDF's
    ``get_text("text")`` flattens these into reading order, so the prose splitter
    mashes the columns into scrambled paragraphs (the date divorced from its event,
    a row split across two atoms) and can even drop a cell. ``find_tables(strategy=
    "lines")`` can't see them (no lines), or sees phantom columns.

    This reconstructs them from the WHITESPACE columns the way a human reads them:
      1. group ``get_text("words")`` into visual lines (by PDF block/line ids);
      2. find a *consistent vertical whitespace river* — a column boundary x that
         several stacked lines all leave empty — establishing the column grid;
      3. assign every word to a column by that grid, folding wrapped continuation
         lines (left-cell text with no right-cell word) into the row above;
      4. emit a clean ``kind="table"`` block + its bbox, same shape as the ruled
         path, so the existing table→atom emitter turns each row into ONE atom with
         all columns together (no scramble, no divorced cell, no lost cell).

    Conservative by construction (won't fire on flowing prose): needs ≥2 rows whose
    boundary aligns within a tight tolerance, both sides populated, columns
    left-aligned. Returns ``([], [])`` on anything it can't prove tabular.
    """
    try:
        import fitz  # type: ignore[import-not-found]
    except Exception:
        return [], []

    try:
        with fitz.open(str(pdf_path)) as doc:
            page = doc[page_index]
            page_w = float(page.rect.width) or 612.0
            try:
                words = page.get_text("words") or []
            except Exception:
                return [], []
    except Exception:
        return [], []
    if len(words) < 4:
        return [], []

    # 1. group words into VISUAL lines by y-band. PyMuPDF gives each column cell
    #    its own (block, line) id even when they share a row, so we cluster by
    #    vertical position instead — words within ~half a line-height of each
    #    other belong to the same visual row (this is what re-unites a date with
    #    its event, or a factor with its weight).
    valid = [w for w in words if len(w) >= 8 and str(w[4]).strip()]
    if len(valid) < 4:
        return [], []
    heights = sorted(w[3] - w[1] for w in valid)
    line_h = heights[len(heights) // 2] or 10.0
    y_tol = max(3.0, 0.6 * line_h)
    valid.sort(key=lambda w: (w[1], w[0]))
    lines: list[dict[str, Any]] = []
    cur: list[Any] = []
    cur_y = None
    for w in valid:
        if cur and abs(w[1] - cur_y) <= y_tol:
            cur.append(w)
        else:
            if cur:
                ws = sorted(cur, key=lambda t: t[0])
                lines.append({"y0": min(t[1] for t in ws), "y1": max(t[3] for t in ws),
                              "x0": min(t[0] for t in ws), "x1": max(t[2] for t in ws),
                              "words": ws})
            cur = [w]
            cur_y = w[1]
    if cur:
        ws = sorted(cur, key=lambda t: t[0])
        lines.append({"y0": min(t[1] for t in ws), "y1": max(t[3] for t in ws),
                      "x0": min(t[0] for t in ws), "x1": max(t[2] for t in ws),
                      "words": ws})
    if len(lines) < 2:
        return [], []
    GAP_MIN = max(38.0, 0.055 * page_w)   # a real column river, not inter-word space
    ALIGN_TOL = 10.0                       # how tightly stacked boundaries must agree

    def _gap_boundary(ws: list[Any]) -> float | None:
        """Left edge of the RIGHT column across the widest inter-word gap, or None.
        The right column's left edge is the STABLE rail (a left cell of varying
        width shifts a gap *midpoint*, but the right column starts at a fixed x)."""
        best = None  # (gap_size, right_col_x0)
        for a, b in zip(ws, ws[1:]):
            gap = b[0] - a[2]
            if gap > GAP_MIN and (best is None or gap > best[0]):
                best = (gap, b[0])
        return None if best is None else best[1]

    # 2. anchor lines: a clear internal whitespace river at a fixed right-col rail.
    anchors: list[tuple[int, float]] = []  # (line_index, right_col_x)
    for i, L in enumerate(lines):
        bx = _gap_boundary(L["words"])
        if bx is not None:
            anchors.append((i, bx))
    if len(anchors) < 2:
        return [], []

    # 3. cluster anchors by their boundary rail (within ALIGN_TOL). Each cluster
    #    of ≥2 stacked anchors that share a rail is one column region.
    anchors.sort(key=lambda t: t[1])
    clusters: list[list[tuple[int, float]]] = []
    for a in anchors:
        if clusters and abs(a[1] - clusters[-1][0][1]) <= ALIGN_TOL:
            clusters[-1].append(a)
        else:
            clusters.append([a])

    blocks: list[dict[str, Any]] = []
    bboxes: list[Any] = []

    for cluster in clusters:
        if len(cluster) < 2:
            continue
        X = sorted(c[1] for c in cluster)[len(cluster) // 2]  # median rail
        anchor_lis = sorted(c[0] for c in cluster)
        top_li, bot_li = anchor_lis[0], anchor_lis[-1]
        # 4. build rows over the anchor span. A line with a RIGHT-column word
        #    opens a row; a left-only line folds into the row above (wrapped
        #    cell). A line whose word straddles the rail is prose → end the
        #    region there (this also separates two stacked tables / strips an
        #    intro sentence that bleeds into the span).
        rows_lr: list[list[str]] = []
        region_lis: list[int] = []
        cont_run = 0
        for li in range(top_li, bot_li + 1):
            ws = lines[li]["words"]
            if any(t[0] < X - 2.0 and t[2] > X + 2.0 for t in ws):
                break  # a word crosses the rail → not a cell boundary → prose
            left = [t for t in ws if t[2] <= X + 1.0]
            right = [t for t in ws if t[0] >= X - 1.0]
            ltxt = " ".join(t[4] for t in left).strip()
            rtxt = " ".join(t[4] for t in right).strip()
            if right:
                rows_lr.append([ltxt, rtxt])
                region_lis.append(li)
                cont_run = 0
            elif rows_lr and left:
                cont_run += 1
                if cont_run > 6:   # a "cell" wrapping >6 lines is really prose
                    break
                rows_lr[-1][0] = (rows_lr[-1][0] + " " + ltxt).strip()
                region_lis.append(li)
            elif left:
                break  # left-only line before any 2-col row → not a clean table
        # require ≥2 rows and both columns genuinely populated across the region.
        if len(rows_lr) < 2:
            continue
        if sum(1 for r in rows_lr if r[0]) < 2 or sum(1 for r in rows_lr if r[1]) < 2:
            continue

        # 5. header detection: a first row whose cells are short, all-alpha
        #    labels (no digits) names the columns; otherwise generic col_N.
        def _is_label(s: str) -> bool:
            return bool(s) and not any(c.isdigit() for c in s) and len(s.split()) <= 3

        # 5a. A LABEL:VALUE FORM has no header — every row is a field pair (Name |
        #     Austin Coryell, Date | …, BK Store Number | 557). The left column is
        #     all field labels and the right column is HETEROGENEOUS values (text,
        #     dates, numbers). Treat each row as a key:value pair so it reads
        #     "Name: Austin Coryell", not a header cross-product
        #     ("Name: Site Number | Austin Coryell: 16404"). A real data grid
        #     (Factor | Weight, all-numeric right column) keeps header behaviour.
        def _form_label(s: str) -> bool:
            s = (s or "").strip()
            return bool(s) and len(s.split()) <= 5 and s[-1:] not in ".!?:" \
                and any(c.isalpha() for c in s)

        def _numlike(s: str) -> bool:
            return bool(re.fullmatch(r"[-$(]?[\d.,/: ]+(?:[ap]\.?m\.?)?\)?%?",
                                     (s or "").strip(), re.I))
        lefts = [r[0] for r in rows_lr]
        rights = [r[1] for r in rows_lr if (r[1] or "").strip()]
        # Require >=4 field rows so this fires on a genuine FORM (Name/Date/Store#/
        # Site#/Arrival/Departure) and NOT on a 2-3 row data table whose first row
        # is a real header ("Type | Qty." over "Cameras | 65") — which would
        # otherwise leak the header pair "Type: Qty." as a noise atom.
        is_form = (len(rows_lr) >= 4
                   and sum(1 for ltxt in lefts if _form_label(ltxt)) >= max(2, 0.7 * len(lefts))
                   and (not rights or sum(1 for v in rights if _numlike(v)) < 0.8 * len(rights)))
        if is_form:
            form_rows = rows_lr
            # Even a form-shaped grid can open with a real COLUMN-HEADER pair
            # ("Type | Qty.", "Equipment Type | Included in Buildout") — drop it
            # when the right cell is a generic header word, so it doesn't leak as
            # a "Type: Qty." noise atom. A genuine field pair (right cell is a
            # VALUE: "Name | Austin Coryell") is kept.
            _GENERIC_HEADER_RIGHT = {
                "qty", "qty.", "quantity", "included in buildout", "total",
                "number", "total number", "status", "description", "unit",
                "price", "amount", "value", "notes", "cost", "rate",
            }
            if form_rows and (form_rows[0][1] or "").strip().lower().rstrip(":") in _GENERIC_HEADER_RIGHT:
                form_rows = form_rows[1:]
            columns = ["col_0"]
            out_rows = [{"col_0": f"{r[0]}: {r[1]}".strip(" :")}
                        for r in form_rows if (r[0] or r[1])]
        else:
            if _is_label(rows_lr[0][0]) and _is_label(rows_lr[0][1]):
                columns = [rows_lr[0][0], rows_lr[0][1]]
                data = rows_lr[1:]
            else:
                columns = ["col_0", "col_1"]
                data = rows_lr
            out_rows = [{columns[0]: r[0], columns[1]: r[1]} for r in data
                        if r[0] or r[1]]
        if len(out_rows) < 1:
            continue
        blocks.append({"kind": "table", "columns": columns, "rows": out_rows,
                       "extraction": "column_whitespace_v1"})
        x0 = min(lines[li]["x0"] for li in region_lis)
        y0 = min(lines[li]["y0"] for li in region_lis)
        x1 = max(lines[li]["x1"] for li in region_lis)
        y1 = max(lines[li]["y1"] for li in region_lis)
        try:
            bboxes.append(fitz.Rect(x0, y0, x1, y1))
        except Exception:
            bboxes.append(None)

    bboxes = [b for b in bboxes if b is not None]
    return blocks, bboxes

# Each profile is a dict so we can attach optional per-profile guards
# without touching every existing entry.  Required keys:
#   header:          tuple[str, ...]   — header tokens, lower
#   atom_kind:       str               — value.kind tag
#   atom_type:       str               — AtomType enum value
#   field_names:     tuple[str, ...]   — value-dict keys for each cell
#   locator_label:   str               — short tag in source_ref.locator
# Optional keys (post-v8 boss review hardening):
#   first_cell_re:   compiled regex    — every row's first cell MUST match
#                                        or the table parser stops early
#   row_stop_re:     compiled regex    — when the FIRST cell matches, stop
#                                        (e.g., page-2 measurement table
#                                        below field-checklist).
_PORT_TOKEN_RE = re.compile(r"^(gi|fa|te|xe|et|eth|mgmt)\d+/\d+(/\d+)?$", re.I)
_RFI_ID_RE = re.compile(r"^rfi-\d{2,4}$", re.I)
_RB_ID_RE = re.compile(r"^rb-\d{2,4}$", re.I)
_MEAS_ID_RE = re.compile(r"^m-\d{2,4}$", re.I)
_FCHK_NUM_RE = re.compile(r"^\d{1,3}$")

_VERTICAL_TABLE_PROFILES: list[dict] = [
    {
        "header": ("#", "survey item", "status", "area", "note"),
        "atom_kind": "field_checklist_row_v2",
        "atom_type": "scope_item",
        "field_names": ("item_no", "item", "status", "area", "note"),
        "locator_label": "field_check",
        "first_cell_re": _FCHK_NUM_RE,  # F2 — only digits
    },
    # Boss-review v9 C002-F3 — Managed Services Acceptance Checklist
    # ("# / Acceptance Item / Status / Owner / Due"). Status values
    # like "Customer Pending" / "Exception" / "blocked by vendor"
    # belong here as ``open_question`` / ``action_item`` atoms, NOT
    # as scope_exclusion atoms.
    {
        "header": ("#", "acceptance item", "status", "owner", "due"),
        "atom_kind": "managed_services_acceptance_checklist_row",
        "atom_type": "open_question",
        "field_names": ("item_no", "item", "status", "owner", "due"),
        "locator_label": "msp_acceptance_checklist",
        "first_cell_re": _FCHK_NUM_RE,
    },
    {
        "header": ("rfi", "issue", "owner", "status", "needed by"),
        "atom_kind": "rfi_row",
        "atom_type": "open_question",
        "field_names": ("rfi_id", "issue", "owner", "status", "needed_by"),
        "locator_label": "rfi",
        "first_cell_re": _RFI_ID_RE,  # F3 — strictly RFI-### only
    },
    {
        "header": ("ref", "measurement", "value", "field note"),
        "atom_kind": "working_measurement_row",
        "atom_type": "quantity",
        "field_names": ("ref", "measurement", "value", "field_note"),
        "locator_label": "measurement",
        "first_cell_re": _MEAS_ID_RE,  # F2 — strictly M-### only
    },
    {
        "header": ("port", "patch", "vlan/use", "note"),
        "atom_kind": "port_vlan_assignment",
        "atom_type": "port_vlan_assignment",
        "field_names": ("port", "patch", "vlan_use", "note"),
        "locator_label": "port_vlan",
        "first_cell_re": _PORT_TOKEN_RE,  # F4 — must be Gi/Fa/Te/etc switch port
    },
    {
        "header": ("runbook", "trigger", "owner", "status", "evidence"),
        "atom_kind": "runbook_row",
        "atom_type": "action_item",
        "field_names": ("runbook_id", "trigger", "owner", "status", "evidence"),
        "locator_label": "runbook",
        "first_cell_re": _RB_ID_RE,  # only RB-### rows
    },
]

def _vertical_table_atoms_from_text(
    *,
    project_id: str,
    artifact_id: str,
    filename: str,
    page_number: int,
    text: str,
    parser_version: str,
) -> list[EvidenceAtom]:
    """Detect vertical-listed tables on a PDF page.

    Boss-review F3+F4: the original ``_field_checklist_atoms_from_text``
    required all 5 cells on one line. PyMuPDF on hand-form/scanned-feel
    PDFs returns each cell on its OWN line. We detect headers on
    consecutive lines, then chunk subsequent lines into N-row groups.
    """
    out: list[EvidenceAtom] = []
    lines = [ln.rstrip() for ln in text.splitlines()]
    norm = [normalize_text(ln).strip() for ln in lines]
    i = 0
    while i < len(lines):
        # Skip empties cheaply.
        if not norm[i]:
            i += 1
            continue
        for profile in _VERTICAL_TABLE_PROFILES:
            header_tokens = profile["header"]
            atom_kind = profile["atom_kind"]
            atom_type_str = profile["atom_type"]
            field_names = profile["field_names"]
            locator_label = profile["locator_label"]
            first_cell_re = profile.get("first_cell_re")
            n = len(header_tokens)
            # Try to align the next n non-empty lines to header_tokens.
            cand: list[int] = []
            j = i
            while j < len(lines) and len(cand) < n:
                if norm[j]:
                    cand.append(j)
                j += 1
            if len(cand) < n:
                continue
            if not all(norm[cand[k]] == header_tokens[k] for k in range(n)):
                continue
            # Header matched. Read row groups.
            row_idx = 0
            cursor = cand[-1] + 1
            while cursor < len(lines):
                row_lines: list[int] = []
                while cursor < len(lines) and len(row_lines) < n:
                    if norm[cursor]:
                        # Boss-review v8 F2/F3/F4/F5 — STOP if the
                        # FIRST cell of a new row matches a known new-
                        # table header. We only apply this check at
                        # row boundaries (len(row_lines)==0) so we
                        # don't accidentally cut a row mid-way when a
                        # data cell happens to share a header word.
                        if len(row_lines) == 0 and _NEW_TABLE_HEADER_RE.match(lines[cursor].strip()):
                            break
                        row_lines.append(cursor)
                    cursor += 1
                if len(row_lines) < n:
                    break
                row_values = [lines[ix].strip() for ix in row_lines]
                first = row_values[0]
                # Heuristic + per-profile guard — the first cell must
                # match the profile's expected pattern (Gi…, RFI-###,
                # M-###, RB-###, or an integer for field-checklist).
                if not first or first.isupper() and len(first.split()) > 4:
                    break
                if first_cell_re is not None and not first_cell_re.match(first):
                    # Stop the table; the row that failed the guard is
                    # likely the start of a different section.
                    break
                row_dict = dict(zip(field_names, row_values))
                # Determine atom type — "OPEN" status → constraint not scope_item.
                atype = atom_type_str
                status = row_dict.get("status", "")
                if atom_kind == "field_checklist_row_v2" and status.upper() in {"OPEN", "BLOCKED", "PENDING", "EXCEPTION", "RFI"}:
                    atype = "constraint"
                # Build atom.
                try:
                    resolved_atom_type = AtomType(atype)
                except ValueError:
                    resolved_atom_type = AtomType.scope_item
                row_id = row_values[0] or f"row_{row_idx}"
                source_ref = SourceRef(
                    id=stable_id("src", artifact_id, "pdf", page_number, locator_label, row_id),
                    artifact_id=artifact_id,
                    artifact_type=ArtifactType.pdf,
                    filename=filename,
                    locator={
                        "page": page_number,
                        "vertical_table": locator_label,
                        "row_index": row_idx,
                        "row_id": row_id,
                    },
                    extraction_method=f"pdf_vertical_table_v1::{atom_kind}",
                    parser_version=parser_version,
                )
                pretty = " | ".join(f"{fn}: {row_values[k]}" for k, fn in enumerate(field_names))
                out.append(
                    EvidenceAtom(
                        id=stable_id(
                            "atm", project_id, artifact_id, atom_kind,
                            page_number, row_idx, *row_values,
                        ),
                        project_id=project_id,
                        artifact_id=artifact_id,
                        atom_type=resolved_atom_type,
                        raw_text=pretty,
                        normalized_text=normalize_text(pretty),
                        value={
                            "kind": atom_kind,
                            "page": page_number,
                            "row_index": row_idx,
                            **row_dict,
                        },
                        entity_keys=[],
                        source_refs=[source_ref],
                        receipts=[],
                        authority_class=AuthorityClass.customer_current_authored,
                        confidence=0.86,
                        review_status=ReviewStatus.auto_accepted,
                        review_flags=[],
                        parser_version=parser_version,
                    )
                )
                row_idx += 1
            # After processing the table, advance i past it.
            i = cursor
            break
        else:
            i += 1
    return out
