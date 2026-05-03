"""LLM content extraction from detect_standalone overlays.

# Organizational rule system (colors force structure)

The overlay colors are rules; each forces a specific content role. The LLM
only ever sees clean text — no coordinates, no RGB, no detection stats.

| color   | id pattern                                       | content role                                 | forcing rule                                                                                                                 |
| ------- | ------------------------------------------------ | -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| PURPLE  | ``titleblk*``                                    | document identity                            | Wrappers that enclose any PURPLE → TITLE-BLOCK sidebar, feeds the document header only. Never appears in the body.           |
| BLUE    | ``vN`` (each wrapper)                            | SECTION                                      | Every non-title-block BLUE wrapper becomes one section (or several sub-sections, see next rule). Full-page outer wrappers skipped. |
| BLUE    | ``vN_secM_title``                                | sub-section boundary                         | If a wrapper has multiple ``_secM_title`` children, the wrapper is sliced into sub-sections at those y-values.               |
| BLUE    | ``vN_title`` / ``vN_secM_title``                 | section heading                              | Used as the display title of the section / sub-section it scopes.                                                            |
| CYAN    | ``colhdr_*`` or ``mccol_*_hdr_*``              | **column-splitters for a TABLE**             | A section that contains CYAN is forced into TABLE mode. CYAN x-spans define columns for the ORANGE data cells inside.        |
| ORANGE  | any (``color == ORANGE``)                        | data cells / prose tokens                    | Words below the CYAN band are distributed to columns by x-overlap. When a section has no CYAN, words are emitted as prose.   |
| GREEN   | ``minitable_*_mtcelld``                          | abbreviation cell                            | Two GREEN cells per RED row = one ``symbol: meaning`` pair.                                                                  |
| RED     | ``minitable_*_mtrow``                            | abbreviation row scope                       | See GREEN.                                                                                                                    |
| BLUE    | ``textsec_N_title`` + ``textsec_N_body``         | prose (notes)                                | Titled prose block.                                                                                                           |
| —       | ``line_repair_*``                                | discarded                                    | Not content.                                                                                                                  |

Extraction strategy:

- **Text at wrapper level.** We extract words from inside the BLUE wrapper
  (always a reliable bbox) using ``page.get_text('words', clip=rect)``.
- **Columns from CYAN.** For table sections we split those words into
  columns using the CYAN x-spans (the detection step already determined
  these; we respect them as forcing rules).
- **Rows by y-clustering.** Words beneath the CYAN band are clustered into
  rows by y-center.
- **Notes / prose.** Sections without CYAN are rendered as their raw
  wrapper text, in reading order. Standalone ``textsec_N_title`` /
  ``textsec_N_body`` pairs (from title detection) are also emitted as
  ``kind: notes`` when their title centroid is **not** inside an emitted
  table slice—so general-notes strips appear even when CYAN forces wall
  schedules into ``kind: table``.
- **Abbreviations.** GREEN cells grouped by RED rows → ``symbol: meaning``,
  with a ``sections`` item ``kind: abbreviations`` (``entries``) in sheet order,
  and the same data in the top-level ``abbreviations`` array. The right margin
  ``TITLE:`` line is the **sheet title**; it is hoisted to
  ``document.sheet_title`` and the rest of the title block (firm, scale, job)
  is **dropped from ``sections``** — those fields are pure metadata, not body
  copy. ``full_text`` still contains the raw page text for callers that need it.
- **Responsibility matrix → ``contractor_matrix``.** If a notes section title
  contains *RESPONSIBILITY* and *MATRIX*, extraction upgrades it to structured
  ``kind: contractor_matrix`` (three columns + optional ``footer_note`` after
  ``NOTE:``). Rules: (1) use **unsorted** ``get_text("text", clip=…)`` for
  that section — **not** ``sort=True`` (sort interleaves multi-column layout);
  (2) column text stops at the first ``NOTE:`` so the next section on the page
  does not append to the last column; (3) if the PDF text stream omits
  *ELECTRICAL CONTRACTOR TO PROVIDE* but electrical bullets are merged into
  the security list, split at a small set of lead-in patterns (e.g. ``4X4``,
  ``KNOX BOX``, or the *MISCELLANEOUS … LABELS … CONDUIT* line); (4) if
  header-based parsing fails, **fallback** assigns words in the clip to three
  *x*-clusters (1D k-means) and reads each column top-to-bottom.

**Presentation order** (``sections`` in JSON and Markdown): title/kind **tiers**
put *General notes* and *code summary* before *schedules*, and the *title-block
margin* last. On **mixed** pages (prose, ``mccol``, contractor matrix, **and**
at least one CYAN schedule table), all **body** schedule tables
(``kind: table`` with default tier) are forced **after** any non-table body
section; keyword-titled ``table`` blocks (general / code / index) stay in the
early group by tier. Within a group, top-then-left order is kept. The
``abbreviations`` **section** (when present) is ordered before the title-block
margin. The top-level ``abbreviations`` field mirrors the same list for
callers that read the root only.

Artifacts (same base path as the overlay image / JSON):

- ``<base>.extraction.json`` — structured, content-only
- ``<base>.extraction.md``   — Markdown, one H2 per detected section
"""
from __future__ import annotations

__all__ = [
    "extract_from_overlay_json",
    "write_extraction_artifacts",
]

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_from_overlay_json(
    overlay_json: dict[str, Any] | str | Path,
    pdf_path: str | Path | None = None,
) -> dict[str, Any]:
    if not isinstance(overlay_json, dict):
        overlay_json = json.loads(Path(overlay_json).read_text())
    assert isinstance(overlay_json, dict)
    pdf = pdf_path or overlay_json.get("pdf")
    if not pdf or not Path(str(pdf)).is_file():
        raise FileNotFoundError(f"PDF not found: {pdf!r}")
    page_index = int(overlay_json.get("page", 0))
    return _build_document(overlay_json, str(pdf), page_index)


def write_extraction_artifacts(
    out_base: str | Path,
    doc: dict[str, Any],
    *,
    write_json: bool = True,
    write_markdown: bool = True,
) -> dict[str, str]:
    out_base = Path(out_base)
    stem = (
        out_base.with_suffix("")
        if out_base.suffix.lower() in (".json", ".png", ".md")
        else out_base
    )
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    # ----- Crop symbol images for every table row (first column) -----
    internals = doc.pop("_internals", None)
    symbols_rel_dir = f"{stem.name}.symbols"
    symbols_abs_dir = stem.parent / symbols_rel_dir
    if internals is not None:
        symbols_abs_dir.mkdir(parents=True, exist_ok=True)
        # Drop crops from prior runs so row indices never leave stale secXX_rowNNN.png
        # gaps (e.g. after table row-count changes).
        for stale in symbols_abs_dir.glob("sec*_row*.png"):
            try:
                stale.unlink()
            except OSError:
                pass
        page = internals["page_ref"]
        doc_ref = internals["doc_ref"]
        scale = internals["scale"]
        rotated_cw = internals["rotated_cw"]
        page_height_pt = internals["page_height_pt"]
        try:
            for si, section in enumerate(doc.get("sections") or [], 1):
                section.pop("_presentation_sort", None)
                anchors = section.pop("_row_anchors", None)
                if section.get("kind") != "table" or not anchors:
                    continue
                for ri, (row, row_anchors) in enumerate(
                    zip(section.get("rows") or [], anchors), 1
                ):
                    if not isinstance(row, dict) or "symbol" not in row:
                        continue
                    sym_cell = row_anchors[0] if row_anchors else None
                    if sym_cell is None:
                        continue
                    fn = f"sec{si:02d}_row{ri:03d}.png"
                    out_png = symbols_abs_dir / fn
                    ok = _crop_region_to_png(
                        page, sym_cell["px_bbox"], scale, rotated_cw,
                        page_height_pt, out_png,
                        render_scale=3.0, pad_pt=1.0,
                    )
                    if ok:
                        row["symbol"]["image"] = f"{symbols_rel_dir}/{fn}"
        finally:
            try:
                doc_ref.close()
            except Exception:
                pass

    if write_json:
        p = Path(str(stem) + ".extraction.json")
        p.write_text(
            json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        paths["json"] = str(p)
    if write_markdown:
        p2 = Path(str(stem) + ".extraction.md")
        p2.write_text(_render_markdown(doc), encoding="utf-8")
        paths["markdown"] = str(p2)
    return paths


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _import_fitz():
    import fitz

    return fitz


_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub(" ", s.replace("\r", " ")).strip()


def _pdf_rect(px_bbox, scale: float, pad: float = 0.0, *, rotated_cw: bool = False, page_height_pt: float | None = None):
    """Detector-space px_bbox → PyMuPDF PDF-point rect.

    When ``rotated_cw`` is True the detector rotated the raster 90° CW during
    render (landscape content on a portrait page). Content rotation forward:
    portrait (x_p, y_p) → detector landscape (H - y_p, x_p) where H is the
    portrait page height in points. Inverse: (x_d, y_d) → (y_d, H - x_d).

    A rect (xd0, yd0, xd1, yd1) transforms to (yd0, H - xd1, yd1, H - xd0).
    """
    fitz = _import_fitz()
    x0, y0, x1, y1 = px_bbox
    xd0, yd0, xd1, yd1 = x0 / scale, y0 / scale, x1 / scale, y1 / scale
    if rotated_cw and page_height_pt is not None:
        H = float(page_height_pt)
        return fitz.Rect(
            yd0 - pad,
            H - xd1 - pad,
            yd1 + pad,
            H - xd0 + pad,
        )
    return fitz.Rect(xd0 - pad, yd0 - pad, xd1 + pad, yd1 + pad)


def _crop_region_to_png(
    page,
    px_bbox,
    scale: float,
    rotated_cw: bool,
    page_height_pt: float,
    out_path: Path,
    render_scale: float = 3.0,
    pad_pt: float = 1.0,
) -> bool:
    """Render just the pixels inside ``px_bbox`` (detector-landscape space)
    to a PNG at ``out_path``. Returns True on success.

    Used to save symbol-column glyphs so an LLM can see what each symbol
    looks like.
    """
    fitz = _import_fitz()
    rect = _pdf_rect(
        px_bbox, scale, pad=pad_pt,
        rotated_cw=rotated_cw, page_height_pt=page_height_pt,
    )
    if rect.is_empty or rect.is_infinite:
        return False
    try:
        mat = fitz.Matrix(render_scale, render_scale)
        pix = page.get_pixmap(matrix=mat, clip=rect, alpha=False)
    except Exception:
        return False
    if pix.width <= 2 or pix.height <= 2:
        return False
    try:
        if rotated_cw:
            # Re-orient to match the landscape overlay (detector space).
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            img = img.transpose(Image.ROTATE_270)  # CW 90°
            out_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(out_path)
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            pix.save(str(out_path))
    except Exception:
        return False
    return True


def _derotate_word(w: dict, page_height_pt: float) -> dict:
    """Map a PyMuPDF portrait word back to detector landscape PDF points.

    Inverse of the rect transform: PyMuPDF (x_p, y_p) → detector (H - y_p, x_p).
    """
    H = float(page_height_pt)
    return {
        "x0": H - w["y1"],
        "y0": w["x0"],
        "x1": H - w["y0"],
        "y1": w["x1"],
        "text": w["text"],
    }


def _bbox_overlaps(a, b) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return ax1 > bx0 and bx1 > ax0 and ay1 > by0 and by1 > ay0


def _bbox_contains(outer, inner) -> bool:
    ox0, oy0, ox1, oy1 = outer
    ix0, iy0, ix1, iy1 = inner
    return ox0 <= ix0 and oy0 <= iy0 and ox1 >= ix1 and oy1 >= iy1


def _render_scale(data: dict) -> float:
    try:
        return float((data.get("debug_stats") or {}).get("render_scale_used") or 2.5)
    except Exception:
        return 2.5


def _rotated_cw(data: dict) -> bool:
    return bool((data.get("debug_stats") or {}).get("rotated_cw"))


# ---------------------------------------------------------------------------
# Document-level header parsing (PURPLE-driven)
# ---------------------------------------------------------------------------


_SHEET_NUM_RE = re.compile(r"^[A-Z]\d{2,4}(?:\.\d)?$")
_DATE_RE = re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4}$")
_JOB_RE = re.compile(r"^\d{2,4}[A-Z]{1,3}\d{3,8}$")
_PHONE_RE = re.compile(r"^\d{3}-\d{3}-\d{4}$")
_TECH_BLOCK = {
    "RJ45", "CAT6", "CAT6A", "CAT5", "CAT5E", "CMP", "CMR", "CMX",
    "POE", "120VAC", "208VAC", "240VAC", "277VAC", "480VAC",
    "IPTV", "MATV", "OM3", "OM4", "LC", "VOIP", "WIFI", "NTI", "ASD",
}


def _document_header_from_text(plain_text: str) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    if not plain_text:
        return meta
    tokens: list[str] = []
    for line in re.split(r"\s{2,}|\n", plain_text):
        for tok in line.split():
            t = tok.strip(",.:;")
            if t:
                tokens.append(t)

    for tok in tokens:
        if tok in _TECH_BLOCK:
            continue
        if "sheet_number" not in meta and _SHEET_NUM_RE.fullmatch(tok):
            meta["sheet_number"] = tok
    # Prefer two-digit-year dates (issue dates like 12/15/23) over four-digit
    # years (plot dates). If none of the former is found, fall back to any.
    for tok in tokens:
        if "date" not in meta and _DATE_RE.fullmatch(tok) and re.search(r"/\d{2}$", tok):
            meta["date"] = tok
    for tok in tokens:
        if "date" not in meta and _DATE_RE.fullmatch(tok):
            meta["date"] = tok
    for tok in tokens:
        if tok in _TECH_BLOCK:
            continue
        if (
            "job_number" not in meta
            and _JOB_RE.fullmatch(tok)
            and tok != meta.get("sheet_number")
        ):
            meta["job_number"] = tok
    for tok in tokens:
        if "phone" not in meta and _PHONE_RE.fullmatch(tok):
            meta["phone"] = tok
            break

    concat = " ".join(plain_text.split())
    sn = meta.get("sheet_number")
    jn = meta.get("job_number")
    if sn:
        stop = r"RESPONSIBILITY|MATRIX|SHEET|NO\.|DATE|REVISIONS|PROJECT|CLIENT|ARCHITECT|SEAL|ISSUED|CONSULTANT"
        pat = rf"{re.escape(sn)}\s+([A-Z][A-Z0-9 &/\-]{{2,35}}?)(?=\s+(?:{stop})|\s{{2,}}|$)"
        if jn:
            m = re.search(rf"{re.escape(jn)}\s+" + pat, concat)
            if m:
                meta["sheet_title"] = m.group(1).strip().rstrip(",.-")
        if "sheet_title" not in meta:
            m = re.search(pat, concat)
            if m:
                meta["sheet_title"] = m.group(1).strip().rstrip(",.-")

    _STOP = r"CLIENT|ARCHITECT|PROJECT|SEAL|REVISIONS|ISSUED|SHEET|DATE|JOB|CONSULTANT|SCALE|NORTH|REV\."
    _name = r"[A-Z][A-Z0-9 &'.\-,]{3,60}?"
    for label, key in (("PROJECT", "project"), ("CLIENT", "client"), ("ARCHITECT", "architect")):
        m = re.search(
            rf"\b{label}\b\s+({_name})(?=\s+(?:{_STOP})\b|\s+\d|\s{{2,}}|$)",
            concat,
        )
        if m:
            val = m.group(1).strip().rstrip(",.-")
            if val.upper() not in {"CLIENT", "ARCHITECT", "PROJECT"}:
                meta[key] = val
    return meta


def _prose_header_from_full_text(full_text: str) -> dict[str, Any]:
    """Universal header lines from raw ``page.get_text`` (GC work orders, memos).

    Architectural title blocks often omit center-sheet prose; this scans the
    flattened page text *before* ``SCOPE OF WORK`` / major section cues so
    ``document`` carries title, address, units count, and start date.
    """
    if not full_text:
        return {}
    norm = re.sub(r"[\s\u00a0]+", " ", full_text.strip())
    scope_m = re.search(r"\sSCOPE OF WORK\b", norm, re.I)
    if scope_m:
        head_region = norm[: scope_m.start()].strip()
    else:
        cut = re.search(
            r"\s(?:UNITS|COMMON AREA|NOTES|DESCRIPTION)\b",
            norm,
            re.I,
        )
        head_region = norm[: cut.start()].strip() if cut else norm[:1800]

    out: dict[str, Any] = {}
    if not head_region:
        return out

    m = re.search(
        r"(Work Order\s*[–-]\s*.+?)(?=\s+\d{3,5}\s+[A-Z])",
        head_region,
        re.I,
    )
    if m:
        out["work_order_title"] = m.group(1).strip()

    m = re.search(
        r"(\d{3,5}\s+[A-Z0-9][^,]{2,80}?,\s*[^,]+?,\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?)",
        head_region,
    )
    if m:
        out["site_address"] = re.sub(r"\s+", " ", m.group(1).strip())

    m = re.search(r"Total\s+Units\s*:\s*(\d+)", head_region, re.I)
    if m:
        out["total_units"] = int(m.group(1))

    m = re.search(
        r"Est\.\s*Start\s+Date\s*[–-]\s*(\d{1,2}/\d{1,2}/\d{2,4})",
        head_region,
        re.I,
    )
    if m:
        out["start_date"] = m.group(1).strip()

    hb_parts: list[str] = []
    if out.get("work_order_title"):
        hb_parts.append(out["work_order_title"])
    if out.get("site_address"):
        hb_parts.append(out["site_address"])
    if out.get("total_units") is not None:
        hb_parts.append(f"Total Units: {out['total_units']}")
    if out.get("start_date"):
        hb_parts.append(f"Est. Start Date – {out['start_date']}")
    if hb_parts:
        out["header_block"] = "\n".join(hb_parts)

    return out


# Split a single-line RFP / memo cover when PyMuPDF flattens the text layer.
_COVER_LINE_SPLIT_TAIL = re.compile(
    r"\s+(?=Request for Proposal(?:\s*\(?RFP\)?)?\b|Confidential\b|"
    r"(?:\d{4}\s*[\u2013\u2014\-\u2212]?\s*\d{4}(?:\s+Deployment)?)\b)",
    re.I,
)


def _split_flat_cover_title_line(flat: str) -> list[str]:
    """Split one flattened cover string into ordered title + subtitle fragments."""
    flat = re.sub(r"\s+", " ", (flat or "").strip())
    if not flat:
        return []
    parts = [p.strip() for p in _COVER_LINE_SPLIT_TAIL.split(flat) if p.strip()]
    if len(parts) <= 1:
        return [flat]
    first = parts[0]
    m = re.match(r"^(.+?\([^)]+\))\s+([A-Z].+)$", first)
    if m:
        return [m.group(1).strip(), m.group(2).strip()] + parts[1:]
    return parts


def _parse_cover_page_title_subtitles(raw_text: str) -> tuple[str, list[str]]:
    """Return ``(primary_title, subtitle_lines)`` from title-block cell text."""
    raw = (raw_text or "").strip()
    if not raw:
        return "", []
    lines = [re.sub(r"\s+", " ", ln.strip()) for ln in raw.splitlines() if ln.strip()]
    if len(lines) >= 2:
        return lines[0], lines[1:]
    flat = lines[0] if lines else raw
    segs = _split_flat_cover_title_line(flat)
    if not segs:
        return "", []
    return segs[0], segs[1:]


def _cover_page_tail_after_cell(page, cover_cell_text: str) -> str:
    """Text on the page after the title-box clip (e.g. *Confidential*, deployment)."""
    whole = _norm(page.get_text("text") or "")
    if not whole:
        return ""
    cell = _norm((cover_cell_text or "").split("\n\n")[0].strip())
    if cell:
        if whole.startswith(cell):
            return whole[len(cell) :].strip()
        idx = whole.find(cell)
        if idx >= 0:
            return whole[idx + len(cell) :].strip()
    m = re.search(r"\(\s*RFP\s*\)\s*(.+)$", whole, re.I | re.S)
    if m:
        return m.group(1).strip()
    return ""


def _split_cover_tail_segments(tail: str) -> list[str]:
    """Split trailing cover lines (*Confidential*, deployment years, …)."""
    tail = (tail or "").strip()
    if not tail:
        return []
    parts = [p.strip() for p in _COVER_LINE_SPLIT_TAIL.split(tail) if p.strip()]
    if len(parts) > 1:
        return [_polish_cover_year_line(p) for p in parts]
    m = re.match(r"^(Confidential)\s+(.+)$", tail, re.I)
    if m:
        return [
            m.group(1),
            _polish_cover_year_line(m.group(2).strip()),
        ]
    return [_polish_cover_year_line(tail)]


def _polish_cover_year_line(s: str) -> str:
    """Normalize thin/no-break separators between four-digit years (PDF quirk)."""
    if not s:
        return s
    return re.sub(
        r"(\d{4})\s*[\u202f\u00a0\u2009\u2007\u2013\u2014\-\u2212.]+\s*(\d{4})",
        r"\1–\2",
        s,
        count=1,
    )


# --- Flattened RFP body pages (Introduction + Project Scope) -----------------

_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[“\"A-Z])")


def _split_sentences_for_readability(text: str) -> list[str]:
    """Turn a flattened paragraph into short strings (sentence-ish chunks)."""
    t = _norm(text)
    if not t:
        return []
    parts = [p.strip() for p in _SENTENCE_BREAK.split(t) if p.strip()]
    return parts if len(parts) > 1 else ([t] if t else [])


def _detach_trailing_prose_from_bullets(items: list[str]) -> tuple[list[str], list[str]]:
    """When the last ``•`` chunk still contains post-list paragraphs, peel them off."""
    if not items:
        return [], []
    last = items[-1]
    for pat in (
        r"\s+(This program will\b.*)$",
        r"\s+(Partnership with\b.*)$",
        r"\s+(TSC is seeking\b.*)$",
        r"\s+(The partner\(s\) must demonstrate prior\b.*)$",
    ):
        m = re.search(pat, last, re.I | re.S)
        if m:
            fixed = last[: m.start()].rstrip()
            tail = m.group(1).strip()
            out = list(items[:-1]) + ([fixed] if fixed else [])
            extra = _split_sentences_for_readability(tail)
            return out, extra
    return items, []


def _prose_and_bullets_from_block(block: str) -> tuple[list[str], list[str], list[str]]:
    """Split ``block`` into opening sentences, ``•`` bullet lines, and closing sentences."""
    block = (block or "").strip()
    if not block:
        return [], [], []
    if "•" not in block:
        return _split_sentences_for_readability(block), [], []
    first = block.index("•")
    head = block[:first].strip()
    tail = block[first:].strip()
    raw_items = re.findall(r"•\s*([^\•]+)", tail)
    items = [x.strip() for x in raw_items if x.strip()]
    items, extra_sents = _detach_trailing_prose_from_bullets(items)
    opening = _split_sentences_for_readability(head)
    return opening, items, extra_sents


def _build_body_markdown(
    paragraphs: list[str],
    bullets: list[str],
    closing: list[str] | None = None,
) -> str:
    chunks: list[str] = []
    if paragraphs:
        chunks.append("\n\n".join(paragraphs))
    if bullets:
        chunks.append("\n".join(f"- {b}" for b in bullets))
    if closing:
        chunks.append("\n\n".join(closing))
    return "\n\n".join(chunks).strip()


_SQUARE_BULLET_RE = re.compile(r"[▪\u25aa\u25ab\u2023]")

# ``Label:`` lines that introduce the following rows (PDF ``o`` / ``▪`` siblings) are not
# peer “content bullets”; they scope the nested list. Applied as ``role: "list_intro"``.
_COLON_LIST_INTRO_MAX_CHARS = 120
_COLON_LIST_INTRO_MAX_WORDS = 18


def _is_colon_list_intro_label(text: str) -> bool:
    """Whether *text* is a list introducer (``Includes:``, ``Lift usage required for:``).

    Heuristic: trailing colon, modest length, few words before the colon, not a
    multi-clause line. Content bullets rarely end with ``:``; when they do, they
    are usually longer or sentence-like.
    """
    t = (text or "").strip()
    if not t.endswith(":"):
        return False
    core = t[:-1].strip()
    if not core or ";" in core:
        return False
    if len(t) > _COLON_LIST_INTRO_MAX_CHARS:
        return False
    n_words = len(core.split())
    if n_words > _COLON_LIST_INTRO_MAX_WORDS:
        return False
    return True


def _annotate_colon_list_intro_roles(node: dict[str, Any]) -> None:
    """Set ``role`` to ``list_intro`` on nodes whose text introduces child bullets."""
    for ch in node.get("children") or []:
        if isinstance(ch, dict):
            _annotate_colon_list_intro_roles(ch)
    ch = node.get("children") or []
    tx = (node.get("text") or "").strip()
    if ch and _is_colon_list_intro_label(tx):
        node["role"] = "list_intro"


def _split_on_square_bullets(s: str) -> dict[str, Any] | None:
    """If *s* uses small-square bullets, return ``{text, children}`` else ``None``."""
    s = (s or "").strip()
    if not s or not _SQUARE_BULLET_RE.search(s):
        return None
    parts = [p.strip() for p in _SQUARE_BULLET_RE.split(s) if p.strip()]
    if len(parts) < 2:
        return None
    head, *rest = parts
    return {
        "text": head,
        "children": [{"text": r, "children": []} for r in rest],
    }


def _split_tsc_bullet_item_fragments(raw: str) -> list[str]:
    """Split merged ``Petsense Stores (PTS) • …`` tails into separate top bullets."""
    raw = (raw or "").strip()
    if not raw:
        return []
    if "Petsense Stores (PTS)" in raw and "•" in raw:
        bits = re.split(r"\s+(?=Petsense Stores \(PTS\)\s*•\s*)", raw)
        bits = [b.strip() for b in bits if b.strip()]
        if len(bits) > 1:
            out: list[str] = []
            for b in bits:
                out.extend(_split_tsc_bullet_item_fragments(b))
            return out
    return [raw]


def _stitch_pts_heading_into_following(items: list[Any]) -> list[str]:
    """When ``…runs Petsense Stores (PTS)`` shares a line with the next ``•`` bullet, merge."""
    raw_items = [str(x) for x in items]
    out: list[str] = []
    i = 0
    while i < len(raw_items):
        s = raw_items[i].rstrip()
        if re.search(r"(?i)\s+Petsense Stores \(PTS\)\s*$", s) and i + 1 < len(raw_items):
            head = re.sub(r"(?i)\s+Petsense Stores \(PTS\)\s*$", "", s).rstrip()
            nxt = raw_items[i + 1].lstrip()
            out.append(head)
            out.append(f"Petsense Stores (PTS) {nxt}")
            i += 2
            continue
        out.append(raw_items[i])
        i += 1
    return out


def _maybe_split_pts_existing_scope(root: dict[str, Any]) -> dict[str, Any]:
    """Turn ``Petsense Stores (PTS) Existing:`` into PTS → Existing/Scope hierarchy."""
    tx = (root.get("text") or "").strip()
    m = re.match(
        r"(?i)^(Petsense Stores\s*\(PTS\))\s+(Existing|Scope):\s*$",
        tx,
    )
    if not m:
        return root
    label = m.group(2)
    lab = "Existing:" if label.lower() == "existing" else "Scope:"
    return {
        "text": m.group(1).strip(),
        "children": [{"text": lab, "children": list(root.get("children") or [])}],
    }


def _parse_single_tsc_top_bullet(s: str) -> dict[str, Any]:
    """One ``•``-segment: optional `` o `` level-2 rows and ``▪`` level-3 rows."""
    s = re.sub(r"^[•\u2022]\s*", "", (s or "").strip())
    if not s:
        return {"text": "", "children": []}
    if not re.search(r"\s+o\s+", s, re.I):
        sq = _split_on_square_bullets(s)
        return sq if sq else {"text": s, "children": []}
    parts = [p.strip() for p in re.split(r"\s+o\s+", s, flags=re.I) if p.strip()]
    head = parts[0]
    children: list[dict[str, Any]] = []
    for p in parts[1:]:
        sq = _split_on_square_bullets(p)
        if sq:
            children.append(sq)
        else:
            children.append({"text": p, "children": []})
    head_sq = _split_on_square_bullets(head)
    if head_sq:
        return {
            "text": head_sq["text"],
            "children": (head_sq.get("children") or []) + children,
        }
    return {"text": head, "children": children}


def _split_closing_quote_then_policy_sentence(t: str) -> tuple[str, str | None]:
    """Split ``…\"ready\" Any long … will be the responsibility…`` when PDF drops the callout box."""
    if len(t) < 70:
        return t, None
    best: tuple[str, str] | None = None

    def consider(head: str, tail: str) -> None:
        nonlocal best
        h, ta = head.strip(), tail.strip()
        if len(h) > 100 or len(ta) < 50:
            return
        if not re.search(
            r"(?i)\b(will be|shall be|must be|responsibility of|liable for|indemnif|warranty|disclaimer)\b",
            ta,
        ):
            return
        best = (h, ta)

    for m in re.finditer(r"(\u201d)\s+([A-Z])", t):
        consider(t[: m.end(1)], t[m.start(2) :])

    if "\u201d" not in t:
        for m in re.finditer(r'(")\s+([A-Z])', t):
            if t[: m.start(1)].count('"') % 2 == 0:
                continue
            consider(t[: m.end(1)], t[m.start(2) :])

    if best:
        return best
    return t, None


def _split_glued_callout_suffix(text: str) -> tuple[str, str | None]:
    """When PDF flattening joins a callout after a bullet phrase, return ``(main, callout)``."""
    t = (text or "").strip()
    if not t:
        return t, None

    m = re.search(
        r"(?i)(?<=\S)\s+(NOTE|WARNING|CAUTION|IMPORTANT|REMINDER|ALERT)\s*:\s+",
        t,
    )
    if m:
        head = t[: m.start()].strip()
        tail = t[m.start() :].strip()
        if head and len(tail) > 5:
            return head, tail

    m2 = re.search(r"(?<=\S)[\s\u00a0]*[\u26a0\u26a1\u2757][\uFE0F\u20E3]?\s+", t)
    if m2 and m2.start() > 4 and m2.end() < len(t) - 15:
        head = t[: m2.start()].strip()
        tail = t[m2.end() :].strip()
        if head and len(tail) > 15:
            return head, tail

    m4 = re.search(r"(?i)(?<=\bTSC)\s+(Partner\s+is\s+responsible\b)", t)
    if m4:
        head = t[: m4.start()].strip()
        tail = t[m4.start(1) :].strip()
        if head and len(tail) > 25:
            return head, tail

    return _split_closing_quote_then_policy_sentence(t)


def _peel_glued_callout_segments(text: str) -> list[str]:
    """Left-to-right: main bullet text, then zero or more callout-only strings."""
    cur = (text or "").strip()
    if not cur:
        return []
    out: list[str] = []
    while True:
        head, tail = _split_glued_callout_suffix(cur)
        if tail and head:
            out.append(head)
            cur = tail
        else:
            out.append(cur)
            break
    return out


def _expand_glued_note_callouts_in_node(node: dict[str, Any]) -> None:
    """Split glued callouts (NOTE:, ⚠, policy sentences) that were concatenated onto a bullet line."""
    ch = node.get("children")
    if not isinstance(ch, list) or not ch:
        return
    new_ch: list[dict[str, Any]] = []
    for c in ch:
        if not isinstance(c, dict):
            continue
        _expand_glued_note_callouts_in_node(c)
        tx = str(c.get("text") or "")
        segments = _peel_glued_callout_segments(tx)
        if len(segments) == 1:
            new_ch.append(c)
        else:
            c["text"] = segments[0]
            new_ch.append(c)
            for seg in segments[1:]:
                new_ch.append({"text": seg, "children": [], "role": "note_callout"})
    node["children"] = new_ch


def _expand_glued_note_callouts_in_tree(roots: list[dict[str, Any]]) -> None:
    """Split glued callouts on any depth; at root list, peeled callouts follow as siblings."""
    i = 0
    while i < len(roots):
        n = roots[i]
        if not isinstance(n, dict):
            i += 1
            continue
        _expand_glued_note_callouts_in_node(n)
        tx = str(n.get("text") or "")
        segments = _peel_glued_callout_segments(tx)
        if len(segments) > 1:
            n["text"] = segments[0]
            for j, seg in enumerate(segments[1:]):
                roots.insert(
                    i + 1 + j,
                    {"text": seg, "children": [], "role": "note_callout"},
                )
            i += len(segments)
        else:
            i += 1


def _render_bullet_tree_md(roots: list[dict[str, Any]], depth: int = 0) -> str:
    lines: list[str] = []
    pad = "  " * depth
    for node in roots:
        tx = (node.get("text") or "").strip()
        if not tx:
            continue
        ch = node.get("children") or []
        role = node.get("role")
        if role == "note_callout":
            lines.append(f"{pad}> {_md_cell(tx)}")
            continue
        if role == "list_intro" and ch:
            lines.append(f"{pad}**{_md_cell(tx)}**")
            sub = _render_bullet_tree_md(ch, depth + 1)
            if sub.strip():
                lines.append(sub)
            continue
        lines.append(f"{pad}- {_md_cell(tx)}")
        if ch:
            lines.append(_render_bullet_tree_md(ch, depth + 1))
    return "\n".join(x for x in lines if x)


def _find_petsense_pts_root_index(roots: list[dict[str, Any]]) -> int | None:
    for i, r in enumerate(roots):
        tx = (r.get("text") or "").strip()
        if re.match(r"(?i)^Petsense Stores \(PTS\)\s*$", tx):
            return i
    return None


def _take_tsc_subsection_header(paragraphs: list[str]) -> tuple[str, list[str]]:
    """TSC subsection title + remaining prose. Long run-in lines keep prose; title is canonical."""
    for i, p in enumerate(paragraphs):
        ps = str(p).strip()
        if not ps:
            continue
        if re.search(r"(?i)tractor supply", ps) and (
            re.search(r"\(tsc\)", ps) or re.search(r"\bTSC\b", ps)
        ):
            if re.match(r"(?i)^Tractor Supply Stores \(TSC\)\s*$", ps):
                return ps, [
                    str(x).strip()
                    for j, x in enumerate(paragraphs)
                    if j != i and str(x).strip()
                ]
            return "Tractor Supply Stores (TSC)", [
                str(x).strip() for x in paragraphs if str(x).strip()
            ]
    return "Tractor Supply Stores (TSC)", [
        str(x).strip() for x in paragraphs if str(x).strip()
    ]


def _maybe_promote_store_subsections(
    sec: dict[str, Any], roots: list[dict[str, Any]]
) -> list[dict[str, Any]] | None:
    """When the PDF nests PTS bullets (Scope, etc.) as peers of ``Petsense Stores (PTS)``,
    emit ``subsections`` so each bold block is a unit: header + ``bullet_tree`` only under it.
    """
    pts_i = _find_petsense_pts_root_index(roots)
    if pts_i is None:
        return None
    paras = sec.get("paragraphs")
    if not isinstance(paras, list):
        paras = []
    if not any(
        re.search(r"(?i)tractor supply", str(p))
        and (
            re.search(r"\(tsc\)", str(p), re.I) or re.search(r"\bTSC\b", str(p))
        )
        for p in paras
    ):
        return None
    pts = roots[pts_i]
    tail = roots[pts_i + 1 :]
    merged_pts_items: list[dict[str, Any]] = list(pts.get("children") or []) + tail
    tsc_block = roots[:pts_i]
    hdr, rest = _take_tsc_subsection_header([str(x) for x in paras])
    sec["paragraphs"] = rest
    out: list[dict[str, Any]] = []
    if tsc_block:
        out.append({"header": hdr, "bullet_tree": tsc_block})
    out.append({"header": "Petsense Stores (PTS)", "bullet_tree": merged_pts_items})
    return out


def _render_subsections_md(subs: list[dict[str, Any]], depth: int = 0) -> str:
    parts: list[str] = []
    for sub in subs:
        h = (sub.get("header") or "").strip()
        chunk: list[str] = []
        if h:
            if depth == 0:
                chunk.append(f"**{_md_cell(h)}**")
            else:
                level = min(2 + depth, 6)
                chunk.append(f"{'#' * level} {_md_cell(h)}")
        st = (sub.get("subtitle") or "").strip()
        if st:
            chunk.append(f"### {_md_cell(st)}")
        ps = sub.get("paragraphs")
        if isinstance(ps, list) and ps:
            para_txt = "\n\n".join(
                _md_cell(str(p)) for p in ps if str(p).strip()
            ).strip()
            if para_txt:
                chunk.append(para_txt)
        bt = sub.get("bullet_tree")
        if isinstance(bt, list) and bt:
            md = _render_bullet_tree_md(bt, 0)
            if md.strip():
                chunk.append(md)
        else:
            bullets = sub.get("bullet_items")
            if isinstance(bullets, list) and bullets:
                flat = "\n".join(
                    f"- {_md_cell(str(b))}"
                    for b in bullets
                    if str(b).strip()
                )
                if flat.strip():
                    chunk.append(flat)
        nested = sub.get("subsections")
        if isinstance(nested, list) and nested:
            sm = _render_subsections_md(nested, depth + 1)
            if sm.strip():
                chunk.append(sm)
        if chunk:
            parts.append("\n\n".join(x for x in chunk if str(x).strip()))
    return "\n\n".join(parts).strip()


def _attach_nested_bullet_tree_to_section(sec: dict[str, Any]) -> None:
    """Parse ``o`` / ``▪`` into ``bullet_tree`` or ``subsections``; drop flat ``bullet_items``.

    TSC vs PTS store blocks become ``subsections`` each with ``header`` + ``bullet_tree`` so
    PTS content (Scope, etc.) is not a sibling of the PTS header in JSON.

    ``body`` keeps remaining prose (``paragraphs`` + ``closing_paragraphs``) only.
    """
    items = sec.get("bullet_items")
    if not isinstance(items, list) or not items:
        return
    items = _stitch_pts_heading_into_following(items)
    blob = "\n".join(str(x) for x in items)
    if not re.search(r"(?:\s+o\s+|[▪\u25aa\u25ab])", blob, re.I):
        return
    roots: list[dict[str, Any]] = []
    for raw in items:
        for frag in _split_tsc_bullet_item_fragments(str(raw)):
            roots.append(_maybe_split_pts_existing_scope(_parse_single_tsc_top_bullet(frag)))
    roots = [r for r in roots if (r.get("text") or "").strip() or r.get("children")]
    if not roots:
        return
    for r in roots:
        _annotate_colon_list_intro_roles(r)
    _expand_glued_note_callouts_in_tree(roots)
    subs = _maybe_promote_store_subsections(sec, roots)
    if subs is not None:
        sec["subsections"] = subs
    else:
        sec["bullet_tree"] = roots
    sec.pop("bullet_items", None)
    paras = [str(p).strip() for p in (sec.get("paragraphs") or []) if str(p).strip()]
    head = "\n\n".join(paras).strip() if paras else ""
    closing = sec.get("closing_paragraphs")
    body_parts: list[str] = []
    if head:
        body_parts.append(head)
    if isinstance(closing, list) and closing:
        tail = "\n\n".join(str(c).strip() for c in closing if str(c).strip())
        if tail:
            body_parts.append(tail)
    sec["body"] = "\n\n".join(body_parts).strip()


def _structure_introduction_project_scope_sections(
    full_text: str,
) -> list[dict[str, Any]] | None:
    """Detect TSC-style *Introduction* / *Project Scope* on one flattened line.

    Emits two ``kind: notes`` sections with ``paragraphs`` + ``bullet_items`` for
    LLM-friendly JSON (no reliance on visible-box wrappers).
    """
    t = _norm(full_text)
    if len(t) < 400 or not re.match(r"^Introduction\b", t, re.I):
        return None
    m = re.search(r"\bProject\s+Scope\b", t, re.I)
    if not m:
        return None
    intro_block = t[: m.start()].strip()
    scope_block = t[m.end() :].strip()
    intro_block = re.sub(r"^Introduction\s+", "", intro_block, flags=re.I).strip()
    scope_block = re.sub(r"^Project\s+Scope\s+", "", scope_block, flags=re.I).strip()
    if not intro_block or not scope_block:
        return None

    out: list[dict[str, Any]] = []
    for title, body, y0 in (
        ("Introduction", intro_block, 10.0),
        ("Project Scope", scope_block, 500.0),
    ):
        paras, bullets, closing = _prose_and_bullets_from_block(body)
        if not paras and not bullets:
            continue
        sec: dict[str, Any] = {
            "kind": "notes",
            "title": title,
            "paragraphs": paras,
            "bullet_items": bullets,
            "body": _build_body_markdown(paras, bullets, closing or None),
        }
        if closing:
            sec["closing_paragraphs"] = closing
        sec["_presentation_sort"] = _section_presentation_sort(
            title, "notes", None, (0.0, y0, 1.0, y0 + 1.0)
        )
        _attach_nested_bullet_tree_to_section(sec)
        out.append(sec)
    return out or None


# --- TSC RFP pages after Introduction (continuation + follow-on headings) -----
# Canonical title lists live in ``headings.builtin``; merge per-overlay JSON via
# ``heading_template`` (see ``headings.template`` + ``document.heading_analysis``).

from orbitbrief_page_os.segmentation.headings.builtin import (
    TSC_FOLLOWON_HEADINGS as _TSC_FOLLOWON_HEADINGS,
    TSC_MAJOR_BAND_SECTION_TITLES as _TSC_MAJOR_BAND_SECTION_TITLES,
)

# Minor PDF headings (small bold) that continue a *major* block started on the prior page.
# Keys must match ``title`` from split text; values = required ``title`` of last section on page N-1.
_TSC_MINOR_SUBTITLE_CONTINUES_MAJOR: dict[str, str] = {
    "Operational Expectations": "Warehousing & Inventory Management",
}

# ``_TSC_FOLLOWON_HEADINGS`` also lists strings that are *major* splitters, but a few
# are reused visually as small-bold lines under another major (same PDF page).
_SUBTITLE_OK_ALSO_MAJOR_HEADING: frozenset[str] = frozenset({"Inventory Management"})

# Same-page major (blue) followed by bold sub-heads split as separate ``notes`` — fold
# them under one section with ``subsections`` (LLM + MD hierarchy).
_TSC_SAME_PAGE_MAJOR_CHILD_TITLES: dict[str, tuple[str, ...]] = {
    "Kitting & Asset Management": (
        "Kitting Requirements",
        "Labeling & Asset Tracking",
        "Quality Control",
        "Shipping & Logistics",
        "Shipping Requirements",
        "Coordination",
    ),
}

# Within a merged ``subsections`` list, some bold lines are *bands* (large/blue in PDF)
# and the following known headers belong *under* that band, not as siblings of the band.
_TSC_INTERMEDIATE_BAND_ABSORBS: dict[str, tuple[str, ...]] = {
    "Shipping & Logistics": ("Shipping Requirements",),
}

_SUBTITLE_SMALL_WORDS = frozenset(
    "a an and as at but by for from in into la le of on or per se the to via vs".split()
)


def _looks_like_run_in_subtitle_line(text: str) -> bool:
    """Short Title-Case line that behaves like a PDF sub-head before bullets."""
    t = (text or "").strip()
    if not t or len(t) > 56:
        return False
    if "\n" in t or "\r" in t:
        return False
    words = t.split()
    if len(words) < 2 or len(words) > 10:
        return False
    if t.count(".") > 1:
        return False
    if t.endswith(".") and len(words) > 4:
        return False
    if "(" in t and ")" in t and len(t) > 40:
        return False
    if any(w and w[0].isdigit() for w in words):
        return False
    for w in words:
        core = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", w)
        if not core:
            return False
        low = core.lower()
        if low in _SUBTITLE_SMALL_WORDS:
            continue
        if core[0].islower():
            return False
    return True


def _is_tsc_prose_colon_intro_before_bullets(text: str) -> bool:
    """True for ``…as follows:`` / short colon lines that introduce the bullet block."""
    t = (text or "").strip()
    if not t.endswith(":"):
        return False
    if len(t) > 220:
        return False
    if _is_colon_list_intro_label(t):
        return True
    tl = t.lower()
    if tl.endswith("as follows:") or tl.endswith("conducted as follows:"):
        return True
    return False


def _maybe_hoist_trailing_colon_intro_into_bullet_tree(sec: dict[str, Any]) -> None:
    """When the last ``paragraphs`` line introduces the following ``bullet_tree`` (colon
    block before ``•`` rows in the PDF), wrap the tree under that line as ``list_intro``.
    """
    if str(sec.get("kind")) != "notes":
        return
    paras = sec.get("paragraphs")
    tree = sec.get("bullet_tree")
    if not isinstance(paras, list) or len(paras) < 1:
        return
    if not isinstance(tree, list) or not tree:
        return
    last = str(paras[-1]).strip()
    if not _is_tsc_prose_colon_intro_before_bullets(last):
        return
    head_paras = [str(p).strip() for p in paras[:-1] if str(p).strip()]
    sec["paragraphs"] = head_paras
    intro = {"text": last, "children": list(tree), "role": "list_intro"}
    sec["bullet_tree"] = [intro]
    _annotate_colon_list_intro_roles(intro)
    _recompute_sec_body_after_paragraph_change(sec)


def _recompute_sec_body_after_paragraph_change(sec: dict[str, Any]) -> None:
    """Refresh ``body`` after ``paragraphs`` / subtitle edits."""
    paras = [str(p).strip() for p in (sec.get("paragraphs") or []) if str(p).strip()]
    closing = sec.get("closing_paragraphs")
    if sec.get("bullet_tree") or sec.get("subsections"):
        head = "\n\n".join(paras).strip() if paras else ""
        parts: list[str] = []
        if head:
            parts.append(head)
        if isinstance(closing, list) and closing:
            tail = "\n\n".join(str(c).strip() for c in closing if str(c).strip())
            if tail:
                parts.append(tail)
        sec["body"] = "\n\n".join(parts).strip()
    else:
        bullets = [str(b) for b in (sec.get("bullet_items") or []) if str(b).strip()]
        sec["body"] = _build_body_markdown(
            paras,
            bullets,
            closing if isinstance(closing, list) else None,
        )


def _maybe_hoist_trailing_paragraph_as_subtitle(sec: dict[str, Any]) -> None:
    """Move a trailing ``paragraphs[-1]`` line into ``subtitle`` when it reads like a
    small-bold sub-head (same page as the section ``title`` major block).
    """
    if str(sec.get("kind")) != "notes":
        return
    if sec.get("subtitle") or sec.get("parent_major_section"):
        return
    paras = sec.get("paragraphs")
    if not isinstance(paras, list) or len(paras) < 2:
        return
    has_list = bool(
        sec.get("bullet_tree")
        or sec.get("subsections")
        or (
            isinstance(sec.get("bullet_items"), list)
            and any(str(x).strip() for x in sec["bullet_items"])
        )
    )
    if not has_list:
        return
    cand = str(paras[-1]).strip()
    if not _looks_like_run_in_subtitle_line(cand):
        return
    title = str(sec.get("title") or "").strip()
    if cand.casefold() == title.casefold():
        return
    if cand in _TSC_FOLLOWON_HEADINGS and cand not in _SUBTITLE_OK_ALSO_MAJOR_HEADING:
        return
    sec["subtitle"] = cand
    sec["paragraphs"] = [str(p).strip() for p in paras[:-1] if str(p).strip()]
    _recompute_sec_body_after_paragraph_change(sec)


def _hoist_run_in_subtitles_all_sections(sections: list[dict[str, Any]]) -> None:
    for sec in sections:
        _maybe_hoist_trailing_paragraph_as_subtitle(sec)


def _subsection_dict_from_absorbed_notes(sec: dict[str, Any]) -> dict[str, Any]:
    """Strip an absorbed ``notes`` section down to a ``subsections`` entry."""
    title = str(sec.get("title") or "").strip()
    out: dict[str, Any] = {"header": title}
    for k in (
        "subtitle",
        "paragraphs",
        "bullet_tree",
        "bullet_items",
        "closing_paragraphs",
    ):
        v = sec.get(k)
        if v is None or v == [] or v == "":
            continue
        out[k] = v
    return out


def _nest_tsc_subsections_under_intermediate_bands(
    subs: list[dict[str, Any]],
) -> None:
    """Fold configured ``(band header) → (child headers…)`` chains into nested ``subsections``.

    Extends ``_TSC_INTERMEDIATE_BAND_ABSORBS`` for other PDFs where a mid-page band owns
    the next one or more bold sub-heads without changing the top-level major merge.
    """
    if not subs:
        return
    absorb = _TSC_INTERMEDIATE_BAND_ABSORBS
    out: list[dict[str, Any]] = []
    i = 0
    while i < len(subs):
        cur = subs[i]
        h = str(cur.get("header") or "").strip()
        want = absorb.get(h)
        if not want:
            out.append(cur)
            i += 1
            continue
        gathered: list[dict[str, Any]] = []
        j = i + 1
        for w in want:
            if j >= len(subs):
                break
            if str(subs[j].get("header") or "").strip() != w:
                break
            gathered.append(subs[j])
            j += 1
        if gathered:
            cur["subsections"] = gathered
            cur["hierarchy"] = "intermediate_band_with_nested_subheads"
            out.append(cur)
            i = j
        else:
            out.append(cur)
            i += 1
    subs[:] = out
    for s in subs:
        inner = s.get("subsections")
        if isinstance(inner, list) and inner:
            _nest_tsc_subsections_under_intermediate_bands(inner)


def _merge_tsc_same_page_major_with_child_sections(
    sections: list[dict[str, Any]],
) -> None:
    """Fold consecutive child ``notes`` under one major (PDF blue band + bold sub-heads)."""
    for major, expected in _TSC_SAME_PAGE_MAJOR_CHILD_TITLES.items():
        i = None
        for k, s in enumerate(sections):
            if s.get("kind") == "notes" and str(s.get("title") or "") == major:
                i = k
                break
        if i is None:
            continue
        major_sec = sections[i]
        if major_sec.get("bullet_tree") or major_sec.get("subsections"):
            continue
        if major_sec.get("parent_major_section"):
            continue
        j = i + 1
        gathered: list[dict[str, Any]] = []
        for want in expected:
            if j >= len(sections):
                break
            s = sections[j]
            if s.get("kind") != "notes" or str(s.get("title") or "") != want:
                break
            gathered.append(s)
            j += 1
        if not gathered:
            continue
        major_sec["subsections"] = [
            _subsection_dict_from_absorbed_notes(x) for x in gathered
        ]
        _nest_tsc_subsections_under_intermediate_bands(major_sec["subsections"])
        major_sec["hierarchy"] = "major_with_same_page_subsections"
        major_sec.pop("bullet_items", None)
        paras = [
            str(p).strip()
            for p in (major_sec.get("paragraphs") or [])
            if str(p).strip()
        ]
        major_sec["body"] = "\n\n".join(paras).strip() if paras else ""
        del sections[i + 1 : i + 1 + len(gathered)]


def _slug_section_id(title: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", (title or "").lower()).strip("-")
    return s or "section"


def _collect_tsc_followon_heading_hits(
    t: str,
    headings: tuple[str, ...] | None = None,
) -> list[tuple[int, str]]:
    """Return ``(start, heading)`` pairs left-to-right, dropping nested matches."""
    t = _norm(t)
    hset = headings if headings is not None else _TSC_FOLLOWON_HEADINGS
    candidates = [(t.find(h), h) for h in hset if t.find(h) >= 0]
    if not candidates:
        return []
    candidates.sort(key=lambda x: (x[0], -len(x[1])))
    picked: list[tuple[int, str]] = []
    for j, h in candidates:
        end = j + len(h)
        overlapped = False
        for pj, ph in picked:
            p_end = pj + len(ph)
            if j < p_end and end > pj:
                overlapped = True
                break
        if overlapped:
            continue
        picked.append((j, h))
    picked.sort(key=lambda x: x[0])
    return picked


def _first_tsc_followon_heading(
    text: str,
    headings: tuple[str, ...] | None = None,
) -> tuple[int, str] | None:
    hits = _collect_tsc_followon_heading_hits(text, headings)
    return hits[0] if hits else None


def _prior_page_last_structured_section_title(doc: Any, page_index: int) -> str | None:
    """Best-effort title of the last notes-like section on *page_index - 1*."""
    if page_index <= 0:
        return None
    try:
        prev = _norm(doc[page_index - 1].get_text("text") or "")
    except Exception:
        return None
    if not prev:
        return None
    secs = _structure_introduction_project_scope_sections(prev)
    if secs:
        return str(secs[-1].get("title") or "") or None
    fol = _structure_tsc_followon_sections(prev)
    if fol:
        return str(fol[-1].get("title") or "") or None
    return None


def _structure_tsc_followon_sections(
    remainder: str,
    headings: tuple[str, ...] | None = None,
    major_band: frozenset[str] | None = None,
) -> list[dict[str, Any]]:
    """Split *remainder* at known TSC headings into ``kind: notes`` sections."""
    t = (remainder or "").strip()
    if not t:
        return []
    hset = headings if headings is not None else _TSC_FOLLOWON_HEADINGS
    mset = major_band if major_band is not None else _TSC_MAJOR_BAND_SECTION_TITLES
    hits = _collect_tsc_followon_heading_hits(t, hset)
    if not hits:
        return []
    out: list[dict[str, Any]] = []
    for i, (j, h) in enumerate(hits):
        end = hits[i + 1][0] if i + 1 < len(hits) else len(t)
        chunk = t[j:end].strip()
        body = chunk[len(h) :].strip() if chunk.startswith(h) else chunk
        paras, bullets, closing = _prose_and_bullets_from_block(body)
        if not paras and not bullets and not closing:
            continue
        y0 = float(j)
        sec: dict[str, Any] = {
            "kind": "notes",
            "title": h,
            "paragraphs": paras,
            "bullet_items": bullets,
            "body": _build_body_markdown(paras, bullets, closing or None),
        }
        if closing:
            sec["closing_paragraphs"] = closing
        sec["_presentation_sort"] = _section_presentation_sort(
            h, "notes", None, (0.0, y0, 1.0, y0 + 1.0)
        )
        _attach_nested_bullet_tree_to_section(sec)
        if h in mset:
            sec["heading_tier"] = "major"
        _maybe_hoist_trailing_colon_intro_into_bullet_tree(sec)
        out.append(sec)
    return out


def _annotate_tsc_subtitle_under_parent_major(
    sections: list[dict[str, Any]],
    doc: Any,
    page_index: int,
    pdf_path: str,
) -> None:
    """When the first section title is a known *minor* subtitle that belongs under the
    prior page's last *major* heading (e.g. Operational Expectations under Warehousing),
    set ``parent_major_section`` + ``continues_from`` so JSON/Markdown reflect hierarchy.
    """
    if page_index <= 0 or not sections:
        return
    first = sections[0]
    if str(first.get("kind")) != "notes":
        return
    title = str(first.get("title") or "").strip()
    parent_required = _TSC_MINOR_SUBTITLE_CONTINUES_MAJOR.get(title)
    if not parent_required:
        return
    prior_last = _prior_page_last_structured_section_title(doc, page_index)
    if not prior_last or prior_last != parent_required:
        return
    sl = _slug_section_id(parent_required)
    first["parent_major_section"] = parent_required
    first["parent_major_slug"] = sl
    first["hierarchy"] = "subtitle_continuation_under_major"
    first["continues_from"] = {
        "page_index": page_index - 1,
        "section_title": parent_required,
        "section_slug": sl,
        "relation": "logical_subtitle_under_prior_page_major",
        "subtitle": title,
        "source_pdf": pdf_path,
        "merge_key": f"p{page_index - 1}:{sl}",
        "reader_note": (
            f"This heading continues the «{parent_required}» block from the prior page "
            "(minor heading under the same major section as in the PDF)."
        ),
    }


def _prefix_is_tsc_hollow_bullet_continuation(
    prefix: str,
    followon_headings: tuple[str, ...] | None = None,
) -> bool:
    """True when *prefix* is a leading `` o `` run (page break mid nested list), no ``•``."""
    p = _norm(prefix or "")
    if len(p) < 8:
        return False
    if "•" in p or "\u2022" in p:
        return False
    if not re.match(r"(?i)^o\s+\S", p):
        return False
    low = p.lower()
    fh = followon_headings if followon_headings is not None else _TSC_FOLLOWON_HEADINGS
    for h in fh:
        hl = h.strip().lower()
        if len(hl) >= 12 and low.startswith(hl):
            return False
    return True


def _parse_tsc_o_only_bullet_run(prefix: str) -> list[dict[str, Any]]:
    """Parse ``o A o B`` (no top ``•`` row) into flat ``bullet_tree`` nodes."""
    p = _norm(prefix)
    p = re.sub(r"(?i)^o\s+", "", p).strip()
    if not p:
        return []
    parts = re.split(r"\s+o\s+", p, flags=re.I)
    return [{"text": x.strip(), "children": []} for x in parts if x.strip()]


def _maybe_tsc_continuation_and_followon_sections(
    full_text: str,
    doc: Any,
    page_index: int,
    pdf_path: str,
    *,
    followon_headings: tuple[str, ...] | None = None,
    major_band: frozenset[str] | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """When a page does not start with *Introduction*, peel prose before the first
    known follow-on heading and attach it logically to the prior page's last section.

    Returns ``(continuation_document_meta_or_None, new_sections)``.
    """
    if page_index <= 0:
        return None, []
    t = _norm(full_text)
    if not t or re.match(r"^Introduction\b", t, re.I):
        return None, []
    hit = _first_tsc_followon_heading(t, followon_headings)
    if not hit:
        return None, []
    j, heading = hit
    if j == 0:
        follow = _structure_tsc_followon_sections(t, followon_headings, major_band)
        return None, follow
    prefix = t[:j].strip()
    remainder = t[j:].strip()
    prior_title = _prior_page_last_structured_section_title(doc, page_index)
    attach_title = prior_title or "Project Scope"

    if _prefix_is_tsc_hollow_bullet_continuation(prefix, followon_headings):
        tree = _parse_tsc_o_only_bullet_run(prefix)
        _expand_glued_note_callouts_in_tree(tree)
        body = _render_bullet_tree_md(tree, 0).strip()
        cont_sec: dict[str, Any] = {
            "kind": "notes_continuation",
            "title": None,
            "paragraphs": [],
            "bullet_tree": tree,
            "body": body,
            "continues_from": {
                "page_index": page_index - 1,
                "section_title": attach_title,
                "section_slug": _slug_section_id(attach_title),
                "relation": "tsc_hollow_bullet_continuation",
                "source_pdf": pdf_path,
                "merge_key": f"p{page_index - 1}:{_slug_section_id(attach_title)}",
                "reader_note": (
                    f"Hollow-circle (`` o ``) lines continue a nested list from section "
                    f"«{attach_title}» on the prior PDF page (typically children of a "
                    "colon intro such as “Align delivery timing with:”); merge these "
                    "nodes under that list intro when assembling the full document."
                ),
            },
            "_presentation_sort": _section_presentation_sort(
                "Continuation", "notes_continuation", None, (0.0, -1.0, 1.0, 0.0)
            ),
        }
        meta = {
            "prior_page_index": page_index - 1,
            "attach_to_section_title": attach_title,
            "attach_to_section_slug": _slug_section_id(attach_title),
            "continuation_bullet_nodes": len(tree),
            "next_heading_on_this_page": heading,
            "source_pdf": pdf_path,
            "merge_key": f"p{page_index - 1}:{_slug_section_id(attach_title)}",
        }
        follow = _structure_tsc_followon_sections(
            remainder, followon_headings, major_band
        )
        return meta, [cont_sec] + follow

    if len(prefix) < 12:
        follow = _structure_tsc_followon_sections(t, followon_headings, major_band)
        return None, follow
    paras = _split_sentences_for_readability(prefix)
    cont_sec = {
        "kind": "notes_continuation",
        "title": None,
        "paragraphs": paras,
        "bullet_items": [],
        "body": "\n\n".join(paras).strip(),
        "continues_from": {
            "page_index": page_index - 1,
            "section_title": attach_title,
            "section_slug": _slug_section_id(attach_title),
            "relation": "same_logical_section",
            "source_pdf": pdf_path,
            "merge_key": f"p{page_index - 1}:{_slug_section_id(attach_title)}",
            "reader_note": (
                "These sentences continue the prior page before the next heading; "
                "merge with that section when assembling a full document."
            ),
        },
        "_presentation_sort": _section_presentation_sort(
            "Continuation", "notes_continuation", None, (0.0, -1.0, 1.0, 0.0)
        ),
    }
    meta = {
        "prior_page_index": page_index - 1,
        "attach_to_section_title": attach_title,
        "attach_to_section_slug": _slug_section_id(attach_title),
        "sentence_count": len(paras),
        "next_heading_on_this_page": heading,
        "source_pdf": pdf_path,
        "merge_key": f"p{page_index - 1}:{_slug_section_id(attach_title)}",
    }
    follow = _structure_tsc_followon_sections(remainder, followon_headings, major_band)
    return meta, [cont_sec] + follow


def _extract_tbstruct_cover_text(
    page,
    boxes: list[dict],
    scale: float,
    *,
    rotated_cw: bool,
    page_height_pt: float,
) -> str:
    """Text inside synthetic ``tbstruct_*`` / ``tbtext_*`` cells (cover / margin).

    When the detector finds no ``v\\d+`` body wrappers, these cells still carry
    the real cover copy; ``page.get_text('text')`` is often one flattened line on
    rotated pages, so we clip the cell and use word-rebuilt lines.
    """
    tb_parents = [
        b
        for b in boxes
        if re.fullmatch(r"tbstruct_\d+", str(b.get("box_id") or ""))
        and b.get("color") == "BLUE"
    ]
    if not tb_parents:
        return ""
    parent_ids = {str(b.get("box_id")) for b in tb_parents}
    parts: list[str] = []
    for b in sorted(boxes, key=lambda x: (x.get("px_bbox") or [0, 0, 0, 0])[1]):
        bid = str(b.get("box_id") or "")
        pid = str(b.get("parent_box_id") or "")
        if (
            bid.startswith("tbtext_")
            and pid in parent_ids
            and b.get("color") == "ORANGE"
        ):
            try:
                r = _pdf_rect(
                    b["px_bbox"],
                    scale,
                    pad=1.0,
                    rotated_cw=rotated_cw,
                    page_height_pt=page_height_pt,
                )
                flat = (_text_in_rect(page, r) or "").strip()
                if flat:
                    t = flat
                else:
                    t = _text_in_rect_preserve_lines(
                        page,
                        r,
                        rotated_cw=rotated_cw,
                        page_height_pt=page_height_pt,
                    ).strip()
                if t:
                    parts.append(t)
            except Exception:
                pass
    if parts:
        return "\n\n".join(parts)
    for w in sorted(tb_parents, key=lambda x: x["px_bbox"][1]):
        try:
            r = _pdf_rect(
                w["px_bbox"],
                scale,
                pad=1.0,
                rotated_cw=rotated_cw,
                page_height_pt=page_height_pt,
            )
            flat = (_text_in_rect(page, r) or "").strip()
            if flat:
                t = flat
            else:
                t = _text_in_rect_preserve_lines(
                    page,
                    r,
                    rotated_cw=rotated_cw,
                    page_height_pt=page_height_pt,
                ).strip()
            if t:
                parts.append(t)
        except Exception:
            pass
    return "\n\n".join(parts) if parts else ""


def _split_title_line_from_margin(margin_text: str) -> tuple[str, str | None]:
    """Strip architectural ``TITLE:`` (single- or multi-line) from a right title-block
    strip. The value belongs in ``document.sheet_title`` — a *label for the whole
    sheet* (e.g. \"General notes, code summary, …\"), not a narrative \"General
    notes\" section. Returns (cleaned margin text, title or None)."""
    lines = (margin_text or "").splitlines(keepends=False)
    n = len(lines)
    if not n:
        return (margin_text or ""), None
    new_lines: list[str] = []
    title: str | None = None
    i = 0
    while i < n:
        raw = lines[i] or ""
        m1 = re.match(r"^TITLE:\s*(.*)$", raw, re.I)
        if not m1:
            new_lines.append(raw)
            i += 1
            continue
        sub = (m1.group(1) or "").strip()
        if sub:
            title = sub
            i += 1
            continue
        i += 1
        buf: list[str] = []
        while i < n:
            seg = (lines[i] or "").strip()
            if not seg:
                if buf:
                    break
                i += 1
                continue
            if _SHEET_NUM_RE.fullmatch(seg) or re.match(
                r"^G?\d{2,4}\s*$", seg
            ):
                break
            if re.match(
                r"^(?:SCALE|PROJECT|DATE|DRAWN|N\.?T\.S|REVISIONS|PROJECT|CHECKED)\b",
                (lines[i] or ""),
                re.I,
            ) and not buf:
                break
            buf.append(seg)
            i += 1
        if buf:
            title = re.sub(r"\s+", " ", " ".join(buf))
    return "\n".join(new_lines).strip(), (title.strip() if (title and title.strip()) else None) or None


def _drop_duplicate_matrix_fingerprints(
    sections: list[dict[str, Any]],
) -> None:
    """Remove later copies of the same ``kind: matrix`` block (same title,
    columns, grid). Stacked BLUE wrappers sometimes re-emit the same checklist."""
    seen: set[tuple[Any, ...]] = set()
    i = 0
    while i < len(sections):
        s = sections[i]
        if s.get("kind") == "matrix":
            cols = tuple(s.get("columns") or ())
            grid = tuple(tuple(r) for r in (s.get("grid") or []))
            fp = (s.get("title"), cols, grid)
            if fp in seen:
                sections.pop(i)
                continue
            seen.add(fp)
        i += 1


# ---------------------------------------------------------------------------
# Rule engine — wrapper → section
# ---------------------------------------------------------------------------


def _is_title_block_wrapper(w: dict, purple: list[dict]) -> bool:
    """Any wrapper that overlaps (or contains) a PURPLE titleblk box is part of
    the title-block sidebar. Those are rendered as document metadata, not body."""
    wb = w["px_bbox"]
    for p in purple:
        if _bbox_overlaps(wb, p["px_bbox"]):
            return True
    return False


def _is_full_page(w: dict, page_w: int, page_h: int) -> bool:
    x0, y0, x1, y1 = w["px_bbox"]
    return (x1 - x0) >= 0.85 * page_w and (y1 - y0) >= 0.80 * page_h


def _sub_section_slices(
    wrapper: dict,
    sec_titles: list[dict],
    all_boxes: list[dict] | None = None,
) -> list[tuple[int, int, dict | None]]:
    """Return (y_top, y_bot, title_box) slices for a wrapper.

    If the wrapper has ``v{N}_secM_title`` children, each title starts a slice;
    the slice extends to the next title (or the wrapper's bottom).
    For ``mccol_N_group``, the child ``mccol_N_title`` may sit *above* the
    group bbox; we still attach it and expand the slice y_top upward.
    Otherwise returns a single slice covering the full wrapper.
    """
    wx0, wy0, wx1, wy1 = wrapper["px_bbox"]
    wid = str(wrapper["box_id"])

    m_mcc = re.match(r"^mccol_(\d+)_group$", wid)
    if m_mcc and all_boxes is not None:
        tid = f"mccol_{m_mcc.group(1)}_title"
        title_cands = [
            t
            for t in all_boxes
            if str(t.get("box_id")) == tid and str(t.get("parent_box_id")) == wid
        ]
        if title_cands:
            t = min(title_cands, key=lambda x: x["px_bbox"][1])
            ty0 = t["px_bbox"][1]
            y_top = min(wy0, ty0)
            y_bot = wy1
            return [(y_top, y_bot, t)]

    mine = [
        t for t in sec_titles
        if re.fullmatch(rf"{re.escape(wid)}_sec\d+_title", t["box_id"])
        and _bbox_contains(wrapper["px_bbox"], t["px_bbox"])
    ]
    mine.sort(key=lambda t: t["px_bbox"][1])
    if not mine:
        return [(wy0, wy1, None)]
    slices: list[tuple[int, int, dict | None]] = []
    for i, t in enumerate(mine):
        y_top = t["px_bbox"][1]
        y_bot = mine[i + 1]["px_bbox"][1] if i + 1 < len(mine) else wy1
        slices.append((y_top, y_bot, t))
    # If the first subsection starts well below the wrapper top, treat the
    # gap as a prefix slice (rare: v has content above its first sec_title).
    if slices and slices[0][0] - wy0 >= 20:
        slices.insert(0, (wy0, slices[0][0], None))
    return slices


def _words_in_rect(page, rect, *, rotated_cw: bool = False, page_height_pt: float | None = None):
    try:
        raw = page.get_text("words", clip=rect) or []
    except Exception:
        raw = []
    # PyMuPDF words: (x0, y0, x1, y1, "text", block, line, wordno)
    raw_words = [
        {"x0": w[0], "y0": w[1], "x1": w[2], "y1": w[3], "text": w[4]}
        for w in raw
        if w[4].strip()
    ]
    if rotated_cw and page_height_pt is not None:
        return [_derotate_word(w, page_height_pt) for w in raw_words]
    return raw_words


def _text_in_rect(page, rect) -> str:
    try:
        return _norm(page.get_text("text", clip=rect) or "")
    except Exception:
        return ""


def _text_in_rect_preserve_lines(page, rect, *, rotated_cw: bool = False, page_height_pt: float | None = None) -> str:
    """For rotated pages, reconstruct multi-line text from words.

    PyMuPDF's ``get_text('text', clip=...)`` returns glyphs in logical-line
    order based on the native page orientation, which is wrong when the
    content was rendered in a rotated raster (landscape sheet on a portrait
    page). We therefore rebuild the text from words, clustering by the
    detector's Y (which is PyMuPDF's X after the inverse rotation).
    """
    if rotated_cw and page_height_pt is not None:
        words = _words_in_rect(page, rect, rotated_cw=True, page_height_pt=page_height_pt)
        if not words:
            return ""
        rows = _cluster_rows(words, gap=2.0)
        return "\n".join(
            " ".join(w["text"] for w in sorted(row, key=lambda w: w["x0"]))
            for row in rows
        )
    try:
        raw = page.get_text("text", clip=rect) or ""
    except Exception:
        return ""
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in raw.splitlines()]
    return "\n".join([ln for ln in lines if ln])


def _notes_body_reading_order(
    page,
    sect_rect,
    title_text: str,
    *,
    rotated_cw: bool,
    page_height_pt: float,
) -> str:
    """Notes body: for *Responsibility matrix*, prefer top-to-bottom, left-to-right
    text (``sort=True``) so ``CABLING|SECURITY|ELECTRICAL`` headers are present
    for structured parsing. Otherwise use preserve-lines extraction."""
    tu = (title_text or "").upper()
    if "RESPONSIBILITY" in tu and "MATRIX" in tu:
        # Do **not** use sort=True: it orders by (y, x) and interleaves the three
        # columns line-by-line, which breaks CABLING | SECURITY | ELECTRICAL
        # structure. Unsorted text follows draw order, which usually matches
        # column-then-below reading for this block (same as full page text).
        try:
            t = page.get_text("text", clip=sect_rect) or ""
        except (TypeError, Exception):
            t = ""
        if t.strip():
            return t
    return _text_in_rect_preserve_lines(
        page, sect_rect,
        rotated_cw=rotated_cw, page_height_pt=page_height_pt,
    )


def _cluster_rows(words: list[dict], gap: float = 3.0) -> list[list[dict]]:
    if not words:
        return []
    ws = sorted(words, key=lambda w: (w["y0"], w["x0"]))
    rows: list[list[dict]] = [[ws[0]]]
    y_key = (ws[0]["y0"] + ws[0]["y1"]) / 2
    row_h = ws[0]["y1"] - ws[0]["y0"]
    for w in ws[1:]:
        cy = (w["y0"] + w["y1"]) / 2
        if abs(cy - y_key) <= max(row_h * 0.7, gap):
            rows[-1].append(w)
            y_key = sum((x["y0"] + x["y1"]) / 2 for x in rows[-1]) / len(rows[-1])
            row_h = max(row_h, w["y1"] - w["y0"])
        else:
            rows.append([w])
            y_key = cy
            row_h = w["y1"] - w["y0"]
    return rows


def _assign_to_column(word: dict, col_x_spans: list[tuple[float, float]]) -> int:
    cx = (word["x0"] + word["x1"]) / 2
    best = 0
    best_d = 1e9
    for j, (x0, x1) in enumerate(col_x_spans):
        if x0 - 1 <= cx <= x1 + 1:
            return j
        d = min(abs(cx - x0), abs(cx - x1))
        if d < best_d:
            best_d = d
            best = j
    return best


_VARIANT_WORDS = {1: "single", 2: "double", 3: "triple", 4: "quadruple"}


def _variant_word(n: int) -> str:
    """Human-friendly label for how many sub-rows a table record carries.

    1 → ``single``, 2 → ``double``, 3 → ``triple``, 4 → ``quadruple``,
    everything else → ``N-way``. Included in the JSON so an LLM can
    categorise a row without counting variants itself.
    """
    if n in _VARIANT_WORDS:
        return _VARIANT_WORDS[n]
    return f"{n}-way"


def _nonempty_cell_strings(cells: list) -> set[str]:
    return {(c or "").strip() for c in (cells or []) if (c or "").strip()}


def _drop_y_band_fragment_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove bands that are almost certainly a y-clustering split of a row
    above: no symbol, no description, but the same data cells as a *recent* row
    (including a **sub-variant** of a double multi-line record — compare against
    **every** variant, not just the first)."""
    out: list[dict[str, Any]] = []
    for r in rows:
        st = (r.get("symbol_text") or "").strip()
        vlist = r.get("variants") or []
        v0 = (vlist[0] if vlist else []) or []
        desc = (v0[0] or "").strip() if v0 else ""
        if st or desc or not v0 or not out:
            out.append(r)
            continue
        fs = _nonempty_cell_strings(v0)
        is_frag = False
        if len(fs) < 1:
            out.append(r)
            continue
        # F 2 + TV combo rows: y-banding often splits the *fiber* stack out as a
        # separate band (no symbol / no description) while the multi-variant row
        # above only captured the copper half — drop the dangling fiber line.
        tail = " ".join((c or "") for c in v0)
        if "2 STRAND" in tail and "FIBER" in tail and "LC DUPLEX" in tail:
            last_sym = (out[-1].get("symbol_text") or "").upper()
            if "F 2" in last_sym and "TV" in last_sym:
                continue
        # (1) Same-by-column against any variant of recent rows
        for prev in out[-4:]:
            for pv in (prev.get("variants") or [[]]):
                if not isinstance(pv, list) or len(pv) < 2 or len(v0) < 2:
                    continue
                nmatch = 0
                for a, b in zip(v0[1:12], pv[1:12]):
                    ta, tb = (a or "").strip(), (b or "").strip()
                    if ta and ta == tb:
                        nmatch += 1
                if nmatch >= 3:
                    is_frag = True
                    break
                # (2) Sub-rows: ≥3 shared cell strings, with at least one *specific*
                # (long) string — avoids false positives on ``N/A`` / ``1`` stacks.
                ps = _nonempty_cell_strings(pv)
                inter = fs & ps
                if len(inter) >= 3 and any(len(s) > 14 for s in inter):
                    is_frag = True
                    break
            if is_frag:
                break
        if is_frag:
            continue
        out.append(r)
    return out


def _dedupe_cyan_spans(spans_px: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge cyan x-spans that heavily overlap (detector sometimes emits
    duplicate rings for the same column)."""
    if not spans_px:
        return []
    spans = sorted(spans_px)
    merged: list[tuple[float, float]] = [spans[0]]
    for x0, x1 in spans[1:]:
        mx0, mx1 = merged[-1]
        overlap = max(0.0, min(x1, mx1) - max(x0, mx0))
        min_w = max(1.0, min(x1 - x0, mx1 - mx0))
        if overlap >= 0.65 * min_w:
            merged[-1] = (min(x0, mx0), max(x1, mx1))
        else:
            merged.append((x0, x1))
    return merged


def _build_table_from_cells(
    page,
    section_px_bbox,
    cyan_boxes: list[dict],
    orange_cells: list[dict],
    scale: float,
    *,
    rotated_cw: bool,
    page_height_pt: float,
    data_y_top_px: float | None = None,
) -> dict[str, Any] | None:
    """Rule-driven table build:

    * **CYAN** defines the column schema (x-spans + header labels).
    * **ORANGE** cells ARE the data rows — one cell per (row, column) slot.

    Row detection uses orange-cell y-band clustering (not word clusters),
    which yields one row per physical orange row even when a cell spans
    multiple lines of text.

    Each returned row carries a reference to its *first-column* orange cell
    (used later to crop the symbol image).
    """
    if not cyan_boxes or not orange_cells:
        return None

    cy_px_spans = sorted([(c["px_bbox"][0], c["px_bbox"][2]) for c in cyan_boxes])
    cy_px_spans = _dedupe_cyan_spans(cy_px_spans)
    col_x_spans_pt = [(x0 / scale, x1 / scale) for (x0, x1) in cy_px_spans]
    col_x_spans_px = cy_px_spans

    hy_top = min(c["px_bbox"][1] for c in cyan_boxes) / scale
    hy_bot = max(c["px_bbox"][3] for c in cyan_boxes) / scale

    # --- Column labels: stay strictly inside the cyan header band so we
    # don't pull section-title text (which sits above) into the labels.
    hy_top_wide = hy_top - 1.5
    hy_bot_wide = hy_bot + 1.5
    col_labels: list[str] = []
    for x0, x1 in col_x_spans_pt:
        px_hdr = (x0 * scale, hy_top_wide * scale, x1 * scale, hy_bot_wide * scale)
        r = _pdf_rect(
            px_hdr, scale, pad=0.5,
            rotated_cw=rotated_cw, page_height_pt=page_height_pt,
        )
        hdr_words = _words_in_rect(
            page, r, rotated_cw=rotated_cw, page_height_pt=page_height_pt
        )
        if hdr_words:
            hdr_words.sort(key=lambda w: (w["y0"], w["x0"]))
            col_labels.append(" ".join(w["text"] for w in hdr_words))
        else:
            col_labels.append(_text_in_rect(page, r))
    # If a label still looks empty, try a slight widening (headers with
    # descenders / multi-line labels) — but capped small so we don't grab
    # the section title or data rows.
    for i, (x0, x1) in enumerate(col_x_spans_pt):
        if col_labels[i].strip():
            continue
        px_hdr = (x0 * scale, (hy_top - 4.0) * scale, x1 * scale, (hy_bot + 4.0) * scale)
        r = _pdf_rect(
            px_hdr, scale, pad=0.5,
            rotated_cw=rotated_cw, page_height_pt=page_height_pt,
        )
        hdr_words = _words_in_rect(
            page, r, rotated_cw=rotated_cw, page_height_pt=page_height_pt
        )
        if hdr_words:
            hdr_words.sort(key=lambda w: (w["y0"], w["x0"]))
            col_labels[i] = " ".join(w["text"] for w in hdr_words)

    # --- Filter orange cells inside this section below the header band
    sx0, sy0, sx1, sy1 = section_px_bbox
    top_px = data_y_top_px if data_y_top_px is not None else hy_bot * scale + 1
    top_px = max(top_px, hy_bot * scale + 1)
    # Helpers
    def _in_section(c: dict) -> bool:
        cx0, cy0, cx1, cy1 = c["px_bbox"]
        cx = (cx0 + cx1) / 2.0
        cy = (cy0 + cy1) / 2.0
        return (
            sx0 - 1 <= cx <= sx1 + 1
            and cy >= top_px - 1
            and cy1 <= sy1 + 4
        )

    cells = [c for c in orange_cells if _in_section(c)]
    if not cells:
        return None

    # Drop oversized cells (nested wrappers): anything taller than ~4× the
    # median cell height is almost certainly a panel wrapper, not a data cell.
    raw_heights = sorted((c["px_bbox"][3] - c["px_bbox"][1]) for c in cells)
    med_h_raw = raw_heights[len(raw_heights) // 2] if raw_heights else 12.0
    cells = [
        c for c in cells
        if (c["px_bbox"][3] - c["px_bbox"][1]) <= max(30.0, med_h_raw * 4.0)
    ]
    if not cells:
        return None

    # Dedupe near-duplicate cell detections (the detector occasionally emits
    # two overlapping orange rings for the same data cell, which would
    # incorrectly double the variant count below). Only true "same-sized
    # overlap" pairs are collapsed — nested cells (small cell inside a
    # wrapper) are NOT treated as duplicates.
    def _dedupe_cells(cs: list[dict]) -> list[dict]:
        kept: list[dict] = []
        for c in cs:
            cx0, cy0, cx1, cy1 = c["px_bbox"]
            ca = max(1, (cx1 - cx0) * (cy1 - cy0))
            is_dup = False
            for k in kept:
                kx0, ky0, kx1, ky1 = k["px_bbox"]
                ka = max(1, (kx1 - kx0) * (ky1 - ky0))
                ix0, iy0 = max(cx0, kx0), max(cy0, ky0)
                ix1, iy1 = min(cx1, kx1), min(cy1, ky1)
                if ix1 > ix0 and iy1 > iy0:
                    iarea = (ix1 - ix0) * (iy1 - iy0)
                    # Require the intersection to cover most of BOTH cells
                    # and the two cells to be of comparable size.
                    if (
                        iarea / max(ca, ka) >= 0.7
                        and min(ca, ka) / max(ca, ka) >= 0.7
                    ):
                        is_dup = True
                        break
            if not is_dup:
                kept.append(c)
        return kept

    cells = _dedupe_cells(cells)

    # --- Cluster orange cells into rows by y-band.
    heights = sorted((c["px_bbox"][3] - c["px_bbox"][1]) for c in cells)
    med_h = heights[len(heights) // 2] if heights else 12.0
    # Slightly looser y-banding: tall multi-line body cells and short data cells
    # in the *same* table row can have >8px center-y delta; a tight band_thresh
    # spawns a junk band with no description/symbol (duplicate of the row above).
    band_thresh = max(10.0, 0.72 * med_h)

    cells.sort(key=lambda c: ((c["px_bbox"][1] + c["px_bbox"][3]) / 2.0, c["px_bbox"][0]))
    bands: list[list[dict]] = []
    band_y: list[float] = []
    for c in cells:
        cy = (c["px_bbox"][1] + c["px_bbox"][3]) / 2.0
        if bands and abs(cy - band_y[-1]) <= band_thresh:
            bands[-1].append(c)
            band_y[-1] = sum((x["px_bbox"][1] + x["px_bbox"][3]) / 2.0 for x in bands[-1]) / len(bands[-1])
        else:
            bands.append([c])
            band_y.append(cy)

    # --- Cache cell text so we extract each cell at most once
    def _cell_text(c: dict) -> str:
        r = _pdf_rect(
            c["px_bbox"], scale, pad=0.8,
            rotated_cw=rotated_cw, page_height_pt=page_height_pt,
        )
        words = _words_in_rect(
            page, r, rotated_cw=rotated_cw, page_height_pt=page_height_pt,
        )
        words.sort(key=lambda w: (w["y0"], w["x0"]))
        t = " ".join(w["text"] for w in words).strip()
        if not t:
            t = _text_in_rect(page, r)
        return re.sub(r"\s+", " ", t).strip()

    def _assign_col(c: dict) -> int:
        cx_pt = (c["px_bbox"][0] + c["px_bbox"][2]) / 2.0 / scale
        j = 0
        best = 1e9
        for idx, (x0, x1) in enumerate(col_x_spans_pt):
            if x0 - 1 <= cx_pt <= x1 + 1:
                return idx
            d = min(abs(cx_pt - x0), abs(cx_pt - x1))
            if d < best:
                best = d
                j = idx
        return j

    # --- Build one "record" per band; a record can carry N sub-row variants
    rows_out: list[dict[str, Any]] = []
    col0_w_px = (col_x_spans_pt[0][1] - col_x_spans_pt[0][0]) * scale if col_x_spans_pt else 40.0
    for band in bands:
        # Group cells by column
        by_col: dict[int, list[dict]] = defaultdict(list)
        for c in band:
            j = _assign_col(c)
            by_col[j].append(c)
        # Sort each column's stack top-to-bottom
        for j in by_col:
            by_col[j].sort(key=lambda c: c["px_bbox"][1])

        # Extract symbol-column cells separately — they never contribute to
        # variant counting (symbol glyphs often split into 2-3 tiny pieces).
        symbol_cells = by_col.pop(0, [])

        # Group each column's stacked cells by vertical gap. The detector in
        # these schedules emits one orange cell per logical cell (even for
        # multi-line text), so any non-trivial vertical gap between two
        # stacked cells signals separate variants — we keep a TIGHT threshold
        # so 4-5 px gaps (typical between sub-rows) are preserved.
        gap_thresh = 1.5
        col_groups: dict[int, list[list[dict]]] = {}
        for j, col_cells in by_col.items():
            s = sorted(col_cells, key=lambda c: c["px_bbox"][1])
            if not s:
                col_groups[j] = []
                continue
            groups: list[list[dict]] = [[s[0]]]
            for c in s[1:]:
                prev = groups[-1][-1]
                gap = c["px_bbox"][1] - prev["px_bbox"][3]
                if gap <= gap_thresh:
                    groups[-1].append(c)
                else:
                    groups.append([c])
            col_groups[j] = groups

        # Variant count = max number of GROUPS across non-symbol columns.
        # A single tall cell / one multi-line group contributes just 1.
        non_sym_group_counts = [len(g) for g in col_groups.values()]
        variant_count = max(non_sym_group_counts) if non_sym_group_counts else 1
        variant_count = max(1, variant_count)

        # Build each variant's cells by selecting the v-th group per column.
        body_len = max(0, len(col_x_spans_pt) - 1)

        def _group_text(group: list[dict]) -> str:
            txt = " ".join(_cell_text(c) for c in group)
            return re.sub(r"\s+", " ", txt).strip()

        variants: list[list[str]] = []
        for vi in range(variant_count):
            vc = [""] * body_len
            for j, groups in col_groups.items():
                ri = j - 1
                if ri < 0 or ri >= body_len or not groups:
                    continue
                if len(groups) == 1:
                    # Spans all variants — repeat the same text.
                    vc[ri] = _group_text(groups[0])
                elif vi < len(groups):
                    vc[ri] = _group_text(groups[vi])
            variants.append(vc)

        # Symbol text is the concatenation of all narrow cells in col 0.
        symbol_text = " ".join(_cell_text(c) for c in symbol_cells).strip()
        symbol_text = re.sub(r"\s+", " ", symbol_text)

        # Symbol image anchor: prefer the narrowest cell in the symbol column, but
        # never require a max-width filter — wide glyphs (e.g. TV + triangle + #)
        # can exceed 1.6× col width; without a fallback we skip PNG crops entirely.
        sym_anchor: dict | None = None
        max_w_sym = max(col0_w_px * 1.6, 40.0)
        valid_sym = [
            c for c in symbol_cells
            if (c["px_bbox"][2] - c["px_bbox"][0]) <= max_w_sym
        ]
        pool = valid_sym if valid_sym else symbol_cells
        if pool:
            sym_anchor = min(
                pool, key=lambda c: c["px_bbox"][2] - c["px_bbox"][0]
            )
        # If the detector put no orange in the symbol column for this band, still
        # cut the column-0 y-span so the writer never skips ``secXX_rowNNN.png``.
        if sym_anchor is None and band:
            x0s, x1s = col_x_spans_px[0]
            ys: list[float] = []
            for c in band:
                ys.extend([c["px_bbox"][1], c["px_bbox"][3]])
            if ys:
                sym_anchor = {
                    "px_bbox": [x0s, min(ys) - 2.0, x1s, max(ys) + 2.0],
                }

        # Drop any trailing variants that are entirely empty (detector noise),
        # and skip the whole record if nothing meaningful remains.
        variants = [v for v in variants if any((c or "").strip() for c in v)]
        if not variants and not symbol_text:
            continue
        if not variants:
            # symbol-only stub (e.g. bare icon). Keep as single empty variant.
            variants = [[""] * body_len]
        variant_count = len(variants)

        rows_out.append(
            {
                "symbol_text": symbol_text,
                "symbol_anchor": sym_anchor,
                "variant_count": variant_count,
                "variants": variants,
            }
        )

    rows_out = _drop_y_band_fragment_rows(rows_out)

    return {
        "headers": col_labels,
        "col_x_spans_px": col_x_spans_px,
        "rows": rows_out,
    }


def _dedupe_orange_cells_simple(cs: list[dict]) -> list[dict]:
    """Collapse duplicate orange rings for the same logical cell (same as table builder)."""
    kept: list[dict] = []
    for c in cs:
        cx0, cy0, cx1, cy1 = c["px_bbox"]
        ca = max(1, (cx1 - cx0) * (cy1 - cy0))
        is_dup = False
        for k in kept:
            kx0, ky0, kx1, ky1 = k["px_bbox"]
            ka = max(1, (kx1 - kx0) * (ky1 - ky0))
            ix0, iy0 = max(cx0, kx0), max(cy0, ky0)
            ix1, iy1 = min(cx1, kx1), min(cy1, ky1)
            if ix1 > ix0 and iy1 > iy0:
                iarea = (ix1 - ix0) * (iy1 - iy0)
                if iarea / max(ca, ka) >= 0.7 and min(ca, ka) / max(ca, ka) >= 0.7:
                    is_dup = True
                    break
        if not is_dup:
            kept.append(c)
    return kept


def _uniform_cell_text(
    page,
    cell_box: dict,
    scale: float,
    *,
    rotated_cw: bool,
    page_height_pt: float,
) -> str:
    r = _pdf_rect(
        cell_box["px_bbox"], scale, pad=0.8,
        rotated_cw=rotated_cw, page_height_pt=page_height_pt,
    )
    words = _words_in_rect(
        page, r, rotated_cw=rotated_cw, page_height_pt=page_height_pt,
    )
    words.sort(key=lambda w: (w["y0"], w["x0"]))
    t = " ".join(w["text"] for w in words).strip()
    if not t:
        t = _text_in_rect(page, r)
    return re.sub(r"\s+", " ", t).strip()


def _split_bands_into_uniform_table_chunks(
    bands: list[list[dict]],
    band_y: list[float],
) -> list[list[list[dict]]]:
    """Split y-clustered orange bands when a **different grid** starts below.

    Long GC/commercial sheets stack unrelated tables (checklist + quantities)
    in one detector wrapper without CYAN guides. Signals:

    - **Column-count jump** — new header row has a different cell count than
      the active table (e.g. 10-wide scope vs 6-wide Units).
    - **Large vertical gap** — whitespace between stacked tables (only after
      at least one full header+data pair so we never split the scope header
      from its first data row).
    """
    if len(bands) < 2:
        return []
    gaps = [band_y[i] - band_y[i - 1] for i in range(1, len(band_y))]
    med_g = sorted(gaps)[len(gaps) // 2] if gaps else 14.0
    gap_split = max(32.0, 2.35 * med_g)

    chunks: list[list[list[dict]]] = []
    split_start = 0
    hdr_n = len(bands[0])

    for i in range(1, len(bands)):
        gap = band_y[i] - band_y[i - 1]
        ln = len(bands[i])
        col_jump = abs(ln - hdr_n) >= 4
        big_gap = gap > gap_split and i > 1
        if col_jump or big_gap:
            chunks.append(bands[split_start:i])
            split_start = i
            hdr_n = len(bands[i])
    chunks.append(bands[split_start:])
    return [c for c in chunks if len(c) >= 2]


def _column_x_bounds_from_centers(
    centers: list[float],
    chunk_x0: float,
    chunk_x1: float,
) -> list[tuple[float, float]]:
    """Return ``n`` column x-intervals (pixel space) from column centers."""
    n = len(centers)
    if n == 0:
        return []
    out: list[tuple[float, float]] = []
    for j in range(n):
        left = chunk_x0 if j == 0 else (centers[j - 1] + centers[j]) / 2.0
        right = chunk_x1 if j == n - 1 else (centers[j] + centers[j + 1]) / 2.0
        out.append((left, right))
    return out


def _matrix_cell_text_for_column_slice(
    page,
    x0_px: float,
    x1_px: float,
    y0_px: float,
    y1_px: float,
    section_px_bbox: list[float],
    scale: float,
    *,
    rotated_cw: bool,
    page_height_pt: float,
) -> str:
    """Words in a header-row slice when an orange ring missed a narrow column."""
    sx0, sy0, sx1, sy1 = section_px_bbox
    clip = [
        max(sx0, min(x0_px, x1_px)),
        max(sy0, min(y0_px, y1_px)),
        min(sx1, max(x0_px, x1_px)),
        min(sy1, max(y0_px, y1_px)),
    ]
    if clip[2] <= clip[0] + 1.5 or clip[3] <= clip[1] + 1.5:
        return ""
    r = _pdf_rect(
        clip, scale, pad=0.25,
        rotated_cw=rotated_cw, page_height_pt=page_height_pt,
    )
    words = _words_in_rect(
        page, r, rotated_cw=rotated_cw, page_height_pt=page_height_pt,
    )
    words.sort(key=lambda w: (w["y0"], w["x0"]))
    t = " ".join(w["text"] for w in words if str(w.get("text", "")).strip()).strip()
    if not t:
        t = (_text_in_rect(page, r) or "").strip()
    return re.sub(r"\s+", " ", t).strip()


def _uniform_matrix_header_column_centers_from_words(
    page,
    *,
    chunk_x0: float,
    chunk_x1: float,
    y0_hdr: float,
    y1_hdr: float,
    section_px_bbox: list[float],
    scale: float,
    rotated_cw: bool,
    page_height_pt: float,
) -> list[float] | None:
    """Infer extra column boundaries from PDF words when orange rings skip a column.

    Uses adaptive gap-splitting on the header text line that contains *Type*
    (typical quantity tables). Skips scope-style checklist rows without *Type*.
    """
    sx0, sy0, sx1, sy1 = section_px_bbox
    y1_eff = min(sy1, y1_hdr + max(52.0, (y1_hdr - y0_hdr) * 1.75))
    clip = [
        max(sx0, chunk_x0),
        max(sy0, y0_hdr - 4.0),
        min(sx1, chunk_x1),
        min(sy1, y1_eff),
    ]
    if clip[2] <= clip[0] + 2.0 or clip[3] <= clip[1] + 2.0:
        return None
    r = _pdf_rect(
        clip, scale, pad=0.2,
        rotated_cw=rotated_cw, page_height_pt=page_height_pt,
    )
    words = _words_in_rect(
        page, r, rotated_cw=rotated_cw, page_height_pt=page_height_pt,
    )
    if len(words) < 5:
        return None
    rows = _cluster_rows(words, gap=3.0)
    best: list[dict] = []
    best_sc = -1.0
    for row in rows:
        s = " ".join(str(w.get("text", "")) for w in row).lower()
        sc = float(len(row))
        if re.search(r"\btype\b", s) and re.search(
            r"qty|homerun|location|unit|jack|drop|count", s
        ):
            sc += 500.0
        elif re.search(r"\btype\b", s):
            sc += 50.0
        if sc > best_sc:
            best_sc = sc
            best = row
    if not best:
        return None
    main = sorted(best, key=lambda w: w["x0"])
    joined = " ".join(str(w.get("text", "")) for w in main)
    if not re.search(r"(?i)\btype\b", joined):
        return None
    if len(main) < 2:
        return None
    gaps = [
        main[i + 1]["x0"] - main[i]["x1"]
        for i in range(len(main) - 1)
    ]
    small_gaps = [g for g in gaps if g <= 14.0]
    med_s = (
        sorted(small_gaps)[len(small_gaps) // 2]
        if small_gaps
        else 3.0
    )
    # ``med`` included column-sized gaps (~17px), inflating the threshold and
    # merging adjacent columns (e.g. *Units* + *Data Jacks*). Base the split
    # only on **within-cell** word gaps (≈2–4px in raster space).
    thresh = max(10.0, 5.2 * med_s)
    runs: list[list[dict]] = [[main[0]]]
    for w in main[1:]:
        g = w["x0"] - runs[-1][-1]["x1"]
        if g <= thresh:
            runs[-1].append(w)
        else:
            runs.append([w])
    if len(runs) < 5:
        return None
    return [
        sum((w["x0"] + w["x1"]) / 2.0 for w in run) / float(len(run))
        for run in runs
    ]


def _maybe_expand_total_drop_count_uniform_column(
    page,
    columns: list[str],
    centers: list[float],
    chunk_x0: float,
    chunk_x1: float,
    y0_hdr: float,
    y1_hdr: float,
    section_px_bbox: list[float],
    scale: float,
    *,
    rotated_cw: bool,
    page_height_pt: float,
) -> tuple[list[str], list[float], list[tuple[float, float]]] | None:
    """Append *Total Drop Count* when PDF text has it but orange rings stop earlier.

    Common on GC quantity tables: *Data Jacks* has a ring but the next header
    (*Total Drop Count*) does not; cells still align when geometry uses the
    widest orange row.
    """
    if len(columns) != len(centers):
        return None
    n = len(columns)
    if n != 5:
        return None
    joined = " ".join(columns).lower()
    if "total drop" in joined or "drop count" in joined:
        return None
    if not any(
        k in joined for k in ("data jack", "qty of unit", "homerun", "hr location")
    ):
        return None

    sx0, sy0, sx1, sy1 = section_px_bbox
    pad_px = 30.0
    clip_px = [
        max(sx0, chunk_x0 - pad_px),
        max(sy0, y0_hdr - 12.0),
        min(sx1, chunk_x1 + pad_px),
        min(sy1, y1_hdr + max(60.0, (y1_hdr - y0_hdr) * 2.25)),
    ]
    clip_pdf = _pdf_rect(
        clip_px, scale, pad=2.0,
        rotated_cw=rotated_cw, page_height_pt=page_height_pt,
    )

    new_cx_px: float | None = None
    label = "Total Drop Count"
    if not rotated_cw:
        try:
            hits: list = []
            for phrase in ("Total Drop Count", "Total Drop"):
                hits = page.search_for(phrase, clip=clip_pdf, quads=False) or []
                if hits:
                    if phrase == "Total Drop":
                        label = "Total Drop Count"
                    break
        except Exception:
            hits = []
        if hits:
            r = hits[0]
            cx_pdf = (float(r.x0) + float(r.x1)) / 2.0
            new_cx_px = cx_pdf * scale

    if new_cx_px is None and n >= 3:
        dx = centers[-1] - centers[-2]
        if dx > 4.0:
            cand = centers[-1] + dx * 0.97
            if cand <= chunk_x1 + 95.0:
                new_cx_px = cand

    if new_cx_px is None:
        return None

    pairs = list(zip(centers, columns)) + [(new_cx_px, label)]
    pairs.sort(key=lambda p: p[0])
    centers_out = [p[0] for p in pairs]
    columns_out = [p[1] for p in pairs]
    x_bounds = _column_x_bounds_from_centers(centers_out, chunk_x0, chunk_x1)
    return columns_out, centers_out, x_bounds


def _fill_uniform_matrix_row_empty_cells_from_slices(
    page,
    row: list[str],
    x_bounds: list[tuple[float, float]],
    y0_px: float,
    y1_px: float,
    section_px_bbox: list[float],
    scale: float,
    *,
    rotated_cw: bool,
    page_height_pt: float,
    columns: list[str],
) -> None:
    """Fill orange-mapping gaps using PDF words in each column × row slice."""
    if len(row) != len(x_bounds):
        return
    row_h = max(10.0, y1_px - y0_px)
    pad_y = min(5.0, row_h * 0.12)
    y0_u = y0_px - pad_y
    y1_u = y1_px + pad_y
    for j, xb in enumerate(x_bounds):
        if (row[j] or "").strip():
            continue
        x0b, x1b = xb
        span = x1b - x0b
        pad_x = max(2.5, min(14.0, span * 0.08))
        t = _matrix_cell_text_for_column_slice(
            page,
            x0b - pad_x,
            x1b + pad_x,
            y0_u,
            y1_u,
            section_px_bbox,
            scale,
            rotated_cw=rotated_cw,
            page_height_pt=page_height_pt,
        )
        if not t:
            continue
        col_lab = (columns[j] if j < len(columns) else "").lower()
        tok = t.strip().split()
        if len(tok) == 1:
            one = tok[0]
            if len(one) <= 2 and one.isdigit():
                if any(k in col_lab for k in ("location", "type")):
                    continue
            row[j] = one
        elif len(tok) >= 2 and all(re.fullmatch(r"\d+", x) for x in tok):
            row[j] = tok[-1]
        else:
            row[j] = t


def _uniform_matrix_from_band_chunk(
    page,
    tbands: list[list[dict]],
    scale: float,
    *,
    section_px_bbox: list[float],
    rotated_cw: bool,
    page_height_pt: float,
) -> dict[str, Any] | None:
    """Build one matrix dict from ``tbands`` (header band + ≥1 data row).

    Column **geometry** uses the **widest** orange row in the chunk: narrow
    right-hand headers (e.g. *Total Drop Count*) often lack an orange ring on
    the title row but still have rings on data rows — using only the header
    row would drop that column.
    """
    if len(tbands) < 2:
        return None

    n_geom = max(len(b) for b in tbands)
    # Schedules range from 2 cols (*Type* / *Qty.*) through 10+ col scope matrices.
    if n_geom < 2:
        return None

    hdr = sorted(tbands[0], key=lambda c: c["px_bbox"][0])
    if len(hdr) < 2:
        return None
    hdr_hs = [c["px_bbox"][3] - c["px_bbox"][1] for c in hdr]
    med_hdr_h = sorted(hdr_hs)[len(hdr_hs) // 2]
    max_hdr_h = max(hdr_hs)
    if med_hdr_h > 52 and max_hdr_h > 88:
        return None
    if max_hdr_h > 88:
        return None

    geom_candidates = [i for i, b in enumerate(tbands) if len(b) == n_geom]
    geom_i = min(geom_candidates)
    geom_row = sorted(tbands[geom_i], key=lambda c: c["px_bbox"][0])
    centers = [
        (c["px_bbox"][0] + c["px_bbox"][2]) / 2.0 for c in geom_row
    ]
    n = n_geom

    chunk_x0 = min(c["px_bbox"][0] for band in tbands for c in band)
    chunk_x1 = max(c["px_bbox"][2] for band in tbands for c in band)
    y0_hdr = min(c["px_bbox"][1] for c in hdr)
    y1_hdr = max(c["px_bbox"][3] for c in hdr)

    w_centers = _uniform_matrix_header_column_centers_from_words(
        page,
        chunk_x0=chunk_x0,
        chunk_x1=chunk_x1,
        y0_hdr=y0_hdr,
        y1_hdr=y1_hdr,
        section_px_bbox=section_px_bbox,
        scale=scale,
        rotated_cw=rotated_cw,
        page_height_pt=page_height_pt,
    )
    if w_centers and len(w_centers) > n:
        centers = w_centers
        n = len(centers)

    x_bounds = _column_x_bounds_from_centers(centers, chunk_x0, chunk_x1)
    acc_hdr: dict[int, list[str]] = defaultdict(list)
    for c in hdr:
        cx = (c["px_bbox"][0] + c["px_bbox"][2]) / 2.0
        j = min(range(n), key=lambda i: abs(cx - centers[i]))
        acc_hdr[j].append(
            _uniform_cell_text(
                page, c, scale,
                rotated_cw=rotated_cw, page_height_pt=page_height_pt,
            )
        )

    columns: list[str] = []
    for j in range(n):
        parts = [p for p in acc_hdr.get(j, []) if p and p.strip()]
        columns.append(" ".join(parts).strip())

    for j in range(n):
        if columns[j].strip():
            continue
        x0b, x1b = x_bounds[j]
        columns[j] = _matrix_cell_text_for_column_slice(
            page,
            x0b,
            x1b,
            y0_hdr,
            y1_hdr,
            section_px_bbox,
            scale,
            rotated_cw=rotated_cw,
            page_height_pt=page_height_pt,
        )

    maybe_td = _maybe_expand_total_drop_count_uniform_column(
        page,
        columns,
        centers,
        chunk_x0,
        chunk_x1,
        y0_hdr,
        y1_hdr,
        section_px_bbox,
        scale,
        rotated_cw=rotated_cw,
        page_height_pt=page_height_pt,
    )
    chunk_x1_eff = chunk_x1
    if maybe_td:
        columns, centers, x_bounds = maybe_td
        n = len(centers)
        if columns and re.search(
            r"(?i)total\s*drop|drop\s*count",
            columns[-1] or "",
        ):
            last_c = float(centers[-1])
            prev_c = float(centers[-2]) if len(centers) >= 2 else last_c - 44.0
            chunk_x1_eff = min(
                section_px_bbox[2],
                max(float(chunk_x1), last_c + max(56.0, (last_c - prev_c) * 0.75)),
            )
            x_bounds = _column_x_bounds_from_centers(centers, chunk_x0, chunk_x1_eff)

    if not any(x.strip() for x in columns):
        return None

    grid: list[list[str]] = []
    for band in tbands[1:]:
        acc: dict[int, list[str]] = defaultdict(list)
        for c in band:
            cx = (c["px_bbox"][0] + c["px_bbox"][2]) / 2.0
            j = min(range(n), key=lambda i: abs(cx - centers[i]))
            acc[j].append(
                _uniform_cell_text(
                    page, c, scale,
                    rotated_cw=rotated_cw, page_height_pt=page_height_pt,
                )
            )
        row = []
        for j in range(n):
            parts = [p for p in acc.get(j, []) if p and p.strip()]
            row.append(" ".join(parts).strip())
        y0b = min(c["px_bbox"][1] for c in band)
        y1b = max(c["px_bbox"][3] for c in band)
        _fill_uniform_matrix_row_empty_cells_from_slices(
            page,
            row,
            x_bounds,
            y0b,
            y1b,
            section_px_bbox,
            scale,
            rotated_cw=rotated_cw,
            page_height_pt=page_height_pt,
            columns=columns,
        )
        grid.append(row)

    px_top = min(c["px_bbox"][1] for band in tbands for c in band)
    px_bot = max(c["px_bbox"][3] for band in tbands for c in band)
    return {
        "columns": columns,
        "grid": grid,
        "_chunk_px_top": px_top,
        "_chunk_px_bot": px_bot,
    }


_SUBMATRIX_TITLE_HINT = re.compile(
    r"(?i)^[\s*]*(?:Units|Common Area|Equipment|Notes|Schedule)\s*[\s*]*$"
)


def _infer_uniform_matrix_follow_on_title(
    page,
    sect_px_bbox: list[float],
    scale: float,
    *,
    y_gap_top: float,
    y_gap_bottom: float,
    rotated_cw: bool,
    page_height_pt: float,
) -> str:
    """Heading text between two stacked uniform-matrix chunks (e.g. *Units*)."""
    sx0, sy_lo, sx1, sy_hi = sect_px_bbox
    a, b = sorted((float(y_gap_top), float(y_gap_bottom)))
    sy0 = max(sy_lo, a)
    sy1 = min(sy_hi, b)
    if sy1 - sy0 < 5.0:
        return ""
    exp_px = [sx0, sy0, sx1, sy1]
    r = _pdf_rect(
        exp_px, scale, pad=2.0,
        rotated_cw=rotated_cw, page_height_pt=page_height_pt,
    )
    raw = (
        _text_in_rect_preserve_lines(
            page, r,
            rotated_cw=rotated_cw, page_height_pt=page_height_pt,
        )
        or ""
    )
    for ln in raw.splitlines():
        t = re.sub(r"^\s*\**|\**\s*$", "", ln.strip())
        if _SUBMATRIX_TITLE_HINT.match(t):
            return re.sub(r"\s+", " ", t).strip()
    candidates: list[str] = []
    for ln in raw.splitlines():
        t = re.sub(r"^\s*\**|\**\s*$", "", ln.strip())
        if not t or len(t) > 72:
            continue
        if re.match(r"(?i)^scope\s+of\s+work$", t):
            continue
        if len(t.split()) <= 6 and re.match(r"^[\w\s\-/&]+$", t):
            candidates.append(t)
    return re.sub(r"\s+", " ", candidates[-1]).strip() if candidates else ""


_NEXT_SECTION_LINE_HINT = re.compile(
    r"(?i)^(common area|equipment|notes|schedule|scope of work)\s*$"
)

# When PDF text has no newline before the next bold section title, the tail rect can
# include ``Common Area Type Qty...`` on the same line as narrative prose.
_TAIL_INLINE_NEXT_SECTION = re.compile(
    r"(?i)(?<=[\w\)\]\d\|])\s+"
    r"(common area|equipment|notes|schedule)\s+"
    r"(?:type|qty\.?|hr)\b"
)


def _truncate_tail_at_next_schedule_heading_blob(text: str) -> str:
    """Cut merged-line bleed before another subsection title (*Unit AP Installation*, …)."""
    t = text.strip()
    if not t:
        return t
    # Next schedule merged onto one line with no leading whitespace — drop entirely.
    if re.match(r"(?i)Closet\s+Build\s*out\s+Type\b", t):
        return ""
    if re.match(r"(?i)Unit\s+AP\s+Installation\s+Type\b", t):
        return ""
    cuts: list[int] = []
    for pat in (
        r"(?i)\s+Unit\s+AP\s+Installation\b",
        r"(?i)\s+Closet\s+Build\s*out\b",
        r"(?i)\s+Fiber\s+backbone\b(?=\s+From\b)",
    ):
        m = re.search(pat, t)
        if m:
            cuts.append(m.start())
    if cuts:
        t = t[: min(cuts)].rstrip()
    return t


def _uniform_matrix_columns_look_like_qty_table(columns: list[str] | None) -> bool:
    """Non-scope schedules: 2-col Type/Qty, 3-col fiber summaries, 4+ col unit counts, …"""
    if not columns:
        return False
    joined = " ".join((c or "").strip() for c in columns).lower()
    n = len(columns)
    # Two-column take-off (*Type* | *Qty.*) — e.g. AP installation, closet build-out.
    if n == 2 and re.search(r"\btype\b", joined) and re.search(r"\bqty", joined):
        return True
    # Fiber backbone (*From* | *Number of IDFs* | *Total Runs*) — often no ``type``.
    if re.search(r"\bfrom\b", joined) and re.search(
        r"idf|mdf|runs|total\s+runs|number\s+of",
        joined,
    ):
        return True
    if not re.search(r"\btype\b", joined):
        return False
    return bool(
        re.search(
            r"qty|homerun|hr\s+location|\bhr\b|location|drop|count|unit|jack|"
            r"keystone|data|mount|wall|ceiling|in\s+wall",
            joined,
        )
    )


def _should_skip_tabular_notes_noise(section: dict[str, Any]) -> bool:
    """Drop ``notes`` sections that are really table text dumped from sliver wrappers."""
    if str(section.get("kind")) != "notes":
        return False
    body = (section.get("body") or "").strip()
    title = (section.get("title") or "").strip()
    if len(body) < 45:
        return False
    lo = body.lower()
    if (
        re.search(r"\bstudio\b", lo)
        and re.search(r"\b1 bed\b", lo)
        and "idf" in lo
        and "total" in lo
    ):
        return True
    if re.search(r"data drop.*keystone|keystone", lo) and re.search(
        r"\bap homerun\b|\bdual cat6\b", lo
    ):
        return True
    tu = title.upper()
    if tu in ("TYPE", "QTY.", "QTY", "HR", "COUNT", "HR LOCATION"):
        lines = [ln for ln in body.splitlines() if ln.strip()]
        if len(lines) >= 6 and sum(1 for c in body if c.isdigit()) >= 8:
            return True
    if re.match(r"(?i)^cat6\)\s*$", title) and "keystone" in lo:
        return True
    return False


def _infer_gc_work_order_matrix_banner_title(
    page,
    sect_px: list[float],
    scale: float,
    *,
    rotated_cw: bool,
    page_height_pt: float,
    current: str | None,
) -> str:
    """Prefer *SCOPE OF WORK* / *Units* / *Common Area* over column labels (*TYPE*, fragments)."""
    sx0, sy0, sx1, sy1 = [float(x) for x in sect_px]
    span = sy1 - sy0
    h = min(130.0, max(40.0, span * 0.42))
    # Subsection titles (*Common Area*, *Units*) often sit **above** the slice top
    # (wrapper slice starts at the table header row).
    head_top = max(0.0, sy0 - 96.0)
    head_px = [sx0, head_top, sx1, min(sy1, sy0 + h)]
    if head_px[3] <= head_px[1] + 3.0:
        return (current or "").strip()
    raw = (
        _text_in_rect_preserve_lines(
            page,
            _pdf_rect(
                head_px, scale, pad=0.5,
                rotated_cw=rotated_cw, page_height_pt=page_height_pt,
            ),
            rotated_cw=rotated_cw, page_height_pt=page_height_pt,
        )
        or ""
    )
    t = _norm(raw)
    if re.search(r"(?i)scope\s+of\s+work", t):
        return "SCOPE OF WORK"
    if re.search(r"(?i)fiber\s+backbone", t):
        return "Fiber backbone"
    if re.search(r"(?i)unit\s+ap\s+installation", t):
        return "Unit AP Installation"
    if re.search(r"(?i)closet\s+build\s*out", t):
        return "Closet Build out"
    # ``Common Area`` quantity block (not *Common Area Wiring* in SCOPE checklist).
    if re.search(r"(?im)(^|\n)\s*Common\s+Area\s*(\n|$)", t) or (
        re.search(r"(?i)\bCommon\s+Area\b", t[:360])
        and not re.search(r"(?i)common\s+area\s+wiring", t[:140])
    ):
        return "Common Area"
    if re.search(r"(?im)(^|\n)\s*Units\s*(\n|$)", t) or re.search(
        r"(?i)\bUnits\s+Type\s", t[:360]
    ):
        return "Units"
    cur = (current or "").strip()
    cu = cur.upper()
    junk_title = cu in (
        "TYPE",
        "QTY.",
        "QTY",
        "HR",
        "COUNT",
        "CAT6)",
        "CAT6",
    ) or (len(cur) <= 6 and re.match(r"(?i)^type$", cur)) or bool(
        re.match(r"(?i)^cat6", cur)
    )
    if junk_title:
        if re.search(r"(?i)\bCommon\s+Area\b", t):
            return "Common Area"
        if re.search(r"(?i)\bunits\b", t):
            return "Units"
    return cur


def _interstitial_is_spilled_next_chunk_table(text: str) -> bool:
    """True when extracted text is actually the next matrix's title/header/table
    spill (gap rect above orange grid, or tail rect after a matrix whose follow-on
    table lives in a different wrapper slice)."""
    t = re.sub(r"^\s*\**|\**\s*$", "", (text or "").strip())
    if len(t) < 10:
        return False
    head = t[:120]
    return bool(re.match(r"(?is)^units\s+type\s+", head))


def _join_matrix_notes(a: str | None, b: str | None) -> str | None:
    a = (a or "").strip()
    b = (b or "").strip()
    if not b:
        return a or None
    if not a:
        return b or None
    if b in a or a in b:
        return a
    return f"{a}\n\n{b}"


def _extract_tail_notes_before_next_heading(
    page,
    sect_px_bbox: list[float],
    scale: float,
    *,
    y_top_px: float,
    page_h_px: int,
    rotated_cw: bool,
    page_height_pt: float,
) -> str:
    """Prose below a matrix when the next section uses different orange clusters."""
    sx0, sy0, sx1, sy1 = sect_px_bbox
    sy_a = max(sy0, float(y_top_px))
    # Synthetic wrappers (e.g. ``mccol_*``) often end **above** the last orange
    # row; narrative then sits outside ``sy1`` but still belongs to this section.
    sy_b = min(float(page_h_px) - 1.0, sy_a + 520.0)
    if sy_b - sy_a < 8.0:
        return ""
    exp_px = [sx0, sy_a, sx1, sy_b]
    r = _pdf_rect(
        exp_px, scale, pad=1.0,
        rotated_cw=rotated_cw, page_height_pt=page_height_pt,
    )
    raw = (
        _text_in_rect_preserve_lines(
            page, r,
            rotated_cw=rotated_cw, page_height_pt=page_height_pt,
        )
        or ""
    )
    raw = _norm(raw)
    lines = [ln.rstrip() for ln in raw.splitlines()]
    cut = len(lines)
    for i, ln in enumerate(lines):
        tl = re.sub(r"^\s*\**|\**\s*$", "", ln.strip())
        if _NEXT_SECTION_LINE_HINT.match(tl):
            cut = i
            break
    text = "\n".join(lines[:cut]).strip()
    while text.startswith("\n"):
        text = text.lstrip("\n")
    m_inline = _TAIL_INLINE_NEXT_SECTION.search(text)
    if m_inline:
        text = text[: m_inline.start()].rstrip()
    text = _truncate_tail_at_next_schedule_heading_blob(text)
    if _interstitial_is_spilled_next_chunk_table(text):
        return ""
    if len(text) < 8:
        return ""
    return text


def _extract_matrix_interstitial_notes(
    page,
    sect_px_bbox: list[float],
    scale: float,
    *,
    y_gap_top: float,
    y_gap_bottom: float,
    rotated_cw: bool,
    page_height_pt: float,
) -> str:
    """Prose between two stacked matrix chunks (e.g. unit wiring narrative).

    Vertical span is the slice **below** the upper chunk's orange bbox and
    **above** the lower chunk's orange bbox — outside both grids.
    """
    sx0, sy_lo, sx1, sy_hi = sect_px_bbox
    a, b = sorted((float(y_gap_top), float(y_gap_bottom)))
    sy0 = max(sy_lo, a + 0.5)
    sy1 = min(sy_hi, b - 0.5)
    if sy1 - sy0 < 6.0:
        return ""
    exp_px = [sx0, sy0, sx1, sy1]
    r = _pdf_rect(
        exp_px, scale, pad=1.5,
        rotated_cw=rotated_cw, page_height_pt=page_height_pt,
    )
    raw = (
        _text_in_rect_preserve_lines(
            page, r,
            rotated_cw=rotated_cw, page_height_pt=page_height_pt,
        )
        or ""
    )
    raw = _norm(raw)
    lines = [ln.rstrip() for ln in raw.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    while lines:
        tl = lines[0].strip()
        tl = re.sub(r"^\s*\**|\**\s*$", "", tl)
        if _NEXT_SECTION_LINE_HINT.match(tl):
            lines.pop(0)
            continue
        break
    while lines:
        tl = lines[-1].strip()
        tl = re.sub(r"^\s*\**|\**\s*$", "", tl)
        if _NEXT_SECTION_LINE_HINT.match(tl):
            lines.pop()
            continue
        break
    text = "\n".join(lines).strip()
    if _interstitial_is_spilled_next_chunk_table(text):
        return ""
    if len(text) < 8:
        return ""
    return text


def _try_uniform_matrices_from_oranges(
    page,
    section_px_bbox: list[float],
    orange_cells: list[dict],
    scale: float,
    *,
    rotated_cw: bool,
    page_height_pt: float,
    data_y_top_px: float | None,
) -> list[dict[str, Any]]:
    """No CYAN column headers: infer one or more uniform matrices from orange cells.

    Top band with ≥5 cells → header row; bands below → data rows, possibly split
    into multiple tables when column counts or vertical gaps change.
    """
    sx0, sy0, sx1, sy1 = section_px_bbox

    def _cent_in(pxb: list[float] | tuple[float, ...]) -> bool:
        cx = (pxb[0] + pxb[2]) / 2.0
        cy = (pxb[1] + pxb[3]) / 2.0
        return sx0 - 3 <= cx <= sx1 + 3 and sy0 - 3 <= cy <= sy1 + 3

    cells = [c for c in orange_cells if _cent_in(c["px_bbox"])]
    if data_y_top_px is not None:
        cells = [
            c
            for c in cells
            if (c["px_bbox"][1] + c["px_bbox"][3]) / 2.0 >= float(data_y_top_px) - 10.0
        ]
    # Header + at least one data row: 2×2 minimum for 2-col tables.
    if len(cells) < 4:
        return []

    cells = _dedupe_orange_cells_simple(cells)

    heights = sorted((c["px_bbox"][3] - c["px_bbox"][1]) for c in cells)
    med_h = heights[len(heights) // 2] if heights else 12.0
    band_thresh = max(10.0, 0.72 * med_h)

    cells.sort(
        key=lambda c: (
            (c["px_bbox"][1] + c["px_bbox"][3]) / 2.0,
            c["px_bbox"][0],
        )
    )
    bands: list[list[dict]] = []
    band_y: list[float] = []
    for c in cells:
        cy = (c["px_bbox"][1] + c["px_bbox"][3]) / 2.0
        if bands and abs(cy - band_y[-1]) <= band_thresh:
            bands[-1].append(c)
            band_y[-1] = sum(
                (x["px_bbox"][1] + x["px_bbox"][3]) / 2.0 for x in bands[-1]
            ) / len(bands[-1])
        else:
            bands.append([c])
            band_y.append(cy)

    if len(bands) < 2:
        return []

    tb_chunks = _split_bands_into_uniform_table_chunks(bands, band_y)
    out: list[dict[str, Any]] = []
    for tbands in tb_chunks:
        one = _uniform_matrix_from_band_chunk(
            page, tbands, scale,
            section_px_bbox=section_px_bbox,
            rotated_cw=rotated_cw, page_height_pt=page_height_pt,
        )
        if one:
            out.append(one)
    return out


def _section_has_scope_heading_text(
    title_text: str,
    page,
    sect_px_bbox: list[float],
    scale: float,
    *,
    rotated_cw: bool,
    page_height_pt: float,
) -> bool:
    if title_text and re.search(r"scope\s+of\s+work", title_text, re.I):
        return True
    sx0, sy0, sx1, sy1 = sect_px_bbox
    # Title often sits *above* the orange grid bbox — search upward.
    exp_px = [sx0, max(0.0, sy0 - 220.0), sx1, sy1]
    r = _pdf_rect(
        exp_px, scale, pad=1.0,
        rotated_cw=rotated_cw, page_height_pt=page_height_pt,
    )
    blob = (
        _text_in_rect_preserve_lines(
            page, r,
            rotated_cw=rotated_cw, page_height_pt=page_height_pt,
        )
        or ""
    )[:2500]
    return bool(re.search(r"scope\s+of\s+work", blob, re.I))


def _extract_scope_section_title(
    page,
    sect_px_bbox: list[float],
    scale: float,
    *,
    rotated_cw: bool,
    page_height_pt: float,
) -> str:
    sx0, sy0, sx1, sy1 = sect_px_bbox
    exp_px = [sx0, max(0.0, sy0 - 220.0), sx1, sy1]
    r = _pdf_rect(
        exp_px, scale, pad=1.0,
        rotated_cw=rotated_cw, page_height_pt=page_height_pt,
    )
    raw = (
        _text_in_rect_preserve_lines(
            page, r,
            rotated_cw=rotated_cw, page_height_pt=page_height_pt,
        )
        or ""
    )
    for ln in raw.splitlines():
        if re.search(r"scope\s+of\s+work", ln, re.I):
            return re.sub(r"\s+", " ", ln.strip())
    return ""


# ---------------------------------------------------------------------------
# Abbreviations (GREEN cells grouped by RED rows)
# ---------------------------------------------------------------------------


def _build_abbreviations(
    page,
    abbr_cells: list[dict],
    abbr_rows: list[dict],
    scale: float,
    *,
    rotated_cw: bool,
    page_height_pt: float,
    all_boxes: list[dict] | None = None,
) -> (
    list[dict[str, str]] | tuple[list[dict[str, str]], dict[str, list[dict[str, str]]]]
):
    """Build symbol/meaning pairs from GREEN/RED minitables.

    When ``all_boxes`` is provided, rows whose *row* bbox center lies inside
    a ``mccol_N_col_M`` column are **excluded** from the returned *global* list
    and bucketed in the second return value: ``"N|M"`` →
    :class:`[{\"symbol\", \"meaning\"}, ...]` (N = ``mccol`` group index, M =
    column index) so they can be attached to the right discipline in a drawing
    index. Otherwise behavior is unchanged (all pairs global)."""
    if not abbr_rows or not abbr_cells:
        if all_boxes is not None:
            return [], {}
        return []

    by_row: dict[str, list[dict]] = defaultdict(list)
    for c in abbr_cells:
        best_row = None
        best_area = -1.0
        for r in abbr_rows:
            if _bbox_overlaps(c["px_bbox"], r["px_bbox"]):
                cx0, cy0, cx1, cy1 = c["px_bbox"]
                rx0, ry0, rx1, ry1 = r["px_bbox"]
                ix0, iy0 = max(cx0, rx0), max(cy0, ry0)
                ix1, iy1 = min(cx1, rx1), min(cy1, ry1)
                a = (ix1 - ix0) * (iy1 - iy0)
                if a > best_area:
                    best_area = a
                    best_row = r["box_id"]
        if best_row:
            by_row[best_row].append(c)

    def _mccol_bucket(
        rbox: list[float] | None,
    ) -> str | None:
        if not rbox or not all_boxes:
            return None
        x0, y0, x1, y1 = rbox
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        for b in all_boxes:
            m = re.match(r"^mccol_(\d+)_col_(\d+)$", str(b.get("box_id", "")))
            if not m:
                continue
            px0, py0, px1, py1 = b["px_bbox"]
            if px0 <= cx <= px1 and py0 <= cy <= py1:
                return f"{m.group(1)}|{m.group(2)}"
        return None

    global_pairs: list[dict[str, str]] = []
    scoped: dict[str, list[dict[str, str]]] = defaultdict(list)
    row_sorted = sorted(abbr_rows, key=lambda r: r["px_bbox"][1])
    for r in row_sorted:
        cells = sorted(by_row.get(r["box_id"], []), key=lambda c: c["px_bbox"][0])
        texts: list[str] = []
        for c in cells:
            rect = _pdf_rect(
                c["px_bbox"], scale, pad=0.5,
                rotated_cw=rotated_cw, page_height_pt=page_height_pt,
            )
            t = _text_in_rect(page, rect)
            if t:
                texts.append(t)
        if len(texts) < 1:
            continue
        if len(texts) >= 2:
            p = {
                "symbol": texts[0].strip(),
                "meaning": " ".join(t.strip() for t in texts[1:] if t),
            }
        else:
            p = {"symbol": texts[0].strip(), "meaning": ""}
        bkt = _mccol_bucket(r.get("px_bbox") if r else None)
        if bkt is not None:
            scoped[bkt].append(p)
        else:
            global_pairs.append(p)

    if all_boxes is not None:
        return global_pairs, dict(scoped)
    return global_pairs


# ---------------------------------------------------------------------------
# Mccol (multi-column contact / index) structured extraction
# ---------------------------------------------------------------------------


def _split_banner_and_section_heading(combined: str) -> tuple[str, str]:
    """Split ``"BANNER — HEADING"`` (em dash, en dash, or ASCII hyphen) into
    the sheet title line vs the in-section label (e.g. PROJECT TEAM)."""
    t = (combined or "").strip()
    for sep in (" — ", " – ", " - "):
        if sep in t:
            a, b = t.split(sep, 1)
            a, b = a.strip(), b.strip()
            if a and b:
                return a, b
    return t, ""


def _extract_mccol_column_boxes(
    page,
    wrapper: dict,
    all_boxes: list[dict],
    scale: float,
    *,
    rotated_cw: bool,
    page_height_pt: float,
) -> list[dict[str, str]] | None:
    """One entry per ``mccol_N_col_M`` (left-to-right), with CYAN header as
    ``label`` and text below the header in ``body`` (per-column clip)."""
    wid = str(wrapper.get("box_id", ""))
    m = re.match(r"^mccol_(\d+)_group$", wid)
    if not m:
        return None
    n = m.group(1)
    prefix = f"mccol_{n}"
    col_re = re.compile(rf"^{re.escape(prefix)}_col_(\d+)$")
    cols: list[tuple[int, dict]] = []
    for b in all_boxes:
        bid = str(b.get("box_id", ""))
        mat = col_re.match(bid)
        if not mat or str(b.get("parent_box_id")) != wid:
            continue
        cols.append((int(mat.group(1)), b))
    if not cols:
        return None
    cols.sort(key=lambda x: (x[0], x[1]["px_bbox"][0]))
    out: list[dict[str, str]] = []
    for idx, col_box in cols:
        cx0, cy0, cx1, cy1 = col_box["px_bbox"]
        col_id = str(col_box.get("box_id", ""))
        data_top = float(cy0)
        label = ""
        hdr_id = f"{prefix}_hdr_{idx}"
        for h in all_boxes:
            if str(h.get("box_id")) != hdr_id or str(h.get("parent_box_id")) != col_id:
                continue
            hx0, hy0, hx1, hy1 = h["px_bbox"]
            rlabel = _pdf_rect(
                [hx0, hy0, hx1, hy1], scale, pad=0.5,
                rotated_cw=rotated_cw, page_height_pt=page_height_pt,
            )
            label = re.sub(
                r"[\s\u00a0]+", " ", _text_in_rect(page, rlabel) or ""
            ).strip()
            label = re.sub(r":\s*$", "", label)
            data_top = hy1 + 2.0
            break
        rdata = _pdf_rect(
            [cx0, data_top, cx1, cy1], scale, pad=0.5,
            rotated_cw=rotated_cw, page_height_pt=page_height_pt,
        )
        body = _text_in_rect_preserve_lines(
            page, rdata,
            rotated_cw=rotated_cw, page_height_pt=page_height_pt,
        )
        body = re.sub(r"[\r\t]+", "", body)
        body = re.sub(r"\n{3,}", "\n\n", body).strip()
        out.append(
            {
                "id": f"col_{idx}",
                "label": label,
                "body": body,
            }
        )
    return out


def _rows_from_drawing_index_body(body: str) -> list[dict[str, str]]:
    """Alternating non-empty lines: sheet id then title (per-column drawing list)."""
    lines = [ln.strip() for ln in (body or "").splitlines() if ln.strip()]
    if not lines:
        return []
    out: list[dict[str, str]] = []
    for i in range(0, len(lines) - 1, 2):
        out.append({"sheet": lines[i], "title": lines[i + 1]})
    if len(lines) % 2 == 1:
        out.append({"sheet": lines[-1], "title": ""})
    return out


def _apply_scoped_abbreviations_to_mccol_sections(
    sections: list[dict[str, Any]],
    abbr_scoped: dict[str, list[dict[str, str]]],
) -> None:
    """Attach minitable pairs to the ``mccol`` column that contains them; add
    ``rows`` with ``sheet`` / ``title``; for drawing index, clear duplicate ``body``."""
    for sec in sections:
        if sec.get("kind") != "mccol" or "boxes" not in sec:
            continue
        mxi = sec.get("mccol_index")
        if mxi is None:
            continue
        sh_u = (sec.get("section_heading") or "").strip().upper()
        is_drawing = "DRAWING" in sh_u and "SHEET" in sh_u
        for bx in sec.get("boxes") or []:
            if not isinstance(bx, dict):
                continue
            m = re.match(r"^col_(\d+)$", str(bx.get("id", "")))
            if not m:
                continue
            cj = int(m.group(1))
            key = f"{mxi}|{cj}"
            raw = list(abbr_scoped.get(key) or [])
            rows: list[dict[str, str]] = []
            for p in raw:
                sym = (p.get("symbol") or "").strip()
                mean = (p.get("meaning") or "").strip()
                if sym or mean:
                    rows.append({"sheet": sym, "title": mean})
            if not rows and is_drawing and (bx.get("body") or "").strip():
                rows = _rows_from_drawing_index_body(bx.get("body", ""))
            if rows:
                bx["rows"] = rows
                if is_drawing:
                    bx["body"] = ""
            elif is_drawing and (bx.get("body") or "").strip():
                p2 = _rows_from_drawing_index_body(bx.get("body", ""))
                if p2:
                    bx["rows"] = p2
                    bx["body"] = ""


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def _wrapper_id(b: dict) -> str | None:
    bid = str(b["box_id"])
    if re.fullmatch(r"v\d+", bid):
        return bid
    p = str(b.get("parent_box_id") or "")
    m = re.match(r"^(v\d+)(?:_|$)", p)
    if m:
        return m.group(1)
    m = re.match(r"^(v\d+)_", bid)
    if m:
        return m.group(1)
    return None


def _presentation_tier(title: str, kind: str, placement: str | None) -> int:
    """Integer tier: **lower = earlier** in the emitted section list. Geometry
    (y, x) is applied *within* a tier. Designed for mixed sheets: overarching
    narrative / code context before schedule tables; margin strip last.
    """
    if (placement or "").lower() == "trailing_margin":
        return 100
    if str(kind) == "notes_continuation":
        return -10
    u = re.sub(r"[\s_]+", " ", (title or "")).upper().strip()
    if not u:
        u = ""

    if str(kind) == "abbreviations":
        return 60

    def ovw() -> bool:
        if re.search(
            r"(^|[^A-Z0-9])(GENERAL (NOTE|INFORMATION|N\.?O\.?T\.?E)|KEY NOTE|"
            r"STANDARD NOTE|BASIS OF DESIGN|DESIGN INTENT|PROJECT NARRATIVE)([^A-Z0-9]|$)",
            u,
        ):
            return True
        if re.match(r"^GENERAL NOTE", u) or re.match(r"^GENERAL INFORMATION", u):
            return True
        if "NOT FOR CONSTRUCTION" in u or re.search(
            r"\bN\.?\s*F\.?\s*C\.?\b", u
        ) or re.search(
            r"\bNOT\s+FOR\s+CONST(?:RUCTION|\.)\b", u
        ):
            return True
        return False

    def code() -> bool:
        if re.search(
            r"\b(BUILDING CODE|CODE SUMMARY|LIFE SAFETY|FIRE RATED|FIRE-?RATED|"
            r"FIRE RATING|FIRE-?RATING|OCCUP(AN)?CY( LOAD| LOADS)?|ACCESSIB(ILITY|LE)|"
            r"ENERGY COMPL(IANCE|I)|FIRE EGRESS|ZONING|PERMIT|I\.?B\.?C|"
            r"N\.?F\.?P\.?A|LOCAL (REQUIRE|ORDIN)|CODE COMPL(IANCE|I))\b",
            u,
        ):
            return True
        if "CODE" in u and "SUMMARY" in u:
            return True
        return False

    def index_roster() -> bool:
        if re.search(
            r"\b(PROJECT TEAM|PROJECT DIRECTORY|DRAWING INDEX|LIST OF DRAW(ING|INGS)\b|"
            r"DRAWING LIST|DRAWING SHEET|SHEET (LIST|INDEX|INDEX OF)|\bKEY PLAN|"
            r"COVER SHEET|TABLE OF CONT(ENTS|ENT))\b",
            u,
        ):
            return True
        return False

    def abbr_legend() -> bool:
        return bool(
            re.search(
                r"\b(ABBREVIATION|GLOSSARY|ENCLOSURE LEGEND|SYMBOL LEGEND)\b",
                u,
            ) or re.match(
                r"^LEGEND\b", u
            ) or re.match(
                r"^ABBREVIAT", u
            )
        )

    if kind == "contractor_matrix":
        return 12 if code() else 30
    if ovw():
        return 5
    if code():
        return 12
    if index_roster():
        return 18
    if abbr_legend() and (kind in ("table", "notes", "mccol", "matrix")):
        return 60
    if str(kind) == "mccol":
        return 18
    if str(kind) == "matrix":
        # Uniform grids (SCOPE OF WORK checklists, wide matrices without CYAN).
        return 16
    if str(kind) == "table":
        # High tier: in mixed pages, all schedule tables (tier ≥
        # ``_SCHEDULE_TABLE_TIER``) are emitted **after** prose/mccol/contractor.
        return 50
    if str(kind) == "notes":
        return 25
    return 32


def _section_presentation_sort(
    title: str | None,
    kind: str,
    placement: str | None,
    px_bbox: tuple[float, float, float, float],
) -> list[float]:
    x0, y0, _x1, _y1 = px_bbox
    t = _presentation_tier(
        (title or "").strip(),
        str(kind),
        (placement or "").lower() or None,
    )
    return [float(t), float(y0), float(x0)]


# Tables with tier < this in ``_presentation_sort`` are "early" (title is still
# general / code / index) and stay in reading order *with* narrative blocks. All
# other ``kind: table`` sections (schedules) are **always** sorted after the
# non-table body on mixed sheets so TYPE/wall blocks never open the doc when
# prose or mccol is present.
_SCHEDULE_TABLE_TIER = 30.0


def _section_sort_key(sec: dict[str, Any]) -> tuple[float, float, float, int]:
    p = sec.get("_presentation_sort")
    if (
        isinstance(p, (list, tuple)) and len(p) == 3
        and all(isinstance(n, (int, float)) for n in p)
    ):
        return (float(p[0]), float(p[1]), float(p[2]), 0)
    return (1e6, 0.0, 0.0, 0)


def _is_body_schedule_table(sec: dict[str, Any]) -> bool:
    if sec.get("kind") != "table":
        return False
    p = sec.get("_presentation_sort")
    if not (isinstance(p, (list, tuple)) and len(p) >= 1):
        return True
    return float(p[0]) >= _SCHEDULE_TABLE_TIER


def _sort_sections_by_presentation(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not sections:
        return sections

    body = [s for s in sections if (s.get("placement") or "") != "trailing_margin"]
    margin = [s for s in sections if (s.get("placement") or "") == "trailing_margin"]
    if not body:
        return sorted(margin, key=_section_sort_key)

    abbr_secs = [s for s in body if s.get("kind") == "abbreviations"]
    body_core = [s for s in body if s.get("kind") != "abbreviations"]

    sched = [s for s in body_core if _is_body_schedule_table(s)]
    other = [s for s in body_core if not _is_body_schedule_table(s)]
    if sched and other:
        return (
            sorted(other, key=_section_sort_key)
            + sorted(sched, key=_section_sort_key)
            + sorted(abbr_secs, key=_section_sort_key)
            + sorted(margin, key=_section_sort_key)
        )
    return (
        sorted(body_core, key=_section_sort_key)
        + sorted(abbr_secs, key=_section_sort_key)
        + sorted(margin, key=_section_sort_key)
    )


def _refine_work_order_presentation(sections: list[dict[str, Any]]) -> None:
    """On GC **work order** PDFs, ``mccol`` (tier 18) was always sorted **before**
    ``notes`` (tier 25) even when a notes block sits *between* mccol strips
    (interleaved layout).  Use one presentation tier and keep geometry so
    ``_sort_sections_by_presentation`` orders **top-to-bottom** like the sheet.
    """
    if not sections:
        return
    joined = "\n".join(
        " ".join(
            [
                str(s.get("title") or ""),
                str(s.get("banner_title") or ""),
                str(s.get("section_heading") or ""),
            ]
        )
        for s in sections
    ).upper()
    if "WORK ORDER" not in joined:
        return
    for sec in sections:
        p = sec.get("_presentation_sort")
        if not (isinstance(p, (list, tuple)) and len(p) >= 3):
            continue
        _t, y0, x0 = float(p[0]), float(p[1]), float(p[2])
        # Single tier: reading order follows (y, x) only.
        sec["_presentation_sort"] = [10.0, y0, x0]


def _extract_textsec_note_sections(
    page,
    boxes: list[dict],
    scale: float,
    *,
    rotated_cw: bool,
    page_height_pt: float,
    table_bboxes: list[tuple[float, float, float, float]],
) -> list[dict[str, Any]]:
    """Sections from ``textsec_N_title`` / ``textsec_N_body`` overlays.

    Wrapper iteration alone misses these when CYAN forces sibling strips into
    ``kind: table``. Skips pairs whose title centroid sits inside an emitted
    table slice so header bands / row labels stay with the schedule.
    """
    pad = 6.0
    expanded: list[tuple[float, float, float, float]] = [
        (x0 - pad, y0 - pad, x1 + pad, y1 + pad)
        for (x0, y0, x1, y1) in table_bboxes
    ]

    def _centroid_in_tables(px_bbox: list[float] | tuple[float, ...]) -> bool:
        cx = (float(px_bbox[0]) + float(px_bbox[2])) / 2.0
        cy = (float(px_bbox[1]) + float(px_bbox[3])) / 2.0
        for x0, y0, x1, y1 in expanded:
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                return True
        return False

    pairs: dict[int, dict[str, dict]] = {}
    for b in boxes:
        bid = str(b.get("box_id") or "")
        m = re.match(r"^textsec_(\d+)_title$", bid)
        if m:
            pairs.setdefault(int(m.group(1)), {})["title"] = b
        m2 = re.match(r"^textsec_(\d+)_body$", bid)
        if m2:
            pairs.setdefault(int(m2.group(1)), {})["body"] = b

    out: list[dict[str, Any]] = []
    for n in sorted(pairs.keys()):
        pair = pairs[n]
        tb = pair.get("title")
        if not tb or "px_bbox" not in tb:
            continue
        if expanded and _centroid_in_tables(tb["px_bbox"]):
            continue

        trect = _pdf_rect(
            tb["px_bbox"], scale, pad=0.5,
            rotated_cw=rotated_cw, page_height_pt=page_height_pt,
        )
        raw_title = _text_in_rect(page, trect) or ""
        title_text = _trim_section_title(
            re.sub(r"[\s\u00a0]+", " ", raw_title).strip()
        )

        body_box = pair.get("body")
        body_text = ""
        if body_box and "px_bbox" in body_box:
            brect = _pdf_rect(
                body_box["px_bbox"], scale, pad=0.5,
                rotated_cw=rotated_cw, page_height_pt=page_height_pt,
            )
            body_text = (
                _text_in_rect_preserve_lines(
                    page, brect,
                    rotated_cw=rotated_cw, page_height_pt=page_height_pt,
                )
                or ""
            ).strip()

        if not title_text and not body_text:
            continue
        if title_text and body_text.lower().startswith(title_text.lower()):
            body_text = body_text[len(title_text):].lstrip("\n ").strip()

        sec: dict[str, Any] = {
            "kind": "notes",
            "title": title_text or None,
            "body": body_text,
        }

        if body_box and "px_bbox" in body_box:
            bx = body_box["px_bbox"]
            sx0 = min(tb["px_bbox"][0], bx[0])
            sy0 = min(tb["px_bbox"][1], bx[1])
            sx1 = max(tb["px_bbox"][2], bx[2])
            sy1 = max(tb["px_bbox"][3], bx[3])
        else:
            sx0, sy0, sx1, sy1 = tb["px_bbox"]

        sec["_presentation_sort"] = _section_presentation_sort(
            sec.get("title"),
            "notes",
            sec.get("placement"),
            (float(sx0), float(sy0), float(sx1), float(sy1)),
        )
        out.append(sec)
    return out


def _build_document(data: dict[str, Any], pdf_path: str, page_index: int) -> dict[str, Any]:
    fitz = _import_fitz()
    doc = fitz.open(pdf_path)
    if page_index < 0 or page_index >= len(doc):
        doc.close()
        raise IndexError(f"page {page_index} out of range")
    page = doc[page_index]

    boxes: list[dict] = list(data.get("boxes") or [])
    scale = _render_scale(data)
    rotated_cw = _rotated_cw(data)
    page_w = int(data.get("image_width") or 0)
    page_h = int(data.get("image_height") or 0)
    page_height_pt = float(page.rect.height)

    # Group boxes by role
    v_wrappers = [
        b for b in boxes
        if b.get("color") == "BLUE" and re.fullmatch(r"v\d+", str(b["box_id"]))
    ]
    mccol_groups = [
        b for b in boxes
        if b.get("color") == "BLUE" and re.match(r"^mccol_\d+_group$", str(b.get("box_id", "")))
    ]
    wrappers_all = v_wrappers + mccol_groups
    sec_titles = [b for b in boxes if re.fullmatch(r"v\d+_sec\d+_title", str(b["box_id"]))]
    colhdrs = [
        b for b in boxes
        if str(b["box_id"]).startswith("colhdr_")
        or re.match(r"^mccol_\d+_hdr_\d+$", str(b.get("box_id", "")))
    ]
    purple = [b for b in boxes if (b.get("color") == "PURPLE") or str(b["box_id"]).startswith("titleblk")]
    abbr_cells = [b for b in boxes if str(b["box_id"]).startswith("minitable_") and "mtcelld" in str(b["box_id"])]
    abbr_rows = [b for b in boxes if str(b["box_id"]).startswith("minitable_") and "mtrow" in str(b["box_id"])]
    orange_cells_all = [
        b for b in boxes
        if b.get("color") == "ORANGE"
        and not str(b["box_id"]).startswith("line_repair_")
    ]

    # Page plain text (LLM fallback + document-header source)
    try:
        full_text = _norm(page.get_text("text") or "")
    except Exception:
        full_text = ""

    from orbitbrief_page_os.segmentation.headings.template import (
        resolve_effective_headings,
    )

    _fh, _major_band = resolve_effective_headings(data)

    # Build document header from PURPLE region text (fallback: full page)
    tb_text_parts: list[str] = []
    for p in purple:
        try:
            r = _pdf_rect(
                p["px_bbox"], scale, pad=40.0,
                rotated_cw=rotated_cw, page_height_pt=page_height_pt,
            )
            t = _text_in_rect(page, r)
            if t:
                tb_text_parts.append(t)
        except Exception:
            pass
    tb_wrappers = [w for w in wrappers_all if _is_title_block_wrapper(w, purple)]
    for w in tb_wrappers:
        r = _pdf_rect(
            w["px_bbox"], scale, pad=1.0,
            rotated_cw=rotated_cw, page_height_pt=page_height_pt,
        )
        t = _text_in_rect_preserve_lines(
            page, r, rotated_cw=rotated_cw, page_height_pt=page_height_pt
        )
        if t:
            tb_text_parts.append(t)
    header_text_scope = "\n".join(tb_text_parts) if tb_text_parts else full_text
    header = _document_header_from_text(header_text_scope)
    prose_meta = _prose_header_from_full_text(full_text)
    for k, v in prose_meta.items():
        if k not in header or header.get(k) in (None, ""):
            header[k] = v
    if "date" not in header and prose_meta.get("start_date"):
        header["date"] = prose_meta["start_date"]

    # --- Sections ---------------------------------------------------------
    # A wrapper is title-block if it overlaps a purple stamp OR sits fully in
    # the right-hand sidebar (rightmost ~15% of the page width) where the
    # project / sheet-metadata columns live.
    sidebar_x = 0.85 * page_w

    def is_sidebar(w: dict) -> bool:
        x0, _, x1, _ = w["px_bbox"]
        return x0 >= sidebar_x or x1 - x0 < 0.12 * page_w and x0 >= 0.8 * page_w

    body_wrappers = [
        w for w in wrappers_all
        if not _is_full_page(w, page_w, page_h)
        and not _is_title_block_wrapper(w, purple)
        and not is_sidebar(w)
        # Synthetic tier strips duplicate orange-matrix tables as empty ``mccol``.
        and not (
            re.match(r"^mccol_\d+_group$", str(w.get("box_id", "")))
            and w.get("synthetic") is True
        )
    ]
    body_wrappers.sort(key=lambda w: (w["px_bbox"][1], w["px_bbox"][0]))
    _has_v5 = any(x.get("box_id") == "v5" for x in body_wrappers)

    sections: list[dict[str, Any]] = []
    table_bboxes: list[tuple[float, float, float, float]] = []
    for w in body_wrappers:
        # Master CYAN for this wrapper: colhdrs entirely inside the wrapper.
        wrapper_cyan = [c for c in colhdrs if _bbox_contains(w["px_bbox"], c["px_bbox"])]
        slices = _sub_section_slices(w, sec_titles, boxes)
        for (y_top, y_bot, title_box) in slices:
            sect_px = [w["px_bbox"][0], y_top, w["px_bbox"][2], y_bot]
            if w.get("box_id") == "v6" and w.get("parent_box_id") == "v1" and _has_v5:
                peer = next(
                    (x for x in body_wrappers if x.get("box_id") == "v5"),
                    None,
                )
                if peer is not None:
                    wa, pa = w["px_bbox"], peer["px_bbox"]
                    if abs(wa[1] - pa[1]) < 4.0 and abs(wa[3] - pa[3]) < 4.0:
                        sect_px[0] = min(wa[0], pa[0])
                        sect_px[2] = max(wa[2], pa[2])
            sect_rect = _pdf_rect(
                sect_px, scale, pad=0.5,
                rotated_cw=rotated_cw, page_height_pt=page_height_pt,
            )
            # Prefer cyan physically inside this slice; otherwise **inherit
            # the wrapper's schema** so sibling sub-sections share columns.
            local_cy = [c for c in wrapper_cyan if _bbox_overlaps(sect_px, c["px_bbox"])]
            sect_cy = local_cy if local_cy else wrapper_cyan
            if w.get("box_id") == "v6" and w.get("parent_box_id") == "v1":
                # v6 can carry spurious CYAN (decorative sub-columns); this block is
                # always the 3-column responsibility matrix, not a data table.
                sect_cy = []

            # Section title
            title_text = ""
            if title_box is not None:
                title_text = _text_in_rect(
                    page,
                    _pdf_rect(
                        title_box["px_bbox"], scale, pad=1.0,
                        rotated_cw=rotated_cw, page_height_pt=page_height_pt,
                    ),
                )
            if not title_text:
                lines = _text_in_rect_preserve_lines(
                    page, sect_rect,
                    rotated_cw=rotated_cw, page_height_pt=page_height_pt,
                ).splitlines()
                for ln in lines:
                    t = ln.strip()
                    if t and t.upper() == t and len(t) >= 4:
                        title_text = t
                        break
                if not title_text and lines:
                    title_text = lines[0]
            title_text = _trim_section_title(title_text)
            title_text = _infer_gc_work_order_matrix_banner_title(
                page, sect_px, scale,
                rotated_cw=rotated_cw, page_height_pt=page_height_pt,
                current=title_text,
            )
            m_w = re.match(r"^mccol_(\d+)_group$", str(w.get("box_id", "")))
            if m_w:
                hid = f"mccol_{m_w.group(1)}_heading"
                for hx in boxes:
                    if str(hx.get("box_id")) == hid and str(
                        hx.get("parent_box_id")
                    ) == str(w.get("box_id")):
                        hread = _text_in_rect(
                            page,
                            _pdf_rect(
                                hx["px_bbox"], scale, pad=0.5,
                                rotated_cw=rotated_cw, page_height_pt=page_height_pt,
                            ),
                        )
                        hread = re.sub(r"[\s\u00a0]+", " ", (hread or "")).strip()
                        if hread and hread.upper() not in (title_text or "").upper():
                            title_text = (
                                f"{title_text} — {hread}"
                                if title_text
                                else hread
                            )
                        break

            # Data region: skip the sub-section's own title row
            data_top_px = y_top
            if title_box is not None:
                data_top_px = max(data_top_px, title_box["px_bbox"][3] + 2)

            kind: str
            payload: dict[str, Any]
            _row_anchors: list[list[dict | None]] = []
            extra_matrix_sections: list[dict[str, Any]] = []
            matrix_primary_chunk_top: float | None = None

            def _fallback_notes_or_matrix() -> None:
                nonlocal kind, payload, title_text, matrix_primary_chunk_top
                matrix_primary_chunk_top = None
                extra_matrix_sections.clear()
                ums = _try_uniform_matrices_from_oranges(
                    page, sect_px, orange_cells_all, scale,
                    rotated_cw=rotated_cw, page_height_pt=page_height_pt,
                    data_y_top_px=data_top_px,
                )
                scope_like = _section_has_scope_heading_text(
                    title_text, page, sect_px, scale,
                    rotated_cw=rotated_cw, page_height_pt=page_height_pt,
                )
                ncol = len(ums[0]["columns"]) if ums else 0
                qty_sched = (
                    ums
                    and _uniform_matrix_columns_look_like_qty_table(
                        ums[0].get("columns")
                    )
                )
                if ums and (
                    scope_like or ncol >= 8 or (ncol >= 2 and qty_sched)
                ):
                    kind = "matrix"
                    payload = {"columns": ums[0]["columns"], "grid": ums[0]["grid"]}
                    matrix_primary_chunk_top = float(ums[0]["_chunk_px_top"])
                    st = _extract_scope_section_title(
                        page, sect_px, scale,
                        rotated_cw=rotated_cw, page_height_pt=page_height_pt,
                    )
                    if st:
                        title_text = st
                    if len(ums) > 1:
                        inter = _extract_matrix_interstitial_notes(
                            page, sect_px, scale,
                            y_gap_top=ums[0]["_chunk_px_bot"] + 0.5,
                            y_gap_bottom=ums[1]["_chunk_px_top"] - 0.5,
                            rotated_cw=rotated_cw,
                            page_height_pt=page_height_pt,
                        )
                        if inter:
                            payload["notes"] = inter
                    for i in range(1, len(ums)):
                        prev = ums[i - 1]
                        um = ums[i]
                        tit = _infer_uniform_matrix_follow_on_title(
                            page, sect_px, scale,
                            y_gap_top=prev["_chunk_px_bot"] + 1.0,
                            y_gap_bottom=um["_chunk_px_top"] - 1.0,
                            rotated_cw=rotated_cw,
                            page_height_pt=page_height_pt,
                        )
                        xd: dict[str, Any] = {
                            "title": tit or None,
                            "columns": um["columns"],
                            "grid": um["grid"],
                            "_chunk_px_top": float(um["_chunk_px_top"]),
                        }
                        if i + 1 < len(ums):
                            notes_below = _extract_matrix_interstitial_notes(
                                page, sect_px, scale,
                                y_gap_top=um["_chunk_px_bot"] + 0.5,
                                y_gap_bottom=ums[i + 1]["_chunk_px_top"] - 0.5,
                                rotated_cw=rotated_cw,
                                page_height_pt=page_height_pt,
                            )
                            if notes_below:
                                xd["notes"] = notes_below
                        extra_matrix_sections.append(xd)
                    if len(ums) == 2:
                        tail = _extract_tail_notes_before_next_heading(
                            page, sect_px, scale,
                            y_top_px=ums[1]["_chunk_px_bot"] + 0.5,
                            page_h_px=page_h,
                            rotated_cw=rotated_cw,
                            page_height_pt=page_height_pt,
                        )
                        if tail and extra_matrix_sections:
                            x0 = extra_matrix_sections[-1]
                            x0["notes"] = _join_matrix_notes(x0.get("notes"), tail)
                    elif len(ums) == 1:
                        tail = _extract_tail_notes_before_next_heading(
                            page, sect_px, scale,
                            y_top_px=ums[0]["_chunk_px_bot"] + 0.5,
                            page_h_px=page_h,
                            rotated_cw=rotated_cw,
                            page_height_pt=page_height_pt,
                        )
                        if tail:
                            payload["notes"] = _join_matrix_notes(
                                payload.get("notes"), tail
                            )
                    return
                kind = "notes"
                payload = {
                    "body": _notes_body_reading_order(
                        page, sect_rect, title_text,
                        rotated_cw=rotated_cw, page_height_pt=page_height_pt,
                    ),
                }

            if sect_cy:
                tbl = _build_table_from_cells(
                    page, sect_px, sect_cy, orange_cells_all, scale,
                    rotated_cw=rotated_cw, page_height_pt=page_height_pt,
                    data_y_top_px=data_top_px,
                )
                if tbl and (any(h for h in tbl["headers"]) or tbl["rows"]):
                    kind = "table"
                    # Promote the leftmost column to a dedicated *symbol* slot
                    # (image + text abbrev) and keep the rest as data columns.
                    # Each record may carry multiple variants (sub-rows) when
                    # cells stack inside a data column — e.g. a combo outlet
                    # listing a DATA row and a FIBER row under one symbol.
                    symbol_col_label = (tbl["headers"][0] if tbl["headers"] else "")
                    headers_body = tbl["headers"][1:] if tbl["headers"] else []
                    rows_body: list[dict[str, Any]] = []
                    anchor_list: list[list[dict | None]] = []
                    for r in tbl["rows"]:
                        vc = r["variant_count"]
                        variants = r["variants"]
                        row_dict: dict[str, Any] = {
                            "symbol": {
                                "label": r["symbol_text"],
                                "image": None,
                            },
                            "variant_count": vc,
                            "variant_label": _variant_word(vc),
                        }
                        if vc == 1 and variants:
                            row_dict["cells"] = variants[0]
                        else:
                            row_dict["variants"] = [
                                {"cells": v} for v in variants
                            ]
                        rows_body.append(row_dict)
                        anchor_list.append([r["symbol_anchor"]])
                    _row_anchors = anchor_list
                    payload = {
                        "symbol_column_label": symbol_col_label,
                        "headers": headers_body,
                        "rows": rows_body,
                    }
                else:
                    _fallback_notes_or_matrix()
            else:
                _fallback_notes_or_matrix()

            m_g = re.match(r"^mccol_(\d+)_group$", str(w.get("box_id", "")))
            if m_g and kind == "notes":
                mb = _extract_mccol_column_boxes(
                    page, w, boxes, scale,
                    rotated_cw=rotated_cw, page_height_pt=page_height_pt,
                )
                if mb:
                    ban, sh = _split_banner_and_section_heading(title_text)
                    kind = "mccol"
                    line_parts: list[str] = []
                    for bx in mb:
                        lab = (bx.get("label") or "").strip() or (
                            f"Column {bx.get('id', '')}"
                        )
                        btxt = (bx.get("body") or "").strip()
                        line_parts.append(f"**{lab}**\n\n{btxt}".strip() if btxt else f"**{lab}**")
                    body_joined = "\n\n---\n\n".join(line_parts)
                    payload = {
                        "banner_title": ban,
                        "section_heading": sh,
                        "boxes": mb,
                        "body": body_joined,
                    }

            # Strip the title text off the body (avoid duplication)
            if kind == "notes" and payload.get("body") and title_text:
                body = payload["body"]
                if body.lower().startswith(title_text.lower()):
                    body = body[len(title_text):].lstrip("\n ")
                    payload["body"] = body

            section_entry = {
                "kind": kind,
                "title": title_text or None,
                **payload,
            }
            m_cc = re.match(r"^mccol_(\d+)_group$", str(w.get("box_id", "")))
            if m_cc:
                section_entry["mccol_group_id"] = str(w.get("box_id"))
                section_entry["mccol_index"] = int(m_cc.group(1))
            _apply_responsibility_matrix(section_entry, page, sect_rect)
            # Stash internal anchors for later symbol extraction
            if section_entry.get("kind") == "table" and _row_anchors:
                section_entry["_row_anchors"] = _row_anchors
            out_sec = _demote_table_if_prose(section_entry)
            if _should_skip_tabular_notes_noise(out_sec):
                continue
            if out_sec.get("kind") == "table":
                table_bboxes.append(
                    (
                        float(sect_px[0]),
                        float(sect_px[1]),
                        float(sect_px[2]),
                        float(sect_px[3]),
                    )
                )
            sx0, sy0, sx1, sy1 = (
                float(sect_px[0]),
                float(sect_px[1]),
                float(sect_px[2]),
                float(sect_px[3]),
            )
            ps_bbox = (
                sx0,
                float(matrix_primary_chunk_top)
                if (
                    out_sec.get("kind") == "matrix"
                    and matrix_primary_chunk_top is not None
                )
                else sy0,
                sx1,
                sy1,
            )
            out_sec["_presentation_sort"] = _section_presentation_sort(
                out_sec.get("title") or title_text or None,
                str(out_sec.get("kind") or "notes"),
                out_sec.get("placement"),
                ps_bbox,
            )
            sections.append(out_sec)
            for xm in extra_matrix_sections:
                ex_top = xm.pop("_chunk_px_top", None)
                notes_x = xm.pop("notes", None)
                x_sec: dict[str, Any] = {
                    "kind": "matrix",
                    "title": xm.get("title"),
                    "columns": xm["columns"],
                    "grid": xm["grid"],
                }
                if notes_x:
                    x_sec["notes"] = notes_x
                _apply_responsibility_matrix(x_sec, page, sect_rect)
                x_ps_bbox = (
                    sx0,
                    float(ex_top) if ex_top is not None else sy0,
                    sx1,
                    sy1,
                )
                x_sec["_presentation_sort"] = _section_presentation_sort(
                    x_sec.get("title"),
                    "matrix",
                    out_sec.get("placement"),
                    x_ps_bbox,
                )
                sections.append(_demote_table_if_prose(x_sec))

    sections.extend(
        _extract_textsec_note_sections(
            page,
            boxes,
            scale,
            rotated_cw=rotated_cw,
            page_height_pt=page_height_pt,
            table_bboxes=table_bboxes,
        )
    )

    # --- Minitable abbreviations (global + scoped) ---------------------------
    has_mccol = any(
        re.match(r"^mccol_\d+_group$", str(b.get("box_id", "")))
        for b in boxes
    )
    abbr_out = _build_abbreviations(
        page, abbr_cells, abbr_rows, scale,
        rotated_cw=rotated_cw, page_height_pt=page_height_pt,
        all_boxes=boxes if has_mccol else None,
    )
    if isinstance(abbr_out, tuple):
        abbreviations, abbr_scoped = abbr_out
    else:
        abbreviations = abbr_out
        abbr_scoped = None
    if not isinstance(abbreviations, list):
        abbreviations = []
    if abbr_scoped is None:
        abbr_scoped = {}

    if abbreviations:
        y0a = 0.0
        x0a = 0.0
        if abbr_rows:
            y0a = min(r["px_bbox"][1] for r in abbr_rows)
            x0a = min(r["px_bbox"][0] for r in abbr_rows)
        ab_sec: dict[str, Any] = {
            "kind": "abbreviations",
            "title": "Abbreviations",
            "entries": list(abbreviations),
        }
        ab_sec["_presentation_sort"] = _section_presentation_sort(
            "Abbreviations", "abbreviations", None, (x0a, y0a, x0a + 1, y0a + 1)
        )
        sections.append(ab_sec)

    # Right-hand title-block strip: lift the architectural ``TITLE:`` line into
    # ``document.sheet_title`` and drop the rest. Firm / scale / job are pure
    # title-block metadata; emitting them as a body section was just noise that
    # downstream LLMs treated as a real "notes" block. (Removed Apr 2026.)
    margin_w = [
        w
        for w in v_wrappers
        if is_sidebar(w)
        and not _is_full_page(w, page_w, page_h)
        and not _is_title_block_wrapper(w, purple)
    ]
    if margin_w:
        margin_w.sort(key=lambda w: (w["px_bbox"][1], w["px_bbox"][0]))
        m_parts: list[str] = []
        for mw in margin_w:
            r = _pdf_rect(
                mw["px_bbox"], scale, pad=0.5,
                rotated_cw=rotated_cw, page_height_pt=page_height_pt,
            )
            t = _text_in_rect_preserve_lines(
                page, r,
                rotated_cw=rotated_cw, page_height_pt=page_height_pt,
            ).strip()
            if t:
                m_parts.append(t)
        if m_parts:
            raw_m = "\n\n---\n\n".join(m_parts)
            _margin_body_unused, block_title = _split_title_line_from_margin(raw_m)
            if block_title:
                header["sheet_title"] = block_title
            # NOTE: no section is appended; ``trailing_margin`` is intentionally
            # not part of ``sections`` anymore. If a caller still wants the raw
            # margin text, ``full_text`` on the root remains available.

    # Cover / RFP title page: detector emits ``tbstruct_*`` + ``tbtext_*`` but no
    # ``v\\d+`` wrappers, so nothing clips the boxed title. Reconstruct lines from
    # the cell (rotated pages need word clustering) and split title vs subtitles.
    # Skip when many tbstruct cells exist (e.g. drawing sidebar) — that is not a
    # single-box cover and would concatenate unrelated strips.  Page 0 allows up
    # to two legacy margin cells; centered covers often suppress those cells and
    # rely on full-page text instead.
    tb_struct_n = sum(
        1
        for b in boxes
        if re.fullmatch(r"tbstruct_\d+", str(b.get("box_id") or ""))
    )
    cover_tb_ok = tb_struct_n <= 1 or (page_index == 0 and tb_struct_n <= 2)
    if not body_wrappers and cover_tb_ok:
        cover_raw = _extract_tbstruct_cover_text(
            page,
            boxes,
            scale,
            rotated_cw=rotated_cw,
            page_height_pt=page_height_pt,
        )
        if len(cover_raw.strip()) < 40 and page_index == 0:
            try:
                alt = (page.get_text("text") or "").strip()
            except Exception:
                alt = ""
            if len(alt) > len(cover_raw.strip()):
                cover_raw = alt
        if cover_raw.strip():
            c_title, c_subs = _parse_cover_page_title_subtitles(cover_raw)
            tail = _cover_page_tail_after_cell(page, cover_raw)
            if tail:
                for ex in _split_cover_tail_segments(tail):
                    exs = ex.strip()
                    if not exs or exs == c_title or exs in c_subs:
                        continue
                    c_subs.append(exs)
            c_subs = [_polish_cover_year_line(x) for x in c_subs]
            if c_title:
                header["title"] = c_title
            if c_subs:
                header["subtitles"] = c_subs
            if c_title or c_subs:
                full_text = "\n".join(
                    [x for x in ([c_title] if c_title else []) + c_subs if x]
                )

    if not sections and full_text:
        rfp_secs = _structure_introduction_project_scope_sections(full_text)
        if rfp_secs:
            sections.extend(rfp_secs)
            header["structured_layout"] = "introduction_project_scope"

    if not sections and page_index > 0 and full_text.strip():
        cmeta, more_secs = _maybe_tsc_continuation_and_followon_sections(
            full_text,
            doc,
            page_index,
            pdf_path,
            followon_headings=_fh,
            major_band=_major_band,
        )
        if more_secs:
            sections.extend(more_secs)
            if cmeta:
                header["continuation"] = cmeta
            header["structured_layout"] = (
                header.get("structured_layout") or "tsc_followon"
            )

    if sections and page_index > 0:
        _annotate_tsc_subtitle_under_parent_major(
            sections, doc, page_index, pdf_path
        )

    _drop_duplicate_matrix_fingerprints(sections)
    _refine_work_order_presentation(sections)
    sections = _sort_sections_by_presentation(sections)

    if sections:
        _hoist_run_in_subtitles_all_sections(sections)
        _merge_tsc_same_page_major_with_child_sections(sections)

    if abbr_scoped:
        _apply_scoped_abbreviations_to_mccol_sections(sections, abbr_scoped)

    # Stash info the writer will need to crop symbol images next to the MD/JSON
    # artifacts. The anchor list is popped in write_extraction_artifacts so the
    # final JSON stays coordinate-free.
    extraction_internals = {
        "page_ref": page,  # keep page open until artifacts are written
        "doc_ref": doc,
        "scale": scale,
        "rotated_cw": rotated_cw,
        "page_height_pt": page_height_pt,
    }

    # --- Build output ----------------------------------------------------
    out: dict[str, Any] = {
        "document": {
            "source_pdf": pdf_path,
            "page_index": page_index,
            **{
                k: header[k]
                for k in (
                    "sheet_number",
                    "sheet_title",
                    "title",
                    "subtitles",
                    "project",
                    "client",
                    "architect",
                    "date",
                    "job_number",
                    "phone",
                    "work_order_title",
                    "site_address",
                    "total_units",
                    "start_date",
                    "header_block",
                    "structured_layout",
                    "continuation",
                )
                if k in header
            },
        },
        "sections": sections,
        "abbreviations": abbreviations,
        "full_text": full_text,
        "_internals": extraction_internals,
    }
    try:
        from orbitbrief_page_os.segmentation.headings.compose import (
            attach_heading_analysis,
        )

        attach_heading_analysis(
            out,
            page,
            full_text,
            data,
            followon_used=_fh,
            major_used=_major_band,
            page_index=page_index,
        )
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# Responsibility / contractor matrix (3-column bullet blocks + NOTE)
# ---------------------------------------------------------------------------


_RM_HEADERS = [
    (r"CABLING\s+CONTRACTOR\s+TO\s+PROVIDE", "cabling_contractor"),
    (r"SECURITY\s+CONTRACTOR\s+TO\s+PROVIDE", "security_contractor"),
    (r"ELECTRICAL\s+CONTRACTOR\s+TO\s+PROVIDE", "electrical_contractor"),
]


def _first_electrical_bullet_index(items: list[str]) -> int | None:
    """When the matrix omits the electrical column header in the text stream, its
    bullets are often concatenated to the security column. Split at the first
    line that clearly belongs in the electrical scope."""
    for i, it in enumerate(items):
        u = it.upper()
        if re.match(r"^4X4\b", it, re.I):
            return i
        if re.match(r"^KNOX\s+BOX", u):
            return i
        if "MISCELLANEOUS" in u and "LABELS" in u and "CONDUIT" in u:
            return i
    return None


def _split_merged_security_electrical(
    columns: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if any(c.get("id") == "electrical_contractor" for c in columns):
        return columns
    out = []
    for c in columns:
        if c.get("id") != "security_contractor":
            out.append(c)
            continue
        items = list(c.get("items") or [])
        j = _first_electrical_bullet_index(items)
        if j is not None and j > 0:
            out.append({**c, "items": items[:j]})
            out.append(
                {
                    "id": "electrical_contractor",
                    "heading": "ELECTRICAL CONTRACTOR TO PROVIDE",
                    "items": items[j:],
                }
            )
        else:
            out.append(c)
    return out


def _normalize_bullet_text(s: str) -> str:
    s = s.replace("\r", " ").replace("\n", " ")
    s = re.sub(r"[\s\u00a0]+", " ", s).strip()
    return s


def _split_bullet_items(chunk: str) -> list[str]:
    """Split a column body into list items (PDF bullets: •, , U+f0b7, etc.)."""
    if not chunk or not chunk.strip():
        return []
    t = chunk
    for u in ("\uf0b7", "", "\u2022", "·", "\u25cf", "\u25aa"):
        t = t.replace(u, "•")
    # Collapse runs of the same bullet char
    t = re.sub(r"(•\s*)+", "•", t)
    parts = re.split(r"•", t)
    out: list[str] = []
    for p in parts:
        p = _normalize_bullet_text(p)
        if len(p) < 2:
            continue
        # Drop stray header fragments
        if re.match(r"^(CABLING|SECURITY|ELECTRICAL)\s+CONTRACTOR\b", p, re.I):
            continue
        out.append(p)
    return out


def _kmeans1d_three_columns(xs: list[float]) -> list[int]:
    """Assign each x to 0,1,2 (left → right) using 1D k-means; returns per-x labels."""
    n = len(xs)
    if n < 3:
        return [0] * n
    lo, hi = min(xs), max(xs)
    span = max(1e-3, hi - lo)
    centers = [lo + (i + 0.5) * span / 3.0 for i in range(3)]
    labels = [0] * n
    for _ in range(30):
        for i, x in enumerate(xs):
            best_j = 0
            best_d = 1e18
            for j in range(3):
                d = (x - centers[j]) ** 2
                if d < best_d:
                    best_d = d
                    best_j = j
            labels[i] = best_j
        new_c: list[float] = []
        for j in range(3):
            cl = [xs[i] for i in range(n) if labels[i] == j]
            new_c.append(sum(cl) / len(cl) if cl else centers[j])
        new_c.sort()
        if all(abs(new_c[j] - centers[j]) < 0.2 for j in range(3)):
            break
        centers = new_c
    order = sorted(range(3), key=lambda j: centers[j])
    remap = {order[j]: j for j in range(3)}
    return [remap[labels[i]] for i in range(n)]


def _parse_responsibility_matrix_spatial(page, sect_rect) -> dict[str, Any] | None:
    """When string parsing fails, assign words to three x-clusters (k-means) and
    read each column top-to-bottom. Excludes the NOTE: footer band by y."""
    try:
        words = page.get_text("words", clip=sect_rect) or []
    except Exception:
        return None
    if len(words) < 8:
        return None
    recs: list[tuple[float, float, float, float, str]] = []
    for w in words:
        if len(w) < 5:
            continue
        x0, y0, x1, y1 = float(w[0]), float(w[1]), float(w[2]), float(w[3])
        tx = str(w[4]).strip()
        if not tx:
            continue
        recs.append((x0, y0, x1, y1, tx))
    if not recs:
        return None
    ymin = min((r[1] + r[3]) / 2.0 for r in recs)
    ymax = max((r[1] + r[3]) / 2.0 for r in recs)
    y_note_top: float | None = None
    for r in recs:
        if re.match(r"^NOTE:?$", r[4], re.I):
            y_note_top = r[1]
            break
    if y_note_top is not None:
        recs = [r for r in recs if (r[1] + r[3]) / 2.0 < y_note_top - 0.5]
    if len(recs) < 6:
        return None
    xcs = [(r[0] + r[2]) / 2.0 for r in recs]
    labels = _kmeans1d_three_columns(xcs)
    col_words: list[list[tuple[float, str]]] = [[], [], []]
    for r, col_i in zip(recs, labels):
        yc = (r[1] + r[3]) / 2.0
        col_words[col_i].append((yc, r[4]))
    head = (
        "CABLING CONTRACTOR TO PROVIDE",
        "SECURITY CONTRACTOR TO PROVIDE",
        "ELECTRICAL CONTRACTOR TO PROVIDE",
    )
    columns: list[dict[str, Any]] = []
    for ci in range(3):
        col_words[ci].sort(key=lambda t: t[0])
        blob = " ".join(t[1] for t in col_words[ci])
        hb = re.sub(_RM_HEADERS[ci][0], " ", blob, count=1, flags=re.I)
        items = _split_bullet_items(hb)
        if not items:
            continue
        columns.append(
            {
                "id": _RM_HEADERS[ci][1],
                "heading": head[ci],
                "items": items,
            }
        )
    if len(columns) < 2:
        return None
    footer_note = ""
    try:
        all_words = page.get_text("words", clip=sect_rect) or []
    except Exception:
        all_words = words
    foot_recs: list[tuple[float, float, float, float, str]] = []
    for w in all_words:
        if len(w) < 5:
            continue
        x0, y0, x1, y1 = float(w[0]), float(w[1]), float(w[2]), float(w[3])
        tx = str(w[4]).strip()
        if not tx:
            continue
        foot_recs.append((x0, y0, x1, y1, tx))
    y_lo = ymax - 0.18 * max(1.0, ymax - ymin) if foot_recs else 0.0
    bottom = " ".join(
        t[4]
        for t in sorted(
            (r for r in foot_recs if (r[1] + r[3]) / 2.0 >= y_lo),
            key=lambda r: (r[1], r[0]),
        )
    )
    m_fn = re.search(
        r"\bNOTE:\s*(ULTIMATELY\s+THE\s+GENERAL\s+CONTRACTOR\b[^.]*\.)",
        bottom,
        re.I,
    )
    if m_fn:
        footer_note = _normalize_bullet_text(m_fn.group(1).strip())
    return {
        "layout": "three_column_bullets",
        "columns": columns,
        "footer_note": footer_note,
    }


def _parse_contractor_matrix(body: str) -> dict[str, Any] | None:
    """Parse a *Responsibility matrix* block: up to three contractor columns with
    bullet lists + optional ``NOTE:`` footer. Does not require CYAN/orange.

    If the CABLING header is missing but bullets appear before the SECURITY
    header, the leading span is treated as the cabling column.
    """
    if not body or not body.strip():
        return None
    t = body.strip()
    t = re.sub(r"^RESPONSIBILITY\s+MATRIX\s+", "", t, flags=re.I)
    # PDFs often break headers across lines; one line for header / column discovery.
    t_flat = re.sub(r"\s+", " ", t)

    footer_note = ""
    m_fn = re.search(
        r"\bNOTE:\s*(ULTIMATELY\s+THE\s+GENERAL\s+CONTRACTOR\b[^.]*\.)",
        t_flat,
        re.I,
    )
    if m_fn:
        footer_note = _normalize_bullet_text(m_fn.group(1).strip())

    # Column text is only the block *above* ``NOTE:``; the PDF text stream after
    # the matrix often repeats headers or the next section title and must not
    # merge into the last column.
    m_note = re.search(r"\bNOTE:\s*", t_flat, re.I)
    if m_note:
        t_main = t_flat[: m_note.start()].strip()
    else:
        t_main = t_flat
    t_main = re.sub(r"\s+", " ", t_main)

    pos: list[tuple[int, int, str, str]] = []
    for pat, rid in _RM_HEADERS:
        m = re.search(pat, t_main, re.I)
        if m:
            pos.append((m.start(), m.end(), rid, m.group(0).strip()))
    if not pos:
        return None
    pos.sort(key=lambda x: x[0])

    columns: list[dict[str, Any]] = []

    # Leading bullets (no CABLING line in the text stream)
    if pos[0][0] > 8 and pos[0][2] != "cabling_contractor":
        lead = t_main[: pos[0][0]]
        lead_items = _split_bullet_items(lead)
        if lead_items:
            columns.append(
                {
                    "id": "cabling_contractor",
                    "heading": "CABLING CONTRACTOR TO PROVIDE",
                    "items": lead_items,
                }
            )

    for i, (_s, e0, rid, heading) in enumerate(pos):
        chunk_end = pos[i + 1][0] if i + 1 < len(pos) else len(t_main)
        chunk = t_main[e0:chunk_end]
        items = _split_bullet_items(chunk)
        if not items:
            continue
        columns.append(
            {
                "id": rid,
                "heading": " ".join(heading.split()),
                "items": items,
            }
        )

    if len(columns) < 2:
        return None
    columns = _split_merged_security_electrical(columns)
    if len(columns) < 2:
        return None
    return {
        "layout": "three_column_bullets",
        "columns": columns,
        "footer_note": footer_note,
    }


def _apply_responsibility_matrix(
    section: dict[str, Any],
    page: Any = None,
    sect_rect: Any = None,
) -> None:
    """In-place: upgrade ``notes`` + *Responsibility matrix* into ``contractor_matrix``."""
    if section.get("kind") != "notes":
        return
    title = section.get("title")
    title_u = (title or "").upper()
    if "RESPONSIBILITY" not in title_u or "MATRIX" not in title_u:
        return
    body = (section.get("body") or "").strip()
    parsed = _parse_contractor_matrix(body)
    if (not parsed or len(parsed.get("columns") or []) < 2) and page is not None and sect_rect is not None:
        parsed = _parse_responsibility_matrix_spatial(page, sect_rect)
    if not parsed or len(parsed.get("columns") or []) < 2:
        return
    section.clear()
    section.update(
        {
            "kind": "contractor_matrix",
            "title": title,
            "layout": parsed["layout"],
            "columns": parsed["columns"],
            "footer_note": parsed.get("footer_note") or "",
        }
    )


# ---------------------------------------------------------------------------
# Markdown rendering (content-only)
# ---------------------------------------------------------------------------


def _trim_section_title(t: str) -> str:
    """Cut section titles at the first column-header phrase so we don't carry
    the header row into the heading (e.g. ``SYMBOL DESCRIPTION CABLE COUNT``).
    """
    if not t:
        return t
    t = re.sub(r"\s+", " ", t).strip()
    # Common header-row starters that indicate end of the real section title
    cut_markers = [
        "SYMBOL DESCRIPTION", "SYMBOL  DESCRIPTION",
        "SECURITY CONTRACTOR TO PROVIDE", "CABLING CONTRACTOR TO PROVIDE",
        "ELECTRICAL CONTRACTOR TO PROVIDE",
        "DESCRIPTION CABLE",
    ]
    for marker in cut_markers:
        idx = t.find(marker)
        if idx > 4:
            t = t[:idx].rstrip(" -,.")
            break
    if len(t) > 80:
        t = t[:77].rstrip() + "…"
    return t


def _row_variants(r) -> list[list[str]]:
    """Normalise either row shape (``cells`` or ``variants``) to a list of
    variant cell-lists. Single-variant rows return a 1-item list.
    """
    if not isinstance(r, dict):
        return [r]
    if "variants" in r and r["variants"]:
        return [(v.get("cells") or []) for v in r["variants"]]
    return [r.get("cells") or []]


def _demote_table_if_prose(section: dict) -> dict:
    """Tables whose rows concatenate into natural-language prose are demoted
    to ``notes``.
    """
    if section.get("kind") != "table":
        return section
    rows = section.get("rows") or []
    if not rows:
        return section

    # Symbol legend sheets (CCTV, structured cabling, etc.) use long sentences
    # in cells. The short-cell heuristic below mis-classifies them as prose and
    # strips table rows — which also drops symbol-image extraction.
    title_u = (section.get("title") or "").upper()
    if "SYMBOL LEGEND" in title_u:
        return section

    meaningful_rows = 0
    for r in rows:
        for cs in _row_variants(r):
            filled = [c for c in cs if c and c.strip()]
            if not filled:
                continue
            short = sum(1 for c in filled if len(c.split()) <= 3 and len(c) <= 20)
            if short >= max(3, int(0.4 * len(filled))):
                meaningful_rows += 1
                break
    if meaningful_rows < max(1, int(0.5 * len(rows))):
        body_lines: list[str] = []
        for r in rows:
            label = ""
            if isinstance(r, dict):
                label = (r.get("symbol") or {}).get("label") or ""
            for cs in _row_variants(r):
                tokens = ([label] if label else []) + [c for c in cs if c]
                body_lines.append(" ".join(tokens).strip())
        return {
            "kind": "notes",
            "title": section.get("title"),
            "body": "\n".join(body_lines).strip(),
        }
    return section


def _md_cell(s: str) -> str:
    return (s or "").replace("\n", " ").replace("|", "\\|").strip()


def _render_markdown(d: dict[str, Any]) -> str:
    L: list[str] = []
    doc = d.get("document") or {}
    cover_title = (doc.get("title") or "").strip()
    cover_subs = [
        str(x).strip()
        for x in (doc.get("subtitles") or [])
        if isinstance(x, str) and str(x).strip()
    ]
    sn = doc.get("sheet_number") or ""
    st = doc.get("sheet_title") or ""
    if cover_title:
        L.append(f"# {_md_cell(cover_title)}\n\n")
        if cover_subs:
            L.append("**Subtitles**\n\n")
            for line in cover_subs:
                L.append(f"- {_md_cell(line)}\n")
            L.append("\n")
    elif sn and st:
        L.append(f"# Sheet {sn} — {st}\n\n")
    elif sn:
        L.append(f"# Sheet {sn}\n\n")
    elif st:
        L.append(f"# {st}\n\n")
    else:
        L.append("# Sheet\n\n")

    kv_rows = []
    for label, key in (
        ("Work order", "work_order_title"),
        ("Site / address", "site_address"),
        ("Total units", "total_units"),
        ("Start date", "start_date"),
        ("Project", "project"),
        ("Client", "client"),
        ("Architect", "architect"),
        ("Date", "date"),
        ("Job Number", "job_number"),
        ("Phone", "phone"),
    ):
        v = doc.get(key)
        if v:
            kv_rows.append(f"- **{label}:** {v}")
    if kv_rows:
        L.append("\n".join(kv_rows) + "\n\n")

    # Sections: presentation order (title tier, then y/x), not raw detection order
    sections = d.get("sections") or []
    for i, s in enumerate(sections, 1):
        kind = s.get("kind")
        if kind == "mccol" and (s.get("section_heading") or "").strip():
            title = s["section_heading"].strip()
        elif kind == "notes_continuation":
            cf = s.get("continues_from") or {}
            st = cf.get("section_title") or "prior section"
            pidx = cf.get("page_index")
            title = (
                f"Continuation (after page {int(pidx) + 1} — «{st}»)"
                if pidx is not None
                else f"Continuation — «{st}»"
            )
        else:
            title = s.get("title") or f"Section {i}"
        pm = (s.get("parent_major_section") or "").strip()
        sub_here = (s.get("subtitle") or "").strip()
        if kind == "notes_continuation":
            L.append(f"## {_md_cell(str(title))}\n\n")
        elif pm and str(kind) == "notes":
            L.append(f"## {_md_cell(pm)}\n\n")
            L.append(f"### {_md_cell(str(title))}\n\n")
            cf0 = s.get("continues_from") or {}
            mk0 = str(cf0.get("merge_key") or "")
            if mk0:
                L.append(
                    f"> **Hierarchy** (`{mk0}`): subsection «{_md_cell(str(title))}» under major "
                    f"«{_md_cell(pm)}» (PDF `page_index` **{cf0.get('page_index', '')}**). "
                    f"{cf0.get('reader_note', '')}\n\n"
                )
        elif sub_here and str(kind) == "notes":
            L.append(f"## {_md_cell(str(title))}\n\n")
            L.append(f"### {_md_cell(sub_here)}\n\n")
        else:
            title_m = _md_cell(str(title))
            if str(kind) == "notes" and str(s.get("heading_tier") or "") == "major":
                L.append(f"# {title_m}\n\n")
            else:
                L.append(f"## {title_m}\n\n")
        if kind == "notes_continuation":
            cf = s.get("continues_from") or {}
            mk = cf.get("merge_key", "")
            L.append(
                f"> **Cross-page link** (`{mk}`): attach to section "
                f"«{cf.get('section_title', '')}» on PDF `page_index` **{cf.get('page_index')}**. "
                f"{cf.get('reader_note', '')}\n\n"
            )
            paras = s.get("paragraphs") or []
            for p in paras:
                ps = str(p).strip()
                if ps:
                    L.append(ps + "\n\n")
            bt_cont = s.get("bullet_tree")
            if isinstance(bt_cont, list) and bt_cont:
                trc = _render_bullet_tree_md(bt_cont, 0)
                if trc.strip():
                    L.append(trc + "\n\n")
            continue
        if kind == "abbreviations":
            for e in s.get("entries") or []:
                if not isinstance(e, dict):
                    continue
                sym = e.get("symbol", "")
                mean = e.get("meaning", "")
                if sym and mean:
                    L.append(f"- **{sym}**: {mean}\n")
                elif sym:
                    L.append(f"- **{sym}**\n")
            L.append("\n")
        elif kind == "table":
            headers = s.get("headers") or []
            rows = s.get("rows") or []

            def row_symbol(r):
                if isinstance(r, dict):
                    return r.get("symbol") or {}
                return {}

            def symbol_md(sym: dict) -> str:
                img = sym.get("image")
                lbl = sym.get("label") or ""
                if img and lbl:
                    return f"![]({img}) *{_md_cell(lbl)}*"
                if img:
                    return f"![]({img})"
                if lbl:
                    return f"*{_md_cell(lbl)}*"
                return ""

            any_symbol = any(
                row_symbol(r).get("image") or row_symbol(r).get("label") for r in rows
            )

            # ----- Pipe-table view (each variant = one visual row) -----
            if headers or any_symbol:
                cols = (["Symbol"] if any_symbol else []) + [
                    (_md_cell(h) or "—") for h in headers
                ]
                L.append("| " + " | ".join(cols) + " |\n")
                L.append("|" + "|".join("---" for _ in cols) + "|\n")
                for r in rows:
                    sym = row_symbol(r)
                    variants_cells = _row_variants(r)
                    vcount = len(variants_cells)
                    for vi, vc in enumerate(variants_cells):
                        first = (vi == 0)
                        last = (vi == vcount - 1)
                        sym_cell = ""
                        if any_symbol:
                            if first and vcount == 1:
                                sym_cell = symbol_md(sym)
                            elif first:
                                sym_cell = f"{symbol_md(sym)} *(1 of {vcount})*".strip()
                            else:
                                sym_cell = f"↳ *( {vi+1} of {vcount})*"
                        line = []
                        if any_symbol:
                            line.append(sym_cell)
                        line.extend(_md_cell(v) for v in vc)
                        L.append("| " + " | ".join(line) + " |\n")
                L.append("\n")

            # ----- LLM-friendly record list -----
            if rows:
                L.append(
                    "**Records (each with its symbol image/label and variant count; "
                    "multi-variant records list each sub-row):**\n\n"
                )
                for idx, r in enumerate(rows, 1):
                    sym = row_symbol(r)
                    img = sym.get("image")
                    lbl = sym.get("label") or ""
                    vcount = r.get("variant_count", 1) if isinstance(r, dict) else 1
                    vlabel = r.get("variant_label") if isinstance(r, dict) else None
                    head_parts: list[str] = []
                    if img or lbl:
                        sym_desc = "Symbol: "
                        if img:
                            sym_desc += f"![]({img})"
                        if lbl:
                            sym_desc += f' (abbrev "{_md_cell(lbl)}")'
                        head_parts.append(sym_desc.strip())
                    if vcount > 1:
                        head_parts.append(
                            f"variants: {vlabel or vcount} ({vcount} sub-rows)"
                        )
                    variants_cells = _row_variants(r)
                    if vcount == 1:
                        parts = list(head_parts)
                        for h, v in zip(headers, variants_cells[0]):
                            if v and v.strip():
                                parts.append(f"{_md_cell(h)}: {_md_cell(v)}")
                        if parts:
                            L.append(f"{idx}. " + "; ".join(parts) + "\n")
                    else:
                        if head_parts:
                            L.append(f"{idx}. " + "; ".join(head_parts) + "\n")
                        for vi, vc in enumerate(variants_cells):
                            sub_parts = []
                            for h, v in zip(headers, vc):
                                if v and v.strip():
                                    sub_parts.append(f"{_md_cell(h)}: {_md_cell(v)}")
                            if sub_parts:
                                letter = chr(ord("a") + vi)
                                L.append(
                                    f"   {idx}{letter}. " + "; ".join(sub_parts) + "\n"
                                )
                L.append("\n")
        elif kind == "matrix":
            cols = s.get("columns") or []
            grid = s.get("grid") or []
            if cols:
                L.append(
                    "*Uniform column grid (no schedule-style symbol column).*\n\n"
                )
                L.append("| " + " | ".join(_md_cell(c) or "—" for c in cols) + " |\n")
                L.append("|" + "|".join("---" for _ in cols) + "|\n")
                n = len(cols)
                for row in grid:
                    cells = [row[i] if i < len(row) else "" for i in range(n)]
                    L.append(
                        "| " + " | ".join(_md_cell(c) for c in cells) + " |\n"
                    )
                L.append("\n")
            nt = (s.get("notes") or "").strip()
            if nt:
                L.append("### Notes\n\n")
                L.append(nt + "\n\n")
        elif kind == "mccol":
            ban = (s.get("banner_title") or "").strip()
            if ban:
                L.append(f"*{ban}*\n\n")
            for bx in s.get("boxes") or []:
                lab = (bx.get("label") or "").strip() or (bx.get("id") or "—")
                L.append(f"### {lab}\n\n")
                rows = bx.get("rows")
                if rows and isinstance(rows, list):
                    L.append("| Sheet | Title |\n|---|---|\n")
                    for rw in rows:
                        if not isinstance(rw, dict):
                            continue
                        sh = _md_cell(str(rw.get("sheet", "") or ""))
                        ti = _md_cell(str(rw.get("title", "") or ""))
                        L.append(f"| {sh} | {ti} |\n")
                    L.append("\n")
                else:
                    bdy = (bx.get("body") or "").strip()
                    if bdy:
                        L.append(bdy + "\n\n")
        elif kind == "contractor_matrix":
            cols = s.get("columns") or []
            L.append(
                "*Three-column responsibility matrix (contractor → bullet items).*\n\n"
            )
            for col in cols:
                hid = col.get("id") or ""
                head = col.get("heading") or hid.replace("_", " ").title()
                L.append(f"### {head}\n\n")
                for it in col.get("items") or []:
                    if (it or "").strip():
                        L.append(f"- {_md_cell(it)}\n")
                L.append("\n")
            fn = (s.get("footer_note") or "").strip()
            if fn:
                L.append("### Note\n\n")
                L.append(f"> {_md_cell(fn)}\n\n")
        else:  # notes / prose
            paras = s.get("paragraphs")
            bullets = s.get("bullet_items")
            closing = s.get("closing_paragraphs")
            body = s.get("body") or ""
            tree = s.get("bullet_tree")
            subs = s.get("subsections")
            if isinstance(paras, list) and paras and any(str(p).strip() for p in paras):
                for p in paras:
                    ps = str(p).strip()
                    if ps:
                        L.append(ps + "\n\n")
            if isinstance(subs, list) and subs:
                sm = _render_subsections_md(subs)
                if sm.strip():
                    L.append(sm + "\n\n")
            elif isinstance(tree, list) and tree:
                tr = _render_bullet_tree_md(tree, 0)
                if tr.strip():
                    L.append(tr + "\n\n")
            elif isinstance(bullets, list) and bullets:
                for b in bullets:
                    bs = str(b).strip()
                    if bs:
                        L.append(f"- {_md_cell(bs)}\n")
                L.append("\n")
            if isinstance(closing, list) and closing:
                for p in closing:
                    ps = str(p).strip()
                    if ps:
                        L.append(ps + "\n\n")
            else:
                had_list = (
                    (isinstance(subs, list) and bool(subs))
                    or (isinstance(tree, list) and bool(tree))
                    or (
                        isinstance(bullets, list)
                        and any(str(b).strip() for b in bullets)
                    )
                )
                if body and not had_list:
                    para_join = ""
                    if isinstance(paras, list) and paras:
                        para_join = "\n\n".join(
                            str(p).strip() for p in paras if str(p).strip()
                        ).strip()
                    if not (para_join and body.strip() == para_join):
                        L.append(body + "\n\n")

    # Root ``abbreviations`` (same list) when not already emitted as a section
    abbr = d.get("abbreviations") or []
    if abbr and not any(
        (x.get("kind") == "abbreviations") for x in (d.get("sections") or [])
    ):
        L.append("## Abbreviations\n\n")
        for e in abbr:
            sym = e.get("symbol", "")
            mean = e.get("meaning", "")
            if sym and mean:
                L.append(f"- **{sym}**: {mean}\n")
            elif sym:
                L.append(f"- **{sym}**\n")
        L.append("\n")

    # Full-page text as reliable fallback (skip when structured sections duplicate it)
    full = (d.get("full_text") or "").strip()
    doc_meta = d.get("document") or {}
    _sl = doc_meta.get("structured_layout")
    if (
        full
        and _sl not in ("introduction_project_scope", "tsc_followon")
        and not doc_meta.get("continuation")
    ):
        L.append("## Full page text (verbatim)\n\n")
        L.append("```text\n")
        L.append(full)
        L.append("\n```\n")
    return "".join(L)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Extract LLM-ready content (JSON + Markdown) from a detect_standalone overlay JSON."
    )
    ap.add_argument("--json", required=True, help="Path to detect_standalone --json-out file")
    ap.add_argument("--pdf", default=None, help="Override PDF path (else use json['pdf'])")
    ap.add_argument(
        "--out",
        default=None,
        help="Output base path (default: reuse --json path, writes .extraction.json/.extraction.md)",
    )
    args = ap.parse_args(argv)
    jpath = Path(args.json)
    out_base = Path(args.out) if args.out else jpath
    data = json.loads(jpath.read_text(encoding="utf-8"))
    doc = extract_from_overlay_json(data, pdf_path=args.pdf)
    paths = write_extraction_artifacts(out_base, doc)
    for k, v in paths.items():
        print(f"Wrote {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
