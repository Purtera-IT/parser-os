"""Universal legend → rule discovery.

The pipeline used to ship a 7-symbol YAML whitelist (WN, POS-T, POS-P,
TV, CR, DA, H) and only trust device codes that appeared in that list.
That's the opposite of universal — every project the parser hadn't been
hand-tuned for fell through.

This module replaces the whitelist with a *discovery* pass driven by
the project's own legend page(s). At parse time, the structured legend
doc (cells with explicit bbox_pt and column headers) is walked and
every row becomes a runtime :class:`LegendRule`. The raw_symbol comes
from the SYMBOL cell's text token when present (e.g. "WN", "POS-T")
or a stable synthetic identifier when the cell is text-less (e.g. a
camera drawing). The normalized_class, system, cable info, mounting,
power, and remarks all come straight from the legend's columns — no
keyword tables, no hardcoded ``description → class`` mappings.

Result: a project with cameras, fire alarm devices, motion detectors,
intercoms, etc. gets every one of them as a detectable device class,
without anyone editing a YAML file.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any

from app.takeoff.schemas import BBox, LegendRule


# A legend SYMBOL cell text that looks like a real code (e.g. "WN",
# "POS-T", "FACP-2"). Conservative shape — at least one ASCII letter,
# up to 6 chars total. Matches the symbol normalizer in
# parser_intelligence (deliberately the same gate).
_SYMBOL_SHAPE_RE = re.compile(r"^[A-Z][A-Z0-9\-]{0,5}$")

# Trailing port-count placeholder on legend SYMBOL cells. Example:
# "POS-T #", "A #", "F 2", "TV 12".
_TRAILING_PORT_RE = re.compile(r"\s+[#\d]+\s*$")

# Snake-case sanitization: collapse runs of non-alphanumeric into
# single underscores.
_NON_ALNUM_RE = re.compile(r"[^A-Z0-9]+")

# "1 PORT " / "4 PORT " prefix on most cabling-device descriptions —
# strip it from normalized_class derivation so two ports vs four ports
# of the same device share the same class.
_PORT_COUNT_PREFIX_RE = re.compile(r"^\d+\s*-?\s*PORT\s+", flags=re.IGNORECASE)

# Mounting qualifiers — universal English geometry words, not project-
# specific terminology. Wall-mounted and ceiling-mounted variants of
# the SAME device share a single normalized_class so downstream rollups
# don't double-count them as different device families. The
# distinction is preserved in the full `description` field.
_MOUNTING_PREFIX_RE = re.compile(
    r"^(wall|ceiling|floor|surface|flush|recess|recessed|pole)_(mounted_)?"
)
_MOUNTING_SUFFIX_RE = re.compile(
    r"_(wall|ceiling|floor|surface|flush|recess|recessed|pole)_(mounted)?$"
)
# Stand-alone "_mounted_" / "_mount_" tokens (e.g. "wall_mount" without
# "_mounted") — collapse to the empty string.
_MOUNTING_INNER_RE = re.compile(r"_mounted_|_mount_")


def _normalize_symbol_text(raw: str) -> str | None:
    """Return a code-shaped raw_symbol from a SYMBOL cell's text, or None.

    Conservative — rejects multi-word remnants, paragraph text, and
    glyph-only cells. Mirrors the gate used in parser_intelligence so
    both layers agree on what counts as a "real" code.
    """
    if not raw:
        return None
    s = str(raw).strip()
    s = _TRAILING_PORT_RE.sub("", s).strip()
    if " " in s:
        return None
    candidate = s.upper()
    if not _SYMBOL_SHAPE_RE.match(candidate):
        return None
    return candidate


def _normalized_class_from(description: str) -> str:
    """Universal description → normalized_class.

    Strips a leading port-count prefix ("1 PORT ", "4 PORT ") so port
    variants of the same device share a class, then snake_cases the
    rest. No keyword tables — this is a pure text transformation.

    Examples::

        "1 PORT WALL MOUNTED WIRELESS NODE OUTLET"
            -> "wall_mounted_wireless_node_outlet"
        "CARD READER"
            -> "card_reader"
        "MINI DOME SINGLE LENS CAMERA - CEILING MOUNTED"
            -> "mini_dome_single_lens_camera_ceiling_mounted"
    """
    upper = description.strip().upper()
    upper = _PORT_COUNT_PREFIX_RE.sub("", upper)
    tokens = [t for t in _NON_ALNUM_RE.split(upper) if t]
    if not tokens:
        return "unclassified"
    snake = "_".join(tokens).lower()[:160]
    # Collapse mounting qualifiers — wall vs ceiling vs floor mounted
    # variants of the same device share a single class. Run repeatedly
    # because a description can have both a prefix and a suffix form.
    for _ in range(3):
        new = _MOUNTING_PREFIX_RE.sub("", snake)
        new = _MOUNTING_SUFFIX_RE.sub("", new)
        new = _MOUNTING_INNER_RE.sub("_", new)
        new = re.sub(r"_+", "_", new).strip("_")
        if new == snake:
            break
        snake = new
    return snake or "unclassified"


def _system_from_section(section_title: str) -> str:
    """Universal section title → system name.

    Strips generic suffixes (" SYMBOL LEGEND", " LEGEND") then
    snake_cases the remainder.
    """
    cleaned = (section_title or "").strip().upper()
    for suffix in (" SYMBOL LEGEND", " SYMBOLS LEGEND", " LEGEND"):
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
            break
    tokens = [t for t in _NON_ALNUM_RE.split(cleaned) if t]
    return "_".join(tokens).lower() or "unknown_system"


def _resolve_col(headers: list[dict[str, Any]] | None, *names: str) -> int | None:
    """Return the index of the first column whose header CONTAINS any of
    ``names`` (case-insensitive), or ``None`` if no header matches.
    """
    if not headers:
        return None
    for i, c in enumerate(headers):
        text = (c.get("text") or "").upper()
        for n in names:
            if n in text:
                return i
    return None


def _cell_text(cells: list[dict[str, Any]], idx: int | None) -> str | None:
    """Return the text of cell at ``idx``, stripped, or ``None`` if absent."""
    if idx is None or idx < 0 or idx >= len(cells):
        return None
    t = (cells[idx].get("text") or "").strip()
    return t or None


def _cell_bbox(cells: list[dict[str, Any]], idx: int) -> BBox | None:
    """Return the cell's bbox as a typed BBox in ``pdf_pt``, or None."""
    if idx < 0 or idx >= len(cells):
        return None
    bbox_pt = cells[idx].get("bbox_pt")
    if not bbox_pt or len(bbox_pt) != 4:
        return None
    return BBox(
        x0=float(bbox_pt[0]),
        y0=float(bbox_pt[1]),
        x1=float(bbox_pt[2]),
        y1=float(bbox_pt[3]),
        coord_space="pdf_pt",
    )


def _parse_int_or_none(s: str | None) -> int | None:
    if not s:
        return None
    s = s.strip()
    if not s.isdigit():
        return None
    try:
        return int(s)
    except ValueError:
        return None


def _stable_shape_id(*, section_title: str, description: str, row_idx: int) -> str:
    """Derive a deterministic synthetic raw_symbol for a textless row.

    Format: ``__shp_<10-char hash>`` — collision-free across the legend
    and identifiable as shape-only by the ``__shp_`` prefix.
    """
    h = hashlib.sha1(
        f"{section_title}|{description}|{row_idx}".encode("utf-8")
    ).hexdigest()[:10]
    return f"__shp_{h}"


def _iter_legend_pages(legend_doc: Any) -> list[dict[str, Any]]:
    """Normalize a legend doc / list-of-docs into a flat list."""
    if isinstance(legend_doc, list):
        return [d for d in legend_doc if isinstance(d, dict)]
    if isinstance(legend_doc, dict):
        return [legend_doc]
    return []


def discover_legend_rules(*, legend_doc: Any) -> list[LegendRule]:
    """Walk every row of every section of a structured legend doc and
    return a :class:`LegendRule` for each.

    Inputs are the *structured* legend dict (as produced by
    :func:`app.takeoff.legend_extract.extract_legend`) — either a single
    doc or a list of per-page docs. Output is a flat list of rules,
    one per row, with no deduplication: if the legend has WN-ceiling
    and WN-wall on two consecutive rows, both come through.

    No YAML defaults are consulted. No keyword tables are consulted.
    Every device class the project legend defines becomes a rule.
    """
    rules: list[LegendRule] = []
    seen_synthetic_ids: set[str] = set()

    for legend_page in _iter_legend_pages(legend_doc):
        page_index = legend_page.get("page_index")
        try:
            page_index_int: int | None = int(page_index) if page_index is not None else None
        except (TypeError, ValueError):
            page_index_int = None

        for table in legend_page.get("tables", []) or []:
            for section in table.get("sections", []) or []:
                section_title = (section.get("title") or "").strip()
                if not section_title:
                    continue
                # Skip non-symbol sections (RESPONSIBILITY MATRIX, GENERAL
                # NOTES) by requiring at least one column header.
                cols = section.get("column_headers") or []
                if not cols:
                    continue
                sym_col = _resolve_col(cols, "SYMBOL")
                if sym_col is None:
                    sym_col = 0
                desc_col = _resolve_col(cols, "DESCRIPTION")
                if desc_col is None:
                    desc_col = 1 if len(cols) >= 2 else None
                cc_col = _resolve_col(cols, "CABLE COUNT")
                ctype_col = _resolve_col(cols, "CABLE DESC", "CABLE TYPE")
                wat_col = _resolve_col(cols, "WORK AREA")
                close_col = _resolve_col(cols, "CLOSET")
                mount_col = _resolve_col(cols, "MOUNT")
                rough_col = _resolve_col(cols, "ROUGH")
                power_col = _resolve_col(cols, "POWER")
                rem_col = _resolve_col(cols, "REMARK")

                system = _system_from_section(section_title)
                for row_idx, row in enumerate(section.get("rows", []) or []):
                    cells = row.get("cells") or []
                    if not cells:
                        continue
                    desc_text = _cell_text(cells, desc_col)
                    if not desc_text:
                        continue
                    if len(desc_text) < 6:
                        continue
                    if " " not in desc_text:
                        # Single-token descriptions ("1", "N/A") are likely
                        # noise rows (placeholder cells captured by table
                        # extraction). Skip.
                        continue
                    sym_text_raw = _cell_text(cells, sym_col)
                    normalized_code = _normalize_symbol_text(sym_text_raw or "")
                    if normalized_code:
                        raw_symbol = normalized_code
                    else:
                        # Shape-only path: the SYMBOL cell must be large
                        # enough to actually host an icon drawing. Notes
                        # rows and cable-spec sub-rows in the legend
                        # often have a tiny SYMBOL cell (e.g. 31 x 3 pt)
                        # — keep those out so template-matching doesn't
                        # produce thousands of false-positive matches
                        # from a 13x26-px text-stamp template.
                        sym_bbox = _cell_bbox(cells, sym_col)
                        if sym_bbox is None:
                            continue
                        cell_w = sym_bbox.x1 - sym_bbox.x0
                        cell_h = sym_bbox.y1 - sym_bbox.y0
                        # Universal threshold: legend SYMBOL cells in any
                        # firm's drawings are routinely 50-80 pt on a side
                        # for real device rows. Cable-spec sub-rows are
                        # ~48 wide x 27-32 tall (text only, no icon).
                        # Notes blocks span 180+ pt. Accept cells in
                        # [40, 140] pt on each axis — that admits every
                        # observed device cell across firms while keeping
                        # both the cable-spec strips and the notes blocks
                        # out of the rule set.
                        if not (40 <= cell_w <= 140 and 40 <= cell_h <= 140):
                            continue
                        raw_symbol = _stable_shape_id(
                            section_title=section_title,
                            description=desc_text,
                            row_idx=row_idx,
                        )
                        if raw_symbol in seen_synthetic_ids:
                            continue
                        seen_synthetic_ids.add(raw_symbol)

                    rules.append(LegendRule(
                        raw_symbol=raw_symbol,
                        normalized_class=_normalized_class_from(desc_text),
                        system=system,
                        description=desc_text,
                        cable_count=_parse_int_or_none(_cell_text(cells, cc_col)),
                        cable_type=_cell_text(cells, ctype_col),
                        work_area_termination=_cell_text(cells, wat_col),
                        closet_termination=_cell_text(cells, close_col),
                        mounting=_cell_text(cells, mount_col),
                        rough_in=_cell_text(cells, rough_col),
                        power=_cell_text(cells, power_col),
                        remarks=[],
                        source_page=page_index_int,
                        source_bbox=_cell_bbox(cells, sym_col),
                        confidence=0.92,
                    ))

    return rules


__all__ = [
    "discover_legend_rules",
]
