"""Site-roster table extractor.

Real construction / IT-refresh deals declare an "authoritative site
roster" — a table with one row per physical site, listing the site ID,
facility name, street address, MDF/IDF, access window, escort owner,
etc. These rosters are the load-bearing reference for site_tables,
escort scheduling, and cutover planning.

Today's PDF parser flattens each table row into a single ``scope_item``
atom with no structured semantics, so the entity extractor has to
recover site info from prose — which it does badly (Marriott/OPTBOT
case: 0 of 5 canonical site IDs captured, 16 "site" entities made from
prose fragments like "n terminal", "building c", "site id facility").

This module:

1. Detects when a table row block is part of a **site_roster** table
   (by column headers OR explicit ``kind=physical_site`` declaration
   in surrounding prose OR row-shape pattern matching site IDs).
2. Maps each cell to a canonical field (``site_id``, ``facility_name``,
   ``street_address``, ``mdf_idf``, ``access_window``, ``escort_owner``,
   ``phone``, ``email``, ``contact``, ``notes``, …).
3. Returns a structured ``SiteRosterRow`` per row that the PDF parser
   can emit as either an ``entity`` atom (kind=site) or a
   ``physical_site`` row that downstream entity-extraction consumes.

Deterministic. No LLM. No I/O. Works on whatever the upstream PDF
parser produces (``{"columns": [...], "rows": [{col: cell, ...}]}``
shape) — the structure already exists; we just give it semantic
meaning.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence


# Header keywords. The header column is matched case-insensitively;
# the FIRST matching field wins so order matters (more specific
# patterns first). Each value is a tuple of regex patterns that
# the column header must MATCH (substring search).
_FIELD_HEADER_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    # Site identifier (most authoritative)
    ("site_id", ("site id", "site #", "site code", "site key", "location id", "location code", "facility id", "facility code", "store #", "store id")),
    # Facility / building name
    ("facility_name", ("facility name", "facility", "site name", "location name", "building", "building name", "premises name", "store name")),
    # Street address
    ("street_address", ("street address", "address", "physical address", "site address", "location")),
    # City/state
    ("city_state", ("city/state", "city, state", "city / state", "region")),
    # MDF/IDF closet
    ("mdf_idf", ("mdf/idf", "mdf / idf", "mdf", "idf", "closet", "tr ", "main distribution", "telecom room")),
    # Access window / hours
    ("access_window", ("access window", "access hours", "hours", "operating hours", "site hours", "business hours")),
    # Escort / point-of-contact
    ("escort_owner", ("escort owner", "escort", "site owner", "facility owner", "point of contact", "poc")),
    # Site contact
    ("contact", ("contact", "site contact", "primary contact", "facility contact")),
    ("phone", ("phone", "telephone", "tel ", "tel#", "phone #")),
    ("email", ("email", "e-mail", "email address")),
    # Square footage / occupancy
    ("sqft", ("square footage", "sqft", "sq ft", "footprint", "size (sqft)", "size sqft")),
    ("occupancy", ("occupancy", "occupants", "headcount", "users", "seats")),
    # Notes
    ("notes", ("notes", "remarks", "comments")),
)


# Patterns that, when present in the column headers as a SET, signal
# "this is a site roster" with high confidence.
_ROSTER_HEADER_PRESENCE_SIGNALS = (
    {"site_id", "facility_name"},
    {"site_id", "street_address"},
    {"facility_name", "street_address"},
    {"site_id", "mdf_idf"},
)


# Site-ID shape regex — used as a fallback when no header tells us
# which column is the ID. Tries hard to recognize enterprise
# site IDs across formats:
#   ATL-HQ-01, NYC-DC-12, SFO-WEST-05, LON-OFFICE-A2, S001, STORE-142,
#   BLDG-12, B12, MDC-01, ATL_HQ_01 (underscore variant)
_SITE_ID_SHAPE_RE = re.compile(
    r"^(?:"
    r"[A-Z]{2,5}[-_][A-Z0-9]{1,8}(?:[-_][A-Z0-9]{1,6}){0,3}"  # ATL-HQ-01, NYC-DC-12, ATL_HQ_01
    r"|S\d{2,4}|SITE[-_]?\d{1,4}"                              # S001, SITE-12, SITE12
    r"|STORE[-_]?\d{1,4}|LOC[-_]?\d{1,4}"                       # STORE-142, LOC-7
    r"|BLDG[-_]?\d{1,4}|B\d{1,4}"                              # BLDG-12, B12
    r"|MDC[-_]?\d{1,4}|IDC[-_]?\d{1,4}"                        # MDC-01, IDC-3
    r"|DC\d{1,4}"                                              # DC12
    r"|H\d{1,4}|W\d{1,4}"                                      # H1, W3 (rare but real)
    r")$",
    re.IGNORECASE,
)

# Phrase that explicitly declares a table is a site roster.
_KIND_PHYSICAL_SITE_DECLARATION = re.compile(
    r"\bkind\s*=\s*physical_site\b", re.IGNORECASE
)


@dataclass(frozen=True)
class SiteRosterRow:
    """One row of a site roster, with cells mapped to canonical fields."""

    row_index: int
    site_id: str | None
    facility_name: str | None
    street_address: str | None
    mdf_idf: str | None = None
    access_window: str | None = None
    escort_owner: str | None = None
    contact: str | None = None
    phone: str | None = None
    email: str | None = None
    city_state: str | None = None
    sqft: str | None = None
    occupancy: str | None = None
    notes: str | None = None
    extra_fields: tuple[tuple[str, str], ...] = ()
    raw_cells: tuple[tuple[str, str], ...] = ()
    confidence: float = 0.8

    def as_dict(self) -> dict[str, Any]:
        return {
            "row_index": self.row_index,
            "site_id": self.site_id,
            "facility_name": self.facility_name,
            "street_address": self.street_address,
            "mdf_idf": self.mdf_idf,
            "access_window": self.access_window,
            "escort_owner": self.escort_owner,
            "contact": self.contact,
            "phone": self.phone,
            "email": self.email,
            "city_state": self.city_state,
            "sqft": self.sqft,
            "occupancy": self.occupancy,
            "notes": self.notes,
            "extras": dict(self.extra_fields),
            "raw_cells": dict(self.raw_cells),
            "confidence": self.confidence,
        }


# ── Header / row detection ───────────────────────────────────────


def _norm_header(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def map_columns_to_fields(
    columns: Sequence[str],
    *,
    explicit_declaration: bool = False,
) -> dict[int, str]:
    """For each column header, return the canonical field name it
    maps to. Unknown headers are simply omitted from the mapping.

    When ``explicit_declaration`` is True (surrounding prose declares
    ``kind=physical_site``), we additionally treat ambiguous headers
    like "code", "name", "where" / "loc" positionally — leftmost
    untouched column becomes site_id, next becomes facility_name,
    next becomes street_address. This rescues rosters that use
    project-jargon column names instead of the canonical ones.
    """
    out: dict[int, str] = {}
    used_fields: set[str] = set()
    for i, col in enumerate(columns):
        header = _norm_header(str(col))
        if not header:
            continue
        for field_name, keywords in _FIELD_HEADER_PATTERNS:
            if field_name in used_fields:
                continue
            for kw in keywords:
                if kw in header:
                    out[i] = field_name
                    used_fields.add(field_name)
                    break
            if i in out:
                break

    if explicit_declaration:
        # Positional fallbacks for ambiguous headers when caller has
        # told us this IS a site roster.
        position_defaults = ("site_id", "facility_name", "street_address",
                              "mdf_idf", "access_window", "escort_owner")
        pos_iter = iter(position_defaults)
        for i, col in enumerate(columns):
            if i in out:
                continue
            try:
                while True:
                    candidate = next(pos_iter)
                    if candidate not in used_fields:
                        out[i] = candidate
                        used_fields.add(candidate)
                        break
            except StopIteration:
                break
    return out


def looks_like_site_roster(
    *,
    columns: Sequence[str],
    rows: Sequence[Any],
    surrounding_text: str = "",
) -> bool:
    """Heuristic gate: is this table block a site roster?

    Three positive signals (any one is sufficient):
      1. Surrounding prose declares ``kind=physical_site``.
      2. Column headers include ≥2 of: site_id / facility_name /
         street_address / mdf_idf.
      3. ≥3 row's leftmost non-empty cell matches the site-ID shape
         regex (handles rosters that ship without column headers).
    """
    # Signal 1: explicit declaration
    if _KIND_PHYSICAL_SITE_DECLARATION.search(surrounding_text):
        return True

    # Signal 2: column header presence
    col_map = map_columns_to_fields(columns)
    fields_present = set(col_map.values())
    for signal_set in _ROSTER_HEADER_PRESENCE_SIGNALS:
        if signal_set.issubset(fields_present):
            return True

    # Signal 3: row-shape — count rows whose leftmost non-empty cell
    # matches the site-ID shape.
    id_hits = 0
    for row in rows[:20]:
        leftmost = _leftmost_nonempty_cell(row, columns)
        if leftmost and _SITE_ID_SHAPE_RE.match(leftmost.strip()):
            id_hits += 1
    if id_hits >= 3:
        return True

    return False


def _leftmost_nonempty_cell(row: Any, columns: Sequence[str]) -> str | None:
    """Return the value of the leftmost non-empty cell in a row.

    Tolerates dict rows ({col: val}) and list/tuple rows. Whitespace
    is stripped; None / blank cells are skipped.
    """
    if isinstance(row, dict):
        for col in columns:
            val = row.get(col)
            if val is None:
                continue
            s = str(val).strip()
            if s:
                return s
        # If dict has more keys than `columns`, walk them too
        for v in row.values():
            if v is None:
                continue
            s = str(v).strip()
            if s:
                return s
        return None
    if isinstance(row, (list, tuple)):
        for v in row:
            if v is None:
                continue
            s = str(v).strip()
            if s:
                return s
        return None
    s = str(row or "").strip()
    return s or None


def _cell_value(row: Any, columns: Sequence[str], col_idx: int) -> str:
    """Return the cell value at column index ``col_idx`` for ``row``."""
    if isinstance(row, dict):
        if 0 <= col_idx < len(columns):
            v = row.get(columns[col_idx])
            return str(v).strip() if v is not None else ""
        # Fallback: positional access on dict values
        vs = list(row.values())
        if 0 <= col_idx < len(vs):
            v = vs[col_idx]
            return str(v).strip() if v is not None else ""
        return ""
    if isinstance(row, (list, tuple)):
        if 0 <= col_idx < len(row):
            v = row[col_idx]
            return str(v).strip() if v is not None else ""
    return ""


def _infer_site_id_from_row(row: Any, columns: Sequence[str]) -> str | None:
    """When the header doesn't name a Site ID column, sniff each cell
    against the site-ID shape regex and return the first match."""
    if isinstance(row, dict):
        for col in columns:
            val = row.get(col)
            if val is None:
                continue
            s = str(val).strip()
            if s and _SITE_ID_SHAPE_RE.match(s):
                return s
        for v in row.values():
            if v is None:
                continue
            s = str(v).strip()
            if s and _SITE_ID_SHAPE_RE.match(s):
                return s
        return None
    if isinstance(row, (list, tuple)):
        for v in row:
            if v is None:
                continue
            s = str(v).strip()
            if s and _SITE_ID_SHAPE_RE.match(s):
                return s
    return None


# ── Row → SiteRosterRow ─────────────────────────────────────────


def _is_header_row(row: Any, columns: Sequence[str], field_map: dict[int, str]) -> bool:
    """A row whose cell values mostly equal the headers themselves is a
    duplicated header row (some PDF extractors fold the header into
    rows[0]). Skip these so they don't pollute the entity extraction."""
    matches = 0
    total = 0
    for i, col in enumerate(columns):
        total += 1
        if i in field_map:
            cell = _cell_value(row, columns, i)
            if cell and _norm_header(cell) == _norm_header(str(col)):
                matches += 1
    return total > 0 and matches >= max(1, total // 2)


def extract_site_roster(
    *,
    columns: Sequence[str],
    rows: Sequence[Any],
    surrounding_text: str = "",
) -> list[SiteRosterRow]:
    """Pull every physical_site row out of a roster table.

    Returns an empty list when the table doesn't look like a site
    roster — see ``looks_like_site_roster`` for the gate. Callers
    should check that gate first and only invoke ``extract_site_roster``
    when it returns True, but this function is safe to call
    unconditionally.
    """
    if not looks_like_site_roster(
        columns=columns, rows=rows, surrounding_text=surrounding_text
    ):
        return []

    explicit_decl = bool(_KIND_PHYSICAL_SITE_DECLARATION.search(surrounding_text or ""))
    field_map = map_columns_to_fields(columns, explicit_declaration=explicit_decl)
    # If we have no column->field mapping (rare; happens when the
    # roster is shipped without headers), build a positional one:
    # column 0 is treated as site_id, column 1 as facility_name,
    # column 2 as street_address.
    if not field_map and rows:
        defaults = ("site_id", "facility_name", "street_address", "mdf_idf", "access_window", "escort_owner")
        for i, fname in enumerate(defaults):
            if i < len(columns):
                field_map[i] = fname

    out: list[SiteRosterRow] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, (dict, list, tuple)) and row is not None:
            continue
        # Skip header-as-row duplicates
        if _is_header_row(row, columns, field_map):
            continue

        cells: dict[str, str] = {}
        raw_cells: list[tuple[str, str]] = []
        for i, col in enumerate(columns):
            val = _cell_value(row, columns, i)
            if not val:
                continue
            raw_cells.append((str(col), val))
            field_name = field_map.get(i)
            if field_name:
                # First non-empty value wins (don't clobber)
                cells.setdefault(field_name, val)

        # Fallback: infer site_id from row content when we couldn't
        # find one via the column map.
        if "site_id" not in cells:
            sid = _infer_site_id_from_row(row, columns)
            if sid:
                cells["site_id"] = sid

        # Collapse internal whitespace on the site_id when the
        # compact form still looks like a site ID. PDF wrap can
        # split "ATL-WEST-02" -> "ATL-WEST-0\n2" or
        # "ATL_HQ_01" -> "ATL HQ 01 _ _". The compact form is the
        # canonical site_id.
        sid = cells.get("site_id") or ""
        if sid:
            compact = re.sub(r"\s+", "", sid)
            if compact != sid and _SITE_ID_SHAPE_RE.match(compact):
                cells["site_id"] = compact
            else:
                # Also try collapsing whitespace AND restoring
                # underscores stripped by PDF rendering
                compact_underscore = re.sub(r"\s+", "_", sid).strip("_")
                if (
                    compact_underscore != sid
                    and _SITE_ID_SHAPE_RE.match(compact_underscore)
                ):
                    cells["site_id"] = compact_underscore

        # A row with neither a site_id nor a facility_name is just
        # noise — skip it.
        if not cells.get("site_id") and not cells.get("facility_name"):
            continue

        # Bucket unknown fields
        known_fields = {f[0] for f in _FIELD_HEADER_PATTERNS}
        extras: list[tuple[str, str]] = []
        for col_name, val in raw_cells:
            # Skip cells that were already absorbed into canonical fields
            i = list(columns).index(col_name) if col_name in columns else -1
            if i >= 0 and i in field_map:
                continue
            extras.append((col_name, val))

        confidence = 0.85 if cells.get("site_id") else 0.6
        out.append(
            SiteRosterRow(
                row_index=row_index,
                site_id=cells.get("site_id"),
                facility_name=cells.get("facility_name"),
                street_address=cells.get("street_address"),
                mdf_idf=cells.get("mdf_idf"),
                access_window=cells.get("access_window"),
                escort_owner=cells.get("escort_owner"),
                contact=cells.get("contact"),
                phone=cells.get("phone"),
                email=cells.get("email"),
                city_state=cells.get("city_state"),
                sqft=cells.get("sqft"),
                occupancy=cells.get("occupancy"),
                notes=cells.get("notes"),
                extra_fields=tuple(extras),
                raw_cells=tuple(raw_cells),
                confidence=confidence,
            )
        )

    return out


__all__ = [
    "SiteRosterRow",
    "extract_site_roster",
    "looks_like_site_roster",
    "map_columns_to_fields",
]
