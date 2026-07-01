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
# Field-header patterns — order matters. The first matching field
# wins, so more specific patterns go first to prevent generic
# keywords like "location" from claiming the wrong column when a
# more authoritative match exists later in the row.
_FIELD_HEADER_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    # Site identifier (most authoritative)
    ("site_id", ("site id", "site #", "site code", "site key", "location id", "location code", "facility id", "facility code", "store #", "store id", "store number", "site number")),
    # Street address — keep BEFORE facility_name so an "Address" or
    # "Street" header takes the address column even when a "Location"
    # header would also match facility_name's "location name".
    ("street_address", ("street address", "physical address", "site address", "address", "street", "addr")),
    # Facility / building name
    ("facility_name", ("facility name", "facility", "site name", "location name", "building name", "premises name", "store name", "name", "location", "use", "building")),
    # City / state — separate columns BEFORE combined city_state so both can map.
    ("city", ("city", "town", "municipality")),
    ("state", ("state", "st", "province")),
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
    # Zip code as its own column
    ("zip", ("zip", "zipcode", "zip code", "postal", "postcode", "postal code")),
    # Square footage / occupancy
    ("sqft", ("square footage", "sqft", "sq ft", "footprint", "size (sqft)", "size sqft", "size")),
    ("occupancy", ("occupancy", "occupants", "headcount", "users", "seats")),
    # Notes
    ("notes", ("notes", "remarks", "comments")),
)


# Header tokens that disqualify a column from being a street_address
# even though they contain the "address" substring — these are network
# / contact identifiers, not physical addresses.
_NON_STREET_ADDRESS_HEADERS: tuple[str, ...] = (
    "ip address", "ip addr", "mac address", "email address", "e-mail address",
    "ipv4", "ipv6", "url", "web address",
)


# Patterns that, when present in the column headers as a SET, signal
# "this is a site roster" with high confidence.
_ROSTER_HEADER_PRESENCE_SIGNALS = (
    {"site_id", "facility_name"},
    {"site_id", "street_address"},
    {"facility_name", "street_address"},
    {"site_id", "mdf_idf"},
    {"facility_name", "city"},
    {"street_address", "city"},
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
    city: str | None = None
    state: str | None = None
    zip: str | None = None
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
            "city": self.city,
            "state": self.state,
            "zip": self.zip,
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
            # The street_address patterns match the bare "address"
            # substring, which also lives inside "IP Address", "MAC
            # Address" and "Email Address". Those are network / contact
            # identifiers, not a physical address — never let them claim
            # the street_address slot (the ghost-site root cause).
            if field_name == "street_address" and any(
                d in header for d in _NON_STREET_ADDRESS_HEADERS
            ):
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
    # v48 FIX 3: Negative guard — if column headers contain strong
    # non-roster signals (requirements / BOM / risk / acceptance /
    # schedule / spec), REJECT immediately. This prevents the roster
    # path from silently eating these tables and dropping their rows.
    _NON_ROSTER_HEADER_SIGNALS: frozenset[str] = frozenset({
        "requirement", "acceptance", "bom", "line item", "unit price",
        "quantity", "risk id", "probability", "impact", "mitigation",
        "checklist", "checkpoint", "deliverable", "criterion", "criteria",
        "phase", "milestone", "task", "activity", "duration", "predecessor",
        "status", "priority", "category", "description",
        "part number", "part no", "model number", "model no", "sku",
        "unit cost", "extended", "subtotal", "total cost",
        "shall", "must", "will provide",
        # Asset-inventory signals: an asset inventory (Asset ID / Serial /
        # Model / IP Address / MAC Address / Hostname) is NOT a site
        # roster. Without this guard, AST-001 matches the site-ID shape
        # and "IP Address" matches the street-address header, producing
        # ghost physical_site rows. One asset-inventory signal is enough
        # to reject — real site rosters never carry serial/MAC/IP columns.
        "serial", "mac address", "ip address", "asset id", "asset tag",
        "hostname",
    })
    if columns:
        header_blob = " ".join(c.lower() for c in columns)
        if any(sig in header_blob for sig in _NON_ROSTER_HEADER_SIGNALS):
            return False

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
    # matches the site-ID shape. Accept when either
    #   - 3+ rows match (universal high-confidence), or
    #   - 2 rows match AND at least one column maps to a canonical
    #     field (medium-confidence with corroboration), or
    #   - 1 row matches AND we only have 1 row (single-site roster
    #     after the table-prelude has been declared elsewhere).
    id_hits = 0
    inspected = 0
    for row in rows[:20]:
        inspected += 1
        leftmost = _leftmost_nonempty_cell(row, columns)
        if leftmost and _SITE_ID_SHAPE_RE.match(leftmost.strip()):
            id_hits += 1
    if id_hits >= 3:
        return True
    if id_hits >= 2 and fields_present:
        return True
    if id_hits == inspected == 1 and fields_present:
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
                # When site_id was inferred FROM a cell that's also
                # mapped to another canonical field (e.g. header is
                # "Building", cell is "BLDG-1", which is BOTH the
                # building name AND the site ID), DON'T set the
                # other field to the same value — that's a
                # duplicate, not a real facility_name. Clear those
                # so a later column with the real name (like "Use")
                # has a chance to take facility_name via positional
                # fallback / extras.
                for fname, val in list(cells.items()):
                    if fname != "site_id" and val == sid:
                        cells.pop(fname)
                # And — if facility_name is now empty, promote the
                # NEXT non-empty cell to facility_name. This is the
                # universal "first non-id cell becomes the human
                # label" rule that handles "Building, Use,
                # Square footage" with cells "BLDG-1, Office,
                # 120000sf" → site_id="BLDG-1", facility="Office".
                if "facility_name" not in cells:
                    for i, col in enumerate(columns):
                        v = _cell_value(row, columns, i)
                        if not v or v == sid:
                            continue
                        # Skip cells already absorbed into other
                        # canonical fields
                        if any(v == cv for cv in cells.values()):
                            continue
                        # Skip cells that themselves look like a
                        # site_id (don't take a second ID as the name)
                        if _SITE_ID_SHAPE_RE.match(v.strip()):
                            continue
                        cells["facility_name"] = v
                        break

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

        from app.core.address_parse import enrich_location_fields

        loc = enrich_location_fields(
            street_address=cells.get("street_address"),
            city=cells.get("city"),
            state=cells.get("state"),
            zip_code=cells.get("zip"),
            city_state=cells.get("city_state"),
            facility_name=cells.get("facility_name"),
        )
        if loc["street_address"]:
            cells["street_address"] = loc["street_address"]
        if loc["city"]:
            cells["city"] = loc["city"]
        if loc["state"]:
            cells["state"] = loc["state"]
        if loc["zip"]:
            cells["zip"] = loc["zip"]

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
                city=cells.get("city"),
                state=cells.get("state"),
                zip=cells.get("zip"),
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
