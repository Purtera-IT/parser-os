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


# Header keywords, matched case-insensitively against the column header.
#
# MATCHING RULE — most-specific-wins, NOT first-in-list-wins.
# For each column header we score EVERY (field, synonym) pair that matches and
# keep the LONGEST matching synonym; list order is only the tie-break for two
# synonyms of identical length. That is why "City/State" maps to ``city_state``
# (synonym "city/state", 10 chars) and not to ``city`` (synonym "city", 4), and
# why "Site Address" maps to ``street_address`` ("site address", 12) and not to
# ``facility_name``. Under the old first-match rule the answer depended on which
# field happened to sit higher in this tuple, so reordering any pair merely
# moved the collision to a different pair instead of removing it.
#
# Two further rules keep the substring search honest:
#   * a synonym of 3 characters or fewer must match on WORD BOUNDARIES ("ST"
#     is the state column; "Estimated Cost" is not), and
#   * a field is claimed at most once per table — the leftmost column that
#     matches it wins, and later columns fall through to their next-best field.
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
    ("city_state", ("city/state", "city, state", "city / state", "city-state")),
    # Region / zone is an ORGANISATIONAL grouping, not geography. Clayton's
    # Region column reads "TEN" / "SCA" — internal territory codes that are not
    # a city, not a state and not parseable as either. It used to be a
    # ``city_state`` synonym, which stamped those codes onto 419/438 site atoms
    # as if they were places. A roster that carries only a region and no
    # city/state knows no city/state: that is an abstain, not a guess.
    ("region", ("region",)),
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


#: A synonym this short or shorter is matched on word boundaries rather than as
#: a bare substring. "st" is the state column when the header IS "ST"; inside
#: "Estimated Cost" it is three letters of a different word entirely.
_SHORT_SYNONYM_MAX = 3

#: (field, synonym, length, list_position, matcher) for every synonym, ordered
#: most-specific-first so the first match found for a header is the best one.
#: ``matcher`` is a callable ``header -> bool``. Built once at import; the
#: ordering is total and deterministic (longer synonym first, then original list
#: position, then the synonym text) so the mapping never depends on set or dict
#: iteration order.
def _synonym_matcher(synonym: str):
    token = synonym.strip()
    if len(token) <= _SHORT_SYNONYM_MAX:
        pattern = re.compile(r"\b" + re.escape(token) + r"\b")
        return pattern.search
    return lambda header, _s=synonym: _s in header


_SYNONYM_RULES: tuple[tuple[str, str, object], ...] = tuple(
    (field_name, synonym, _synonym_matcher(synonym))
    for _len, _pos, _syn, field_name, synonym in sorted(
        (
            (-len(synonym.strip()), pos, synonym, field_name, synonym)
            for pos, (field_name, synonyms) in enumerate(_FIELD_HEADER_PATTERNS)
            for synonym in synonyms
        )
    )
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


# A roster names SITES. Every presence-signal above pairs a locational field
# with one of these — except ``{street_address, city}``, which names only a
# PLACE. Every invoice, quotation and letterhead in existence carries an
# address and a city, so that pair alone matched them all and minted a ghost
# site per billing block. A signal match now additionally requires the table
# to identify a site, not merely sit at one.
_SITE_IDENTITY_FIELDS: frozenset[str] = frozenset({
    "site_id", "facility_name", "mdf_idf",
})


# Markers of an account / billing header block: the fields a remittance or
# quotation carries and a site roster never does. Negative evidence, not a
# hard reject — "Customer:" alone appears in plenty of real SOW rosters — so
# two *independent* markers are required before a block is ruled a billing
# header. Grouped by concept so that spelling variants of the SAME field
# ("Account #" / "Account No" / "Account Number") count once, not twice.
# Universal vocabulary, no customer terminology.
_BILLING_HEADER_MARKER_GROUPS: tuple[tuple[str, ...], ...] = (
    ("account #", "account no", "account number", "acct #", "acct no"),
    ("exp date", "expiration date", "expires"),
    ("bill to", "billing address", "billed to"),
    ("remit to", "remittance"),
    ("rqstd by", "requested by", "req by"),
    ("customer:", "customer #", "customer no"),
    ("invoice",),
    ("po #", "p o #", "po no", "purchase order"),
    ("terms:", "payment terms"),
    ("quotation", "quote #"),
)

_MIN_BILLING_MARKERS = 2

#: Marker matching is whitespace-tolerant ("Account #", "Account#",
#: "Account  #" are one marker) and punctuation-preserving (the ``#`` and
#: ``:`` ARE the signal — a bare "customer" is just a word).
_BILLING_MARKER_RES: tuple[tuple[re.Pattern[str], ...], ...] = tuple(
    tuple(
        re.compile(r"\b" + r"\s*".join(re.escape(tok) for tok in marker.split()))
        for marker in group
    )
    for group in _BILLING_HEADER_MARKER_GROUPS
)


def _looks_like_billing_header(*parts: str) -> bool:
    """Do these fragments carry two-or-more independent account-header markers?"""
    blob = " ".join(p.lower() for p in parts if p)
    if not blob:
        return False
    hits = sum(
        1 for group in _BILLING_MARKER_RES if any(r.search(blob) for r in group)
    )
    return hits >= _MIN_BILLING_MARKERS


# Fields that only a site roster carries. A ``site_id`` column alone is not
# enough (BOMs, decision logs and port maps all reference a site); one of
# these must be present for a table to be a roster. Shared with xlsx_parser
# so the strict gate and the header finder agree on one definition.
ROSTER_SPECIFIC_FIELDS: frozenset[str] = frozenset({
    "facility_name", "street_address", "mdf_idf",
    "access_window", "escort_owner", "city_state",
})


# Asset inventories (Asset ID / Serial / Model / IP / MAC / Hostname) mint
# ghost sites: AST-001 matches the site-ID shape and "IP Address" matches the
# street-address header. No real site roster ever carries these, so a single
# hit is decisive — reject regardless of what else the header maps.
_HARD_REJECT_HEADER_SIGNALS: frozenset[str] = frozenset({
    "serial", "mac address", "ip address", "asset id", "asset tag", "hostname",
})

# Signals that a table is a requirements list / BOM / risk register / schedule
# rather than a roster. These are NOT decisive on their own: real rosters
# routinely carry an operational column ("Migration Status", "Open/Closed",
# "Type", "Description"). Reject only when the roster evidence is weak — see
# ``_roster_evidence_is_dominant``.
_SOFT_REJECT_HEADER_SIGNALS: frozenset[str] = frozenset({
    "requirement", "acceptance", "bom", "line item", "unit price",
    "quantity", "risk id", "probability", "impact", "mitigation",
    "checklist", "checkpoint", "deliverable", "criterion", "criteria",
    "phase", "milestone", "task", "activity", "duration", "predecessor",
    "status", "priority", "category", "description",
    "part number", "part no", "model number", "model no", "sku",
    "unit cost", "extended", "subtotal", "total cost",
    "shall", "must", "will provide",
})


def _roster_evidence_is_dominant(columns: Sequence[str]) -> bool:
    """Does the header map so strongly to roster fields that one incidental
    non-roster column (a Status / Type / Description) cannot be the truth of
    the table? Requires >=3 canonical fields AND >=2 roster-specific ones."""
    present = set(map_columns_to_fields(columns).values())
    return len(present) >= 3 and len(present & ROSTER_SPECIFIC_FIELDS) >= 2


# A column that structurally IS the row's identifier: no spaces, short, and
# carrying at least one digit (so human names — "David", "Bradshaw" — can
# never be mistaken for site codes).
_CODE_SHAPE_RE = re.compile(r"^(?=[^\d]*\d)[A-Za-z0-9][A-Za-z0-9._/-]*$")
_MAX_ID_LEN = 16
# Pure-digit columns of these uniform widths are phone numbers / ZIPs /
# ZIP+4 / EINs, not site codes. Real store codes vary in width (58, 268, 1054).
_NON_ID_UNIFORM_DIGIT_WIDTHS: frozenset[int] = frozenset({5, 9, 10, 11})


def detect_site_id_column(
    columns: Sequence[str],
    rows: Sequence[Any],
    field_map: dict[int, str],
) -> int | None:
    """Find the site-identifier column by SHAPE when no header names one.

    Every enterprise roster keys its rows on some code — but the header for it
    is site-specific jargon ("HC", "Loc", "Br #", "Unit") that no keyword list
    can enumerate. The identifier is recognisable structurally instead: the
    column is near-fully populated, its values are unique, short, space-free
    and digit-bearing. Returns the leftmost qualifying column index, or None.
    """
    if "site_id" in field_map.values():
        return None  # a header already declared it
    n = len(rows)
    if n < 3:
        return None
    for i, _col in enumerate(columns):
        if i in field_map:
            continue
        vals = [v for v in (_cell_value(r, columns, i).strip() for r in rows) if v]
        if len(vals) < max(3, int(0.9 * n)):
            continue  # identifiers are not sparse
        if len(set(vals)) < 0.95 * len(vals):
            continue  # identifiers are unique
        if any(len(v) > _MAX_ID_LEN or not _CODE_SHAPE_RE.match(v) for v in vals):
            continue
        widths = {len(v) for v in vals if v.isdigit()}
        if (len(widths) == 1 and all(v.isdigit() for v in vals)
                and next(iter(widths)) in _NON_ID_UNIFORM_DIGIT_WIDTHS):
            continue  # uniform-width digits: phone / ZIP / EIN, not a site code
        return i
    return None


def _id_prefix_for(header: str) -> str:
    """Namespace a bare numeric code by its own column header: ``896`` under an
    ``HC`` column becomes ``HC-896``. A naked number is not a usable site id —
    downstream gates read 4-digit integers as years/amounts and reject them,
    and two sheets keyed on different numbering would collide."""
    slug = re.sub(r"[^A-Za-z0-9]+", "", header or "").upper()[:8]
    return slug or "SITE"


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
    #: Organisational territory ("TEN", "SCA", "Midwest"). NOT geography — it
    #: is carried so the column is never lost, but it never feeds city/state.
    region: str | None = None
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
            "region": self.region,
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
    # Machine-generated sheets ship snake_case headers (``Site_Name``,
    # ``Manager_Name``, ``Visit_Date``). Treat ``_`` as a word separator so
    # they map to the same fields as their spaced equivalents — otherwise a
    # perfectly good roster reads as a table of unknown columns.
    return re.sub(r"\s+", " ", re.sub(r"_+", " ", (text or "")).strip().lower())


def find_header_row(rows: Sequence[Any], scan_limit: int = 25) -> int:
    """Index of the row that actually carries the column headers.

    Real workbooks put a title banner above the header (``Site Master`` /
    ``All in-scope sites with dispatch metadata`` / then the real header).
    Anchoring on "first non-blank row" reads the banner as the entire header,
    so the sheet maps to zero fields and the whole roster is invisible.

    Returns the FIRST row that maps to >=2 canonical fields including at least
    one roster-specific one — scanning forward, never into the data, because a
    data row cannot out-score a header it sits beneath. Falls back to the first
    non-blank row when nothing qualifies (previous behaviour).
    """
    first_nonblank = 0
    seen_nonblank = False
    limit = min(scan_limit, len(rows))
    for i in range(limit):
        cells = [str(c or "").strip() for c in (rows[i] or ())]
        nonblank = [c for c in cells if c]
        if not nonblank:
            continue
        if not seen_nonblank:
            first_nonblank, seen_nonblank = i, True
        if len(nonblank) < 2:
            continue  # merged title / caption banner, not a header
        present = set(map_columns_to_fields(cells).values())
        if len(present) >= 2 and (present & ROSTER_SPECIFIC_FIELDS):
            return i
    return first_nonblank


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
        # _SYNONYM_RULES is pre-sorted longest-synonym-first, so the FIRST rule
        # that matches an available field is by construction the most specific
        # reading of this header.
        for field_name, _synonym, matches in _SYNONYM_RULES:
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
            if matches(header):
                out[i] = field_name
                used_fields.add(field_name)
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
    # v48 FIX 3 / v58: Negative guard. Asset-inventory signals are decisive
    # (they mint ghost sites). Requirements / BOM / schedule signals are only
    # decisive when the roster evidence is weak — a real store roster carries
    # "Migration Status" and "Type" columns without ceasing to be a roster,
    # and rejecting on one such hit silently drops the whole site list.
    if columns:
        header_blob = " ".join(c.lower() for c in columns)
        if any(sig in header_blob for sig in _HARD_REJECT_HEADER_SIGNALS):
            return False
        if any(sig in header_blob for sig in _SOFT_REJECT_HEADER_SIGNALS):
            if not _roster_evidence_is_dominant(columns):
                return False

    # Signal 1: explicit declaration
    if _KIND_PHYSICAL_SITE_DECLARATION.search(surrounding_text):
        return True

    # Negative evidence: an account / billing header block carries an address
    # and a city like a roster does, but it is a remittance stub, not a site
    # list. Applied only to the *heuristic* signals below — an explicit
    # ``kind=physical_site`` declaration above still wins.
    if _looks_like_billing_header(surrounding_text, " ".join(columns or ())):
        return False

    # Signal 2: column header presence
    col_map = map_columns_to_fields(columns)
    fields_present = set(col_map.values())
    for signal_set in _ROSTER_HEADER_PRESENCE_SIGNALS:
        if not signal_set.issubset(fields_present):
            continue
        # A roster must identify a SITE, not just a place. Address+city alone
        # describes every letterhead ever printed.
        if not (fields_present & _SITE_IDENTITY_FIELDS):
            continue
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

    # No header named the identifier column ("HC", "Loc", "Br #", "Unit") —
    # find it by shape. Without this every row arrives with site_id=None and
    # is later discarded downstream as an id-less ghost.
    id_prefix = ""
    id_col = detect_site_id_column(columns, rows, field_map)
    if id_col is not None:
        field_map[id_col] = "site_id"
        id_prefix = _id_prefix_for(str(columns[id_col]))

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

        # Namespace a bare numeric code by its column ("896" -> "HC-896").
        if id_prefix and cells.get("site_id", "").isdigit():
            cells["site_id"] = f"{id_prefix}-{cells['site_id']}"

        # A row with no id, no name AND no address is just noise. An address
        # alone IS a site: rosters routinely leave the display name blank for
        # a store that hasn't been branded yet, and dropping those rows loses
        # real, fully-addressed locations.
        if not (cells.get("site_id") or cells.get("facility_name")
                or cells.get("street_address")):
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

        # Three tiers, not two: a row with neither an id nor a facility name is
        # still a site when the address anchors it, and it should not inherit
        # the confidence of a named row.
        if cells.get("site_id"):
            confidence = 0.85
        elif cells.get("facility_name"):
            confidence = 0.6
        else:
            confidence = 0.5  # address-anchored only

        from app.core.address_parse import (
            enrich_location_fields,
            split_city_state_strict,
        )

        # A combined "City/State" cell is split here, GUESS-FREE, before
        # enrichment: only "<name>, <2-letter state>" and "<name>, <full state
        # name>" resolve. "Nashville" on its own and "Springfield, Springfield"
        # resolve to nothing and leave city/state empty rather than inventing a
        # pair. The raw combined value stays on the row as ``city_state``.
        # (Passing the split parts instead of the raw cell also keeps
        # ``enrich_location_fields``' lenient fallback out of this path —
        # roster columns abstain where prose may guess.)
        cs_city, cs_state = split_city_state_strict(cells.get("city_state"))

        loc = enrich_location_fields(
            street_address=cells.get("street_address"),
            city=cells.get("city") or cs_city,
            state=cells.get("state") or cs_state,
            zip_code=cells.get("zip"),
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
                region=cells.get("region"),
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


# ── Roster preference (a RANKING, never a filter) ────────────────
#
# Clayton ships two workbooks that both pass the roster gate: "Exhibit A -
# Retail Locations.xlsx" (the actual store list) and "Clayton Homes CALC.xlsx"
# (a travel-cost model that happens to carry an address column). Nothing told
# the parser which one is the authoritative roster, so the cost model — with
# more rows — dominated the site base.
#
# The fix is a SIGNAL, not a gate. Every roster candidate is scored on how
# site-roster-like its filename and sheet name read, and the score / rank /
# winner are stamped on the emitted atoms' provenance. Downstream consumers can
# then prefer the best roster. NOTHING IS EVER DROPPED: a low-ranked sheet still
# emits every one of its atoms, still carries its evidence, and is still
# reachable — it simply sorts below a better roster. A hard filter here would
# destroy evidence on a heuristic, which is exactly what this module must not do.
#
# Markers are generic roster vocabulary ("retail locations", "site list") and
# generic non-roster vocabulary ("calc", "travel", "pricing") — never a customer
# name. Every marker that matches contributes, so a name that reads as a roster
# in several ways scores above one that reads that way once.

#: Names that say "this table IS the list of sites".
_ROSTER_POSITIVE_MARKERS: tuple[tuple[str, float], ...] = (
    ("retail locations", 3.0),
    ("location roster", 3.0),
    ("site roster", 3.0),
    ("store roster", 3.0),
    ("site list", 3.0),
    ("store list", 3.0),
    ("location list", 3.0),
    ("branch list", 3.0),
    ("site master", 3.0),
    ("site table", 3.0),
    ("exhibit a", 2.0),
    ("locations", 2.0),
    ("roster", 2.0),
    ("sites", 1.0),
    ("stores", 1.0),
    ("branches", 1.0),
    ("facilities", 1.0),
    ("addresses", 1.0),
)

#: Names that say "this table is a model / schedule / price book that merely
#: mentions sites".
_ROSTER_NEGATIVE_MARKERS: tuple[tuple[str, float], ...] = (
    ("calc", -2.0),
    ("travel", -2.0),
    ("cost", -2.0),
    ("pricing", -2.0),
    ("price", -2.0),
    ("rates", -2.0),
    ("rate card", -2.0),
    ("gantt", -2.0),
    ("financial", -2.0),
    ("budget", -2.0),
    ("margin", -2.0),
    ("invoice", -2.0),
    ("quote", -2.0),
    ("estimate", -1.0),
    ("forecast", -1.0),
)

#: Secondary, structural term: a candidate whose header genuinely maps roster
#: fields is more roster-like than one that does not. Deliberately small so a
#: name marker always outweighs it — a "Travel Cost" sheet with an address
#: column must not out-rank a "Retail Locations" sheet.
_ROSTER_FIELD_BONUS = 0.5
_ROSTER_FIELD_BONUS_CAP = 2.0


def roster_preference_score(
    *,
    filename: str = "",
    sheet_name: str = "",
    columns: Sequence[str] = (),
) -> float:
    """How site-roster-like is this candidate? Higher is better; may be negative.

    Deterministic and purely local — it reads only the candidate's own name and
    header, so two parsers scoring the same sheet always agree without needing
    to see each other's work. This is a preference signal for ranking; it never
    decides whether a table is a roster (``looks_like_site_roster`` does that)
    and it never suppresses anything.
    """
    blob = f"{filename} {sheet_name}".lower()
    score = 0.0
    for marker, weight in _ROSTER_POSITIVE_MARKERS:
        if marker in blob:
            score += weight
    for marker, weight in _ROSTER_NEGATIVE_MARKERS:
        if marker in blob:
            score += weight
    if columns:
        mapped = set(map_columns_to_fields(columns).values()) & ROSTER_SPECIFIC_FIELDS
        score += min(len(mapped) * _ROSTER_FIELD_BONUS, _ROSTER_FIELD_BONUS_CAP)
    return score


def rank_roster_candidates(
    candidates: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Rank roster candidates best-first. Returns EVERY candidate, never fewer.

    Each candidate is a mapping that may carry ``filename``, ``sheet_name`` and
    ``columns``; any other keys are passed through untouched so a caller can
    carry its own payload. Each returned dict gains:

    ``roster_score``     the raw preference score. This is the comparable
                         quantity ACROSS calls — two workbooks ranked separately
                         can still be compared on it.
    ``roster_rank``      1 = best WITHIN this call. Dense ranking — tied scores
                         share a rank and the next distinct score takes the next
                         integer.
    ``roster_preferred`` True for a rank-1 candidate whose score is not negative.
                         A negative score means the candidate's own name argues
                         against it being a roster ("Travel Cost", "Rate Card"),
                         and winning a field of one such candidate does not make
                         it a good roster — only the best of a bad set. It still
                         ranks, and it still keeps every atom; it is simply not
                         held out as the roster to prefer.

    Results come back in the CALLER'S original order, so ranking a list twice —
    or ranking a re-ordered list — produces the same per-candidate answer
    (order-invariant and idempotent). Ties are resolved by score alone, never by
    position, so no candidate is privileged for arriving first.
    """
    scored: list[tuple[float, dict[str, Any]]] = []
    for cand in candidates:
        score = roster_preference_score(
            filename=str(cand.get("filename") or ""),
            sheet_name=str(cand.get("sheet_name") or ""),
            columns=cand.get("columns") or (),
        )
        scored.append((score, cand))

    distinct = sorted({s for s, _ in scored}, reverse=True)
    rank_of = {s: i + 1 for i, s in enumerate(distinct)}

    out: list[dict[str, Any]] = []
    for score, cand in scored:
        rank = rank_of[score]
        merged = dict(cand)
        merged["roster_score"] = score
        merged["roster_rank"] = rank
        merged["roster_preferred"] = rank == 1 and score >= 0
        out.append(merged)
    return out


__all__ = [
    "ROSTER_SPECIFIC_FIELDS",
    "SiteRosterRow",
    "detect_site_id_column",
    "extract_site_roster",
    "find_header_row",
    "looks_like_site_roster",
    "map_columns_to_fields",
    "rank_roster_candidates",
    "roster_preference_score",
]
