from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from openpyxl import load_workbook
from openpyxl.utils import get_column_letter

from app.core.ids import stable_id
from app.core.item_identity import merge_parser_value_identity
from app.core.normalizers import normalize_entity_key, normalize_text, parse_quantity
from app.core.segments import ArtifactSegment
from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    EvidenceAtom,
    ParserOutput,
    ReviewStatus,
    SourceRef,
    ParserCapability,
    ParserMatch,
)
from app.parsers.base import BaseParser
from app.parsers.binary_markers import emit_zip_binary_markers
from app.parsers.segmenters import segment_xlsx
from app.parsers.sheet_classifier import (
    SheetDestination,
    SheetRole,
    classify_sheet,
)
from app.parsers.structured_projection import (
    derived_files_for,
    make_page,
    make_section,
    make_structured_document,
    make_table,
    stamp_section_and_block_ids,
)
from app.domain.schemas import DomainPack

parser_name = "xlsx"
parser_version = "xlsx_parser_v2_1"

# Consumed by parser-os-service `_attachments_status` after compile.
ARTIFACT_PARSE_ERROR_PREFIX = "artifact_parse_error:"
STRUCTURED_SCHEMA_XLSX = "orbitbrief.xlsx.structured.v1"
STRUCTURED_SCHEMA_CSV = "orbitbrief.csv.structured.v1"

# Cell-fact patterns (PR2). When a generic row's individual cell text
# matches one of these patterns we ALSO emit a sub-atom typed as
# ``exclusion`` or ``risk``, anchored to the parent row by
# ``value.parent_row_atom_id``. This stops "EOL", "not included",
# "single point of failure", … from being buried inside generic
# row text where the packetizer can't find them.
_EXCLUSION_CELL_RE = re.compile(
    # PR3 (post-v3 review) — dropped "unless noted". That phrase is
    # almost always a SUPPORT-TIER conditional ("8x5 vendor support
    # unless noted"), not a contractual exclusion. The packetizer
    # exclusion gate now also checks the canonical_field so a cell
    # in a support_level / coverage / support_entitlement column
    # never participates in a scope_exclusion packet.
    r"\b(exclud(?:e|ed|es|ing)|not covered|not included|unsupported|"
    r"no sla|outside coverage)\b",
    re.I,
)
_RISK_CELL_RE = re.compile(
    r"\b(eol|end of life|unsupported|critical|high risk|"
    r"single point of failure|mismatch|missing)\b",
    re.I,
)
# PR3 — conditional support boundary phrasing. "8x5 vendor support
# unless noted" / "no SLA if expired" / "coverage exclusion if expired".
# These are SUPPORT TIER descriptions, not contractual exclusions.
_CONDITIONAL_SUPPORT_RE = re.compile(
    r"\b(unless\s+noted|sla\s+exclusion\s+if\s+expired|"
    r"no\s+sla\s+if\s+expired|coverage\s+exclusion\s+if\s+expired|"
    r"unless\s+otherwise\s+noted|where\s+applicable)\b",
    re.I,
)
_SUPPORT_LEVEL_FIELDS: frozenset[str] = frozenset(
    {
        "support_level",
        "coverage",
        "support_entitlement",
        "support_tier",
    }
)


# PR2 (post-v3 review) — per-sheet profile for operational workbooks.
# Used by ``XlsxParser._emit_operational_sheet_rows`` to dispatch
# every row of every named sheet to its proper AtomType + value.kind
# + authority class.
@dataclass(frozen=True)
class _OperationalSheetProfile:
    atom_type: AtomType
    kind: str
    authority_class: AuthorityClass
    confidence: float


_OPERATIONAL_SHEET_PROFILES: dict[str, _OperationalSheetProfile] = {
    "readme": _OperationalSheetProfile(
        AtomType.project_metadata,
        "project_metadata_row",
        AuthorityClass.customer_current_authored,
        0.90,
    ),
    "dashboard": _OperationalSheetProfile(
        AtomType.project_metadata,
        "dashboard_metric_row",
        AuthorityClass.customer_current_authored,
        0.90,
    ),
    "asset inventory": _OperationalSheetProfile(
        AtomType.asset_record,
        "asset_inventory_row",
        AuthorityClass.approved_site_roster,
        0.94,
    ),
    "site survey raw": _OperationalSheetProfile(
        AtomType.site_survey_row,
        "site_survey_row",
        AuthorityClass.customer_current_authored,
        0.92,
    ),
    "port map & vlans": _OperationalSheetProfile(
        AtomType.port_vlan_assignment,
        "port_vlan_assignment_row",
        AuthorityClass.approved_site_roster,
        0.94,
    ),
    "port map vlans": _OperationalSheetProfile(
        AtomType.port_vlan_assignment,
        "port_vlan_assignment_row",
        AuthorityClass.approved_site_roster,
        0.94,
    ),
    "circuit inventory": _OperationalSheetProfile(
        AtomType.circuit_inventory,
        "circuit_inventory_row",
        AuthorityClass.approved_site_roster,
        0.92,
    ),
    "license support": _OperationalSheetProfile(
        AtomType.support_entitlement,
        "support_entitlement_row",
        AuthorityClass.vendor_quote,
        0.92,
    ),
    "noc alert matrix": _OperationalSheetProfile(
        AtomType.alert_route,
        "alert_route_row",
        AuthorityClass.customer_current_authored,
        0.92,
    ),
    "risk register": _OperationalSheetProfile(
        AtomType.risk,
        "risk_register_row",
        AuthorityClass.customer_current_authored,
        0.93,
    ),
    "cutover validation": _OperationalSheetProfile(
        AtomType.cutover_validation,
        "cutover_validation_row",
        AuthorityClass.customer_current_authored,
        0.93,
    ),
    "source refs": _OperationalSheetProfile(
        AtomType.project_metadata,
        "source_reference_row",
        AuthorityClass.machine_extractor,
        0.75,
    ),
}


# Service-line classifier — when a BOM line item description matches
# one of these tokens (case-insensitive substring match), the line is
# routed to `service:` instead of `device:`. Real device names rarely
# include these tokens; service line items almost always do.
_SERVICE_LINE_TOKENS: tuple[str, ...] = (
    "labor", "labour", "hours", "hour", "after-hours", "after hours",
    "support", "supports", "supported",
    "training", "trainings", "adoption",
    "workshop", "workshops", "discovery",
    "design", "designs", "engineering services",
    "professional services", "managed services",
    "consulting", "consultancy", "advisory",
    "hypercare", "warranty", "warrantee",
    "project management", "program management", "pmo",
    "governance", "oversight",
    "implementation", "installation services", "deployment",
    "commissioning", "decommissioning",
    "migration", "cutover", "go-live",
    "documentation services", "as-built", "as built",
    "testing services", "validation", "uat", "acceptance",
    "rfp response", "proposal preparation",
)


def _looks_like_service_line(value: str) -> bool:
    """Return True if a BOM line-item description is a service rather
    than a physical device.

    Routes labor / support / training / hypercare / governance /
    consulting / project-management line items to `service:<slug>`
    instead of `device:<slug>` so the device namespace stays clean.
    """
    lower = value.lower()
    return any(token in lower for token in _SERVICE_LINE_TOKENS)


def _norm_op_sheet(sheet_name: str) -> str:
    return normalize_text(sheet_name).replace("_", " ").strip()


# ── Commercial-sheet money extraction ───────────────────────────────
# Pricing in estimating workbooks is stored as bare numeric cells, not
# "$"-formatted text, so the symbol-based money extractor in
# entity_extraction misses it. These helpers recover money two ways:
#   • money columns  — a header cell naming a money concept (cost / sell /
#     price / rate / total / revenue / margin …) over numeric data cells
#     (master catalogs, rate cards).
#   • label→value    — a money-concept label cell paired with the nearest
#     numeric cell to its right on the same row (deal-financials summaries
#     like "Total Deal Revenue | 21560").
_MONEY_CONCEPT_RE = re.compile(
    r"revenue|cost|margin|price|\bsell\b|\bfee\b|budget|subtotal|amount"
    r"|\brate\b|charge|expense|extended|unit\s*price|total",
    re.I,
)
# Floor that rejects tax multipliers (1.09 / 1.34), margin ratios (0.27),
# and other sub-dollar line noise while keeping genuine prices/totals.
_MIN_MONEY_VALUE = 5.0


def _is_money_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and abs(value) >= _MIN_MONEY_VALUE
    )


def _money_columns(rows: list[list[Any]]) -> set[int]:
    """Column indices whose header names a money concept (% columns excluded)."""
    cols: set[int] = set()
    for row in rows[:20]:
        for idx, c in enumerate(row):
            t = str(c or "").strip()
            if t and "%" not in t and _MONEY_CONCEPT_RE.search(t):
                cols.add(idx)
    return cols


def _row_money_values(row: list[Any], money_cols: set[int]) -> list[float]:
    """Money amounts on a row, via money columns and label→value pairs."""
    vals: list[float] = []
    for idx in money_cols:
        if idx < len(row) and _is_money_number(row[idx]):
            vals.append(float(row[idx]))
    for idx, c in enumerate(row):
        t = str(c or "").strip()
        if not t or "%" in t or not _MONEY_CONCEPT_RE.search(t):
            continue
        for j in range(idx + 1, len(row)):
            if _is_money_number(row[j]):
                vals.append(float(row[j]))
                break
    return vals


# ── Financial-summary (Deal Kit / P&L) structured extraction ─────────
# Deal-kit / estimating workbooks lay the deal economics out as a 2-D
# label→value grid, not a row table, so the generic row emitter mashes
# unrelated cells together. These helpers read the grid the way a human
# does: find a known label, take the value to its right. The vocabulary
# is the standard estimating-workbook field set (revenue/cost/margin by
# category + deal header fields) — universal across the org's deals, no
# customer-specific terms.

# P&L line metric: "<Category> Revenue|Cost|Margin" (with optional
# leading "Total"). Captures the category and which metric.
_PL_METRIC_RE = re.compile(
    r"^(?:total\s+)?(?P<cat>.+?)\s+(?P<metric>revenue|cost|margin)$", re.I
)
# Margin-percent line: "Margin % on <Category>".
_PL_MARGIN_PCT_RE = re.compile(r"^margin\s*%\s*on\s+(?P<cat>.+)$", re.I)

# Deal-header fields → normalized key. Matched against the lowercased,
# whitespace-collapsed label cell.
_DEAL_HEADER_LABELS: dict[str, str] = {
    "oppty #": "opportunity_id",
    "oppty#": "opportunity_id",
    "opportunity #": "opportunity_id",
    "opportunity#": "opportunity_id",
    "sales rep": "sales_rep",
    "customer": "customer",
    "end user": "end_user",
    "quoted w/ partner": "quoted_with_partner",
    "qty of sites": "site_count",
    "# of sites": "site_count",
    "division": "division",
    "project duration (months)": "project_duration",
    "project duration": "project_duration",
    "billing type": "billing_type",
    "region": "region",
    "channel/direct": "channel",
    "enterprise/ technical": "segment",
    "enterprise/technical": "segment",
    "date": "deal_date",
}

# Don't treat these as values when they sit to the right of a label —
# they are themselves column labels in the adjacent block of the grid.
_PL_METRIC_WORDS = frozenset({"revenue", "cost", "margin"})


def _norm_label(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _is_label_cell(text: str) -> bool:
    """True when a cell reads like a label (a header/P&L term), so it is
    never mistaken for the *value* of the label to its left."""
    if text in _DEAL_HEADER_LABELS:
        return True
    if _PL_MARGIN_PCT_RE.match(text):
        return True
    m = _PL_METRIC_RE.match(text)
    return bool(m and m.group("metric").lower() in _PL_METRIC_WORDS)


def _coerce_pl_number(value: Any) -> float | None:
    """Parse a financial cell to a float. Returns None for blanks,
    Excel errors (#DIV/0!), and non-numeric text (TBD / No)."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s or s.startswith("#"):
        return None
    neg = s.startswith("(") and s.endswith(")")
    s2 = s.replace("(", "").replace(")", "").replace("$", "").replace(",", "").replace("%", "").strip()
    if not s2:
        return None
    try:
        n = float(s2)
    except ValueError:
        return None
    return -n if neg else n


def _norm_pl_category(raw: str) -> tuple[str, str]:
    """Return (key, display) for a P&L category label.

    ``key`` is a slug for grouping; ``display`` is a cleaned human label.
    'Total Deal' / 'Deal' collapse to the same key so the summary line and
    the detail block don't split into two categories."""
    disp = re.sub(r"\s+", " ", raw.strip())
    disp = re.sub(r"^total\s+", "", disp, flags=re.I).strip() or disp
    key = re.sub(r"[^a-z0-9]+", "_", disp.lower()).strip("_")
    # 'deal' is the grand-total category — normalize a few synonyms.
    if key in {"deal", "overall_deal", "project"}:
        key = "deal"
        disp = "Deal"
    return key, disp


def _right_neighbor_value(row: list[Any], idx: int) -> Any:
    """The first non-empty cell to the right of ``idx`` in ``row``."""
    for j in range(idx + 1, len(row)):
        v = row[j]
        if v is not None and str(v).strip():
            return v
    return None


def _looks_like_header_label(text: str) -> bool:
    """Structural test for a deal-header *label* cell, independent of the
    canonical vocabulary.

    A header label is a short, mostly-text caption ("PO Number", "Account
    Manager", "Site Contact") sitting to the left of its value. This lets
    the extractor capture deal-kit header fields that aren't in the
    canonical map, so non-standard fields are never silently dropped. P&L
    metric / margin lines are excluded — they belong to the P&L block, not
    the header."""
    t = re.sub(r"\s+", " ", str(text or "").strip()).rstrip(":").strip()
    if not t:
        return False
    if not re.search(r"[A-Za-z]", t):  # need a letter; pure numbers aren't labels
        return False
    if _PL_MARGIN_PCT_RE.match(t):
        return False
    m = _PL_METRIC_RE.match(t)
    if m and m.group("metric").lower() in _PL_METRIC_WORDS:
        return False
    # Labels are captions, not prose: keep them short.
    return len(t.split()) <= 6


# Excel error literals leak into cells as text (#DIV/0!, #REF!, #N/A, …).
# They are never a real field value — a formula failed — so they must
# never be captured as a deal-header field.
_EXCEL_ERROR_RE = re.compile(
    r"^#(?:DIV/0!|REF!|N/A|VALUE!|NAME\?|NULL!|NUM!|SPILL!|CALC!|GETTING_DATA|#+)$",
    re.I,
)


def _coerce_header_value(val: Any) -> str | None:
    """Normalize a header *value* cell to a string, or None when it isn't a
    usable value (blank, an Excel error literal, or itself a label/P&L term)."""
    if val is None:
        return None
    if hasattr(val, "isoformat"):
        try:
            val = val.isoformat()
        except Exception:
            val = str(val)
    sval = re.sub(r"\s+", " ", str(val).strip())
    if not sval:
        return None
    if _EXCEL_ERROR_RE.match(sval):
        return None
    if _is_label_cell(_norm_label(sval)):
        return None
    return sval


def _first_nonblank_header_row(rows: list[list[Any]]) -> int | None:
    for i, row in enumerate(rows[:20]):
        nonblank = [str(c).strip() for c in row if str(c or "").strip()]
        if len(nonblank) >= 2:
            return i
    return None


def _cell_to_text_op(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return str(value).strip()

# Canonical column role -> normalized header tokens (lowercase, punctuation-stripped variants).
HEADER_ALIASES: dict[str, set[str]] = {
    "project": {"project", "job", "job number", "project number"},
    "site": {"site", "facility", "building", "school", "campus", "store", "venue"},
    "facility": {"facility name", "facility"},
    "building": {"building", "bldg", "structure"},
    "floor": {"floor", "level", "fl"},
    "wing": {"wing", "sector"},
    "area": {"area", "region"},
    "zone": {"zone", "district"},
    "room": {"room", "room #", "room number", "space"},
    "location": {"location", "position", "place", "mounting location", "area name"},
    "plate_id": {"plate id", "plate", "outlet id", "drop id", "jack id", "cable id", "plate #"},
    "outlet_id": {"outlet id", "outlet"},
    "drop_id": {"drop id", "drop"},
    "mdf": {"mdf", "mdf id", "main distribution frame"},
    "idf": {"idf", "idf id", "intermediate distribution frame"},
    "closet": {"closet", "telecom closet"},
    "rack": {"rack", "rack id"},
    "device": {"device", "asset", "equipment", "camera", "ap", "reader", "hostname"},
    "device_type": {"device type", "type", "equipment type"},
    "item": {"item", "line item", "line"},
    "description": {"description", "desc", "details", "summary"},
    "quantity": {"qty", "qty.", "quantity", "count", "#", "no", "units", "total qty", "# drops"},
    "count": {"count", "cnt"},
    "uom": {"uom", "unit", "units", "ea", "each"},
    "material": {"material", "wire", "cable"},
    "material_spec": {"material spec", "material / spec", "spec", "cable spec"},
    "cable_category": {"cable category", "cable cat"},
    "cable_type": {"cable type", "cable"},
    "shielding": {"shielding", "utp", "stp"},
    "jacket_rating": {"jacket", "plenum", "riser"},
    "connector": {"connector", "jack type"},
    "termination": {"termination", "terminations"},
    "patch_panel": {"patch panel", "panel"},
    "faceplate": {"faceplate", "wall plate", "plate type"},
    "scope": {"scope", "work type", "work package"},
    "included": {"included", "included?", "in scope?", "in scope", "base bid"},
    "excluded": {"excluded", "excluded?", "out of scope"},
    "access": {"access", "access window", "hours", "site access", "work window"},
    "access_window": {"access window", "hours"},
    "lift": {"lift", "lift required", "elevator"},
    "ceiling_access": {"ceiling access", "ceiling"},
    "after_hours": {"after hours", "after-hours", "nights", "weekends"},
    "escort": {"escort", "escort required"},
    "badge": {"badge", "badge access", "badge required", "mdf badge"},
    "customer_responsibility": {"customer responsibility", "owner responsibility", "customer provides"},
    "vendor_responsibility": {"vendor responsibility", "contractor provides"},
    "notes": {"notes", "comments", "clarifications", "assumptions", "remarks"},
    "open_question": {"open question", "question", "confirm"},
    "test_standard": {"test standard", "test standard required"},
    "certification": {"certification", "testing", "test standard", "certify", "tester export", "fluke"},
    "labeling": {"label", "labeling", "label standard"},
    "as_built": {"as built", "as-built", "redlines"},
    "status": {"status", "state"},
}

RowKind = Literal[
    "blank",
    "title",
    "header",
    "section_header",
    "line_item",
    "total",
    "subtotal",
    "grand_total",
    "note",
    "malformed",
]


@dataclass
class SheetParseModel:
    header_idx: int
    header_map: dict[str, int]
    wide_qty_columns: list[dict[str, Any]]
    header_mode: str
    diagnostics: list[str] = field(default_factory=list)


def _header_cell_tokens(cell: Any) -> set[str]:
    raw = str(cell or "").strip()
    if not raw:
        return set()
    lowered = raw.lower()
    tokens: set[str] = {
        normalize_text(raw).strip(".:?"),
        normalize_text(raw.replace("/", " ")).strip(".:?"),
        normalize_text(raw.replace("/", " ").replace("?", "")).strip(".:?"),
        re.sub(r"\s+", " ", lowered).strip(),
    }
    for part in re.split(r"[/|]", raw):
        t = normalize_text(part).strip(".:?! ")
        if t:
            tokens.add(t)
    return {t for t in tokens if t}


def _map_canonical_header(cell: Any) -> str | None:
    for key, aliases in HEADER_ALIASES.items():
        if _header_cell_tokens(cell) & aliases:
            return key
    return None


def _wide_quantity_header_meta(cell: Any) -> dict[str, Any] | None:
    """If this column header is a material/connector quantity column (wide schedule), return metadata."""
    blob = " ".join(sorted(_header_cell_tokens(cell))).lower()
    if not blob.strip():
        return None
    if re.search(r"\brj[-\s]?45\b", blob) or re.search(r"\bdata jack\b", blob) or blob == "jack count":
        return {
            "item": "RJ45",
            "normalized_item": "rj45",
            "item_kind": "termination",
            "material_family": "connector",
            "cable_category": None,
            "shielding": None,
            "entity_hint": ("connector", "rj45"),
        }
    if re.search(r"\bcat6a\b|\bcat\s*6a\b|category\s*6a", blob):
        return {
            "item": "Cat6A",
            "normalized_item": "cat6a",
            "item_kind": "cable_drop",
            "material_family": "cable",
            "cable_category": "cat6a",
            "shielding": None,
            "entity_hint": ("cable", "cat6a"),
        }
    if "fiber" in blob or "strand" in blob:
        return {
            "item": "Fiber",
            "normalized_item": "fiber",
            "item_kind": "cable_drop",
            "material_family": "fiber",
            "cable_category": None,
            "shielding": None,
            "entity_hint": ("cable", "fiber"),
        }
    if re.search(r"\bcat6\b", blob) and ("utp" in blob or "unshielded" in blob or "non.?shielded" in blob):
        return {
            "item": "Cat6 UTP",
            "normalized_item": "cat6_utp",
            "item_kind": "cable_drop",
            "material_family": "cable",
            "cable_category": "cat6",
            "shielding": "unshielded",
            "entity_hint": ("material", "cat6_utp"),
        }
    if re.search(r"\bcat6\b", blob) and ("stp" in blob or "shielded" in blob):
        return {
            "item": "Cat6 STP",
            "normalized_item": "cat6_stp",
            "item_kind": "cable_drop",
            "material_family": "cable",
            "cable_category": "cat6",
            "shielding": "shielded",
            "entity_hint": ("cable", "cat6_stp"),
        }
    if re.search(r"\bcat6\b", blob) and "utp" not in blob and "stp" not in blob and "shield" not in blob:
        return {
            "item": "Cat6",
            "normalized_item": "cat6",
            "item_kind": "cable_drop",
            "material_family": "cable",
            "cable_category": "cat6",
            "shielding": None,
            "entity_hint": ("cable", "cat6"),
        }
    return None


def _header_map_from_row(row: list[Any]) -> tuple[dict[str, int], list[dict[str, Any]]]:
    header_map: dict[str, int] = {}
    wide: list[dict[str, Any]] = []
    seen_wide_cols: set[int] = set()
    for col_idx, cell in enumerate(row):
        canon = _map_canonical_header(cell)
        if canon and canon not in header_map:
            header_map[canon] = col_idx
        meta = _wide_quantity_header_meta(cell)
        if meta and col_idx not in seen_wide_cols:
            seen_wide_cols.add(col_idx)
            wide.append({"col_idx": col_idx, "header_raw": str(cell or "").strip(), **meta})
    return header_map, wide


def _merge_header_cells(top: Any, bottom: Any) -> str:
    a = str(top or "").strip()
    b = str(bottom or "").strip()
    if a and b:
        return f"{a} {b}".strip()
    return a or b


def _header_map_from_two_rows(row_top: list[Any], row_bot: list[Any]) -> tuple[dict[str, int], list[dict[str, Any]]]:
    width = max(len(row_top), len(row_bot))
    merged: list[str] = []
    for col in range(width):
        top = row_top[col] if col < len(row_top) else None
        bot = row_bot[col] if col < len(row_bot) else None
        merged.append(_merge_header_cells(top, bot))
    return _header_map_from_row(merged)


def _sheet_qualifies(header_map: dict[str, int], wide_qty_columns: list[dict[str, Any]]) -> bool:
    if wide_qty_columns:
        return any(header_map.get(k) is not None for k in ("plate_id", "location", "room", "site", "description", "device"))
    entity_any = any(
        header_map.get(k) is not None
        for k in ("site", "device", "location", "room", "plate_id", "building", "floor", "project")
    )
    if header_map.get("quantity") is not None and entity_any:
        return True
    if entity_any and len(header_map) >= 2:
        return True
    return False


def _detect_header(rows: list[list[Any]], scan_limit: int = 45) -> SheetParseModel:
    diagnostics: list[str] = []
    best_idx: int | None = None
    best_map: dict[str, int] = {}
    best_wide: list[dict[str, Any]] = []
    best_score = -1.0
    best_mode = "single"

    limit = min(scan_limit, len(rows))
    for idx in range(limit):
        row = rows[idx]
        hm, wq = _header_map_from_row(row)
        score = len(hm) + 0.45 * len(wq)
        if _sheet_qualifies(hm, wq) and score > best_score:
            best_score = score
            best_idx = idx
            best_map = hm
            best_wide = list(wq)
            best_mode = "single"
        if idx + 1 < len(rows):
            hm2, wq2 = _header_map_from_two_rows(row, rows[idx + 1])
            score2 = len(hm2) + 0.45 * len(wq2) + 0.05
            if _sheet_qualifies(hm2, wq2) and score2 > best_score:
                best_score = score2
                best_idx = idx
                best_map = hm2
                best_wide = list(wq2)
                best_mode = "pair"

    if best_idx is None:
        diagnostics.append("no_header_found")
        return SheetParseModel(-1, {}, [], "none", diagnostics)

    diagnostics.append(f"header_row={best_idx + 1} mode={best_mode} keys={sorted(best_map.keys())} wide_qty_cols={len(best_wide)}")
    if not best_wide and not best_map.get("quantity"):
        diagnostics.append("no_quantity_bearing_column")
    elif best_wide:
        diagnostics.append(f"domain_wide_quantity_columns:{','.join(w['item'] for w in best_wide)}")
    return SheetParseModel(best_idx, best_map, best_wide, best_mode, diagnostics)


def _is_blank_row(row: list[Any]) -> bool:
    return all(str(c or "").strip() == "" for c in row)


# Sheet names that signal "this tab is human instructions, not data".
# Compared after lowercasing + stripping; substring match so workbooks
# titled e.g. ``Instructions (Read First)`` still hit. Universal across
# all xlsx workbooks; not domain-specific.
_INSTRUCTIONAL_SHEET_NAME_TOKENS: frozenset[str] = frozenset({
    "instruction",
    "instructions",
    "readme",
    "read me",
    "read_me",
    "cover",
    "cover sheet",
    "coversheet",
    "about",
    "overview",
    "guide",
    "how to",
    "how-to",
    "table of contents",
    "toc",
    "legend",
    "key",
    "intro",
    "introduction",
    "title page",
})


def _looks_instructional_sheet(sheet_name: str) -> bool:
    """True when the sheet name signals a non-data instructional tab."""
    name = (sheet_name or "").strip().lower()
    if not name:
        return False
    return any(token in name for token in _INSTRUCTIONAL_SHEET_NAME_TOKENS)


def _has_tabular_shape(rows: list[list[Any]]) -> bool:
    """Cheap structural check: does the sheet *look* like a real table?

    True when there are at least 2 non-blank rows AND the widest row has
    >= 2 non-empty cells. False for pure prose tabs (one row of
    instructions, blank rows, single-column dumps).
    """
    non_blank = [r for r in rows if not _is_blank_row(r)]
    if len(non_blank) < 2:
        return False
    widest = max(
        (sum(1 for c in r if str(c or "").strip()) for r in non_blank),
        default=0,
    )
    return widest >= 2


def _generic_table_from_rows(
    rows: list[list[Any]],
) -> tuple[list[str], list[dict[str, Any]]] | None:
    """Build a full table from a tabular sheet whose header the canonical
    detector could not recognise.

    The canonical ``_detect_header`` only fires for sheets whose columns
    match the device/quantity vocabulary it knows. Bill-of-materials, rate
    cards, financial P&Ls and other commercial tables use arbitrary
    headers, so they previously fell through to a 25-row truncated prose
    dump — silently losing most rows and all structure. This recovers
    them generically: pick the widest of the leading rows as the header,
    keep every non-blank data row (no truncation), and never invents
    vocabulary. Returns ``None`` when the sheet is not genuinely tabular.
    """
    if not _has_tabular_shape(rows):
        return None

    def _filled(row: list[Any]) -> int:
        return sum(1 for c in (row or []) if str(c or "").strip())

    # Width = the widest non-blank row's column span (drives column count).
    width = max((len(r or []) for r in rows if not _is_blank_row(r)), default=0)
    if width < 2:
        return None

    # Header = the first of the leading non-blank rows that is the most
    # populated (titles/notes above the real header have fewer cells).
    header_idx = -1
    best_filled = -1
    seen = 0
    for idx, row in enumerate(rows):
        if _is_blank_row(row):
            continue
        seen += 1
        f = _filled(row)
        if f > best_filled:
            best_filled = f
            header_idx = idx
        if seen >= 8:  # only scan the leading band for a header
            break
    if header_idx < 0:
        return None

    header_row = rows[header_idx] or []
    columns: list[str] = []
    used: dict[str, int] = {}
    for i in range(width):
        raw = str(header_row[i]).strip() if i < len(header_row) and header_row[i] is not None else ""
        name = raw or f"col_{i + 1}"
        if name in used:
            used[name] += 1
            name = f"{name}_{used[name]}"
        else:
            used[name] = 1
        columns.append(name)

    table_rows: list[dict[str, Any]] = []
    for row_idx in range(header_idx + 1, len(rows)):
        row = rows[row_idx] or []
        if _is_blank_row(row):
            continue
        cells: dict[str, Any] = {}
        for col_idx, col_name in enumerate(columns):
            value = row[col_idx] if col_idx < len(row) else ""
            cells[col_name] = "" if value is None else str(value).strip()
        table_rows.append(cells)

    if not table_rows:
        return None
    return columns, table_rows


def _label_text_for_row(row: list[Any], label_col_indices: list[int]) -> str:
    parts: list[str] = []
    for idx in label_col_indices:
        if idx < len(row):
            parts.append(str(row[idx] or "").strip())
    return " ".join(p for p in parts if p).strip()


def _label_column_indices(header_map: dict[str, int]) -> list[int]:
    for key in ("plate_id", "location", "description", "item", "site", "room", "drop_id", "outlet_id"):
        if key in header_map:
            return [header_map[key]]
    return [0]


def _is_total_label(text: str) -> bool:
    t = normalize_text(text).strip()
    if not t:
        return False
    return bool(re.match(r"^(totals?|subtotal|grand\s*total)\b", t, re.I))


def _row_kind(
    row: list[Any],
    header_map: dict[str, int],
    label_indices: list[int],
) -> RowKind:
    if _is_blank_row(row):
        return "blank"
    label = _label_text_for_row(row, label_indices)
    low = label.lower()
    if _is_total_label(label):
        if low.startswith("grand"):
            return "grand_total"
        if low.startswith("subtotal") or low.startswith("sub total"):
            return "subtotal"
        return "total"
    if label.endswith(":") and len(label) < 56:
        return "section_header"
    if len(label) > 80 and sum(1 for c in row if str(c or "").strip()) <= 2:
        return "title"
    if label and not any(str(row[i] or "").strip() for i in range(len(row)) if i not in label_indices):
        return "note"
    return "line_item"


def _parse_schedule_quantity_cell(raw_val: Any) -> dict[str, Any]:
    raw = "" if raw_val is None else str(raw_val).strip()
    out: dict[str, Any] = {
        "quantity_raw": raw,
        "quantity": None,
        "quantity_status": "missing",
        "quantity_min": None,
        "quantity_max": None,
        "uom": None,
        "uncertain": True,
        "review_flags": [],
    }
    if raw == "":
        out["quantity_status"] = "missing"
        return out
    low = normalize_text(raw).replace(",", "")
    if low in {"n/a", "na", "tbd", "pending"}:
        out["quantity_status"] = "tbd" if "tbd" in low or "pending" in low else "not_applicable"
        return out
    if "allowance" in low or low == "lot" or "lot" in low:
        out["quantity_status"] = "allowance"
        out["uncertain"] = True
        return out
    if "included" in low and not re.search(r"\d", raw):
        out["quantity_status"] = "included"
        out["uncertain"] = False
        return out
    mr = re.match(r"^\s*(\d[\d,]*)\s*[-–]\s*(\d[\d,]*)\s*$", raw.replace(",", ""))
    if mr:
        lo = int(mr.group(1))
        hi = int(mr.group(2))
        out["quantity_min"] = lo
        out["quantity_max"] = hi
        out["quantity_status"] = "range"
        out["review_flags"].append("xlsx_parser:range_quantity")
        return out
    mq = re.match(r"^\s*(-?\d[\d,]*(?:\.\d+)?)\s*([a-z%]{0,8})?\s*$", low.replace(",", ""))
    if mq:
        q = float(mq.group(1))
        if q.is_integer():
            q = int(q)
        out["quantity"] = q
        out["uom"] = mq.group(2) or None
        out["quantity_status"] = "zero" if q == 0 else "known"
        out["uncertain"] = False
        return out
    legacy = parse_quantity(raw)
    out.update(
        {
            "quantity": legacy.get("quantity"),
            "uom": legacy.get("unit"),
            "uncertain": bool(legacy.get("uncertain", True)),
        }
    )
    out["quantity_status"] = "ambiguous" if legacy.get("uncertain") else ("known" if legacy.get("quantity") is not None else "ambiguous")
    if out["quantity_status"] == "ambiguous":
        out["review_flags"].append("xlsx_parser:ambiguous_quantity")
    return out


def _row_text_blob(row: list[Any], header_map: dict[str, int]) -> str:
    parts: list[str] = []
    for key in ("notes", "description", "scope", "access", "location", "item"):
        idx = header_map.get(key)
        if idx is not None and idx < len(row):
            parts.append(str(row[idx] or ""))
    parts.extend(str(c or "") for c in row)
    return normalize_text(" ".join(parts)).lower()


def _emit_scope_constraint_atoms(
    row_blob: str,
    site: str,
    device: str,
    floor: str,
    room: str,
    location: str,
    append_atom: Any,
    row_confidence: float,
    *,
    context_columns: dict[str, str],
) -> None:
    cols = context_columns

    def ap(
        atom_type: AtomType,
        raw_text: str,
        value: dict[str, Any],
        confidence: float,
        *,
        review_status: ReviewStatus = ReviewStatus.auto_accepted,
        review_flags: list[str] | None = None,
    ) -> None:
        append_atom(
            atom_type,
            raw_text,
            value,
            confidence,
            review_status=review_status,
            review_flags=review_flags,
            extra_columns=cols,
        )

    if re.search(r"\b(after[-\s]?hours|nights only|weekends only)\b", row_blob):
        ap(
            AtomType.constraint,
            "After-hours access constraint",
            {"constraint_type": "after_hours", "site": site, "device": device, "floor": floor, "room": room, "location": location},
            row_confidence * 0.9,
        )
    if re.search(r"\b(lift required|elevator|customer provides lift|customer provide lift)\b", row_blob):
        if "customer" in row_blob and "lift" in row_blob:
            ap(
                AtomType.action_item,
                "Customer lift responsibility",
                {"action": "customer_provides_lift", "site": site, "device": device, "location": location},
                row_confidence * 0.85,
            )
        ap(
            AtomType.constraint,
            "Lift access constraint",
            {"constraint_type": "lift", "site": site, "device": device, "location": location},
            row_confidence * 0.88,
        )
    if re.search(r"\b(badge|escort|ceiling access)\b", row_blob):
        ap(
            AtomType.constraint,
            "Site access constraint",
            {"constraint_type": "access", "detail": row_blob[:200], "site": site, "location": location},
            row_confidence * 0.85,
        )
    if re.search(r"\b(confirm|unknown|tbd)\b.*\b(badge|mdf|access)\b|\b(badge|mdf|access)\b.*\b(confirm|unknown|tbd)\b", row_blob):
        ap(
            AtomType.open_question,
            "Access confirmation open question",
            {"topic": "badge_or_access", "site": site, "location": location},
            row_confidence * 0.8,
            review_status=ReviewStatus.needs_review,
        )
    if re.search(r"\b(certification required|certify|test standard)\b", row_blob):
        ap(
            AtomType.constraint,
            "Certification requirement",
            {"constraint_type": "certification", "site": site, "location": location},
            row_confidence * 0.88,
        )
    if re.search(r"\blabel(ing)?\b.*\btbd\b|\btbd\b.*\blabel", row_blob):
        ap(
            AtomType.open_question,
            "Labeling standard TBD",
            {"topic": "labeling", "site": site, "location": location},
            row_confidence * 0.78,
            review_status=ReviewStatus.needs_review,
        )
    if re.search(r"\b(excluded|removed|deleted|not included|out of scope)\b", row_blob):
        ap(
            AtomType.exclusion,
            "Scope exclusion from schedule",
            {"exclusion_hint": row_blob[:240], "site": site, "location": location},
            row_confidence * 0.82,
        )


class XlsxParser(BaseParser):
    parser_name = parser_name
    parser_version = parser_version
    capability = ParserCapability(
        parser_name=parser_name,
        parser_version=parser_version,
        supported_extensions=[".xlsx", ".csv"],
        supported_artifact_types=[ArtifactType.xlsx, ArtifactType.csv],
        emitted_atom_types=[
            AtomType.entity,
            AtomType.quantity,
            AtomType.scope_item,
            AtomType.constraint,
            AtomType.exclusion,
            AtomType.open_question,
            AtomType.action_item,
        ],
        supported_domain_packs=["*"],
        requires_binary=False,
        supports_source_replay=True,
    )

    def match(self, path: Path, sample_text: str | None, domain_pack: DomainPack | None) -> ParserMatch:
        del sample_text, domain_pack
        suffix = path.suffix.lower()
        confidence = 0.0
        reasons: list[str] = []
        if suffix in {".xlsx", ".csv"}:
            from app.parsers.spreadsheet_route_signals import (
                path_roster_schedule_hint,
                sniff_operations_workbook_strength,
                sniff_xlsx_roster_schedule_strength,
            )

            confidence = 0.58
            reasons.append(f"spreadsheet_extension:{suffix}")

            # PR1 (post-v3 review) — multi-sheet operations workbook
            # detection. Wins decisively over QuoteParser when the
            # workbook has asset / site / port / circuit / risk /
            # cutover sheets (even if it ALSO has a BOM sheet).
            if suffix == ".xlsx":
                ops_score, ops_reasons = sniff_operations_workbook_strength(path)
                reasons.extend(ops_reasons)
                if ops_score >= 0.55:
                    return ParserMatch(
                        parser_name=self.parser_name,
                        confidence=0.97,
                        reasons=reasons + ["xlsx_match:operations_workbook"],
                        artifact_type=ArtifactType.xlsx,
                    )

            if path_roster_schedule_hint(path):
                confidence += 0.14
                reasons.append("xlsx_match:path_roster_schedule_token")
            try:
                xscore, _xr = sniff_xlsx_roster_schedule_strength(path)
                confidence = min(0.92, confidence + 0.22 * xscore)
                reasons.append(f"xlsx_match:schedule_strength={xscore:.2f}")
            except Exception:  # noqa: BLE001
                reasons.append("xlsx_match:schedule_sniff_failed")
        return ParserMatch(
            parser_name=self.parser_name,
            confidence=confidence,
            reasons=reasons,
            artifact_type=ArtifactType.xlsx if suffix == ".xlsx" else ArtifactType.csv,
        )

    def parse(self, artifact_path: Path) -> list[EvidenceAtom]:
        artifact_id = stable_id("art", str(artifact_path))
        return self.parse_artifact(
            project_id="unknown_project",
            artifact_id=artifact_id,
            path=artifact_path,
        )

    def segment_artifact(self, project_id: str, artifact_id: str, path: Path) -> list[ArtifactSegment]:
        return segment_xlsx(project_id=project_id, artifact_id=artifact_id, path=path, parser_version=self.parser_version)

    def parse_artifact(
        self,
        project_id: str,
        artifact_id: str,
        path: Path,
        domain_pack: DomainPack | None = None,
    ) -> list[EvidenceAtom]:
        """Back-compat wrapper that returns a flat list of atoms.

        Prefer :meth:`parse_artifact_full` — that surfaces the
        ``structured.json`` / ``structured.md`` derived files to the
        compiler so the OrbitBrief envelope can render this workbook
        with the same fidelity it gets for PDFs.
        """
        return self.parse_artifact_full(
            project_id=project_id,
            artifact_id=artifact_id,
            path=path,
            domain_pack=domain_pack,
        ).atoms

    def parse_artifact_full(
        self,
        project_id: str,
        artifact_id: str,
        path: Path,
        domain_pack: DomainPack | None = None,
    ) -> ParserOutput:
        del domain_pack
        suffix = path.suffix.lower()
        if suffix == ".csv":
            atoms, sheets, parse_error = self._parse_csv(
                project_id=project_id, artifact_id=artifact_id, path=path
            )
            schema = STRUCTURED_SCHEMA_CSV
            artifact_type = ArtifactType.csv
        else:
            atoms, sheets, parse_error = self._parse_xlsx(
                project_id=project_id, artifact_id=artifact_id, path=path
            )
            schema = STRUCTURED_SCHEMA_XLSX
            artifact_type = ArtifactType.xlsx
            # COVERAGE BACKSTOP (parse-coverage, no silent whole-file loss):
            # a VALID workbook whose every sheet routed to DROP/empty — e.g. an
            # unfilled RFP price/SLA response template (price column blank, so the
            # sheet_classifier drops it) — would otherwise vanish entirely, losing
            # the whole priced-scope catalog. If the read SUCCEEDED but produced
            # ZERO atoms, fall back to generic per-row extraction over any
            # data-bearing sheet so a populated file can never be silently dropped.
            # Purely additive: fires only when output would be empty, so it cannot
            # regress any deal that already extracts. Instructional/cover sheets
            # are still skipped by _emit_generic_rows' own guard.
            if not atoms and not parse_error:
                backstop: list[EvidenceAtom] = []
                for sh in sheets:
                    rws = sh.get("rows") or []
                    nonempty = [r for r in rws if any(str(c or "").strip() for c in r)]
                    if len(nonempty) >= 3:
                        backstop.extend(self._emit_generic_rows(
                            project_id=project_id, artifact_id=artifact_id,
                            artifact_type=ArtifactType.xlsx, filename=path.name,
                            sheet_name=sh.get("name") or "", rows=rws,
                        ))
                if backstop:
                    atoms = backstop
                    self._coverage_backstop_note = (
                        f"INFO: coverage backstop recovered {len(backstop)} row(s) "
                        f"from {path.name} (every sheet had routed to drop/empty — "
                        f"e.g. an unfilled RFP price/SLA template)"
                    )

        structured_doc = self._build_structured_doc(
            schema=schema,
            artifact_type=artifact_type,
            filename=path.name,
            sheets=sheets,
            parse_error=parse_error,
        )
        stamp_section_and_block_ids(structured_doc, artifact_seed=artifact_id)
        warnings: list[str] = []
        if parse_error:
            warnings.append(f"{ARTIFACT_PARSE_ERROR_PREFIX}{artifact_id}:{parse_error}")
        _bs_note = getattr(self, "_coverage_backstop_note", "")
        if _bs_note:
            warnings.append(_bs_note)
            self._coverage_backstop_note = ""
        # Mark embedded charts / images / drawings / OLE objects in .xlsx so an
        # embedded diagram or logo-as-data can't silently vanish. (CSV has no
        # zip container, so this is a no-op there.)
        if suffix != ".csv":
            atoms = list(atoms) + emit_zip_binary_markers(
                path=path,
                project_id=project_id,
                artifact_id=artifact_id,
                filename=path.name,
                artifact_type=ArtifactType.xlsx,
                parser_version=self.parser_version,
            )
        return ParserOutput(
            atoms=atoms,
            derived_files=derived_files_for(artifact_path=path, structured_doc=structured_doc),
            warnings=warnings,
        )

    @staticmethod
    def _tabular_read_error_code(exc: BaseException, *, tabular: Literal["xlsx", "csv"]) -> str:
        name = type(exc).__name__
        msg = str(exc).lower()
        if tabular == "xlsx" and (name == "BadZipFile" or "zip" in msg or "central directory" in msg):
            return "corrupt_xlsx"
        if tabular == "csv":
            return "corrupt_csv"
        return f"{tabular}_read_error"

    def _parse_xlsx(
        self, project_id: str, artifact_id: str, path: Path
    ) -> tuple[list[EvidenceAtom], list[dict[str, Any]], str | None]:
        try:
            workbook = load_workbook(path, read_only=True, data_only=True)
        except Exception as exc:
            code = self._tabular_read_error_code(exc, tabular="xlsx")
            return [], [], f"{code}:{type(exc).__name__}:{exc}"
        atoms: list[EvidenceAtom] = []
        sheets: list[dict[str, Any]] = []
        for sheet in workbook.worksheets:
            rows = [list(row) for row in sheet.iter_rows(values_only=True)]
            atoms.extend(
                self._parse_sheet_rows(
                    project_id=project_id,
                    artifact_id=artifact_id,
                    filename=path.name,
                    artifact_type=ArtifactType.xlsx,
                    sheet_name=sheet.title,
                    rows=rows,
                )
            )
            sheets.append({"name": sheet.title, "rows": rows})
        return atoms, sheets, None

    def _parse_csv(
        self, project_id: str, artifact_id: str, path: Path
    ) -> tuple[list[EvidenceAtom], list[dict[str, Any]], str | None]:
        try:
            with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
                reader = csv.reader(handle)
                rows = [list(row) for row in reader]
        except Exception as exc:
            code = self._tabular_read_error_code(exc, tabular="csv")
            return [], [], f"{code}:{type(exc).__name__}:{exc}"
        sheet_atoms = self._parse_sheet_rows(
            project_id=project_id,
            artifact_id=artifact_id,
            filename=path.name,
            artifact_type=ArtifactType.csv,
            sheet_name="csv",
            rows=rows,
        )
        return sheet_atoms, [{"name": "csv", "rows": rows}], None

    def _build_structured_doc(
        self,
        *,
        schema: str,
        artifact_type: ArtifactType,
        filename: str,
        sheets: list[dict[str, Any]],
        parse_error: str | None = None,
    ) -> dict[str, Any]:
        """Render every sheet as a page with one section that contains the
        full table.  Empty / structureless sheets become a tiny note so
        downstream consumers can still see they exist.
        """
        pages: list[dict[str, Any]] = []
        for index, sheet in enumerate(sheets):
            sheet_name = sheet["name"]
            rows = sheet["rows"]
            section_blocks: list[dict[str, Any]] = []
            model = _detect_header(rows) if rows else None
            if model and model.header_idx >= 0:
                header_row_idx = model.header_idx
                header_row = rows[header_row_idx] or []
                if model.header_mode == "pair" and header_row_idx + 1 < len(rows):
                    bot = rows[header_row_idx + 1] or []
                    columns = [
                        " ".join(part for part in (str(a or "").strip(), str(b or "").strip()) if part).strip()
                        or f"col_{i + 1}"
                        for i, (a, b) in enumerate(
                            zip(header_row, bot + [None] * (len(header_row) - len(bot)))
                        )
                    ]
                    data_start = header_row_idx + 2
                else:
                    columns = [
                        (str(c).strip() if c is not None else "") or f"col_{i + 1}"
                        for i, c in enumerate(header_row)
                    ]
                    data_start = header_row_idx + 1
                table_rows: list[dict[str, Any]] = []
                for row_idx in range(data_start, len(rows)):
                    row = rows[row_idx] or []
                    if all(str(c or "").strip() == "" for c in row):
                        continue
                    cells: dict[str, Any] = {}
                    for col_idx, col_name in enumerate(columns):
                        if col_idx < len(row):
                            value = row[col_idx]
                        else:
                            value = ""
                        cells[col_name] = "" if value is None else str(value).strip()
                    table_rows.append(cells)
                if table_rows:
                    section_blocks.append(make_table(columns=columns, rows=table_rows))
            if not section_blocks:
                # Sheet has no detectable header; fall back to a raw textual
                # dump so the LLM still has something to look at.
                snippet_lines: list[str] = []
                for row in rows[:25]:
                    line = " | ".join(str(c).strip() for c in (row or []) if c is not None and str(c).strip())
                    if line:
                        snippet_lines.append(line)
                if snippet_lines:
                    section_blocks.append(
                        {"kind": "paragraph", "text": "\n".join(snippet_lines)}
                    )
            section = make_section(heading=sheet_name, level=2, blocks=section_blocks)
            pages.append(
                make_page(
                    page=index,
                    title=sheet_name,
                    sections=[section],
                )
            )
        metadata = [f"sheet: {s['name']}" for s in sheets]
        if parse_error:
            metadata.append(f"parse_error: {parse_error}")
        return make_structured_document(
            schema_version=schema,
            filename=filename,
            artifact_type=artifact_type.value,
            title=filename,
            metadata=metadata,
            pages=pages,
        )

    def _build_source_ref(
        self,
        artifact_id: str,
        artifact_type: ArtifactType,
        filename: str,
        sheet_name: str,
        row_number: int,
        columns: dict[str, str],
    ) -> SourceRef:
        return SourceRef(
            id=stable_id("src", artifact_id, sheet_name, row_number, stable_id("col", *sorted(columns.values()))),
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            filename=filename,
            locator={"sheet": sheet_name, "row": row_number, "columns": columns},
            extraction_method="xlsx_table_mapping_v2_0",
            parser_version=self.parser_version,
        )

    def _maybe_emit_site_roster_atoms(
        self,
        *,
        project_id: str,
        artifact_id: str,
        artifact_type: ArtifactType,
        filename: str,
        sheet_name: str,
        rows: list[list[Any]],
    ) -> list[EvidenceAtom]:
        """Emit ``physical_site`` entity atoms when the sheet shape
        matches a site roster (header row with Site ID / Facility /
        Address / MDF / Escort + structured site-shaped IDs in data
        rows). Returns [] when this sheet is not a roster.
        """
        try:
            from app.parsers.site_roster_extractor import (
                extract_site_roster,
                looks_like_site_roster,
            )
        except Exception:  # pragma: no cover
            return []
        if not rows:
            return []
        # Skip blank leading rows
        first_data_row = 0
        for i, r in enumerate(rows):
            if any(str(c or "").strip() for c in (r or ())):
                first_data_row = i
                break
        body = rows[first_data_row:]
        if len(body) < 2:
            return []
        header_raw = [str(c or "").strip() for c in (body[0] or ())]
        data_rows: list[dict[str, Any]] = []
        for r in body[1:]:
            cells: dict[str, Any] = {}
            for i, v in enumerate(r or ()):
                col = header_raw[i] if i < len(header_raw) and header_raw[i] else f"col_{i}"
                cells[col] = "" if v is None else str(v)
            if any(v for v in cells.values()):
                data_rows.append(cells)
        if not data_rows:
            return []
        try:
            if not looks_like_site_roster(
                columns=header_raw, rows=data_rows, surrounding_text=sheet_name or ""
            ):
                return []
            # Stricter gate for XLSX (where site_id columns appear in
            # BOMs, decisions sheets, port maps, etc.): require at least
            # one additional roster-specific column beyond site_id.
            # Avoids treating a "Site ID | Decision | Approved By"
            # sheet as a roster.
            from app.parsers.site_roster_extractor import map_columns_to_fields
            field_map = map_columns_to_fields(header_raw)
            roster_specific = {
                "facility_name", "street_address", "mdf_idf",
                "access_window", "escort_owner", "city_state",
            }
            if not (set(field_map.values()) & roster_specific):
                return []
            roster_rows = extract_site_roster(
                columns=header_raw, rows=data_rows, surrounding_text=sheet_name or ""
            )
        except Exception:  # pragma: no cover
            return []
        if not roster_rows:
            return []
        out: list[EvidenceAtom] = []
        for site_row in roster_rows:
            sid = (site_row.site_id or "").strip()
            canon_id = sid or site_row.facility_name or ""
            if not canon_id:
                continue
            row_index = first_data_row + 1 + site_row.row_index
            atom_id = stable_id(
                "atm", artifact_id, sheet_name, row_index, "physical_site", canon_id
            )
            text_parts = []
            for label, val in [
                ("site_id", sid or site_row.site_id),
                ("facility", site_row.facility_name),
                ("address", site_row.street_address),
                ("mdf_idf", site_row.mdf_idf),
                ("access", site_row.access_window),
                ("escort", site_row.escort_owner),
                ("contact", site_row.contact),
                ("phone", site_row.phone),
                ("email", site_row.email),
            ]:
                if val:
                    text_parts.append(f"{label}: {val}")
            row_text = " | ".join(text_parts) or canon_id
            source_ref = SourceRef(
                id=stable_id("src", atom_id),
                artifact_id=artifact_id,
                artifact_type=artifact_type,
                filename=filename,
                locator={
                    "sheet": sheet_name,
                    "row": row_index,
                    "extraction": "xlsx_site_roster_v1",
                },
                extraction_method="xlsx_site_roster_v1",
                parser_version=self.parser_version,
            )
            entity_keys: list[str] = []
            if sid:
                slug = re.sub(r"[^a-z0-9]+", "_", sid.lower()).strip("_")
                if slug:
                    entity_keys.append(f"site:{slug}")
            out.append(
                EvidenceAtom(
                    id=atom_id,
                    project_id=project_id,
                    artifact_id=artifact_id,
                    # v53.2 ROOT-CAUSE FIX: physical_site (not entity) so
                    # downstream gates can build the canonical site catalog.
                    atom_type=AtomType.physical_site,
                    raw_text=row_text,
                    normalized_text=row_text.lower(),
                    value={
                        "kind": "physical_site",
                        "id": sid or site_row.site_id,  # canonical id
                        "site_id": sid or site_row.site_id,
                        "name": site_row.facility_name,
                        "facility_name": site_row.facility_name,
                        "address": site_row.street_address,
                        "street_address": site_row.street_address,
                        "mdf_idf": site_row.mdf_idf,
                        "access_window": site_row.access_window,
                        "escort_owner": site_row.escort_owner,
                        "contact": site_row.contact,
                        "phone": site_row.phone,
                        "email": site_row.email,
                        "city_state": site_row.city_state,
                        "zip": site_row.zip,
                        "sqft": site_row.sqft,
                        "occupancy": site_row.occupancy,
                        "notes": site_row.notes,
                        "extras": dict(site_row.extra_fields),
                        # Bind the FULL raw row so EVERY column surfaces (Users,
                        # Rooms, Hardware/Services/Logistics budgets, Notes, ...),
                        # not just the canonical roster fields. _atom_bound_text
                        # renders these "Header: value | ..." at decide-time.
                        "cells": (
                            dict(data_rows[site_row.row_index])
                            if 0 <= site_row.row_index < len(data_rows)
                            else {}
                        ),
                    },
                    entity_keys=sorted(set(entity_keys)),
                    source_refs=[source_ref],
                    receipts=[],
                    authority_class=AuthorityClass.contractual_scope,
                    confidence=site_row.confidence,
                    confidence_raw=site_row.confidence,
                    calibrated_confidence=site_row.confidence,
                    review_status=ReviewStatus.auto_accepted,
                    review_flags=[],
                    parser_version=self.parser_version,
                )
            )
        return out

    # Sheets the classifier routes to COMMERCIAL (rate cards, master
    # catalogs, deal-financials) can be large pricebooks; cap the number
    # of pricing atoms per sheet so a 1000-row catalog can't flood the
    # envelope. Deal totals and rate cards are far smaller than this.
    _COMMERCIAL_ROW_CAP = 200
    # Bulk pricing sheets (rate cards / catalogs) fold every row into the
    # single rollup atom's value.rows rather than emitting per-row atoms.
    # This cap bounds envelope size for pathological sheets while staying
    # far above any realistic rate table.
    _COMMERCIAL_FOLD_CAP = 5000

    def _emit_financial_summary_rows(
        self,
        project_id: str,
        artifact_id: str,
        artifact_type: ArtifactType,
        filename: str,
        sheet_name: str,
        rows: list[list[Any]],
    ) -> list[EvidenceAtom]:
        """Structured extraction for a deal-financials / P&L summary sheet.

        Reads the 2-D label→value grid the way a PM does and emits:

        * one ``deal_metadata`` atom carrying the deal header (OPPTY #,
          customer, sales rep, billing type, duration, region, …);
        * one ``commercial_total`` atom per P&L category (Deal / Labor /
          PMO / Materials / Lift-Rental / Misc) with a structured
          ``value`` (revenue / cost / margin / margin_pct) so the
          OrbitBrief financial section renders without re-parsing text.

        This replaces the row-glue the generic commercial emitter produced
        for these sheets (``OPPTY # | 126 | Total Deal Revenue | 21560``).
        Deterministic, LLM-free, universal across estimating workbooks.
        """
        header: dict[str, Any] = {}
        header_locators: dict[str, dict[str, int]] = {}
        # category key -> {"display", "revenue", "cost", "margin",
        #                  "margin_pct", "row"}
        pl: dict[str, dict[str, Any]] = {}

        for ri, row in enumerate(rows):
            for ci, cell in enumerate(row):
                if cell is None:
                    continue
                # Original-case, whitespace-collapsed text drives P&L
                # category *display*; the lowercased form drives matching.
                orig = re.sub(r"\s+", " ", str(cell).strip())
                label = orig.lower()
                if not label:
                    continue

                # ── deal header field (canonical key) ──
                key = _DEAL_HEADER_LABELS.get(label)
                if key and key not in header:
                    sval = _coerce_header_value(_right_neighbor_value(row, ci))
                    if sval is not None:
                        header[key] = sval
                        header_locators[key] = {"row": ri + 1, "col": ci + 1}
                    continue

                # ── P&L margin-percent line ──
                mp = _PL_MARGIN_PCT_RE.match(orig)
                if mp:
                    ckey, disp = _norm_pl_category(mp.group("cat"))
                    n = _coerce_pl_number(_right_neighbor_value(row, ci))
                    slot = pl.setdefault(ckey, {"display": disp, "row": ri + 1})
                    if n is not None and slot.get("margin_pct") is None:
                        # Store as a percentage; deal-kit fractions (0.2857)
                        # become 28.57, explicit percents pass through.
                        slot["margin_pct"] = round(n * 100, 2) if abs(n) <= 1.5 else round(n, 2)
                    continue

                # ── P&L revenue / cost / margin line ──
                mm = _PL_METRIC_RE.match(orig)
                if mm and mm.group("metric").lower() in _PL_METRIC_WORDS:
                    ckey, disp = _norm_pl_category(mm.group("cat"))
                    metric = mm.group("metric").lower()
                    n = _coerce_pl_number(_right_neighbor_value(row, ci))
                    slot = pl.setdefault(ckey, {"display": disp, "row": ri + 1})
                    if n is not None and slot.get(metric) is None:
                        slot[metric] = round(n, 2)

        # Confidence gate: only trust the structured P&L view when the grid
        # actually reads like a deal-kit financial summary — at least two
        # P&L categories carrying numbers, OR a multi-field deal header.
        # Otherwise (e.g. a country-multiplier matrix mislabelled as a
        # financial summary) fall back to the generic commercial emitter so
        # the sheet's real money rows aren't lost.
        pl_with_numbers = sum(
            1 for s in pl.values()
            if any(s.get(m) is not None for m in ("revenue", "cost", "margin", "margin_pct"))
        )
        if pl_with_numbers < 2 and len(header) < 3:
            return self._emit_commercial_sheet_rows(
                project_id=project_id,
                artifact_id=artifact_id,
                artifact_type=artifact_type,
                filename=filename,
                sheet_name=sheet_name,
                rows=rows,
                classification=classify_sheet(sheet_name, rows),
            )

        # ── generic header enrichment ──
        # The grid is now confirmed to be a deal-kit financial summary, so
        # capture ANY remaining label→value pair structurally (not just the
        # canonical vocabulary). This keeps non-standard header fields
        # (PO #, Account Manager, Site Contact, …) from being dropped. It
        # runs only past the confidence gate, so non-P&L grids — which fall
        # back above — are never affected. P&L metric/margin cells and
        # already-captured canonical fields are skipped.
        _GENERIC_HEADER_CAP = 40
        for ri, row in enumerate(rows):
            for ci, cell in enumerate(row):
                if cell is None or len(header) >= _GENERIC_HEADER_CAP:
                    continue
                orig = re.sub(r"\s+", " ", str(cell).strip())
                label = orig.lower()
                if not label or label in _DEAL_HEADER_LABELS:
                    continue
                if not _looks_like_header_label(orig):
                    continue
                sval = _coerce_header_value(_right_neighbor_value(row, ci))
                if sval is None:
                    continue
                # A genuine header value is atomic — a name, code, number, or
                # date. A multi-word, non-numeric phrase ("Gross Margin Deal
                # Kit", "Overall Deal Kit Summary") is a section title that
                # happens to sit beside a caption, not a field value; skip it
                # so the sweep doesn't mint bogus heading fields.
                if len(sval.split()) >= 4 and not any(ch.isdigit() for ch in sval):
                    continue
                gkey = re.sub(r"[^a-z0-9]+", "_", label).strip("_")
                if not gkey or gkey in header:
                    continue
                header[gkey] = sval
                header_locators[gkey] = {"row": ri + 1, "col": ci + 1}

        atoms: list[EvidenceAtom] = []

        def _src(tag: str, locator: dict[str, Any]) -> SourceRef:
            loc = {"sheet": sheet_name, "extraction": "financial_summary", **locator}
            return SourceRef(
                id=stable_id("src", artifact_id, sheet_name, tag),
                artifact_id=artifact_id,
                artifact_type=artifact_type,
                filename=filename,
                locator=loc,
                extraction_method="financial_summary",
                parser_version=self.parser_version,
            )

        # ── deal header atom ──
        if header:
            parts = [f"{k.replace('_', ' ').title()}: {v}" for k, v in header.items()]
            text = " | ".join(parts)[:4000]
            atom_id = stable_id("atm", artifact_id, "deal_metadata", sheet_name)
            ent_keys: list[str] = []
            if header.get("opportunity_id"):
                ent_keys.append(f"deal:{header['opportunity_id']}")
            if header.get("customer"):
                cust_slug = re.sub(r"[^a-z0-9]+", "_", str(header["customer"]).lower()).strip("_")
                if cust_slug:
                    ent_keys.append(f"customer:{cust_slug}")
            atoms.append(
                EvidenceAtom(
                    id=atom_id,
                    project_id=project_id,
                    artifact_id=artifact_id,
                    atom_type=AtomType.deal_metadata,
                    raw_text=text,
                    normalized_text=text.lower(),
                    value={"kind": "deal_header", "fields": header,
                           "field_locators": header_locators, "sheet_name": sheet_name},
                    entity_keys=ent_keys,
                    source_refs=[_src("deal_metadata", {})],
                    receipts=[],
                    authority_class=AuthorityClass.vendor_quote,
                    confidence=0.8,
                    confidence_raw=0.8,
                    calibrated_confidence=0.8,
                    review_status=ReviewStatus.needs_review,
                    review_flags=[],
                    parser_version=self.parser_version,
                )
            )

        # ── P&L category atoms ──
        # Order: grand-total Deal first, then by source row.
        def _cat_sort(item: tuple[str, dict[str, Any]]):
            k, v = item
            return (0 if k == "deal" else 1, v.get("row", 1_000_000))

        for ckey, slot in sorted(pl.items(), key=_cat_sort):
            rev = slot.get("revenue")
            cost = slot.get("cost")
            margin = slot.get("margin")
            mpct = slot.get("margin_pct")
            # Skip a category that carries no numeric content at all.
            if all(x is None for x in (rev, cost, margin, mpct)):
                continue
            disp = slot["display"]
            money_keys = sorted(
                {f"money:{int(round(v))}" for v in (rev, cost, margin)
                 if isinstance(v, (int, float)) and abs(v) >= _MIN_MONEY_VALUE}
            )

            def _fmt(v: Any) -> str:
                return f"${int(round(v)):,}" if isinstance(v, (int, float)) else "n/a"

            text = (
                f"{disp}: revenue {_fmt(rev)}, cost {_fmt(cost)}, "
                f"margin {_fmt(margin)}"
                + (f" ({mpct:g}%)" if isinstance(mpct, (int, float)) else "")
            )
            atom_id = stable_id("atm", artifact_id, "pl", sheet_name, ckey)
            atoms.append(
                EvidenceAtom(
                    id=atom_id,
                    project_id=project_id,
                    artifact_id=artifact_id,
                    atom_type=AtomType.commercial_total,
                    raw_text=text,
                    normalized_text=text.lower(),
                    value={
                        "kind": "pl_line",
                        "category": disp,
                        "category_key": ckey,
                        "revenue": rev,
                        "cost": cost,
                        "margin": margin,
                        "margin_pct": mpct,
                        "sheet_name": sheet_name,
                    },
                    entity_keys=money_keys,
                    source_refs=[_src(f"pl_{ckey}", {"row": slot.get("row", 0)})],
                    receipts=[],
                    authority_class=AuthorityClass.vendor_quote,
                    confidence=0.78,
                    confidence_raw=0.78,
                    calibrated_confidence=0.78,
                    review_status=ReviewStatus.needs_review,
                    review_flags=[],
                    parser_version=self.parser_version,
                )
            )

        # If the structured pass found nothing usable (an oddly-shaped
        # sheet), fall back to the generic commercial emitter so we never
        # regress to zero atoms.
        if not atoms:
            return self._emit_commercial_sheet_rows(
                project_id=project_id,
                artifact_id=artifact_id,
                artifact_type=artifact_type,
                filename=filename,
                sheet_name=sheet_name,
                rows=rows,
                classification=classify_sheet(sheet_name, rows),
            )
        return atoms

    def _emit_commercial_sheet_rows(
        self,
        project_id: str,
        artifact_id: str,
        artifact_type: ArtifactType,
        filename: str,
        sheet_name: str,
        rows: list[list[Any]],
        classification: Any,
    ) -> list[EvidenceAtom]:
        """Route a non-scope *pricing* sheet to typed commercial atoms.

        Rate cards, master catalogs and deal-financials tabs are not
        customer scope, but they carry pricing the PM needs. Rather than
        drop them, every money-bearing row becomes a typed commercial
        atom (``commercial_total`` for a deal-financials summary,
        ``pricing_assumption`` for rate-card / catalog rows), tagged
        ``vendor_quote`` authority and carrying ``money:`` entity keys so
        it feeds the OrbitBrief ``pricing_clarity`` dimension without ever
        landing in ``scope_truth``.
        """
        from app.parsers.sheet_classifier import SheetRole

        role = classification.role
        atom_type = (
            AtomType.commercial_total
            if role is SheetRole.FINANCIAL_SUMMARY
            else AtomType.pricing_assumption
        )

        # FINANCIAL_SUMMARY sheets carry the deal economics (revenue, cost,
        # margin, per-line labor) — low-cardinality facts a PM reads line by
        # line, so they stay as individual atoms. RATE_CARD / CATALOG /
        # REFERENCE sheets are bulk backing matrices (a 312-row global rate
        # table, a master price book); exploding them into one atom per row
        # drowns the deliverable and costs a per-row LLM pass downstream while
        # producing zero packets. For those we fold every row losslessly into
        # the rollup atom's ``value.rows`` (full drill-down preserved) and
        # emit only the single summary atom.
        collapse_to_summary = role is not SheetRole.FINANCIAL_SUMMARY

        money_cols = _money_columns(rows)
        atoms: list[EvidenceAtom] = []
        all_values: list[float] = []
        folded_rows: list[dict[str, Any]] = []
        for row_idx, row in enumerate(rows):
            cells = [("" if c is None else str(c).strip()) for c in row]
            if not any(cells):
                continue
            values = _row_money_values(row, money_cols)
            if not values:
                # Header / label rows with no dollar figure carry no
                # pricing signal — skip so the commercial view stays clean.
                continue
            all_values.extend(values)
            money_keys = sorted({f"money:{int(round(v))}" for v in values})
            row_text = " | ".join(c for c in cells if c)[:4000]
            label = " ".join(
                c for c in cells if c and not c.replace(",", "").replace(".", "").lstrip("-").isdigit()
            ).strip()[:300]

            if collapse_to_summary:
                # Lossless drill-down payload — kept inside the one rollup
                # atom rather than as a standalone atom.
                if len(folded_rows) < self._COMMERCIAL_FOLD_CAP:
                    folded_rows.append(
                        {
                            "row": row_idx + 1,
                            "label": label,
                            "money_keys": money_keys,
                            "cells": [c for c in cells if c],
                        }
                    )
                continue

            atom_id = stable_id(
                "atm", artifact_id, atom_type.value, sheet_name, row_idx
            )
            src = SourceRef(
                id=stable_id("src", atom_id),
                artifact_id=artifact_id,
                artifact_type=artifact_type,
                filename=filename,
                locator={
                    "sheet": sheet_name,
                    "row": row_idx + 1,
                    "extraction": "commercial_sheet_routing",
                },
                extraction_method="commercial_sheet_routing",
                parser_version=self.parser_version,
            )
            atoms.append(
                EvidenceAtom(
                    id=atom_id,
                    project_id=project_id,
                    artifact_id=artifact_id,
                    atom_type=atom_type,
                    raw_text=row_text,
                    normalized_text=row_text.lower(),
                    value={
                        "label": label,
                        "money_keys": money_keys,
                        "sheet_role": role.value,
                        "sheet_name": sheet_name,
                        "cells": [c for c in cells if c],
                    },
                    entity_keys=money_keys,
                    source_refs=[src],
                    receipts=[],
                    authority_class=AuthorityClass.vendor_quote,
                    confidence=0.7,
                    confidence_raw=0.7,
                    calibrated_confidence=0.7,
                    review_status=ReviewStatus.needs_review,
                    review_flags=[],
                    parser_version=self.parser_version,
                )
            )
            if len(atoms) >= self._COMMERCIAL_ROW_CAP:
                break

        # ``line_count`` reflects every money-bearing row found, whether it
        # became its own atom (financial summary) or was folded (rate card).
        line_count = len(atoms) if not collapse_to_summary else len(folded_rows)
        if line_count == 0:
            return []

        # Roll-up banner: a single summary atom per pricing sheet so the
        # OrbitBrief pricing view can render one readable line (count +
        # $-range + total). For collapsed sheets it also carries the full
        # row matrix in ``value.rows`` for drill-down; for financial-summary
        # sheets the granular rows follow as their own atoms.
        summary = self._commercial_summary_atom(
            project_id=project_id,
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            filename=filename,
            sheet_name=sheet_name,
            role=role,
            atom_type=atom_type,
            line_count=line_count,
            values=all_values,
            folded_rows=folded_rows if collapse_to_summary else None,
        )
        return [summary, *atoms]

    def _commercial_summary_atom(
        self,
        project_id: str,
        artifact_id: str,
        artifact_type: ArtifactType,
        filename: str,
        sheet_name: str,
        role: Any,
        atom_type: AtomType,
        line_count: int,
        values: list[float],
        folded_rows: list[dict[str, Any]] | None = None,
    ) -> EvidenceAtom:
        lo = min(values) if values else 0.0
        hi = max(values) if values else 0.0
        total = sum(values)
        money_keys = sorted(
            {f"money:{int(round(v))}" for v in (lo, hi, total) if v >= _MIN_MONEY_VALUE}
        )
        # ASCII hyphen separator — the OrbitBrief renderer and several JSON
        # consumers round-tripped the en-dash through a non-UTF-8 stage and
        # surfaced it as U+FFFD ("$8�$5,390"). A plain "-" is unambiguous.
        label = (
            f"{sheet_name}: {line_count} pricing line"
            f"{'s' if line_count != 1 else ''}, "
            f"${int(round(lo)):,}-${int(round(hi)):,}"
        )
        atom_id = stable_id(
            "atm", artifact_id, atom_type.value, sheet_name, "summary"
        )
        src = SourceRef(
            id=stable_id("src", atom_id),
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            filename=filename,
            locator={
                "sheet": sheet_name,
                "extraction": "commercial_sheet_routing",
                "rollup": True,
            },
            extraction_method="commercial_sheet_routing",
            parser_version=self.parser_version,
        )
        return EvidenceAtom(
            id=atom_id,
            project_id=project_id,
            artifact_id=artifact_id,
            atom_type=atom_type,
            raw_text=label,
            normalized_text=label.lower(),
            value={
                "is_summary": True,
                "label": label,
                "line_count": line_count,
                "money_min": round(lo, 2),
                "money_max": round(hi, 2),
                "money_sum": round(total, 2),
                "money_keys": money_keys,
                "sheet_role": role.value,
                "sheet_name": sheet_name,
                # Full row matrix for collapsed bulk sheets (rate cards /
                # catalogs). None for financial-summary sheets whose rows
                # are emitted as their own atoms.
                "rows": folded_rows if folded_rows else None,
            },
            entity_keys=money_keys,
            source_refs=[src],
            receipts=[],
            authority_class=AuthorityClass.vendor_quote,
            confidence=0.7,
            confidence_raw=0.7,
            calibrated_confidence=0.7,
            review_status=ReviewStatus.needs_review,
            review_flags=["pricing_rollup"],
            parser_version=self.parser_version,
        )

    def _dropped_sheet_marker(
        self,
        *,
        project_id: str,
        artifact_id: str,
        artifact_type: ArtifactType,
        filename: str,
        sheet_name: str,
        rows: list[list[Any]],
        reason: str,
    ) -> EvidenceAtom:
        """A retained marker for a whole sheet the role-router classified DROP.

        Carries the sheet's rows (capped) so a PM omission complaint can still
        recover the content, and is pre-stamped ``suppressed:sheet_router`` so
        the compiler diverts it into the suppressed sidecar — it never reaches
        scope_truth. This is the parse-time analog of the suppression ledger.
        """
        # Cap retained rows — a DROP sheet is noise; we keep enough to localize
        # an omission complaint without bloating the envelope.
        capped = [[("" if c is None else str(c)) for c in r] for r in rows[:500]]
        atom_id = stable_id("atm", artifact_id, "dropped_sheet", sheet_name)
        src = SourceRef(
            id=stable_id("src", atom_id),
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            filename=filename,
            locator={"sheet": sheet_name, "extraction": "sheet_role_router"},
            extraction_method="sheet_role_router",
            parser_version=self.parser_version,
        )
        return EvidenceAtom(
            id=atom_id,
            project_id=project_id,
            artifact_id=artifact_id,
            atom_type=AtomType.dropped_sheet,
            raw_text=f"[dropped sheet: {sheet_name}] {len(rows)} rows",
            normalized_text=f"dropped sheet {sheet_name}".lower(),
            value={
                "sheet_name": sheet_name,
                "row_count": len(rows),
                "rows": capped,
                "_suppression": {"stage": "sheet_router", "reason": reason},
            },
            entity_keys=[],
            source_refs=[src],
            receipts=[],
            authority_class=AuthorityClass.machine_extractor,
            confidence=0.0,
            confidence_raw=0.0,
            calibrated_confidence=0.0,
            review_status=ReviewStatus.needs_review,
            review_flags=["suppressed:sheet_router"],
            parser_version=self.parser_version,
        )

    def _parse_sheet_rows(
        self,
        project_id: str,
        artifact_id: str,
        filename: str,
        artifact_type: ArtifactType,
        sheet_name: str,
        rows: list[list[Any]],
    ) -> list[EvidenceAtom]:
        if not rows:
            return []

        # Sheet-role router: estimating workbooks carry rate-card backing
        # lists, master price catalogs, deal-financials tabs and empty
        # template sheets that are not customer scope. The classifier
        # routes each sheet so non-scope sheets can't bury real evidence:
        #   • DROP       → empty / cover / lookup-helper noise: emit nothing.
        #   • COMMERCIAL → rate card / catalog / deal financials: emit TYPED
        #     commercial atoms (pricing the PM needs) — never scope_item, so
        #     scope_truth stays clean while pricing still surfaces.
        #   • SCOPE      → fall through to normal row mining.
        classification = classify_sheet(sheet_name, rows)
        if classification.destination is SheetDestination.DROP:
            # Retained-suppression: instead of vanishing, a DROP-classified
            # sheet is emitted as ONE marker atom carrying its rows, stamped
            # suppressed:sheet_router. The compiler diverts pre-suppressed
            # parser atoms into CompileResult.suppressed_atoms, so the sheet
            # never reaches scope but stays auditable + complaint-localizable.
            return [
                self._dropped_sheet_marker(
                    project_id=project_id,
                    artifact_id=artifact_id,
                    artifact_type=artifact_type,
                    filename=filename,
                    sheet_name=sheet_name,
                    rows=rows,
                    reason=getattr(classification, "reason", "") or "sheet_role=DROP",
                )
            ]
        if classification.destination is SheetDestination.COMMERCIAL:
            # Deal-financials / P&L sheets get a structured label→value
            # extractor (clean deal header + per-category P&L atoms);
            # rate cards / catalogs stay on the row/rollup path.
            if classification.role is SheetRole.FINANCIAL_SUMMARY:
                return self._emit_financial_summary_rows(
                    project_id=project_id,
                    artifact_id=artifact_id,
                    artifact_type=artifact_type,
                    filename=filename,
                    sheet_name=sheet_name,
                    rows=rows,
                )
            return self._emit_commercial_sheet_rows(
                project_id=project_id,
                artifact_id=artifact_id,
                artifact_type=artifact_type,
                filename=filename,
                sheet_name=sheet_name,
                rows=rows,
                classification=classification,
            )

        # RF1 — explicit fast-path for known structured-row CSVs.
        # Files named asset_inventory / site_list / risk_register /
        # license_support_matrix / lifecycle route to the typed-row
        # profiler so they produce asset_record / site_roster / risk /
        # support_entitlement / lifecycle_status atoms (PR2). This must
        # run BEFORE the site-roster fast path: an asset_inventory CSV
        # (Asset ID / Serial / IP Address / MAC Address) would otherwise
        # be mis-read as a site roster (AST-001 matches the site-ID shape
        # and "IP Address" matches the street-address header) and emit
        # ghost physical_site atoms. Restricted to .csv to avoid
        # disturbing existing .xlsx fixtures (e.g. demo_project's
        # site_list.xlsx that legacy tests expect on the canonical path).
        _STRUCTURED_NAMES = (
            "asset_inventory", "site_list", "risk_register",
            "license_support", "support_matrix", "lifecycle",
        )
        stem = filename.lower().replace("-", "_").replace(" ", "_")
        if (
            artifact_type == ArtifactType.csv
            and any(s in stem for s in _STRUCTURED_NAMES)
        ):
            return self._emit_generic_rows(
                project_id=project_id,
                artifact_id=artifact_id,
                artifact_type=artifact_type,
                filename=filename,
                sheet_name=sheet_name,
                rows=rows,
            )

        # Site-roster fast path: when this sheet's first non-empty row
        # looks like site_roster headers (Site ID / Facility / Address /
        # MDF / Access / Escort), route the rows through the same
        # extractor the PDF parser uses so XLSX-shipped rosters produce
        # structured ``physical_site`` atoms (not just generic row atoms).
        roster_atoms = self._maybe_emit_site_roster_atoms(
            project_id=project_id,
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            filename=filename,
            sheet_name=sheet_name,
            rows=rows,
        )
        if roster_atoms:
            return roster_atoms

        # PR2 (post-v3 review) — operational-workbook sheet profiles.
        # If the sheet name matches one of the well-known ops sheets
        # (Asset Inventory / Site Survey / Port Map & VLANs / Circuit
        # Inventory / License Support / NOC Alert Matrix / Risk
        # Register / Cutover Validation / README / Dashboard /
        # Source Refs), dispatch every row to its typed AtomType.
        op_profile = _OPERATIONAL_SHEET_PROFILES.get(_norm_op_sheet(sheet_name))
        if op_profile is not None:
            ops_atoms = self._emit_operational_sheet_rows(
                project_id=project_id,
                artifact_id=artifact_id,
                artifact_type=artifact_type,
                filename=filename,
                sheet_name=sheet_name,
                rows=rows,
                profile=op_profile,
            )
            if ops_atoms:
                return ops_atoms
            # Fall through to the legacy path if the ops emitter
            # produced nothing (e.g. blank sheet).

        model = _detect_header(rows)
        if model.header_idx < 0:
            # PRODUCTION_GAPS P0.4: when the canonical-header detector can't
            # find a cabling/networking-style schedule (plate_id, site,
            # quantity columns), fall back to a generic row-as-atom emitter.
            # Real-world XLSX attachments — Q&A logs, fee schedules, vendor
            # response matrices, RFP cost tables — almost never use the
            # canonical column names this parser was originally tuned for.
            # The generic emitter still produces structured atoms so the
            # downstream entity_extraction stage can pull
            # device/vendor/quantity/site keys out of the row text.
            return self._emit_generic_rows(
                project_id=project_id,
                artifact_id=artifact_id,
                artifact_type=artifact_type,
                filename=filename,
                sheet_name=sheet_name,
                rows=rows,
            )
        data_start = model.header_idx + (2 if model.header_mode == "pair" else 1)
        atoms: list[EvidenceAtom] = []
        label_indices = _label_column_indices(model.header_map)

        # v49.2: capture raw header row once for raw_table_row emission.
        # The centralized _enrich_table_atoms() in entity_extraction
        # will classify per row using the column schema registry.
        _raw_headers: list[str] = []
        if 0 <= model.header_idx < len(rows):
            _raw_headers = [str(c or "").strip() for c in rows[model.header_idx]]

        for row_idx in range(data_start, len(rows)):
            row = rows[row_idx]
            if _is_blank_row(row):
                continue
            rk = _row_kind(row, model.header_map, label_indices)
            if rk in {"blank", "header", "title", "section_header"}:
                continue
            if rk == "note":
                continue

            extracted = self._extract_row_values(row, model.header_map)

            # v49.2: emit raw_table_row for centralized classification.
            if _raw_headers:
                _row_cells = [str(c or "").strip() for c in row]
                _row_text = " | ".join(c for c in _row_cells if c)[:4000]
                _rtr_id = stable_id("atm", artifact_id, "raw_table_row", sheet_name, row_idx)
                _rtr_columns = {
                    h: get_column_letter(i + 1)
                    for i, h in enumerate(_raw_headers)
                    if h
                }
                _rtr_src = SourceRef(
                    id=stable_id("src", _rtr_id),
                    artifact_id=artifact_id,
                    artifact_type=artifact_type,
                    filename=filename,
                    locator={"sheet": sheet_name, "row": row_idx + 1, "columns": _rtr_columns, "extraction": "raw_table_row_v49_2", "section_path": [sheet_name] if sheet_name else []},
                    extraction_method="raw_table_row_v49_2",
                    parser_version=self.parser_version,
                )
                atoms.append(EvidenceAtom(
                    id=_rtr_id,
                    project_id=project_id,
                    artifact_id=artifact_id,
                    atom_type=AtomType.raw_table_row,
                    raw_text=_row_text,
                    normalized_text=_row_text.lower(),
                    value={
                        "_columns": list(_raw_headers),
                        "_row": _row_cells,
                        "_table_idx": 0,
                        "_row_idx": row_idx,
                        "_filename": filename,
                        "_sheet": sheet_name,
                        "_artifact_type": "xlsx",
                    },
                    entity_keys=[],
                    source_refs=[_rtr_src],
                    receipts=[],
                    authority_class=AuthorityClass.contractual_scope,
                    confidence=0.80,
                    confidence_raw=0.80,
                    calibrated_confidence=0.80,
                    review_status=ReviewStatus.auto_accepted,
                    review_flags=[],
                    parser_version=self.parser_version,
                ))

            if rk in {"subtotal"}:
                atoms.extend(
                    self._emit_subtotal_row(
                        project_id=project_id,
                        artifact_id=artifact_id,
                        artifact_type=artifact_type,
                        filename=filename,
                        sheet_name=sheet_name,
                        row_number=row_idx + 1,
                        model=model,
                        extracted=extracted,
                        row=row,
                        label_indices=label_indices,
                    )
                )
                continue

            if rk in {"total", "grand_total"}:
                atoms.extend(
                    self._emit_total_row(
                        project_id=project_id,
                        artifact_id=artifact_id,
                        artifact_type=artifact_type,
                        filename=filename,
                        sheet_name=sheet_name,
                        row_number=row_idx + 1,
                        model=model,
                        row=row,
                        label_indices=label_indices,
                    )
                )
                continue

            if rk == "line_item":
                if _is_blank_row(list(extracted.values())) and not model.wide_qty_columns:
                    continue
                atoms.extend(
                    self._emit_line_item_row(
                        project_id=project_id,
                        artifact_id=artifact_id,
                        artifact_type=artifact_type,
                        filename=filename,
                        sheet_name=sheet_name,
                        row_number=row_idx + 1,
                        model=model,
                        extracted=extracted,
                        row=row,
                        label_indices=label_indices,
                    )
                )

        # Universal fallback: when the header-mapped path emitted
        # nothing (sheet had headers but they didn't match any
        # canonical extractor profile — Pricing sheets, summary
        # tables, etc.), fall back to the generic row emitter so the
        # rows aren't silently dropped.
        if not atoms:
            return self._emit_generic_rows(
                project_id=project_id,
                artifact_id=artifact_id,
                artifact_type=artifact_type,
                filename=filename,
                sheet_name=sheet_name,
                rows=rows,
            )
        return atoms

    @staticmethod
    def _canonicalize_header(value: str) -> str:
        """Map a raw column header into a stable canonical key.

        Used by :meth:`_generic_row_profile` to bucket a row into one
        of the structured AtomTypes (risk / asset_record / …) without
        requiring the upstream workbook to use a specific column name.
        """
        raw = normalize_text(str(value or "")).replace("-", " ").replace("_", " ")
        raw = re.sub(r"\s+", " ", raw).strip()

        aliases: dict[str, set[str]] = {
            "site_id": {"site id", "site code", "campus id", "location id"},
            "site_name": {"site", "site name", "campus", "location", "building"},
            "address": {
                "address",
                "street address",
                "service address",
                "site address",
            },
            "access_notes": {"access notes", "access window", "site access"},

            "risk_id": {"risk id", "raid id", "issue id"},
            "severity": {"severity", "priority", "risk rating"},
            "impact": {"impact", "business impact"},
            "likelihood": {"likelihood", "probability"},
            "mitigation": {"mitigation", "mitigation plan", "response plan"},
            "owner": {"owner", "assigned owner", "risk owner"},
            "status": {"status", "raid status"},

            "asset_id": {"asset id", "asset tag", "tag"},
            "serial": {"serial", "serial number", "s n", "sn"},
            "model": {"model", "model number", "part number"},
            "ip_address": {"ip", "ip address", "management ip", "mgmt ip"},
            "mac_address": {"mac", "mac address"},
            "lifecycle": {"lifecycle", "eol", "end of life", "refresh status"},

            "support_level": {"support level", "support tier", "coverage"},
            "contract_id": {
                "contract",
                "contract id",
                "support contract",
                "contract number",
            },
            "renewal_date": {
                "renewal",
                "renewal date",
                "expiration",
                "expiry",
                "expires",
            },
            "support_notes": {
                "support notes",
                "coverage notes",
                "entitlement notes",
            },

            "wan_provider": {
                "wan provider",
                "carrier",
                "isp",
                "circuit provider",
            },
            "circuit_id": {"circuit id", "circuit", "service id"},

            # PR2 (post-v3 review) — operational-workbook columns.
            "asset_type": {"asset type", "device type", "type"},
            "vendor": {"vendor", "manufacturer", "mfr", "make"},
            "manufacturer": {"manufacturer", "mfr", "make"},
            "hostname": {"hostname", "host name", "device name"},
            "mdf": {"mdf", "mdf id", "mdf area", "mdf/area"},
            "idf": {"idf", "idf id"},
            "in_service_date": {
                "in service",
                "in service date",
                "in-service date",
                "deployed",
                "deployment date",
            },
            "refresh_target": {
                "refresh target",
                "refresh date",
                "refresh planned",
            },
            "lifecycle_status": {
                "lifecycle status",
                "lifecycle state",
                "refresh status",
            },
            # NOC alert matrix
            "monitor_id": {"monitor id", "alert id"},
            "runbook_ref": {"runbook ref", "runbook", "playbook ref"},
            "alert_severity": {"alert severity", "alert level"},
            # Cutover validation
            "validation_id": {"validation id", "test id", "check id"},
            "customer_signoff": {"customer signoff", "customer sign off", "signoff"},
            "pass_flag": {"pass flag", "pass/fail", "result"},
            # Port / VLAN
            "switch_hostname": {"switch hostname", "switch name", "switch"},
            "port": {"port", "switch port", "interface"},
            "vlan_id": {"vlan id", "vlan"},
            "patch_panel_port": {"patch panel port", "panel port", "patch port"},
            # BOM detail
            "scope_bucket": {"scope bucket", "bucket", "scope phase"},
            "category": {"category", "line category"},
            "sku": {"sku", "part number", "part"},
            "unit_cost": {"unit cost", "unit price", "cost each"},
            "extended_cost": {"extended cost", "extended price", "ext cost"},
            "quote_status": {"quote status", "status quote", "po status"},
            "procurement_constraint": {
                "procurement constraint",
                "procurement note",
            },
            "distributor": {"distributor", "supplier", "wholesaler"},
            "quote_ref": {"quote ref", "quote number", "quote id"},
            "bom_line": {"bom line", "line number", "line"},
        }

        for canon, names in aliases.items():
            if raw in names:
                return canon
        return raw.replace(" ", "_")

    @staticmethod
    def _generic_row_profile(
        canon_cells: dict[str, str],
        sheet_name: str,
    ) -> tuple[AtomType, str, AuthorityClass, float]:
        """Inspect canonicalized cells + sheet name and pick the best
        structured AtomType / value.kind / authority class / confidence
        for this row. Defaults to ``scope_item`` / table_row / 0.84
        which preserves the legacy behavior."""
        keys = {k for k, v in canon_cells.items() if str(v).strip()}
        sheet = normalize_text(sheet_name)

        # Order matters — most specific first. Lifecycle is more
        # specific than the bare asset_record because it requires both
        # an asset identifier AND a lifecycle/status column.
        if {"risk_id", "severity"} & keys and (
            {"mitigation", "impact", "owner", "status", "likelihood"} & keys
        ):
            return (
                AtomType.risk,
                "risk_register_row",
                AuthorityClass.customer_current_authored,
                0.93,
            )
        if {"support_level", "renewal_date", "contract_id"} & keys:
            return (
                AtomType.support_entitlement,
                "support_entitlement_row",
                AuthorityClass.vendor_quote,
                0.92,
            )
        # PR2 (post-v3 review) — asset_record beats lifecycle_status
        # when the row has BOTH an asset identifier AND a vendor /
        # model / manufacturer / site signal. Pure lifecycle/EOL
        # records without a vendor/model still classify as
        # lifecycle_status.
        if (
            {"asset_id", "serial", "ip_address", "mac_address", "hostname"} & keys
            and (
                {"model", "manufacturer", "vendor", "asset_type", "site_name", "site"}
                & keys
            )
        ):
            return (
                AtomType.asset_record,
                "asset_inventory_row",
                AuthorityClass.approved_site_roster,
                0.94,
            )
        if {"lifecycle", "lifecycle_status", "refresh_target", "status"} & keys and (
            {"asset_id", "model", "serial"} & keys
        ):
            return (
                AtomType.lifecycle_status,
                "lifecycle_status_row",
                AuthorityClass.approved_site_roster,
                0.91,
            )
        if {"asset_id", "serial", "ip_address", "mac_address"} & keys:
            return (
                AtomType.asset_record,
                "asset_inventory_row",
                AuthorityClass.approved_site_roster,
                0.93,
            )
        if {"site_id", "address"} & keys or ("site" in sheet and "address" in keys):
            return (
                AtomType.site_roster,
                "site_roster_row",
                AuthorityClass.approved_site_roster,
                0.94,
            )
        return (
            AtomType.scope_item,
            "table_row",
            AuthorityClass.contractual_scope,
            0.84,
        )

    # ───── PR2 (post-v3 review) — operational sheet emitter ─────

    def _entity_keys_from_operational_row(
        self, cells: dict[str, str]
    ) -> list[str]:
        keys: list[str] = []
        for field in ("site_name", "site", "location"):
            if cells.get(field):
                key = normalize_entity_key("site", cells[field])
                if key:
                    keys.append(key)
        for field in ("mdf", "idf"):
            if cells.get(field):
                keys.append(normalize_entity_key(field, cells[field]))
        for field in ("asset_id", "hostname", "device_name"):
            if cells.get(field):
                keys.append(normalize_entity_key("device", cells[field]))
        for field in ("vendor", "manufacturer"):
            if cells.get(field):
                keys.append(normalize_entity_key("vendor", cells[field]))
        for field in ("part_number", "sku"):
            if cells.get(field):
                keys.append(normalize_entity_key("part_number", cells[field]))
        return sorted(set(keys))

    def _emit_operational_sheet_rows(
        self,
        *,
        project_id: str,
        artifact_id: str,
        artifact_type: ArtifactType,
        filename: str,
        sheet_name: str,
        rows: list[list[Any]],
        profile: _OperationalSheetProfile,
    ) -> list[EvidenceAtom]:
        header_idx = _first_nonblank_header_row(rows)
        if header_idx is None:
            return []

        headers = [
            (str(c).strip() if c is not None else "") or f"col_{i + 1}"
            for i, c in enumerate(rows[header_idx])
        ]
        canonical_headers = [self._canonicalize_header(h) for h in headers]

        atoms: list[EvidenceAtom] = []

        for row_idx in range(header_idx + 1, len(rows)):
            row = rows[row_idx] or []
            if _is_blank_row(row):
                continue

            cells: dict[str, str] = {}
            canonical_cells: dict[str, str] = {}
            cell_columns: dict[str, str] = {}

            for i, header in enumerate(headers):
                if i >= len(row):
                    continue
                value = row[i]
                text = _cell_to_text_op(value)
                if not text:
                    continue
                cells[header] = text
                canonical_cells[canonical_headers[i]] = text
                cell_columns[header] = get_column_letter(i + 1)

            if not cells:
                continue

            raw_text = " | ".join(f"{k}: {v}" for k, v in cells.items())

            value: dict[str, Any] = {
                "kind": profile.kind,
                "sheet": sheet_name,
                "row": row_idx + 1,
                "cells": cells,
                "canonical_cells": canonical_cells,
            }

            # PR2 — structured value payload per row kind. The brain
            # layer can read these without re-parsing.
            if profile.atom_type is AtomType.asset_record:
                value["asset"] = {
                    k: canonical_cells.get(k)
                    for k in (
                        "asset_id", "site_id", "site_name",
                        "asset_type", "vendor", "manufacturer",
                        "model", "quantity", "mdf", "serial",
                        "ip_address", "in_service_date",
                        "refresh_target", "status",
                    )
                    if canonical_cells.get(k)
                }
            elif profile.atom_type is AtomType.port_vlan_assignment:
                value["port"] = {
                    k: canonical_cells.get(k)
                    for k in (
                        "site_id", "switch_hostname", "port",
                        "vlan_id", "patch_panel_port",
                    )
                    if canonical_cells.get(k)
                }
            elif profile.atom_type is AtomType.circuit_inventory:
                value["circuit"] = {
                    k: canonical_cells.get(k)
                    for k in (
                        "circuit_id", "wan_provider", "site_id",
                        "site_name", "lead_time",
                    )
                    if canonical_cells.get(k)
                }
            elif profile.atom_type is AtomType.alert_route:
                value["alert"] = {
                    k: canonical_cells.get(k)
                    for k in (
                        "monitor_id", "runbook_ref", "alert_severity",
                        "owner", "site_id", "site_name",
                    )
                    if canonical_cells.get(k)
                }
            elif profile.atom_type is AtomType.cutover_validation:
                value["validation"] = {
                    k: canonical_cells.get(k)
                    for k in (
                        "validation_id", "customer_signoff", "pass_flag",
                        "owner", "status", "site_id",
                    )
                    if canonical_cells.get(k)
                }
            elif profile.atom_type is AtomType.support_entitlement:
                value["entitlement"] = {
                    k: canonical_cells.get(k)
                    for k in (
                        "support_level", "contract_id", "renewal_date",
                        "support_notes", "vendor",
                    )
                    if canonical_cells.get(k)
                }

            source_ref = self._build_source_ref(
                artifact_id=artifact_id,
                artifact_type=artifact_type,
                filename=filename,
                sheet_name=sheet_name,
                row_number=row_idx + 1,
                columns=cell_columns,
            )
            atom_id = stable_id(
                "atm",
                project_id,
                artifact_id,
                sheet_name,
                str(row_idx + 1),
                profile.kind,
                normalize_text(raw_text)[:160],
            )
            atom = EvidenceAtom(
                id=atom_id,
                project_id=project_id,
                artifact_id=artifact_id,
                atom_type=profile.atom_type,
                raw_text=raw_text,
                normalized_text=normalize_text(raw_text),
                value=value,
                entity_keys=self._entity_keys_from_operational_row(canonical_cells),
                source_refs=[source_ref],
                receipts=[],
                authority_class=profile.authority_class,
                confidence=profile.confidence,
                review_status=ReviewStatus.auto_accepted,
                review_flags=[],
                parser_version=self.parser_version,
            )
            atoms.append(atom)

            atoms.extend(
                self._emit_field_aware_cell_fact_atoms(
                    project_id=project_id,
                    artifact_id=artifact_id,
                    artifact_type=artifact_type,
                    filename=filename,
                    sheet_name=sheet_name,
                    row_number=row_idx + 1,
                    row_atom_id=atom_id,
                    cells=cells,
                    canonical_cells=canonical_cells,
                    cell_columns=cell_columns,
                )
            )

        return atoms

    def _emit_field_aware_cell_fact_atoms(
        self,
        *,
        project_id: str,
        artifact_id: str,
        artifact_type: ArtifactType,
        filename: str,
        sheet_name: str,
        row_number: int,
        row_atom_id: str,
        cells: dict[str, str],
        canonical_cells: dict[str, str],
        cell_columns: dict[str, str],
    ) -> list[EvidenceAtom]:
        """PR3 — field-aware cell-fact emitter. Same as
        :meth:`_emit_cell_fact_atoms` but knows that cells in
        support_level / coverage / support_entitlement columns are
        SUPPORT-TIER prose, not exclusions, and that conditional
        boundary phrasing ("8x5 vendor support unless noted") gets
        its own ``conditional_support_boundary`` atom type."""
        out: list[EvidenceAtom] = []
        for original_field, text in cells.items():
            text_str = str(text).strip()
            if not text_str:
                continue
            canonical_field = self._canonicalize_header(original_field)

            if canonical_field in _SUPPORT_LEVEL_FIELDS:
                # Support-tier prose is not a contractual exclusion;
                # only emit a conditional boundary atom if the cell
                # explicitly carries conditional language.
                if _CONDITIONAL_SUPPORT_RE.search(text_str):
                    atom_type = AtomType.conditional_support_boundary
                    confidence = 0.88
                    flags = ["conditional_support_boundary"]
                else:
                    continue
            elif _CONDITIONAL_SUPPORT_RE.search(text_str):
                atom_type = AtomType.conditional_support_boundary
                confidence = 0.88
                flags = ["conditional_support_boundary"]
            elif _EXCLUSION_CELL_RE.search(text_str):
                atom_type = AtomType.exclusion
                confidence = 0.90
                flags = []
            elif _RISK_CELL_RE.search(text_str):
                atom_type = AtomType.risk
                confidence = 0.88
                flags = []
            else:
                continue

            source_ref = SourceRef(
                id=stable_id(
                    "src", artifact_id, sheet_name, row_number,
                    original_field, "cell_fact",
                ),
                artifact_id=artifact_id,
                artifact_type=artifact_type,
                filename=filename,
                locator={
                    "sheet": sheet_name,
                    "row": row_number,
                    "columns": {original_field: cell_columns.get(original_field)},
                    "parent_row_atom_id": row_atom_id,
                },
                extraction_method="xlsx_cell_fact_v2_field_aware",
                parser_version=self.parser_version,
            )
            out.append(
                EvidenceAtom(
                    id=stable_id(
                        "atm", project_id, artifact_id, sheet_name,
                        row_number, original_field, atom_type.value, text_str,
                    ),
                    project_id=project_id,
                    artifact_id=artifact_id,
                    atom_type=atom_type,
                    raw_text=f"{original_field}: {text_str}",
                    normalized_text=normalize_text(text_str),
                    value={
                        "kind": "cell_fact",
                        "field": original_field,
                        "canonical_field": canonical_field,
                        "parent_row_atom_id": row_atom_id,
                        "text": text_str,
                    },
                    entity_keys=[],
                    source_refs=[source_ref],
                    receipts=[],
                    authority_class=AuthorityClass.customer_current_authored,
                    confidence=confidence,
                    review_status=ReviewStatus.auto_accepted,
                    review_flags=flags,
                    parser_version=self.parser_version,
                )
            )
        return out

    def _emit_cell_fact_atoms(
        self,
        *,
        project_id: str,
        artifact_id: str,
        artifact_type: ArtifactType,
        filename: str,
        sheet_name: str,
        row_number: int,
        row_atom_id: str,
        cells: dict[str, str],
        cell_columns: dict[str, str],
    ) -> list[EvidenceAtom]:
        """Emit cell-level sub-atoms for buried facts inside a generic row.

        Specifically: if a cell mentions an explicit exclusion phrase
        ("not included", "excluded", "outside coverage", …) emit an
        ``exclusion`` atom anchored to that cell. Same for risk
        signals ("EOL", "single point of failure", "missing", …) →
        ``risk`` atom. Each sub-atom carries
        ``value.parent_row_atom_id`` so the packetizer can group them
        with the parent row.
        """
        out: list[EvidenceAtom] = []
        for field, text in cells.items():
            text_str = str(text).strip()
            if not text_str:
                continue
            if _EXCLUSION_CELL_RE.search(text_str):
                atom_type = AtomType.exclusion
                confidence = 0.90
            elif _RISK_CELL_RE.search(text_str):
                atom_type = AtomType.risk
                confidence = 0.88
            else:
                continue

            source_ref = SourceRef(
                id=stable_id(
                    "src", artifact_id, sheet_name, row_number, field, "cell_fact"
                ),
                artifact_id=artifact_id,
                artifact_type=artifact_type,
                filename=filename,
                locator={
                    "sheet": sheet_name,
                    "row": row_number,
                    "columns": {field: cell_columns.get(field)},
                    "parent_row_atom_id": row_atom_id,
                },
                extraction_method="xlsx_cell_fact_v1",
                parser_version=self.parser_version,
            )
            out.append(
                EvidenceAtom(
                    id=stable_id(
                        "atm",
                        project_id,
                        artifact_id,
                        sheet_name,
                        row_number,
                        field,
                        atom_type.value,
                        text_str,
                    ),
                    project_id=project_id,
                    artifact_id=artifact_id,
                    atom_type=atom_type,
                    raw_text=f"{field}: {text_str}",
                    normalized_text=normalize_text(text_str),
                    value={
                        "kind": "cell_fact",
                        "field": field,
                        "parent_row_atom_id": row_atom_id,
                        "text": text_str,
                    },
                    entity_keys=[],
                    source_refs=[source_ref],
                    receipts=[],
                    authority_class=AuthorityClass.customer_current_authored,
                    confidence=confidence,
                    review_status=ReviewStatus.auto_accepted,
                    review_flags=[],
                    parser_version=self.parser_version,
                )
            )
        return out

    def _emit_generic_rows(
        self,
        project_id: str,
        artifact_id: str,
        artifact_type: ArtifactType,
        filename: str,
        sheet_name: str,
        rows: list[list[Any]],
    ) -> list[EvidenceAtom]:
        """Generic fallback: emit one atom per non-empty row.

        Used when ``_detect_header`` can't find a canonical
        cabling/networking schedule.  Treats the first non-empty row
        with mostly text cells as the column-header row (so cell
        values get keyed by column name); falls back to ``col_N``
        labels otherwise.

        Each row becomes a ``scope_item`` atom with ``raw_text`` set
        to ``"col_a: value | col_b: value | ..."`` so the
        ``entity_extraction`` stage can pull entity keys out of the
        text just like for any other atom.

        See PRODUCTION_GAPS.md P0.4.
        """
        # Universal false-positive guard: workbook tabs whose only purpose
        # is to instruct the human filling in the form (e.g. "Instructions",
        # "Read Me", "Cover", "About") are not data. They reach this code
        # path because their prose can't satisfy the canonical header
        # detector. If the sheet name signals an instructional intent and
        # the body has no detectable schedule structure, emit zero atoms
        # rather than fabricating ``scope_item`` atoms from cover text.
        if _looks_instructional_sheet(sheet_name) and not _has_tabular_shape(rows):
            return []

        atoms: list[EvidenceAtom] = []

        # Step 1: pick a header row.  We scan the first 10 non-empty rows
        # and take the one with the most non-numeric, short string cells.
        # If no row qualifies we use ``col_N`` labels.
        header_idx = -1
        header_score = -1
        for idx, row in enumerate(rows[:10]):
            if _is_blank_row(row):
                continue
            non_empty = [c for c in row if str(c or "").strip()]
            if not non_empty:
                continue
            string_cells = sum(
                1 for c in non_empty
                if isinstance(c, str) and len(c) <= 60 and not c.strip().isdigit()
            )
            score = string_cells - max(0, len(non_empty) - string_cells)
            if score >= 2 and score > header_score:
                header_score = score
                header_idx = idx

        if header_idx >= 0:
            header_row = rows[header_idx]
            columns = [
                (str(c).strip() if c is not None else "") or f"col_{i + 1}"
                for i, c in enumerate(header_row)
            ]
            data_start = header_idx + 1
        else:
            # No detectable header — use col_1, col_2, ... and start
            # from row 0 so we don't lose any rows.
            width = max((len(r) for r in rows), default=0)
            columns = [f"col_{i + 1}" for i in range(width)]
            data_start = 0

        # Step 2: emit one atom per non-empty data row.
        for row_idx in range(data_start, len(rows)):
            row = rows[row_idx] or []
            if _is_blank_row(row):
                continue
            # Render the row as "col: value | col: value" — same shape
            # the canonical line-item emitter uses, so OrbitBrief and
            # entity_extraction get a familiar structure.
            bare: list[str] = []
            cell_columns: dict[str, str] = {}
            for col_idx, col_name in enumerate(columns):
                if col_idx >= len(row):
                    break
                value = row[col_idx]
                if value is None:
                    continue
                value_str = str(value).strip()
                if not value_str:
                    continue
                bare.append(value_str)
                cell_columns[col_name] = get_column_letter(col_idx + 1)
            # raw_text is the bare pipe-join of cell VALUES (not "col: value"),
            # matching the docx per-row blob and the schema row_text — so when a
            # raw_table_row routes this same row to a typed atom (deal_metadata,
            # etc.), cross_type_dedup collapses this scope twin into it. Column
            # meaning is preserved in value.cells and rendered back by
            # _atom_bound_text at decide-time, so nothing is lost for display.
            raw_text = " | ".join(bare).strip()
            if not raw_text:
                continue
            # Skip the header row itself so it doesn't reappear as data.
            if row_idx == header_idx:
                continue

            source_ref = self._build_source_ref(
                artifact_id=artifact_id,
                artifact_type=artifact_type,
                filename=filename,
                sheet_name=sheet_name,
                row_number=row_idx + 1,
                columns=cell_columns,
            )
            atom_id = stable_id(
                "atm",
                artifact_id,
                sheet_name,
                str(row_idx + 1),
                normalize_text(raw_text)[:120],
            )
            cells_by_original_col = {
                col: (str(row[i]).strip() if i < len(row) and row[i] is not None else "")
                for i, col in enumerate(columns)
                if i < len(row) and str(row[i] or "").strip()
            }
            # Profile the row: risk register, asset inventory, support
            # entitlement, site roster, lifecycle status, or generic
            # scope item.  Falls back to the legacy
            # (scope_item, table_row, contractual_scope, 0.84) tuple
            # when no profile fits.
            canonical_columns = [self._canonicalize_header(c) for c in columns]
            canon_cells = {
                canonical_columns[i]: str(row[i]).strip()
                for i in range(min(len(row), len(canonical_columns)))
                if row[i] is not None and str(row[i]).strip()
            }
            atom_type, row_kind, authority_class, confidence = (
                self._generic_row_profile(canon_cells, sheet_name)
            )
            value: dict[str, Any] = {
                "kind": row_kind,
                "sheet": sheet_name,
                "row": row_idx + 1,
                "cells": cells_by_original_col,
                "canonical_cells": canon_cells,
            }
            atoms.append(
                EvidenceAtom(
                    id=atom_id,
                    project_id=project_id,
                    artifact_id=artifact_id,
                    atom_type=atom_type,
                    raw_text=raw_text,
                    normalized_text=normalize_text(raw_text),
                    value=value,
                    entity_keys=[],  # populated by core.entity_extraction
                    source_refs=[source_ref],
                    receipts=[],
                    authority_class=authority_class,
                    confidence=confidence,
                    review_status=ReviewStatus.auto_accepted,
                    review_flags=[],
                    parser_version=self.parser_version,
                )
            )

            # v50: also emit a raw_table_row so the central schema registry can
            # TYPE this row (Field/Value -> deal_metadata, BOM -> bom_line, etc.)
            # — the same path the canonical-header and quote emitters already
            # use. Only when a real header row was found (col_N placeholders
            # can't match a schema). The scope_item above stays as the fail-open
            # fallback for rows that match no schema; when one DOES match,
            # cross_type_dedup collapses the scope twin into the typed atom
            # (shared (sheet,row) cell key + identical bare-pipe raw_text).
            if header_idx >= 0:
                _rtr_id = stable_id("atm", artifact_id, "raw_table_row", sheet_name, row_idx)
                _rtr_src = SourceRef(
                    id=stable_id("src", _rtr_id),
                    artifact_id=artifact_id,
                    artifact_type=artifact_type,
                    filename=filename,
                    locator={
                        "sheet": sheet_name,
                        "row": row_idx + 1,
                        "columns": cell_columns,
                        "extraction": "raw_table_row_v49_2",
                        "section_path": [sheet_name] if sheet_name else [],
                    },
                    extraction_method="raw_table_row_v49_2",
                    parser_version=self.parser_version,
                )
                atoms.append(
                    EvidenceAtom(
                        id=_rtr_id,
                        project_id=project_id,
                        artifact_id=artifact_id,
                        atom_type=AtomType.raw_table_row,
                        raw_text=raw_text[:4000],
                        normalized_text=raw_text.lower()[:4000],
                        value={
                            "_columns": list(columns),
                            "_row": [str(c).strip() if c is not None else "" for c in row],
                            "_table_idx": 0,
                            # 1-based to match the scope_item's locator row
                            # (row_idx+1), so the typed atom shares the scope
                            # twin's (sheet,row) cell key and they collapse.
                            "_row_idx": row_idx + 1,
                            "_filename": filename,
                            "_sheet": sheet_name,
                            "_artifact_type": "xlsx",
                        },
                        entity_keys=[],
                        source_refs=[_rtr_src],
                        receipts=[],
                        authority_class=AuthorityClass.contractual_scope,
                        confidence=0.80,
                        confidence_raw=0.80,
                        calibrated_confidence=0.80,
                        review_status=ReviewStatus.auto_accepted,
                        review_flags=[],
                        parser_version=self.parser_version,
                    )
                )

            # Cell-level sub-atoms — emit exclusion / risk atoms for
            # any cells whose text matches the cell-fact patterns.
            atoms.extend(
                self._emit_cell_fact_atoms(
                    project_id=project_id,
                    artifact_id=artifact_id,
                    artifact_type=artifact_type,
                    filename=filename,
                    sheet_name=sheet_name,
                    row_number=row_idx + 1,
                    row_atom_id=atom_id,
                    cells=cells_by_original_col,
                    cell_columns=cell_columns,
                )
            )

            # Week 6 P6.7: when this row is a Q&A pair (the columns
            # include both a question-column and a response-column),
            # also emit a *separate* atom for the question and the
            # response.  Keeps the row-level atom for context but lets
            # the packetizer surface the Q as ``open_question`` and the
            # A as ``customer_instruction`` independently — that's what
            # closes the XLSX_RARE atom_count gap (486 Q&A rows in
            # CalSAWS were producing 486 atoms instead of ~1500).
            for sub_atom in self._emit_qa_row_subatoms(
                project_id=project_id,
                artifact_id=artifact_id,
                artifact_type=artifact_type,
                filename=filename,
                sheet_name=sheet_name,
                columns=columns,
                row=row,
                row_idx=row_idx,
                cell_columns=cell_columns,
            ):
                atoms.append(sub_atom)
        return atoms

    # Names that mark a column as the *question* side of a Q&A row.
    _XLSX_QUESTION_COL_HINTS = (
        "question",
        "concern",
        "inquiry",
        "issue",
        "comment",
        "ask",
    )
    # Names that mark a column as the *response/answer* side.
    _XLSX_RESPONSE_COL_HINTS = (
        "response",
        "answer",
        "reply",
        "resolution",
        "clarification",
    )

    def _emit_qa_row_subatoms(
        self,
        *,
        project_id: str,
        artifact_id: str,
        artifact_type: ArtifactType,
        filename: str,
        sheet_name: str,
        columns: list[str],
        row: list[Any],
        row_idx: int,
        cell_columns: dict[str, str],
    ) -> list[EvidenceAtom]:
        """If this row has Q/A columns, emit one atom per side."""
        q_col_idx = a_col_idx = None
        for i, col in enumerate(columns):
            col_lower = col.lower()
            if q_col_idx is None and any(h in col_lower for h in self._XLSX_QUESTION_COL_HINTS):
                q_col_idx = i
            if a_col_idx is None and any(h in col_lower for h in self._XLSX_RESPONSE_COL_HINTS):
                a_col_idx = i
        # Need both sides to be a real Q&A row.
        if q_col_idx is None or a_col_idx is None or q_col_idx == a_col_idx:
            return []

        out: list[EvidenceAtom] = []
        for kind, col_idx, atom_type, authority in (
            ("question", q_col_idx, AtomType.open_question, AuthorityClass.contractual_scope),
            ("answer", a_col_idx, AtomType.customer_instruction, AuthorityClass.customer_current_authored),
        ):
            if col_idx >= len(row):
                continue
            value = row[col_idx]
            if value is None:
                continue
            value_str = str(value).strip()
            if len(value_str) < 10:
                continue
            col_name = columns[col_idx]
            sub_text = f"{col_name}: {value_str}"
            sub_source_ref = self._build_source_ref(
                artifact_id=artifact_id,
                artifact_type=artifact_type,
                filename=filename,
                sheet_name=sheet_name,
                row_number=row_idx + 1,
                columns={col_name: cell_columns.get(col_name, "")},
            )
            sub_atom_id = stable_id(
                "atm",
                artifact_id,
                sheet_name,
                str(row_idx + 1),
                kind,
                normalize_text(value_str)[:120],
            )
            out.append(
                EvidenceAtom(
                    id=sub_atom_id,
                    project_id=project_id,
                    artifact_id=artifact_id,
                    atom_type=atom_type,
                    raw_text=sub_text,
                    normalized_text=normalize_text(sub_text),
                    value={
                        "kind": "qa_subatom",
                        "qa_side": kind,
                        "sheet": sheet_name,
                        "row": row_idx + 1,
                        "column": col_name,
                    },
                    entity_keys=[],
                    source_refs=[sub_source_ref],
                    receipts=[],
                    authority_class=authority,
                    confidence=0.85,
                    review_status=ReviewStatus.auto_accepted,
                    review_flags=[],
                    parser_version=self.parser_version,
                )
            )
        return out

    def _extract_row_values(self, row: list[Any], header_map: dict[str, int]) -> dict[str, str]:
        extracted: dict[str, str] = {}
        for key, idx in header_map.items():
            value = row[idx] if idx < len(row) else ""
            extracted[key] = str(value).strip() if value is not None else ""
        return extracted

    def _entity_keys_from_extracted(self, extracted: dict[str, str]) -> list[str]:
        keys: list[str] = []
        mapping = [
            ("site", "site"),
            ("building", "building"),
            ("floor", "floor"),
            ("room", "room"),
            ("area", "area"),
            ("zone", "zone"),
            ("location", "location"),
            ("plate_id", "plate"),
            ("outlet_id", "location"),
            ("drop_id", "location"),
            ("mdf", "mdf"),
            ("idf", "idf"),
        ]
        for field, etype in mapping:
            v = extracted.get(field, "").strip()
            if v:
                key = normalize_entity_key(etype, v)
                if key:
                    keys.append(key)
        # Device vs service classification — BOM rows describing labor,
        # training, hypercare, project management, etc. are SERVICE line
        # items, not devices. Routing them to `service:` instead of
        # `device:` keeps the device namespace clean (only physical
        # hardware ends up under `device:`).
        device_value = extracted.get("device", "").strip()
        if device_value:
            etype = "service" if _looks_like_service_line(device_value) else "device"
            key = normalize_entity_key(etype, device_value)
            if key:
                keys.append(key)
        return keys

    def _emit_subtotal_row(
        self,
        project_id: str,
        artifact_id: str,
        artifact_type: ArtifactType,
        filename: str,
        sheet_name: str,
        row_number: int,
        model: SheetParseModel,
        extracted: dict[str, str],
        row: list[Any],
        label_indices: list[int],
    ) -> list[EvidenceAtom]:
        del extracted, row, label_indices
        # Subtotal: no entity/quantity atoms (avoid double-count with grand total).
        _ = model.diagnostics
        return []

    def _emit_total_row(
        self,
        project_id: str,
        artifact_id: str,
        artifact_type: ArtifactType,
        filename: str,
        sheet_name: str,
        row_number: int,
        model: SheetParseModel,
        row: list[Any],
        label_indices: list[int],
    ) -> list[EvidenceAtom]:
        atoms: list[EvidenceAtom] = []
        label_col = label_indices[0] if label_indices else 0
        label_letter = get_column_letter(label_col + 1)

        if model.wide_qty_columns:
            for wcol in model.wide_qty_columns:
                idx = wcol["col_idx"]
                if idx >= len(row):
                    continue
                parsed = _parse_schedule_quantity_cell(row[idx])
                if parsed.get("quantity") is None and parsed.get("quantity_status") not in {"zero"}:
                    continue
                qty = parsed.get("quantity")
                if qty is None and parsed.get("quantity_status") == "zero":
                    qty = 0
                if qty is None:
                    continue
                qcol = get_column_letter(idx + 1)
                columns = {"total_label": label_letter, "quantity": qcol}
                source_ref = self._build_source_ref(artifact_id, artifact_type, filename, sheet_name, row_number, columns)
                entity_keys: list[str] = []
                hint = wcol.get("entity_hint")
                if hint:
                    entity_keys.append(normalize_entity_key(hint[0], hint[1]))
                value = {
                    **parsed,
                    "item": wcol["item"],
                    "normalized_item": wcol["normalized_item"],
                    "item_kind": wcol.get("item_kind"),
                    "material_family": wcol.get("material_family"),
                    "cable_category": wcol.get("cable_category"),
                    "shielding": wcol.get("shielding"),
                    "source_row_type": "total",
                    "aggregate": True,
                }
                value = merge_parser_value_identity(value, raw_text=f"Total {wcol['item']} {qty}")
                atoms.append(
                    EvidenceAtom(
                        id=stable_id("atm", project_id, artifact_id, sheet_name, row_number, "qty_total", wcol["normalized_item"], str(qty)),
                        project_id=project_id,
                        artifact_id=artifact_id,
                        atom_type=AtomType.quantity,
                        raw_text=f"Total {wcol['item']} {qty}",
                        normalized_text=normalize_text(f"total {wcol['item']} {qty}"),
                        value=value,
                        entity_keys=entity_keys,
                        source_refs=[source_ref],
                        authority_class=AuthorityClass.approved_site_roster,
                        confidence=0.94,
                        review_status=ReviewStatus.auto_accepted,
                        review_flags=["xlsx_parser:aggregate_total"],
                        parser_version=self.parser_version,
                    )
                )
            return atoms

        # Traditional single quantity column total row
        qidx = model.header_map.get("quantity")
        if qidx is None or qidx >= len(row):
            return atoms
        parsed = _parse_schedule_quantity_cell(row[qidx])
        qty = parsed.get("quantity")
        if qty is None and parsed.get("quantity_status") != "zero":
            return atoms
        if qty is None and parsed.get("quantity_status") == "zero":
            qty = 0
        columns = {"total_label": label_letter, "quantity": get_column_letter(qidx + 1)}
        source_ref = self._build_source_ref(artifact_id, artifact_type, filename, sheet_name, row_number, columns)
        value = {**parsed, "source_row_type": "total", "aggregate": True, "item": "total", "normalized_item": "total"}
        value = merge_parser_value_identity(value, raw_text=f"Total quantity {qty}")
        atoms.append(
            EvidenceAtom(
                id=stable_id("atm", project_id, artifact_id, sheet_name, row_number, "qty_total_single", str(qty)),
                project_id=project_id,
                artifact_id=artifact_id,
                atom_type=AtomType.quantity,
                raw_text=f"Total quantity {qty}",
                normalized_text=normalize_text(f"total quantity {qty}"),
                value=value,
                entity_keys=[],
                source_refs=[source_ref],
                authority_class=AuthorityClass.approved_site_roster,
                confidence=0.92,
                review_status=ReviewStatus.auto_accepted,
                review_flags=["xlsx_parser:aggregate_total"],
                parser_version=self.parser_version,
            )
        )
        return atoms

    def _emit_line_item_row(
        self,
        project_id: str,
        artifact_id: str,
        artifact_type: ArtifactType,
        filename: str,
        sheet_name: str,
        row_number: int,
        model: SheetParseModel,
        extracted: dict[str, str],
        row: list[Any],
        label_indices: list[int],
    ) -> list[EvidenceAtom]:
        atoms: list[EvidenceAtom] = []
        site = extracted.get("site", "").strip()
        device = extracted.get("device", "").strip()
        floor = extracted.get("floor", "").strip()
        room = extracted.get("room", "").strip()
        location = extracted.get("location", "").strip() or extracted.get("description", "").strip()
        scope = extracted.get("scope", "").strip()
        access = extracted.get("access", "").strip() or extracted.get("access_window", "").strip()
        plate = extracted.get("plate_id", "").strip()
        notes_blob = _row_text_blob(row, model.header_map)

        # Do not treat "total" in notes as skipping — row_kind already line_item.
        label_text = _label_text_for_row(row, label_indices)
        if _is_total_label(label_text):
            return []

        entity_keys = self._entity_keys_from_extracted(extracted)
        major = sum(1 for k in ("site", "device", "location", "plate_id", "room") if extracted.get(k))
        row_confidence = 0.92 if major >= 2 else (0.88 if major == 1 else 0.78)

        def append_atom(
            atom_type: AtomType,
            raw_text: str,
            value: dict[str, Any],
            confidence: float,
            *,
            entity_keys_out: list[str] | None = None,
            review_status: ReviewStatus = ReviewStatus.auto_accepted,
            review_flags: list[str] | None = None,
            extra_columns: dict[str, str] | None = None,
        ) -> None:
            cols = dict(extra_columns or {})
            ek = entity_keys_out if entity_keys_out is not None else entity_keys
            atoms.append(
                EvidenceAtom(
                    id=stable_id("atm", project_id, artifact_id, sheet_name, row_number, atom_type.value, raw_text),
                    project_id=project_id,
                    artifact_id=artifact_id,
                    atom_type=atom_type,
                    raw_text=raw_text,
                    normalized_text=normalize_text(raw_text),
                    value=value,
                    entity_keys=ek,
                    source_refs=[
                        self._build_source_ref(artifact_id, artifact_type, filename, sheet_name, row_number, cols)
                    ],
                    authority_class=AuthorityClass.approved_site_roster,
                    confidence=confidence,
                    review_status=review_status,
                    review_flags=review_flags or [],
                    parser_version=self.parser_version,
                )
            )

        # Entities: precise types, not everything as site
        for field, etype, label in (
            ("site", "site", "Site"),
            ("building", "building", "Building"),
            ("floor", "floor", "Floor"),
            ("room", "room", "Room"),
            ("area", "area", "Area"),
            ("zone", "zone", "Zone"),
            ("location", "location", "Location"),
            ("plate_id", "plate", "Plate"),
            ("mdf", "mdf", "MDF"),
            ("idf", "idf", "IDF"),
            ("device", "device", "Device"),
        ):
            val = extracted.get(field, "").strip()
            if not val:
                continue
            idx = model.header_map.get(field)
            letter = get_column_letter((idx or 0) + 1) if idx is not None else "A"
            single_key = [normalize_entity_key(etype, val)]
            append_atom(
                AtomType.entity,
                f"{label} {val}",
                {"entity_type": etype, "name": val},
                row_confidence,
                entity_keys_out=single_key,
                extra_columns={field: letter},
            )

        if model.wide_qty_columns:
            ctx_cols: dict[str, str] = {}
            for key in ("plate_id", "location", "room", "site", "description"):
                idx = model.header_map.get(key)
                if idx is not None and idx < len(row):
                    ctx_cols[key] = get_column_letter(idx + 1)
            for wcol in model.wide_qty_columns:
                idx = wcol["col_idx"]
                if idx >= len(row):
                    continue
                parsed = _parse_schedule_quantity_cell(row[idx])
                if parsed.get("quantity") is None and parsed.get("quantity_status") not in {"zero", "included", "allowance", "range"}:
                    if str(row[idx] or "").strip() == "":
                        continue
                qty_val = parsed.get("quantity")
                qcol = get_column_letter(idx + 1)
                cols = {**ctx_cols, "quantity": qcol}
                hint = wcol.get("entity_hint")
                hint_keys = [normalize_entity_key(hint[0], hint[1])] if hint else []
                q_entity_keys = list(dict.fromkeys(entity_keys + hint_keys))
                value = {
                    **parsed,
                    "item": wcol["item"],
                    "normalized_item": wcol["normalized_item"],
                    "item_kind": wcol.get("item_kind"),
                    "material_family": wcol.get("material_family"),
                    "cable_category": wcol.get("cable_category"),
                    "shielding": wcol.get("shielding"),
                    "source_row_type": "line_item",
                    "aggregate": False,
                    "plate_id": plate or None,
                    "location": location or None,
                }
                value = merge_parser_value_identity(
                    value,
                    raw_text=f"{wcol['item']} {scope} {location} {row[idx]}".strip(),
                )
                rev = ReviewStatus.needs_review if parsed.get("review_flags") else ReviewStatus.auto_accepted
                atoms.append(
                    EvidenceAtom(
                        id=stable_id(
                            "atm",
                            project_id,
                            artifact_id,
                            sheet_name,
                            row_number,
                            "qty_wide",
                            wcol["normalized_item"],
                            str(qty_val or parsed.get("quantity_status")),
                        ),
                        project_id=project_id,
                        artifact_id=artifact_id,
                        atom_type=AtomType.quantity,
                        raw_text=f"Quantity {wcol['item']} {row[idx]}",
                        normalized_text=normalize_text(f"quantity {wcol['item']} {row[idx]}"),
                        value=value,
                        entity_keys=q_entity_keys,
                        source_refs=[self._build_source_ref(artifact_id, artifact_type, filename, sheet_name, row_number, cols)],
                        authority_class=AuthorityClass.approved_site_roster,
                        confidence=row_confidence * 0.95,
                        review_status=rev,
                        review_flags=parsed.get("review_flags") or [],
                        parser_version=self.parser_version,
                    )
                )
        else:
            quantity_raw = extracted.get("quantity", "").strip()
            if quantity_raw:
                parsed = _parse_schedule_quantity_cell(quantity_raw)
                qidx = model.header_map.get("quantity")
                qcol = get_column_letter((qidx or 0) + 1) if qidx is not None else "D"
                cols = {k: get_column_letter(model.header_map[k] + 1) for k in ("site", "device", "floor", "room", "quantity") if k in model.header_map}
                if "quantity" not in cols:
                    cols["quantity"] = qcol
                rev = ReviewStatus.needs_review if parsed.get("uncertain") or parsed.get("review_flags") else ReviewStatus.auto_accepted
                flags = list(parsed.get("review_flags") or [])
                if parsed.get("uncertain"):
                    flags.append("quantity_uncertain")
                value = {**parsed, "source_row_type": "line_item", "aggregate": False}
                value = merge_parser_value_identity(value, raw_text=f"Quantity {quantity_raw} {scope}")
                atoms.append(
                    EvidenceAtom(
                        id=stable_id("atm", project_id, artifact_id, sheet_name, row_number, "qty", quantity_raw),
                        project_id=project_id,
                        artifact_id=artifact_id,
                        atom_type=AtomType.quantity,
                        raw_text=f"Quantity {quantity_raw}",
                        normalized_text=normalize_text(f"quantity {quantity_raw}"),
                        value=value,
                        entity_keys=entity_keys,
                        source_refs=[self._build_source_ref(artifact_id, artifact_type, filename, sheet_name, row_number, cols)],
                        authority_class=AuthorityClass.approved_site_roster,
                        confidence=row_confidence,
                        review_status=rev,
                        review_flags=flags,
                        parser_version=self.parser_version,
                    )
                )

        if scope or (site and device):
            work_scope = scope if scope else "work_item"
            cols = {k: get_column_letter(model.header_map[k] + 1) for k in ("scope", "site", "device", "floor", "room") if k in model.header_map}
            append_atom(
                AtomType.scope_item,
                f"Scope {work_scope}",
                {"scope": work_scope, "site": site, "device": device, "floor": floor, "room": room, "location": location},
                row_confidence * 0.9,
                extra_columns=cols or {"scope": "E"},
            )

        if access:
            aidx = model.header_map.get("access")
            if aidx is None:
                aidx = model.header_map.get("access_window")
            acol = get_column_letter((aidx or 0) + 1) if aidx is not None else "E"
            append_atom(
                AtomType.constraint,
                f"Access {access}",
                {"access_window": access, "site": site, "device": device, "location": location},
                row_confidence * 0.88,
                extra_columns={"access": acol},
            )

        scope_context_cols: dict[str, str] = {}
        if label_indices:
            scope_context_cols["label"] = get_column_letter(label_indices[0] + 1)
        for nk in ("notes", "description", "scope"):
            nix = model.header_map.get(nk)
            if nix is not None:
                scope_context_cols[nk] = get_column_letter(nix + 1)
        if not scope_context_cols:
            scope_context_cols["context"] = "A"

        def scope_append(
            atom_type: AtomType,
            raw_text: str,
            value: dict[str, Any],
            confidence: float,
            *,
            review_status: ReviewStatus = ReviewStatus.auto_accepted,
            review_flags: list[str] | None = None,
            extra_columns: dict[str, str] | None = None,
        ) -> None:
            append_atom(
                atom_type,
                raw_text,
                value,
                confidence,
                review_status=review_status,
                review_flags=review_flags,
                extra_columns=extra_columns or scope_context_cols,
            )

        _emit_scope_constraint_atoms(
            notes_blob,
            site,
            device,
            floor,
            room,
            location,
            scope_append,
            row_confidence,
            context_columns=scope_context_cols,
        )

        return atoms
