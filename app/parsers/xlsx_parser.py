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
from app.parsers.segmenters import segment_xlsx
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
            atoms, sheets = self._parse_csv(project_id=project_id, artifact_id=artifact_id, path=path)
            schema = STRUCTURED_SCHEMA_CSV
            artifact_type = ArtifactType.csv
        else:
            atoms, sheets = self._parse_xlsx(project_id=project_id, artifact_id=artifact_id, path=path)
            schema = STRUCTURED_SCHEMA_XLSX
            artifact_type = ArtifactType.xlsx

        structured_doc = self._build_structured_doc(
            schema=schema,
            artifact_type=artifact_type,
            filename=path.name,
            sheets=sheets,
        )
        stamp_section_and_block_ids(structured_doc, artifact_seed=artifact_id)
        return ParserOutput(
            atoms=atoms,
            derived_files=derived_files_for(artifact_path=path, structured_doc=structured_doc),
        )

    def _parse_xlsx(
        self, project_id: str, artifact_id: str, path: Path
    ) -> tuple[list[EvidenceAtom], list[dict[str, Any]]]:
        try:
            workbook = load_workbook(path, read_only=True, data_only=True)
        except Exception:
            return [], []
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
        return atoms, sheets

    def _parse_csv(
        self, project_id: str, artifact_id: str, path: Path
    ) -> tuple[list[EvidenceAtom], list[dict[str, Any]]]:
        try:
            with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
                reader = csv.reader(handle)
                rows = [list(row) for row in reader]
        except Exception:
            return [], []
        sheet_atoms = self._parse_sheet_rows(
            project_id=project_id,
            artifact_id=artifact_id,
            filename=path.name,
            artifact_type=ArtifactType.csv,
            sheet_name="csv",
            rows=rows,
        )
        return sheet_atoms, [{"name": "csv", "rows": rows}]

    def _build_structured_doc(
        self,
        *,
        schema: str,
        artifact_type: ArtifactType,
        filename: str,
        sheets: list[dict[str, Any]],
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
        return make_structured_document(
            schema_version=schema,
            filename=filename,
            artifact_type=artifact_type.value,
            title=filename,
            metadata=[f"sheet: {s['name']}" for s in sheets],
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
                    atom_type=AtomType.entity,
                    raw_text=row_text,
                    normalized_text=row_text.lower(),
                    value={
                        "kind": "physical_site",
                        "site_id": sid or site_row.site_id,
                        "facility_name": site_row.facility_name,
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
        # RF1 — explicit fast-path for known structured-row CSVs.
        # Files named asset_inventory / site_list / risk_register /
        # license_support_matrix / lifecycle should route to the
        # typed-row profiler in _emit_generic_rows so they produce
        # asset_record / site_roster / risk / support_entitlement /
        # lifecycle_status atoms (PR2). Restricted to .csv to avoid
        # disturbing existing .xlsx fixtures (e.g. demo_project's
        # site_list.xlsx that legacy tests expect on the canonical
        # path).
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
            parts: list[str] = []
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
                parts.append(f"{col_name}: {value_str}")
                cell_columns[col_name] = get_column_letter(col_idx + 1)
            raw_text = " | ".join(parts).strip()
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
