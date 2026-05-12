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
                sniff_xlsx_roster_schedule_strength,
            )

            confidence = 0.58
            reasons.append(f"spreadsheet_extension:{suffix}")
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
        return atoms

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
            value: dict[str, Any] = {
                "kind": "table_row",
                "sheet": sheet_name,
                "row": row_idx + 1,
                "cells": {
                    col: (str(row[i]).strip() if i < len(row) and row[i] is not None else "")
                    for i, col in enumerate(columns)
                    if i < len(row) and str(row[i] or "").strip()
                },
            }
            atoms.append(
                EvidenceAtom(
                    id=atom_id,
                    project_id=project_id,
                    artifact_id=artifact_id,
                    atom_type=AtomType.scope_item,
                    raw_text=raw_text,
                    normalized_text=normalize_text(raw_text),
                    value=value,
                    entity_keys=[],  # populated by core.entity_extraction
                    source_refs=[source_ref],
                    receipts=[],
                    authority_class=AuthorityClass.contractual_scope,
                    confidence=0.84,
                    review_status=ReviewStatus.auto_accepted,
                    review_flags=[],
                    parser_version=self.parser_version,
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
            ("device", "device"),
        ]
        for field, etype in mapping:
            v = extracted.get(field, "").strip()
            if v:
                keys.append(normalize_entity_key(etype, v))
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
