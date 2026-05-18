"""Per-page Parser Intelligence — the parser's complete understanding.

A ``PageIntelligence`` document is the parser's full report on what it
saw, what it knew before seeing it, and how the two combined. It pulls
every layer of the takeoff pipeline into one cross-referenced view:

  ProjectReference (pages 0-2)        — what the project DECLARES
        legend         — what each symbol means
        schedule       — which vendors/parts implement each component
        spec           — what the project requires
              │
              ▼
  DetectionPlan (per page)            — what to look for HERE
        expected_symbols (from legend.rows[*].SYMBOL)
        project_zones (parsed from prior pages + this page)
        page strategy (router output)
              │
              ▼
  PlanExtract (per page)              — what got found
        devices, keyed_notes, zone_notes, schematic_region
              │
              ▼
  PageIntelligence                    — per-device understanding
        for each device:
          * legend_row  (cable spec, termination, mounting, power, remarks)
          * icon_png    (cropped from the legend page)
          * schedule_cross_refs (component schedule rows matching the system)
          * spec_paragraph_refs (spec paragraphs mentioning the device class)
          * location_context (room_guess, keynote_ref+text, home_run_to)
          * confidence_breakdown (per-signal score + composite)

The whole point: when you read a PageIntelligence document, you see
exactly what the parser knows about every device on the page, traced
back to the page in the project reference that declared it. No
hardcoded knowledge bleeds in.
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from app.takeoff.legend_extract import _resolve_symbol_column_index
from app.takeoff.plan_extract import (
    build_plan_extract,
    find_legend_row_for_symbol,
)


SCHEMA_VERSION = "purtera.lowvoltage.page_intelligence.v1"
READABLE_SCHEMA_VERSION = "purtera.lowvoltage.page_intelligence.readable.v1"


# Bare symbol code shape — letters / digits / hyphens, 1-6 chars, starts
# with a letter. Catches WN, POS-T, FACP-2, IC, KP, DC, etc.
_SYMBOL_SHAPE_RE = re.compile(r"^[A-Z][A-Z0-9\-]{0,5}$")


def _normalize_symbol_code(raw: str) -> str | None:
    """Strip port-count placeholders and validate symbol code shape.

    Legend rows often use ``"POS-T #"`` / ``"A #"`` / ``"TV #"`` /
    ``"F 2"`` / ``"F 2 TV"`` style codes where ``#`` is the per-row
    port-count placeholder. On the actual plans these appear as the
    bare prefix (``POS-T``, ``A``, ``TV``). We strip the trailing
    placeholder and validate the remainder as a code-shaped token.

    Returns the cleaned code, or ``None`` if the input is paragraph
    text / a numeric placeholder / a multi-word note / etc.
    """
    if not raw:
        return None
    s = raw.strip()
    # Common placeholder variants — strip trailing ``" #"`` or ``" 2"``.
    s = re.sub(r"\s+[#\d]+\s*$", "", s).strip()
    # Some rows are ``"F 2 TV"`` style — try just the first token.
    parts = s.split()
    candidate = parts[0] if parts else s
    candidate = candidate.upper().strip()
    if not _SYMBOL_SHAPE_RE.match(candidate):
        return None
    return candidate


# ─────────────────────── DetectionPlan helpers ────────────────────────


def _iter_legend_rows(project_reference: dict[str, Any]) -> list[dict[str, Any]]:
    """Yield every legend row with its column→value dict and section name."""
    legend = project_reference.get("legend")
    if legend is None:
        return []
    legend_list = legend if isinstance(legend, list) else [legend]
    out: list[dict[str, Any]] = []
    for leg in legend_list:
        for table in leg.get("tables", []) or []:
            for section in table.get("sections", []) or []:
                cols = section.get("column_headers") or []
                sym_idx = _resolve_symbol_column_index(cols)
                if sym_idx is None:
                    continue
                sym_col_name = (cols[sym_idx].get("text") or "SYMBOL").strip()
                for row in section.get("rows") or []:
                    cbh = row.get("cells_by_header") or {}
                    sym_text = (cbh.get(sym_col_name) or "").strip()
                    if not sym_text:
                        continue
                    code = _normalize_symbol_code(sym_text)
                    if code is None:
                        continue
                    out.append({
                        "symbol":          code,
                        "raw_symbol_text": sym_text,
                        "section_title":   section.get("title"),
                        "row_cells_by_header": dict(cbh),
                    })
    return out


def _iter_schedule_rows(project_reference: dict[str, Any]) -> list[dict[str, Any]]:
    """Yield every component-schedule row with column→value + section title."""
    schedule = project_reference.get("schedule")
    if schedule is None:
        return []
    schedule_list = schedule if isinstance(schedule, list) else [schedule]
    out: list[dict[str, Any]] = []
    for sch in schedule_list:
        for table in sch.get("tables", []) or []:
            for section in table.get("sections", []) or []:
                for row in section.get("rows") or []:
                    cbh = row.get("cells_by_header") or {}
                    if not cbh:
                        continue
                    out.append({
                        "section_title": section.get("title"),
                        "cells_by_header": dict(cbh),
                    })
    return out


def _iter_spec_paragraphs(project_reference: dict[str, Any]) -> list[dict[str, Any]]:
    """Yield every spec paragraph + bullet with its heading context."""
    spec = project_reference.get("spec")
    if spec is None:
        return []
    spec_list = spec if isinstance(spec, list) else [spec]
    out: list[dict[str, Any]] = []
    for sp in spec_list:
        for section in sp.get("sections", []) or []:
            heading = (section.get("heading") or "").strip()
            for p in section.get("paragraphs") or []:
                p_clean = " ".join((p or "").split())
                if p_clean:
                    out.append({"heading": heading, "kind": "paragraph", "text": p_clean})
            for b in section.get("bullets") or []:
                b_clean = " ".join((b or "").split())
                if b_clean:
                    out.append({"heading": heading, "kind": "bullet", "text": b_clean})
    return out


def _project_zones(project_reference: dict[str, Any], plan_extract: dict[str, Any]) -> list[str]:
    """Collect the set of homerun targets visible to the parser.

    Combines zone targets parsed from THIS page's plan_extract with any
    closet IDs mentioned in the legend / schedule (MDF / IDF-N / TR-N).
    """
    targets: set[str] = set()
    for z in plan_extract.get("zone_notes") or []:
        t = (z.get("target") or "").strip()
        if t:
            targets.add(t)
    # Also pull from any room name patterns in legend / schedule that
    # look like closet identifiers. Tight pattern: the prefix MUST be
    # followed by a hyphen + ID (IDF-2, TR-3, ER-A, BDF-5) OR be the
    # literal "MDF ROOM" phrase. Plain words like TRAY / TRONICS /
    # TROUGHS that happen to start with "TR" are NOT closet refs.
    closet_re = re.compile(
        r"\b("
        r"MDF\s+ROOM|"          # MDF ROOM (literal)
        r"IDF-[A-Z0-9]+|"        # IDF-2, IDF-A, IDF-12
        r"TR-[A-Z0-9]+|"         # TR-3
        r"ER-[A-Z0-9]+|"         # ER-1
        r"BDF-[A-Z0-9]+"         # BDF-5
        r")\b"
    )
    for row in _iter_legend_rows(project_reference) + _iter_schedule_rows(project_reference):
        cbh = row.get("row_cells_by_header") or row.get("cells_by_header") or {}
        for v in cbh.values():
            if not isinstance(v, str):
                continue
            for m in closet_re.finditer(v.upper()):
                t = re.sub(r"\s+", " ", m.group(1).strip())
                if t.startswith("MDF"):
                    t = "MDF ROOM"
                targets.add(t)
    return sorted(targets)


def build_detection_plan(project_reference: dict[str, Any]) -> dict[str, Any]:
    """Build the parser's "what should I look for" plan based on the reference."""
    legend_rows = _iter_legend_rows(project_reference)
    # Deduplicate symbols by uppercase code.
    by_symbol: dict[str, dict[str, Any]] = {}
    for row in legend_rows:
        sym = row["symbol"].strip().upper()
        if sym not in by_symbol:
            by_symbol[sym] = {
                "symbol": sym,
                "section": row.get("section_title"),
                "row": row.get("row_cells_by_header") or {},
            }
    expected_symbols = sorted(by_symbol.values(), key=lambda e: e["symbol"])

    # Pull schedule + spec for cross-reference counts.
    schedule_rows = _iter_schedule_rows(project_reference)
    spec_paragraphs = _iter_spec_paragraphs(project_reference)

    return {
        "expected_symbols": expected_symbols,
        "expected_symbol_codes": [e["symbol"] for e in expected_symbols],
        "schedule_rows_available": len(schedule_rows),
        "spec_paragraphs_available": len(spec_paragraphs),
        "intro_pages": project_reference.get("intro_pages") or {},
    }


# ─────────────────────── Cross-reference lookups ──────────────────────


# Mapping from normalized device class → keywords to look for in schedule
# rows / spec paragraphs. Lets us match (e.g.) a WIRELESS_NODE_OUTLET device
# against schedule entries mentioning "CABLE" or "CAT6" without needing
# an exact part number link.
_CLASS_KEYWORDS: dict[str, tuple[str, ...]] = {
    "wireless_node_outlet": ("CAT6", "WIRELESS", "WAP", "WN", "ACCESS POINT", "WI-FI"),
    "matv_outlet":           ("MATV", "COAX", "RG6", "RG11", "TV", "TELEVISION", "VIDEO"),
    "pos_terminal_outlet":   ("POS", "POINT OF SALE", "TERMINAL"),
    "pos_printer_outlet":    ("POS", "PRINTER", "POINT OF SALE"),
    "access_control_card_reader": ("CARD READER", "ACCESS CONTROL", "PROXIMITY"),
    "duress_alarm_push_button":   ("DURESS", "PANIC", "ALARM"),
    "door_alarm_contact":         ("DOOR", "CONTACT", "ALARM"),
    "house_phone_outlet":         ("HOUSE PHONE", "PHONE", "VOIP", "VOICE"),
    "intercom_remote_station":    ("INTERCOM", "REMOTE STATION"),
    "motion_detector":            ("MOTION", "DETECTOR"),
    "cctv_camera":                ("CAMERA", "CCTV", "MINI DOME", "BULLET"),
}


def _schedule_cross_refs(
    *, device_class: str, schedule_rows: list[dict[str, Any]], limit: int = 5
) -> list[dict[str, Any]]:
    """Find component-schedule rows whose text mentions this device class."""
    keywords = _CLASS_KEYWORDS.get(device_class, ())
    if not keywords:
        return []
    hits: list[dict[str, Any]] = []
    for row in schedule_rows:
        cbh = row.get("cells_by_header") or {}
        haystack = " ".join(str(v).upper() for v in cbh.values())
        if any(k.upper() in haystack for k in keywords):
            hits.append({
                "section": row.get("section_title"),
                "row": dict(cbh),
            })
            if len(hits) >= limit:
                break
    return hits


def _spec_paragraph_refs(
    *, device_class: str, spec_paragraphs: list[dict[str, Any]], limit: int = 3
) -> list[dict[str, Any]]:
    """Find spec paragraphs mentioning anything related to the device class."""
    keywords = _CLASS_KEYWORDS.get(device_class, ())
    if not keywords:
        return []
    hits: list[dict[str, Any]] = []
    for p in spec_paragraphs:
        text_u = p.get("text", "").upper()
        if any(k.upper() in text_u for k in keywords):
            hits.append({
                "heading": p.get("heading"),
                "kind": p.get("kind"),
                "text": p.get("text"),
            })
            if len(hits) >= limit:
                break
    return hits


def _icon_path_for_symbol(
    *, raw_symbol: str, project_reference: dict[str, Any],
) -> str | None:
    """Return the icon PNG filename for a raw symbol, when the reference
    has an ``icon_map`` and a matching legend section/row.

    The icon_map is keyed by ``"<section_1based>/<row_0based>"``. We
    iterate the legend in the same order legend_extract uses (which is
    also the order crop_symbol_icons used) to find the section/row of
    the first legend entry whose SYMBOL matches.
    """
    legend = project_reference.get("legend")
    if legend is None:
        return None
    leg = legend[0] if isinstance(legend, list) else legend
    icon_map: dict[str, str] = leg.get("icon_map") or {}
    if not icon_map:
        return None
    target = raw_symbol.strip().upper()
    section_idx = 0
    for table in leg.get("tables", []) or []:
        for section in table.get("sections", []) or []:
            section_idx += 1
            cols = section.get("column_headers") or []
            sym_idx = _resolve_symbol_column_index(cols)
            if sym_idx is None:
                continue
            sym_col_name = (cols[sym_idx].get("text") or "SYMBOL").strip()
            for row_idx, row in enumerate(section.get("rows") or []):
                cbh = row.get("cells_by_header") or {}
                sym_text = (cbh.get(sym_col_name) or "").strip().upper()
                if sym_text != target:
                    continue
                key = f"{section_idx}/{row_idx}"
                if key in icon_map:
                    return icon_map[key]
    return None


# ─────────────────────── Confidence scoring ─────────────────────────


def _confidence_breakdown(*, device: dict[str, Any], legend_hit: bool) -> dict[str, Any]:
    """Compose a per-device confidence breakdown.

    Signals:
      * legend lookup hit         → 0.4
      * keynote ref resolved      → 0.2
      * home_run_to resolved      → 0.2
      * room_guess captured       → 0.1
      * no review flags           → 0.1
    Maximum: 1.0.
    """
    score = 0.0
    breakdown: dict[str, Any] = {}
    legend_w = 0.4 if legend_hit else 0.0
    breakdown["legend_lookup"] = legend_w
    score += legend_w

    keynote_w = 0.2 if device.get("keynote_text") else (0.1 if device.get("keynote") else 0.0)
    breakdown["keynote_resolution"] = keynote_w
    score += keynote_w

    home_w = 0.2 if device.get("home_run_to") else 0.0
    breakdown["home_run_resolution"] = home_w
    score += home_w

    room_w = 0.1 if device.get("room_guess") else 0.0
    breakdown["room_label"] = room_w
    score += room_w

    flags_w = 0.1 if not (device.get("review_flags") or []) else 0.0
    breakdown["no_review_flags"] = flags_w
    score += flags_w

    breakdown["composite"] = round(min(1.0, score), 3)
    return breakdown


# ────────────────── PageIntelligence top-level builder ──────────────


def build_page_intelligence(
    *,
    pdf_path: Path,
    page_index: int,
    project_reference: dict[str, Any],
) -> dict[str, Any]:
    """Produce the complete intelligence report for one page."""
    started = time.perf_counter()
    plan = build_detection_plan(project_reference)
    extract = build_plan_extract(
        pdf_path=pdf_path,
        page_index=page_index,
        project_reference=project_reference,
    )

    schedule_rows = _iter_schedule_rows(project_reference)
    spec_paragraphs = _iter_spec_paragraphs(project_reference)

    intel_devices: list[dict[str, Any]] = []
    for d in extract.get("devices") or []:
        sym = d.get("symbol", "")
        legend_row_full = find_legend_row_for_symbol(
            raw_symbol=sym, project_reference=project_reference
        )
        legend_row_clean: dict[str, Any] = {}
        legend_section = None
        if legend_row_full:
            legend_section = legend_row_full.get("_section_title")
            legend_row_clean = {
                k: v for k, v in legend_row_full.items() if not k.startswith("_")
            }
        device_class = d.get("normalized_class") or ""
        schedule_refs = _schedule_cross_refs(
            device_class=device_class,
            schedule_rows=schedule_rows,
        )
        spec_refs = _spec_paragraph_refs(
            device_class=device_class,
            spec_paragraphs=spec_paragraphs,
        )
        icon_path = _icon_path_for_symbol(
            raw_symbol=sym, project_reference=project_reference,
        )
        cbreakdown = _confidence_breakdown(device=d, legend_hit=bool(legend_row_full))

        intel_devices.append({
            "id": d.get("id"),
            "symbol": sym,
            "normalized_class": d.get("normalized_class"),
            "system": d.get("system"),
            "intel": {
                "what_it_is": (
                    f"{d.get('normalized_class','?')} — {legend_row_clean.get('DESCRIPTION','?')}"
                ),
                "legend_section": legend_section,
                "legend_row": legend_row_clean,
                "icon_png": icon_path,
                "schedule_cross_refs": schedule_refs,
                "spec_paragraph_refs": spec_refs,
            },
            "location_context": {
                "bbox_pt": d.get("bbox_pt"),
                "room_guess": d.get("room_guess"),
                "home_run_to": d.get("home_run_to"),
                "keynote_ref": d.get("keynote"),
                "keynote_text": d.get("keynote_text"),
                "floor_label": d.get("floor_label"),
                "multiplier": d.get("multiplier"),
            },
            "review_flags": d.get("review_flags") or [],
            "confidence": cbreakdown,
        })

    # Self-verification stats.
    n_total = len(intel_devices)
    n_legend = sum(1 for e in intel_devices if e["intel"].get("legend_row"))
    n_complete = sum(
        1 for e in intel_devices
        if e["intel"].get("legend_row")
        and e["location_context"].get("home_run_to")
        and not e.get("review_flags")
    )
    avg_conf = (
        sum(e["confidence"]["composite"] for e in intel_devices) / n_total
        if n_total else 0.0
    )

    doc: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source_pdf": str(pdf_path),
        "page_index": page_index,
        "sheet": extract.get("sheet") or {},
        "project_knowledge_used": {
            "legend_rows_available": len(_iter_legend_rows(project_reference)),
            "schedule_rows_available": len(schedule_rows),
            "spec_paragraphs_available": len(spec_paragraphs),
            "expected_symbol_codes": plan["expected_symbol_codes"],
            "project_zones": _project_zones(project_reference, extract),
        },
        "detection_plan": {
            "expected_symbols": plan["expected_symbol_codes"],
            "page_strategy": "device_takeoff",  # by router; only floor-plan pages route here
        },
        "what_was_detected": {
            "devices_total": n_total,
            "by_class": dict(Counter(
                e.get("normalized_class") for e in intel_devices if e.get("normalized_class")
            )),
            "by_symbol": dict(Counter(
                e.get("symbol") for e in intel_devices if e.get("symbol")
            )),
            "keyed_notes_extracted": len(extract.get("keyed_notes") or []),
            "zone_notes_resolved": len(extract.get("zone_notes") or []),
        },
        "self_verification": {
            "legend_lookup_hit_rate": (n_legend / n_total) if n_total else 0.0,
            "fully_resolved_rate": (n_complete / n_total) if n_total else 0.0,
            "average_confidence": round(avg_conf, 3),
        },
        "keyed_notes": extract.get("keyed_notes") or [],
        "zone_notes": extract.get("zone_notes") or [],
        "devices": intel_devices,
        "schematic_region": extract.get("schematic_region"),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    return doc


def page_intelligence_to_readable(doc: dict[str, Any]) -> dict[str, Any]:
    """LLM-friendly view — strip bboxes, keep cross-references."""
    devs_out = []
    for d in doc.get("devices") or []:
        loc = d.get("location_context") or {}
        intel = d.get("intel") or {}
        devs_out.append({
            "symbol": d.get("symbol"),
            "class": d.get("normalized_class"),
            "system": d.get("system"),
            "what_it_is": intel.get("what_it_is"),
            "legend_section": intel.get("legend_section"),
            "icon": intel.get("icon_png"),
            "legend_row": intel.get("legend_row"),
            "schedule_cross_refs": intel.get("schedule_cross_refs"),
            "spec_paragraph_refs": intel.get("spec_paragraph_refs"),
            "room": loc.get("room_guess"),
            "routes_to": loc.get("home_run_to"),
            "keynote": loc.get("keynote_ref"),
            "keynote_text": loc.get("keynote_text"),
            "multiplier": loc.get("multiplier"),
            "review_flags": d.get("review_flags") or [],
            "confidence": (d.get("confidence") or {}).get("composite"),
        })
    return {
        "schema_version": READABLE_SCHEMA_VERSION,
        "source": Path(doc.get("source_pdf", "")).name,
        "page_index": doc.get("page_index"),
        "sheet": doc.get("sheet") or {},
        "project_knowledge_used": doc.get("project_knowledge_used") or {},
        "detection_plan": doc.get("detection_plan") or {},
        "what_was_detected": doc.get("what_was_detected") or {},
        "self_verification": doc.get("self_verification") or {},
        "keyed_notes": doc.get("keyed_notes") or [],
        "zone_notes": doc.get("zone_notes") or [],
        "devices": devs_out,
    }


def page_intelligence_to_markdown(doc: dict[str, Any]) -> str:
    """Human-readable markdown report — the operator's view."""
    sheet = doc.get("sheet") or {}
    src = Path(doc.get("source_pdf", "")).name
    kn = doc.get("keyed_notes") or []
    zones = doc.get("zone_notes") or []
    pk = doc.get("project_knowledge_used") or {}
    wd = doc.get("what_was_detected") or {}
    sv = doc.get("self_verification") or {}

    L = []
    L.append(f"# Page {doc.get('page_index')} Intelligence Report — {sheet.get('sheet_number')} {sheet.get('sheet_name')}")
    L.append("")
    L.append(f"- **source**: `{src}`")
    L.append(f"- **schema**: `{doc.get('schema_version')}`")
    L.append(f"- **page_type**: `{sheet.get('page_type')}`   |   in_scope: `{sheet.get('in_scope')}`   |   multiplier: `{sheet.get('multiplier')}`")
    L.append("")

    L.append("## What the parser KNEW before parsing this page")
    L.append("")
    L.append(f"- Legend (from page 1): **{pk.get('legend_rows_available', 0)}** symbol rows defined")
    L.append(f"- Schedule (from page 2): **{pk.get('schedule_rows_available', 0)}** component rows cataloged")
    L.append(f"- Spec (from page 0): **{pk.get('spec_paragraphs_available', 0)}** paragraphs captured")
    L.append(f"- Expected symbol codes: `{', '.join(pk.get('expected_symbol_codes') or [])}`")
    L.append(f"- Known project zones: `{', '.join(pk.get('project_zones') or [])}`")
    L.append("")

    L.append("## What the parser DETECTED on this page")
    L.append("")
    L.append(f"- **{wd.get('devices_total', 0)}** devices total")
    by_class = wd.get('by_class') or {}
    if by_class:
        L.append("- By device class:")
        for cls, n in sorted(by_class.items(), key=lambda kv: (-kv[1], kv[0])):
            L.append(f"   - `{cls}`: {n}")
    L.append(f"- **{wd.get('keyed_notes_extracted', 0)}** keyed-note entries extracted")
    L.append(f"- **{wd.get('zone_notes_resolved', 0)}** zone-routing notes resolved")
    L.append("")

    L.append("## Self-verification")
    L.append("")
    L.append(f"- Legend-lookup hit rate: **{sv.get('legend_lookup_hit_rate', 0):.0%}** ({int(sv.get('legend_lookup_hit_rate', 0) * (wd.get('devices_total') or 0))} of {wd.get('devices_total', 0)})")
    L.append(f"- Fully resolved (legend + zone + no flags): **{sv.get('fully_resolved_rate', 0):.0%}**")
    L.append(f"- Average composite confidence: **{sv.get('average_confidence', 0):.2f}**")
    L.append("")

    if kn:
        L.append("## Keyed notes (extracted from this page)")
        L.append("")
        for entry in kn:
            L.append(f"- **{entry.get('number')}.** {entry.get('text')}")
        L.append("")

    if zones:
        L.append("## Zone routing notes")
        L.append("")
        for z in zones:
            tgt = z.get("target") or "?"
            raw = z.get("raw_text") or ""
            L.append(f"- → **{tgt}**: {raw}")
        L.append("")

    L.append("## Per-device intelligence")
    L.append("")
    seen_devices_by_class: dict[str, int] = {}
    for d in doc.get("devices") or []:
        intel = d.get("intel") or {}
        loc = d.get("location_context") or {}
        cls = d.get("normalized_class") or "?"
        seen_devices_by_class[cls] = seen_devices_by_class.get(cls, 0) + 1
        idx = seen_devices_by_class[cls]
        L.append(f"### {idx}. {d.get('symbol')} ({cls})")
        L.append("")
        if intel.get("icon_png"):
            L.append(f"- **Icon**: `legend_icons/{intel.get('icon_png')}`")
        if intel.get("legend_section"):
            L.append(f"- **Legend section**: {intel.get('legend_section')}")
        L.append(f"- **What it is**: {intel.get('what_it_is')}")
        if loc.get("room_guess"):
            L.append(f"- **Room**: {loc.get('room_guess')}")
        if loc.get("routes_to") or loc.get("home_run_to"):
            L.append(f"- **Routes to**: {loc.get('home_run_to') or loc.get('routes_to')}")
        if loc.get("keynote_ref"):
            kt = loc.get("keynote_text")
            L.append(f"- **Keynote ref {loc.get('keynote_ref')}**: {kt or '(no text resolved)'}")
        # Legend row fields.
        legend_row = intel.get("legend_row") or {}
        if legend_row:
            L.append(f"- **Legend declaration (from page 1)**:")
            interesting_keys = ("DESCRIPTION", "CABLE COUNT", "CABLE DESCRIPTION",
                                "WORK AREA TERMINATION", "CLOSET TERMINATION",
                                "STANDARD MOUNTING HEIGHT (AFF)", "ELECTRICAL ROUGH-IN",
                                "POWER REQUIREMENT", "REMARKS")
            for key in interesting_keys:
                if key in legend_row and legend_row[key]:
                    L.append(f"    - {key}: {legend_row[key]}")
        # Schedule cross-references.
        sched = intel.get("schedule_cross_refs") or []
        if sched:
            L.append(f"- **Schedule cross-refs (from page 2)**: {len(sched)} matching row(s)")
            for s in sched[:2]:
                row = s.get("row") or {}
                desc = row.get("DESCRIPTION") or list(row.values())[0] if row else ""
                L.append(f"    - {s.get('section')}: {str(desc)[:100]}")
        # Spec paragraph references.
        spec_refs = intel.get("spec_paragraph_refs") or []
        if spec_refs:
            L.append(f"- **Spec references (from page 0)**: {len(spec_refs)} matching paragraph(s)")
            for sp in spec_refs[:1]:
                L.append(f"    - ({sp.get('heading') or '?'}): {sp.get('text','')[:140]}")
        # Confidence.
        conf = d.get("confidence") or {}
        L.append(f"- **Confidence**: `{conf.get('composite', 0):.2f}` (legend={conf.get('legend_lookup',0):.2f}, keynote={conf.get('keynote_resolution',0):.2f}, route={conf.get('home_run_resolution',0):.2f}, room={conf.get('room_label',0):.2f})")
        if d.get("review_flags"):
            L.append(f"- **Review flags**: `{', '.join(d['review_flags'])}`")
        L.append("")

    return "\n".join(L).rstrip() + "\n"


def write_page_intelligence(
    *,
    pdf_path: Path,
    page_index: int,
    out_dir: Path,
    project_reference: dict[str, Any],
    filename_stem: str | None = None,
) -> dict[str, Path]:
    """Build and write the intelligence report for one page."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = build_page_intelligence(
        pdf_path=pdf_path,
        page_index=page_index,
        project_reference=project_reference,
    )
    stem = filename_stem or f"page_{page_index:02d}_intelligence"
    full = out_dir / f"{stem}.json"
    readable = out_dir / f"{stem}.readable.json"
    md = out_dir / f"{stem}.md"
    full.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    readable.write_text(
        json.dumps(page_intelligence_to_readable(doc), indent=2),
        encoding="utf-8",
    )
    md.write_text(page_intelligence_to_markdown(doc), encoding="utf-8")
    return {"full": full, "readable": readable, "markdown": md}


__all__ = [
    "SCHEMA_VERSION",
    "READABLE_SCHEMA_VERSION",
    "build_detection_plan",
    "build_page_intelligence",
    "page_intelligence_to_markdown",
    "page_intelligence_to_readable",
    "write_page_intelligence",
]
