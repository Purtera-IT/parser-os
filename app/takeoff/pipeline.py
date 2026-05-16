"""End-to-end takeoff pipeline: PDF path -> :class:`TakeoffDocument`.

This module is the single public entry-point for the takeoff layer:

    from app.takeoff.pipeline import build_low_voltage_takeoff
    takeoff = build_low_voltage_takeoff(pdf_path)

It opens the PDF once with PyMuPDF, walks every page, builds sheet
records, runs the WN/POS/TV/etc. detector on plan pages, fuses
candidates into devices, parses zone notes, applies floor multipliers,
and finally unitizes everything into quote lines.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from app.takeoff.candidate_fusion import fuse_candidates_to_devices
from app.takeoff.corrections import apply_corrections_if_present
from app.takeoff.exports import takeoff_summary
from app.takeoff.legend_extractor import load_default_legend_rules
from app.takeoff.multipliers import floor_label_for_title, multiplier_for_title
from app.takeoff.plan_regions import default_excluded_regions, default_plan_viewport
from app.takeoff.schemas import (
    DeviceInstance,
    LegendRule,
    SheetRecord,
    SymbolCandidate,
    TakeoffDocument,
)
from app.takeoff.sheet_classifier import classify_sheet
from app.takeoff.symbol_candidates import detect_symbol_candidates
from app.takeoff.zones import collect_zone_warnings, parse_zones


def build_low_voltage_takeoff(pdf_path: Path) -> TakeoffDocument:
    """Build a :class:`TakeoffDocument` from a PDF on disk.

    Deterministic — no network, no OCR, no LLM. Opens the PDF once and
    closes it before returning. Errors propagate to the caller; the
    OrbitBriefPdfParser integration wraps this in try/except so a
    takeoff failure can never fail the full parse.
    """
    try:
        import fitz  # PyMuPDF
    except Exception as exc:  # pragma: no cover - env-specific
        raise RuntimeError("PyMuPDF (fitz) is required for the takeoff pipeline") from exc

    pdf_path = Path(pdf_path)
    legend_rules: list[LegendRule] = load_default_legend_rules()
    sheets: list[SheetRecord] = []
    candidates: list[SymbolCandidate] = []
    devices: list[DeviceInstance] = []
    warnings: list[str] = []
    open_questions: list[str] = []

    # Per-class candidate rejection tallies — used for the summary.
    candidates_by_class: dict[str, dict[str, int]] = defaultdict(
        lambda: {"not_in_scope": 0, "non_plan_page": 0, "outside_viewport": 0}
    )

    with fitz.open(str(pdf_path)) as doc:
        legend_source_page = _find_legend_page_index(doc)
        if legend_source_page is not None:
            for rule in legend_rules:
                rule.source_page = legend_source_page
                rule.confidence = max(rule.confidence, 0.92)
        for page_index in range(len(doc)):
            page = doc[page_index]
            page_text = ""
            try:
                page_text = page.get_text("text") or ""
            except Exception:
                page_text = ""
            sheet = classify_sheet(page_index, page_text)
            if sheet.sheet_name:
                levels, mult = multiplier_for_title(sheet.sheet_name)
                sheet.levels_represented = levels
                sheet.multiplier = mult
                sheet.floor_label = floor_label_for_title(sheet.sheet_name)
            sheet.plan_viewport = default_plan_viewport(page)
            sheet.excluded_regions = default_excluded_regions(page)
            sheets.append(sheet)

            # Detect candidates on device-bearing pages (and emit rejected
            # candidates on legend / detail pages so the audit trail is
            # complete).
            page_candidates = detect_symbol_candidates(
                page=page, sheet=sheet, legend_rules=legend_rules
            )
            candidates.extend(page_candidates)

            # Tally rejected candidates by class for the summary.
            for cand in page_candidates:
                if cand.rejection_reason is None or cand.normalized_class is None:
                    continue
                bucket = candidates_by_class[cand.normalized_class]
                reason = cand.rejection_reason
                if reason.startswith("sheet ") and "not in scope" in reason:
                    bucket["not_in_scope"] += 1
                elif reason.startswith("page_type="):
                    bucket["non_plan_page"] += 1
                elif "outside plan_viewport" in reason or "excluded_region" in reason:
                    bucket["outside_viewport"] += 1

            # Zones + fusion only run on device-bearing in-scope pages.
            if sheet.page_type in {"floor_plan", "typical_plan"} and sheet.in_scope:
                zones = parse_zones(page_text)
                sheet_devices = fuse_candidates_to_devices(
                    candidates=page_candidates,
                    sheet=sheet,
                    zones=zones,
                    legend_rules=legend_rules,
                )
                devices.extend(sheet_devices)

                # Per-sheet warnings (missing-zone, OCR-typo).
                sheet_warnings = collect_zone_warnings(
                    sheet_number=sheet.sheet_number,
                    sheet_name=sheet.sheet_name,
                    sheet_levels=sheet.levels_represented,
                    zones=zones,
                )
                warnings.extend(sheet_warnings)

                # Promote ambiguous-zone device flags to open questions.
                if any("ambiguous_homerun_zone" in d.review_flags for d in sheet_devices):
                    open_questions.append(
                        f"ambiguous home-run zone on {sheet.sheet_number or 'unknown sheet'}: "
                        f"multiple zone notes and device level cannot be resolved"
                    )

    # Apply (optional) human corrections file if present alongside the PDF.
    candidates, devices, applied_corrections = apply_corrections_if_present(
        pdf_path=pdf_path,
        candidates=candidates,
        devices=devices,
        sheets=sheets,
        legend_rules=legend_rules,
    )
    if applied_corrections:
        warnings.append(
            f"corrections_applied: {len(applied_corrections)} entries"
        )

    # Unitize.
    from app.takeoff.quote_unitizer import quote_lines_for_devices

    quote_lines = quote_lines_for_devices(devices)
    summary = takeoff_summary(sheets, devices, candidates_by_class=candidates_by_class)

    # Promote per-class zone-coverage open questions for downstream review.
    for w in warnings:
        if w.startswith("missing_homerun_zone_for_levels"):
            open_questions.append(w)

    return TakeoffDocument(
        source_pdf=pdf_path.name,
        sheets=sheets,
        legend_rules=legend_rules,
        candidates=candidates,
        devices=devices,
        quote_lines=quote_lines,
        warnings=warnings,
        open_questions=open_questions,
        summary=summary,
    )


def _find_legend_page_index(doc) -> int | None:
    """Best-effort scan for a 'SYMBOLS & LEGENDS' sheet (typically T0.01)."""
    for i in range(len(doc)):
        try:
            text = (doc[i].get_text("text") or "").upper()
        except Exception:
            continue
        if "SYMBOLS & LEGENDS" in text or "SYMBOLS AND LEGENDS" in text:
            return i
    return None


__all__ = ["build_low_voltage_takeoff"]
