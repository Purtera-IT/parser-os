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

import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from app.takeoff.candidate_fusion import fuse_candidates_to_devices
from app.takeoff.corrections import apply_corrections_if_present
from app.takeoff.exports import takeoff_summary
from app.takeoff.keynotes import KeynoteTable, parse_keynote_table_spatial
from app.takeoff.legend_discovery import discover_legend_rules
from app.takeoff.legend_extract import extract_legend as _extract_legend_doc
from app.takeoff.legend_extractor import load_default_legend_rules, rules_by_symbol
from app.takeoff.legend_self_extractor import (
    extract_legend_from_page,
    merge_with_defaults,
)
from app.takeoff.multipliers import floor_label_for_title, multiplier_for_title
from app.takeoff.nearby_text import collect_nearby_text, collect_room_labels
from app.takeoff.pdf_native import extract_page_words
from app.takeoff.plan_regions import default_excluded_regions, default_plan_viewport
from app.takeoff.schemas import (
    DeviceInstance,
    LegendRule,
    SheetRecord,
    SymbolCandidate,
    TakeoffDocument,
)
from app.takeoff.ocr_signals import OCREngineHandle, ocr_candidates_for_page
from app.takeoff.shape_signals import (
    ShapeTemplate,
    extract_shape_only_templates_from_legend_doc,
    extract_templates_from_legend,
    shape_candidates_for_page,
)
from app.takeoff.sheet_classifier import classify_sheet
from app.takeoff.spatial_zones import ZoneRegion, build_zone_regions
from app.takeoff.symbol_candidates import detect_symbol_candidates
from app.takeoff.typical_plan_expander import (
    TypicalPlanReport,
    build_expansion_summary,
    count_room_types_on_floor,
    expand_typical_plan,
)
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
    # ``legend_rules`` is populated by walking the project's OWN legend
    # page(s) below — no YAML defaults are consulted. The empty list
    # here is the universal starting point: if the PDF has no legend
    # page, the parser detects nothing rather than detecting the wrong
    # things from a stale hardcoded whitelist.
    legend_rules: list[LegendRule] = []
    sheets: list[SheetRecord] = []
    candidates: list[SymbolCandidate] = []
    devices: list[DeviceInstance] = []
    warnings: list[str] = []
    open_questions: list[str] = []
    typical_reports: list[TypicalPlanReport] = []
    # Buffer (sheet, page) for floor_plan-type sheets — used to scan
    # for typical-room labels in the second pass while the PDF is
    # still open.
    floor_plan_pages: list[tuple[SheetRecord, Any]] = []
    typical_expansion: dict[str, Any] = {}

    # Per-class candidate rejection tallies — used for the summary.
    candidates_by_class: dict[str, dict[str, int]] = defaultdict(
        lambda: {"not_in_scope": 0, "non_plan_page": 0, "outside_viewport": 0}
    )

    shape_pass_enabled = _env_flag_enabled("PARSER_OS_ENABLE_SHAPE_SIGNALS")
    shape_pass_results: dict[int, list[SymbolCandidate]] = {}
    shape_pass_warnings: list[str] = []

    ocr_pass_enabled = _env_flag_enabled("PARSER_OS_ENABLE_OCR_SIGNALS")
    ocr_pass_results: dict[int, list[SymbolCandidate]] = {}
    ocr_pass_warnings: list[str] = []
    ocr_engine_handle: OCREngineHandle | None = None
    if ocr_pass_enabled:
        from app.takeoff.ocr_signals import _load_engine
        ocr_engine_handle = _load_engine()
        if ocr_engine_handle.name == "none":
            ocr_pass_enabled = False
            ocr_pass_warnings.append(
                "ocr_signals_skipped_no_engine: neither easyocr nor "
                "pytesseract is installed"
            )

    with fitz.open(str(pdf_path)) as doc:
        legend_source_page = _find_legend_page_index(doc)

        # ─── Universal rule discovery from THIS project's legend ───
        #
        # The pipeline used to load a 7-symbol YAML whitelist (WN, CR,
        # TV, POS-T, POS-P, DA, H) and only trust device codes that
        # appeared in that list. That was the OPPOSITE of universal —
        # projects with cameras, motion detectors, fire alarm panels,
        # intercoms, keypads, etc. all fell through.
        #
        # Now the project's OWN legend page is the source of truth. We
        # extract a structured legend doc once (cells with explicit
        # bboxes + column headers), walk every row of every section,
        # and emit one LegendRule per row — with raw_symbol = text
        # token when present, or a stable synthetic __shp_<hash> when
        # the cell is text-less (cameras etc.). The normalized_class,
        # system, cable info, mounting, power all come straight from
        # the legend's columns. No YAML, no keyword tables.
        legend_doc: dict[str, Any] | None = None
        if legend_source_page is not None:
            try:
                legend_doc = _extract_legend_doc(
                    pdf_path=pdf_path,
                    page_index=legend_source_page,
                )
            except Exception as exc:  # pragma: no cover - env-specific
                warnings.append(f"legend_extract_failed: {exc!r}")
                legend_doc = None
            if isinstance(legend_doc, dict) and legend_doc.get("tables"):
                try:
                    legend_rules = discover_legend_rules(legend_doc=legend_doc)
                except Exception as exc:  # pragma: no cover - env-specific
                    warnings.append(f"legend_discovery_failed: {exc!r}")
                    legend_rules = []
                if legend_rules:
                    text_coded = sum(
                        1 for r in legend_rules if not r.raw_symbol.startswith("__shp_")
                    )
                    shape_only = len(legend_rules) - text_coded
                    warnings.append(
                        "legend_discovery: "
                        f"{len(legend_rules)} rule(s) discovered from page "
                        f"{legend_source_page} ({text_coded} text-coded, "
                        f"{shape_only} shape-only)"
                    )
        if not legend_rules:
            warnings.append(
                "legend_discovery_empty: no rules discovered from PDF legend "
                "— the parser will detect nothing. Verify a legend page exists "
                "and the extractor classified it correctly."
            )

        # Phase B: shape-template extraction. Templates are cropped from
        # the legend page for BOTH text-coded rules (anchored on the
        # text token, via the original extractor) AND shape-only
        # synthetic rules (anchored on the rule's source_bbox).
        shape_templates: list[ShapeTemplate] = []
        if shape_pass_enabled and legend_source_page is not None and legend_rules:
            try:
                shape_templates = extract_templates_from_legend(
                    doc[legend_source_page], legend_rules
                )
            except Exception as exc:  # pragma: no cover - env-specific
                shape_pass_warnings.append(
                    f"shape_signals_template_extraction_failed: {exc}"
                )
            if isinstance(legend_doc, dict) and legend_doc.get("tables"):
                try:
                    _so_templates, _ = extract_shape_only_templates_from_legend_doc(
                        pdf_path=pdf_path,
                        legend_doc=legend_doc,
                        rules=legend_rules,
                    )
                    if _so_templates:
                        shape_templates.extend(_so_templates)
                        warnings.append(
                            f"shape_only_templates_cropped: {len(_so_templates)} "
                            "textless legend row(s) ready for template-match"
                        )
                except Exception as exc:  # pragma: no cover - env-specific
                    shape_pass_warnings.append(
                        f"shape_signals_textless_extraction_failed: {exc}"
                    )
            if not shape_templates:
                shape_pass_warnings.append(
                    "shape_signals_no_templates_extracted: legend page "
                    "found but no usable icon templates"
                )
        elif shape_pass_enabled and legend_source_page is None:
            shape_pass_warnings.append(
                "shape_signals_no_legend_page: cannot extract templates"
            )
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

            # Pre-compute page words + keynote table once. The keynote
            # table is per-page (each plan has its own) and the words
            # feed both nearby-text capture and keynote-ref lookups.
            # The spatial parser uses bbox-pairing — pairs numbers with
            # descriptions by geometry instead of sequence order, which
            # handles Marriott's column-split keynote table correctly.
            page_words = extract_page_words(page)
            keynote_table = parse_keynote_table_spatial(
                page_index=page_index,
                page_text=page_text,
                page_words=page_words,
            )

            # Detect candidates on device-bearing pages (and emit rejected
            # candidates on legend / detail pages so the audit trail is
            # complete).
            page_candidates = detect_symbol_candidates(
                page=page, sheet=sheet, legend_rules=legend_rules
            )

            # Populate nearby_text on each accepted candidate — this is
            # how we capture room labels like "EXISTING MDF ROOM" that
            # live a few points to the side of the device symbol.
            for cand in page_candidates:
                if cand.rejection_reason is not None:
                    continue
                cand.nearby_text = collect_nearby_text(
                    bbox=cand.bbox,
                    page_words=page_words,
                    own_symbol=cand.raw_symbol,
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
                # Phase C: derive spatial regions for SINGLE-LEVEL
                # multi-zone sheets so fusion can pick a zone based on
                # device position instead of falling through to
                # ``ambiguous_homerun_zone``.
                #
                # Multi-LEVEL multi-zone sheets (T1.06 / T1.10 — each
                # zone covers a different level) intentionally remain
                # ambiguous: a single WN drawn on a typical-floor sheet
                # represents many physical WNs (one per floor) which
                # may route to DIFFERENT IDFs. Picking one IDF based
                # on its spatial position would be wrong.
                zone_regions: list[ZoneRegion] = []
                if len(zones) > 1:
                    all_zones_share_level = all(
                        z.applies_to_all_levels for z in zones
                    )
                    if all_zones_share_level:
                        zone_regions = build_zone_regions(page=page, zones=zones)

                # Phase B: run the shape-template pass alongside the
                # text candidate detection — only if templates were
                # extracted and the env flag opted in.
                sheet_shape_cands: list[SymbolCandidate] = []
                if shape_templates:
                    sheet_shape_cands = shape_candidates_for_page(
                        page=page,
                        sheet=sheet,
                        templates=shape_templates,
                        rules_by_symbol=rules_by_symbol(legend_rules),
                    )
                    # Carry rejected shape candidates over to the doc-
                    # level audit list so reviewers can see the false
                    # positives the pass produced. Accepted shape-only
                    # ones are added below too so they're visible.
                    candidates.extend(sheet_shape_cands)
                    shape_pass_results[sheet.page_index] = sheet_shape_cands

                # Phase D: optional OCR pass — only when env flag opted
                # in AND an engine was successfully loaded above.
                if ocr_pass_enabled and ocr_engine_handle is not None:
                    ocr_cands, ocr_reason = ocr_candidates_for_page(
                        page=page,
                        sheet=sheet,
                        legend_rules=legend_rules,
                        engine=ocr_engine_handle,
                    )
                    if ocr_reason:
                        ocr_pass_warnings.append(ocr_reason)
                    if ocr_cands:
                        candidates.extend(ocr_cands)
                        ocr_pass_results[sheet.page_index] = ocr_cands
                        # OCR candidates currently go straight into
                        # candidates (audit trail). Fusion with text /
                        # shape is left for a follow-up commit — for
                        # now they're tracked as ocr-only needs_review
                        # markers.

                sheet_devices = fuse_candidates_to_devices(
                    candidates=page_candidates,
                    sheet=sheet,
                    zones=zones,
                    legend_rules=legend_rules,
                    shape_candidates=sheet_shape_cands,
                    zone_regions=zone_regions,
                    page_words=page_words,
                    keynote_table=keynote_table,
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

                # Phase A: hold floor-plan sheets so we can scan them
                # for typical-room labels after T4.xx has been parsed.
                if sheet.page_type == "floor_plan":
                    floor_plan_pages.append((sheet, page))

                # Phase A: typical-plan device-per-room dictionary.
                if sheet.page_type == "typical_plan":
                    report = expand_typical_plan(
                        page=page,
                        sheet=sheet,
                        candidates=page_candidates,
                    )
                    if report is not None and report.panels:
                        typical_reports.append(report)

        # Phase A: now that all typical-plan sheets have been parsed,
        # scan guest-room floor pages for native-text room labels
        # matching the room types discovered on the typical-plans
        # (K1, K2, QQ1, QQ2, …). Floors without any visible room
        # labels surface as ``unresolved_floors`` rather than silently
        # being ignored. Done inside the with-block because we need
        # the page objects.
        if typical_reports:
            known_room_types: set[str] = set()
            for r in typical_reports:
                known_room_types.update(p.room_type for p in r.panels)

            floor_room_counts: dict[str, dict[str, int]] = {}
            for f_sheet, f_page in floor_plan_pages:
                if f_sheet.sheet_number is None:
                    continue
                counts = count_room_types_on_floor(f_page, known_room_types)
                # Only record floors that actually look like guest-room
                # floors — i.e. at least one room label present. Other
                # floor_plan sheets (lobby, ballroom, …) are not in
                # scope for expansion.
                if any(v > 0 for v in counts.values()):
                    floor_room_counts[f_sheet.sheet_number] = counts
                elif _looks_like_guest_room_sheet(f_sheet):
                    # Sheet is in the typical-floor range but no labels
                    # were found — typically T1.10 / T1.11 / T1.12 on
                    # the Marriott set. Record the floor as unresolved.
                    floor_room_counts[f_sheet.sheet_number] = {
                        r: 0 for r in known_room_types
                    }

            typical_expansion = build_expansion_summary(
                typical_reports=typical_reports,
                floor_room_counts=floor_room_counts,
                sheet_records=sheets,
            )

            # Surface gaps explicitly — the operator should not be
            # left guessing why a "typical plan + 5 guest-room floors"
            # set produced zero expanded counts.
            unresolved = typical_expansion.get("unresolved_floors") or []
            if unresolved:
                open_questions.append(
                    "typical_plan_keycount_missing: no per-floor "
                    f"{'/'.join(sorted(known_room_types))} room labels "
                    f"parsed on {', '.join(unresolved)}; operator must "
                    "supply room counts before typical-plan expansion "
                    "can be applied"
                )
            elif not (typical_expansion.get("expanded_device_totals") or {}):
                warnings.append(
                    "typical_plan_expansion_empty: typical plan parsed but no "
                    "floors mapped to expansion"
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
    # Split candidates so the summary can compute
    # text_only_count / shape_only_count / cross_validated_count.
    #
    # A candidate produced by ``detect_symbol_candidates`` (native PDF
    # text) starts with source_methods=["pdf_native_text"] and gains
    # "shape_template" iff fusion matched it to a shape candidate.
    #
    # A candidate produced by ``shape_candidates_for_page`` starts
    # with source_methods=["shape_template"] and is upgraded by
    # fusion to also list "pdf_native_text" when it was matched.
    #
    # To avoid double-counting a cross-validated PAIR, we feed only the
    # native-text candidates into ``text_candidates`` and only the
    # shape candidates that did NOT get matched (i.e. shape-template
    # in methods but NOT pdf_native_text) into ``shape_candidates``.
    text_only_candidates = []
    shape_only_candidates = []
    for c in candidates:
        sm = set(c.source_methods or [])
        if "pdf_native_text" in sm and "shape_template" in sm:
            # Either origin — counts once in text_only_candidates as
            # a cross-validated text candidate.
            if c.id.startswith("cand"):  # candidate id prefix
                text_only_candidates.append(c)
        elif "pdf_native_text" in sm:
            text_only_candidates.append(c)
        elif "shape_template" in sm:
            shape_only_candidates.append(c)
    summary = takeoff_summary(
        sheets,
        devices,
        candidates_by_class=candidates_by_class,
        text_candidates=text_only_candidates,
        shape_candidates=shape_only_candidates,
    )
    if typical_expansion:
        summary["typical_plan_expansion"] = typical_expansion
    if shape_pass_enabled:
        summary["shape_signals"] = {
            "enabled": True,
            "templates_extracted": [t.raw_symbol for t in shape_templates] if shape_templates else [],
            "pages_scanned": len(shape_pass_results),
            "total_shape_candidates": sum(len(v) for v in shape_pass_results.values()),
        }
    if shape_pass_warnings:
        warnings.extend(shape_pass_warnings)
    if ocr_pass_enabled or ocr_pass_warnings:
        summary["ocr_signals"] = {
            "enabled": bool(ocr_pass_enabled),
            "engine": ocr_engine_handle.name if ocr_engine_handle else "none",
            "pages_scanned": len(ocr_pass_results),
            "total_ocr_candidates": sum(len(v) for v in ocr_pass_results.values()),
        }
    if ocr_pass_warnings:
        warnings.extend(ocr_pass_warnings)

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


def _env_flag_enabled(var_name: str) -> bool:
    """Truthy parse of an env var: 1/true/yes/on (case-insensitive)."""
    raw = os.environ.get(var_name, "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


_GUEST_ROOM_TITLE_HINTS = (
    "GUESTROOM",
    "GUEST ROOM",
    "LEVEL 19",
    "LEVEL 24",
    "LEVEL 25",
    "LEVEL 5",
    "LEVEL 17",
)


def _looks_like_guest_room_sheet(sheet: SheetRecord) -> bool:
    """Heuristic: does this floor_plan sheet look like a guest-room floor?

    Used to decide whether a floor_plan sheet *should* have produced
    typical-room labels but didn't, so it should be surfaced as
    ``unresolved`` instead of silently ignored. The heuristic checks for
    typical guest-room sheet titles or for a non-trivial sheet
    multiplier (``> 1``) — both are strong signals that the floor is
    one of several repeated typical levels.
    """
    if sheet.multiplier and sheet.multiplier > 1:
        return True
    name = (sheet.sheet_name or "").upper()
    return any(h in name for h in _GUEST_ROOM_TITLE_HINTS)


def _find_legend_page_index(doc) -> int | None:
    """Best-effort scan for a 'SYMBOLS & LEGENDS' sheet (typically T0.01).

    Scoring: a page that is classified as a ``legend`` sheet (via
    ``classify_sheet``) AND mentions 'SYMBOLS & LEGENDS' in its title
    wins outright. Otherwise fall back to the first page whose text
    contains the keyword. This prevents the SPEC page (T0.00) — which
    may reference 'SYMBOLS & LEGENDS' in a table of contents — from
    being chosen over the actual legend page (T0.01).
    """
    title_candidate: int | None = None
    text_candidate: int | None = None
    for i in range(len(doc)):
        try:
            text = (doc[i].get_text("text") or "")
        except Exception:
            continue
        text_upper = text.upper()
        if "SYMBOLS & LEGENDS" not in text_upper and "SYMBOLS AND LEGENDS" not in text_upper:
            continue
        if text_candidate is None:
            text_candidate = i
        sheet = classify_sheet(i, text)
        if sheet.page_type == "legend":
            title_candidate = i
            break
    return title_candidate if title_candidate is not None else text_candidate


__all__ = ["build_low_voltage_takeoff"]
