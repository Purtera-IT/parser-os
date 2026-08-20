"""Legend-first schematic pre-pass.

Fifteen percent of the original module, serving one document kind most deals never contain, and coupled to almost nothing: measured, it needed one symbol from the rest of the module and one symbol depended on it. Schematics are their own problem -- a legend page is the answer key for its own drawing set, so grounding is per-document nearest-legend matching rather than global classification, and none of that reasoning belongs beside paragraph and table extraction.
"""

from __future__ import annotations

from app.core.schemas import EvidenceAtom
from app.domain.schemas import DomainPack
from pathlib import Path
from typing import Any

# The identical helper also lives in orbitbrief_pdf, but importing it from
# there would make this module depend on the one that imports it. This is the
# canonical definition and the two are the same expression; the call sites
# here pass a Path, which is what it expects.
from app.parsers.structured_projection import derived_dir_for
import os
import re


# Filter list for "orphan token" harvesting — common column-header
# words and English filler that shouldn't be treated as symbols.
_LEGEND_TOKEN_BLOCKLIST: frozenset[str] = frozenset({
    "ABOVE", "AFF", "ARCH", "BACK", "CABLE", "CAT6", "CEILING",
    "CLOSET", "CMP", "COAX", "COMPONENT", "COMPONENTS", "CONDUIT",
    "CONTROL", "COOPER", "COPPER", "COUNT", "COUNTER", "COVER",
    "DESCRIPTION", "DEVICE", "DOCK", "DOOR", "DRAWING", "DRAWINGS",
    "ELECTRICAL", "ENTRY", "EQUIP", "ETC", "FINISH", "FLUSH",
    "FRAME", "FROM", "GROUP", "HARDWARE", "HEIGHT", "INSERT",
    "INSTALLATION", "JACK", "LIST", "LOAD", "LOWER", "MANUFACTURERS",
    "MOUNT", "MOUNTED", "MOUNTING", "MUD", "NIC", "NOT", "NOTE",
    "NOTES", "N/A", "NA", "NORMAL", "NUMBER", "OUTLET", "OWNER",
    "PANEL", "PART", "PATCH", "PER", "PLANS", "POE", "PORT",
    "POWER", "PROVIDE", "READER", "REFER", "REMARKS", "REQUIREMENT",
    "REQUIREMENTS", "RING", "RISER", "ROOM", "ROOMS", "ROUGH",
    "ROUGH-IN", "SCHEDULE", "SECONDARY", "SECURITY", "SEE",
    "SHIELDED", "SHOWN", "SIZE", "SPACE", "STANDARD", "STRANDED",
    "STUB", "SUITE", "SYMBOL", "SYMBOLS", "SYSTEM", "TERMINATION",
    "TYPE", "TYPES", "TYPICAL", "TYPICALLY", "UNDER", "UNLESS",
    "UPS", "USE", "USED", "VAULT", "VERIFY", "WALL", "WAREHOUSE",
    "WIRE", "WITH", "WORK", "ZONE",
    "AND", "OR", "FOR", "THE", "ARE", "WAS", "WERE", "ALL", "ANY",
    "PER", "VERIFY", "TBD",
    "A", "B", "C", "D", "E", "F", "G",
    # ----- column letters used as grid coordinates -----
    "A#", "A #",
})

def _augment_legend_with_orphan_tokens(
    *,
    legend: Any,
    per_page_legend_bbox: dict[int, tuple[float, float, float, float]],
    per_page_blocks: dict[int, list[Any]],
) -> Any:
    """Harvest short uppercase tokens from a legend's bbox region.

    The row-parser pairs blocks into (symbol, description) rows but
    occasionally misses the symbol token (multi-column legends with
    wide gaps, columns of nothing-but-icon swatches, etc.). For each
    legend, scan its bbox for short standalone uppercase tokens that
    don't already appear as ``normalized_symbol_text`` in the legend
    entries, and append a synthetic ParsedLegendEntry for each.

    Filtered by ``_LEGEND_TOKEN_BLOCKLIST`` to keep English filler /
    column-header words out of the symbol vocabulary.
    """
    from app.parsers.schematic_models import ParsedLegend, ParsedLegendEntry
    import re

    legend_bbox = per_page_legend_bbox.get(legend.page_index)
    if legend_bbox is None:
        return legend
    blocks = per_page_blocks.get(legend.page_index) or []
    if not blocks:
        return legend

    have: set[str] = set()
    for e in legend.entries:
        s = (e.normalized_symbol_text or "").strip().upper()
        if s:
            have.add(s)
    new_entries: list[ParsedLegendEntry] = list(legend.entries)
    seen_new: set[str] = set()
    # Pattern: short uppercase alphanum tokens, optionally with -, /, or digits
    pat = re.compile(r"^[A-Z][A-Z0-9/\-]{0,5}$")
    for b in blocks:
        bbox = getattr(b, "bbox", None)
        if not bbox or len(bbox) != 4:
            continue
        # Must lie inside the legend bbox
        if not (
            legend_bbox[0] <= bbox[0]
            and bbox[2] <= legend_bbox[2]
            and legend_bbox[1] <= bbox[1]
            and bbox[3] <= legend_bbox[3]
        ):
            continue
        text = (getattr(b, "text", "") or "").strip()
        if not text or len(text) > 6:
            continue
        upper = text.upper()
        if upper in have or upper in seen_new:
            continue
        if upper in _LEGEND_TOKEN_BLOCKLIST:
            continue
        if not pat.match(upper):
            continue
        # Looks like a real legend symbol — synthesize an entry.
        try:
            entry = ParsedLegendEntry.make(
                page_index=legend.page_index,
                label_text=upper,
                normalized_label=upper.lower(),
                raw_symbol_text=upper,
                normalized_symbol_text=upper,
                symbol_bbox_pdf=tuple(float(x) for x in bbox),
                confidence=0.6,
            )
        except (TypeError, ValueError):
            continue
        new_entries.append(entry)
        seen_new.add(upper)

    if not seen_new:
        return legend
    # Rebuild the ParsedLegend with the new entry tuple. Use make() so
    # legend_id rolls forward to reflect the new entry set.
    return ParsedLegend.make(
        page_index=legend.page_index,
        sheet_number=legend.sheet_number,
        title=legend.title,
        scope=legend.scope,
        entries=tuple(new_entries),
        continuation_refs=legend.continuation_refs,
        source_ref_locator=dict(legend.source_ref_locator),
        confidence=legend.confidence,
        warnings=legend.warnings,
    )

def _run_schematic_pre_pass(
    *,
    project_id: str,
    artifact_id: str,
    path: Path,
    parser_version: str,
    domain_pack: DomainPack | None,
) -> tuple[list[EvidenceAtom], list[dict[str, Any]]]:
    """Legend-first schematic pre-pass for a PDF (PR5).

    Returns ``(atoms, derived_files)``.  ``atoms`` is a deterministic
    list of ``schematic_*`` atoms; ``derived_files`` is a list of
    ``ParserDerivedFile`` dicts to attach to ``ParserOutput``.

    Behavior is conservative — if no legend is parsed anywhere in the
    document AND the active domain pack declares no detection
    targets, the pre-pass returns empty results so non-schematic PDFs
    are untouched (preserves the determinism + provenance contracts
    for the existing test grid).
    """
    try:
        import fitz  # type: ignore[import-not-found]
    except Exception:  # pragma: no cover
        return [], []
    from app.parsers.schematic_atom_emitters import (
        collect_all,
        emit_detection_atom,
        emit_keyed_note_atom,
        emit_legend_atom,
        emit_line_run_atom,
        emit_room_atom,
        emit_schedule_row_atom,
        emit_sheet_metadata_atom,
        emit_target_set_atom,
        emit_warning_atom,
        intersect_with_pack,
    )
    from app.parsers.schematic_models import DetectionTarget, DetectionTargetSet, SchematicWarning
    from orbitbrief_page_os.segmentation.schematic.legend_locator import (
        locate_legend_candidates,
        page_text_blocks,
    )
    from orbitbrief_page_os.segmentation.schematic.legend_parser import parse_legend
    from orbitbrief_page_os.segmentation.schematic.legend_resolver import (
        LegendResolver,
        extract_sheet_number,
    )
    from orbitbrief_page_os.segmentation.schematic.symbol_detector import detect_symbols
    from orbitbrief_page_os.segmentation.schematic.raster import is_text_poor_page
    from orbitbrief_page_os.segmentation.schematic import ocr as schematic_ocr
    from orbitbrief_page_os.segmentation.schematic.page_kind_classifier import (
        LEGEND_TABLE,
        SCHEDULE_BOM,
        SPEC_PROSE,
        SCHEMATIC_DRAWING,
        UNKNOWN as PAGE_UNKNOWN,
        classify_page_kind,
    )

    try:
        doc = fitz.open(str(path))
    except Exception:  # pragma: no cover
        return [], []

    resolver = LegendResolver()
    per_page_blocks: dict[int, list[Any]] = {}
    per_page_legend_bbox: dict[int, tuple[float, float, float, float]] = {}
    parsed_legends: list[Any] = []

    atoms: list[EvidenceAtom] = []
    legend_records: list[dict[str, Any]] = []
    target_records: list[dict[str, Any]] = []
    detection_records: list[dict[str, Any]] = []

    declared_emitted: set[tuple[str, str]] = set()
    legend_gap_emitted: set[tuple[str, str]] = set()
    pack_has_targets_for_warning = bool(domain_pack and domain_pack.detection_targets)
    try:
        for page_index in range(doc.page_count):
            try:
                page_obj = doc.load_page(page_index)
                blocks = page_text_blocks(page_obj)
            except Exception:
                blocks = []
                page_obj = None
            per_page_blocks[page_index] = blocks
            # Raster fallback: if the page has effectively no text layer
            # AND the active pack expects schematic content, try local
            # OCR to recover legend rows. When OCR is unavailable, emit
            # an ``ocr_unavailable`` warning so the page doesn't silently
            # parse as blank. When OCR IS available, convert recognized
            # words into TextBlocks in PDF-point space and feed them to
            # the rest of the legend pipeline.
            if (
                page_obj is not None
                and pack_has_targets_for_warning
                and not blocks
                and is_text_poor_page(page_obj)
            ):
                if not schematic_ocr.is_available():
                    atoms.append(
                        emit_warning_atom(
                            warning=schematic_ocr.status_warning(
                                page_index=page_index, sheet_number=None
                            ),
                            project_id=project_id,
                            artifact_id=artifact_id,
                            filename=path.name,
                            parser_version=parser_version,
                            page=page_obj,
                        )
                    )
                else:
                    from orbitbrief_page_os.segmentation.schematic.raster import (
                        render_page_to_ndarray,
                    )
                    from app.parsers.schematic_models import SCHEMATIC_REPLAY_DPI

                    arr = render_page_to_ndarray(page_obj, dpi=SCHEMATIC_REPLAY_DPI)
                    if arr is not None:
                        words = schematic_ocr.ocr_words(arr)
                        ocr_blocks = schematic_ocr.words_to_textblocks(
                            words, page_dpi=SCHEMATIC_REPLAY_DPI
                        )
                        if ocr_blocks:
                            blocks = ocr_blocks
                            per_page_blocks[page_index] = ocr_blocks
                            atoms.append(
                                emit_warning_atom(
                                    warning=SchematicWarning.make(
                                        warning_type="ocr_recovered",
                                        page_index=page_index,
                                        sheet_number=None,
                                        detail=(
                                            f"Raster page parsed via OCR "
                                            f"({len(ocr_blocks)} text rows recovered)."
                                        ),
                                        extras={
                                            "ocr_word_count": len(words),
                                            "ocr_block_count": len(ocr_blocks),
                                        },
                                    ),
                                    project_id=project_id,
                                    artifact_id=artifact_id,
                                    filename=path.name,
                                    parser_version=parser_version,
                                    page=page_obj,
                                )
                            )
            # ── Page-kind routing (PR: Marriott multi-legend fix) ──
            # Classify the page so we (a) skip prose/schedule pages
            # and (b) extract MULTIPLE legends from legend-table
            # pages instead of bailing on the first match.
            classification = classify_page_kind(
                page_index=page_index, page=page_obj, blocks=blocks
            )
            page_kind = classification.kind

            # SPEC_PROSE + SCHEDULE_BOM pages have no schematic content;
            # skip the entire legend/symbol flow. The generic PDF parser
            # (table/text extraction) handles these pages.
            if page_kind in (SPEC_PROSE, SCHEDULE_BOM):
                # Still ingest into resolver so cross-doc state is
                # consistent (it just produces no legends/targets).
                page_bbox_for_ingest_skip: tuple[float, float, float, float] | None = None
                if page_obj is not None:
                    try:
                        r = page_obj.rect
                        page_bbox_for_ingest_skip = (
                            float(r.x0), float(r.y0), float(r.x1), float(r.y1)
                        )
                    except Exception:  # pragma: no cover
                        page_bbox_for_ingest_skip = None
                resolver.ingest_page(
                    page_index=page_index,
                    blocks=blocks,
                    legend=None,
                    page_bbox=page_bbox_for_ingest_skip,
                )
                continue

            legend = None
            candidates = locate_legend_candidates(page_index=page_index, blocks=blocks)
            ordered = sorted(
                (c for c in candidates if c.score >= 0.45),
                key=lambda c: (-c.score, c.page_index, c.bbox[1], c.bbox[0]),
            )
            chosen_bbox: tuple[float, float, float, float] | None = None
            sheet = extract_sheet_number(blocks)

            # LEGEND_TABLE pages contain MULTIPLE legends (Marriott
            # T0.01 = Structured Cabling + Intrusion + Access Control
            # + CCTV). Extract every non-bogus candidate; promote
            # scope to ``global`` since the legend applies to all
            # subsequent drawing pages with the same domain.
            if page_kind == LEGEND_TABLE:
                page_legends: list[Any] = []
                seen_legend_ids: set[str] = set()
                seen_bbox_centers: list[tuple[float, float]] = []
                # Marriott T0.01 has FOUR legend tables (STRUCTURED
                # CABLING + INTRUSION DETECTION + ACCESS CONTROL +
                # CCTV) — the locator normalizes their headers to the
                # same string ("symbol legend"), so deduping by header
                # text used to collapse all four into one. Instead,
                # dedupe by the parsed legend_id (entry-set hash) and
                # by bbox-center proximity so distinct legends survive.
                BBOX_DUPE_PT = 36.0
                for cand in ordered:
                    cx = (cand.bbox[0] + cand.bbox[2]) / 2.0
                    cy = (cand.bbox[1] + cand.bbox[3]) / 2.0
                    if any(
                        abs(cx - sc[0]) <= BBOX_DUPE_PT and abs(cy - sc[1]) <= BBOX_DUPE_PT
                        for sc in seen_bbox_centers
                    ):
                        continue
                    parsed = parse_legend(
                        candidate=cand,
                        page_blocks=blocks,
                        sheet_number=sheet,
                        scope="global",
                    )
                    if parsed is None:
                        continue
                    if parsed.legend_id in seen_legend_ids:
                        continue
                    seen_legend_ids.add(parsed.legend_id)
                    seen_bbox_centers.append((cx, cy))
                    page_legends.append(parsed)
                    if chosen_bbox is None:
                        chosen_bbox = cand.bbox

                # Promote the in-loop ``legend`` to the first parsed
                # (for the ``if legend is not None`` block below); the
                # rest get appended directly to parsed_legends.
                if page_legends:
                    legend = page_legends[0]
                    parsed_legends.extend(page_legends[1:])
            else:
                # SCHEMATIC_DRAWING / COVER_TITLE / UNKNOWN — keep
                # current "first non-empty candidate wins" behavior.
                for cand in ordered:
                    scope = "global" if (cand.header_text and "symbols & legends" in cand.header_text) else "page"
                    legend = parse_legend(
                        candidate=cand,
                        page_blocks=blocks,
                        sheet_number=sheet,
                        scope=scope,  # type: ignore[arg-type]
                    )
                    if legend is not None:
                        chosen_bbox = cand.bbox
                        break
            page_bbox_for_ingest: tuple[float, float, float, float] | None = None
            if page_obj is not None:
                try:
                    r = page_obj.rect
                    page_bbox_for_ingest = (float(r.x0), float(r.y0), float(r.x1), float(r.y1))
                except Exception:  # pragma: no cover
                    page_bbox_for_ingest = None
            resolver.ingest_page(
                page_index=page_index,
                blocks=blocks,
                legend=legend,
                page_bbox=page_bbox_for_ingest,
            )
            if legend is not None:
                parsed_legends.append(legend)
                if chosen_bbox is not None:
                    per_page_legend_bbox[page_index] = chosen_bbox

        pack_has_targets = bool(domain_pack and domain_pack.detection_targets)
        if not parsed_legends and not pack_has_targets:
            return [], []

        # Vision-LLM symbol detection bootstrap. Extract legend symbol
        # crops once per document so they can be reused across every
        # SCHEMATIC_DRAWING page during the per-page detection loop.
        # Opt-in via PARSER_OS_VISION_DETECT=1 so default compiles stay
        # byte-stable for the existing test grid.
        vision_legend_crops: list[Any] = []
        vision_enabled = os.environ.get("PARSER_OS_VISION_DETECT") == "1"
        vision_cache_path: Path | None = None
        if vision_enabled and parsed_legends:
            try:
                from orbitbrief_page_os.segmentation.schematic.legend_symbol_crops import (
                    extract_legend_symbol_crops,
                )
                from orbitbrief_page_os.segmentation.schematic.vision_symbol_detector import (
                    is_vision_endpoint_reachable,
                )
            except Exception:  # pragma: no cover
                extract_legend_symbol_crops = None  # type: ignore[assignment]
                is_vision_endpoint_reachable = None  # type: ignore[assignment]
            if extract_legend_symbol_crops is not None and is_vision_endpoint_reachable is not None:
                if is_vision_endpoint_reachable():
                    crops_out_dir = derived_dir_for(path)
                    try:
                        crops_out_dir.mkdir(parents=True, exist_ok=True)
                    except OSError:  # pragma: no cover
                        pass
                    try:
                        vision_legend_crops = extract_legend_symbol_crops(
                            legends=parsed_legends,
                            pdf_path=path,
                            out_dir=crops_out_dir,
                        )
                    except Exception:  # pragma: no cover
                        vision_legend_crops = []
                    vision_cache_path = path.parent / ".orbitbrief_vision_detect_cache.jsonl"

        for legend in parsed_legends:
            try:
                legend_page = doc.load_page(legend.page_index)
            except Exception:  # pragma: no cover
                legend_page = None
            atoms.append(
                emit_legend_atom(
                    legend=legend,
                    project_id=project_id,
                    artifact_id=artifact_id,
                    filename=path.name,
                    parser_version=parser_version,
                    page=legend_page,
                )
            )
            legend_records.append(
                {
                    "legend_id": legend.legend_id,
                    "page": legend.page_index,
                    "sheet_number": legend.sheet_number,
                    "scope": legend.scope,
                    "entries": [
                        {
                            "entry_id": e.entry_id,
                            "symbol": e.raw_symbol_text,
                            "label": e.label_text,
                            "normalized_label": e.normalized_label,
                            "count_column": e.count_column,
                        }
                        for e in legend.entries
                    ],
                }
            )

        # Per-page resolution + target-set emission. Pages without a sheet
        # number AND without a parsed legend on them are skipped: this
        # is the discriminator that prevents non-drawing PDFs from being
        # spammed with ``missing_legend`` warnings.
        for page_index in sorted(per_page_blocks):
            blocks = per_page_blocks[page_index]
            sheet = extract_sheet_number(blocks)
            own_legend = any(l.page_index == page_index for l in parsed_legends)
            # The pack-with-targets case: even if a drawing-like page has
            # no extractable sheet number, the active domain pack
            # expects schematic context. Routing it through the resolver
            # surfaces a ``missing_legend`` warning instead of silently
            # dropping the page (boss-review fix).
            pack_expects_schematic = bool(domain_pack and domain_pack.detection_targets)
            page_text_density = sum(len((b.text or "").strip()) for b in blocks)
            # Image-only drawing detection: if the page has effectively no
            # text BUT the document has parsed legends from other pages
            # AND the active pack expects schematic content, we still want
            # to run the glyph-template matcher against the raster page so
            # symbol counts come back instead of vanishing silently.
            raster_only_page = (
                pack_expects_schematic
                and parsed_legends
                and not blocks
            )
            if sheet is None and not own_legend and not (
                pack_expects_schematic and page_text_density >= 40
            ) and not raster_only_page:
                continue
            try:
                page = doc.load_page(page_index)
            except Exception:  # pragma: no cover
                page = None
            resolved = resolver.resolve_for_page(page_index)
            for warning in resolved.warnings:
                atoms.append(
                    emit_warning_atom(
                        warning=warning,
                        project_id=project_id,
                        artifact_id=artifact_id,
                        filename=path.name,
                        parser_version=parser_version,
                        page=page,
                    )
                )
            if resolved.legend is None:
                continue
            if domain_pack is not None:
                targets, gaps = intersect_with_pack(
                    legend=resolved.legend, pack=domain_pack
                )
                pack_id_for_set = domain_pack.pack_id
            else:
                targets, gaps = [], []
                pack_id_for_set = "legend_only"

            # When the active pack doesn't intersect the legend (e.g.
            # fiber pack vs telecom legend), synthesize one target per
            # legend entry so the text-tag + vision detectors have
            # something to look for. This keeps the parser universal:
            # a real DD with WN/CR/TV symbols produces detections
            # regardless of which domain pack is loaded.
            if not targets:
                # Augment the legend with orphan symbol tokens.
                # Real legends have one column of short symbol tokens
                # (WN / CR / ZN / DC / FACP-2 / MATV / etc.) but the
                # row-parser occasionally fails to pair a token with
                # its description, so the resulting entry list omits
                # the symbol. Scan the legend bbox for standalone
                # uppercase tokens and synthesize entries for any
                # that aren't already represented.
                augmented_legend = _augment_legend_with_orphan_tokens(
                    legend=resolved.legend,
                    per_page_legend_bbox=per_page_legend_bbox,
                    per_page_blocks=per_page_blocks,
                )
                synthesized: list[DetectionTarget] = []
                for entry in augmented_legend.entries:
                    key_seed = (
                        entry.normalized_symbol_text
                        or entry.normalized_label
                        or entry.entry_id
                    )
                    if not key_seed:
                        continue
                    tk = key_seed.lower().strip()
                    ek = f"device:{tk}".replace(" ", "_")
                    try:
                        synthesized.append(
                            DetectionTarget(
                                target_key=tk,
                                entity_key=ek,
                                completeness="informational",
                                expected_modalities=("text_tag", "vision_llm"),
                                legend_entry_id=entry.entry_id,
                                aliases=tuple(
                                    a for a in (
                                        entry.raw_symbol_text or "",
                                        entry.normalized_symbol_text or "",
                                        entry.label_text or "",
                                        entry.normalized_label or "",
                                    ) if a
                                ),
                            )
                        )
                    except ValueError:
                        continue
                targets = synthesized
                pack_id_for_set = "legend_only"
                # Replace resolved.legend with the augmented copy so
                # downstream code (symbol detector, atom emitters)
                # see the harvested entries too.
                import dataclasses as _dc
                try:
                    resolved = _dc.replace(resolved, legend=augmented_legend)
                except (TypeError, ValueError):  # pragma: no cover
                    pass

            target_set = DetectionTargetSet.make(
                page_index=page_index,
                sheet_number=sheet,
                pack_id=pack_id_for_set,
                legend_id=resolved.legend.legend_id,
                targets=tuple(targets),
                legend_gap_target_keys=tuple(gaps),
            )
            page_bbox: tuple[float, float, float, float] | None = None
            if page is not None:
                try:
                    rect = page.rect
                    page_bbox = (float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))
                except Exception:  # pragma: no cover
                    page_bbox = None
            atoms.append(
                emit_target_set_atom(
                    target_set=target_set,
                    project_id=project_id,
                    artifact_id=artifact_id,
                    filename=path.name,
                    parser_version=parser_version,
                    page=page,
                    page_bbox=page_bbox,
                )
            )
            target_records.append(
                {
                    "page": page_index,
                    "sheet_number": sheet,
                    "legend_id": resolved.legend.legend_id,
                    "rationale": resolved.rationale,
                    "priority": resolved.priority,
                    "targets": [t.target_key for t in targets],
                    "legend_gap_target_keys": list(gaps),
                }
            )
            # legend_gap warnings: pack declared the target as
            # load-bearing but the resolved legend doesn't mention it.
            # Attach the legend's bbox so source_replay still verifies
            # the receipt against pixels (rather than emitting a
            # locator with only a page index).
            legend_bbox_for_gap = per_page_legend_bbox.get(resolved.legend.page_index)
            legend_page_for_gap = None
            try:
                legend_page_for_gap = doc.load_page(resolved.legend.page_index)
            except Exception:  # pragma: no cover
                pass
            for gap_key in gaps:
                # Dedupe: emit each (legend_id, target_key) gap once
                # regardless of how many drawing pages resolve to the
                # same legend.
                dedup = (resolved.legend.legend_id, gap_key)
                if dedup in legend_gap_emitted:
                    continue
                legend_gap_emitted.add(dedup)
                atoms.append(
                    emit_warning_atom(
                        warning=SchematicWarning.make(
                            warning_type="legend_gap",
                            page_index=resolved.legend.page_index,
                            sheet_number=resolved.legend.sheet_number,
                            detail=f"Pack '{domain_pack.pack_id}' declares load-bearing target '{gap_key}' but legend has no matching entry",
                            target_key=gap_key,
                            legend_id=resolved.legend.legend_id,
                            bbox_pdf=legend_bbox_for_gap,
                        ),
                        project_id=project_id,
                        artifact_id=artifact_id,
                        filename=path.name,
                        parser_version=parser_version,
                        page=legend_page_for_gap,
                    )
                )

            # Symbol detection (PR6) — run when we have a resolved
            # legend AND either:
            #   (a) a non-empty pack target set, OR
            #   (b) vision-LLM detection is enabled + crops exist
            #
            # The original guard only allowed pack-matched targets,
            # which silenced vision detection whenever the domain
            # pack didn't match the legend vocabulary (e.g. running
            # the fiber pack against a security/telecom legend). The
            # vision detector matches directly against legend
            # entries, so it can fire even when no pack target
            # intersects.
            vision_can_run = bool(vision_enabled and vision_legend_crops)
            if page is None or (not target_set.targets and not vision_can_run):
                continue
            try:
                legend_page = doc.load_page(resolved.legend.page_index)
            except Exception:  # pragma: no cover
                continue
            excluded: list[tuple[float, float, float, float]] = []
            if resolved.legend.page_index in per_page_legend_bbox:
                if resolved.legend.page_index == page_index:
                    excluded.append(per_page_legend_bbox[resolved.legend.page_index])
            # Additional exclusion zones — title block, drawing index,
            # keyed notes, and schedules. Without these, a "PTZ" inside
            # "PTZ ROOM" or a schedule cell gets counted as a detection.
            from orbitbrief_page_os.segmentation.schematic.exclusion_zones import (
                detect_exclusion_zones,
            )
            from orbitbrief_page_os.segmentation.schematic.sheet_metadata import (
                parse_sheet_metadata,
            )
            from orbitbrief_page_os.segmentation.schematic.rooms import (
                Room,
                assign_detections_to_rooms,
                detect_rooms,
            )
            from orbitbrief_page_os.segmentation.schematic.keyed_notes import (
                detect_keyed_notes,
            )

            zones = detect_exclusion_zones(blocks, page_bbox=page_bbox)
            for zone in zones:
                excluded.append(zone.bbox)

            # Sheet metadata atom — one per drawing page that carries
            # an extractable title block.
            title_block_bbox = next(
                (z.bbox for z in zones if z.label == "title_block"),
                None,
            )
            try:
                sheet_meta = parse_sheet_metadata(
                    page_index=page_index,
                    blocks=blocks,
                    sheet_number=sheet,
                    title_block_bbox=title_block_bbox,
                )
            except Exception:  # pragma: no cover
                sheet_meta = None
            if sheet_meta is not None:
                # Suppress fieldless sheet_metadata atoms: a sheet
                # number alone is already captured elsewhere
                # (target_set, legend, detections). Only emit when
                # at least one substantive title-block field was
                # parsed.
                substantive = any([
                    sheet_meta.sheet_title,
                    sheet_meta.project_name,
                    sheet_meta.scale,
                    sheet_meta.issue_date,
                    sheet_meta.revision,
                    sheet_meta.drafter,
                    sheet_meta.checker,
                    sheet_meta.approver,
                    sheet_meta.client,
                ])
                if substantive:
                    atoms.append(
                        emit_sheet_metadata_atom(
                            metadata=sheet_meta,
                            project_id=project_id,
                            artifact_id=artifact_id,
                            filename=path.name,
                            parser_version=parser_version,
                            page=page,
                        )
                    )

            # Room / zone atoms — pulled from blocks outside the
            # excluded zones so we don't pick up schedule-row room IDs.
            try:
                rooms_on_page: list[Room] = detect_rooms(
                    page_index=page_index,
                    sheet_number=sheet,
                    blocks=blocks,
                    excluded_bboxes=tuple(excluded),
                )
            except Exception:  # pragma: no cover
                rooms_on_page = []
            for room in rooms_on_page:
                atoms.append(
                    emit_room_atom(
                        room=room,
                        project_id=project_id,
                        artifact_id=artifact_id,
                        filename=path.name,
                        parser_version=parser_version,
                        page=page,
                    )
                )

            # Keyed-notes atoms — both the note rows and their resolved
            # body callouts. The exclusion-zone pass already keeps the
            # block out of symbol detection; this turns the contents
            # into reviewable atoms.
            try:
                keyed_notes_on_page = detect_keyed_notes(
                    page_index=page_index,
                    sheet_number=sheet,
                    blocks=blocks,
                )
            except Exception:  # pragma: no cover
                keyed_notes_on_page = []
            for note in keyed_notes_on_page:
                atoms.append(
                    emit_keyed_note_atom(
                        note=note,
                        project_id=project_id,
                        artifact_id=artifact_id,
                        filename=path.name,
                        parser_version=parser_version,
                        page=page,
                    )
                )

            # Construction schedule rows — door / camera / equipment /
            # fixture / panel schedules.  Each row joins to a detection
            # by tag downstream (after detect_symbols runs).
            from orbitbrief_page_os.segmentation.schematic.schedules import (
                detect_schedules,
                join_schedule_rows_to_detections,
            )

            try:
                schedule_rows_on_page = detect_schedules(
                    page_index=page_index,
                    sheet_number=sheet,
                    blocks=blocks,
                )
            except Exception:  # pragma: no cover
                schedule_rows_on_page = []
            for row in schedule_rows_on_page:
                atoms.append(
                    emit_schedule_row_atom(
                        row=row,
                        project_id=project_id,
                        artifact_id=artifact_id,
                        filename=path.name,
                        parser_version=parser_version,
                        page=page,
                    )
                )

            # Prose-with-symbol suppression: any text block whose text
            # contains a legend symbol but ISN'T a standalone label
            # (e.g. "PTZ ROOM", "Card Reader Suite") must be added to
            # the exclusion set so the glyph_template matcher does
            # not catch the symbol's pixels inside the prose word.
            # The text-tag matcher already filters via
            # _block_text_is_standalone_symbol; glyph_template needs
            # the bboxes excluded explicitly because it operates on
            # rendered pixels.
            from orbitbrief_page_os.segmentation.schematic.symbol_detector import (
                _block_text_is_standalone_symbol,
            )

            legend_symbol_tokens: dict[str, Any] = {
                (e.normalized_symbol_text or "").upper(): e
                for e in resolved.legend.entries
                if e.normalized_symbol_text
            }
            for blk in blocks:
                text = (blk.text or "").strip()
                if not text:
                    continue
                upper = text.upper()
                if not any(
                    sym in upper.split() or sym + " " in upper or " " + sym in upper or upper == sym
                    for sym in legend_symbol_tokens
                ):
                    continue
                if not _block_text_is_standalone_symbol(text, legend_symbol_tokens):
                    excluded.append(blk.bbox)
            detections = detect_symbols(
                page=page,
                page_index=page_index,
                sheet_number=sheet,
                blocks=blocks,
                target_set=target_set,
                legend=resolved.legend,
                legend_page=legend_page,
                excluded_bboxes=tuple(excluded),
            )
            # Vision-LLM augmentation for SCHEMATIC_DRAWING pages.
            # On real schematics the symbol IS an icon, not text — the
            # text-tag detector returns 0 hits. Vision detector finds
            # icons via region proposals + qwen2.5vl match against the
            # legend symbol crops. Only runs when the endpoint is
            # reachable + at least one legend crop was extracted.
            classification_for_page = classify_page_kind(
                page_index=page_index, page=page, blocks=blocks
            ) if page is not None else None
            page_kind_for_vision = (
                classification_for_page.kind if classification_for_page else PAGE_UNKNOWN
            )
            if (
                vision_enabled
                and vision_legend_crops
                and page_kind_for_vision in (SCHEMATIC_DRAWING, PAGE_UNKNOWN)
                and page is not None
            ):
                try:
                    from orbitbrief_page_os.segmentation.schematic.region_proposals import (
                        propose_regions,
                    )
                    from orbitbrief_page_os.segmentation.schematic.vision_symbol_detector import (
                        detect_symbols_via_vision,
                    )
                except Exception:  # pragma: no cover
                    propose_regions = None  # type: ignore[assignment]
                    detect_symbols_via_vision = None  # type: ignore[assignment]
                if propose_regions is not None and detect_symbols_via_vision is not None:
                    try:
                        proposals = propose_regions(page=page, page_index=page_index)
                    except Exception:  # pragma: no cover
                        proposals = []
                    if proposals:
                        try:
                            vision_dets = detect_symbols_via_vision(
                                page=page,
                                page_index=page_index,
                                region_proposals=proposals,
                                legend_crops=vision_legend_crops,
                                cache_path=vision_cache_path,
                            )
                        except Exception:  # pragma: no cover
                            vision_dets = []
                        # Convert VisionDetection → SymbolDetection so the
                        # downstream emit pipeline treats them uniformly
                        # with the text_tag detections.
                        from app.parsers.schematic_models import SymbolDetection as _SymbolDetection
                        entry_by_id = {
                            e.entry_id: e
                            for l in parsed_legends
                            for e in l.entries
                        }
                        target_by_entry_id: dict[str, Any] = {}
                        for t in target_set.targets:
                            if t.legend_entry_id:
                                target_by_entry_id[t.legend_entry_id] = t
                        for vd in vision_dets:
                            entry = entry_by_id.get(vd.matched_entry_id)
                            if entry is None:
                                continue
                            target = target_by_entry_id.get(vd.matched_entry_id)
                            # When the active pack doesn't intersect the
                            # legend (e.g. running the fiber pack on a
                            # security/telecom legend), synthesize a
                            # target_key from the entry itself so the
                            # vision detection isn't dropped.
                            if target is not None:
                                target_key = target.target_key
                                entity_key = target.target_key
                            else:
                                target_key = (
                                    entry.normalized_label
                                    or (entry.normalized_symbol_text or "")
                                    or entry.entry_id
                                )
                                entity_key = f"device:{target_key}".lower().replace(" ", "_")
                            try:
                                sd = _SymbolDetection.make(
                                    page_index=page_index,
                                    sheet_number=sheet,
                                    target_key=target_key,
                                    entity_key=entity_key,
                                    legend_entry_id=entry.entry_id,
                                    bbox_pdf=vd.bbox_pdf,
                                    crop_sha256="",
                                    modality="vision_llm",
                                    confidence=vd.confidence,
                                    nearby_text=vd.matched_label_text,
                                )
                            except (TypeError, ValueError):
                                continue
                            detections.append(sd)
            # Assign each detection to its nearest room (when rooms
            # were detected on this page). The mapping is recorded
            # on the detection atom's value so downstream consumers
            # can group counts by room without re-running geometry.
            detection_room_map: dict[str, str] = {}
            if rooms_on_page:
                try:
                    detection_room_map = assign_detections_to_rooms(
                        detections, rooms_on_page
                    )
                except Exception:  # pragma: no cover
                    detection_room_map = {}

            # Mounting-height callouts — attach the nearest one to each
            # detection so a CR atom carries "48 AFF" without the
            # reviewer opening the PDF.
            from orbitbrief_page_os.segmentation.schematic.callouts import (
                attach_callouts_to_detections,
                detect_callouts,
            )

            try:
                callouts_on_page = detect_callouts(blocks, excluded_bboxes=tuple(excluded))
                detection_callout_map = attach_callouts_to_detections(
                    detections, callouts_on_page
                )
            except Exception:  # pragma: no cover
                detection_callout_map = {}

            # Mounting-height inheritance chain (PM-critical):
            #   1. nearest inline callout (set above)
            #   2. schedule row's "mounting" / "mounting_height" field
            #   3. legend entry's MOUNTING / MOUNTING HEIGHT attribute
            #   4. keyed-note default ("All devices mounted at X AFF
            #      unless noted") — derived once per page
            import re as _re

            keyed_note_default_height: str | None = None
            for note in keyed_notes_on_page:
                m = _re.search(
                    r"(?:mounted|mounting)\s+(?:at|height)?\s*"
                    r"([0-9]+(?:\.[0-9]+)?\s*(?:\"|in|inches)?\s*"
                    r"a\.?f\.?f\.?|"
                    r"[0-9]+\s*'\s*-\s*[0-9]+(?:\s*[0-9]+/[0-9]+)?\s*\"|"
                    r"ceiling|"
                    r"verify\s+w/?\s*arch)",
                    note.text,
                    _re.IGNORECASE,
                )
                if m:
                    keyed_note_default_height = m.group(1).strip()
                    break

            legend_mounting_by_entry: dict[str, str] = {}
            legend_responsibility_by_entry: dict[str, str] = {}
            legend_remarks_by_entry: dict[str, str] = {}
            for entry in resolved.legend.entries:
                attrs = dict(entry.attributes)
                m_val = (
                    attrs.get("mounting_height")
                    or attrs.get("mounting")
                )
                if m_val:
                    legend_mounting_by_entry[entry.entry_id] = m_val
                # Responsibility / by-others markers — explicit
                # ``responsibility`` column wins; otherwise scan
                # the remarks column for the conventional phrases.
                resp_val: str | None = attrs.get("responsibility")
                remarks_text = attrs.get("remarks") or ""
                if not resp_val and remarks_text:
                    upper = remarks_text.upper()
                    for marker in ("NIC", "BY OWNER", "BY GC", "BY OTHERS", "NOT IN CONTRACT"):
                        if marker in upper:
                            resp_val = marker
                            break
                if resp_val:
                    legend_responsibility_by_entry[entry.entry_id] = resp_val
                if remarks_text:
                    legend_remarks_by_entry[entry.entry_id] = remarks_text

            # Schedule-row joins — pass 1 is nearby_text tag match,
            # pass 2 is spatial join when a TAG block sits within
            # ~2 inches of the detection center.
            try:
                detection_schedule_map = join_schedule_rows_to_detections(
                    schedule_rows_on_page,
                    detections,
                    blocks=blocks,
                )
            except Exception:  # pragma: no cover
                detection_schedule_map = {}

            # Line runs — conduit / cable / riser polylines, snapped
            # to nearby detections. Emitted AFTER detections so the
            # snap targets are deterministic.
            from orbitbrief_page_os.segmentation.schematic.line_runs import (
                detect_line_runs,
            )

            try:
                line_runs_on_page = detect_line_runs(
                    page=page,
                    page_index=page_index,
                    sheet_number=sheet,
                    detections=detections,
                    excluded_bboxes=tuple(excluded),
                )
            except Exception:  # pragma: no cover
                line_runs_on_page = []
            for line_run in line_runs_on_page:
                atoms.append(
                    emit_line_run_atom(
                        line_run=line_run,
                        project_id=project_id,
                        artifact_id=artifact_id,
                        filename=path.name,
                        parser_version=parser_version,
                        page=page,
                    )
                )

            for det in detections:
                room_id = detection_room_map.get(det.detection_id)
                callout = detection_callout_map.get(det.detection_id)
                schedule_row = detection_schedule_map.get(det.detection_id)
                atom = emit_detection_atom(
                    detection=det,
                    project_id=project_id,
                    artifact_id=artifact_id,
                    filename=path.name,
                    parser_version=parser_version,
                )
                updates: dict[str, Any] = {}
                new_value = dict(atom.value)
                new_entity_keys = list(atom.entity_keys)
                if room_id:
                    new_value["located_in_room_id"] = room_id
                    # Look up the room's human-readable label/number
                    # so downstream consumers don't have to join on
                    # the opaque room hash.
                    room_obj = next(
                        (r for r in rooms_on_page if r.room_id == room_id),
                        None,
                    )
                    if room_obj is not None:
                        new_value["located_in_room_label"] = room_obj.label
                        if room_obj.number:
                            new_value["located_in_room_number"] = room_obj.number
                            new_value["located_in_room_display"] = (
                                f"{room_obj.label} {room_obj.number}"
                            )
                        else:
                            new_value["located_in_room_display"] = room_obj.label
                    new_entity_keys.append(f"room:{room_id}")
                # Mounting-height inheritance chain.
                resolved_height: str | None = None
                height_source: str | None = None
                if callout is not None:
                    resolved_height = callout.text
                    height_source = "inline_callout"
                    new_value["callout_bbox"] = list(callout.bbox)
                if schedule_row is not None:
                    new_value["schedule_row_id"] = schedule_row.row_id
                    new_value["schedule_tag"] = schedule_row.tag
                    new_value["schedule_kind"] = schedule_row.schedule_kind
                    new_value["schedule_fields"] = dict(schedule_row.fields)
                    new_entity_keys.append(f"schedule_tag:{schedule_row.tag}")
                    if resolved_height is None:
                        sched_height = (
                            schedule_row.fields_dict().get("mounting_height")
                            or schedule_row.fields_dict().get("mounting")
                        )
                        if sched_height:
                            resolved_height = sched_height
                            height_source = "schedule"
                # Legend column fallback.
                if resolved_height is None and det.legend_entry_id:
                    legend_height = legend_mounting_by_entry.get(det.legend_entry_id)
                    if legend_height:
                        resolved_height = legend_height
                        height_source = "legend_column"
                # Keyed-note default fallback ("X AFF unless noted").
                if resolved_height is None and keyed_note_default_height:
                    resolved_height = keyed_note_default_height
                    height_source = "keyed_note_default"
                if resolved_height is not None:
                    new_value["mounting_height"] = resolved_height
                    new_value["mounting_height_source"] = height_source

                # Responsibility / NIC markers (PM-critical for scope).
                if det.legend_entry_id:
                    resp_val = legend_responsibility_by_entry.get(det.legend_entry_id)
                    if resp_val:
                        new_value["responsibility"] = resp_val
                        new_entity_keys.append(
                            f"responsibility:{resp_val.lower().replace(' ', '_')}"
                        )
                    remarks_val = legend_remarks_by_entry.get(det.legend_entry_id)
                    if remarks_val:
                        new_value["legend_remarks"] = remarks_val
                # Trigger the update when ANY field was added or
                # ANY new entity_key was appended.  The earlier code
                # only checked the room/callout/schedule trio, which
                # silently dropped keyed-note-default heights,
                # legend-column heights, and responsibility markers
                # on detections with no room/callout/schedule.
                if new_value != atom.value or new_entity_keys != list(atom.entity_keys):
                    updates["value"] = new_value
                    updates["entity_keys"] = sorted(set(new_entity_keys))
                if updates:
                    atom = atom.model_copy(update=updates)
                atoms.append(atom)
                detection_records.append(
                    {
                        "detection_id": det.detection_id,
                        "page": det.page_index,
                        "target_key": det.target_key,
                        "modality": det.modality,
                        "bbox": list(det.bbox_pdf),
                        "crop_sha256": det.crop_sha256,
                        "confidence": det.confidence,
                        "located_in_room_id": room_id,
                        "mounting_height": callout.text if callout else None,
                        "schedule_row_id": schedule_row.row_id if schedule_row else None,
                        "schedule_tag": schedule_row.tag if schedule_row else None,
                    }
                )

            # Schematic quantity aggregation (PR7) — turn detection
            # counts into ``AtomType.quantity`` atoms and emit a
            # declared-count atom from any legend row that has a
            # count_column. Same-sheet conflicts are paired by
            # ``_build_schematic_quantity_edges`` in the graph builder.
            from app.parsers.schematic_atom_emitters import (
                emit_declared_count_atom,
                emit_detected_count_atom,
            )

            counts_by_target: dict[str, list] = {}
            for det in detections:
                counts_by_target.setdefault(det.target_key, []).append(det)
            for target in target_set.targets:
                hits = counts_by_target.get(target.target_key, [])
                detected_atom = emit_detected_count_atom(
                    page_index=page_index,
                    sheet_number=sheet,
                    target=target,
                    detections=hits,
                    project_id=project_id,
                    artifact_id=artifact_id,
                    filename=path.name,
                    parser_version=parser_version,
                )
                if detected_atom is not None:
                    atoms.append(detected_atom)

                # legend_orphan: load-bearing target declared by the
                # legend but zero detections on this drawing body.
                # Boss-review fix — previously declared but never emitted.
                if (
                    not hits
                    and target.completeness == "load_bearing"
                    and target.legend_entry_id is not None
                ):
                    orphan_entry = next(
                        (e for e in resolved.legend.entries if e.entry_id == target.legend_entry_id),
                        None,
                    )
                    orphan_bbox = orphan_entry.symbol_bbox_pdf if orphan_entry else None
                    atoms.append(
                        emit_warning_atom(
                            warning=SchematicWarning.make(
                                warning_type="legend_orphan",
                                page_index=page_index,
                                sheet_number=sheet,
                                detail=(
                                    f"Legend entry for load-bearing target "
                                    f"'{target.target_key}' produced zero detections on this page."
                                ),
                                target_key=target.target_key,
                                legend_id=resolved.legend.legend_id,
                                legend_entry_id=target.legend_entry_id,
                                bbox_pdf=orphan_bbox,
                            ),
                            project_id=project_id,
                            artifact_id=artifact_id,
                            filename=path.name,
                            parser_version=parser_version,
                            page=legend_page,
                        )
                    )

                if target.legend_entry_id is None:
                    continue
                # Walk the legend for the declared count for this entry.
                # Emit the declared atom only once per (target, legend_entry)
                # pair — without this guard the same declared count would
                # be re-emitted for every drawing page that resolves to
                # the same legend.
                dedup_key = (target.target_key, target.legend_entry_id)
                if dedup_key in declared_emitted:
                    continue
                for entry in resolved.legend.entries:
                    if entry.entry_id != target.legend_entry_id:
                        continue
                    if entry.count_column is None:
                        continue
                    declared = emit_declared_count_atom(
                        page_index=resolved.legend.page_index,
                        sheet_number=resolved.legend.sheet_number,
                        target=target,
                        declared_count=entry.count_column,
                        entry=entry,
                        project_id=project_id,
                        artifact_id=artifact_id,
                        filename=path.name,
                        parser_version=parser_version,
                        page=legend_page,
                    )
                    if declared is not None:
                        atoms.append(declared)
                        declared_emitted.add(dedup_key)
                    else:
                        # Provenance gate refused (no symbol bbox or no
                        # crop hash available). Emit a low-confidence
                        # warning so the count isn't silently lost.
                        atoms.append(
                            emit_warning_atom(
                                warning=SchematicWarning.make(
                                    warning_type="weak_declared_count_provenance",
                                    page_index=resolved.legend.page_index,
                                    sheet_number=resolved.legend.sheet_number,
                                    detail=(
                                        f"Legend declared count={entry.count_column} for target "
                                        f"'{target.target_key}' but the row had no replayable bbox; "
                                        f"declared-count atom suppressed."
                                    ),
                                    target_key=target.target_key,
                                    legend_id=resolved.legend.legend_id,
                                    legend_entry_id=target.legend_entry_id,
                                ),
                                project_id=project_id,
                                artifact_id=artifact_id,
                                filename=path.name,
                                parser_version=parser_version,
                                page=legend_page,
                            )
                        )
                    break

            # ``unknown_symbol`` warnings: tokens that look like
            # legend-style symbol tags but matched no legend entry.
            atoms.extend(
                _unknown_symbol_warnings(
                    blocks=blocks,
                    page_index=page_index,
                    sheet=sheet,
                    legend=resolved.legend,
                    excluded_bboxes=excluded,
                    project_id=project_id,
                    artifact_id=artifact_id,
                    filename=path.name,
                    parser_version=parser_version,
                    page=page,
                )
            )
    finally:
        try:
            doc.close()
        except Exception:  # pragma: no cover
            pass

    derived_relative = derived_dir_for(path).name
    derived_files: list[dict[str, Any]] = [
        {
            "relative_path": f"{derived_relative}/schematic_legends.json",
            "content_kind": "json",
            "content_json": {"schema_version": "schematic.legends.v1", "legends": legend_records},
        },
        {
            "relative_path": f"{derived_relative}/schematic_targets.json",
            "content_kind": "json",
            "content_json": {"schema_version": "schematic.targets.v1", "pages": target_records},
        },
        {
            "relative_path": f"{derived_relative}/schematic_detections.json",
            "content_kind": "json",
            "content_json": {"schema_version": "schematic.detections.v1", "detections": detection_records},
        },
    ]
    # Optional debug-overlay sidecars. The flag is opt-in via the
    # ``PARSER_OS_SCHEMATIC_OVERLAYS`` env var so default compiles
    # still produce byte-identical output. When set, one PNG per
    # drawing page is written under ``<stem>.derived/overlays/`` and
    # an ``schematic_overlays.json`` manifest is added so downstream
    # consumers (OrbitBrief envelope renderer, debug viewer) can
    # find them deterministically.
    if os.environ.get("PARSER_OS_SCHEMATIC_OVERLAYS") == "1" and parsed_legends:
        try:
            from orbitbrief_page_os.segmentation.schematic.debug_overlay import render_overlay
        except Exception:
            render_overlay = None  # type: ignore[assignment]
        if render_overlay is not None:
            try:
                overlay_doc = fitz.open(str(path))
            except Exception:  # pragma: no cover
                overlay_doc = None
            overlay_manifest: list[dict[str, Any]] = []
            target_pages = sorted({rec["page"] for rec in target_records})
            if overlay_doc is not None:
                try:
                    for page_index in target_pages:
                        try:
                            overlay_page = overlay_doc.load_page(page_index)
                        except Exception:  # pragma: no cover
                            continue
                        page_detections = [
                            d for d in detection_records if d.get("page") == page_index
                        ]
                        legends_here = [
                            l for l in parsed_legends if l.page_index == page_index
                        ]
                        # debug_overlay.render_overlay expects SymbolDetection
                        # records, not raw dicts — rebuild lightweight stand-ins.
                        from app.parsers.schematic_models import SymbolDetection

                        dets: list[SymbolDetection] = []
                        for d in page_detections:
                            bbox = d.get("bbox") or [0, 0, 1, 1]
                            try:
                                dets.append(
                                    SymbolDetection.make(
                                        page_index=int(d.get("page", page_index)),
                                        sheet_number=None,
                                        target_key=str(d.get("target_key", "")),
                                        entity_key=str(d.get("target_key", "")),
                                        legend_entry_id=None,
                                        bbox_pdf=(
                                            float(bbox[0]),
                                            float(bbox[1]),
                                            float(bbox[2]),
                                            float(bbox[3]),
                                        ),
                                        crop_sha256=str(d.get("crop_sha256") or ""),
                                        modality=d.get("modality") or "text_tag",
                                        confidence=float(d.get("confidence") or 0.0),
                                    )
                                )
                            except ValueError:
                                continue
                        out_rel = f"{derived_relative}/overlays/page_{page_index:04d}.png"
                        out_path = path.parent / out_rel.replace("/", os.sep)
                        result = render_overlay(
                            page=overlay_page,
                            legends_on_page=legends_here,
                            detections=dets,
                            out_path=out_path,
                        )
                        if result is not None:
                            overlay_manifest.append(
                                {
                                    "page": page_index,
                                    "relative_path": out_rel,
                                    "legend_count": result.legend_count,
                                    "detection_count": result.detection_count,
                                    "width": result.width,
                                    "height": result.height,
                                }
                            )
                finally:
                    try:
                        overlay_doc.close()
                    except Exception:  # pragma: no cover
                        pass
            derived_files.append(
                {
                    "relative_path": f"{derived_relative}/schematic_overlays.json",
                    "content_kind": "json",
                    "content_json": {
                        "schema_version": "schematic.overlays.v1",
                        "overlays": overlay_manifest,
                    },
                }
            )
    return collect_all(atoms), derived_files

# Tokens that look symbol-shaped but are conventionally noise on
# construction drawings — column-grid bubbles (single letters), simple
# integer keyed-note numbers (handled separately by the keyed-notes
# pass when present), the page's own sheet number, and a small set of
# common page metadata tokens.  Boss-review fix: previously every
# repeated short ALL-CAPS token became an unknown_symbol.
_UNKNOWN_TOKEN_IGNORES = {
    "NIC",
    "NTS",
    "NA",
    "TBD",
    "REF",
    "REV",
    "SEE",
    "MAX",
    "MIN",
    "TYP",
    "EQ",
    "AFF",
    "OC",
    "DWG",
    "SHT",
    "GC",
    "EC",
    "MC",
    "PC",
    "AV",
    "FA",
    "AC",
    "SC",
    "BMS",
    "AHU",
    "VAV",
    "PDU",
    "UPS",
    "ATS",
    "MDF",
    "IDF",
    "TR",
    "ER",
    "MEP",
}

def _unknown_symbol_warnings(
    *,
    blocks: list[Any],
    page_index: int,
    sheet: str | None,
    legend: Any,
    excluded_bboxes: list[tuple[float, float, float, float]],
    project_id: str,
    artifact_id: str,
    filename: str,
    parser_version: str,
    page: Any | None = None,
) -> list[EvidenceAtom]:
    """Emit ``unknown_symbol`` warnings for legend-style tokens with no match.

    Conservative: only short ALL-CAPS tokens (length 2-5) that appear
    repeatedly on the page.  The boss review caught that the previous
    implementation flagged ordinary drawing furniture — sheet numbers,
    grid bubbles, keyed-note integers, common drawing abbreviations —
    as unknown symbols, drowning the real warnings.  This version
    suppresses each of those classes.
    """
    import re as _re

    from app.parsers.schematic_atom_emitters import emit_warning_atom
    from app.parsers.schematic_models import SchematicWarning

    known: set[str] = {
        (e.normalized_symbol_text or "").upper()
        for e in legend.entries
        if e.normalized_symbol_text
    }
    sheet_token = (sheet or "").upper()

    def _looks_like_grid_bubble(tok: str) -> bool:
        # A single letter or single digit is a grid label, not a symbol.
        return len(tok) == 1

    def _looks_like_keyed_note_integer(tok: str) -> bool:
        # Bare 1-3 digit integers are typically keyed-note markers.
        return tok.isdigit() and 1 <= len(tok) <= 3

    def _looks_like_sheet_number(tok: str) -> bool:
        # The page's own sheet number repeats in the title block / index.
        return tok == sheet_token or _re.match(r"^[A-Z]{1,3}\d+(?:\.\d+)?$", tok) is not None

    counts: dict[str, int] = {}
    first_bbox: dict[str, tuple[float, float, float, float]] = {}
    for blk in blocks:
        if any(_bbox_intersects(blk.bbox, ex) for ex in excluded_bboxes):
            continue
        for m in _re.finditer(r"\b[A-Z0-9][A-Z0-9\-]{1,4}\b", blk.text):
            tok = m.group(0).upper()
            if tok in known:
                continue
            if tok in _UNKNOWN_TOKEN_IGNORES:
                continue
            if _looks_like_grid_bubble(tok):
                continue
            if _looks_like_keyed_note_integer(tok):
                continue
            if _looks_like_sheet_number(tok):
                continue
            counts[tok] = counts.get(tok, 0) + 1
            first_bbox.setdefault(tok, blk.bbox)
    out: list[EvidenceAtom] = []
    for tok, n in sorted(counts.items()):
        if n < 3:  # ignore noise — only flag clearly repeated tokens
            continue
        out.append(
            emit_warning_atom(
                warning=SchematicWarning.make(
                    warning_type="unknown_symbol",
                    page_index=page_index,
                    sheet_number=sheet,
                    detail=f"Token {tok!r} appears {n} times on page but is not in the resolved legend.",
                    bbox_pdf=first_bbox[tok],
                    extras={"token": tok, "count": n},
                ),
                project_id=project_id,
                artifact_id=artifact_id,
                filename=filename,
                parser_version=parser_version,
                page=page,
            )
        )
    return out

def _bbox_intersects(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])
