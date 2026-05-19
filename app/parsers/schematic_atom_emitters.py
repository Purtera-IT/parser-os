"""Project schematic dataclasses onto ``EvidenceAtom`` instances.

Used by ``OrbitbriefPdfParser`` (PR5) and the symbol detector (PR6).
Each emitter is small and side-effect-free; the caller decides where
the produced atoms go in the ``ParserOutput`` stream.
"""
from __future__ import annotations

from typing import Any, Iterable

from app.core.ids import stable_id
from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    EvidenceAtom,
    ReviewStatus,
    SourceRef,
)
from app.domain.schemas import DetectionTargetSpec, DomainPack
from app.parsers.schematic_models import (
    BBOX_UNITS_PDF_POINTS,
    SCHEMATIC_REPLAY_DPI,
    DetectionTarget,
    DetectionTargetSet,
    ParsedLegend,
    ParsedLegendEntry,
    SchematicWarning,
    SymbolDetection,
    crop_sha256_of_pixels,
)


def compute_crop_sha256(page: Any, bbox: tuple[float, float, float, float]) -> str | None:
    """Render a PDF bbox at the schematic replay DPI and hash the pixels.

    Returns ``None`` if fitz is unavailable or the bbox is degenerate.
    Callers use the result to fill ``SourceRef.locator['crop_sha256']``
    so source_replay can re-verify the exact region this atom claims.
    """
    try:
        import fitz  # type: ignore[import-not-found]
    except Exception:  # pragma: no cover
        return None
    x0, y0, x1, y1 = (float(b) for b in bbox)
    if not (x1 > x0 and y1 > y0):
        return None
    try:
        zoom = SCHEMATIC_REPLAY_DPI / 72.0
        pix = page.get_pixmap(
            matrix=fitz.Matrix(zoom, zoom),
            clip=fitz.Rect(x0, y0, x1, y1),
            alpha=False,
            colorspace=fitz.csRGB,
        )
        return crop_sha256_of_pixels(pix.samples, pix.width, pix.height, pix.n)
    except Exception:  # pragma: no cover
        return None


def build_replayable_locator(
    *,
    page_index: int,
    bbox: tuple[float, float, float, float] | None,
    page: Any | None,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a ``SourceRef.locator`` carrying page + bbox + crop_sha256.

    When ``page`` and ``bbox`` are both available, the resulting locator
    is replayable by ``_verify_pdf_bbox_crop`` (the bbox is re-rendered
    and the hash is recomputed). When ``bbox`` is missing we degrade to a
    page-only locator — the atom is still emitted but is not replayable;
    callers should pair that case with an explicit
    ``low_provenance`` flag in ``extras`` so audits can find it later.
    """
    loc: dict[str, Any] = {"page": page_index}
    if extras:
        for k, v in extras.items():
            loc[k] = v
    if bbox is not None:
        x0, y0, x1, y1 = (float(b) for b in bbox)
        if x1 > x0 and y1 > y0:
            loc["bbox"] = [x0, y0, x1, y1]
            loc["bbox_units"] = BBOX_UNITS_PDF_POINTS
            if page is not None:
                crop = compute_crop_sha256(page, (x0, y0, x1, y1))
                if crop:
                    loc["crop_sha256"] = crop
    return loc


# ─────────────────────── shared SourceRef builder ─────────────────


def _source_ref(
    *,
    artifact_id: str,
    filename: str,
    locator: dict,
    parser_version: str,
    extraction_method: str,
    suffix: str,
) -> SourceRef:
    sid = stable_id("sr", artifact_id, extraction_method, suffix)
    return SourceRef(
        id=sid,
        artifact_id=artifact_id,
        artifact_type=ArtifactType.pdf,
        filename=filename,
        locator=locator,
        extraction_method=extraction_method,
        parser_version=parser_version,
    )


# ─────────────────────── intersection logic ───────────────────────


def intersect_with_pack(
    *,
    legend: ParsedLegend,
    pack: DomainPack,
) -> tuple[list[DetectionTarget], list[str]]:
    """Map parsed legend entries onto domain-pack detection targets.

    Returns ``(targets, legend_gap_target_keys)``.

    Each legend entry resolves to *at most one* pack target — the one
    whose aliases produce the strongest specificity score against
    the entry's symbol text and label.  This prevents a broad alias
    bucket (e.g. ``device_aliases.ip_camera`` containing the generic
    token ``camera``) from accidentally claiming a "PTZ CAMERA"
    legend entry on behalf of every camera subtype.

    A pack target with ``completeness="load_bearing"`` that no legend
    entry resolves to is added to ``legend_gap_target_keys`` so the
    parser can emit a ``legend_gap`` warning.
    """
    matched: dict[str, DetectionTarget] = {}
    for entry in legend.entries:
        best = _best_target_for_entry(entry, pack)
        if best is None:
            continue
        spec, score = best
        if spec.key in matched:
            continue
        matched[spec.key] = DetectionTarget(
            target_key=spec.key,
            entity_key=spec.entity_key,
            completeness=spec.completeness,
            expected_modalities=tuple(spec.modalities),
            ontology_key=spec.ontology_key,
            legend_entry_id=entry.entry_id,
            aliases=tuple(pack.resolved_target_aliases(spec)),
            parent_entity_keys=tuple(spec.parent_entity_keys),
        )
    gaps: list[str] = []
    for spec in pack.detection_targets:
        if spec.completeness != "load_bearing":
            continue
        if spec.key in matched:
            continue
        gaps.append(spec.key)
    targets = sorted(matched.values(), key=lambda t: t.target_key)
    return targets, sorted(gaps)


def _best_target_for_entry(
    entry: ParsedLegendEntry,
    pack: DomainPack,
) -> tuple[DetectionTargetSpec, int] | None:
    """Return ``(target, score)`` for the pack target that best matches the entry.

    Explicit aliases (target ``aliases:``) and the target's own key
    rank ahead of aliases pulled in by ``aliases_from``.  This
    prevents a wide ``device_aliases.*`` super-bucket (e.g.
    ``ip_camera`` containing every camera subtype's variants) from
    making subtype targets indistinguishable — the parent bucket is
    only a *fallback* for matching, not the authority for picking
    the most specific target.

    Specificity (per alias):

      - 6 = explicit alias / key-alias exactly equals entry symbol
      - 5 = explicit alias / key-alias equals a candidate string
      - 4 = explicit multi-word alias whose tokens all appear in candidate
      - 3 = explicit single-word alias that appears as a candidate token
      - 2 = inherited alias (from aliases_from) — equal to or in candidate
      - 0 = no match
    """
    candidates = []
    if entry.normalized_symbol_text:
        candidates.append(entry.normalized_symbol_text.strip())
    if entry.normalized_label:
        candidates.append(entry.normalized_label.strip())
    if entry.label_text:
        candidates.append(entry.label_text.strip().lower())
    candidates = [c for c in candidates if c]
    if not candidates:
        return None

    sym = (entry.normalized_symbol_text or "").strip()
    best: tuple[DetectionTargetSpec, int] | None = None
    for spec in pack.detection_targets:
        explicit_aliases: list[str] = [a.strip().lower() for a in spec.aliases]
        explicit_aliases.append(spec.key.lower().replace("_", " ").strip())
        explicit_aliases = [a for a in explicit_aliases if a]
        inherited_aliases: list[str] = []
        for ref in spec.aliases_from:
            head, sep, tail = ref.partition(".")
            if head != "device_aliases" or not sep or not tail:
                continue
            for raw in pack.device_aliases.get(tail, []) or []:
                inherited_aliases.append(raw.strip().lower())

        best_score = 0
        for alias in explicit_aliases:
            score = _alias_specificity(alias, sym, candidates, explicit=True)
            if score > best_score:
                best_score = score
        if best_score < 3:
            for alias in inherited_aliases:
                score = _alias_specificity(alias, sym, candidates, explicit=False)
                if score > best_score:
                    best_score = score
        if best_score <= 0:
            continue
        if best is None:
            best = (spec, best_score)
            continue
        if best_score > best[1]:
            best = (spec, best_score)
        elif best_score == best[1]:
            if (len(spec.key), spec.key) < (len(best[0].key), best[0].key):
                best = (spec, best_score)
    return best


def _alias_specificity(alias: str, symbol: str, candidates: list[str], *, explicit: bool) -> int:
    if not alias:
        return 0
    if explicit and symbol and alias == symbol:
        return 6
    if explicit and alias in candidates:
        return 5
    alias_tokens = _tokenize(alias)
    for cand in candidates:
        cand_tokens = set(_tokenize(cand))
        if len(alias_tokens) >= 2 and alias_tokens and all(tok in cand_tokens for tok in alias_tokens):
            return 4 if explicit else 2
    if len(alias_tokens) == 1:
        single = alias_tokens[0]
        for cand in candidates:
            cand_tokens = set(_tokenize(cand))
            if single in cand_tokens:
                return 3 if explicit else 2
    return 0


def _tokenize(text: str) -> list[str]:
    import re

    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t]


# ─────────────────────── atom emitters ────────────────────────────


def emit_legend_atom(
    *,
    legend: ParsedLegend,
    project_id: str,
    artifact_id: str,
    filename: str,
    parser_version: str,
    page: Any | None = None,
) -> EvidenceAtom:
    # The legend's source_ref_locator already carries the bbox the
    # locator landed on; rebuild the locator through the replayable
    # helper so a crop_sha256 lands on it too.
    raw_loc = legend.locator_dict()
    bbox_tuple: tuple[float, float, float, float] | None = None
    bbox_raw = raw_loc.get("bbox")
    if isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) == 4:
        bbox_tuple = (float(bbox_raw[0]), float(bbox_raw[1]), float(bbox_raw[2]), float(bbox_raw[3]))
    locator = build_replayable_locator(
        page_index=legend.page_index,
        bbox=bbox_tuple,
        page=page,
        extras={
            "sheet_number": legend.sheet_number,
            "legend_id": legend.legend_id,
            "scope": legend.scope,
            "layer": raw_loc.get("layer"),
        },
    )
    src = _source_ref(
        artifact_id=artifact_id,
        filename=filename,
        locator=locator,
        parser_version=parser_version,
        extraction_method="schematic_legend_parser",
        suffix=legend.legend_id,
    )
    raw = (
        f"{legend.title or 'LEGEND'} ({legend.sheet_number or '?'}): "
        + ", ".join(
            f"{e.raw_symbol_text or '?'}={e.label_text}" for e in legend.entries[:8]
        )
    )
    return EvidenceAtom(
        id=stable_id("atom_schematic_legend", artifact_id, legend.legend_id),
        project_id=project_id,
        artifact_id=artifact_id,
        atom_type=AtomType.schematic_legend,
        raw_text=raw,
        normalized_text=raw.lower(),
        value={
            "legend_id": legend.legend_id,
            "page": legend.page_index,
            "sheet_number": legend.sheet_number,
            "scope": legend.scope,
            "entry_count": len(legend.entries),
            "entries": [
                {
                    "entry_id": e.entry_id,
                    "symbol": e.raw_symbol_text,
                    "label": e.label_text,
                    "count_column": e.count_column,
                }
                for e in legend.entries
            ],
        },
        entity_keys=[f"schematic_legend:{legend.legend_id}"],
        source_refs=[src],
        authority_class=AuthorityClass.machine_extractor,
        confidence=legend.confidence,
        review_status=ReviewStatus.auto_accepted,
        parser_version=parser_version,
    )


def emit_target_set_atom(
    *,
    target_set: DetectionTargetSet,
    project_id: str,
    artifact_id: str,
    filename: str,
    parser_version: str,
    page: Any | None = None,
    page_bbox: tuple[float, float, float, float] | None = None,
) -> EvidenceAtom:
    # Target-set atom is page-scoped (it's a declaration about a whole
    # drawing page, not a single region). Use the page rectangle as the
    # bbox so source_replay can still verify *something* against the
    # rendered page even though the atom doesn't pin a specific glyph.
    locator = build_replayable_locator(
        page_index=target_set.page_index,
        bbox=page_bbox,
        page=page,
        extras={
            "sheet_number": target_set.sheet_number,
            "pack_id": target_set.pack_id,
            "legend_id": target_set.legend_id,
        },
    )
    src = _source_ref(
        artifact_id=artifact_id,
        filename=filename,
        locator=locator,
        parser_version=parser_version,
        extraction_method="schematic_target_set",
        suffix=f"{target_set.page_index}:{target_set.legend_id or ''}",
    )
    target_keys = [t.target_key for t in target_set.targets]
    raw = (
        f"Detection targets for page {target_set.page_index}"
        f" sheet {target_set.sheet_number or '?'}:"
        f" {len(target_keys)} target(s) ({', '.join(target_keys[:8])})"
    )
    return EvidenceAtom(
        id=stable_id(
            "atom_schematic_target_set",
            artifact_id,
            target_set.page_index,
            target_set.legend_id or "",
            tuple(target_keys),
        ),
        project_id=project_id,
        artifact_id=artifact_id,
        atom_type=AtomType.schematic_detection_target_set,
        raw_text=raw,
        normalized_text=raw.lower(),
        value={
            "page": target_set.page_index,
            "sheet_number": target_set.sheet_number,
            "pack_id": target_set.pack_id,
            "legend_id": target_set.legend_id,
            "targets": [
                {
                    "target_key": t.target_key,
                    "entity_key": t.entity_key,
                    "completeness": t.completeness,
                    "modalities": list(t.expected_modalities),
                    "ontology_key": t.ontology_key,
                    "legend_entry_id": t.legend_entry_id,
                }
                for t in target_set.targets
            ],
            "legend_gap_target_keys": list(target_set.legend_gap_target_keys),
        },
        entity_keys=sorted({f"detection_target:{t.target_key}" for t in target_set.targets}),
        source_refs=[src],
        authority_class=AuthorityClass.machine_extractor,
        confidence=0.9,
        review_status=ReviewStatus.auto_accepted,
        parser_version=parser_version,
    )


def emit_sheet_metadata_atom(
    *,
    metadata: Any,
    project_id: str,
    artifact_id: str,
    filename: str,
    parser_version: str,
    page: Any | None = None,
) -> EvidenceAtom:
    """Project a ``SheetMetadata`` record into a ``schematic_sheet_metadata`` atom."""
    locator = build_replayable_locator(
        page_index=metadata.page_index,
        bbox=metadata.bbox,
        page=page,
        extras={"sheet_number": metadata.sheet_number},
    )
    src = _source_ref(
        artifact_id=artifact_id,
        filename=filename,
        locator=locator,
        parser_version=parser_version,
        extraction_method="schematic_sheet_metadata",
        suffix=f"{metadata.page_index}:{metadata.sheet_number or ''}",
    )
    fields = {
        "sheet_number": metadata.sheet_number,
        "sheet_title": metadata.sheet_title,
        "project_name": metadata.project_name,
        "scale": metadata.scale,
        "issue_date": metadata.issue_date,
        "revision": metadata.revision,
        "drafter": metadata.drafter,
        "checker": metadata.checker,
        "approver": metadata.approver,
        "client": metadata.client,
    }
    present = {k: v for k, v in fields.items() if v}
    raw = (
        f"Sheet {metadata.sheet_number or '?'}: "
        + ", ".join(f"{k}={v!r}" for k, v in sorted(present.items()))
    )
    return EvidenceAtom(
        id=stable_id(
            "atom_schematic_sheet_metadata",
            artifact_id,
            metadata.page_index,
            metadata.sheet_number or "",
        ),
        project_id=project_id,
        artifact_id=artifact_id,
        atom_type=AtomType.schematic_sheet_metadata,
        raw_text=raw,
        normalized_text=raw.lower(),
        value={"page": metadata.page_index, **fields},
        entity_keys=sorted(
            {f"sheet:{metadata.sheet_number}"} if metadata.sheet_number else set()
        ),
        source_refs=[src],
        authority_class=AuthorityClass.machine_extractor,
        confidence=0.85,
        review_status=ReviewStatus.auto_accepted,
        parser_version=parser_version,
    )


def emit_room_atom(
    *,
    room: Any,
    project_id: str,
    artifact_id: str,
    filename: str,
    parser_version: str,
    page: Any | None = None,
) -> EvidenceAtom:
    """Project a ``Room`` record into a ``schematic_room`` atom."""
    locator = build_replayable_locator(
        page_index=room.page_index,
        bbox=room.bbox,
        page=page,
        extras={"sheet_number": room.sheet_number, "room_id": room.room_id},
    )
    src = _source_ref(
        artifact_id=artifact_id,
        filename=filename,
        locator=locator,
        parser_version=parser_version,
        extraction_method="schematic_room",
        suffix=room.room_id,
    )
    raw = f"Room {room.label!r} on page {room.page_index}"
    return EvidenceAtom(
        id=stable_id("atom_schematic_room", artifact_id, room.room_id),
        project_id=project_id,
        artifact_id=artifact_id,
        atom_type=AtomType.schematic_room,
        raw_text=raw,
        normalized_text=raw.lower(),
        value={
            "room_id": room.room_id,
            "page": room.page_index,
            "sheet_number": room.sheet_number,
            "label": room.label,
            "number": room.number,
        },
        entity_keys=[f"room:{room.room_id}"],
        source_refs=[src],
        authority_class=AuthorityClass.machine_extractor,
        confidence=room.confidence,
        review_status=ReviewStatus.auto_accepted,
        parser_version=parser_version,
    )


def emit_keyed_note_atom(
    *,
    note: Any,
    project_id: str,
    artifact_id: str,
    filename: str,
    parser_version: str,
    page: Any | None = None,
) -> EvidenceAtom:
    locator = build_replayable_locator(
        page_index=note.page_index,
        bbox=note.bbox,
        page=page,
        extras={"sheet_number": note.sheet_number, "note_number": note.number},
    )
    src = _source_ref(
        artifact_id=artifact_id,
        filename=filename,
        locator=locator,
        parser_version=parser_version,
        extraction_method="schematic_keyed_note",
        suffix=f"{note.page_index}:{note.number}",
    )
    raw = f"Keyed note {note.number}: {note.text}"
    return EvidenceAtom(
        id=stable_id("atom_schematic_keyed_note", artifact_id, note.page_index, note.number),
        project_id=project_id,
        artifact_id=artifact_id,
        atom_type=AtomType.schematic_keyed_note,
        raw_text=raw,
        normalized_text=raw.lower(),
        value={
            "number": note.number,
            "text": note.text,
            "page": note.page_index,
            "sheet_number": note.sheet_number,
            "callout_count": len(note.callout_bboxes),
        },
        entity_keys=[f"keyed_note:{note.page_index}:{note.number}"],
        source_refs=[src],
        authority_class=AuthorityClass.machine_extractor,
        confidence=note.confidence,
        review_status=ReviewStatus.auto_accepted,
        parser_version=parser_version,
    )


def emit_warning_atom(
    *,
    warning: SchematicWarning,
    project_id: str,
    artifact_id: str,
    filename: str,
    parser_version: str,
    page: Any | None = None,
) -> EvidenceAtom:
    base_loc = warning.locator_dict()
    bbox_raw = base_loc.get("bbox")
    bbox_tuple: tuple[float, float, float, float] | None = None
    if isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) == 4:
        bbox_tuple = (float(bbox_raw[0]), float(bbox_raw[1]), float(bbox_raw[2]), float(bbox_raw[3]))
    locator = build_replayable_locator(
        page_index=warning.page_index,
        bbox=bbox_tuple,
        page=page,
        extras={k: v for k, v in base_loc.items() if k not in {"bbox", "bbox_units", "page"}},
    )
    src = _source_ref(
        artifact_id=artifact_id,
        filename=filename,
        locator=locator,
        parser_version=parser_version,
        extraction_method="schematic_warning",
        suffix=warning.warning_id,
    )
    raw = f"[{warning.warning_type}] page {warning.page_index}: {warning.detail}"
    return EvidenceAtom(
        id=stable_id("atom_schematic_warning", artifact_id, warning.warning_id),
        project_id=project_id,
        artifact_id=artifact_id,
        atom_type=AtomType.schematic_warning,
        raw_text=raw,
        normalized_text=raw.lower(),
        value={
            "warning_id": warning.warning_id,
            "warning_type": warning.warning_type,
            "page": warning.page_index,
            "sheet_number": warning.sheet_number,
            "legend_id": warning.legend_id,
            "legend_entry_id": warning.legend_entry_id,
            "target_key": warning.target_key,
            "detail": warning.detail,
        },
        entity_keys=[f"schematic_warning:{warning.warning_type}"],
        source_refs=[src],
        authority_class=AuthorityClass.machine_extractor,
        confidence=0.5,
        review_status=ReviewStatus.needs_review,
        parser_version=parser_version,
    )


def emit_detection_atom(
    *,
    detection: SymbolDetection,
    project_id: str,
    artifact_id: str,
    filename: str,
    parser_version: str,
) -> EvidenceAtom:
    """Project a ``SymbolDetection`` onto a ``schematic_symbol_detection`` atom.

    The ``SourceRef.locator`` carries page + bbox in PDF points and
    the deterministic 200-DPI crop hash so source_replay can verify.
    """
    locator = detection.locator_dict()
    locator["bbox_units"] = BBOX_UNITS_PDF_POINTS
    src = _source_ref(
        artifact_id=artifact_id,
        filename=filename,
        locator=locator,
        parser_version=parser_version,
        extraction_method="schematic_symbol_detector",
        suffix=detection.detection_id,
    )
    raw = (
        f"{detection.target_key} @ page {detection.page_index}"
        f" sheet {detection.sheet_number or '?'} via {detection.modality}"
        + (f' near "{detection.nearby_text}"' if detection.nearby_text else "")
    )
    return EvidenceAtom(
        id=stable_id("atom_schematic_detection", artifact_id, detection.detection_id),
        project_id=project_id,
        artifact_id=artifact_id,
        atom_type=AtomType.schematic_symbol_detection,
        raw_text=raw,
        normalized_text=raw.lower(),
        value={
            "detection_id": detection.detection_id,
            "page": detection.page_index,
            "sheet_number": detection.sheet_number,
            "target_key": detection.target_key,
            "entity_key": detection.entity_key,
            "legend_entry_id": detection.legend_entry_id,
            "modality": detection.modality,
            "bbox": list(detection.bbox_pdf),
            "crop_sha256": detection.crop_sha256,
            "nearby_text": detection.nearby_text,
        },
        entity_keys=[detection.entity_key],
        source_refs=[src],
        authority_class=AuthorityClass.machine_extractor,
        confidence=detection.confidence,
        review_status=ReviewStatus.auto_accepted,
        parser_version=parser_version,
    )


def emit_detected_count_atom(
    *,
    page_index: int,
    sheet_number: str | None,
    target: DetectionTarget,
    detections: list[SymbolDetection],
    project_id: str,
    artifact_id: str,
    filename: str,
    parser_version: str,
) -> EvidenceAtom | None:
    """Aggregate a list of same-target detections into a ``quantity`` atom.

    Returns ``None`` when ``detections`` is empty so the parser
    does not emit zero-count quantity atoms that the conflict
    gate would then have to filter out.  Each emitted atom
    carries bbox provenance pointing at the *first* detection's
    crop region — that satisfies the schematic same-artifact
    exception in the packetizer.
    """
    if not detections:
        return None
    primary = detections[0]
    locator = {
        "page": page_index,
        "sheet_number": sheet_number,
        "bbox": list(primary.bbox_pdf),
        "bbox_units": BBOX_UNITS_PDF_POINTS,
        "crop_sha256": primary.crop_sha256,
        "schematic_target_key": target.target_key,
        "schematic_role": "detected",
    }
    src = _source_ref(
        artifact_id=artifact_id,
        filename=filename,
        locator=locator,
        parser_version=parser_version,
        extraction_method="schematic_detected_count",
        suffix=f"{page_index}:{target.target_key}",
    )
    raw = f"Detected count for {target.target_key} on page {page_index} = {len(detections)}"
    return EvidenceAtom(
        id=stable_id(
            "atom_schematic_quantity_detected",
            artifact_id,
            page_index,
            target.target_key,
        ),
        project_id=project_id,
        artifact_id=artifact_id,
        atom_type=AtomType.quantity,
        raw_text=raw,
        normalized_text=raw.lower(),
        value={
            "quantity": len(detections),
            "schematic_target_key": target.target_key,
            "schematic_role": "detected",
            "schematic_sheet_number": sheet_number,
            "schematic_page": page_index,
            "detection_ids": [d.detection_id for d in detections],
        },
        entity_keys=sorted({
            target.entity_key,
            f"schematic_count:{target.target_key}",
            *target.parent_entity_keys,
        }),
        source_refs=[src],
        authority_class=AuthorityClass.machine_extractor,
        confidence=0.9,
        review_status=ReviewStatus.auto_accepted,
        parser_version=parser_version,
    )


def emit_declared_count_atom(
    *,
    page_index: int,
    sheet_number: str | None,
    target: DetectionTarget,
    declared_count: float,
    entry: ParsedLegendEntry,
    project_id: str,
    artifact_id: str,
    filename: str,
    parser_version: str,
    page: Any | None = None,
) -> EvidenceAtom | None:
    """Emit a ``quantity`` atom for a declared count from a legend row.

    Returns ``None`` when the legend entry has no real symbol bbox to
    pin against. The earlier implementation faked a ``(0, 0, 1, 1)``
    bbox so the locator would always look replayable; that produced
    a crop hash for a 1-pt corner of the page which no future review
    could re-verify. Callers should react to ``None`` by emitting a
    ``schematic_warning`` instead so the count is still surfaced but
    not laundered through fake provenance.

    The atom carries ``schematic_role="declared"`` so the graph
    builder pairs it with a same-(sheet, target) detected atom and
    emits a schematic quantity contradiction edge when they disagree.
    """
    bbox = entry.symbol_bbox_pdf
    if bbox is None:
        return None
    locator = build_replayable_locator(
        page_index=page_index,
        bbox=bbox,
        page=page,
        extras={
            "sheet_number": sheet_number,
            "schematic_target_key": target.target_key,
            "schematic_role": "declared",
            "legend_entry_id": entry.entry_id,
        },
    )
    # The atom must carry a real crop hash; if compute failed, refuse
    # to emit so the packetizer's narrow same-artifact exception
    # cannot certify a conflict without verifiable provenance.
    if "crop_sha256" not in locator:
        return None
    src = _source_ref(
        artifact_id=artifact_id,
        filename=filename,
        locator=locator,
        parser_version=parser_version,
        extraction_method="schematic_declared_count",
        suffix=f"{page_index}:{target.target_key}:{entry.entry_id}",
    )
    raw = (
        f"Declared count for {target.target_key} on sheet "
        f"{sheet_number or '?'} = {declared_count}"
    )
    return EvidenceAtom(
        id=stable_id(
            "atom_schematic_quantity_declared",
            artifact_id,
            page_index,
            target.target_key,
            entry.entry_id,
        ),
        project_id=project_id,
        artifact_id=artifact_id,
        atom_type=AtomType.quantity,
        raw_text=raw,
        normalized_text=raw.lower(),
        value={
            "quantity": declared_count,
            "schematic_target_key": target.target_key,
            "schematic_role": "declared",
            "schematic_sheet_number": sheet_number,
            "schematic_page": page_index,
            "legend_entry_id": entry.entry_id,
        },
        entity_keys=sorted({
            target.entity_key,
            f"schematic_count:{target.target_key}",
            *target.parent_entity_keys,
        }),
        source_refs=[src],
        authority_class=AuthorityClass.machine_extractor,
        confidence=0.8,
        review_status=ReviewStatus.auto_accepted,
        parser_version=parser_version,
    )


def collect_all(atoms: Iterable[EvidenceAtom]) -> list[EvidenceAtom]:
    """Stable-sort schematic atoms for deterministic emission order.

    Total sort key (each tier is a fallback when the previous tier ties):

      1. atom-type priority
      2. page (drawn from ``value['page']``, ``value['schematic_page']``,
         or the first ``SourceRef.locator['page']`` — quantity atoms
         carry ``schematic_page`` rather than ``page``, and a flat
         ``value.get('page', 0)`` lookup would collapse them all to 0)
      3. sheet number (so multi-sheet drawings stay grouped by sheet)
      4. target key (load-bearing detections sort by target alphabetically)
      5. atom type value (locked-in tiebreaker between same-priority types)
      6. atom id (final guaranteed tiebreaker)
    """
    pri = {
        AtomType.schematic_sheet_metadata: 0,
        AtomType.schematic_legend: 1,
        AtomType.schematic_room: 2,
        AtomType.schematic_keyed_note: 3,
        AtomType.schematic_note_callout: 4,
        AtomType.schematic_detection_target_set: 5,
        AtomType.schematic_symbol_detection: 6,
        AtomType.quantity: 7,
        AtomType.schematic_warning: 8,
    }

    def _page(atom: EvidenceAtom) -> int:
        value = atom.value if isinstance(atom.value, dict) else {}
        for key in ("page", "schematic_page"):
            v = value.get(key)
            if isinstance(v, int):
                return v
        for src in atom.source_refs or []:
            loc = src.locator if isinstance(src.locator, dict) else {}
            v = loc.get("page")
            if isinstance(v, int):
                return v
        return 0

    def _sheet(atom: EvidenceAtom) -> str:
        value = atom.value if isinstance(atom.value, dict) else {}
        for key in ("sheet_number", "schematic_sheet_number"):
            v = value.get(key)
            if isinstance(v, str):
                return v
        return ""

    def _target(atom: EvidenceAtom) -> str:
        value = atom.value if isinstance(atom.value, dict) else {}
        v = value.get("target_key") or value.get("schematic_target_key")
        return str(v) if v else ""

    return sorted(
        atoms,
        key=lambda a: (
            pri.get(a.atom_type, 99),
            _page(a),
            _sheet(a),
            _target(a),
            a.atom_type.value,
            a.id,
        ),
    )
