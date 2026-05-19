"""Project schematic dataclasses onto ``EvidenceAtom`` instances.

Used by ``OrbitbriefPdfParser`` (PR5) and the symbol detector (PR6).
Each emitter is small and side-effect-free; the caller decides where
the produced atoms go in the ``ParserOutput`` stream.
"""
from __future__ import annotations

from typing import Iterable

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
    DetectionTarget,
    DetectionTargetSet,
    ParsedLegend,
    ParsedLegendEntry,
    SchematicWarning,
    SymbolDetection,
)


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
) -> EvidenceAtom:
    locator = {
        "page": legend.page_index,
        "sheet_number": legend.sheet_number,
        "legend_id": legend.legend_id,
        "scope": legend.scope,
        **legend.locator_dict(),
    }
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
) -> EvidenceAtom:
    locator = {
        "page": target_set.page_index,
        "sheet_number": target_set.sheet_number,
        "pack_id": target_set.pack_id,
        "legend_id": target_set.legend_id,
    }
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


def emit_warning_atom(
    *,
    warning: SchematicWarning,
    project_id: str,
    artifact_id: str,
    filename: str,
    parser_version: str,
) -> EvidenceAtom:
    locator = warning.locator_dict()
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
        entity_keys=[target.entity_key, f"schematic_count:{target.target_key}"],
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
) -> EvidenceAtom:
    """Emit a ``quantity`` atom for a declared count from a legend row.

    The bbox is the legend entry's symbol bbox (where the count column
    sat on the legend sheet). It carries ``schematic_role="declared"``
    so the graph builder pairs it with a same-(sheet, target) detected
    atom and emits a schematic quantity contradiction edge when they
    disagree.
    """
    bbox = entry.symbol_bbox_pdf or (0.0, 0.0, 1.0, 1.0)
    locator = {
        "page": page_index,
        "sheet_number": sheet_number,
        "bbox": list(bbox),
        "bbox_units": BBOX_UNITS_PDF_POINTS,
        "schematic_target_key": target.target_key,
        "schematic_role": "declared",
        "legend_entry_id": entry.entry_id,
    }
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
        entity_keys=[target.entity_key, f"schematic_count:{target.target_key}"],
        source_refs=[src],
        authority_class=AuthorityClass.machine_extractor,
        confidence=0.8,
        review_status=ReviewStatus.auto_accepted,
        parser_version=parser_version,
    )


def collect_all(atoms: Iterable[EvidenceAtom]) -> list[EvidenceAtom]:
    """Stable-sort schematic atoms for deterministic emission order.

    PDF parser appends schematic atoms after structured atoms; this
    helper keeps the schematic block internally sorted so two compiles
    of the same PDF produce byte-identical envelopes.
    """
    pri = {
        AtomType.schematic_legend: 0,
        AtomType.schematic_detection_target_set: 1,
        AtomType.schematic_symbol_detection: 2,
        AtomType.quantity: 4,
        AtomType.schematic_warning: 5,
    }
    return sorted(
        atoms,
        key=lambda a: (pri.get(a.atom_type, 99), a.value.get("page", 0), a.id),
    )
