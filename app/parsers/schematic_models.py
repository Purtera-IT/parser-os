"""Schematic parsing data contracts (PR1 of the schematic upgrade).

Frozen, hashable dataclasses for the legend-first schematic pipeline.
They are intentionally separate from the Pydantic atom models in
``app.core.schemas`` so the page-OS layer can build them without
introducing a Pydantic dependency on every legend parser inner loop.

The pipeline shape:

    PDF page
      ├─ legend located + parsed       → ParsedLegend (entries=[ParsedLegendEntry, ...])
      ├─ targets resolved              → list[DetectionTarget]  (from legend ∩ pack.detection_targets)
      ├─ symbols detected              → list[SymbolDetection]
      └─ failures/warnings             → list[SchematicWarning]

Each of these is later projected into an ``EvidenceAtom`` (atom_type
in ``schematic_legend`` / ``schematic_detection_target_set`` /
``schematic_symbol_detection`` / ``schematic_warning``). The atom's
``SourceRef.locator`` carries::

    {
      "page": <int, 0-based PDF page index>,
      "sheet_number": <str | None>,         # e.g. "T0.01"
      "bbox": [x0, y0, x1, y1],             # PDF points, origin top-left
      "bbox_units": "pdf_points",
      "crop_sha256": <hex string | None>,   # deterministic 200-dpi crop hash
      "legend_id": <str | None>,
      "legend_entry_id": <str | None>,
    }

so that ``source_replay`` can re-render the page at a fixed DPI, crop
the bbox, and hash-verify the receipt independently of any LLM.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from app.core.ids import stable_id


# Public constants reused by parser, detector, and source_replay so a
# single change point governs the deterministic render contract.
BBOX_UNITS_PDF_POINTS = "pdf_points"
SCHEMATIC_REPLAY_DPI = 200
SCHEMATIC_REPLAY_PADDING_PT = 0.0


# ─────────────────────────── legend ────────────────────────────────


@dataclass(frozen=True)
class ParsedLegendEntry:
    """A single row inside a parsed schematic legend.

    ``raw_symbol_text`` is the literal symbol token if the legend cell
    is text-only (``WN``, ``CR``); ``symbol_crop_sha256`` is the
    deterministic crop hash when the symbol is a glyph/swatch. Both
    are optional but at least one must be present for the entry to be
    classified.

    ``attributes`` carries the additional columns that real construction
    legends use beyond symbol/description/count — mounting height,
    cable count, work-area termination, closet termination, rough-in,
    power requirement, NIC marker, manufacturer/model, etc. Keys are
    canonical normalized strings ("mounting_height", "cable_count",
    "rough_in", "power", "remarks", "nic", "mfg", "model"); values
    are the raw text from the cell.
    """

    entry_id: str
    label_text: str
    normalized_label: str
    raw_symbol_text: str | None = None
    normalized_symbol_text: str | None = None
    symbol_bbox_pdf: tuple[float, float, float, float] | None = None
    symbol_crop_sha256: str | None = None
    vector_fingerprint: str | None = None
    count_column: float | None = None
    notes: tuple[str, ...] = ()
    attributes: tuple[tuple[str, str], ...] = ()
    source_ref_locator: tuple[tuple[str, Any], ...] = ()
    confidence: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence out of range: {self.confidence!r}")
        if self.raw_symbol_text is None and self.symbol_crop_sha256 is None:
            raise ValueError("ParsedLegendEntry needs raw_symbol_text or symbol_crop_sha256")

    @classmethod
    def make(
        cls,
        *,
        page_index: int,
        label_text: str,
        normalized_label: str,
        raw_symbol_text: str | None = None,
        normalized_symbol_text: str | None = None,
        symbol_bbox_pdf: tuple[float, float, float, float] | None = None,
        symbol_crop_sha256: str | None = None,
        vector_fingerprint: str | None = None,
        count_column: float | None = None,
        notes: tuple[str, ...] = (),
        attributes: dict[str, str] | None = None,
        source_ref_locator: dict[str, Any] | None = None,
        confidence: float = 0.0,
    ) -> "ParsedLegendEntry":
        entry_id = stable_id(
            "legend_entry",
            page_index,
            normalized_symbol_text or "",
            normalized_label,
            symbol_crop_sha256 or "",
        )
        locator_pairs: tuple[tuple[str, Any], ...] = tuple(
            sorted((str(k), v) for k, v in (source_ref_locator or {}).items())
        )
        attr_pairs: tuple[tuple[str, str], ...] = tuple(
            sorted((str(k), str(v)) for k, v in (attributes or {}).items())
        )
        return cls(
            entry_id=entry_id,
            label_text=label_text,
            normalized_label=normalized_label,
            raw_symbol_text=raw_symbol_text,
            normalized_symbol_text=normalized_symbol_text,
            symbol_bbox_pdf=symbol_bbox_pdf,
            symbol_crop_sha256=symbol_crop_sha256,
            vector_fingerprint=vector_fingerprint,
            count_column=count_column,
            notes=tuple(notes),
            attributes=attr_pairs,
            source_ref_locator=locator_pairs,
            confidence=confidence,
        )

    def locator_dict(self) -> dict[str, Any]:
        return dict(self.source_ref_locator)

    def attributes_dict(self) -> dict[str, str]:
        return dict(self.attributes)


LegendScope = Literal["page", "sheet", "sheet_range", "global", "continuation"]


@dataclass(frozen=True)
class ParsedLegend:
    """All entries from a single legend block on a single page.

    A legend's ``scope`` tells the resolver whether it applies only to
    the page it lives on, to a single sheet, to a range of sheets, to
    the entire document (a global legend sheet like T0.01), or whether
    it is a continuation block referring back to another legend.
    """

    legend_id: str
    page_index: int
    sheet_number: str | None
    title: str | None
    scope: LegendScope
    entries: tuple[ParsedLegendEntry, ...]
    continuation_refs: tuple[str, ...] = ()
    source_ref_locator: tuple[tuple[str, Any], ...] = ()
    confidence: float = 0.0
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence out of range: {self.confidence!r}")

    @classmethod
    def make(
        cls,
        *,
        page_index: int,
        sheet_number: str | None,
        title: str | None,
        scope: LegendScope,
        entries: tuple[ParsedLegendEntry, ...] | list[ParsedLegendEntry],
        continuation_refs: tuple[str, ...] = (),
        source_ref_locator: dict[str, Any] | None = None,
        confidence: float = 0.0,
        warnings: tuple[str, ...] = (),
    ) -> "ParsedLegend":
        entries_tuple = tuple(entries)
        legend_id = stable_id(
            "legend",
            page_index,
            sheet_number or "",
            title or "",
            scope,
            tuple(e.entry_id for e in entries_tuple),
        )
        locator_pairs: tuple[tuple[str, Any], ...] = tuple(
            sorted((str(k), v) for k, v in (source_ref_locator or {}).items())
        )
        return cls(
            legend_id=legend_id,
            page_index=page_index,
            sheet_number=sheet_number,
            title=title,
            scope=scope,
            entries=entries_tuple,
            continuation_refs=tuple(continuation_refs),
            source_ref_locator=locator_pairs,
            confidence=confidence,
            warnings=tuple(warnings),
        )

    def locator_dict(self) -> dict[str, Any]:
        return dict(self.source_ref_locator)


# ─────────────────────────── targets ───────────────────────────────


Completeness = Literal["load_bearing", "informational"]
Modality = Literal["text_tag", "glyph_template", "vector_shape", "zone", "line_run"]


@dataclass(frozen=True)
class DetectionTarget:
    """A single symbol the parser must hunt for on a drawing page.

    ``target_key`` is pack-local (``security_camera.ptz``);
    ``entity_key`` is the canonical entity slug emitted on detection
    atoms; ``ontology_key`` optionally pins the target to a node in
    the active pack's ontology YAML; ``legend_entry_id`` links back to
    the row of the parsed legend that established the target for this
    drawing page (when resolved from the legend rather than purely
    from the pack); ``parent_entity_keys`` lets a subtype roll up
    to a broader bucket so cross-artifact quantity conflicts catch
    BOMs that use the parent key.
    """

    target_key: str
    entity_key: str
    completeness: Completeness
    expected_modalities: tuple[Modality, ...]
    ontology_key: str | None = None
    legend_entry_id: str | None = None
    aliases: tuple[str, ...] = ()
    parent_entity_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.target_key:
            raise ValueError("DetectionTarget.target_key cannot be empty")
        if not self.entity_key:
            raise ValueError("DetectionTarget.entity_key cannot be empty")
        if not self.expected_modalities:
            raise ValueError("DetectionTarget needs at least one expected_modality")


@dataclass(frozen=True)
class DetectionTargetSet:
    """The full set of targets the parser will look for on one drawing page.

    The set is the deterministic intersection of:
      - the parsed legend (legend rows mapped to pack targets), and
      - the active domain pack's ``detection_targets`` for any
        load-bearing targets the legend omitted (those omissions
        surface as ``legend_gap`` warnings instead of vanishing).
    """

    page_index: int
    sheet_number: str | None
    pack_id: str
    legend_id: str | None
    targets: tuple[DetectionTarget, ...]
    legend_gap_target_keys: tuple[str, ...] = ()

    @classmethod
    def make(
        cls,
        *,
        page_index: int,
        sheet_number: str | None,
        pack_id: str,
        legend_id: str | None,
        targets: tuple[DetectionTarget, ...] | list[DetectionTarget],
        legend_gap_target_keys: tuple[str, ...] = (),
    ) -> "DetectionTargetSet":
        targets_tuple = tuple(sorted(targets, key=lambda t: t.target_key))
        return cls(
            page_index=page_index,
            sheet_number=sheet_number,
            pack_id=pack_id,
            legend_id=legend_id,
            targets=targets_tuple,
            legend_gap_target_keys=tuple(sorted(legend_gap_target_keys)),
        )


# ─────────────────────────── detection ─────────────────────────────


@dataclass(frozen=True)
class SymbolDetection:
    """One detected instance of a target symbol on a drawing body.

    A detection always carries a bbox in PDF points and a 200-DPI
    crop hash so ``source_replay`` can re-render the page and verify
    that the pixels at this region still hash the same way.
    """

    detection_id: str
    page_index: int
    sheet_number: str | None
    target_key: str
    entity_key: str
    legend_entry_id: str | None
    bbox_pdf: tuple[float, float, float, float]
    crop_sha256: str
    modality: Modality
    confidence: float
    nearby_text: str | None = None
    extras: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence out of range: {self.confidence!r}")
        x0, y0, x1, y1 = self.bbox_pdf
        if not (x1 > x0 and y1 > y0):
            raise ValueError(f"bbox not strictly positive: {self.bbox_pdf!r}")

    @classmethod
    def make(
        cls,
        *,
        page_index: int,
        sheet_number: str | None,
        target_key: str,
        entity_key: str,
        legend_entry_id: str | None,
        bbox_pdf: tuple[float, float, float, float],
        crop_sha256: str,
        modality: Modality,
        confidence: float,
        nearby_text: str | None = None,
        extras: dict[str, Any] | None = None,
    ) -> "SymbolDetection":
        detection_id = stable_id(
            "schematic_detection",
            page_index,
            target_key,
            tuple(round(float(v), 3) for v in bbox_pdf),
            crop_sha256,
            modality,
        )
        return cls(
            detection_id=detection_id,
            page_index=page_index,
            sheet_number=sheet_number,
            target_key=target_key,
            entity_key=entity_key,
            legend_entry_id=legend_entry_id,
            bbox_pdf=tuple(float(v) for v in bbox_pdf),  # type: ignore[arg-type]
            crop_sha256=crop_sha256,
            modality=modality,
            confidence=confidence,
            nearby_text=nearby_text,
            extras=tuple(sorted((str(k), v) for k, v in (extras or {}).items())),
        )

    def locator_dict(self) -> dict[str, Any]:
        return {
            "page": self.page_index,
            "sheet_number": self.sheet_number,
            "bbox": list(self.bbox_pdf),
            "bbox_units": BBOX_UNITS_PDF_POINTS,
            "crop_sha256": self.crop_sha256,
            "legend_entry_id": self.legend_entry_id,
            "target_key": self.target_key,
        }


# ─────────────────────────── warnings ──────────────────────────────


WarningType = Literal[
    "missing_legend",
    "weak_legend",
    "legend_gap",
    "legend_orphan",
    "unknown_symbol",
    "ambiguous_legend_reference",
    "unresolved_legend_reference",
    "ocr_unavailable",
    "low_ocr_confidence",
    "schematic_quantity_contradiction",
]


@dataclass(frozen=True)
class SchematicWarning:
    """A structured failure-mode signal emitted alongside schematic atoms.

    The PDF parser turns each warning into a ``schematic_warning``
    atom with ``review_status=needs_review``; downstream consumers
    (gold-compare, debug overlays) can group by ``warning_type``.
    """

    warning_id: str
    warning_type: WarningType
    page_index: int
    sheet_number: str | None
    detail: str
    bbox_pdf: tuple[float, float, float, float] | None = None
    legend_id: str | None = None
    legend_entry_id: str | None = None
    target_key: str | None = None
    extras: tuple[tuple[str, Any], ...] = ()

    @classmethod
    def make(
        cls,
        *,
        warning_type: WarningType,
        page_index: int,
        sheet_number: str | None,
        detail: str,
        bbox_pdf: tuple[float, float, float, float] | None = None,
        legend_id: str | None = None,
        legend_entry_id: str | None = None,
        target_key: str | None = None,
        extras: dict[str, Any] | None = None,
    ) -> "SchematicWarning":
        warning_id = stable_id(
            "schematic_warning",
            warning_type,
            page_index,
            sheet_number or "",
            target_key or "",
            legend_entry_id or "",
            detail,
        )
        return cls(
            warning_id=warning_id,
            warning_type=warning_type,
            page_index=page_index,
            sheet_number=sheet_number,
            detail=detail,
            bbox_pdf=tuple(float(v) for v in bbox_pdf) if bbox_pdf else None,  # type: ignore[arg-type]
            legend_id=legend_id,
            legend_entry_id=legend_entry_id,
            target_key=target_key,
            extras=tuple(sorted((str(k), v) for k, v in (extras or {}).items())),
        )

    def locator_dict(self) -> dict[str, Any]:
        loc: dict[str, Any] = {"page": self.page_index, "warning_type": self.warning_type}
        if self.sheet_number:
            loc["sheet_number"] = self.sheet_number
        if self.bbox_pdf:
            loc["bbox"] = list(self.bbox_pdf)
            loc["bbox_units"] = BBOX_UNITS_PDF_POINTS
        if self.legend_id:
            loc["legend_id"] = self.legend_id
        if self.legend_entry_id:
            loc["legend_entry_id"] = self.legend_entry_id
        if self.target_key:
            loc["target_key"] = self.target_key
        return loc


# ─────────────────────────── helpers ───────────────────────────────


def crop_sha256_of_pixels(pixels: bytes, width: int, height: int, channels: int) -> str:
    """Deterministic hash of a rendered crop.

    Hashes the raw raster bytes prefixed by a header so two different
    crops with identical pixel payloads but different dimensions never
    collide. Used by both the symbol detector (at emit time) and
    source_replay (at verify time).
    """
    import hashlib

    header = f"schematic_crop_v1|{width}x{height}x{channels}|".encode("utf-8")
    return hashlib.sha256(header + pixels).hexdigest()
