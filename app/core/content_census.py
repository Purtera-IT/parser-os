"""Content census: a per-modality region inventory + coverage invariant.

This generalizes the span ledger (``app/core/span_ledger.py``) from *text
spans* to *regions of any modality*. The span ledger answers "which detected
spans did a stage drop, and was that a GATE or a SEAM?" — but it can only see
content the parser already iterated over. The census closes the deeper gap: it
inventories **every region of every kind** in an artifact using a reader
*independent of the extractor*, then enforces one invariant:

    every region must be COVERED (produced >=1 atom) or MARKED (produced a
    "needs OCR / vision / manual review" marker) — never UNCOVERED.

An UNCOVERED region is silent loss: content that exists in the file but the
parser neither extracted nor flagged. Because the census denominator comes
from an independent reader, this catches the **never-detected** class —
content controls, textboxes, headers/footers, embedded images, OLE objects,
schematic regions — that the parser's own field of view can't see.

Detection vs extraction
-----------------------
Detection (this census) can be made *total and guaranteed* across modalities:
we can always know a region exists, even a schematic we cannot yet read.
Extraction quality is a separate, improving frontier. The MARKED status is the
bridge: an un-extracted region never vanishes; it becomes a located marker the
PM sees ("image/schematic here, vision pass required").
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum


class RegionKind(str, Enum):
    """The kind of content a region holds. Modality-agnostic."""

    TEXT = "text"                     # paragraph / cell / text-run
    TABLE = "table"                   # structured grid
    IMAGE = "image"                   # raster/vector picture (needs OCR/vision)
    CHART = "chart"                   # plotted data (needs data extraction)
    EMBEDDED_OBJECT = "embedded"      # OLE / nested artifact -> recurse
    SHAPE = "shape"                   # drawing / textbox / SmartArt
    CONTENT_CONTROL = "content_control"  # w:sdt and similar
    HEADER_FOOTER = "header_footer"
    NOTE = "note"                     # speaker notes / footnotes / comments
    OTHER = "other"


class CoverageStatus(str, Enum):
    COVERED = "covered"        # produced >=1 atom
    MARKED = "marked"          # produced a needs-review marker (e.g. image)
    UNCOVERED = "uncovered"    # silent loss -> invariant violation


_TEXT_KINDS = {RegionKind.TEXT, RegionKind.TABLE, RegionKind.HEADER_FOOTER, RegionKind.NOTE}


@dataclass(frozen=True)
class Region:
    """One inventoried region of an artifact, from an independent reader."""

    region_id: str
    artifact: str
    kind: RegionKind
    location: str          # human-readable: "body/p12", "sdt/contacts", "media/image1.png"
    text: str = ""         # for text-bearing regions (the reconciliation key)
    note: str = ""         # extra detail for binary regions

    @property
    def modality(self) -> str:
        return "text" if self.kind in _TEXT_KINDS else "binary"


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


@dataclass
class ContentCensus:
    """Region inventory for one artifact plus the coverage invariant.

    Populate with ``register`` (from an independent reader), then call
    ``reconcile`` with the parser's emitted atoms. Reconciliation is by
    *content*, not parser internals, so the census stays an honest,
    independent denominator.
    """

    artifact: str
    regions: dict[str, Region] = field(default_factory=dict)
    _covered: set[str] = field(default_factory=set)
    _marked: set[str] = field(default_factory=set)

    # -- inventory -------------------------------------------------------
    def register(self, region: Region) -> None:
        self.regions[region.region_id] = region

    # -- reconciliation against emitted atoms ----------------------------
    def reconcile(self, atoms: list) -> None:
        """Decide each region's status from the atoms the parser produced.

        * text region  -> COVERED if its text (or a 40-char chunk) appears in
          some atom; else UNCOVERED.
        * binary region (image/embedded/chart/shape) -> MARKED if some marker
          atom references it (a "needs OCR/vision/manual review" atom);
          COVERED if a real atom references it; else UNCOVERED.
        """
        atom_blob = " || ".join(_norm(getattr(a, "raw_text", "") or "") for a in atoms)
        marker_refs: set[str] = set()
        real_refs: set[str] = set()
        for a in atoms:
            val = getattr(a, "value", None) or {}
            ref = None
            if isinstance(val, dict):
                ref = val.get("region_ref") or val.get("media") or val.get("filename")
            is_marker = isinstance(val, dict) and "marker" in str(val.get("kind", ""))
            if ref:
                (marker_refs if is_marker else real_refs).add(str(ref))

        for rid, region in self.regions.items():
            if region.modality == "text":
                t = _norm(region.text)
                if t and (t in atom_blob or t[:40] in atom_blob):
                    self._covered.add(rid)
            else:
                loc = region.location
                if loc in real_refs or rid in real_refs:
                    self._covered.add(rid)
                elif loc in marker_refs or rid in marker_refs:
                    self._marked.add(rid)

    # -- analysis --------------------------------------------------------
    def status(self, region_id: str) -> CoverageStatus:
        if region_id in self._covered:
            return CoverageStatus.COVERED
        if region_id in self._marked:
            return CoverageStatus.MARKED
        return CoverageStatus.UNCOVERED

    def uncovered(self) -> list[Region]:
        return [r for rid, r in self.regions.items() if self.status(rid) is CoverageStatus.UNCOVERED]

    def invariant_ok(self) -> bool:
        """True iff every region is COVERED or MARKED (nothing silently lost)."""
        return not self.uncovered()

    def coverage_by_kind(self) -> dict[str, tuple[int, int]]:
        """{kind: (accounted_for, total)} where accounted = covered or marked."""
        out: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for rid, r in self.regions.items():
            out[r.kind.value][1] += 1
            if self.status(rid) is not CoverageStatus.UNCOVERED:
                out[r.kind.value][0] += 1
        return {k: (v[0], v[1]) for k, v in out.items()}

    # -- reporting -------------------------------------------------------
    def report(self, *, width: int = 95, sample: int = 8) -> str:
        lines = ["=" * width, f"CONTENT CENSUS  —  {self.artifact}", "=" * width]
        total = len(self.regions)
        unc = self.uncovered()
        accounted = total - len(unc)
        verdict = "PASS" if self.invariant_ok() else "*** FAIL (silent loss) ***"
        lines.append(
            f"coverage invariant: {accounted}/{total} regions accounted for "
            f"(covered or marked)  ->  {verdict}"
        )
        lines.append("")
        lines.append(f"{'kind':18}{'accounted':>12}{'total':>8}")
        lines.append("-" * 40)
        for kind, (acc, tot) in sorted(self.coverage_by_kind().items()):
            lines.append(f"{kind:18}{acc:>12}{tot:>8}")
        lines.append("")
        if unc:
            lines.append("-" * width)
            lines.append(f"UNCOVERED REGIONS (silent loss): {len(unc)}")
            lines.append("-" * width)
            for r in unc[:sample]:
                desc = r.text or r.note or "(binary)"
                lines.append(f"  [{r.kind.value}] {r.location}: {desc[:width - 25]}")
            if len(unc) > sample:
                lines.append(f"  ... +{len(unc) - sample} more")
        return "\n".join(lines)
