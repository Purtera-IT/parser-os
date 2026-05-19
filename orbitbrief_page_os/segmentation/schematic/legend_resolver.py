"""Document-level legend resolution (PR4 of the schematic upgrade).

A drawing set typically has many sheets and only one or two legend
sheets. Floor plans, riser diagrams, and detail sheets all need to
know which legend applies. The resolver builds the cross-sheet index
once per compile and answers the question:

    for drawing page N, what legend governs my symbol detections?

Resolution priority (deterministic — first hit wins):

1. **In-page legend** (a ``ParsedLegend`` whose ``page_index`` is N).
2. **Explicit reference** (``see sheet E-001 for legend`` on this page
   or a continuation block ``symbols continued from sheet T0.01``).
3. **Same discipline global legend** (a legend on a sheet whose
   number shares the discipline prefix — ``T`` / ``E`` / ``FA`` /
   ``AC`` / ``SC`` / ``AV`` / ``M`` — with this page's sheet number).
4. **Project-global legend** (a single legend whose ``scope`` is
   ``global``).

If two or more legends tie at the same priority, the lowest
``page_index`` wins and an ``ambiguous_legend_reference`` warning is
emitted. If no legend resolves, a ``missing_legend`` warning is
emitted instead — and the drawing page is *not* given a detection
target set in PR5.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from app.parsers.schematic_models import ParsedLegend, SchematicWarning
from orbitbrief_page_os.segmentation.schematic.legend_locator import (
    LegendCandidate,
    TextBlock,
)


_SHEET_NUMBER_RE = re.compile(
    r"\b([A-Z]{1,3})[\s\-_]?([0-9]+\.[0-9]+|[0-9]{1,4})\b"
)
_DRAWING_INDEX_HEADER_RE = re.compile(
    r"\b(drawing\s+index|sheet\s+index|sheet\s+list)\b", re.IGNORECASE
)
_INDEX_ROW_RE = re.compile(
    r"^\s*([A-Z]{1,3}[\s\-]?[0-9.]+)\s+(.+?)\s*$"
)
_LEGEND_REF_INLINE_RE = re.compile(
    r"see\s+(?:sheet|dwg|drawing)\s+([A-Z]+[\s\-_]?[0-9.]+)\s+(?:for\s+)?(?:legend|symbols?)",
    re.IGNORECASE,
)
_LEGEND_CONTINUED_RE = re.compile(
    r"(?:symbols?|legend)\s+continued\s+from\s+sheet\s+([A-Z]+[\s\-_]?[0-9.]+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ResolvedLegend:
    """The legend chosen for a drawing page, plus how we got there.

    ``priority`` mirrors the order list above (1=in-page, 4=global)
    so callers can record receipts in derived files. ``warnings``
    contains structured warnings the resolver emitted while making
    the decision (e.g. ``ambiguous_legend_reference``).
    """

    page_index: int
    sheet_number: str | None
    legend: ParsedLegend | None
    priority: int
    rationale: str
    warnings: tuple[SchematicWarning, ...]


def _norm_sheet(token: str | None) -> str | None:
    if not token:
        return None
    t = token.strip().upper().replace(" ", "").replace("-", "").replace("_", "")
    return t or None


def _discipline_prefix(sheet: str | None) -> str | None:
    if not sheet:
        return None
    m = re.match(r"^([A-Z]+)", sheet)
    return m.group(1) if m else None


def extract_sheet_number(blocks: Sequence[TextBlock]) -> str | None:
    """Pull a sheet number from a page's text blocks.

    Prefers candidates in the bottom-right quadrant (typical title-block
    location).  Ties resolve by largest y0 (lower on the page), then
    largest x0 (further right), then the alphabetically smallest token
    so the result is fully deterministic.

    Earlier versions of this function inverted the scoring — they
    appended ``(-y0, -x0, token)`` and returned ``cands[-1]``, which
    actually preferred top-left incidental references over the
    title-block. The boss review caught that.
    """
    cands: list[tuple[float, float, str]] = []
    for blk in blocks:
        for m in _SHEET_NUMBER_RE.finditer(blk.text):
            token = m.group(1) + m.group(2)
            cands.append((blk.bbox[1], blk.bbox[0], token))
    if not cands:
        return None
    # Sort by y desc, x desc, token asc — last element pulled from a
    # reverse-sorted list is what we want, but we use max() over the
    # full keyspace so the tie-break is explicit and total.
    best = max(cands, key=lambda c: (c[0], c[1], -1))
    # Tie-breaker on identical y/x: choose the alphabetically smallest
    # token for determinism (matters when a page accidentally repeats).
    best_y, best_x, _ = best
    same_xy = [c for c in cands if c[0] == best_y and c[1] == best_x]
    chosen = min(c[2] for c in same_xy)
    return _norm_sheet(chosen)


def parse_drawing_index(blocks: Sequence[TextBlock]) -> dict[str, str]:
    """Return ``{sheet_number: short_title}`` for a drawing-index page.

    Empty dict if the page does not look like a drawing index. Used
    by the resolver to discover which sheet should host the legend
    even before the legend page itself is parsed.
    """
    is_index = any(_DRAWING_INDEX_HEADER_RE.search(b.text) for b in blocks)
    if not is_index:
        return {}
    out: dict[str, str] = {}
    for blk in blocks:
        m = _INDEX_ROW_RE.match(blk.text)
        if not m:
            continue
        sheet_raw, title = m.group(1), m.group(2).strip()
        sheet = _norm_sheet(sheet_raw)
        if not sheet:
            continue
        # Avoid grabbing the index header line itself.
        if title and not _DRAWING_INDEX_HEADER_RE.search(title):
            out[sheet] = title
    return out


def detect_inline_references(blocks: Sequence[TextBlock]) -> dict[str, list[str]]:
    """Return ``{kind: [sheet_number, ...]}`` for legend cross-refs on a page.

    Kinds: ``see_sheet`` (active reference) and ``continuation``
    (passive reference from a continuation block).
    """
    out: dict[str, list[str]] = {"see_sheet": [], "continuation": []}
    for blk in blocks:
        m = _LEGEND_REF_INLINE_RE.search(blk.text)
        if m:
            ref = _norm_sheet(m.group(1))
            if ref:
                out["see_sheet"].append(ref)
        m2 = _LEGEND_CONTINUED_RE.search(blk.text)
        if m2:
            ref = _norm_sheet(m2.group(1))
            if ref:
                out["continuation"].append(ref)
    return out


@dataclass
class _PageInfo:
    page_index: int
    sheet_number: str | None
    legend: ParsedLegend | None
    references: dict[str, list[str]]


class LegendResolver:
    """Index of all legends and sheet-references in a single PDF.

    Construction is cheap; the heavy work runs inside ``ingest_page``
    as the PDF parser walks each page. After all pages are ingested,
    ``resolve_for_page`` returns the deterministic legend choice.
    """

    def __init__(self) -> None:
        self._pages: dict[int, _PageInfo] = {}
        self._sheet_to_page: dict[str, int] = {}
        self._page_to_sheet: dict[int, str] = {}
        self._global_legends: list[ParsedLegend] = []
        self._drawing_index: dict[str, str] = {}

    # ─────────── ingestion ───────────

    def ingest_page(
        self,
        *,
        page_index: int,
        blocks: Sequence[TextBlock],
        legend: ParsedLegend | None = None,
    ) -> None:
        sheet = extract_sheet_number(blocks)
        refs = detect_inline_references(blocks)
        self._pages[page_index] = _PageInfo(
            page_index=page_index,
            sheet_number=sheet,
            legend=legend,
            references=refs,
        )
        if sheet:
            # First-page-wins for the sheet → page map so a duplicate
            # sheet number does not silently move pointers around.
            self._sheet_to_page.setdefault(sheet, page_index)
            self._page_to_sheet[page_index] = sheet
        if legend is not None and legend.scope == "global":
            self._global_legends.append(legend)
        # Drawing-index discovery — accept the first index page we see.
        if not self._drawing_index:
            idx = parse_drawing_index(blocks)
            if idx:
                self._drawing_index = idx

    @property
    def drawing_index(self) -> dict[str, str]:
        return dict(self._drawing_index)

    @property
    def sheet_to_page(self) -> dict[str, int]:
        return dict(self._sheet_to_page)

    @property
    def global_legends(self) -> list[ParsedLegend]:
        return list(self._global_legends)

    # ─────────── resolution ───────────

    def resolve_for_page(self, page_index: int) -> ResolvedLegend:
        info = self._pages.get(page_index)
        warnings: list[SchematicWarning] = []
        sheet = info.sheet_number if info else None

        # Priority 1 — in-page legend.
        if info and info.legend is not None:
            return ResolvedLegend(
                page_index=page_index,
                sheet_number=sheet,
                legend=info.legend,
                priority=1,
                rationale="in_page_legend",
                warnings=tuple(warnings),
            )

        candidates: list[tuple[int, ParsedLegend, str]] = []
        ambiguous_buckets: dict[int, list[ParsedLegend]] = {}

        # Priority 2 — explicit references on this page.
        if info:
            for kind in ("see_sheet", "continuation"):
                for ref in info.references.get(kind, []):
                    ref_page = self._sheet_to_page.get(ref)
                    if ref_page is None:
                        warnings.append(
                            SchematicWarning.make(
                                warning_type="unresolved_legend_reference",
                                page_index=page_index,
                                sheet_number=sheet,
                                detail=f"Reference to sheet {ref} not found in document",
                                extras={"kind": kind, "ref": ref},
                            )
                        )
                        continue
                    ref_legend = self._pages[ref_page].legend
                    if ref_legend is None:
                        warnings.append(
                            SchematicWarning.make(
                                warning_type="unresolved_legend_reference",
                                page_index=page_index,
                                sheet_number=sheet,
                                detail=f"Sheet {ref} has no parsed legend",
                                extras={"kind": kind, "ref": ref},
                            )
                        )
                        continue
                    candidates.append((2, ref_legend, f"explicit_{kind}:{ref}"))

        # Priority 3 — same-discipline global legend.
        my_prefix = _discipline_prefix(sheet) if sheet else None
        if my_prefix:
            for legend in self._global_legends:
                lp = _discipline_prefix(legend.sheet_number)
                if lp == my_prefix:
                    candidates.append((3, legend, f"same_discipline:{lp}"))

        # Priority 4 — project-global legend.
        for legend in self._global_legends:
            candidates.append((4, legend, "project_global"))

        # Pick lowest priority bucket that has entries.
        if not candidates:
            warnings.append(
                SchematicWarning.make(
                    warning_type="missing_legend",
                    page_index=page_index,
                    sheet_number=sheet,
                    detail="No in-page legend, no resolvable cross-sheet reference, no global legend.",
                )
            )
            return ResolvedLegend(
                page_index=page_index,
                sheet_number=sheet,
                legend=None,
                priority=99,
                rationale="missing_legend",
                warnings=tuple(warnings),
            )

        # Group by priority and pick deterministically inside the best bucket.
        by_priority: dict[int, list[tuple[ParsedLegend, str]]] = {}
        for prio, legend, rationale in candidates:
            by_priority.setdefault(prio, []).append((legend, rationale))
        best_priority = min(by_priority)
        bucket = by_priority[best_priority]
        # Deduplicate by legend_id while preserving order so warnings count once.
        seen: set[str] = set()
        unique: list[tuple[ParsedLegend, str]] = []
        for entry in bucket:
            if entry[0].legend_id in seen:
                continue
            seen.add(entry[0].legend_id)
            unique.append(entry)
        if len(unique) > 1:
            warnings.append(
                SchematicWarning.make(
                    warning_type="ambiguous_legend_reference",
                    page_index=page_index,
                    sheet_number=sheet,
                    detail=(
                        f"{len(unique)} legends tied at priority {best_priority}; "
                        f"selecting lowest page_index deterministically."
                    ),
                    extras={
                        "tied_legend_ids": [legend.legend_id for legend, _ in unique],
                    },
                )
            )
        unique.sort(key=lambda pair: (pair[0].page_index, pair[0].legend_id))
        chosen, rationale = unique[0]
        return ResolvedLegend(
            page_index=page_index,
            sheet_number=sheet,
            legend=chosen,
            priority=best_priority,
            rationale=rationale,
            warnings=tuple(warnings),
        )
