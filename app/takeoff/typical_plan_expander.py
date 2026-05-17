"""Typical-plan multiplier resolution.

Background:
    A "typical plan" sheet (Cooper-Carry T-sets use T4.xx for this)
    contains several sub-plans for the repeating room types that occur
    on every guest-room floor — e.g. T4.00 holds four panels:

        1  K1 - GUESTROOM PLAN
        2  K2 - GUESTROOM PLAN
        3  QQ1 - GUESTROOM PLAN
        4  QQ2 - GUESTROOM PLAN

    Each panel shows the devices needed in ONE instance of that room
    type. The actual building has dozens of these rooms distributed
    across the guest-room floors (T1.06 …  T1.12). The MVP picks up
    only the literal device tokens drawn inside T4.00 — that's ~15 TVs.
    The *real* TV count is roughly (rooms_per_floor × floor_count),
    easily an order of magnitude larger.

This module:

1. Parses panel titles from a typical-plan sheet's text and assigns
   each native-text device candidate to a panel bbox.
2. Counts devices per (room_type, normalized_class).
3. Detects per-floor room counts by scanning guest-room floor sheets
   for native-text occurrences of the room-type codes (K1, K2, QQ1,
   QQ2, etc.) — this is a v0 heuristic; when no countable label is
   present we report the floor as ``unresolved``.
4. Multiplies the typical-room device counts by the per-floor room
   counts, then by the floor's existing sheet multiplier, and emits a
   summary block + a list of unresolved floors.

The expander never invents counts. If a guest-room floor has no
detectable room-type label, the floor is added to ``unresolved_floors``
and the operator is asked to fill in the room counts manually.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from app.takeoff.schemas import BBox, SheetRecord, SymbolCandidate

# ─── Panel title parsing ───
# Two flavours of panel title appear on Cooper-Carry T4-style sheets:
#
#   1   K1 - GUESTROOM PLAN
#   2   K2 - GUESTROOM PLAN
#
# The number is the panel index (1..N) and the alpha code is the room
# type. The two pieces sometimes sit on adjacent lines in the extracted
# text, so we use a tolerant pattern.
_PANEL_TITLE_RE = re.compile(
    r"(?P<index>\d+)\s*\n+\s*(?P<room>[A-Z]{1,3}\d?)\s*[-–]\s*GUESTROOM\s+PLAN",
    re.IGNORECASE,
)

# Inline form: "K1 - GUESTROOM PLAN" without a preceding number — fall
# back to enumerating in document order.
_PANEL_TITLE_INLINE_RE = re.compile(
    r"(?P<room>[A-Z]{1,3}\d?)\s*[-–]\s*GUESTROOM\s+PLAN",
    re.IGNORECASE,
)

# Room-type code recognizer. The Marriott PDF uses K1, K2, QQ1, QQ2,
# but Cooper-Carry T-sets at other hotels also use KK1, S1, ADA, etc.
_ROOM_CODE_RE = re.compile(r"^[A-Z]{1,3}\d?$")


@dataclass
class TypicalPanel:
    """A single sub-plan panel inside a typical-plan sheet."""

    index: int
    room_type: str
    bbox: BBox | None = None  # crude vertical strip; None if unknown
    title_bbox: BBox | None = None
    device_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class TypicalPlanReport:
    """Per-typical-plan-sheet expansion report."""

    sheet_number: str
    page_index: int
    panels: list[TypicalPanel] = field(default_factory=list)

    def typical_room_device_counts(self) -> dict[str, dict[str, int]]:
        return {p.room_type: dict(p.device_counts) for p in self.panels}


# ───── Panel title parsing ─────────────────────────────────────────


def parse_panel_titles_from_text(text: str) -> list[tuple[int | None, str]]:
    """Return [(panel_index_or_None, room_type), …] in document order.

    The numbered form wins when present; otherwise we fall back to
    enumerating the inline form.
    """
    if not text:
        return []
    panels: list[tuple[int | None, str]] = []
    # Pass 1: numbered form (most reliable).
    for m in _PANEL_TITLE_RE.finditer(text):
        idx = int(m.group("index"))
        room = m.group("room").upper()
        if _ROOM_CODE_RE.match(room):
            panels.append((idx, room))
    if panels:
        return panels
    # Pass 2: inline form — assign synthetic indices in order.
    seen: set[str] = set()
    for m in _PANEL_TITLE_INLINE_RE.finditer(text):
        room = m.group("room").upper()
        if not _ROOM_CODE_RE.match(room):
            continue
        if room in seen:
            continue
        seen.add(room)
        panels.append((len(panels) + 1, room))
    return panels


def panel_title_positions(
    page: Any,
    panels: list[tuple[int | None, str]],
) -> list[tuple[int, str, BBox]]:
    """Locate each panel title's bbox on the page.

    For each ``(idx, room_type)`` tuple, scan the page words for a
    sequence ``idx``  ``room_type`` (numbers and codes adjacent in PDF
    order). Returns ``(idx, room_type, title_bbox)`` for every panel
    successfully located.

    The bbox is the FULL title bounding rectangle — index + room +
    "- GUESTROOM PLAN" suffix — so the caller has a better estimate
    of where the panel's plan area actually sits. Cooper-Carry T-sets
    typically center the plan above its title, so the plan's
    horizontal center sits roughly at the title's horizontal center.
    """
    try:
        words = page.get_text("words")
    except Exception:
        return []

    out: list[tuple[int, str, BBox]] = []
    seen_pairs: set[tuple[str, float, float]] = set()
    for idx, room in panels:
        if idx is None:
            continue
        # Find every word that equals str(idx).
        for w in words:
            if w[4] != str(idx):
                continue
            ix0, iy0, ix1, iy1 = w[0], w[1], w[2], w[3]
            # Search for a word with text == room within ~120pt to the
            # right or below.
            best: tuple[float, tuple] | None = None
            for w2 in words:
                if w2[4].upper() != room:
                    continue
                rx0, ry0, rx1, ry1 = w2[0], w2[1], w2[2], w2[3]
                dx = rx0 - ix1
                dy = ry0 - iy0
                if -20 <= dy <= 80 and 0 <= dx <= 120:
                    dist = abs(dx) + abs(dy)
                    if best is None or dist < best[0]:
                        best = (dist, (rx0, ry0, rx1, ry1))
            if best is None:
                continue
            rx0, ry0, rx1, ry1 = best[1]
            x0 = min(ix0, rx0)
            y0 = min(iy0, ry0)
            x1 = max(ix1, rx1)
            y1 = max(iy1, ry1)
            # Extend x1 to include the "GUESTROOM PLAN" suffix if it
            # sits on the same y as the room code (within ~8pt) AND
            # within a tight horizontal gap (so we don't swallow the
            # next panel's suffix). We accept consecutive suffix words
            # only when each one starts within 12pt of the previous
            # word's right edge.
            suffix_tokens = ("-", "GUESTROOM", "PLAN")
            cursor_right = rx1
            for tok in suffix_tokens:
                candidate = None
                for w3 in words:
                    if w3[4].upper() != tok:
                        continue
                    if abs(w3[1] - ry0) > 8.0:
                        continue
                    if not (cursor_right - 2.0 <= w3[0] <= cursor_right + 12.0):
                        continue
                    if candidate is None or w3[0] < candidate[0]:
                        candidate = (w3[0], w3[2])
                if candidate is None:
                    break
                cursor_right = candidate[1]
                x1 = max(x1, cursor_right)
            key = (room, round(x0, 1), round(y0, 1))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            out.append((idx, room, BBox(x0=x0, y0=y0, x1=x1, y1=y1, coord_space="pdf_pt")))
            break
    return out


def derive_panel_bboxes(
    page: Any,
    titled: list[tuple[int, str, BBox]],
) -> list[TypicalPanel]:
    """Crude v0 panel bboxes from the title positions.

    Strategy: panel titles in T-sets sit BELOW the device area. So a
    panel's plan region runs from a y above the title to the title's
    own y, horizontally bounded by the midpoints between adjacent
    panel titles in the same row.

    We sort by ``(y_band, x)`` to handle multi-row layouts (e.g. a
    2×2 grid of panels). For the common Marriott case of four panels
    in a single horizontal row, this produces four vertical strips of
    the page each centered on its title.
    """
    if not titled:
        return []

    try:
        rect = page.rect
        page_w = float(rect.width)
        page_h = float(rect.height)
    except Exception:
        page_w, page_h = 3024.0, 2160.0

    # Group titles by approximate y-row (tolerance: 100pt at 3x scale).
    sorted_titled = sorted(titled, key=lambda t: (round(t[2].y0 / 100.0), t[2].x0))

    # For Marriott-style single-row, all titles will share one band.
    rows: list[list[tuple[int, str, BBox]]] = []
    current_row: list[tuple[int, str, BBox]] = []
    current_y: float | None = None
    for tup in sorted_titled:
        y = tup[2].y0
        if current_y is None or abs(y - current_y) < 100.0:
            current_row.append(tup)
            current_y = y if current_y is None else current_y
        else:
            rows.append(current_row)
            current_row = [tup]
            current_y = y
    if current_row:
        rows.append(current_row)

    panels: list[TypicalPanel] = []
    for row_titles in rows:
        # Sort row by x.
        row_titles.sort(key=lambda t: t[2].x0)
        title_ys = [t[2].y0 for t in row_titles]
        row_top_y = min(title_ys)
        plan_top = 0.05 * page_h
        plan_bottom = row_top_y - 4.0
        # Vertical splits between titles. Cooper-Carry T-sets layout
        # each panel as an equally-sized horizontal slab with its
        # title centered underneath the plan. We use the median
        # title-to-title pitch as the slab width — this is more
        # robust than the simple midpoint method, which over-allocates
        # to the leftmost and rightmost panels when titles are not
        # exactly evenly spaced.
        x_centers = [(t[2].x0 + t[2].x1) / 2.0 for t in row_titles]
        if len(x_centers) >= 2:
            pitches = sorted(
                x_centers[i + 1] - x_centers[i] for i in range(len(x_centers) - 1)
            )
            slab_width = pitches[len(pitches) // 2]
        else:
            slab_width = page_w
        half = slab_width / 2.0
        for i, (idx, room, title_bbox) in enumerate(row_titles):
            cx = x_centers[i]
            left = max(0.0, cx - half)
            right = min(page_w, cx + half)
            # Don't overlap the previous panel — clamp to that
            # boundary's midpoint.
            if i > 0:
                left = max(left, (x_centers[i - 1] + cx) / 2.0)
            if i < len(row_titles) - 1:
                right = min(right, (cx + x_centers[i + 1]) / 2.0)
            panel_bbox = BBox(
                x0=left,
                y0=plan_top,
                x1=right,
                y1=plan_bottom,
                coord_space="pdf_pt",
            )
            panels.append(
                TypicalPanel(
                    index=idx,
                    room_type=room,
                    bbox=panel_bbox,
                    title_bbox=title_bbox,
                )
            )
    return panels


# ─────────────── Device assignment ─────────────────────────────────


def assign_candidates_to_panels(
    panels: list[TypicalPanel],
    candidates: list[SymbolCandidate],
) -> None:
    """Tally device counts per panel from accepted candidates.

    A candidate is assigned to a panel iff its center lies inside the
    panel's bbox. Candidates outside every panel are silently dropped
    — they're likely titleblock symbols already rejected upstream.
    """
    for p in panels:
        p.device_counts = {}
    for cand in candidates:
        if cand.rejection_reason is not None:
            continue
        if cand.normalized_class is None:
            continue
        cx, cy = cand.bbox.center()
        for p in panels:
            if p.bbox is None:
                continue
            if (
                p.bbox.x0 <= cx <= p.bbox.x1
                and p.bbox.y0 <= cy <= p.bbox.y1
            ):
                p.device_counts[cand.normalized_class] = (
                    p.device_counts.get(cand.normalized_class, 0) + 1
                )
                break


# ─── Per-floor room counts ───


def count_room_types_on_floor(
    page: Any,
    room_types: Iterable[str],
) -> dict[str, int]:
    """Count native-text occurrences of each room-type code on a page.

    A "match" is a word whose text equals the room code exactly (case
    insensitive). The count for QQ1 on T1.06 is the number of QQ1
    labels NTI placed on the plan — typically one label per room.

    This deliberately ignores companion labels like ``K1 L`` / ``K1 R``
    so a single K1 instance counts once.
    """
    try:
        words = page.get_text("words")
    except Exception:
        return {r: 0 for r in room_types}
    upper_rooms = {r.upper() for r in room_types}
    counts: dict[str, int] = {r: 0 for r in upper_rooms}
    seen: set[tuple[str, int, int]] = set()
    for w in words:
        txt = w[4].upper()
        if txt not in upper_rooms:
            continue
        # Dedupe near-coincident words (within 8pt).
        key = (txt, int(w[0] / 8.0), int(w[1] / 8.0))
        if key in seen:
            continue
        seen.add(key)
        counts[txt] += 1
    return counts


# ─── Aggregation ────


def expand_typical_plan(
    *,
    page: Any,
    sheet: SheetRecord,
    candidates: list[SymbolCandidate],
) -> TypicalPlanReport | None:
    """Build a :class:`TypicalPlanReport` for one typical-plan sheet.

    Returns ``None`` if the sheet has no parseable panel titles — the
    caller treats that as "this typical-plan sheet didn't expose any
    sub-plans we know how to expand".
    """
    if sheet.page_type != "typical_plan":
        return None
    try:
        text = page.get_text("text") or ""
    except Exception:
        text = ""
    titles = parse_panel_titles_from_text(text)
    if not titles:
        return None
    titled_with_pos = panel_title_positions(page, titles)
    panels = derive_panel_bboxes(page, titled_with_pos)
    if not panels:
        return None
    assign_candidates_to_panels(panels, candidates)
    return TypicalPlanReport(
        sheet_number=sheet.sheet_number or f"page_{sheet.page_index}",
        page_index=sheet.page_index,
        panels=panels,
    )


def build_expansion_summary(
    *,
    typical_reports: list[TypicalPlanReport],
    floor_room_counts: dict[str, dict[str, int]],
    sheet_records: list[SheetRecord],
) -> dict[str, Any]:
    """Compute the expanded device totals across all floors.

    Returns a structured summary suitable for assigning to
    ``TakeoffDocument.summary['typical_plan_expansion']``.

    ``floor_room_counts`` is keyed by sheet_number (e.g. "T1.06") and
    maps to {room_type: rooms_per_floor}.

    The summary tracks:
        typical_plan_pages          per-sheet panel layout dump
        typical_room_device_counts  {room_type: {class: per_room_count}}
        floor_room_counts           {sheet_number: {room_type: count}}
        expanded_device_totals      {class: extended_total} across all floors
        per_floor_expansion         {sheet_number: {class: extended_for_that_floor}}
        unresolved_floors           list of guest-room sheets w/ no room counts
    """
    typical_room_device_counts: dict[str, dict[str, int]] = {}
    for r in typical_reports:
        for p in r.panels:
            tdc = typical_room_device_counts.setdefault(p.room_type, {})
            for cls, n in p.device_counts.items():
                tdc[cls] = tdc.get(cls, 0) + n

    sheet_by_number = {s.sheet_number: s for s in sheet_records if s.sheet_number}

    expanded_totals: dict[str, int] = {}
    per_floor_expansion: dict[str, dict[str, int]] = {}
    unresolved: list[str] = []

    # Determine the guest-room floors expected to be expanded — every
    # ``floor_plan`` sheet that has at least one room-type token on it.
    guest_room_floors = sorted(floor_room_counts.keys())
    for sheet_number in guest_room_floors:
        rooms = floor_room_counts.get(sheet_number, {})
        sheet = sheet_by_number.get(sheet_number)
        multiplier = sheet.multiplier if sheet else 1
        if not any(v > 0 for v in rooms.values()):
            unresolved.append(sheet_number)
            continue
        floor_total: dict[str, int] = {}
        for room_type, room_count in rooms.items():
            per_room = typical_room_device_counts.get(room_type, {})
            for cls, per in per_room.items():
                ext = per * int(room_count) * int(multiplier)
                floor_total[cls] = floor_total.get(cls, 0) + ext
                expanded_totals[cls] = expanded_totals.get(cls, 0) + ext
        per_floor_expansion[sheet_number] = floor_total

    typical_plan_pages: list[dict[str, Any]] = []
    for r in typical_reports:
        typical_plan_pages.append(
            {
                "sheet": r.sheet_number,
                "page_index": r.page_index,
                "panels": [
                    {
                        "index": p.index,
                        "room_type": p.room_type,
                        "bbox": p.bbox.model_dump() if p.bbox else None,
                        "device_counts": p.device_counts,
                    }
                    for p in r.panels
                ],
            }
        )

    return {
        "typical_plan_pages": typical_plan_pages,
        "typical_room_device_counts": typical_room_device_counts,
        "floor_room_counts": floor_room_counts,
        "expanded_device_totals": expanded_totals,
        "per_floor_expansion": per_floor_expansion,
        "unresolved_floors": unresolved,
    }


__all__ = [
    "TypicalPanel",
    "TypicalPlanReport",
    "parse_panel_titles_from_text",
    "panel_title_positions",
    "derive_panel_bboxes",
    "assign_candidates_to_panels",
    "count_room_types_on_floor",
    "expand_typical_plan",
    "build_expansion_summary",
]
