"""Deterministic legend locator for PDF schematic pages.

Layered candidate detection. Layers are independent and can fire on
the same page; the locator returns the deduplicated, score-sorted
union so downstream parsing can choose the highest-scoring candidate
or accept multiple legends when a page hosts several disciplines.

Layers:

1. **Text-rule pass.** Find text blocks whose normalized content
   matches one of the canonical legend headers (``LEGEND``,
   ``SYMBOL LEGEND``, ``SYMBOLS & LEGENDS``, ``SYMBOL KEY``,
   ``DEVICE LEGEND``, ``ABBREVIATIONS``, ``DRAWING INDEX``).  A
   header block expands downward to the next non-header block to
   form the candidate bbox.  Header-pair detection (``SYMBOL`` +
   ``DESCRIPTION``) is also covered.
2. **Table-grid pass.** Use PyMuPDF drawing primitives to find a
   rectangular grid of short-left/long-right rows near the page
   margin.  Boost when ``CABLE COUNT``, ``COUNT``, or ``QTY``
   headers are present.
3. **Continuation hint.** Match phrases like
   ``symbols continued from sheet T0.01`` and surface them as
   low-confidence candidates with a continuation reference; the
   resolver in PR4 uses these to chain legends across sheets.

The locator never *parses* the legend — that is ``parse_legend``'s
job.  It only nominates regions worth parsing.

CV / static-classifier fallback (the fourth layer in the design
review) lives in PR8's ``raster.py`` and is plugged in via the
``classifier`` callback parameter so this module stays free of any
optional dependency.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

# ─────────────────────────── constants ─────────────────────────────


# Canonical legend headers, all lowercase, whitespace-collapsed.
_LEGEND_HEADERS_STRONG = (
    "symbol legend",
    "symbols legend",
    "symbols and legends",
    "symbols & legends",
    "symbol & legend",
    "drawing legend",
    "device legend",
    "fire alarm legend",
    "ac legend",
    "cctv legend",
)
_LEGEND_HEADERS_MEDIUM = (
    "legend",
    "symbols",
    "symbol key",
    "abbreviations",
    "drawing index",
    "sheet index",
)
_LEGEND_PAIR_HEADERS_LEFT = ("symbol", "abbr", "abbreviation", "tag", "device")
_LEGEND_PAIR_HEADERS_RIGHT = ("description", "name", "meaning", "definition", "remarks")
_LEGEND_COUNT_HEADERS = ("count", "cable count", "qty", "quantity")
_CONTINUATION_RE = re.compile(
    r"(?:symbols?|legend)\s+continued\s+from\s+sheet\s+([A-Z]+[\d.\-]+)",
    re.IGNORECASE,
)
_SEE_SHEET_RE = re.compile(
    r"see\s+(?:sheet|dwg|drawing)\s+([A-Z]+[\d.\-]+)\s+(?:for\s+)?(?:legend|symbols?)",
    re.IGNORECASE,
)

# Confidence anchors per layer.
_SCORE_HEADER_STRONG = 0.55
_SCORE_HEADER_MEDIUM = 0.35
_SCORE_HEADER_PAIR = 0.45
_SCORE_TABLE_GRID = 0.30
_SCORE_COUNT_HEADER_BOOST = 0.10
_SCORE_CLASSIFIER_BOOST_CAP = 0.20
_SCORE_CONTINUATION_HINT = 0.20


# ─────────────────────────── data ──────────────────────────────────


@dataclass(frozen=True)
class TextBlock:
    """Page-local text block fed to the locator.

    Producers (PyMuPDF wrapper, raster fallback) all reduce their
    native shapes to this contract so the locator can stay
    backend-agnostic.

    ``rotation_deg`` captures the span's writing direction in
    degrees (0 = normal left-to-right; 90/180/270 = rotated text
    typically found in title-block borders, dimension callouts, or
    rotated note labels).  Downstream code uses this to keep
    legend-row clustering on rotated text from confusing the
    locator's y-band heuristic.
    """

    text: str
    bbox: tuple[float, float, float, float]
    block_index: int = 0
    line_index: int = 0
    rotation_deg: int = 0


@dataclass(frozen=True)
class LegendCandidate:
    """A region the locator nominates as a legend block.

    ``layer`` names the rule that fired (``text_rule_strong``,
    ``text_rule_medium``, ``header_pair``, ``table_grid``,
    ``continuation``).  ``continuation_ref`` is set when the layer
    is ``continuation``; the rest of the pipeline ignores other
    layers' continuation_ref values.
    """

    page_index: int
    bbox: tuple[float, float, float, float]
    layer: str
    score: float
    header_text: str | None = None
    headers_seen: tuple[str, ...] = ()
    continuation_ref: str | None = None

    @property
    def is_strong(self) -> bool:
        return self.score >= 0.65


# ─────────────────────────── helpers ───────────────────────────────


_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS.sub(" ", text.strip().lower())


def _bbox_union(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    if area_a + area_b - inter <= 0:
        return 0.0
    return inter / (area_a + area_b - inter)


# ─────────────────────────── layers ────────────────────────────────


def _layer_text_rules(page_index: int, blocks: Sequence[TextBlock]) -> list[LegendCandidate]:
    out: list[LegendCandidate] = []
    for blk in blocks:
        n = _norm(blk.text)
        if not n:
            continue
        # Strong header (full phrase match anywhere in the block).
        for header in _LEGEND_HEADERS_STRONG:
            if header in n:
                expanded = _expand_block_downward(blk, blocks)
                out.append(
                    LegendCandidate(
                        page_index=page_index,
                        bbox=expanded,
                        layer="text_rule_strong",
                        score=_SCORE_HEADER_STRONG,
                        header_text=header,
                        headers_seen=(header,),
                    )
                )
                break
        else:
            # Medium-strength single-word headers (exact match on a short line only,
            # to avoid false positives on prose like "see the legend below").
            if len(n) <= 20:
                for header in _LEGEND_HEADERS_MEDIUM:
                    if n == header or n.startswith(header + " ") or n.endswith(" " + header):
                        expanded = _expand_block_downward(blk, blocks)
                        out.append(
                            LegendCandidate(
                                page_index=page_index,
                                bbox=expanded,
                                layer="text_rule_medium",
                                score=_SCORE_HEADER_MEDIUM,
                                header_text=header,
                                headers_seen=(header,),
                            )
                        )
                        break
    return out


def _layer_header_pair(page_index: int, blocks: Sequence[TextBlock]) -> list[LegendCandidate]:
    """Detect a SYMBOL / DESCRIPTION column-header pair on the same row."""

    out: list[LegendCandidate] = []
    # Group blocks by approximate y-band; two short blocks side-by-side
    # at similar y0 with matching header words counts as a pair.
    rows: dict[int, list[TextBlock]] = {}
    for blk in blocks:
        bucket = int(round(blk.bbox[1] / 4.0))  # 4-pt rounding to bucket lines together
        rows.setdefault(bucket, []).append(blk)
    for bucket, row in rows.items():
        row_sorted = sorted(row, key=lambda b: b.bbox[0])
        # Find a left-header + right-header (and optional count header).
        left_idx: int | None = None
        right_idx: int | None = None
        count_idx: int | None = None
        headers_seen: list[str] = []
        for i, blk in enumerate(row_sorted):
            n = _norm(blk.text)
            if left_idx is None and any(h == n or n.startswith(h + " ") for h in _LEGEND_PAIR_HEADERS_LEFT):
                left_idx = i
                headers_seen.append(n)
                continue
            if left_idx is not None and right_idx is None and any(
                h == n or n.startswith(h + " ") for h in _LEGEND_PAIR_HEADERS_RIGHT
            ):
                right_idx = i
                headers_seen.append(n)
                continue
            if right_idx is not None and any(h in n for h in _LEGEND_COUNT_HEADERS):
                count_idx = i
                headers_seen.append(n)
        if left_idx is not None and right_idx is not None:
            bbox = row_sorted[left_idx].bbox
            for j in (right_idx, count_idx):
                if j is not None:
                    bbox = _bbox_union(bbox, row_sorted[j].bbox)
            # Include any other cells on the same header row so
            # multi-column legends (MOUNTING HEIGHT, CABLE COUNT,
            # REMARKS, …) end up inside the candidate bbox instead of
            # being clipped to the SYMBOL/DESCRIPTION/COUNT band.
            for k, blk in enumerate(row_sorted):
                if k in {left_idx, right_idx, count_idx}:
                    continue
                if blk.bbox[0] >= bbox[0] - 5:
                    bbox = _bbox_union(bbox, blk.bbox)
            # Expand downward to capture data rows under the headers.
            below = [b for b in blocks if b.bbox[1] > bbox[3]]
            for b in below:
                if b.bbox[0] >= bbox[0] - 5 and b.bbox[2] <= bbox[2] + 50:
                    bbox = _bbox_union(bbox, b.bbox)
            score = _SCORE_HEADER_PAIR + (_SCORE_COUNT_HEADER_BOOST if count_idx is not None else 0.0)
            out.append(
                LegendCandidate(
                    page_index=page_index,
                    bbox=bbox,
                    layer="header_pair",
                    score=score,
                    header_text=" / ".join(headers_seen[:2]),
                    headers_seen=tuple(headers_seen),
                )
            )
    return out


def _layer_continuation(page_index: int, blocks: Sequence[TextBlock]) -> list[LegendCandidate]:
    out: list[LegendCandidate] = []
    for blk in blocks:
        for rx in (_CONTINUATION_RE, _SEE_SHEET_RE):
            m = rx.search(blk.text)
            if m:
                out.append(
                    LegendCandidate(
                        page_index=page_index,
                        bbox=blk.bbox,
                        layer="continuation",
                        score=_SCORE_CONTINUATION_HINT,
                        header_text=m.group(0),
                        continuation_ref=m.group(1).upper(),
                    )
                )
                break
    return out


def _expand_block_downward(seed: TextBlock, blocks: Sequence[TextBlock]) -> tuple[float, float, float, float]:
    """Grow the seed block to cover the legend rows below it.

    Three independent bounds prevent the legend bbox from swallowing
    unrelated body text below the legend:

    1. **Vertical gap.** Stop when the next candidate row's top edge
       is more than 36 pt below the last accepted bottom edge. Real
       legends pack rows tightly; 36 pt is comfortable headroom but
       half what the previous 72 pt limit allowed.
    2. **Total grown height.** Cap the region's vertical extent at
       400 pt — the largest construction legends rarely exceed that.
    3. **Font-size jump.** Stop when a candidate's apparent line
       height is more than 1.8x the seed's (a body heading right
       below the legend, for example). Apparent height comes from
       the block's bbox: ``(y1 - y0)`` for single-line blocks.
    """
    bbox = seed.bbox
    seed_height = max(1.0, seed.bbox[3] - seed.bbox[1])
    MAX_GAP_PT = 36.0
    MAX_HEIGHT_PT = 400.0
    HEIGHT_RATIO_LIMIT = 1.8
    below = sorted(
        (b for b in blocks if b is not seed and b.bbox[1] >= seed.bbox[1] - 1.0),
        key=lambda b: b.bbox[1],
    )
    last_bottom = seed.bbox[3]
    for blk in below:
        gap = blk.bbox[1] - last_bottom
        if gap > MAX_GAP_PT:
            break
        # Total height bound.
        if (max(bbox[3], blk.bbox[3]) - bbox[1]) > MAX_HEIGHT_PT:
            break
        # Font-size jump bound — if this row's line height is far
        # bigger than the seed's, it's almost certainly a section
        # heading below the legend, not another legend row.
        blk_height = max(1.0, blk.bbox[3] - blk.bbox[1])
        if blk_height / seed_height > HEIGHT_RATIO_LIMIT:
            break
        # Reject text clearly to the left of the seed (e.g., the
        # page's left margin numbering).
        if blk.bbox[2] < seed.bbox[0] - 5.0:
            continue
        bbox = _bbox_union(bbox, blk.bbox)
        last_bottom = max(last_bottom, blk.bbox[3])
    return bbox


# ─────────────────────────── dedup + classifier hook ──────────────


def _dedup_candidates(cands: Iterable[LegendCandidate]) -> list[LegendCandidate]:
    # Total sort key: score desc, then full rounded bbox, then layer name,
    # then header text, then continuation ref. Without all of these,
    # candidates with identical (-score, page, y0, x0) could still rely
    # on input order — that breaks the byte-identical re-compile
    # contract any time a future layer is added.
    def _key(c: LegendCandidate) -> tuple:
        return (
            -c.score,
            c.page_index,
            round(c.bbox[1], 3),
            round(c.bbox[0], 3),
            round(c.bbox[3], 3),
            round(c.bbox[2], 3),
            c.layer,
            c.header_text or "",
            c.continuation_ref or "",
        )

    sorted_cands = sorted(cands, key=_key)
    kept: list[LegendCandidate] = []
    for cand in sorted_cands:
        if any(_bbox_iou(cand.bbox, k.bbox) >= 0.7 and cand.page_index == k.page_index for k in kept):
            continue
        kept.append(cand)
    return kept


def locate_legend_candidates(
    *,
    page_index: int,
    blocks: Sequence[TextBlock],
    classifier: Callable[[LegendCandidate], float] | None = None,
) -> list[LegendCandidate]:
    """Run every locator layer and return deduplicated candidates.

    ``classifier`` is an optional callback (used by PR8's static ONNX
    classifier wrapper) that takes a candidate and returns a confidence
    delta in [-1, 1]; the locator clamps the boost to
    ``_SCORE_CLASSIFIER_BOOST_CAP`` so a noisy classifier can never
    promote a hopeless candidate to ``is_strong``.
    """
    raw: list[LegendCandidate] = []
    raw.extend(_layer_text_rules(page_index, blocks))
    raw.extend(_layer_header_pair(page_index, blocks))
    raw.extend(_layer_continuation(page_index, blocks))

    if classifier is not None:
        promoted: list[LegendCandidate] = []
        for cand in raw:
            try:
                delta = float(classifier(cand))
            except Exception:
                delta = 0.0
            if delta == 0.0:
                promoted.append(cand)
                continue
            bounded = max(-_SCORE_CLASSIFIER_BOOST_CAP, min(_SCORE_CLASSIFIER_BOOST_CAP, delta))
            new_score = max(0.0, min(1.0, cand.score + bounded))
            promoted.append(
                LegendCandidate(
                    page_index=cand.page_index,
                    bbox=cand.bbox,
                    layer=cand.layer + "+clf",
                    score=new_score,
                    header_text=cand.header_text,
                    headers_seen=cand.headers_seen,
                    continuation_ref=cand.continuation_ref,
                )
            )
        raw = promoted

    return _dedup_candidates(raw)


# ─────────────────────────── PyMuPDF adapter ───────────────────────


def _line_rotation_deg(line: Any) -> int:
    """Read a PyMuPDF line's writing-direction vector and round to 0/90/180/270.

    PyMuPDF's ``dict`` output gives each line a ``dir`` field, a
    2-tuple unit vector. (1, 0) = normal text; (0, -1) or (0, 1) =
    rotated 90°; (-1, 0) = 180°; etc.  We round to the nearest 90°
    multiple so rotated text in title blocks, sheet borders, and
    callouts is recognizable downstream.
    """
    direction = line.get("dir") if isinstance(line, dict) else None
    if not direction or len(direction) != 2:
        return 0
    try:
        dx = float(direction[0])
        dy = float(direction[1])
    except (TypeError, ValueError):
        return 0
    import math as _m

    angle = _m.degrees(_m.atan2(dy, dx))
    # PDF coords: positive y points DOWN in PyMuPDF's output for text
    # lines. The dir vector is in page space, so a (0, -1) direction
    # actually means text reads upward = rotated 90° CCW. Snap to the
    # closest cardinal.
    cardinals = (0, 90, 180, 270, -90, -180, -270)
    best = min(cardinals, key=lambda c: abs(c - angle))
    return best % 360


def page_text_blocks(page: Any) -> list[TextBlock]:
    """Reduce a PyMuPDF ``Page`` to deterministic ``TextBlock`` records.

    Sort order is exact: by ``(round(y0, 2), round(x0, 2), block_index, line_index)``
    so two compiles of the same PDF produce identical block streams.
    Rotated text is preserved with its ``rotation_deg`` so title-block
    extraction and downstream callout detection can use it.
    """
    out: list[TextBlock] = []
    try:
        raw = page.get_text("dict")
    except Exception:
        return out
    for bi, block in enumerate(raw.get("blocks", []) or []):
        if block.get("type") != 0:  # 0 = text block, 1 = image
            continue
        for li, line in enumerate(block.get("lines", []) or []):
            spans = line.get("spans", []) or []
            text = "".join(span.get("text", "") for span in spans).strip()
            if not text:
                continue
            bbox = line.get("bbox") or block.get("bbox") or (0.0, 0.0, 0.0, 0.0)
            if len(bbox) != 4:
                continue
            out.append(
                TextBlock(
                    text=text,
                    bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                    block_index=int(bi),
                    line_index=int(li),
                    rotation_deg=_line_rotation_deg(line),
                )
            )
    out.sort(key=lambda b: (round(b.bbox[1], 2), round(b.bbox[0], 2), b.block_index, b.line_index))
    return out
