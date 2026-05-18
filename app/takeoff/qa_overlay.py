"""Render QA overlay PNGs that visualize the takeoff on each page.

Each overlay shows, for a single page:

* the plan viewport (blue rectangle)
* excluded regions / titleblock (gray rectangle)
* accepted device candidates (green) — outlined to the FULL legend
  symbol footprint (text + icon strokes), not just the text token
* rejected candidates (red) with their rejection reason
* keynote bubble callouts (orange) — the ``(N)`` hexagons drawn next
  to devices, rendered as a separate visual class so a reviewer can
  see at a glance which device references which keyed note

Overlays are written under ``<PDF_STEM>.derived/qa_overlays/`` and the
QA pass silently no-ops when Pillow is not available.

By default this renders ONLY floor-plan-ish pages that have at least one
accepted device (spec/legend/detail/riser pages are skipped). Pass
``include_rejected_pages=True`` or ``accepted_only=False`` to widen the
scope. The render is deliberately not optimized for file-size, so it stays
fast — call this from a CLI / offline review path, not from the hot
parse path.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from app.takeoff.schemas import BBox, DeviceInstance, SheetRecord, SymbolCandidate, TakeoffDocument
from app.takeoff.symbol_footprint import (
    NO_OP_FOOTPRINT,
    SymbolFootprint,
    build_symbol_footprints,
    inflate_bbox,
)


_DEFAULT_ZOOM = 1.0  # 1.0x the native PDF size — readable, fast to render.

# Page types we skip by default. Floor plans / typical plans / equipment rooms /
# component schedules / unknown pages CAN have accepted devices in some projects,
# so they stay eligible unless the caller flips ``accepted_only`` themselves.
_NON_DEVICE_PAGE_TYPES = frozenset({"spec", "legend", "detail", "riser"})


def _qa_dir(pdf_path: Path) -> Path:
    return pdf_path.with_name(f"{pdf_path.stem}.derived") / "qa_overlays"


def write_qa_overlays(
    *,
    pdf_path: Path,
    takeoff: TakeoffDocument,
    zoom: float = _DEFAULT_ZOOM,
    include_rejected_pages: bool = False,
    accepted_only: bool = True,
    max_pages: int | None = None,
) -> dict[str, Any]:
    """Render QA overlay PNGs for pages of interest.

    Parameters
    ----------
    pdf_path:
        Source PDF. Overlays land under ``<pdf_stem>.derived/qa_overlays/``.
    takeoff:
        The :class:`TakeoffDocument` produced by ``build_low_voltage_takeoff``.
    zoom:
        Pixmap scale factor. Default ``1.0`` (native pt). Bump to ``1.5`` /
        ``2.0`` when zoomed-in detail is needed for human review.
    include_rejected_pages:
        When ``False`` (default), skip pages whose ``page_type`` is
        spec / legend / detail / riser. Set ``True`` to include them — useful
        when debugging why legend symbols are being miscounted.
    accepted_only:
        When ``True`` (default), skip pages that have candidates but no
        accepted device instances. Set ``False`` to render every page that
        has any candidate at all (rejected included).
    max_pages:
        Optional cap on the number of pages rendered, applied AFTER filtering.

    Returns
    -------
    A small summary dict::

        {
          "pages_requested": int,           # pages we tried to render after filters
          "pages_written":   int,           # PNGs actually saved
          "skipped_non_device_pages": int,  # pages filtered out by page_type / accepted_only
          "elapsed_seconds": float,
        }

    Silently no-ops (returning zeros) when PyMuPDF or Pillow is unavailable.
    """
    started = time.perf_counter()
    summary: dict[str, Any] = {
        "pages_requested": 0,
        "pages_written": 0,
        "skipped_non_device_pages": 0,
        "elapsed_seconds": 0.0,
    }

    try:
        import fitz  # noqa: F401
        from PIL import Image, ImageDraw  # noqa: F401
    except Exception:
        summary["elapsed_seconds"] = time.perf_counter() - started
        return summary

    # Build per-symbol footprint inflation factors from the legend.
    # This runs once per overlay pass; codes not in the legend fall back
    # to NO_OP_FOOTPRINT (text bbox only) inside _render_page_overlay.
    footprints = _build_footprints_for(pdf_path=pdf_path, takeoff=takeoff)

    sheet_index = {s.page_index: s for s in takeoff.sheets}
    device_index: dict[int, list[DeviceInstance]] = {}
    for d in takeoff.devices:
        device_index.setdefault(d.page_index, []).append(d)
    candidate_index: dict[int, list[SymbolCandidate]] = {}
    for c in takeoff.candidates:
        candidate_index.setdefault(c.page_index, []).append(c)

    # Decide which pages to render — dispatch by page_type via the router.
    # legend pages get a different overlay (legend_table_match) and other
    # non-device pages are skipped. The ``include_rejected_pages`` flag
    # widens scope to spec/detail/riser too.
    from app.takeoff.page_type_router import overlay_strategy_for

    pages_with_candidates = sorted({c.page_index for c in takeoff.candidates})
    selected_device: list[int] = []
    selected_legend: list[int] = []
    for page_index in pages_with_candidates:
        sheet = sheet_index.get(page_index)
        strategy = overlay_strategy_for(sheet) if sheet is not None else "skip"

        if strategy == "device_takeoff":
            if accepted_only and not device_index.get(page_index):
                summary["skipped_non_device_pages"] += 1
                continue
            selected_device.append(page_index)
        elif strategy == "legend_table_match":
            # Legend pages always render when the env flag is on — they
            # don't have ``accepted_only`` semantics because legend rows
            # are matched, not accepted/rejected.
            selected_legend.append(page_index)
        elif strategy == "skip":
            if include_rejected_pages and sheet is not None and sheet.in_scope:
                # Caller explicitly opted in to see skipped pages — render
                # them with the device overlay even though page_type would
                # normally have skipped them.
                selected_device.append(page_index)
            else:
                summary["skipped_non_device_pages"] += 1

    if max_pages is not None:
        # Cap applies across both strategy buckets, device first then legend.
        room = max(0, max_pages)
        if len(selected_device) >= room:
            selected_device = selected_device[:room]
            selected_legend = []
        else:
            selected_legend = selected_legend[: room - len(selected_device)]

    summary["pages_requested"] = len(selected_device) + len(selected_legend)
    if not (selected_device or selected_legend):
        summary["elapsed_seconds"] = time.perf_counter() - started
        return summary

    out_dir = _qa_dir(pdf_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Device-takeoff overlays (the established flow).
    with fitz.open(str(pdf_path)) as doc:  # type: ignore[name-defined]
        for page_index in selected_device:
            try:
                png = _render_page_overlay(
                    doc=doc,
                    page_index=page_index,
                    sheet=sheet_index.get(page_index),
                    devices=device_index.get(page_index, []),
                    candidates=candidate_index.get(page_index, []),
                    footprints=footprints,
                    out_dir=out_dir,
                    zoom=zoom,
                )
                if png is not None:
                    summary["pages_written"] += 1
            except Exception:  # pragma: no cover - never fail the parse
                continue

    # Legend overlay — uses the segmentation pipeline to match symbol
    # rows. Imported lazily so a module-level import doesn't pull in
    # orbitbrief_page_os when no legend pages are scheduled.
    if selected_legend:
        try:
            from app.takeoff.legend_overlay import render_legend_overlay
        except Exception:  # pragma: no cover - env-specific
            render_legend_overlay = None  # type: ignore[assignment]
        if render_legend_overlay is not None:
            for page_index in selected_legend:
                try:
                    out_path = out_dir / f"page_{page_index:04d}_legend.png"
                    lg_summary = render_legend_overlay(
                        pdf_path=pdf_path,
                        page_index=page_index,
                        out_path=out_path,
                        legend_rules=takeoff.legend_rules,
                    )
                    if lg_summary.get("output"):
                        summary["pages_written"] += 1
                except Exception:  # pragma: no cover - never fail the parse
                    continue

    summary["elapsed_seconds"] = time.perf_counter() - started
    return summary


def _build_footprints_for(
    *,
    pdf_path: Path,
    takeoff: TakeoffDocument,
) -> dict[str, SymbolFootprint]:
    """Build per-symbol footprints from the legend page(s) in ``pdf_path``.

    Finds every sheet classified as ``page_type == "legend"`` in
    ``takeoff.sheets``, extracts its legend doc on the fly, and unions
    the resulting footprint maps. Codes the detector actually cares
    about (from ``takeoff.legend_rules`` plus everything actually
    detected) form the ``known_codes`` filter.

    Returns ``{}`` on any failure — the overlay then falls back to
    :data:`NO_OP_FOOTPRINT` (text bbox only) for every symbol.
    """
    # Set of codes the detector actually cares about.
    rules = getattr(takeoff, "legend_rules", None) or []
    known_codes: set[str] = set()
    for r in rules:
        sym = getattr(r, "raw_symbol", None) or getattr(r, "symbol", None)
        if sym:
            known_codes.add(str(sym).upper())
    for cand in getattr(takeoff, "candidates", []) or []:
        sym = getattr(cand, "raw_symbol", None)
        if sym:
            known_codes.add(str(sym).upper())
    if not known_codes:
        return {}

    # Find every legend page in the PDF.
    legend_pages = [
        s.page_index for s in (takeoff.sheets or [])
        if (s.page_type or "").lower() == "legend"
    ]
    if not legend_pages:
        return {}

    try:
        from app.takeoff.legend_extract import extract_legend
    except Exception:  # pragma: no cover - env-specific
        return {}

    merged: dict[str, SymbolFootprint] = {}
    for page_index in legend_pages:
        try:
            legend_doc = extract_legend(pdf_path=pdf_path, page_index=page_index)
        except Exception:  # pragma: no cover - never fail rendering
            continue
        if not isinstance(legend_doc, dict) or not legend_doc.get("tables"):
            continue
        fp_map = build_symbol_footprints(
            pdf_path=pdf_path,
            legend_doc=legend_doc,
            known_codes=known_codes,
        )
        for code, fp in fp_map.items():
            merged.setdefault(code, fp)  # first legend page wins
    return merged


def _bbox_for_candidate(
    *,
    candidate: SymbolCandidate,
    footprint: SymbolFootprint,
) -> tuple[float, float, float, float]:
    """Return the inflated PDF-pt bbox for ``candidate`` using ``footprint``."""
    b = candidate.bbox
    return inflate_bbox(
        text_x0=b.x0, text_y0=b.y0, text_x1=b.x1, text_y1=b.y1,
        footprint=footprint,
    )


def _find_keynote_bubble_bbox(
    *,
    page: Any,
    device: DeviceInstance,
    search_radius_pt: float = 80.0,
) -> tuple[float, float, float, float] | None:
    """Locate the keynote-callout bbox on the plan for ``device``.

    Looks for a numeric text token equal to ``device.keynote`` whose
    bbox center sits within ``search_radius_pt`` PDF points of the
    device's bbox center. Returns the bbox of the closest such token,
    or ``None`` if the device has no keynote or no matching token sits
    within radius. The bbox is the TEXT bbox of the digit(s) inside
    the bubble — small, but enough to outline the callout on the
    overlay.
    """
    target = (device.keynote or "").strip()
    if not target:
        return None
    cx, cy = device.bbox.center()
    closest: tuple[float, float, float, float] | None = None
    closest_dist = float("inf")
    try:
        words = page.get_text("words") or []
    except Exception:  # pragma: no cover - never fail rendering
        return None
    for w in words:
        if len(w) < 5:
            continue
        wx0, wy0, wx1, wy1, wt = w[0], w[1], w[2], w[3], w[4]
        cleaned = str(wt).strip().strip("()")
        if cleaned != target:
            continue
        bx = (wx0 + wx1) / 2.0
        by = (wy0 + wy1) / 2.0
        dx, dy = bx - cx, by - cy
        d2 = dx * dx + dy * dy
        if d2 > search_radius_pt * search_radius_pt:
            continue
        if d2 < closest_dist:
            closest_dist = d2
            closest = (float(wx0), float(wy0), float(wx1), float(wy1))
    return closest


def _device_for_candidate(
    *,
    candidate: SymbolCandidate,
    devices: list[DeviceInstance],
) -> DeviceInstance | None:
    """Match a candidate to its accepted device by page+center coordinate."""
    if candidate.rejection_reason is not None:
        return None
    key = (
        candidate.page_index,
        round(candidate.bbox.center()[0], 1),
        round(candidate.bbox.center()[1], 1),
    )
    for d in devices:
        if (
            d.page_index,
            round(d.bbox.center()[0], 1),
            round(d.bbox.center()[1], 1),
        ) == key:
            return d
    return None


# Color palette — three distinct hues so a reviewer can read the overlay
# at a glance: device fully-resolved (green), device flagged for review
# or rejected (red), keynote callout bubble (orange).
_GREEN  = (40, 180, 60)
_RED    = (210, 50, 50)
_ORANGE = (255, 140, 0)
_BLUE   = (20, 60, 200)
_GRAY   = (160, 160, 160)


def _render_page_overlay(
    *,
    doc: Any,
    page_index: int,
    sheet: SheetRecord | None,
    devices: list[DeviceInstance],
    candidates: list[SymbolCandidate],
    footprints: dict[str, SymbolFootprint],
    out_dir: Path,
    zoom: float,
) -> Path | None:
    import fitz
    from PIL import Image, ImageDraw

    page = doc[page_index]
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix)
    mode = "RGB" if pix.alpha == 0 else "RGBA"
    img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
    if mode != "RGB":
        img = img.convert("RGB")
    draw = ImageDraw.Draw(img)

    def _scale_bbox(bbox: BBox) -> tuple[float, float, float, float]:
        return (bbox.x0 * zoom, bbox.y0 * zoom, bbox.x1 * zoom, bbox.y1 * zoom)

    def _scale_pt(x0: float, y0: float, x1: float, y1: float) -> tuple[float, float, float, float]:
        return (x0 * zoom, y0 * zoom, x1 * zoom, y1 * zoom)

    # 1) Frame: plan viewport (blue) + excluded regions (gray).
    if sheet is not None:
        if sheet.plan_viewport is not None:
            draw.rectangle(_scale_bbox(sheet.plan_viewport), outline=_BLUE, width=3)
        for region in sheet.excluded_regions or []:
            draw.rectangle(_scale_bbox(region), outline=_GRAY, width=2)

    # 2) Devices/candidates: inflate to the full legend footprint.
    keynote_bubbles_drawn: set[tuple[float, float, float, float]] = set()
    for c in candidates:
        fp = footprints.get((c.raw_symbol or "").upper(), NO_OP_FOOTPRINT)
        x0, y0, x1, y1 = _bbox_for_candidate(candidate=c, footprint=fp)
        rect = _scale_pt(x0, y0, x1, y1)
        is_accepted = c.rejection_reason is None
        color = _GREEN if is_accepted else _RED
        draw.rectangle(rect, outline=color, width=2)

        # 3) Label above the inflated box: just the symbol code + zone.
        label_parts = [c.raw_symbol]
        device = _device_for_candidate(candidate=c, devices=devices)
        if device is not None and device.home_run_to:
            label_parts.append(f"-> {device.home_run_to}")
        label = " ".join(label_parts)
        text_x = rect[0] + 2
        text_y = max(0, rect[1] - 14)
        draw.text((text_x, text_y), label, fill=color)

        # 4) Keynote bubble: highlight the (N) callout drawn near the
        # device in orange so it reads as a separate visual class.
        if device is not None:
            kb = _find_keynote_bubble_bbox(page=page, device=device)
            if kb is not None:
                kx0, ky0, kx1, ky1 = kb
                # Inflate the bubble bbox slightly so the orange ring sits
                # outside the digit glyph rather than clipping it.
                pad = 3.0
                bubble_rect = _scale_pt(kx0 - pad, ky0 - pad, kx1 + pad, ky1 + pad)
                key = (round(bubble_rect[0]), round(bubble_rect[1]),
                       round(bubble_rect[2]), round(bubble_rect[3]))
                if key not in keynote_bubbles_drawn:
                    draw.rectangle(bubble_rect, outline=_ORANGE, width=2)
                    draw.text(
                        (bubble_rect[0] + 2, max(0, bubble_rect[1] - 14)),
                        f"note {device.keynote}",
                        fill=_ORANGE,
                    )
                    keynote_bubbles_drawn.add(key)

    out_path = out_dir / f"page_{page_index:04d}_takeoff.png"
    img.save(out_path, format="PNG")
    return out_path


__all__ = ["write_qa_overlays"]
