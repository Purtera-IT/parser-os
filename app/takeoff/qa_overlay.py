"""Render QA overlay PNGs that visualize the takeoff on each page.

Each overlay shows, for a single page:

* the plan viewport (blue rectangle)
* excluded regions / titleblock (gray rectangle)
* accepted device candidates (green) with their raw symbol + zone
* rejected candidates (red) with their rejection reason

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

    sheet_index = {s.page_index: s for s in takeoff.sheets}
    device_index: dict[int, list[DeviceInstance]] = {}
    for d in takeoff.devices:
        device_index.setdefault(d.page_index, []).append(d)
    candidate_index: dict[int, list[SymbolCandidate]] = {}
    for c in takeoff.candidates:
        candidate_index.setdefault(c.page_index, []).append(c)

    # Decide which pages to render.
    pages_with_candidates = sorted({c.page_index for c in takeoff.candidates})
    selected: list[int] = []
    for page_index in pages_with_candidates:
        sheet = sheet_index.get(page_index)
        if (
            not include_rejected_pages
            and sheet is not None
            and sheet.page_type in _NON_DEVICE_PAGE_TYPES
        ):
            summary["skipped_non_device_pages"] += 1
            continue
        if accepted_only and not device_index.get(page_index):
            summary["skipped_non_device_pages"] += 1
            continue
        selected.append(page_index)

    if max_pages is not None:
        selected = selected[: max(0, max_pages)]

    summary["pages_requested"] = len(selected)
    if not selected:
        summary["elapsed_seconds"] = time.perf_counter() - started
        return summary

    out_dir = _qa_dir(pdf_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    with fitz.open(str(pdf_path)) as doc:  # type: ignore[name-defined]
        for page_index in selected:
            try:
                png = _render_page_overlay(
                    doc=doc,
                    page_index=page_index,
                    sheet=sheet_index.get(page_index),
                    devices=device_index.get(page_index, []),
                    candidates=candidate_index.get(page_index, []),
                    out_dir=out_dir,
                    zoom=zoom,
                )
                if png is not None:
                    summary["pages_written"] += 1
            except Exception:  # pragma: no cover - never fail the parse
                continue

    summary["elapsed_seconds"] = time.perf_counter() - started
    return summary


def _render_page_overlay(
    *,
    doc: Any,
    page_index: int,
    sheet: SheetRecord | None,
    devices: list[DeviceInstance],
    candidates: list[SymbolCandidate],
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

    def _scaled(bbox: BBox) -> tuple[float, float, float, float]:
        return (
            bbox.x0 * zoom,
            bbox.y0 * zoom,
            bbox.x1 * zoom,
            bbox.y1 * zoom,
        )

    if sheet is not None:
        if sheet.plan_viewport is not None:
            draw.rectangle(_scaled(sheet.plan_viewport), outline=(20, 60, 200), width=3)
        for region in sheet.excluded_regions or []:
            draw.rectangle(_scaled(region), outline=(160, 160, 160), width=2)

    device_keys = {(d.page_index, round(d.bbox.center()[0], 1), round(d.bbox.center()[1], 1)) for d in devices}
    for c in candidates:
        rect = _scaled(c.bbox)
        is_accepted = c.rejection_reason is None
        color = (40, 180, 60) if is_accepted else (210, 50, 50)
        draw.rectangle(rect, outline=color, width=2)
        label = c.raw_symbol
        if is_accepted:
            # Match candidate -> device for zone annotation.
            key = (c.page_index, round(c.bbox.center()[0], 1), round(c.bbox.center()[1], 1))
            if key in device_keys:
                dev = next((d for d in devices
                            if (d.page_index, round(d.bbox.center()[0], 1), round(d.bbox.center()[1], 1)) == key),
                           None)
                if dev and dev.home_run_to:
                    label = f"{c.raw_symbol} -> {dev.home_run_to}"
        draw.text((rect[0] + 4, max(0, rect[1] - 12)), label, fill=color)

    out_path = out_dir / f"page_{page_index:04d}_takeoff.png"
    img.save(out_path, format="PNG")
    return out_path


__all__ = ["write_qa_overlays"]
