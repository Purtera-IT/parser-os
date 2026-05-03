"""Assemble ``document.heading_analysis`` and optional ``title_pdf_style`` on sections."""

from __future__ import annotations

from typing import Any

from .heuristics import generic_heading_candidates
from .pdf_spans import extract_pdf_span_line_signals, match_title_to_span_style
from .vlm import maybe_vlm_heading_hints


def attach_heading_analysis(
    out: dict[str, Any],
    page: Any,
    full_text: str,
    overlay: dict[str, Any],
    *,
    followon_used: tuple[str, ...],
    major_used: frozenset[str],
    page_index: int,
) -> None:
    """Populate ``document.heading_analysis``; add ``title_pdf_style`` on ``notes`` sections."""
    doc = out.setdefault("document", {})
    if overlay.get("skip_heading_analysis"):
        return

    span_lines = extract_pdf_span_line_signals(page)
    generic = generic_heading_candidates(span_lines)
    vlm = maybe_vlm_heading_hints(
        page_index=page_index,
        span_lines=span_lines,
        generic_candidates=generic,
    )

    doc["heading_analysis"] = {
        "version": 1,
        "followon_headings_effective": list(followon_used),
        "major_band_titles_effective": sorted(major_used),
        "pdf_span_line_count": len(span_lines),
        "pdf_span_lines": span_lines[:400],
        "generic_heading_candidates": generic[:60],
        "full_text_char_len": len(full_text or ""),
        "vlm": vlm,
    }

    for sec in out.get("sections") or []:
        if str(sec.get("kind")) != "notes":
            continue
        title = str(sec.get("title") or "").strip()
        if not title:
            continue
        style = match_title_to_span_style(title, span_lines)
        if style:
            sec["title_pdf_style"] = style
