"""Generic heading-like line heuristics (no project-specific strings)."""

from __future__ import annotations

import re
from typing import Any


def score_line_as_generic_heading(text: str) -> tuple[float, list[str]]:
    """Return ``(0..1 score, reason tags)`` for a single logical line."""
    t = re.sub(r"[\s\u00a0]+", " ", (text or "").strip())
    reasons: list[str] = []
    if not t or len(t) < 6 or len(t) > 120:
        return 0.0, reasons
    if t.count(".") >= 2 or ";" in t:
        return 0.0, reasons
    words = t.split()
    if len(words) < 2 or len(words) > 14:
        return 0.0, reasons
    if re.search(r"\d{4}", t):
        return 0.0, reasons
    if t.endswith(":") and len(words) <= 18:
        reasons.append("colon_tail")
    if t.endswith("."):
        return 0.0, reasons
    title_like = 0
    for w in words:
        core = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", w)
        if not core:
            continue
        if core[0].isupper():
            title_like += 1
    ratio = title_like / max(len(words), 1)
    if ratio >= 0.55:
        reasons.append("title_case_ratio")
    if len(t) <= 52 and ratio >= 0.45:
        reasons.append("short_line")
    if not reasons:
        return 0.0, reasons
    score = 0.25 * min(1.0, ratio * 1.2)
    if "colon_tail" in reasons:
        score += 0.15
    if "short_line" in reasons:
        score += 0.2
    return min(1.0, score), reasons


def generic_heading_candidates(span_lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach heuristic scores to each PDF span line (filter to score > 0)."""
    out: list[dict[str, Any]] = []
    for row in span_lines:
        tx = str(row.get("text") or "").strip()
        if not tx:
            continue
        sc, reasons = score_line_as_generic_heading(tx)
        if sc <= 0:
            continue
        item = {**row, "heading_heuristic_score": round(sc, 3), "heading_heuristic_reasons": reasons}
        out.append(item)
    out.sort(key=lambda x: (-float(x.get("heading_heuristic_score") or 0), len(str(x.get("text")))))
    return out[:80]
