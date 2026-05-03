"""Per-overlay JSON heading templates (extend / replace built-in TSC lists)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .builtin import TSC_FOLLOWON_HEADINGS, TSC_MAJOR_BAND_SECTION_TITLES


def load_overlay_heading_template(overlay: dict[str, Any]) -> dict[str, Any] | None:
    """Return template dict from ``overlay["heading_template"]`` (path or object)."""
    raw = overlay.get("heading_template")
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        p = Path(raw.strip())
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    return None


def resolve_effective_headings(
    overlay: dict[str, Any],
) -> tuple[tuple[str, ...], frozenset[str]]:
    """Merge ``heading_template`` with built-ins (extras, optional full replace)."""
    tpl = load_overlay_heading_template(overlay) or {}
    follow = TSC_FOLLOWON_HEADINGS
    major = TSC_MAJOR_BAND_SECTION_TITLES

    rep = tpl.get("followon_headings_replace")
    if isinstance(rep, list) and rep:
        follow = tuple(str(x).strip() for x in rep if str(x).strip())
    else:
        extra = tpl.get("followon_headings_extra")
        if isinstance(extra, list) and extra:
            add = tuple(str(x).strip() for x in extra if str(x).strip())
            follow = tuple(dict.fromkeys(list(follow) + list(add)))

    mex = tpl.get("major_band_titles_extra")
    if isinstance(mex, list) and mex:
        major = major | frozenset(str(x).strip() for x in mex if str(x).strip())

    return follow, major
