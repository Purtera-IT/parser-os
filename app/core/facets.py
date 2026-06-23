"""OrbitBrief facet sections — the 7 PM-dashboard sections, assigned by the
contrastive facet head (guess-free).

The 7 facets (WORK / SITE / COMMERCIAL / COMPLIANCE / PARTY / TIMING / META) are
the original "dashboard sections" design target. The facet head (contrastive
mode=facet, held-out **0.925**, 81% @ 0.95 precision) predicts an atom's facet
directly from its decide-text — independent of the production ``AtomType``
taxonomy (the two vocabularies differ, so an atom_type->facet map can't be used).

This module groups the deal's atoms into those 7 sections for the envelope. Atoms
the head can't confidently place (the ~19% it abstains on, or OOD) land in
``uncategorized`` rather than being force-assigned — guess-free, never a wrong
section.

SAFE BY DESIGN: if the facet head (torch/model) is absent, or the feature flag is
off, :func:`build_facet_sections` returns ``{"enabled": False, "sections": []}``
and the envelope simply omits the sections — byte-identical to today. OFF by
default (``SOWSMITH_FACET_SECTIONS``).
"""
from __future__ import annotations

import os
from typing import Any

# Order = display order of the dashboard sections.
FACETS = ("WORK", "SITE", "COMMERCIAL", "COMPLIANCE", "PARTY", "TIMING", "META")


def _facet_dir() -> str:
    return os.environ.get("SOWSMITH_CONTRASTIVE_FACET_DIR", "_contrastive_facet")


def _enabled() -> bool:
    return os.environ.get("SOWSMITH_FACET_SECTIONS", "").strip().lower() in (
        "1", "true", "yes", "on")


def _load_facet_head():
    """The promoted facet store (mode=facet), or None if absent / wrong mode."""
    try:
        from app.core.contrastive_type_knn import load_promoted

        head = load_promoted(registry_dir=_facet_dir())
        return head if (head is not None and head.mode == "facet") else None
    except Exception:
        return None


def assign_facets(texts: list[str]) -> list[str | None]:
    """One facet per text via the facet head, or ``None`` (abstain / OOD). Returns
    all-``None`` if the head is absent or not a facet store. Never raises."""
    if not texts:
        return []
    head = _load_facet_head()
    if head is None:
        return [None] * len(texts)
    try:
        verdicts = head.classify_batch(texts)
        return [v[0] if v is not None else None for v in verdicts]
    except Exception:
        return [None] * len(texts)


def build_facet_sections(atoms: list[Any]) -> dict[str, Any]:
    """Group atoms into the 7 facet dashboard sections via the facet head.

    Returns ``{"enabled", "sections": [{facet, atom_ids, count}], "uncategorized_*"}``.
    Disabled / no head / empty -> ``{"enabled": False, "sections": []}`` so the
    envelope omits it. Never raises (a failure degrades to disabled)."""
    if not _enabled() or not atoms:
        return {"enabled": False, "sections": []}
    head = _load_facet_head()
    if head is None:
        # head absent -> stay disabled so the envelope is byte-identical to today
        return {"enabled": False, "sections": []}
    try:
        from app.core.typed_atom_classifier import _atom_decide_text

        texts = [_atom_decide_text(a) for a in atoms]
        verdicts = head.classify_batch(texts)
        facets = [v[0] if v is not None else None for v in verdicts]
        buckets: dict[str, list[str]] = {f: [] for f in FACETS}
        uncat: list[str] = []
        for atom, facet in zip(atoms, facets):
            aid = getattr(atom, "id", None)
            if aid is None:
                continue
            (buckets[facet] if facet in buckets else uncat).append(aid)
        sections = [
            {"facet": f, "atom_ids": sorted(buckets[f]), "count": len(buckets[f])}
            for f in FACETS if buckets[f]
        ]
        return {
            "enabled": True,
            "sections": sections,
            "uncategorized_atom_ids": sorted(uncat),
            "uncategorized_count": len(uncat),
        }
    except Exception:
        return {"enabled": False, "sections": []}
