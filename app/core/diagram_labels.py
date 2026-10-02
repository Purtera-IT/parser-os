"""A short label read off a diagram is a label, not a site fact.

Deal 010246's install drawings came back with their callouts -- "Solar
Panel", "Cell Modem", "Junction Box", "Pole 3" -- typed
``site_infrastructure``, the type for "MDF/IDF id, bandwidth/circuit, cable
plant condition". A callout names a part of a picture; it states nothing
about the site on its own. So after typing, a short label that came from a
picture or a drawing and was typed site_infrastructure becomes a reject-able
``diagram_label`` (deal_metadata, chatter, rejected_by ``diagram_label``),
keeping the model's guess as an alternative type for the labeler.

A line that does state an infrastructure fact ("MDF in room 112", "100 Mbps
circuit", "42U rack") is not a bare callout and keeps its type.
"""
from __future__ import annotations

import re
from typing import Any

#: Extraction methods / value markers of text read off a picture or drawing.
_PICTURE_METHODS = ("pdf_image_vision", "linked_picture", "dwg", "dxf", "vsdx", "image_ocr", "ocr_image")
_PICTURE_ARTIFACTS = {"image", "vsdx"}

#: What makes a short line an infrastructure FACT rather than a callout.
_INFRA_FACT_RE = re.compile(
    r"\b(?:mdf|idf|demarc|mbps|gbps|kbps|circuit|bandwidth|isp|fiber|fibre|"
    r"\d+\s*(?:u|ru)\b|cat\s*\d|conduit|riser|telco|backbone|uplink|vlan|ip)\b",
    re.I,
)
_VERB_RE = re.compile(
    r"\b(?:is|are|was|were|be|will|shall|must|should|can|has|have|needs?|requires?|"
    r"provides?|install(?:ed|s)?|mount(?:ed|s)?|connect(?:ed|s)?|runs?|located)\b",
    re.I,
)


def _type(a: Any) -> str:
    t = getattr(a, "atom_type", None)
    return str(getattr(t, "value", t) or "")


def from_a_picture(atom: Any) -> bool:
    """Was this atom read off a picture or a drawing sheet?"""
    v = getattr(atom, "value", None)
    if isinstance(v, dict) and (v.get("via") in ("pdf_image_vision", "linked_picture_vision")
                                or v.get("image_kind") or v.get("on_drawing")):
        return True
    for ref in (getattr(atom, "source_refs", None) or [])[:1]:
        method = str(getattr(ref, "extraction_method", "") or "").lower()
        if any(m in method for m in _PICTURE_METHODS):
            return True
        at = getattr(ref, "artifact_type", None)
        if str(getattr(at, "value", at) or "").lower() in _PICTURE_ARTIFACTS:
            return True
        loc = getattr(ref, "locator", None) or {}
        if isinstance(loc, dict) and (loc.get("on_drawing") or loc.get("bbox_pdf_points") and loc.get("layer")):
            return True
    return False


def is_callout_label(text: str) -> bool:
    """A short noun label: at most five words, no sentence, no infra fact."""
    t = " ".join(str(text or "").split()).strip(" -:•")
    if not t or len(t) > 40:
        return False
    words = t.split()
    if not (1 <= len(words) <= 5):
        return False
    if re.search(r"[.!?;]\s*$", t) or _VERB_RE.search(t) or _INFRA_FACT_RE.search(t):
        return False
    return True


def retype_diagram_labels(atoms: list[Any], *, types: frozenset[str] = frozenset({"site_infrastructure"})) -> int:
    """Retype short picture / drawing callouts typed ``types`` to a reject-able
    diagram_label. Returns how many were retyped."""
    from app.core.deal_chatter import CHATTER_FLAG
    from app.core.schemas import AtomType

    n = 0
    for a in atoms:
        if _type(a) not in types:
            continue
        if not from_a_picture(a) or not is_callout_label(getattr(a, "raw_text", "") or ""):
            continue
        v = dict(a.value) if isinstance(getattr(a, "value", None), dict) else {}
        alt = list(v.get("alt_atom_types") or [])
        if _type(a) not in alt:
            alt.append(_type(a))
        v.update({"kind": "diagram_label", "chatter": True, "rejected_by": "diagram_label",
                  "alt_atom_types": alt})
        a.value = v
        a.atom_type = AtomType.deal_metadata
        flags = list(getattr(a, "review_flags", None) or [])
        for f in (CHATTER_FLAG, "diagram_label"):
            if f not in flags:
                flags.append(f)
        a.review_flags = flags
        a.entity_keys = [k for k in (getattr(a, "entity_keys", None) or []) if not str(k).startswith("site:")]
        n += 1
    return n


__all__ = ["from_a_picture", "is_callout_label", "retype_diagram_labels"]
