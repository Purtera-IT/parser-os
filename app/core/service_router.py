"""Service-type router head: deal scope -> managed-service pack.

A trained contrastive kNN head (bge-base, GPU fine-tuned on DeepSeek scope
labels) that classifies a deal's SCOPE into its primary managed-service pack —
replacing the brief-gen keyword router that mis-routed TV installs to
``datacenter``. Writes ``envelope.service_routing`` so Orbitbrief-Core's
pack_prior router can use it as a guess-free **confident-override** prior.

Specialist over the learnable classes (audio_visual / low_voltage_cabling /
staff_augmentation / wireless); abstains (no prediction) on everything else and
on out-of-distribution scopes (nearest neighbor beyond ``sim_floor``), so the
brief-gen keyword router stays in charge there.

After the neural vote, an evidence-anchor gate (mirrors OrbitBrief domain-pack
``required_anchor_regex_any``) must also clear before a specialist label is
emitted. Without that gate, UPS battery / APC SKU prose can embed near wireless
exemplars (``AP`` substring / power vocabulary) and confidently mis-route a
non-WLAN deal — Stinson battery install scored wireless@0.92 with zero WLAN
anchors. The gate abstains rather than guess, keeping keyword pack_prior in
charge.

Off by default. Enable with ``SOWSMITH_SERVICE_ROUTING=1``; head dir via
``SOWSMITH_SERVICE_ROUTER_DIR`` (default ``_contrastive_router``). The dir holds
``store.npz`` + ``knn_meta.json`` + ``best/`` encoder (the GPU artifact).
"""
from __future__ import annotations

import os
import re
from typing import Any

_CAP = 40  # atoms sampled for the scope summary — matches _label_service_types._scope_summary

# Atom types that are BOM/pricing/commercial noise, NOT scope-of-work. The full
# parser emits XLSX BOM line-items as hundreds of `pricing_assumption` atoms that
# drown the actual scope (a TV install reads as 313 cabling/wifi material rows vs
# 52 scope atoms) and flip the route to wireless/cabling. The labeler's parse
# never surfaced that BOM, so the head learned from scope prose — exclude these
# at inference so the representation matches training.
_NOISE_TYPES = frozenset({
    "pricing_assumption", "commercial_total", "rate_card", "line_item",
})

# Evidence anchors required before the neural head may emit a specialist pack.
# Patterns align with OrbitBrief ``domain_packs.yaml`` required_anchor_regex_any
# so parser-os and Orbit cannot disagree on what counts as real WLAN/AV/cabling
# scope. Empty tuple = no gate (label passes through).
_PACK_EVIDENCE_ANCHORS: dict[str, tuple[tuple[str, ...], int]] = {
    "wireless": (
        (
            r"\b(access\s+points?|wlc|wireless\s+controller|wireless\s+lan\s+controller|wlan\s+controller)\b",
            r"\b(wi[- ]?fi\s+heatmap|wireless\s+heatmap|rf\s+heatmap)\b",
            r"\b802\.11(?:ac|ax|be|n|g)\b",
            r"\b(wlan\s+(?:install|deployment|design|cabling)|ap\s+install|ap\s+(?:cabling|drop)s?|access\s+point\s+(?:install|drop|cable)s?)\b",
            r"\b(ssid|wpa[23]|802\.1x\s+wireless|radius\s+wireless|wireless\s+psk)\b",
            r"\b(meraki\s+mr\d{2,3}|aruba\s+ap[- ]?\d{2,3}|cisco\s+(?:cw|air[- ]?(?:ap|cap))\d{2,4}|mist\s+ap\d|ruckus\s+r\d{3}|catalyst\s+9166|cw9166|cw9162|cw9164)\b",
            r"\b(wi[- ]?fi|wlan|wireless\s+(?:ap|access|lan|network|survey|design|install))\b",
        ),
        2,
    ),
    "audio_visual": (
        (
            r"\b(display|projector|video\s+wall|microphone(?:\s+array)?)\b",
            r"\b(biamp|q[- ]?sys|crestron|extron|qsc|shure|symetrix|tesira|polycom|cisco\s+(?:room\s+kit|webex)|logitech\s+rally)\b",
            r"\b(dante|aes67|\bndi\b|sdi|hdbaset)\b",
            r"\b(teams\s+room|zoom\s+room|google\s+meet\s+room|huddle\s+room|conference\s+room\s+(?:av|audio|video)|classroom\s+(?:av|audio|video))\b",
            r"\b(control\s+(?:processor|panel)|video\s+codec|av\s+codec|av\s+(?:rack|equipment|design|drawings?))\b",
        ),
        3,
    ),
    "low_voltage_cabling": (
        (
            r"\b(cat\s*[56]a?|category\s*[56]a?)\b",
            r"\b(fiber|fibre|om[34]|os2|single[- ]?mode|multi[- ]?mode)\b",
            r"\b(cable\s+tray|j[- ]?hook|ladder\s+rack|cable\s+pathway)\b",
            r"\b(permanent\s+link|channel\s+test|fluke|certif(?:y|ication))\b",
            r"\b(idf|mdf|telecom\s+room|tr\s+build[- ]?out)\b",
            r"\b(low[- ]?voltage\s+cabl(?:e|ing)|structured\s+cabl(?:e|ing)|horizontal\s+cabl(?:e|ing))\b",
        ),
        2,
    ),
}


def _router_dir() -> str:
    return os.environ.get("SOWSMITH_SERVICE_ROUTER_DIR", "_contrastive_router")


def _enabled() -> bool:
    return os.environ.get("SOWSMITH_SERVICE_ROUTING", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


_HEAD: Any = None
_LOADED = False


def _load_head():
    global _HEAD, _LOADED
    if _LOADED:
        return _HEAD
    _LOADED = True
    try:
        from app.core.contrastive_type_knn import load_promoted

        _HEAD = load_promoted(registry_dir=_router_dir())
    except Exception:
        _HEAD = None
    return _HEAD


def _scope_summary(atoms: list[Any], documents: list[dict]) -> str:
    """Rebuild the labeler's scope-summary representation from the envelope so the
    head sees in-distribution input (FILES line + diverse-stride atom-body sample)."""
    names = " | ".join(
        str(d.get("filename") or "").rsplit(".", 1)[0]
        for d in documents
        if d.get("filename")
    )[:200]

    def _atype(a) -> str:
        return str(getattr(a, "atom_type", None) or (a.get("atom_type") if isinstance(a, dict) else "") or "")

    # Scope-of-work atoms only; BOM/pricing rows dominate the parse and misroute.
    scope_atoms = [a for a in atoms if _atype(a) not in _NOISE_TYPES]
    if len(scope_atoms) < 5:  # guard: thin scope -> fall back to all atoms
        scope_atoms = list(atoms)
    bodies = [
        t for a in scope_atoms if (t := str(getattr(a, "text", "") or "").strip())
    ]
    if len(bodies) > _CAP:
        bodies = bodies[:: max(1, len(bodies) // _CAP)][:_CAP]
    return f"FILES: {names}\nSCOPE ATOMS:\n" + "\n".join(f"- {b[:160]}" for b in bodies)


def _corpus_text(atoms: list[Any], documents: list[dict]) -> str:
    """Flat text used by the evidence-anchor gate (filenames + atom bodies)."""
    parts: list[str] = []
    for d in documents or []:
        if isinstance(d, dict):
            parts.append(str(d.get("filename") or ""))
            parts.append(str(d.get("name") or ""))
    for a in atoms or []:
        if isinstance(a, dict):
            parts.append(str(a.get("text") or a.get("raw_text") or ""))
        else:
            parts.append(str(getattr(a, "text", "") or getattr(a, "raw_text", "") or ""))
    return "\n".join(parts)


def _evidence_anchor_satisfied(label: str, corpus_text: str) -> bool:
    """True when *label* needs no gate, or corpus clears its required anchors."""
    key = str(label or "").strip().lower()
    spec = _PACK_EVIDENCE_ANCHORS.get(key)
    if not spec:
        return True
    patterns, min_hits = spec
    if not corpus_text.strip():
        return False
    distinct: set[str] = set()
    for pattern in patterns:
        try:
            compiled = re.compile(pattern, re.I)
        except re.error:
            continue
        for m in compiled.finditer(corpus_text):
            distinct.add(m.group(0).lower())
            if len(distinct) >= min_hits:
                return True
    return False


def build_service_routing(atoms: list[Any], documents: list[dict]) -> dict[str, Any]:
    """Classify the deal scope into its primary service pack, or abstain.

    Returns ``{"enabled": False}`` when off / no head, ``{"enabled": True,
    "primary": None, "abstained": True}`` when the head is unsure (OOD or below
    tau), else ``{"enabled": True, "primary": <pack>, "confidence": <float>}``.
    Guess-free by construction (the head abstains rather than guess)."""
    if not _enabled():
        return {"enabled": False}
    head = _load_head()
    if head is None:
        return {"enabled": False}
    summary = _scope_summary(atoms, documents)
    try:
        res = head.classify(summary)
    except Exception:
        return {"enabled": False}
    if not res:
        return {"enabled": True, "primary": None, "confidence": 0.0, "abstained": True}
    label, conf = res
    # The head was trained with an explicit ``other`` class (every service it
    # can't route confidently). Treat that — and the parser's AMBIGUOUS abstain
    # target — as "no opinion" so brief-gen's keyword router stays in charge.
    if str(label).lower() in ("other", "ambiguous"):
        return {"enabled": True, "primary": None, "confidence": round(float(conf), 4), "abstained": True}
    # Neural vote alone is not enough for specialist packs — require real
    # equipment/scope anchors so UPS/APC battery installs cannot become wireless.
    corpus = _corpus_text(atoms, documents) or summary
    if not _evidence_anchor_satisfied(str(label), corpus):
        return {
            "enabled": True,
            "primary": None,
            "confidence": round(float(conf), 4),
            "abstained": True,
            "abstain_reason": "missing_evidence_anchors",
            "neural_primary": str(label),
            "source": "service_router_head",
        }
    return {
        "enabled": True,
        "primary": label,
        "secondary": [],
        "confidence": round(float(conf), 4),
        "source": "service_router_head",
    }
