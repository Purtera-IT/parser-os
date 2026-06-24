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

Off by default. Enable with ``SOWSMITH_SERVICE_ROUTING=1``; head dir via
``SOWSMITH_SERVICE_ROUTER_DIR`` (default ``_contrastive_router``). The dir holds
``store.npz`` + ``knn_meta.json`` + ``best/`` encoder (the GPU artifact).
"""
from __future__ import annotations

import os
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
    try:
        res = head.classify(_scope_summary(atoms, documents))
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
    return {
        "enabled": True,
        "primary": label,
        "secondary": [],
        "confidence": round(float(conf), 4),
        "source": "service_router_head",
    }
