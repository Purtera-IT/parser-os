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

import hashlib
import os
import re
import zlib
from typing import Any

SCOPE_SUMMARY_VERSION = 2  # bump when the head-facing representation changes

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
    """Rebuild the scope-summary representation the head is scored on.

    FILES line + a consistent 40-atom sample of the scope bodies.

    NOTE FOR WHOEVER TRAINS THE NEXT HEAD: the sampler changed (see below), so
    this is representation v2 and a head fitted on v1's stride sample is out of
    distribution against it. ``SCOPE_SUMMARY_VERSION`` is written into the
    routing provenance for exactly that reason -- a version recorded next to
    the input is the difference between noticing a mismatch and puzzling over a
    head that quietly got worse.
    """
    names = " | ".join(
        str(d.get("filename") or "").rsplit(".", 1)[0]
        for d in documents
        if d.get("filename")
    )[:200]

    def _atype(a) -> str:
        # atom_type is an AtomType *enum* on EvidenceAtom objects (str(enum) would
        # give "AtomType.pricing_assumption", not "pricing_assumption") — use .value.
        at = getattr(a, "atom_type", None)
        if at is None and isinstance(a, dict):
            at = a.get("atom_type")
        return at.value if hasattr(at, "value") else str(at or "")

    def _text(a) -> str:
        # EvidenceAtom stores text under `raw_text` (NOT `text` — `_compact_atom`
        # maps raw_text->"text" only in the envelope dict). Missing this returned
        # "" for every real atom -> empty scope summary -> always routed wireless.
        for attr in ("raw_text", "normalized_text", "text"):
            v = getattr(a, attr, None)
            if v:
                return str(v).strip()
        if isinstance(a, dict):
            for k in ("text", "raw_text", "normalized_text", "body"):
                if a.get(k):
                    return str(a[k]).strip()
        return ""

    # Scope-of-work atoms only; BOM/pricing rows dominate the parse and misroute.
    scope_atoms = [a for a in atoms if _atype(a) not in _NOISE_TYPES]
    if len(scope_atoms) < 5:  # guard: thin scope -> fall back to all atoms
        scope_atoms = list(atoms)
    bodies = [t for a in scope_atoms if (t := _text(a))]
    if len(bodies) > _CAP:
        # Consistent sampling: keep the _CAP atoms with the smallest stable
        # hash OF THEIR OWN TEXT, then restore document order.
        #
        # ``bodies[:: len // _CAP]`` is a step function. The divisor holds
        # across a band of deal sizes and then increments, and at that moment
        # the sample changes wholesale -- so one atom, arriving at the wrong
        # count, could replace almost the entire input to the route. Measured
        # over deals of 150-400 atoms, comparing each size against one atom
        # more (overlap of the selected positions):
        #
        #     sampler        worst 1-atom   mean 1-atom   cliffs below 0.5
        #     stride              0.053         0.974            7
        #     proportional        0.333         0.333          250
        #     this                0.951         0.994            0
        #
        # Proportional spacing was tried first and is worse: it removes the
        # cliffs by making EVERY addition churn the sample. Hashing each atom's
        # own text decouples the decision from the deal's size entirely -- an
        # atom's membership depends on nothing but itself, so adding an
        # unrelated document leaves every existing pick where it was.
        #
        # crc32 rather than hash(): PYTHONHASHSEED randomises str hashing per
        # process, which would make the router's input differ between two runs
        # over identical bytes.
        keyed = sorted(
            range(len(bodies)),
            key=lambda i: (zlib.crc32(bodies[i].encode("utf-8", "replace")), i),
        )
        bodies = [bodies[i] for i in sorted(keyed[:_CAP])]
    return f"FILES: {names}\nSCOPE ATOMS:\n" + "\n".join(f"- {b[:160]}" for b in bodies)


def _conf_ceiling() -> float:
    """Reported-confidence ceiling for the router head. Its raw similarity score
    saturates near 1.0 while held-out accuracy is far lower, so a raw 1.0 is
    dangerously overconfident: downstream (OrbitBrief's gap checklist) would
    treat a confidently-wrong domain label as certain. Clamp the REPORTED
    confidence until the calibrated contrastive head lands. Configurable via
    SOWSMITH_ROUTER_CONF_CEILING; default 0.8."""
    try:
        return float(os.environ.get("SOWSMITH_ROUTER_CONF_CEILING", "0.8"))
    except ValueError:
        return 0.8


def _corpus_text(atoms: list[Any], documents: list[dict]) -> str:
    """Flat text used by the evidence-anchor gate. ATOM BODIES ONLY.

    This included the filenames, which defeated the gate's whole purpose. The
    gate exists because a neural vote can be confidently wrong -- the docstring
    at the top of this module records Stinson's battery install scoring
    wireless@0.92 with zero WLAN anchors -- so it demands real evidence before
    a specialist label may be emitted. A filename is not evidence; it is the
    one attribute of a document anybody can set to anything.

    Measured on a UPS/battery deal whose scope contains no wireless language at
    all:

        documents named attachment_b.pdf / scope_of_work.docx   gate: False
        the same deal, files named
          "Wireless AP Survey.pdf" / "Wi-Fi Heatmap Report.docx" gate: TRUE

    Renaming two files walked the deal straight through the guard built to stop
    exactly that misroute. (Underscored names happened not to fire, because a
    regex word boundary does not break on "_" -- so whether the hole opened
    depended on the customer's naming habit, which is worse than it failing
    outright.)

    ``documents`` is kept in the signature: callers pass it, and the parameter
    documents that the omission is deliberate rather than an oversight.
    """
    del documents
    parts: list[str] = []
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


def _shadow(
    atoms: list[Any],
    documents: list[dict],
    *,
    base: tuple[str, float] | None,
    deal_id: str,
    project_id: str,
    head_result: tuple[str, float] | None,
    base_observed: bool = True,
    summary: str | None = None,
) -> dict[str, Any]:
    """Record both routers' answers and return the queue signal.

    Never raises and never decides: if this fails, routing is unaffected. The
    base's answer is what ships; the head rides along in provenance so its own
    output can never become its training target.
    """
    try:
        from app.core import router_shadow

        if summary is None:
            try:
                summary = _scope_summary(atoms, documents)
            except Exception:
                summary = ""
        record = router_shadow.record(
            deal_id=deal_id,
            project_id=project_id,
            base_label=(base[0] if base else None),
            base_confidence=(base[1] if base else 0.0),
            head_label=(head_result[0] if head_result else None),
            head_confidence=(head_result[1] if head_result else 0.0),
            base_observed=base_observed,
            provenance={"scope_summary_version": SCOPE_SUMMARY_VERSION},
            scope_summary=summary or "",
        )
        return {"shadow": record.as_dict(), "ask_pm": record.ask_pm,
                "ask_reason": record.ask_reason}
    except Exception:  # noqa: BLE001
        return {}


def known_packs() -> list[str]:
    """The labels a PM may choose between when correcting a route.

    A correction chip needs candidates or it cannot offer anything to pick.
    Taken from the loaded head's own classes when there is one, and otherwise
    from the packs the evidence gate knows about -- so the chip works before a
    head exists, which is the situation today and for a while yet.
    """
    packs = set(_PACK_EVIDENCE_ANCHORS)
    head = _load_head()
    labels = getattr(head, "y", None)
    if labels is not None:
        try:
            packs.update(
                str(v) for v in set(labels.tolist())
                if str(v).lower() not in ("ambiguous", "other")
            )
        except Exception:  # noqa: BLE001 - a malformed store must not break routing
            pass
    packs.add("staff_augmentation")
    return sorted(packs)


def build_service_routing(
    atoms: list[Any],
    documents: list[dict],
    *,
    base: tuple[str, float] | None = None,
    base_observed: bool = True,
    deal_id: str = "",
    project_id: str = "",
) -> dict[str, Any]:
    """Classify the deal scope into its primary service pack, or abstain.

    Returns ``{"enabled": False}`` when off / no head, ``{"enabled": True,
    "primary": None, "abstained": True}`` when the head is unsure (OOD or below
    tau), else ``{"enabled": True, "primary": <pack>, "confidence": <float>}``.
    Guess-free by construction (the head abstains rather than guess)."""
    # The head being off or absent is the NORMAL case right now, and it is
    # precisely when collecting matters most: the blob registry holds one
    # training row of one class, so every compile that routes a deal and does
    # not record it is a training example thrown away. Shadow first, decide
    # after.
    head = _load_head() if _enabled() else None
    if head is None:
        shadow = _shadow(atoms, documents, base=base, deal_id=deal_id,
                         project_id=project_id, head_result=None,
                         base_observed=base_observed)
        return {"enabled": False, "candidates": known_packs(), **shadow}
    # Record the EXACT string the head embeds. Every router eval so far has
    # rebuilt this from envelope.json and measured a different function: replaying
    # the shipped head over envelope-derived summaries reproduced only 2 of 9 live
    # routes (2026-08-10), because production strides 40 atoms over the FULL parse
    # while the envelope keeps a compacted subset. Without this the router's input
    # is unobservable and no accuracy number means anything. Bounded by
    # construction: _CAP atoms x 160 chars + a 200-char FILES line, ~6.6 KB.
    try:
        summary = _scope_summary(atoms, documents)
    except Exception:
        summary = ""
    prov = {
        "scope_summary": summary,
        "scope_summary_sha256": hashlib.sha256(summary.encode("utf-8")).hexdigest()[:16],
        "scope_summary_chars": len(summary),
        "scope_summary_version": SCOPE_SUMMARY_VERSION,
    }
    try:
        res = head.classify(summary)
    except Exception:
        return {"enabled": False}
    if not res:
        shadow = _shadow(atoms, documents, base=base, deal_id=deal_id,
                         project_id=project_id, head_result=None,
                         base_observed=base_observed, summary=summary)
        return {"enabled": True, "primary": None, "confidence": 0.0,
                "abstained": True, "candidates": known_packs(), **shadow, **prov}
    label, conf = res
    # Calibration guard: report a CLAMPED confidence (raw kept for transparency).
    # The head's raw score saturates near 1.0 even though its held-out accuracy
    # is far lower, so an unclamped 1.0 would make OrbitBrief treat a
    # confidently-wrong domain label as certain and select the wrong gap
    # checklist. Real high confidence returns with the calibrated head.
    raw_conf = round(float(conf), 4)
    conf = round(min(raw_conf, _conf_ceiling()), 4)
    # The head was trained with an explicit ``other`` class (every service it
    # can't route confidently). Treat that — and the parser's AMBIGUOUS abstain
    # target — as "no opinion" so brief-gen's keyword router stays in charge.
    if str(label).lower() in ("other", "ambiguous"):
        shadow = _shadow(atoms, documents, base=base, deal_id=deal_id,
                         project_id=project_id, head_result=None,
                         base_observed=base_observed, summary=summary)
        return {"enabled": True, "primary": None, "confidence": conf,
                "raw_confidence": raw_conf, "abstained": True,
                "candidates": known_packs(), **shadow, **prov}
    # A neural vote alone is not enough for a specialist pack -- require real
    # equipment/scope anchors so UPS/APC battery installs cannot become wireless.
    # Stinson scored wireless@0.92 with zero WLAN anchors. The gate abstains
    # rather than guess, leaving the keyword pack_prior in charge.
    corpus = _corpus_text(atoms, documents) or summary
    if not _evidence_anchor_satisfied(str(label), corpus):
        return {
            "enabled": True,
            "primary": None,
            "confidence": conf,
            "raw_confidence": raw_conf,
            "abstained": True,
            "abstain_reason": "missing_evidence_anchors",
            "neural_primary": str(label),
            "source": "service_router_head",
            "candidates": known_packs(),
            **_shadow(atoms, documents, base=base, deal_id=deal_id,
                      project_id=project_id, head_result=None,
                         base_observed=base_observed, summary=summary),
            **prov,
        }
    return {
        "enabled": True,
        "primary": label,
        "secondary": [],
        "confidence": conf,
        "raw_confidence": raw_conf,
        "source": "service_router_head",
        "candidates": known_packs(),
        **_shadow(atoms, documents, base=base, deal_id=deal_id,
                  project_id=project_id, head_result=(str(label), conf),
                  base_observed=base_observed, summary=summary),
        **prov,
    }
