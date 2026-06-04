"""Base learning: turn the lexical gates INTO the store, so the regex can die.

The pipeline carries ~600+ hand-curated keyword/regex entries that encode real
human knowledge — "a school/hospital/warehouse is a site", "'the facility' /
'bid opening' / a standards-code prefix is NOT a site", "Cisco/Aruba/Genetec
are vendors". That knowledge is correct; the *mechanism* (exact/substring regex)
is what fails — it never generalizes to the next deal's phrasing and it can't be
corrected by a PM.

This module harvests that knowledge as the store's **base learning layer**:
every keyword phrase becomes an embedding exemplar of a scoped
:class:`~app.core.feedback_store.Correction`. The store then makes the same
calls *semantically* — so "elementary school" generalizes to "the lower school",
"PreK-5 building", etc., which the frozenset never matched — and every entry is
now an inspectable, override-able row instead of buried code.

Crucially this does NOT duplicate the lists: each :class:`GateSpec` *imports the
live constant*, so the gate's own frozenset is the single source of truth. The
regex stays in place as the training data; it is deleted only once
:func:`verify_gate` proves the seeded store reproduces it on a held-out probe
set with **zero collateral** (eval invariant D), run against the live embedding
model. The frozenset becomes training data, then it's thrown away.

Deterministic in tests: the store's embedder is injected, so seeding,
resolution, and verification are exercised with no network.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Callable

from app.core.feedback_store import (
    SCOPE_GLOBAL,
    Correction,
    FeedbackStore,
    seed_default_corrections,
)

# The site decision the geo-fallback / detection gates all serve: is this
# surface string a real physical job site, or not a site at all?
RELATION_IS_SITE = "is_physical_site"
SITE_CANDIDATES = ["site", "not_site"]

# The vendor decision the entity gates serve: is this token a product/service
# vendor brand (so it must never be minted as a site or a customer)?
RELATION_IS_VENDOR = "is_vendor_token"
VENDOR_CANDIDATES = ["vendor", "not_vendor"]


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:48] or "x"


@dataclass
class GateSpec:
    """A declarative description of one lexical gate, ready to seed.

    Attributes:
        id: short stable id used to namespace the seeded correction ids.
        relation: the decide() relation this gate governs.
        candidates: the closed verdict set for that relation.
        groups: ``verdict -> exemplar phrases``. Each phrase becomes one
            single-exemplar correction (a kNN prototype), so resolution is
            nearest-neighbor over the gate's vocabulary — the semantic version
            of the substring match the regex did.
        scope / threshold: applied to every correction seeded from this spec.
        source: where the knowledge lives (``"module:CONSTANT"``) — kept on each
            correction's ``complaint_id`` for provenance and for knowing which
            regex a passing verification lets us delete.
    """

    id: str
    relation: str
    candidates: list[str]
    groups: dict[str, list[str]]
    scope: str = SCOPE_GLOBAL
    threshold: float = 0.82
    source: str = ""


def _safe_import(module: str, name: str):
    """Import a live gate constant, tolerating a rename/removal so bootstrap
    never hard-fails on one missing list."""
    try:
        import importlib

        return getattr(importlib.import_module(module), name, None)
    except Exception:  # pragma: no cover - defensive
        return None


def _phrases(obj) -> list[str]:
    """Coerce a frozenset/list/dict-of-aliases gate constant into a flat,
    de-duplicated list of exemplar phrases."""
    out: list[str] = []
    if obj is None:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and k.strip():
                out.append(k)
            if isinstance(v, (list, tuple, set, frozenset)):
                out.extend(str(x) for x in v if isinstance(x, str) and x.strip())
    elif isinstance(obj, (list, tuple, set, frozenset)):
        out.extend(str(x) for x in obj if isinstance(x, str) and x.strip())
    # De-dup, preserve order.
    seen: set[str] = set()
    uniq: list[str] = []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def default_gate_specs() -> list[GateSpec]:
    """Build specs from the live gate constants. Defensive: a renamed/removed
    constant simply contributes no exemplars rather than breaking bootstrap."""
    site_yes = (
        _phrases(_safe_import("app.core.site_detection", "_STRONG_FACILITY_TAILS"))
        + _phrases(_safe_import("app.core.site_llm_verify", "_STRONG_FACILITY_ANCHORS"))
        + _phrases(_safe_import("app.core.site_llm_verify", "_WEAK_FACILITY_ANCHORS"))
    )
    site_no = (
        _phrases(_safe_import("app.core.site_detection", "_SITE_BLOCKLIST"))
        + _phrases(_safe_import("app.core.site_llm_verify", "_OBVIOUS_NON_SITES"))
        + _phrases(_safe_import("app.core.site_llm_verify", "_CODE_SHAPE_PREFIX_DENYLIST"))
    )
    vendor_yes = (
        _phrases(_safe_import("app.core.site_llm_verify", "_VENDOR_SIGNAL_TOKENS"))
        + _phrases(_safe_import("app.core.entity_extraction", "_CROSS_PACK_VENDORS"))
    )

    specs: list[GateSpec] = []
    if site_yes or site_no:
        specs.append(
            GateSpec(
                id="site",
                relation=RELATION_IS_SITE,
                candidates=SITE_CANDIDATES,
                groups={"site": site_yes, "not_site": site_no},
                source="site_detection+site_llm_verify",
            )
        )
    if vendor_yes:
        specs.append(
            GateSpec(
                id="vendor",
                relation=RELATION_IS_VENDOR,
                candidates=VENDOR_CANDIDATES,
                groups={"vendor": vendor_yes},
                source="site_llm_verify._VENDOR_SIGNAL_TOKENS+entity_extraction._CROSS_PACK_VENDORS",
            )
        )
    return specs


def seed_corrections_from_spec(spec: GateSpec) -> list[Correction]:
    """One single-exemplar correction per phrase (a kNN prototype). Ids are
    content-stable so re-seeding REPLACES rather than duplicates."""
    out: list[Correction] = []
    for verdict, phrases in spec.groups.items():
        for phrase in phrases:
            # Slug for human readability + a short content hash so two distinct
            # phrases that slug the same (truncation/punctuation) never collide
            # and silently drop one entry's knowledge.
            digest = hashlib.sha1(phrase.encode("utf-8")).hexdigest()[:8]
            cid = f"gate:{spec.id}:{verdict}:{_slug(phrase)}_{digest}"
            out.append(
                Correction(
                    id=cid,
                    relation=spec.relation,
                    verdict=verdict,
                    scope=spec.scope,
                    exemplars=[phrase],
                    threshold=spec.threshold,
                    instruction=f"Bootstrapped from lexical gate '{spec.source}'.",
                    created_by="gate_bootstrap",
                    complaint_id=f"bootstrap:{spec.source}",
                )
            )
    return out


def bootstrap_store(
    store: FeedbackStore, specs: list[GateSpec] | None = None
) -> int:
    """Seed ``store`` with corrections harvested from the lexical gates.

    Returns the number of corrections written. Idempotent (content-stable ids).
    """
    specs = specs if specs is not None else default_gate_specs()
    n = 0
    for spec in specs:
        for c in seed_corrections_from_spec(spec):
            store.add(c)
            n += 1
    return n


# ── verification: a gate earns its deletion ──────────────────────────────
@dataclass
class GateVerification:
    """How faithfully the seeded store reproduces a gate, on a probe set.

    ``safe_to_delete`` is the deletion criterion: the store must reproduce the
    gate's verdicts at high rate with ZERO cross-verdict misassignments
    (collateral). Run this with the LIVE embedder on HELD-OUT phrasings before
    removing the regex — resolving the gate's own entries (which are in the
    store) is only a sanity floor.
    """

    spec_id: str
    relation: str
    total: int = 0
    reproduced: int = 0
    misassigned: int = 0
    silent: int = 0
    misassign_examples: list[str] = field(default_factory=list)

    @property
    def reproduction_rate(self) -> float:
        return self.reproduced / self.total if self.total else 0.0

    @property
    def safe_to_delete(self) -> bool:
        return self.misassigned == 0 and self.reproduction_rate >= 0.95

    def summary(self) -> str:
        flag = "SAFE-TO-DELETE" if self.safe_to_delete else "KEEP-REGEX"
        return (
            f"[{flag}] gate '{self.spec_id}' ({self.relation}): "
            f"{self.reproduced}/{self.total} reproduced "
            f"({self.reproduction_rate:.0%}), {self.misassigned} misassigned, "
            f"{self.silent} silent"
        )


def verify_gate(
    store: FeedbackStore,
    spec: GateSpec,
    *,
    probes: dict[str, list[str]] | None = None,
) -> GateVerification:
    """Check that ``store`` reproduces ``spec`` on a probe set.

    Args:
        probes: ``verdict -> phrases`` to test. Defaults to the spec's own
            exemplars (sanity floor). Supply HELD-OUT paraphrases here for a
            real generalization + collateral verification.
    """
    from app.core.decide import DecisionScope

    probes = probes if probes is not None else spec.groups
    v = GateVerification(spec_id=spec.id, relation=spec.relation)
    for expected, phrases in probes.items():
        for phrase in phrases:
            d = store.resolve(
                relation=spec.relation,
                text=phrase,
                candidates=spec.candidates,
                context="",
                scope=DecisionScope(),
                instruction="",
                relations=None,
            )
            v.total += 1
            if d is None or d.verdict is None:
                v.silent += 1
            elif d.verdict == expected:
                v.reproduced += 1
            else:
                v.misassigned += 1
                if len(v.misassign_examples) < 10:
                    v.misassign_examples.append(f"{phrase!r}: →{d.verdict} (want {expected})")
    return v


def bootstrap_default_store(
    db_path: str, *, embed_fn: Callable | None = None, verify: bool = True
) -> int:
    """Create/populate a feedback-store DB with the base learning layer plus the
    built-in global corrections (PurTera). For ops to prime
    ``SOWSMITH_FEEDBACK_STORE_DB``. Returns total corrections written.

    By default (``verify=True``) each harvested gate is **verify-gated before it
    is written**: a spec is seeded into the live store only if the store
    reproduces it cleanly (``safe_to_delete`` — high reproduction, ZERO
    collateral) against the configured embedder. This is the same criterion the
    regex deletion is gated on, applied at seed time, so a gate whose knowledge
    is contextual rather than lexical (site-ness — bare facility tokens collide
    with non-sites on the real embedder) is **skipped, not leaked** as
    collateral-prone bare-token corrections. The contextual knowledge still
    enters the store the right way: as rich exemplars from real-deal / PM
    corrections. Set ``verify=False`` to force-seed every gate (e.g. with a
    deterministic embedder in tests)."""
    store = FeedbackStore(db_path, embed_fn=embed_fn)
    n = 0
    for spec in default_gate_specs():
        if verify:
            # Verify on a throwaway twin so a failing gate never touches the
            # live store. Self-probe is a sufficient floor: a contextual gate
            # fails to reproduce even its own bare tokens on the real embedder.
            twin = store.evaluation_twin(extra=seed_corrections_from_spec(spec))
            if not verify_gate(twin, spec).safe_to_delete:
                continue
        n += bootstrap_store(store, [spec])
    n += seed_default_corrections(store)
    return n


__all__ = [
    "GateSpec",
    "GateVerification",
    "RELATION_IS_SITE",
    "RELATION_IS_VENDOR",
    "SITE_CANDIDATES",
    "VENDOR_CANDIDATES",
    "default_gate_specs",
    "seed_corrections_from_spec",
    "bootstrap_store",
    "verify_gate",
    "bootstrap_default_store",
]


if __name__ == "__main__":  # pragma: no cover - ops entry point
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "feedback_store.db"
    written = bootstrap_default_store(target)
    print(f"bootstrapped {written} corrections into {target}")
