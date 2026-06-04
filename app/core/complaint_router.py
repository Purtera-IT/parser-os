"""Complaint root-cause router: decide *what kind of fix* a PM complaint needs.

The feedback store can only ever fix one class of mistake: a wrong judgment at a
``decide()`` **seam**. It keys a learned correction on an atom's embedding and
re-resolves the next semantically-similar atom. That is genuinely self-healing —
but only when the offending content actually *reached* a decide() seam.

A PM, however, files every complaint the same way: "this is wrong." Some of those
are seam mistakes (self-healing). Others are:

* **GATE bugs** — a deterministic code rule (a prose keep/drop test, the xlsx
  sheet-role router, an OCR step) killed the content before any seam saw it. A
  store correction has nothing to attach to and cannot re-admit it. The honest
  fix is a localized code change at the named rule.
* **Never-detected** — the parser never produced a region for this content at
  all (the independent content census reports it UNCOVERED). There is no atom,
  no drop record, nothing to learn from. The fix is extractor code.

If we route all three into :func:`app.core.complaint_intake.intake` and commit a
store correction, the GATE-bug and never-detected complaints get a correction
that *looks* applied but silently never fires — the PM re-reports the same miss
next deal. This module prevents that by classifying the root cause **first**,
using the provenance the pipeline already records:

* the **span ledger** (:class:`app.core.span_ledger.SpanLedger`) tags every
  content drop ``GATE`` or ``SEAM`` — the mechanical "parser issue vs decide
  issue" answer;
* the **retained suppressed atoms** carry ``value["_suppression"]`` and
  ``decision_provenance`` (a decide()-sourced suppression is a SEAM; a
  parser-stage suppression is a GATE);
* the **content census** (:class:`app.core.content_census.ContentCensus`) is the
  independent denominator that catches content no stage ever recorded.

The router emits a :class:`RoutingVerdict` with two orthogonal booleans —
``learnable`` (a store correction will actually enforce the fix) and ``code_fix``
(a localized code change is required) — so the "and if both" case is first-class:
a recurring GATE bug can get an *immediate* kNN band-aid (learnable) while still
flagging the real code fix (code_fix), instead of one masking the other.

Pure and side-effect-free. No LLM, no I/O. ``route()`` never raises; an
unlocalizable complaint comes back ``UNLOCALIZED`` so the caller can ask the PM
for a better anchor rather than committing a correction that can never fire.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.core.complaint_intake import (
    Complaint,
    ComplaintResolution,
    LocalizedAtom,
    _localize,
    intake,
)
from app.core.feedback_store import FeedbackStore


class RootCause(str, Enum):
    """Where the mistake actually lives — which determines what can fix it."""

    SEAM_CORRECTION = "seam_correction"  # wrong decide() judgment → store learns
    GATE_BUG = "gate_bug"                # deterministic rule dropped it → code fix
    NEEDS_EXTRACTOR = "needs_extractor"  # never detected (census UNCOVERED) → code
    UNLOCALIZED = "unlocalized"          # can't find it anywhere → ask the PM


@dataclass
class RoutingVerdict:
    """The router's diagnosis of a single complaint.

    Attributes:
        complaint: the PM's report, unchanged.
        root_cause: the classified bucket.
        learnable: True iff a feedback-store correction can actually enforce the
            fix (the content reached, or can reach, a decide() seam). When True
            the caller should commit ``resolution.proposed_correction``.
        code_fix: True iff a localized code change is required to fully fix it.
            For a pure SEAM correction this is False; for a GATE bug it is True;
            for a recurring GATE bug it can be True *alongside* ``learnable`` —
            the kNN band-aid plus the real fix.
        recoverable: True iff the content still exists somewhere (a retained
            suppressed atom / dropped-sheet marker), so a PM can pull it now even
            before any code fix lands. False for never-detected content.
        fix_target: a concrete pointer to the fix — the GATE rule/stage name for
            a code fix, the census region location for a never-detected miss, or
            the decide() relation for a seam correction.
        evidence: one-line human-readable justification of the classification.
        localized: atoms the complaint was matched to (accepted + suppressed).
        resolution: the proposed (uncommitted) store correction when
            ``learnable``; ``None`` otherwise (committing it would never fire).
    """

    complaint: Complaint
    root_cause: RootCause
    learnable: bool
    code_fix: bool
    recoverable: bool
    fix_target: str
    evidence: str
    localized: list[LocalizedAtom] = field(default_factory=list)
    resolution: ComplaintResolution | None = None

    @property
    def fix_shape(self) -> str:
        """Plain-English summary of what action this verdict calls for."""
        if self.learnable and self.code_fix:
            return (
                "store correction (immediate, self-healing) AND a localized code "
                f"fix at: {self.fix_target}"
            )
        if self.learnable:
            return "store correction (self-healing); no code change needed"
        if self.root_cause is RootCause.NEEDS_EXTRACTOR:
            return f"extractor code change — content never detected at: {self.fix_target}"
        if self.root_cause is RootCause.GATE_BUG:
            return f"localized code fix at the deterministic rule: {self.fix_target}"
        return "needs a better anchor from the PM (could not localize)"


# ── provenance probes ─────────────────────────────────────────────────
def _decision_sourced(atom: Any) -> bool:
    """True iff this atom's outcome was set at a decide() seam.

    A decide()-driven suppression stamps ``decision_provenance`` with a tier
    (``store``/``llm``/``fallback``). Its presence means the content *reached* a
    seam — so a learned correction can re-resolve it. A parser-stage suppression
    (e.g. the xlsx sheet-role router) leaves no such stamp.
    """
    prov = getattr(atom, "decision_provenance", None)
    if isinstance(prov, dict) and prov.get("source"):
        return True
    return False


def _ledger_kind_for(text: str, ledger: Any) -> str:
    """Return ``"gate"`` / ``"seam"`` / ``""`` for the drop matching ``text``.

    Consults the span ledger's *lost* records (a span that never produced any
    downstream representation) and the broader drop list. Matching is by text
    containment, mirroring complaint localization. The ledger is the
    authoritative "parser issue vs decide issue" signal.
    """
    if ledger is None or not text:
        return ""
    q = text.strip().lower()
    if not q:
        return ""

    def _match(records: list[Any]) -> str:
        for d in records:
            raw = (getattr(d, "raw_text", "") or "").strip().lower()
            if not raw:
                continue
            if q in raw or raw in q:
                kind = getattr(d, "kind", None)
                return getattr(kind, "value", str(kind or "")).lower()
        return ""

    # Prefer lost records (no representation at all) — those are the true misses.
    try:
        hit = _match(list(ledger.lost_records()))
    except Exception:
        hit = ""
    if hit:
        return hit
    try:
        return _match(list(getattr(ledger, "drops", []) or []))
    except Exception:
        return ""


def _census_uncovered_location(text: str, census: Any) -> str:
    """If ``text`` matches an UNCOVERED census region, return its location.

    The census is the independent denominator: a region it reports UNCOVERED was
    *never detected* by the parser — there is no atom and (usually) no ledger
    drop, so neither the store nor a gate-loosening helps. Only extractor code
    can recover it. Returns ``""`` when the census covers (or doesn't know about)
    the text.

    Expects a census the caller has already ``reconcile``-d against the emitted
    atoms; ``ContentCensus.uncovered()`` then lists exactly the silently-lost
    regions. We never call ``reconcile`` here (it needs the atom set and mutates
    census state), so routing stays a pure read.
    """
    if census is None or not text:
        return ""
    q = text.strip().lower()
    if not q:
        return ""
    try:
        uncovered = list(census.uncovered())
    except Exception:
        return ""
    for r in uncovered:
        rtext = (getattr(r, "text", "") or "").strip().lower()
        loc = getattr(r, "location", "") or getattr(r, "region_id", "")
        if rtext and (q in rtext or rtext in q):
            return loc
        # Binary region (image/embedded): no text to match on, but the PM may
        # name its location ("the schematic on page 3"); match the locator.
        rloc = (loc or "").strip().lower()
        if rloc and rloc in q:
            return loc
    return ""


# ── the router ─────────────────────────────────────────────────────────
def route(
    complaint: Complaint,
    *,
    result: Any = None,
    ledger: Any = None,
    census: Any = None,
    store: FeedbackStore | None = None,
) -> RoutingVerdict:
    """Classify a complaint's root cause and decide whether the store can fix it.

    Args:
        complaint: the PM's report (relation + desired verdict + text/atom_id).
        result: the ``CompileResult`` (accepted + retained-suppressed atoms).
        ledger: optional :class:`SpanLedger` — the authoritative GATE/SEAM tag.
        census: optional :class:`ContentCensus` — the independent
            never-detected denominator.
        store: optional feedback store (passed through to ``intake``).

    Returns:
        A :class:`RoutingVerdict`. When ``learnable`` is True it carries a
        proposed (uncommitted) correction in ``resolution`` ready for
        ``complaint_intake.confirm``. Never raises.
    """
    try:
        localized = _localize(complaint, result)
    except Exception:  # pragma: no cover - localization must not break routing
        localized = []

    text = complaint.text or (localized[0].text if localized else "")
    ledger_kind = _ledger_kind_for(text, ledger)
    uncovered_loc = _census_uncovered_location(text, census)

    accepted = [l for l in localized if l.bucket == "accepted"]
    suppressed = [l for l in localized if l.bucket == "suppressed"]

    # ── 1. Localized to an ACCEPTED atom ────────────────────────────────
    # It reached a decide() seam and was kept/typed. Whether the PM says it was
    # wrongly kept or misclassified, the remedy is a learned verdict at that
    # seam. Pure SEAM correction — self-healing, no code change.
    if accepted:
        return _seam_verdict(
            complaint, localized, result, store,
            evidence=(
                f"matched {len(accepted)} accepted atom(s); the decide() seam "
                f"for relation '{complaint.relation}' produced this outcome, so "
                "a learned correction re-resolves it (self-healing)."
            ),
        )

    # ── 2. Localized to a SUPPRESSED (retained) atom ────────────────────
    # The content survived as a marker, so it is recoverable now. Whether a store
    # correction can *re-admit* it depends on HOW it was suppressed:
    #   • at a decide() seam (decision_provenance present, or ledger=SEAM)
    #     → SEAM correction, learnable.
    #   • by a deterministic parser stage (ledger=GATE, or no decision stamp)
    #     → GATE bug; the store can't re-admit it. Code fix at that stage.
    if suppressed:
        loc = suppressed[0]
        seam_like = ledger_kind == "seam" or any(
            _decision_sourced(l.atom) for l in suppressed
        )
        if seam_like:
            return _seam_verdict(
                complaint, localized, result, store,
                evidence=(
                    f"matched a suppressed atom dropped at a decide() seam "
                    f"('{loc.suppression_stage or 'decide'}'); a learned "
                    "correction re-admits it (self-healing)."
                ),
            )
        # GATE-suppressed. Recoverable (the marker holds the content) but only a
        # code fix at the deterministic stage truly fixes it. If it ALSO recurs
        # semantically we still offer the store as a band-aid (learnable+code).
        stage = loc.suppression_stage or ledger_kind or "deterministic parser gate"
        return RoutingVerdict(
            complaint=complaint,
            root_cause=RootCause.GATE_BUG,
            learnable=False,
            code_fix=True,
            recoverable=True,
            fix_target=stage,
            evidence=(
                f"matched a suppressed atom dropped by a deterministic stage "
                f"('{stage}', reason: {loc.suppression_reason or 'n/a'}). The "
                "store acts only at decide() seams, so it cannot re-admit this; "
                "loosen the gate in code (the content is retained, so the PM can "
                "recover it now)."
            ),
            localized=localized,
        )

    # ── 3. Not localized to any atom ────────────────────────────────────
    # No accepted or suppressed atom carries this text. Two sub-cases:
    #   • census says a region is UNCOVERED → the parser never detected it.
    #     Nothing to learn from. Extractor code.
    #   • ledger has a GATE lost-record → a deterministic stage dropped it
    #     before any atom existed. Code fix at that rule.
    if uncovered_loc:
        return RoutingVerdict(
            complaint=complaint,
            root_cause=RootCause.NEEDS_EXTRACTOR,
            learnable=False,
            code_fix=True,
            recoverable=False,
            fix_target=uncovered_loc,
            evidence=(
                f"the content census reports region '{uncovered_loc}' UNCOVERED "
                "— the parser never produced an atom for it. There is nothing for "
                "the store to learn from; this needs extractor code."
            ),
            localized=localized,
        )

    if ledger_kind == "gate":
        return RoutingVerdict(
            complaint=complaint,
            root_cause=RootCause.GATE_BUG,
            learnable=False,
            code_fix=True,
            recoverable=False,
            fix_target=_ledger_rule_for(text, ledger) or "deterministic GATE",
            evidence=(
                "the span ledger shows this content died at a deterministic GATE "
                "with no surviving atom; a store correction cannot reach it. "
                "Localized code fix at the named rule."
            ),
            localized=localized,
        )

    if ledger_kind == "seam":
        # Ledger says a seam dropped it but we have no retained atom to anchor to.
        # Still learnable in principle, but the PM must give a better anchor for
        # the exemplar; surface as seam with whatever text we have.
        return _seam_verdict(
            complaint, localized, result, store,
            evidence=(
                "the span ledger attributes this loss to a decide() seam; a "
                "learned correction applies, but provide an exact snippet for a "
                "tighter exemplar."
            ),
        )

    # ── 4. Nothing matched anywhere ─────────────────────────────────────
    return RoutingVerdict(
        complaint=complaint,
        root_cause=RootCause.UNLOCALIZED,
        learnable=False,
        code_fix=False,
        recoverable=False,
        fix_target="",
        evidence=(
            "could not localize the complaint to any accepted atom, suppressed "
            "atom, ledger drop, or census region. Ask the PM for an atom_id or an "
            "exact snippet before committing any correction."
        ),
        localized=localized,
    )


def _ledger_rule_for(text: str, ledger: Any) -> str:
    """Return the GATE rule name for the lost record matching ``text``."""
    if ledger is None or not text:
        return ""
    q = text.strip().lower()
    try:
        for d in ledger.lost_records():
            raw = (getattr(d, "raw_text", "") or "").strip().lower()
            if raw and (q in raw or raw in q):
                rule = getattr(d, "rule", "") or getattr(d, "stage", "")
                return str(rule)
    except Exception:
        pass
    return ""


def _seam_verdict(
    complaint: Complaint,
    localized: list[LocalizedAtom],
    result: Any,
    store: FeedbackStore | None,
    *,
    evidence: str,
) -> RoutingVerdict:
    """Build a learnable verdict, attaching a proposed (uncommitted) correction."""
    resolution: ComplaintResolution | None
    try:
        resolution = intake(complaint, result=result, store=store)
    except Exception:  # pragma: no cover - intake must not break routing
        resolution = None
    return RoutingVerdict(
        complaint=complaint,
        root_cause=RootCause.SEAM_CORRECTION,
        learnable=True,
        code_fix=False,
        recoverable=True,
        fix_target=complaint.relation,
        evidence=evidence,
        localized=localized,
        resolution=resolution,
    )


__all__ = [
    "RootCause",
    "RoutingVerdict",
    "route",
]
