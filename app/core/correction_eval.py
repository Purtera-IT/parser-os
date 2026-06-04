"""The correction gate: a fix must earn its way in.

A learned correction is powerful and therefore dangerous — a rule that fixes
"PurTera is not a site" must not, as a side effect, start dropping a *real*
site that happens to embed nearby. So no correction is committed on a human's
say-so alone: it is first evaluated against a hold-out of probes and must clear
nine invariants. This module is that gate.

The nine invariants (the contract the whole learning system is judged on):

    A  Complaint understood     — the proposal has something to embed and (when
                                   a result was given) localized the atom.
    B  Fix efficacy = 100%      — the exact thing complained about now resolves
                                   to the desired verdict.
    C  Generalization           — paraphrases of the complaint also resolve
                                   correctly (it's a meaning rule, not a string).
    D  No collateral            — every control probe that should be untouched
                                   resolves identically with and without the
                                   correction. THE HARD ONE; a single regression
                                   fails the gate.
    E  Reversible / inspectable — the correction is a row you can read and
                                   disable; disabling stops it firing.
    F  Faster, not slower       — a store hit is decided without the LLM
                                   (``source == "store"``), so it can only
                                   remove latency, never add it.
    G  Safe fallback            — with the embedding endpoint unreachable the
                                   correction stays silent (returns ``None``),
                                   never a guess.
    H  Conflict handling        — scope precedence holds: a narrower correction
                                   wins where it applies and nowhere else.
    I  Provenance intact        — a decision the correction drives cites its id.

Evaluation runs on an :meth:`FeedbackStore.evaluation_twin` — a throwaway copy —
so a failing candidate never touches the live store. ``gated_confirm`` commits
to the real store only on a clean pass.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from app.core.complaint_intake import ComplaintResolution
from app.core.decide import DecisionScope
from app.core.feedback_store import Correction, FeedbackStore


@dataclass
class Probe:
    """One hold-out case. ``expect`` is the verdict the store should return, or
    ``None`` meaning "the store must stay silent here" (used for control /
    collateral probes that the correction must not disturb)."""

    text: str
    relation: str
    expect: str | None
    candidates: list[str]
    scope: DecisionScope = field(default_factory=DecisionScope)


@dataclass
class InvariantResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class EvalReport:
    results: list[InvariantResult]

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    def failed(self) -> list[str]:
        return [r.name for r in self.results if not r.passed]

    def summary(self) -> str:
        marks = " ".join(
            f"{r.name}{'✓' if r.passed else '✗'}" for r in self.results
        )
        return f"{'PASS' if self.passed else 'FAIL'} [{marks}]"


def _active_copy(correction: Correction) -> Correction:
    c = copy.deepcopy(correction)
    c.status = "active"
    return c


def _resolve(store: FeedbackStore, p: Probe):
    return store.resolve(
        relation=p.relation,
        text=p.text,
        candidates=p.candidates,
        context="",
        scope=p.scope,
        instruction="",
        relations=None,
    )


def evaluate(
    store: FeedbackStore,
    candidate: Correction,
    *,
    fix_probes: list[Probe],
    generalization_probes: list[Probe] | None = None,
    collateral_probes: list[Probe] | None = None,
    resolution: ComplaintResolution | None = None,
) -> EvalReport:
    """Run the nine invariants for ``candidate`` against the probe sets.

    All resolution happens on twins of ``store`` (never the live store):

    * ``baseline`` = current active corrections only.
    * ``trial`` = baseline + ``candidate`` (forced active).

    Args:
        fix_probes: the exact-complaint cases (invariant B). Their ``expect`` is
            the desired verdict.
        generalization_probes: paraphrase cases (invariant C), same ``expect``.
        collateral_probes: controls that must be UNCHANGED (invariant D); their
            ``expect`` is ignored — what matters is baseline == trial.
        resolution: the originating intake resolution (invariant A), optional.
    """
    gen = generalization_probes or []
    coll = collateral_probes or []
    cand = _active_copy(candidate)

    baseline = store.evaluation_twin()
    trial = store.evaluation_twin(extra=[cand])

    results: list[InvariantResult] = []

    # A — complaint understood: there is something to embed.
    has_exemplar = bool(cand.exemplars) and any(e.strip() for e in cand.exemplars)
    localized_ok = resolution is None or bool(resolution.localized) or has_exemplar
    results.append(
        InvariantResult(
            "A",
            has_exemplar and localized_ok,
            "exemplar present" if has_exemplar else "no exemplar to embed",
        )
    )

    # B — fix efficacy 100%: every fix probe hits the desired verdict.
    b_fail = [
        p.text for p in fix_probes
        if (d := _resolve(trial, p)) is None or d.verdict != p.expect
    ]
    results.append(
        InvariantResult("B", not b_fail, f"{len(fix_probes) - len(b_fail)}/{len(fix_probes)} fixed")
    )

    # C — generalization: paraphrases resolve correctly.
    c_fail = [
        p.text for p in gen
        if (d := _resolve(trial, p)) is None or d.verdict != p.expect
    ]
    results.append(
        InvariantResult("C", not c_fail, f"{len(gen) - len(c_fail)}/{len(gen)} generalized")
    )

    # D — no collateral: controls resolve identically with/without candidate.
    changed: list[str] = []
    for p in coll:
        bd = _resolve(baseline, p)
        td = _resolve(trial, p)
        bv = bd.verdict if bd else None
        tv = td.verdict if td else None
        if bv != tv:
            changed.append(f"{p.text!r}: {bv}→{tv}")
    results.append(
        InvariantResult(
            "D", not changed,
            "no controls disturbed" if not changed else "; ".join(changed),
        )
    )

    # E — reversible / inspectable: the row exists and disabling stops it.
    e_pass = False
    e_detail = "candidate not found in trial store"
    if trial.get(cand.id) is not None:
        trial.set_status(cand.id, "disabled")
        if fix_probes:
            after = _resolve(trial, fix_probes[0])
            # After disabling, this correction must not be the one deciding.
            e_pass = after is None or after.correction_id != cand.id
        else:
            e_pass = True
        e_detail = "disable stops firing" if e_pass else "still fires when disabled"
        trial.set_status(cand.id, "active")  # restore for any later use
    results.append(InvariantResult("E", e_pass, e_detail))

    # F — faster not slower: a fix-probe hit is decided by the store, no LLM.
    f_pass = True
    f_detail = "store-decided (no LLM)"
    for p in fix_probes:
        d = _resolve(trial, p)
        if d is None or d.source != "store":
            f_pass = False
            f_detail = "fix probe not store-decided"
            break
    results.append(InvariantResult("F", f_pass, f_detail))

    # G — safe fallback: unreachable endpoint → silent.
    offline = FeedbackStore(
        ":memory:", embed_fn=store._embed_fn, reachable_fn=lambda: False
    )
    offline.add(cand)
    g_pass = all(_resolve(offline, p) is None for p in fix_probes) if fix_probes else True
    results.append(
        InvariantResult("G", g_pass, "silent when offline" if g_pass else "guessed offline")
    )

    # H — conflict handling: a deal-scoped candidate must not leak to other
    # deals. Only meaningful when the candidate is deal-scoped.
    h_pass = True
    h_detail = "scope respected"
    if cand.scope == "deal" and fix_probes:
        other = DecisionScope(deal_id=f"{cand.scope_key}__not")
        leak = []
        for p in fix_probes:
            elsewhere = Probe(p.text, p.relation, p.expect, p.candidates, other)
            d = _resolve(trial, elsewhere)
            if d is not None and d.correction_id == cand.id:
                leak.append(p.text)
        h_pass = not leak
        h_detail = "no cross-deal leak" if h_pass else "leaked to another deal"
    results.append(InvariantResult("H", h_pass, h_detail))

    # I — provenance intact: a driven decision cites the correction id.
    i_pass = True
    i_detail = "cites correction id"
    if fix_probes:
        d = _resolve(trial, fix_probes[0])
        i_pass = d is not None and d.correction_id == cand.id
        i_detail = "cites id" if i_pass else "no correction id on decision"
    results.append(InvariantResult("I", i_pass, i_detail))

    return EvalReport(results=results)


def gated_confirm(
    store: FeedbackStore,
    resolution: ComplaintResolution,
    *,
    fix_probes: list[Probe],
    generalization_probes: list[Probe] | None = None,
    collateral_probes: list[Probe] | None = None,
) -> tuple[bool, EvalReport]:
    """Evaluate the proposed correction; commit to ``store`` only on a clean
    pass of all nine invariants. Returns ``(committed, report)``.

    This is the safe replacement for a bare ``confirm()`` when probe sets are
    available: a correction that would damage a control case is refused before
    it can ever fire in production.
    """
    candidate = resolution.proposed_correction
    report = evaluate(
        store,
        candidate,
        fix_probes=fix_probes,
        generalization_probes=generalization_probes,
        collateral_probes=collateral_probes,
        resolution=resolution,
    )
    if not report.passed:
        return False, report
    from app.core.complaint_intake import confirm

    confirm(store, resolution)
    return True, report


__all__ = [
    "Probe",
    "InvariantResult",
    "EvalReport",
    "evaluate",
    "gated_confirm",
]
