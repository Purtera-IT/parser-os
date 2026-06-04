"""PM feedback loop — the human-in-the-loop endpoints (upgrade #4).

A PM who spots a wrong judgment must be able to teach the parser without writing
code, and a learned rule must never commit on the PM's say-so alone — it clears
the same nine-invariant verify-gate every other correction does.

Three endpoints, all under ``/projects/{project_id}/feedback``:

* ``POST /rule``        — plain English → :func:`plain_rule_compiler.compile_rule`
                          (synthesize a structured rule, gate it, commit on a
                          clean pass). The end-to-end "type a sentence" loop.
* ``POST /complaint``   — a structured complaint about a specific atom →
                          :func:`complaint_intake.intake` (localize against the
                          saved compile result's accepted + suppressed atoms) →
                          :func:`correction_eval.gated_confirm` when probe sets
                          are supplied. Without probes it returns the *proposed*
                          (uncommitted) correction so the PM can add controls —
                          a commit is never made ungated.
* ``GET  /corrections`` — the learned rules currently in force (provenance).

The feedback store is shared process-wide via :func:`decide.get_store`; it is
activated by ``SOWSMITH_FEEDBACK_STORE_DB`` (same switch the compiler uses), so
these endpoints are inert until a store is wired — they 409 rather than guess.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.complaint_intake import (
    KIND_WRONGLY_KEPT,
    Complaint,
    confirm,
    intake,
)
from app.core.correction_eval import Probe, gated_confirm
from app.core.decide import DecisionScope, get_store
from app.core.feedback_store import SCOPE_DEAL, SCOPE_GLOBAL, SCOPE_PACK
from app.core.plain_rule_compiler import compile_rule
from app.storage.repositories import _load_compile_result

router = APIRouter(prefix="/projects", tags=["feedback"])


def _require_store():
    """Return the wired feedback store, or 409 if none is active. Attempts the
    env-driven wiring the compiler uses, so a fresh process self-activates."""
    store = get_store()
    if store is None:
        # Try the same opt-in wiring the compiler performs.
        try:
            from app.core.compiler import _maybe_wire_feedback_store

            _maybe_wire_feedback_store()
            store = get_store()
        except Exception:  # pragma: no cover - defensive
            store = None
    if store is None:
        raise HTTPException(
            status_code=409,
            detail=(
                "No feedback store is active. Set SOWSMITH_FEEDBACK_STORE_DB to "
                "enable the learning loop."
            ),
        )
    return store


def _scope_obj(scope: str, scope_key: str) -> DecisionScope:
    if scope == SCOPE_DEAL:
        return DecisionScope(deal_id=scope_key)
    if scope == SCOPE_PACK:
        return DecisionScope(pack_id=scope_key)
    return DecisionScope()


# ── request / response models ────────────────────────────────────────
class RuleRequest(BaseModel):
    sentence: str = Field(..., description="The PM's plain-English rule.")
    created_by: str = ""
    scope: str = SCOPE_GLOBAL
    scope_key: str = ""


class ProbeSpec(BaseModel):
    text: str
    candidates: list[str] = Field(default_factory=list)


class ComplaintRequest(BaseModel):
    relation: str
    desired_verdict: str
    candidates: list[str] = Field(default_factory=list)
    text: str = ""
    atom_id: str = ""
    kind: str = KIND_WRONGLY_KEPT
    scope: str = SCOPE_GLOBAL
    scope_key: str = ""
    note: str = ""
    created_by: str = ""
    # Optional verify-gate probe sets. A commit happens ONLY when controls are
    # supplied (so invariant D — no collateral — can actually be checked).
    paraphrases: list[str] = Field(default_factory=list)
    controls: list[ProbeSpec] = Field(default_factory=list)


class FeedbackResponse(BaseModel):
    committed: bool
    correction_id: str
    relation: str
    verdict: str
    report: str
    diagnosis: str = ""
    preview: str = ""
    failed_invariants: list[str] = Field(default_factory=list)


# ── endpoints ─────────────────────────────────────────────────────────
@router.post("/{project_id}/feedback/rule", response_model=FeedbackResponse)
def feedback_rule(project_id: str, req: RuleRequest) -> FeedbackResponse:
    store = _require_store()
    try:
        compiled = compile_rule(
            req.sentence,
            store,
            created_by=req.created_by or project_id,
            default_scope=req.scope,
            default_scope_key=req.scope_key,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    return FeedbackResponse(
        committed=compiled.committed,
        correction_id=compiled.proposal_correction_id,
        relation=compiled.proposal.relation,
        verdict=compiled.proposal.verdict,
        report=compiled.report.summary(),
        diagnosis=compiled.explain(),
        failed_invariants=compiled.report.failed(),
    )


@router.post("/{project_id}/feedback/complaint", response_model=FeedbackResponse)
def feedback_complaint(project_id: str, req: ComplaintRequest) -> FeedbackResponse:
    store = _require_store()
    try:
        result = _load_compile_result(project_id)
    except KeyError:
        raise HTTPException(
            status_code=404, detail=f"Unknown project '{project_id}'"
        ) from None

    complaint = Complaint(
        relation=req.relation,
        desired_verdict=req.desired_verdict,
        text=req.text,
        atom_id=req.atom_id,
        kind=req.kind,
        scope=req.scope,
        scope_key=req.scope_key,
        note=req.note,
        created_by=req.created_by or project_id,
    )
    resolution = intake(complaint, result=result, store=store)
    corr = resolution.proposed_correction

    # Without controls we cannot prove "no collateral" (invariant D), so we never
    # commit ungated: return the proposed correction for the PM to complete.
    if not req.controls:
        return FeedbackResponse(
            committed=False,
            correction_id=corr.id,
            relation=corr.relation,
            verdict=corr.verdict,
            report="UNGATED (no control probes supplied)",
            diagnosis=resolution.diagnosis,
            preview=resolution.preview,
        )

    candidates = req.candidates or [req.desired_verdict]
    sc = _scope_obj(req.scope, req.scope_key)
    exemplar = (corr.exemplars[0] if corr.exemplars else req.text).strip()

    fix_probes = [
        Probe(exemplar, req.relation, req.desired_verdict, list(candidates), sc)
    ]
    gen_probes = [
        Probe(p.strip(), req.relation, req.desired_verdict, list(candidates), sc)
        for p in req.paraphrases
        if p.strip()
    ]
    coll_probes = [
        Probe(
            c.text.strip(),
            req.relation,
            None,
            list(c.candidates or candidates),
            sc,
        )
        for c in req.controls
        if c.text.strip()
    ]

    committed, report = gated_confirm(
        store,
        resolution,
        fix_probes=fix_probes,
        generalization_probes=gen_probes,
        collateral_probes=coll_probes,
    )
    return FeedbackResponse(
        committed=committed,
        correction_id=corr.id,
        relation=corr.relation,
        verdict=corr.verdict,
        report=report.summary(),
        diagnosis=resolution.diagnosis,
        preview=resolution.preview,
        failed_invariants=report.failed(),
    )


class CorrectionView(BaseModel):
    id: str
    relation: str
    verdict: str
    scope: str
    scope_key: str
    exemplars: list[str]
    instruction: str
    status: str
    hit_count: int
    created_by: str


@router.get("/{project_id}/feedback/corrections")
def list_feedback_corrections(
    project_id: str, status: str = "active"
) -> dict[str, Any]:
    store = _require_store()
    if not hasattr(store, "list_corrections"):
        raise HTTPException(
            status_code=501, detail="store does not support listing corrections"
        )
    rows = store.list_corrections(status=status or None)
    items = [
        CorrectionView(
            id=c.id,
            relation=c.relation,
            verdict=c.verdict,
            scope=c.scope,
            scope_key=c.scope_key,
            exemplars=list(c.exemplars),
            instruction=c.instruction,
            status=c.status,
            hit_count=c.hit_count,
            created_by=c.created_by,
        )
        for c in rows
    ]
    return {"total": len(items), "items": items}
