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
* ``POST /correction`` — one-tap PM chip correction (mirrors frontend
  ``HEAD_CORRECTIONS``): localize → confirm → instant store + gold training row.

The feedback store is shared process-wide via :func:`decide.get_store`; it is
activated by ``SOWSMITH_FEEDBACK_STORE_DB`` (same switch the compiler uses), so
these endpoints are inert until a store is wired — they 409 rather than guess.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.complaint_intake import (
    KIND_MISCLASSIFIED,
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

# Mirrors purpulse-frontend src/lib/orbitbrief/headCorrections.ts HEAD_CORRECTIONS.
HEAD_REGISTRY: dict[str, str] = {
    "type": "atom_type",
    "admission": "admission",
    "gap": "gap_valid",
    "conflict": "edge_relation",
    "site": "same_site",
    "norm": "value_norm",
    "router": "service_routing",
    "facet": "facet",
    "image": "pdf_image_kind",
}


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
    fired_instantly: bool = False


class CorrectionRequest(BaseModel):
    """One-tap chip payload from Scope Cockpit (BFF maps camelCase → snake_case)."""

    head: str
    deal_id: str = ""
    compile_id: str = ""
    target_id: str = ""
    text: str
    old_value: str = ""
    new_value: str
    scope: str = "deal"
    context: str = ""
    relations: dict[str, Any] = Field(default_factory=dict)
    candidates: list[str] = Field(default_factory=list)
    pm: str = ""


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
    # Compile result is best-effort context for localization. The stateless
    # service has no local sqlite DB — a missing/unopenable DB must not 500 an
    # otherwise-valid complaint (same pattern as /feedback/correction).
    try:
        result = _load_compile_result(project_id)
    except KeyError:
        raise HTTPException(
            status_code=404, detail=f"Unknown project '{project_id}'"
        ) from None
    except Exception:
        result = None

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


@router.post("/{project_id}/feedback/correction", response_model=FeedbackResponse)
def feedback_correction(project_id: str, req: CorrectionRequest) -> FeedbackResponse:
    """PM one-tap chip → instant FeedbackStore correction + gold training row.

  Chip corrections are explicit PM verdict picks (not free-text rules), so we
  commit via :func:`complaint_intake.confirm` without the full probe gate —
  the PM's selection *is* the gold label. Training rows are written inside
  ``confirm()``; the next compile's ``decide()`` store tier can fire immediately.
    """
    store = _require_store()
    relation = HEAD_REGISTRY.get(req.head.strip().lower())
    if not relation:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown head '{req.head}'. Expected one of: {sorted(HEAD_REGISTRY)}",
        )
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text is required")
    if req.new_value is None or not str(req.new_value).strip():
        raise HTTPException(status_code=422, detail="new_value is required")

    scope, scope_key = _scope_from_chip(req.scope, project_id)
    note = (
        f"chip:{req.head} {req.old_value!r}→{req.new_value!r}"
        + (f" ctx={req.context}" if req.context else "")
    )
    complaint = Complaint(
        relation=relation,
        desired_verdict=str(req.new_value).strip(),
        text=req.text.strip(),
        atom_id=req.target_id.strip(),
        kind=KIND_MISCLASSIFIED,
        scope=scope,
        scope_key=scope_key,
        note=note,
        created_by=req.pm.strip() or project_id,
    )
    # Compile result is best-effort CONTEXT for the correction (intake handles
    # None). The service is stateless — results live in blob, not a local sqlite
    # DB — so a missing/unopenable DB must NOT crash an otherwise-valid PM fix.
    try:
        result = _load_compile_result(project_id)
    except Exception:  # KeyError (unknown project) | sqlite OperationalError | etc.
        result = None

    resolution = intake(complaint, result=result, store=store)
    try:
        corr = confirm(store, resolution)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None

    fired = False
    exemplar = (corr.exemplars[0] if corr.exemplars else req.text).strip()
    candidates = req.candidates or [corr.verdict]
    try:
        hit = store.resolve(
            relation=relation,
            text=exemplar,
            candidates=list(candidates),
            context=req.context or "",
            scope=_scope_obj(scope, scope_key),
            instruction=corr.instruction,
            relations=corr.relations or req.relations,
        )
        fired = hit is not None and hit.verdict == corr.verdict
    except Exception:  # pragma: no cover - probe only
        fired = False

    return FeedbackResponse(
        committed=True,
        correction_id=corr.id,
        relation=corr.relation,
        verdict=corr.verdict,
        report="CHIP committed (instant-learning)",
        diagnosis=resolution.diagnosis,
        preview=resolution.preview,
        fired_instantly=fired,
    )


def _scope_from_chip(scope: str, project_id: str) -> tuple[str, str]:
    s = (scope or "deal").strip().lower()
    if s == "global":
        return SCOPE_GLOBAL, ""
    if s == "pack":
        return SCOPE_PACK, project_id
    return SCOPE_DEAL, project_id


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
