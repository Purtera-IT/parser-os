"""PM correction → training-ready Correction, generic across ALL heads.

One universal entry point the product calls when a PM fixes anything in the brief.
Every head (current and future) maps to a single ``Correction`` row in the
:class:`~app.core.feedback_store.FeedbackStore`, which:
  * fires INSTANTLY via ``decide()`` on the next similar atom (no retrain), and
  * banks as gold for the nightly eval-gated retrain (``app.learning.retrain``).

A new head needs ONE line in ``HEAD_REGISTRY`` — nothing else. The frontend
mirrors this registry (see purpulse.app ``src/lib/orbitbrief/headCorrections.ts``);
the two MUST stay in sync (test: ``_test_pm_feedback.py``).
"""
from __future__ import annotations
import hashlib
import time
from dataclasses import dataclass
from typing import Any

from app.core.feedback_store import Correction, SCOPE_DEAL, SCOPE_GLOBAL


@dataclass(frozen=True)
class HeadSpec:
    relation: str          # the decide() relation this head's corrections govern
    kind: str              # what the PM is pointing at: atom | edge | gap | entity | deal
    label: str             # human label for the UI / provenance
    mode: str = "classify" # "classify" (verdict ∈ candidate set, resolve()-driven)
                           #  | "extract" (verdict is a value the head extracts; stored
                           #     as gold for retrain, applied at extraction time not resolve)


# ── the single source of truth for EVERY trainable head ───────────────────────
# Add a row here + the mirror in headCorrections.ts and a new head is fully wired
# into the correction loop (UI affordance → store → instant-learn → retrain).
HEAD_REGISTRY: dict[str, HeadSpec] = {
    "type":      HeadSpec("atom_type",       "atom",   "Atom type"),
    "admission": HeadSpec("admission",       "atom",   "Keep / drop"),
    "gap":       HeadSpec("gap_valid",       "gap",    "Gap"),
    "conflict":  HeadSpec("edge_relation",   "edge",   "Cross-doc conflict"),
    "site":      HeadSpec("same_site",       "entity", "Site identity"),
    "norm":      HeadSpec("value_norm",      "atom",   "Value / amount", mode="extract"),
    "router":    HeadSpec("service_routing", "deal",   "Workstream / domain"),
    "facet":     HeadSpec("facet",           "atom",   "Brief section"),
}

# Deal-scoped PM corrections fire readily WITHIN that deal (the PM explicitly fixed
# it here, so the blast radius is one deal). Global corrections keep the high bar.
_THRESHOLD_DEAL = 0.74
_THRESHOLD_GLOBAL = 0.82


def _cid(head: str, deal_id: str, target_id: str, new_value: str) -> str:
    h = hashlib.sha1(f"{head}|{deal_id}|{target_id}|{new_value}".encode()).hexdigest()[:12]
    return f"pm_{head}_{h}"


def pm_correction_to_correction(payload: dict[str, Any]) -> Correction:
    """Map the universal PM-correction payload → a Correction row. Pure (no I/O).

    payload (the exact JSON the frontend POSTs):
      head:      str   one of HEAD_REGISTRY
      dealId:    str
      targetId:  str   atom_id | edge_id | gap_id | entity_id
      text:      str   the exemplar the PM corrected (atom text / edge "a || b")
      oldValue:  str   what the head said (for provenance / wrongful-override stats)
      newValue:  str   what the PM says it is  → the verdict the head must learn
      scope:     str   "deal" (default) | "global"  (global = applies to all deals)
      pm:        str   who corrected (optional)
      context:   str   optional neighbor/section context (improves the prototype)
      relations: dict  optional structured grounding (e.g. {"authoritative":"a"})
    """
    head = payload["head"]
    spec = HEAD_REGISTRY.get(head)
    if spec is None:
        raise ValueError(f"unknown head {head!r}; add it to HEAD_REGISTRY")
    deal_id = str(payload.get("dealId") or "")
    target_id = str(payload.get("targetId") or "")
    new_value = str(payload["newValue"])
    text = (payload.get("text") or "").strip()
    if not text:
        raise ValueError("PM correction needs `text` (the exemplar to learn from)")
    scope = SCOPE_GLOBAL if payload.get("scope") == "global" else SCOPE_DEAL
    exemplar = (text if not payload.get("context")
                else f"{text}\n[ctx] {payload['context']}")
    now = time.time()
    return Correction(
        id=_cid(head, deal_id, target_id, new_value),
        relation=spec.relation,
        verdict=new_value,
        scope=scope,
        scope_key=("" if scope == SCOPE_GLOBAL else deal_id),
        exemplars=[exemplar],
        threshold=(_THRESHOLD_GLOBAL if scope == SCOPE_GLOBAL else _THRESHOLD_DEAL),
        relations=dict(payload.get("relations") or {}),
        instruction=f"PM {spec.label}: {payload.get('oldValue','?')} → {new_value}",
        complaint_id=target_id or None,
        created_by=str(payload.get("pm") or "pm"),
        created_at=now,
        updated_at=now,
    )


def apply_pm_correction(store, payload: dict[str, Any]) -> str:
    """Ingest a PM correction into the live store. Returns the correction id.

    The fix is honored on the NEXT similar atom immediately (store.resolve) and
    is picked up by the nightly retrain. Works for every head in HEAD_REGISTRY.
    """
    corr = pm_correction_to_correction(payload)
    store.add(corr)
    # Mirror to blob so the worker (which runs decide() during compile) and the
    # nightly retrain see this correction too, and it survives container
    # recycles. Gated + best-effort: a no-op unless SOWSMITH_FEEDBACK_BLOB is on.
    try:
        from app.core import feedback_blob as _fb

        _fb.upload_correction(corr)
    except Exception:  # pragma: no cover - mirroring must never break a fix
        pass
    # Durable training signal: store.add() above only makes the fix fire on the
    # NEXT similar atom (instant learning). For the head to durably LEARN it, the
    # nightly eval-gated retrain needs a gold TrainingRow — which this path never
    # wrote (the "+ gold row for the nightly retrain" promise was unkept). Log
    # one gold row per exemplar, mirroring complaint_intake.confirm, and mirror
    # the rows to blob so they reach the worker's training log. Never raises;
    # no-op when SOWSMITH_TRAINING_LOG_DB is unset.
    try:
        from app.core.training_log import TEACHER_PM, TrainingRow, log_rows

        _deal_id = corr.scope_key if corr.scope == SCOPE_DEAL else ""
        _rows = [
            TrainingRow(
                relation=corr.relation,
                label=corr.verdict,
                raw_text=ex,
                label_kind="judgment",
                teacher=TEACHER_PM,
                confidence=1.0,
                scope=corr.scope,
                scope_key=corr.scope_key,
                deal_id=_deal_id,
                complaint_id=corr.complaint_id,
                provenance={"stage": "pm_correction", "instruction": corr.instruction},
            )
            for ex in corr.exemplars
            if ex and ex.strip()
        ]
        if _rows:
            log_rows(_rows)
            try:
                from app.core import feedback_blob as _fb2

                _fb2.upload_training_rows(corr.id, _rows)
            except Exception:  # pragma: no cover
                pass
    except Exception:  # pragma: no cover - training-log is additive, never fatal
        pass
    return corr.id
