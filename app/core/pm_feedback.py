"""PM correction → training-ready Correction, generic across ALL heads.

One universal entry point the product calls when a PM fixes anything in the brief.
Every head (current and future) maps to a single ``Correction`` row in the
:class:`~app.core.feedback_store.FeedbackStore`, which:
  * fires INSTANTLY via ``decide()`` on the next similar atom (no retrain), and
  * banks as gold for the nightly eval-gated retrain (``app.learning.retrain``).

A new head needs ONE line in ``HEAD_REGISTRY`` — nothing else. The frontend
mirrors this registry (see purpulse.app ``src/lib/orbitbrief/headCorrections.ts``);
the two MUST stay in sync (test: ``tests/test_routes_feedback.py``). The API
layer (``app.api.routes_feedback``) derives its head→relation mapping from this
registry — never duplicate it.
"""
from __future__ import annotations
import dataclasses
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
    "image":     HeadSpec("pdf_image_kind",  "atom",   "Image kind"),
    # A house style is a judgment like any other: "SLO, not SLA, because we
    # guarantee objectives". The verdict IS the preferred wording, so this is
    # an extract head — nothing chooses between a closed set, something is
    # rewritten. Applied only to text WE author; never to quoted evidence.
    "terminology": HeadSpec("preferred_term", "deal", "Preferred wording", mode="extract"),
}

# Deal-scoped PM corrections fire readily WITHIN that deal (the PM explicitly fixed
# it here, so the blast radius is one deal). Global corrections keep the high bar.
_THRESHOLD_DEAL = 0.74
_THRESHOLD_GLOBAL = 0.82


#: A repeated judgment is stronger evidence than a single one, so the same
#: correction made again widens rather than replaces. Never below the bar a
#: single in-deal correction already clears.
_EXEMPLAR_CAP = 12
_THRESHOLD_STEP = 0.02
_THRESHOLD_FLOOR = _THRESHOLD_DEAL
#: Which deals this judgment has been reached on, carried inside `relations`.
_DEALS_KEY = "pm_seen_deals"


def _cid(head: str, deal_id: str, target_id: str, new_value: str, scope: str = SCOPE_DEAL) -> str:
    # A GLOBAL correction is a judgment about the thing, not about the deal it
    # was noticed on: the same rule rejected on three deals must be ONE
    # correction with three exemplars, not three corrections with one each.
    # Live: `mode.av_install.cable_conceal_drywall` was dismissed on deals
    # 3bbb2efa, 221a2bae and 02557291 with identical wording.
    key_deal = "" if scope == SCOPE_GLOBAL else deal_id
    h = hashlib.sha1(f"{head}|{key_deal}|{target_id}|{new_value}".encode()).hexdigest()[:12]
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
      rationale: str   optional WHY, recorded on the correction but kept OUT of
                       the exemplar. A reason describes the judgment, not the
                       thing judged: appending "because our NOC hosts it" to
                       "who provides the customer bridge" moved the prototype
                       far enough that the question it was meant to catch
                       missed by 0.08 cosine.
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
        id=_cid(head, deal_id, target_id, new_value, scope),
        relation=spec.relation,
        verdict=new_value,
        scope=scope,
        scope_key=("" if scope == SCOPE_GLOBAL else deal_id),
        exemplars=[exemplar],
        threshold=(_THRESHOLD_GLOBAL if scope == SCOPE_GLOBAL else _THRESHOLD_DEAL),
        relations={**dict(payload.get("relations") or {}), _DEALS_KEY: [deal_id] if deal_id else []},
        # The PM's own reason rides with the correction, so wherever it fires
        # the brief can say whose judgment this was and why they made it.
        instruction=(
            f"PM {spec.label}: {payload.get('oldValue','?')} → {new_value}"
            + (
                f" — {str(payload.get('rationale') or payload.get('context') or '').strip()}"
                if (payload.get("rationale") or payload.get("context"))
                else ""
            )
        ),
        complaint_id=target_id or None,
        created_by=str(payload.get("pm") or "pm"),
        created_at=now,
        updated_at=now,
    )


def _merge_with_existing(store, corr: Correction) -> Correction:
    """Fold a repeat of the same judgment into the correction it repeats.

    ``store.add`` is INSERT-OR-REPLACE by id, so without this the second time
    a PM makes the same call it overwrites the first and the store learns
    nothing from the repetition. Each new exemplar widens the prototype, and
    the threshold relaxes one step per extra exemplar down to the bar a single
    deal-scoped correction already clears — evidence buys reach, never a free
    pass.
    """
    try:
        prior = store.get(corr.id)
    except Exception:
        prior = None
    if prior is None:
        return corr
    seen = list(getattr(prior, "exemplars", None) or [])
    for ex in corr.exemplars:
        if ex and ex not in seen:
            seen.append(ex)
    seen = seen[:_EXEMPLAR_CAP]

    # Evidence is measured in DEALS, not in repetitions. Six clicks on one
    # wording is one judgment; the same judgment reached independently on
    # three deals is three. Live: the pathway ask was rejected six times
    # across three deals with identical wording.
    rel = dict(getattr(prior, "relations", None) or {})
    deals = list(rel.get(_DEALS_KEY) or [])
    for d in (dict(getattr(corr, "relations", None) or {}).get(_DEALS_KEY) or []):
        if d and d not in deals:
            deals.append(d)
    rel.update(dict(getattr(corr, "relations", None) or {}))
    rel[_DEALS_KEY] = deals[:_EXEMPLAR_CAP]

    evidence = max(len(seen), len(deals))
    base = float(getattr(corr, "threshold", _THRESHOLD_DEAL))
    relaxed = max(_THRESHOLD_FLOOR, round(base - _THRESHOLD_STEP * (evidence - 1), 4))
    return dataclasses.replace(corr, exemplars=seen, threshold=relaxed, relations=rel)


def apply_pm_correction(store, payload: dict[str, Any]) -> str:
    """Ingest a PM correction into the live store. Returns the correction id.

    The fix is honored on the NEXT similar atom immediately (store.resolve) and
    is picked up by the nightly retrain. Works for every head in HEAD_REGISTRY.
    """
    corr = pm_correction_to_correction(payload)
    corr = _merge_with_existing(store, corr)
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
