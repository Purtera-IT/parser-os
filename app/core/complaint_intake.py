"""Complaint intake: turn a PM's "this is wrong" into a durable correction.

A complaint is the raw signal of a human catching a mistake: "PurTera is our
company, it shouldn't be a site", "you missed the Santa Fe location", "this row
is a price-book line, not deal scope". The job of this module is to convert that
into a :class:`app.core.feedback_store.Correction` the store can enforce forever
— **without ever acting on its own**. A human always confirms before anything is
committed.

The pipeline is five explicit steps:

1. **Localize** — find the atom(s) the complaint is about, across BOTH the
   accepted atoms and the *retained suppressed* atoms (Phase 1). Without
   retention an omission complaint ("you dropped X") had nothing to point at;
   now the dropped atom is right there, flagged with the stage that removed it.
2. **Diagnose** — read why the wrong outcome happened: for a suppressed atom,
   the stage and reason are stamped on it (``value["_suppression"]``); for a
   misclassification, the atom's current type/verdict.
3. **Generalize** — build a *proposed* correction: a relation-grounded verdict
   with the offending text as its embedding exemplar, scoped as the PM asked
   (deal / pack / global). This is the leap from "fix this one" to "never again,
   anywhere semantically similar".
4. **Confirm** — surface the diagnosis + a dry-run preview to the human. Nothing
   is written. ``confirm()`` is a separate, explicit call.
5. **Commit** — only on confirm: add the active correction to the store.

Pure and side-effect-free until ``confirm()``. No LLM, no I/O (the store does
its own). ``intake()`` never raises on bad input — it returns a resolution whose
``localized`` list may be empty so the caller can ask the PM for a better anchor.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from app.core.feedback_store import (
    SCOPE_DEAL,
    SCOPE_GLOBAL,
    SCOPE_PACK,
    Correction,
    FeedbackStore,
)

# Complaint kinds. They differ only in where the offending atom lives and how we
# phrase the diagnosis; the remedy is always the same shape — a relation-grounded
# verdict correction.
KIND_WRONGLY_DROPPED = "wrongly_dropped"   # a real thing was suppressed
KIND_WRONGLY_KEPT = "wrongly_kept"         # a non-thing was accepted (e.g. PurTera site)
KIND_MISCLASSIFIED = "misclassified"       # accepted but wrong type/verdict


@dataclass
class Complaint:
    """A PM's report that a specific judgment was wrong.

    Attributes:
        relation: the decide() family the bad judgment belongs to
            (e.g. ``"physical_site"``, ``"atom_type"``).
        desired_verdict: what the answer SHOULD have been — one of the verdicts
            that decision uses. This becomes the correction's verdict.
        text: the snippet the complaint is about. Used both to localize the atom
            and as the correction's embedding exemplar. If omitted, ``atom_id``
            must be given and the atom's own text is used.
        atom_id: optional exact anchor when the PM clicked a specific atom.
        kind: one of the ``KIND_*`` constants (advisory; drives wording only).
        scope/scope_key: where the resulting correction should apply.
        note: the PM's free-text explanation (kept for audit).
    """

    relation: str
    desired_verdict: str
    text: str = ""
    atom_id: str = ""
    kind: str = KIND_WRONGLY_KEPT
    scope: str = SCOPE_GLOBAL
    scope_key: str = ""
    note: str = ""
    created_by: str = ""
    id: str = field(default_factory=lambda: f"cmp_{uuid.uuid4().hex[:12]}")
    created_at: float = field(default_factory=time.time)


@dataclass
class LocalizedAtom:
    """One atom the complaint plausibly refers to, with where it ended up."""

    atom_id: str
    text: str
    bucket: str  # "accepted" | "suppressed"
    suppression_stage: str = ""
    suppression_reason: str = ""
    atom: Any = None


@dataclass
class ComplaintResolution:
    """The result of intake: evidence + a *proposed* (uncommitted) correction.

    ``proposed_correction.status == "proposed"`` until :func:`confirm` flips it
    to ``"active"`` and writes it. ``preview`` is a human-readable dry run.
    """

    complaint: Complaint
    localized: list[LocalizedAtom]
    diagnosis: str
    proposed_correction: Correction
    preview: str
    committed: bool = False


# ── helpers ──────────────────────────────────────────────────────────
def _atom_type_str(atom: Any) -> str:
    at = getattr(atom, "atom_type", None)
    return at.value if hasattr(at, "value") else str(at or "")


def _atom_text(atom: Any) -> str:
    return (
        getattr(atom, "raw_text", None)
        or getattr(atom, "normalized_text", None)
        or getattr(atom, "text", None)
        or ""
    )


def _suppression_of(atom: Any) -> tuple[str, str]:
    """Return ``(stage, reason)`` stamped by the Phase 1 suppression ledger."""
    val = getattr(atom, "value", None) or {}
    if isinstance(val, dict):
        supp = val.get("_suppression") or {}
        if isinstance(supp, dict):
            return str(supp.get("stage") or ""), str(supp.get("reason") or "")
    return "", ""


def _matches(query: str, atom_text: str) -> bool:
    q, a = query.strip().lower(), atom_text.strip().lower()
    if not q or not a:
        return False
    return q in a or a in q


def _localize(complaint: Complaint, result: Any) -> list[LocalizedAtom]:
    """Find atoms the complaint refers to across accepted + suppressed sets."""
    accepted = list(getattr(result, "atoms", None) or []) if result is not None else []
    suppressed = (
        list(getattr(result, "suppressed_atoms", None) or [])
        if result is not None
        else []
    )
    out: list[LocalizedAtom] = []
    for bucket, atoms in (("accepted", accepted), ("suppressed", suppressed)):
        for a in atoms:
            aid = getattr(a, "id", "") or ""
            atext = _atom_text(a)
            hit = (complaint.atom_id and aid == complaint.atom_id) or (
                not complaint.atom_id and _matches(complaint.text, atext)
            )
            if not hit:
                continue
            stage, reason = _suppression_of(a) if bucket == "suppressed" else ("", "")
            out.append(
                LocalizedAtom(
                    atom_id=aid,
                    text=atext,
                    bucket=bucket,
                    suppression_stage=stage,
                    suppression_reason=reason,
                    atom=a,
                )
            )
    return out


def _diagnose(complaint: Complaint, localized: list[LocalizedAtom]) -> str:
    if not localized:
        return (
            "Could not localize the complaint to any accepted or suppressed atom. "
            "The correction will still be created from the complaint text, but "
            "provide an atom_id or exact snippet for a tighter anchor."
        )
    parts: list[str] = []
    for loc in localized:
        if loc.bucket == "suppressed":
            stage = loc.suppression_stage or "an upstream stage"
            reason = loc.suppression_reason or "no reason recorded"
            parts.append(
                f"atom {loc.atom_id} was SUPPRESSED by '{stage}' "
                f"({reason}); the deal lost it silently before retention."
            )
        else:
            cur = _atom_type_str(loc.atom)
            parts.append(
                f"atom {loc.atom_id} was ACCEPTED as '{cur or 'unknown'}'."
            )
    return " ".join(parts)


def _exemplar_text(complaint: Complaint, localized: list[LocalizedAtom]) -> str:
    if complaint.text.strip():
        return complaint.text.strip()
    for loc in localized:
        if loc.text:
            return loc.text
    return ""


def intake(
    complaint: Complaint, *, result: Any = None, store: FeedbackStore | None = None
) -> ComplaintResolution:
    """Localize, diagnose, and propose — but never commit.

    Args:
        complaint: the PM's report.
        result: the :class:`CompileResult` the complaint is about (so we can
            search its accepted + suppressed atoms). Optional; without it the
            correction is built from the complaint text alone.
        store: unused for proposal, accepted for symmetry / future dedup checks.

    Returns:
        A :class:`ComplaintResolution` with an uncommitted proposed correction.
        Never raises.
    """
    try:
        localized = _localize(complaint, result)
    except Exception:  # pragma: no cover - localization must not break intake
        localized = []
    diagnosis = _diagnose(complaint, localized)
    exemplar = _exemplar_text(complaint, localized)

    proposed = Correction(
        id=f"corr_{uuid.uuid4().hex[:12]}",
        relation=complaint.relation,
        verdict=complaint.desired_verdict,
        scope=complaint.scope,
        scope_key=complaint.scope_key,
        exemplars=[exemplar] if exemplar else [],
        instruction=complaint.note,
        complaint_id=complaint.id,
        created_by=complaint.created_by,
        status="proposed",  # NOT active until confirm()
    )

    scope_desc = {
        SCOPE_GLOBAL: "every deal (global)",
        SCOPE_PACK: f"the '{complaint.scope_key}' pack",
        SCOPE_DEAL: f"deal '{complaint.scope_key}'",
    }.get(complaint.scope, complaint.scope)
    n = len(localized)
    preview = (
        f"PROPOSED (not committed): on relation '{complaint.relation}', text "
        f"semantically like {exemplar!r} will resolve to "
        f"'{complaint.desired_verdict}' across {scope_desc}. "
        f"Localized {n} matching atom(s). Call confirm() to enforce, or discard."
    )
    return ComplaintResolution(
        complaint=complaint,
        localized=localized,
        diagnosis=diagnosis,
        proposed_correction=proposed,
        preview=preview,
    )


def confirm(store: FeedbackStore, resolution: ComplaintResolution) -> Correction:
    """Commit the proposed correction to the store as ACTIVE.

    This is the only step that mutates anything. Refuses a correction with no
    exemplar (nothing to embed → it could never fire). Idempotent per
    resolution: a second call is a no-op that returns the same correction.
    """
    if resolution.committed:
        return resolution.proposed_correction
    c = resolution.proposed_correction
    if not c.exemplars or not any(e.strip() for e in c.exemplars):
        raise ValueError(
            "cannot confirm a correction with no exemplar text to embed; "
            "supply complaint.text or a localizable atom_id"
        )
    c.status = "active"
    c.updated_at = time.time()
    store.add(c)
    resolution.committed = True

    # Teacher-logging tap #3 (gold): a confirmed PM correction is the scarce,
    # high-weight signal. Log one TrainingRow per exemplar so the student head
    # learns the *rule* (delexicalized) the PM just taught. Never raises, no-op
    # when training-log is off (SOWSMITH_TRAINING_LOG_DB unset).
    try:
        from app.core.training_log import TEACHER_PM, TrainingRow, log_rows
        _deal_id = c.scope_key if c.scope == "deal" else ""
        _rows = [
            TrainingRow(
                relation=c.relation,
                label=c.verdict,
                raw_text=ex,
                label_kind="judgment",
                teacher=TEACHER_PM,
                confidence=1.0,
                scope=c.scope,
                scope_key=c.scope_key,
                deal_id=_deal_id,
                complaint_id=c.complaint_id,
                provenance={"stage": "pm_correction", "instruction": c.instruction},
            )
            for ex in c.exemplars
            if ex and ex.strip()
        ]
        log_rows(_rows)
    except Exception:
        pass

    return c


def reject(resolution: ComplaintResolution) -> None:
    """Discard a proposed correction. Pure marker — nothing was written."""
    resolution.proposed_correction.status = "rejected"


__all__ = [
    "Complaint",
    "ComplaintResolution",
    "LocalizedAtom",
    "intake",
    "confirm",
    "reject",
    "KIND_WRONGLY_DROPPED",
    "KIND_WRONGLY_KEPT",
    "KIND_MISCLASSIFIED",
]
