"""Span-admission seam — make *recall* text-ruleable.

The keep/drop seam (multi_entity_llm) can only ever SUPPRESS a confident learned
reject ("never fabricate a keep"). That keeps the LLM honest but leaves a gap: a
PM cannot teach the system to *catch* a class of content it missed (milestones,
quantities, requirements…) by writing a sentence — because there is no decision
seam where such an admission could be learned.

This module adds that seam. For each retained-but-unpromoted atom (anything the
pipeline kept — ``_keep`` or a mis-typed atom — and therefore already has text and
an embedding), it asks decide() whether the atom should be ADMITTED into a span
relation. The contract mirrors keep/drop exactly, inverted:

* **STORE-ONLY** (``llm=False``): admission is never an LLM call, never latency.
* **Guess-free**: an atom is admitted ONLY on a confident learned match to that
  relation's corrections/anchors (cosine ≥ the correction's threshold). Abstain →
  NOT admitted; the atom is left exactly as it was. So similarity alone can never
  fabricate a span the documents don't support.
* **Text-ruleable**: a PM teaches recall by adding a Correction with
  ``relation == verdict == <span relation>`` and a sentence + an example. No code
  change — the seam reads the same store.

Because it only ever reads atoms that were already parsed/retained, it CANNOT
recover content that never became an atom (an unparsed sheet, an OCR failure, a
rolled-up table) — that remains a parser/coverage fix. It converts the middle
class (parsed-but-unpromoted) from code into text.
"""
from __future__ import annotations

import os
from typing import Iterable, Optional

from app.core.decide import DecisionScope, decide

# Span relations the seam may admit into. Universal — extend by adding a
# correction whose relation == verdict == the relation name; no code change here.
ADMISSIBLE_RELATIONS: tuple[str, ...] = (
    "milestones",
    "requirements",
    "quantities",
    "acceptance_criteria",
    "penalties",
    "compliance_obligations",
    "risks",
    "certifications",
    "stakeholders",
    "commercial_line_items",
)


def _instruction(relation: str) -> str:
    return (
        f"Does this text belong to the '{relation}' relation? "
        f"Admit ONLY on a confident learned match; otherwise abstain."
    )


def admit_atom(
    text: str,
    relation: str,
    *,
    scope: Optional[DecisionScope] = None,
    instruction: str = "",
) -> bool:
    """True iff the store confidently says ``text`` should be admitted as ``relation``.

    Pure store decision (``llm=False``). Abstain/uncertain → False (not admitted).
    Never raises into the pipeline.
    """
    if not text or not text.strip():
        return False
    try:
        d = decide(
            relation,
            text.strip()[:600],
            [relation],
            instruction=instruction or _instruction(relation),
            scope=scope or DecisionScope(),
            llm=False,
        )
    except Exception:  # pragma: no cover - the seam must never break a compile
        return False
    return d is not None and d.source == "store" and d.verdict == relation


def admit_relations(
    text: str,
    *,
    relations: Iterable[str] = ADMISSIBLE_RELATIONS,
    scope: Optional[DecisionScope] = None,
) -> list[str]:
    """All span relations this atom should be admitted into (often 0, sometimes 1+).

    An atom can legitimately satisfy more than one relation (e.g. a clause that is
    both an acceptance_criterion and a requirement); the seam returns every
    confident admission and the caller decides how to bank them.
    """
    return [r for r in relations if admit_atom(text, r, scope=scope)]


def admit_atoms(
    atoms: Iterable,
    *,
    text_of=lambda a: getattr(a, "raw_text", None) or getattr(a, "text", "") or "",
    relations: Iterable[str] = ADMISSIBLE_RELATIONS,
    scope: Optional[DecisionScope] = None,
) -> list[tuple[object, list[str]]]:
    """Map each atom to the span relations the store admits it into.

    Returns only atoms with ≥1 admission, as ``(atom, [relations])`` pairs, so the
    caller can emit the new span rows/atoms. Atoms with no confident admission are
    omitted (left untouched — guess-free).
    """
    rels = tuple(relations)
    out: list[tuple[object, list[str]]] = []
    for a in atoms:
        hits = admit_relations(text_of(a), relations=rels, scope=scope)
        if hits:
            out.append((a, hits))
    return out


# ── compiler-facing: re-type retained atoms in place ─────────────────
# At the atom level the "retained / unpromoted" bucket is these generic
# AtomTypes (the keep/drop ``_keep`` label is training-only). An atom that
# survived as one of these is a candidate for recovery into a specific type.
WEAK_ATOM_TYPES: frozenset = frozenset({
    "scope_item", "entity", "deal_metadata", "site_implementation_note",
})

# Communication / non-deal metadata atoms are intentionally typed as
# deal_metadata with a structured ``value.kind``. They must not be
# "recovered" into stakeholder / requirement / scope. Covers email chrome
# and transcript greeting/intro/logistics (``conversation_meta``).
_PROTECTED_META_KINDS: frozenset = frozenset({
    "email_addressee",
    "email_body_context",
    "email_header",
    "conversation_meta",
})


def _is_protected_email_atom(atom) -> bool:
    """True for non-deal communication metadata (email + transcript chrome)."""
    val = getattr(atom, "value", None)
    if not isinstance(val, dict):
        return False
    return str(val.get("kind") or "") in _PROTECTED_META_KINDS

# The specific types a retained atom may be recovered into — the span-recall
# targets plus the commercial categories. Only valid AtomType values.
RECOVERABLE_ATOM_TYPES: tuple = (
    "milestone_phase", "lead_time_constraint", "blackout_date_range",
    "integration_checkpoint", "cutover_step",
    "requirement", "submission_req", "acceptance_criterion",
    "electrical_acceptance_test", "eval_criterion",
    "quantity", "compliance_rule", "compliance_classification", "bonding_insurance",
    "deliverable", "dependency", "change_order_rule", "payment_term",
    "site_access_window", "pricing_assumption",
    "bom_line", "material", "expense", "pmo", "license_subscription", "service_line",
)


def readmit_atom_types(
    atoms,
    *,
    weak: frozenset = WEAK_ATOM_TYPES,
    candidates: tuple = RECOVERABLE_ATOM_TYPES,
    scope: Optional[DecisionScope] = None,
) -> int:
    """Re-type retained/weak atoms into a recovered AtomType. Prefers the trained
    ADMISSION HEADS (logistic, precision-tuned, embedder-pinned) when a registry
    exists — they generalise far past cosine (≈0.83/0.83 held-out vs ≈0.17 kNN).
    Falls back to the STORE decide() kNN path when no heads are available. Both
    are guess-free (abstain → atom untouched) and never call the LLM. Mutates
    ``atom.atom_type`` in place; returns the count re-typed. The compiler stage
    is flag-gated (``SOWSMITH_SPAN_ADMISSION``) so it is a no-op until enabled.
    """
    heads = _load_admission_heads()
    if heads:
        return _readmit_via_heads(atoms, heads, weak)
    return _readmit_via_store(atoms, weak, list(candidates), scope)


def _load_admission_heads():
    try:
        from app.core.admission_head import AdmissionRegistry
        from app.core.embedding_retrieval import _embed_model
        reg = os.environ.get("SOWSMITH_ADMISSION_REGISTRY", "_admission_heads")
        return AdmissionRegistry(reg).load_all(embed_model=_embed_model())
    except Exception:
        return {}


def _readmit_via_heads(atoms, heads, weak) -> int:
    """Embed weak atoms once, score every head, re-type to the highest-confidence
    relation whose probability clears that head's precision-tuned threshold."""
    import numpy as np
    from app.core.schemas import AtomType
    from app.core.admission_head import RELATION_TO_ATOM_TYPE
    from app.core.embedding_retrieval import embed_texts

    targets = [
        a
        for a in atoms
        if getattr(getattr(a, "atom_type", None), "value", getattr(a, "atom_type", None)) in weak
        and (getattr(a, "raw_text", "") or "").strip()
        and not _is_protected_email_atom(a)
    ]
    if not targets:
        return 0
    texts = [(a.raw_text or "").strip()[:600] for a in targets]
    try:
        V = np.asarray(embed_texts(texts), dtype=np.float32)
    except Exception:
        return 0
    if V.ndim != 2 or V.shape[0] != len(targets):
        return 0
    n = 0
    items = [(r, h) for r, h in heads.items() if r in RELATION_TO_ATOM_TYPE]
    for i, a in enumerate(targets):
        best_rel, best_margin = None, 0.0
        for rel, h in items:
            p = float(h.proba(V[i])[0])
            if p >= h.threshold and (p - h.threshold) >= best_margin:
                best_rel, best_margin = rel, p - h.threshold
        if best_rel is not None:
            try:
                a.atom_type = AtomType(RELATION_TO_ATOM_TYPE[best_rel])
                n += 1
            except Exception:
                pass
    return n


def _readmit_via_store(atoms, weak, cand, scope) -> int:
    from app.core.schemas import AtomType
    n = 0
    for a in atoms:
        cur = getattr(a, "atom_type", None)
        cur_v = getattr(cur, "value", cur)
        if cur_v not in weak:
            continue
        if _is_protected_email_atom(a):
            continue
        text = (getattr(a, "raw_text", "") or "").strip()
        if not text:
            continue
        try:
            d = decide(
                "atom_type", text[:600], cand,
                instruction=("Re-type this retained atom into its specific type "
                             "only if confident; otherwise abstain."),
                scope=scope or DecisionScope(), llm=False,
            )
        except Exception:  # pragma: no cover
            continue
        if d is not None and d.source == "store" and d.verdict and d.verdict != cur_v:
            try:
                a.atom_type = AtomType(d.verdict)
                n += 1
            except Exception:
                pass
    return n


__all__ = [
    "ADMISSIBLE_RELATIONS", "admit_atom", "admit_relations", "admit_atoms",
    "WEAK_ATOM_TYPES", "RECOVERABLE_ATOM_TYPES", "readmit_atom_types",
]
