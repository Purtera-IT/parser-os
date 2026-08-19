"""Noise-suppression compile stage — store-driven drop of reference/template atoms.

Consumes the universal gate seeded by :mod:`app.core.noise_suppression_seed`
(relation ``atom_noise_admission`` → ``drop`` | ``keep``). For each noise-prone
atom it asks the feedback store (``decide(..., llm=False)`` — store-only, never
an LLM call) whether the atom is reference/template content that should be
dropped from deal scope.

Hard contracts (mirror the rest of the learning-loop seams):

* **Guess-free.** Only a CONFIDENT learned ``drop`` (``source == "store"``)
  suppresses an atom. Store-undecided / no store / unreachable embedder → the
  atom is kept untouched. The gate can only ever remove an error a taught rule
  confidently matches; it never introduces a model-driven judgment.
* **Opt-in.** Off unless ``SOWSMITH_NOISE_SUPPRESSION`` is truthy, so default
  compiles and the whole test suite are byte-identical until enabled.
* **Lossless.** This function only PARTITIONS atoms; the compiler diverts the
  dropped set into the retained-suppression ledger (``capture_suppressed``), so
  every dropped atom is still auditable and localizable for omission complaints.
* **Bounded blast radius.** Only atom types that empirically carry the noise
  (rate cards, materials catalogs, rate-label-as-person) are ever examined, so a
  scope_item / constraint / decision atom is never at risk.
"""

from __future__ import annotations

import os

from app.core.decide import DecisionScope, decide, get_store
from app.core.schemas import AtomType, EvidenceAtom
from app.core.noise_suppression_seed import (
    NOISE_CANDIDATES,
    NOISE_DROP_VERDICT,
    PERSON_NOISE_RELATION,
    PRICING_NOISE_RELATION,
)

# Each noise-prone atom type is examined against its OWN type-scoped gate, so a
# concept can only ever fire on the type it was taught for. This is what makes a
# real after-hours DEAL RATE (pricing_assumption) safe: it is only checked
# against the rate-card/catalog gate, never the rate-label-as-person gate.
_TYPE_TO_RELATION: dict[AtomType, str] = {
    AtomType.pricing_assumption: PRICING_NOISE_RELATION,
    AtomType.stakeholder: PERSON_NOISE_RELATION,
}

# Everything else is passed through untouched (zero collateral risk).
NOISE_PRONE_TYPES: frozenset[AtomType] = frozenset(_TYPE_TO_RELATION)

_INSTRUCTION = (
    "Decide whether this atom is reference/template content (a master rate-card "
    "or materials-catalog row, or a billing time-window / rate label mis-read as "
    "a person) that does not belong to this deal's scope, versus real deal "
    "evidence to keep."
)


def is_enabled() -> bool:
    return os.environ.get("SOWSMITH_NOISE_SUPPRESSION", "").strip().lower() not in (
        "",
        "0",
        "false",
        "no",
        "off",
    )


def _atom_text(atom: EvidenceAtom) -> str:
    text = getattr(atom, "raw_text", "") or getattr(atom, "normalized_text", "") or ""
    if not text:
        val = getattr(atom, "value", None)
        if isinstance(val, dict):
            text = str(val.get("text") or "")
    return text.strip()


def suppress_noise_atoms(
    atoms: list[EvidenceAtom],
    *,
    project_id: str = "",
) -> tuple[list[EvidenceAtom], list[EvidenceAtom]]:
    """Partition ``atoms`` into (kept, dropped) using the store-only noise gate.

    Returns the kept list (order-preserved) and the dropped list. A no-op
    returning ``(atoms, [])`` when disabled, when no store is wired, or when the
    store has nothing confident to say — so callers can always use the result
    unconditionally.
    """
    if not is_enabled():
        return atoms, []
    if get_store() is None:
        return atoms, []

    scope = DecisionScope(deal_id=project_id or "")
    kept: list[EvidenceAtom] = []
    dropped: list[EvidenceAtom] = []

    for atom in atoms:
        atom_type = getattr(atom, "atom_type", None)
        relation = _TYPE_TO_RELATION.get(atom_type)
        if relation is None:
            kept.append(atom)
            continue
        text = _atom_text(atom)
        if not text:
            kept.append(atom)
            continue
        try:
            d = decide(
                relation,
                text,
                list(NOISE_CANDIDATES),
                instruction=_INSTRUCTION,
                scope=scope,
                llm=False,
            )
        except Exception:  # pragma: no cover - gate must never break a compile
            kept.append(atom)
            continue
        if d.source == "store" and d.verdict == NOISE_DROP_VERDICT:
            dropped.append(atom)
        else:
            kept.append(atom)

    return kept, dropped


__all__ = ["suppress_noise_atoms", "is_enabled", "NOISE_PRONE_TYPES"]
