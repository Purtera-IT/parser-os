"""Task admission — the compile consumer for the Deal Kit's ``admission`` head.

A PM who removes a proposed task with "not this job" teaches
``admission → drop`` on the task's own sentence (Deal Kit, publish lessons).
Until this stage existed that lesson was banked and never asked for: the next
compile of the same documents proposed the same line again.

Mirrors :mod:`app.core.noise_suppression`: store-only (``llm=False``), guess-free
(only a CONFIDENT learned ``drop`` removes a task; abstention keeps it), lossless
(the compiler diverts dropped atoms into the suppression ledger), and scoped to
the deal so a lesson taught at deal scope holds there and a global one holds
everywhere.
"""

from __future__ import annotations

from typing import Any

from app.core.decide import DecisionScope, decide, get_store
from app.core.task_tier_classifier import _atom_text, _atom_type_str

ADMISSION_RELATION = "admission"
ADMISSION_CANDIDATES: tuple[str, str] = ("keep", "drop")
_INSTRUCTION = "Is this line work this deal quotes (keep), or something a PM has said is not this job (drop)?"


def drop_taught_out_tasks(atoms: list[Any], *, project_id: str = "") -> tuple[list[Any], list[Any]]:
    """Partition task atoms into (kept, dropped) by the learned admission gate."""
    if get_store() is None:
        return atoms, []
    scope = DecisionScope(deal_id=str(project_id or ""))
    kept: list[Any] = []
    dropped: list[Any] = []
    for atom in atoms:
        if _atom_type_str(atom) != "task":
            kept.append(atom)
            continue
        text = _atom_text(atom)
        if not text:
            kept.append(atom)
            continue
        try:
            d = decide(
                ADMISSION_RELATION, text[:600], list(ADMISSION_CANDIDATES),
                instruction=_INSTRUCTION, llm=False, scope=scope,
            )
        except Exception:  # pragma: no cover - a gate never breaks a compile
            kept.append(atom)
            continue
        if d is not None and d.source == "store" and d.verdict == "drop":
            dropped.append(atom)
        else:
            kept.append(atom)
    return kept, dropped


__all__ = ["drop_taught_out_tasks", "ADMISSION_RELATION", "ADMISSION_CANDIDATES"]
