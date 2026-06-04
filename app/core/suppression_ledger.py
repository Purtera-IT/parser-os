"""Retained-suppression ledger — atoms a stage removes are kept, not lost.

Six compile stages can DROP atoms (duplicate collapse, execution-boilerplate
drop, semantic/cross-type dedup, vendor-site suppression, …). Today a dropped
atom simply vanishes from the atom list. That makes two things impossible:

* **Omission complaints.** When a PM says "you missed the loading dock", the
  dock atom must still exist somewhere — flagged, not in the accepted set — so
  the complaint can be localized back to the stage that removed it. A silent
  drop is unlocalizable.
* **Auditability.** "Why isn't X in the brief?" should be answerable by pointing
  at the stage + reason that suppressed it, not by re-deriving the pipeline.

This module is the retention primitive. It never decides *what* to drop — the
stages still own that judgment. It only records the drop: given the atom list
*before* and *after* a stage, it diffs by id, stamps each removed atom with a
``suppressed:<stage>`` review flag and a ``_suppression`` provenance marker
(stage + reason), and returns those atoms so the compiler can carry them in a
sidecar (``CompileResult.suppressed_atoms``) that downstream consumers ignore.

Pure function, no I/O, no LLM. The accepted ``atoms`` set the compiler keeps is
unchanged — this only captures what would otherwise have been thrown away.
"""

from __future__ import annotations

from typing import Any

SUPPRESSION_FLAG_PREFIX = "suppressed:"


def _atom_id(atom: Any) -> str:
    return str(getattr(atom, "id", "") or "")


def capture_suppressed(
    before: list[Any],
    after: list[Any],
    *,
    stage: str,
    reason: str,
) -> list[Any]:
    """Return atoms present in ``before`` but absent from ``after``, stamped.

    A stage that collapses/drops atoms hands its input (``before``) and output
    (``after``); any atom whose id disappeared was suppressed. Each suppressed
    atom is mutated in place (it is no longer in the accepted set, so mutation
    is safe) to record *why* it was removed:

    * ``review_flags`` gains ``"suppressed:<stage>"`` (idempotent), and
    * ``value["_suppression"]`` records ``{"stage", "reason"}`` when ``value``
      is a dict, preserving any existing keys.

    Args:
        before: the atom list as it entered the stage.
        after: the atom list the stage produced.
        stage: the stage name (e.g. ``"semantic_dedup"``), for the flag/marker.
        reason: a short human-readable cause, surfaced to the PM on audit.

    Returns:
        The list of suppressed atoms (a subset of ``before``), order-preserved.
        Empty when the stage dropped nothing.
    """
    after_ids = {_atom_id(a) for a in after}
    flag = f"{SUPPRESSION_FLAG_PREFIX}{stage}"
    suppressed: list[Any] = []
    for atom in before:
        aid = _atom_id(atom)
        if aid and aid in after_ids:
            continue
        # Record the suppression flag (idempotent).
        flags = list(getattr(atom, "review_flags", None) or [])
        if flag not in flags:
            flags = sorted(set(flags + [flag]))
            try:
                atom.review_flags = flags
            except Exception:  # pragma: no cover - defensive (frozen/odd atom)
                pass
        # Record the structured provenance marker when value is a dict.
        val = getattr(atom, "value", None)
        if isinstance(val, dict):
            val["_suppression"] = {"stage": stage, "reason": reason}
        suppressed.append(atom)
    return suppressed


def merge_suppressed(
    ledger: list[Any],
    newly_suppressed: list[Any],
) -> None:
    """Append ``newly_suppressed`` to ``ledger`` in place, de-duped by id.

    A given atom can only be suppressed once (the first stage that removes it
    wins); a later stage never sees it again because it is no longer in the
    accepted set. This de-dupe is belt-and-suspenders against an atom id that
    is somehow recorded twice.
    """
    seen = {_atom_id(a) for a in ledger}
    for atom in newly_suppressed:
        aid = _atom_id(atom)
        if aid and aid in seen:
            continue
        seen.add(aid)
        ledger.append(atom)


__all__ = ["capture_suppressed", "merge_suppressed", "SUPPRESSION_FLAG_PREFIX"]
