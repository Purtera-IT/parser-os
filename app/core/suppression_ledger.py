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
    note_suppressed(stage, suppressed)
    return suppressed


#: One line per suppressed atom, in order, for the compile trace: which stage
#: removed which text. Reset by the compiler at the start of a compile.
DROP_NOTES: list[str] = []


def note_suppressed(stage: str, atoms: list[Any], *, cap: int = 400) -> None:
    for a in atoms or []:
        if len(DROP_NOTES) >= cap:
            return
        t = getattr(a, "atom_type", None)
        t = str(getattr(t, "value", t) or "")
        text = " ".join(str(getattr(a, "raw_text", None) or getattr(a, "normalized_text", None) or "").split())
        DROP_NOTES.append(f"INFO: dropped[{stage}] {t}: {text[:140]}")


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


def _line_key(text: Any) -> str:
    import re

    return " ".join(re.sub(r"[^0-9a-z]+", " ", str(text or "").lower()).split())


#: Below this a line is punctuation or a page number, not content.
_MIN_GUARD_KEY = 3


def keep_unsurvived_lines(
    before: list[Any],
    after: list[Any],
    *,
    stage: str,
    eligible: Any = None,
) -> tuple[list[Any], list[Any]]:
    """Put back every removed atom whose text no kept atom contains.

    A collapse/rollup gate removes a line on the claim that another atom
    already says it. On two-column PDF pages that claim was false: a spec
    panel and numbered steps 6-15 were collapsed as "duplicates" of a
    neighbouring line (or rolled into a "N table rows" count) and survived
    only in the suppression ledger, where nobody can see or label them.

    So the claim is checked: a removed atom stays removed only when some atom
    in ``after`` carries its text (case and punctuation folded, substring
    match on ``raw_text``). Otherwise it is returned to the list, next to the
    atom that preceded it in ``before``, and marked ``_survivor_guard``.
    ``eligible(atom)`` limits which removed atoms the guard may restore.

    Returns ``(after_with_restored, restored)``.
    """
    after_ids = {id(a) for a in after}
    removed = [a for a in before if id(a) not in after_ids]
    if not removed:
        return list(after), []
    corpus = "\x00".join(_line_key(getattr(a, "raw_text", "") or getattr(a, "normalized_text", "")) for a in after)
    restored: list[Any] = []
    for atom in removed:
        if eligible is not None and not eligible(atom):
            continue
        key = _line_key(getattr(atom, "raw_text", "") or getattr(atom, "normalized_text", ""))
        if len(key) < _MIN_GUARD_KEY or key in corpus:
            continue
        val = getattr(atom, "value", None)
        if isinstance(val, dict):
            val["_survivor_guard"] = {
                "stage": stage,
                "reason": "no kept atom contains this line, so it was not removed",
            }
        restored.append(atom)
    if not restored:
        return list(after), []

    # Re-insert each restored atom after the nearest preceding atom (in the
    # original order) that is still in the list.
    restored_ids = {id(a) for a in restored}
    follow: dict[int, list[Any]] = {}
    lead: list[Any] = []
    anchor = None
    for atom in before:
        if id(atom) in after_ids:
            anchor = atom
        elif id(atom) in restored_ids:
            if anchor is None:
                lead.append(atom)
            else:
                follow.setdefault(id(anchor), []).append(atom)
    out: list[Any] = list(lead)
    for atom in after:
        out.append(atom)
        out.extend(follow.get(id(atom), ()))
    return out, restored


__all__ = [
    "capture_suppressed", "merge_suppressed", "keep_unsurvived_lines", "SUPPRESSION_FLAG_PREFIX",
]
