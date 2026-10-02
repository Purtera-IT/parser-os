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

import threading as _threading
from typing import Any

SUPPRESSION_FLAG_PREFIX = "suppressed:"

#: ``value`` key on an atom a stage drops on purpose rather than folds (a
#: hallucinated site, a PMO line that is not a quote line): no survivor.
DROPPED_NOT_FOLDED_KEY = "_dropped_not_folded"

#: ``value`` key on a suppressed atom: the kept atom it was folded into.
SURVIVOR_KEY = "_survivor"

#: Stages that remove an atom ON PURPOSE -- chrome, a heading, boilerplate, a
#: gate's verdict. The atom's content lives on nowhere, by design, so its
#: entry is marked ``kind: "drop"`` with the stage's reason. Every other
#: stage is a FOLD: the atom's content lives on in another atom, which its
#: entry must name under :data:`SURVIVOR_KEY`.
DROP_STAGES = frozenset({
    "section_heading",            # parser: a heading is its section's path
    "sheet_router",               # parser: a sheet routed DROP
    "pricing_rollup_rows_emitted",  # parser: count banner; every row is its own atom
    "chrome",                     # signature images, e-sign stamps
    "execution_boilerplate_drop",
    "document_job_scope",
    "atom_type_sanity",           # repeated page footers, label-only rows
    "site_geo_fallback",          # vendor letterhead address
    "entity_resolution",          # vendor letterhead address (post backfill)
    "task_admission",
    "noise_suppression",
    "drawing_pairs",
    "substance_gate",
})


def mark_dropped_not_folded(atom: Any, why: str) -> None:
    """Mark ``atom`` as removed on purpose, with ``why``: it names no survivor."""
    v = getattr(atom, "value", None)
    if isinstance(v, dict):
        v[DROPPED_NOT_FOLDED_KEY] = why


def suppression_kind(atom: Any, stage: str = "") -> str:
    """``"drop"`` or ``"fold"`` for a suppressed atom (see :data:`DROP_STAGES`)."""
    val = getattr(atom, "value", None)
    val = val if isinstance(val, dict) else {}
    if val.get(DROPPED_NOT_FOLDED_KEY):
        return "drop"
    if not stage:
        stage = str((val.get("_suppression") or {}).get("stage") or "")
    if not stage:
        stage = next((str(f)[len(SUPPRESSION_FLAG_PREFIX):] for f in (getattr(atom, "review_flags", None) or [])
                      if str(f).startswith(SUPPRESSION_FLAG_PREFIX)), "")
    return "drop" if stage in DROP_STAGES else "fold"


def _stamp_kind(atom: Any, stage: str = "") -> None:
    val = getattr(atom, "value", None)
    if not isinstance(val, dict):
        return
    sup = val.get("_suppression")
    if not isinstance(sup, dict):
        return
    kind = suppression_kind(atom, stage or str(sup.get("stage") or ""))
    sup["kind"] = kind
    if kind == "drop" and val.get(DROPPED_NOT_FOLDED_KEY) and not sup.get("drop_reason"):
        sup["drop_reason"] = str(val[DROPPED_NOT_FOLDED_KEY])


# Which atom each fold went into: ``id(loser) -> (loser, winner)``, per thread.
# A stage's suppressed entry must name a survivor that is still standing, and
# only the fold itself knows which one it was -- by the time the compiler sees
# the stage's output, a site merged into another site's record shares none of
# its words (010353: three site atoms suppressed with survivor null). Read and
# cleared by :func:`take_folds`.
_FOLDS = _threading.local()


def note_folded_into(loser: Any, winner: Any) -> None:
    """Record that ``loser`` was folded into ``winner`` (see :func:`take_folds`)."""
    if loser is None or winner is None or loser is winner:
        return
    reg = getattr(_FOLDS, "map", None)
    if reg is None:
        reg = _FOLDS.map = {}
    reg[id(loser)] = (loser, winner)


def take_folds() -> dict[int, tuple[Any, Any]]:
    """The folds recorded since the last call, then forget them."""
    reg = getattr(_FOLDS, "map", None) or {}
    _FOLDS.map = {}
    return reg


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
            _stamp_kind(atom, stage)
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
        _stamp_kind(atom)
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


def _is_site(atom: Any) -> bool:
    t = getattr(atom, "atom_type", None)
    return str(getattr(t, "value", t) or "") == "physical_site"


def _named_survivor(atom: Any) -> str:
    val = getattr(atom, "value", None)
    val = val if isinstance(val, dict) else {}
    sv = val.get(SURVIVOR_KEY) or val.get("duplicate_of") or {}
    return str(sv.get("atom_id") or "") if isinstance(sv, dict) else ""


def settle_ledger(atoms: list[Any], suppressed: list[Any]) -> tuple[list[Any], list[Any], dict[str, int]]:
    """End of compile: every suppressed fold names a survivor that stands.

    A survivor can itself be folded or dropped by a later stage. For each
    suppressed atom that is a FOLD (see :data:`DROP_STAGES`):

    * its survivor stands -> nothing to do;
    * its survivor was folded on -> the chain is followed and the entry
      re-pointed at the atom that stands (``via`` keeps the first name);
    * the chain ends at an atom a stage DROPPED -> the fold goes with it,
      recorded as a drop whose reason names that atom and stage (a site goes
      with it only when the dropped atom is a site too);
    * it names no survivor -> a standing atom with the same words is named;
    * otherwise it comes back to the atom list: no atom vanishes without a
      recorded survivor.

    Returns ``(atoms, suppressed, counts)``.
    """
    from app.core.cross_doc_copies import _text_key

    final_ids = {str(getattr(a, "id", "") or ""): a for a in atoms}
    supp_by_id = {str(getattr(a, "id", "") or ""): a for a in suppressed}
    by_text: dict[str, list[Any]] = {}
    for a in atoms:
        by_text.setdefault(_text_key(a), []).append(a)
    counts = {"repointed": 0, "named_by_text": 0, "dropped_with_survivor": 0, "restored": 0}
    restored: list[Any] = []
    for s in suppressed:
        if suppression_kind(s) == "drop":
            continue
        val = getattr(s, "value", None)
        if not isinstance(val, dict):
            restored.append(s)
            continue
        sup = val.get("_suppression") if isinstance(val.get("_suppression"), dict) else {}
        stage = str(sup.get("stage") or "")
        start = _named_survivor(s)
        cur, seen, end_drop, standing = start, set(), None, None
        while cur and cur not in seen and len(seen) < 64:
            seen.add(cur)
            if cur in final_ids:
                standing = final_ids[cur]
                break
            nxt = supp_by_id.get(cur)
            if nxt is None:
                break
            if suppression_kind(nxt) == "drop":
                end_drop = nxt
                break
            cur = _named_survivor(nxt)
        if standing is None and not start:
            key = _text_key(s)
            same = by_text.get(key, ()) if key else ()
            own = str(getattr(s, "artifact_id", "") or "")
            standing = next((k for k in same if str(getattr(k, "artifact_id", "") or "") == own),
                            next(iter(same), None))
            if standing is not None:
                counts["named_by_text"] += 1
        if standing is not None:
            sid = str(getattr(standing, "id", "") or "")
            if sid != start:
                rec = {"atom_id": sid, "artifact_id": str(getattr(standing, "artifact_id", "") or ""),
                       "stage": stage}
                if start:
                    rec["via"] = start
                    counts["repointed"] += 1
                val[SURVIVOR_KEY] = rec
            continue
        if end_drop is not None and (not _is_site(s) or _is_site(end_drop)):
            dv = getattr(end_drop, "value", None)
            dsup = (dv.get("_suppression") if isinstance(dv, dict) else None) or {}
            sup = dict(sup)
            sup["kind"] = "drop"
            sup["drop_reason"] = (
                f"folded into {getattr(end_drop, 'id', '')}, which {dsup.get('stage') or 'a stage'} dropped: "
                f"{dsup.get('reason') or (dv or {}).get(DROPPED_NOT_FOLDED_KEY) or ''}"
            ).strip()
            val["_suppression"] = sup
            counts["dropped_with_survivor"] += 1
            continue
        restored.append(s)
    if not restored:
        return atoms, suppressed, counts
    back = {id(a) for a in restored}
    for a in restored:
        flags = [f for f in (getattr(a, "review_flags", None) or []) if not str(f).startswith(SUPPRESSION_FLAG_PREFIX)]
        try:
            a.review_flags = flags
        except Exception:  # pragma: no cover
            pass
        val = getattr(a, "value", None)
        if isinstance(val, dict):
            sup = val.pop("_suppression", None) or {}
            val.pop(SURVIVOR_KEY, None)
            val["_restored"] = {
                "stage": str(sup.get("stage") or ""),
                "reason": "folded into no atom that stands at the end of the compile",
            }
    counts["restored"] = len(restored)
    return list(atoms) + restored, [a for a in suppressed if id(a) not in back], counts


__all__ = [
    "settle_ledger",
    "capture_suppressed", "merge_suppressed", "keep_unsurvived_lines", "SUPPRESSION_FLAG_PREFIX",
]
