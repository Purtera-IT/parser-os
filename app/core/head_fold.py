"""Which copy of a repeated line survives is the heads' call, not a dedup's.

A dedup stage folds copies of one line onto a survivor it picks by position
(the authored original, the earliest file). That is extraction: it finds that
the copies are the same line (``rel:same_as``). Which copy the deal keeps is a
judgment, and the keep head (``read:co_action``) owns it. The audit of
2026-10-22 (sitefix/audit_out/overrides.md, override points 1 and 8) found 826
head-kept quoted copies the heads were never shown, 30 of them lost outright
because the survivor the stage picked was one the head rejects ("He also
confirmed that he will need us to run the cabling too", 010043: quoted copy
keep 0.89, authored survivor reject).

So a fold is provisional until the heads have read every copy:

* the folded copies stay in the suppression ledger, naming their survivor
  (``_survivor``), as before;
* :func:`fold_losers` lists them, and the compiler hands them to the deal's
  read (``c3_type_base.also_read``), so each copy is read in its own context;
* :func:`fold_by_heads` then settles each fold: when the survivor is not one
  the keep head keeps and a folded copy is, that copy takes the survivor's
  place in the atom list and the survivor goes to the ledger naming it.

With no keep reading (the brief's heads off, the read failed) nothing moves.
"""
from __future__ import annotations

from typing import Any, Iterable

from app.core.suppression_ledger import SUPPRESSION_FLAG_PREFIX, SURVIVOR_KEY

#: Keep head reading at or above this is a keep (the brief's MAYBE_BAR).
KEEP_BAR = 0.5
CO_ACTION = "read:co_action"

#: Fold stages that run before the deal's read: their copies are read with the
#: deal and the survivor is chosen right after it (quoted_chatter_dedup's
#: survivors are admission-held lines, settled over that list).
# One list with the training export (c3_record.read_with_deal), so the base
# trains on exactly the lines this read holds.
from c3_record import ALSO_READ_STAGES as PRE_HEAD_FOLDS  # noqa: E402

#: Fold stages after the read whose losers the read already holds: settled
#: right after them, before entities, edges and packets are built. Stakeholder
#: and site merges are left alone: they merge the copies' fields into one
#: record (0 head-keep lost on the audited deals).
POST_HEAD_FOLDS = frozenset({"semantic_dedup"})

#: Every fold the heads settle, for the late re-read.
FINAL_FOLDS = PRE_HEAD_FOLDS | POST_HEAD_FOLDS

#: ``value`` key on a copy the heads kept over the stage's survivor.
HEAD_FOLD_KEY = "head_fold"


def _value(atom: Any) -> dict[str, Any]:
    v = getattr(atom, "value", None)
    return v if isinstance(v, dict) else {}


def _id(atom: Any) -> str:
    return str(getattr(atom, "id", "") or "")


def _stage(atom: Any) -> str:
    s = _value(atom).get("_suppression")
    return str(s.get("stage") or "") if isinstance(s, dict) else ""


def _survivor_id(atom: Any) -> str:
    s = _value(atom).get(SURVIVOR_KEY)
    return str(s.get("atom_id") or "") if isinstance(s, dict) else ""


def fold_losers(ledger: Iterable[Any], stages: frozenset[str] = PRE_HEAD_FOLDS) -> list[Any]:
    """The ledger's folded copies from ``stages`` that name a survivor."""
    return [a for a in ledger or () if _stage(a) in stages and _survivor_id(a)]


def _default_reading(atom_id: str, head: str) -> tuple[str, float] | None:
    from app.core.c3_type_base import line_reading

    return line_reading(atom_id, head)


def _default_type_drops(atom_id: str) -> bool:
    from app.core.c3_type_base import type_head_drops

    return type_head_drops(atom_id)


def _keep_p(reading: tuple[str, float] | None) -> float | None:
    if reading and reading[0] == "keep" and float(reading[1] or 0.0) >= KEEP_BAR:
        return float(reading[1])
    return None


def _unsuppress(atom: Any) -> None:
    flags = [f for f in (getattr(atom, "review_flags", None) or []) if not str(f).startswith(SUPPRESSION_FLAG_PREFIX)]
    try:
        atom.review_flags = flags
    except Exception:  # pragma: no cover - defensive
        pass
    v = _value(atom)
    v.pop("_suppression", None)
    v.pop(SURVIVOR_KEY, None)


def _suppress(atom: Any, winner: Any, stage: str) -> None:
    flag = f"{SUPPRESSION_FLAG_PREFIX}{stage}"
    flags = list(getattr(atom, "review_flags", None) or [])
    if flag not in flags:
        try:
            atom.review_flags = sorted(set(flags + [flag]))
        except Exception:  # pragma: no cover - defensive
            pass
    v = getattr(atom, "value", None)
    if isinstance(v, dict):
        v["_suppression"] = {"stage": stage, "kind": "fold",
                             "reason": "the keep head (read:co_action) kept another copy of this line"}
        v[SURVIVOR_KEY] = {"atom_id": _id(winner), "artifact_id": str(getattr(winner, "artifact_id", "") or ""),
                           "stage": stage}


def _carry(winner: Any, loser: Any) -> None:
    """The survivor's evidence goes with the line: its source refs, receipts
    and entity keys (what a dedup had merged into it) join the kept copy's."""
    for attr in ("source_refs", "receipts", "entity_keys"):
        mine = getattr(winner, attr, None)
        theirs = getattr(loser, attr, None) or []
        if not isinstance(mine, list):
            continue
        for x in theirs:
            if x not in mine:
                try:
                    mine.append(x)
                except Exception:  # pragma: no cover - defensive
                    break


def fold_by_heads(atoms: list[Any], ledger: list[Any], *, stages: frozenset[str] = PRE_HEAD_FOLDS,
                  reading=None, type_drops=None) -> tuple[list[Any], list[tuple[Any, Any]]]:
    """Settle each fold from ``stages`` by the keep head.

    For every survivor in ``atoms`` with folded copies in ``ledger``: when the
    keep head read the survivor and does not keep it, and keeps one of its
    copies, the surest kept copy takes the survivor's place (same position)
    and the survivor goes to the ledger naming it; the other copies re-point at
    it. ``ledger`` is changed in place. ``reading(atom_id, head)`` defaults to
    this compile's base read. Returns ``(atoms, [(kept_copy, old_survivor)])``.
    """
    reading = reading or _default_reading
    type_drops = type_drops or _default_type_drops

    def kept_p(aid: str) -> float | None:
        # A copy the type head reads as chatter (small_talk / _keep at its
        # SURE bar) is the type head's drop: co_action never learned those
        # lines, so its keep there decides nothing.
        return None if type_drops(aid) else _keep_p(reading(aid, CO_ACTION))

    pos = {_id(a): i for i, a in enumerate(atoms)}
    groups: dict[str, list[Any]] = {}
    for loser in fold_losers(ledger, stages):
        sid = _survivor_id(loser)
        if sid in pos:
            groups.setdefault(sid, []).append(loser)
    if not groups:
        return atoms, []
    out = list(atoms)
    swaps: list[tuple[Any, Any]] = []
    for sid, losers in groups.items():
        survivor = out[pos[sid]]
        said = reading(sid, CO_ACTION)
        if said is None or kept_p(sid) is not None:
            continue  # unread survivor: nothing to arbitrate; kept survivor: the fold stands
        kept = [(p, c) for c in losers if (p := kept_p(_id(c))) is not None]
        if not kept:
            continue
        _, best = max(kept, key=lambda x: x[0])
        stage = _stage(best)
        _unsuppress(best)
        _value(best)[HEAD_FOLD_KEY] = {"over": sid, "stage": stage, "head": CO_ACTION,
                                       "survivor_said": [said[0], round(float(said[1] or 0.0), 3)]}
        _carry(best, survivor)
        out[pos[sid]] = best
        ledger[:] = [a for a in ledger if a is not best]
        _suppress(survivor, best, stage)
        ledger.append(survivor)
        for other in losers:
            if other is not best and isinstance(getattr(other, "value", None), dict):
                other.value[SURVIVOR_KEY] = {"atom_id": _id(best),
                                             "artifact_id": str(getattr(best, "artifact_id", "") or ""),
                                             "stage": _stage(other)}
        swaps.append((best, survivor))
    return out, swaps


__all__ = ["FINAL_FOLDS", "HEAD_FOLD_KEY", "KEEP_BAR", "POST_HEAD_FOLDS", "PRE_HEAD_FOLDS", "fold_by_heads",
           "fold_losers"]
