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
#: deal and the survivor is chosen right after it.
PRE_HEAD_FOLDS = frozenset({"quoted_history_dedup"})

#: Fold stages settled at the end of the compile, after every stage has run.
FINAL_FOLDS = PRE_HEAD_FOLDS

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


def fold_by_heads(atoms: list[Any], ledger: list[Any], *, stages: frozenset[str] = PRE_HEAD_FOLDS,
                  reading=None) -> tuple[list[Any], list[tuple[Any, Any]]]:
    """Settle each fold from ``stages`` by the keep head.

    For every survivor in ``atoms`` with folded copies in ``ledger``: when the
    keep head read the survivor and does not keep it, and keeps one of its
    copies, the surest kept copy takes the survivor's place (same position)
    and the survivor goes to the ledger naming it; the other copies re-point at
    it. ``ledger`` is changed in place. ``reading(atom_id, head)`` defaults to
    this compile's base read. Returns ``(atoms, [(kept_copy, old_survivor)])``.
    """
    reading = reading or _default_reading
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
        if said is None or _keep_p(said) is not None:
            continue  # unread survivor: nothing to arbitrate; kept survivor: the fold stands
        kept = [(p, c) for c in losers if (p := _keep_p(reading(_id(c), CO_ACTION))) is not None]
        if not kept:
            continue
        _, best = max(kept, key=lambda x: x[0])
        stage = _stage(best)
        _unsuppress(best)
        _value(best)[HEAD_FOLD_KEY] = {"over": sid, "stage": stage, "head": CO_ACTION,
                                       "survivor_said": [said[0], round(float(said[1] or 0.0), 3)]}
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


__all__ = ["FINAL_FOLDS", "HEAD_FOLD_KEY", "KEEP_BAR", "PRE_HEAD_FOLDS", "fold_by_heads", "fold_losers"]
