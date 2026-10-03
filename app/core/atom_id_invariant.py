"""End of compile: no two atoms share an id.

An atom id is the key every later reader joins on -- the label walk, the
envelope's ``atom_ids``, the suppression ledger's survivors, the cross-document
copies. Every stage that removes or restores atoms also keys by id (or by the
Python object), so an atom that reaches the list twice looks like ONE atom to
all of them and nothing downstream notices. 010353's intake "Job site" line went
out twice that way: semantic_dedup kept it as a demoted site and then appended
the same object again as the deal's "site-blind" pick.

So the rule is checked once, at the end, as a bug detector:

* strict (the test suite sets ``SOWSMITH_STRICT_INVARIANTS=1``): raise
  :class:`DuplicateAtomIdError`, so the stage that did it fails a test;
* production: never fail a customer's compile over it. A repeat of the very
  same object is dropped (it holds nothing the first does not), a different
  atom that collides on the id is kept (dropping it could delete words), and
  either way a ``WARNING`` names the id so the stage can be found.
"""
from __future__ import annotations

import os
from collections import Counter
from typing import Any

STRICT_ENV = "SOWSMITH_STRICT_INVARIANTS"


class DuplicateAtomIdError(AssertionError):
    """Two atoms in one compile result carry the same id."""


def _strict() -> bool:
    return str(os.environ.get(STRICT_ENV) or "").strip().lower() in {"1", "true", "yes"}


def duplicate_atom_ids(atoms: list[Any]) -> dict[str, int]:
    """``{atom_id: count}`` for every id held by more than one entry."""
    counts = Counter(str(getattr(a, "id", "") or "") for a in atoms)
    return {aid: n for aid, n in counts.items() if aid and n > 1}


def enforce_unique_atom_ids(atoms: list[Any], *, strict: bool | None = None) -> tuple[list[Any], list[str]]:
    """Check that no two atoms share an id. Returns ``(atoms, warnings)``.

    Strict mode raises on any duplicate. Otherwise a repeat of the same object
    is dropped and a distinct atom with a colliding id is kept; both are
    reported.
    """
    dups = duplicate_atom_ids(atoms)
    if not dups:
        return atoms, []
    if strict if strict is not None else _strict():
        sample = ", ".join(f"{aid} x{n}" for aid, n in sorted(dups.items())[:10])
        raise DuplicateAtomIdError(f"compile emitted {len(dups)} duplicate atom id(s): {sample}")
    out: list[Any] = []
    seen_objects: set[int] = set()
    repeats: Counter[str] = Counter()
    collisions: Counter[str] = Counter()
    first_by_id: dict[str, Any] = {}
    for atom in atoms:
        aid = str(getattr(atom, "id", "") or "")
        if aid not in dups:
            out.append(atom)
            continue
        if id(atom) in seen_objects:
            repeats[aid] += 1
            continue
        seen_objects.add(id(atom))
        if aid in first_by_id:
            collisions[aid] += 1
        else:
            first_by_id[aid] = atom
        out.append(atom)
    warnings = [
        f"WARNING: duplicate atom id {aid}: dropped {n} repeat(s) of the same atom"
        for aid, n in sorted(repeats.items())
    ] + [
        f"WARNING: duplicate atom id {aid}: {n + 1} different atoms share it (kept all)"
        for aid, n in sorted(collisions.items())
    ]
    return out, warnings


__all__ = ["DuplicateAtomIdError", "STRICT_ENV", "duplicate_atom_ids", "enforce_unique_atom_ids"]
