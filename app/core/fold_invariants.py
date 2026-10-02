"""The two rules that sit above every collapse in the compile.

A dozen stages fold atoms together: `collapse_duplicate_atoms`,
`cross_type_dedup_atoms`, `semantic_dedup_atoms`, `_suppress_table_row_blob_doubles`,
`merge_signature_rows`, `dedupe_stakeholder_atoms`, the rollups. Each one has its
own notion of what a duplicate is, and each one is right about its own case.

What none of them had was a shared floor. So the same defect was found and
fixed eight separate times, in eight separate shapes:

  phase 1-2  81 per-plate cable quantities folded onto one plate's copy
             39 folds where the numbers differed, one of them a bid deadline
             a location's quantities deleted by a twin from another location
  phase 3    a totals block reduced to its revenue line, losing cost, margin
             and margin percentage
             four HubSpot notes and seven message headers each collapsed to one
             a `quantity` retyped into prose that did not state the number
             a contract row on a signature page deleted as a signature
             the same row deleted again, one stage later, as a "duplicate" of
             the signature record that replaced it

Every one is the same sentence: **a fold may not delete what only the loser
held.** That rule was already written down -- it is the promotion gate on the
suppression head in `_BETS_LEDGER.md`, which says a learned head "may never
fold atoms that state different numbers, or that belong to different places."
It was a gate on a head that does not exist yet, while the code re-derived it
by hand at each fold site and missed one every time.

This module is that rule, once.

Two invariants
--------------
**Different numbers are different facts.** Formatting is not a number:
"1,200" and "1200" are one quantity, and so are "2.50" and "2.5", so folding
them must still be allowed. A figure the loser states and the survivor does
not is a deletion.

**Different identities are different facts.** Two atoms with the same words
about different plates, sites, rooms, notes or messages are two facts. Identity
is read from the atom's structured value and its entity keys, never guessed
from its text.

What this module does NOT decide
--------------------------------
Whether a given stage should refuse a fold, merge the loser in, or keep both.
That is policy and it differs per stage -- `cross_type_dedup_atoms` exempts a
money column whose value already lives in the winner, because there the loser's
figure is not lost. This module answers only the factual question each of those
policies needs: *what would this fold delete?*
"""
from __future__ import annotations

import json as _json
import re
from typing import Any

#: A figure, with its thousands separators and decimal part attached.
FIGURE_RE = re.compile(r"\d+(?:[.,]\d+)*")

#: Value keys that say WHICH thing an atom is about. Two atoms with the same
#: words and different values here are about different things.
IDENTITY_FIELDS: tuple[str, ...] = (
    "plate_id",
    "site",
    "site_key",
    "room",
    "location",
    # Added in phase 3: a HubSpot note and an email message are identities in
    # exactly the same sense, and keying them on a constant collapsed every
    # note on a deal into one.
    "hubspot_note_id",
    "message_id",
)


def _canonical(raw: str) -> str:
    """One figure, with formatting removed and the quantity preserved."""
    cleaned = raw.replace(",", "")
    try:
        value = float(cleaned)
    except ValueError:
        return cleaned
    return str(int(value)) if value.is_integer() else str(value)


def figures_in_order(text: str) -> tuple[str, ...]:
    """Every figure in the text, in order. Order matters to a bucket key."""
    return tuple(_canonical(m) for m in FIGURE_RE.findall(text or ""))


def figures(text: str) -> frozenset[str]:
    """Every figure in the text, as a set. Order does not matter to a guard."""
    return frozenset(figures_in_order(text))


def figures_stated(atom: Any) -> frozenset[str]:
    """Every figure the atom states, in its words OR its structured value.

    Both halves matter, and the second half is the one that gets forgotten. A
    raw table row's trailing "| $21,560.00" is a money column whose amount
    already lives in the typed sibling's `value`; the typed atom is not poorer
    for lacking it in text, and folding the row away loses nothing. The same
    fold becomes a deletion the moment the survivor's value does not carry the
    figure either.
    """
    blob = str(
        getattr(atom, "raw_text", None) or getattr(atom, "text", None) or ""
    )
    value = getattr(atom, "value", None)
    if isinstance(value, dict):
        try:
            blob = blob + " " + _json.dumps(value, default=str)
        except Exception:
            blob = blob + " " + str(value)
    return figures(blob)


_EMAIL_RE = __import__("re").compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def emails_stated(atom: Any) -> frozenset[str]:
    """Every email address the atom states, in its words OR its value.

    An address is how a contact is reached, and it carries no digit, so
    :func:`figures_stated` never protected it: a contact row "Jane Roe | PM |
    jane@acme.com" and its bare twin "Jane Roe | PM" key alike once the key
    is cut, and the row with the address was the one folded away.
    """
    blob = str(getattr(atom, "raw_text", None) or getattr(atom, "text", None) or "")
    value = getattr(atom, "value", None)
    if isinstance(value, dict):
        try:
            blob = blob + " " + _json.dumps(value, default=str)
        except Exception:
            blob = blob + " " + str(value)
    return frozenset(m.group(0).lower().rstrip(".") for m in _EMAIL_RE.finditer(blob))


def emails_only_the_loser_states(winner: Any, loser: Any) -> frozenset[str]:
    """Addresses that leave the compile if ``loser`` is folded into ``winner``."""
    return emails_stated(loser) - emails_stated(winner)


def identity(atom: Any) -> tuple[str, ...]:
    """What distinguishes this atom from another with the SAME words."""
    ident: list[str] = []
    value = getattr(atom, "value", None)
    if isinstance(value, dict):
        for field in IDENTITY_FIELDS:
            got = value.get(field)
            if got not in (None, "", [], {}):
                ident.append(f"{field}={got}")
    keys = getattr(atom, "entity_keys", None)
    if keys:
        # An atom tied to a different entity is about a different thing, and
        # entity keys are already canonical, so they compare cleanly.
        ident.extend(sorted(str(k) for k in keys))
    return tuple(ident)


# ---- the two questions a fold site actually asks ---------------------------


def figures_only_the_loser_states(winner: Any, loser: Any) -> frozenset[str]:
    """Figures that leave the compile if ``loser`` is folded into ``winner``."""
    return figures_stated(loser) - figures_stated(winner)


def identities_differ(winner: Any, loser: Any) -> bool:
    """True when the two atoms are about demonstrably different things.

    Only a stated identity counts. An atom that records no plate, site, room,
    note or message is not thereby "somewhere else" -- it is simply unplaced,
    and refusing every fold involving one would stop dedup entirely.
    """
    a, b = identity(winner), identity(loser)
    return bool(a) and bool(b) and a != b


def refuse_fold(winner: Any, loser: Any) -> str | None:
    """A reason this fold would delete a fact, or None when it is safe.

    The default floor, for a stage with no exemption of its own. A stage that
    knows better about its own case -- a money column already carried in the
    winner's value, a per-document index that is not a fact -- asks the two
    questions above directly instead.
    """
    lost = figures_only_the_loser_states(winner, loser)
    if lost:
        return "states figures the survivor does not: " + ", ".join(sorted(lost)[:6])
    gone = emails_only_the_loser_states(winner, loser)
    if gone:
        return "states email addresses the survivor does not: " + ", ".join(sorted(gone)[:6])
    if identities_differ(winner, loser):
        return f"different identity: {identity(loser)} vs {identity(winner)}"
    return None
