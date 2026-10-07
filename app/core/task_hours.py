"""Learned hours for a unit of work -- taught from finished Deal Kits, not listed.

A Deal Kit is the answer to "how long does this take": 000020 Binghamton's
Estimate_Detail gives 0.75 h to "Remove the 1518 SonicWall from service, label
it, and store as a backup"; 000061 MBrany's Install ROM gives "3 h per cable
drop". Each becomes a correction on relation ``task_hours`` whose exemplar is
the work line and whose verdict carries the hours:

    hours=0.75                          fixed hours for that piece of work
    hours=3;per=cable drop              hours per unit; scaled by the quantity
    hours=16;per=camera;qty=3           the kit's own numbers: 16 h for 3 cameras
    hours=4;role=L1 EUC                 optionally the role that did it

At compile time every task atom asks the store (nearest taught exemplar,
guess-free, no LLM). A confident match stamps ``estimated_hours`` and where the
number came from on the atom; abstain leaves it alone. Per-unit hours scale by
a number written next to the taught unit ("39 AP locations" for "per AP
location"), matched on the unit's own words -- the unit comes from the kit, so
there is no list of nouns here. A per-unit match with no quantity in the text
records the rate and no total, rather than guessing a count.

A per-unit lesson carries the kit's numbers, not a rounded rate. 010043's kit
priced 16 h for 3 cameras; taught as ``hours=5.33;per=camera`` the compile
stamped 15.99 h on the same three cameras. Taught as ``hours=16;per=camera;
qty=3`` the rate is 16/3 at full precision and the total is the kit's 16.

The atom labeler's `hours` card asks a narrower question: what hours does THIS
LINE STATE? Estimates are not base training -- the Deal Kit has its own models
for those -- so a card verdict carries where its number came from:

    unstated                            the line states no hours
    hours=8;basis=stated                the line itself says 8 hours
    hours=2;per=drop;qty=24;basis=stated
    hours=6;basis=estimate              our estimate -- company layer only
    hours=6;basis=deal_kit              read off the Deal Kit -- company layer only

``hours_judgment_problem`` is the card's grammar. Only ``basis=stated`` (and
``unstated``) train the base head; human_labels routes the rest, and every
older verdict with no basis, to Purtera's ``co_hours_estimate``.
``parse_hours_verdict`` stays the kit's grammar and reads a card verdict too
(the basis rides along as a key); ``unstated`` parses to None, so it never
stamps hours.
"""

from __future__ import annotations

import re
from typing import Any

RELATION = "task_hours"

#: The card verdict for a line that states no hours -- the head's "no value".
UNSTATED = "unstated"
#: Where a card verdict's number came from. Only ``stated`` trains the base:
#: an estimate, or hours read off the Deal Kit, train the company layer.
BASIS_STATED = "stated"
BASES = (BASIS_STATED, "estimate", "deal_kit")

_NUM_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w.])")
_WORD_RE = re.compile(r"[a-z0-9]+")


def encode_hours_verdict(hours: float, *, per: str = "", qty: float | None = None, role: str = "",
                         basis: str = "") -> str:
    parts = [f"hours={float(hours):g}"]
    if per.strip():
        parts.append(f"per={per.strip()}")
        if qty is not None and float(qty) > 0:
            parts.append(f"qty={float(qty):g}")
    if role.strip():
        parts.append(f"role={role.strip()}")
    if basis.strip():
        parts.append(f"basis={basis.strip()}")
    return ";".join(parts)


def parse_hours_verdict(verdict: str) -> dict[str, Any] | None:
    out: dict[str, Any] = {}
    for part in str(verdict or "").split(";"):
        key, sep, val = part.partition("=")
        if sep:
            out[key.strip()] = val.strip()
    try:
        out["hours"] = float(out["hours"])
    except (KeyError, ValueError):
        return None
    if out["hours"] < 0:
        return None
    # The quantity the taught hours covered; only meaningful with a unit.
    try:
        qty = float(out.get("qty", ""))
    except ValueError:
        qty = 0.0
    if qty > 0 and out.get("per"):
        out["qty"] = qty
    else:
        out.pop("qty", None)
    return out


def hours_judgment_basis(verdict: str) -> str | None:
    """Where a labeling-card `hours` verdict's number came from.

    ``unstated`` for a line that states no hours; ``stated`` / ``estimate`` /
    ``deal_kit`` from the verdict's basis; ``""`` for a number saved before the
    card asked (an older verdict); None when the verdict does not parse."""
    v = str(verdict or "").strip()
    if v.lower() == UNSTATED:
        return UNSTATED
    parsed = parse_hours_verdict(v)
    if parsed is None:
        return None
    return str(parsed.get("basis") or "").strip().lower()


def hours_judgment_problem(verdict: str) -> str | None:
    """Why a NEW labeling-card `hours` verdict is refused, or None when valid.

    Valid: ``unstated`` alone, or ``hours=N[;per=..][;qty=N][;role=..]`` with a
    basis in BASES. Only ``basis=stated`` trains the base head; the mirror
    (human_labels) routes the other two to the company layer."""
    basis = hours_judgment_basis(verdict)
    if basis is None:
        return ("hours takes `unstated`, or hours=N[;per=<unit>][;qty=N][;role=<role>];basis=<"
                + "|".join(BASES) + "> -- e.g. hours=3;per=cable drop;basis=stated")
    if basis == UNSTATED or basis in BASES:
        return None
    if not basis:
        return "hours needs a basis: basis=" + "|".join(BASES) + " (only hours the line states train the base)"
    return f"basis={basis} is not one of " + ", ".join(BASES)


def _stem(word: str) -> str:
    w = word.lower()
    for suffix in ("ies", "es", "s"):
        if len(w) > len(suffix) + 2 and w.endswith(suffix):
            return w[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return w


def quantity_for_unit(text: str, unit: str) -> float | None:
    """The number written in front of the taught unit, or None.

    "Install EMT conduit drops at 39 AP locations" with unit "AP location" -> 39.
    The number must be followed, within four words, by the unit's last word
    (its head noun), compared by stem so "drops" matches "drop"."""
    unit_words = [_stem(w) for w in _WORD_RE.findall(unit.lower())]
    if not unit_words:
        return None
    head = unit_words[-1]
    words = list(re.finditer(r"\S+", text))
    for i, w in enumerate(words):
        m = _NUM_RE.fullmatch(w.group(0).strip("(),:;"))
        if not m:
            continue
        following = [_stem(x) for tok in words[i + 1 : i + 5] for x in _WORD_RE.findall(tok.group(0).lower())]
        if head in following:
            try:
                return float(m.group(1))
            except ValueError:
                return None
    return None


def _plural(unit: str, n: float) -> str:
    """The unit as it reads after a count: "3 cameras", "1 cable drop"."""
    u = unit.strip()
    if n == 1 or not u or u.endswith("s"):
        return u
    return u + "s"


def _atom_type(atom: Any) -> str:
    at = getattr(atom, "atom_type", None)
    return str(getattr(at, "value", at) or "")


def estimate_task_hours(atoms: list[Any], *, store: Any = None) -> int:
    """Stamp learned hours on task atoms. Returns how many were stamped."""
    try:
        from app.core.decide import DecisionScope, decide, get_store
    except Exception:  # pragma: no cover
        return 0
    store = store if store is not None else get_store()
    if store is None:
        return 0
    try:
        verdicts = sorted({
            str(c.verdict) for c in store.all_corrections(active_only=True)
            if getattr(c, "relation", "") == RELATION and parse_hours_verdict(c.verdict)
        })
    except Exception:
        return 0
    if not verdicts:
        return 0
    stamped = 0
    for atom in atoms:
        if _atom_type(atom) != "task":
            continue
        # Hours belong to the quote line. A mention folded into a fuller
        # statement of the same work (task_tier_classifier.fold_task_mentions)
        # is not priced again.
        _val = getattr(atom, "value", None)
        if isinstance(_val, dict) and _val.get("folded_into"):
            continue
        text = str(getattr(atom, "raw_text", "") or "").strip()
        if not text:
            continue
        # The deal the atom belongs to, so a lesson taught for this deal is found.
        scope = DecisionScope(deal_id=str(getattr(atom, "project_id", "") or ""))
        d = decide(RELATION, text[:600], verdicts, instruction="Hours for this unit of work, taught from finished Deal Kits.", llm=False, scope=scope)
        if d is None or d.source != "store" or not d.verdict:
            continue
        parsed = parse_hours_verdict(d.verdict)
        if not parsed:
            continue
        value = getattr(atom, "value", None)
        if not isinstance(value, dict):
            value = {}
            try:
                atom.value = value
            except Exception:
                continue
        per = str(parsed.get("per") or "")
        rate = parsed["hours"]
        if per:
            taught_qty = parsed.get("qty")
            # The kit's numbers divide at full precision; only the display rounds.
            if taught_qty:
                rate = parsed["hours"] / float(taught_qty)
            qty = quantity_for_unit(text, per)
            value["hours_per_unit"] = round(rate, 4)
            value["hours_unit"] = per
            value["estimated_hours"] = round(rate * qty, 2) if qty is not None else None
            taught = f"{parsed['hours']:g} h for {taught_qty:g} {_plural(per, taught_qty)} = " if taught_qty else ""
            shown = f"{rate:.4g}" if taught_qty else f"{rate:g}"
            value["hours_basis"] = taught + f"{shown} h per {per}" + (f" x {qty:g}" if qty is not None else " (quantity not stated)")
        else:
            value["estimated_hours"] = rate
            value["hours_basis"] = f"{rate:g} h"
        if parsed.get("role"):
            value["hours_role"] = parsed["role"]
        value["hours_correction_id"] = getattr(d, "correction_id", None)
        value["hours_confidence"] = round(float(getattr(d, "confidence", 0.0) or 0.0), 3)
        # Pedigree: how many taught lines stand behind this rate, so the kit
        # can say "learned from N kit lines" instead of asking to be believed.
        try:
            c = store.get(str(getattr(d, "correction_id", "") or "")) if hasattr(store, "get") else None
            ex = list(getattr(c, "exemplars", None) or []) if c is not None else []
            if ex:
                value["hours_evidence"] = len(ex)
                value["hours_taught_by"] = str(getattr(c, "created_by", "") or "")
        except Exception:
            pass
        stamped += 1
    return stamped


__all__ = [
    "RELATION", "UNSTATED", "BASIS_STATED", "BASES", "encode_hours_verdict", "parse_hours_verdict",
    "hours_judgment_basis", "hours_judgment_problem", "quantity_for_unit", "estimate_task_hours",
]
