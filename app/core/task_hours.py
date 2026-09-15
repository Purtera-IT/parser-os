"""Learned hours for a unit of work -- taught from finished Deal Kits, not listed.

A Deal Kit is the answer to "how long does this take": 000020 Binghamton's
Estimate_Detail gives 0.75 h to "Remove the 1518 SonicWall from service, label
it, and store as a backup"; 000061 MBrany's Install ROM gives "3 h per cable
drop". Each becomes a correction on relation ``task_hours`` whose exemplar is
the work line and whose verdict carries the hours:

    hours=0.75                          fixed hours for that piece of work
    hours=3;per=cable drop              hours per unit; scaled by the quantity
    hours=4;role=L1 EUC                 optionally the role that did it

At compile time every task atom asks the store (nearest taught exemplar,
guess-free, no LLM). A confident match stamps ``estimated_hours`` and where the
number came from on the atom; abstain leaves it alone. Per-unit hours scale by
a number written next to the taught unit ("39 AP locations" for "per AP
location"), matched on the unit's own words -- the unit comes from the kit, so
there is no list of nouns here. A per-unit match with no quantity in the text
records the rate and no total, rather than guessing a count.
"""

from __future__ import annotations

import re
from typing import Any

RELATION = "task_hours"

_NUM_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w.])")
_WORD_RE = re.compile(r"[a-z0-9]+")


def encode_hours_verdict(hours: float, *, per: str = "", role: str = "") -> str:
    parts = [f"hours={float(hours):g}"]
    if per.strip():
        parts.append(f"per={per.strip()}")
    if role.strip():
        parts.append(f"role={role.strip()}")
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
    return out


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
            qty = quantity_for_unit(text, per)
            value["hours_per_unit"] = rate
            value["hours_unit"] = per
            value["estimated_hours"] = round(rate * qty, 2) if qty is not None else None
            value["hours_basis"] = f"{rate:g} h per {per}" + (f" x {qty:g}" if qty is not None else " (quantity not stated)")
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


__all__ = ["RELATION", "encode_hours_verdict", "parse_hours_verdict", "quantity_for_unit", "estimate_task_hours"]
