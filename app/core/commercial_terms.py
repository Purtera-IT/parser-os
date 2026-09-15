"""The commercial shape of a kit, learned from finished Deal Kits -- not listed.

A Deal Kit's sheet says Fixed or Hourly; its Gantt carries a PM line and a
travel line. None of that is written in the deal's documents, so nothing can
be extracted from them -- but it recurs: the same kind of request gets the
same shape of kit. 010198 (one register and a printer, after hours) was Fixed
with a PC line; 010095 (install two device servers) was Hourly with travel.

Each finished kit teaches one correction on relation ``commercial_terms``
whose exemplar is the deal's request -- a priced task line -- and whose
verdict carries the kit's shape:

    billing=fixed;pm_hours=2;pc_hours=1;travel_days=1     (any subset)

At compile time every task atom asks the store (nearest taught request,
guess-free, no LLM). A confident match stamps ``value.commercial_terms`` on
the atom with the parsed shape, its pedigree (how many kit lines stand behind
it, who taught it) and the store's confidence; the Deal Kit prefill takes the
best-supported stamp for the quote's scalars and says where it came from.
Abstain leaves nothing: a shape nobody taught is not guessed.
"""

from __future__ import annotations

from typing import Any

RELATION = "commercial_terms"
BILLING_TYPES = ("fixed", "t_and_m", "milestone")
_NUMBER_KEYS = ("pm_hours", "pc_hours", "travel_days")


def encode_commercial_verdict(*, billing: str = "", pm_hours: float | None = None, pc_hours: float | None = None,
                              travel_days: float | None = None) -> str:
    """The kit's shape as one verdict string; keys the kit did not set are left out."""
    parts = []
    b = str(billing or "").strip().lower()
    if b in BILLING_TYPES:
        parts.append(f"billing={b}")
    for key, val in (("pm_hours", pm_hours), ("pc_hours", pc_hours), ("travel_days", travel_days)):
        if val is None:
            continue
        try:
            f = float(val)
        except (TypeError, ValueError):
            continue
        if f >= 0:
            parts.append(f"{key}={f:g}")
    return ";".join(parts)


def parse_commercial_verdict(verdict: str) -> dict[str, Any] | None:
    """The shape a verdict carries, or None when it carries nothing usable."""
    out: dict[str, Any] = {}
    for part in str(verdict or "").split(";"):
        key, sep, val = part.partition("=")
        if not sep:
            continue
        key, val = key.strip(), val.strip()
        if key == "billing":
            if val.lower() in BILLING_TYPES:
                out["billing"] = val.lower()
        elif key in _NUMBER_KEYS:
            try:
                f = float(val)
            except ValueError:
                continue
            if f >= 0:
                out[key] = f
    return out or None


def _atom_type(atom: Any) -> str:
    at = getattr(atom, "atom_type", None)
    return str(getattr(at, "value", at) or "")


def stamp_commercial_terms(atoms: list[Any], *, store: Any = None) -> int:
    """Stamp the learned commercial shape on task atoms. Returns how many were stamped."""
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
            if getattr(c, "relation", "") == RELATION and parse_commercial_verdict(c.verdict)
        })
    except Exception:
        return 0
    if not verdicts:
        return 0
    stamped = 0
    for atom in atoms:
        if _atom_type(atom) != "task":
            continue
        _val = getattr(atom, "value", None)
        # A mention folded into a fuller statement of the same work is not the request.
        if isinstance(_val, dict) and _val.get("folded_into"):
            continue
        text = str(getattr(atom, "raw_text", "") or "").strip()
        if not text:
            continue
        scope = DecisionScope(deal_id=str(getattr(atom, "project_id", "") or ""))
        d = decide(RELATION, text[:600], verdicts, instruction="The commercial shape finished Deal Kits gave this kind of request.",
                   llm=False, scope=scope)
        if d is None or d.source != "store" or not d.verdict:
            continue
        parsed = parse_commercial_verdict(d.verdict)
        if not parsed:
            continue
        value = getattr(atom, "value", None)
        if not isinstance(value, dict):
            value = {}
            try:
                atom.value = value
            except Exception:
                continue
        value["commercial_terms"] = parsed
        value["commercial_terms_correction_id"] = getattr(d, "correction_id", None)
        value["commercial_terms_confidence"] = round(float(getattr(d, "confidence", 0.0) or 0.0), 3)
        # Pedigree: how many taught requests stand behind this shape, and who taught it.
        try:
            c = store.get(str(getattr(d, "correction_id", "") or "")) if hasattr(store, "get") else None
            ex = list(getattr(c, "exemplars", None) or []) if c is not None else []
            if ex:
                value["commercial_terms_evidence"] = len(ex)
                value["commercial_terms_taught_by"] = str(getattr(c, "created_by", "") or "")
        except Exception:
            pass
        stamped += 1
    return stamped


__all__ = ["RELATION", "BILLING_TYPES", "encode_commercial_verdict", "parse_commercial_verdict", "stamp_commercial_terms"]
