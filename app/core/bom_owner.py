"""Who supplies a hardware line: us, or the customer?

A Deal Kit's materials are what we buy and sell. A hardware list in the
deal's documents is often the customer's own -- the equipment they bought
through their reseller and want us to install. Live 010095 (2026-09-15): a
note listing the Lantronix device server, the serial cable, a patch cable
and a ThinkCentre reached the Deal Kit prefill as four BOM rows; the kit
ordered nothing, because SHI supplied every one of them. 010043: the
Verkada cloud licence CDW sells became a BOM row the kit never priced.

The question is answered per line through the decide() chokepoint:

    STORE (a PM or a finished kit taught "the customer furnishes this")
      -> LLM  ->  nothing stamped

Only a store hit or a confident model verdict stamps ``value.supplied_by``;
an undecided line carries no verdict and the prefill treats it as it always
did. Nothing is dropped from the atoms: the line stays evidence of what is
being installed, whoever pays for it.
"""

from __future__ import annotations

import os
from typing import Any

RELATION = "bom_owner"
CANDIDATES = ["we_supply", "customer_furnished"]
INSTRUCTION = (
    "A hardware or materials line from a deal's documents. Decide who SUPPLIES it: "
    "we_supply when we buy and sell it as part of our quote, customer_furnished when "
    "the customer or their reseller provides it and we only install, configure or "
    "connect it. A line the customer lists as equipment they have, bought or are "
    "shipping is customer_furnished; a line from our own quote or a materials request "
    "to us is we_supply. If the document does not say, answer unknown."
)
_MIN_CONF = 0.85


def enabled() -> bool:
    return os.environ.get("SOWSMITH_BOM_OWNER", "1").strip().lower() not in ("0", "false", "no", "off")


def _atom_type(atom: Any) -> str:
    at = getattr(atom, "atom_type", None)
    return str(getattr(at, "value", at) or "")


def _context(atom: Any) -> str:
    v = getattr(atom, "value", None) or {}
    bits = []
    if isinstance(v, dict):
        for k in ("title", "intro", "lead_in", "section_header"):
            x = v.get(k)
            if isinstance(x, list):
                x = " ".join(str(i) for i in x)
            if x:
                bits.append(f"{k}: {str(x)[:200]}")
        if v.get("author_affiliation"):
            bits.append(f"written by: {v['author_affiliation']}")
    fn = str(getattr(atom, "source_filename", "") or "")
    if fn:
        bits.append(f"document: {fn[:120]}")
    return "\n".join(bits)


def stamp_bom_owners(atoms: list[Any], *, project_id: str = "") -> tuple[int, list[dict[str, Any]]]:
    """Stamp ``value.supplied_by`` on bom_line atoms the store or model can
    call. Returns (stamped, verdicts). Never raises."""
    try:
        from app.core.decide import DecisionScope, decide
    except Exception:  # pragma: no cover
        return 0, []
    scope = DecisionScope(deal_id=str(project_id or ""))
    stamped = 0
    verdicts: list[dict[str, Any]] = []
    for a in atoms:
        if _atom_type(a) != "bom_line":
            continue
        text = " ".join(str(getattr(a, "raw_text", "") or "").split())
        if not text:
            continue
        try:
            d = decide(RELATION, text[:600], CANDIDATES, instruction=INSTRUCTION, context=_context(a)[:1200],
                       scope=scope, exclude_created_by=("teacher",))
        except Exception:
            d = None
        verdict = getattr(d, "verdict", None)
        conf = float(getattr(d, "confidence", 0.0) or 0.0)
        source = getattr(d, "source", "fallback")
        decided = verdict in CANDIDATES and (source == "store" or conf >= _MIN_CONF)
        verdicts.append({"text": text[:120], "verdict": verdict if decided else None, "model_verdict": verdict,
                         "confidence": round(conf, 3), "source": source})
        if not decided:
            continue
        v = getattr(a, "value", None)
        if not isinstance(v, dict):
            v = {}
            try:
                a.value = v
            except Exception:
                continue
        v["supplied_by"] = verdict
        v["supplied_by_source"] = source
        v["supplied_by_confidence"] = round(conf, 3)
        stamped += 1
    return stamped, verdicts


__all__ = ["RELATION", "CANDIDATES", "INSTRUCTION", "enabled", "stamp_bom_owners"]
