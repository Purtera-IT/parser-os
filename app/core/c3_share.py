"""Who decided what in one compile: the C3 heads, a rule, or the LLM.

``summarize`` counts every decision a compile makes, per kind, from what the compile already
records (the atoms' ``c3_label``, the brief's ``by`` / ``section_by`` / ``parts``, the quote's
``by``, the typing counts in ``deflect_counts.c3.typing`` and the LLM calls per stage), so the
share of decisions the heads made can be read off any result.json, old or new:

    line_type   head: the base typed it (authority / assign / fill) or said the parser type stands
                (keep); override: a later rule changed a head-typed line; llm: lines typing sent to
                the typing LLM; rule: the parser's own type, no head reading
    keep        the brief's keep / leave-out decision per line (``decided_by.keep``)
    section     the brief section each kept line went to (``section_by``)
    brief_part  the brief's parts (``decided_by.parts``; older results: heads_used / fell_back)
    site        physical_site lines: head-typed vs the parser's roster / rules
    site_role   whose address a site is (vendor_role_proposal / _decision source)
    quote       the quote's numbers and answers (``by`` of techs, hours, rate, day type, billing,
                travel, sites)
    llm_stage   LLM calls per stage that DECIDE something (prose writing and value fills excluded)

Works on objects (the compiler) and on JSON dicts (a downloaded result.json)."""
from __future__ import annotations

from collections import Counter
from typing import Any

#: Stages whose LLM calls write prose or fill values the heads' type asked for, not decisions.
PROSE_STAGES = ("c3_brief", "c3_quote", "parse_prefetch")


def _get(o: Any, k: str, default: Any = None) -> Any:
    return o.get(k, default) if isinstance(o, dict) else getattr(o, k, default)


def _type(a: Any) -> str:
    t = _get(a, "atom_type", "")
    return str(getattr(t, "value", t) or "")


def _kind(by: Any) -> str:
    """head | rule | llm | none, from a ``by`` string."""
    b = str(by or "")
    if not b or b.startswith("none"):
        return "none"
    if b.startswith(("base:", "purtera:", "heads:", "head:", "judge:", "c3")):
        return "head"
    if b.startswith(("llm", "teacher")):
        return "llm"
    return "rule"


def line_types(atoms: list[Any], typing: dict[str, Any] | None) -> Counter:
    c: Counter = Counter()
    for a in atoms:
        v = _get(a, "value") or {}
        lab = v.get("c3_label") if isinstance(v, dict) else None
        if isinstance(lab, dict) and lab.get("act"):
            if lab["act"] == "keep" or not lab.get("type") or lab.get("type") == _type(a):
                c["head"] += 1
            else:
                c["override"] += 1
        else:
            c["rule"] += 1
    t = typing or {}
    to_llm = max(0, int(t.get("lines") or 0) - int(t.get("taken") or 0))
    # The lines typing sent to the LLM carry no c3_label: move them from rule to llm.
    moved = min(to_llm, c["rule"])
    c["rule"] -= moved
    c["llm"] += moved
    return c


def sites(atoms: list[Any]) -> tuple[Counter, Counter]:
    site, role = Counter(), Counter()
    for a in atoms:
        if _type(a) != "physical_site":
            continue
        v = _get(a, "value") or {}
        v = v if isinstance(v, dict) else {}
        lab = v.get("c3_label")
        site["head" if isinstance(lab, dict) and lab.get("act") else "rule"] += 1
        src = (v.get("vendor_role_proposal") or {}).get("source") or (v.get("_decision") or {}).get("source")
        if v.get("geo_mention_source"):
            src = v.get("geo_mention_source")
        if src:
            role["head" if src == "c3" else "llm" if src in ("llm", "fallback") else "rule"] += 1
    return site, role


def brief_parts(brief: dict[str, Any]) -> Counter:
    d = (brief or {}).get("decided_by") or {}
    parts = d.get("parts")
    c: Counter = Counter()
    if isinstance(parts, dict):
        for p in parts.values():
            c[str((p or {}).get("by") or "rule")] += 1
    else:
        c["head"] += len(d.get("heads_used") or [])
        c["rule"] += len(d.get("fell_back") or [])
    return c


def keeps(brief: dict[str, Any]) -> Counter:
    c: Counter = Counter()
    for k, n in (((brief or {}).get("decided_by") or {}).get("keep") or {}).items():
        # admission:* (col:admission) and type_head:* (the type head's small_talk) are head decisions too.
        c["head" if str(k).startswith(("head:", "base:", "admission:", "type_head:")) else "rule"] += int(n or 0)
    return c


def sections(brief: dict[str, Any]) -> Counter:
    c: Counter = Counter()
    for items in ((brief or {}).get("sections") or {}).values():
        for it in items or []:
            if isinstance(it, dict):
                c[_kind(it.get("section_by"))] += 1
    return c


def quote(q: dict[str, Any] | None) -> Counter:
    c: Counter = Counter()
    if not isinstance(q, dict) or not q:
        return c
    for line in q.get("lines") or []:
        for k, by in ((line or {}).get("by") or {}).items():
            c[_kind(by) if not str(by).startswith("fallback") else "rule"] += 1
    for k in ("billing", "travel", "sites"):
        part = q.get(k)
        if isinstance(part, dict):
            by = str(part.get("by") or "")
            c[_kind(by) if not by.startswith("fallback") else "rule"] += 1
    return c


def llm_stages(calls_by_stage: dict[str, Any] | None) -> dict[str, int]:
    return {s: int(r.get("llm_calls") or 0) for s, r in (calls_by_stage or {}).items()
            if int((r or {}).get("llm_calls") or 0) and s not in PROSE_STAGES}


def summarize(atoms: list[Any], brief: dict[str, Any] | None, capabilities: dict[str, Any] | None) -> dict[str, Any]:
    caps = capabilities or {}
    c3 = ((caps.get("deflect_counts") or {}).get("c3") or {})
    brief = brief or {}
    site, role = sites(atoms)
    kinds = {
        "line_type": line_types(atoms, c3.get("typing")),
        "keep": keeps(brief),
        "section": sections(brief),
        "brief_part": brief_parts(brief),
        "site": site,
        "site_role": role,
        "quote": quote(brief.get("quote")),
    }
    out: dict[str, Any] = {}
    total: Counter = Counter()
    for k, c in kinds.items():
        c = Counter({x: n for x, n in c.items() if n and x != "none"})
        n = sum(c.values())
        out[k] = {**dict(c), "share_head": round(c["head"] / n, 3) if n else None}
        total.update(c)
    n = sum(total.values())
    out["all"] = {**dict(total), "share_head": round(total["head"] / n, 3) if n else None}
    out["llm_decision_calls"] = llm_stages(caps.get("calls_by_stage"))
    return out
