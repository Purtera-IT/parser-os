"""The rule-fire ledger: every time a rule overrides, merges, vetoes or fills something the super heads decide.

The heads decide each line's type; rules downstream still re-type, merge, veto or fill. Each such rule is to be
replaced by a head through targeted refits, and to know when, every fire is counted with what the head said and
what the rule did:

    fire(rule_id, atom, head_answer, rule_answer, note="")

"Agree" means the head's own answer for that line already matched what the rule did. ``rule_answer`` is a type
("physical_site"), or ``not:<type>`` when the rule's act is that the line is NOT that (a veto: the head agrees when
it said anything else). ``head_answer`` None means no head answers this question for this line (a merge of two
site lines is a pair judgment the type head does not make; a quote field the serve filled by fallback): the fire
counts as ``no_head``, never as agreement.

A rule moves Active -> Shadow (the head decides, the rule only logs) once the head agrees on ~95% of its fires over
20+ unseen deals, then Retired. ``SOWSMITH_RULE_MODES`` sets the mode per rule (``rule_id:active|shadow``, comma
separated; ``prefix.*`` matches every rule under a prefix; default active). ``shadow(rule_id)`` tells a call site
to log without acting. A rule whose call site cannot skip its act safely is registered ``can_shadow=False``: it
always acts, and its record says the shadow ask was not honoured.

The records land in the compile result and the envelope as ``rule_fires`` =
``{rule_id: {fires, agree_with_head, disagree, no_head, mode, stage, examples[<=5 atom ids]}}`` and, per deal, in the brief
record as ``decided_by.rule_fires``. ``scripts/rule_ledger_report.py`` sums them over deals.
"""

from __future__ import annotations

import os
import threading
from typing import Any

#: Every rule instrumented: what it does, whether its call site honours shadow mode, and its ``stage``:
#: ``upstream`` acts before or at the heads' decision, on parser output; ``downstream`` edits a fact the heads already
#: decided (merging, vetoing, retyping, dropping sites after the heads). Owner, 2026-10-29: downstream rules are to
#: go -- their fires become refit labels, or the rule moves upstream. ``default``: the mode when the env names none.
#: Ids under ``quote.`` are minted per field and pick (``quote.hours.fallback:family_median``): log only, upstream
#: (the serve fills what no head answered).
RULES: dict[str, dict[str, Any]] = {
    "party_address_veto.signature_page": {
        "stage": "downstream",
        "what": "a physical_site on a signature page (or whose street is in the signature block) becomes deal_metadata",
        "can_shadow": True},
    "party_address_veto.signature_keys": {
        "stage": "downstream",
        "what": "a non-site line on a signature page loses its site: keys", "can_shadow": True},
    # Replaced by repeat_address.follow_full_copy (upstream): shadow by default, so its fires show what the
    # upstream rule missed.
    "party_address_veto.follow_party_copies": {
        "stage": "downstream", "default": "shadow",
        "what": "a bare copy of a stakeholder/signatory address is retyped stakeholder", "can_shadow": True},
    "repeat_address.follow_full_copy": {
        "stage": "upstream",
        "what": "a bare repeat of an address takes the heads' answer on its full-context copy, at the heads' read",
        "can_shadow": True},
    "quote_sites.fold_contained": {
        "stage": "downstream",
        "what": "a quote site whose keys sit inside ONE other site's key is folded into it", "can_shadow": True},
    "quote_sites.fold_mention": {
        "stage": "downstream",
        "what": "a quote site whose keys sit inside SEVERAL sites is dropped as a mention", "can_shadow": True},
    "overwrite_bar.extracted": {
        "stage": "upstream",
        "what": "a sure head answer is refused because the parser extracted values the new type drops (0.85 bar)",
        "can_shadow": True},
    "overwrite_bar.derived_finding": {
        "stage": "upstream",
        "what": "our sheet summary (derived_finding) never becomes physical_site", "can_shadow": True},
    "site_keys.join_split_address": {
        "stage": "downstream",
        "what": "adjacent street / City ST ZIP site lines take the first line's key", "can_shadow": True},
    "site_keys.street_copy": {
        "stage": "downstream",
        "what": "a street-only site line written inside one keyed site line takes its key", "can_shadow": True},
    "site_keys.mint_from_head": {
        "stage": "upstream",
        "what": "a head physical_site line with no key gets one from its words", "can_shadow": True},
    "site_keys.join_same_place": {
        "stage": "upstream",
        "what": "site lines naming one street and ZIP or one store number take one key; no head answers same-site yet",
        "can_shadow": True},
    "site_keys.names_no_place": {
        "stage": "upstream",
        "what": "on a deal whose site lines name places, a site line naming no street, ZIP or id is no roster site",
        "can_shadow": True},
    "site_keys.drop_unbacked": {
        "stage": "downstream",
        "what": "a site key no site line backs is dropped from a line", "can_shadow": True},
    "semantic_dedup.site_alias": {
        "stage": "downstream",
        "what": "a name-only physical_site is removed as an alias of a located site", "can_shadow": True},
    "semantic_dedup.hallucinated_site": {
        "stage": "downstream",
        "what": "a physical_site whose name, address and facility are one string is dropped as an LLM ghost "
                "(never a site-list row read cell by cell)", "can_shadow": True},
    "semantic_dedup.site_demote": {
        "stage": "downstream",
        "what": "a physical_site loses its site claim (kept for its words, typed deal_metadata)", "can_shadow": True},
    # Log only: the ghost drop and the group merge feed the winner's merged value in the same pass; skipping them
    # would leave two sites sharing one canonical id.
    "semantic_dedup.site_ghost": {
        "stage": "downstream",
        "what": "a physical_site no canonical site resolves is dropped (covered by a kept site)", "can_shadow": False},
    "semantic_dedup.site_merge": {
        "stage": "downstream",
        "what": "a physical_site is merged into its canonical group's winner", "can_shadow": False},
    "site_geo_fallback.address": {
        "stage": "downstream",
        "what": "a physical_site is minted from a City/State/ZIP address in a line's text", "can_shadow": True},
    "site_geo_fallback.mention": {
        "stage": "downstream",
        "what": "a physical_site is minted from a named place judged job_site (geo_mention_role)", "can_shadow": True},
    "site_geo_fallback.vendor_suppress": {
        "stage": "downstream",
        "what": "a physical_site read as the vendor's letterhead address is suppressed", "can_shadow": True},
    # Log only: the head's own act (a locked line a rule re-typed goes back); each fire shows a rule upstream
    # that overrode the head.
    # A PM's own fix on the deal page (Scope Evidence / Sources reader): the top authority, and an approved label.
    "pm_fix.not_a_site": {"stage": "pm", "what": "a PM marked this line not a site, with a why", "can_shadow": False},
    "pm_fix.address": {"stage": "pm", "what": "a PM gave this site's right address, with a why", "can_shadow": False},
    "pm_fix.set_aside": {"stage": "pm", "what": "a PM set this line aside", "can_shadow": False},
    "pm_fix.looks_right": {"stage": "pm", "what": "a PM confirmed the heads' answer on this line", "can_shadow": False},
    "authority_final.restore": {
        "stage": "upstream",
        "what": "apply_authority(final=True) puts back the head's type a rule changed", "can_shadow": False},
    # The envelope's site list (orbitbrief_core.build_site_readiness) reads the heads' roster; its old merges
    # only log what they would have merged (shadow by default: "default": "shadow").
    "site_readiness.alias_merge": {
        "stage": "downstream", "default": "shadow",
        "what": "two sites the heads' roster keeps apart are one row (collect_site_alias_groups)", "can_shadow": True},
    "site_readiness.name_collapse": {
        "stage": "downstream", "default": "shadow",
        "what": "a roster site is folded by name / prefix / initials into another row (FIX 5c canonical_map)",
        "can_shadow": True},
    "quote.site_row_hours": {
        "stage": "downstream",
        "what": "the site list's Total Hrs column replaces the quote line's hours (average per site)",
        "can_shadow": True},
}

_MAX_EXAMPLES = 5
_LOCK = threading.Lock()
_FIRES: dict[str, dict[str, Any]] = {}
#: (rule, atom id) already counted: a rule run more than once per compile (the party veto runs four times; in
#: shadow it meets the same line each time) counts a line once.
_SEEN: set[tuple[str, str]] = set()


def reset() -> None:
    """Forget the previous compile's fires (the compiler calls it at setup)."""
    with _LOCK:
        _FIRES.clear()
        _SEEN.clear()


def _modes() -> dict[str, str]:
    out: dict[str, str] = {}
    for part in os.environ.get("SOWSMITH_RULE_MODES", "").split(","):
        k, _, v = part.strip().rpartition(":")
        if k and v.strip().lower() in ("active", "shadow"):
            out[k.strip()] = v.strip().lower()
    return out


def asked_mode(rule_id: str) -> str:
    """The mode ``SOWSMITH_RULE_MODES`` asks for this rule: an exact id, else the longest ``prefix.*``."""
    modes = _modes()
    if rule_id in modes:
        return modes[rule_id]
    best = ""
    for k in modes:
        if k.endswith(".*") and rule_id.startswith(k[:-1]) and len(k) > len(best):
            best = k
    return modes.get(best) or (RULES.get(rule_id) or {}).get("default", "active")


def stage(rule_id: str) -> str:
    """``upstream`` or ``downstream`` (see ``RULES``)."""
    got = (RULES.get(rule_id) or {}).get("stage")
    return got or ("upstream" if rule_id.startswith("quote.") else "downstream")


def take(prefix: str) -> dict[str, dict[str, Any]]:
    """Remove and return the records of every rule under ``prefix`` (the envelope's own rules, which run after the
    compile result was written)."""
    out = {k: v for k, v in fires().items() if k.startswith(prefix)}
    with _LOCK:
        for k in out:
            _FIRES.pop(k, None)
        for key in [x for x in _SEEN if x[0].startswith(prefix)]:
            _SEEN.discard(key)
    return out


def can_shadow(rule_id: str) -> bool:
    return bool((RULES.get(rule_id) or {}).get("can_shadow"))


def shadow(rule_id: str) -> bool:
    """True when the call site must log the fire and not act. Always False for a log-only rule."""
    return can_shadow(rule_id) and asked_mode(rule_id) == "shadow"


def mode(rule_id: str) -> str:
    asked = asked_mode(rule_id)
    if asked == "shadow" and not can_shadow(rule_id):
        return "active (shadow not honoured: log only)"
    return asked


def agrees(head_answer: Any, rule_answer: Any) -> bool | None:
    """Did the head already say what the rule did? None when no head answered."""
    if head_answer is None or head_answer == "":
        return None
    h, r = str(head_answer), str(rule_answer)
    if r.startswith("not:"):
        return h != r[4:]
    return h == r


def head_of(atom: Any) -> str | None:
    """The type head's answer for this line: its read for the deal, else the label it left on the value
    (``c3_label``, ``c3_base_says``). ``_keep`` (the parser's type stands) is the line's type now, so call it
    before the rule acts. None when the heads did not read the line."""
    aid = str(getattr(atom, "id", "") or "")
    got_type = None
    try:
        from app.core.c3_type_base import _DEAL

        # A line repeat_address gave its full-context copy's answer: the head's own read of THIS line.
        got = (_DEAL.get("own_reads") or {}).get(aid) or (_DEAL.get("types") or {}).get(aid)
        if isinstance(got, (list, tuple)) and got and got[0]:
            got_type = str(got[0])
    except Exception:  # pragma: no cover - never break a rule over its ledger
        pass
    v = getattr(atom, "value", None)
    if got_type is None and isinstance(v, dict):
        for k in ("c3_label", "c3_base_says"):
            lab = v.get(k)
            # A label a rule wrote (follow_party_copies' "follows_copy") is not the head's answer.
            if isinstance(lab, dict) and lab.get("type") and lab.get("act") != "follows_copy":
                got_type = str(lab["type"])
                break
    if got_type == "_keep":
        t = getattr(atom, "atom_type", None)
        return str(getattr(t, "value", t) or "") or None
    return got_type


def _atom_id(atom: Any) -> str:
    if atom is None:
        return ""
    if isinstance(atom, str):
        return atom
    return str(getattr(atom, "id", "") or "")


def fire(rule_id: str, atom: Any, head_answer: Any, rule_answer: Any, note: str = "") -> bool | None:
    """Record one fire of ``rule_id`` on ``atom`` (an atom, an atom id, or None for a deal-level fire).
    A line counts once per rule per compile. Returns the agreement (None: no head answered). Never raises."""
    try:
        ok = agrees(head_answer, rule_answer)
        aid = _atom_id(atom)
        with _LOCK:
            if aid:
                if (rule_id, aid) in _SEEN:
                    return ok
                _SEEN.add((rule_id, aid))
            rec = _FIRES.get(rule_id)
            if rec is None:
                rec = _FIRES[rule_id] = {"fires": 0, "agree_with_head": 0, "disagree": 0, "no_head": 0,
                                         "_dis": [], "_any": [], "note": ""}
            rec["fires"] += 1
            rec["agree_with_head" if ok else "disagree" if ok is False else "no_head"] += 1
            if aid:
                bucket = rec["_dis"] if ok is False else rec["_any"]
                if len(bucket) < _MAX_EXAMPLES and aid not in bucket:
                    bucket.append(aid)
            if note and not rec["note"]:
                rec["note"] = str(note)[:200]
        # The line's own trail: which rule touched it, against what the heads said, so a PM can see the flow of a
        # decision on the line itself (Scope's Evidence, the Sources reader), not only in the deal's tallies.
        v = getattr(atom, "value", None) if atom is not None and not isinstance(atom, str) else None
        if isinstance(v, dict):
            trail = v.get("rule_trace") if isinstance(v.get("rule_trace"), list) else []
            if len(trail) < 12:
                trail.append({"rule": rule_id, "stage": stage(rule_id), "mode": mode(rule_id),
                              "head": None if head_answer is None else str(head_answer),
                              "did": None if rule_answer is None else str(rule_answer),
                              "agrees": ok, "note": str(note)[:160] if note else ""})
                v["rule_trace"] = trail
        return ok
    except Exception:  # pragma: no cover
        return None


def fires() -> dict[str, dict[str, Any]]:
    """This compile's records: ``{rule_id: {fires, agree_with_head, disagree, no_head, mode, examples}}``.
    Examples list disagreements first (the lines a refit should look at)."""
    with _LOCK:
        snap = {k: dict(v, _dis=list(v["_dis"]), _any=list(v["_any"])) for k, v in _FIRES.items()}
    out: dict[str, dict[str, Any]] = {}
    for rid in sorted(snap):
        r = snap[rid]
        ex = (r["_dis"] + [a for a in r["_any"] if a not in r["_dis"]])[:_MAX_EXAMPLES]
        rec = {"fires": r["fires"], "agree_with_head": r["agree_with_head"], "disagree": r["disagree"],
               "no_head": r["no_head"], "mode": mode(rid), "stage": stage(rid), "examples": ex}
        if r["note"]:
            rec["note"] = r["note"]
        out[rid] = rec
    return out


__all__ = ["RULES", "agrees", "asked_mode", "can_shadow", "fire", "fires", "head_of", "mode", "reset", "shadow", "stage",
           "take"]
