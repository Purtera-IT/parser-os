"""Brief v2: the super heads decide what goes in a deal's brief, one LLM call
writes it, every sentence cited.

Each part of the brief is decided by a head when the head registry
(``c3_record/heads.json``) says that head beat guessing on unseen deals, and
by a fast rule from the parse otherwise. Every item says which one decided it
(``by``), so a labeling round that promotes a head shows up here as items
moving from ``rule:`` to ``base:`` or ``purtera:`` with nothing else changed.

    part        head (when trusted)          fallback
    ----------  ---------------------------  ------------------------------------
    keep        read:co_action (purtera)     the line's type is brief material
    left out    (why) co_reason, noise_class rejected / small_talk types
    section     read:sow_section             section by line type
    service     deal:primary_service         the parse's domain pack
    questions   col:wants / read:intake_gap  open_question lines, missing_info packets
    owner       read:needed_by (purtera)     project_manager
    conflicts   rel:contradicts              packets with contradicting lines

Switch: ``SOWSMITH_C3_BRIEF=on`` (the compile's one base read then also asks
for these heads); ``SOWSMITH_C3_BRIEF_WRITE=on`` adds the writing call.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import Counter, defaultdict
from typing import Any

log = logging.getLogger(__name__)

KEEP_BAR = 0.9           # co_action keep at or above this: in the brief body (71% right on unseen 010043)
MAYBE_BAR = 0.5          # keep between MAYBE_BAR and KEEP_BAR: "also mentioned"
SECTION_BAR = 0.4
NOISE_BAR = 0.6
SERVICE_BAR = 0.6
QUESTION_BAR = 0.5
QUESTION_KEY = "jdg:gap"
#: Catalog questions at or above this p_valid are asked. Chosen on 010043
#: (validation deal): its top 7 were all valid, the head's own 0.5 cut found 2
#: of 13. SOWSMITH_C3_QUESTION_BAR overrides.
QUESTION_CATALOG_BAR = float(os.environ.get("SOWSMITH_C3_QUESTION_BAR", "0.33"))
CONFLICT_BAR = 0.5
MAX_FACTS_PER_SECTION = 25
MAX_LABEL_TASKS = 40

#: Fallback section by line type, the order a SOW reads in.
SECTION_BY_TYPE: dict[str, str] = {}
for _section, _types in {
    "overview": ("deal_metadata", "project_metadata", "deal_state", "quote_status", "lifecycle_status"),
    "scope_of_work": ("scope_item", "task", "service_line", "cutover_step", "integration_checkpoint",
                      "requirement", "quantity", "material", "bom_line", "vendor_line_item", "site_roster",
                      "physical_site", "site_attribute", "site_infrastructure", "site_room_mix",
                      "site_implementation_note", "asset_record", "pmo"),
    "deliverables": ("deliverable",),
    "assumptions": ("assumption", "pricing_assumption"),
    "out_of_scope": ("exclusion",),
    "acceptance": ("acceptance_criterion", "electrical_acceptance_test", "cutover_validation"),
    "timeline": ("milestone_phase", "deadline", "blackout_date_range", "site_access_window",
                 "lead_time_constraint"),
    "fees": ("commercial_total", "payment_term", "expense", "site_budget", "license_subscription"),
    "change_control": ("change_order_rule",),
    "terms": ("contract_term", "compliance", "compliance_rule", "bonding_insurance", "support_entitlement"),
    "customer_responsibilities": ("dependency", "customer_instruction", "site_access_restriction"),
    "provider_responsibilities": ("meeting_commitment", "action_item", "decision", "constraint", "risk",
                                  "mitigation"),
    "contacts": ("stakeholder", "signatory", "approval_authority"),
}.items():
    for _t in _types:
        SECTION_BY_TYPE[_t] = _section

#: Types that never belong in a brief whatever a head says.
NEVER = frozenset({"small_talk", "raw_utterance", "raw_table_row", "dropped_sheet", "entity",
                   "schematic_detection_target_set", "schematic_symbol_detection", "schematic_line_run"})
_PLACEHOLDER = re.compile(r"\s*\[(image|figure|table) awaiting", re.I)
QUESTION_WANTS = ("chase-artifact", "chase-conversation", "confirm-with-customer", "decide-internally")


def enabled() -> bool:
    return os.environ.get("SOWSMITH_C3_BRIEF", "").strip().lower() in ("1", "true", "yes", "on")


def writing_enabled() -> bool:
    return os.environ.get("SOWSMITH_C3_BRIEF_WRITE", "").strip().lower() in ("1", "true", "yes", "on")


def _trusted(key: str) -> bool:
    forced = {k.strip() for k in os.environ.get("SOWSMITH_C3_BRIEF_TRUST", "").split(",") if k.strip()}
    if key in forced:
        return True
    from c3_record.heads import trusted

    return trusted(key)


def _by(key: str) -> str:
    from c3_record.heads import layer

    return f"{layer(key) or 'base'}:{key}"


def _type(atom: Any) -> str:
    t = getattr(atom, "atom_type", "")
    return str(getattr(t, "value", t) or "")


def _deal_kit(atom: Any) -> bool:
    try:
        from app.core.c3_type_base import _DEAL, skipped

        return skipped(atom, _DEAL.get("sha"))
    except Exception:
        return False


def build(atoms: list[Any], packets: list[Any] = (), *, service_fallback: str | None = None,
          readings: dict[str, Any] | None = None) -> dict[str, Any]:
    """The structured brief: no LLM. ``readings`` is the compile's full read
    (``c3_type_base.full_read()``); without it every part falls back."""
    started = time.monotonic()
    readings = readings or {}
    reads: dict[str, dict[str, list]] = readings.get("reads") or {}
    links: list[dict[str, Any]] = readings.get("links") or []
    deal: dict[str, list] = readings.get("deal") or {}
    have_reads = bool(reads)
    by_id = {str(a.id): a for a in atoms}

    def r(atom_id: str, head: str) -> tuple[str, float]:
        v = (reads.get(atom_id) or {}).get(head)
        return (str(v[0]), float(v[1] or 0.0)) if v and v[0] is not None else ("_absent", 0.0)

    use = {k: have_reads and _trusted(k) for k in (
        "read:co_action", "read:co_reason", "read:noise_class", "read:sow_section", "col:wants",
        "read:intake_gap", "read:needed_by")}
    use["deal:primary_service"] = bool(deal) and _trusted("deal:primary_service")
    use["rel:contradicts"] = bool(links) and _trusted("rel:contradicts")

    # ---- keep / left out
    # co_action decides. noise_class and co_reason only say WHY a line was
    # left out: noise_class names a kind of noise for every line it reads
    # (its labels are all noise kinds), so it is no gate -- run first it threw
    # out 28 of the 57 lines the labeler kept on 000002, each read keep >= 0.998.
    kept, maybe, left_out, label_tasks = [], [], [], []
    why = Counter()

    def _reason(aid: str, fallback: str) -> tuple[str, str]:
        if use["read:co_reason"]:
            c, pc = r(aid, "read:co_reason")
            if c != "_absent" and pc >= 0.5:
                return c, _by("read:co_reason")
        if use["read:noise_class"]:
            n, pn = r(aid, "read:noise_class")
            if n != "_absent" and pn >= NOISE_BAR:
                return n, _by("read:noise_class")
        return fallback, "rule"

    for a in atoms:
        aid, t = str(a.id), _type(a)
        status = str(getattr(getattr(a, "review_status", ""), "value", getattr(a, "review_status", "")))
        if _deal_kit(a):
            # The base is never shown a Deal Kit or estimating sheet (it is
            # what the kit model is trained to produce), so its reading of one
            # means nothing: 45 of 135 false keeps on 010043 were kit lines.
            left_out.append({"id": aid, "reason": "old_manual_deal_kit", "by": "rule:document"})
            why["rule:deal_kit"] += 1
            continue
        act, p = r(aid, "read:co_action") if use["read:co_action"] else ("_absent", 0.0)
        if status == "rejected" or (t in NEVER and not (t == "small_talk" and act == "keep" and p >= 0.9)):
            left_out.append({"id": aid, "reason": "rejected" if status == "rejected" else f"type {t}",
                             "by": "rule:type"})
            why["rule:never"] += 1
            continue
        if act != "_absent":
            if act == "keep" and p >= KEEP_BAR:
                kept.append((a, p, _by("read:co_action")))
                why["head:keep"] += 1
            elif act == "keep" and p >= MAYBE_BAR:
                maybe.append((a, p))
                why["head:maybe"] += 1
            else:
                reason, rby = _reason(aid, act)
                left_out.append({"id": aid, "reason": reason, "by": rby, "p": round(p, 3)})
                why["head:" + act] += 1
            # The lines the head is least sure of are the best ones to label.
            if 0.3 <= p < 0.9 and act == "keep" or 0.3 <= p < 0.6:
                label_tasks.append({"id": aid, "head": "read:co_action", "base_says": act, "p": round(p, 3),
                                    "text": a.raw_text[:200]})
        elif t in SECTION_BY_TYPE:
            kept.append((a, None, "rule:type"))
            why["rule:keep"] += 1
        else:
            left_out.append({"id": aid, "reason": f"type {t}", "by": "rule:type"})
            why["rule:drop"] += 1

    # ---- sections
    sections: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for a, p, by in kept:
        aid = str(a.id)
        sec, ps = r(aid, "read:sow_section") if use["read:sow_section"] else ("_absent", 0.0)
        if sec != "_absent" and ps >= SECTION_BAR:
            sec_by = _by("read:sow_section")
        else:
            sec, sec_by = SECTION_BY_TYPE.get(_type(a), "other"), "rule:type"
        sections[sec].append({"id": aid, "text": a.raw_text[:400], "type": _type(a),
                              "keep_p": None if p is None else round(p, 3), "by": by, "section_by": sec_by})
    for sec in sections:
        sections[sec].sort(key=lambda x: -(x["keep_p"] or 0.5))

    # ---- service
    # The head answers when trusted, and also when the parse has no real pack
    # to fall back on ("default_pack" names none): on 000002 the head said
    # low_voltage_cabling, the labeler's answer, and the rule said default_pack.
    svc = deal.get("deal:primary_service") if deal else None
    no_rule = service_fallback in (None, "", "default_pack")
    if svc and svc[0] not in (None, "_absent") and (
            (use["deal:primary_service"] and float(svc[1] or 0) >= SERVICE_BAR) or no_rule):
        service = {"value": svc[0], "p": round(float(svc[1] or 0), 3), "by": _by("deal:primary_service")}
    else:
        service = {"value": service_fallback, "by": "rule:domain_pack",
                   "base_says": (deal.get("deal:primary_service") or [None])[0]}

    # ---- questions
    # 1. Base: every catalog question (c3_record/questions.json) asked of the
    #    whole deal by the Questions head; it says whether THIS deal needs it
    #    and nothing answers it yet. Kept at or above the bar, best first.
    #    On 010043 its scores ranked the labeler's valid ones first (AUC 0.92;
    #    its top 7 were all valid); the head's own 0.5 cut found 2 of 13.
    # 2. The deal's own open asks: lines that ask something, which co_action
    #    keeps (call chatter is typed open_question too).
    questions: list[dict[str, Any]] = []
    seen_q: set[str] = set()
    asked = readings.get("questions") or []
    if asked:
        try:
            from c3_record.questions_catalog import by_text

            meta = by_text()
        except Exception:
            meta = {}
        svc_value = str(service.get("value") or "")
        for q in sorted(asked, key=lambda x: -float(x.get("p_valid") or 0)):
            p_valid = float(q.get("p_valid") or 0)
            if p_valid < QUESTION_CATALOG_BAR:
                break
            m = meta.get(q["text"]) or {}
            if m.get("applies_to") and svc_value and svc_value not in m["applies_to"]:
                continue
            questions.append({"id": "q:" + str(m.get("id") or len(questions)), "text": q["text"],
                              "family": m.get("family", ""), "layer": m.get("layer", "base"),
                              "p": round(p_valid, 3), "by": _by(QUESTION_KEY), "owner": "project_manager"})
    for a in atoms:
        aid = str(a.id)
        hit = None
        if use["col:wants"]:
            w, pw = r(aid, "col:wants")
            if w in QUESTION_WANTS and pw >= QUESTION_BAR:
                hit = (w, _by("col:wants"))
        if not hit and use["read:intake_gap"]:
            g, pg = r(aid, "read:intake_gap")
            if g == "true" and pg >= QUESTION_BAR:
                hit = ("intake_gap", _by("read:intake_gap"))
        if not hit and _type(a) == "open_question":
            # Call chatter is typed open_question too ("can you meet us in
            # Tampa?"): the line must also be one co_action keeps.
            act, pa = r(aid, "read:co_action") if use["read:co_action"] else ("keep", 1.0)
            if act == "keep" and pa >= MAYBE_BAR and not _deal_kit(a) and not _PLACEHOLDER.match(a.raw_text):
                hit = ("open_question", "rule:type+" + _by("read:co_action") if use["read:co_action"] else "rule:type")
        if hit:
            owner = r(aid, "read:needed_by")[0] if use["read:needed_by"] else "_absent"
            questions.append({"id": aid, "text": a.raw_text[:300], "wants": hit[0], "by": hit[1],
                              "family": "deal_ask", "layer": "base",
                              "owner": owner if owner != "_absent" else "project_manager"})
            seen_q.add(aid)
    for pk in packets or ():
        fam = str(getattr(getattr(pk, "family", ""), "value", getattr(pk, "family", "")))
        if fam != "missing_info":
            continue
        ids = [i for i in (list(pk.governing_atom_ids) + list(pk.supporting_atom_ids)) if i in by_id]
        if not ids or ids[0] in seen_q:
            continue
        a = by_id[ids[0]]
        act, pa = r(ids[0], "read:co_action") if use["read:co_action"] else ("keep", 1.0)
        if act != "keep" or pa < MAYBE_BAR or _deal_kit(a) or _PLACEHOLDER.match(a.raw_text):
            continue
        seen_q.add(ids[0])
        # The packet's reason is boilerplate ("Open question from transcript
        # requires clarification."); the line itself is the question.
        questions.append({"id": ids[0], "text": a.raw_text[:300], "wants": "missing_info",
                          "family": "deal_ask", "layer": "base",
                          "by": "rule:packet", "owner": "project_manager"})

    # ---- conflicts
    conflicts: list[dict[str, Any]] = []
    if use["rel:contradicts"]:
        for ln in links:
            if ln.get("relation") == "contradicts" and float(ln.get("p") or 0) >= CONFLICT_BAR \
                    and ln.get("src") in by_id and ln.get("dst") in by_id:
                conflicts.append({"ids": [ln["src"], ln["dst"]], "p": round(float(ln["p"]), 3),
                                  "by": _by("rel:contradicts")})
    else:
        for pk in packets or ():
            bad = [i for i in pk.contradicting_atom_ids if i in by_id]
            good = [i for i in pk.governing_atom_ids if i in by_id]
            if bad and good:
                conflicts.append({"ids": [good[0], bad[0]], "about": pk.anchor_key, "by": "rule:packet"})

    label_tasks.sort(key=lambda x: abs(x["p"] - 0.45))
    brief = {
        "version": 2,
        "service": service,
        "sections": dict(sections),
        "also_mentioned": [{"id": str(a.id), "text": a.raw_text[:200], "keep_p": round(p, 3)}
                           for a, p in sorted(maybe, key=lambda x: -x[1])[:150]],
        "questions": questions,
        "conflicts": conflicts[:40],
        "left_out": {"count": len(left_out), "by_reason": dict(Counter(x["reason"] for x in left_out).most_common(12))},
        "decided_by": {
            "heads_used": sorted(k for k, v in use.items() if v),
            "fell_back": sorted(k for k, v in use.items() if not v),
            "base_read": have_reads,
            "keep": dict(why),
        },
        "label_next": label_tasks[:MAX_LABEL_TASKS],
        "assemble_ms": round((time.monotonic() - started) * 1000, 1),
    }
    _stamp_layers(brief)
    return brief


def _layer_of(by: str) -> str:
    """base: decided by a universal head (any company); purtera: by a
    company head; rule: by a fallback rule until a head is trusted."""
    head = by.split("+")[-1]
    return "purtera" if head.startswith("purtera:") else "base" if head.startswith("base:") else "rule"


def _stamp_layers(brief: dict[str, Any]) -> None:
    counts: Counter = Counter()
    items = [x for v in brief["sections"].values() for x in v] + brief["questions"] + brief["conflicts"]
    for x in items:
        x["decided_by_layer"] = _layer_of(str(x.get("by") or ""))
        counts[x["decided_by_layer"]] += 1
    brief["service"]["decided_by_layer"] = _layer_of(str(brief["service"].get("by") or ""))
    brief["decided_by"]["layers"] = dict(counts)


_PROMPT = (
    "Write a project manager's brief from these facts. Every sentence must cite the [ids] of the facts it "
    "uses, and say nothing that is not in a fact. Return JSON only: {\"summary\": str, \"sections\": "
    "{section: [{\"text\": str, \"cites\": [id]}]}, \"open_questions\": [{\"question\": str, \"owner\": str, "
    "\"cites\": [id]}]}.\n\nSERVICE: %s\nFACTS BY SECTION: %s\nOPEN QUESTIONS: %s\nCONFLICTS: %s"
)


def write(brief: dict[str, Any], *, model: str | None = None) -> dict[str, Any]:
    """One LLM call turns the structured brief into prose. Sentences whose
    citations are not facts of this brief are dropped and counted."""
    from app.core import llm_client

    facts = {s: [f"[{x['id']}] {x['text']}" for x in v[:MAX_FACTS_PER_SECTION]]
             for s, v in brief["sections"].items()}
    qs = [f"[{q['id']}] (owner: {q['owner']}) {q['text']}" for q in brief["questions"][:30]]
    cs = [x["ids"] for x in brief["conflicts"][:15]]
    known = {x["id"] for v in brief["sections"].values() for x in v} | {q["id"] for q in brief["questions"]}
    started = time.monotonic()
    raw = llm_client.complete(_PROMPT % (brief["service"].get("value"), json.dumps(facts), json.dumps(qs),
                                         json.dumps(cs)),
                              max_tokens=4000, model=model or os.environ.get("SOWSMITH_C3_BRIEF_MODEL") or None,
                              timeout=int(os.environ.get("SOWSMITH_C3_BRIEF_TIMEOUT", "120")))
    out: dict[str, Any] = {"write_ms": round((time.monotonic() - started) * 1000, 1)}
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        out["error"] = "no JSON from the writing call"
        return out
    try:
        doc = json.loads(m.group(0))
    except Exception as exc:
        out["error"] = f"bad JSON: {exc}"[:200]
        return out
    dropped = 0
    for sec, sentences in list((doc.get("sections") or {}).items()):
        keep = []
        for s in sentences or []:
            cites = [str(c).strip("[]") for c in (s.get("cites") or [])] if isinstance(s, dict) else []
            if cites and all(c in known for c in cites):
                keep.append({"text": s.get("text", ""), "cites": cites})
            else:
                dropped += 1
        doc["sections"][sec] = keep
    out.update(summary=doc.get("summary", ""), sections=doc.get("sections") or {},
               open_questions=doc.get("open_questions") or [], uncited_dropped=dropped)
    return out


def run(atoms: list[Any], packets: list[Any] = (), *, service_fallback: str | None = None) -> dict[str, Any]:
    """Build (and, when switched on, write) the brief from this compile's read."""
    from app.core import c3_type_base

    brief = build(atoms, packets, service_fallback=service_fallback, readings=c3_type_base.full_read())
    if writing_enabled():
        try:
            brief["written"] = write(brief)
        except Exception as exc:
            brief["written"] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
    return brief
