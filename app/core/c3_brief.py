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
#: The owner head moves a line out of customer responsibilities above this.
OWNER_BAR = 0.75
NOISE_BAR = 0.6
SERVICE_BAR = 0.6
QUESTION_BAR = 0.5
QUESTION_KEY = "jdg:gap"
#: Catalog questions kept per topic (site access, schedule, commercial ...),
#: best first, before the writing call drops the ones the deal answers.
#: 0 = no cap: the head decides how many (its scores are calibrated).
QUESTIONS_PER_FAMILY = int(os.environ.get("SOWSMITH_C3_QUESTIONS_PER_FAMILY", "0"))
EVIDENCE_LINES = 5
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


def _derived(atom: Any) -> bool:
    """A line the parser made up from other lines, not one a document has."""
    try:
        for ref in getattr(atom, "source_refs", None) or []:
            loc = getattr(ref, "locator", None) or (ref.get("locator") if isinstance(ref, dict) else None)
            if isinstance(loc, dict) and loc.get("derived"):
                return True
    except Exception:
        pass
    return any(str(f).startswith("derived_") for f in (getattr(atom, "review_flags", None) or []))


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


def _attach_evidence(questions: list[dict[str, Any]], atoms: list[Any]) -> None:
    """The deal lines closest to each catalog question (the parse's own
    embeddings, cached by the compile's warm-up), so the writing call can see
    whether the deal already answers it and word it for this job."""
    if not questions:
        return
    pool = [a for a in atoms if _type(a) not in NEVER and not _deal_kit(a) and len(a.raw_text.strip()) > 12]
    if not pool:
        return
    try:
        import numpy as np

        from app.core.embedding_retrieval import embed_texts

        lines = embed_texts([a.raw_text.strip()[:600] for a in pool])
        qv = embed_texts([q["text"] for q in questions])
        sims = qv @ lines.T
    except Exception as exc:
        log.info("C3 brief evidence skipped: %s", type(exc).__name__)
        return
    for q, row in zip(questions, sims):
        # Distinct lines only: the same clause repeated across drafts ("PAYMENT
        # TERMS" twice, one travel sentence three times on 010038) filled every
        # slot, and the line that answers the question ("Net 30") was crowded out.
        ev, seen = [], set()
        for i in np.argsort(-row):
            if row[i] <= 0 or len(ev) >= EVIDENCE_LINES:
                break
            key = re.sub(r"\W+", " ", pool[i].raw_text.lower()).strip()[:160]
            if key in seen:
                continue
            seen.add(key)
            ev.append({"id": str(pool[i].id), "text": pool[i].raw_text[:240], "sim": round(float(row[i]), 3)})
        q["evidence"] = ev


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
        # A line the parser wrote itself ("stage: awaiting site survey --
        # Derived from the step outstanding") says nothing a document says;
        # on 010038 two such lines became "a site survey is pending" on a job
        # already done. Labeled co_action ignore / parser_derived.
        if _derived(a):
            left_out.append({"id": aid, "reason": "parser_derived", "by": "rule:derived"})
            why["rule:derived"] += 1
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
        # Our technician's work is never the customer's responsibility. On
        # 010038 the section head, unsure (0.42-0.56), filed "Remove the TVs
        # and mounts from the OLD site" there while the owner head said
        # technician at 0.80-0.94; on the labeled deal the owner head said
        # technician at >= 0.75 on 82 of 94 scope lines and on no customer line.
        if sec == "customer_responsibilities" and have_reads:
            own, po = r(aid, "read:task_owner_role")
            if own == "technician" and po >= OWNER_BAR:
                sec, sec_by = "scope_of_work", "base:read:task_owner_role"
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
        # Trade questions only for a service someone is sure of: the head at
        # SERVICE_BAR, or the parse's own pack. 010043 read imac at 0.52 (the
        # labeler said security_camera) and was asked IMAC questions.
        svc_value = str(service.get("value") or "")
        if service.get("by", "").startswith(("base:", "purtera:")) and float(service.get("p") or 0) < SERVICE_BAR:
            svc_value = ""
            service["unsure"] = True
        per_family: Counter = Counter()
        bar = (float(os.environ["SOWSMITH_C3_QUESTION_BAR"]) if os.environ.get("SOWSMITH_C3_QUESTION_BAR")
               else readings.get("question_bar") or QUESTION_CATALOG_BAR)
        for q in sorted(asked, key=lambda x: -float(x.get("p_valid") or 0)):
            p_valid = float(q.get("p_valid") or 0)
            if p_valid < bar:
                break
            m = meta.get(q["text"]) or {}
            if m.get("applies_to") and svc_value not in m["applies_to"]:
                continue
            fam = m.get("family", "")
            if QUESTIONS_PER_FAMILY and per_family[fam] >= QUESTIONS_PER_FAMILY:
                continue
            per_family[fam] += 1
            questions.append({"id": "q:" + str(m.get("id") or len(questions)), "text": m.get("display") or q["text"],
                              "asked_as": q["text"],
                              "family": m.get("family", ""), "layer": m.get("layer", "base"),
                              "p": round(p_valid, 3), "by": _by(QUESTION_KEY), "owner": "project_manager"})
    _t_ev = time.monotonic()
    _attach_evidence([q for q in questions if q["id"].startswith("q:")], atoms)
    evidence_ms = round((time.monotonic() - _t_ev) * 1000, 1)
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
        "evidence_ms": evidence_ms,
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
    "Write a short project manager's brief from these facts: a summary of at most 3 sentences, and at most 3 "
    "short sentences for each of at most 6 sections (the ones a PM reads first: scope, customer "
    "responsibilities, out of scope, schedule, fees, terms) (the facts themselves are listed under the brief, so do not restate every "
    "one). Every sentence must cite the [ids] of the facts it uses, and say nothing that is not in a fact. "
    "Return JSON only: {\"summary\": str, \"sections\": "
    "{section: [{\"text\": str, \"cites\": [id]}]}, \"open_questions\": [{\"question\": str, \"owner\": str, "
    "\"cites\": [id]}]}.\n\nSERVICE: %s\nFACTS BY SECTION: %s\nOPEN QUESTIONS: %s\nCONFLICTS: %s"
)

_ASK_PROMPT = (
    "Which of these open questions about a field-services job do the given deal lines already ANSWER? A line "
    "answers a question only when it states the specific answer (a date, a number, a name, a yes or no, the "
    "term itself). A line that only mentions the topic, a table heading, or a line saying it will be decided "
    "later does NOT answer it. A contract's effective date is not a start or go-live date, and a payment "
    "milestone is not a schedule. Return JSON only, listing ONLY the answered ones: "
    "{\"answered\": [{\"id\": question id, \"by\": line id}]}. Return {\"answered\": []} if none.\n"
    "SERVICE: %s\nQUESTIONS: %s"
)
#: Questions per answered-check call. At 40, one 010038 call answered 9 and
#: the same call again answered none; at 12 it answered 6-8 every time.
ASK_BATCH = int(os.environ.get("SOWSMITH_C3_BRIEF_ASK_BATCH", "12"))
#: The prose answer's token cap. 2,000 cut off 010043's brief (it runs ~1,800).
PROSE_TOKENS = int(os.environ.get("SOWSMITH_C3_BRIEF_PROSE_TOKENS", "3500"))


def _json_of(raw: str) -> dict[str, Any] | None:
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _salvage(raw: str) -> dict[str, Any] | None:
    """A prose answer cut off at the token cap, closed after its last complete
    sentence object. The answer runs ~1,800 tokens on a big deal (010043:
    6,200-6,800 chars), at the old 2,000 cap, and one cut-off answer used to
    lose the whole brief."""
    start = (raw or "").find("{")
    if start < 0:
        return None
    stack: list[str] = []
    in_str = esc = False
    best: tuple[int, list[str]] | None = None
    for i in range(start, len(raw)):
        ch = raw[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()
            if ch == "}" and len(stack) >= 2:
                best = (i + 1, list(stack))
    if best is None:
        return None
    end, open_ = best
    try:
        return json.loads(raw[start:end] + "".join(reversed(open_)))
    except Exception:
        return None


def _in_failed(q: dict[str, Any], batches: list[list[dict[str, Any]]], raws: list[str]) -> bool:
    """Whether this question's batch came back unusable (it then counts as
    unchecked, never dropped)."""
    for b, raw in zip(batches, raws):
        if any(x["id"] == q["id"] for x in b):
            return _json_of(raw) is None
    return False


def _complete(prompt: str, model: str | None, max_tokens: int) -> str:
    """One call, retried once when it comes back empty or unparseable
    (llm_client returns "" on any failure, a throttled burst included: one of
    010043's two question batches came back empty and its 32 went unchecked)."""
    from app.core import llm_client

    for attempt in range(2):
        raw = llm_client.complete(prompt, max_tokens=max_tokens,
                                  model=model or os.environ.get("SOWSMITH_C3_BRIEF_MODEL") or None,
                                  timeout=int(os.environ.get("SOWSMITH_C3_BRIEF_TIMEOUT", "60")))
        if _json_of(raw) is not None:
            return raw
        if attempt == 0:
            time.sleep(1.0)
    return raw


def write(brief: dict[str, Any], *, model: str | None = None) -> dict[str, Any]:
    """The brief's prose and the question check, as parallel LLM calls: one
    for the prose, one per batch of catalog questions. Sentences whose
    citations are not facts of this brief are dropped and counted; a question
    a batch did not mark answered (citing a line that answers it) stays asked,
    so a failed or partial call loses nothing."""
    from concurrent.futures import ThreadPoolExecutor

    facts = {s: [f"[{x['id']}] {x['text']}" for x in v[:MAX_FACTS_PER_SECTION]]
             for s, v in brief["sections"].items()}
    cs = [x["ids"] for x in brief["conflicts"][:15]]
    known = {x["id"] for v in brief["sections"].values() for x in v} | {q["id"] for q in brief["questions"]}
    catalog = [q for q in brief["questions"] if q["id"].startswith("q:")]
    qs = [f"[{q['id']}] (owner: {q['owner']}) {q['text']}" for q in brief["questions"]
          if not q["id"].startswith("q:")][:30]
    lines_seen = {e["id"] for q in catalog for e in q.get("evidence") or []}
    service = brief["service"].get("value")
    batches = [catalog[i:i + ASK_BATCH] for i in range(0, len(catalog), ASK_BATCH)]
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=1 + len(batches)) as pool:
        prose_f = pool.submit(_complete, _PROMPT % (service, json.dumps(facts), json.dumps(qs), json.dumps(cs)),
                              model, PROSE_TOKENS)
        ask_f = [pool.submit(_complete, _ASK_PROMPT % (service, json.dumps(
            [{"id": q["id"], "q": q["text"],
              "lines": [f"[{e['id']}] {e['text']}" for e in q.get("evidence") or []]} for q in b])),
            model, 600) for b in batches]
        prose_raw = prose_f.result()
        ask_raw = [f.result() for f in ask_f]
    out: dict[str, Any] = {"write_ms": round((time.monotonic() - started) * 1000, 1), "calls": 1 + len(batches)}
    doc = _json_of(prose_raw)
    dropped = 0
    if doc is None:
        doc = _salvage(prose_raw)
        if doc is not None:
            out["salvaged"] = True
    if doc is None:
        out["error"] = "the prose call returned no usable JSON"
        doc = {}
    for sec, sentences in list((doc.get("sections") or {}).items()):
        keep = []
        for s in sentences or []:
            cites = [str(c).strip("[]") for c in (s.get("cites") or [])] if isinstance(s, dict) else []
            if cites and all(c in known for c in cites):
                keep.append({"text": s.get("text", ""), "cites": cites})
            else:
                dropped += 1
        doc["sections"][sec] = keep
    by_qid = {q["id"]: q for q in catalog}
    asked, answered, failed = [], [], 0

    def ok(ids):
        return [str(i).strip("[]") for i in ids or [] if str(i).strip("[]") in lines_seen]
    for raw in ask_raw:
        got = _json_of(raw)
        if got is None:
            failed += 1
            continue
        for c in got.get("answered") or []:
            q = by_qid.get(str(c.get("id")))
            by = ok([c.get("by")] if not isinstance(c.get("by"), list) else c.get("by"))
            if q is not None and by and q["id"] not in {a["id"] for a in answered}:
                answered.append({"id": q["id"], "question": q["text"], "answered_by": by})
    done = {a["id"] for a in answered}
    for q in catalog:
        if q["id"] not in done and not (failed and _in_failed(q, batches, ask_raw)):
            asked.append({"id": q["id"], "question": q["text"], "family": q.get("family"),
                          "owner": q["owner"], "p": q.get("p"),
                          "cites": [e["id"] for e in (q.get("evidence") or [])[:1]]})
    checked = {a["id"] for a in asked} | {a["id"] for a in answered}
    unchecked = [q for q in catalog if q["id"] not in checked]
    for q in unchecked:
        asked.append({"id": q["id"], "question": q["text"], "family": q.get("family"), "owner": q["owner"],
                      "p": q.get("p"), "cites": [], "unchecked": True})
    asked.sort(key=lambda a: -(a.get("p") or 0))
    out.update(summary=doc.get("summary", ""), sections=doc.get("sections") or {},
               open_questions=doc.get("open_questions") or [], uncited_dropped=dropped,
               catalog_questions=asked, answered_in_deal=answered, unchecked=len(unchecked),
               failed_ask_calls=failed)
    return out


def envelope_questions(brief: dict[str, Any]) -> list[dict[str, Any]]:
    """The base's catalog questions as the envelope's open questions (kind
    ``catalog``): the ones to ask (best first) and the ones the deal already
    answers (review_status ``answered``, with the line). ``query`` is the
    wording the Questions head reads; question_fit trains on it."""
    by_id = {q["id"]: q for q in brief.get("questions") or [] if str(q.get("id", "")).startswith("q:")}
    w = brief.get("written") or {}
    asked = w.get("catalog_questions")
    if asked is None:
        asked = [{"id": q["id"], "question": q["text"], "p": q.get("p")} for q in by_id.values()]
    out = []
    for a in asked:
        q = by_id.get(a["id"]) or {}
        cites = a.get("cites") or []
        # No atom_id: the labeling API keys a card by its line when it has one,
        # and catalog questions often share their closest line, so they would
        # fold into one card. Keyed by text instead; the lines go in `cites`.
        out.append({"atom_id": None, "cites": cites, "artifact_id": None, "text": a["question"],
                    "query": q.get("asked_as") or q.get("text") or a["question"],
                    "review_status": "needs_review", "kind": "catalog", "question_id": a["id"],
                    "family": q.get("family"), "layer": q.get("layer", "base"), "p": a.get("p"),
                    "decided_by": q.get("by", "base:jdg:gap")})
    for a in w.get("answered_in_deal") or []:
        q = by_id.get(a["id"]) or {}
        out.append({"atom_id": None, "cites": a.get("answered_by") or [], "artifact_id": None, "text": a["question"],
                    "query": q.get("asked_as") or a["question"], "review_status": "answered",
                    "kind": "catalog", "question_id": a["id"], "family": q.get("family"),
                    "layer": q.get("layer", "base"), "p": q.get("p"), "answered_by": a.get("answered_by"),
                    "decided_by": q.get("by", "base:jdg:gap")})
    return out


#: Sections a later SOW restates in full, so an earlier version's clause is
#: not a second fact but an old one.
CONTRACT_SECTIONS = frozenset({"scope_of_work", "customer_responsibilities", "provider_responsibilities",
                               "out_of_scope", "assumptions", "fees", "terms", "change_control",
                               "acceptance", "deliverables", "timeline"})
_SOW_NAME = re.compile(r"\bsow\b|statement of work|ready for signature|fully executed|\bsigned\b", re.I)


def _sow_like(doc: dict[str, Any]) -> bool:
    kind = str(doc.get("kind") or "").lower()
    return kind in ("sow_draft", "signed_sow", "sow", "proposal") or (
        kind in ("file", "") and bool(_SOW_NAME.search(str(doc.get("filename") or ""))))


def _drop_superseded(brief: dict[str, Any], atoms: list[Any], documents: list[dict[str, Any]]) -> int:
    """Keep only the latest SOW's clauses in the contract sections. On 010038
    the brief was handed the June draft ($7,890, travel billable), the July
    change order ($1,260) and the signed SOW ($9,150, no travel) as equals and
    wrote "conflicting amounts exist" and "travel is billed". Documents are in
    arrival order (c3_type_base.deal_documents); every SOW version and change
    order before the latest SOW is superseded. Their lines are moved to
    ``superseded``, not dropped, so a PM can see what changed. Returns lines moved."""
    order = [d for d in documents if isinstance(d, dict)]

    def contract(d: dict[str, Any]) -> bool:
        return _sow_like(d) or str(d.get("kind") or "").lower() == "change_order"

    by_doc: dict[str, list[Any]] = {}
    for a in atoms:
        by_doc.setdefault(str(getattr(a, "artifact_id", "") or ""), []).append(a)
    header: dict[int, str] = {}
    for i, d in enumerate(order):
        if contract(d) and not d.get("authored_at"):
            try:
                from app.core.orbitbrief_envelope import _document_header_date

                header[i] = str(_document_header_date(by_doc.get(str(d.get("doc_id")), [])) or "")
            except Exception:
                header[i] = ""

    def when(i: int) -> tuple:
        # The date the document says it was written (its record, else its own
        # header, as the envelope reads it), else its arrival. On 010038 the
        # signed SOW was attached to nothing, so arrival put it before the July
        # change order it folds in.
        d = order[i]
        return (str(d.get("authored_at") or header.get(i) or d.get("sent_at") or ""), i)

    sows = sorted((i for i, d in enumerate(order) if _sow_like(d)), key=when)
    if not sows:
        return 0
    latest = sows[-1]
    old = {str(order[i].get("doc_id")) for i, d in enumerate(order)
           if i != latest and contract(d) and when(i) < when(latest)}
    if not old:
        return 0
    doc_of = {str(a.id): str(getattr(a, "artifact_id", "") or "") for a in atoms}
    latest_id = str(order[latest].get("doc_id"))
    moved = []
    for sec in list(brief.get("sections") or {}):
        items = brief["sections"][sec]
        # Only where the latest SOW speaks to this section itself; else the
        # older version is all the deal has on it, and it stays.
        has_latest = any(doc_of.get(str(it.get("id"))) == latest_id for it in items)
        keep = []
        for it in items:
            iid = str(it.get("id"))
            # "Documents disagree: $7,890 vs $9,150" between a superseded
            # version and the latest is not a conflict, wherever it is filed.
            stale_conflict = iid.startswith("xdoc_conflict") and doc_of.get(iid) in old
            if stale_conflict or (sec in CONTRACT_SECTIONS and has_latest and doc_of.get(iid) in old):
                moved.append({**it, "section": sec, "superseded_by": str(order[latest].get("filename") or "")})
            else:
                keep.append(it)
        brief["sections"][sec] = keep
    brief["superseded"] = moved
    # Every reader of the lines, not only the brief: the Deal Kit and the SOW
    # read atoms directly and saw the draft's $7,890 beside the signed $9,150.
    # The line stays; it says which document replaced it.
    by = str(order[latest].get("filename") or order[latest].get("doc_id") or "")
    for a in atoms:
        if str(getattr(a, "artifact_id", "") or "") in old:
            v = getattr(a, "value", None)
            if isinstance(v, dict):
                v["superseded_by"] = by
            else:
                try:
                    a.value = {"superseded_by": by}
                except Exception:
                    pass
    brief["superseded_documents"] = sorted(old)
    return len(moved)


def run(atoms: list[Any], packets: list[Any] = (), *, service_fallback: str | None = None) -> dict[str, Any]:
    """Build (and, when switched on, write) the brief from this compile's read."""
    from app.core import c3_type_base

    brief = build(atoms, packets, service_fallback=service_fallback, readings=c3_type_base.full_read())
    try:
        _drop_superseded(brief, atoms, c3_type_base._DEAL.get("documents") or [])
    except Exception as exc:  # pragma: no cover - never lose the brief over this
        log.info("C3 brief superseded check skipped: %s", type(exc).__name__)
    # The base's line type for every line that names a site, for the roster
    # filter (site_identity.filter_roster): a site only ever mentioned in lines
    # the base does not read as a site is a mention, not a job site.
    _types = c3_type_base._DEAL.get("types") or {}
    brief["site_line_types"] = {
        str(a.id): _types[str(a.id)] for a in atoms
        if str(a.id) in _types and any(str(k).startswith("site:") for k in (getattr(a, "entity_keys", None) or []))}
    brief["contacts"] = contacts(atoms)
    if writing_enabled():
        try:
            brief["written"] = write(brief)
        except Exception as exc:
            brief["written"] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
        w = brief.get("written")
        if isinstance(w, dict) and isinstance(w.get("sections"), dict):
            details = job_details(atoms)
            brief["job_details"] = details
            if details:
                # Right after scope: the lines a PM prices from.
                secs = dict(w["sections"])
                ordered = {}
                for k, v in secs.items():
                    ordered[k] = v
                    if k in ("scope", "scope_of_work"):
                        ordered["job_details"] = details
                if "job_details" not in ordered:
                    ordered = {"job_details": details, **ordered}
                w["sections"] = ordered
            if brief["contacts"]:
                w["sections"]["contacts"] = [{"text": c["line"], "cites": c["cites"]} for c in brief["contacts"]]
    return brief


#: Line types a PM prices from. The written brief is three sentences a
#: section, so on the graded pre-quote deals it dropped what drives the price:
#: 010043's cable runs (154 / 88 / 146 ft), camera models and brick mounting;
#: 010038's ceiling, backboard and access window. These are the deal's own
#: lines, cited, no LLM wording.
DETAIL_TYPES = ("quantity", "bom_line", "vendor_line_item", "site_attribute", "site_infrastructure",
                "site_access_window", "site_access_restriction", "site_implementation_note", "deadline",
                "lead_time_constraint", "blackout_date_range", "material", "site_room_mix")
MAX_DETAILS = 15
#: Lines that are never a pricing detail, whatever they mention.
_NOT_DETAIL = frozenset({"small_talk", "stakeholder", "deal_metadata", "deal_state", "entity", "list_header",
                         "raw_utterance", "signatory", "approval_authority", "answered_question",
                         # Questions are asked in the questions list; terms are fees.
                         "open_question", "internal_question", "payment_term", "commercial_total",
                         "contract_term", "pricing_assumption"})
_DETAIL_SIGNAL = re.compile(
    r"\b\d+(\.\d+)?\s*(ft|feet|foot|'|in|inch|u|units?|x|qty|aps?|tvs?|cameras?|drops?|runs?|sites?|days?|hours?)\b"
    r"|\bqty\b|\bmodel\b|\b(ceiling|vaulted|brick|concrete|drop[- ]tile|lift|ladder|scissor|rack|backboard|"
    r"plywood|mount(s|ing)?|conduit|idf|mdf|poe|outlet|power|access|window|after[- ]hours|escort|badge|"
    r"elevator|loading dock|move[- ]in|deadline|asap|urgent|week of)\b", re.I)
_VENDOR_TOTAL = re.compile(r"\b(sub ?total|grand total|sales tax|tax|shipping|lease|fmv)\b", re.I)
_PLACEHOLDER_LINE = re.compile(r"^\[(image|drawing|figure|table)[^\]]*\]", re.I)


def job_details(atoms: list[Any]) -> list[dict[str, Any]]:
    """The concrete lines a PM prices from: quantities, equipment and models,
    lengths, site conditions, access windows and deadlines, in document order,
    one per distinct line, each citing itself. Chosen by what the line says,
    not its type: on 010038 the ceiling, access window and rack lines were
    typed scope_item / dependency."""
    first, rest, seen = [], [], set()
    for a in atoms:
        t = _type(a)
        if t in _NOT_DETAIL:
            continue
        v = a.value if isinstance(getattr(a, "value", None), dict) else {}
        if v.get("superseded_by") or _deal_kit(a):
            continue
        text = re.sub(r"\s+", " ", (getattr(a, "raw_text", "") or "")).strip()
        if len(text) < 6 or len(text) > 300 or _PLACEHOLDER_LINE.match(text) or _VENDOR_TOTAL.search(text) \
                or text.rstrip().endswith("?"):
            continue
        if not (_DETAIL_SIGNAL.search(text) or t in ("bom_line", "vendor_line_item")):
            continue
        key = re.sub(r"\W+", " ", text.lower()).strip()[:120]
        if key in seen:
            continue
        seen.add(key)
        (first if t in DETAIL_TYPES else rest).append({"text": text, "cites": [str(a.id)]})
    return (first + rest)[:MAX_DETAILS]


_NOT_A_NAME = re.compile(r"^(hi|hey|hello|dear|thanks|thank you|regards)\b|\d|@|\b(site|facility|office|"
                         r"suite|street|ave|inc|llc|corp|group|services)\b", re.I)
_PHONE = re.compile(r"(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}")
MAX_CONTACTS = 10
#: A job title read as a name ("Seller Representative", "Client Executive").
_TITLE_WORD = re.compile(r"\b(representative|executive|manager|director|president|officer|buyer|seller|"
                         r"engineer|technician|coordinator|administrator|sales|account|client|customer|team)\b", re.I)
#: A greeting ("Trent-") or a signature's link text read as a role.
_JUNK_ROLE = re.compile(r"linkedin|^\w+\s*-$|^\W*$", re.I)
#: A real address, not an inline image's content id ("image002.png@01dd...").
_EMAIL = re.compile(r"(?!image\d)[\w.+-]+@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")


def contacts(atoms: list[Any]) -> list[dict[str, Any]]:
    """The deal's people, from the lines read as stakeholders: a real name with
    a role, email or phone, one entry per person, most-mentioned first. No LLM.
    The graders found every name the brief left out (Rob Cole, Rafael
    Fernandez ...) already on stakeholder lines; the writer just dropped them.
    A greeting ("Hi Tanner"), a place or a date is not a person."""
    people: dict[str, dict[str, Any]] = {}
    for a in atoms:
        if _type(a) != "stakeholder":
            continue
        v = a.value if isinstance(getattr(a, "value", None), dict) else {}
        if v.get("superseded_by") or str(v.get("role") or "").lower() == "filler":
            continue
        name = re.sub(r"\s+", " ", str(v.get("name") or "")).strip(" ,;:|")
        if not name or len(name.split()) < 2 or len(name) > 40 or _NOT_A_NAME.search(name):
            continue
        if _TITLE_WORD.search(name):
            continue
        role = str(v.get("role") or v.get("title") or "").strip()
        if role.lower() in ("stakeholder", "contact") or _NOT_A_NAME.match(role) or _JUNK_ROLE.search(role):
            role = ""
        email = str(v.get("email") or "").strip()
        if not _EMAIL.fullmatch(email):
            email = ""
        phone = str(v.get("phone") or "").strip() or ((_PHONE.search(a.raw_text or "") or [None])[0] or "")
        if not (role or email or phone):
            continue
        p = people.setdefault(name.lower(), {"name": name, "role": "", "email": "", "phone": "", "n": 0, "cites": []})
        p["n"] += 1
        p["role"] = p["role"] or role
        p["email"] = p["email"] or email
        p["phone"] = p["phone"] or phone
        if len(p["cites"]) < 2:
            p["cites"].append(str(a.id))
    out = sorted(people.values(), key=lambda p: -p["n"])[:MAX_CONTACTS]
    for p in out:
        bits = [x for x in (p["role"], p["email"], p["phone"]) if x]
        p["line"] = p["name"] + (" (" + ", ".join(bits) + ")" if bits else "")
    return out
