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
#: Line types the base reads that are never brief material, whatever co_action
#: says: a question the deal already answered, a heading. On the graded
#: pre-quote deals the base read "Do you have the model of the camera?" as
#: answered_question at 0.9985 and "Site Contacts:" as list_header at 0.997,
#: and the brief still filed them as a customer responsibility and overview.
NOT_BRIEF_TYPES = frozenset({"answered_question", "list_header", "internal_question"})
BASE_TYPE_BAR = 0.5
#: The supplier head (col:supplier) at or above this says who furnishes a
#: priced line. A document whose priced lines the customer furnishes is the
#: customer's own order (010043: the CDW order confirmation for the cameras the
#: customer bought), not this job's fees.
SUPPLIER_BAR = 0.6
#: The head that decides whether a line is in the brief at all. col:admission
#: (test A, 2026-10-28: held-out 0.726 vs live 0.530) answers "is this a fact
#: of the deal"; read:co_action keeps only its company-action meaning (we
#: drop / ignore a real fact). Before 10-28 co_action alone decided, and it
#: was never taught that chatter is not kept: 412 parser-rejected chatter lines
#: came back on 010394. SOWSMITH_C3_KEEP_HEAD=read:co_action restores that.
KEEP_HEAD = os.environ.get("SOWSMITH_C3_KEEP_HEAD", "col:admission").strip() or "col:admission"
#: The type head's small_talk at or above this is final: no other head keeps the line.
SMALL_TALK_BAR = float(os.environ.get("SOWSMITH_C3_SMALL_TALK_BAR", "0.8"))
#: Admission keep at or above this goes in the body, between MAYBE_BAR and it "also mentioned".
#: Its keeps of the audit's labeled real lines sat at 0.53-0.98 (most 0.7-0.93) on 2026-10-28;
#: KEEP_BAR (0.9) was calibrated for co_action and sent most of them to "also mentioned".
ADMISSION_BODY_BAR = float(os.environ.get("SOWSMITH_C3_ADMISSION_BODY_BAR", "0.7"))
#: co_action reject / ignore at or above this drops a line admission kept (a company action).
COMPANY_DROP_BAR = 0.5
#: Lines of a NEVER type (raw_utterance, small_talk, entity ...): the keep head decides first, the
#: type rule only where it abstains. SOWSMITH_C3_NEVER_ADMISSION=off restores rule-first.
#: Held-out test A (7 deals, 311 labeled NEVER-type lines, 2026-10-28): admission drop at >= 0.5
#: is right 0.929, the rule's drop on the same lines the same 0.929 (both say drop), so its drops
#: change no outcome. Its keeps lose to the rule below 0.95 (0.37 vs 0.63 at >= 0.5) and win only
#: at >= 0.99 (10 of 11 right vs 1 of 11, 8 of them one deal, drafted gold): keep is off by default
#: (SOWSMITH_C3_NEVER_KEEP_BAR 1.01) until more labels back it.
NEVER_ADMISSION = os.environ.get("SOWSMITH_C3_NEVER_ADMISSION", "on").strip().lower() not in (
    "0", "false", "no", "off")
NEVER_DROP_BAR = float(os.environ.get("SOWSMITH_C3_NEVER_DROP_BAR", "0.5"))
NEVER_KEEP_BAR = float(os.environ.get("SOWSMITH_C3_NEVER_KEEP_BAR", "1.01"))
MAX_FACTS_PER_SECTION = 25
#: "Also mentioned" lines carried with their text (the rest are listed by id).
ALSO_MENTIONED_SHOWN = int(os.environ.get("SOWSMITH_C3_ALSO_MENTIONED_SHOWN", "300"))
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
_PLACEHOLDER = re.compile(r"\s*\[(image|figure|table|chart|drawing|embedded object) (awaiting|read:|skipped:)", re.I)
QUESTION_WANTS = ("chase-artifact", "chase-conversation", "confirm-with-customer", "decide-internally")


#: Lines a vision model read off a picture (its description, a caption, OCR'd
#: logo text): what the parser made of a picture, not a line a document has.
_VISION_VIA = frozenset({"pdf_image_vision", "linked_picture_vision"})
_VISION_METHODS = ("pdf_image_vision", "docx_image_vision", "linked_picture_vision")


def _vision(atom: Any) -> bool:
    v = getattr(atom, "value", None)
    if isinstance(v, dict) and (v.get("via") in _VISION_VIA or v.get("kind") == "image_fact"):
        return True
    for ref in getattr(atom, "source_refs", None) or []:
        m = getattr(ref, "extraction_method", None) or (ref.get("extraction_method") if isinstance(ref, dict) else "")
        if str(m or "").startswith(_VISION_METHODS):
            return True
    return False


#: SOWSMITH_C3_ADMISSIBILITY_GATE (default on): lines from a document that arrived after the deal was
#: decided lost / archived (deal_stage admissible_for "neither") are left out of the brief, visibly.
def admissibility_gate() -> bool:
    return os.environ.get("SOWSMITH_C3_ADMISSIBILITY_GATE", "1").strip().lower() not in ("0", "false", "no", "off")


def document_gate() -> list[dict[str, Any]]:
    """Every document of the deal with what decides whether it feeds the brief and the quote: the
    document-job verdict (and who made it: a PM's lesson, the C3 head, the model judge, or nothing),
    any rule that kept a document the head called another job, and its admissibility at the deal's
    stage when it arrived (``deal_stage``, the CRM stage timeline). ``feeds`` is the outcome."""
    try:
        from app.core.c3_type_base import _DEAL
        from app.core.document_lifecycle import deal_stage as _ds
    except Exception:  # pragma: no cover
        return []
    verdict_of: dict[str, dict[str, Any]] = {}
    for v in _DEAL.get("document_job") or []:
        for fn in v.get("filenames") or []:
            verdict_of[str(fn)] = v
    timeline = _DEAL.get("stage_timeline")
    out = []
    for d in _DEAL.get("documents") or []:
        if not isinstance(d, dict):
            continue
        fn = str(d.get("filename") or "")
        v = verdict_of.get(fn) or {}
        stage = _ds.annotate(d.get("lifecycle"), authored_at=d.get("authored_at"), direction=d.get("direction"),
                             timeline=timeline) if timeline else {}
        spared = [k for k in ("drawing_spared", "content_spared", "share_rule_spared") if v.get(k)]
        job = None
        if v:
            src = str(v.get("source") or "")
            # This document's own reading (the conversation's verdict is its documents' worst case:
            # one other_job reading used to show on every document of the thread, 010340's 0.896 x 7).
            own = (v.get("c3_docs") or {}).get(fn)
            job = {"verdict": v.get("verdict"),
                   "by": ("pm_lesson" if src == "store" else "head:jdg:document_job" if src == "c3"
                          else "head:jdg:document_job (below its bar)" if src == "c3_below_bar"
                          else "model_judge" if v.get("model_verdict") and src not in ("fallback", "") else "none"),
                   "head": list(own) if own else ([v.get("c3_verdict"), v.get("c3_confidence")]
                                                   if v.get("c3_verdict") else None),
                   "conversation_head": [v.get("c3_verdict"), v.get("c3_confidence")] if v.get("c3_verdict") else None,
                   "bar": v.get("c3_bar"),
                   "confidence": v.get("confidence"), "kept_by_rule": spared or None, "reason": v.get("reason") or None}
        after = admissibility_gate() and stage.get("admissible_for") == "neither"
        # The deal record's documents (c3_type_base.deal_documents) carry their artifact id as ``doc_id``;
        # read as "artifact_id" alone every row had "" and ``build`` barred nothing (wave 4: 010222's
        # 11 after-decision documents and 000083's 3 still fed the brief and the quote).
        out.append({"artifact_id": str(d.get("artifact_id") or d.get("doc_id") or ""), "filename": fn, "document_job": job,
                    "deal_stage": {"stage_at_arrival": stage.get("stage_at_arrival"),
                                   "admissible_for": stage.get("admissible_for"), "why": stage.get("why")} if stage else None,
                    "feeds": not ((job or {}).get("verdict") == "other_job" or after),
                    "why_not": ("document_job other_job" if (job or {}).get("verdict") == "other_job"
                                else f"deal_stage: {stage.get('why')}" if after else None)})
    return out


def _derived(atom: Any) -> bool:
    """A line the parser made up from other lines (or read off a picture), not one a document has.
    Picture readings go through this guard like every other parser-made line (010309: 24 readings of a
    vendor's install guide, "The image shows ...", were kept in scope and out of scope); the brief
    hands them to the writer as figures instead."""
    if _vision(atom):
        return True
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


def type_by_head(atom: Any) -> bool:
    """The line's current type is the type head's: typing labelled it
    (``c3_label``, act keep or the same type) or the base confirmed the
    parser's type (``c3_confirm``). The same test ``c3_share.line_types``
    counts as the head's."""
    v = getattr(atom, "value", None)
    if not isinstance(v, dict):
        return False
    t = _type(atom)
    lab = v.get("c3_label")
    if isinstance(lab, dict) and lab.get("act"):
        return lab["act"] == "keep" or not lab.get("type") or lab.get("type") == t
    conf = v.get("c3_confirm")
    return isinstance(conf, dict) and conf.get("type") == t


def _deal_kit(atom: Any) -> bool:
    try:
        from app.core.c3_type_base import _DEAL, skipped

        return skipped(atom, _DEAL.get("sha"))
    except Exception:
        return False


def _base_type(atom_id: str) -> tuple[str, float]:
    """The base's own type reading of a line (its one read of the deal)."""
    try:
        from app.core.c3_type_base import _DEAL

        v = (_DEAL.get("types") or {}).get(atom_id)
    except Exception:
        v = None
    return (str(v[0]), float(v[1] or 0.0)) if v and v[0] is not None else ("_absent", 0.0)


def _when_of(atom: Any) -> str:
    """The day the line's message or note was written (YYYY-MM-DD), or ''."""
    try:
        from datetime import datetime, timezone

        from app.core.c3_type_base import _own_time

        t = _own_time(atom)
        return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d") if t else ""
    except Exception:
        return ""


def _heading_of(atom: Any) -> str:
    """The document heading a line sits under ("What We Ask YOU Provide"): a
    bare "Ethernet cabling" says nothing about who supplies it without it."""
    try:
        from app.core.c3_type_base import _section

        path = _section(atom)
    except Exception:
        return ""
    return path.split(" > ")[-1][:80] if path else ""


def _not_brief(atom: Any) -> str:
    """The type that keeps a line out of the brief, or ''."""
    bt, pb = _base_type(str(atom.id))
    if bt in NOT_BRIEF_TYPES and pb >= BASE_TYPE_BAR:
        return bt
    return _type(atom) if _type(atom) in NOT_BRIEF_TYPES else ""


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


def _answer(v: Any) -> tuple[str, float]:
    """``(answer, p)`` from one read entry. A set-valued head (read:needed_by) is served as
    ``[[answers over 0.5], {answer: p}]``: its surest answer, or ``_absent`` when none is over 0.5
    (the entry used to be read as the string "['project_manager']")."""
    if not v or v[0] is None:
        return ("_absent", 0.0)
    if isinstance(v[0], (list, tuple)):
        ps = v[1] if len(v) > 1 and isinstance(v[1], dict) else {}
        names = [str(x) for x in v[0]]
        if not names:
            return ("_absent", 0.0)
        best = max(names, key=lambda n: float(ps.get(n) or 0.0))
        return (best, float(ps.get(best) or 0.0))
    return (str(v[0]), float(v[1] or 0.0))


def _answered(reads: dict[str, dict[str, list]], links: list[dict[str, Any]], deal: dict[str, list]) -> dict[str, bool]:
    """Which brief heads this compile's read actually answered on at least one line (or the deal)."""
    got: set[str] = set()
    for heads in (reads or {}).values():
        got.update(k for k, v in (heads or {}).items() if v and v[0] is not None)
    out = {k: k in got for k in ("read:co_action", "read:co_reason", "read:noise_class", "read:sow_section",
                                 "col:wants", "read:intake_gap", "read:needed_by", "col:admission")}
    sv = (deal or {}).get("deal:primary_service")
    out["deal:primary_service"] = bool(sv) and sv[0] not in (None, "_absent")
    out["rel:contradicts"] = any(ln.get("relation") == "contradicts" for ln in links or [])
    return out


#: Brief parts no head answers today: always a rule (or the CRM), said so in ``decided_by.no_head``.
NO_HEAD_PARTS = {
    "job_details": "rule: job_details() picks the lines to list by type (no head chooses them)",
    "contacts": "rule: contacts() reads names, roles, emails and phones off signature and header lines",
    "deal": "CRM record (deal and account name)",
    "figures": "rule: priced lines by type",
}


def _part(key: str, used: bool, answered: bool, view: Any, have_reads: bool) -> dict[str, str]:
    """How one brief part was decided, and why a head did not decide it."""
    if used:
        return {"by": "head", "why": "trusted head answered"}
    if not have_reads:
        return {"by": "rule", "why": "no base read this compile"}
    try:
        from c3_record.view import heads as _view_heads, name as _view_name

        v = _view_name(view)
        served = bool(_view_heads(v, [key])) if v else True
    except Exception:
        v, served = "", True
    if not served:
        return {"by": "rule", "why": f"not served in the {v} view (c3_record/view.py masks it, server side)"}
    if not answered:
        return {"by": "rule", "why": ("no contradicts link in the read" if key.startswith("rel:")
                                      else "served, but no answer on any line")}
    try:
        from c3_record.heads import head as _head

        h = _head(key) or {}
    except Exception:
        h = {}
    return {"by": "rule", "why": f"registry status {h.get('status') or 'missing'} "
                                 f"(c3_record/heads.json holdout {h.get('holdout')})"}


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
        return _answer((reads.get(atom_id) or {}).get(head))

    use = {k: have_reads and _trusted(k) for k in (
        "read:co_action", "read:co_reason", "read:noise_class", "read:sow_section", "col:wants",
        "read:intake_gap", "read:needed_by")}
    use["deal:primary_service"] = bool(deal) and _trusted("deal:primary_service")
    use["rel:contradicts"] = bool(links) and _trusted("rel:contradicts")
    # A head the registry trusts still decides nothing when this read never
    # answered it: the planning view does not serve the SOW / quote heads
    # (read:sow_section, read:intake_gap, deal:*) and the quoting view not the
    # PM heads (read:needed_by, read:task_owner_role). Until 10-28 those parts
    # were counted as head-decided while every line fell to the rule.
    answered = _answered(reads, links, deal)
    for k in list(use):
        use[k] = use[k] and answered.get(k, False)
    # The keep head decides when this read answered it (the registry has no
    # labeled bar for it yet; test A measured it, see KEEP_HEAD).
    use["col:admission"] = have_reads and KEEP_HEAD == "col:admission" and answered.get("col:admission", False)
    parts = {k: _part(k, use[k], answered.get(k, False), readings.get("view"), have_reads) for k in use}

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

    kept_unsure: list[dict[str, Any]] = []
    from app.core.c3_type_base import out_of_quoting_view

    out_of_view = out_of_quoting_view()
    docs = document_gate()
    barred = {d["artifact_id"]: d["why_not"] for d in docs if not d["feeds"] and d["artifact_id"]}
    from app.core.c3_type_base import _doc_id

    for a in atoms:
        aid, t = str(a.id), _type(a)
        art = _doc_id(a)
        if art in barred:
            # The document does not feed this deal's brief or quote (another job, or arrived after the
            # deal was decided); its lines stay in the compile for the heads and are listed here.
            left_out.append({"id": aid, "reason": "after_decision" if barred[art].startswith("deal_stage")
                             else "other_job", "by": "rule:deal_stage" if barred[art].startswith("deal_stage")
                             else "document_job"})
            why["document:" + ("after_decision" if barred[art].startswith("deal_stage") else "other_job")] += 1
            continue
        status = str(getattr(getattr(a, "review_status", ""), "value", getattr(a, "review_status", "")))
        if _deal_kit(a):
            # The base is never shown a Deal Kit or estimating sheet (it is
            # what the kit model is trained to produce), so its reading of one
            # means nothing: 45 of 135 false keeps on 010043 were kit lines.
            left_out.append({"id": aid, "reason": "old_manual_deal_kit", "by": "rule:document"})
            why["rule:deal_kit"] += 1
            continue
        if aid in out_of_view:
            # A line the quoting read leaves out (our own paper, after the cut, a pricing workbook ...):
            # the view keeps it from the heads by design, so it never reaches the brief by the type rule.
            left_out.append({"id": aid, "reason": "out_of_view:" + out_of_view[aid], "by": "view:quoting"})
            why["view:out_of_view:" + out_of_view[aid]] += 1
            continue
        # A line the parser wrote itself ("stage: awaiting site survey --
        # Derived from the step outstanding") says nothing a document says;
        # on 010038 two such lines became "a site survey is pending" on a job
        # already done. Labeled co_action ignore / parser_derived.
        if _derived(a):
            pic = _vision(a)
            left_out.append({"id": aid, "reason": "picture_reading" if pic else "parser_derived",
                             "by": "rule:derived"})
            why["rule:picture" if pic else "rule:derived"] += 1
            continue
        act, p = r(aid, "read:co_action") if use["read:co_action"] else ("_absent", 0.0)
        adm, pa = r(aid, "col:admission") if use["col:admission"] else ("_absent", 0.0)
        # One head per decision. The type head's sure small_talk is the type
        # head's answer: no keep head brings the line back (010394 "You're on
        # mute, buddy." small_talk 0.99 was kept by co_action 0.957).
        bt, pb = _base_type(aid)
        if bt == "small_talk" and pb >= SMALL_TALK_BAR:
            left_out.append({"id": aid, "reason": "small_talk", "by": "base:col:label_type", "p": round(pb, 3)})
            why["type_head:small_talk"] += 1
            continue
        keep_act, keep_p, keep_key = (("keep" if adm == "keep" else "reject"), pa, "col:admission") \
            if adm in ("keep", "drop") else (act, p, "read:co_action")
        # small_talk the parser typed (the type head unsure) is a type, not a
        # keep decision: when the keep head keeps the line it stays. 010426's
        # only date, "the three weeks after Thanksgiving", read small_talk 0.80
        # / keep 0.64 and was dropped outright.
        never_admitted = False
        if NEVER_ADMISSION and status != "rejected" and t in NEVER and keep_key == "col:admission":
            if adm == "drop" and pa >= NEVER_DROP_BAR:
                left_out.append({"id": aid, "reason": "not_admitted", "by": "base:col:admission",
                                 "p": round(pa, 3)})
                why["admission:drop_never_type"] += 1
                continue
            never_admitted = adm == "keep" and pa >= NEVER_KEEP_BAR
        if status == "rejected" or (t in NEVER and not never_admitted
                                    and not (t == "small_talk" and keep_act == "keep" and keep_p >= MAYBE_BAR)):
            left_out.append({"id": aid, "reason": "rejected" if status == "rejected" else f"type {t}",
                             "by": "rule:type"})
            why["rule:never"] += 1
            continue
        nb = _not_brief(a)
        if nb:
            left_out.append({"id": aid, "reason": f"base type {nb}", "by": "base:type"})
            why["base:type"] += 1
            continue
        if keep_key == "col:admission":
            # Admission says it is a fact of the deal; co_action may still say
            # the company drops or ignores it (its company-action meaning).
            if keep_act == "keep" and act in ("reject", "ignore") and p >= COMPANY_DROP_BAR:
                reason, rby = _reason(aid, act)
                left_out.append({"id": aid, "reason": reason, "by": rby if rby != "rule" else _by("read:co_action"),
                                 "p": round(p, 3), "admission": round(pa, 3)})
                why["head:company_" + act] += 1
                continue
            if keep_act == "keep" and pa >= ADMISSION_BODY_BAR:
                kept.append((a, pa, "base:col:admission"))
                why["admission:keep"] += 1
            elif keep_act == "keep" and pa >= MAYBE_BAR:
                maybe.append((a, pa))
                why["admission:maybe"] += 1
            else:
                left_out.append({"id": aid, "reason": "not_admitted", "by": "base:col:admission",
                                 "p": round(pa, 3)})
                why["admission:drop"] += 1
                if keep_act == "keep":
                    kept_unsure.append({"id": aid, "text": a.raw_text[:400], "p": round(pa, 3)})
            if 0.3 <= pa < 0.7:
                label_tasks.append({"id": aid, "head": "col:admission", "base_says": adm, "p": round(pa, 3),
                                    "text": a.raw_text[:200]})
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
                # The head's own choice was keep, only below the bar: 010171's
                # "We would like to have the Logitech Sight installed in this
                # specific room" (keep 0.39). The brief body leaves it out; the
                # SOW writer sees it, marked unsure.
                if act == "keep":
                    kept_unsure.append({"id": aid, "text": a.raw_text[:400], "p": round(p, 3)})
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

    # What the keep decision kept (body or "also mentioned") and who decided it.
    kept_by_now = {str(a.id): by for a, _, by in kept}
    kept_by_now.update({str(a.id): ("base:col:admission" if use["col:admission"] and r(str(a.id), "col:admission")[0]
                                    in ("keep", "drop") else _by("read:co_action")) for a, _ in maybe})
    kept_ids_now = set(kept_by_now)
    no_keep_head = not (use["col:admission"] or use["read:co_action"])

    # ---- sections
    # jdg:sow_section (app/core/sow_rubric_judge.py): the placement rubric
    # applied per fact. shadow records its answer; on lets it place the fact.
    from app.core import sow_rubric_judge as _judge

    judge_mode = _judge.mode()
    judged: dict[str, Any] = {}
    if judge_mode != "off" and kept:
        svc0 = deal.get("deal:primary_service") if deal else None
        job = str((svc0 or [None])[0] or service_fallback or "")
        if job in ("_absent", "default_pack"):
            job = ""
        t0 = time.monotonic()
        judged = _judge.judge_many([{"id": str(a.id), "text": a.raw_text[:300], "job": job,
                                     "under": _heading_of(a)} for a, _, _ in kept])
        log.info("c3_brief: jdg:sow_section %s judged %d/%d facts in %.1fs", judge_mode, len(judged), len(kept),
                 time.monotonic() - t0)
    sections: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for a, p, by in kept:
        aid = str(a.id)
        v = judged.get(aid)
        if judge_mode == "on" and v is not None and v.section not in (None, "_absent"):
            sec, ps, sec_by = v.section, float(v.section_conf), "judge:jdg:sow_section"
        else:
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
        # The same for a task the section head filed as out of scope or an
        # assumption: "Remove the Printer from the OLD site" (010038) read
        # out_of_scope 0.54 and technician 0.84; it is our requested task.
        #
        # The owner head arbitrates between HEADS only. A section the rule
        # derived from a type the type head read (labelled or confirmed) is
        # the type head's placement: an exclusion read 0.96-0.99, a pricing
        # term 0.99 or a customer duty keeps its section whatever the owner
        # head says (010362, 010395, 010355, 010245 had them all moved to
        # scope_of_work: against a rule placement ``po > ps`` was ``po > 0``).
        # Where no head typed or placed a DUTY (customer_responsibilities),
        # the owner head may fill, sure (OWNER_BAR), and the item says so
        # (``section_fill``). An exclusion or an assumption is no duty: who
        # "does" "Lifts are excluded" is not a question (010355: technician
        # 0.82 on two parser-typed ROM exclusions), so a rule placement there
        # is only ever moved against the section head.
        typed_by_head = type_by_head(a)
        placed_by_head = sec_by != "rule:type"
        fill = None
        if sec in ("out_of_scope", "assumptions", "customer_responsibilities") and have_reads \
                and (placed_by_head or (not typed_by_head and sec == "customer_responsibilities")):
            own, po = r(aid, "read:task_owner_role")
            if own == "technician":
                if placed_by_head:
                    # "Mount the Aruba APs" (010038) read technician 0.70,
                    # customer responsibility 0.65, and the brief gave our work
                    # to the customer.
                    move = po > ps or (sec == "customer_responsibilities" and po >= OWNER_BAR)
                else:
                    move = po >= OWNER_BAR
                    fill = "no head typed or placed this line; the owner head filled it" if move else None
                if move:
                    sec, sec_by = "scope_of_work", "base:read:task_owner_role"
        item = {"id": aid, "text": a.raw_text[:400], "type": _type(a), "when": _when_of(a),
                "under": _heading_of(a),
                "keep_p": None if p is None else round(p, 3), "by": by, "section_by": sec_by,
                "type_by": "head" if typed_by_head else "rule"}
        if fill:
            item["section_fill"] = fill
        if v is not None:
            item["judge"] = {"section": v.section, "owner": v.owner, "p": round(float(v.section_conf), 3)}
        sections[sec].append(item)
    _customer_furnished(sections, by_id, r if have_reads else None)
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
        if not use["deal:primary_service"]:
            # The pack rule names no service (default_pack), so the head's answer stands even
            # untrusted: it decided this part, unsure, and is counted as the head's.
            parts["deal:primary_service"] = {"by": "head", "why": "untrusted head answered; the rule names no "
                                             "service (default_pack) - " + parts["deal:primary_service"]["why"]}
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
            # Tampa?"): the line must also be one the keep head keeps.
            if (aid in kept_ids_now or no_keep_head) and not _deal_kit(a) and not _PLACEHOLDER.match(a.raw_text):
                hit = ("open_question", "rule:type+" + kept_by_now.get(aid, "rule:type"))
        # A question the base reads as already answered is not asked.
        if hit and _not_brief(a):
            hit = None
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
        if (ids[0] not in kept_ids_now and not no_keep_head) or _deal_kit(a) or _PLACEHOLDER.match(a.raw_text) or _not_brief(a):
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
    unsure_ids = {k["id"] for k in kept_unsure}
    brief = {
        "version": 2,
        "service": service,
        "sections": dict(sections),
        "also_mentioned": [{"id": str(a.id), "text": a.raw_text[:200], "keep_p": round(p, 3), "when": _when_of(a)}
                           for a, p in sorted(maybe, key=lambda x: -x[1])[:ALSO_MENTIONED_SHOWN]],
        # Nothing the keep head kept vanishes: the cap above only bounds the
        # text carried; every "also mentioned" line is counted and listed by id
        # (010152: 591 lines, 441 cut without a trace).
        "also_mentioned_total": len(maybe),
        "also_mentioned_more_ids": [str(a.id) for a, _ in sorted(maybe, key=lambda x: -x[1])[ALSO_MENTIONED_SHOWN:]],
        "questions": questions,
        "conflicts": conflicts[:40],
        "kept_unsure": sorted(kept_unsure, key=lambda x: -x["p"])[:120],
        # ids: the quote heads must not count off a line the heads left out (or an old Deal Kit line).
        "left_out": {"count": len(left_out), "by_reason": dict(Counter(x["reason"] for x in left_out).most_common(12)),
                     "ids": [str(x["id"]) for x in left_out],
                     # The lines a quoting read leaves out, by the view's reason
                     # (c3_type_base.out_of_quoting_view); ours_in_quoting_view: those of them ours.
                     "out_of_view": _by_view_reason(left_out),
                     "ours_in_quoting_view": _by_view_reason(left_out).get("ours", []),
                     # The lines a head (not a rule) left out, by id: the SOW
                     # may not bring them back (c3_sow.facts). A keep below
                     # the bar is not one of them (kept_unsure).
                     "head_dropped": sorted(x["id"] for x in left_out
                                            if str(x.get("by") or "").startswith(("base:", "purtera:"))
                                            and x["id"] not in unsure_ids)},
        "decided_by": {
            "heads_used": sorted(k for k, v in parts.items() if v["by"] == "head"),
            "fell_back": sorted(k for k, v in parts.items() if v["by"] != "head"),
            # Each part: who decided it and, for a rule, why no head did.
            "parts": parts,
            "no_head": dict(NO_HEAD_PARTS),
            "view": readings.get("view") or "",
            "base_read": have_reads,
            "keep": dict(why),
            # Which head decided whether a line is in the brief at all.
            "keep_head": ("col:admission" if use["col:admission"] else
                          "read:co_action" if use["read:co_action"] else "rule:type"),
        },
        "label_next": label_tasks[:MAX_LABEL_TASKS],
        # Every line's keep decision, compactly: who decided (a head, a rule, the view, the document
        # gate), at what p, and the keep heads' own readings, so an audit or a PM tool can tell an
        # admission drop from a co_action company drop from a rule (wave 4: only counts were kept).
        "line_decisions": _line_decisions(kept, maybe, left_out, reads, kept_by_now),
        # What decided which documents feed the brief and the quote, per document.
        "documents": docs,
        "assemble_ms": round((time.monotonic() - started) * 1000, 1),
        "evidence_ms": evidence_ms,
    }
    _stamp_layers(brief)
    return brief


_PRICE = re.compile(r"\$\s?\d")
_PRICED_TYPES = frozenset({"vendor_line_item", "bom_line", "commercial_total", "rate_card"})


def _customer_furnished(sections: dict[str, list[dict[str, Any]]], by_id: dict[str, Any], r) -> int:
    """Move a document's priced lines out of fees and scope when the supplier
    head says the customer furnishes them: the document is the customer's own
    order, and its totals are what the customer paid, not this job's price.
    Decided per document by the head's sure answers on its priced lines
    (010043's CDW order: both camera models and the junction box customer,
    the mount and license us; the grader read the $6,184 as our fee).
    Returns lines moved."""
    if r is None or not sections.get("fees"):
        return 0
    doc_of = {iid: str(getattr(by_id.get(iid), "artifact_id", "") or "") for v in sections.values()
              for iid in (x["id"] for x in v)}
    fee_docs = {doc_of[x["id"]] for x in sections["fees"] if doc_of.get(x["id"])}
    priced: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for sec, items in sections.items():
        for it in items:
            d = doc_of.get(it["id"])
            if d in fee_docs and (sec == "fees" or it.get("type") in _PRICED_TYPES or _PRICE.search(it.get("text") or "")):
                priced[d].append((sec, it))
    moved: list[tuple[str, dict[str, Any]]] = []
    for d, items in priced.items():
        seen_ids: set[str] = set()
        items = [(sec, it) for sec, it in items if sec != "job_details" and not (it["id"] in seen_ids or seen_ids.add(it["id"]))]
        sure = [(sec, it, r(it["id"], "col:supplier")) for sec, it in items]
        # Not ours is the question: the head splits a reseller's order between
        # customer and third_party (010043: 3 customer, 2 third party, 3 us).
        sure = [(sec, it, v) for sec, it, v in sure if v[1] >= SUPPLIER_BAR]
        ours = [x for x in sure if x[2][0] == "us"]
        if sure and len(ours) * 2 < len(sure):
            moved.extend(items)
    if not moved:
        return 0
    gone = {id(it) for _, it in moved}
    for sec in list(sections):
        sections[sec] = [x for x in sections[sec] if id(x) not in gone]
    for _, it in moved:
        sections["customer_furnished"].append({**it, "section_by": "base:col:supplier"})
    return len(moved)


#: Columns of ``line_decisions.rows``.
LINE_DECISION_COLS = ("id", "outcome", "by", "p", "col:admission", "read:co_action")


def _reading(reads: dict[str, dict[str, list]], aid: str, head: str) -> str | None:
    """``"answer:p"`` of one head on one line, or None when the read did not answer it."""
    ans, p = _answer((reads.get(aid) or {}).get(head))
    return None if ans == "_absent" and not p else f"{ans}:{p:.3f}"


def _line_decisions(kept: list[tuple[Any, Any, str]], maybe: list[tuple[Any, float]],
                    left_out: list[dict[str, Any]], reads: dict[str, dict[str, list]],
                    kept_by: dict[str, str] | None = None) -> dict[str, Any]:
    """``{"cols": LINE_DECISION_COLS, "rows": [...]}``: one row per line the keep loop decided.
    ``outcome`` is ``kept`` (the body), ``also`` (kept below the body bar) or the left-out reason."""
    def row(aid: str, outcome: str, by: str, p: Any) -> list[Any]:
        return [aid, outcome, by, None if p is None else round(float(p), 3),
                _reading(reads, aid, "col:admission"), _reading(reads, aid, "read:co_action")]

    rows = [row(str(a.id), "kept", by, p) for a, p, by in kept]
    rows += [row(str(a.id), "also", (kept_by or {}).get(str(a.id)) or "", p) for a, p in maybe]
    rows += [row(str(x["id"]), str(x.get("reason") or ""), str(x.get("by") or ""), x.get("p")) for x in left_out]
    return {"cols": list(LINE_DECISION_COLS), "rows": rows}


def reconcile_line_decisions(brief: dict[str, Any]) -> int:
    """Make the ledger say where each kept line ended up. ``line_decisions``
    is written by the keep loop; a later step can still move a kept line out
    of the body (``_drop_superseded`` moves an older SOW version's clauses to
    ``superseded``). On 010362 92 lines read "kept" and sat in no section and
    not in also-mentioned, with no reason. A kept or also line found in
    ``superseded`` now reads ``superseded`` (by ``rule:superseded``); one found
    nowhere reads ``kept_unplaced``. Returns rows changed."""
    ld = brief.get("line_decisions") or {}
    rows = ld.get("rows") or []
    if not rows:
        return 0
    placed = {str(x.get("id")) for v in (brief.get("sections") or {}).values() for x in v if isinstance(x, dict)}
    placed |= {str(x.get("id")) for x in brief.get("also_mentioned") or [] if isinstance(x, dict)}
    placed |= {str(i) for i in brief.get("also_mentioned_more_ids") or []}
    stale = {str(x.get("id")): x for x in brief.get("superseded") or [] if isinstance(x, dict)}
    n = 0
    for row in rows:
        if row[1] not in ("kept", "also") or str(row[0]) in placed:
            continue
        old = stale.get(str(row[0]))
        if old is not None:
            row[1], row[2] = "superseded", "rule:superseded"
        else:
            row[1], row[2] = "kept_unplaced", "none:" + str(row[2] or "")
        n += 1
    if n:
        ld.setdefault("moved_after_keep", 0)
        ld["moved_after_keep"] += n
    return n


def _by_view_reason(left_out: list[dict[str, Any]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for x in left_out:
        r = str(x.get("reason") or "")
        if r.startswith("out_of_view:"):
            out.setdefault(r.split(":", 1)[1], []).append(str(x["id"]))
    return out


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


#: The written sections and the head sections each one is written from. A
#: sentence may only cite its own section's lines: the writer was handed every
#: line and six fixed headings, and on the graded deals it moved 12 of 15
#: sentences out of the section the heads chose and wrote "painting the
#: backboard is not included" under out of scope from an overview line.
WRITER_SECTIONS: dict[str, tuple[str, ...]] = {
    "scope": ("scope_of_work", "deliverables", "provider_responsibilities"),
    "job_details": ("job_details",),
    "customer_responsibilities": ("customer_responsibilities",),
    "customer_furnished": ("customer_furnished",),
    "out_of_scope": ("out_of_scope",),
    "schedule": ("timeline",),
    "assumptions": ("assumptions",),
    "fees": ("fees",),
    "terms": ("terms", "change_control", "acceptance"),
}

_PROMPT = (
    "Write the brief a project manager at PurTera prices a field-services job from. PurTera is the provider "
    "being asked to quote and do the work. The account in the CRM record is who asked PurTera for the quote (often a "
    "reseller or partner such as CDW); the end customer is who the work is for. Never say the account or the "
    "end customer does the work.\n"
    "SUMMARY: 3 or 4 sentences on the whole job, from any facts and the CRM record: who the account is and who "
    "the end customer is when named, what work it is with its counts, where, when (dates, windows, deadlines) and, "
    "only when a fact says it, what is being asked for (a budgetary price, a quote, a SOW). Leave out anything the "
    "facts do not give; never guess what is being asked for. The CRM record's deal name is a fact too: keep its "
    "specifics (trade, cable type, counts, place, end customer) in the summary, in plain words. Never mention "
    "the CRM, a deal name or number, an id, or these instructions in the text.\n"
    "SECTIONS: write each section given under SECTIONS ONLY from that section's own facts, up to 5 short "
    "sentences. Cover every distinct fact a PM would price from once in the whole brief: never repeat a fact "
    "another sentence already gave, and merge facts that say the same thing. Never write a section that is not "
    "given, never move a fact to another section, and never say something is excluded, unknown or not stated "
    "unless that section's fact says so. A line's [under: ...] heading is the list it sits in ('What We Ask YOU "
    "Provide' means PurTera supplies it). Never say "
    "fact says so. job_details (up to 10 sentences): every count (including counts read off the figures), "
    "model, size, length, material PurTera must supply, extra work asked for, site condition, and access window "
    "or deadline the facts give, in the third person; never restate a fact Scope already gave. customer_furnished: equipment the customer already bought or "
    "supplies; it is not this job's price: say what was bought, how many, and its total by the label the "
    "fact gives (GRAND TOTAL), never whether that total includes tax.\n"
    "PRICING QUESTIONS: up to 10 questions a PM must ask before pricing THIS job that the facts leave open: "
    "specific to its work, site, counts, access, materials, who supplies what, and timing. Consider, and ask only "
    "when the facts leave it open: the street address and floor; union or prevailing-wage labor; mounting "
    "height and lift or ladder; who supplies, receives and stages the equipment; whether PurTera connects, "
    "configures and tests or only mounts; removal and disposal of old equipment; one trip or several; site "
    "hours, escort and access. Never ask what ANY "
    "fact given here (overview, sections or also kept) already answers, never ask contract boilerplate; cite "
    "the facts each question comes from.\n"
    "RULES: every sentence cites the [ids] of the facts it uses and says nothing that is not in them, except a "
    "total that follows by plain arithmetic from cited facts (89 APs x 2 drops = 178 drops), shown as the sum. Never "
    "combine two facts into a claim neither states (a note that payments exclude tax does not make a grand total "
    "tax-free); copy counts, amounts, sizes and dates exactly as a fact gives them; name a total by the label its "
    "fact gives it (GRAND TOTAL, SUBTOTAL) and never say whether it includes tax unless that same fact says so; "
    "when facts disagree, give both and say they disagree. A later fact replaces an earlier one on the same "
    "point, and a question a later fact answers is no longer open. A fact may start with the day it was written "
    "(YYYY-MM-DD): turn 'tomorrow', 'next Tuesday' and the like into the date. Give places as City, ST. Say "
    "plainly what PurTera supplies (labor, lift, cable ...) and what the customer supplies. Ignore email-client "
    "banners and notices (\"Learn why this is important\", \"You don't often get email from\"); they are not "
    "about the deal.\n"
    "Return JSON only: {\"summary\": str, \"sections\": {section: [{\"text\": str, \"cites\": [id]}]}, "
    "\"open_questions\": [{\"question\": str, \"owner\": str, \"cites\": [id]}], "
    "\"pricing_questions\": [{\"question\": str, \"cites\": [id]}]}.\n\n"
    "SERVICE: %s\nCRM RECORD: %s\nOVERVIEW FACTS (summary only): %s\nSECTIONS: %s\n"
    "ALSO KEPT (less sure, or read off the job's figures and photos by a vision pass; use them only in the "
    "summary and job_details, and only when they add a date, a "
    "deadline, a contact, a count, what is being asked for, or a change to the scope such as work added): %s\n"
    "OPEN QUESTIONS (asked in the deal; return under open_questions only those a PM must still get answered to "
    "price the job, reworded plainly; drop nudges and chasers such as 'have you had a chance to review', 'any "
    "update', questions the account asks PurTera (turnaround, availability; PurTera answers those), and notes or "
    "labels copied from a spreadsheet): %s\nCONFLICTS: %s"
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
SECTION_SENTENCES = 8
DETAIL_SENTENCES = 10
#: Catalog families shown under the pricing questions: the ones about this
#: job. Contract terms and compliance are asked of every deal (labeled valid on
#: every one), so they stay for labeling and the SOW, not the pre-quote page.
SHOWN_FAMILIES = frozenset({"site_access", "schedule", "quantities_scope", "network_it", "people_contacts"})
CATALOG_SHOW = int(os.environ.get("SOWSMITH_C3_BRIEF_CATALOG_SHOW", "0"))
ASK_BATCH = int(os.environ.get("SOWSMITH_C3_BRIEF_ASK_BATCH", "12"))
#: The prose answer's token cap. 2,000 cut off 010043's brief (it runs ~1,800).
PROSE_TOKENS = int(os.environ.get("SOWSMITH_C3_BRIEF_PROSE_TOKENS", "6000"))


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


def _complete(prompt: str, model: str | None, max_tokens: int, effort: str | None = None) -> str:
    """One call, retried once when it comes back empty or unparseable
    (llm_client returns "" on any failure, a throttled burst included: one of
    010043's two question batches came back empty and its 32 went unchecked)."""
    from app.core import llm_client

    tries = max(1, int(os.environ.get("SOWSMITH_C3_LLM_TRIES", "4")))
    for attempt in range(tries):
        raw = llm_client.complete(prompt, max_tokens=max_tokens,
                                  model=model or os.environ.get("SOWSMITH_C3_BRIEF_MODEL") or None,
                                  timeout=int(os.environ.get("SOWSMITH_C3_BRIEF_TIMEOUT", "60")),
                                  reasoning_effort=effort)
        if _json_of(raw) is not None:
            return raw
        if attempt < tries - 1:
            # A throttled burst comes back empty: three deals compiling at once
            # left 010082's SOW task list with one line (2026-10-18).
            time.sleep(2.0 * (attempt + 1))
    return raw


_INLINE_ID = re.compile(r"\b(?:atm|xdoc_conflict|pmq|ent|q)_[0-9A-Za-z_]+")


def _service_for_writer(brief: dict[str, Any]) -> str:
    """The service the writer is told: none when the head was unsure. Told
    "wireless" at 0.36, the writer called 010426's monitor install a
    wireless project in three sentences."""
    svc = brief.get("service") or {}
    if svc.get("unsure") or not svc.get("value"):
        return "not known; describe the job only from the facts"
    return str(svc["value"])


_RELATIVE = {"today": 0, "tonight": 0, "tomorrow": 1, "yesterday": -1}
_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


def _dated(text: str, when: str | None) -> str:
    """A relative day in a line, given its date in brackets, worked out from the
    day the line was written: "tomorrow" in a 2026-10-14 email is 2026-10-15
    (010440: the model, left to it, wrote Oct 14)."""
    if not when:
        return text
    try:
        from datetime import date, timedelta

        d0 = date.fromisoformat(when[:10])
    except Exception:
        return text

    def sub(m):
        w = m.group(0).lower()
        if w in _RELATIVE:
            d = d0 + timedelta(days=_RELATIVE[w])
        else:
            wd = _WEEKDAYS.index(w.split()[-1])
            ahead = (wd - d0.weekday()) % 7 or 7
            if w.startswith("next "):
                ahead = ahead if ahead > 0 else 7
            d = d0 + timedelta(days=ahead)
        return f"{m.group(0)} ({d.isoformat()})"

    return re.sub(r"\b(?:today|tonight|tomorrow|yesterday|(?:next |this )?(?:monday|tuesday|wednesday|thursday|"
                  r"friday|saturday|sunday))\b", sub, text, flags=re.I)


def _cites_of(sentence: Any) -> list[str]:
    """A written sentence's cited ids: its ``cites`` list, a comma string, or
    -- when the model put them only in the text, as it does on about one call
    in three (010043 lost every section that way) -- the ids in the text."""
    if not isinstance(sentence, dict):
        return []
    raw = sentence.get("cites")
    if isinstance(raw, str):
        raw = re.split(r"[,\s]+", raw)
    cites = [str(c).strip("[] ") for c in (raw or []) if str(c).strip("[] ")]
    if not cites:
        cites = _INLINE_ID.findall(str(sentence.get("text") or ""))
    return list(dict.fromkeys(cites))


def write(brief: dict[str, Any], *, model: str | None = None) -> dict[str, Any]:
    """The brief's prose and the question check, as parallel LLM calls: one
    for the prose, one per batch of catalog questions. Sentences whose
    citations are not facts of this brief are dropped and counted; a question
    a batch did not mark answered (citing a line that answers it) stays asked,
    so a failed or partial call loses nothing."""
    from concurrent.futures import ThreadPoolExecutor

    heads = brief["sections"]
    facts: dict[str, list[str]] = {}
    own: dict[str, set[str]] = {}
    for ws, srcs in WRITER_SECTIONS.items():
        items = [x for src in srcs for x in heads.get(src) or []][:MAX_FACTS_PER_SECTION]
        if items:
            facts[ws] = [f"[{x['id']}] " + (f"({x['when']}) " if x.get("when") else "")
                         + (f"[under: {x['under']}] " if x.get("under") else "") + _dated(x["text"], x.get("when"))
                         for x in items]
            own[ws] = {x["id"] for x in items}
    overview = [f"[{x['id']}] {x['text']}" for x in (heads.get("overview") or [])[:MAX_FACTS_PER_SECTION]]
    also_items = list(brief.get("also_mentioned") or [])[:30] + list(brief.get("figures") or [])
    also = [f"[{x['id']}] " + (f"({x['when']}) " if x.get("when") else "") + _dated(x["text"], x.get("when"))
            for x in also_items]
    if also_items:
        own.setdefault("job_details", set()).update(x["id"] for x in also_items)
    # An overview line can carry the job's details (010038's follow-up email:
    # ladders on site, no COI, a backboard, the June 24-30 window); written
    # only into the summary, they were lost.
    own.setdefault("job_details", set()).update(x["id"] for x in (heads.get("overview") or []))
    cs = [x["ids"] for x in brief["conflicts"][:15]]
    known = {x["id"] for k, v in heads.items() if k != "job_details" for x in v} | {q["id"] for q in brief["questions"]}
    catalog = [q for q in brief["questions"] if q["id"].startswith("q:")]
    qs = [f"[{q['id']}] (owner: {q['owner']}) {q['text']}" for q in brief["questions"]
          if not q["id"].startswith("q:")][:30]
    lines_seen = {e["id"] for q in catalog for e in q.get("evidence") or []}
    service = _service_for_writer(brief)
    batches = [catalog[i:i + ASK_BATCH] for i in range(0, len(catalog), ASK_BATCH)]
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=1 + len(batches)) as pool:
        prose_f = pool.submit(_complete, _PROMPT % (service, json.dumps(brief.get("deal") or {}),
                                                    json.dumps(overview), json.dumps(facts), json.dumps(also),
                                                    json.dumps(qs), json.dumps(cs)),
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
    # The model sometimes nests the question lists inside "sections" (010038):
    # they were read as sentences and dropped as uncited.
    secs_in = doc.get("sections") if isinstance(doc.get("sections"), dict) else {}
    for k in ("pricing_questions", "open_questions"):
        if k in secs_in and not doc.get(k):
            doc[k] = secs_in.pop(k)
        else:
            secs_in.pop(k, None)
    moved = 0
    for sec, sentences in list((doc.get("sections") or {}).items()):
        keep = []
        allowed = own.get(sec, set())
        for s in sentences or []:
            cites = _cites_of(s)
            if cites and all(c in allowed for c in cites):
                keep.append({"text": s.get("text", ""), "cites": cites})
            elif cites and all(c in known for c in cites):
                moved += 1  # a fact the heads put in another section
            else:
                dropped += 1
        if keep:
            doc["sections"][sec] = keep[:DETAIL_SENTENCES if sec == "job_details" else SECTION_SENTENCES]
        else:
            doc["sections"].pop(sec, None)
    out["moved_dropped"] = moved
    scope_cites = {c for x in (doc.get("sections") or {}).get("scope") or [] for c in x.get("cites") or []}
    jd = (doc.get("sections") or {}).get("job_details")
    if jd and scope_cites:
        doc["sections"]["job_details"] = [x for x in jd if not set(x.get("cites") or []) <= scope_cites]
        if not doc["sections"]["job_details"]:
            doc["sections"].pop("job_details")
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
    summary = doc.get("summary", "")
    if isinstance(summary, list):
        # The model sometimes writes the summary as sentence objects like the
        # sections (010426): the page printed the list itself.
        summary = " ".join(str(x.get("text") if isinstance(x, dict) else x).strip() for x in summary if x)
    pricing = []
    for q in doc.get("pricing_questions") or []:
        if isinstance(q, dict) and str(q.get("question") or "").strip():
            cites = [c for c in _cites_of({"cites": q.get("cites"), "text": q.get("question")}) if c in known]
            pricing.append({"question": strip_cites(str(q["question"])), "owner": "project_manager",
                            "cites": cites, "kind": "pricing"})
    out.update(summary=str(summary or ""), sections=doc.get("sections") or {},
               # Pricing questions first: what the PM must ask to price. The
               # deal's own asks after, only those the writer kept as still
               # open ("Have you had a chance to review this proposal?" is a
               # nudge, not a question to price from).
               open_questions=pricing[:10] + list(doc.get("open_questions") or []), uncited_dropped=dropped,
               # The deal's own and its pricing questions come first; the head's
               # catalog questions after them, the most likely few (all of them
               # stay for labeling).
               catalog_questions=[q for q in asked if q.get("family") in SHOWN_FAMILIES][:CATALOG_SHOW],
               catalog_questions_all=asked, answered_in_deal=answered, unchecked=len(unchecked),
               failed_ask_calls=failed)
    return out


def envelope_questions(brief: dict[str, Any]) -> list[dict[str, Any]]:
    """The base's catalog questions as the envelope's open questions (kind
    ``catalog``): the ones to ask (best first) and the ones the deal already
    answers (review_status ``answered``, with the line). ``query`` is the
    wording the Questions head reads; question_fit trains on it."""
    by_id = {q["id"]: q for q in brief.get("questions") or [] if str(q.get("id", "")).startswith("q:")}
    w = brief.get("written") or {}
    asked = w.get("catalog_questions_all", w.get("catalog_questions"))
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


def _crm_deal(project_dir: Any) -> dict[str, str]:
    """The CRM's name for the deal and its account (who we quote), never its
    amount: before the quote that amount IS the quote."""
    if not project_dir:
        return {}
    try:
        from app.core.orbitbrief_envelope import _load_manifest_crm

        crm = _load_manifest_crm(project_dir) or {}
    except Exception:
        return {}
    return {k: str(crm[k]) for k in ("deal_name", "account_name") if crm.get(k)}


def run(atoms: list[Any], packets: list[Any] = (), *, service_fallback: str | None = None,
        project_dir: Any = None, on_built: Any = None) -> dict[str, Any]:
    """Build (and, when switched on, write) the brief from this compile's read. ``on_built(brief)`` is
    called once the sections, job details and deal are final and before the writer runs (the compile
    starts the quote heads there, beside the writer's calls); its errors never cost the brief."""
    from app.core import c3_type_base

    brief = build(atoms, packets, service_fallback=service_fallback, readings=c3_type_base.full_read())
    # The heads leave contacts and the deal's identity to the CRM
    # (contact_from_crm); the writer is told what the CRM calls the deal and
    # who the account is, or the brief names neither (5 of 6 graded deals).
    brief["deal"] = _crm_deal(project_dir)
    # Who furnishes each kept line, as the supplier head reads it: the SOW's
    # supply lists follow it (c3_sow) instead of guessing from wording.
    for v in list((brief.get("sections") or {}).values()) + [brief.get("also_mentioned") or []]:
        for x in v:
            r = c3_type_base.line_reading(str(x["id"]), "col:supplier")
            if r:
                x["supplier"] = [r[0], round(r[1], 3)]
            # And who does it (read:task_owner_role): the SOW's party for a
            # duty follows it where the head answered (c3_sow._follow_owner).
            o = c3_type_base.line_reading(str(x["id"]), "read:task_owner_role")
            if o:
                x["owner"] = [o[0], round(o[1], 3)]
    # Lines a vision pass read off the job's own figures and photos. The heads
    # never saw such lines in training and file them as parser-derived; until
    # they are labeled, the writer gets them as what they are (010038's floor
    # plans: 3 APs, 2 TVs, 1 NUC/TAP, 1 printer), and each is a label task.
    kept_now = {str(x["id"]) for v in (brief.get("sections") or {}).values() for x in v}
    ours_now = {i for ids in ((brief.get("left_out") or {}).get("out_of_view") or {}).values() for i in ids}
    figs, seen_f = [], set()
    for a in atoms:
        v = a.value if isinstance(getattr(a, "value", None), dict) else {}
        key = re.sub(r"\W+", " ", str(a.raw_text).lower()).strip()
        if _vision(a) and str(a.id) not in kept_now and str(a.id) not in ours_now and key not in seen_f:
            seen_f.add(key)
            figs.append({"id": str(a.id), "text": a.raw_text[:300], "when": _when_of(a)})
    brief["figures"] = figs[:24]
    brief.setdefault("label_next", []).extend(
        {"id": f["id"], "head": "read:co_action", "base_says": "figure", "p": None, "text": f["text"][:200]}
        for f in figs[:10])
    try:
        _drop_superseded(brief, atoms, c3_type_base._DEAL.get("documents") or [])
    except Exception as exc:  # pragma: no cover - never lose the brief over this
        log.info("C3 brief superseded check skipped: %s", type(exc).__name__)
    reconcile_line_decisions(brief)
    # The base's line type for every line that names a site, for the roster
    # filter (site_identity.filter_roster): a site only ever mentioned in lines
    # the base does not read as a site is a mention, not a job site.
    _types = c3_type_base._DEAL.get("types") or {}
    # Only lines that name a site in their own words: a key copied onto every
    # line of a note or SOW by document provenance (or the single-site anchor)
    # made the From/To/signature lines "site lines" (010316, 010227).
    from app.core.site_keys_follow_heads import _copied

    brief["site_line_types"] = {
        str(a.id): _types[str(a.id)] for a in atoms
        if str(a.id) in _types and any(str(k).startswith("site:") and not _copied(a, str(k))
                                       for k in (getattr(a, "entity_keys", None) or []))}
    if c3_type_base._DEAL.get("site_keys"):
        brief["site_keys"] = c3_type_base._DEAL["site_keys"]
    brief["contacts"] = contacts(contact_lines(atoms))
    # Only lines the heads kept in the body: picked by wording alone it put
    # back lines co_action dropped -- 9 of 15 on 010038, among them "concrete"
    # and the stale "if they have a scissor lift/ ladder". A line the keep
    # head only half kept ("also mentioned") does not drive the job details or
    # the quote either (010363: an optional site and "10 Site-to-Site IPsec
    # VPN Tunnels" at keep 0.69 / 0.87 became the quote's job lines).
    kept_ids = {str(x["id"]) for v in (brief.get("sections") or {}).values() for x in v}
    details = job_details(atoms, kept_ids)
    brief["job_details"] = details
    if details:
        # The writer words these as facts (counts, models, conditions); pasted
        # as they were, a grader read customer emails in the first person.
        brief.setdefault("sections", {})["job_details"] = [
            {"id": d["cites"][0], "text": d["text"]} for d in details]
    if on_built is not None:
        try:
            on_built(brief)
        except Exception as exc:  # pragma: no cover - the caller falls back to calling after the brief
            log.info("C3 brief on_built hook failed: %s", type(exc).__name__)
    if writing_enabled():
        try:
            brief["written"] = write(brief)
        except Exception as exc:
            brief["written"] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
        w = brief.get("written")
        _check_written(brief, atoms)
        if isinstance(w, dict) and isinstance(w.get("sections"), dict):
            secs = w["sections"]
            written_details = w["sections"].get("job_details") or _write_job_details(brief, details)
            if written_details:
                details = written_details
                w["job_details_written"] = True
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


def _check_written(brief: dict[str, Any], atoms: list[Any]) -> None:
    """The writer's checks, as two chains that run side by side: the sentences
    (verify, then cover what the verified brief left out) and the open questions
    (the pricing ones the facts answer, then the ones the deal answers). Each
    chain is in its old order; neither reads what the other writes (sentences
    live in ``written.sections``, questions in ``written.open_questions``), so
    the brief is the same as running all four one after another -- which put
    four or more LLM round trips end to end on every compile."""

    def _sentences() -> None:
        try:
            _verify_written(brief)
        except Exception as exc:  # pragma: no cover
            log.info("C3 brief verify pass skipped: %s", type(exc).__name__)
        try:
            _cover_missing_facts(brief)
        except Exception as exc:  # pragma: no cover
            log.info("C3 brief coverage pass skipped: %s", type(exc).__name__)

    def _questions() -> None:
        try:
            _drop_questions_the_facts_answer(brief)
        except Exception as exc:  # pragma: no cover
            log.info("C3 brief pricing-question check skipped: %s", type(exc).__name__)
        try:
            _drop_answered_open_questions(brief, atoms)
        except Exception as exc:  # pragma: no cover - a check never costs the brief
            log.info("C3 brief open-question check skipped: %s", type(exc).__name__)

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="c3-brief-check") as pool:
        jobs = [pool.submit(_sentences), pool.submit(_questions)]
        for job in jobs:
            job.result()


_DETAILS_PROMPT = (
    "Turn these deal lines into the job details a project manager prices a field-services job from: at most 8 "
    "short sentences. Count each kind of equipment or work item (\"3 access points\", \"2 TVs\"); give models, "
    "measurements and cable lengths, site conditions (ceiling, wall, mounting, lift or ladder) and access or "
    "timing limits. Copy every count and size exactly as a line gives it and never add lines' counts together "
    "unless they are the same kind of item. Write in the third person; never quote or copy a line, never write \"I\" or \"we\". When "
    "lines disagree, the later line wins (lines are in order). Say nothing a line does not say, and cite the "
    "[ids] each sentence uses. Return JSON only: {\"sentences\": [{\"text\": str, \"cites\": [id]}]}."
    "\n\nSERVICE: %s\nLINES:\n%s"
)


def _write_job_details(brief: dict[str, Any], details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The job-details lines worded as facts by one small call (the main
    writer, told to add the section, kept to its six). Sentences citing
    anything but these lines are dropped; nothing back means the raw lines stay."""
    if not details:
        return []
    ids = {c for d in details for c in d.get("cites") or []}
    lines = "\n".join(f"[{d['cites'][0]}] {d['text']}" for d in details if d.get("cites"))
    raw = _complete(_DETAILS_PROMPT % (_service_for_writer(brief), lines), None, 900)
    got = _json_of(raw) or {}
    out = []
    for snt in got.get("sentences") or []:
        if not isinstance(snt, dict):
            continue
        cites = [str(c).strip("[]") for c in snt.get("cites") or []]
        text = str(snt.get("text") or "").strip()
        if text and cites and all(c in ids for c in cites) and not re.search(r"\b(I|we|our)\b", text):
            out.append({"text": text, "cites": cites})
    return out[:8]


_FACTS_ANSWER_PROMPT = (
    "Here are a field-services deal's facts and a list of questions. Which questions do the facts ALREADY "
    "answer (state the answer, or settle it: a fact that ladders are on site answers whether a lift is needed)? "
    "Return JSON only: {\"answered\": [question id, ...]}.\nFACTS:\n%s\nQUESTIONS: %s"
)


_COVER_PROMPT = (
    "Below is a brief a project manager prices a field-services job from, and every fact it was written from. "
    "List the facts a PM needs to price the job that the brief does NOT already state: counts (also counts "
    "read off figures), equipment and materials PurTera must supply or the customer supplies, extra work "
    "asked for (anything PurTera is asked to supply or install), access windows, dates and deadlines, site "
    "conditions, and an on-site contact a line names as such. Add only what a line actually states; when no "
    "line states something, add nothing for it. A line's [under: ...] heading says what list it is in ('What We "
    "Ask YOU Provide' means PurTera supplies it). Never write that something is unknown, not stated, not given, "
    "not mentioned or not requested, and never call a salesperson the on-site contact. "
    "Write each as "
    "one short third-person sentence that cites the [ids] it comes from and says nothing more than they do; "
    "a later fact replaces an earlier one; never restate what the brief already says. At most 8. Return JSON "
    "only: {\"add\": [{\"text\": str, \"cites\": [id]}]}.\nBRIEF:\n%s\nFACTS:\n%s"
)


_VERIFY_PROMPT = (
    "For each numbered sentence, say whether the lines it cites state what it says. A faithful rewording and a "
    "total by plain arithmetic from the cited lines count as stated; anything added, guessed, merged into a new "
    "claim, or attributed to the wrong party does not. Return JSON only: {\"supported\": [sentence numbers]}.\n%s"
)


def _verified(sentences: list[dict[str, Any]], lines: dict[str, str]) -> list[dict[str, Any]]:
    """Keep only added sentences their cited lines state exactly: on 010426 the
    added ones called the reseller's salesperson the on-site contact."""
    if not sentences:
        return []
    body = "\n".join(f"{i}. {x['text']}\n" + "\n".join(f"   [{c}] {lines.get(c, '')[:300]}" for c in x["cites"])
                     for i, x in enumerate(sentences))
    got = _json_of(_complete(_VERIFY_PROMPT % body, None, 300)) or {}
    ok = {int(i) for i in got.get("supported") or [] if str(i).isdigit()}
    return [x for i, x in enumerate(sentences) if i in ok]


def _verify_written(brief: dict[str, Any]) -> int:
    """Every written sentence checked against the lines it cites; one they do
    not state is dropped (010426's "installing 45 desks" for monitors on arms,
    an after-hours "exclusion" nobody made). Returns sentences dropped."""
    w = brief.get("written") or {}
    secs = w.get("sections")
    if not isinstance(secs, dict):
        return 0
    lines: dict[str, str] = {}
    for v in (brief.get("sections") or {}).values():
        for x in v:
            lines[str(x["id"])] = (f"[under: {x['under']}] " if x.get("under") else "") + str(x["text"])
    for x in list(brief.get("also_mentioned") or []) + list(brief.get("figures") or []):
        lines[str(x["id"])] = str(x["text"])
    flat = [(k, i, x) for k, v in secs.items() if k != "contacts" for i, x in enumerate(v)
            if isinstance(x, dict) and x.get("cites")]
    if not flat:
        return 0
    keep = _verified([x for _, _, x in flat][:60], lines)
    keep_ids = {id(x) for x in keep}
    gone = 0
    for k in list(secs):
        if k == "contacts":
            continue
        before = secs[k]
        secs[k] = [x for x in before if id(x) in keep_ids or not (isinstance(x, dict) and x.get("cites"))]
        gone += len(before) - len(secs[k])
        if not secs[k]:
            secs.pop(k)
    w["verify_dropped"] = gone
    return gone


def _cover_missing_facts(brief: dict[str, Any]) -> int:
    """A second look for pricing facts the brief left out: the writer, given
    every line, still dropped 010038's floor-plan counts, the backboard it is
    asked to supply and the June 24-30 access window. Added sentences go under
    job details and must cite lines the brief was given."""
    w = brief.get("written") or {}
    secs = w.get("sections")
    if not isinstance(secs, dict):
        return 0
    items: list[dict[str, Any]] = []
    for v in (brief.get("sections") or {}).values():
        items += v
    items += list(brief.get("also_mentioned") or [])[:30] + list(brief.get("figures") or [])
    known = {str(x["id"]) for x in items}
    facts = "\n".join(f"[{x['id']}] " + (f"({x['when']}) " if x.get("when") else "")
                      + (f"[under: {x['under']}] " if x.get("under") else "") + _dated(str(x["text"])[:300], x.get("when"))
                      for x in {str(x["id"]): x for x in items}.values())[:26000]
    text = "\n".join([str(w.get("summary") or "")] + [str(x.get("text") or "") for v in secs.values() for x in v])
    got = _json_of(_complete(_COVER_PROMPT % (text[:8000], facts), None, 1200)) or {}
    add = []
    for x in got.get("add") or []:
        cites = [c for c in _cites_of(x) if c in known]
        t = strip_cites(str(x.get("text") or "") if isinstance(x, dict) else "")
        if t and cites:
            add.append({"text": t, "cites": cites})
    add = _verified(add[:8], {str(x["id"]): str(x["text"]) for x in items})
    if add:
        secs["job_details"] = list(secs.get("job_details") or []) + add
        w["covered_added"] = len(add[:8])
    return len(add)


def _drop_questions_the_facts_answer(brief: dict[str, Any]) -> int:
    """The writer's pricing questions checked against every fact it was given:
    on 010038, 9 of 12 asked what the deal's own lines already said (ladders on
    site, no COI, the access window)."""
    w = brief.get("written") or {}
    qs = w.get("open_questions") or []
    pricing = [(i, q) for i, q in enumerate(qs) if q.get("kind") == "pricing"]
    if not pricing:
        return 0
    lines = []
    for v in (brief.get("sections") or {}).values():
        lines += [x["text"] for x in v]
    lines += [x["text"] for x in (brief.get("also_mentioned") or [])[:30]]
    lines += [x["text"] for x in brief.get("figures") or []]
    facts = "\n".join(f"- {t[:300]}" for t in dict.fromkeys(lines))[:24000]
    raw = _complete(_FACTS_ANSWER_PROMPT % (facts, json.dumps({f"p{i}": q["question"] for i, q in pricing})),
                    None, 400)
    got = _json_of(raw) or {}
    gone = {str(x) for x in got.get("answered") or []}
    if not gone:
        return 0
    w["open_questions"] = [q for i, q in enumerate(qs) if f"p{i}" not in gone]
    w["pricing_answered_in_deal"] = [q for i, q in enumerate(qs) if f"p{i}" in gone]
    return len(gone)


def _drop_answered_open_questions(brief: dict[str, Any], atoms: list[Any]) -> int:
    """The deal's own questions the deal goes on to answer are not open. On the
    graded pre-quote deals the brief re-asked "Can you confirm the address and
    quantity?" and "Did he mention the cable lengths?" that later emails had
    answered. Each question gets its closest lines and the same answered check
    the catalog gets; an answered one moves to ``answered_open_questions``."""
    w = brief.get("written")
    qs = (w or {}).get("open_questions") or []
    if not qs:
        return 0
    probe = [{"id": f"d{i}", "text": strip_cites(str(q.get("question") or ""))} for i, q in enumerate(qs)]
    _attach_evidence(probe, atoms)
    service = _service_for_writer(brief)
    answered: dict[str, list[str]] = {}
    batches = [probe[b:b + ASK_BATCH] for b in range(0, len(probe), ASK_BATCH)]

    def _ask(batch: list[dict[str, Any]]) -> str:
        return _complete(_ASK_PROMPT % (service, json.dumps(
            [{"id": q["id"], "q": q["text"], "lines": [f"[{e['id']}] {e['text']}" for e in q.get("evidence") or []]}
             for q in batch])), None, 600)

    # The batches are independent calls: asked together, read back in batch order.
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=max(1, len(batches)), thread_name_prefix="c3-brief-ask") as pool:
        raws = list(pool.map(_ask, batches))
    for batch, raw in zip(batches, raws):
        got = _json_of(raw) or {}
        seen = {e["id"] for q in batch for e in q.get("evidence") or []}
        for c in got.get("answered") or []:
            by = c.get("by") if isinstance(c.get("by"), list) else [c.get("by")]
            by = [str(x).strip("[]") for x in by if str(x).strip("[]") in seen]
            if by:
                answered[str(c.get("id"))] = by
    keep, gone = [], []
    for i, q in enumerate(qs):
        (gone if f"d{i}" in answered else keep).append({**q, **({"answered_by": answered[f"d{i}"]}
                                                                 if f"d{i}" in answered else {})})
    w["open_questions"] = keep
    w["answered_open_questions"] = gone
    return len(gone)


def strip_cites(text: str) -> str:
    return re.sub(r"\s*\[(?:[a-z]+_[0-9a-zA-Z_]+(?:,\s*)?)+\]", "", text or "").strip()


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
    r"elevator|loading dock|move[- ]in|deadline|asap|urgent|week of|weeks? (after|before)|thanksgiving|christmas|"
    r"holiday|new year|(next|this) (week|month)|january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\b", re.I)
_VENDOR_TOTAL = re.compile(r"\b(sub ?total|grand total|sales tax|tax|shipping|lease|fmv)\b", re.I)
_PLACEHOLDER_LINE = re.compile(r"^\[(image|drawing|figure|table)[^\]]*\]", re.I)


def job_details(atoms: list[Any], kept: set[str] | None = None) -> list[dict[str, Any]]:
    """The concrete lines a PM prices from: quantities, equipment and models,
    lengths, site conditions, access windows and deadlines, in document order,
    one per distinct line, each citing itself. Chosen by what the line says,
    not its type: on 010038 the ceiling, access window and rack lines were
    typed scope_item / dependency."""
    first, rest, seen = [], [], set()
    for a in atoms:
        t = _type(a)
        if kept is not None and str(a.id) not in kept:
            continue
        # A line co_action kept is eligible whatever its type reads (small_talk
        # is a type, not a keep decision); the other types are never details.
        if (t in _NOT_DETAIL and not (t == "small_talk" and kept is not None)) or _not_brief(a):
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


_ONSITE = re.compile(r"(?i:\b(?:(?:site|on-?site|onsite|local)\s+contact|contact\s+person|point\s+of\s+contact)\s*(?:is|will be|:)?)\s+"
                     r"(?P<name>[A-Z][a-z]+(?:\s+[A-Z]\.?)?(?:\s+[A-Z][a-zA-Z'-]+){1,2})")


def _line_of(atom: Any) -> int | None:
    refs = getattr(atom, "source_refs", None) or []
    loc = getattr(refs[0], "locator", None) if refs else None
    loc = loc if isinstance(loc, dict) else {}
    try:
        return int(loc.get("line_start")) if loc.get("line_start") is not None else None
    except (TypeError, ValueError):
        return None


def contact_lines(atoms: list[Any]) -> list[Any]:
    """The lines contacts() reads: not those a quoting read leaves out (c3_type_base.out_of_quoting_view)
    -- our own people's signatures on our quote, or someone who only wrote after the cut, are not the
    deal's contacts at quoting time."""
    from app.core.c3_type_base import out_of_quoting_view

    out = out_of_quoting_view()
    return [a for a in atoms if str(getattr(a, "id", "")) not in out] if out else list(atoms)


def contacts(atoms: list[Any]) -> list[dict[str, Any]]:
    """The deal's people, from the lines read as stakeholders: a real name with
    a role, email or phone, one entry per person, most-mentioned first. No LLM.
    The graders found every name the brief left out (Rob Cole, Rafael
    Fernandez ...) already on stakeholder lines; the writer just dropped them.
    A greeting ("Hi Tanner"), a place or a date is not a person."""
    people: dict[str, dict[str, Any]] = {}
    by_art: dict[str, list[Any]] = {}
    for a in atoms:
        by_art.setdefault(str(getattr(a, "artifact_id", "")), []).append(a)
    for i, a in enumerate(atoms):
        m = _ONSITE.search(str(getattr(a, "raw_text", "") or ""))
        if m:
            # "Contact Person is Rafael Fernandez" names the role itself: the
            # on-site contact (010387), which no stakeholder field carried.
            name = re.sub(r"\s+", " ", m.group("name")).strip(" ,.;:")
            if 2 <= len(name.split()) <= 4 and not _NOT_A_NAME.search(name):
                p = people.setdefault(name.lower(), {"name": name, "role": "", "email": "", "phone": "",
                                                     "n": 0, "cites": []})
                p["n"] += 2
                p["role"] = "on-site contact"
                p["cites"].append(str(a.id))
                # Their number is the next line of the same document by line
                # number ("Cell Number: 904-386-5066", 010387); the compile's
                # own atom order is not the document's.
                here = _line_of(a)
                if here is not None:
                    for b in by_art.get(str(getattr(a, "artifact_id", "")), []):
                        lb = _line_of(b)
                        if lb is None or not here < lb <= here + 3:
                            continue
                        bt = str(getattr(b, "raw_text", "") or "").strip()
                        ph = _PHONE.search(bt)
                        if ph and not p["phone"] and re.fullmatch(
                                r"(?i)\s*(cell|mobile|phone|tel|direct)?\s*(number|no\.?|#)?\s*:?\s*[\d()+.\s-]{10,20}\s*", bt):
                            p["phone"] = ph.group(0)
                            break
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
        phone = str(v.get("phone") or "").strip()
        if not phone:
            # A number written before the name is not theirs: "Cell Number:
            # 904-386-5066, Bill Ort, AVP" is the on-site contact's cell run
            # into Bill's signature (010387).
            ph = _PHONE.search(a.raw_text or "")
            at = (a.raw_text or "").find(name.split()[0]) if name else -1
            phone = ph.group(0) if ph and (at < 0 or ph.start() > at) else ""
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
