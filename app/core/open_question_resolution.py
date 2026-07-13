"""Open-question resolution + gap-question generation.

The extractor mints an ``open_question`` atom from any text ending in
``?`` (vendor discovery FAQs, "TBD", RFI rows). That's naive: a question
whose answer sits two inches away in the same deal — "What size TVs?"
answered by an LG 65" display atom, "inventory count?" answered by a
scope requirement — gets surfaced to the PM as an unresolved *blocker*.
Meanwhile the genuine gaps (no named site contact, no dates, unconfirmed
dependencies) are never asked because nobody typed them with a "?".

This module flips that:

1. ``resolve_open_questions`` — an ``open_question`` is marked *answered*
   when another (non-question) atom in the corpus shares an
   answer-bearing entity key with it. Answered questions stay in the
   atom stream (provenance) but are flagged so the dashboard drops them
   as blockers. Uses the linkage the extractor already computed
   (``entity_keys``) — deterministic, no LLM.

2. ``generate_gap_questions`` — turns the *missing* high-priority SRL
   fields (site contact, schedule dates, key dependencies) into explicit
   questions the PM should chase. Driven by the SRL schema, not by literal
   "?" characters.
"""

from __future__ import annotations

import re
from typing import Any

# Entity-key prefixes that, when shared between an open_question and some
# other atom, mean the question's subject is concretely covered. Generic
# location/org/stakeholder keys are excluded: a question merely mentioning
# a site isn't "answered" just because a site atom exists.
_ANSWER_BEARING_PREFIXES: tuple[str, ...] = (
    "device:",
    "requirement:",
    "quantity:",
    "money:",
    "sku:",
    "acceptance:",
    "acceptance_criterion:",
    "milestone:",
    "phase:",
    "service:",
    "service_line:",
    "scope:",
    "task:",
)

ANSWERED_FLAG = "answered_in_corpus"
NOISE_FLAG = "not_pm_actionable_question"

_TRANSCRIPT_SPEAKER_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\s*\[\d{1,2}:\d{2}\]")
# Literal "?" noise that is NOT a PM blocker when it appears in free-prose FAQ
# dumps. Transcript diarized turns are exempt — they are conversation-graph
# atoms (pricing Q→A, remote-vs-onsite, etc.) and must stay in the stream.
_UNHELPFUL_QUESTION_RE = re.compile(
    r"\b("
    r"have\s+the\s+what|"
    r"anything\s+else\s+you\s+need"
    r")\b",
    re.I,
)
_ANSWER_AFTER_QUESTION_RE = re.compile(
    r"\?\s+(?:yeah|yep|yes|no|got it|it'?s|i think|basically|daniel|jacob|eddie)\b",
    re.I,
)


def _atom_type_str(atom: Any) -> str:
    at = getattr(atom, "atom_type", None)
    return at.value if hasattr(at, "value") else str(at or "")


def _answer_bearing_keys(keys: Any) -> set[str]:
    out: set[str] = set()
    for k in keys or []:
        ks = str(k)
        if any(ks.startswith(p) for p in _ANSWER_BEARING_PREFIXES):
            out.add(ks)
    return out


def is_answered_question(atom: Any) -> bool:
    val = getattr(atom, "value", None)
    if isinstance(val, dict) and val.get("answered") is True:
        return True
    return ANSWERED_FLAG in (getattr(atom, "review_flags", None) or [])


def is_unhelpful_pm_question(atom: Any) -> bool:
    """True when a literal question is transcript/dialogue noise, not a PM gap.

    Diarized hybrid-transcript turns (``block_kind=transcript_turn``) are
    NEVER unhelpful here — they belong to the conversation graph (pricing
    Q→A, signature routing, remote-vs-onsite) and must remain auditable
    and head-eligible. The PM-gap filter only targets free-prose FAQ noise.
    """
    if _atom_type_str(atom) != "open_question":
        return False
    # Conversation-graph turns from hybrid rewrite — keep always.
    refs = getattr(atom, "source_refs", None) or []
    if refs:
        loc = getattr(refs[0], "locator", None) or {}
        if isinstance(loc, dict) and (
            loc.get("block_kind") == "transcript_turn"
            or loc.get("hybrid_plan")
            or loc.get("utterance_index") is not None
        ):
            return False
    val = getattr(atom, "value", None) or {}
    if isinstance(val, dict):
        if val.get("kind") == "visual_page_marker":
            return False
        if val.get("answered") is True:
            return True
        if val.get("kind") == "transcript_turn" or val.get("speaker"):
            return False
    text = str(getattr(atom, "raw_text", None) or getattr(atom, "text", None) or "")
    if not text:
        return False
    low = text.lower()
    if len(text) > 300:
        return True
    if _TRANSCRIPT_SPEAKER_RE.search(text):
        return True
    if _ANSWER_AFTER_QUESTION_RE.search(text):
        return True
    if _UNHELPFUL_QUESTION_RE.search(text):
        return True
    # "Do you have resources..." is a sales/resource ask that task backfill turns
    # into scope; it is not a PM blocker once task evidence exists.
    if low.startswith("do you have resources") and "ubiquiti install" in low:
        return True
    return False


def filter_unhelpful_open_questions(atoms: list[Any]) -> tuple[list[Any], list[Any]]:
    """Remove non-actionable literal questions from the active atom stream."""
    from app.core.schemas import ReviewStatus

    kept: list[Any] = []
    dropped: list[Any] = []
    for atom in atoms:
        if is_unhelpful_pm_question(atom):
            val = getattr(atom, "value", None)
            if isinstance(val, dict):
                val["suppressed_as"] = "not_pm_actionable_question"
            flags = list(getattr(atom, "review_flags", None) or [])
            if NOISE_FLAG not in flags:
                atom.review_flags = sorted(set(flags + [NOISE_FLAG]))
            if getattr(atom, "review_status", None) == ReviewStatus.needs_review:
                atom.review_status = ReviewStatus.auto_accepted
            dropped.append(atom)
            continue
        kept.append(atom)
    return kept, dropped


def resolve_open_questions(atoms: list[Any]) -> int:
    """Flag open_question atoms whose answer exists elsewhere in the corpus.

    Mutates in place. Returns the number flagged answered. Never touches
    visual-page-marker questions (those are genuine review items).
    """
    from app.core.schemas import ReviewStatus

    # Build the set of answer-bearing keys owned by NON-question atoms.
    answered_keys: set[str] = set()
    for atom in atoms:
        if _atom_type_str(atom) == "open_question":
            continue
        answered_keys |= _answer_bearing_keys(getattr(atom, "entity_keys", None))

    if not answered_keys:
        return 0

    resolved = 0
    for atom in atoms:
        if _atom_type_str(atom) != "open_question":
            continue
        val = getattr(atom, "value", None) or {}
        if isinstance(val, dict) and val.get("kind") == "visual_page_marker":
            continue
        q_keys = _answer_bearing_keys(getattr(atom, "entity_keys", None))
        if not q_keys or q_keys.isdisjoint(answered_keys):
            continue
        # Mark answered.
        if isinstance(getattr(atom, "value", None), dict):
            atom.value["answered"] = True
            atom.value["answered_by_keys"] = sorted(q_keys & answered_keys)
        flags = list(getattr(atom, "review_flags", None) or [])
        if ANSWERED_FLAG not in flags:
            atom.review_flags = sorted(set(flags + [ANSWERED_FLAG]))
        # An answered question is no longer a blocker -> auto_accepted.
        if getattr(atom, "review_status", None) == ReviewStatus.needs_review:
            atom.review_status = ReviewStatus.auto_accepted
        resolved += 1
    return resolved


# --- gap-question generation from the SRL schema -----------------------

# High-priority missing-field -> PM question. Only the fields whose
# absence genuinely blocks or risks delivery; we don't nag for every one
# of the 41 SRL fields. Keyed by SRL field_id.
_GAP_QUESTIONS: dict[str, str] = {
    "site_contact": "Who is the on-site contact (name + phone) for each location?",
    "project_sponsor": "Who is the customer project sponsor / decision-maker?",
    "project_manager": "Who is the vendor-side project manager / coordinator?",
    "kickoff_date": "What is the kickoff / project start date?",
    "cutover_date": "What is the cutover / go-live date?",
    "phase_milestones": "What are the phase milestones and their dates?",
    "blackout_windows": "Are there blackout / maintenance windows to avoid?",
    "site_access_terms": "What are the site access, escort, and badging terms?",
    "work_hours": "What are the allowed work hours / windows on site?",
    "payment_terms": "What are the payment / invoicing terms?",
    "acceptance_criteria": "What are the customer acceptance criteria and sign-off gate?",
    "device_qty_per_site": "What is the confirmed device count per site?",
    "customer_responsibilities": "What customer-side responsibilities / dependencies are required (power, network, credentials)?",
}


# --- universal PM head-start checklist (deal-agnostic) ------------------
# The standing questions a senior PM/quoter asks on ANY low-voltage / field-
# services deal before it can be priced and scheduled. Each entry carries
# topic keywords; the question is offered only when the deal's atoms do NOT
# already cover that topic (universal presence check, no per-deal tuning), so
# a complete deal sees few and a thin deal gets a full head-start. "Even better
# than the PM would": this is the consolidated checklist, never forgotten.
_HEADSTART: list[tuple[str, str, str, tuple[str, ...]]] = [
    # people / contacts
    ("site_contact","People","Who is the on-site contact (name + mobile) at each location?",("on-site contact","site contact","poc")),
    ("sponsor","People","Who is the customer project sponsor / decision-maker?",("sponsor","decision maker","decision-maker")),
    ("vendor_pm","People","Who is the vendor-side project manager / coordinator?",("project manager","coordinator","pm fee","projmgmt")),
    ("escalation","People","What is the escalation path and after-hours contact?",("escalation","after hours","after-hours")),
    ("security_contact","People","Who approves site/badge access and security clearance?",("badge","clearance","security contact")),
    # schedule / dates
    ("kickoff","Schedule","What is the confirmed kickoff / start date?",("kickoff","kick-off","start date","commencement")),
    ("golive","Schedule","What is the cutover / go-live / completion date?",("go-live","go live","cutover","completion date")),
    ("milestones","Schedule","What are the phase milestones and their dates?",("milestone","phase ","schedule")),
    ("blackout","Schedule","Are there blackout / freeze / maintenance windows to avoid?",("blackout","freeze window","maintenance window")),
    ("workhours","Schedule","What are the permitted work hours (incl. after-hours/weekend)?",("work hours","working hours","business hours","after hours")),
    ("duration","Schedule","What is the expected project duration / number of trips?",("duration","weeks","number of trips","mobilization")),
    ("leadtime","Schedule","What are material / equipment lead times and order-by dates?",("lead time","lead-time","order by","procurement")),
    # site access / logistics
    ("access_terms","Site","What are the site access, escort, and badging terms?",("access","escort","badge")),
    ("parking_loading","Site","Is there parking, a loading dock, and freight access?",("parking","loading dock","freight","dock")),
    ("staging","Site","Where can equipment be staged / stored on site?",("staging","storage","stage equipment")),
    ("elevator","Site","Is elevator / lift / hoisting access available for materials?",("elevator","lift","hoist")),
    ("idf_mdf","Site","Where are the IDF/MDF / telecom closets and are they ready?",("idf","mdf","closet","telecom room")),
    ("power_avail","Site","Is adequate power / circuits available at each work area?",("power","circuit","outlet","ups")),
    ("ceiling_walls","Site","Are ceilings/walls/pathways suitable and accessible for the install?",("ceiling","pathway","conduit","wall structure")),
    ("env_conditions","Site","Are there environmental constraints (clean room, occupied space, noise)?",("clean room","occupied","noise","dust")),
    ("multi_site","Site","Is the site list / address / count confirmed for every location?",("site list","locations","addresses","per site")),
    # scope / quantities
    ("device_count","Scope","Is the confirmed device/drop count per site locked?",("device count","quantity","qty","drops","units")),
    ("make_model","Scope","Are exact makes/models and part numbers confirmed?",("make and model","model","part number","sku")),
    ("mounting","Scope","Who supplies mounts/brackets/hardware and to what spec?",("mount","bracket","hardware","anchor")),
    ("cabling_spec","Scope","What cabling type/category, lengths, and termination standard apply?",("cat6","cat5","cabling","termination","fiber")),
    ("removal_disposal","Scope","What is the removal / haul-away / ITAD / e-waste process?",("removal","haul","disposal","itad","e-waste","decommission")),
    ("config_scope","Scope","What configuration/programming/testing is in scope per device?",("configuration","programming","testing","provisioning")),
    ("labeling","Scope","What labeling / documentation / as-built standard is required?",("labeling","as-built","as built","documentation standard")),
    ("inventory","Scope","Is an inventory / asset-tag count required on arrival?",("inventory","asset tag","asset-tag","serial capture")),
    # network / IT
    ("network_ready","IT","Will the network/VLAN/IP/DHCP be provisioned before arrival?",("vlan","ip address","dhcp","network ready","switchport")),
    ("credentials","IT","Will customer credentials / system access be provided?",("credential","login","access to system","account")),
    ("firmware","IT","Is there a firmware / image / golden-config standard to apply?",("firmware","image","golden config","baseline config")),
    ("integration","IT","What upstream systems must this integrate with (VMS, ACS, DNS)?",("integration","vms","acs","dns","headend")),
    # commercial
    ("payment_terms","Commercial","What are the payment / invoicing / milestone-billing terms?",("payment","net 30","invoice","billing")),
    ("rate_basis","Commercial","Is pricing T&M, fixed-fee, or per-device, and what's the rate basis?",("t&m","fixed fee","per device","rate")),
    ("change_order","Commercial","What is the change-order / out-of-scope rate and approval process?",("change order","change-order","out of scope","additional work")),
    ("travel_expense","Commercial","Are travel / per-diem / freight expenses billable and capped?",("travel","per diem","per-diem","expense")),
    ("min_callout","Commercial","Is there a minimum call-out / trip charge / cancellation fee?",("minimum","call-out","callout","cancellation")),
    ("tax","Commercial","Who is responsible for taxes / duties / permits?",("tax","duty","duties","permit")),
    # compliance / risk
    ("insurance","Compliance","What insurance / COI limits and bonding are required?",("insurance","coi","bonding","liability")),
    ("prevailing_wage","Compliance","Does prevailing wage / union / Davis-Bacon labor apply?",("prevailing wage","union","davis-bacon","certified payroll")),
    ("background","Compliance","Are background checks / drug screening / NDAs required?",("background check","drug screen","nda","fingerprint")),
    ("safety","Compliance","What site safety / PPE / OSHA / LOTO requirements apply?",("safety","ppe","osha","lockout","loto")),
    ("data_privacy","Compliance","Are there data-privacy / security compliance constraints (HIPAA, PCI, CJIS)?",("hipaa","pci","cjis","gdpr","compliance")),
    ("permits","Compliance","Are permits / AHJ approvals / inspections required?",("permit","ahj","inspection","authority having jurisdiction")),
    # quality / acceptance
    ("acceptance","Quality","What are the acceptance criteria and sign-off gate per site?",("acceptance","sign-off","sign off","completion criteria")),
    ("testing_proof","Quality","What test results / photos / proof-of-completion are required?",("test results","photos","proof of completion","punch list")),
    ("warranty","Quality","What workmanship warranty / defect-remediation window applies?",("warranty","defect","remediation","workmanship")),
    ("training","Quality","Is end-user training / knowledge transfer in scope?",("training","knowledge transfer","handoff")),
    ("closeout","Quality","What closeout package (as-builts, warranties, manuals) is required?",("closeout","close-out","as-built","o&m manual")),
    # dependencies
    ("customer_deps","Dependencies","What customer-side prerequisites are required (power, network, access, materials)?",("customer responsib","customer provided","customer-provided","prerequisite")),
    ("third_party","Dependencies","Are there third-party / OEM / GC dependencies or coordination?",("third party","third-party","general contractor","oem coordination")),
    ("spares","Dependencies","Are spares / attic-stock / replacement units required?",("spare","attic stock","attic-stock","replacement unit")),
    ("warranty_oem","Dependencies","Are OEM warranties / RMA handling defined?",("rma","oem warranty","manufacturer warranty")),
]


def universal_head_start(atoms: list[Any]) -> list[dict[str, Any]]:
    """Deal-agnostic PM head-start: the FULL standing quoting/scheduling checklist
    a senior PM asks on any field-services deal, each tagged ``covered`` when the
    deal's atoms already address that topic (presence check over atom text). The
    PM always gets the full head-start (≥50 items); covered ones are pre-checked
    and the rest are genuine gaps to chase. Universal — same checklist every deal."""
    corpus = "\n".join(
        (getattr(a, "raw_text", "") or "") + " " + (getattr(a, "normalized_text", "") or "")
        for a in (atoms or [])
    ).lower()
    out: list[dict[str, Any]] = []
    for fid, cat, q, kws in _HEADSTART:
        covered = any(kw in corpus for kw in kws)
        out.append({"field_id": fid, "category": cat, "summary": q,
                    "kind": "headstart", "covered": covered})
    return out


def generate_gap_questions(srl_checklist: dict[str, Any]) -> list[dict[str, Any]]:
    """From an SRL missing checklist, produce explicit gap questions.

    Returns a list of ``{"field_id","category","summary","kind"}`` dicts,
    one per high-priority missing field. Ordered by the curated map.
    """
    if not srl_checklist:
        return []
    missing_ids = {m.get("field_id") for m in (srl_checklist.get("missing") or [])}
    out: list[dict[str, Any]] = []
    for field_id, question in _GAP_QUESTIONS.items():
        if field_id in missing_ids:
            out.append({
                "field_id": field_id,
                "summary": question,
                "kind": "generated_gap",
            })
    return out


__all__ = [
    "resolve_open_questions",
    "filter_unhelpful_open_questions",
    "generate_gap_questions",
    "universal_head_start",
    "is_answered_question",
    "is_unhelpful_pm_question",
    "ANSWERED_FLAG",
    "NOISE_FLAG",
]
