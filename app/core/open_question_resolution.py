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
    "generate_gap_questions",
    "is_answered_question",
    "ANSWERED_FLAG",
]
