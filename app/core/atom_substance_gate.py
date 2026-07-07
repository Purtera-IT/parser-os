"""Universal substance gate — drop context-free atom fragments.

An atom only has value to a downstream head (``quote_line_head``,
``site_facility_head``, stakeholder/roster logic, …) if it carries the CONTEXT
that makes it actionable. A bare person name with no role/affiliation is
useless to every head; transcript backchannel ("Yeah.", "Okay.") is not scope.
These fragments inflate atom counts, drag quality scores down, and give the
heads nothing to work with.

This stage removes two classes of context-free fragment, deterministically and
UNIVERSALLY — it keys off STRUCTURE and general role/substance vocabulary, never
a specific name, deal, or domain term:

1. ``drop_contextless_stakeholders`` — a ``stakeholder`` atom that is just a
   name (no role token, no email, no affiliation, no approval/responsibility
   cue) is not a usable stakeholder record. The classifier (or a parser) tagged
   a salutation / sign-off / speaker label as a person; without context it is
   noise, so it is dropped.

2. ``drop_nonsubstantive_fragments`` — a short prose atom whose entire content
   (after removing a leading "Speaker [mm:ss]" transcript label) is
   backchannel / filler ("Yeah.", "Got it.", "Sounds good.") carries no
   deal substance and is dropped.

Both are LOSSLESS at the compiler level: the compiler routes the dropped set
into the retained-suppression ledger, so every removed atom stays auditable.
Conservative by construction — anything that shows a shred of role/substance is
kept, and only ``stakeholder`` / generic-prose types are ever examined.
"""

from __future__ import annotations

import re
from typing import Any

# ── general role / title vocabulary (universal, not a name list) ──
# The presence of any of these tokens near a name means the atom carries the
# ROLE context a head needs. This is deliberately broad and domain-neutral.
_ROLE_TOKENS: frozenset[str] = frozenset(
    {
        "manager", "director", "lead", "leader", "engineer", "architect",
        "analyst", "coordinator", "supervisor", "administrator", "admin",
        "officer", "president", "vp", "svp", "evp", "ceo", "cfo", "cto",
        "coo", "cio", "ciso", "owner", "sponsor", "stakeholder", "approver",
        "executive", "exec", "principal", "partner", "consultant",
        "specialist", "technician", "tech", "foreman", "superintendent",
        "estimator", "buyer", "procurement", "contractor", "subcontractor",
        "installer", "electrician", "designer", "planner", "scheduler",
        "representative", "rep", "liaison", "contact", "poc", "head",
        "chief", "associate", "assistant", "clerk", "receptionist",
        "operator", "dispatcher", "agent", "advisor", "auditor",
        "controller", "treasurer", "secretary", "chair", "chairman",
        "chairperson", "board", "founder", "cofounder", "principal",
        "steward", "custodian", "facilities", "operations", "ops",
        "procurement", "purchasing", "accounts", "billing", "sales",
    }
)

# multi-word role phrases (checked as substrings on the lowered text)
_ROLE_PHRASES: tuple[str, ...] = (
    "project manager", "account executive", "account manager",
    "point of contact", "general contractor", "site manager",
    "site supervisor", "vice president", "team lead", "team member",
    "program manager", "product manager", "operations manager",
    "field engineer", "sales engineer", "solutions architect",
    "network engineer", "security officer", "facility manager",
    "facilities manager", "office manager", "it manager", "it director",
    "decision maker", "key contact", "primary contact",
)

# approval / responsibility / relationship cues — a name attached to one of
# these is a real stakeholder mention ("approved by …", "… will handle …").
_RELATION_RE = re.compile(
    r"\b(?:approv\w*|sign(?:ed|s)?[\s\-]?off|signoff|authoriz\w*|"
    r"responsible|reports?\s+to|report(?:ing)?\s+to|will\s+(?:handle|lead|manage|own|approve)|"
    r"in\s+charge|accountable|oversee\w*|manages?|leads?|owns?|"
    r"decision\s+maker|primary\s+contact|point\s+of\s+contact)\b",
    re.IGNORECASE,
)

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"\b(?:\+?\d[\d().\-\s]{7,}\d)\b")

# a leading "Speaker Name [mm:ss]" transcript label to strip before the
# substance test, so "Jacob Vander-Plaats [03:05] Yeah." is judged on "Yeah.".
_SPEAKER_LABEL_RE = re.compile(
    r"^[A-Z][A-Za-z.'\-]*(?:\s+[A-Z][A-Za-z.'\-]*){0,3}\s*\[\d{1,2}:\d{2}(?::\d{2})?\]\s*"
)

# backchannel / filler tokens (universal conversational acknowledgements).
_FILLER_TOKENS: frozenset[str] = frozenset(
    {
        "yeah", "yea", "yep", "yup", "yes", "no", "nope", "nah", "ok", "okay",
        "sure", "right", "uh-huh", "uhhuh", "mhm", "mm-hmm", "mmhmm", "mm",
        "hmm", "huh", "oh", "ah", "um", "uh", "er", "well", "so", "like",
        "gotcha", "cool", "nice", "great", "awesome", "perfect", "exactly",
        "totally", "absolutely", "definitely", "agreed", "understood",
        "correct", "indeed", "true", "fine", "good", "alright", "okey",
        "thanks", "thank", "welcome", "please", "sorry", "hi", "hey", "hello",
        "bye", "goodbye", "cheers", "anyway", "anyways", "basically",
    }
)

_STAKEHOLDER = "stakeholder"
# Types eligible for the filler test — only generic prose buckets, so a typed
# scope/exclusion/quantity/task/BOM atom is never at risk.
_FILLER_ELIGIBLE = frozenset({"scope_item", "entity", "note"})


def _atom_type_str(atom: Any) -> str:
    at = getattr(atom, "atom_type", None)
    return at.value if hasattr(at, "value") else str(at or "")


def _atom_text(atom: Any) -> str:
    text = (
        getattr(atom, "raw_text", None)
        or getattr(atom, "normalized_text", None)
        or ""
    )
    if not text:
        val = getattr(atom, "value", None)
        if isinstance(val, dict):
            text = str(val.get("text") or val.get("name") or "")
    return str(text).strip()


def _atom_value(atom: Any) -> dict:
    val = getattr(atom, "value", None)
    return val if isinstance(val, dict) else {}


def _has_role_context(text: str, value: dict, entity_keys: list[str]) -> bool:
    """True when a stakeholder atom carries enough context to be actionable:
    a role/title, email, phone, affiliation, or approval/responsibility cue."""
    # Structured fields the classifier may have filled.
    for field in ("role", "title", "email", "position", "affiliation", "org", "organization", "department"):
        v = value.get(field)
        if isinstance(v, str) and v.strip():
            return True
    # Entity keys that anchor the person to an org / email / role.
    for k in entity_keys or []:
        ks = str(k)
        if ks.startswith(("org:", "email:", "role:", "company:", "vendor:")):
            return True
    lowered = text.lower()
    if _EMAIL_RE.search(text) or _PHONE_RE.search(text):
        return True
    if _RELATION_RE.search(text):
        return True
    for phrase in _ROLE_PHRASES:
        if phrase in lowered:
            return True
    tokens = {re.sub(r"[^a-z]", "", t) for t in lowered.split()}
    if tokens & _ROLE_TOKENS:
        return True
    return False


def _looks_like_bare_name(text: str) -> bool:
    """True when the text is essentially just a personal name / salutation
    fragment — short, name-shaped tokens, no digits, no role words. This is the
    shape of a useless bare-name stakeholder ("Eddie,", "Tom Amble.",
    "Patrick Kelly")."""
    stripped = text.strip().strip(".,;:!?")
    if not stripped:
        return True
    words = stripped.split()
    # A name is a handful of tokens at most; anything longer is a sentence.
    if len(words) > 4:
        return False
    for w in words:
        # allow initials / hyphenated / apostrophe'd name tokens only
        if not re.fullmatch(r"[A-Za-z][A-Za-z.'\-]*", w):
            return False
    return True


def drop_contextless_stakeholders(atoms: list[Any]) -> tuple[list[Any], list[Any]]:
    """Partition into (kept, dropped). A ``stakeholder`` atom with no
    role/affiliation/contact context AND a bare-name shape is dropped — it is a
    salutation / sign-off / speaker label mis-typed as a person and is useless
    to every downstream head. Everything else is kept untouched."""
    kept: list[Any] = []
    dropped: list[Any] = []
    for atom in atoms:
        if _atom_type_str(atom) != _STAKEHOLDER:
            kept.append(atom)
            continue
        text = _atom_text(atom)
        value = _atom_value(atom)
        entity_keys = list(getattr(atom, "entity_keys", None) or [])
        if _has_role_context(text, value, entity_keys):
            kept.append(atom)
            continue
        if _looks_like_bare_name(text):
            dropped.append(atom)
            continue
        kept.append(atom)
    return kept, dropped


def _content_tokens(text: str) -> list[str]:
    probe = _SPEAKER_LABEL_RE.sub("", text).strip()
    return [re.sub(r"[^a-z0-9]", "", t) for t in probe.lower().split() if t.strip()]


def drop_nonsubstantive_fragments(atoms: list[Any]) -> tuple[list[Any], list[Any]]:
    """Partition into (kept, dropped). A generic-prose atom (scope_item / entity
    / note) whose entire content — after stripping a leading transcript speaker
    label — is backchannel/filler is dropped as non-substantive. Conservative:
    only fires when EVERY content token is filler, so any real deal word keeps
    the atom."""
    kept: list[Any] = []
    dropped: list[Any] = []
    for atom in atoms:
        if _atom_type_str(atom) not in _FILLER_ELIGIBLE:
            kept.append(atom)
            continue
        text = _atom_text(atom)
        tokens = [t for t in _content_tokens(text) if t]
        # Only judge short utterances; a long paragraph is never "just filler".
        if not tokens or len(tokens) > 5:
            kept.append(atom)
            continue
        if all(t in _FILLER_TOKENS for t in tokens):
            dropped.append(atom)
            continue
        kept.append(atom)
    return kept, dropped


def apply_substance_gate(atoms: list[Any]) -> tuple[list[Any], list[Any]]:
    """Run both drops. Returns (kept, dropped). ``dropped`` is the union across
    passes; the compiler routes it into the retained-suppression ledger."""
    kept, dropped_a = drop_contextless_stakeholders(atoms)
    kept, dropped_b = drop_nonsubstantive_fragments(kept)
    return kept, dropped_a + dropped_b


__all__ = [
    "apply_substance_gate",
    "drop_contextless_stakeholders",
    "drop_nonsubstantive_fragments",
]
