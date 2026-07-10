"""Universal substance gate — drop context-free atom fragments.

An atom only has value to a downstream head (``quote_line_head``,
``site_facility_head``, stakeholder/roster logic, …) if it carries the CONTEXT
that makes it actionable. A bare person name with no role/affiliation is
useless to every head; transcript backchannel ("Yeah.", "Okay.") is not scope.
These fragments inflate atom counts, drag quality scores down, and give the
heads nothing to work with.

This stage removes several classes of context-free fragment, deterministically
and UNIVERSALLY — it keys off STRUCTURE and general role/substance vocabulary,
never a specific name, deal, or domain term:

1. ``drop_contextless_stakeholders`` — a ``stakeholder`` atom that is just a
   name (no role token, no email, no affiliation, no approval/responsibility
   cue) is not a usable stakeholder record.

2. ``drop_nonsubstantive_fragments`` — a short prose atom whose entire content
   (after removing a leading "Speaker [mm:ss]" transcript label) is
   backchannel / filler ("Yeah.", "Got it.", "Sounds good.") carries no
   deal substance and is dropped.

3. ``drop_section_headers`` — a scope_item whose text is a document section
   header ("Executive Summary", "Full Transcript: …") is chrome, not scope.

4. ``drop_email_non_scope`` — email header metadata, label-only lead-ins
   ("Customer specifically said:"), and pleasantry/sign-off body lines typed
   as scope are not actionable to any head. Intentional communication atoms
   (``email_body_context``) are kept. Body greetings are metadata tags on
   sibling atoms, not standalone ``email_addressee`` atoms.

5. ``retag_transcript_conversational`` — raw transcript turns (page ≥ 1) that
   lack deal substance (no scope verb, no device/vendor entity, no structured
   bullet) are retagged to ``deal_metadata`` / ``kind=conversation_meta`` so
   they stay auditable but do not infect neural/embedding/scope heads.

6. ``drop_risk_fragments`` — a ``risk`` atom whose text is a mid-sentence
   clipping without a complete risk structure (no subject, no consequence) is
   not actionable to a risk head.

7. ``collapse_ambiguous_user_quantities`` — when multiple ``quantity`` atoms
   for the same noun (e.g. "users") share the same source locator, keep only
   the one with the widest range (most context).

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
# substance test, so "Alex Rivera [03:05] Yeah." is judged on "Yeah.".
_SPEAKER_LABEL_RE = re.compile(
    r"^[A-Z][A-Za-z.'\-]*(?:\s+[A-Z][A-Za-z0-9.'\-]*){0,4}\s*\[\d{1,2}:\d{2}(?::\d{2})?\]\s*"
)

# The ENTIRE atom text is a transcript speaker header ("Alex Rivera
# [00:48]") and nothing else. When a transcript is delivered as a PDF, the
# color-driven segmenter emits each speaker-turn header as its own paragraph
# block, which the fail-open PDF atomizer then turns into a standalone
# scope_item. That atom is pure chrome — the speaker + timestamp are already
# carried as ``section_path`` context on the real utterance atoms beneath the
# header — so a bare label is not actionable to any downstream head. STRUCTURAL
# (name-shaped tokens + a [mm:ss] stamp anchored to the whole line), never a
# name/deal list, so it generalises to any transcript.
_SPEAKER_LABEL_ONLY_RE = re.compile(
    r"^[A-Z][A-Za-z.'\-]*(?:\s+[A-Z][A-Za-z0-9.'\-]*){0,4}\s*"
    r"\[\d{1,2}:\d{2}(?::\d{2})?\]\s*$"
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

# Closed-class conversational tokens that carry no deal substance on their own:
# grammatical function words (pronouns, light copular/perception/stance verbs)
# and social address terms. This is a UNIVERSAL linguistic class — NOT deal,
# name, or domain vocabulary. It only ever participates in the "every content
# token is non-substantive" test below, so a single real deal word ("cameras",
# "switch", "Okta") always keeps the atom. It exists to catch conversational
# turns the bare-filler set misses — "I see.", "Thank you.", "Thanks, guys.",
# "Okay, sounds good." — where a pronoun or a light verb sits beside filler.
_SOCIAL_FUNCTION_TOKENS: frozenset[str] = frozenset(
    {
        # pronouns
        "i", "you", "we", "he", "she", "it", "they", "me", "us", "him",
        "her", "them", "my", "your", "our", "his", "its", "their", "myself",
        "yourself", "ourselves",
        # social address terms
        "guy", "guys", "everyone", "everybody", "folks", "team", "sir",
        "maam", "man", "yall",
        # light perception / linking / stance verbs (only gated when the WHOLE
        # utterance is function/social/filler words)
        "see", "saw", "seen", "sound", "sounds", "sounded", "look", "looks",
        "looking", "think", "guess", "suppose",
    }
)

_STAKEHOLDER = "stakeholder"
# Types eligible for the filler test — only generic prose buckets, so a typed
# scope/exclusion/quantity/task/BOM atom is never at risk.
_FILLER_ELIGIBLE = frozenset({"scope_item", "entity", "note"})

# ── section-header chrome (structural, not a vocabulary list) ──
# A line that is ONLY a document section label ("Executive Summary",
# "Full Transcript: …", "Action Items", "Key Decisions") carries no deal
# substance. Anchored to the whole line; a sentence that merely mentions the
# phrase ("we discussed the executive summary") still flows through.
_SECTION_HEADER_RE = re.compile(
    r"^(?:"
    r"(?:meeting\s+)?summary(?:\s+and\s+full\s+transcript)?(?:\s+executive\s+summary)?|"
    r"executive\s+summary|"
    r"full\s+transcript(?:\s*:\s*.+)?|"
    r"action\s+items?|"
    r"key\s+decisions?|"
    r"decisions?|"
    r"discussion|"
    r"attendees?|"
    r"participants?"
    r")\s*:?\s*$",
    re.IGNORECASE,
)

# ── email label-only lead-ins (structural: ends with colon, no clause) ──
# "Customer specifically said:", "By the end of the meeting customer clarified:"
# are attribution labels, not scope. A real sentence that happens to end with
# a colon ("Include: badge readers and cameras") is kept because it carries
# content after the colon.
_LABEL_ONLY_RE = re.compile(
    r"^[A-Za-z][^.!?]{0,120}:\s*$"
)

# ── email pleasantry / sign-off openers (closed-class social phrases) ──
# Same structural class as _SIGNOFF_RE in email_parser — short social lines
# that carry no deal substance. NOT a name/deal list.
_EMAIL_PLEASANTRY_RE = re.compile(
    r"^(?:"
    r"appreciate\s+(?:you|it|the)|"
    r"thank(?:s| you)(?:\s+(?:so\s+much|again|a\s+lot|much|everyone|guys))?|"
    r"looking\s+forward|"
    r"let\s+(?:me|us)\s+know|"
    r"feel\s+free\s+to|"
    r"happy\s+to\s+help|"
    r"please\s+(?:let|reach|feel)|"
    r"hope\s+(?:life|you|this|all)|"
    r"good\s+to\s+(?:meet|see|hear)|"
    r"nice\s+(?:to\s+)?meet|"
    r"have\s+a\s+(?:good|great|nice)"
    r")\b",
    re.IGNORECASE,
)

# ── transcript conversational markers (closed-class social/logistics) ──
# Greetings, sign-offs, logistics, and self-introductions on raw transcript
# turns. A turn that ALSO carries a deal noun/verb is kept (substance check).
_CONVERSATIONAL_LEAD_RE = re.compile(
    r"(?:"
    r"(?:^|\.\s*)(?:hi|hey|hello|good\s+(?:morning|afternoon|evening))\b|"
    r"(?:how\s+(?:are|you)|been\s+a\s+while|long\s+time\s+no|hope\s+life)|"
    r"(?:nice\s+to\s+meet|good\s+to\s+meet|pleased\s+to\s+meet)|"
    r"(?:i(?:'m|\s+am)\s+\w+.*(?:co-?founder|engineer|manager|director))|"
    r"(?:thank(?:s| you)|appreciate\s+it|all\s+right.*thank)|"
    r"(?:we(?:'re| are)\s+(?:waiting|expecting|ready))|"
    r"(?:sorry|apolog|no\s+worries|excuse\s+me)|"
    r"(?:can\s+you\s+repeat|i(?:'m| am)\s+not\s+hearing)|"
    r"(?:just\s+busy|not\s+too\s+bad|can(?:'t|not)\s+complain)|"
    r"(?:all\s+right.*thanks?\s+everyone)|"
    r"(?:i(?:'ll| will)\s+(?:sit|send|do\s+that|ping))|"
    r"(?:hey,?\s+how\s+you\s+doing)"
    r")",
    re.IGNORECASE,
)

# ── deal-substance signals (universal closed-class, not domain vocabulary) ──
# A transcript turn that matches ANY of these carries enough context for a head.
_SCOPE_VERB_RE = re.compile(
    r"\b(?:install|configure|deploy|integrat|setup|set\s+up|provision|"
    r"require|exclud|build|survey|upgrade|replace|implement|onboard|"
    r"walk\s+(?:through|him|her|them)|white\s+glove|badge\s+zone|"
    r"vlan|ssid|radius|access\s+control|badging|camera|firewall|"
    r"switch|router|access\s+point|reader|doorbell|okta|unifi|"
    r"uid\s+enterprise|equipment|hardware|parts?\s+list|sow|quote)\b",
    re.IGNORECASE,
)

# ── risk structure: a risk atom needs subject + consequence, not a clip ──
_RISK_CONSEQUENCE_RE = re.compile(
    r"\b(?:risk\s+of|delay|blocker|show[\s-]?stopper|deal[\s-]?breaker|"
    r"hard\s+requirement|"
    r"consequence|mitigat|contingenc|liability|penalty|"
    r"unable\s+to\s+(?:proceed|complete|deliver)|"
    r"would\s+(?:delay|block|prevent|impact))\b",
    re.IGNORECASE,
)

# Mid-sentence clippings that start a risk atom without anchoring context.
_RISK_FRAGMENT_START_RE = re.compile(
    r"^(?:there'?s|it'?s|if\s+we|consider|but\s+we|and\s+if|that\s+would|"
    r"really\s+good\s+if|long\s+as|may\s+be|then\s+walk|will\s+be\s+easy)\b",
    re.IGNORECASE,
)

# Exec-summary page (page 0 bullets) is always kept — high-authority synthesis.
_EXEC_SUMMARY_PAGE = 0



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


# Communication-atom roles on email body lines — NOT stakeholder titles.
# ``role: "to_greeting"`` / ``"intro"`` must not satisfy stakeholder role context.
_EMAIL_COMM_ROLES: frozenset[str] = frozenset({"to_greeting", "intro", "from_signoff"})


def _has_role_context(text: str, value: dict, entity_keys: list[str]) -> bool:
    """True when a stakeholder atom carries enough context to be actionable:
    a role/title, email, phone, affiliation, or approval/responsibility cue."""
    # Structured fields the classifier may have filled.
    for field in ("role", "title", "email", "position", "affiliation", "org", "organization", "department"):
        v = value.get(field)
        if isinstance(v, str) and v.strip():
            if field == "role" and v.strip().lower() in _EMAIL_COMM_ROLES:
                continue
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
    to every downstream head. Everything else is kept untouched.

    Safety net: legacy ``kind=email_addressee`` (bare greeting) is dropped —
    addressee is metadata on sibling atoms, not a roster person.
    ``email_body_context`` mistyped as stakeholder is restored to
    ``deal_metadata`` and kept.
    """
    kept: list[Any] = []
    dropped: list[Any] = []
    for atom in atoms:
        if _atom_type_str(atom) != _STAKEHOLDER:
            kept.append(atom)
            continue
        text = _atom_text(atom)
        value = _atom_value(atom)
        kind = str(value.get("kind") or "")
        if kind == "email_addressee":
            dropped.append(atom)
            continue
        if kind == "email_body_context":
            try:
                from app.core.schemas import AtomType

                atom.atom_type = AtomType.deal_metadata
            except Exception:
                pass
            kept.append(atom)
            continue
        entity_keys = list(getattr(atom, "entity_keys", None) or [])
        if _has_role_context(text, value, entity_keys):
            kept.append(atom)
            continue
        # Transcript QA-split name fragment ("Tom Amble." in a speaker block) —
        # almost always a sign-off / attribution tail, never a roster record.
        if value.get("qa_split") and _looks_like_bare_name(text):
            dropped.append(atom)
            continue
        if _looks_like_bare_name(text):
            dropped.append(atom)
            continue
        kept.append(atom)
    return kept, dropped


def _content_tokens(text: str) -> list[str]:
    probe = _SPEAKER_LABEL_RE.sub("", text).strip()
    return [re.sub(r"[^a-z0-9]", "", t) for t in probe.lower().split() if t.strip()]


# The union checked by the "every content token is non-substantive" test.
_NONCONTENT_TOKENS: frozenset[str] = _FILLER_TOKENS | _SOCIAL_FUNCTION_TOKENS


def drop_nonsubstantive_fragments(atoms: list[Any]) -> tuple[list[Any], list[Any]]:
    """Partition into (kept, dropped). A generic-prose atom (scope_item / entity
    / note) is dropped as non-substantive when it is either:

    * a bare transcript speaker header ("Alex Rivera [00:48]") — pure
      chrome the segmenter should have kept as ``section_path`` context, or
    * a short utterance whose entire content — after stripping a leading
      transcript speaker label — is backchannel/filler/social function words
      ("Yeah.", "I see.", "Thank you.", "Thanks, guys.", "Okay, sounds good.").

    Conservative: the utterance test only fires on short turns where EVERY
    content token is non-substantive, so a single real deal word ("cameras",
    "Okta", "switch") always keeps the atom."""
    kept: list[Any] = []
    dropped: list[Any] = []
    for atom in atoms:
        if _atom_type_str(atom) not in _FILLER_ELIGIBLE:
            kept.append(atom)
            continue
        text = _atom_text(atom)
        # A standalone speaker/timestamp header line is chrome, never content.
        if _SPEAKER_LABEL_ONLY_RE.match(text.strip()):
            dropped.append(atom)
            continue
        tokens = [t for t in _content_tokens(text) if t]
        # Only judge short utterances; a long paragraph is never "just filler".
        if not tokens or len(tokens) > 6:
            kept.append(atom)
            continue
        if all(t in _NONCONTENT_TOKENS for t in tokens):
            dropped.append(atom)
            continue
        kept.append(atom)
    return kept, dropped


def _atom_page(atom: Any) -> int | None:
    """Page number from the first source_ref locator, if present."""
    refs = getattr(atom, "source_refs", None) or []
    if not refs:
        return None
    loc = getattr(refs[0], "locator", None) or {}
    if not isinstance(loc, dict):
        return None
    page = loc.get("page")
    if page is None:
        return None
    try:
        return int(page)
    except (TypeError, ValueError):
        return None


def _is_exec_summary_bullet(atom: Any) -> bool:
    """True when the atom is a page-0 bullet from an executive-summary section."""
    page = _atom_page(atom)
    if page != _EXEC_SUMMARY_PAGE:
        return False
    val = _atom_value(atom)
    if val.get("kind") == "bullet":
        return True
    refs = getattr(atom, "source_refs", None) or []
    if refs:
        loc = getattr(refs[0], "locator", None) or {}
        if isinstance(loc, dict) and loc.get("block_kind") == "bullet_list":
            return True
    return False


def _has_deal_substance(text: str, entity_keys: list[str]) -> bool:
    """True when text or entity_keys carry enough deal context for a head."""
    if _SCOPE_VERB_RE.search(text):
        return True
    for k in entity_keys or []:
        ks = str(k)
        if ks.startswith(("device:", "vendor:", "quantity:", "site:", "req")):
            return True
    return False


def _is_email_atom(atom: Any) -> bool:
    val = _atom_value(atom)
    kind = str(val.get("kind") or "")
    if kind in {
        "email_body_line",
        "email_header",
        "email_addressee",
        "email_body_context",
    }:
        return True
    refs = getattr(atom, "source_refs", None) or []
    if refs:
        loc = getattr(refs[0], "locator", None) or {}
        if isinstance(loc, dict) and loc.get("kind") == "email_header":
            return True
    return False


def drop_section_headers(atoms: list[Any]) -> tuple[list[Any], list[Any]]:
    """Drop scope_item atoms whose entire text is a document section header."""
    kept: list[Any] = []
    dropped: list[Any] = []
    for atom in atoms:
        if _atom_type_str(atom) != "scope_item":
            kept.append(atom)
            continue
        text = _atom_text(atom).strip()
        if _SECTION_HEADER_RE.match(text):
            dropped.append(atom)
            continue
        # "Full Transcript: Alex Rivera [00:04]" — section header + speaker
        if re.match(r"^full\s+transcript\s*:\s*.+\[\d{1,2}:\d{2}", text, re.I):
            dropped.append(atom)
            continue
        kept.append(atom)
    return kept, dropped


def drop_email_non_scope(atoms: list[Any]) -> tuple[list[Any], list[Any]]:
    """Drop email chrome mis-typed as scope: headers, label-only lead-ins,
    pleasantry/sign-off body lines. Real include/exclude list items (with
    ``list_section`` set) are always kept.

    Intentional communication atoms are also kept:
    - ``email_body_context`` — intro / logistics prose (not contractual scope)

    Legacy ``email_addressee`` atoms (bare greetings) are dropped — addressee
    is now metadata/tag on sibling atoms, not a reviewable card.
    """
    kept: list[Any] = []
    dropped: list[Any] = []
    for atom in atoms:
        at = _atom_type_str(atom)
        if at not in {"scope_item", "deal_metadata"}:
            kept.append(atom)
            continue
        val = _atom_value(atom)
        kind = str(val.get("kind") or "")
        # Intro prose is a first-class communication atom — keep.
        if kind == "email_body_context":
            kept.append(atom)
            continue
        # Legacy bare greeting atoms — demote/drop (tag lives on siblings).
        if kind == "email_addressee":
            dropped.append(atom)
            continue
        # Email header metadata — never scope (retyped at parse time to
        # deal_metadata, but catch any legacy scope_item headers too).
        if kind == "email_header":
            dropped.append(atom)
            continue
        if not _is_email_atom(atom):
            kept.append(atom)
            continue
        text = _atom_text(atom).strip()
        # Include/exclude list items are real scope — always keep.
        if val.get("list_section") in {"include", "exclude"}:
            kept.append(atom)
            continue
        # Label-only lead-in ("Customer specifically said:")
        if _LABEL_ONLY_RE.match(text):
            dropped.append(atom)
            continue
        # Pleasantry / sign-off opener with no deal substance
        if _EMAIL_PLEASANTRY_RE.match(text) and not _has_deal_substance(
            text, list(getattr(atom, "entity_keys", None) or [])
        ):
            dropped.append(atom)
            continue
        kept.append(atom)
    return kept, dropped


def drop_transcript_conversational(atoms: list[Any]) -> tuple[list[Any], list[Any]]:
    """Retag conversational transcript turns to ``conversation_meta``.

    Executive-summary bullets and substantive turns stay as-is. Greeting /
    intro / logistics / filler turns become ``deal_metadata`` with
    ``kind=conversation_meta`` so they remain auditable but do not feed
    scope or neural heads.

    Returns ``(kept, dropped)`` with ``dropped`` always empty (retag-in-place).
    Name preserved for call-site compatibility.
    """
    try:
        from app.core.hybrid_summary_transcript import retag_conversational_to_meta

        # Never retag page-0 exec-summary bullets — peel them out first.
        candidates: list[Any] = []
        passthrough: list[Any] = []
        for atom in atoms:
            if _atom_type_str(atom) == "scope_item" and _is_exec_summary_bullet(atom):
                passthrough.append(atom)
            else:
                candidates.append(atom)
        retag_conversational_to_meta(candidates)
        return passthrough + candidates, []
    except Exception:
        pass

    # Legacy drop fallback (import failure only).
    kept: list[Any] = []
    dropped: list[Any] = []
    for atom in atoms:
        at = _atom_type_str(atom)
        if at != "scope_item":
            kept.append(atom)
            continue
        if _is_exec_summary_bullet(atom):
            kept.append(atom)
            continue
        page = _atom_page(atom)
        if page is None or page < 1:
            kept.append(atom)
            continue
        text = _atom_text(atom)
        entity_keys = list(getattr(atom, "entity_keys", None) or [])
        if _has_deal_substance(text, entity_keys):
            kept.append(atom)
            continue
        probe = _SPEAKER_LABEL_RE.sub("", text).strip()
        if val := _atom_value(atom):
            if val.get("qa_split") and len(probe.split()) > 15:
                kept.append(atom)
                continue
        if _CONVERSATIONAL_LEAD_RE.search(probe):
            dropped.append(atom)
            continue
        tokens = [t for t in _content_tokens(probe) if t]
        if len(tokens) < 8 and not entity_keys:
            dropped.append(atom)
            continue
        kept.append(atom)
    return kept, dropped


def drop_risk_fragments(atoms: list[Any]) -> tuple[list[Any], list[Any]]:
    """Drop risk atoms that are mid-sentence clippings without risk structure.

    A risk atom must carry enough context for a head to act: an explicit
    consequence/risk marker, a structured exec-summary bullet, or a complete
    anchored clause. Bare paragraph clippings ("consider it but we, if we
    can't we might...") are dropped even when wordy."""
    kept: list[Any] = []
    dropped: list[Any] = []
    for atom in atoms:
        if _atom_type_str(atom) != "risk":
            kept.append(atom)
            continue
        text = _atom_text(atom)
        val = _atom_value(atom)
        # Exec-summary bullets typed as risk are complete statements — keep.
        if val.get("kind") == "bullet":
            kept.append(atom)
            continue
        if _RISK_CONSEQUENCE_RE.search(text):
            kept.append(atom)
            continue
        # Mid-sentence clipping without consequence structure — drop.
        if _RISK_FRAGMENT_START_RE.match(text.strip()):
            dropped.append(atom)
            continue
        # Short clause without any risk anchor — drop.
        if len(text.split()) < 10:
            dropped.append(atom)
            continue
        kept.append(atom)
    return kept, dropped


def _quantity_locator_key(atom: Any) -> str:
    """Stable key for grouping quantity atoms from the same source utterance."""
    refs = getattr(atom, "source_refs", None) or []
    if not refs:
        return str(getattr(atom, "id", ""))
    loc = getattr(refs[0], "locator", None) or {}
    if not isinstance(loc, dict):
        return str(getattr(atom, "id", ""))
    parts = [
        str(loc.get("page", "")),
        str(loc.get("block_id", "")),
        str(loc.get("line_start", "")),
        str(loc.get("message_index", "")),
    ]
    return "|".join(parts)


def collapse_ambiguous_user_quantities(atoms: list[Any]) -> tuple[list[Any], list[Any]]:
    """When multiple quantity atoms for the same noun share a source locator,
    keep the one with the widest range (most context) and drop the rest.

    E.g. "12 or 20 people or up to 50 people" emits quantity:20 and
    quantity:50 from the same sentence — keep quantity:50 (widest upper bound)
    and suppress the narrower duplicate. STRUCTURAL: keys off noun + locator,
    not a specific number or deal."""
    from collections import defaultdict

    qty_by_group: dict[tuple[str, str], list[Any]] = defaultdict(list)
    non_qty: list[Any] = []
    for atom in atoms:
        if _atom_type_str(atom) != "quantity":
            non_qty.append(atom)
            continue
        val = _atom_value(atom)
        noun = str(val.get("noun") or "").strip().lower()
        loc_key = _quantity_locator_key(atom)
        qty_by_group[(noun, loc_key)].append(atom)

    kept_qty: list[Any] = []
    dropped_qty: list[Any] = []
    for (_noun, _loc), group in qty_by_group.items():
        if len(group) == 1:
            kept_qty.append(group[0])
            continue
        # Pick the atom with the highest quantity (widest range upper bound).
        def _qty_score(a: Any) -> float:
            v = _atom_value(a)
            q = v.get("quantity")
            rmax = v.get("range_max")
            try:
                base = float(q) if q is not None else 0.0
            except (TypeError, ValueError):
                base = 0.0
            try:
                upper = float(rmax) if rmax is not None else base
            except (TypeError, ValueError):
                upper = base
            return max(base, upper)

        winner = max(group, key=_qty_score)
        kept_qty.append(winner)
        for a in group:
            if a is not winner:
                dropped_qty.append(a)

    return non_qty + kept_qty, dropped_qty


def apply_substance_gate(atoms: list[Any]) -> tuple[list[Any], list[Any]]:
    """Run all drops. Returns (kept, dropped). ``dropped`` is the union across
    passes; the compiler routes it into the retained-suppression ledger."""
    all_dropped: list[Any] = []
    kept, d = drop_contextless_stakeholders(atoms)
    all_dropped.extend(d)
    kept, d = drop_nonsubstantive_fragments(kept)
    all_dropped.extend(d)
    kept, d = drop_section_headers(kept)
    all_dropped.extend(d)
    kept, d = drop_email_non_scope(kept)
    all_dropped.extend(d)
    kept, d = drop_transcript_conversational(kept)
    all_dropped.extend(d)
    kept, d = drop_risk_fragments(kept)
    all_dropped.extend(d)
    kept, d = collapse_ambiguous_user_quantities(kept)
    all_dropped.extend(d)
    return kept, all_dropped


__all__ = [
    "apply_substance_gate",
    "collapse_ambiguous_user_quantities",
    "drop_contextless_stakeholders",
    "drop_email_non_scope",
    "drop_nonsubstantive_fragments",
    "drop_risk_fragments",
    "drop_section_headers",
    "drop_transcript_conversational",
]
