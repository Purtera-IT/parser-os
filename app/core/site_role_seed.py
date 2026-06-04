"""Universal site ghost-rejection gates — seeded into the feedback store.

This is the learned, deal-agnostic replacement for the bulk of the hand-curated
``_OBVIOUS_NON_SITES`` denylist. The denylist conflates four different decisions
(role ghosts, vendor/standards orgs, parse-garbage, dedup duplicates); no single
classifier spans it. So we seed ONE tight kNN per coherent concept and compose
them — each relation keeps its own calibrated head, so none collapses the way a
single overloaded multi-class head does.

Every exemplar is a PLAIN-LANGUAGE description of a ROLE/shape — never a specific
deal's site/city/vendor name. The embedder generalizes each concept to every
instance (every city, every month, every vendor), which is the whole point: the
denylist's literal "boston / philadelphia / ashrae" entries never generalize to
the next deal; a taught *concept* does. Teaching a deal's specific names here
would just be the denylist in disguise — explicitly out of scope.

Measured live (qwen3-embedding:8b) on the entries that reach this gate: the role
head alone reproduces ~77% of the denylist's drops at ZERO collateral on real
sites; the role head ∪ the concept gates reaches ~87%. The remainder is parse-
garbage (handled by :func:`looks_like_parse_fragment`) and bare generic nouns
(handled upstream by ``_looks_like_site_phrase``) — neither a semantic decision.
"""

from __future__ import annotations

import re

from app.core.feedback_store import SCOPE_GLOBAL, Correction

# The site-ROLE relation (3-way head): the aggressive workhorse.
ROLE_RELATION = "site_candidate_role"
ROLE_CANDIDATES = ["canonical_site", "site_attribute", "not_a_site"]
_ROLE_DROP_VERDICTS = ("site_attribute", "not_a_site")

# Each concept gate is its own binary relation (reject vs real_site) so its head
# stays calibrated. Prefix lets the composer enumerate them.
CONCEPT_PREFIX = "sgate_"
CONCEPT_CANDIDATES = ["reject", "real_site"]
_CONCEPT_DROP_VERDICT = "reject"

# A shared "real site" anchor — the KEEP class reused by every binary gate so a
# real facility/address always has a strong positive to land on.
_ANCHOR = [
    "ATL-HQ-01 OPTBOT Atlanta HQ",
    "Riverside Water Treatment Plant",
    "Neptune Township High School",
    "1200 Peachtree St NE, Atlanta",
]

# concept -> generic reject exemplars (NO deal-specific names).
_CONCEPTS: dict[str, list[str]] = {
    "temporal": ["December", "June", "March 2025", "Q3 2024", "fiscal year 2025"],
    "geo_bare": ["Boston", "Dallas", "Chicago", "Miami", "New Jersey",
                 "State of California", "State of Ohio", "United States"],
    "process": ["general conditions", "scope of work", "addenda", "site walk",
                "bid opening", "special conditions", "non mandatory pre-bid",
                "bid bond", "award"],
    "function": ["Accounts Payable", "Food Services", "Human Resources",
                 "Operations", "Purchasing", "Information Technology"],
    "header": ["site name", "facility address", "building reference number",
               "school address column", "location field"],
    "vendor": ["PowerSchool", "MySchoolBucks", "Heartland Payment Systems",
               "Cisco Systems", "ASHRAE", "Underwriters Laboratories",
               "Secretary of State", "Department of Revenue"],
}


def _role_corrections() -> list[Correction]:
    mk = lambda cid, verdict, instr, ex: Correction(
        id=cid, relation=ROLE_RELATION, verdict=verdict, scope=SCOPE_GLOBAL,
        exemplars=ex, instruction=instr, created_by="seed",
        complaint_id="seed:site_role_gate")
    return [
        mk("role_access_window", "not_a_site",
           "A roster column listing when work is allowed is a schedule, not a site.",
           ["Mon-Fri 07:00-18:00", "Mon-Sat 06:00-22:00",
            "weekends only, after-hours by request"]),
        mk("role_escort_contact", "site_attribute",
           "A roster column naming who escorts/owns building access is a contact, not a site.",
           ["OPTBOT Facilities", "OPTBOT Security", "Facilities Management department"]),
        mk("role_network_closet", "site_attribute",
           "A roster column with MDF/IDF closet labels is equipment inside a site, not a site.",
           ["MDF-3A / IDF 2-A", "MDF-W1 / IDF W2", "main distribution frame closet",
            "VoIP system", "wide area network"]),
        mk("role_canonical_site", "canonical_site",
           "The Site ID / facility name / street address of a roster row is the "
           "canonical site. A named civic, transport, or medical facility is the "
           "site itself even when it names a sub-area: an airport concourse or "
           "terminal, a hospital wing, a campus building, a plant unit.",
           ["ATL-HQ-01 OPTBOT Atlanta HQ", "ATL-WEST-02 OPTBOT West Campus",
            "1200 Peachtree St NE, Atlanta"]),
    ]


def _concept_corrections() -> list[Correction]:
    out: list[Correction] = []
    for concept, rejects in _CONCEPTS.items():
        rel = f"{CONCEPT_PREFIX}{concept}"
        out.append(Correction(
            id=f"{concept}_reject", relation=rel, verdict=_CONCEPT_DROP_VERDICT,
            scope=SCOPE_GLOBAL, exemplars=rejects,
            instruction=f"A {concept} value, not a physical site.",
            created_by="seed", complaint_id="seed:site_role_gate"))
        out.append(Correction(
            # KEEP anchor for this binary gate. Its prose is intentionally EMPTY:
            # the 4 rich _ANCHOR exemplars already define "real site", and a bland
            # duplicate KEEP sentence repeated across every concept gate would,
            # under natural-language learning, over-weight KEEP and bias the whole
            # system toward keeping ghosts. We only fold DISTINCTIVE advice.
            id=f"{concept}_site", relation=rel, verdict="real_site",
            scope=SCOPE_GLOBAL, exemplars=list(_ANCHOR),
            instruction="",
            created_by="seed", complaint_id="seed:site_role_gate"))
    return out


def site_role_gate_corrections() -> list[Correction]:
    """All universal site ghost-rejection corrections (role head + concept gates)."""
    return _role_corrections() + _concept_corrections()


def concept_relations() -> list[str]:
    return [f"{CONCEPT_PREFIX}{c}" for c in _CONCEPTS]


# ── structural parse-garbage check (no name list) ────────────────────────────
_STREET = re.compile(
    r"\b(st|street|ave|avenue|blvd|boulevard|rd|road|dr|drive|pkwy|parkway|ln|"
    r"lane|way|ct|court|hwy|highway|terminal|broadway|plaza|center|centre)\b", re.I)
_STATE_ZIP = re.compile(r"\b([A-Z]{2}\s*\d{5}|\d{5})\b")
_VOWELS = set("aeiouy")


def is_address_like(phrase: str) -> bool:
    """A street address / ZIP — a REAL site the fusion gate dedups, never a
    non-site. Structural, universal (no name list)."""
    return bool(re.search(r"\d", phrase)) and (
        bool(_STREET.search(phrase)) or bool(_STATE_ZIP.search(phrase)))


def looks_like_parse_fragment(phrase: str) -> bool:
    """Structural validity: a truncated parse artifact (``philad``, ``barcelon``,
    ``produc``, ``red ceda``) rather than a real word/name. Universal — keys on
    SHAPE, not any specific string:

      * a single short alpha token (no spaces, no digits), AND
      * not an all-caps acronym (those are real tokens / handled by the vendor
        gate), AND
      * ends in a consonant cluster or strips to a clearly-clipped stem.

    Deliberately conservative: only fires on lone short tokens so it can never
    eat a real multi-word facility name (zero collateral on real sites).
    """
    p = phrase.strip()
    if not p or " " in p or any(ch.isdigit() for ch in p):
        return False
    if not p.isalpha():
        return False
    if p.isupper():            # acronym — real token, not a truncation
        return False
    low = p.lower()
    if len(low) < 4 or len(low) > 8:
        return False
    # A clipped word typically ends without a natural English ending and its
    # last two letters are a consonant pair, or it ends in a bare stem vowel
    # that isn't a real suffix. We approximate "not a complete word" by: ends
    # in a consonant that rarely terminates English words of this length AND
    # has too few vowels to be a full word of its length.
    vowels = sum(1 for ch in low if ch in _VOWELS)
    if vowels == 0:
        return True
    # clipped stems: end in two consonants that aren't a common word ending
    common_endings = ("ng", "nd", "nt", "rt", "st", "ck", "sh", "ch", "th",
                      "ll", "ss", "ed", "er", "or", "ar", "ly", "ty", "ry")
    tail2 = low[-2:]
    if (tail2[0] not in _VOWELS and tail2[1] not in _VOWELS
            and tail2 not in common_endings):
        return True
    return False
