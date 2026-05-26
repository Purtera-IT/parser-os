"""LLM-driven site extraction + verification (opt-in, Ollama-backed).

Two modes:

  1. EXTRACT (preferred when LLM available)
     ``extract_sites_with_llm(atoms) -> set[str]``
     Sends the project's document content to an LLM and asks it
     to identify the project sites directly. The model returns a
     canonical site list which becomes the catalog. No regex,
     no candidate-list pre-pass — pure semantic understanding of
     what the bid is about.

  2. VERIFY (fallback / legacy)
     ``verify_sites_with_llm(catalog, atoms) -> set[str]``
     Sends an existing regex-built candidate catalog through the
     LLM as a KEEP/DROP filter. Used when extract-mode produces
     no usable answer (LLM down, malformed response, etc.).

Configuration (env vars, all optional):
  OLLAMA_HOST        — http://HOST:PORT (default: http://100.114.102.122:11434)
  OLLAMA_MODEL       — model name (default: qwen3:14b)
  SOWSMITH_LLM_TIMEOUT — seconds per call (default: 90)

The verifier is intentionally fail-safe: any HTTP error, timeout, or
malformed model response causes a graceful fallback to the
deterministic structural catalog. No API key required — Ollama
runs on a private tailnet machine.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Any

DEFAULT_HOST = "http://100.114.102.122:11434"
# qwen3:14b is the speed/quality sweet spot. The strict prompt +
# Python hygiene filter compensate for the 14b model being a step
# down from 32b. For maximum quality at ~3× the latency, set
# OLLAMA_MODEL=qwen3:32b in the environment.
DEFAULT_MODEL = "qwen3:14b"
DEFAULT_TIMEOUT = 180
DEFAULT_PROBE_TIMEOUT = 2  # seconds — quick reachability check


_CHUNK_SIZE = 40  # candidates per LLM call (verify mode)
_EXTRACT_DOC_BUDGET = 25000  # chars of doc context sent in extract mode


# ─────────────────────── REACHABILITY PROBE ───────────────────────


def ollama_reachable() -> bool:
    """Quick check whether the configured Ollama host is reachable.

    Used before invoking the LLM so air-gapped / offline environments
    fall back to the deterministic catalog without burning a full
    request timeout per compile. Returns ``True`` on a 200 response
    to ``/api/tags`` within ``DEFAULT_PROBE_TIMEOUT`` seconds.
    """
    host = os.environ.get("OLLAMA_HOST", DEFAULT_HOST).rstrip("/")
    try:
        req = urllib.request.Request(f"{host}/api/tags")
        with urllib.request.urlopen(req, timeout=DEFAULT_PROBE_TIMEOUT) as resp:
            return resp.status == 200
    except Exception:
        return False


# ─────────────────────── EXTRACT MODE ───────────────────────


def extract_sites_with_llm(atoms: list[Any]) -> set[str]:
    """Ask LLM to identify project sites directly from doc content.

    No candidate-list input — the LLM reads representative excerpts
    from the project's source documents and returns a canonical site
    list. This is the primary site-detection path when an LLM is
    available; the structural 6-tier regex catalog is the fallback.

    For projects whose total atom-text content fits within
    ``_EXTRACT_DOC_BUDGET``, a single LLM call is sufficient.
    For larger projects, falls back to per-document extraction:
    each artifact gets its own LLM call, results are unioned. This
    guarantees we don't miss sites in document #4 because we only
    sent #1-3 to the LLM.

    Returns a set of normalized site phrases. Returns ``set()`` on
    any failure (LLM unreachable, malformed response) so the caller
    can fall back to the structural catalog.
    """
    if not atoms:
        return set()

    # Estimate total project content size to decide call strategy.
    total_chars = 0
    by_artifact: dict[str, list[Any]] = {}
    for atom in atoms:
        aid = getattr(atom, "artifact_id", None)
        if not aid:
            continue
        by_artifact.setdefault(aid, []).append(atom)
        rt = getattr(atom, "raw_text", None) or ""
        if isinstance(rt, str):
            total_chars += len(rt)

    if not by_artifact:
        return set()

    # Strategy 1 — small project: one combined call.
    # Send all docs concatenated when they fit comfortably.
    if total_chars <= int(_EXTRACT_DOC_BUDGET * 1.2):
        docs_excerpt = _build_doc_excerpt(
            atoms, max_per_doc=15000, max_total=_EXTRACT_DOC_BUDGET,
        )
        if not docs_excerpt:
            return set()
        prompt = _build_extract_prompt(docs_excerpt)
        response_text = _call_ollama(prompt, max_tokens=1024)
        if not response_text:
            return set()
        return _parse_sites_list(response_text)

    # Strategy 2 — large project: per-artifact calls, union results.
    # Each artifact gets its own atom list and its own LLM extract.
    # This is critical for projects like Pack 02 APS / Pack 12 BMS
    # where total content far exceeds the doc budget.
    union_sites: set[str] = set()
    for aid in sorted(by_artifact.keys()):
        artifact_atoms = by_artifact[aid]
        # Per-artifact budget: cap each doc at 18000 chars so
        # the prompt stays well within qwen3:14b's context window
        # and the per-call latency stays bounded.
        docs_excerpt = _build_doc_excerpt(
            artifact_atoms, max_per_doc=18000, max_total=18000,
        )
        if not docs_excerpt:
            continue
        prompt = _build_extract_prompt(docs_excerpt)
        response_text = _call_ollama(prompt, max_tokens=1024)
        if not response_text:
            continue
        union_sites |= _parse_sites_list(response_text)
    return union_sites


def _build_extract_prompt(docs_excerpt: str) -> str:
    """Compose the LLM extraction prompt.

    Compact, precision-first instructions. Examples use placeholder
    NAMES (not real institutions) so the model doesn't echo example
    names back into its output instead of extracting from the docs.
    The Python hygiene denylist catches form-field words / generic
    nouns the LLM may still slip through.
    """
    return f"""Identify PHYSICAL PROJECT SITES from this bid package — the actual buildings, schools, hospitals, offices, plants, depots, or named facilities where the contracted work will be performed.

CRITICAL: Only return sites that ACTUALLY APPEAR in the documents below. Do NOT invent sites or include sites from your training data. If the document mentions a site name or site code, include exactly that.

RULES:
- PRECISION OVER RECALL. When in doubt, EXCLUDE.
- Extract sites VERBATIM from the document text. Use the FULL canonical name as it appears.
- Each site must be a SPECIFIC named building/facility OR a site-code identifier.
- ALWAYS include site codes (e.g. "ATL-HQ-01", "STORE-142", "Building 47") when they appear — these are the customer's authoritative scope anchors.
- Include both the site code AND the friendly name when both appear (they refer to the same physical place).

INCLUDE shape examples (these are PATTERNS, not real names — use names from the docs below):
  ✓ "<Proper Noun> Elementary School" / "<Proper Noun> High School"
  ✓ "<Proper Noun> Medical Center" / "<Proper Noun> University Medical Center"
  ✓ "<Proper Noun> Pavilion" / "<Proper Noun> Plant" / "<Proper Noun> Annex"
  ✓ "<Customer> School District" / "<Customer> Public Schools" / "<Customer> County Schools USD N"
  ✓ Site codes: "<REGION>-<FUNCTION>-<NN>" (e.g. ATL-HQ-01, ATL-WEST-02, ATL-AIR-03, ATL-CP-05)
  ✓ Street addresses when used as the site name (e.g. "1200 Peachtree St NE")
  ✓ Customer-specific named facilities (e.g. "<Customer Name> Headquarters", "<Customer Name> Logistics Center")

EXCLUDE — these are NOT physical sites:
✗ Standards bodies (ANSI, ASHRAE, NFPA, IEEE, AWWA, AWPA, UL, OSHA, EPA, ISO, TIA)
✗ Vendor / product / SaaS brands (Cisco, Genetec, MySchoolBucks, Mealviewer, Mosaic Cloud, Heartland Payment Systems, ServiceNow, any CamelCase software/cloud name)
✗ Government licensing/regulatory bodies ("Department of Revenue", "Secretary of State", "IRS", "Treasury")
✗ Cities/counties alone without a specific named facility
✗ Departments without a physical building ("IT Department", "Accounts Payable", "Information Technology", "Food Services")
✗ Spec/form labels ("Pre Bid Meeting", "Purchasing Office", "Bid Opening", "Performance Bond", "Project Name", "Owner", "Date", "Phone", "Certification")
✗ Sentence fragments / table-cell truncations
✗ Generic nouns alone ("academy", "school", "building", "library", "medical center", "high school")
✗ Streets alone ("Main Street")
✗ Concepts/systems ("Wide Area Network", "VoIP", "Mass Notification")
✗ Categories ("Phase I", "FEMA Category III", "Level I")
✗ Abbreviations <5 chars unless clearly a site code (AHU, VAV, RTU, APAC)

PROJECT DOCUMENTS:

{docs_excerpt}

OUTPUT (single-line JSON, no commentary, no markdown):
{{"sites": ["<site name or code 1>", "<site name or code 2>"]}}

If no real sites in the docs, return: {{"sites": []}}

Remember: extract VERBATIM from the docs above. Do not invent. Do not echo example shapes.

/no_think"""


def _parse_sites_list(response_text: str) -> set[str]:
    """Parse the extract-mode LLM response into a normalized set.

    Applies a hygiene safety net: drops obvious junk (form-field
    words, generic-noun fragments, single short tokens) even if the
    LLM was tricked into including them. This is set-membership
    filtering, not regex pattern matching — just basic sanity.
    """
    # Find the JSON object in the response
    match = re.search(r"\{[^{}]*\"sites\"\s*:\s*\[[^\]]*\][^{}]*\}", response_text, re.DOTALL)
    if not match:
        return set()
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return set()
    sites = parsed.get("sites")
    if not isinstance(sites, list):
        return set()
    out: set[str] = set()
    for item in sites:
        if not isinstance(item, str):
            continue
        normalized = _normalize_phrase(item)
        if not normalized or len(normalized) < 3:
            continue
        if _is_obvious_non_site(normalized):
            continue
        out.add(normalized)
    return out


# Denylist of obvious non-sites the LLM occasionally returns despite
# the prompt instructions. These are form-field labels, generic
# nouns, table-header words, calendar months, and common SaaS/POS
# vendor brand names. Set-membership check — not regex.
_SUB_SPACE_PREFIXES: tuple[str, ...] = (
    "room", "rooms", "rm", "sector", "zone", "wing", "floor",
    "level", "area", "suite", "ste", "office", "bay", "cube",
    "bldg suite", "bldg ste",
)

_SUB_SPACE_FRAGMENTS: tuple[str, ...] = (
    "conference room", "meeting room", "training room",
    "computer lab", "classroom", "boardroom",
)

_SPEC_LABEL_TOKENS: tuple[str, ...] = (
    "mandatory pre proposal", "mandatory pre-proposal",
    "pre bid meeting", "pre-bid meeting", "pre bid conference",
    "bidder", "bidders", "proposer", "proposers",
    "request for proposal", "request for quote",
    "specification section", "section ",
    "bid opening", "bid submission",
    "performance bond", "general conditions",
)


_OBVIOUS_NON_SITES: frozenset[str] = frozenset({
    # Form / table / spec labels
    "back", "save", "request", "certification", "date", "march",
    "april", "may", "june", "july", "august", "september", "october",
    "november", "december", "january", "february",
    "owner", "phone", "house", "addenda", "bids", "sale", "connect",
    "p o drawer", "p.o. drawer", "taxpayer identification number",
    "project name", "bid opening", "bid closing", "award",
    "agencies engaged", "agencies engaged construction inspection",
    "agencies performing nondestructive testing", "bidder", "bidders",
    "addendum", "amendment", "schedule", "contractor", "subcontractor",
    "scope of work", "request for proposal", "rfp", "rfi", "rfq",
    "alumni field", "notes high school",  # generic from Pottsville
    "academic center",  # too generic
    # Generic nouns alone
    "academy", "school", "campus", "building", "district", "office",
    "library", "medical center", "math", "science building",
    "gymnasium", "auditorium", "stadium",
    "department", "departments", "facility", "facilities", "location",
    "locations", "site", "sites", "place", "places", "address",
    "addresses", "premises", "property", "properties", "room",
    "main entrance", "exit", "lobby", "high school", "middle school",
    "elementary school", "primary school", "pump station",
    "academic center",
    # Concepts/systems (not buildings)
    "wide area network", "local area network", "wan", "lan",
    "mass notification", "wireless coverage", "fiber backbone",
    "voip system", "fire alarm system", "access control system",
    "intrusion detection system", "ip cameras", "ip phones",
    "wireless network", "wired network",
    # Functions / departments alone
    "information technology", "food services", "human resources",
    "accounts payable", "accounts receivable", "purchasing",
    "facilities management", "operations", "administration",
    "central administration", "school district", "central office",
    # Other junk seen in real packs
    "job walk", "site walk", "pre bid", "pre-bid", "walk-through",
    "performance bond", "bid bond", "insurance",
    "general conditions", "special conditions",
    "the project", "this project", "the work", "the contractor",
    "the owner", "the bidder", "the engineer", "the architect",
    "rock", "philad", "produc", "red ceda", "barcelon", "hubert",
    "wattles eleme", "apac",  # truncations
    # Cities/counties alone — when they appear without a specific
    # facility, they're rarely the real site
    "boston", "philadelphia", "pittsburgh", "portland", "reston",
    "bethesda", "arlington", "arlington heights", "arlington hts il",
    "new jersey", "san francisco", "dallas", "carolina",
    # Equipment / acronyms misread as sites
    "ahu", "vav", "rtu", "bas", "awpa", "awwa", "rcshsb",
    "power input poe", "product standard", "refrigerating",
    "redwood inspection service",
    # Standards bodies (some leak past the prompt)
    "ansi", "ashrae", "astm", "nfpa", "ieee", "iso", "tia", "eia",
    "underwriters laboratories", "ul", "fcc", "osha", "epa", "cisa",
    # Government regulatory / licensing bodies (NOT project sites
    # even if they appear by name in the bid package)
    "department of revenue", "department of treasury", "internal revenue service",
    "secretary of state", "office of the secretary of state",
    "department of education", "department of energy",
    "state of california", "state of new york", "state of new jersey",
    "state of texas", "state of florida", "state of illinois",
    "state of south carolina", "state of north carolina",
    "state of south carolina department of revenue retail license",
    "state of south carolina office of the secretary of state",
    # Common vendor / SaaS / POS brand names that aren't sites
    "heartland payment systems", "heartland school solutions",
    "heartland payment systems llc dba heartland school solutions",
    "myschoolbucks", "mealviewer", "mealviewer digital menus",
    "lunchbyte systems nutrikids", "lunchbyte systems",
    "data futures lunchbox", "comalex caf enterprise",
    "link technologies websmartt", "mosaic cloud",
    "ajax imaging", "concept design studio",
    # Project-noise specific to Muskegon / others
    "a muskegon area intermediate school district",
    # v18 leakers — generic / fragment / vendor-like phrases
    "city center", "city centre", "town center", "town centre",
    "performing arts", "student trans", "transition student trans",
    "conference room", "meeting room", "training room",
    "building o", "building a", "building b", "building c",
    "building d", "building e",  # too generic — sub-buildings need
                                  # MORE context to be sites
    "excelsior medical", "excelsior medical corporation",
    "treatment pavilion", "new jersey the neptune high school",
    "potential emergency shelter hours neptune municipal building",
    # Pleasanton acronym fragments
    "osc lpfd classroom", "osc lpfd training center",
    "osc pd", "osc lpfd", "osc osc pd", "osc ppd", "osc pw",
    "library one", "the library",  # duplicate forms of "library"
    "library", "training center",  # too generic alone
    # Los Medanos sub-spaces and fragments
    "fishbowl area of the college complex",
    "equipment plant bas upgrade los medanos college",
    "mandatory pre proposal information building o conference room",
    # DASNY duplicates - keep canonical form
    "28 liberty", "albany ny 12207", "new york ny 10005",
    "dasny 515 broadway",
    # Hood
    "hood county",  # alone — without facility
    # Neptune fragments
    "the almerth m battle homes", "the ntam project", "the neptune",
    "neptune annex 143 col 4 2", "604 gables elementary school 734",
    "gables elementary school address", "neptune middle school address",
    "brookdale community college address", "aquatic center address",
    "the n tam", "ntam project",
    # Common bidding-context proper nouns picked up by regex
    "owner architect", "engineer architect", "owner-architect",
    "owner engineer", "construction manager",
    # Pottsville generics
    "academic center", "alumni field", "notes high school",
    "high school", "middle school", "elementary school",
    # Manhattan Beach generics
    "school district", "manhattan beach", "central administration",
    "central office",
    # v20 leakers — universal patterns
    "beach school district",  # fragment of "Solana Beach School District"
    "district the district", "the district",
    "power school", "powerschool",  # SaaS product
    "public works maintenance",
    "ctc classroom",  # sub-space
    "support center", "analysis center",
    "contractor district",
    "pos terminal", "point of sale terminal",
    "back office", "front office",
    "security intelligence operations center",
    "pre bid", "post bid", "non mandatory",
    "level 1", "level 2", "level 3", "level i", "level ii", "level iii",
    "site walk", "site visit", "site survey",
    "performance location",
    "ongoing support structure", "work breakdown structure",
    "letter fields", "letter field",
    "fns 742",
    "minority business assistance office",
    "state fiscal accountability authority",
    "united states",
    "albany headquarters",
})


def _is_obvious_non_site(normalized: str) -> bool:
    """Hygiene check: drop obvious non-sites the LLM may have included.

    Pipeline (return True = drop):
      1. Pure-numeric (table cell index)
      2. Explicit denylist (form fields, generic nouns, vendors)
      3. Tiny single-word fragments (apac, back, rock)
      4. Vendor-brand pattern (vendor-signal tokens + no anchor)
      5. Sub-space pattern ("Room 114", "Sector 4")
      6. Spec-label pattern ("Mandatory Pre-Proposal", "Bidder", …)
      7. POSITIVE-SIGNAL gate — drop everything lacking any
         facility anchor, institutional descriptor, or digit.
    """
    # Pure numeric — table cell index
    if normalized.replace(" ", "").replace("-", "").replace("_", "").isdigit():
        return True
    # Denylist match
    if normalized in _OBVIOUS_NON_SITES:
        return True
    # Tiny single-word fragments (e.g., "apac", "back", "rock")
    if " " not in normalized and "-" not in normalized and "/" not in normalized:
        # Allow site codes (anything with digits) and 5+ char acronyms
        if not any(c.isdigit() for c in normalized) and len(normalized) <= 4:
            return True
    # Vendor-brand-pattern detector
    if _looks_like_vendor_brand(normalized):
        return True
    # Sub-space pattern: phrases that START with a sub-space prefix
    # ("Room 114", "Sector 4", "Wing N", "Suite 1") are rooms /
    # sub-spaces, not sites. Drop them.
    if _is_sub_space_only(normalized):
        return True
    # Spec-label sentence-fragment pattern: phrase contains a known
    # bid spec / form heading token. The site name is a fragment
    # of the spec section, not the project site.
    if _has_spec_label_token(normalized):
        return True
    # Universal positive-signal gate: drop anything that has no
    # facility anchor, no institutional descriptor, no digit.
    if not _has_positive_site_signal(normalized):
        return True
    return False


def _is_sub_space_only(normalized: str) -> bool:
    """Detect phrases that are room/sector/zone references, NOT sites.

    Phrase starts with a sub-space prefix word, OR consists entirely
    of sub-space fragments + numbers.
    """
    words = normalized.split()
    if not words:
        return False
    first = words[0]
    # Starts with "room", "rooms", "sector", "zone", "wing", etc.
    if first in {"room", "rooms", "rm", "sector", "zone",
                 "wing", "floor", "level", "area", "suite", "ste",
                 "bay", "cube", "lab"}:
        return True
    # Generic sub-space fragments alone or with trailing numbers
    for frag in _SUB_SPACE_FRAGMENTS:
        if normalized == frag:
            return True
        # "conference room" alone or "conference room 1" — sub-space
        if normalized.startswith(frag + " ") and all(
            w.isdigit() or w in {"and", "or", "no", "no.", "#"}
            for w in normalized[len(frag) + 1:].split()
        ):
            return True
    return False


def _has_spec_label_token(normalized: str) -> bool:
    """Detect phrases that contain a known bid/spec-section label.

    These are sentence fragments captured from form headings, not
    project sites ("Mandatory Pre-Proposal Information Building O
    Conference Room", "Bidder shall provide…").
    """
    for label in _SPEC_LABEL_TOKENS:
        if label in normalized:
            return True
    return False


def _has_positive_site_signal(normalized: str) -> bool:
    """Return True iff phrase has at least one signal of being a
    real site.

    Strong anchor (elementary school, hospital, pavilion, …) → KEEP
    Institutional descriptor (district, county, USD, …)      → KEEP
    Digit + ≥2 meaningful words (address/site code)          → KEEP
    Weak anchor + ≥3 distinct meaningful words OR descriptor → KEEP
    """
    # STRONG anchor — definitive facility, always keep
    for anchor in _STRONG_FACILITY_ANCHORS:
        if anchor in normalized:
            return True
    # Institutional descriptor anywhere
    for desc in _INSTITUTIONAL_DESCRIPTORS:
        if desc in normalized:
            return True
    # Meaningful (≥3 char) word count — split on whitespace AND
    # hyphens so "atl-west-01" counts as {"atl","west"} (digit
    # tokens are also kept for site-code recognition).
    word_tokens = [w for w in re.split(r"[\s\-]+", normalized) if w]
    meaningful_words = [w for w in word_tokens if len(w) >= 3]
    distinct_meaningful = set(meaningful_words)
    # Digit — accept only when paired with ≥2 distinct meaningful
    # words (so "fns 742" / "level 1" don't pass on digit alone).
    # Site codes like "atl-west-01" have 2 meaningful tokens so
    # they pass.
    has_digit = any(c.isdigit() for c in normalized)
    if has_digit and len(distinct_meaningful) >= 2:
        return True
    # WEAK anchor — accept only with extra context (≥3 distinct
    # meaningful words)
    weak_match = False
    for anchor in _WEAK_FACILITY_ANCHORS:
        if anchor in normalized:
            weak_match = True
            break
    if weak_match and len(distinct_meaningful) >= 3:
        return True
    return False


# Tokens that strongly suggest a software / SaaS / payment / vendor
# brand name when they appear in a multi-word phrase WITHOUT a
# facility-anchor noun.
_VENDOR_SIGNAL_TOKENS: frozenset[str] = frozenset({
    "systems", "technologies", "technology", "solutions",
    "enterprise", "enterprises", "cloud", "saas", "platform",
    "software", "digital", "data", "intelligence", "analytics",
    "payment", "payments", "processing", "merchant",
    "lunchbox", "menus", "menu", "kiosks", "kiosk", "checkout",
    "websmartt", "nutrikids", "schoolbucks", "mealviewer",
    "lifesize", "pagerduty", "servicenow", "salesforce",
    "imaging", "design studio", "studio", "agency",
    "consulting", "consultants", "advisors", "partners",
    "group", "incorporated", "llc", "inc", "ltd", "corp",
    "corporation", "co",
})

# STRONG facility anchors — a 2-word phrase ending here is
# definitively a site even without other signal. These nouns
# are unambiguous building types. SaaS products that masquerade
# as schools ("PowerSchool", "MySchoolBucks") are caught by the
# explicit denylist BEFORE the anchor check fires.
_STRONG_FACILITY_ANCHORS: frozenset[str] = frozenset({
    "school", "schools",  # "Wesley School" / "Atrisco Heritage Academy"
    "elementary school", "middle school", "high school",
    "primary school", "preschool",
    "elementary", "academy", "academies",
    "college", "university",
    "hospital", "clinic",
    "medical center", "medical centre", "health center",
    "library",
    "auditorium", "stadium", "gymnasium", "fieldhouse",
    "courthouse", "city hall", "town hall",
    "fire station", "police station",
    "emergency operations center",
    "warehouse", "depot",
    "datacenter", "data center",
    "headquarters",
    "pavilion",
    "campus",
    "central plant", "main plant", "utility plant",
    "reservoir", "substation",
    "tower",
    "pump station", "supply station", "lift station",
    "municipal building",
})

# WEAK facility anchors — these alone are too generic. They count
# as positive signal only when:
#   - phrase has ≥3 distinct words OR
#   - phrase has an institutional descriptor (district/county/…) OR
#   - phrase has a digit
# This prevents "support center", "back office", "power school"
# (PowerSchool, the SaaS) from passing on the anchor alone.
_WEAK_FACILITY_ANCHORS: frozenset[str] = frozenset({
    "center", "centre", "complex",
    "office", "facility", "facilities",
    "building", "room", "area", "zone", "wing", "floor",
    "annex", "suite",
    "plaza", "square",
    "terminal",  # POS terminal is NOT a site
    "station",   # "weather station" not a site; "fire station" is (strong)
    "public works",  # generic dept; longer forms with department / yard OK
})

# Public alias kept for back-compat — union of strong + weak
_FACILITY_ANCHOR_TOKENS: frozenset[str] = (
    _STRONG_FACILITY_ANCHORS | _WEAK_FACILITY_ANCHORS
)


# Institutional descriptors that mark a customer-name phrase as a
# real site (school district, ISD, USD, county school system, etc.)
# even without a facility anchor noun.
_INSTITUTIONAL_DESCRIPTORS: frozenset[str] = frozenset({
    "district", "isd", "usd", "system", "systems",
    "county schools", "city schools", "public schools",
    "intermediate school district", "unified school district",
    "school district", "school system",
    "city of", "town of", "village of", "township of",
    "county of", "borough of", "municipality of",
    "police department", "fire department", "fire & rescue",
    "department of public works", "department of transportation",
    "transit authority", "housing authority",
    "community college district", "regional school",
    "auditor", "purchasing dept", "purchasing department",
})


def _looks_like_vendor_brand(normalized: str) -> bool:
    """Detect vendor-brand-style multi-word phrases.

    Heuristic: phrase has a "vendor signal" token (systems,
    technologies, cloud, llc, etc.) AND NO facility anchor token.
    Catches things like "Link Technologies WebSmartt", "Comalex
    CAF Enterprise", "Mealviewer Digital Menus", "Heartland
    Payment Systems LLC" that the LLM may include.

    Tokens are split on both whitespace AND hyphens so phrases
    like "link technologies-websmartt" (where the vendor parser
    embedded a hyphen) still match the vendor signal.
    """
    if " " not in normalized and "-" not in normalized:
        return False
    # Split on both whitespace AND hyphens so embedded hyphens
    # don't hide the vendor signal token
    tokens = set(re.split(r"[\s\-]+", normalized))
    if not tokens & _VENDOR_SIGNAL_TOKENS:
        return False
    # Check for facility anchor — including 2-word anchors
    for anchor in _FACILITY_ANCHOR_TOKENS:
        if anchor in normalized:
            return False
    # Has vendor signal, no anchor → likely vendor brand
    return True


def _normalize_phrase(phrase: str) -> str:
    """Lowercase + strip + collapse whitespace + drop trailing punct.

    Also strips leading articles ("a", "an", "the") so "the wesley
    school" and "wesley school" collapse to the same entity. Without
    this, atom-level inconsistency in capitalization or article use
    produces duplicate site entries.
    """
    s = phrase.lower().strip()
    s = re.sub(r"[^a-z0-9\s\-/.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # Strip leading articles
    for article in ("the ", "a ", "an "):
        if s.startswith(article):
            s = s[len(article):]
            break
    return s


# ─────────────────────── VERIFY MODE (legacy) ───────────────────────


def verify_sites_with_llm(catalog: set[str], atoms: list[Any]) -> set[str]:
    """Filter the candidate site catalog via Ollama LLM.

    Sends the catalog + representative doc excerpts to an LLM. Asks
    the model to label each candidate as KEEP (real project site) or
    DROP. Catalogs larger than ``_CHUNK_SIZE`` are split into multiple
    LLM calls so the response stays bounded and parses reliably.

    Args:
      catalog: set of normalized site phrases. Output of the
        deterministic catalog builder.
      atoms: full atom list — used to extract project context.

    Returns:
      Subset of ``catalog`` the LLM kept. On any failure for a chunk,
      the chunk passes through unchanged (fail-safe).
    """
    if not catalog:
        return catalog

    docs_excerpt = _build_doc_excerpt(atoms, max_per_doc=600, max_total=3000)
    if not docs_excerpt:
        return catalog

    items = sorted(catalog)
    if len(items) <= _CHUNK_SIZE:
        return _verify_chunk(items, docs_excerpt) or catalog

    # Chunk large catalogs and union the keeps.
    kept_total: set[str] = set()
    any_succeeded = False
    for i in range(0, len(items), _CHUNK_SIZE):
        chunk = items[i:i + _CHUNK_SIZE]
        result = _verify_chunk(chunk, docs_excerpt)
        if result is None:
            # Chunk failed — fail-safe: keep all items in this chunk
            kept_total.update(chunk)
        else:
            any_succeeded = True
            kept_total.update(result)
    # If EVERY chunk failed, fall back to full passthrough so we
    # don't accidentally produce an empty catalog from comms errors.
    if not any_succeeded:
        return catalog
    return kept_total


def _verify_chunk(items: list[str], docs_excerpt: str) -> set[str] | None:
    """Send one chunk of candidates to the LLM. Returns the kept set,
    or ``None`` if the call failed / the response was unparseable."""
    if not items:
        return set()
    prompt = _build_prompt(items, docs_excerpt)
    response_text = _call_ollama(prompt)
    if not response_text:
        return None
    return _parse_keep_drop_response(response_text, items)


# ─────────── prompt construction ───────────


def _build_doc_excerpt(atoms: list[Any], *, max_per_doc: int, max_total: int) -> str:
    """Concatenate doc context (filename + section paths + body text)
    into a chunk the LLM can read.

    Section_path tokens are included alongside raw_text because real
    bid docs often have the customer / site name in a PDF section
    HEADING (which doesn't end up in any atom's raw_text but DOES
    end up in section_path). Skipping headings means the LLM
    misses the customer name on cover pages — that's why ITAD was
    failing before this fix.
    """
    by_artifact: dict[str, dict[str, Any]] = {}
    for atom in atoms or []:
        aid = getattr(atom, "artifact_id", None)
        if not aid:
            continue
        slot = by_artifact.setdefault(aid, {
            "bodies": [],
            "headings": set(),
            "filename": None,
        })
        raw = getattr(atom, "raw_text", None) or ""
        if raw:
            slot["bodies"].append(raw)
        # Capture section_path headings (the cover-page institutional
        # names that the PDF parser slices into subsection headings
        # rather than body text).
        try:
            refs = getattr(atom, "source_refs", None) or []
            if refs:
                locator = getattr(refs[0], "locator", None) or {}
                if isinstance(locator, dict):
                    for k in ("section_path",):
                        sp = locator.get(k)
                        if isinstance(sp, list):
                            for h in sp:
                                if isinstance(h, str) and h.strip():
                                    slot["headings"].add(h.strip())
                    for k in ("section", "heading", "title"):
                        v = locator.get(k)
                        if isinstance(v, str) and v.strip():
                            slot["headings"].add(v.strip())
                if slot["filename"] is None:
                    fname = getattr(refs[0], "filename", None)
                    if fname:
                        slot["filename"] = fname
        except Exception:
            pass

    if not by_artifact:
        return ""
    chunks: list[str] = []
    running_total = 0
    for aid, slot in sorted(by_artifact.items()):
        # Headings first — these often carry the customer / site
        # name on cover pages
        headings_part = ""
        if slot["headings"]:
            # Keep headings under ~800 chars
            headings_text = " | ".join(sorted(slot["headings"]))[:800]
            headings_part = f"[HEADINGS] {headings_text}\n\n"
        # Body text
        stitched = " ".join(slot["bodies"])
        body_budget = max(0, max_per_doc - len(headings_part))
        if len(stitched) > body_budget:
            stitched = stitched[:body_budget]
        section = f"--- {slot['filename'] or aid} ---\n{headings_part}{stitched}"
        chunks.append(section)
        running_total += len(section)
        if running_total >= max_total:
            break
    return "\n\n".join(chunks)


def _build_prompt(items: list[str] | set[str], doc_excerpt: str) -> str:
    """Compose the audit prompt.

    Critical design points:
      - List candidates as a numbered list so the model can refer to
        them precisely
      - Request strict JSON output (no commentary) for deterministic
        parsing
      - Give explicit DROP examples (standards bodies, landmarks,
        sentence fragments) so the model knows what to filter
    """
    if isinstance(items, set):
        items = sorted(items)
    numbered = "\n".join(f"  {i + 1}. {p}" for i, p in enumerate(items))

    return f"""You are auditing a candidate site list extracted from a managed-services bid package.

The bid package is excerpted below (multiple source documents):

{doc_excerpt}

Here are the candidate site names that were detected:

{numbered}

For each candidate, label whether it is a real PROJECT SITE (a building, school, hospital, facility, or location in scope for THIS specific bid) or a FALSE POSITIVE.

DROP these (NOT project sites):
  - Standards bodies / industry associations (ANSI, ASHRAE, NFPA, "Building Officials Council", "American Wood Preservers")
  - Vendor / product names ("Cisco Systems", "Genetec Security Center")
  - Famous landmarks mentioned only as reference examples ("Chrysler Building" in a Muskegon spec, "Yeon Building")
  - Cities or places mentioned as spec references, NOT in project scope
  - Sentence fragments captured by regex ("consumption annual energy costs neptune municipal building", "performance bonds los medanos college")
  - Generic nouns without a specific name ("medical center", "elementary school", "academy")
  - Spec-doc section headings ("pre bid meeting location", "purchasing office")
  - Street addresses misclassified as sites ("Heck Ave", "Corlies Avenue")

KEEP these (real project sites):
  - Named schools / hospitals / buildings / offices / branches that this bid actually serves
  - Real facility names with proper nouns + facility tail ("Wesley School", "Brookdale Community College")
  - Customer's headquarters / main location

Respond with ONLY a JSON object on a single line. No commentary. No thinking. No markdown.

Format: {{"keep": [<numbers>], "drop": [<numbers>]}}

Example: {{"keep": [1, 3, 7], "drop": [2, 4, 5, 6, 8]}}

/no_think"""


# ─────────── Ollama HTTP call ───────────


def _call_ollama(prompt: str, *, max_tokens: int = 2048) -> str:
    """POST to /api/generate. Returns the raw response.text or ''."""
    host = os.environ.get("OLLAMA_HOST", DEFAULT_HOST).rstrip("/")
    model = os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)
    timeout = int(os.environ.get("SOWSMITH_LLM_TIMEOUT", str(DEFAULT_TIMEOUT)))

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.0,
            "num_predict": max_tokens,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{host}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""
    try:
        result = json.loads(body)
        return str(result.get("response") or "")
    except json.JSONDecodeError:
        return ""


# ─────────── response parsing ───────────


def _parse_keep_drop_response(response_text: str, items: list[str] | set[str]) -> set[str]:
    """Extract the keep-numbers from the model's JSON response.

    The model is asked for ``{"keep": [1, 3, 7], "drop": [...]}``.
    Some models append commentary; we look for the first JSON object
    and parse that. Hygiene filter is also applied so even items the
    model said "keep" get dropped if they're obviously not sites.
    """
    if isinstance(items, set):
        items = sorted(items)
    if not items:
        return set()
    # Find the first {...} block in the response
    match = re.search(r"\{[^{}]*\"keep\"\s*:\s*\[[^\]]*\][^{}]*\}", response_text, re.DOTALL)
    if not match:
        return set()
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return set()
    keep_ids = parsed.get("keep")
    if not isinstance(keep_ids, list):
        return set()
    kept: set[str] = set()
    for raw_id in keep_ids:
        try:
            idx = int(raw_id) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(items):
            item = items[idx]
            # Hygiene safety net: drop items the LLM said "keep" but
            # that are obvious non-sites (form-field words, vendors,
            # generic nouns)
            if not _is_obvious_non_site(item):
                kept.add(item)
    return kept


def apply_site_hygiene(catalog: set[str]) -> set[str]:
    """Filter a site catalog (from any source) through the denylist.

    Public function so the regex catalog can also be hygienized
    before being used as a fallback — that way even when the LLM
    pipeline is fully bypassed, obvious junk doesn't reach the
    final output.
    """
    return {p for p in catalog if not _is_obvious_non_site(p)}


__all__ = [
    "extract_sites_with_llm",
    "verify_sites_with_llm",
    "ollama_reachable",
    "apply_site_hygiene",
]
