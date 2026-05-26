"""Document-structure-aware site detection (Option D).

The proper-noun regex matcher in entity_extraction emits a ``site:*``
key for every capitalized phrase in any atom. On real bid docs that
produces 60-180 false positives per pack (standards-body names,
random landmarks, header fragments, sentence pieces, generic nouns).

This module provides a deterministic, structural alternative:

  Tier 1 — LOCATIONS SECTION
    Real RFPs / SOWs / Specs have an explicit exhibit listing every
    project site ("Exhibit B — Locations", "Site List", "Facility
    Schedule", "Service Locations"). Find that section; every
    qualifying line under it is a site declaration.

  Tier 2 — ADDRESS-ANCHORED
    A capitalized phrase within ~120 chars of a US street address is
    very likely to be the site at that address. The address is the
    structural anchor that distinguishes real sites from incidental
    proper nouns.

  Tier 3 — STRICT REGEX FALLBACK
    Only when Tiers 1+2 both produce nothing (small atoms, missing
    locations exhibit, no addresses). Phrase must end with a strong
    facility tail noun (``elementary school``, ``medical center``,
    ``fire station``) AND have no negative tokens.

The output is an **authoritative site catalog** for the project —
a set of normalized site phrases. ``_emit_proper_nouns`` consults
this catalog before emitting any ``site:`` key. Outside the catalog,
proper nouns get routed to ``vendor:`` / ``stakeholder:`` /
dropped instead of falsely tagged as sites.

Pure functions, no I/O. Built once per compile, passed through
``enrich_atoms``.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

# Headings that indicate the section IS the authoritative site list.
# Match against normalized section path tokens (lower-cased).
_LOCATIONS_SECTION_PHRASES: tuple[str, ...] = (
    "locations",
    "site list",
    "sites",
    "facility list",
    "facilities list",
    "facility schedule",
    "service locations",
    "project sites",
    "list of sites",
    "list of locations",
    "building list",
    "buildings list",
    "schools list",
    "school list",
    "list of buildings",
    "list of schools",
    "list of facilities",
    "location schedule",
    "address list",
    "addresses",
    "site directory",
    "site addresses",
    "facility addresses",
    "site information",
    # Common exhibit / attachment labels — when paired with a
    # site/location keyword anywhere in the parent section path
    "exhibit a — locations",
    "exhibit a - locations",
    "exhibit b — locations",
    "exhibit b - locations",
    "exhibit c — locations",
    "exhibit c - locations",
    "attachment a — sites",
    "attachment a - sites",
    "schedule a — locations",
    "schedule a - locations",
    "schedule b — locations",
    "schedule b - locations",
)

# US street address — strict pattern requiring a street number AND a
# street suffix. We do NOT require city/state/zip because those often
# wrap to the next line in PDFs.
_STREET_SUFFIX = (
    r"st\.?|street|rd\.?|road|ave\.?|avenue|blvd\.?|boulevard|"
    r"dr\.?|drive|ln\.?|lane|ct\.?|court|pl\.?|place|"
    r"pkwy\.?|parkway|hwy\.?|highway|way|terr\.?|terrace|"
    r"cir\.?|circle|sq\.?|square|trl\.?|trail|"
    r"row|run|loop|alley|"
    r"plaza|crossing|crescent|"
    r"n\.?|s\.?|e\.?|w\.?|north|south|east|west|"
    r"nw|ne|sw|se"
)
_US_ADDRESS_RE = re.compile(
    rf"\b\d{{1,6}}[\s,.]+(?:[A-Z0-9][A-Za-z0-9'.\-]*\s+){{1,5}}(?:{_STREET_SUFFIX})\b",
    re.IGNORECASE,
)
# US zip code (5 or 5+4)
_US_ZIP_RE = re.compile(r"\b\d{5}(?:-\d{4})?\b")

# Strong facility tail nouns — phrases ending in these are high-
# confidence site candidates even without an address. These are the
# words that ALMOST NEVER appear in incidental proper nouns.
_STRONG_FACILITY_TAILS: frozenset[str] = frozenset({
    "school", "elementary", "middle school", "high school", "primary",
    "academy", "university", "college", "campus",
    "hospital", "clinic", "medical center", "medical centre",
    "fire station", "police station", "library",
    "courthouse", "city hall", "town hall",
    "warehouse", "depot", "terminal",
    "datacenter", "data center",
    "headquarters", "hq",
    "fieldhouse", "stadium", "auditorium", "gymnasium",
    "annex",
    "center", "centre",   # too generic alone, but with positive prefix is OK
})

# Phrases that look like sites by surface but never are. Add as we
# find more in real packs.
_SITE_BLOCKLIST: frozenset[str] = frozenset({
    "each facility", "this facility", "the facility", "every facility",
    "the building", "this building", "the site", "this site",
    "the property", "the location", "this location",
    "all sites", "all locations", "all buildings",
    "block 909",
    # Section / header fragments seen in the corpus
    "consumption", "energy costs", "estimated electric consumption",
    "annual energy costs", "facility description address",
    "estimated annual consumption",
    "covid 19", "covid-19",
    "fema category", "fema category i", "fema category ii",
    "fema category iii", "fema category iv",
    "food sales", "first aid", "first aid squad",
    "business improvement district",
    "central utilities plant",
    "atlantic county utilities authority",
    "camden county municipal utilities authority",
    "defense advanced projects agency",
    "department of defense", "department of energy",
    "department of education", "department of transportation",
    "bid opening", "bid closing", "bid submission",
    "quality assurance", "quality control",
    "go live", "go live per building",
    "note add", "notify testing agency",
    "level i", "level ii", "level iii", "level iv",
    "phase i", "phase ii", "phase iii", "phase iv",
    "pre on site", "pre on-site",
})


def _normalize(phrase: str) -> str:
    """Lowercase, trim, collapse whitespace, strip punctuation."""
    s = phrase.lower().strip()
    s = re.sub(r"[^a-z0-9\s\-/.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _has_address_nearby(atom_text: str, phrase_start: int, phrase_end: int, window: int = 120) -> bool:
    """Address co-location: a US street address within ``window`` chars of the phrase."""
    pre = atom_text[max(0, phrase_start - window):phrase_start]
    post = atom_text[phrase_end:min(len(atom_text), phrase_end + window)]
    return bool(_US_ADDRESS_RE.search(pre) or _US_ADDRESS_RE.search(post) or
                _US_ZIP_RE.search(pre) or _US_ZIP_RE.search(post))


def _atom_section_text(atom: Any) -> str:
    """Concatenate the atom's section_path into a single lowercase string."""
    try:
        refs = getattr(atom, "source_refs", None) or []
        if not refs:
            return ""
        locator = getattr(refs[0], "locator", None) or {}
        if not isinstance(locator, dict):
            return ""
        section_path = locator.get("section_path")
        if isinstance(section_path, list) and section_path:
            return " ".join(str(x).lower() for x in section_path if x)
        for k in ("section", "heading", "title", "subsection"):
            v = locator.get(k)
            if isinstance(v, str) and v:
                return v.lower()
    except Exception:
        return ""
    return ""


def _atom_is_in_locations_section(atom: Any) -> bool:
    """Tier 1: atom's section_path matches a known Locations heading."""
    section = _atom_section_text(atom)
    if not section:
        return False
    for needle in _LOCATIONS_SECTION_PHRASES:
        if needle in section:
            return True
    return False


def _looks_like_site_phrase(phrase: str) -> bool:
    """Filter out obviously non-site capitalized phrases.

    Drops sentence fragments, generic nouns, headers, blocklisted
    phrases, standards-body names, and phrases with embedded verb/
    header tokens that signal it's a description not a name.
    """
    norm = _normalize(phrase)
    if not norm or len(norm) < 4:
        return False
    if norm in _SITE_BLOCKLIST:
        return False
    # Multi-clause / header fragments: phrases containing common
    # PDF-header glue tokens are header strings, not names
    _HEADER_GLUE = (
        "description address", "address ", "energy costs",
        "annual consumption", "annual energy", "estimated electric",
        "consumption ", " consumption", "estimated ", " estimated",
        "tag ", " tag", "zone group", "device type", "performance bond",
        "purchasing office", "pre bid", "meeting location",
        "facility description", "building reference", "building summary",
        "campus overview", "contracts manager", "likely installation",
        "rf tag", "pressure sensor", "ah tag", "ah ",
        " summary", "overview ", "reference ",
        "required high", "required low", "required medium",
    )
    if any(t in norm for t in _HEADER_GLUE):
        return False
    # Phrases that start with a generic article/word
    for stop in ("the ", "a ", "an ", "each ", "this ", "that ", "these ", "those "):
        if norm.startswith(stop):
            norm = norm[len(stop):]
            if len(norm) < 4:
                return False
    # Standards-body / industry-association tail words. Phrases
    # ending in these are organizations, not sites.
    _NON_SITE_TAILS = {
        "council", "association", "society", "institute",
        "federation", "consortium", "alliance", "bureau", "publications",
        "officials", "manufacturers", "preservers", "contractors",
        "engineers", "architects", "designers", "specifiers",
        "authority", "agency", "administration", "commission",
        "department", "ministry", "office", "press", "studio",
        "category", "categories", "level", "phase", "grade",
        "consumption", "costs", "summary", "overview",
        # Street suffixes — "Corlies Avenue", "Heck Ave", "Neptune
        # Boulevard" are streets, not sites. The address: emitter
        # captures them separately.
        "ave", "avenue", "street", "st", "road", "rd", "drive", "dr",
        "boulevard", "blvd", "way", "lane", "ln", "court", "ct",
        "place", "pl", "highway", "hwy", "parkway", "pkwy",
        "circle", "cir", "trail", "trl", "terrace", "loop", "run",
        # Generic plural nouns
        "schools", "buildings", "facilities", "campuses", "offices",
        "stations", "centers", "centres", "rooms",
    }
    last_word = norm.rsplit(" ", 1)[-1] if " " in norm else norm
    if last_word in _NON_SITE_TAILS:
        return False
    # Single-word phrases that match a generic facility noun without
    # a specific name in front (e.g. "academy" alone, "school" alone,
    # "campus" alone, "hospital" alone) are too generic to be a site.
    _GENERIC_ALONE = {
        "academy", "school", "campus", "hospital", "clinic",
        "facility", "building", "office", "warehouse", "annex",
        "tower", "plaza", "park", "center", "centre",
        "elementary", "elementary school", "high school",
        "middle school", "primary school", "community college",
        "data center", "datacenter",
        "main entrance", "critical facility",
    }
    if norm in _GENERIC_ALONE:
        return False
    # Phrases with too many words are almost always sentence
    # fragments, not names. Real site names rarely exceed 7 words
    # ("Lakeshore Learning Center Career Tech Center" = 6 words is
    # the upper bound seen in real packs).
    word_count = len(norm.split())
    if word_count > 7:
        return False
    # Phrases that contain certain operational verbs/headers are
    # spec-sheet glue not names
    _GLUE_TOKENS = {
        "installation", "summary", "overview", "estimated",
        "required", "calculated", "computed", "approximate",
    }
    tokens = set(norm.split())
    if tokens & _GLUE_TOKENS and word_count >= 3:
        return False
    return True


def find_authoritative_site_phrases(atoms: Iterable[Any]) -> set[str]:
    """Build the project-wide authoritative-site catalog.

    Returns a set of normalized phrases. The proper-noun emitter
    consults this set: a capitalized phrase becomes a ``site:*`` key
    only if its normalized form is IN this set.

    Three signals contribute to inclusion:

      Tier 1 — atom is in a section whose path contains a Locations
               heading (most authoritative)
      Tier 2 — atom's raw_text has a US street address; capitalized
               phrases within 120 chars of the address are sites
      Tier 3 — phrase ends with a strong facility-tail noun
               (Wesley Elementary School, Memorial Hospital, Fire
               Station 4); high-precision pattern that matches
               named institutions

    All three tiers contribute to the same catalog. An atom can be
    in multiple tiers.
    """
    catalog: set[str] = set()

    # Tier 1: every atom in a Locations section contributes its
    # raw_text as a candidate site list. We split on commas / newlines
    # / semicolons since these sections often list sites comma-
    # separated or one-per-line.
    for atom in atoms:
        if not _atom_is_in_locations_section(atom):
            continue
        raw = (getattr(atom, "raw_text", None) or "").strip()
        if not raw:
            # Try value.text as fallback
            val = getattr(atom, "value", None) or {}
            if isinstance(val, dict):
                raw = val.get("text") or val.get("content") or ""
        if not raw:
            continue
        # Split by line/comma/semicolon for multi-site list cells
        for piece in re.split(r"[,;\n\r]+|\s{3,}", raw):
            piece = piece.strip(" .-•*\t")
            if 4 <= len(piece) <= 120 and any(c.isupper() for c in piece):
                if _looks_like_site_phrase(piece):
                    catalog.add(_normalize(piece))

    # Tier 2: address-anchored phrases. For every atom containing a
    # US street address, capture proper-noun phrases within 120 chars
    # of the address span.
    proper_run = re.compile(
        r"\b([A-Z][A-Za-z0-9'.\-]+(?:\s+[A-Z][A-Za-z0-9'.\-]+){0,6})\b"
    )
    for atom in atoms:
        raw = getattr(atom, "raw_text", None) or ""
        if not raw:
            continue
        for addr_match in _US_ADDRESS_RE.finditer(raw):
            a_start, a_end = addr_match.span()
            # Scan 120 chars on each side for capitalized phrases
            pre_window = raw[max(0, a_start - 120):a_start]
            post_window = raw[a_end:min(len(raw), a_end + 120)]
            for window in (pre_window, post_window):
                for m in proper_run.finditer(window):
                    phrase = m.group(1)
                    if _looks_like_site_phrase(phrase):
                        catalog.add(_normalize(phrase))

    # Tier 3: phrases ending with strong facility tails — high
    # precision even without structural anchor. We require the
    # phrase to be 2-6 words AND end with a strong tail.
    strong_tail_re = re.compile(
        r"\b([A-Z][A-Za-z0-9'.\-]+(?:\s+[A-Z][A-Za-z0-9'.\-]+){1,5}\s+"
        r"(?:Elementary|Middle|High|Primary|Charter)\s+School)\b"
    )
    medical_re = re.compile(
        r"\b([A-Z][A-Za-z0-9'.\-]+(?:\s+[A-Z][A-Za-z0-9'.\-]+){0,5}\s+"
        r"(?:Medical|Health|Memorial|General|Regional|Children's|Veterans|Community)"
        r"\s+(?:Center|Centre|Hospital|Clinic))\b"
    )
    station_re = re.compile(
        r"\b((?:Fire|Police|Emergency|Power)\s+Station\s*(?:No\.?\s*)?\d{0,3}|"
        r"[A-Z][A-Za-z0-9'.\-]+(?:\s+[A-Z][A-Za-z0-9'.\-]+){0,3}\s+Library)\b"
    )
    for atom in atoms:
        raw = getattr(atom, "raw_text", None) or ""
        if not raw:
            continue
        for regex in (strong_tail_re, medical_re, station_re):
            for m in regex.finditer(raw):
                phrase = m.group(1)
                if _looks_like_site_phrase(phrase):
                    catalog.add(_normalize(phrase))

    return catalog


def phrase_is_in_catalog(phrase: str, catalog: set[str]) -> bool:
    """Check whether a phrase matches anything in the catalog.

    Match modes (anchored, not buried-substring):
      - exact normalized match
      - catalog entry is a PREFIX of the phrase (catalog "Wesley
        School" matches atom "Wesley School Annex")
      - catalog entry is a SUFFIX of the phrase (catalog "Memorial
        Hospital" matches atom "St. Mary Memorial Hospital")
      - phrase is a PREFIX or SUFFIX of catalog entry (atom uses
        shorter form than the catalog entry — accept)

    We deliberately do NOT match when the catalog entry appears as
    an internal substring of a much longer phrase, because that
    flags spec-sheet glue like "consumption annual energy costs
    **neptune municipal building**" via the legitimate catalog
    entry "neptune municipal building".
    """
    norm = _normalize(phrase)
    if not norm:
        return False
    if norm in catalog:
        return True
    norm_words = norm.split()
    if not norm_words:
        return False
    for c in catalog:
        if len(c) < 8:
            continue
        c_words = c.split()
        if not c_words:
            continue
        # Catalog is prefix of phrase: ``c_words`` is a prefix of
        # ``norm_words``
        if len(c_words) < len(norm_words) and norm_words[:len(c_words)] == c_words:
            return True
        # Catalog is suffix of phrase
        if len(c_words) < len(norm_words) and norm_words[-len(c_words):] == c_words:
            return True
        # Phrase is prefix of catalog (atom uses shorter form)
        if len(norm_words) < len(c_words) and c_words[:len(norm_words)] == norm_words:
            return True
        # Phrase is suffix of catalog
        if len(norm_words) < len(c_words) and c_words[-len(norm_words):] == norm_words:
            return True
    return False


__all__ = [
    "find_authoritative_site_phrases",
    "phrase_is_in_catalog",
]
