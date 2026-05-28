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

# v48: populated by extract_sites_with_llm structured path.
# Maps any site name/alias → full site attribute dict
# {id, names, address, mdf_idf, access_window, escort, users, rooms, notes}.
# Read by entity_extraction.py / envelope projector to enrich site entities
# with their attributes (currently a marker for future wiring).
_llm_site_attr_cache: dict[str, dict] = {}

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
    """Lowercase, trim, collapse whitespace, strip punctuation.

    v53.6: normalize hyphens AND underscores to spaces so "ATL-HQ-01"
    and "atl_hq_01" (the entity_key slug form) compare equal. Previously
    only underscores collapsed → "atl-hq-01" stayed with hyphens while
    "atl hq 01" (from slug) had spaces, so phrase_is_in_catalog never
    matched site_keys to catalog entries.
    """
    s = phrase.lower().strip()
    # First, replace hyphens AND underscores AND slashes with spaces
    # so site IDs and slug forms collapse to the same shape.
    s = re.sub(r"[\-_/]", " ", s)
    # Then strip all other punctuation
    s = re.sub(r"[^a-z0-9\s.]", " ", s)
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

    Six independent structural signals contribute. Each is a high-
    precision pattern real bid docs use to declare sites. A phrase
    in ANY tier joins the catalog.

      Tier 1 — LOCATIONS SECTION HEADING IN PATH
        atom.section_path contains a Locations / Site List /
        Facility Schedule / Exhibit B — Locations heading; atom
        text is parsed as site declarations
      Tier 2 — ADDRESS-ANCHORED PROPER NOUN
        atom has a US street address; capitalized phrases within
        120 chars are sites
      Tier 3 — STRONG FACILITY-TAIL PATTERN
        "X Elementary/Middle/High School", "X Medical Center / Hospital",
        "Fire Station N", "X Library"
      Tier 4 — SECTION HEADING IS THE SITE NAME (NEW)
        Section headings in ALL CAPS that contain a strong facility
        tail ("WESLEY SCHOOL", "CRAIG CAMPUS", "MEMORIAL HOSPITAL"
        as standalone headings) are themselves site declarations.
        Real spec docs structure with one section per site.
      Tier 5 — EXPLICIT LABEL "SITES: A, B, C" (NEW)
        Body text matching ``(?:Sites?|Locations?|Buildings?|
        Facilities|Schools?|Stores?|Branches?|Premises?)\\s*[:—-]\\s*``
        followed by a comma/semicolon/newline-separated list of
        proper-noun phrases. Each list item becomes a site.
      Tier 6 — "THE FOLLOWING X" SEMANTIC PATTERN (NEW)
        Body text matching ``(?:the following|these|listed)\\s+
        (?:sites|locations|buildings|facilities|schools)(?:\\s+
        (?:are|include|covered|listed))?\\s*[:—-]?`` — the next
        paragraph / list is the site list.
    """
    # Track which artifact(s) and which tier(s) discovered each phrase.
    # ``evidence[phrase] = {"artifacts": {a1, a2}, "tiers": {1, 4, 5}, "mentions": int}``
    # Final filter (cross-doc validation): keep a phrase iff
    # (a) any structural tier {1, 4, 5, 6} fired for it, OR
    # (b) it was mentioned in ≥2 distinct artifacts, OR
    # (c) it was mentioned ≥3 times even in a single artifact.
    # This kills the long tail of "Chrysler Building" / "Yeon Building"
    # type singletons that pass the regex tiers but are spec
    # references rather than real project sites.
    evidence: dict[str, dict[str, Any]] = {}

    def _record(phrase: str, atom: Any, tier: int) -> None:
        if not _looks_like_site_phrase(phrase):
            return
        norm = _normalize(phrase)
        if not norm:
            return
        e = evidence.setdefault(norm, {"artifacts": set(), "tiers": set(), "mentions": 0})
        e["mentions"] += 1
        e["tiers"].add(tier)
        aid = getattr(atom, "artifact_id", None) if atom is not None else None
        if aid:
            e["artifacts"].add(aid)

    atom_list = list(atoms) if not isinstance(atoms, list) else atoms

    # Track corpus-wide artifact count so we can tune the cross-doc
    # threshold (a single-doc pack can't require 2-artifact evidence).
    artifact_ids = {getattr(a, "artifact_id", None) for a in atom_list}
    artifact_ids.discard(None)
    artifact_count = len(artifact_ids)

    # ─── Tier 0: physical_site atoms ARE the authoritative catalog ───
    # v53.2: any atom emitted by a site-roster parser (PDF/XLSX/DOCX)
    # carries explicit id / name / aliases. These are ground truth —
    # add them to the catalog with the strongest tier so the central
    # gate accepts site:* keys that match these IDs/names AND rejects
    # anything else when this set is non-empty.
    # v53.5: ALSO read site IDs from site_allocation / bom_line atoms
    # value.site / value.site_id fields. These are the BOM rows from
    # spreadsheet parsers (xlsx_parser) that ARE reliably extracting
    # site codes like "ATL-HQ-01" from the BOM columns. They're more
    # specific than LLM site_clusters which often have truncated
    # canonical_names like "ATL-HQ". By including them in the catalog,
    # the central gate accepts site:atl_hq_01 keys.
    for atom in atom_list:
        atype = getattr(atom, "atom_type", None)
        atype_str = atype.value if hasattr(atype, "value") else str(atype or "")
        val = getattr(atom, "value", None) or {}
        if not isinstance(val, dict):
            continue
        if atype_str == "physical_site":
            # v53.11: reject ALL/placeholder IDs at catalog level so
            # they don't enter as canonicals (was leaking via
            # graph_expansion-bridged physical_site atoms with id='ALL').
            _GENERIC_PLACEHOLDERS = {
                "all", "various", "tbd", "n/a", "na", "none", "unknown", "",
                "all sites", "all locations", "various sites",
                "site all", "site various",
            }
            for k in ("id", "site_id"):
                sid = val.get(k)
                if sid and isinstance(sid, str) and sid.strip():
                    s = sid.strip()
                    if s.lower() in _GENERIC_PLACEHOLDERS:
                        continue
                    _record(s, atom, tier=0)
            for k in ("name", "facility_name"):
                nm = val.get(k)
                if nm and isinstance(nm, str) and nm.strip():
                    s = nm.strip()
                    if s.lower() in _GENERIC_PLACEHOLDERS:
                        continue
                    _record(s, atom, tier=0)
            for nm in (val.get("names") or val.get("aliases") or val.get("alternative_names") or []):
                if isinstance(nm, str) and nm.strip():
                    s = nm.strip()
                    if s.lower() in _GENERIC_PLACEHOLDERS:
                        continue
                    _record(s, atom, tier=0)
        elif atype_str in ("site_allocation", "site_attribute", "site_access_window",
                           "site_access_restriction", "site_room_mix",
                           "site_infrastructure", "site_implementation_note",
                           "site_budget", "task", "milestone_phase",
                           "integration_checkpoint", "cutover_step"):
            # These atoms reference a site_id in value.site / value.site_id —
            # those references are reliable site IDs (came from a structured
            # parser column).
            for k in ("site", "site_id", "scope", "applies_to"):
                sid = val.get(k)
                if sid and isinstance(sid, str) and sid.strip():
                    s = sid.strip()
                    # Skip placeholders like "all" / "various"
                    if s.lower() in {"all", "various", "tbd", "n/a", "none", ""}:
                        continue
                    _record(s, atom, tier=0)

    # Tier 1: every atom in a Locations section contributes its
    # raw_text as a candidate site list. We split on commas / newlines
    # / semicolons since these sections often list sites comma-
    # separated or one-per-line.
    for atom in atom_list:
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
                _record(piece, atom, 1)

    # Tier 2: address-anchored phrases. For every atom containing a
    # US street address, capture proper-noun phrases within 120 chars
    # of the address span.
    proper_run = re.compile(
        r"\b([A-Z][A-Za-z0-9'.\-]+(?:\s+[A-Z][A-Za-z0-9'.\-]+){0,6})\b"
    )
    for atom in atom_list:
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
                    _record(phrase, atom, 2)

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
    for atom in atom_list:
        raw = getattr(atom, "raw_text", None) or ""
        if not raw:
            continue
        for regex in (strong_tail_re, medical_re, station_re):
            for m in regex.finditer(raw):
                phrase = m.group(1)
                _record(phrase, atom, 3)

    # Tier 4: SECTION HEADING IS THE SITE NAME
    # Real spec docs are often organized one-section-per-site, with
    # the section heading being the building name in caps (e.g.
    # "WESLEY SCHOOL", "DISTRICT WIDE CRAIG CAMPUS", "MEMORIAL
    # HOSPITAL"). Walk every atom's section_path tokens: if any
    # token ends with a strong facility tail, it's a site declaration.
    _STRONG_HEADING_TAILS = {
        "school", "elementary", "middle", "high", "primary", "academy",
        "campus", "hospital", "clinic", "library", "warehouse",
        "headquarters", "pavilion", "annex", "complex", "wing",
        "fieldhouse", "stadium", "auditorium", "gymnasium",
        "courthouse", "auditorium", "stadium",
    }
    for atom in atom_list:
        try:
            refs = getattr(atom, "source_refs", None) or []
            if not refs:
                continue
            locator = getattr(refs[0], "locator", None) or {}
            if not isinstance(locator, dict):
                continue
            section_path = locator.get("section_path")
            if not isinstance(section_path, list):
                continue
            for raw_heading in section_path:
                if not isinstance(raw_heading, str):
                    continue
                heading = raw_heading.strip()
                if not (2 <= len(heading.split()) <= 7):
                    continue
                # Skip headings that are mostly punctuation or
                # section numbers ("1.1 GENERAL", "SECTION 00 11 13")
                if re.match(r"^[\d.\-\s]+$", heading):
                    continue
                if heading.lower().startswith(("section ", "division ", "article ", "part ", "appendix ", "exhibit ", "attachment ", "schedule ", "chapter ")):
                    continue
                last = heading.split()[-1].lower().rstrip(":,.")
                if last in _STRONG_HEADING_TAILS:
                    _record(heading, atom, 4)
        except Exception:
            continue

    # Tier 5: EXPLICIT "SITES:" / "LOCATIONS:" LABEL
    # Body text like "Sites: Wesley School, Career Tech Center,
    # Lakeshore Learning Center" or "Buildings: A, B, C" — parse
    # the post-colon list as sites.
    _LABEL_LIST_RE = re.compile(
        r"\b(?:sites?|locations?|buildings?|facilities|facility list|"
        r"schools?|premises|properties|stores?|branches?|campuses)\s*"
        r"[:—\-]\s*(.{4,600})",
        re.IGNORECASE,
    )
    for atom in atom_list:
        raw = getattr(atom, "raw_text", None) or ""
        if not raw or len(raw) < 12:
            continue
        for m in _LABEL_LIST_RE.finditer(raw):
            list_text = m.group(1)
            # Cut off at the next likely paragraph break
            stop = re.search(r"\n\s*\n|\. [A-Z]|;\s*[A-Z][a-z]+\s", list_text)
            if stop:
                list_text = list_text[:stop.start()]
            for piece in re.split(r"[,;\n\r]+|\s{3,}|\bAND\b|\band\b", list_text):
                piece = piece.strip(" .-•*\t()[]")
                if 4 <= len(piece) <= 120 and any(c.isupper() for c in piece):
                    _record(piece, atom, 5)

    # Tier 6: "THE FOLLOWING X" SEMANTIC PATTERN
    # Body text like "the following sites are included: ...",
    # "covers these locations:", "listed buildings include:".
    _SEMANTIC_LIST_RE = re.compile(
        r"\b(?:the following|these|listed|including)\s+"
        r"(?:sites?|locations?|buildings?|facilities|schools?|stores?|"
        r"premises|properties|campuses|branches?)"
        r"(?:\s+(?:are|include|listed|covered|served))?"
        r"\s*[:—\-]?\s*(.{4,800})",
        re.IGNORECASE,
    )
    for atom in atom_list:
        raw = getattr(atom, "raw_text", None) or ""
        if not raw or len(raw) < 20:
            continue
        for m in _SEMANTIC_LIST_RE.finditer(raw):
            list_text = m.group(1)
            stop = re.search(r"\n\s*\n|\. [A-Z]|;\s*[A-Z][a-z]+\s", list_text)
            if stop:
                list_text = list_text[:stop.start()]
            for piece in re.split(r"[,;\n\r]+|\s{3,}|\bAND\b|\band\b", list_text):
                piece = piece.strip(" .-•*\t()[]")
                if 4 <= len(piece) <= 120 and any(c.isupper() for c in piece):
                    _record(piece, atom, 6)

    # ─────────── CROSS-DOC VALIDATION (Phase A) ───────────
    # A candidate site survives iff ANY of:
    #   (a) it was discovered via a HIGH-PRECISION tier:
    #         Tier 1 — Locations-section atom
    #         Tier 2 — Address-anchored (has a real address ±120 chars)
    #         Tier 4 — Section heading IS the site name
    #         Tier 5 — Explicit "Sites:" / "Locations:" label list
    #         Tier 6 — "The following sites are…" semantic pattern
    #   (b) it was mentioned in ≥2 distinct artifacts — real
    #       project sites appear in multiple docs (SOW + BOM +
    #       schedule, etc.) whereas spec references / random
    #       landmarks appear in only one
    #   (c) it was mentioned ≥2 times even in a single artifact
    #       (so a doc that lists the site once in body + once in a
    #       table footer still survives)
    # Drop iff ONLY discovered via Tier 3 (strong facility tail)
    # AND seen exactly once. Tier 3 alone is the weakest signal
    # because a phrase like "Chrysler Building" matches the tail
    # regex but isn't anchored to anything project-specific.
    # v53.2: tier 0 is physical_site atoms — they're the strongest
    # signal (already parsed from an authoritative site roster table)
    # so they always enter the catalog regardless of mention count.
    _HIGH_PRECISION_TIERS: set[int] = {0, 1, 2, 4, 5, 6}
    catalog: set[str] = set()
    # Track the physical_site canonical set so we can preserve it
    # if LLM extraction below decides to replace `catalog` wholesale.
    physical_site_phrases: set[str] = set()
    for phrase, ev in evidence.items():
        tiers = ev["tiers"]
        n_arts = len(ev["artifacts"])
        n_mentions = ev["mentions"]
        if tiers & _HIGH_PRECISION_TIERS:
            catalog.add(phrase)
        elif n_arts >= 2:
            catalog.add(phrase)
        elif n_mentions >= 2:
            catalog.add(phrase)
        # else: drop (single mention, only Tier 3 — the weak case)
        if 0 in tiers:
            physical_site_phrases.add(phrase)

    # ─────────── LLM EXTRACTION (Phase B — default-on) ───────────
    # The LLM reads the actual doc content (including section_path
    # HEADINGS so cover-page institutional names like "Geary County
    # Schools USD 475" reach the model) and produces the canonical
    # site list.
    #
    # The LLM-extracted set is the SOURCE OF TRUTH when available —
    # we do NOT merge with the regex catalog, because that introduces
    # false positives (table fragments, vendor names, etc.) that
    # the verify pass doesn't always catch.
    #
    # Behavior:
    #   default — try LLM (Ollama on tailnet), fall back to regex
    #             catalog if not reachable
    #   SOWSMITH_SITE_LLM_DISABLE=1 — force regex-only (air-gapped /
    #                                  test environments)
    #   SOWSMITH_SITE_LLM_VERIFY=1 — legacy alias that still enables
    #                                  the LLM path; kept for back-compat
    #
    # The structural 6-tier regex catalog is the deterministic
    # fallback for when the LLM is unreachable or returns nothing.
    import os
    llm_disabled = bool(os.environ.get("SOWSMITH_SITE_LLM_DISABLE"))
    hygiene_fn = None
    if not llm_disabled:
        try:
            from app.core.site_llm_verify import (
                extract_sites_with_llm,
                verify_sites_with_llm,
                ollama_reachable,
                apply_site_hygiene,
            )
            hygiene_fn = apply_site_hygiene
            # Quick reachability probe so offline environments don't
            # burn a 180-second request timeout per compile.
            if ollama_reachable():
                # Primary path: LLM reads the docs and tells us the
                # sites directly. No regex involvement.
                # v48: extract_sites_with_llm now returns list[dict] of
                # structured site objects with attributes (mdf_idf,
                # access_window, etc.). Build the set[str] catalog from
                # all ids + names so phrase_is_in_catalog() works
                # unchanged. Cache the full attribute dicts so entity
                # enrichment can later attach them to site entities.
                llm_site_objects = extract_sites_with_llm(atom_list)
                # v48 supplemental: complete truncated PDF values
                try:
                    from app.core.entity_resolution import complete_truncated_site_values
                    llm_site_objects = complete_truncated_site_values(llm_site_objects)
                except Exception:
                    pass
                if llm_site_objects:
                    catalog = set()
                    for _s in llm_site_objects:
                        if _s.get("id"):
                            catalog.add(_s["id"])
                        catalog.update(_s.get("names") or [])
                    # v53.2: physical_site atoms (parsed from the
                    # authoritative roster table) are GROUND TRUTH —
                    # never let the LLM extraction drop them. Union back in.
                    catalog |= physical_site_phrases
                    # Cache structured attrs keyed by id + any alias.
                    import app.core.site_detection as _self
                    _self._llm_site_attr_cache = {
                        key: _s
                        for _s in llm_site_objects
                        for key in ([_s.get("id")] + list(_s.get("names") or []))
                        if key
                    }
                    # v53.2: also seed the LLM cache with physical_site
                    # atom rows. build_site_readiness uses this cache as
                    # alias_name → canonical_id; without seeding it from
                    # physical_site, a roster-emitted atom doesn't
                    # contribute to alias collapse.
                    for atom in atom_list:
                        atype = getattr(atom, "atom_type", None)
                        atype_str = atype.value if hasattr(atype, "value") else str(atype or "")
                        if atype_str != "physical_site":
                            continue
                        val = getattr(atom, "value", None) or {}
                        if not isinstance(val, dict):
                            continue
                        sid = val.get("id") or val.get("site_id") or ""
                        if not sid:
                            continue
                        names = []
                        for k in ("name", "facility_name"):
                            v = val.get(k)
                            if v and isinstance(v, str):
                                names.append(v.strip())
                        for nm in (val.get("names") or val.get("aliases") or []):
                            if isinstance(nm, str) and nm.strip():
                                names.append(nm.strip())
                        site_obj = {"id": sid, "names": names}
                        # Map both the id and every name to this canonical.
                        _self._llm_site_attr_cache[sid] = site_obj
                        for nm in names:
                            _self._llm_site_attr_cache[nm] = site_obj
                elif catalog:
                    # Extract returned nothing — try verify on the
                    # structural regex catalog as a fallback polish.
                    verified = verify_sites_with_llm(catalog, atom_list)
                    if verified:
                        catalog = verified
        except Exception:
            # Any LLM failure: keep the deterministic catalog.
            pass

    # Always apply hygiene to the final catalog. Even when LLM is
    # disabled / unreachable, this drops obvious form-field words,
    # generic nouns, and vendor brand names from the regex catalog
    # before returning.
    if hygiene_fn is None:
        try:
            from app.core.site_llm_verify import apply_site_hygiene as _hyg
            catalog = _hyg(catalog)
        except Exception:
            pass
    else:
        catalog = hygiene_fn(catalog)

    # v53.2: physical_site atom entries bypass hygiene — they're
    # ground truth and hygiene's vendor-name/proper-noun heuristics
    # can mis-drop legitimate facility names like "Brady Training".
    catalog |= physical_site_phrases

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
    # v53.6: also check normalized-form match against normalized
    # catalog entries. Catalog stores original case ("ATL-HQ") but
    # the phrase parameter is normalized to lowercase+space ("atl hq").
    norm_catalog = {_normalize(c) for c in catalog}
    if norm in norm_catalog:
        return True
    norm_words = norm.split()
    if not norm_words:
        return False
    for c in catalog:
        nc = _normalize(c)
        # v53.6: was `if len(c) < 8: continue` which dropped legitimate
        # short site IDs like "atl hq" (6 chars). Lower threshold to 4
        # AND allow shorter when the catalog entry has structure (digit
        # or hyphen — typical of site IDs like B12, MDC-01, ATL-HQ).
        c_has_id_shape = any(ch.isdigit() or ch == "-" for ch in c)
        min_len = 4 if c_has_id_shape else 6
        if len(nc) < min_len:
            continue
        c_words = nc.split()
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
