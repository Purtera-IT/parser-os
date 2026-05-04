"""Ontology gap detector — finds candidate aliases / patterns missing from the active domain pack.

The point of this module is to make Parser OS *teach itself* over time: when
the user runs a compile and reviews the output, they should immediately see
"these device-like phrases appeared in the artifacts but no pack knew about
them" — those become candidate aliases for tomorrow's pack.

Two streams of gaps are detected:

1. **Vocab gaps** — phrases that look like a device, exclusion, constraint,
   action, or instruction in the raw text but didn't trigger any pack
   pattern.  Detected with a small set of heuristics (token shape, plural
   forms, regex windows) plus negative-evidence (phrase isn't already in
   any device_alias / exclusion_pattern / etc).

2. **Entity gaps** — atoms whose ``entity_keys`` contain ``*:unknown``
   sentinels.  These are atoms the entity layer couldn't resolve.  We
   surface the raw_text and a "best guess" device classifier so the user
   can pick the right canonical name and add it to the pack.

The output is deterministic and ordered, so reruns produce stable gap
lists you can diff between sessions to track your pack maturity.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

from app.core.normalizers import normalize_text
from app.core.schemas import AtomType, EvidenceAtom
from app.domain.schemas import DomainPack


# Words that are obviously generic — don't surface them as candidate device aliases.
_GENERIC_NOUNS: frozenset[str] = frozenset(
    {
        "the", "and", "for", "with", "from", "that", "this", "will", "was", "are",
        "after", "before", "please", "site", "scope", "scopes", "project", "team",
        "vendor", "customer", "client", "owner", "device", "devices", "system",
        "systems", "phase", "schedule", "approval", "approve", "review", "note",
        "notes", "list", "lists", "thing", "things", "stuff", "item", "items",
        "page", "pages", "section", "sections", "draft", "version", "rev",
        "revision", "addendum", "subject", "from", "sent", "to", "cc", "bcc",
        "kindly", "thanks", "thank", "regards", "sincerely", "best", "team",
    }
)

# Tokens that strongly suggest a managed-services device or material when they appear adjacent.
_DEVICE_SHAPE_HINTS: tuple[str, ...] = (
    "panel", "controller", "reader", "camera", "switch", "router", "firewall",
    "ap", "access point", "antenna", "speaker", "amplifier", "horn", "strobe",
    "detector", "mic", "microphone", "display", "monitor", "tv", "codec",
    "projector", "drop", "jack", "outlet", "receptacle", "circuit", "breaker",
    "ups", "rack", "cabinet", "panel", "transformer", "disconnect", "valve",
    "actuator", "vav", "ahu", "rtu", "vrf", "thermostat", "sensor", "rectifier",
)

# Phrases that look like exclusion/constraint/instruction wording when they
# anchor a sentence but didn't fire a pack pattern.  Each tuple is (regex, gap_kind).
_EXCLUSION_SHAPE = re.compile(
    r"\b(?:not\s+included|by\s+(?:others|gc|ec|owner|client|landlord|tenant|"
    r"mech|hvac|electrical|carrier|leasing|lessor)|"
    r"out\s+of\s+scope|outside\s+scope|excluded|exclude\s+from|n\.?i\.?c\.?|"
    r"allowance(?:\s+only)?|alternate|tbd|to\s+be\s+determined|future\s+(?:phase|scope)|"
    r"phase\s+(?:2|3|ii|iii)|if\s+requested|placeholder|reuse\s+existing|"
    r"existing\s+to\s+remain|change\s+order(?:\s+required)?)\b",
    re.IGNORECASE,
)

_CONSTRAINT_SHAPE = re.compile(
    r"\b(?:after[-\s]?hours|night\s+work|weekends?|escort\s+required|badge\s+required|"
    r"work\s+window|access\s+window|maintenance\s+window|shutdown\s+window|"
    r"loading\s+dock|hot\s+work\s+permit|loto|lock\s+out\s+tag\s+out|"
    r"confined\s+space|fall\s+protection|harness\s+required|hard\s+hat|ppe\s+required|"
    r"prevailing\s+wage|union\s+labor|certified\s+payroll)\b",
    re.IGNORECASE,
)

_INSTRUCTION_SHAPE = re.compile(
    r"\b(?:please\s+(?:add|remove|include|provide|install|coordinate|confirm|update|"
    r"replace|relocate)|kindly\s+(?:add|remove)|approved\s+to\s+proceed|"
    r"proceed\s+with|we\s+(?:want|need|approve|agree)|hold\s+off|cancel\s+scope|"
    r"revise\s+scope|reduce\s+scope|increase\s+scope|move\s+forward)\b",
    re.IGNORECASE,
)

# Capitalized phrases that look like proper nouns, e.g. "Building A West", "Suite 200".
_PROPER_NOUN_PHRASE = re.compile(r"\b([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+){0,3})\b")

# Vendor-shape detection.  Flags 1- to 4-word capitalized brand-names
# that don't appear in any pack alias.  See PRODUCTION_GAPS P1.5.
_VENDOR_SHAPE = re.compile(r"\b([A-Z][a-zA-Z0-9.&\-]+(?:\s+[A-Z][a-zA-Z0-9.&\-]+){0,3})\b")
_VENDOR_INDICATOR_WORDS = (
    " technologies", " technology", " systems", " corporation", " inc",
    " incorporated", " llc", " ltd", " limited", " co.", " communications",
    " electric", " electronics", " industries", " international", " global",
    " networks", " labs", " solutions", " group", " enterprises", " manufacturing",
    " controls", " security", " automation", " software",
)
# A table of well-known vendors so we can score "this surface looks like
# a vendor" even without trailing words like "Inc".  Keys are lowercase
# canonical ids; values are the surface-form aliases the matcher
# recognizes.  Mirrors the cross-pack vendor catalog in
# app.core.entity_extraction so the gap detector and the entity
# extractor agree on what's already known.
_KNOWN_VENDORS: frozenset[str] = frozenset(
    {
        "cisco", "aruba", "juniper", "ubiquiti", "extreme", "fortinet", "ruckus",
        "genetec", "milestone", "axis", "pelco", "sony", "hanwha", "bosch",
        "avigilon", "exacqvision", "exacq", "live earth", "briefcam",
        "lenel", "mercury", "hid", "xceedid", "aptiq", "schlage", "securitron",
        "lifesafety power", "talk-a-phone", "talkaphone", "nedap",
        "lg", "planar", "atlona", "chief", "middle atlantic", "furman",
        "evolution", "legrand", "muxlab", "comprehensive",
        "tridium", "niagara", "vykon", "jace", "honeywell", "notifier",
        "siemens", "johnson controls", "metasys", "simplex", "schneider",
        "alc", "automated logic", "distech", "trane", "carrier", "edwards",
        "mircom", "zenitel", "grandstream", "aiphone", "rave", "blackberry",
        "regroup", "alertus", "dell", "hpe", "lenovo", "microsoft", "vmware",
    }
)

# Part-number / SKU shape — matches Cisco-style (CW9166I-B), HPE
# (J9145A), DICENTIS (DCNM-DVT908), Streamvault (SV-2030E-AC), …
_PART_NUMBER_SHAPE = re.compile(
    r"\b("
    r"[A-Z][A-Z0-9]{1,9}(?:[-/][A-Z0-9]{1,12}){1,6}"
    r"|"
    r"[A-Z]{2,5}[0-9]{2,6}[A-Z]{0,3}"
    r")\b"
)

# Stop-list for proper-noun runs that should never be flagged as a
# vendor or site (RFP boilerplate, government bodies, column headers,
# etc.).  Comparison is case-insensitive and substring-aware so
# "The Secure Networks Act" matches the bare entry "Secure Networks
# Act".
_GAP_STOPLIST_ENTRIES: tuple[str, ...] = (
    "Federal Communications Commission",
    "Federal Communications",
    "Department of Justice",
    "Department of Industrial Relations",
    "Department of Homeland Security",
    "United States Department",
    "State of California",
    "State of Texas",
    "City of Mobile",
    "City of Chicago",
    "City of Milwaukee",
    "City of Santa Monica",
    "Public Records Act",
    "Iran Contracting Act",
    "Secure Networks Act",
    "Patriot Act",
    "Americans with Disabilities Act",
    "Civil Rights Act",
    "Freedom of Access Act",
    "California Public Records",
    "Public Contract Code",
    "Labor Code",
    "Education Code",
    "Government Code",
    "Penal Code",
    "Public Works",
    "Property Damage",
    "Bodily Injury",
    "Personal Injury",
    "Workers Compensation",
    "Workers' Compensation",
    "General Liability",
    "Auto Liability",
    "Professional Liability",
    "Umbrella Liability",
    # RFP / form column-header words (very common in BOM tables that
    # look "vendor-shaped" because they sit next to a SKU).
    "Part Number",
    "Part Numbers",
    "Description",
    "Descriptions",
    "Qty",
    "Quantity",
    "Manufacturer",
    "Manufacturers",
    "Model",
    "Models",
    "Item",
    "Items",
    "Software",
    "Hardware",
    "Service Provider",
    "Service Substitution",
    "Service Level Agreement",
    "Annual Support",
    "Recurring Annual",
    "Support Costs",
    "Tax ID",
    "Federal Taxpayer Number",
    "Authorized Representative",
    "Cost Proposal",
    "Cost Proposals",
    "Project Description",
    "Equipment List",
    "Equipment Service",
    "Cover Letter",
    "Letter of Agreement",
    "Letter of Transmittal",
    "Section Header",
    "Page",
    "Pages",
    "Schedule",
    "Schedules",
    "Title Page",
    "Table of Contents",
    "International Baccalaureate Diploma",
    "Technology Services Department",
)
_GAP_STOPLIST_LOWER: frozenset[str] = frozenset(s.lower() for s in _GAP_STOPLIST_ENTRIES)


def _phrase_in_gap_stoplist(phrase: str) -> bool:
    """True iff ``phrase`` should be filtered as a non-vendor non-site
    proper-noun run.

    Comparison is case-insensitive and accepts plural / leading-article
    variants (so "The Secure Networks Act" filters via "Secure Networks
    Act" and "Federal Communications Commissions" filters via the
    singular form).
    """
    if not phrase:
        return False
    needle = phrase.lower().strip()
    if needle.startswith("the "):
        needle = needle[4:]
    if needle.endswith("s") and not needle.endswith("ss"):
        # Try the singular form too.
        singular = needle[:-1]
        if singular in _GAP_STOPLIST_LOWER:
            return True
    if needle in _GAP_STOPLIST_LOWER:
        return True
    # Substring containment (right side): "Federal Communications
    # Commissions" contains "Federal Communications" prefix.
    for entry in _GAP_STOPLIST_LOWER:
        if entry in needle or needle in entry:
            # Only short overlaps are coincidence; require the entry
            # itself to be ≥ 6 chars to reduce false suppression.
            if len(entry) >= 6:
                return True
    return False


def _flatten_pack_phrases(pack: DomainPack) -> dict[str, set[str]]:
    """Return ``{kind: {normalized_phrase, ...}}`` for everything the pack already knows."""
    flat: dict[str, set[str]] = {
        "device": set(),
        "site": set(),
        "constraint": set(),
        "exclusion": set(),
        "instruction": set(),
        "action": set(),
    }
    for canonical, aliases in (pack.device_aliases or {}).items():
        flat["device"].add(normalize_text(canonical.replace("_", " ")))
        for alias in aliases:
            flat["device"].add(normalize_text(alias))
    for entity in pack.entity_types or []:
        target = entity.name.lower()
        if target in flat:
            flat[target].add(normalize_text(entity.name))
            for alias in entity.aliases:
                flat[target].add(normalize_text(alias))
    for patterns in (pack.constraint_patterns or {}).values():
        for pattern in patterns:
            flat["constraint"].add(normalize_text(pattern))
    for pattern in pack.exclusion_patterns or []:
        flat["exclusion"].add(normalize_text(pattern))
    for pattern in pack.customer_instruction_patterns or []:
        flat["instruction"].add(normalize_text(pattern))
    for actions in (pack.action_aliases or {}).values():
        for pattern in actions:
            flat["action"].add(normalize_text(pattern))
    return flat


def _phrase_already_known(phrase: str, kind: str, pack_phrases: dict[str, set[str]]) -> bool:
    needle = normalize_text(phrase)
    if not needle:
        return True
    bucket = pack_phrases.get(kind) or set()
    if needle in bucket:
        return True
    for known in bucket:
        if known and (known in needle or needle in known):
            return True
    return False


def _atom_blob(atom: EvidenceAtom) -> str:
    parts = [atom.raw_text or "", atom.normalized_text or ""]
    if isinstance(atom.value, dict):
        parts.append(" ".join(str(v) for v in atom.value.values() if isinstance(v, (str, int, float))))
    return " ".join(parts)


def _candidate_device_phrases(text: str) -> list[str]:
    """Return phrases that look like device mentions (heuristic)."""
    hits: list[str] = []
    lowered = " " + normalize_text(text) + " "
    for hint in _DEVICE_SHAPE_HINTS:
        for m in re.finditer(rf"\b([a-z][a-z0-9-]*\s+){{0,2}}{re.escape(hint)}s?\b", lowered):
            phrase = m.group(0).strip()
            if any(g in phrase.split() for g in _GENERIC_NOUNS):
                tokens = [t for t in phrase.split() if t not in _GENERIC_NOUNS]
                phrase = " ".join(tokens)
            phrase = re.sub(r"^(the|a|an|our|your|their|its)\s+", "", phrase)
            if phrase and len(phrase) >= 3:
                hits.append(phrase)
    return hits


def _candidate_site_phrases(text: str) -> list[str]:
    hits: list[str] = []
    for m in _PROPER_NOUN_PHRASE.finditer(text or ""):
        phrase = m.group(1).strip()
        lowered = phrase.lower()
        if any(t in lowered for t in ("building", "bldg", "campus", "wing", "suite", "site", "store", "branch", "floor", "level", "school", "office", "plant", "warehouse", "facility")):
            hits.append(phrase)
    return hits


# Single-word capitalized tokens that are likely vendor names but have
# no SKU neighbor or "Inc." suffix.  We surface these conservatively:
# a token must be ≥ 4 chars, start with a capital, contain a lowercase
# letter (so "PARKING" / "RFP" / "FCC" are out), and not be on a
# common-noun stop-list.
_SINGLE_TOKEN_VENDOR_SHAPE = re.compile(r"\b([A-Z][a-zA-Z\-]{3,18}(?:[A-Z][a-z]+)*)\b")
_SINGLE_TOKEN_COMMON_NOUNS: frozenset[str] = frozenset(
    {
        "Section", "Article", "Page", "Notice", "Exhibit", "Attachment",
        "Appendix", "Chapter", "Schedule", "Table", "Figure", "Title",
        "Vendor", "Customer", "Owner", "Bidder", "Offeror", "Respondent",
        "Contractor", "Supplier", "Manufacturer", "District", "City",
        "County", "State", "Country", "Federal", "Local", "Public",
        "Department", "Office", "Agency", "Authority", "Board",
        "Committee", "Council", "Court", "University", "College",
        "School", "Hospital", "Center", "Centre", "Building", "Floor",
        "Room", "Site", "Project", "Phase", "Step", "Item", "Type",
        "Model", "Make", "Brand", "Quantity", "Amount", "Price", "Cost",
        "Total", "Subtotal", "Date", "Time", "Year", "Month", "Day",
        "Week", "Address", "Email", "Phone", "Fax", "Number", "Name",
        "Title", "Role", "Position", "Yes", "No", "Maybe", "TBD",
        "All", "None", "Each", "Every", "Some", "Any", "Many", "Few",
        "Description", "Specification", "Requirement", "Constraint",
        "Decision", "Action", "Note", "Question", "Answer", "Comment",
        "Approved", "Required", "Pending", "Active", "Closed", "Open",
        "Completed", "Finished", "Started", "Continued", "Renewed",
        "America", "American", "United", "States", "World", "Global",
        "International", "National", "Regional", "Local", "Municipal",
        "Northern", "Southern", "Eastern", "Western", "Central",
        "Modify", "Update", "Replace", "Install", "Remove", "Add",
        "Reuse", "Existing", "New", "Old", "Future", "Past", "Current",
        "Several", "Multiple", "Various", "Different", "Similar", "Same",
        "Power", "Voltage", "Amperage", "Wattage", "Bandwidth", "Speed",
        "Network", "Internet", "Wireless", "Wireless", "Wired",
        "Camera", "Server", "Workstation", "Computer", "Laptop", "Tablet",
        "Phone", "Mobile", "Desktop", "Display", "Monitor", "Speaker",
        "Microphone", "Switch", "Router", "Cable", "Connector",
        "Outlet", "Receptacle", "Drop", "Jack", "Plug",
        "Software", "Hardware", "License", "Licence", "Subscription",
        "Service", "Support", "Maintenance", "Warranty", "Insurance",
        "Indemnity", "Liability", "Compliance", "Certification",
        # VT-RFP-specific common nouns.
        "Andrews", "Drillfield", "Pratt", "Perry", "Surveillance",
        "Implementation", "Integration", "Solution", "Capability",
        "Functionality", "Performance", "Scalability",
        # Months and weekdays.
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
        "Saturday", "Sunday",
    }
)


def _candidate_single_token_vendor_phrases(text: str, pack: DomainPack) -> list[str]:
    """Find bare single-word capitalized tokens likely to be vendor names.

    Targets cases like "ThyssenKrupp", "ESRI", "ArcSDE", "Splunk",
    "Genetec" — vendors that appear without an "Inc." suffix or a
    nearby SKU.  We require the token to be either:
    - CamelCase with ≥1 lowercase letter ("ThyssenKrupp", "OpenGov"), OR
    - 4–8 characters, all uppercase ("ESRI", "USAC", "FCC"), AND
    - not in the common-noun stop-list, AND
    - not in the cross-pack vendor catalog (already known), AND
    - not a single dictionary word (rough heuristic via stoplist).
    """
    if not text:
        return []
    hits: list[str] = []
    for m in _SINGLE_TOKEN_VENDOR_SHAPE.finditer(text):
        token = m.group(1)
        if token in _SINGLE_TOKEN_COMMON_NOUNS:
            continue
        # Requires either a CamelCase pattern or all-caps acronym.
        is_camel = (
            any(c.isupper() for c in token[1:])  # has internal uppercase
            and any(c.islower() for c in token)  # plus lowercase = CamelCase
        )
        is_acronym = (
            len(token) <= 8
            and token == token.upper()
            and len(token) >= 4
        )
        if not (is_camel or is_acronym):
            continue
        if _phrase_in_gap_stoplist(token):
            continue
        # Skip if already in a known vendor surface form.
        lowered = token.lower()
        is_known = False
        for vendor in _KNOWN_VENDORS:
            if vendor in lowered:
                is_known = True
                break
        if is_known:
            continue
        # Skip if the token is the start of a longer multi-word
        # proper-noun phrase (caught by the multi-word detector
        # already).
        ctx_after = text[m.end(): m.end() + 30]
        if re.match(r"\s+[A-Z][a-z]+\s+[A-Z]", ctx_after):
            continue
        hits.append(token)
    return hits


def _candidate_vendor_phrases(text: str, pack: DomainPack) -> list[str]:
    """Return phrases that look like vendor / brand names.

    Heuristic: a capitalized 1-4 word run that
    - is *not* in the cross-pack `_KNOWN_VENDORS` table and
    - is *not* in the `_GAP_STOPLIST`, and
    - either ends in a vendor-indicator word ("Technologies", "Inc.", …)
      OR is a single capitalized token that's also followed by a
      part-number SKU within a small window (likely vendor-of-SKU).

    See PRODUCTION_GAPS P1.5.  Without this, real vendor-name gaps
    like "Talk-A-Phone", "Streamvault", "NEDAP" never surfaced.
    """
    if not text:
        return []
    hits: list[str] = []
    for m in _VENDOR_SHAPE.finditer(text):
        phrase = m.group(1).strip()
        lowered = phrase.lower()
        if _phrase_in_gap_stoplist(phrase):
            continue
        # Skip phrases that are already in the known-vendors table —
        # they're handled by the entity extractor as `vendor:*`.
        is_known = False
        for vendor in _KNOWN_VENDORS:
            if vendor in lowered:
                is_known = True
                break
        if is_known:
            continue
        # Skip phrases that look like SKUs themselves — they belong
        # in part_number gaps, not vendor gaps.
        if _PART_NUMBER_SHAPE.fullmatch(phrase):
            continue
        # Strong signal A: vendor-indicator word follows.
        padded = " " + lowered + " "
        has_indicator = any(ind in padded for ind in _VENDOR_INDICATOR_WORDS)
        # Strong signal B: a SKU-shaped token is within 60 chars to
        # the right (vendor-of-SKU pattern).
        has_sku_neighbor = False
        match_end = m.end()
        window = text[match_end : match_end + 60]
        if _PART_NUMBER_SHAPE.search(window):
            has_sku_neighbor = True
        if not (has_indicator or has_sku_neighbor):
            continue
        # Skip already-known device aliases / examples (those will
        # surface as device gaps elsewhere if relevant).
        if any(
            normalize_text(alias) == normalize_text(phrase)
            for aliases in (pack.device_aliases or {}).values()
            for alias in aliases
        ):
            continue
        # De-noise common construction-doc phrases that pass the
        # vendor-indicator check coincidentally.
        if any(stop in phrase for stop in ("Notice ", "Section ", "Article ", "Page ")):
            continue
        hits.append(phrase)
    return hits


def _candidate_part_number_phrases(text: str, pack: DomainPack) -> list[str]:
    """Return SKU-shaped tokens not already in the pack.

    A SKU is a strong signal of a brand-specific device and the most
    direct path to expanding `device_aliases`.  Conservative: only
    flags multi-segment hyphenated SKUs (≥1 hyphen) so we don't surface
    every uppercase acronym.
    """
    if not text:
        return []
    pack_skus: set[str] = set()
    for aliases in (pack.device_aliases or {}).values():
        for alias in aliases:
            for m in _PART_NUMBER_SHAPE.finditer(alias):
                pack_skus.add(m.group(1).upper())

    hits: list[str] = []
    for m in _PART_NUMBER_SHAPE.finditer(text):
        sku = m.group(1)
        # Need a hyphen or slash to be a strong SKU.  Plain alphanum
        # acronyms like "RFP" or "NFPA" are too generic.
        if "-" not in sku and "/" not in sku:
            continue
        if sku.upper() in pack_skus:
            continue
        hits.append(sku)
    return hits


def detect_ontology_gaps(
    *,
    atoms: list[EvidenceAtom],
    pack: DomainPack,
) -> dict[str, Any]:
    """Analyze atoms against the active pack and return a structured gap report.

    Shape::
        {
          "summary": {"vocab_gap_count": int, "entity_gap_count": int},
          "vocab_gaps": [
              {"kind": "device", "phrase": "horn strobe addressable",
               "atom_ids": [...], "sample_text": "..."}
          ],
          "entity_gaps": [
              {"unknown_key": "device:unknown", "atom_ids": [...],
               "sample_text": "...", "best_guess": "ip camera"}
          ],
        }
    """
    pack_phrases = _flatten_pack_phrases(pack)

    # ── Vocab gaps: bucket candidate phrases that aren't already in the pack.
    vocab_buckets: dict[str, dict[str, list[str]]] = {
        "device": defaultdict(list),
        "site": defaultdict(list),
        "vendor": defaultdict(list),
        "part_number": defaultdict(list),
        "exclusion": defaultdict(list),
        "constraint": defaultdict(list),
        "instruction": defaultdict(list),
    }
    sample_text_by_phrase: dict[tuple[str, str], str] = {}

    for atom in atoms:
        blob = _atom_blob(atom)
        if not blob.strip():
            continue
        # Devices
        for phrase in _candidate_device_phrases(blob):
            if not _phrase_already_known(phrase, "device", pack_phrases):
                vocab_buckets["device"][phrase].append(atom.id)
                sample_text_by_phrase.setdefault(("device", phrase), atom.raw_text)
        # Sites
        for phrase in _candidate_site_phrases(blob):
            if not _phrase_already_known(phrase, "site", pack_phrases):
                vocab_buckets["site"][phrase].append(atom.id)
                sample_text_by_phrase.setdefault(("site", phrase), atom.raw_text)
        # Vendors (P1.5 — use the cross-pack catalog, not the
        # narrowly-typed pack_phrases bucket)
        for phrase in _candidate_vendor_phrases(blob, pack):
            vocab_buckets["vendor"][phrase].append(atom.id)
            sample_text_by_phrase.setdefault(("vendor", phrase), atom.raw_text)
        # Single-word vendor candidates (Week 5 — surfaces tokens like
        # "ThyssenKrupp" / "ESRI" / "ArcSDE" that have no SKU neighbor
        # and no "Inc." suffix).
        for phrase in _candidate_single_token_vendor_phrases(blob, pack):
            vocab_buckets["vendor"][phrase].append(atom.id)
            sample_text_by_phrase.setdefault(("vendor", phrase), atom.raw_text)
        # Part numbers (P1.5)
        for sku in _candidate_part_number_phrases(blob, pack):
            vocab_buckets["part_number"][sku].append(atom.id)
            sample_text_by_phrase.setdefault(("part_number", sku), atom.raw_text)
        # Exclusion / constraint / instruction shapes
        for kind, regex in (
            ("exclusion", _EXCLUSION_SHAPE),
            ("constraint", _CONSTRAINT_SHAPE),
            ("instruction", _INSTRUCTION_SHAPE),
        ):
            for m in regex.finditer(blob):
                phrase = m.group(0).strip().lower()
                if not _phrase_already_known(phrase, kind, pack_phrases):
                    vocab_buckets[kind][phrase].append(atom.id)
                    sample_text_by_phrase.setdefault((kind, phrase), atom.raw_text)

    vocab_gaps: list[dict[str, Any]] = []
    for kind, buckets in vocab_buckets.items():
        for phrase, atom_ids in buckets.items():
            vocab_gaps.append(
                {
                    "kind": kind,
                    "phrase": phrase,
                    "occurrences": len(atom_ids),
                    "atom_ids": sorted(set(atom_ids))[:8],
                    "sample_text": (sample_text_by_phrase.get((kind, phrase)) or "").strip()[:300],
                }
            )

    # ── Entity gaps: atoms grounded on `*:unknown` sentinels.
    entity_gaps_buckets: dict[str, list[EvidenceAtom]] = defaultdict(list)
    for atom in atoms:
        for key in atom.entity_keys:
            if key.endswith(":unknown"):
                entity_gaps_buckets[key].append(atom)
    entity_gaps: list[dict[str, Any]] = []
    for key, group in entity_gaps_buckets.items():
        sample = sorted(group, key=lambda a: a.id)[0]
        # Best-guess: count which device aliases appear in any atom raw_text.
        guess: str | None = None
        if key.startswith("device:"):
            counts: Counter[str] = Counter()
            for atom in group:
                lowered = normalize_text(atom.raw_text)
                for canonical, aliases in (pack.device_aliases or {}).items():
                    for alias in aliases:
                        if normalize_text(alias) in lowered:
                            counts[canonical] += 1
                            break
            if counts:
                guess = counts.most_common(1)[0][0]
        entity_gaps.append(
            {
                "unknown_key": key,
                "occurrences": len(group),
                "atom_ids": sorted(a.id for a in group)[:12],
                "sample_text": (sample.raw_text or "").strip()[:300],
                "best_guess_canonical": guess,
            }
        )

    # Deterministic ordering: most occurrences first, then phrase asc.
    vocab_gaps.sort(key=lambda row: (-row["occurrences"], row["kind"], row["phrase"]))
    entity_gaps.sort(key=lambda row: (-row["occurrences"], row["unknown_key"]))

    return {
        "summary": {
            "vocab_gap_count": len(vocab_gaps),
            "entity_gap_count": len(entity_gaps),
            "by_kind": dict(Counter(row["kind"] for row in vocab_gaps)),
            "active_pack_id": pack.pack_id,
            "active_pack_version": pack.version,
        },
        "vocab_gaps": vocab_gaps,
        "entity_gaps": entity_gaps,
    }


def render_gaps_markdown(report: dict[str, Any]) -> str:
    """Render the gap report as a human-readable markdown checklist for review.

    Each gap line is a checkbox a reviewer can tick once they've added it to
    the appropriate pack — or strike through if it's noise.
    """
    summary = report.get("summary") or {}
    lines: list[str] = []
    lines.append("# Ontology gaps — candidates to add to the domain pack")
    lines.append("")
    lines.append(
        f"_Active pack: **{summary.get('active_pack_id', 'unknown')}** "
        f"v{summary.get('active_pack_version', '?')} • "
        f"{summary.get('vocab_gap_count', 0)} vocab gaps • "
        f"{summary.get('entity_gap_count', 0)} entity gaps_"
    )
    lines.append("")
    lines.append("Tick each item once you've decided what to do:")
    lines.append("- [ ] add to pack")
    lines.append("- [ ] dismiss (true negative / noise)")
    lines.append("- [ ] needs more samples before deciding")
    lines.append("")

    # Vocab gaps grouped by kind
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in report.get("vocab_gaps", []):
        by_kind[row["kind"]].append(row)
    for kind in sorted(by_kind):
        rows = by_kind[kind]
        lines.append(f"## {kind.title()} candidates ({len(rows)})")
        lines.append("")
        for row in rows[:80]:
            lines.append(
                f"- [ ] **`{row['phrase']}`** — {row['occurrences']}× • "
                f"sample: _{(row['sample_text'] or '').replace(chr(10), ' ')[:160]}_"
            )
        lines.append("")

    # Entity gaps
    entity_gaps = report.get("entity_gaps") or []
    if entity_gaps:
        lines.append(f"## Entity resolution gaps ({len(entity_gaps)})")
        lines.append("")
        lines.append("Atoms below were tagged `*:unknown` because no pack alias matched.  "
                     "Pick a canonical name and add the missing alias to the pack.")
        lines.append("")
        for row in entity_gaps:
            guess_note = f" • best guess: **`{row['best_guess_canonical']}`**" if row.get("best_guess_canonical") else ""
            lines.append(
                f"- [ ] `{row['unknown_key']}` ({row['occurrences']}×){guess_note}"
            )
            lines.append(f"  - sample: _{(row.get('sample_text') or '').replace(chr(10), ' ')[:200]}_")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


__all__ = ["detect_ontology_gaps", "render_gaps_markdown"]
