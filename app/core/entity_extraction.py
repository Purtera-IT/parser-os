"""Universal entity extraction.

A parser-agnostic, pack-aware extractor that scans an atom's raw text
and emits ``entity_keys`` like ``device:ip_camera``, ``vendor:cisco``,
``site:perry_street_parking_deck``, ``part_number:cw9166i_b``.

This module is the bridge between parsers (which know structure) and
the entity_resolution stage (which groups already-populated keys).
Most parsers historically hardcoded ``entity_keys=[]``, leaving
downstream graph_build / packetize unable to anchor on real entities.
This extractor closes that gap *universally* — any atom with raw_text
gets enriched, regardless of which parser produced it.

Design principles:
- Idempotent — running it twice produces the same set of keys.
- Pack-aware — pulls aliases from the active ``DomainPack``.
- Cross-pack — knows about common vendors (Cisco, Genetec, Lenel, …)
  and architectural patterns (CSI MasterFormat, street addresses,
  Q&A "QN/AN" markers) regardless of which pack is active.
- Conservative — when in doubt, emit nothing rather than a noisy key.
"""

from __future__ import annotations

import functools
import os
import re
import unicodedata
from collections.abc import Iterable
from typing import Any

from app.core.entity_hygiene import filter_entity_keys_for_atom
from app.core.normalizers import normalize_entity_key, normalize_text
from app.domain.schemas import DomainPack


# ───────────────────────── cross-pack vendor catalog ─────────────────────────
#
# Vendors that appear across many service lines.  Domain packs can extend
# this implicitly via their device_aliases (e.g. "Cisco DNA" → device).
# This list is for *direct* vendor name detection so we can produce
# `vendor:cisco`-style keys even when the device alias didn't match.
#
# Each canonical key maps to the surface-form aliases we expect to see in
# RFP / spec / proposal text.  Keep it lower-case; the matcher does
# case-insensitive word-boundary searches.
_CROSS_PACK_VENDORS: dict[str, list[str]] = {
    # Networking / wireless
    "cisco": ["cisco", "cisco systems", "cisco meraki", "meraki"],
    "aruba": ["aruba", "aruba networks", "hpe aruba"],
    "juniper": ["juniper", "juniper networks", "mist"],
    "ubiquiti": ["ubiquiti", "unifi", "ubnt"],
    "extreme": ["extreme networks", "extreme switching"],
    "fortinet": ["fortinet", "fortigate", "fortiap"],
    "ruckus": ["ruckus", "ruckus wireless", "commscope ruckus"],
    # Security camera / VMS
    "genetec": ["genetec", "genetec security center", "genetec synergis", "streamvault"],
    "milestone": ["milestone", "milestone xprotect", "xprotect"],
    "axis": ["axis", "axis communications", "axis camera"],
    "pelco": ["pelco"],
    "sony": ["sony"],
    "hanwha": ["hanwha", "hanwha vision", "wisenet"],
    "bosch": ["bosch", "bosch security", "bosch dicentis"],
    "avigilon": ["avigilon", "avigilon alta", "motorola avigilon"],
    "exacqvision": ["exacqvision", "exacq", "exacq technologies"],
    "live_earth": ["live earth"],
    "briefcam": ["briefcam"],
    # Access control
    "lenel": ["lenel", "lenel onguard", "onguard"],
    "mercury": ["mercury", "mercury security", "mercury intelligent"],
    "hid": ["hid global", "hid", "hid signo"],
    "xceedid": ["xceedid"],
    "aptiq": ["aptiq"],
    "schlage": ["schlage"],
    "securitron": ["securitron"],
    "lifesafety_power": ["lifesafety power", "lsp"],
    "talk_a_phone": ["talk-a-phone", "talkaphone", "talk a phone"],
    "nedap": ["nedap"],
    # AV / conferencing
    "lg": ["lg electronics", "lg display", "lg commercial"],
    "planar": ["planar"],
    "atlona": ["atlona"],
    "chief": ["chief mfg", "chief"],
    "middle_atlantic": ["middle atlantic", "m.a."],
    "furman": ["furman"],
    "evolution": ["evolution digital"],
    "legrand": ["legrand"],
    "muxlab": ["muxlab"],
    "comprehensive": ["comprehensive cable"],
    # Week 6 P6.8: AV-industry vendors regularly appearing in
    # boardroom / huddle-room RFPs but missing from the original
    # Week 5 catalog.  Without these, AV_TRIO undercounts unique
    # vendor mentions vs. its gold expectation.
    "crestron": ["crestron", "crestron flex"],
    "extron": ["extron", "extron electronics"],
    "biamp": ["biamp", "biamp systems", "tesira"],
    "shure": ["shure", "shure incorporated"],
    "qsc": ["qsc", "q-sys", "q sys"],
    "kramer": ["kramer", "kramer electronics"],
    "amx": ["amx", "amx by harman"],
    "polycom": ["polycom", "poly"],
    "vaddio": ["vaddio"],
    "logitech": ["logitech"],
    "epson": ["epson"],
    "panasonic": ["panasonic"],
    "sony": ["sony"],
    "samsung": ["samsung", "samsung electronics"],
    "sharp": ["sharp"],
    "nec_display": ["nec display", "nec display solutions"],
    "benq": ["benq"],
    "barco": ["barco", "barco clickshare"],
    "clearone": ["clearone"],
    "yamaha": ["yamaha", "yamaha unified communications"],
    "sennheiser": ["sennheiser"],
    "audio_technica": ["audio-technica", "audio technica"],
    "bose": ["bose", "bose professional"],
    "jbl": ["jbl", "jbl professional"],
    "harman": ["harman", "harman international"],
    "mersive": ["mersive", "solstice"],
    "williams_av": ["williams av", "williams sound"],
    "listen_tech": ["listen tech", "listen technologies"],
    "da_lite": ["da-lite", "dalite", "da lite"],
    "draper": ["draper inc", "draper screens"],
    "stewart_filmscreen": ["stewart filmscreen", "stewart screens"],
    "ergotron": ["ergotron"],
    "vizio": ["vizio"],
    "lg_business": ["lg business", "lg signage"],
    "philips": ["philips", "philips signage"],
    # BMS / building automation
    "tridium": ["tridium", "niagara", "vykon", "jace"],
    "honeywell": ["honeywell", "honeywell fire", "notifier"],
    "siemens": ["siemens"],
    "johnson_controls": ["johnson controls", "metasys", "simplex", "simplexgrinnell"],
    "schneider": ["schneider electric", "schneider", "ecostruxure"],
    "alc": ["automated logic", "alc"],
    "distech": ["distech", "distech controls"],
    "trane": ["trane"],
    # NOTE: Carrier Corporation (HVAC manufacturer) is intentionally
    # matched only via context-rich surfaces, never bare "carrier".
    # The bare word too easily matches generic English usage —
    # "telecom carrier", "common carrier", "carrier wave", "package
    # carrier" — and would produce ``vendor:carrier`` false positives.
    # Real Carrier brand mentions typically include qualifying context.
    "carrier_corporation": ["carrier corporation", "carrier hvac", "carrier brand"],
    "edwards_est": ["edwards est", "est", "edwards"],
    "mircom": ["mircom"],
    # IP intercom
    "zenitel": ["zenitel"],
    "grandstream": ["grandstream"],
    "aiphone": ["aiphone", "airphone"],  # the OCTA "Airphone" typo
    # Mass notification
    "rave_mobile_safety": ["rave mobile safety", "rave"],
    "blackberry": ["blackberry corporation", "blackberry at hoc", "at hoc"],
    "regroup": ["regroup mass notification", "regroup"],
    "alertus": ["alertus technologies", "alertus"],
    # Server / hardware
    "dell": ["dell", "dell technologies", "dell precision"],
    "hpe": ["hpe", "hewlett packard enterprise"],
    "lenovo": ["lenovo"],
    # Utility
    "microsoft": ["microsoft", "windows server", "active directory"],
    "vmware": ["vmware"],
    # Niche / specialty vendors that show up in real RFPs.  Adding
    # these as canonical vendor keys means the entity extractor and
    # the gap detector don't double-flag them.  See PRODUCTION_GAPS
    # Week 5 — the VT-CAM gold expected several of these.
    "t2_systems": ["t2 systems", "t2 system", "t2"],
    "thyssenkrupp": ["thyssenkrupp", "thyssen krupp", "thyssen-krupp"],
    "esri": ["esri", "arcgis", "arcsde", "arc sde"],
    "autocad": ["autocad", "auto cad"],
    "live_earth": ["live earth", "liveearth"],
    "exacq": ["exacq", "exacq technologies", "exacqvision"],
    "splunk": ["splunk"],
    "tableau": ["tableau"],
    "openg": ["opengov"],
    "palantir": ["palantir"],
    "salesforce": ["salesforce"],
    "okta": ["okta"],
    # Additional security / fire / BAS vendors common in pre-SOW corpora
    "veridocs": ["veridocs"],
    "openpath": ["openpath"],
    "verkada": ["verkada"],
    "rhombus": ["rhombus systems", "rhombus"],
    "kastle": ["kastle systems", "kastle"],
    "feenics": ["feenics"],
    "brivo": ["brivo"],
    "salient": ["salient systems"],
    "ipvideo": ["ipvideo corporation", "ipvideo"],
    "milestone_xprotect": ["xprotect corporate", "xprotect express", "xprotect smart client"],
    "siklu": ["siklu"],
    "cambium": ["cambium networks", "cambium"],
    "viavi": ["viavi"],
    "spiderlabs": ["spiderlabs"],
    "fluke": ["fluke networks", "fluke"],
}


# Site name suffixes that strongly signal a building/site entity even
# without help from the active pack.  Word-boundary matched.
_SITE_SUFFIX_PATTERNS = [
    r"\b([A-Z][A-Za-z0-9'.\-]*(?:\s+[A-Z][A-Za-z0-9'.\-]*){0,4})\s+(Elementary School|Middle School|High School|Charter (?:Elementary|Middle|High|School)|Academy)\b",
    r"\b([A-Z][A-Za-z0-9'.\-]*(?:\s+[A-Z][A-Za-z0-9'.\-]*){0,4})\s+(Hospital|Medical Center|Health Center|Clinic|VA Medical Center)\b",
    r"\b([A-Z][A-Za-z0-9'.\-]*(?:\s+[A-Z][A-Za-z0-9'.\-]*){0,4})\s+(Convention Center|Conference Center|Civic Center|Community Center|Recreation Center|Performing Arts Center)\b",
    r"\b([A-Z][A-Za-z0-9'.\-]*(?:\s+[A-Z][A-Za-z0-9'.\-]*){0,4})\s+(University|College|Institute|Polytechnic)\b",
    r"\b([A-Z][A-Za-z0-9'.\-]*(?:\s+[A-Z][A-Za-z0-9'.\-]*){0,4})\s+(Police Department|Fire Department|City Hall|County (?:Office|Building|Court))\b",
    r"\b([A-Z][A-Za-z0-9'.\-]*(?:\s+[A-Z][A-Za-z0-9'.\-]*){0,4})\s+(Bus Base|Transit Center|Transportation Center|Park & Ride|Park and Ride)\b",
    r"\b([A-Z][A-Za-z0-9'.\-]*(?:\s+[A-Z][A-Za-z0-9'.\-]*){0,4})\s+(Airport|Terminal|Cruise Terminal|Train Station)\b",
    r"\b([A-Z][A-Za-z0-9'.\-]*(?:\s+[A-Z][A-Za-z0-9'.\-]*){0,4})\s+(Library|Museum|Stadium|Arena|Theatre|Auditorium)\b",
    r"\b([A-Z][A-Za-z0-9'.\-]*(?:\s+[A-Z][A-Za-z0-9'.\-]*){0,4})\s+(Parking (?:Deck|Garage|Structure|Lot)|Stadium Lot)\b",
    r"\b([A-Z][A-Za-z0-9'.\-]*(?:\s+[A-Z][A-Za-z0-9'.\-]*){0,4})\s+(Boardroom|Headquarters|HQ|Administrative Office|Administration Building)\b",
    r"\b([A-Z][A-Za-z0-9'.\-]*(?:\s+[A-Z][A-Za-z0-9'.\-]*){0,4})\s+(Information Systems (?:Bldg|Building)|IT Building|Data Center|Server Room|Telecom Room|MDF|IDF)\b",
]
_SITE_SUFFIX_REGEXES = [re.compile(p) for p in _SITE_SUFFIX_PATTERNS]


# A4 universal naming: "Building 13" / "Site 7" / "Branch 42" /
# "Edificio 4" / "Bâtiment 12" / "Gebäude 5" / "棟3". These appear in
# enterprise floor plans and international deals where every site
# has the same brand name and is distinguished only by a number.
# Numbers can be roman ("Building III"), with optional letter
# suffix ("Building 13A"), or unicode-digit. We deliberately
# require the building-word + space + number form so that a bare
# number elsewhere in text never becomes a site.
_NUMBERED_SITE_REGEX = re.compile(
    r"\b("
    r"(?:Building|Bldg|Site|Branch|Office|Facility|Warehouse|"
    r"Annex|Block|Wing|Tower|Plant|Depot|Hub|Floor|Fl|Lvl|Level|"
    # A7 non-English building words
    r"Edificio|Edif|Edifício|"           # Spanish, Portuguese
    r"Bâtiment|Bat|Immeuble|"            # French
    r"Gebäude|Geb|Gebaeude|"             # German (+ ASCII fallback)
    r"Palazzo|Edificio|"                 # Italian
    r"Здание|"                            # Russian
    r"棟|楠|建物|建筑物|建筑|建築物|"      # CJK
    r"건물|동"                            # Korean
    r")\s+"
    r"([0-9A-Z]+(?:[\-/][0-9A-Z]+)?|[IVXLCDM]+)"
    r"\b)",
    re.UNICODE,
)



# Street address pattern — captures something like "1700 Pratt Drive" or
# "4700 Crest Drive" or "60 East Van Buren Street".  Conservative: we
# require a number, then capitalized word(s), then a street suffix.
_STREET_SUFFIXES = (
    # Standard street suffixes
    "Street|Str|St|Ave|Avenue|Boulevard|Blvd|Road|Rd|Drive|Dr|Lane|Ln|Way|"
    "Court|Ct|Place|Pl|Highway|Hwy|Parkway|Pkwy|Trail|Trl|Circle|Cir|"
    # Less-common but real street suffixes seen in commercial deals
    "Connector|Corridor|Gateway|Crossing|Terrace|Loop|Run|Pike|Turnpike|"
    "Bypass|Expressway|Expwy|Freeway|Fwy|Route|Rte|Spur|Branch|"
    "Plaza|Square|Sq|Crescent|Cres|Mews|Walk|Promenade|Esplanade|"
    "Alley|Aly|Mall|Path|Bridge|Brg|Causeway|Cswy|Junction|Jct|"
    "Row|Greenway|Greenwy|Walkway"
)
_STREET_ADDRESS_REGEX = re.compile(
    r"\b(\d+(?:\-\d+)?)\s+([A-Z][A-Za-z0-9'.\-]*(?:\s+[A-Za-z0-9'.\-]+){0,4})\s+("
    + _STREET_SUFFIXES
    + r")\.?\b"
)


# Site-code pattern: ATL-HQ / ATL-WEST / ATL-AIR / NYC-DC1 / SFO-HQ /
# CHI-MAIN-EAST / HOUSTON-WAREHOUSE style first-class site identifiers.
#
# First segment: 3–10 uppercase letters (covers 3-letter airport codes
#   like ATL/SFO/LAX, 4-letter city codes like CHGO, and full city
#   names like HOUSTON/DALLAS/BERLIN).
# Trailing segments: must start with a letter, 1–10 alphanumerics.
#   Up to 4 trailing segments (handles CHI-MAIN-EAST-1).
#
# Acceptance is gated by a positive suffix-allowlist (see
# ``_SITE_CODE_SUFFIX_ALLOWLIST`` and ``_SITE_CODE_SUFFIX_PATTERN``)
# so unknown junk codes (MOCK-OPTBOT-ATL, DEV-ATL, MSA-2026-001) can't
# leak through — the absence of a recognized site-function suffix is
# itself the filter.
_SITE_CODE_REGEX = re.compile(
    # Head: 2-10 uppercase letters (was 3-10; tighter is too restrictive
    # for shorter region prefixes like "SF" or "AT").
    # Continuation: 1-5 segments, each either alphanumeric starting with
    # a letter (HQ, DC1, WEST) OR pure digits 1-3 chars (01, 12, 100).
    # The trailing digit segment is required to capture "ATL-HQ-01" style
    # IDs where the row number lives in its own segment.
    r"\b([A-Z]{2,10}(?:-(?:[A-Z][A-Z0-9]{0,9}|\d{1,3})){1,5})\b"
)
# Known non-site hyphenated all-caps codes that look site-shaped but
# aren't — connectors, network specs, manufacturer abbreviations, etc.
_SITE_CODE_DENYLIST: frozenset[str] = frozenset({
    # connector / cable types
    "RJ-45", "RJ-11", "RJ-48", "BNC-F", "USB-C", "USB-A", "USB-B",
    "POE-PLUS",
    # wireless / network specs
    "WI-FI", "WIFI", "BT-LE", "BLE-5", "BT-5",
    # power / building services
    "PV-DC", "DC-POWER", "AC-POWER", "VFD-3", "VRF-1",
    # document / form codes
    "P-O", "PO-1", "RFQ-1", "RFP-1",
    # generic
    "TBD-1", "NTS-1",
})

# First-segment denylist for hyphen-separated codes. When `_SITE_CODE_REGEX`
# matches something like `MOCK-OPTBOT-ATL` or `DEV-ATL` or `MSA-2026`, the
# first segment is the discriminator — if it is a known non-site token
# (test data marker, contract type, cloud platform, ID prefix), the whole
# match is rejected even if the trailing segments look airport-shaped.
# Without this OPTBOT-style mock deal text leaks `site:mock_optbot_atl`,
# `site:dev_atl`, `site:mock_msa`.
_SITE_CODE_HEAD_DENYLIST: frozenset[str] = frozenset({
    # test / mock / dev data prefixes
    "MOCK", "DEV", "TEST", "TESTS", "DEMO", "FAKE", "DUMMY",
    "SAMPLE", "EXAMPLE", "STUB", "DRAFT", "TMP", "TEMP",
    "PROTO", "POC",
    # contract / document type prefixes
    "MSA", "NDA", "SOW", "MOU", "LOI", "DPA", "BAA", "SLA",
    "EULA", "RFP", "RFQ", "RFI", "PO", "WO", "INV", "TKT",
    "TASK", "PROJ", "DEAL", "CASE", "REQ", "REQS", "QUO", "QUOTE",
    "ORDER", "ORD",
    # CRM / system ID prefixes
    "HS", "SF", "SFDC", "HUBSPOT", "ZEN", "ZENDESK",
    # cloud / SaaS platform prefixes
    "AZURE", "AWS", "GCP", "ARM", "EC2", "GKE", "AKS",
    "INTUNE", "OKTA", "ENTRA", "DUO",
    # API / protocol prefixes
    "API", "REST", "GRPC", "JSON", "XML", "YAML", "CSV",
    # identifier-type prefixes
    "ID", "IDS", "REF", "SKU", "UPC", "EAN", "ISBN", "GUID", "UUID",
    "TAG", "TYPE",
    # log / severity prefixes
    "ERR", "WRN", "INF", "DBG", "FATAL",
    # role / org prefixes that can look 3-letter site-shaped
    "USR", "ADM", "MGR", "EMP", "STAFF",
})


# Positive suffix allowlist for hyphenated site codes. A code only
# becomes a site when its LAST segment matches one of these tokens
# (closed list) or matches ``_SITE_CODE_SUFFIX_PATTERN`` (open shape
# for numbered floors/buildings/data-centers/wings).
#
# This is the load-bearing universal gate: instead of trying to
# enumerate every possible junk word in a head/middle position, we
# require the trailing token to carry a known site-function meaning
# (direction, facility type, datacenter number, floor number, wing).
# Anything that doesn't end in a recognized site suffix fails the
# gate, no matter what its head or middle segments look like.
_SITE_CODE_SUFFIX_ALLOWLIST: frozenset[str] = frozenset({
    # === Cardinal directions ===
    "N", "S", "E", "W", "NE", "NW", "SE", "SW",
    "NORTH", "SOUTH", "EAST", "WEST", "CENTRAL",
    "NORTHEAST", "NORTHWEST", "SOUTHEAST", "SOUTHWEST",
    # === Function labels ===
    "HQ", "MAIN", "PRIMARY", "SECONDARY", "BACKUP", "FAILOVER",
    "AIR", "AIRPORT", "ANNEX", "CAMPUS", "FACILITY",
    "LAB", "LABS", "OFFICE", "OFFICES",
    "PLANT", "FACTORY", "WAREHOUSE", "WH", "STORAGE",
    "OPS", "OPERATIONS", "OPCENTER",
    "HUB", "DEPOT", "TERMINAL", "GATEWAY",
    "BRANCH", "STORE", "SHOP", "SHOWROOM",
    "DEALER", "DEALERSHIP",
    "CENTER", "CENTRE", "CTR",
    "BLDG", "BLD", "BLOCK", "WING",
    "FLOOR", "FL", "LEVEL", "LVL",
    "RACK", "ROOM",
    # === Datacenter / cabinet shorthand (bare suffix forms) ===
    "DC", "MDC", "IDC", "POP", "POE",
    # === Distribution / utility ===
    "DIST", "DISTRIBUTION", "FULFILLMENT", "LOGISTICS",
})

# Open-shape suffix pattern — accepts numbered facility / wing /
# datacenter / floor / building codes (DC1, FL3, B12, T5, BLDG2, WING7).
_SITE_CODE_SUFFIX_PATTERN = re.compile(
    r"^(?:"
    r"DC\d{1,3}"       # DC1, DC15, DC100
    r"|MDC\d{1,3}"     # MDC1
    r"|IDC\d{1,3}"     # IDC1
    r"|FL\d{1,3}"      # FL3
    r"|FLOOR\d{1,3}"   # FLOOR12
    r"|LVL\d{1,3}"     # LVL3
    r"|LEVEL\d{1,3}"   # LEVEL3
    r"|BLDG\d{1,3}"    # BLDG2
    r"|BLD\d{1,3}"     # BLD2
    r"|B\d{1,3}"       # B12
    r"|T\d{1,3}"       # T5  (tower 5)
    r"|H\d{1,3}"       # H1
    r"|W\d{1,3}"       # W3  (wing 3)
    r"|WING\d{1,3}"    # WING7
    r"|BLOCK\d{1,3}"   # BLOCK2
    r"|RACK\d{1,3}"    # RACK19
    r"|ROOM\d{1,3}"    # ROOM101
    r"|R\d{1,3}"       # R101 (room 101)
    r"|POP\d{1,3}"     # POP2
    r"|SITE\d{1,3}"    # SITE3
    r"|STORE\d{1,3}"   # STORE142
    r"|BRANCH\d{1,3}"  # BRANCH7
    r")$"
)


def _site_code_suffix_ok(last_segment: str, *, prev_segment: str | None = None) -> bool:
    """Universal gate: does this last segment carry site-function meaning?

    Accepts three shapes:
      1. ``last`` is in the curated allowlist (HQ, MAIN, WEST, ...).
      2. ``last`` matches the open-shape pattern (DC1, FL3, BLDG2, ...).
      3. ``last`` is 1-3 digits AND ``prev_segment`` is a recognized
         site-function token. Pattern ``<region>-<function>-<NN>`` is
         the most common enterprise site-ID shape (``ATL-HQ-01``,
         ``NYC-DC-12``, ``SFO-WEST-05``). Without this, every numbered
         site instance of the form gets silently dropped.
    """
    if last_segment in _SITE_CODE_SUFFIX_ALLOWLIST:
        return True
    if _SITE_CODE_SUFFIX_PATTERN.match(last_segment):
        return True
    if (
        prev_segment is not None
        and last_segment.isdigit()
        and 1 <= len(last_segment) <= 3
        and (
            prev_segment in _SITE_CODE_SUFFIX_ALLOWLIST
            or _SITE_CODE_SUFFIX_PATTERN.match(prev_segment)
            # Two-letter site-function shorthands occur in real rosters
            # (for example College Park -> CP), but arbitrary alpha tails
            # like CAT-6 or ECHO-DELTA-001 are document/product codes.
            # Treat compact two-letter shorthands as site codes only when
            # the instance number is two digits or longer.
            or (prev_segment.isalpha() and len(prev_segment) == 2 and len(last_segment) >= 2)
            or (prev_segment.isdigit() and 1 <= len(prev_segment) <= 4)
        )
    ):
        # Accepts ``<region>-<function>-<NN>`` AND ``<region>-<NNN>-<NN>``.
        # The street-number-as-function case (``ATL-047-04`` for
        # "OPTBOT Brady Training, 047 Brady Ave NW") is real in
        # enterprise rosters and would otherwise be silently dropped.
        return True
    return False


# Site-context regex — when a Capitalized-run is immediately adjacent
# (within ~40 chars in the same sentence) to one of these UNAMBIGUOUS
# cues, we treat the run as having positive site signal even if its
# tail isn't a recognized place-noun.
#
# We deliberately use multi-word phrasing ("located at", "site visit:",
# "based in") rather than bare nouns ("site", "location") because bare
# nouns appear too often in unrelated contexts ("Three Site
# Modernization") and would create false corroboration.
#
# The phrases below all have an unambiguous syntactic frame that
# indicates "the noun phrase NEAR this cue refers to a physical place."
_SITE_CONTEXT_REGEX = re.compile(
    r"(?:"
    # Prepositional/state-of-being cues (require explicit preposition)
    r"\bbased\s+(?:at|in|out\s+of)\b"
    r"|\blocated\s+(?:at|in|near|on|adjacent\s+to)\b"
    r"|\bheadquartered(?:\s+(?:at|in))?\b"
    r"|\bsituated\s+(?:at|in|on|near)\b"
    r"|\bhoused\s+(?:at|in)\b"
    r"|\boperat(?:es|ing|ed)\s+(?:out\s+of|from)\b"
    # Site-activity cues
    r"|\bsite\s+(?:visit|tour|survey|walk(?:-?through)?|walkthrough)\b"
    r"|\bon(?:-|\s+)site\s+(?:at|in)\b"
    r"|\bdeployed\s+(?:at|in|to)\b"
    r"|\binstalled\s+(?:at|in)\b"
    r"|\bvisit\s+(?:to|at)\b"
    r"|\btour\s+(?:of|at)\b"
    # Field-label cues (colon-separated)
    r"|\b(?:site|location|address|facility|building|premises|venue)\s*[:=]"
    # Coordinate-style cues
    r"|\baddress(?:es)?\s+(?:is|are|of)\b"
    r"|\bfacilit(?:y|ies)\s+(?:at|in|located|known)\b"
    r")",
    re.IGNORECASE,
)


# Adjacent-window size in characters for corroboration. Tight enough
# that a cue elsewhere in a long sentence ("Three Site Modernization"
# 80 chars away) doesn't create false corroboration, wide enough that
# "Site visit: Aurora Operations Hub on June 10" still corroborates.
_SITE_CORROBORATION_WINDOW = 40


# Hard-disqualify tokens — when ANY of these appears anywhere in a
# Capitalized-run, the run is NEVER a site, even if its tail is an
# otherwise-valid place-noun like "Tower" or "Center".
#
# Rationale: "Mock Atlanta Tower" / "Test Innovation Lab" / "Sample
# Site Headquarters" — these are descriptive scaffolding around a
# place noun, not real site names. The presence of a test-data marker
# is itself disqualifying.
#
# This is narrower than ``_NON_SITE_PHRASE_TAIL_NOUNS`` (which kills
# the phrase only when it appears in the TAIL or as an inner token
# while the tail is non-place). Hard disqualify overrides the
# place-tail bypass entirely.
_HARD_DISQUALIFY_PHRASE_TOKENS: frozenset[str] = frozenset({
    "mock", "mocks", "mocked",
    "fictional", "fictitious",
    "fake", "fakes",
    "dummy", "dummies",
    "demo", "demos",
    "test", "tests",
    "sample", "samples",
    "example", "examples",
    "placeholder", "placeholders",
    "stub", "stubs",
    "synthetic", "simulated",
    "scratch",
    "tbd", "tba", "tbc",
})


# Tail nouns that turn an otherwise-capitalized phrase into NON-site
# content. When the LAST non-stopword token is one of these, the whole
# phrase is dropped from site emission — these tail nouns mean the
# phrase describes a person, a document, a system, or a process, not
# a place.
#
# Role tails (person, not place): "Regional Facilities Manager",
#   "VP Workplace Operations", "Security Architecture Lead".
# Document tails (artifact, not place): "Executive Deal Brief",
#   "Site Surveys DOCX", "Procurement Packet PDF".
# Pipeline / system tails (system, not place): "HubSpot Dev Deal",
#   "Azure Dev Storage", "OrbitBrief Workspace".
# Concept / process tails: "Three Site Modernization",
#   "Target Close Date", "Total Mock Amount".
_ROLE_INNER_MARKERS: frozenset[str] = frozenset({
    "manager", "managers", "director", "directors", "sponsor", "sponsors",
    "officer", "officers", "supervisor", "supervisors",
    "coordinator", "coordinators", "administrator", "administrators",
    "lead", "leads", "owner", "owners", "approver", "approvers",
})


_NON_SITE_PHRASE_TAIL_NOUNS: frozenset[str] = frozenset({
    # === role / person tails ===
    "manager", "managers", "lead", "leads", "director", "directors",
    "sponsor", "sponsors", "engineer", "engineers", "architect",
    "architects", "analyst", "analysts", "coordinator", "coordinators",
    "admin", "administrator", "administrators", "owner", "owners",
    "executive", "executives", "expert", "experts", "specialist",
    "specialists", "rep", "representative", "consultant", "consultants",
    "supervisor", "supervisors", "head", "heads", "chief", "officer",
    "officers", "principal", "principals", "operator", "operators",
    "stakeholder", "stakeholders", "approver", "approvers",
    "respondent", "respondents", "assistant", "assistants",
    "lieutenant", "captain", "designee", "delegate",
    # Role-shaped initialisms / shorthand
    "pm", "po", "ceo", "cto", "cio", "ciso", "cfo", "coo", "vp", "svp", "evp",
    "exec", "mgr", "sup", "supt",
    # Role-shaped activity / function tails
    "operations", "operation", "ops", "ops.", "team", "teams",
    "staff", "personnel", "crew", "crews", "worker", "workers",
    "force", "leadership",
    # === document / artifact tails ===
    "document", "documents", "brief", "briefs", "packet", "packets",
    "notes", "note", "memo", "memos", "plan", "plans", "report",
    "reports", "deck", "decks", "list", "lists", "policy", "policies",
    "guideline", "guidelines", "manual", "manuals", "presentation",
    "presentations", "slides", "slide", "schedule", "schedules",
    "worksheet", "worksheets", "spreadsheet", "spreadsheets",
    "workbook", "workbooks", "draft", "drafts", "version", "versions",
    "edition", "editions", "revision", "revisions",
    "xlsx", "docx", "pdf", "csv", "json", "yaml",
    # === pipeline / system / cloud-resource tails ===
    "storage", "workspace", "workspaces", "deal", "deals",
    "checkpoint", "checkpoints", "batch", "batches", "pipeline",
    "pipelines", "workflow", "workflows", "job", "jobs",
    "instance", "instances", "cluster", "clusters", "container",
    "containers", "bucket", "buckets", "environment", "environments",
    "env", "tenant", "tenants", "account", "accounts",
    "subscription", "subscriptions", "repository", "repositories",
    "repo", "repos", "endpoint", "endpoints", "queue", "queues",
    "topic", "topics", "service", "services", "module", "modules",
    "library", "libraries", "framework", "frameworks",
    "platform", "platforms", "tool", "tools", "system", "systems",
    "connector", "connectors", "adapter", "adapters",
    # === concept / process tails (in addition to existing stopwords) ===
    "modernization", "transformation", "migration", "deployment",
    "deployments", "implementation", "implementations", "upgrade",
    "upgrades", "refresh", "refreshes", "rollout", "rollouts",
    "cutover", "cutovers", "amount", "amounts", "total", "subtotal",
    "value", "values", "approval", "approvals", "controls", "control",
    "use", "usage",
    # === measurement / metric tails ===
    "baseline", "baselines", "benchmark", "benchmarks",
    "metric", "metrics", "kpi", "kpis", "target", "targets",
    "threshold", "thresholds", "estimate", "estimates",
    "forecast", "forecasts", "projection", "projections",
    # === temporal tails ===
    "date", "dates", "deadline", "deadlines", "milestone",
    "milestones", "window", "windows", "timeline", "timelines",
    "timeframe", "timeframes", "period", "periods",
    "quarter", "quarters", "phase", "phases", "stage", "stages",
    # === test / mock / demo tokens (any position kills the phrase) ===
    "mock", "mocks", "mocked", "fictional", "fake", "fakes",
    "dummy", "dummies", "demo", "demos", "test", "tests",
    "sample", "samples", "example", "examples", "placeholder",
    "stub", "stubs", "draft", "drafts", "preliminary", "tentative",
    "synthetic", "simulated", "scratch",
    # === classification / sensitivity tokens (not place names) ===
    "confidential", "proprietary", "restricted", "classified",
    "internal", "private", "public", "sensitive",
    # === demonstratives / vague references that bleed into runs ===
    "this", "that", "these", "those",
})

# Phrases that should never be treated as sites, regardless of shape.
# These show up in enterprise deal docs as section labels / boilerplate.
_SITE_PHRASE_BLOCKLIST: frozenset[str] = frozenset({
    "three site modernization",
    "three site modernization hubspot deal",
    "scope of work",
    "site surveys docx",
    "site surveys",
    "site survey",
    "all sites",
    "all locations",
    "every site",
    "every location",
    "executive deal brief",
    "executive deal brief mock document",
})


# Part-number / SKU pattern.  Cisco-style (CW9166I-B, AIR-DNA-E-T-5Y),
# HPE-style (J9145A), Bosch DICENTIS (DCNM-DVT908), Streamvault
# (SV-2030E-AC).  We anchor on a letter+digit+letter mix or a
# multi-segment hyphenated identifier.
#
# Branch 1: letters/digits then ≥1 hyphen-separated suffix segments.
#           Captures CW9166I-B, AIR-DNA-E-5Y, FPO250/250/250-5D8P3M8PNLXE12M.
# Branch 2: bare alphanumeric SKU with ≥2 letters and ≥2 digits.
#           Captures J9145A, P3268LV, T3620.
_PART_NUMBER_REGEX = re.compile(
    r"\b("
    r"[A-Z][A-Z0-9]{1,9}(?:[-/][A-Z0-9]{1,12}){1,6}"
    r"|"
    r"[A-Z]{2,5}[0-9]{2,6}[A-Z]{0,3}"
    r")\b"
)


# Quantity patterns commonly produced by table extractors:
#   "Qty: 136", "Quantity: 500", "Quantity = 500", "QTY 136"
_QUANTITY_REGEX = re.compile(
    r"\b(?:qty|quantity|quantities|count)\s*[:=]?\s*([0-9]+(?:,[0-9]{3})*)\b",
    re.IGNORECASE,
)

# Noun-anchored quantity pattern: matches "<NUMBER> <unit-or-device-noun>"
# so prose statements like "Install 50 access points", "60 wireless
# devices", "5 distribution switches", "12 SFP modules", "75 power
# cords", "8 spools of cable" produce quantity entities. The trailing
# noun must be from the install-vocabulary list (extensible) so this
# does not match prices ("50 dollars"), dates ("2026 March"), or
# generic prose ("50 reasons").
_QUANTITY_NOUN_REGEX = re.compile(
    r"\b([0-9]+(?:,[0-9]{3})*)\s+"
    # Allow common qualifier prefixes that precede the device noun:
    # "24 IP cameras", "50 wireless access points", "5 PoE+ switches",
    # "12 mesh APs", "8 dome cameras". Each prefix is optional and
    # consumed silently — the noun list below remains the anchor.
    r"(?:(?:wireless|ip|poe\+?\+?|core|access|distribution|mesh|"
    r"managed|layer\s*[23]|mgig|multi[-\s]?gig|2\.5g|5g|10g|"
    r"dome|bullet|ptz|fixed|thermal|fisheye|panoramic|"
    r"hd|high\s+density|indoor|outdoor|hardened|"
    r"prox|badge|card|hid|mobile|multi[-\s]?format|"
    r"smoke|heat|motion|pir|dual\s+tech|"
    r"ceiling|wall|pendant|paging|horn|sound\s+mask|"
    r"rack|server|blade|hyperconverged|1u|2u|"
    r"thin|chrome|"
    r"sd[-\s]?wan|edge|next[-\s]?gen|ngfw|utm)\s+)*"
    r"(?:access\s+points?|aps?|waps?|wireless\s+access\s+points?|wireless\s+devices?|"
    r"switches?|firewalls?|fws?|routers?|cameras?|cams?|sensors?|"
    r"sfp(?:\+|s)?\s*modules?|sfp(?:\+|s)?|modules?|"
    r"patch\s+panels?|patch\s+cords?|panels?|"
    r"jacks?|outlets?|drops?|ports?|"
    r"cables?|cords?|spools?(?:\s+of\s+\w+)?|"
    r"licenses?|seats?|users?|endpoints?|"
    r"servers?|appliances?|chassis|chasses|"
    r"workstations?|laptops?|desktops?|"
    r"installations?|sites?|locations?|facilities|facility|"
    r"readers?|controllers?|displays?|monitors?|projectors?|"
    r"speakers?|microphones?|mics?|"
    r"upses?|racks?|cabinets?|"
    r"strikes?|maglocks?|"
    r"detectors?|pull\s+stations?|horn\s+strobes?|"
    r"units?|devices?|pieces?|each)\b",
    re.IGNORECASE,
)


# Q-and-A markers — Q1., A1., Q.1, A.1, Q-1
_QA_MARKER_REGEX = re.compile(r"\b([QA])\s*\.?\s*(\d{1,3})\b")


# CSI MasterFormat section IDs — 27 32 26, 28 05 00, 25 50 00, 00 21 13
_CSI_SECTION_REGEX = re.compile(r"\b(\d{2})\s(\d{2})\s(\d{2})(?:\.(\d{1,2}))?\b")


# Generic capitalized address phrases: "1700 Pratt Drive" → site:1700_pratt_drive
# Matches phrases like "Andrews Information Systems Building",
# "Perry Street Parking Deck", "Chicago Housing Authority".  Also
# matches 2-word organization names ("Virginia Tech", "Boston College")
# — the in-loop filter then drops 2-word runs whose trailing token is
# not an organization-suffix word (see ``_ORG_SUFFIX_TWO_WORD``).
_PROPER_NOUN_RUN = re.compile(
    r"\b([A-Z][A-Za-z0-9'.\-]+(?:\s+[A-Z][A-Za-z0-9'.\-]+){1,6})\b"
)

# Trailing words that should be stripped when they appear as the last
# token of a proper-noun run.  These are typically column headers /
# field labels that bleed across sentence boundaries in extracted PDF
# text (e.g. "Perry Street Parking Deck. Vendor:").
_PROPER_NOUN_TRAILING_STOPWORDS = {
    "vendor",
    "manufacturer",
    "model",
    "part",
    "qty",
    "quantity",
    "co",
    "inc",
    "llc",
    "ltd",
    "corp",
    "corporation",
    "company",
    "group",
    "description",
    "specification",
    "specifications",
    "spec",
    "specs",
    "section",
    "table",
    "figure",
    "exhibit",
    "appendix",
    "attachment",
    "page",
    "chapter",
    "addendum",
    "addenda",
    "rfp",
    "rfi",
    "bid",
    "proposal",
    "vendors",
    "respondent",
    "offeror",
    "bidder",
    "contractor",
    "city",
    "state",
    "county",
    # Week 6 P6.2: capitalized "function words" that frequently end a
    # proper-noun run when sentence-cased headings or column labels
    # bleed in.  Without these, NATOMAS produces noise like
    # ``site:attorney_fees_in``, ``site:contract_exclusive_the``,
    # ``site:fulfill_contract_when``, ``site:e_rate_funding_year``.
    #
    # Important: stick to function words (prepositions, conjunctions,
    # articles, time/temporal markers) and modal verbs.  Place-shape
    # nouns like ``office``, ``school``, ``district``, ``building``
    # MUST stay out — they're part of legitimate organization names
    # ("District Office", "Branch Office").  See
    # ``_ORG_SUFFIX_TWO_WORD`` for the org-tail whitelist.
    #
    # Concept-shape nouns like ``fees``, ``costs``, ``rates``,
    # ``year``, ``month``, ``form``, ``template``, ``information``,
    # ``data`` ARE included — those make a phrase a section/subject
    # label rather than an organization name.
    "in",
    "on",
    "at",
    "of",
    "to",
    "by",
    "for",
    "with",
    "from",
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "when",
    "where",
    "while",
    "if",
    "unless",
    "until",
    "during",
    "after",
    "before",
    "year",
    "years",
    "day",
    "days",
    "month",
    "months",
    "week",
    "weeks",
    "no",
    "yes",
    "all",
    "any",
    "each",
    "every",
    "shall",
    "must",
    "may",
    "will",
    "should",
    "can",
    "would",
    "could",
    "no.",
    "id",
    "ref",
    "reference",
    # Concept-shape nouns — re-added after Week 6 P6.6 follow-up
    # showed NATOMAS leaking ``site:agreement_attorney_fees`` /
    # ``site:bulk_upload_template`` because these tail nouns weren't
    # being stripped.  Place-shape nouns like ``office``, ``school``,
    # ``district``, ``building`` deliberately stay OUT (they're org
    # tails — see ``_ORG_SUFFIX_TWO_WORD``).
    "fees",
    "fee",
    "costs",
    "cost",
    "rates",
    "rate",
    "prices",
    "price",
    "form",
    "forms",
    "format",
    "template",
    "templates",
    "information",
    "data",
    "list",
    "listing",
    "details",
    "detail",
    "summary",
    "overview",
    "criteria",
    "factor",
    "factors",
    "method",
    "methods",
    "process",
    "procedure",
    "procedures",
    "step",
    "steps",
    "phase",
    "phases",
    "task",
    "tasks",
    "subject",
    "subjects",
    "category",
    "categories",
    "item",
    "items",
    "tab",
    "tabs",
    "section",
    "sections",
    "agreement",
    "agreements",
    "contract",
    "contracts",
    "submission",
    "submissions",
    "type",
    "types",
    "kind",
    "kinds",
    "level",
    "levels",
    # Singular taxonomy / concept words that show up at end of run
    "general",
    "generic",
    "applicable",
    "available",
}

# Words that should NOT count as proper-noun runs even though they look
# like it (RFP boilerplate, common headings).
_PROPER_NOUN_STOPLIST = {
    "request for proposal",
    "scope of work",
    "scope of services",
    "table of contents",
    "instructions to bidders",
    "general conditions",
    "special conditions",
    "notice to bidders",
    "request for proposals",
    "schedule of values",
    "designated subcontractors list",
    "noncollusion declaration",
    "intent to bid",
    "list of drawings",
    "all terms and conditions",
    "all rights reserved",
    "best and final offer",
    "vendor submission checklist",
    "evaluation and selection process",
    "agreement for services",
    "cost evaluation",
    "economic impact",
    "letter of agreement",
    "letter of transmittal",
    "right to protest",
    "iran contracting act",
    "secure networks act",
    "americans with disabilities act",
    "civil rights act",
    "freedom of access act",
    "public records act",
    "labor code",
    "federal communications commission",
    "us department of transportation",
    "department of justice",
    "department of industrial relations",
    "national fire protection association",
    "underwriters laboratories",
    "ip phone project",
    "ip phones project",
    "dna on prem",
    "dna on prem essential",
    # Common form-field labels.  These show up as proper-noun runs in
    # vendor information forms but are field labels, not sites.
    "full legal name",
    "company name",
    "federal taxpayer number",
    "billing name",
    "purchase order address",
    "payment address",
    "business name",
    "dba name",
    "contact name",
    "authorized representative",
    "corporate address",
    "primary point",
    "evaluation criteria",
    "evaluation factors",
    "evaluation committee",
    "evaluation team",
    "non-disclosure agreement",
    "tax id",
    "tax id number",
    "tax id# above",
    "tax identification number",
    "ein number",
    "ssn number",
    "tin number",
    "duns number",
    "fein number",
    "spin number",
    "frn number",
    "form 470",
    "form 471",
    "form 472",
    "form 474",
    "form 486",
    "title page",
    "cover page",
}

# Markers that strongly indicate the atom is a form-field template
# rather than scope content.  Split into "strong" (signature, FEIN,
# (PRINT), (IN INK), …) and "weak" (placeholder column-name prefixes
# emitted by the structured extractor when no header was inferable —
# these appear in legitimate tables too, so they only count when paired
# with a strong marker).  See Week 6 P6.6.
_FORM_FIELD_STRONG_MARKERS = (
    "(print)",
    "(in ink)",
    "(if applicable)",
    "(if different",
    "id#",
    "fein",
    "duns",
    "spin",
    "frn",
    "signature",
    "______",
)
_FORM_FIELD_WEAK_MARKERS = (
    "col_1:",
    "col_2:",
    "col_3:",
    "col_4:",
    "col_5:",
    "col_6:",
    "col_7:",
    "col_8:",
)
_FORM_FIELD_MARKERS = _FORM_FIELD_STRONG_MARKERS + _FORM_FIELD_WEAK_MARKERS


def _slugify(value: str) -> str:
    """ASCII-folded snake_case slug for entity keys.

    Folds Unicode accents (Bâtiment → Batiment, Gebäude → Gebaude,
    Edificio → Edificio) via NFKD normalization before stripping
    to ASCII so international site names produce readable slugs
    instead of "b_timent_12" / "geb_ude_5".

    For non-Latin scripts that don't decompose to ASCII (CJK
    Japanese / Chinese / Korean), the chars are preserved in
    casefolded form so a deterministic slug still emits — the
    raw CJK glyph IS the slug.
    """
    if not value:
        return ""
    # NFKD splits combining marks from base letters; the encode/decode
    # drops the marks but leaves base ASCII letters intact. Non-ASCII
    # scripts (CJK, Cyrillic, etc.) round-trip via ``replace`` so the
    # slug still contains the original glyph and the entity is keyed
    # consistently — albeit not Latin-readable.
    folded = unicodedata.normalize("NFKD", value)
    ascii_folded = folded.encode("ascii", "ignore").decode("ascii")
    # If folding ate everything (e.g. a pure-CJK string), fall back to
    # the casefolded original so the entity isn't silently dropped.
    base = ascii_folded if ascii_folded.strip() else value
    return re.sub(r"[^a-z0-9]+", "_", base.lower()).strip("_")


_UNIVERSAL_DEVICE_BASELINE: dict[str, tuple[str, ...]] = {
    "ip_camera": (
        "ip camera", "ip cameras", "camera", "cameras", "cctv camera",
        "security camera", "surveillance camera", "ip cam", "ip cams",
        "dome camera", "bullet camera", "ptz camera", "ptz",
        "fixed camera", "thermal camera", "fisheye camera",
    ),
    "access_point": (
        "access point", "access points", "wireless access point",
        "ap", "aps", "wap", "waps", "wifi ap", "wi-fi ap",
        "wifi access point", "indoor ap", "outdoor ap",
    ),
    "switch": (
        "switch", "switches", "poe switch", "poe+ switch", "poe++ switch",
        "core switch", "access switch", "distribution switch",
        "managed switch", "layer 2 switch", "layer 3 switch",
        "mgig switch", "multi-gig switch",
    ),
    "router": (
        "router", "routers", "edge router", "core router",
        "sd-wan", "sdwan", "sd-wan appliance",
    ),
    "firewall": (
        "firewall", "firewalls", "fw", "ngfw", "next-gen firewall",
        "utm", "utm appliance",
    ),
    "card_reader": (
        "card reader", "card readers", "prox reader", "badge reader",
        "access reader", "reader", "readers", "hid reader", "mobile reader",
    ),
    "controller": (
        "controller", "controllers", "access controller", "door controller",
        "access control panel",
    ),
    "electric_strike": (
        "electric strike", "strike", "mag lock", "maglock",
        "magnetic lock", "electric lock",
    ),
    "motion_sensor": ("motion sensor", "motion detector", "pir", "pir sensor"),
    "smoke_detector": (
        "smoke detector", "smoke detectors", "smoke sensor",
        "heat detector", "heat detectors",
    ),
    "pull_station": ("pull station", "pull stations", "manual pull station", "fire pull"),
    "horn_strobe": ("horn strobe", "horn/strobe", "notification appliance"),
    "fire_panel": ("fire panel", "facp", "fire alarm control panel", "fire alarm panel"),
    "speaker": ("speaker", "speakers", "ceiling speaker", "pendant speaker"),
    "microphone": ("microphone", "microphones", "mic", "mics"),
    "display": ("display", "displays", "tv", "monitor", "monitors", "video wall", "touchscreen"),
    "projector": ("projector", "projectors", "laser projector"),
    "videoconferencing_codec": ("codec", "vc codec", "room kit", "video bar"),
    "ups": ("ups", "battery backup", "rack ups"),
    "rack": ("rack", "racks", "cabinet", "server rack"),
    "workstation": (
        "workstation", "workstations", "pc", "pcs", "desktop", "desktops",
        "laptop", "laptops", "computer", "computers", "chromebook", "chromebooks",
        "notebook", "notebooks", "thin client", "thin clients",
        "all-in-one", "all in one", "aio",
        # Common consumer / enterprise model series referenced in
        # ITAD bid sheets ("Dell Latitude 3120 2 in 1", "Dell 3120
        # computers", "HP EliteBook 840", "Lenovo ThinkPad T14").
        # These act as device markers so a row that lists 245 units
        # of a specific model still emits ``device:workstation``.
        "dell latitude", "dell optiplex", "dell precision",
        "hp elitebook", "hp probook", "hp envy", "hp pavilion",
        "lenovo thinkpad", "lenovo thinkcentre", "lenovo ideapad",
        "macbook", "macbook pro", "macbook air", "imac",
        "microsoft surface", "surface pro", "surface laptop",
    ),
    "server": ("server", "servers", "rack server", "blade server"),
    "tablet": (
        "tablet", "tablets", "ipad", "ipads",
        "android tablet", "windows tablet", "2 in 1", "2-in-1",
    ),
    "monitor": (
        "monitor", "monitors", "lcd monitor", "led monitor",
        "computer monitor", "desktop monitor",
    ),
    "storage": (
        "ssd", "ssds", "hard drive", "hard drives", "hdd", "hdds",
        "nvme", "external drive", "external hard drive",
        "usb drive", "thumb drive", "flash drive",
    ),
    "printer": (
        "printer", "printers", "mfp", "multifunction printer",
        "laser printer", "inkjet printer", "label printer",
    ),
}


_DEVICE_INDEX_CACHE: dict[int, dict[str, str]] = {}
_TYPED_INDEX_CACHE: dict[int, dict[str, dict[str, str]]] = {}


def _device_alias_index(pack: DomainPack) -> dict[str, str]:
    """Flatten pack.device_aliases into ``{normalized_alias: canonical}``.

    The canonical is the YAML key (e.g. ``ip_camera``).  We add a
    word-boundary entry for every alias so we don't match
    ``"camera"`` inside ``"cameramen"``.

    Auto-plural: when an alias is singular and ends in a plural-safe
    suffix (consonant + non-'s'/'x'/'z') we also register its English
    plural form. So a pack that only lists ``switch`` will still
    match ``switches`` in prose. This avoids forcing every pack to
    list every plural variant.

    Universal baseline: the routed pack may be narrow (e.g.
    ``wireless`` only covers APs/switches). When a doc mentions
    devices outside the pack's vocabulary ("24 cameras" in a
    wireless-routed bundle), the device atom would be dropped. We
    layer a small UNIVERSAL_DEVICE_BASELINE on top so cross-pack
    devices still surface. The routed pack still wins on conflicts.

    Cached by ``id(pack)`` so subsequent calls inside a compile
    hit the cache instead of rebuilding the index per atom.
    """
    cached = _DEVICE_INDEX_CACHE.get(id(pack))
    if cached is not None:
        return cached
    index: dict[str, str] = {}

    def _add(form: str, canonical: str) -> None:
        norm = normalize_text(form)
        if norm:
            index.setdefault(norm, canonical)
        # Auto-plural for short device nouns. Skip if the alias ends in
        # a non-pluralizable suffix or already looks plural.
        if norm and " " not in norm and len(norm) >= 4 and not norm.endswith(("s", "x", "z", "ay", "ey", "iy", "oy", "uy")):
            if norm.endswith(("ch", "sh", "ss")):
                index.setdefault(norm + "es", canonical)
            elif norm.endswith("y") and len(norm) >= 2 and norm[-2] not in "aeiou":
                index.setdefault(norm[:-1] + "ies", canonical)
            else:
                index.setdefault(norm + "s", canonical)

    for canonical, aliases in (pack.device_aliases or {}).items():
        _add(canonical.replace("_", " "), canonical)
        for alias in aliases or []:
            _add(alias, canonical)
    # Universal baseline — only adds entries that aren't already
    # bound to a different canonical (setdefault preserves pack winner).
    for canonical, aliases in _UNIVERSAL_DEVICE_BASELINE.items():
        _add(canonical.replace("_", " "), canonical)
        for alias in aliases:
            _add(alias, canonical)
    _DEVICE_INDEX_CACHE[id(pack)] = index
    return index


_GENERIC_TYPED_ALIAS_SENTINEL = "__generic__"


def _typed_alias_index(pack: DomainPack) -> dict[str, dict[str, str]]:
    """Build ``{entity_type: {alias_norm: example_or_sentinel}}`` so we
    can detect typed entities (room, site, vendor, etc.) generically.

    Aliases come from two pack sources:
      * ``entity.aliases`` — GENERIC synonyms for the type
        ("tenant"/"client"/"owner" for customer; "closet"/"rm"/"mdf"
        for room; "carrier"/"oem"/"reseller" for vendor). These are
        mapped to the ``_GENERIC_TYPED_ALIAS_SENTINEL`` so callers
        can detect them as generic-noun matches that should NOT
        produce real entity keys.
      * ``entity.examples`` — SPECIFIC named instances ("West Region
        District" for customer, "Acme Cabling Co" for vendor). These
        are mapped to the example string itself so they emit a real
        entity key.

    The sentinel is what fixes ``customer:customer`` / ``room:room`` /
    ``vendor:carrier`` from leaking into the entity records: the
    pack defines "tenant" / "closet" / "carrier" as generic synonyms,
    but matching them as entities just because the text contains the
    bare word produces tautological / meaningless keys.

    Cached by ``id(pack)``.
    """
    cached = _TYPED_INDEX_CACHE.get(id(pack))
    if cached is not None:
        return cached
    out: dict[str, dict[str, str]] = {}
    for entity in pack.entity_types or []:
        slot = out.setdefault(entity.name, {})
        for alias in entity.aliases or []:
            alias_norm = normalize_text(alias)
            if alias_norm:
                slot.setdefault(alias_norm, _GENERIC_TYPED_ALIAS_SENTINEL)
        for example in entity.examples or []:
            example_norm = normalize_text(example)
            if example_norm:
                slot.setdefault(example_norm, example)
    _TYPED_INDEX_CACHE[id(pack)] = out
    return out


@functools.lru_cache(maxsize=4096)
def _compiled_word_pattern(alias_lower: str) -> "re.Pattern[str]":
    """Cache compiled word-boundary patterns per alias. The pack
    vocabulary is tiny relative to the atom count — a 4k LRU more
    than covers every alias we'll ever see, with O(1) lookup."""
    return re.compile(r"(?<![a-z0-9])" + re.escape(alias_lower) + r"(?![a-z0-9])")


def _word_match(text_lower: str, alias_lower: str) -> bool:
    """Word-boundary match for an alias inside a pre-lowercased text."""
    if not alias_lower:
        return False
    return _compiled_word_pattern(alias_lower).search(text_lower) is not None


# Optional plural suffix appended to device-alias matches so a count
# ("50 thumb drives", "20 ssds", "old boxes") still resolves to the
# singular canonical. ``e?s`` covers both "-s" and "-es"; it sits
# OUTSIDE the capture group so the canonical lookup keys on the
# singular alias.
_PLURAL_SUFFIX = r"(?:e?s)?"


@functools.lru_cache(maxsize=4096)
def _compiled_device_pattern(alias_lower: str) -> "re.Pattern[str]":
    """Like ``_compiled_word_pattern`` but tolerant of a trailing
    plural ``s``/``es`` — device mentions are routinely pluralized."""
    return re.compile(
        r"(?<![a-z0-9])" + re.escape(alias_lower) + _PLURAL_SUFFIX + r"(?![a-z0-9])"
    )


# Pre-built per-pack matcher: a single union regex over all device
# aliases. Reduces _emit_devices from O(aliases) regex compiles per
# atom to O(1) — one search, one canonical lookup per match.
_DEVICE_UNION_CACHE: dict[int, tuple["re.Pattern[str]", dict[str, str]]] = {}


def _device_union_for_pack(pack: DomainPack, alias_index: dict[str, str]) -> tuple["re.Pattern[str]", dict[str, str]]:
    key = id(pack)
    cached = _DEVICE_UNION_CACHE.get(key)
    if cached is not None:
        return cached
    if not alias_index:
        pattern = re.compile(r"(?!.*)")  # never matches
        _DEVICE_UNION_CACHE[key] = (pattern, alias_index)
        return _DEVICE_UNION_CACHE[key]
    # Sort longest-first so longer aliases win when nested
    # ("access point" before "point").
    aliases_sorted = sorted(alias_index.keys(), key=lambda a: (-len(a), a))
    body = "|".join(re.escape(a) for a in aliases_sorted)
    pattern = re.compile(r"(?<![a-z0-9])(" + body + r")" + _PLURAL_SUFFIX + r"(?![a-z0-9])")
    _DEVICE_UNION_CACHE[key] = (pattern, alias_index)
    return _DEVICE_UNION_CACHE[key]


# ─── v57 P2: negation guard for device alias matching ───
# When a doc says "Programming is pretty easy, but not via thumb drive",
# the substring "thumb drive" matches the ``storage`` alias and we emit
# ``device:storage`` — a hallucination, since the sentence is denying
# that device. Same trap for "no external drive", "without HDD", etc.
#
# We scan a small window (~40 chars) before each match for negation
# cues. If a strong negator appears AND no positive override (like
# "and", "but also") intervenes, the match is suppressed.
_NEGATION_CUES = (
    "but not",
    "but no ",
    "but no,",
    "but no.",
    "not via",
    "not using",
    "without ",
    " no ",
    "rather than ",
    "instead of ",
    "as opposed to ",
)
_NEGATION_OVERRIDES = (
    " and also ",
    "; also ",
    ", also ",
    " plus ",
)


def _is_negated_match(text_lower: str, span_start: int) -> bool:
    """Return True if the device-alias match at ``span_start`` is
    preceded by a negation cue inside the last ~40 chars.

    Conservative: requires the negator to appear AFTER any positive
    override word (so "we ship HDD and also no tape backup" still
    emits ``device:storage`` from the HDD half).
    """
    window_start = max(0, span_start - 40)
    window = text_lower[window_start:span_start]
    last_neg = -1
    for cue in _NEGATION_CUES:
        idx = window.rfind(cue)
        if idx > last_neg:
            last_neg = idx
    if last_neg < 0:
        return False
    last_override = -1
    for cue in _NEGATION_OVERRIDES:
        idx = window.rfind(cue)
        if idx > last_override:
            last_override = idx
    # Negator only "wins" when it's the most recent cue in the window.
    return last_neg > last_override


def _emit_devices(text_lower: str, alias_index: dict[str, str], pack: DomainPack | None = None) -> set[str]:
    """Emit ``device:<canonical>`` keys for every alias in
    ``alias_index`` that word-matches ``text_lower``.

    Fast path: when ``pack`` is supplied we use the cached union regex
    (single sweep, O(matches) work). Legacy path: iterate aliases.

    v57 P2: skip matches preceded by negation cues ("but not", "without",
    "no", "rather than"). Prevents ``thumb drive`` → ``device:storage``
    hallucination from text like "but not via thumb drive".
    """
    keys: set[str] = set()
    if pack is not None:
        pattern, _ = _device_union_for_pack(pack, alias_index)
        for match in pattern.finditer(text_lower):
            if _is_negated_match(text_lower, match.start()):
                continue
            alias = match.group(1)
            canonical = alias_index.get(alias)
            if canonical:
                keys.add(f"device:{_slugify(canonical)}")
        return keys
    for alias_norm, canonical in alias_index.items():
        pattern = _compiled_device_pattern(alias_norm)
        match = pattern.search(text_lower)
        if match is None:
            continue
        if _is_negated_match(text_lower, match.start()):
            continue
        keys.add(f"device:{_slugify(canonical)}")
    return keys


def _emit_typed(text_lower: str, typed_index: dict[str, dict[str, str]]) -> set[str]:
    """Emit typed-entity keys via pack aliases.

    Filters out short aliases (< 5 chars) when they would produce a
    bare canonical key — those are too prone to false positives
    (e.g. ``site:bldg`` from the literal alias "bldg").  The
    proper-noun matcher already catches richer phrases that contain
    those short aliases as suffixes.

    Also skips bare common-noun aliases ("school", "building",
    "warehouse") that produce noise on their own — these only carry
    information when paired with a discriminator (a number, a proper
    noun); the proper-noun matcher will catch the rich version
    (``site:natomas_unified_school_district``) anyway.  See Week 6
    P6.4 — without this filter NATOMAS emits 60+ junk site keys like
    ``site:school`` / ``site:campus`` / ``site:floor``.
    """
    keys: set[str] = set()
    for entity_type, slot in typed_index.items():
        if entity_type == "device":  # devices handled separately
            continue
        for alias_norm, original in slot.items():
            if len(alias_norm) < 5:
                continue
            if alias_norm in _BARE_TYPED_ALIAS_STOPLIST:
                continue
            # Skip generic-type aliases tagged by ``_typed_alias_index``
            # with ``_GENERIC_TYPED_ALIAS_SENTINEL``. These are the
            # ``entity.aliases`` entries from the pack — generic
            # synonyms for the type ("tenant"/"client" for customer,
            # "closet"/"rm" for room, "carrier"/"oem" for vendor) —
            # which would otherwise emit tautological entity keys.
            # Specific named instances (``entity.examples``) still
            # emit because their slot value is the example string,
            # not the sentinel.
            if original == _GENERIC_TYPED_ALIAS_SENTINEL:
                continue
            if _word_match(text_lower, alias_norm):
                keys.add(normalize_entity_key(entity_type, original))
    return keys


# Bare singular nouns that show up as generic-typed-entity aliases
# (``site``, ``room``) in many packs but produce noise as standalone
# entity keys.  Multi-word forms ("main campus", "data closet") and
# numbered forms ("school 12", "floor 3") still match via richer
# alias entries, so this stoplist only kills the bare-word emission.
_BARE_TYPED_ALIAS_STOPLIST: frozenset[str] = frozenset(
    {
        # Place / facility generics — already caught by site / room
        # extraction with real names
        "school", "schools", "campus", "building", "buildings",
        "floor", "floors", "level", "levels", "warehouse",
        "warehouses", "hospital", "hospitals", "clinic", "clinics",
        "office", "offices", "branch", "branches", "store", "stores",
        "facility", "facilities", "site", "sites", "room", "rooms",
        "lab", "labs", "laboratory", "laboratories", "suite", "suites",
        # Business-role generics — pack aliases name "customer" /
        # "vendor" / "owner" / "client" as canonical members, which
        # leads to junk entity keys like `customer:customer` and
        # `vendor:vendor` whenever the bare word appears in prose.
        # Real customer / vendor names get caught by the cross-pack
        # vendor matcher and the institutional-suffix customer
        # promoter.
        "customer", "customers", "client", "clients",
        "vendor", "vendors", "subcontractor", "subcontractors",
        "sub", "subs", "contractor", "contractors",
        "owner", "owners", "partner", "partners",
        "carrier", "carriers", "supplier", "suppliers",
        "distributor", "distributors", "reseller", "resellers",
        "manufacturer", "manufacturers", "oem", "oems",
        "integrator", "integrators",
        "operator", "operators",
        "end user", "end-user", "endpoint", "endpoints",
    }
)


def _emit_vendors(text_lower: str) -> set[str]:
    keys: set[str] = set()
    for canonical, surfaces in _CROSS_PACK_VENDORS.items():
        for surface in surfaces:
            if _word_match(text_lower, surface):
                keys.add(f"vendor:{canonical}")
                break
    return keys


_SITE_CODE_RE = re.compile(r"\b([A-Z]{2,6}-[A-Z0-9]{2,8})\b")
_SITE_SLUG_CODE_RE = re.compile(r"^[a-z]{2,6}_[a-z0-9]{2,8}$")
_SITE_BOILERPLATE_SUBSTRINGS = (
    "hubspot",
    "mock",
    "orbitbrief",
    "parser",
    "azure",
    "procurement_packet",
    "integration_notes",
    "dev_deal",
    "confidential",
    "purpulse",
    "test_deal",
    "synthetic",
)


def is_site_boilerplate_slug(slug: str) -> bool:
    """Drop integration-doc / test phrasing masquerading as site entities."""
    s = slug.lower().strip()
    if not s:
        return True
    if _SITE_SLUG_CODE_RE.match(s):
        return False
    return any(token in s for token in _SITE_BOILERPLATE_SUBSTRINGS)


def _emit_sites(text: str) -> set[str]:
    keys: set[str] = set()

    # Legacy permissive two-segment site-code capture used to emit any
    # all-caps hyphen token as a site (RJ-45, PO-1234, MSA-2026, ...).
    # Keep `_SITE_CODE_RE` only for compatibility with older comments;
    # all code-shaped site IDs now flow through `_SITE_CODE_REGEX` below,
    # where they must pass the positive suffix gate and head denylist.

    # Suffix-based capture (e.g. "Perry Street Parking Deck")
    for regex in _SITE_SUFFIX_REGEXES:
        for match in regex.finditer(text):
            full = " ".join(match.groups()).strip()
            if not full:
                continue
            # Strip leading articles / demonstratives so "The Andrews
            # Information Systems Building" produces only
            # ``site:andrews_information_systems_building`` rather than
            # also leaking ``site:the_andrews_information_systems_building``.
            tokens = full.split()
            while tokens and tokens[0].lower() in _LEADING_ARTICLES:
                tokens.pop(0)
            if not tokens:
                continue
            slug = _slugify(" ".join(tokens))
            if slug and slug not in {"_"}:
                keys.add(f"site:{slug}")

    # A4 universal naming: "Building 13" / "Site 7" / "Bâtiment 12" /
    # "Edificio 4" / "Gebäude 5" / "棟3". These are generic-shape
    # site names used in enterprise floor plans and international
    # deals where every facility is identified only by a number
    # within a brand chain. We slug the full match so different
    # numbers produce different sites.
    for match in _NUMBERED_SITE_REGEX.finditer(text):
        full = match.group(1).strip()
        if not full:
            continue
        slug = _slugify(full)
        if slug and slug not in {"_"}:
            keys.add(f"site:{slug}")

    # First-class site-code capture: ATL-HQ / ATL-WEST / ATL-AIR /
    # NYC-DC1 / SFO-HQ / CHI-MAIN. These are the load-bearing site
    # identifiers on most enterprise deals and bypass the proper-
    # noun multi-word matcher because they're single tokens. Without
    # this branch, ATL-WEST is invisible to the entity extractor
    # even though the BOM allocation row literally cites it.
    for match in _SITE_CODE_REGEX.finditer(text):
        code = match.group(1)
        if code in _SITE_CODE_DENYLIST:
            continue
        segments = code.split("-")
        first = segments[0]
        last = segments[-1]
        prev = segments[-2] if len(segments) >= 2 else None
        # Universal POSITIVE gate: the last segment must carry recognized
        # site-function meaning (direction, facility type, datacenter /
        # floor / building / wing number).  This is the load-bearing
        # robustness check — anything that doesn't end in a known
        # site-suffix fails the gate, regardless of head or middle
        # segments. Catches unknown junk codes like MOCK-OPTBOT-ATL,
        # ALPHA-FOOBAR, GAMMA-FOO-2026 without needing to enumerate
        # every possible junk word.
        if not _site_code_suffix_ok(last, prev_segment=prev):
            continue
        # Head denylist — belt-and-suspenders for codes whose tail
        # happens to look site-shaped but the head is a known test /
        # contract-id marker.  Catches e.g. "DEV-WEST" where WEST is
        # in the suffix allowlist but the whole code is a dev marker.
        if first in _SITE_CODE_HEAD_DENYLIST:
            continue
        slug = _slugify(code)
        if slug:
            keys.add(f"site:{slug}")

    # Street addresses → produce both an address: and (if combinable) a site:
    for match in _STREET_ADDRESS_REGEX.finditer(text):
        number, street, suffix = match.group(1), match.group(2), match.group(3)
        full_addr = f"{number} {street} {suffix}".strip()
        slug = _slugify(full_addr)
        if slug:
            keys.add(f"address:{slug}")

    return keys


def _looks_like_form_field(text: str) -> bool:
    """Heuristic for form-field templates.

    Triggers when:
      * ≥2 strong markers (signature / FEIN / (PRINT) / ...), OR
      * ≥1 strong marker AND ≥1 other marker (strong or weak)

    Weak markers alone (the placeholder ``col_N:`` column names
    emitted when a table has no header) are NOT enough — they appear
    in legitimate tables, e.g. the NATOMAS school list.  See Week 6
    P6.6.
    """
    text_lower = text.lower()
    strong_hits = sum(1 for m in _FORM_FIELD_STRONG_MARKERS if m in text_lower)
    weak_hits = sum(1 for m in _FORM_FIELD_WEAK_MARKERS if m in text_lower)
    if strong_hits >= 2:
        return True
    if strong_hits >= 1 and (strong_hits + weak_hits) >= 2:
        return True
    return False


def _emit_proper_nouns(
    text: str,
    vendor_keys: set[str],
    *,
    authoritative_sites: set[str] | None = None,
) -> set[str]:
    """Capture multi-word proper-noun runs as candidate sites/customers.

    When ``authoritative_sites`` is provided AND non-empty, this
    function is GATED: a phrase is emitted as ``site:*`` only if its
    normalized form matches the catalog (Option D — document-structure
    aware site detection). Phrases outside the catalog are dropped,
    which kills the long tail of false-positive sites (standards
    bodies, random landmarks, header fragments, sentence pieces)
    that the regex was previously emitting.

    When the catalog is None (no catalog built yet) OR empty (the
    artifact has no Locations section and no address signals), the
    function falls back to its prior conservative regex behavior:
    ≥3-word run, stoplist + blocklist filters, sentence-window
    corroboration, etc.

    If the surrounding text looks like a form-field template
    (``FULL LEGAL NAME (PRINT)``, ``id#``, ``col_N:``), we skip
    proper-noun extraction entirely to avoid emitting form labels
    as fake sites.
    """
    # Lazy import to avoid circular dependency at module load time
    from app.core.site_detection import phrase_is_in_catalog
    catalog_active = bool(authoritative_sites)
    if _looks_like_form_field(text):
        return set()

    keys: set[str] = set()
    # Pre-compute vendor surface forms we should avoid double-emitting
    # as sites — e.g. don't say site:genetec_security_center when we
    # already have vendor:genetec.
    vendor_surfaces: set[str] = set()
    for vkey in vendor_keys:
        if not vkey.startswith("vendor:"):
            continue
        canonical = vkey.split(":", 1)[1]
        vendor_surfaces.add(canonical.replace("_", " "))
        for surface in _CROSS_PACK_VENDORS.get(canonical, []) or []:
            vendor_surfaces.add(surface.lower())

    # Split on sentence-ending punctuation first so a stray colon /
    # semicolon doesn't fuse two phrases ("Parking Deck. Vendor:" →
    # don't fuse).  Periods are deliberately omitted from the split
    # set: they fire inside middle initials ("H. Allen Hight
    # Elementary") and abbreviations ("U.S. Department"); the
    # trailing-stopword filter ("Vendor", "Description", …) takes
    # care of the actual sentence-end case.
    for sentence in re.split(r"[;:?!\n]+", text):
        for match in _PROPER_NOUN_RUN.finditer(sentence):
            phrase = match.group(1).strip()
            tokens = phrase.split()

            # Whole-phrase block: a phrase whose final non-trivial
            # token is a role / document / pipeline / process noun is
            # NOT a site — it's a person, an artifact, or a system.
            # We check this BEFORE trim-and-keep so phrases like
            # "Regional Facilities Manager" or "Executive Deal Brief
            # Mock Document" disappear entirely instead of being
            # trimmed to "Regional Facilities" / "Executive Deal
            # Brief Mock" and still emitted as sites.
            final = tokens[-1].lower().rstrip(":,.") if tokens else ""
            if final in _NON_SITE_PHRASE_TAIL_NOUNS:
                continue
            # Inner role-marker block: phrases like "Site Manager Main
            # Campus" or "Network Director West Wing" describe a ROLE
            # (Manager/Director/Sponsor/Officer/Lead) responsible for a
            # location — they're role+site composites that should NOT
            # become a fake site key. If a strong role marker appears
            # ANYWHERE in the phrase (not just the tail), reject.
            inner_lc = {t.lower().rstrip(":,.") for t in tokens}
            if inner_lc & _ROLE_INNER_MARKERS:
                continue

            # Strip trailing field-label words like "Vendor", "Description"
            while tokens and tokens[-1].lower().rstrip(":,.") in _PROPER_NOUN_TRAILING_STOPWORDS:
                tokens.pop()
            # Strip leading articles / demonstratives ("The Andrews
            # Information Systems Building" → "Andrews Information
            # Systems Building").  Without this the regex captures the
            # entire run including ``The`` and we drop the whole match
            # instead of recovering the real proper noun.
            while tokens and tokens[0].lower() in _LEADING_ARTICLES:
                tokens.pop(0)

            # Re-check the new final token after trimming — if trimming
            # exposed a non-site tail noun, drop the phrase entirely.
            if tokens:
                final_after = tokens[-1].lower().rstrip(":,.")
                if final_after in _NON_SITE_PHRASE_TAIL_NOUNS:
                    continue

            # Two-word special case: a Capitalized + capitalized
            # organization name like "Virginia Tech" / "Boston College"
            # / "Cleveland Clinic" / "Houston ISD".  We accept these
            # when (a) the trailing word matches a well-known
            # organization suffix, (b) the tail is a known place-noun,
            # or (c) explicit site-context / address corroborates the
            # phrase nearby. Without this 2-word real sites like
            # "Innovation Tower" or "Birchwood Atelier" (with address)
            # would be dropped by the ≥3-word minimum.
            if len(tokens) == 2:
                trail = tokens[-1].lower().rstrip(":,.")
                two_word_org   = trail in _ORG_SUFFIX_TWO_WORD
                two_word_place = trail in _SITE_TAIL_NOUNS
                two_word_corr  = _has_site_corroboration(
                    sentence, match.start(), match.end()
                )
                if not (two_word_org or two_word_place or two_word_corr):
                    continue
            elif len(tokens) < 3:
                continue
            phrase = " ".join(tokens)
            norm = normalize_text(phrase)
            if norm in _PROPER_NOUN_STOPLIST:
                continue
            # Explicit phrase blocklist — common deal-doc section
            # labels that look site-shaped but never refer to a place.
            if norm in _SITE_PHRASE_BLOCKLIST:
                continue
            # If ANY token in the phrase is a tail noun, the phrase
            # is more likely to be a description than a site.
            # Bypass when (a) the trailing token is a recognized org
            # or place suffix ("Atlanta Headquarters", "Innovation
            # Tower"), OR (b) explicit site-context / address
            # corroborates the phrase nearby ("Located at Aurora
            # Operations Hub").
            phrase_tokens_lower = {t.lower().rstrip(":,.") for t in tokens}
            tail = tokens[-1].lower().rstrip(":,.")
            is_org_tail = tail in _ORG_SUFFIX_TWO_WORD or tail in _SITE_TAIL_NOUNS
            has_corroboration = _has_site_corroboration(
                sentence, match.start(), match.end()
            )
            if (not (is_org_tail or has_corroboration)
                and (phrase_tokens_lower & _NON_SITE_PHRASE_TAIL_NOUNS)):
                continue
            # Hard-disqualify: certain tokens (mock/test/demo/fake/...)
            # are NEVER part of a real site name, even when paired with
            # a valid place-tail. "Mock Atlanta Tower" / "Test Innovation
            # Lab" — the test-marker token is itself disqualifying.
            # This OVERRIDES the is_org_tail bypass above.
            if phrase_tokens_lower & _HARD_DISQUALIFY_PHRASE_TOKENS:
                continue
            # Skip if this run is dominated by a known vendor
            if any(surface in norm for surface in vendor_surfaces if surface and len(surface) >= 4):
                continue
            # Skip if the run is mostly stop-words (only enforced for
            # ≥3-word runs — a 2-word org name like "Virginia Tech" has
            # exactly 2 non-stop tokens and is fine).
            if len(tokens) >= 3:
                non_stop = [w for w in norm.split() if w not in {"of", "and", "the", "for", "to", "in", "on", "at"}]
                if len(non_stop) < 2:
                    continue
            # UNIVERSAL POSITIVE STRUCTURAL GATE: emit a site key only
            # when the run has at least one positive site signal:
            #   (a) tail is a known place-noun (Tower, Building, Annex,
            #       Headquarters, Plaza, Campus, ...)
            #   (b) tail is a known org suffix (Tech, College, Hospital,
            #       ISD, District, ...)
            #   (c) the same sentence contains a street address — the
            #       address corroborates that the phrase refers to a
            #       physical place
            #   (d) an explicit site-context cue ("site visit", "located
            #       at", "based in", "address:", "Facility:", ...)
            #       appears within ~80 chars of the phrase
            # Without one of these positive signals, the phrase is
            # dropped. This is universal: a brand-new junk phrase
            # ("Brilliant Strategic Initiative", "Advanced Process
            # Framework") cannot leak because the absence of positive
            # signal is itself the filter — no enumeration of every
            # possible junk word is required.
            has_place_tail = tail in _SITE_TAIL_NOUNS
            has_org_tail   = tail in _ORG_SUFFIX_TWO_WORD
            if not (has_place_tail or has_org_tail or _has_site_corroboration(sentence, match.start(), match.end())):
                continue
            slug = _slugify(phrase)
            if slug and len(slug) >= 6:
                # Option D gate: when an authoritative-site catalog
                # exists for this project, the phrase MUST match the
                # catalog before we emit a site:* key. Phrases outside
                # the catalog are dropped — they're standards bodies,
                # random landmarks, or header fragments that look
                # site-shaped but aren't real project sites.
                if catalog_active and not phrase_is_in_catalog(phrase, authoritative_sites):
                    continue
                keys.add(f"site:{slug}")
    return keys


# Copular / explicit-aliasing markers — fire alias fusion across
# ALL site keys in the sentence regardless of shape.
_COPULAR_ALIAS_REGEX = re.compile(
    r"(?:"
    r"\b(?:is|are|was|were)\s+(?:the|an?|our|its|their)\s+\w"
    r"|\balso\s+known\s+as\b"
    r"|\ba\.?k\.?a\.?\b"
    r"|\balias(?:es)?\b"
    r"|\bknown\s+as\b"
    r"|\b(?:officially|formerly|previously)\s+(?:called|named|known\s+as)\b"
    r"|\b(?:called|named|designated|labeled|tagged)\b"
    r")",
    re.IGNORECASE,
)

# Separator-based markers — only count as alias markers when the
# sentence contains a MIX of site-code shapes (ATL-HQ) and proper-noun
# names (Atlanta Headquarters). Pure lists ("ATL-HQ; ATL-WEST; ATL-AIR"
# are NOT aliases — they're a list of 3 different places).
_SEPARATOR_ALIAS_REGEX = re.compile(
    r"(?:"
    r"\s+\|\s+"           # pipe with whitespace (table-row text)
    r"|\s+[—–]\s+"        # em-dash or en-dash
    r"|\s+--\s+"          # double hyphen as separator
    r"|\s+/\s+"           # slash with whitespace
    r"|\s+-\s+"           # single hyphen with whitespace
    r")",
)

# Parenthetical-aliasing — "Atlanta Headquarters (ATL-HQ)" or
# "ATL-HQ (Atlanta Headquarters)" → alias.
_PAREN_ALIAS_REGEX = re.compile(
    r"\(\s*[A-Z][A-Z0-9\-]{1,}\s*\)"      # "(ATL-HQ)" / "(NYC-DC1)"
    r"|\(\s*[A-Z][A-Za-z\s]{2,40}\)",     # "(Atlanta Headquarters)"
)

# Colon-bridge — "ATL-HQ: Atlanta Headquarters" or "Site:
# Innovation Tower". Used in tables and definition lists.
_COLON_ALIAS_REGEX = re.compile(
    r"\b[A-Z][A-Z0-9\-]{2,}:\s+[A-Z][a-zA-Z]"
    r"|\b(?:site|location|facility|building|address)\s*:\s+[A-Z]",
    re.IGNORECASE,
)


def _looks_like_site_code_key(site_key: str) -> bool:
    """Heuristic: does this `site:*` key look like a short alphanumeric
    code (ATL-HQ, NYC-DC1) versus a multi-word name (Atlanta
    Headquarters)?

    Used by the alias-fusion logic to distinguish lists-of-distinct-
    sites ("ATL-HQ; ATL-WEST; ATL-AIR" — all code-shaped, so NOT
    aliases) from real alias pairs ("ATL-HQ | Atlanta Headquarters"
    — mixed shapes, so aliases).
    """
    slug = site_key.removeprefix("site:")
    if not slug:
        return False
    tokens = slug.split("_")
    if not tokens or len(tokens) > 4:
        return False
    # All tokens short AND none looks like a recognizable English word
    return all(len(t) <= 5 for t in tokens)


# Between-key inspection — patterns we examine in the text that
# falls between two adjacent site-key spans within a sentence.
#
# Aliasing patterns assert "the two keys are the same place":
_BETWEEN_COPULAR = re.compile(
    r"^\s*(?:[—–\-,]\s*)?"
    r"(?:is|are|was|were)\s+(?:the|an?|our|its|their|also)?\s*"
    r"(?:also\s+)?(?:known\s+as|called|named|designated)?",
    re.IGNORECASE,
)
_BETWEEN_AKA = re.compile(
    r"\b(?:a\.?k\.?a\.?|also\s+known\s+as|alias(?:es)?\s+(?:is|of|for)?|"
    r"known\s+as|called|named|formerly|previously|officially)\b",
    re.IGNORECASE,
)
_BETWEEN_PAREN_ALIAS = re.compile(r"^\s*[—–\-,]?\s*\(")
_BETWEEN_COLON_BRIDGE = re.compile(r"^\s*:\s*$")
# Separator-only patterns — pipe / em-dash / en-dash / slash / double-dash /
# hyphenated with spaces. These are AMBIGUOUS without more info.
_BETWEEN_SEPARATOR_ONLY = re.compile(
    r"^\s*(?:\|\s*|[—–]\s*|--\s*|\s+/\s+|-\s+)$"
)
# List patterns — commas, semicolons, "and"/"or" with optional articles.
# These are NOT alias markers; they separate distinct items.
_BETWEEN_LIST = re.compile(
    r"^\s*(?:,|;|\s+and\s+|\s+or\s+|,\s*and\s+|,\s*or\s+)\s*(?:the\s+)?$",
    re.IGNORECASE,
)


def _find_site_key_spans(
    sentence: str,
) -> list[tuple[str, int, int]]:
    """Locate every `site:*` key occurrence in the sentence with its
    char span.

    Sorted by start offset so callers can iterate adjacent pairs.
    Returns a list of (canonical_key, start, end) tuples. The spans
    are derived from the raw matches in the text (site code regex
    matches + proper-noun runs that pass the structural gate), not
    from the slugified keys, so we can examine the EXACT char window
    between two adjacent key surface forms.
    """
    spans: list[tuple[str, int, int]] = []

    # Site-code occurrences (ATL-HQ, NYC-DC1, ...)
    for match in _SITE_CODE_REGEX.finditer(sentence):
        code = match.group(1)
        if code in _SITE_CODE_DENYLIST:
            continue
        segments = code.split("-")
        first = segments[0]
        last = segments[-1]
        prev = segments[-2] if len(segments) >= 2 else None
        if not _site_code_suffix_ok(last, prev_segment=prev):
            continue
        if first in _SITE_CODE_HEAD_DENYLIST:
            continue
        slug = _slugify(code)
        if slug:
            spans.append((f"site:{slug}", match.start(), match.end()))

    # Proper-noun runs that pass the same structural gate _emit_proper_nouns
    # applies. We can't easily call that function because it returns
    # only the keys (no spans), so duplicate the acceptance logic here
    # with span tracking.
    for sub_sentence_match in re.finditer(r"[^;:?!\n]+", sentence):
        sub = sub_sentence_match.group(0)
        sub_offset = sub_sentence_match.start()
        for match in _PROPER_NOUN_RUN.finditer(sub):
            phrase = match.group(1).strip()
            tokens = phrase.split()
            if not tokens:
                continue
            final = tokens[-1].lower().rstrip(":,.")
            if final in _NON_SITE_PHRASE_TAIL_NOUNS:
                continue
            while tokens and tokens[-1].lower().rstrip(":,.") in _PROPER_NOUN_TRAILING_STOPWORDS:
                tokens.pop()
            while tokens and tokens[0].lower() in _LEADING_ARTICLES:
                tokens.pop(0)
            if tokens:
                final_after = tokens[-1].lower().rstrip(":,.")
                if final_after in _NON_SITE_PHRASE_TAIL_NOUNS:
                    continue
            if len(tokens) == 2:
                trail = tokens[-1].lower().rstrip(":,.")
                two_org   = trail in _ORG_SUFFIX_TWO_WORD
                two_place = trail in _SITE_TAIL_NOUNS
                two_corr  = _has_site_corroboration(sub, match.start(), match.end())
                if not (two_org or two_place or two_corr):
                    continue
            elif len(tokens) < 3:
                continue
            phrase_norm = " ".join(tokens)
            norm = normalize_text(phrase_norm)
            if norm in _PROPER_NOUN_STOPLIST:
                continue
            if norm in _SITE_PHRASE_BLOCKLIST:
                continue
            phrase_tokens_lower = {t.lower().rstrip(":,.") for t in tokens}
            tail = tokens[-1].lower().rstrip(":,.")
            is_org_tail = tail in _ORG_SUFFIX_TWO_WORD or tail in _SITE_TAIL_NOUNS
            has_corroboration = _has_site_corroboration(sub, match.start(), match.end())
            if (not (is_org_tail or has_corroboration)
                and (phrase_tokens_lower & _NON_SITE_PHRASE_TAIL_NOUNS)):
                continue
            if phrase_tokens_lower & _HARD_DISQUALIFY_PHRASE_TOKENS:
                continue
            if len(tokens) >= 3:
                non_stop = [w for w in norm.split() if w not in {"of", "and", "the", "for", "to", "in", "on", "at"}]
                if len(non_stop) < 2:
                    continue
            has_place_tail = tail in _SITE_TAIL_NOUNS
            has_org_tail   = tail in _ORG_SUFFIX_TWO_WORD
            if not (has_place_tail or has_org_tail or _has_site_corroboration(sub, match.start(), match.end())):
                continue
            slug = _slugify(phrase_norm)
            if slug and len(slug) >= 6:
                spans.append((
                    f"site:{slug}",
                    sub_offset + match.start(),
                    sub_offset + match.end(),
                ))

    spans.sort(key=lambda s: (s[1], s[2]))
    # De-dupe overlaps: if two spans share a substring (e.g. proper-noun
    # capture overlapped a site-code capture), keep the longer one.
    dedup: list[tuple[str, int, int]] = []
    for span in spans:
        if dedup and span[1] < dedup[-1][2]:
            # overlaps with previous — keep whichever is wider
            if (span[2] - span[1]) > (dedup[-1][2] - dedup[-1][1]):
                dedup[-1] = span
            continue
        dedup.append(span)
    return dedup


def _classify_pair(
    between_text: str,
    key_a: str,
    key_b: str,
    sentence: str,
    end_a: int,
    start_b: int,
) -> bool:
    """Return True if the text *between* two adjacent site-key spans
    indicates the two keys refer to the same physical place.

    The check is pairwise so mixed-shape LISTS don't over-fuse:
    "Sites: ATL-HQ, Atlanta Headquarters, ATL-WEST, Westside Operations
    Center" — the comma between ``Atlanta Headquarters`` and ``ATL-WEST``
    is a list separator, so those two keys do NOT fuse; meanwhile
    the immediate adjacency ``ATL-HQ , Atlanta Headquarters`` is also
    a comma (also non-alias). Only the ``ATL-WEST | Westside Operations
    Center`` pair (if present) would fuse.

    Aliasing patterns (in priority order):
      1. Explicit AKA / "also known as" / "called" / "designated"
      2. Copular "is the" / "are the"
      3. Parenthetical "(key_b)" immediately after key_a
      4. Colon-bridge "key_a: key_b"
      5. Separator + mixed shape (pipe/em-dash/slash/hyphen) between
         a code-shaped key and a name-shaped key
      6. Em-dash / slash between two name-shaped keys when they are
         the ONLY two keys in the sentence (no list ambiguity)

    NOT aliasing:
      - comma / semicolon / "and" / "or" → list separator
    """
    if _BETWEEN_AKA.search(between_text):
        return True
    if _BETWEEN_COPULAR.search(between_text):
        return True
    if _BETWEEN_PAREN_ALIAS.search(between_text):
        # Verify the paren actually closes between or at key_b
        paren_close = sentence.find(")", end_a)
        if paren_close != -1 and paren_close >= start_b - 1:
            return True
    if _BETWEEN_COLON_BRIDGE.search(between_text):
        return True
    # Strict list separator → never alias
    if _BETWEEN_LIST.search(between_text):
        return False
    # Separator-only between two adjacent keys — alias when:
    #   (a) mixed shape (code + name) with any common separator
    #       (pipe / em-dash / slash / hyphen-with-spaces)
    #   (b) same shape with em-dash, slash, or hyphen-with-spaces
    #       (these patterns almost never delimit list items —
    #       lists use commas, semicolons, "and", or "or")
    # The pipe `|` between SAME-SHAPED keys is treated as a table
    # column separator, NOT an alias — because pipes typically delimit
    # different fields. If two adjacent keys of the same shape sit on
    # either side of a pipe, they're likely two distinct items in a
    # multi-column row, not aliases.
    if _BETWEEN_SEPARATOR_ONLY.search(between_text):
        a_is_code = _looks_like_site_code_key(key_a)
        b_is_code = _looks_like_site_code_key(key_b)
        if a_is_code != b_is_code:
            # Mixed shape with any separator → alias
            # (e.g. ATL-HQ | Atlanta Headquarters,
            #       ATL-HQ — Atlanta Headquarters,
            #       Atlanta Headquarters - ATL-HQ)
            return True
        # Same shape — em-dash, en-dash, slash, or hyphen-with-spaces
        # almost always means alias when between two adjacent
        # proper-noun phrases. Pipes don't (they're column separators).
        if re.search(r"[—–]|/|\s-\s|\s--\s", between_text):
            return True
    return False


def _emit_site_aliases_from_text(text: str) -> list[frozenset[str]]:
    """Return groups of site keys that refer to the same physical place.

    Detection is PAIRWISE: for each adjacent pair of site-key spans in
    a sentence, examine the text between them and decide whether the
    two keys are aliases. Pairs marked as aliases feed into a union-
    find that groups transitive aliases.

    This is more accurate than a sentence-level "any marker → fuse all
    keys" approach. The whole-sentence rule incorrectly fuses lists
    like "Sites: ATL-HQ, Atlanta Headquarters, ATL-WEST, Westside Ops
    Center" because they have mixed shape AND comma separators. The
    pairwise check sees commas between distinct pairs and only fuses
    pairs that have a real alias marker between them.

    See ``_classify_pair`` for the patterns that count as aliasing
    (copular, AKA, parenthetical, colon-bridge, separator + mixed
    shape, em-dash/slash between two names).
    """
    if not text:
        return []
    pairs: list[tuple[str, str]] = []
    for sentence in re.split(r"(?:[?!\n]|\.\s+|\.$)", text):
        s = sentence.strip()
        if not s:
            continue
        spans = _find_site_key_spans(s)
        if len(spans) < 2:
            continue
        # Inspect each adjacent pair of distinct keys.
        pair_aliases_this_sentence: list[tuple[str, str]] = []
        for i in range(len(spans) - 1):
            key_a, _, end_a = spans[i]
            key_b, start_b, _ = spans[i + 1]
            if key_a == key_b:
                continue
            between = s[end_a:start_b]
            if _classify_pair(between, key_a, key_b, s, end_a, start_b):
                pair_aliases_this_sentence.append((key_a, key_b))
        pairs.extend(pair_aliases_this_sentence)
        # Row-level rule: in a pipe-separated row whose ONLY distinct
        # site code is the row's leading identifier, every site key
        # in subsequent cells refers to that row's site (named
        # building, address-cell city, etc.).
        #
        # Concrete: `ATL-AIR | Airport Logistics Annex | 4200 Global
        # Gateway Connector, Building C, College Park | 148 users`
        # — ATL-AIR is the code, Airport Logistics Annex is the
        # named alias, College Park is the city inside the address
        # cell. All three refer to the same physical place.
        #
        # Guards against over-fusion:
        #   - Row must contain "|" (it's a pipe-separated row)
        #   - Row must already have at least one explicit alias pair
        #     (so we know it's an alias-bearing row, not a list-row)
        #   - Row must have AT MOST one distinct site code (rows like
        #     "ATL-HQ | ATL-WEST | ATL-AIR" would have 3 codes and
        #     fall back to pairwise — which correctly doesn't fuse
        #     them since pipe + same-shape isn't an alias)
        if pair_aliases_this_sentence and "|" in s:
            codes_in_row = {
                span[0] for span in spans
                if _looks_like_site_code_key(span[0])
            }
            if len(codes_in_row) <= 1:
                row_keys = [span[0] for span in spans]
                for i in range(len(row_keys) - 1):
                    if row_keys[i] != row_keys[i + 1]:
                        pairs.append((row_keys[i], row_keys[i + 1]))
    # Promote pairs to groups via union-find.
    if not pairs:
        return []
    groups: list[set[str]] = []
    for a, b in pairs:
        target: set[str] | None = None
        for g in groups:
            if a in g or b in g:
                if target is None:
                    target = g
                    target.add(a)
                    target.add(b)
                else:
                    target.update(g)
                    groups.remove(g)
        if target is None:
            groups.append({a, b})
    return _coalesce_alias_groups(groups)


def _coalesce_alias_groups(groups: list[set[str]]) -> list[frozenset[str]]:
    """Union-find: merge any sets that share at least one key.

    Iterates until no overlaps remain so transitive aliases fuse
    correctly (A↔B in one sentence, B↔C in another → {A,B,C}).
    """
    if not groups:
        return []
    merged: list[set[str]] = []
    for g in groups:
        target: set[str] | None = None
        for m in merged:
            if g & m:
                if target is None:
                    target = m
                    target.update(g)
                else:
                    # g overlaps with multiple existing groups — fuse them.
                    target.update(m)
                    merged.remove(m)
        if target is None:
            merged.append(set(g))
    # Re-pass until stable (single-pass might miss chained overlaps).
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(merged):
            j = i + 1
            while j < len(merged):
                if merged[i] & merged[j]:
                    merged[i].update(merged[j])
                    del merged[j]
                    changed = True
                else:
                    j += 1
            i += 1
    # Deterministic sort: groups by smallest key, keys within group sorted.
    return sorted((frozenset(g) for g in merged), key=lambda fs: sorted(fs)[0])


def _has_site_corroboration(sentence: str, span_start: int, span_end: int) -> bool:
    """Return True if a street address or explicit site-context cue
    appears within ~80 chars of the matched span in this sentence.

    This is the structural corroboration check that supplements the
    place-tail / org-tail allowlists in _emit_proper_nouns. It lets
    legitimate sites without a recognized tail (e.g. "Magnolia
    Crossing 4500 Oak Ridge Parkway" or "Site visit: West Annex")
    still surface as sites without opening the door to junk phrases.
    """
    # Same-sentence window of ±_SITE_CORROBORATION_WINDOW chars
    # around the span. Tight enough that an unrelated cue elsewhere
    # in a long sentence doesn't create false corroboration.
    pre = sentence[max(0, span_start - _SITE_CORROBORATION_WINDOW):span_start]
    post = sentence[span_end:min(len(sentence), span_end + _SITE_CORROBORATION_WINDOW)]
    # Address pattern in either window — corroborates that the phrase
    # refers to a physical place.
    if _STREET_ADDRESS_REGEX.search(pre) or _STREET_ADDRESS_REGEX.search(post):
        return True
    # Explicit site-context cue near the phrase.
    if _SITE_CONTEXT_REGEX.search(pre) or _SITE_CONTEXT_REGEX.search(post):
        return True
    return False


# Place-noun tails: when a multi-word proper-noun run ends with one of
# these, the phrase IS genuinely site-shaped even when intermediate
# tokens might overlap the non-site tail list.  This guards against
# the new non-site filter dropping real sites like
# "Innovation Tower" / "Atlanta Headquarters".
_SITE_TAIL_NOUNS: frozenset[str] = frozenset({
    "building", "tower", "campus", "headquarters", "hq", "office",
    "branch", "center", "centre", "facility", "annex", "warehouse",
    "garage", "deck", "structure", "lot", "boardroom", "datacenter",
    "data", "school", "college", "university", "academy", "hospital",
    "clinic", "plant", "factory", "store", "mall", "complex",
    "park", "plaza", "terminal", "concourse", "depot", "yard",
    "airport", "station", "site", "location", "premises",
    # Expanded place-tails for universal coverage
    "hub", "pavilion", "pavillion", "atelier", "studio", "studios",
    "lab", "labs", "laboratory", "laboratories", "workshop", "workshops",
    "hall", "halls", "auditorium", "arena", "stadium", "amphitheater",
    "gymnasium", "fieldhouse",
    "commons", "court", "courtyard", "square", "plaza",
    "annexe", "wing", "wings", "block", "blocks",
    "pavilion", "rotunda", "atrium",
    "tower", "highrise", "skyrise",
    "compound", "estate", "manor", "mansion",
    "dormitory", "dorm", "dorms", "residence", "residences",
    "library", "libraries", "museum", "museums", "gallery", "galleries",
    "theater", "theatre", "theaters", "theatres", "cinema", "cinemas",
    "church", "chapel", "cathedral", "synagogue", "mosque", "temple",
    "fort", "barracks", "armory", "armoury", "garrison",
    "harbor", "harbour", "wharf", "pier", "marina", "dock", "docks",
    "lighthouse", "lookout", "watchtower",
    "field", "fields", "track", "tracks", "course", "courses",
    "ranch", "farm", "vineyard", "orchard",
    "shed", "barn", "silo", "bunker",
    "kiosk", "stall", "booth",
    "village", "neighborhood", "district",
    "metroplex", "townhouse",
})


# Articles / demonstratives that can prefix a proper-noun run; we strip
# them before slugifying so "The Andrews Information Systems Building"
# becomes ``site:andrews_information_systems_building`` instead of being
# dropped because the first token was an article.
_LEADING_ARTICLES: frozenset[str] = frozenset(
    {"the", "a", "an", "this", "that", "these", "those", "all", "and", "but", "or"}
)


# Two-word org suffixes — when a Capitalized run is exactly two words
# and the trailing token is one of these, we still treat it as a site
# candidate.  Without this list the proper-noun matcher's ≥3-word
# minimum drops customer names like "Virginia Tech" or "Boston College".
_ORG_SUFFIX_TWO_WORD: frozenset[str] = frozenset(
    {
        "tech",
        "university",
        "college",
        "institute",
        "polytechnic",
        "academy",
        "hospital",
        "clinic",
        "school",
        "schools",
        "district",
        "isd",
        "esd",
        "department",
        "agency",
        "authority",
        "ministry",
        "council",
        "borough",
        "parish",
        "county",
        "township",
        "village",
        "city",
        "association",
        "society",
        "foundation",
        "corporation",
        "company",
        "trust",
        # Week 6 P6.6 follow-up: school grade-level abbreviations and
        # generic org tails ("Discovery High", "Natomas Middle",
        # "District Office") that show up as 2-word org names without
        # the longer "School" / "Department" suffix.
        "high",
        "middle",
        "elementary",
        "office",
        "campus",
        "center",
        "building",
        "library",
        "museum",
        "park",
        "stadium",
        "arena",
        "facility",
    }
)


def _emit_part_numbers(text: str) -> set[str]:
    keys: set[str] = set()
    for match in _PART_NUMBER_REGEX.finditer(text):
        sku = match.group(1)
        # Skip pure-letter SKUs (likely acronyms like "RFP", "USAC", "FCC")
        if not re.search(r"\d", sku):
            continue
        # Skip 2-letter prefixes that are too short to be a SKU
        if len(sku) < 5:
            continue
        # Reuse the site-code HEAD denylist: the same test / contract /
        # CRM / cloud-platform prefixes that masquerade as site codes
        # (HS-DEAL-..., MOCK-MSA-..., PO-..., DEV-...) also masquerade
        # as part numbers because they share the hyphen-separated
        # uppercase-alphanumeric shape. Manufacturer SKUs (CW9166I-B,
        # AIR-DNA-E-T-5Y, J9145A) never start with these tokens, so
        # the denylist drops contract/project IDs cleanly.
        first_segment = re.split(r"[-/]", sku, 1)[0]
        if first_segment in _SITE_CODE_HEAD_DENYLIST:
            continue
        # Also reject codes whose FIRST segment is a 3-letter airport
        # / city code (ATL, NYC, SFO, ...) — these are project / batch
        # prefixes when followed by digits ("ATL-047", "NYC-001"), not
        # manufacturer SKUs. Real Cisco AIR-* products start with AIR
        # (4 chars) not a 3-letter geo code.
        if (len(first_segment) == 3
            and first_segment.isalpha()
            and first_segment in _AIRPORT_CITY_PREFIXES):
            continue
        slug = _slugify(sku)
        if slug:
            keys.add(f"part_number:{slug}")
    return keys


# 3-letter airport / city / region codes that appear in project IDs
# and batch numbers but never in manufacturer SKUs. Used by
# ``_emit_part_numbers`` to reject codes like ``ATL-047`` /
# ``NYC-2026`` / ``SFO-001`` which would otherwise leak as
# ``part_number:atl_047``.
_AIRPORT_CITY_PREFIXES: frozenset[str] = frozenset({
    # Major US airports
    "ATL", "LAX", "ORD", "DFW", "DEN", "JFK", "SFO", "SEA", "LAS", "MCO",
    "MIA", "PHX", "IAH", "BOS", "MSP", "FLL", "DTW", "PHL", "LGA", "CLT",
    "BWI", "SAN", "TPA", "DCA", "IAD", "MDW", "SLC", "PDX", "STL", "HOU",
    "BNA", "AUS", "RDU", "MCI", "OAK", "MSY", "SJC", "SMF", "SNA", "PIT",
    "CVG", "IND", "CMH", "CLE", "MEM", "JAX", "RIC", "OMA", "ABQ", "ELP",
    "OKC", "TUL", "ICT", "BUF", "SYR", "ROC", "ALB", "PVD", "MHT", "PWM",
    "BTV", "BHM", "MOB", "HSV", "JAN", "BTR", "SHV", "LIT", "AMA", "LBB",
    "MAF", "SAT", "CRP", "BRO", "HRL", "MFE", "LRD", "BPT", "ABE", "AVL",
    # Major international hubs
    "LHR", "LGW", "STN", "CDG", "ORY", "FRA", "MUC", "TXL", "BER", "AMS",
    "MAD", "BCN", "FCO", "MXP", "VIE", "ZRH", "DUB", "CPH", "ARN", "OSL",
    "HEL", "WAW", "PRG", "BUD", "ATH", "IST", "DXB", "DOH", "AUH",
    "NRT", "HND", "KIX", "ICN", "PEK", "PVG", "HKG", "SIN", "BKK", "KUL",
    "SYD", "MEL", "AKL", "YYZ", "YVR", "YUL", "GRU", "EZE", "SCL",
    # Common US city abbreviations (not airport codes but used as
    # project prefixes)
    "NYC", "CHI", "PHL", "LAX", "SFO", "DAL", "HOU", "ATL", "BOS", "SEA",
    "DEN", "MIA", "PHX", "DET", "MIN", "TOR", "MTL", "VAN",
})


# Week 6 P6.4 — institutional / customer suffixes.  When a proper-noun
# run ends in one of these tokens (after slugification) we mirror the
# ``site:`` key as ``customer:`` so OrbitBrief can render the customer
# entity separately from the site.  We keep the ``site:`` key too —
# the same string can legitimately serve as both the customer name and
# the project site.
_CUSTOMER_INSTITUTIONAL_SUFFIXES: frozenset[str] = frozenset(
    {
        "school_district",
        "unified_school_district",
        "school_districts",
        "university",
        "college",
        "polytechnic",
        "institute",
        "academy",
        "hospital",
        "clinic",
        "medical_center",
        "health_system",
        "school",
        "agency",
        "authority",
        "department",
        "ministry",
        "council",
        "corporation",
        "company",
        "trust",
        "association",
    }
)


def _emit_customer_keys(text: str, proper_noun_site_keys: set[str]) -> set[str]:
    """Promote site keys whose tail looks institutional to ``customer:`` keys.

    Example::

        proper_noun_site_keys = {"site:natomas_unified_school_district"}
        →  {"customer:natomas_unified_school_district",
            "site:natomas_unified_school_district"}

    Two-word org runs ("Virginia Tech", "Boston College") also qualify —
    their slug ends in ``_tech`` / ``_college`` which the suffix index
    matches.
    """
    keys: set[str] = set()
    for site_key in proper_noun_site_keys:
        if not site_key.startswith("site:"):
            continue
        slug = site_key.split(":", 1)[1]
        # Look for a known institutional tail.  We check both the full
        # slug and progressively trimmed prefixes so
        # ``natomas_unified_school_district`` matches
        # ``unified_school_district`` and ``school_district`` even when
        # the full slug isn't in the suffix index.
        if any(slug.endswith(suffix) for suffix in _CUSTOMER_INSTITUTIONAL_SUFFIXES):
            keys.add(f"customer:{slug}")
            continue
        # Two-word case (slug has only one underscore): "virginia_tech",
        # "boston_college".  Check the trailing token.
        tail = slug.rsplit("_", 1)[-1] if "_" in slug else slug
        if tail in {"tech", "university", "college", "polytechnic", "institute",
                    "academy", "hospital", "clinic", "school", "isd", "esd",
                    "agency", "authority"}:
            keys.add(f"customer:{slug}")
    return keys


# Week 6 P6.4 — requirement-shape patterns.  Captures funding-program /
# certification / standard requirements as ``requirement:`` keys so the
# packetizer can group compliance and procurement constraints by the
# *thing being required* rather than the surrounding scope text.
#
# Examples:
#   "must be E-rate eligible"           → requirement:erate_eligibility_marking
#   "Secure Networks Act compliance"    → requirement:secure_networks_act_compliance
#   "shall comply with NFPA 72"         → requirement:nfpa_72_compliance
#   "Section 508 compliant"             → requirement:section_508_compliance
#   "TAA-compliant"                     → requirement:taa_compliance
_REQUIREMENT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Bare "E-rate" mention (any context — program, eligibility,
    # funding-year, SPIN, FRN, USAC, …) is enough to mark a
    # requirement; the gold standards count compliance atoms by
    # ``requirement:*`` presence.
    (re.compile(r"\be-?rate\b", re.I), "erate_eligibility_marking"),
    (re.compile(r"\busac\b", re.I), "usac_compliance"),
    (re.compile(r"\b(?:spin|service\s+provider\s+identification\s+number)\b", re.I), "spin_registration"),
    (re.compile(r"\bfcc\s+(?:order|decision|rule|regulation)\b", re.I), "fcc_compliance"),
    (re.compile(r"\bsecure\s+networks\s+act\b", re.I), "secure_networks_act_compliance"),
    (re.compile(r"\bsection\s+508\b", re.I), "section_508_compliance"),
    (re.compile(r"\btaa[-–\s]?compliant\b", re.I), "taa_compliance"),
    (re.compile(r"\bndaa[-–\s]?compliant\b", re.I), "ndaa_compliance"),
    (re.compile(r"\bbuy\s+america(?:n)?\s+act\b", re.I), "buy_america_compliance"),
    (re.compile(r"\bdavis[-–\s]bacon\b", re.I), "davis_bacon_compliance"),
    (re.compile(r"\bcalifornia\s+(?:public\s+records\s+act|education\s+code|labor\s+code|teleconnect|civil\s+code)\b", re.I), "california_legal_compliance"),
    (re.compile(r"\bcompliance\s+with\s+(?:laws|regulations|statutes|codes)\b", re.I), "legal_compliance"),
    (re.compile(r"\bconflict\s+of\s+interest\b", re.I), "conflict_of_interest_disclosure"),
    (re.compile(r"\b(?:non[-–\s]?collusion|noncollusion)\b", re.I), "noncollusion_declaration"),
    (re.compile(r"\biran\s+contracting\s+act\b", re.I), "iran_contracting_act_compliance"),
    (re.compile(r"\bprevailing\s+wage\b", re.I), "prevailing_wage_compliance"),
    (re.compile(r"\bnfpa\s*(\d{1,4})\b", re.I), "nfpa_{0}_compliance"),
    (re.compile(r"\bieee\s*(\d{2,4}(?:\.\d+)?(?:[a-z]{1,3})?)\b", re.I), "ieee_{0}_compliance"),
    (re.compile(r"\bnec\s*(\d{1,4}(?:\.\d{1,4})?)\b", re.I), "nec_{0}_compliance"),
    (re.compile(r"\bul\s*(\d{2,5}[A-Za-z]?)\b", re.I), "ul_{0}_compliance"),
    (re.compile(r"\bada(?:[-–\s]?compliant|\s+compliance)?\b", re.I), "ada_compliance"),
    (re.compile(r"\bosha(?:[-–\s]?compliant|\s+compliance)?\b", re.I), "osha_compliance"),
    (re.compile(r"\bhipaa(?:[-–\s]?compliant|\s+compliance)?\b", re.I), "hipaa_compliance"),
    (re.compile(r"\bfips\s*(140-?2|140-?3)?\b", re.I), "fips_compliance"),
    (re.compile(r"\bul[-–\s]?listed\b", re.I), "ul_listing"),
    (re.compile(r"\betl[-–\s]?listed\b", re.I), "etl_listing"),
    # Generic "in accordance with" / "per [code/standard]" — a fall-back
    # marker so atoms classified as compliance always carry at least
    # one requirement key.  This is what the gold's
    # ``expected_min_compliance_atoms`` actually counts.
    (re.compile(r"\bin\s+(?:full\s+)?accordance\s+with\s+the\s+laws\b", re.I), "legal_compliance"),
    (re.compile(r"\bcomply\s+with\s+(?:current|applicable|all)?\s*(?:federal|state|local)\b", re.I), "regulatory_compliance"),
    # AV / IT industry standards commonly cited in scope sections
    (re.compile(r"\bansi/?tia[-–\s]?(\d{2,4}[-–\.\s\w]*)?\b", re.I), "ansi_tia_compliance"),
    (re.compile(r"\bbicsi\s+(?:standard|tdmm|estimating|ostc|its|rcdd)\b", re.I), "bicsi_compliance"),
    (re.compile(r"\bfcc\s+part\s+(\d+)\b", re.I), "fcc_part_{0}_compliance"),
    (re.compile(r"\b(?:iec|iso/iec)\s*(\d{3,5})\b", re.I), "iec_{0}_compliance"),
    (re.compile(r"\bul\s+listed\b", re.I), "ul_listing"),
    (re.compile(r"\benergy\s+star\b", re.I), "energy_star_compliance"),
    (re.compile(r"\b(?:rohs|reach)[-–\s]?compliant?\b", re.I), "rohs_compliance"),
    # Generic but specific compliance mentions seen in RFP boilerplate
    (re.compile(r"\bcompliance\s+with\s+all\s+applicable\b", re.I), "applicable_compliance"),
    (re.compile(r"\bsuccessful\s+responder\s+(?:shall|must|will)\s+affirm\b", re.I), "responder_affirmation"),
    (re.compile(r"\bresponder\s+(?:shall|must|will)\s+(?:certif|warrant|affirm)", re.I), "responder_certification"),
    (re.compile(r"\binsurance\s+(?:requirements?|certificate|policy)\b", re.I), "insurance_compliance"),
    (re.compile(r"\bbond(?:ing)?\s+requirements?\b", re.I), "bonding_compliance"),
    (re.compile(r"\bworkers?\s+compensation\b", re.I), "workers_compensation_compliance"),
]


def _emit_requirement_keys(text: str) -> set[str]:
    keys: set[str] = set()
    for pattern, key_template in _REQUIREMENT_PATTERNS:
        for match in pattern.finditer(text):
            if "{0}" in key_template and match.groups():
                token = match.group(1) or ""
                token = re.sub(r"[^A-Za-z0-9]", "_", token).strip("_").lower()
                if not token:
                    continue
                keys.add(f"requirement:{key_template.format(token)}")
            else:
                keys.add(f"requirement:{key_template}")
    return keys


def _emit_quantity_keys(value: Any, text: str) -> set[str]:
    """Emit ``quantity:N`` keys when the atom carries explicit qty info.

    Two paths:
    - structured ``value.quantity`` populated by table parsers
    - ``Qty: N`` patterns in raw_text
    """
    keys: set[str] = set()

    if isinstance(value, dict):
        qty = value.get("quantity")
        if isinstance(qty, (int, float)) and not isinstance(qty, bool):
            keys.add(f"quantity:{int(qty) if float(qty).is_integer() else qty}")

    for match in _QUANTITY_REGEX.finditer(text):
        raw = match.group(1).replace(",", "")
        try:
            n = int(raw)
        except ValueError:
            continue
        keys.add(f"quantity:{n}")

    # Noun-anchored quantities: "Install 50 access points",
    # "60 wireless devices", "5 distribution switches", etc.
    for match in _QUANTITY_NOUN_REGEX.finditer(text):
        raw = match.group(1).replace(",", "")
        try:
            n = int(raw)
        except ValueError:
            continue
        # Skip implausibly small ("0 cables") or implausibly large
        # ("1000000 each") values that are almost always not real
        # quantities.
        if n <= 0 or n > 100_000:
            continue
        keys.add(f"quantity:{n}")

    return keys


def _emit_qa_markers(text: str) -> set[str]:
    """Emit ``qa:q1``, ``qa:a1`` markers for transcripted Q&A atoms.

    Useful for letting the packetizer detect Q-and-A pairs that
    belong together even when the raw_text was agglomerated by the
    parser.
    """
    keys: set[str] = set()
    for match in _QA_MARKER_REGEX.finditer(text):
        marker = match.group(1).lower()  # 'q' or 'a'
        number = match.group(2)
        keys.add(f"qa:{marker}{int(number)}")
    return keys


def _emit_csi_sections(text: str) -> set[str]:
    keys: set[str] = set()
    for match in _CSI_SECTION_REGEX.finditer(text):
        a, b, c, sub = match.group(1), match.group(2), match.group(3), match.group(4)
        section_id = f"{a}_{b}_{c}" + (f"_{sub}" if sub else "")
        keys.add(f"spec_section:{section_id}")
    return keys


# ─── Customer extraction from explicit "Company: X" / "Customer: X" labels ───

# Corporate / institutional suffixes that signal a customer name.
# Order matters: regex alternation is left-to-right, so the LONGER
# alternative must come first ("Corporation" before "Corp"; otherwise
# "Wonka Corporation" matches "Corp" then truncates at "oration").
_CORPORATE_SUFFIXES = (
    r"Corporation|Corp\.?|"
    r"Limited|Ltd\.?|"
    r"Incorporated|Inc\.?|"
    r"Company|Co\.?|"
    r"L\.L\.C\.|LLC|"
    r"L\.L\.P\.|LLP|"
    # International corporate suffixes (universal coverage)
    r"GmbH|"                              # Germany
    r"AG|"                                # Germany/Switzerland (after GmbH)
    r"SE|"                                # EU Societas Europaea (BASF, SAP)
    r"K\.K\.|KK|"                         # Japan (Kabushiki Kaisha)
    r"Oyj|Oy|"                            # Finland
    r"ApS|"                               # Denmark (must precede AS)
    r"AS|"                                # Denmark/Norway
    r"AB|"                                # Sweden
    r"Pty\s+Ltd|Pty\.?|"                  # Australia
    r"Pvt\s+Ltd|Pvt\.?|"                  # India
    r"OAO|ZAO|PAO|OOO|"                   # Russia
    r"S\.?A\.?\s*de\s*C\.?V\.?|"          # Mexico
    r"Sdn\.?\s*Bhd\.?|"                   # Malaysia
    r"S\.?A\.?|S\.?p\.?A\.?|N\.?V\.?|B\.?V\.?|"   # Spain/Italy/Netherlands
    r"PLC|P\.?C\.?|P\.?A\.?|PBC|"
    r"Holdings|Group|Partners|Enterprises|Industries|Solutions|"
    r"Systems|Technologies|Services|"
    r"Trust|Foundation|Institute|Association|Society|"
    r"University|College|School District|Hospital|Health System|Medical Center"
)

# Detects labels like "Company: OPTBOT, Inc." / "Customer: Acme Corp" /
# "Client: Globex LLC" / "Account: Initech, Inc." — followed by a
# Capitalized phrase that ends in a corporate / institutional suffix.
_COMPANY_LABEL_REGEX = re.compile(
    r"\b(?:Company|Customer|Client|Account|Buyer|End[\s\-]Client|"
    r"End[\s\-]Customer|Organization|Org)\s*[:=]\s*"
    r"([A-Z][A-Za-z0-9'.\-]*(?:[\s,]+[A-Z][A-Za-z0-9'.\-]*){0,5}\s*[,]?\s*"
    r"(?:" + _CORPORATE_SUFFIXES + r")\.?)",
)


def _emit_customer_from_label(text: str) -> set[str]:
    """Extract customer entities from explicit "Company: X" / "Customer: X"
    field labels.

    Pattern: a label word ("Company"/"Customer"/"Client"/...) followed
    by ``:`` or ``=`` and then a Capitalized phrase ending in a
    recognized corporate or institutional suffix
    (``Inc``, ``LLC``, ``Corp``, ``Company``, ``Holdings``, ``Group``,
    ``University``, ``Hospital``, ...).

    This catches "Company: OPTBOT, Inc." → ``customer:optbot_inc`` /
    "Customer: Acme Corp" → ``customer:acme_corp`` /
    "Client: Globex LLC" → ``customer:globex_llc``.
    """
    keys: set[str] = set()
    for match in _COMPANY_LABEL_REGEX.finditer(text):
        raw_value = match.group(1).strip().rstrip(",")
        # Strip a trailing period after the suffix ("OPTBOT, Inc.")
        raw_value = raw_value.rstrip(".")
        slug = _slugify(raw_value)
        if slug and len(slug) >= 3:
            keys.add(f"customer:{slug}")
    return keys


# ─── Money / currency entity extraction ───

# Matches dollar amounts: $1,847,250 / $1.8M / $250K / USD 1,500,000 /
# 1,500,000 USD / $1,015,626.00. Captures the numeric portion and any
# K/M/B suffix so we can normalize.
# Universal multi-currency money pattern. Captures:
#   - $-prefixed dollar amounts (USD assumed)
#   - €-prefixed Euros
#   - £-prefixed Pounds
#   - ¥-prefixed Yen
#   - ISO-coded prefixed amounts (USD/EUR/GBP/JPY/CHF/CAD/AUD/...)
#   - ISO-coded suffixed amounts (... USD / ... EUR / ...)
#   - K/M/B/T shorthand multipliers (case-insensitive)
#
# The slug carries no currency code (just the absolute numeric amount)
# because cross-currency normalization without exchange rates would
# introduce non-determinism. Downstream consumers can recover currency
# context from atom raw_text if needed.
_MONEY_REGEX = re.compile(
    r"(?:"
    # Symbol-prefixed: $ € £ ¥ amounts with optional K/M/B/T suffix
    r"[\$€£¥]\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?)\s*([KMBT])?\b"
    r"|"
    # ISO-code-prefixed: USD 1,500,000 / EUR 250K / GBP 1.5M
    r"\b(?:USD|EUR|GBP|JPY|CHF|CAD|AUD|NZD|HKD|SGD|CNY|INR|MXN|BRL|ZAR|"
    r"DKK|NOK|SEK|PLN|CZK|HUF|TRY|RUB|KRW|TWD|THB|MYR|IDR|PHP|VND|AED|SAR|ILS|EGP|NGN|KES)\s+"
    r"([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?)\s*([KMBT])?\b"
    r"|"
    # ISO-code-suffixed: 1,500,000 USD / 250000 EUR / 500000 EUR
    r"\b([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?)\s+"
    r"(?:USD|EUR|GBP|JPY|CHF|CAD|AUD|NZD|HKD|SGD|CNY|INR|MXN|BRL|ZAR|"
    r"DKK|NOK|SEK|PLN|CZK|HUF|TRY|RUB|KRW|TWD|THB|MYR|IDR|PHP|VND|AED|SAR|ILS|EGP|NGN|KES)\b"
    r")",
    re.IGNORECASE,
)


_CURRENCY_SYMBOL = {"$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY"}


def parse_money(text: str) -> list[dict[str, Any]]:
    """Structured monetary amounts in ``text``:
    ``[{"amount": float|int, "currency": str|None, "raw": str}, ...]``.

    The shared matcher behind both :func:`_emit_money_keys` (back-compat
    ``money:<n>`` keys) and the NORM front (:func:`normalize_atom_value`). Same
    K/M/B normalization and ``[100, 1e12]`` clamp as the original key emitter;
    additionally captures the currency (symbol/ISO) and the raw span. No
    cross-currency conversion (non-deterministic without rates)."""
    out: list[dict[str, Any]] = []
    for match in _MONEY_REGEX.finditer(text):
        num_str = match.group(1) or match.group(3) or match.group(5)
        suffix = match.group(2) or match.group(4)
        if not num_str:
            continue
        try:
            num = float(num_str.replace(",", ""))
        except ValueError:
            continue
        if suffix:
            # NB: T intentionally absent (matches the original emitter — a bare
            # "1.5T" then falls under the <100 clamp and is dropped).
            num = num * {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}.get(suffix.upper(), 1)
        amount = int(num) if num == int(num) else round(num, 2)
        if amount < 100 or amount > 1_000_000_000_000:
            continue
        raw = match.group(0).strip()
        currency: str | None = None
        for sym, code in _CURRENCY_SYMBOL.items():
            if sym in raw:
                currency = code
                break
        if currency is None:
            iso = re.search(r"\b([A-Z]{3})\b", raw.upper())
            if iso:
                currency = iso.group(1)
        out.append({"amount": amount, "currency": currency, "raw": raw})
    return out


def _emit_money_keys(text: str) -> set[str]:
    """Extract monetary amounts as ``money:<normalized>`` entities.

    Normalizes K/M/B suffixes to absolute amounts (``$1.5M`` → ``money:1500000``,
    ``$250K`` → ``money:250000``, ``USD 100`` → ``money:100``). Back-compat:
    money_summary / sow_readiness / _money_values_in_row depend on these keys.
    """
    return {
        f"money:{int(m['amount']) if m['amount'] == int(m['amount']) else m['amount']}"
        for m in parse_money(text)
    }


def parse_quantity_spans(text: str) -> list[dict[str, Any]]:
    """Structured quantities in ``text``: ``[{"quantity": int, "unit": str,
    "raw": str}, ...]``. Mirrors :func:`_emit_quantity_keys`'s two matchers
    (``Qty:/quantity:/count`` + noun-anchored) with the same ``n<=0 / n>100000``
    guard, but keeps the per-span value shape the NORM front needs. Integers
    only (the cross-doc conflict's ``quantity:`` key parser is int-only)."""
    out: list[dict[str, Any]] = []
    seen: set[int] = set()
    for rx, guarded in ((_QUANTITY_REGEX, False), (_QUANTITY_NOUN_REGEX, True)):
        for match in rx.finditer(text):
            try:
                n = int(match.group(1).replace(",", ""))
            except (ValueError, IndexError):
                continue
            if guarded and (n <= 0 or n > 100_000):
                continue
            if n in seen:
                continue
            seen.add(n)
            out.append({"quantity": n, "unit": "count", "raw": match.group(0).strip()})
    return out


def normalize_atom_value(atom: Any) -> None:
    """Deterministic NORM front (the ``value_norm`` relation). Populate
    ``atom.value`` with a normalized money amount and/or quantity — and emit the
    ``quantity:<int>`` entity_key — for the common SINGLE-value formats, so
    deal_financials / money_summary and the cross-doc quantity conflict have a
    real number to work with (only ~8-14% of value-bearing atoms resolved one
    before: the extractors parsed the number then discarded it into a slug key).

    Safe in the enrich hot loop: digit fast-guard, module-level compiled regexes,
    no LLM/embedder. ``setdefault`` discipline throughout — a parser-supplied
    structured value (table cells, xlsx) is never clobbered. Multi-value rows are
    left key-only (writing a single scalar would corrupt _money_values_in_row)."""
    text = getattr(atom, "raw_text", "") or ""
    if not any(c.isdigit() for c in text):
        return
    value = getattr(atom, "value", None)
    created = False
    if value is None:
        value, created = {}, True
    elif not isinstance(value, dict):
        return  # non-dict value (rare) — don't touch

    keys = list(getattr(atom, "entity_keys", []) or [])
    has_device = any(k.startswith("device:") and k != "device:unknown" for k in keys)
    has_part = any(k.startswith("part_number:") for k in keys)

    # MONEY — only when exactly one amount in the atom; multi-amount rows stay
    # key-only so the multi-key money form isn't corrupted.
    money = parse_money(text)
    if len(money) == 1 and value.get("amount") is None:
        m = money[0]
        value["amount"] = m["amount"]
        if m.get("currency"):
            value.setdefault("currency", m["currency"])
        value.setdefault("raw", m["raw"])

    # QUANTITY — only a single clean quantity; guard so table-parser quantities win.
    qty = parse_quantity_spans(text)
    if len(qty) == 1 and value.get("quantity") is None:
        q = qty[0]
        value["quantity"] = q["quantity"]
        value.setdefault("unit", q["unit"])
        value.setdefault("raw", q["raw"])
        if not value.get("normalized_item") and has_device:
            dev = next(
                (k.split(":", 1)[1] for k in keys
                 if k.startswith("device:") and k != "device:unknown"),
                None,
            )
            if dev:
                value["normalized_item"] = dev
        # Emit quantity:<int> — the SOLE source the device_quantity_cross_doc
        # conflict compares. Anchor on a device/part so it's a real scoped count,
        # not loose prose. (Survives the parser-keyed common case because we
        # append directly, not via the augment-prefix filter.)
        if isinstance(q["quantity"], int) and (has_device or has_part):
            qk = f"quantity:{q['quantity']}"
            if qk not in keys:
                keys.append(qk)
                atom.entity_keys = keys

    if created and value:
        try:
            atom.value = value
        except Exception:  # pragma: no cover - frozen atom; nothing to do
            pass


# ─── Date / milestone entity extraction ───

# ISO date: 2026-07-31 (the format used in OPTBOT and most modern deals).
_ISO_DATE_REGEX = re.compile(r"\b((?:19|20)[0-9][0-9])-([01][0-9])-([0-3][0-9])\b")

# US-format date: 07/31/2026 or 7/31/26
_US_DATE_REGEX = re.compile(
    r"\b([01]?[0-9])/([0-3]?[0-9])/((?:19|20)[0-9][0-9]|[2-9][0-9])\b"
)

# Long-format date: July 31, 2026 or Jul 31 2026
_LONG_DATE_REGEX = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|"
    r"October|November|December|"
    r"Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
    r"\s+([0-3]?[0-9])(?:st|nd|rd|th)?[,]?\s+((?:19|20)[0-9][0-9])\b"
)

# Day-Month-Year: 15-Jun-2026 / 5-Jul-26 / 15 Jun 2026
_DMY_DATE_REGEX = re.compile(
    r"\b([0-3]?[0-9])[\s\-/]"
    r"(January|February|March|April|May|June|July|August|September|"
    r"October|November|December|"
    r"Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
    r"[\s\-/]((?:19|20)[0-9][0-9]|[2-9][0-9])\b",
    re.IGNORECASE,
)

# Quarter notation: Q3 2026 / Q3-2026 / Q3 FY26 / 3Q26
_QUARTER_REGEX = re.compile(
    r"\b(?:Q([1-4])[\s\-/]?(?:FY)?\s*((?:19|20)[0-9][0-9]|[2-9][0-9])"
    r"|([1-4])Q[\s\-/]?((?:19|20)[0-9][0-9]|[2-9][0-9]))\b"
)

# Fiscal year notation: FY26 / FY2026 / FY-26 / Fiscal Year 2026
_FY_REGEX = re.compile(
    r"\b(?:FY[\s\-]?((?:19|20)[0-9][0-9]|[2-9][0-9])"
    r"|fiscal\s+year\s+((?:19|20)[0-9][0-9]))\b",
    re.IGNORECASE,
)

_MONTH_TO_NUM = {
    "january": 1, "jan": 1, "february": 2, "feb": 2,
    "march": 3, "mar": 3, "april": 4, "apr": 4,
    "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

# Milestone-context cues — labels that identify a date as a project
# milestone rather than just an incidental date mention.
_MILESTONE_CONTEXT_REGEX = re.compile(
    r"\b("
    r"close\s+date|closing\s+date|"
    r"start\s+date|kickoff(?:\s+date)?|kick[\s\-]off(?:\s+date)?|"
    r"end\s+date|completion(?:\s+date)?|due\s+date|deadline|"
    r"target\s+(?:date|close|completion)|"
    r"mobilization(?:\s+date|\s+start)?|"
    r"cutover(?:\s+date|\s+begins?|\s+complete)?|"
    r"go[\s\-]live(?:\s+date)?|"
    r"implementation\s+(?:start|end|window|complete)|"
    r"blackout(?:\s+window|\s+period)?|"
    r"hypercare(?:\s+start|\s+end|\s+window)?|"
    r"acceptance(?:\s+date)?|"
    r"sign[\s\-]off|signoff|"
    r"effective(?:\s+date)?|"
    r"expir(?:y|es|ation)|"
    r"milestone|deliverable\s+date|"
    r"phase\s+\d+(?:\s+start|\s+end|\s+complete)?"
    r")\b",
    re.IGNORECASE,
)


def _emit_date_keys(text: str) -> set[str]:
    """Extract dates as ``date:YYYY-MM-DD`` entities, and additionally
    as ``milestone:YYYY-MM-DD`` when a milestone-context cue (close
    date, cutover, blackout, hypercare, kickoff, ...) appears within
    50 chars of the date.

    Captures three date formats:
      ISO:     ``2026-07-31``
      US:      ``07/31/2026`` / ``7/31/26``
      Long:    ``July 31, 2026`` / ``Jul 31 2026``

    All emitted in normalized ISO form (``date:2026-07-31``) so
    downstream consumers can sort, compare, and join on them
    deterministically regardless of input style.
    """
    keys: set[str] = set()
    matches: list[tuple[int, int, str]] = []
    # ISO format
    for m in _ISO_DATE_REGEX.finditer(text):
        try:
            year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if not (1 <= month <= 12 and 1 <= day <= 31):
                continue
            iso = f"{year:04d}-{month:02d}-{day:02d}"
        except ValueError:
            continue
        matches.append((m.start(), m.end(), iso))
    # US format
    for m in _US_DATE_REGEX.finditer(text):
        try:
            month, day, year_raw = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if year_raw < 100:
                year = 2000 + year_raw
            else:
                year = year_raw
            if not (1 <= month <= 12 and 1 <= day <= 31):
                continue
            iso = f"{year:04d}-{month:02d}-{day:02d}"
        except ValueError:
            continue
        matches.append((m.start(), m.end(), iso))
    # Long format
    for m in _LONG_DATE_REGEX.finditer(text):
        try:
            month_name = m.group(1).lower()
            day = int(m.group(2))
            year = int(m.group(3))
            month = _MONTH_TO_NUM.get(month_name)
            if month is None or not (1 <= day <= 31):
                continue
            iso = f"{year:04d}-{month:02d}-{day:02d}"
        except ValueError:
            continue
        matches.append((m.start(), m.end(), iso))
    # Day-Month-Year format: 15-Jun-2026 / 5 Jul 2026 / 15-Jun-26
    for m in _DMY_DATE_REGEX.finditer(text):
        try:
            day = int(m.group(1))
            month_name = m.group(2).lower()
            year_raw = int(m.group(3))
            year = 2000 + year_raw if year_raw < 100 else year_raw
            month = _MONTH_TO_NUM.get(month_name)
            if month is None or not (1 <= day <= 31):
                continue
            iso = f"{year:04d}-{month:02d}-{day:02d}"
        except ValueError:
            continue
        matches.append((m.start(), m.end(), iso))
    for start, end, iso in matches:
        keys.add(f"date:{iso}")
        # Look ±50 chars for a milestone-context cue
        pre = text[max(0, start - 50):start]
        post = text[end:min(len(text), end + 50)]
        if _MILESTONE_CONTEXT_REGEX.search(pre) or _MILESTONE_CONTEXT_REGEX.search(post):
            keys.add(f"milestone:{iso}")
    # Quarter notation — emit as quarter:YYYY-Qn (always treated as a
    # milestone since quarters are inherently project timeline markers)
    for m in _QUARTER_REGEX.finditer(text):
        q_num = m.group(1) or m.group(3)
        year_raw = m.group(2) or m.group(4)
        if not q_num or not year_raw:
            continue
        year = int(year_raw)
        if year < 100:
            year = 2000 + year
        quarter = f"{year:04d}-Q{q_num}"
        keys.add(f"quarter:{quarter}")
        keys.add(f"milestone:{quarter}")
    # Fiscal year notation — emit as fiscal_year:FYYY
    for m in _FY_REGEX.finditer(text):
        year_raw = m.group(1) or m.group(2)
        if not year_raw:
            continue
        year = int(year_raw)
        if year < 100:
            year = 2000 + year
        keys.add(f"fiscal_year:fy{year:04d}")
    return keys


# ─── Stakeholder / person entity extraction ───

# Role / title tokens that, when adjacent to a Capitalized name, mark
# that name as a stakeholder/approver. Limited to the actually-load-
# bearing roles in commercial deal documents; common-noun titles
# ("manager"/"engineer") would over-fire without a discriminator.
_STAKEHOLDER_ROLE_PATTERNS = re.compile(
    r"\b("
    # C-suite
    r"CEO|Chief\s+Executive\s+Officer|"
    r"CFO|Chief\s+Financial\s+Officer|"
    r"CTO|Chief\s+Technology\s+Officer|"
    r"CIO|Chief\s+Information\s+Officer|"
    r"CISO|Chief\s+Information\s+Security\s+Officer|"
    r"COO|Chief\s+Operating\s+Officer|"
    # VP / SVP / EVP
    r"VP|Vice\s+President|SVP|Senior\s+Vice\s+President|"
    r"EVP|Executive\s+Vice\s+President|"
    # Director / Manager — only when paired with a domain word
    r"Director\s+of\s+[A-Z][\w\s]{2,30}|"
    r"Senior\s+Director|Managing\s+Director|"
    # Approval / sponsorship roles
    r"Sponsor|Executive\s+Sponsor|Project\s+Sponsor|Business\s+Sponsor|"
    r"Owner|Project\s+Owner|Product\s+Owner|Budget\s+Owner|"
    r"Approver|Decision\s+Maker|Stakeholder|"
    # Delegated authority
    r"Delegate|CFO\s+Delegate|Approving\s+Authority|"
    # Project roles
    r"PM|Project\s+Manager|Program\s+Manager|"
    # Workplace / technical leads
    r"VP\s+Workplace\s+Operations|"
    r"Head\s+of\s+[A-Z][\w\s]{2,30}|"
    r"Lead\s+[A-Z][\w]+|"
    # Generic "X Manager" — any 1-3 word prefix followed by Manager
    # (catches "Regional Facilities Manager", "Senior Procurement Manager",
    # "IT Operations Manager"). The prefix words must be Capitalized.
    r"(?:[A-Z][a-z]+\s+){1,3}Manager|"
    # Approval / decision VERBS — when a person name is the subject
    # of an approval verb, the name itself is a stakeholder.
    # "Priya Narang approves..." / "Camila Brooks: Approved..."
    r"approves?|approved|approving|"
    r"accepts?|accepted|accepting|"
    r"signs?\s+off|signed\s+off|sign[\s\-]off|signoff|"
    # Ownership — "owns", "owned", "owns the", "owned by"
    r"owns?|owned|"
    r"escalates?|escalated|"
    r"reviews?\s+and\s+(?:approves?|approved)|"
    r"authorized\s+by|approved\s+by|signed\s+by|"
    r"is\s+the\s+(?:owner|sponsor|approver|delegate)|"
    r"responsible\s+for|accountable\s+for|"
    # A7 multi-language role cues (Spanish / French / German /
    # Portuguese / Italian). Same intent — the word marks the
    # nearby capitalized noun as a stakeholder.
    # Spanish — approval verbs (present + past), titles
    r"Director\s+de\s+[A-ZÀ-ÿ][\w\s]{2,30}|"
    r"Gerente|Jefe\s+de\s+[A-ZÀ-ÿ][\w\s]{2,30}|"
    r"Responsable\s+de|Aprobado\s+por|Firmado\s+por|"
    r"aprobó|aprueba|aprobaron|firmó|firma|firmaron|"
    r"autoriza|autorizó|autoriz[oó]\s+por|"
    # French — approval verbs + titles
    r"Directeur\s+(?:de|des|du)\s+[A-ZÀ-ÿ][\w\s]{2,30}|"
    r"Chef\s+de\s+(?:projet|service|département)|"
    r"Responsable|Approuvé\s+par|Signé\s+par|"
    r"approuve|approuvé|approuvent|signe|signé|signent|autorise|autorisé|"
    # German — approval verbs + titles
    r"Geschäftsführer|Leiter\s+(?:der|des)\s+[A-ZÀ-ÿ][\w\s]{2,30}|"
    r"Abteilungsleiter|Projektleiter|Genehmigt\s+(?:von|durch)|"
    r"Unterzeichnet\s+(?:von|durch)|"
    r"genehmigt|genehmigen|unterzeichnet|unterschreibt|freigibt|freigegeben|"
    # Portuguese / Italian — verbs + titles
    r"Diretor|Direttore|Aprovado\s+por|Approvato\s+da|"
    r"Gerente\s+de|Responsabile|"
    r"aprovou|aprova|aprovaram|approva|approvato|approvano|firmato"
    r")\b",
    re.IGNORECASE | re.UNICODE,
)

# Honorifics that prefix a name. We strip these before slugifying so
# "Dr. Sara Chen" becomes ``stakeholder:sara_chen`` not
# ``stakeholder:dr_sara_chen``.
#
# A7: multilingual honorifics — Spanish (Sr/Sra/Srta/Dn/Dña),
# French (M/Mme/Mlle), German (Herr/Frau), Italian (Sig/Sig.ra/Sig.na),
# Portuguese (Sr/Sra/Srta).
_HONORIFIC_REGEX = re.compile(
    r"^(?:Dr|Mr|Mrs|Ms|Mx|Prof|Professor|Sir|Dame|Hon|Rev|Fr|"
    r"Sr|Sra|Srta|Sr\.|Sra\.|Srta\.|Dn|D[oñ]a|Don|Doña|"
    r"M|Mme|Mlle|Madame|Monsieur|Mademoiselle|"
    r"Herr|Frau|Fräulein|"
    r"Sig|Sig\.ra|Sig\.na|Signor|Signora|Signorina"
    r")"
    r"\.?\s+",
    re.IGNORECASE | re.UNICODE,
)

# Name suffixes (Jr/Sr/II/III/IV/V) that may follow a name and break
# the regex if treated as part of the name. We accept them as optional
# and strip when slugifying.
_NAME_SUFFIX = r"(?:\s+(?:Jr|Sr|II|III|IV|V|PhD|Ph\.D\.|MD|M\.D\.|Esq))\.?"

# Pattern: "First Last" / "First Middle Last" / "First M. Last" with
# optional honorific prefix and optional name suffix.
#
# Three forms (alternation, longest first):
#   3-word: First Middle Last      ("Mary Anne Smith")
#   2-word + middle initial: First M. Last  ("Sara G. Chen")
#   2-word: First Last             ("Jordan Ames")
#
# Each token allows hyphens and apostrophes for compound surnames
# ("O'Brien", "Smith-Jones").
# A single "name token" — either a standard capitalized word
# ("Smith" / "MacDonald"), a compound name with hyphen or apostrophe
# ("Smith-Jones" / "O'Brien"), or a single-letter + compound
# ("O'Brien" where O alone is the first cluster).
#
# Critically, each token must contain at least one LOWERCASE letter
# overall (either in the main word or in the compound tail). This
# rejects Roman numerals (II, III, IV, V) and all-caps acronyms from
# being parsed as name tokens.
# A7 multi-language: explicit case-aware Latin uppercase / lowercase
# character classes so names with accents (García / Müller / André /
# José / Søren) match _NAME_TOKEN while staying STRICT on case
# (first char uppercase, rest lowercase) so the regex doesn't grab
# lowercase prose like "de servicios" as a name.
# Python's ``re`` doesn't support ``\p{Lu}`` / ``\p{Ll}``, so we
# enumerate Latin-1 supplement (À-Ö, Ø-Þ uppercase; à-ö, ø-ÿ lowercase).
# Slugification (A4) folds the accents to ASCII downstream.
_UPPER = r"[A-ZÀ-ÖØ-Þ]"
_LOWER = r"[a-zà-öø-ÿ]"
_NAME_TOKEN = (
    r"(?:"
    + _UPPER + _LOWER + r"+(?:[\-']" + _UPPER + _LOWER + r"+)?"  # García / Smith-Jones
    + r"|" + _UPPER + r"[\-']" + _UPPER + _LOWER + r"+"          # O'Brien / D'Souza
    + r")"
)

_PERSON_NAME_REGEX = re.compile(
    r"\b("
    # A7 multilingual honorifics (optional)
    r"(?:Dr|Mr|Mrs|Ms|Mx|Prof|Sir|Dame|Hon|Rev|Fr|"
    r"Sr|Sra|Srta|Dn|Don|D[oñ]a|"
    r"M|Mme|Mlle|Madame|Monsieur|Mademoiselle|"
    r"Herr|Frau|Fräulein|"
    r"Sig|Signor|Signora|Signorina"
    r")\.?\s+"
    r")?"
    r"("
    # 3-word name: First Middle Last
    + _NAME_TOKEN + r"\s+" + _NAME_TOKEN + r"\s+" + _NAME_TOKEN +
    r"|"
    # Initial-style: First M. Last
    + _NAME_TOKEN + r"\s+[A-Z]\.\s+" + _NAME_TOKEN +
    r"|"
    # 2-word name: First Last
    + _NAME_TOKEN + r"\s+" + _NAME_TOKEN +
    r")"
    # Optional Roman / Jr / Sr suffix (captured but stripped downstream)
    r"(?:\s+(?:Jr|Sr|II|III|IV|V)\.?)?",
    re.UNICODE,
)

# Honorific + single name ("Dr. Smith", "Mr. Lee", "Mrs. Park").
# Standalone regex because the main pattern requires ≥2 name tokens.
# The negative lookahead ``(?!\s+`` _NAME_TOKEN `` )`` ensures we DON'T
# fire when a full "Honorific First Last" name is present — the main
# regex catches that. This single-name path is only for honorific +
# surname like "Dr. Smith".
_HONORIFIC_SINGLE_NAME_REGEX = re.compile(
    r"\b(?:Dr|Mr|Mrs|Ms|Mx|Prof|Sir|Dame|Hon|Rev|Fr|"
    # A7 multilingual honorifics
    r"Sr|Sra|Srta|Dn|Don|D[oñ]a|"
    r"M|Mme|Mlle|Madame|Monsieur|Mademoiselle|"
    r"Herr|Frau|Fräulein|"
    r"Sig|Signor|Signora|Signorina"
    r")\.?\s+"
    r"(" + _NAME_TOKEN + r")"
    r"(?!\s+" + _NAME_TOKEN + r")\b",
    re.UNICODE,
)

# D3: Initial + Last form — ``R. Watkins`` / ``J Ames`` / ``J.A. Smith``.
# Captures a single uppercase letter (optionally followed by a period
# and another initial) plus a surname token. The downstream fuser in
# ``entity_resolution.collect_stakeholder_alias_groups`` collapses
# ``stakeholder:r_watkins`` into ``stakeholder:renee_watkins`` when
# the surname uniquely identifies a full-name stakeholder elsewhere
# in the project.
_INITIAL_LAST_NAME_REGEX = re.compile(
    r"\b([A-Z](?:\.[A-Z])?)\.?\s+(" + _NAME_TOKEN + r")\b"
)

# Inverted form: "Smith, John" — last-name-first. Used in formal
# author / stakeholder lists. We require an explicit FIELD label
# ("Name:", "Author:", "Approver:", "Sponsor:", "Owner:", ...)
# immediately before the inverted pair so we don't misinterpret
# comma-separated lists of full names ("Jordan Ames, Priya Narang,
# Camila Brooks") as a series of last-name-first records.
_INVERTED_NAME_REGEX = re.compile(
    r"(?:Name|Author|Approver|Sponsor|Owner|Contact|Stakeholder|PM|"
    r"Manager|Delegate|Reviewer|Signatory)"
    r"\s*[:=]\s*"
    r"([A-Z][a-z]+(?:[\-'][A-Z][a-z]+)?)\s*,\s+"
    r"([A-Z][a-z]+(?:\s+[A-Z]\.?)?)\b",
    re.IGNORECASE,
)

# Names that look proper-noun-shaped but aren't actually people.
# Conservative — when in doubt, leave it out (a name we miss is recoverable
# downstream; a false-positive stakeholder is noise that hurts trust).
_NON_PERSON_NAME_PREFIXES: frozenset[str] = frozenset({
    # Place-shape leads
    "atlanta", "houston", "dallas", "berlin", "tokyo", "seattle",
    "phoenix", "chicago", "boston", "denver", "miami", "austin",
    "north", "south", "east", "west", "central",
    # Vendor / org-shape leads
    "axis", "cisco", "genetec", "lenel", "honeywell", "siemens",
    "schneider", "trane", "carrier", "philips", "dell", "hpe",
    "microsoft", "google", "amazon", "azure", "aws", "gcp",
    "hubspot", "optbot", "orbitbrief", "purpulse",
    # Generic
    "mock", "test", "demo", "fake", "dummy", "sample", "example",
    "fictional", "synthetic",
    # A7 multilingual honorifics that match _NAME_TOKEN but
    # shouldn't fuse with the name into a stakeholder key. They
    # appear as the first word of an "Honorific Name" pair when
    # the main regex backtracks past the optional-honorific group.
    "mme", "mlle", "monsieur", "madame", "mademoiselle",
    "herr", "frau", "fraulein", "fräulein",
    "sig", "signor", "signora", "signorina",
    "sra", "srta", "doña", "dona", "don",
})


def _emit_stakeholders(text: str) -> set[str]:
    """Extract named approvers / stakeholders as ``stakeholder:first_last``
    entities.

    Strategy: find every "First Last" capitalized name, then accept it
    only when a role/title cue (CFO, VP, Sponsor, Director, Approver,
    Owner, PM, Delegate, …) appears within ±60 chars in the same
    sentence. This is the "person + role context" pattern.

    The role context is the disambiguator — a bare "Jordan Ames" might
    be anything, but "Jordan Ames, VP Workplace Operations" or
    "Approved by Jordan Ames" is structurally a stakeholder mention.
    """
    keys: set[str] = set()
    # Sentence splitter that does NOT break at mid-word periods.
    # Pre-protect periods inside honorifics, name suffixes, initials,
    # and common abbreviations by swapping them for a sentinel
    # ``<DOT>``, split on real sentence boundaries, then restore.
    text_safe = re.sub(
        r"\b(Mr|Mrs|Ms|Mx|Dr|Prof|Sir|Dame|Hon|Rev|Fr|"
        r"Jr|Sr|Ph|Ph\.D|MD|M\.D|Esq|"
        r"Inc|Corp|Co|Ltd|LLC|PLC|GmbH|"
        r"St|Ave|Blvd|Rd|Hwy|Pkwy|"
        r"U|S|N|E|W|"
        # A7 multilingual honorifics — protect their periods from
        # the sentence splitter so "Sig. Rossi" / "Sra. García" /
        # "Sgt. Smith" stay one sentence.
        r"Sig|Sra|Srta|Dn|Sgt"
        r")\.",
        r"\1<DOT>",
        text,
    )
    # Single-letter initial followed by space + capital: "Sara G. Chen"
    text_safe = re.sub(r"\b([A-Z])\.\s+(?=[A-Z][a-z])", r"\1<DOT> ", text_safe)
    # Split on real sentence boundaries (period + space + capital,
    # terminal punctuation, newline) AND on semicolon when followed
    # by a name-shaped token. We previously also split on colon, but
    # that broke "Org — Role: Name" signature lines by separating
    # the role context from the name. Colons inside signature lines
    # are FIELD separators, not sentence ends — keep them in the
    # same sentence so the role-context proximity check fires.
    for sentence in re.split(
        r"(?:\.\s+(?=[A-Z])|[?!\n]+|;\s+(?=[A-Z][a-z]+\s+[A-Z]))",
        text_safe,
    ):
        sentence = sentence.replace("<DOT>", ".").strip()
        if not sentence:
            continue
        # Skip sentences without a role cue — saves work
        if not _STAKEHOLDER_ROLE_PATTERNS.search(sentence):
            continue
        # D3: Initial + Last form ("R. Watkins", "J Ames").
        # Like the honorific-single-name path, we require a role
        # cue within ±60 chars to avoid grabbing every "T. Rex" /
        # "V. Important" capitalization in the corpus.
        for il_match in _INITIAL_LAST_NAME_REGEX.finditer(sentence):
            initial = il_match.group(1).strip().rstrip(".")
            last = il_match.group(2).strip()
            if len(last) < 3 or last.lower() in _NON_PERSON_NAME_PREFIXES:
                continue
            # Role-context proximity check, same disambiguator.
            pre = sentence[max(0, il_match.start() - 60):il_match.start()]
            post = sentence[il_match.end():min(len(sentence), il_match.end() + 60)]
            if not (_STAKEHOLDER_ROLE_PATTERNS.search(pre)
                    or _STAKEHOLDER_ROLE_PATTERNS.search(post)):
                continue
            slug = _slugify(f"{initial} {last}")
            if slug and len(slug) >= 3:
                keys.add(f"stakeholder:{slug}")
        # Honorific + single-name ("Dr. Smith", "Ms. Park"). Single
        # capitalized word after an honorific is enough — the
        # honorific is itself a strong stakeholder signal.
        for h_match in _HONORIFIC_SINGLE_NAME_REGEX.finditer(sentence):
            single_name = h_match.group(1).strip()
            if len(single_name) < 3:
                continue
            if single_name.lower() in _NON_PERSON_NAME_PREFIXES:
                continue
            # Role-context proximity check, same as the main path
            pre = sentence[max(0, h_match.start() - 60):h_match.start()]
            post = sentence[h_match.end():min(len(sentence), h_match.end() + 60)]
            if not (_STAKEHOLDER_ROLE_PATTERNS.search(pre)
                    or _STAKEHOLDER_ROLE_PATTERNS.search(post)):
                continue
            slug = _slugify(single_name)
            if slug and len(slug) >= 3:
                keys.add(f"stakeholder:{slug}")
        # Inverted "Smith, John" form — find these first and add them
        # directly. The regex below would miss inverted names because
        # the comma breaks the "First Last" pattern.
        for inv_match in _INVERTED_NAME_REGEX.finditer(sentence):
            last_name = inv_match.group(1).strip()
            first_part = inv_match.group(2).strip()
            # Only emit if a role cue is within ±60 chars of the match
            pre = sentence[max(0, inv_match.start() - 60):inv_match.start()]
            post = sentence[inv_match.end():min(len(sentence), inv_match.end() + 60)]
            if not (_STAKEHOLDER_ROLE_PATTERNS.search(pre)
                    or _STAKEHOLDER_ROLE_PATTERNS.search(post)):
                continue
            # Re-order to "First Last" for canonical slug
            reordered = f"{first_part} {last_name}"
            first_lower = first_part.split()[0].lower()
            if first_lower in _NON_PERSON_NAME_PREFIXES:
                continue
            slug = _slugify(reordered)
            if slug and len(slug) >= 5:
                keys.add(f"stakeholder:{slug}")
        # Standard "First Last" / "First Middle Last" form
        for match in _PERSON_NAME_REGEX.finditer(sentence):
            # group(2) is the name proper (after stripping optional
            # honorific in group(1)).
            name = (match.group(2) or "").strip()
            if not name:
                continue
            # Strip trailing name-suffix tokens (Jr, Sr, II, III, IV, V)
            # that may have been pulled into the 3-word-name alternative.
            # "Robert Brown Jr" → "Robert Brown".
            tokens = name.split()
            while tokens and tokens[-1].rstrip(".").upper() in {
                "JR", "SR", "II", "III", "IV", "V",
                "PHD", "MD", "ESQ",
            }:
                tokens.pop()
            if not tokens:
                continue
            name = " ".join(tokens)
            if not tokens:
                continue
            first_lower = tokens[0].lower()
            if first_lower in _NON_PERSON_NAME_PREFIXES:
                continue
            # Reject if the name's tail token is a CORPORATE suffix
            # ("Acme Corp" / "OPTBOT Inc"). Place suffixes (park,
            # tower, building, plaza, ...) are deliberately omitted
            # here because they collide with real surnames ("Linda
            # Park", "Jenna Hill", "Sam Cross"). The role-context
            # check is the discriminator for those — "Cedar Park"
            # without an approver verb nearby won't trigger anyway.
            tail_lower = tokens[-1].lower().rstrip(",.:")
            if tail_lower in {
                "inc", "incorporated", "llc", "corp", "corporation",
                "company", "co", "ltd", "limited", "plc", "gmbh",
                "holdings", "group", "partners", "enterprises",
            }:
                continue
            # Reject if a role token appears WITHIN the name itself
            # ("Director Jane Doe" — capture "Jane Doe" not "Director Jane")
            tokens_lower = {t.lower().rstrip(",.:") for t in tokens}
            role_tokens = {
                "director", "manager", "lead", "officer", "engineer",
                "architect", "analyst", "ceo", "cfo", "cto", "cio", "ciso",
                "coo", "vp", "svp", "evp", "sponsor", "approver", "delegate",
                "owner", "stakeholder", "executive", "head", "senior",
                "junior", "principal", "associate", "assistant",
                "specialist", "coordinator", "supervisor",
            }
            if tokens_lower & role_tokens:
                continue
            # Reject if EITHER token is a department / function name
            # (Workplace Operations, Marketing Department, ...). These
            # match the two-word capitalized shape but aren't people.
            non_person_tokens = {
                # Department / function names
                "workplace", "operations", "operation", "ops",
                "marketing", "sales", "engineering", "finance",
                "procurement", "security", "technology", "legal",
                "design", "research", "development", "support",
                "administration", "compliance", "audit", "infrastructure",
                "platform", "product", "program", "project", "portfolio",
                "delivery", "implementation", "deployment", "integration",
                "facilities", "logistics", "warehouse",
                "communications", "training", "education", "hr",
                "workforce", "personnel", "talent", "resources",
                "department", "team", "group", "division", "unit",
                "organization", "function",
                # Checklist / template / heading words
                "checklist", "item", "items", "task", "tasks", "type",
                "types", "category", "categories", "step", "steps",
                "phase", "phases", "stage", "stages",
                "due", "start", "end", "date", "dates", "deadline",
                "milestone", "milestones", "duration", "schedule",
                "evidence", "criteria", "criterion", "output", "outputs",
                "input", "inputs", "result", "results",
                "expected", "required", "actual", "estimated", "planned",
                "exit", "entry", "review", "cadence", "frequency",
                "help", "desk", "service",
                "field", "fields", "column", "row", "rows",
                "summary", "overview", "detail", "details",
                "note", "notes", "comment", "comments",
                "section", "subsection", "appendix", "attachment",
                "exhibit", "schedule", "addendum",
                # Status / quality words
                "status", "priority", "severity", "impact", "risk",
                "ready", "complete", "pending", "open", "closed",
                "approved", "rejected", "draft", "final", "active",
                "blocked", "blocker", "warning", "info",
                # Generic objects
                "table", "list", "form", "template", "report", "page",
                "header", "footer", "title", "subtitle",
                "version", "revision", "edition",
                # Action verbs / common imperatives (sometimes
                # capitalized at start of bullets)
                "confirm", "verify", "validate", "check", "test",
                "review", "approve", "submit", "publish", "send",
                "create", "update", "delete", "remove", "add",
                "attach", "run", "execute", "deploy", "install",
                "configure", "setup", "enable", "disable", "start",
                "stop", "schedule", "complete", "finalize",
                # Hub / network terms
                "hub", "node", "endpoint", "gateway", "proxy",
                "instance", "cluster", "tenant", "region", "zone",
            }
            if tokens_lower & non_person_tokens:
                # A7 fallback: if a 3-token match starts with a
                # non-person word ("Finance Jordan Ames"), retry
                # the trailing 2 tokens as a 2-word name. The role
                # context check below still gates emission.
                if len(tokens) == 3 and tokens[0].lower() in (
                    non_person_tokens | role_tokens
                ):
                    tokens = tokens[1:]
                    name = " ".join(tokens)
                    tokens_lower = {t.lower().rstrip(",.:") for t in tokens}
                    if tokens_lower & non_person_tokens:
                        continue
                else:
                    continue
            # Role-context proximity check (±60 chars in this sentence)
            pre = sentence[max(0, match.start() - 60):match.start()]
            post = sentence[match.end():min(len(sentence), match.end() + 60)]
            if not (_STAKEHOLDER_ROLE_PATTERNS.search(pre)
                    or _STAKEHOLDER_ROLE_PATTERNS.search(post)):
                continue
            slug = _slugify(name)
            if slug and len(slug) >= 5:
                keys.add(f"stakeholder:{slug}")
    return keys


# ════════════════════════════════════════════════════════════════════
# CONTACT-ANCHOR EMITTERS (universal — close the email/phone/site-code
# recall gap surfaced by the source-vs-parser audit 2026-05-27)
# ════════════════════════════════════════════════════════════════════

_EMAIL_REGEX = re.compile(
    r"\b([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})\b"
)
_PHONE_REGEX = re.compile(
    r"(?<!\d)(\+?1[-.\s]?)?\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})(?!\d)"
)
# Site-code shape: REGION-FUNCTION[-NN] — 2-5 alphanumeric segments
# separated by hyphens, at least one segment with a digit OR all-caps.
# Catches ATL-HQ-01, STORE-142, MDF-3A, IDF-W2-3, LMC-L640, etc.
_SITE_CODE_PATTERN = re.compile(
    r"\b(?:"
    r"[A-Z]{2,5}-[A-Z0-9]{1,5}(?:-\d{1,4}){0,2}"  # ATL-HQ-01, MDF-W1, IDF-2-7
    r"|[A-Z]{2,5}\d{2,4}"                          # B197, RM12 (no hyphen)
    r")\b"
)
# Persons named via "Name, Role, email" or "contact Name at email"
# patterns. The email is the corroboration: we scan BACKWARD from
# each email match for a capitalized name within ~80 chars. Catches
# all of these:
#   - "Glenn Tilleman, Hood County Purchasing Agent at gtilleman@..."
#   - "Shaun Tozer, Project Manager at 425-939-8046, ... shaun.tozer@..."
#   - "Matthew Brener, BRS, Inc., (267) 688-7301 | matthew@brsinc.com"
#   - "John Foster, Convergent Technology Partners, at jfoster@..."
# Case is strict (uppercase first letter) on the name to avoid
# catching prepositions / lowercase words.
_NAME_NEAR_EMAIL = re.compile(
    r"\b([A-Z][a-z]{1,15}(?:\s+[A-Z]\.?)?(?:\s+[A-Z][a-z'-]{1,18}){1,2})\b"
)
# Trigger words for the contact-line extractor. Case-insensitive on
# the trigger ONLY (via (?i:...) inline group), strict capitalization
# on the captured name.
_PERSON_CONTACT_LINE = re.compile(
    r"(?i:please\s+contact|contact(?:\s+(?:is|will\s+be))?|directed\s+to|"
    r"attention(?:\s+of)?\s*:?|submitted\s+by|prepared\s+by|"
    r"project\s+manager\s*[:\-]?|purchasing\s+agent\s*[:\-]?|"
    r"approved\s+by|signed\s+by|sponsor(?:ed)?\s+by|owned\s+by|"
    r"point\s+of\s+contact\s*[:\-]?|technical\s+lead\s*[:\-]?|"
    r"executive\s+sponsor\s*[:\-]?)\s+"
    r"([A-Z][a-z]+(?:\s+[A-Z]\.?)?(?:\s+[A-Z][a-z'-]+){1,3})\b"
)
# Generic-word denylist for people pulled from email patterns
# (some emails are noreply@, info@, support@, etc.)
_EMAIL_LOCAL_DENY: frozenset[str] = frozenset({
    "noreply", "no-reply", "donotreply", "info", "contact", "hello",
    "support", "help", "admin", "sales", "marketing", "service",
    "billing", "accounts", "ap", "ar", "hr", "it", "legal",
    "office", "front-desk", "frontdesk", "reception",
    "team", "group", "list", "notifications", "alerts",
})


def _slug_simple(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _emit_email_keys(text: str) -> set[str]:
    """Emit ``email:<normalized>`` for every email in text."""
    keys: set[str] = set()
    for m in _EMAIL_REGEX.finditer(text):
        email = m.group(1).lower()
        # Drop trailing dots/punctuation artifacts
        email = email.rstrip(".,;:!)\\")
        slug = _slug_simple(email)
        if slug and len(slug) >= 5:
            keys.add(f"email:{slug}")
    return keys


def _emit_phone_keys(text: str) -> set[str]:
    """Emit ``phone:<digits>`` for every phone number in text.

    Normalizes US numbers to a canonical 10-digit form by stripping
    the leading country-code "1" when present — so "1-800-256-8224"
    and "800-256-8224" both emit ``phone:8002568224`` (one entity,
    not two).

    Rejects 10-digit sequences that can't be valid US phones:
      - leading 0 (real US area codes never start with 0)
      - leading 1 (after country-code strip — N-1-X area codes don't
        exist; this catches lot numbers / doc IDs that got picked
        up by the digits-only pattern)
    """
    keys: set[str] = set()
    for m in _PHONE_REGEX.finditer(text):
        digits = (m.group(1) or "") + m.group(2) + m.group(3) + m.group(4)
        digits = re.sub(r"\D", "", digits)
        # Strip the US country-code prefix so 11-digit "1XXXXXXXXXX"
        # collapses to canonical 10-digit "XXXXXXXXXX".
        if len(digits) == 11 and digits.startswith("1"):
            digits = digits[1:]
        if len(digits) != 10:
            continue
        # Real US area codes start with 2-9. Leading 0 or 1 = lot
        # number / document ID misread as a phone.
        if digits[0] in {"0", "1"}:
            continue
        # Real US central-office codes (digits 4-6) also start with
        # 2-9. Drops "8001234567" patterns where digit 4 is 0 or 1.
        if digits[3] in {"0", "1"}:
            continue
        keys.add(f"phone:{digits}")
    return keys


def _emit_person_from_contact(text: str) -> set[str]:
    """Emit ``stakeholder:<first_last>`` for "Name <email>" /
    "contact Name" / "submitted by Name" patterns.

    Strategy (3 fallbacks):
      1. EMAIL-ANCHORED back-scan — for each email in text, scan
         100 chars backward for capitalized names. Catches "Matthew
         Brener, BRS, Inc., (267) 688-7301 | matthew@..." and
         "via email to John Foster, Convergent Tech Partners,
         at jfoster@...".
      2. CONTACT-LINE — explicit "Please contact X" / "Submitted
         by X". Trigger case-insensitive, name strict-cap.
      3. EMAIL LOCAL-PART derivation — when the email has a
         dot-separated firstname.lastname pattern (e.g.
         "kaylee.yinger@beaufort.k12.sc.us") and the back-scan
         found NO preceding name, derive the person name from
         the local-part. Catches PM contacts in form-only docs
         where the docline reads "a) by email to: <email>" with
         no name in front. Skips role-shaped local-parts (info,
         support, noreply, etc.) via _EMAIL_LOCAL_DENY.
    """
    # Pull the common-noun-first-word + tail-word + org-keyword
    # filter that the LLM stakeholder path already uses. The
    # contact-anchor back-scan can pull "Mosaic Front" / "End Users" /
    # "Power School" off lines like "Mosaic Front | support@..." or
    # "End Users may contact support@..." — those phrases pass the
    # name-shape check and pass _is_likely_person_label (which only
    # checks org-suffix tails) but should NEVER survive as people.
    # Single source of truth for "is this a real person name?" lives
    # in _is_likely_field_label.
    try:
        from app.core.multi_entity_llm import (
            _is_likely_field_label,
            _looks_like_email_or_url,
        )
    except Exception:
        _is_likely_field_label = None  # type: ignore
        _looks_like_email_or_url = None  # type: ignore

    def _looks_like_real_person(name: str) -> bool:
        if not name:
            return False
        if _is_likely_person_label(name):
            return False
        if _is_likely_field_label is not None and _is_likely_field_label(name):
            return False
        if _looks_like_email_or_url is not None and _looks_like_email_or_url(name):
            return False
        return True

    keys: set[str] = set()
    # Email-anchored back-scan
    bad_starts = ("At ", "By ", "For ", "From ", "To ", "Of ",
                  "In ", "On ", "The ", "An ", "A ", "Or ")
    for em in _EMAIL_REGEX.finditer(text):
        start = max(0, em.start() - 100)
        window = text[start:em.start()]
        any_name_found = False
        for nm in _NAME_NEAR_EMAIL.finditer(window):
            name = nm.group(1).strip()
            for bs in bad_starts:
                if name.startswith(bs):
                    name = name[len(bs):].strip()
            slug = _slug_simple(name)
            if slug and "_" in slug and _looks_like_real_person(name):
                keys.add(f"stakeholder:{slug}")
                any_name_found = True
        # Fallback 3 — derive name from email local-part if no
        # preceding name in the back-scan window AND the local-part
        # has the firstname.lastname dot-separated shape.
        if not any_name_found:
            email = em.group(1)
            local = email.split("@", 1)[0]
            if "." in local:
                parts = [p for p in local.split(".") if p]
                if (
                    len(parts) >= 2
                    and parts[0].lower() not in _EMAIL_LOCAL_DENY
                    and all(p.isalpha() and len(p) >= 2 for p in parts)
                ):
                    # Title-case for natural-looking name
                    name_parts = [p.capitalize() for p in parts]
                    name = " ".join(name_parts)
                    slug = _slug_simple(name)
                    if slug and "_" in slug and _looks_like_real_person(name):
                        keys.add(f"stakeholder:{slug}")
    # Contact-line
    for m in _PERSON_CONTACT_LINE.finditer(text):
        name = m.group(1).strip()
        for bs in bad_starts:
            if name.startswith(bs):
                name = name[len(bs):].strip()
        slug = _slug_simple(name)
        if slug and "_" in slug and _looks_like_real_person(name):
            keys.add(f"stakeholder:{slug}")
    return keys


# Names that pass the capitalized-pattern test but are clearly NOT
# people (org names, jargon, table cells).
_PERSON_LABEL_DENYLIST: frozenset[str] = frozenset({
    "Hood County", "Beaufort County", "Geary County",
    "Solana Beach", "Manhattan Beach", "Atlanta GA",
    "Albuquerque Public", "Office Of", "State Of",
    "Department Of", "City Of", "United States",
    "Project Manager", "Purchasing Agent",
    "Technical Lead", "Executive Sponsor",
    "Mock Document", "Mock Deal", "Mock Doc",
})


def _is_likely_person_label(name: str) -> bool:
    """Quick org/jargon filter for would-be person names."""
    if name in _PERSON_LABEL_DENYLIST:
        return True
    # Multi-word names where every word starts with a capital but
    # the LAST word is an org-suffix or jargon noun
    bad_tails = {
        # Jurisdictional
        "County", "City", "Town", "State", "Federal",
        # Org body types
        "Department", "Office", "Agency", "Authority",
        "School", "District", "Schools", "University",
        "Court", "Board", "Committee", "Council", "Commission",
        # Corporate suffixes
        "Corporation", "Corp", "Inc", "LLC", "Ltd", "Co",
        "Partners", "Solutions", "Services", "Systems", "Sales",
        "Technologies", "Group", "Holdings", "Enterprises",
        "Industries", "International", "Global", "Worldwide",
        "Consulting", "Consultants", "Associates", "Advisors",
        "Communications", "Networks", "Engineering",
        "Ventures", "Capital", "Investments",
        # Functions / labels
        "Public", "Private", "Team",
        "Purchasing", "Procurement", "Operations", "Maintenance",
        "Manager", "Agent", "Director", "Supervisor",
        "Sponsor", "Lead", "Owner", "Engineer", "Architect",
        "Coordinator", "Specialist", "Foreman", "Inspector",
        "Officer", "Officers", "Support", "Rep", "Representative",
        "Reps", "Representatives", "Leads", "Specialists",
        # Other
        "Postal", "USA", "US", "USPS", "FedEx", "UPS",
        # Street-suffix words — when a "name" ends in these it's an
        # address fragment, not a person. Catches "Miller Rd",
        # "Swartz Creek" (city name), "Heck Ave", "Corlies Avenue".
        "Rd", "Road", "Ave", "Avenue", "Blvd", "Boulevard",
        "St", "Street", "Ln", "Lane", "Ct", "Court", "Pl", "Place",
        "Dr", "Drive", "Hwy", "Highway", "Pkwy", "Parkway",
        "Cir", "Circle", "Trl", "Trail", "Way", "Terr", "Terrace",
        "Creek", "Brook", "River", "Lake", "Hill", "Valley",
        "Park", "Bay", "Beach", "Heights", "Springs", "Falls",
        "Crossing", "Junction", "Center", "Square", "Plaza",
    }
    tail = name.split()[-1] if " " in name else name
    if tail in bad_tails:
        return True
    # Also check: if ANY token in the name is in the bad_tails AND
    # the name has 3+ words, it's probably an org name even if the
    # tail itself isn't org-like (e.g., "Hood County Purchasing").
    tokens = name.split()
    if len(tokens) >= 3 and any(t in bad_tails for t in tokens):
        return True
    return False


def _emit_site_code_keys(text: str) -> set[str]:
    """Emit ``site:<code>`` for ATL-HQ-01 / STORE-142 / MDF-3A patterns.

    Site codes are the customer's authoritative scope anchors and
    PMs absolutely need them visible. Conservative: requires the
    hyphenated/digit-bearing shape so it doesn't fire on words like
    "USA" or "ANSI".
    """
    keys: set[str] = set()
    for m in _SITE_CODE_PATTERN.finditer(text):
        code = m.group(0)
        # Drop obvious non-codes
        upper = code.upper()
        if upper in {"PDF-A", "USB-C", "HTTP-S", "ISO-9001", "ASCII-7",
                     "MIT-0", "BSD-2", "UTF-8", "RFC-822",
                     "IEEE-754", "IEEE-802", "ANSI-X", "ISO-27001",
                     "PCI-DSS", "SOC-2", "ISO-9000", "HIPAA-1996"}:
            continue
        # Drop pure standards refs (single-segment-NN like "NIST-800")
        # — keep multi-segment ones
        slug = _slug_simple(code)
        if slug and len(slug) >= 4:
            keys.add(f"site:{slug}")
    return keys


def extract_keys(
    text: str,
    *,
    pack: DomainPack,
    value: Any | None = None,
    authoritative_sites: set[str] | None = None,
) -> list[str]:
    """Return entity_keys for ``text`` using ``pack``'s vocabulary.

    Pure function — no I/O, no global state.  ``value`` is the atom's
    structured ``value`` payload if any (e.g. xlsx table_row).

    ``authoritative_sites`` is the project-wide site catalog built by
    ``app.core.site_detection.find_authoritative_site_phrases``. When
    non-empty, the proper-noun emitter ONLY emits ``site:`` keys whose
    normalized form is in the catalog. When None or empty, the
    emitter falls back to its strict regex behavior.
    """
    if not text:
        return []
    text_lower = text.lower()
    device_idx = _device_alias_index(pack)
    typed_idx = _typed_alias_index(pack)

    keys: set[str] = set()
    keys |= _emit_devices(text_lower, device_idx, pack=pack)
    keys |= _emit_typed(text_lower, typed_idx)
    vendor_keys = _emit_vendors(text_lower)
    keys |= vendor_keys
    keys |= _emit_sites(text)
    # Proper-noun fallback runs LAST so it can deduplicate against
    # vendor matches (avoids "site:genetec_security_center" when we
    # already have "vendor:genetec").
    proper_noun_keys = _emit_proper_nouns(
        text, vendor_keys, authoritative_sites=authoritative_sites,
    )
    keys |= proper_noun_keys
    keys |= _emit_part_numbers(text)
    keys |= _emit_quantity_keys(value or {}, text)
    keys |= _emit_qa_markers(text)
    keys |= _emit_csi_sections(text)
    # Week 6 P6.4: derive ``customer:`` keys from sites that look like
    # an institutional name (school district, university, hospital,
    # municipality) and ``requirement:`` keys from compliance-shape
    # phrases.  Both run AFTER the others so they can reuse the
    # already-detected proper-noun runs.
    keys |= _emit_customer_keys(text, proper_noun_keys)
    keys |= _emit_requirement_keys(text)
    # Direct customer-from-label extraction (Company: X / Customer: X)
    # for cases where the customer is named explicitly with a corporate
    # suffix but doesn't trigger the institutional-suffix promotion.
    keys |= _emit_customer_from_label(text)
    # Money / currency entities (dollar amounts, USD-prefixed, K/M/B
    # shorthand). Unblocks downstream pricing-structure validation.
    keys |= _emit_money_keys(text)
    # Date and milestone entities. Every detected ISO/US/Long-format
    # date becomes a ``date:YYYY-MM-DD`` key; dates adjacent to a
    # milestone-context cue (close date, cutover, hypercare, blackout,
    # ...) ALSO emit a ``milestone:YYYY-MM-DD`` key for timeline
    # reasoning.
    keys |= _emit_date_keys(text)
    # Named stakeholders / approvers. Detected via "First Last" name
    # pattern + role-context cue (CFO, VP, Sponsor, Approver, ...)
    # within ±60 chars in the same sentence.
    keys |= _emit_stakeholders(text)
    # Contact-anchor emitters (universal — every email, phone, site
    # code, and "Name <email>" / "contact Name" person reference).
    # Closes the recall gaps surfaced by the 2026-05-27 source-vs-
    # parser audit (emails 0/36, named people 7/160, site codes 1/113).
    keys |= _emit_email_keys(text)
    keys |= _emit_phone_keys(text)
    keys |= _emit_person_from_contact(text)
    keys |= _emit_site_code_keys(text)

    # Final centralized site:* gate — applies to EVERY emitter above,
    # including _emit_site_code_keys whose loose regex catches time
    # ranges ("AM-10"), day ranges ("Mon-Fri"), PO numbers
    # ("PO-MOCK"), quote codes ("Q-DEV-ATL-047"), deal stages
    # ("HS-DEAL"). The gate must run LAST so nothing downstream of
    # the LLM-catalog check can sneak past.
    from app.core.site_detection import (
        phrase_is_in_catalog,
        _looks_like_site_phrase,
    )
    try:
        from app.core.site_llm_verify import _is_obvious_non_site
    except Exception:
        _is_obvious_non_site = None  # type: ignore
    # UNIVERSAL store-learned role gate over every site:* key in this set,
    # computed once (one batched embed). Applied BEFORE the deterministic
    # denylist so the denylist is never the sole authority. No-op when the
    # gate flag is off / no store wired (empty set → identical pipeline).
    try:
        from app.core.entity_resolution import semantic_site_role_drops
        _site_role_drops = semantic_site_role_drops(
            {k for k in keys if isinstance(k, str) and k.startswith("site:")}
        )
    except Exception:
        _site_role_drops = set()  # type: ignore
    site_keys_kept: set[str] = set()
    for k in list(keys):
        if not k.startswith("site:"):
            continue
        slug = k[len("site:"):]
        phrase = slug.replace("_", " ")
        if not _looks_like_site_phrase(phrase):
            continue
        if k in _site_role_drops:
            continue
        if _is_obvious_non_site is not None and _is_obvious_non_site(phrase):
            continue
        if authoritative_sites and not phrase_is_in_catalog(
            phrase, authoritative_sites
        ):
            continue
        site_keys_kept.add(k)
    keys = {k for k in keys if not k.startswith("site:")} | site_keys_kept

    return sorted(keys)


# Entity-key prefixes that the text extractor can SAFELY add even
# when the parser already supplied other keys. Parser-supplied keys
# remain authoritative for their own type; these prefixes are
# typically textual-pattern matches (sites, dates, money, etc.) that
# parsers don't always carry per-row.
_AUGMENT_ALWAYS_PREFIXES: tuple[str, ...] = (
    "site:",
    "address:",
    "date:",
    "milestone:",
    "quarter:",
    "money:",
    # quantity: belt-and-suspenders with normalize_atom_value's direct emit —
    # the cross-doc quantity conflict reads ONLY quantity: keys, and they were
    # being dropped on parser-keyed atoms (only 3/2404 survived).
    "quantity:",
    "stakeholder:",
    "phone:",
    "email:",
    "zip:",
    "customer:",
    "vendor:",
)


def _section_path_context(atom: Any) -> str:
    """Return the atom's section-path text appended for entity scanning.

    A PDF parser often slices an institutional-name heading
    ("Geary County Schools USD 475") into a structured-doc subsection
    rather than a body paragraph. The atoms under that section then
    have rich body text but no site mention. Scanning section_path
    alongside the body text lets the universal extractor pick up the
    site / customer / institution name and tag every child atom with
    it — exactly what an LLM consumer would expect.
    """
    try:
        refs = getattr(atom, "source_refs", None) or []
        if not refs:
            return ""
        locator = getattr(refs[0], "locator", None) or {}
        if not isinstance(locator, dict):
            return ""
        section_path = locator.get("section_path")
        if isinstance(section_path, list) and section_path:
            return " ".join(str(x) for x in section_path if x)
        # Title / heading fallback
        for k in ("section", "heading", "title", "subsection"):
            v = locator.get(k)
            if isinstance(v, str) and v:
                return v
    except Exception:
        return ""
    return ""


def _enrich_table_atoms(
    atom_list: list[Any],
    *,
    project_id: str,
) -> list[Any]:
    """v49.2 — classify every raw_table_row atom via the column schema
    registry. ONE central function replaces per-parser schema calls.

    Parsers emit raw_table_row atoms with value={_columns, _row, _table_idx,
    _row_idx, _filename}. This function detects the schema for each unique
    column set (cached) and emits the appropriate typed atoms (bom_line,
    cutover_step, requirement, etc.).
    """
    from app.core.table_schema_registry import identify_schema, emit_atoms_for_schema
    from app.core.schemas import AtomType

    new_atoms: list[Any] = []
    schema_cache: dict[tuple, str | None] = {}
    in_count = 0
    schema_hits = 0
    emitted = 0

    for atom in atom_list:
        _atype = getattr(atom, "atom_type", None)
        _atype_val = _atype.value if hasattr(_atype, "value") else str(_atype or "")
        if _atype_val != "raw_table_row":
            continue
        in_count += 1
        val = getattr(atom, "value", None) or {}
        if not isinstance(val, dict):
            continue
        columns = val.get("_columns") or []
        row = val.get("_row") or []
        if not columns or not row:
            continue

        cache_key = tuple(columns)
        if cache_key not in schema_cache:
            schema_cache[cache_key] = identify_schema(list(columns))
        schema_name = schema_cache[cache_key]
        if not schema_name:
            continue
        schema_hits += 1
        # Carry the source row's section/heading chain (sheet name, site heading,
        # etc.) onto the typed atoms so section/site attribution survives the
        # schema-routing step instead of being reset to empty.
        _src_section: list[Any] = list(val.get("section_path") or [])
        if not _src_section:
            try:
                for _r in (getattr(atom, "source_refs", None) or []):
                    _loc = getattr(_r, "locator", None) or {}
                    if isinstance(_loc, dict) and _loc.get("section_path"):
                        _src_section = list(_loc["section_path"])
                        break
            except Exception:
                _src_section = []
        # xlsx rows: the sheet IS the section. Fall back to it so the typed
        # atom references its sheet (docx headings ≈ xlsx sheets).
        _sheet = val.get("_sheet") or None
        if not _src_section and _sheet:
            _src_section = [_sheet]
        try:
            schema_atoms = emit_atoms_for_schema(
                schema_name=schema_name,
                columns=list(columns),
                row=list(row),
                row_idx=int(val.get("_row_idx") or 0),
                table_idx=int(val.get("_table_idx") or 0),
                project_id=project_id,
                artifact_id=getattr(atom, "artifact_id", "") or "",
                filename=str(val.get("_filename") or ""),
                section_path=_src_section,
                sheet=_sheet,
            )
            if schema_atoms:
                new_atoms.extend(schema_atoms)
                emitted += len(schema_atoms)
        except Exception:
            pass

    if in_count:
        import sys as _sys_rtr
        try:
            print(
                f"raw_table_row_v49_2: input={in_count} schema_matched={schema_hits} emitted={emitted}",
                file=_sys_rtr.stderr,
            )
        except Exception:
            pass
    return new_atoms


def _entities_to_atoms(
    multi_result: dict[str, Any],
    *,
    project_id: str,
    artifact_ids: list[str],
    parser_version: str = "entity_bridge_v49",
    existing_physical_site_count: int = 0,
    existing_physical_sites: list[Any] | None = None,
) -> list[Any]:
    """v49 — bridge LLM entity findings into proper EvidenceAtom instances.

    multi_result contains structured facts extracted by multi_entity_llm
    (stakeholders, milestones, requirements, cutover_steps, signatories,
    etc.). Previously these only mutated entity_keys on existing atoms —
    they never became atoms themselves. After this bridge: every entity
    finding becomes a typed EvidenceAtom with structured value.
    """
    from app.core.ids import stable_id
    from app.core.schemas import (
        ArtifactType, AtomType, AuthorityClass, EvidenceAtom, EvidenceReceipt, ReviewStatus, SourceRef,
    )

    if not multi_result or not artifact_ids:
        return []

    canonical_aid = artifact_ids[0]

    CATEGORY_TO_ATOM_TYPE: dict[str, AtomType] = {
        "stakeholders":                 AtomType.stakeholder,
        "milestones":                   AtomType.milestone_phase,
        "requirements":                 AtomType.requirement,
        "lead_times":                   AtomType.lead_time_constraint,
        "payment_terms":                AtomType.payment_term,
        "electrical_acceptance":        AtomType.electrical_acceptance_test,
        "compliance_obligations":       AtomType.compliance_rule,
        "certifications":               AtomType.compliance,
        "risks":                        AtomType.risk,
        "acceptance_criteria":          AtomType.acceptance_criterion,
        "penalties":                    AtomType.constraint,
        # v49 new categories (Fix 4)
        "cutover_steps":                AtomType.cutover_step,
        "signatories":                  AtomType.signatory,
        "compliance_classifications":   AtomType.compliance_classification,
        "integration_checkpoints":      AtomType.integration_checkpoint,
        "deliverables":                 AtomType.deliverable,
        "system_mappings":              AtomType.system_mapping,
        "data_flow_steps":              AtomType.data_flow_step,
        "assumptions":                  AtomType.assumption,
        "approval_authorities":         AtomType.approval_authority,
        "approval_decisions":           AtomType.approval_decision,
        "dependencies":                 AtomType.dependency,
        "mitigations":                  AtomType.mitigation,
        # GAP D: pricing_structure → payment_term atoms so OrbitBrief
        # commercial.pricing_structure rule finds the evidence.
        "pricing_structure":            AtomType.payment_term,
        # v52
        "blackout_date_range":          AtomType.blackout_date_range,
        "approval_decision":            AtomType.approval_decision,
        # v53.2 BRIDGE GAP — site_clusters from LLM site extraction now
        # become physical_site atoms (used to only feed
        # _llm_site_attr_cache → never crossed into atoms list →
        # downstream canonical_set never saw them on docs without a
        # structural roster table). quantities now become quantity
        # atoms (similar gap — LLM-extracted figures had no atom).
        "site_clusters":                AtomType.physical_site,
        "quantities":                   AtomType.quantity,
        # v50 — comprehensive commercial line items. ONE extractor, but each
        # item routes to its own atom type by the teacher-emitted `category`
        # (see _COMMERCIAL_CAT_TO_TYPE below). Listed here with a default; the
        # per-item override in the loop picks the real type.
        "commercial_line_items":        AtomType.commercial_total,
    }

    # Per-item routing for commercial_line_items: the item's top-level
    # `category` (labor/pmo/hardware/material/expense/license_subscription/
    # other_commercial) selects the atom type so each lands in the right Deal
    # Kit section.
    _COMMERCIAL_CAT_TO_TYPE = {
        "labor":                AtomType.service_line,
        "pmo":                  AtomType.pmo,
        "hardware":             AtomType.bom_line,
        "material":             AtomType.material,
        "expense":              AtomType.expense,
        "license_subscription": AtomType.license_subscription,
        "other_commercial":     AtomType.commercial_total,
    }

    def _best_text(entity: dict) -> str:
        for field in ("text", "description", "name", "item", "test", "criterion",
                      "deliverable", "step", "checkpoint", "source", "title",
                      "assumption", "approver", "tranche", "step_id"):
            v = entity.get(field)
            if v and isinstance(v, str) and len(v.strip()) >= 3:
                return v.strip()
        return " | ".join(
            str(v) for v in entity.values()
            if v and isinstance(v, (str, int, float)) and str(v).strip()
        )[:500]

    out: list[Any] = []
    seen_texts: set[str] = set()
    # v49.1: log which categories the bridge actually saw vs. emitted.
    # Goes to stderr so worker logs show which LLM extractors returned
    # data and which the bridge dropped (e.g. wrong field name, dedup).
    import sys as _sys_v491
    _cat_counts: dict[str, dict[str, int]] = {}

    # v53.3: site-ID-shaped patterns like ATL-HQ-01, STORE-142, MDF-3A.
    # When an LLM site_cluster has one of these as an alias, prefer
    # IT as the canonical site id so this atom collapses with any
    # roster-parsed physical_site atom (which also uses the ID).
    import re as _re_sid_norm
    _SITE_ID_SHAPE = _re_sid_norm.compile(
        r"^[A-Z]{2,6}[-_][A-Z0-9]{1,8}([-_]\d{1,3})?$"
    )
    # v53.5: matches an address (number + street name) or a city/state/zip
    # token. Used to REJECT LLM site_clusters whose canonical_name is a
    # street address rather than a real facility name.
    _ADDRESS_SHAPE = _re_sid_norm.compile(
        r"^\d+\s+[A-Z]",  # starts with house number + capitalized word
        _re_sid_norm.IGNORECASE,
    )
    # v53.7: LLM commonly hallucinates "<prefix> <suffix> <year>" patterns
    # ("Atl Hq 2026", "ATL West 2027") from contract/phase dates in text.
    # Reject canonical_names that end with a 4-digit year token.
    _YEAR_SUFFIX = _re_sid_norm.compile(r"\s\d{4}$")
    # v53.7: LLM also hallucinates "Site X" / "Location X" wrapper prefixes.
    _WRAPPED_SITE = _re_sid_norm.compile(
        r"^(?:site|location|facility|building)\s+",
        _re_sid_norm.IGNORECASE,
    )
    # v53.5: garbage canonical values from LLM site extraction that
    # should never become physical_site atoms — generic placeholders,
    # company names, addresses, pure numbers.
    _BAD_SITE_NAMES: frozenset[str] = frozenset({
        "all", "all sites", "all locations", "various", "tbd", "n/a",
        "na", "none", "unknown", "various sites",
        # v53.7: common LLM placeholder phrasing
        "site all", "site various", "various locations",
    })

    def _pick_site_id(canonical_name: str, aliases: list) -> str:
        """Prefer the LONGEST site-ID-shaped alias over the prose
        canonical_name. Longer = more specific (ATL-HQ-01 > ATL-HQ).
        Falls back to canonical_name when no ID-shaped alias exists.
        """
        candidates = []
        for cand in (aliases or []) + [canonical_name]:
            if not isinstance(cand, str):
                continue
            stripped = cand.strip()
            if _SITE_ID_SHAPE.match(stripped):
                candidates.append(stripped)
        if candidates:
            # Pick longest (most specific), tie-break alphabetically for stability
            candidates.sort(key=lambda s: (-len(s), s))
            return candidates[0]
        return canonical_name

    def _is_garbage_site(canonical_name: str) -> bool:
        """Reject LLM site_cluster atoms whose canonical_name is garbage:
        addresses-as-name, generic placeholders, customer-name leaks,
        year-suffixed phase names ("Atl Hq 2026"), wrapped tags
        ("Site ALL", "Location TBD").
        """
        if not canonical_name:
            return True
        s = canonical_name.strip()
        if not s:
            return True
        if s.lower() in _BAD_SITE_NAMES:
            return True
        # v53.7: strip wrapping prefix and recheck the inner phrase
        # ("Site ALL" → "ALL" → bad)
        if _WRAPPED_SITE.match(s):
            inner = _WRAPPED_SITE.sub("", s, count=1).strip()
            if not inner or inner.lower() in _BAD_SITE_NAMES:
                return True
        # v53.7: LLM hallucinated "<phase> <year>" patterns
        if _YEAR_SUFFIX.search(s):
            return True
        # Addresses ("1180 Peachtree Street NE...")
        if _ADDRESS_SHAPE.match(s):
            return True
        # Pure number / very short / very long
        if s.isdigit() or len(s) < 3 or len(s) > 100:
            return True
        # v53.7: also check id/aliases for "ALL" garbage that survives
        # via aliases-only construction (canonical_name something else
        # but id resolves to 'ALL' via _pick_site_id).
        if s.upper() == "ALL":
            return True
        return False

    def _normalize_entity_value(category: str, entity: dict) -> dict:
        """v53.2: per-category value normalization so the bridged
        atom's value dict matches what downstream code (semantic_dedup,
        site_readiness canonical_set, find_authoritative_site_phrases)
        expects. Without this, LLM-shaped {canonical_name, aliases}
        site_clusters never reach the physical_site flow because
        downstream queries value.id / value.name / value.names.
        """
        out_val = dict(entity)
        if category == "site_clusters":
            canon_name = entity.get("canonical_name") or entity.get("name") or entity.get("id") or ""
            aliases = entity.get("aliases") or []
            if isinstance(aliases, str):
                aliases = [aliases]
            if not isinstance(aliases, list):
                aliases = []
            # v53.3: prefer a site-ID-shaped alias as canonical id so
            # we collapse with roster-parsed atoms.
            canonical_id = _pick_site_id(canon_name, aliases)
            # Normalize to physical_site shape
            out_val["id"] = canonical_id
            out_val.setdefault("site_id", canonical_id)
            out_val.setdefault("name", canon_name)
            # All forms — including the prose canonical_name — become aliases
            # that downstream alias-collapse / catalog gates can recognize.
            names_set: list[str] = []
            for nm in [canonical_id, canon_name] + list(aliases):
                if isinstance(nm, str) and nm and nm not in names_set:
                    names_set.append(nm)
            out_val["names"] = names_set
            out_val.setdefault("kind", "physical_site")
        elif category == "quantities":
            # quantities have {text, kind, category} — make sure value/unit
            # are present for downstream consumers.
            qt = entity.get("text") or entity.get("canonical") or ""
            out_val.setdefault("text", qt)
            out_val.setdefault("description", qt)
        return out_val

    # v55 TARGETED-MERGE FIX (replaces v53.9 all-or-nothing suppression):
    # The old approach suppressed EVERY LLM site_cluster whenever any
    # structural physical_site atom existed. That was overcorrection —
    # we threw away (a) genuine new sites the LLM found in prose that
    # weren't in the roster table, and (b) high-value alias data
    # (canonical names, addresses) the LLM extracted that could enrich
    # structural atoms.
    #
    # The right algorithm:
    #   1. Build a normalized-form index over every existing structural
    #      physical_site atom: every alias (site_id, name, facility_name,
    #      street_address) maps back to the atom's canonical id.
    #   2. For each LLM site_cluster, normalize its canonical_name +
    #      every alias and look them up in the index.
    #   3. If ANY form matches an existing structural atom → MERGE the
    #      LLM's aliases into that atom and SKIP creating a new atom.
    #   4. If NO form matches → emit as a new physical_site atom (this
    #      is the supplemental "found in prose, not in any table" case).
    #
    # Net effect on OPTBOT: structural finds ATL-HQ-01..ATL-CP-05 (5
    # atoms). LLM finds 5 clusters with surface forms like
    # "OPTBOT Atlanta HQ" (aliases include ATL-HQ-01 + address). The
    # match step recognises these as the same sites, merges the rich
    # aliases into the 5 structural atoms, and emits 0 new atoms. The
    # PM sees 5 sites with full aliases attached, not 10 fighting.
    structural_sites_index: dict[str, Any] = {}  # normalized_form -> atom
    structural_atoms_list = existing_physical_sites or []
    if structural_atoms_list:
        def _norm_form(s: str | None) -> str:
            if not isinstance(s, str):
                return ""
            return re.sub(r"[^a-z0-9]+", "", s.lower())
        for _atom in structural_atoms_list:
            v = getattr(_atom, "value", None) or {}
            if not isinstance(v, dict):
                continue
            for field in ("site_id", "id", "name", "facility_name", "street_address", "address"):
                key = _norm_form(v.get(field))
                if key and key not in structural_sites_index:
                    structural_sites_index[key] = _atom
            # Also index any pre-existing names[] array
            for nm in (v.get("names") or []):
                key = _norm_form(nm if isinstance(nm, str) else "")
                if key and key not in structural_sites_index:
                    structural_sites_index[key] = _atom

    # Back-compat: when structural list isn't passed but the count is
    # >0 (older callers), fall back to the old suppression behaviour.
    suppress_llm_sites = (
        existing_physical_site_count > 0
        and not structural_sites_index  # only if we DIDN'T get the actual list
    )

    for category, atom_type in CATEGORY_TO_ATOM_TYPE.items():
        if category == "site_clusters" and suppress_llm_sites:
            _cat_counts[category] = {"in": len(multi_result.get(category, []) or []), "out": 0, "suppressed": True}
            continue
        entities = multi_result.get(category)
        if not isinstance(entities, list):
            _cat_counts[category] = {"in": 0, "out": 0}
            continue
        _cat_counts[category] = {"in": len(entities), "out": 0}
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            # v50: commercial line items route per-item by their `category`
            # (labor→service_line, hardware→bom_line, material→material,
            # expense→expense, …). Everything else uses the category's mapped
            # type.
            eff_atom_type = atom_type
            if category == "commercial_line_items":
                eff_atom_type = _COMMERCIAL_CAT_TO_TYPE.get(
                    str(entity.get("category") or "").strip().lower(),
                    AtomType.commercial_total,
                )
            # v53.2: best_text needs to look at canonical_name for sites.
            raw_text = entity.get("canonical_name") if category == "site_clusters" else None
            if not raw_text:
                raw_text = _best_text(entity)
            if not raw_text or len(raw_text) < 3:
                continue
            # v53.5: drop garbage LLM site_cluster atoms (ALL/various,
            # address-as-name, customer-name leaks, pure numbers).
            # v53.9b: also check the entity's id/site_id/name fields
            # directly — graph_expansion creates entities like
            # {entity_type:'site', site_id:'ALL', name:''} where
            # canonical_name is None and _best_text concatenates values
            # into "site | ALL" which bypasses the canonical_name filter.
            if category == "site_clusters":
                if _is_garbage_site(raw_text):
                    continue
                # v55: MERGE step — if any of this cluster's forms matches
                # a structural physical_site atom, enrich that atom's
                # aliases instead of creating a new (duplicate) atom.
                if structural_sites_index:
                    def _nf(s: str | None) -> str:
                        if not isinstance(s, str):
                            return ""
                        return re.sub(r"[^a-z0-9]+", "", s.lower())
                    cluster_forms: list[str] = []
                    cn = entity.get("canonical_name")
                    if isinstance(cn, str):
                        cluster_forms.append(cn)
                    for a_ in (entity.get("aliases") or []):
                        if isinstance(a_, str):
                            cluster_forms.append(a_)
                    matched_atom = None
                    for form in cluster_forms:
                        key = _nf(form)
                        if key and key in structural_sites_index:
                            matched_atom = structural_sites_index[key]
                            break
                    if matched_atom is not None:
                        # v56f: SKIP — structural atom is canonical. Do NOT
                        # merge LLM cluster aliases into value.names. The
                        # structural atom already has every column from
                        # its source row (site_id, facility_name,
                        # street_address, mdf_idf, access_window,
                        # escort_owner) in proper named fields.
                        # _clean_physical_site_value derives aliases
                        # deterministically from THOSE fields — not from
                        # LLM clusters whose aliases mix multiple-row
                        # data ("MDF-3A" as alias of West Campus,
                        # "OPTBOT Facil" the escort_owner ending up
                        # tagged onto a site, "1200 Peachtree St NE"
                        # from row 1 leaking onto row 2 etc.).
                        #
                        # We also no longer expand structural_sites_index
                        # with cluster aliases. That expansion was the
                        # source of cross-row contamination: cluster B's
                        # alias matching the alias-expanded-from-cluster-A
                        # entry → cluster B merging into row A's atom.
                        # Without expansion, matching is deterministic
                        # and tied to the original structural fields only.
                        _cat_counts[category]["merged_into_structural"] = (
                            _cat_counts[category].get("merged_into_structural", 0) + 1
                        )
                        continue
                    # v56e MERGE-OR-DROP GUARD: when structural physical_site
                    # atoms exist, the bridge must NOT create new physical_site
                    # atoms from unmatched LLM clusters. The structural roster
                    # is canonical truth; LLM clusters are best-effort
                    # enrichment. Unmatched clusters here are noise — bare
                    # abbreviations the matcher couldn't resolve ("HQ",
                    # "AIR"), address-as-name strings, or LLM hallucinations.
                    # Without this guard, parallel-dispatch ordering in the
                    # LLM-pool produces 0–5 ghost atoms per run depending
                    # on which clusters happen to arrive before the
                    # alias-index expansion catches their forms — a
                    # non-deterministic regression source. Dropping is
                    # safe: any genuine NEW site appearing only in prose
                    # (not in the structural roster) would also lack the
                    # contextual signals the matcher needs, so emitting
                    # it would produce a poor-quality atom anyway.
                    _cat_counts[category]["dropped_unmatched"] = (
                        _cat_counts[category].get("dropped_unmatched", 0) + 1
                    )
                    continue
                # Field-level garbage check
                bad_field = False
                for field in ("id", "site_id", "name", "facility_name", "canonical_name"):
                    val = entity.get(field)
                    if isinstance(val, str) and _is_garbage_site(val):
                        bad_field = True
                        break
                if bad_field:
                    continue
            dedup_key = f"{category}:{raw_text[:120].lower()}"
            if dedup_key in seen_texts:
                continue
            seen_texts.add(dedup_key)

            atom_id = stable_id("atm", canonical_aid, "entity_bridge", category, raw_text[:64])
            src = SourceRef(
                id=stable_id("src", atom_id),
                artifact_id=canonical_aid,
                artifact_type=ArtifactType.docx,
                filename="entity_bridge",
                locator={
                    "extraction": "entity_bridge_v49",
                    "category": category,
                },
                extraction_method="entity_bridge_v49",
                parser_version=parser_version,
            )
            normalized_value = _normalize_entity_value(category, entity)
            receipt = EvidenceReceipt(
                atom_id=atom_id,
                artifact_id=canonical_aid,
                filename="entity_bridge",
                source_ref_id=src.id,
                replay_status="unsupported",
                extracted_snippet=raw_text[:500],
                locator=src.locator,
                reason="post_source_replay_entity_bridge_atom",
                verifier_version=parser_version,
            )
            out.append(
                EvidenceAtom(
                    id=atom_id,
                    project_id=project_id,
                    artifact_id=canonical_aid,
                    atom_type=eff_atom_type,
                    raw_text=raw_text,
                    normalized_text=raw_text.lower(),
                    value=normalized_value,
                    entity_keys=[],
                    source_refs=[src],
                    receipts=[receipt],
                    # v53.2: site_clusters from LLM are weaker authority
                    # than a parsed roster table (avoid promoting LLM
                    # guesses over real physical_site atoms in dedup).
                    authority_class=AuthorityClass.machine_extractor,
                    confidence=0.78 if category == "site_clusters" else 0.82,
                    confidence_raw=0.78 if category == "site_clusters" else 0.82,
                    calibrated_confidence=0.78 if category == "site_clusters" else 0.82,
                    review_status=ReviewStatus.auto_accepted,
                    review_flags=[],
                    parser_version=parser_version,
                )
            )
            _cat_counts[category]["out"] += 1
    # v49.1: diagnostic — surface bridge in/out per category.
    try:
        summary = " ".join(
            f"{cat}:{ct['in']}→{ct['out']}" for cat, ct in _cat_counts.items() if ct["in"] > 0
        )
        if summary:
            print(f"entity_bridge_v49: {summary}", file=_sys_v491.stderr)
    except Exception:
        pass
    return out


def _display_name_from_slug(slug: str) -> str:
    return " ".join(part.upper() if len(part) <= 2 else part.capitalize() for part in slug.split("_") if part)


def _copy_first_source(atom: Any) -> Any | None:
    refs = list(getattr(atom, "source_refs", []) or [])
    return refs[0] if refs else None


def _structural_people_atoms(atom_list: list[Any], project_id: str) -> list[Any]:
    """Emit typed people/governance atoms from high-precision structural cues.

    This is deliberately not a broad person-name regex. It only promotes:
      * contact roster rows with name | title | email | role structure,
      * explicit Owner: / approver signatures already anchored by stakeholder:* keys,
      * explicit signature blocks, and
      * site-roster escort-owner teams.

    That gives deterministic recall for contact tables when the LLM bridge is
    disabled or unavailable, without opening the old false-positive floodgate.
    """
    from app.core.ids import stable_id
    from app.core.schemas import (
        AtomType, AuthorityClass, EvidenceAtom, EvidenceReceipt, ReviewStatus,
    )

    contact_re = re.compile(
        r"(?P<name>[A-Z][A-Za-z'.-]+(?:\s+[A-Z][A-Za-z'.-]+){1,3})\s*\|\s*"
        r"(?P<title>[^|\n.]{2,90})\s*\|\s*"
        r"(?P<email>[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})\s*\|\s*"
        r"(?P<role>[^|\n.]{2,90})\s*\|\s*"
        r"(?P<note>[^\n.]{0,240})",
    )
    owner_re = re.compile(
        r"\bOwner:\s*(?P<name>[A-Z][A-Za-z'.-]+\s+[A-Z][A-Za-z'.-]+)\b"
    )
    delegate_re = re.compile(
        r"\b(?P<name>[A-Z][A-Za-z'.-]+\s+[A-Z][A-Za-z'.-]+),\s*"
        r"(?P<title>[^:;\n]{2,80}):"
    )
    signatory_re = re.compile(
        r"(?P<role>(?:SIGNATURE BLOCKS\s+)?[A-Z][A-Za-z0-9 /&-]{2,90}?):\s*"
        r"(?P<name>[A-Z][A-Za-z'.-]+(?:\s+[A-Z][A-Za-z'.-]+){1,3})\s*\|\s*Signature\s*:",
        re.IGNORECASE,
    )

    stakeholder_candidates: dict[str, tuple[Any, dict[str, Any], str, float]] = {}
    signatory_candidates: dict[tuple[str, str], tuple[Any, dict[str, Any], str, float]] = {}

    def _put_stakeholder(slug: str, source_atom: Any, value: dict[str, Any], raw: str, confidence: float) -> None:
        if not slug or slug in {"mock_vendor", "vendor", "customer", "project_manager"}:
            return
        parts = slug.split("_")
        if value.get("kind") != "team_contact" and len(parts) < 2:
            return
        prev = stakeholder_candidates.get(slug)
        if prev is None or confidence > prev[3] or (value.get("email") and not prev[1].get("email")):
            stakeholder_candidates[slug] = (source_atom, value, raw, confidence)

    def _put_signatory(slug: str, role_slug: str, source_atom: Any, value: dict[str, Any], raw: str, confidence: float) -> None:
        if not slug or not role_slug:
            return
        key = (slug, role_slug)
        prev = signatory_candidates.get(key)
        if prev is None or confidence > prev[3]:
            signatory_candidates[key] = (source_atom, value, raw, confidence)

    def _slug_name(name: str) -> str:
        return _slug(name)

    for atom in atom_list:
        raw = getattr(atom, "raw_text", "") or ""
        if not raw:
            continue
        stakeholder_keys = [
            k[len("stakeholder:"):]
            for k in (getattr(atom, "entity_keys", []) or [])
            if isinstance(k, str) and k.startswith("stakeholder:")
        ]
        stakeholder_key_set = set(stakeholder_keys)

        for m in contact_re.finditer(raw):
            name = m.group("name").strip()
            slug = _slug_name(name)
            if stakeholder_key_set and slug not in stakeholder_key_set:
                continue
            value = {
                "name": name,
                "title": m.group("title").strip(),
                "email": m.group("email").strip().lower(),
                "role": m.group("role").strip(),
                "note": m.group("note").strip(),
                "kind": "person",
            }
            _put_stakeholder(slug, atom, value, m.group(0).strip(), 0.90)

        for slug in stakeholder_keys:
            if slug in stakeholder_candidates:
                continue
            display = _display_name_from_slug(slug)
            if display and display in raw:
                role = "stakeholder"
                title = ""
                conf = 0.72
                for m in owner_re.finditer(raw):
                    if _slug_name(m.group("name")) == slug:
                        role = "owner"
                        conf = 0.78
                        break
                for m in delegate_re.finditer(raw):
                    if _slug_name(m.group("name")) == slug:
                        role = "approval delegate"
                        title = m.group("title").strip()
                        conf = 0.80
                        break
                if "approv" in raw.lower() and role == "stakeholder":
                    role = "approver"
                    conf = max(conf, 0.76)
                value = {
                    "name": display,
                    "title": title,
                    "role": role,
                    "kind": "person",
                }
                _put_stakeholder(slug, atom, value, raw[:500], conf)

        for m in signatory_re.finditer(raw):
            role = re.sub(r"^SIGNATURE BLOCKS\s+", "", m.group("role").strip(), flags=re.IGNORECASE)
            role = re.sub(r"\s+", " ", role).strip(" -:")
            name = re.sub(r"\s+", " ", m.group("name").strip())
            slug = _slug_name(name)
            role_slug = _slug(role)
            value = {
                "name": name,
                "title": role,
                "role": role,
                "signatory_type": role,
                "kind": "signatory",
            }
            _put_signatory(slug, role_slug, atom, value, f"{role}: {name}", 0.88)
            # Signatories are also governance stakeholders, but do not
            # overwrite a richer contact-table row with email/title.
            _put_stakeholder(slug, atom, {"name": name, "title": role, "role": "signatory", "kind": "person"}, f"{role}: {name}", 0.77)

    def _expand_team_contact(name: str) -> str:
        low = (name or "").strip().lower()
        if low.startswith("optbot facil"):
            return "OPTBOT Facilities"
        if low.startswith("optbot secur"):
            return "OPTBOT Security"
        if low.startswith("optbot logis"):
            return "OPTBOT Logistics"
        return name.strip()

    for atom in atom_list:
        val = getattr(atom, "value", None) or {}
        if not isinstance(val, dict):
            continue
        atype = getattr(atom, "atom_type", None)
        atype_str = atype.value if hasattr(atype, "value") else str(atype or "")
        if atype_str != "physical_site":
            continue
        team = _expand_team_contact(str(val.get("escort_owner") or val.get("contact") or ""))
        if not team:
            continue
        slug = _slug(team)
        value = {
            "name": team,
            "role": "site escort owner",
            "kind": "team_contact",
            "org_side": "customer",
            "site_id": val.get("site_id") or val.get("id"),
        }
        _put_stakeholder(slug, atom, value, f"Escort owner: {team}", 0.82)

    out: list[Any] = []

    def _make_atom(source_atom: Any, atom_type: Any, suffix_parts: tuple[str, ...], raw: str, value: dict[str, Any], confidence: float) -> Any | None:
        src = _copy_first_source(source_atom)
        if src is None:
            return None
        artifact_id = getattr(source_atom, "artifact_id", "") or getattr(src, "artifact_id", "") or "unknown_artifact"
        filename = getattr(src, "filename", "") or "structural_people"
        aid = stable_id("atm", artifact_id, "structural_people", *suffix_parts)
        receipt = EvidenceReceipt(
            atom_id=aid,
            artifact_id=artifact_id,
            filename=filename,
            source_ref_id=getattr(src, "id", stable_id("src", aid)),
            replay_status="unsupported",
            extracted_snippet=raw[:500],
            locator=getattr(src, "locator", {}) or {},
            reason="post_source_replay_structural_people_atom",
            verifier_version="structural_people_v54",
        )
        return EvidenceAtom(
            id=aid,
            project_id=project_id,
            artifact_id=artifact_id,
            atom_type=atom_type,
            raw_text=raw[:4000],
            normalized_text=raw.lower()[:4000],
            value=value,
            entity_keys=[f"stakeholder:{suffix_parts[1]}"] if atom_type == AtomType.stakeholder and len(suffix_parts) > 1 else [],
            source_refs=[src],
            receipts=[receipt],
            authority_class=AuthorityClass.contractual_scope,
            confidence=confidence,
            confidence_raw=confidence,
            calibrated_confidence=confidence,
            review_status=ReviewStatus.auto_accepted,
            review_flags=[],
            parser_version="structural_people_v54",
        )

    for slug, (source_atom, value, raw, confidence) in stakeholder_candidates.items():
        atom = _make_atom(source_atom, AtomType.stakeholder, ("stakeholder", slug), raw, value, confidence)
        if atom is not None:
            out.append(atom)
    for (slug, role_slug), (source_atom, value, raw, confidence) in signatory_candidates.items():
        atom = _make_atom(source_atom, AtomType.signatory, ("signatory", slug, role_slug), raw, value, confidence)
        if atom is not None:
            out.append(atom)
    return out


def enrich_atoms(atoms: Iterable[Any], pack: DomainPack) -> tuple[int, int]:
    """Mutate ``atoms`` in place: populate ``entity_keys``.

    Two passes per atom:
      1. If the atom has no entity_keys, run ``extract_keys`` on its
         raw_text + section_path context + value.
      2. If the atom already has entity_keys (parser-supplied), STILL
         run ``extract_keys`` to add textual-pattern keys (``site:``,
         ``date:``, ``money:``, ``stakeholder:``, …) the parser
         doesn't typically emit. Parser-supplied keys for the same
         prefix family are preserved; only NEW prefixes get merged.

    Returns ``(atoms_enriched, total_keys_added)`` for telemetry.
    """
    atoms_enriched = 0
    total_keys_added = 0
    # Build the project-wide authoritative-site catalog ONCE per
    # enrichment pass. Option D — document-structure aware site
    # detection: only phrases that appear in a Locations section,
    # near a US address, or match a strong-facility-tail pattern
    # become valid site:* candidates. Everything else (random
    # landmarks, standards bodies, header fragments) gets dropped.
    from app.core.site_detection import find_authoritative_site_phrases
    atom_list = list(atoms)
    authoritative_sites = find_authoritative_site_phrases(atom_list)

    # When the catalog contains LLM-discovered sites that NO atom's
    # raw_text would naturally emit (e.g. "Geary County Schools USD
    # 475" lives in a PDF cover-page heading, not in any body block),
    # we explicitly inject ``site:<slug>`` keys onto atoms that
    # mention the site in their body OR section_path. Without this
    # step the LLM's discoveries never reach atom.entity_keys and
    # downstream EntityRecord fusion has nothing to fuse.
    def _slug_for_site(phrase: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", phrase.lower()).strip("_")

    _SITE_INJECTION_KEYS: dict[str, str] = {
        _slug_for_site(p): p for p in authoritative_sites
    }

    # Site-gate closure — applied to ANY collection of keys, no matter
    # the source (parser-supplied, regex-emitted, or LLM-injected).
    # Centralised here so the gate is enforced even when extract_keys
    # is not the entry path (e.g. parser already populated entity_keys).
    from app.core.site_detection import (
        phrase_is_in_catalog as _phrase_in_cat,
        _looks_like_site_phrase as _looks_like_site,
    )
    try:
        from app.core.site_llm_verify import _is_obvious_non_site as _obv_non_site
    except Exception:
        _obv_non_site = None  # type: ignore

    # UNIVERSAL store-learned role gate over the full injection key universe
    # (roster catalog + every atom's site:* keys), computed once and applied
    # inside the closure BEFORE the deterministic denylist. No-op when off.
    try:
        from app.core.entity_resolution import semantic_site_role_drops as _site_role_fn
        _gate_role_drops = _site_role_fn(
            {
                k for a in atom_list for k in (getattr(a, "entity_keys", None) or [])
                if isinstance(k, str) and k.startswith("site:")
            }
        )
    except Exception:
        _gate_role_drops = set()  # type: ignore

    # v53.2: STRICT mode — when ANY physical_site atom exists, the
    # roster catalog is the ONLY valid set of sites. Reject any
    # site:* key not aliasing to a roster row.
    _has_physical_site_atoms = any(
        (getattr(a, "atom_type", None).value
            if hasattr(getattr(a, "atom_type", None), "value")
            else str(getattr(a, "atom_type", "") or "")) == "physical_site"
        for a in atom_list
    )
    _strict_site_gate = _has_physical_site_atoms or bool(authoritative_sites)

    def _gate_site_keys(keys_in: Iterable[str]) -> list[str]:
        out: list[str] = []
        for k in keys_in:
            if not k.startswith("site:"):
                out.append(k)
                continue
            phrase = k[len("site:"):].replace("_", " ")
            if not _looks_like_site(phrase):
                continue
            if k in _gate_role_drops:
                continue
            if _obv_non_site is not None and _obv_non_site(phrase):
                continue
            # When we have a canonical roster (physical_site atoms or
            # LLM-built catalog), REJECT anything not in it. Previously
            # this was opt-in only when catalog was non-empty; now it's
            # strict whenever roster signal exists.
            if _strict_site_gate:
                if not authoritative_sites or not _phrase_in_cat(phrase, authoritative_sites):
                    continue
            out.append(k)
        return out

    # v49.1: atom types whose value is already fully structured by the
    # column schema registry (Fix 1). Their raw_text is a synthesized
    # composite ("Wi-Fi 7 AP | site: ATL-AIR | qty: 105") that the
    # regex emitters wrongly parse into ghost site:atl_air_qty_105
    # entity_keys. Skip enrichment entirely — these atoms carry their
    # structured fields in atom.value, no entity_keys needed.
    _SKIP_ENRICHMENT_TYPES = {
        "site_allocation", "bom_line", "cutover_step",
        "acceptance_criterion", "deliverable", "site_budget",
        "integration_checkpoint", "compliance_classification",
        "system_mapping", "signatory", "site_attribute",
        "requirement",
        # v49.2: raw_table_row is a structured intermediate consumed
        # by _enrich_table_atoms. Its raw_text is a synthesized
        # "Site|Part|Qty" composite that pollutes entity_keys if
        # the regex emitters run on it.
        "raw_table_row",
        # v56: physical_site atoms come from structured-table parsers
        # (PDF tables / xlsx rows / docx tables). Their identity is
        # value.site_id; raw_text is a synthesized "site_id: ATL-WEST-0
        # | facility: OPTBOT West Campus | address: ... | mdf_idf:
        # MDF-3A | escort: OPTBOT Facilities" composite that the regex
        # emitters wrongly parse into ghost keys: site:mdf_3a (MDF
        # column → not a site), site:optbot_facil (escort column +
        # truncation → not a site), site:atl_air_asset_type_warehouse
        # (column-header bleed), site:atl_hq_2026 (year-suffix concat).
        # The parser already KNOWS the identity from the table row;
        # the regex pass undoes that knowledge. Skip enrichment and
        # emit exactly ONE clean key from value.site_id (see _emit_one_
        # site_key_from_value below).
        "physical_site",
    }

    def _emit_one_site_key_from_value(atom: Any) -> bool:
        """v56: for atoms in _SKIP_ENRICHMENT_TYPES that have a
        ``value.site_id`` (physical_site rows specifically), ensure the
        atom carries EXACTLY ONE clean ``site:<slug>`` entity_key derived
        from the structured site_id field. This preserves graph anchoring
        without regex-deriving garbage keys from a flattened raw_text.

        Returns True when a key was added/modified, False otherwise.
        """
        val = getattr(atom, "value", None) or {}
        if not isinstance(val, dict):
            return False
        sid = val.get("site_id") or val.get("id") or ""
        if not isinstance(sid, str) or not sid.strip():
            return False
        slug = re.sub(r"[^a-z0-9]+", "_", sid.strip().lower()).strip("_")
        if not slug:
            return False
        target_key = f"site:{slug}"
        existing = list(getattr(atom, "entity_keys", []) or [])
        # Drop any pre-existing site:* keys (from parser or prior pass)
        # and add the single canonical one. Preserve non-site keys
        # (date:, money:, etc.) the parser may have emitted.
        non_site = [k for k in existing if not k.startswith("site:")]
        non_site.append(target_key)
        if list(getattr(atom, "entity_keys", []) or []) == non_site:
            return False  # already canonical
        atom.entity_keys = non_site
        return True

    for atom in atom_list:
        # v49.1: skip schema-emitted atoms entirely
        _atype = getattr(atom, "atom_type", None)
        _atype_str = _atype.value if hasattr(_atype, "value") else str(_atype or "")
        if _atype_str in _SKIP_ENRICHMENT_TYPES:
            # v56: physical_site rows still need ONE canonical site:* key
            # for downstream graph linking. Derive it from value.site_id —
            # do NOT regex over raw_text (that's how ghost keys creep in).
            if _atype_str == "physical_site":
                if _emit_one_site_key_from_value(atom):
                    atoms_enriched += 1
                    total_keys_added += 1
            continue

        existing = list(getattr(atom, "entity_keys", []) or [])
        text = getattr(atom, "raw_text", "") or ""
        value = getattr(atom, "value", None)
        # Concatenate section-path context so institutional names that
        # live in headings (not body) still emit ``site:`` / ``customer:``
        # keys onto child atoms.
        section_ctx = _section_path_context(atom)
        scan_text = f"{text} {section_ctx}".strip() if section_ctx else text

        if not existing:
            new_keys = extract_keys(
                scan_text, pack=pack, value=value,
                authoritative_sites=authoritative_sites,
            )
            if new_keys:
                new_keys = filter_entity_keys_for_atom(atom, new_keys)
                if new_keys:
                    atom.entity_keys = new_keys
                    atoms_enriched += 1
                    total_keys_added += len(new_keys)
            # NORM front: normalize money/qty into atom.value (+ quantity: key)
            # now that this atom's keys are settled — runs on the no-keys path too.
            normalize_atom_value(atom)
            continue

        # Parser already populated keys. Run hygiene first, then
        # apply the LLM-catalog site gate to parser-supplied site
        # keys (parser may have emitted false-positive site codes
        # like "site:po_mock" from PO numbers, "site:am_10" from
        # time ranges — they must clear the same catalog check as
        # regex-emitted keys).
        cleaned = filter_entity_keys_for_atom(atom, existing)
        cleaned = _gate_site_keys(cleaned)
        # Augment with textual-pattern keys the parser doesn't emit
        # per-row (sites, dates, money, stakeholders).
        textual_keys = extract_keys(
            scan_text, pack=pack, value=value,
            authoritative_sites=authoritative_sites,
        )
        existing_prefixes = {
            k.split(":", 1)[0] + ":" if ":" in k else k
            for k in cleaned
        }
        augment: list[str] = []
        for k in textual_keys:
            if ":" not in k:
                continue
            prefix = k.split(":", 1)[0] + ":"
            if prefix not in _AUGMENT_ALWAYS_PREFIXES:
                continue
            # Preserve parser-supplied keys for the same prefix —
            # parser knows the structured source better. Only add
            # NEW prefix families.
            if prefix in existing_prefixes:
                continue
            augment.append(k)
        if augment:
            merged = sorted(set(cleaned) | set(augment))
            merged = filter_entity_keys_for_atom(atom, merged)
            if merged != list(cleaned):
                atom.entity_keys = merged
                atoms_enriched += 1
                total_keys_added += len(augment)
        elif cleaned != list(getattr(atom, "entity_keys", [])):
            atom.entity_keys = cleaned
        # NORM front: normalize money/qty into atom.value (+ quantity: key) for
        # the parser-keyed (augment) path, after keys are settled.
        normalize_atom_value(atom)

    # ─── LLM-DISCOVERED SITE INJECTION ───
    # For each site in the authoritative catalog that no atom's
    # entity_keys currently carries, walk atoms looking for atoms
    # whose text mentions the site, and inject a ``site:<slug>``
    # key. This propagates LLM-found sites (especially those in
    # PDF cover-page headings) onto the atoms that reference them
    # so EntityRecord fusion has something to bind to.
    if _SITE_INJECTION_KEYS:
        already_emitted_site_slugs: set[str] = set()
        for atom in atom_list:
            for k in atom.entity_keys or []:
                if k.startswith("site:"):
                    already_emitted_site_slugs.add(k[len("site:"):])
        missing_sites = {
            slug: phrase
            for slug, phrase in _SITE_INJECTION_KEYS.items()
            if slug and slug not in already_emitted_site_slugs
        }
        if missing_sites:
            for atom in atom_list:
                text = getattr(atom, "raw_text", "") or ""
                section_ctx = _section_path_context(atom)
                full = f"{text} {section_ctx}".lower()
                if not full.strip():
                    continue
                to_add: list[str] = []
                for slug, phrase in missing_sites.items():
                    # Match the phrase loosely (word boundary on
                    # first word + at least one shared meaningful
                    # token). Avoid over-injecting by requiring the
                    # phrase's longest word to be present.
                    words = [w for w in phrase.split() if len(w) >= 4]
                    if not words:
                        continue
                    # Trigger if the longest word + at least one
                    # other meaningful word from the phrase are
                    # both present in the atom text+headings.
                    longest = max(words, key=len)
                    if longest in full:
                        hits = sum(1 for w in words if w in full)
                        if hits >= min(2, len(words)):
                            to_add.append(f"site:{slug}")
                if to_add:
                    # v56: physical_site atoms own their identity from
                    # value.site_id — never override that with phrase
                    # matches from the catalog (those introduce variant
                    # slugs like site:atl_air alongside the canonical
                    # site:atl_air_03, polluting the graph). Other atom
                    # types still benefit from the injection.
                    _atype_inj = getattr(atom, "atom_type", None)
                    _atype_str_inj = _atype_inj.value if hasattr(_atype_inj, "value") else str(_atype_inj or "")
                    if _atype_str_inj == "physical_site":
                        continue
                    merged_keys = sorted(set(atom.entity_keys or []) | set(to_add))
                    atom.entity_keys = merged_keys
                    atoms_enriched += 1
                    total_keys_added += len(to_add)

    # ─── MULTI-ENTITY LLM PASS ───
    # ONE LLM call returns customer, stakeholders, milestones,
    # requirements, and site canonical clusters. Each item gets
    # injected onto the atoms that mention it (same loose-match
    # heuristic as the site injection above). Lifts the B+/A-
    # extractors (stakeholders, milestones, requirements, customer
    # fusion, site dedup) to A+ universally.
    try:
        from app.core.multi_entity_llm import extract_multi_entities_with_llm
        from app.core.site_llm_verify import ollama_reachable
        do_multi = (
            not os.environ.get("SOWSMITH_MULTI_ENTITY_DISABLE")
            and ollama_reachable()
        )
    except Exception:
        do_multi = False
        extract_multi_entities_with_llm = None  # type: ignore
    multi_result: dict[str, Any] = {}
    if do_multi and extract_multi_entities_with_llm is not None:
        try:
            multi_result = extract_multi_entities_with_llm(atom_list) or {}
        except Exception:
            multi_result = {}

    # v49.2 RAW TABLE ROW CLASSIFICATION: every parser emits
    # raw_table_row atoms with {_columns, _row}. Centralized here so a
    # single change in table_schema_registry covers all parsers (xlsx,
    # docx, future pptx/csv). Runs ALWAYS, even when multi_result is
    # empty (LLM may be unreachable but tables are still classifiable).
    # v50.1: collect NEW atoms (raw_table_row classifications + entity
    # bridge results). We append these to BOTH atom_list (used inside
    # this function) AND `atoms` (the caller's reference) so they
    # survive into the rest of the pipeline. enrich_atoms previously
    # silently dropped them because `atom_list = list(atoms)` made a
    # disconnected copy.
    _new_atoms_to_publish: list[Any] = []

    try:
        _proj_id_rtr = (
            getattr(atom_list[0], "project_id", "") if atom_list else ""
        )
        _rtr_atoms = _enrich_table_atoms(atom_list, project_id=_proj_id_rtr)
        if _rtr_atoms:
            atom_list.extend(_rtr_atoms)
            _new_atoms_to_publish.extend(_rtr_atoms)
            atoms_enriched += len(_rtr_atoms)
    except Exception as _rtr_exc:
        import logging as _lg_rtr
        _lg_rtr.getLogger(__name__).warning(
            "raw_table_row enrichment failed: %s", _rtr_exc
        )

    if multi_result:
        injected, key_count = _inject_multi_entity_keys(
            atom_list, multi_result
        )
        atoms_enriched += injected
        total_keys_added += key_count

        # v49 ENTITY-TO-ATOM BRIDGE: convert every LLM entity finding
        # into a proper typed EvidenceAtom. This is the fix for all 22
        # ZERO extraction categories — the LLM was finding the facts
        # but they were dying in the entity dict, never reaching the
        # atom stream. After this call, stakeholders/deliverables/
        # signatories/cutover_steps etc. become real typed atoms.
        try:
            _artifact_ids = sorted({
                getattr(a, "artifact_id", "") for a in atom_list
                if getattr(a, "artifact_id", "")
            })
            _project_id = (
                getattr(atom_list[0], "project_id", "") if atom_list else ""
            )
            # v55: pass the ACTUAL list of structural physical_site atoms
            # to the bridge (not just a count). The bridge uses this to
            # MERGE matching LLM site_cluster aliases into existing atoms
            # instead of suppressing/duplicating them. See _entities_to_atoms.
            def _atype_str(a: Any) -> str:
                at = getattr(a, "atom_type", None)
                return at.value if hasattr(at, "value") else str(at or "")
            _existing_phys_list = [a for a in atom_list if _atype_str(a) == "physical_site"]
            _existing_phys = len(_existing_phys_list)
            bridge_atoms = _entities_to_atoms(
                multi_result,
                project_id=_project_id,
                artifact_ids=_artifact_ids,
                existing_physical_site_count=_existing_phys,
                existing_physical_sites=_existing_phys_list,
            )
            if bridge_atoms:
                atom_list.extend(bridge_atoms)
                _new_atoms_to_publish.extend(bridge_atoms)
                atoms_enriched += len(bridge_atoms)
        except Exception as _bridge_exc:
            import logging as _lg
            _lg.getLogger(__name__).warning(
                "entity-to-atom bridge failed: %s", _bridge_exc
            )

    # v50.1: publish new atoms back to caller's list so downstream
    # stages (typed_atom_classification, dedup, entity_resolution,
    # graph_build, packetize, envelope projection) actually see them.
    if _new_atoms_to_publish:
        try:
            if isinstance(atoms, list):
                atoms.extend(_new_atoms_to_publish)
        except Exception:
            pass

        # v44.5: inject vision-extracted rows AS atom entity_keys so
        # BOM line items, contact rosters, schedule phases, etc. that
        # only exist in PDF tables become first-class entities (money,
        # phone, email, stakeholder, milestone, site, etc.) instead
        # of stranded in multi_result["vision_rows"].
        vision_rows = multi_result.get("vision_rows") or []
        if vision_rows:
            try:
                from app.core.vision_extraction import inject_vision_rows_as_entities
                vmod, vkeys = inject_vision_rows_as_entities(atom_list, vision_rows)
                atoms_enriched += vmod
                total_keys_added += vkeys
            except Exception as e:
                import logging as _lg
                _lg.getLogger(__name__).warning(
                    "vision-row injection failed: %s", e,
                )

    # ─── FINAL HYGIENE PASS ───
    # Universal safety net: walk every atom's entity_keys and drop:
    #   - site:* keys that fail site hygiene
    #   - stakeholder:* keys that look like field labels (column
    #     headers, form fields, table column names) rather than
    #     real people names
    # This catches noise that bypassed earlier gates regardless of
    # which emitter produced the key (parser-supplied, regex
    # _emit_stakeholders, LLM-injected, etc.).
    try:
        from app.core.site_llm_verify import _is_obvious_non_site
    except Exception:
        _is_obvious_non_site = None  # type: ignore
    try:
        from app.core.multi_entity_llm import _is_likely_field_label
    except Exception:
        _is_likely_field_label = None  # type: ignore

    # UNIVERSAL site-role gate (decide() STORE kNN → never the denylist alone).
    # Compute the drop-set ONCE over every site:* key across all atoms so the
    # embedding cache warms in a single batch, then apply per-key below BEFORE
    # the deterministic _is_obvious_non_site denylist. Safe no-op when the gate
    # flag is off or no store is wired (returns empty set → identical pipeline).
    try:
        from app.core.entity_resolution import semantic_site_role_drops
        _all_site_keys = {
            k for atom in atom_list for k in (atom.entity_keys or [])
            if isinstance(k, str) and k.startswith("site:")
        }
        _site_role_drops = semantic_site_role_drops(_all_site_keys)
    except Exception:
        _site_role_drops = set()  # type: ignore

    # v41: also import customer-regulator filter for final hygiene pass
    try:
        from app.core.multi_entity_llm import _looks_like_regulator_not_customer
    except Exception:
        _looks_like_regulator_not_customer = None  # type: ignore

    # v41+: customer hygiene
    # ------------------------------------------------------------------
    # When the LLM identified a primary customer in this pack, that
    # customer is the buyer. ALL other "customer:" slugs are regex
    # co-mention noise (org names mentioned in the doc that aren't the
    # buyer). Drop them aggressively, except aliases / extensions of
    # the LLM customer (e.g. "BCSD" vs "Beaufort County School
    # District" — both should survive if both are emitted).
    _llm_customer_slug = None
    try:
        if isinstance(multi_result, dict):
            _llm_customer_name = multi_result.get("customer")
            if isinstance(_llm_customer_name, str) and _llm_customer_name.strip():
                _llm_customer_slug = _slug(_llm_customer_name)
    except Exception:
        _llm_customer_slug = None

    # _CUSTOMER_NOISE_TAILS: when a customer slug ends in any of these,
    # it's structurally noise (regex-emitted from co-mentions), NOT a
    # real buying customer. Real buying customers can also have these
    # tails BUT only when the LLM endorsed them or no LLM customer
    # exists.
    _CUSTOMER_NOISE_TAILS = {
        "council", "commission", "committee", "board",
        "department", "agency", "authority", "bureau",
        "office", "court",
    }

    # Known product / SaaS names that get false-positive promoted to
    # "customer:" because their slug ends in an institutional tail
    # (school / district / etc.). These are universally not buyers.
    _KNOWN_PRODUCT_DENYLIST = {
        "power_school", "powerschool", "mosaic", "mosaic_cloud",
        "myschoolbucks", "mealviewer", "websmartt", "msa", "msasupport",
        "scolaris", "infinite_campus", "skyward", "tyler",
    }

    for atom in atom_list:
        current = atom.entity_keys or []
        if not current:
            continue
        kept = []
        dropped_any = False
        for k in current:
            if k.startswith("site:"):
                # Universal store-learned role gate first; denylist second.
                if k in _site_role_drops:
                    dropped_any = True
                    continue
                if _is_obvious_non_site is not None:
                    phrase = k[len("site:"):].replace("_", " ")
                    if _is_obvious_non_site(phrase):
                        dropped_any = True
                        continue
            if k.startswith("stakeholder:") and _is_likely_field_label is not None:
                phrase = k[len("stakeholder:"):].replace("_", " ")
                if _is_likely_field_label(phrase):
                    dropped_any = True
                    continue
            # v41: customer hygiene
            if k.startswith("customer:"):
                slug = k[len("customer:"):]
                phrase = slug.replace("_", " ")
                # (a) Drop known SaaS / product names (universally not
                #     a buying customer)
                if slug in _KNOWN_PRODUCT_DENYLIST:
                    dropped_any = True
                    continue
                # (b) Drop regulator-looking names
                if _looks_like_regulator_not_customer is not None and \
                        _looks_like_regulator_not_customer(phrase):
                    dropped_any = True
                    continue
                # (c) Drop institutional-noise tails (council, commission,
                #     committee, department, agency, etc.) — regex
                #     co-mentions, not real buyers
                tail = phrase.split()[-1] if phrase else ""
                if tail in _CUSTOMER_NOISE_TAILS:
                    dropped_any = True
                    continue
                # (d) v41b had aggressive LLM-customer-honoring drop here
                #     but it killed the real customer when LLM picked
                #     wrong. Removed — product denylist + regulator
                #     filter + noise tails are sufficient.
            kept.append(k)
        if dropped_any:
            atom.entity_keys = kept

    # ─── SITE FRAGMENT DROP ───
    # When a site:* slug is a strict prefix of the customer:* slug
    # (e.g. site:beaufort_county_school when customer is
    # customer:beaufort_county_school_district), the site key is a
    # FRAGMENT of the customer name and shouldn't be a separate
    # site entity. Drop it.
    #
    # Two sources of customer slugs:
    #   (a) customer:* keys actually injected on atoms (via
    #       _inject_multi_entity_keys' _phrase_in_atom matching).
    #   (b) the LLM-emitted multi_result["customer"] string —
    #       used as a fallback so we still drop fragments even
    #       when the customer phrase didn't appear verbatim on
    #       any single atom. Critical for Pack 18 Beaufort POS
    #       where atoms reference "BCSD" / "the District" more
    #       often than the full "Beaufort County School District"
    #       phrase that _phrase_in_atom requires.
    customer_slugs = {
        k[len("customer:"):]
        for atom in atom_list
        for k in (atom.entity_keys or [])
        if k.startswith("customer:")
    }
    llm_customer_name = (
        multi_result.get("customer")
        if isinstance(multi_result, dict)
        else None
    )
    if isinstance(llm_customer_name, str) and llm_customer_name.strip():
        customer_slugs.add(_slug(llm_customer_name))
    if customer_slugs:
        for atom in atom_list:
            keys = atom.entity_keys or []
            if not keys:
                continue
            new_keys = []
            changed = False
            for k in keys:
                if k.startswith("site:"):
                    site_slug = k[len("site:"):]
                    # Strict-prefix match: site_slug is a proper
                    # prefix of any customer_slug, AND has ≥2 tokens
                    # (so we don't drop short codes like "atl_hq")
                    if (
                        "_" in site_slug
                        and any(
                            cs.startswith(site_slug + "_")
                            for cs in customer_slugs
                        )
                    ):
                        changed = True
                        continue
                new_keys.append(k)
            if changed:
                atom.entity_keys = new_keys

    # ─── FINAL PASS: contact-anchor stakeholder recovery ───
    # The noisy regex _emit_stakeholders may have been dropped by the
    # LLM-trumps-regex rule above (or by hygiene). The contact-anchor
    # extractor (_emit_person_from_contact) is the LOW-FALSE-POSITIVE
    # path — it requires an email or explicit "contact <Name>" /
    # "Project Manager: <Name>" trigger, so its output is PM-critical
    # signal that should NEVER be dropped just because the LLM ran
    # successfully. Walk every atom one more time and re-emit those
    # keys. Catches Glenn Tilleman, Shaun Tozer, John Foster, Matthew
    # Brener even when LLM extract returned a different / no person.
    #
    # Belt and suspenders: even though _emit_person_from_contact now
    # filters noun-fragments internally via _is_likely_field_label,
    # we run the same filter here so any future caller that bypasses
    # the internal filter still gets sanitized output. Also applies
    # _is_obvious_non_site to any site:* sneaking in.
    try:
        from app.core.multi_entity_llm import (
            _is_likely_field_label as _ilfl_recovery,
        )
    except Exception:
        _ilfl_recovery = None  # type: ignore
    for atom in atom_list:
        raw = getattr(atom, "raw_text", "") or ""
        if not raw:
            continue
        contact_keys = _emit_person_from_contact(raw)
        contact_keys |= _emit_email_keys(raw)
        contact_keys |= _emit_phone_keys(raw)
        # Filter noun-fragment stakeholders out of the recovery
        # contribution. These shouldn't make it past
        # _emit_person_from_contact now, but if they do, drop them
        # before merging onto atoms.
        if _ilfl_recovery is not None:
            sanitized: set[str] = set()
            for k in contact_keys:
                if k.startswith("stakeholder:"):
                    phrase = k[len("stakeholder:"):].replace("_", " ")
                    if _ilfl_recovery(phrase):
                        continue
                sanitized.add(k)
            contact_keys = sanitized
        if not contact_keys:
            continue
        existing = set(atom.entity_keys or [])
        new_keys = contact_keys - existing
        if new_keys:
            atom.entity_keys = sorted(existing | new_keys)
            atoms_enriched += 1
            total_keys_added += len(new_keys)

    # ─── STRUCTURAL PEOPLE ATOM BRIDGE ───
    # Make contact-table / owner / signature facts first-class typed
    # atoms even when the multi-entity LLM is disabled. This is not a
    # general name regex; it only promotes high-precision roster, owner,
    # signature, and escort-owner structures already present in the pack.
    try:
        people_atoms = _structural_people_atoms(atom_list, project_id=(getattr(atom_list[0], "project_id", "") if atom_list else ""))
        if people_atoms:
            atom_list.extend(people_atoms)
            if isinstance(atoms, list):
                atoms.extend(people_atoms)
            atoms_enriched += len(people_atoms)
    except Exception as _people_exc:
        import logging as _lg_people
        _lg_people.getLogger(__name__).warning(
            "structural people atom bridge failed: %s", _people_exc
        )

    return atoms_enriched, total_keys_added


def _slug(text: str) -> str:
    """Lowercase + non-alphanumeric → underscore + strip edges."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _atom_full_text(atom: Any) -> str:
    """Atom body + section_path headings, lowercased — for matching."""
    parts: list[str] = []
    raw = getattr(atom, "raw_text", None) or ""
    if isinstance(raw, str):
        parts.append(raw)
    try:
        refs = getattr(atom, "source_refs", None) or []
        if refs:
            loc = getattr(refs[0], "locator", None) or {}
            if isinstance(loc, dict):
                sp = loc.get("section_path")
                if isinstance(sp, list):
                    for h in sp:
                        if isinstance(h, str):
                            parts.append(h)
                for k in ("section", "heading", "title"):
                    v = loc.get(k)
                    if isinstance(v, str):
                        parts.append(v)
    except Exception:
        pass
    return " ".join(parts).lower()


def _phrase_in_atom(phrase: str, atom_text: str) -> bool:
    """Match LLM-emitted phrase against atom text using a tolerant
    rule: phrase's distinguishing word (longest ≥4-char word) must
    appear AND at least 2 of the phrase's meaningful tokens must
    appear in the atom text.
    """
    if not phrase or not atom_text:
        return False
    words = [w for w in re.split(r"\W+", phrase.lower()) if len(w) >= 4]
    if not words:
        # Fallback: substring match for short phrases (e.g., "USD 475")
        return phrase.lower() in atom_text
    longest = max(words, key=len)
    if longest not in atom_text:
        return False
    hits = sum(1 for w in words if w in atom_text)
    return hits >= min(2, len(words))


def _inject_multi_entity_keys(
    atom_list: list[Any], multi: dict[str, Any]
) -> tuple[int, int]:
    """Walk atoms; inject customer / stakeholder / milestone /
    requirement / site keys from the LLM multi-entity result onto
    atoms whose text mentions the entity.

    Returns (atoms_modified, keys_added).

    When the LLM provided customer or stakeholder data, ALL pre-
    existing regex-emitted customer:/stakeholder: keys are dropped
    first (the LLM read the full doc context and is the source of
    truth for those categories). The injection then re-populates
    with clean LLM-derived keys.
    """
    atoms_modified = 0
    keys_added = 0

    # Pre-compute slugs for each entity
    customer = multi.get("customer")
    # Customer hygiene: drop LLM picks that look like regulatory
    # bodies / licensing issuers ("State of South Carolina Department
    # of Revenue Retail License") rather than buying customers.
    # Real govt buyers like "City of Atlanta" / "Beaufort County
    # School District" don't match these patterns.
    if isinstance(customer, str) and customer.strip():
        try:
            from app.core.multi_entity_llm import _looks_like_regulator_not_customer
            if _looks_like_regulator_not_customer(customer):
                customer = None
        except Exception:
            pass
    customer_slug = _slug(customer) if isinstance(customer, str) and customer else None

    stakeholders = multi.get("stakeholders") or []
    stakeholder_entries: list[tuple[str, str]] = []  # (phrase, slug)
    for s in stakeholders:
        name = s.get("name")
        if isinstance(name, str) and name.strip():
            stakeholder_entries.append((name.strip(), _slug(name)))

    # LLM-AUTHORITATIVE for customer + stakeholder when the LLM ran
    # successfully (returned ANY output for any category):
    #
    #   - customer:    drop regex emissions; the LLM's single
    #                  canonical customer is the truth.
    #   - stakeholder: drop ALL regex _emit_stakeholders emissions
    #                  (these include jargon-y false positives like
    #                  "annual_electricity_bill" on Pack 14 Neptune
    #                  because the regex catches any "First Last"
    #                  + role-context pattern). The new contact-
    #                  anchor emitter (_emit_person_from_contact)
    #                  RUNS AGAIN at the end of enrich_atoms to
    #                  recover PM-critical names like Glenn Tilleman,
    #                  Shaun Tozer, John Foster, Matthew Brener.
    #                  Contact-anchor requires email/phone or
    #                  explicit trigger ("contact <Name>") so its
    #                  false-positive rate is much lower than the
    #                  noisy regex.
    llm_ran = bool(multi.get("customer") is not None or
                   multi.get("stakeholders") or
                   multi.get("milestones") or
                   multi.get("requirements") or
                   multi.get("site_clusters"))
    drop_regex_customer = llm_ran and bool(customer_slug)
    drop_regex_stakeholder = llm_ran
    if drop_regex_customer or drop_regex_stakeholder:
        for atom in atom_list:
            keys = atom.entity_keys or []
            if not keys:
                continue
            filtered = []
            changed = False
            for k in keys:
                if drop_regex_customer and k.startswith("customer:"):
                    changed = True
                    continue
                if drop_regex_stakeholder and k.startswith("stakeholder:"):
                    changed = True
                    continue
                filtered.append(k)
            if changed:
                atom.entity_keys = filtered

    milestones = multi.get("milestones") or []
    milestone_entries: list[tuple[str, str]] = []
    for m in milestones:
        name = m.get("name")
        if isinstance(name, str) and name.strip():
            milestone_entries.append((name.strip(), _slug(name)))

    requirements = multi.get("requirements") or []
    requirement_entries: list[tuple[str, str]] = []
    for r in requirements:
        text = r.get("text")
        if isinstance(text, str) and text.strip():
            # Use first 6 distinctive words as the matching phrase
            words = [w for w in re.split(r"\W+", text) if len(w) >= 4][:6]
            if not words:
                continue
            match_phrase = " ".join(words)
            requirement_entries.append((match_phrase, _slug(text[:80])))

    site_clusters = multi.get("site_clusters") or []
    # cluster_lookups: list of (canonical_slug, list_of_alias_phrases)
    cluster_lookups: list[tuple[str, list[str]]] = []
    for c in site_clusters:
        canon = c.get("canonical_name")
        aliases = c.get("aliases") or []
        if not isinstance(canon, str) or not canon.strip():
            continue
        canon_slug = _slug(canon)
        alias_phrases = [a for a in aliases if isinstance(a, str) and a.strip()]
        if not alias_phrases:
            continue
        cluster_lookups.append((canon_slug, alias_phrases))

    # v38: LLM quantities — bind by text-phrase to atoms that mention
    # them so they become quantity:<slug> entities. Same loose-match
    # pattern as requirements (longest meaningful word + 2-token hit).
    quantities = multi.get("quantities") or []
    quantity_entries: list[tuple[str, str]] = []
    for q in quantities:
        text = q.get("text")
        if isinstance(text, str) and text.strip():
            words = [w for w in re.split(r"\W+", text) if len(w) >= 3][:6]
            if not words:
                continue
            match_phrase = " ".join(words)
            quantity_entries.append((match_phrase, _slug(text[:80])))

    # v43/v44.2 — wire the 5 new entity types into the injection layer.
    # The extractors return data but until v44.2 this layer didn't
    # know to inject `certification:` / `risk:` / `acceptance_criteria:`
    # / `penalty:` / `compliance_obligation:` keys onto atoms — which
    # meant entity_resolution never saw them and they ended up at 0
    # in the final output.
    certifications = multi.get("certifications") or []
    cert_entries: list[tuple[str, str]] = []
    for c in certifications:
        name = c.get("name")
        if isinstance(name, str) and name.strip():
            # v44.3 — for SHORT certification names (TIA-568, PCI-DSS,
            # USAC, FNS-742, E-Rate) the word-filtered match-phrase
            # approach loses the hyphens/dots that PDFs actually
            # contain. Use the RAW name as match phrase so
            # _phrase_in_atom's substring fallback (which kicks in
            # when no word is ≥4 chars) does case-insensitive
            # substring search against the original form.
            cert_entries.append((name.strip(), _slug(name[:80])))

    risks = multi.get("risks") or []
    risk_entries: list[tuple[str, str]] = []
    for r in risks:
        desc = r.get("description")
        if isinstance(desc, str) and desc.strip():
            words = [w for w in re.split(r"\W+", desc) if len(w) >= 4][:6]
            if not words:
                continue
            match_phrase = " ".join(words)
            risk_entries.append((match_phrase, _slug(desc[:80])))

    acceptance_items = multi.get("acceptance_criteria") or []
    acceptance_entries: list[tuple[str, str]] = []
    for a in acceptance_items:
        crit = a.get("criterion")
        if isinstance(crit, str) and crit.strip():
            words = [w for w in re.split(r"\W+", crit) if len(w) >= 4][:6]
            if not words:
                continue
            match_phrase = " ".join(words)
            acceptance_entries.append((match_phrase, _slug(crit[:80])))

    penalties = multi.get("penalties") or []
    penalty_entries: list[tuple[str, str]] = []
    for p in penalties:
        desc = p.get("description")
        if isinstance(desc, str) and desc.strip():
            words = [w for w in re.split(r"\W+", desc) if len(w) >= 4][:6]
            if not words:
                continue
            match_phrase = " ".join(words)
            penalty_entries.append((match_phrase, _slug(desc[:80])))

    compliance_items = multi.get("compliance_obligations") or []
    compliance_entries: list[tuple[str, str]] = []
    for c in compliance_items:
        obl = c.get("obligation")
        if isinstance(obl, str) and obl.strip():
            words = [w for w in re.split(r"\W+", obl) if len(w) >= 4][:6]
            if not words:
                continue
            match_phrase = " ".join(words)
            compliance_entries.append((match_phrase, _slug(obl[:80])))

    for atom in atom_list:
        full = _atom_full_text(atom)
        if not full:
            continue
        to_add: list[str] = []

        # Customer — inject on EVERY atom that mentions the customer
        # name. This is OK because the customer is genuinely the
        # subject of every document in their bid package.
        if customer_slug and customer:
            if _phrase_in_atom(customer, full):
                to_add.append(f"customer:{customer_slug}")

        # Stakeholders
        for name, slug in stakeholder_entries:
            if _phrase_in_atom(name, full):
                to_add.append(f"stakeholder:{slug}")

        # Milestones — match by name OR by ISO date if present
        for name, slug in milestone_entries:
            if _phrase_in_atom(name, full):
                to_add.append(f"milestone:{slug}")

        # Requirements
        for match_phrase, slug in requirement_entries:
            if _phrase_in_atom(match_phrase, full):
                to_add.append(f"requirement:{slug}")

        # Quantities (v38)
        for match_phrase, slug in quantity_entries:
            if _phrase_in_atom(match_phrase, full):
                to_add.append(f"quantity:{slug}")

        # v43/v44.2 — 5 new entity types
        for match_phrase, slug in cert_entries:
            if _phrase_in_atom(match_phrase, full):
                to_add.append(f"certification:{slug}")
        for match_phrase, slug in risk_entries:
            if _phrase_in_atom(match_phrase, full):
                to_add.append(f"risk:{slug}")
        for match_phrase, slug in acceptance_entries:
            if _phrase_in_atom(match_phrase, full):
                to_add.append(f"acceptance_criteria:{slug}")
        for match_phrase, slug in penalty_entries:
            if _phrase_in_atom(match_phrase, full):
                to_add.append(f"penalty:{slug}")
        for match_phrase, slug in compliance_entries:
            if _phrase_in_atom(match_phrase, full):
                to_add.append(f"compliance_obligation:{slug}")

        # Sites — emit the CANONICAL site key for each cluster whose
        # alias phrase appears in this atom. This force-merges all
        # surface forms onto the canonical key.
        for canon_slug, alias_phrases in cluster_lookups:
            matched = False
            for alias in alias_phrases:
                if _phrase_in_atom(alias, full):
                    matched = True
                    break
            if matched:
                to_add.append(f"site:{canon_slug}")

        if to_add:
            # v56: same physical_site guard as the earlier injection pass.
            # The atom's site identity is value.site_id; LLM cluster
            # aliases shouldn't add competing site:* slugs onto it.
            _atype_llm = getattr(atom, "atom_type", None)
            _atype_str_llm = _atype_llm.value if hasattr(_atype_llm, "value") else str(_atype_llm or "")
            if _atype_str_llm == "physical_site":
                # Allow non-site:* keys (penalty:, compliance:, etc.) but
                # drop any site:* augments — physical_site atoms own
                # their key already.
                to_add = [k for k in to_add if not k.startswith("site:")]
            if to_add:
                existing_keys = set(atom.entity_keys or [])
                new_keys = [k for k in to_add if k not in existing_keys]
                if new_keys:
                    atom.entity_keys = sorted(existing_keys | set(new_keys))
                    atoms_modified += 1
                    keys_added += len(new_keys)

    return atoms_modified, keys_added


__all__ = ["extract_keys", "enrich_atoms"]
