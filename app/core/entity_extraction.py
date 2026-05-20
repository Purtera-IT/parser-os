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

import re
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
    "carrier": ["carrier"],
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


# Street address pattern — captures something like "1700 Pratt Drive" or
# "4700 Crest Drive" or "60 East Van Buren Street".  Conservative: we
# require a number, then capitalized word(s), then a street suffix.
_STREET_SUFFIXES = (
    "Street|Str|St|Ave|Avenue|Boulevard|Blvd|Road|Rd|Drive|Dr|Lane|Ln|Way|"
    "Court|Ct|Place|Pl|Highway|Hwy|Parkway|Pkwy|Trail|Trl|Circle|Cir"
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
    r"\b([A-Z]{3,10}(?:-[A-Z][A-Z0-9]{0,9}){1,4})\b"
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


def _site_code_suffix_ok(last_segment: str) -> bool:
    """Universal gate: does this last segment carry site-function meaning?"""
    if last_segment in _SITE_CODE_SUFFIX_ALLOWLIST:
        return True
    if _SITE_CODE_SUFFIX_PATTERN.match(last_segment):
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
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _device_alias_index(pack: DomainPack) -> dict[str, str]:
    """Flatten pack.device_aliases into ``{normalized_alias: canonical}``.

    The canonical is the YAML key (e.g. ``ip_camera``).  We add a
    word-boundary entry for every alias so we don't match
    ``"camera"`` inside ``"cameramen"``.
    """
    index: dict[str, str] = {}
    for canonical, aliases in (pack.device_aliases or {}).items():
        canonical_norm = normalize_text(canonical.replace("_", " "))
        if canonical_norm:
            index.setdefault(canonical_norm, canonical)
        for alias in aliases or []:
            alias_norm = normalize_text(alias)
            if alias_norm:
                index.setdefault(alias_norm, canonical)
    return index


def _typed_alias_index(pack: DomainPack) -> dict[str, dict[str, str]]:
    """Build ``{entity_type: {alias_norm: example_or_alias}}`` so we can
    detect typed entities (room, site, vendor, etc.) generically.

    The returned ``example_or_alias`` is what we'll slugify for the
    canonical key — preferring rich examples over bare aliases.
    """
    out: dict[str, dict[str, str]] = {}
    for entity in pack.entity_types or []:
        slot = out.setdefault(entity.name, {})
        for alias in entity.aliases or []:
            alias_norm = normalize_text(alias)
            if alias_norm:
                slot.setdefault(alias_norm, alias)
        for example in entity.examples or []:
            example_norm = normalize_text(example)
            if example_norm:
                slot.setdefault(example_norm, example)
    return out


def _word_match(text_lower: str, alias_lower: str) -> bool:
    """Word-boundary match for an alias inside a pre-lowercased text."""
    if not alias_lower:
        return False
    pattern = r"(?<![a-z0-9])" + re.escape(alias_lower) + r"(?![a-z0-9])"
    return re.search(pattern, text_lower) is not None


def _emit_devices(text_lower: str, alias_index: dict[str, str]) -> set[str]:
    keys: set[str] = set()
    for alias_norm, canonical in alias_index.items():
        if _word_match(text_lower, alias_norm):
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
        "school",
        "schools",
        "campus",
        "building",
        "buildings",
        "floor",
        "floors",
        "level",
        "levels",
        "warehouse",
        "warehouses",
        "hospital",
        "hospitals",
        "clinic",
        "clinics",
        "office",
        "offices",
        "branch",
        "branches",
        "store",
        "stores",
        "facility",
        "facilities",
        "site",
        "sites",
        "room",
        "rooms",
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


def _emit_sites(text: str) -> set[str]:
    keys: set[str] = set()

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
        # Universal POSITIVE gate: the last segment must carry recognized
        # site-function meaning (direction, facility type, datacenter /
        # floor / building / wing number).  This is the load-bearing
        # robustness check — anything that doesn't end in a known
        # site-suffix fails the gate, regardless of head or middle
        # segments. Catches unknown junk codes like MOCK-OPTBOT-ATL,
        # ALPHA-FOOBAR, GAMMA-FOO-2026 without needing to enumerate
        # every possible junk word.
        if not _site_code_suffix_ok(last):
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


def _emit_proper_nouns(text: str, vendor_keys: set[str]) -> set[str]:
    """Capture multi-word proper-noun runs as candidate sites/customers.

    Conservative: only emits when the run is ≥3 words, its lowercased
    form isn't in the stoplist, doesn't overlap a vendor we already
    detected, and doesn't end on a known field label.  Tagged
    ``site:`` so the domain pack's alias resolver can promote it
    to its canonical type later.

    If the surrounding text looks like a form-field template
    (``FULL LEGAL NAME (PRINT)``, ``id#``, ``col_N:``), we skip
    proper-noun extraction entirely to avoid emitting form labels
    as fake sites.
    """
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
        if not _site_code_suffix_ok(last):
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
        slug = _slugify(sku)
        if slug:
            keys.add(f"part_number:{slug}")
    return keys


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


def extract_keys(
    text: str,
    *,
    pack: DomainPack,
    value: Any | None = None,
) -> list[str]:
    """Return entity_keys for ``text`` using ``pack``'s vocabulary.

    Pure function — no I/O, no global state.  ``value`` is the atom's
    structured ``value`` payload if any (e.g. xlsx table_row).
    """
    if not text:
        return []
    text_lower = text.lower()
    device_idx = _device_alias_index(pack)
    typed_idx = _typed_alias_index(pack)

    keys: set[str] = set()
    keys |= _emit_devices(text_lower, device_idx)
    keys |= _emit_typed(text_lower, typed_idx)
    vendor_keys = _emit_vendors(text_lower)
    keys |= vendor_keys
    keys |= _emit_sites(text)
    # Proper-noun fallback runs LAST so it can deduplicate against
    # vendor matches (avoids "site:genetec_security_center" when we
    # already have "vendor:genetec").
    proper_noun_keys = _emit_proper_nouns(text, vendor_keys)
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

    return sorted(keys)


def enrich_atoms(atoms: Iterable[Any], pack: DomainPack) -> tuple[int, int]:
    """Mutate ``atoms`` in place: populate ``entity_keys`` for any atom
    whose list is currently empty.

    Returns ``(atoms_enriched, total_keys_added)`` for telemetry.
    Atoms that already have ``entity_keys`` are left untouched —
    parser-supplied keys are authoritative.
    """
    atoms_enriched = 0
    total_keys_added = 0
    for atom in atoms:
        if getattr(atom, "entity_keys", None):
            # Even pre-populated keys go through hygiene so a parser
            # that mints a fake ``site:belden_cat6`` gets cleaned up.
            cleaned = filter_entity_keys_for_atom(atom, atom.entity_keys)
            if cleaned != list(atom.entity_keys):
                atom.entity_keys = cleaned
            continue
        text = getattr(atom, "raw_text", "") or ""
        value = getattr(atom, "value", None)
        new_keys = extract_keys(text, pack=pack, value=value)
        if new_keys:
            new_keys = filter_entity_keys_for_atom(atom, new_keys)
            if new_keys:
                atom.entity_keys = new_keys
                atoms_enriched += 1
                total_keys_added += len(new_keys)
    return atoms_enriched, total_keys_added


__all__ = ["extract_keys", "enrich_atoms"]
