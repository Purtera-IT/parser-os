"""Is this device-alias hit really that device, or an ordinary word?

The device vocabulary carries short generic words because they ARE device
names in the right sentence: "cabinet" is a rack in "12U cabinet in the IDF",
"panel" is an access controller beside a door, "tower" is a desktop next to a
keyboard. In a construction-camera install they are not: "mount the camera on
the electrical cabinet", "the tower crane", "20 amp breaker", "ship via UPS
ground". Word-boundary matching alone tagged every one of those, so a deal
about cameras on poles came back full of ``device:rack`` and friends.

Each rule here is narrow and names the ordinary-word reading it rejects. A
rule never removes an alias from the vocabulary, it only declines ONE match
whose surroundings say it is something else, so the same word keeps tagging
when it is used as a device name.

Rule kinds, per alias (optionally per canonical):

* ``needs``        -- the word is too generic to stand alone; the atom text
                       must also carry a cue from that device's world.
* ``not_after``    -- the word right before it makes it something else
                       ("kitchen cabinet", "patch panel", "bike rack").
* ``not_before``   -- the text right after it does ("cam lock", "speaker 1:").
* ``unit``         -- after a number it is a unit of measure ("50 pcs",
                       "25 lb", "10 meter", "20 amp").
* ``verb``         -- used as a verb ("will monitor the site", "switch the
                       power off", "display the feed").

Plus two atom-level guards used by the enricher: legal boilerplate and
chatter-flagged atoms never mint device keys.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_WORD_BEFORE_RE = re.compile(r"([a-z0-9][a-z0-9'&]*)[\s/-]*$")


@dataclass(frozen=True)
class _Rule:
    needs: "re.Pattern[str] | None" = None
    not_after: frozenset[str] = field(default_factory=frozenset)
    not_before: "re.Pattern[str] | None" = None
    unit: bool = False
    unit_singular_only: bool = False  # "2 amps" may be two amplifiers
    verb: "re.Pattern[str] | None" = None  # objects that follow a verb use
    canonicals: frozenset[str] | None = None  # None = any canonical
    # Text in another world (``veto``) and none of the device's own
    # (``veto_unless``): "send the invoices to AP" is accounts payable.
    veto: "re.Pattern[str] | None" = None
    veto_unless: "re.Pattern[str] | None" = None


def _cue(*words: str) -> "re.Pattern[str]":
    return re.compile(r"(?<![a-z0-9])(?:" + "|".join(words) + r")(?![a-z0-9])")


# ── cue vocabularies ────────────────────────────────────────────────────
_IT_RACK_CUES = _cue(
    r"mdf", r"idf", r"\d{1,2}\s?r?u", r"19(?:\"|”|'')", r"19[- ]?(?:inch|in\.?)",
    r"racks?", r"server(?:s)?", r"network(?:ing)?", r"data (?:closet|room|center|centre|cabinet)",
    r"telecom", r"comms?", r"communications?", r"patch panels?", r"switch(?:es)?",
    r"pdus?", r"ups", r"routers?", r"firewalls?", r"fiber", r"cabling", r"cat ?6a?", r"cat ?5e?",
    r"nvr", r"ethernet", r"poe", r"structured cabling", r"equipment room", r"lan", r"wan",
    r"head ?end", r"core switch(?:es)?", r"uplinks?", r"two[- ]post", r"four[- ]post", r"2[- ]post", r"4[- ]post",
)
_ACCESS_CONTROL_CUES = _cue(
    r"doors?", r"access control", r"card readers?", r"readers?", r"badges?", r"credentials?",
    r"locks?", r"maglocks?", r"mag locks?", r"strikes?", r"egress", r"rex", r"acs", r"lenel",
    r"genetec", r"hid", r"mercury", r"keyscan", r"brivo", r"openpath", r"verkada", r"s2",
    r"intrusion", r"alarm", r"fire alarm", r"facp", r"keypad", r"zones?",
)
_COMPUTER_CUES = _cue(
    r"pcs?", r"computers?", r"desktops?", r"cpus?", r"workstations?", r"dell", r"hp",
    r"lenovo", r"optiplex", r"thinkcentre", r"keyboards?", r"mouse", r"mice", r"monitors?",
    r"imacs?", r"laptops?", r"notebooks?", r"chromebooks?", r"usb(?:-c)?", r"thunderbolt",
    r"docking", r"hdmi", r"displayport", r"windows", r"imaging", r"image", r"surface",
    r"mini[- ]tower", r"mid[- ]tower", r"full[- ]tower", r"sff", r"ram", r"ssd",
)
_DOOR_HARDWARE_CUES = _cue(
    r"doors?", r"locks?", r"latch(?:es)?", r"frames?", r"access", r"hardware", r"egress",
    r"card", r"readers?", r"fail[- ]?(?:safe|secure)", r"12 ?v(?:dc)?", r"24 ?v(?:dc)?",
    r"hes", r"von duprin", r"mortise", r"electrified", r"keeper", r"jamb", r"strike plate",
)
_AUDIO_CUES = _cue(
    r"speakers?", r"subwoofers?", r"amps?", r"amplifiers?", r"audio", r"sound", r"dsp",
    r"bass", r"woofer", r"mixer", r"microphones?", r"mics?", r"pa", r"paging", r"av",
    r"a/v", r"qsc", r"biamp", r"bose", r"jbl", r"crown", r"ohms?", r"watts?", r"70 ?v",
)
_BMS_CUES = _cue(
    r"bms", r"bas", r"niagara", r"jace", r"tridium", r"bacnet", r"modbus", r"ddc",
    r"hvac", r"ahus?", r"vavs?", r"controllers?", r"supervisory", r"head ?end", r"metasys",
    r"ecostruxure", r"desigo", r"webctrl", r"alerton",
)
_POS_CUES = _cue(
    r"pos", r"point of sale", r"cash", r"till", r"receipt", r"payment", r"pin ?pad",
    r"checkout", r"terminal", r"lane", r"retail", r"store",
)
_FIRE_ALARM_CUES = _cue(
    r"strobes?", r"horns?", r"fire", r"alarm", r"notification", r"facp", r"nac",
    r"candela", r"cd", r"smoke", r"pull stations?", r"ada",
)
_STORAGE_MEDIA_CUES = _cue(
    r"backup", r"backups", r"lto\d?", r"librar(?:y|ies)", r"drives?", r"cartridges?",
    r"media", r"data", r"tape library", r"archive", r"degauss\w*", r"shred\w*", r"sanitiz\w*",
)
_NET_TAP_CUES = _cue(
    r"network tap", r"fiber", r"coax", r"rf", r"splitters?", r"taps? (?:port|module)",
    r"span", r"packet", r"monitoring port", r"optical", r"das",
)
_DAS_CUES = _cue(r"das", r"antennas?", r"bda", r"rf", r"donor", r"remote units?", r"in-building", r"cellular", r"signal")
_WIRELESS_CUES = _cue(
    r"wi-?fi", r"wireless", r"ssid", r"poe", r"ceiling", r"mount(?:ed|ing)?", r"coverage",
    r"802\.11\w*", r"antennas?", r"heat ?maps?", r"survey", r"wlc", r"controller",
    r"switch(?:es)?", r"meraki", r"aruba", r"ruckus", r"ubiquiti", r"mist", r"access points?",
    r"wap", r"aps", r"\d+\s*aps?", r"radios?", r"band", r"ghz",
)
_FINANCE_CUES = _cue(
    r"invoices?", r"invoicing", r"payments?", r"payable", r"billing", r"remit\w*",
    r"accounts", r"vendor (?:setup|form)", r"w-?9", r"po", r"purchase orders?", r"net ?\d{2}",
)

# Determiners / pronouns / objects that follow a verb use of the word.
_VERB_OBJECT = (
    r"the|a|an|all|any|our|your|their|its|this|that|these|those|it|them|"
    r"him|her|us|me|each|every|some|both|everything|anything"
)
_MONITOR_VERB_AFTER = re.compile(
    r"^\s+(?:" + _VERB_OBJECT + r"|progress|status|activity|traffic|usage|"
    r"performance|conditions|construction|the site|site|closely|remotely|"
    r"continuously|24/7|and|for|whether|how)(?![a-z0-9])"
)
_DISPLAY_VERB_AFTER = re.compile(r"^\s+(?:" + _VERB_OBJECT + r")(?![a-z0-9])")
_SWITCH_VERB_AFTER = re.compile(
    r"^\s+(?:" + _VERB_OBJECT + r"|over|off|on|back|out|from|to|between|gears|vendors|carriers|providers)(?![a-z0-9])"
)
_STRIKE_VERB_AFTER = re.compile(r"^\s+(?:" + _VERB_OBJECT + r"|out|through|off|a balance)(?![a-z0-9])")
_REGISTER_VERB_AFTER = re.compile(
    r"^\s+(?:" + _VERB_OBJECT + r"|for|with|online|to|at|on|now|today|by|in|as|your)(?![a-z0-9])"
)

# A word immediately before a verb use: modal / auxiliary / subject pronoun.
_VERB_INTRO = frozenset({
    "will", "shall", "can", "could", "should", "must", "may", "might", "would",
    "please", "we", "they", "i", "you", "he", "she", "also", "then", "to",
    "don't", "dont", "cannot", "can't", "won't", "wont", "let's", "lets",
    "help", "helps", "able", "who", "easily", "continue", "continues", "ll",
})

_ELECTRICAL_NOUN_AFTER = re.compile(
    r"^\s*(?:circuits?|breakers?|service|load|draw|fuses?|outlets?|receptacles?|plugs?|"
    r"rating|rated|dedicated|panel|disconnect|feed|power|max|minimum|maximum)(?![a-z0-9])"
)

_RULES: dict[str, tuple[_Rule, ...]] = {
    # "cabinet" is a rack only in an IT room. Electrical, kitchen, filing,
    # millwork and enclosure cabinets are construction words.
    "cabinet": (
        _Rule(
            needs=_IT_RACK_CUES,
            not_after=frozenset({
                "electrical", "electric", "kitchen", "file", "filing", "storage", "base",
                "upper", "lower", "medicine", "supply", "fire", "extinguisher", "hose",
                "utility", "meter", "power", "breaker", "wood", "wooden", "custom",
                "millwork", "display", "trophy", "tool", "janitor", "janitorial", "mop",
                "gun", "liquor", "bathroom", "vanity", "lighting", "irrigation", "traffic",
                "control", "transformer", "switchgear", "battery", "panel", "pedestal",
                "hvac", "mechanical", "sprinkler", "valve", "classroom", "teacher",
                "charging", "laptop", "chromebook", "tablet", "locking", "cleaning",
            }),
            canonicals=frozenset({"rack", "server_rack", "network_rack"}),
        ),
    ),
    # Bare "rack" is the default IT reading; only a non-IT modifier changes it.
    "rack": (
        _Rule(
            not_after=frozenset({
                "bike", "bicycle", "coat", "shoe", "spice", "wine", "drying", "roof",
                "pallet", "luggage", "magazine", "towel", "dish", "tire", "hat", "gun",
                "bread", "baking", "oven", "clothes", "clothing", "garment", "brochure",
                "literature", "pipe", "ladder", "truck", "boot", "glove", "key", "tray",
                "lumber", "drywall",
            }),
            not_before=re.compile(r"^\s*(?:of lamb|and pinion)(?![a-z0-9])"),
        ),
    ),
    # "panel" is an access-control panel only near doors / readers / locks.
    "panel": (
        _Rule(
            needs=_ACCESS_CONTROL_CUES,
            canonicals=frozenset({"controller", "access_controller", "control_panel"}),
        ),
        _Rule(
            not_after=frozenset({
                "patch", "solar", "touch", "front", "rear", "side", "back", "wall",
                "ceiling", "acoustic", "glass", "lcd", "led", "flat", "fiber",
                "blank", "blanking", "fence", "door", "sandwich", "expert", "review",
                "interview", "discussion", "metal", "plywood", "wood", "display",
            }),
            not_before=re.compile(r"^\s*(?:discussion|of experts|interview)(?![a-z0-9])"),
        ),
    ),
    # A tower is a desktop only next to computers.
    "tower": (
        _Rule(
            needs=_COMPUTER_CUES,
            not_after=frozenset({
                "crane", "cell", "cellular", "water", "cooling", "light", "lighting",
                "camera", "radio", "scaffold", "scaffolding", "observation", "office",
                "clock", "bell", "transmission", "guard", "hose", "lattice", "monopole",
                "wireless", "antenna", "communications", "comm", "telecom", "trailer",
                "mobile", "surveillance", "security", "jobsite", "construction", "steel",
            }),
            not_before=re.compile(r"^\s*(?:crane|cranes|camera|cameras|light|lights|site|sites)(?![a-z0-9])"),
            canonicals=frozenset({"desktop", "workstation"}),
        ),
    ),
    # Electric strike: a door-hardware word; otherwise a date, a lightning
    # strike or "strike through".
    "strike": (
        _Rule(
            needs=_DOOR_HARDWARE_CUES,
            not_after=frozenset({"lightning", "labor", "union", "air", "first", "bird", "pre"}),
            verb=_STRIKE_VERB_AFTER,
        ),
    ),
    "es": (_Rule(needs=_DOOR_HARDWARE_CUES),),
    # Docking station vs loading dock.
    "dock": (
        _Rule(
            needs=_COMPUTER_CUES,
            not_after=frozenset({"loading", "receiving", "shipping", "boat", "truck", "dry", "freight"}),
            not_before=re.compile(r"^\s*(?:doors?|levelers?|plates?|area|height|lock|locks|access)(?![a-z0-9])"),
        ),
    ),
    "reader": (
        _Rule(
            not_after=frozenset({
                "meter", "screen", "pdf", "adobe", "acrobat", "e", "news", "proof",
                "mind", "map", "plan", "blueprint", "barcode", "check", "magazine",
                "avid", "book", "kindle",
            }),
        ),
    ),
    "controller": (
        _Rule(
            not_after=frozenset({
                "financial", "finance", "corporate", "assistant", "city", "county",
                "state", "division", "regional", "project", "plant", "group", "deputy",
            }),
        ),
    ),
    "supervisor": (_Rule(needs=_BMS_CUES, canonicals=frozenset({"controller"})),),
    "ups": (
        _Rule(
            not_after=frozenset({
                "follow", "set", "pick", "sign", "touch", "hook", "line", "mock",
                "pop", "start", "check", "back", "clean", "warm", "roll", "look",
                "write", "mark", "call", "round", "tune", "close", "mix", "make",
                "lock", "build", "pull", "push", "sit", "speed", "cover", "ramp",
                "wrap", "catch", "hold", "via", "by", "thru", "through", "with",
                "drop", "lay", "grown", "ups", "fedex", "usps", "dhl", "ship",
                "shipped", "shipping", "ups's", "pin",
            }),
            not_before=re.compile(
                r"^\s*(?:ground|next day|2nd day|second day|air|store|tracking|"
                r"shipping|shipment|delivery|account|label|labels|freight|express|"
                r"truck|driver|pickup|worldwide|surepost)(?![a-z0-9])"
            ),
        ),
    ),
    # Display / monitor / switch / register are also common verbs.
    "monitor": (_Rule(verb=_MONITOR_VERB_AFTER),),
    "display": (_Rule(verb=_DISPLAY_VERB_AFTER),),
    "switch": (
        _Rule(
            verb=_SWITCH_VERB_AFTER,
            not_after=frozenset({
                "light", "wall", "dimmer", "toggle", "limit", "kill", "safety",
                "transfer", "pressure", "float", "flow", "rocker", "power", "key",
                "reed", "tamper", "door", "push", "pull", "foot", "micro", "selector",
                "disconnect", "ats", "bait", "bait-and",
            }),
        ),
    ),
    "register": (_Rule(needs=_POS_CUES, verb=_REGISTER_VERB_AFTER),),
    # "Speaker 1:" is a transcript label, a guest speaker is a person.
    "speaker": (
        _Rule(
            not_after=frozenset({"guest", "keynote", "featured", "main", "motivational", "public", "native", "fluent", "next"}),
            # "Speaker 1:", "Speaker B)", or a bare "Speaker 2" label line.
            not_before=re.compile(r"^\s+(?:\d{1,2}|[a-z])\s*(?:[:)]|$)"),
        ),
    ),
    "sub": (_Rule(needs=_AUDIO_CUES, not_before=re.compile(r"^\s*(?:-?\s*contract\w*|panel|par|total|section|floor|base|grade)"), canonicals=frozenset({"speaker"})),),
    "card": (
        _Rule(
            not_after=frozenset({
                "business", "credit", "debit", "sd", "microsd", "micro", "report",
                "gift", "sim", "graphics", "video", "network", "wifi", "rate", "score",
                "index", "playing", "post", "green", "wild", "memory", "sound",
                "punch", "comment", "trump", "calling", "greeting", "flash", "line",
            }),
        ),
    ),
    "badge": (_Rule(not_after=frozenset({"merit", "name"})),),
    "tape": (
        _Rule(
            needs=_STORAGE_MEDIA_CUES,
            not_after=frozenset({"caution", "duct", "electrical", "painter's", "painters", "masking", "red", "measuring", "double-sided", "warning", "packing", "scotch", "floor"}),
            not_before=re.compile(r"^\s*(?:measure|off|it|up|down)(?![a-z0-9])"),
        ),
    ),
    "tap": (_Rule(needs=_NET_TAP_CUES),),
    "storage": (
        _Rule(
            not_before=re.compile(r"^\s*(?:rooms?|closets?|space|areas?|units?|containers?|sheds?|cabinets?|racks?|bins?|trailers?|yard|facility|locker)(?![a-z0-9])"),
            not_after=frozenset({"cold", "onsite", "on-site", "offsite", "off-site", "material", "materials", "equipment", "secure", "temporary", "bike"}),
        ),
    ),
    "smoke": (_Rule(not_before=re.compile(r"^\s*(?:test|tests|testing|and mirrors|break|signals?)(?![a-z0-9])"), not_after=frozenset({"no", "cigarette"})),),
    "na": (_Rule(needs=_FIRE_ALARM_CUES),),
    "hs": (_Rule(needs=_FIRE_ALARM_CUES),),
    "ped": (_Rule(needs=_POS_CUES),),
    "poi": (_Rule(needs=_DAS_CUES),),
    "ground": (
        _Rule(
            not_before=re.compile(r"^\s*(?:floor|level|lot|up|-up|breaking|work|crew|clearance|surface|conditions|mounted|mount|transportation|shipping|service)(?![a-z0-9])"),
            not_after=frozenset({"ups", "above", "below", "break", "broke", "common", "high", "level", "solid", "middle", "fedex"}),
        ),
    ),
    "ap": (_Rule(not_before=re.compile(r"^\s*(?:department|dept|team|clerk|contact|invoices?|inbox|e-?mail|@|aging|automation|specialist|manager|portal|style|news|exam|class|course|test|credit)(?![a-z0-9])"), not_after=frozenset({"accounts"}), veto=_FINANCE_CUES, veto_unless=_WIRELESS_CUES),),
    # Units of measure after a number.
    "pc": (_Rule(unit=True),),
    "lb": (_Rule(unit=True, not_after=frozenset({"per"})),),
    "dia": (_Rule(unit=True),),
    "meter": (_Rule(unit=True, not_before=re.compile(r"^\s*(?:long|run|length|cable|cables|of)(?![a-z0-9])")),),
    "amp": (_Rule(unit=True, unit_singular_only=True, not_before=_ELECTRICAL_NOUN_AFTER),),
    "cam": (_Rule(not_before=re.compile(r"^\s*(?:locks?|levers?|buckles?|straps?|latch(?:es)?|followers?|shafts?)(?![a-z0-9])")),),
    "server": (_Rule(not_after=frozenset({"process", "food", "restaurant"}), not_before=re.compile(r"^\s+of\s+(?:this|the|any|such)\s+(?:notice|process|summons)")),),
}

_NUMBER_BEFORE_RE = re.compile(r"(?:\d|\d\s*[\"”']|\d\s*-)\s*$")
_CLAUSE_START_RE = re.compile(r"(?:^|[.;:!?\n•*•(\[]|\s-\s|^\s*\d+[.)])\s*$")


def _word_before(text_lower: str, start: int) -> str:
    m = _WORD_BEFORE_RE.search(text_lower[:start])
    return m.group(1) if m else ""


def _is_unit_after_number(text_lower: str, start: int, surface: str, original: str | None) -> bool:
    before = text_lower[max(0, start - 12):start]
    if not _NUMBER_BEFORE_RE.search(before):
        return False
    # "10 PCs" in the original casing is ten computers; "10 pcs" is pieces.
    if original is not None and len(original) > 1 and original[-1] == "s" and original[:-1].isupper():
        return False
    return True


def _is_verb_use(text_lower: str, start: int, end_full: int, after_re: "re.Pattern[str]") -> bool:
    after = text_lower[end_full:end_full + 40]
    word = _word_before(text_lower, start)
    if word in _VERB_INTRO and word != "to":
        # "we will monitor ..."; "please display ..."
        return True
    if not after_re.match(after):
        return False
    if word == "to" or not word:
        return True
    # "monitor progress and display the feed": a singular form after a
    # conjunction, taking an object, continues a verb list.
    if word in ("and", "or") and end_full - start == len(text_lower[start:end_full].rstrip("s")):
        return True
    return bool(_CLAUSE_START_RE.search(text_lower[:start]))


def device_match_is_spurious(
    text_lower: str,
    start: int,
    end_alias: int,
    end_full: int,
    alias: str,
    canonical: str,
    *,
    original: str | None = None,
) -> bool:
    """True when the alias hit at ``text_lower[start:end_full]`` is an ordinary
    word, not the device ``canonical``.

    ``end_alias`` is the end of the alias proper, ``end_full`` includes any
    plural suffix the matcher allowed. ``original`` is the original-case
    surface of the match when the caller has it.
    """
    rules = _RULES.get(alias)
    if rules is None and alias.endswith("es"):
        rules = _RULES.get(alias[:-2])
    if rules is None and alias.endswith("s"):
        rules = _RULES.get(alias[:-1])
    if not rules:
        return False
    canon = (canonical or "").lower()
    word_before = None
    for rule in rules:
        if rule.canonicals is not None and canon not in rule.canonicals:
            continue
        if word_before is None:
            word_before = _word_before(text_lower, start)
        # Hyphen-joined compound ("follow-ups", "pick-ups").
        if rule.not_after and start > 0 and text_lower[start - 1] == "-" and word_before in rule.not_after:
            return True
        if rule.not_after and word_before in rule.not_after:
            return True
        if rule.not_before is not None and rule.not_before.match(text_lower[end_full:end_full + 40]):
            return True
        if rule.unit and not (rule.unit_singular_only and end_full > end_alias) and _is_unit_after_number(text_lower, start, text_lower[start:end_full], original):
            return True
        if rule.verb is not None and _is_verb_use(text_lower, start, end_full, rule.verb):
            return True
        if rule.veto is not None:
            masked = text_lower[:start] + " " * (end_full - start) + text_lower[end_full:]
            if rule.veto.search(masked) and not (rule.veto_unless is not None and rule.veto_unless.search(masked)):
                return True
        if rule.needs is not None:
            # The cue must appear somewhere OTHER than the alias itself.
            masked = text_lower[:start] + " " * (end_full - start) + text_lower[end_full:]
            if not rule.needs.search(masked):
                return True
    return False


# ── atom-level guards ───────────────────────────────────────────────────
_LEGAL_CUES = re.compile(
    r"\b(?:indemnif\w*|hold harmless|liabilit(?:y|ies)|liable|hereunder|herein|hereto|"
    r"thereof|whereas|governing law|arbitration|force majeure|consequential damages|"
    r"intellectual property|confidential information|sole remedy|notwithstanding|"
    r"limitation of liability|jurisdiction|severab\w*|assigns|successors)\b",
    re.I,
)


def is_legal_boilerplate(text: str) -> bool:
    """Contract boilerplate: two or more distinct legal-term cues. A device
    named in an indemnity clause is not equipment on the job."""
    if not text:
        return False
    found = {m.group(0).lower() for m in _LEGAL_CUES.finditer(text)}
    return len(found) >= 2


def iter_valid_alias_matches(pattern: "re.Pattern[str]", text_lower: str, alias: str, canonical: str, original_text: str | None = None):
    """Yield the matches of a single-alias ``pattern`` that are not spurious."""
    for m in pattern.finditer(text_lower):
        orig = None
        if original_text is not None and len(original_text) == len(text_lower):
            orig = original_text[m.start():m.end()]
        if device_match_is_spurious(text_lower, m.start(), m.end(), m.end(), alias, canonical, original=orig):
            continue
        yield m


__all__ = ["device_match_is_spurious", "is_legal_boilerplate", "iter_valid_alias_matches"]
