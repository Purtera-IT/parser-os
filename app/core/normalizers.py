from __future__ import annotations

import json
import re
from typing import Any

from app.domain import get_active_domain_pack


# Insanity-perf: normalize_text is called ~50M times during a single
# compile. Precompile the patterns here so each call is a single C-side
# regex match rather than a recompile lookup through the global cache.
_WHITESPACE_RE = re.compile(r"\s+")
_NON_ENTITY_CHAR_RE = re.compile(r"[^a-z0-9 ._-]")


def normalize_text(value: str) -> str:
    if value is None:
        return ""
    value = value.strip().lower()
    value = _WHITESPACE_RE.sub(" ", value)
    return value


def normalize_entity(value: str) -> str:
    value = normalize_text(value)
    value = _NON_ENTITY_CHAR_RE.sub("", value)
    return value


_GENERIC_SITE_PSEUDO_VALUES: frozenset[str] = frozenset({
    "",
    "-",
    "—",
    "n/a",
    "na",
    "none",
    "null",
    "tbd",
    "tba",
    "tbc",
    "all",
    "any",
    "various",
    "multiple",
    "site",
    "location",
    "address",
    "see above",
    "see below",
    "see notes",
    "see attached",
    "as noted",
    "unknown",
    # Column-header / row-label noise that gets surfaced as a "site"
    # when prose mentions the column name (e.g. "Site ID Facility"
    # is a header row caught by the proper-noun matcher).
    "site id",
    "site id facility",
    "site code",
    "site number",
    "facility name",
    "access window",
    "escort owner",
    "mdf",
    "idf",
    "mdf idf",
    # Telecom-closet identifiers that aren't physical sites —
    # they're rooms WITHIN a site (already captured via mdf_idf on
    # the physical_site atom). The bare slug shouldn't surface as
    # its own site entity.
    "mdf 1", "mdf 2", "mdf 3", "mdf 4", "mdf 5", "mdf 6",
    "mdf a", "mdf b", "mdf c", "mdf d", "mdf w", "mdf w1", "mdf w2",
    "mdf cp", "mdf e",
    "idf 1", "idf 2", "idf 3", "idf 4", "idf 5",
    "idf a", "idf b", "idf w", "idf w1", "idf w2",
    "warehouse rf",
    "n terminal",
    "ic 001", "ic 002", "ic 003",
    "am 3", "am 4", "am 5",
    "atlanta ga", "atlanta",
    # Bare ATL-style prefixes are legitimate aliases for numbered site
    # rows (ATL-HQ -> ATL-HQ-01). Keep them as entity keys; the physical_site
    # semantic deduper performs alias-to-canonical collapse later.
    # Building / closet identifiers
    "building c", "building d", "building e",
})


def normalize_entity_key(entity_type: str, value: str) -> str:
    normalized = normalize_text(value)
    if entity_type == "site":
        # Generic pseudo-values like "ALL" / "N/A" / "Various" should not
        # produce site entities — they show up in xlsx allocation tables
        # to mean "applies everywhere", not "site named ALL".
        if normalized in _GENERIC_SITE_PSEUDO_VALUES:
            return ""
        # Also reject the underscore-slug form ("site_id_facility",
        # "n_terminal", "mdf_w1"). The slug form is what shows up in
        # canonical_keys after entity_resolution.
        if normalized.replace("_", " ") in _GENERIC_SITE_PSEUDO_VALUES:
            return ""
        site_aliases = {
            "west-wing": "west wing",
            "bldg a west": "west wing",
            "building a west": "west wing",
            "main campus north": "main campus",
        }
        normalized = site_aliases.get(normalized, normalized)
    if entity_type == "device":
        pack = get_active_domain_pack()
        pack_device_aliases: dict[str, str] = {}
        for canonical, aliases in pack.device_aliases.items():
            for alias in aliases:
                pack_device_aliases[normalize_text(alias)] = canonical.replace("_", " ")
        device_aliases = {
            "ip cam": "ip camera",
            "ip cams": "ip camera",
            "ip cameras": "ip camera",
            "camera": "ip camera",
            "aps": "access point",
            "ap": "access point",
        }
        device_aliases.update(pack_device_aliases)
        normalized = device_aliases.get(normalized, normalized)
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    return f"{entity_type}:{normalized}"


def parse_quantity(value: Any) -> dict[str, Any]:
    raw = "" if value is None else str(value).strip()
    if raw == "":
        return {"quantity": None, "unit": "count", "raw": raw, "uncertain": True}

    normalized = normalize_text(raw).replace(",", "")
    match = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*([a-z%]+)?\s*$", normalized)
    if not match:
        return {"quantity": None, "unit": "count", "raw": raw, "uncertain": True}

    number_raw = match.group(1)
    unit = match.group(2) or "count"
    quantity = float(number_raw)
    if quantity.is_integer():
        quantity = int(quantity)
    return {"quantity": quantity, "unit": unit, "raw": raw, "uncertain": False}


def normalize_transcript_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\t+", " ", text)
    text = re.sub(r"[ ]+", " ", text)
    return text.strip()


# ── the Otter / Rev / Zoom dialect: the speaker gets its OWN line ─────────
#
# Those exporters write
#
#     Cliff Creech
#     we need forty sites by Q3.
#
# where Teams and the meeting PDFs write ``Cliff Creech: we need forty...``.
# Everything downstream -- detect_speaker, the utterance segmenter,
# speaker_role, the meeting_decision family -- speaks the colon dialect, so
# rather than teach each of them a second grammar, this folds the dialect
# into the one they already know. The router calls the SAME function to ask
# whether folding would produce turns, which is the point: routing and
# parsing cannot disagree about what the file is, because they are reading
# the same evidence through the same code.
#
# A name token is a capital followed by letters, or a capital and a period
# for an initial. A trailing period on a multi-letter word is NOT a name
# token, so an ordinary sentence cannot be mistaken for a speaker line.
_STANDALONE_SPEAKER_RE = re.compile(
    r"^(?P<name>[A-Z](?:[A-Za-z'\-]+|\.)(?:[ \t]+[A-Z](?:[A-Za-z0-9'\-]+|\.)){0,3})"
    r"(?P<role>[ \t]*\([^)]{1,40}\))?"
    r"(?:[ \t]+\[?(?P<ts>\d{1,2}:\d{2}(?::\d{2})?)\]?)?$"
)
_SPEAKER_LINE_MAX_CHARS = 48


def fold_standalone_speaker_lines(text: str) -> tuple[str, dict[str, Any]]:
    """Rewrite own-line speaker labels as ``Name: body``.

    Returns the folded text and the statistics the decision was made on, so a
    caller that only wants to KNOW (the router) and a caller that wants the
    rewrite (the parser) share one implementation.

    Four conditions have to hold together, and each one is there to exclude a
    specific document that would otherwise look like a transcript:

    * **a name line is followed by a non-name line.** An attendee roster is
      names all the way down, so every line disqualifies the one above it and
      the roster folds to nothing. This is the strongest of the four.
    * **names recur.** Turn-taking is what a transcript IS. Section headings
      in a spec are title-case and short, but each appears once; requiring
      1.5 folds per distinct name separates a conversation from an outline.
    * **few distinct names.** A meeting has a handful of participants; a
      specification has dozens of headings.
    * **a quarter of the document.** Below that it is a document with some
      capitalised lines in it, not a transcript.

    Line numbering is preserved -- the speaker line is blanked, never deleted
    -- so ``line_start`` in every locator still points at the true source
    line and receipt replay keeps verifying.
    """
    lines = text.splitlines()
    filled = [i for i, ln in enumerate(lines) if ln.strip()]

    candidates: dict[int, Any] = {}
    for i in filled:
        stripped = lines[i].strip()
        if len(stripped) > _SPEAKER_LINE_MAX_CHARS:
            continue
        match = _STANDALONE_SPEAKER_RE.match(stripped)
        if match:
            candidates[i] = match

    folds: list[tuple[int, int]] = []
    for position, i in enumerate(filled):
        if i not in candidates or position + 1 >= len(filled):
            continue
        following = filled[position + 1]
        if following in candidates:
            # A name under a name is a roster line, not a turn.
            continue
        folds.append((i, following))

    names = {candidates[i].group("name").strip() for i, _ in folds}
    stats: dict[str, Any] = {
        "folds": len(folds),
        "distinct_speakers": len(names),
        "line_count": len(filled),
        "density": (len(folds) / len(filled)) if filled else 0.0,
        "qualifies": False,
    }
    stats["qualifies"] = (
        len(folds) >= 4
        and 0 < len(names) <= 12
        and len(folds) >= len(names) * 1.5
        and stats["density"] >= 0.25
    )
    if not stats["qualifies"]:
        return text, stats

    out = list(lines)
    for i, following in folds:
        match = candidates[i]
        speaker = match.group("name").strip()
        role = (match.group("role") or "").strip()
        if role:
            # Keep the parenthetical: "(Purtera)" is who the speaker works
            # for, and dropping it here would lose it silently.
            speaker = f"{speaker} {role}"
        stamp = match.group("ts")
        body = out[following].strip()
        # ``Name [mm:ss]: body`` is the dialect _NAME_BRACKET_TS_RE already
        # reads, so a timestamped export keeps its clock.
        out[following] = f"{speaker} [{stamp}]: {body}" if stamp else f"{speaker}: {body}"
        out[i] = ""
    return "\n".join(out), stats


def parse_timestamp(line: str) -> str | None:
    match = re.search(r"\[(\d{2}:\d{2}:\d{2})\]", line)
    if match:
        return match.group(1)
    # Meeting-export clocks are often [mm:ss] without hours.
    match = re.search(r"\[(\d{1,2}:\d{2})\]", line)
    if match:
        return match.group(1)
    match = re.search(r"\b(\d{2}:\d{2}:\d{2})\b", line)
    if match:
        return match.group(1)
    return None


# ``Alex Rivera [03:15]`` / ``Alex Rivera [03:15]: body`` — common PDF/meeting
# export diarization. Structural (Name tokens + bracket clock), not a name list.
# Initials ("J.") allowed; trailing period on multi-letter words is not (avoids
# "Hey.\\nTrent … [00:56]" being parsed as a speaker).
# Only [ \t] between name tokens — never newlines — so sticky section
# headers on the previous line cannot be absorbed into the speaker.
_NAME_BRACKET_TS_RE = re.compile(
    r"^(?P<speaker>[A-Z](?:[A-Za-z'\-]+|\.)(?:[ \t]+[A-Z](?:[A-Za-z0-9'\-]+|\.)){0,4})"
    r"[ \t]*\[(?P<ts>\d{1,2}:\d{2}(?::\d{2})?)\][ \t]*:?[ \t]*(?P<body>.*)$"
)


def detect_speaker(line: str) -> str | None:
    line = line.strip()
    # Name [mm:ss] body  (meeting PDF / HubSpot transcript export)
    name_ts = _NAME_BRACKET_TS_RE.match(line)
    if name_ts:
        return name_ts.group("speaker").strip()
    # [00:00:01] Speaker: text
    match = re.match(r"^\[(?:\d{2}:\d{2}:\d{2})\]\s*([A-Za-zÀ-ÿ][^:]{1,79}):\s*.+$", line)
    if match:
        return match.group(1).strip()
    # Speaker: text — anchor on a letter to avoid matching a leading
    # bracketed timestamp like ``[00:00:01] Welcome everyone.`` which
    # otherwise yielded speaker="[00" and crashed the segment splitter.
    match = re.match(r"^([A-Za-zÀ-ÿ][^:]{1,79}):\s*.+$", line)
    if match:
        key = match.group(1).strip()
        if key.lower() not in {"decision", "decisions", "action items", "open questions", "ai"}:
            return key
    return None


# Meeting-summary / agenda section labels (Title-Case or ALL CAPS in source).
# Structural closed class — not a deal/customer vocabulary. Used by transcript
# segmenters AND the PDF text-rich splitter so header+bullet pages nest bullets
# under the right section_path instead of gluing headers into prose.
_MEETING_SECTION_ALIASES: dict[str, str] = {
    "decisions": "Decisions",
    "decision": "Decisions",
    "key decisions": "Key Decisions",
    "key decision": "Key Decisions",
    "action items": "Action Items",
    "action item": "Action Items",
    "open questions": "Open Questions",
    "open question": "Open Questions",
    "notes": "Notes",
    "discussion": "Discussion",
    "executive summary": "Executive Summary",
    "attendees": "Attendees",
    "participants": "Participants",
    "next steps": "Next Steps",
    "agenda": "Agenda",
    "follow ups": "Follow Ups",
    "follow-ups": "Follow Ups",
    "follow up": "Follow Ups",
}


def detect_section(line: str) -> str | None:
    """Return canonical meeting-section label when ``line`` is ONLY that header.

    Matches Title-Case / ALL CAPS / trailing-colon variants. Returns None for
    ordinary prose that merely mentions the phrase.
    """
    cleaned = normalize_text(line).rstrip(":")
    return _MEETING_SECTION_ALIASES.get(cleaned)


def meeting_section_slug(label: str | None) -> str | None:
    """Normalize a meeting section label to a ``list_section`` slug."""
    if not label:
        return None
    cleaned = normalize_text(label).rstrip(":")
    canonical = _MEETING_SECTION_ALIASES.get(cleaned)
    if not canonical:
        for display in set(_MEETING_SECTION_ALIASES.values()):
            if normalize_text(display) == cleaned:
                canonical = display
                break
    if not canonical:
        return None
    return re.sub(r"[^a-z0-9]+", "_", canonical.lower()).strip("_")


def split_transcript_segments(text: str) -> list[dict[str, Any]]:
    normalized = normalize_transcript_text(text)
    # Fold HERE rather than inside normalize_transcript_text: that function
    # strips, it is called twice on the way in (segment_transcript normalises
    # and then split_transcript_segments normalises again), and the second
    # strip ate the blank line the fold leaves behind -- shifting every line
    # number by one and breaking receipt replay on exactly the files the fold
    # was added to support. Folding against the same list whose indices become
    # line numbers makes that class of bug unrepresentable.
    normalized, _fold_stats = fold_standalone_speaker_lines(normalized)
    lines = normalized.splitlines()
    segments: list[dict[str, Any]] = []
    current_section: str | None = None
    utterance_index = 0

    for index, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        section = detect_section(stripped)
        if section:
            current_section = section
            continue

        speaker = detect_speaker(stripped)
        timestamp = parse_timestamp(stripped)
        content = stripped
        name_ts = _NAME_BRACKET_TS_RE.match(stripped)
        if name_ts:
            # ``Alex Rivera [03:15] body`` — body is everything after the stamp.
            content = (name_ts.group("body") or "").strip()
            if not timestamp:
                timestamp = name_ts.group("ts")
        elif speaker:
            # remove timestamp prefix and speaker label from content
            content = re.sub(r"^\[(?:\d{2}:\d{2}(?::\d{2})?)\]\s*", "", content)
            # Defensive: a future regex change could yield a "speaker"
            # whose label isn't followed by ``:`` in the line. Skip the
            # split rather than crashing — keep the full line as content.
            if ":" in content:
                content = content.split(":", 1)[1].strip()
        elif stripped.startswith("- "):
            content = stripped[2:].strip()
        else:
            # Bracketed-timestamp leader (``[00:00:01] body``) with no
            # speaker label: strip the timestamp so the body reads
            # cleanly downstream.
            content = re.sub(r"^\[(?:\d{2}:\d{2}(?::\d{2})?)\]\s*", "", content)

        segments.append(
            {
                "utterance_index": utterance_index,
                "line_start": index,
                "line_end": index,
                "speaker": speaker,
                "timestamp_start": timestamp,
                "timestamp_end": None,
                "section": current_section,
                "text": content,
            }
        )
        utterance_index += 1
    return segments


def extract_meeting_entities(text: str) -> list[str]:
    lowered = normalize_text(text)
    entity_keys: set[str] = set()
    pack = get_active_domain_pack()

    for match in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s+(Campus|Wing|Building|Store|Site))\b", text):
        phrase = match.group(1)
        site_key = normalize_entity_key("site", phrase)
        if site_key:
            entity_keys.add(site_key)

    if "main campus" in lowered:
        entity_keys.add("site:main_campus")
    if "west wing" in lowered:
        entity_keys.add("site:west_wing")

    if re.search(r"\bip\s*cameras?\b", lowered):
        entity_keys.add(normalize_entity_key("device", "IP Camera"))
    if re.search(r"\baccess point\b|\baps?\b", lowered):
        entity_keys.add(normalize_entity_key("device", "access point"))
    for canonical, aliases in pack.device_aliases.items():
        for alias in aliases:
            if re.search(rf"\b{re.escape(normalize_text(alias))}\b", lowered):
                entity_keys.add(f"device:{canonical}")
                break

    return sorted(entity_keys)


def looks_like_diarized_transcript_json(raw_text: str) -> bool:
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        return False
    if isinstance(payload, dict):
        if isinstance(payload.get("utterances"), list):
            return True
        if isinstance(payload.get("segments"), list):
            return True
    if isinstance(payload, list):
        return all(isinstance(item, dict) and ("speaker" in item or "text" in item) for item in payload[:5])
    return False
