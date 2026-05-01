from __future__ import annotations

import json
import re
from typing import Any

from app.domain import get_active_domain_pack


def normalize_text(value: str) -> str:
    if value is None:
        return ""
    value = value.strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value


def normalize_entity(value: str) -> str:
    value = normalize_text(value)
    value = re.sub(r"[^a-z0-9 ._-]", "", value)
    return value


def normalize_entity_key(entity_type: str, value: str) -> str:
    normalized = normalize_text(value)
    if entity_type == "site":
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


def parse_timestamp(line: str) -> str | None:
    match = re.search(r"\[(\d{2}:\d{2}:\d{2})\]", line)
    if match:
        return match.group(1)
    match = re.search(r"\b(\d{2}:\d{2}:\d{2})\b", line)
    if match:
        return match.group(1)
    return None


def detect_speaker(line: str) -> str | None:
    line = line.strip()
    # [00:00:01] Speaker: text
    match = re.match(r"^\[(?:\d{2}:\d{2}:\d{2})\]\s*([^:]{2,80}):\s*.+$", line)
    if match:
        return match.group(1).strip()
    # Speaker: text
    match = re.match(r"^([^:]{2,80}):\s*.+$", line)
    if match:
        key = match.group(1).strip()
        if key.lower() not in {"decision", "decisions", "action items", "open questions", "ai"}:
            return key
    return None


def detect_section(line: str) -> str | None:
    cleaned = normalize_text(line).rstrip(":")
    if cleaned in {"decisions", "action items", "open questions", "notes", "discussion"}:
        return cleaned.title()
    return None


def split_transcript_segments(text: str) -> list[dict[str, Any]]:
    normalized = normalize_transcript_text(text)
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
        if speaker:
            # remove timestamp prefix and speaker label from content
            content = re.sub(r"^\[(?:\d{2}:\d{2}:\d{2})\]\s*", "", content)
            content = content.split(":", 1)[1].strip()
        elif stripped.startswith("- "):
            content = stripped[2:].strip()

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
        entity_keys.add(normalize_entity_key("site", phrase))

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
