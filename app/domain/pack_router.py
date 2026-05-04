"""Project-level domain-pack router.

Picks the best ``DomainPack`` for a project directory.  Order of
precedence:

1. **Explicit override** — ``--domain-pack <pack_id>`` from the CLI
   (handled at the caller level by passing ``domain_pack`` to
   ``load_domain_pack``); this module only fires when the caller
   passes ``None``.
2. **Project config** — ``<project>/project.yaml`` with
   ``service_line:`` or ``domain_pack:`` keys.
3. **SOURCE_NOTES.md** — a free-text declaration like ``Service line:
   security_camera`` (case-insensitive, multiple synonyms accepted).
4. **Filename heuristics** — keywords in the artifact filenames that
   strongly imply a service line (e.g. ``cabling`` → ``copper_cabling``,
   ``camera`` → ``security_camera``, ``wireless`` → ``wireless``).
5. **Content scoring** — read the first ~5 KB of each parseable file
   (PDF, XLSX, txt) and tally hits against each pack's
   ``device_aliases`` and ``entity_types[].aliases``; pick the
   highest-scoring pack with a margin over default.
6. **Fallback** — ``default_pack``.

The router is conservative: when scores tie or are too low to be
confident, it returns the default pack so downstream stages stay
predictable.  All decisions are reported via ``RoutingDecision`` so
operators can see *why* a pack was selected.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from app.domain.loader import (
    DEFAULT_PACK_ID,
    _candidate_pack_path,
    load_domain_pack,
)
from app.domain.schemas import DomainPack


# Service-line synonyms commonly seen in RFP / SOURCE_NOTES text.
# Each canonical pack id maps to the surface forms a human might write.
_SERVICE_LINE_SYNONYMS: dict[str, list[str]] = {
    "security_camera": [
        "security_camera",
        "security camera",
        "video surveillance",
        "vms",
        "cctv",
        "video_surveillance",
        "surveillance",
    ],
    "access_control": [
        "access_control",
        "access control",
        "eacs",
        "card access",
        "door access",
        "physical access control",
    ],
    "wireless": [
        "wireless",
        "wifi",
        "wi-fi",
        "wlan",
        "e-rate wireless",
    ],
    "networking": [
        "networking",
        "managed vpn",
        "vpn",
        "lan",
        "wan",
        "ethernet switching",
        "structured network",
    ],
    "copper_cabling": [
        "copper_cabling",
        "copper cabling",
        "cabling",
        "structured cabling",
        "cat6",
        "cat 6",
        "cat-6",
        "ip phone cabling",
    ],
    "av": [
        "av",
        "audio visual",
        "audio-visual",
        "audiovisual",
        "audio_visual",
        "av integration",
        "video conferencing",
    ],
    "bms": [
        "bms",
        "building management",
        "building automation",
        "bas",
        "ddc",
        "facility controls",
        "integrated automation",
    ],
    "paging": [
        "paging",
        "mass notification",
        "emergency notification",
        "mass communication",
        "emns",
        "wide-area public address",
    ],
    "fire_safety": [
        "fire_safety",
        "fire safety",
        "fire alarm",
        "fire detection",
        "life safety",
        "nfpa 72",
    ],
    "das": [
        "das",
        "distributed antenna system",
        "in-building das",
        "public safety das",
    ],
    "electrical": [
        "electrical",
        "electrical power",
        "panel schedule",
        "power distribution",
    ],
    "itad": [
        "itad",
        "it asset disposition",
        "asset disposition",
        "secure data destruction",
        "e-waste recycling",
    ],
}

# Filename keywords that bias pack selection.  Used as a tiebreaker
# when content scoring is ambiguous.
_FILENAME_KEYWORDS: dict[str, list[str]] = {
    "security_camera": ["camera", "surveillance", "cctv", "vms", "video_analytics"],
    "access_control": ["access_control", "card_reader", "door_access", "eacs"],
    "wireless": ["wireless", "wifi", "wlan", "e-rate", "erate", "ap_install"],
    "networking": ["network", "vpn", "lan", "wan", "switch", "managed_vpn"],
    "copper_cabling": ["cabling", "cat6", "cat-6", "structured_cabling", "ip_phone"],
    "av": ["audio_visual", "audiovisual", "av_", "boardroom", "conference_room"],
    "bms": ["bas", "bms", "building_automation", "hvac_controls", "integrated_automation"],
    "paging": ["mass_notification", "emergency_notification", "paging", "mass_comm"],
    "fire_safety": ["fire_alarm", "fire_safety", "nfpa72", "smoke_detector"],
    "das": ["das", "distributed_antenna", "ibw", "in_building_wireless"],
    "electrical": ["panel_schedule", "electrical_distribution"],
    "itad": ["itad", "asset_disposition", "asset_inventory", "data_destruction"],
}


@dataclass
class RoutingDecision:
    """Why we picked a particular pack — for telemetry and review folders."""

    pack_id: str
    source: str  # 'explicit' | 'project_yaml' | 'source_notes' | 'filename' | 'content' | 'default'
    confidence: float
    rationale: str
    alternatives: list[tuple[str, float]] = field(default_factory=list)


def _read_project_yaml(project_dir: Path) -> dict[str, Any] | None:
    for name in ("project.yaml", "project.yml"):
        candidate = project_dir / name
        if candidate.is_file():
            try:
                payload = yaml.safe_load(candidate.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    return payload
            except Exception:  # pragma: no cover — never fail compile on config read
                return None
    return None


def _read_source_notes(project_dir: Path) -> str:
    candidate = project_dir / "SOURCE_NOTES.md"
    if not candidate.is_file():
        return ""
    try:
        return candidate.read_text(encoding="utf-8", errors="ignore")
    except Exception:  # pragma: no cover
        return ""


def _service_line_to_pack_id(service_line: str) -> str | None:
    """Normalize free-text service line → canonical pack id."""
    if not service_line:
        return None
    needle = service_line.strip().lower().replace("-", "_").replace(" ", "_")
    # Direct match against canonical names
    for pack_id in _SERVICE_LINE_SYNONYMS:
        if needle == pack_id:
            return pack_id
    # Synonym match (using normalized synonyms)
    raw = service_line.strip().lower()
    for pack_id, surfaces in _SERVICE_LINE_SYNONYMS.items():
        for surface in surfaces:
            if raw == surface or surface in raw:
                return pack_id
    return None


def _route_from_project_yaml(project_dir: Path) -> RoutingDecision | None:
    payload = _read_project_yaml(project_dir)
    if not payload:
        return None
    pack_id_explicit = payload.get("domain_pack") or payload.get("pack_id")
    if isinstance(pack_id_explicit, str) and pack_id_explicit.strip():
        pack_id = pack_id_explicit.strip()
        return RoutingDecision(
            pack_id=pack_id,
            source="project_yaml",
            confidence=1.0,
            rationale=f"project.yaml declared domain_pack: {pack_id}",
        )
    service_line = payload.get("service_line")
    if isinstance(service_line, str) and service_line.strip():
        pack_id = _service_line_to_pack_id(service_line) or DEFAULT_PACK_ID
        return RoutingDecision(
            pack_id=pack_id,
            source="project_yaml",
            confidence=0.95,
            rationale=f"project.yaml declared service_line: {service_line!r} → {pack_id}",
        )
    return None


_SOURCE_NOTES_PATTERNS = [
    # **Service line**: `copper_cabling`
    re.compile(r"\*\*service\s*line\*\*\s*:\s*`?([a-z0-9_\- ]{2,40})`?", re.IGNORECASE),
    # Service line: copper_cabling
    re.compile(r"^[\s\-\*]*service\s*line\s*[:=]\s*([a-z0-9_\- ]{2,40})", re.IGNORECASE | re.MULTILINE),
    # service_line: copper_cabling
    re.compile(r"\bservice_line\s*[:=]\s*([a-z0-9_\- ]{2,40})", re.IGNORECASE),
]


def _route_from_source_notes(project_dir: Path) -> RoutingDecision | None:
    notes = _read_source_notes(project_dir)
    if not notes:
        return None
    for pattern in _SOURCE_NOTES_PATTERNS:
        match = pattern.search(notes)
        if not match:
            continue
        raw_value = match.group(1).strip().strip("`'\"")
        pack_id = _service_line_to_pack_id(raw_value)
        if pack_id:
            return RoutingDecision(
                pack_id=pack_id,
                source="source_notes",
                confidence=0.90,
                rationale=f"SOURCE_NOTES.md declared service line {raw_value!r} → {pack_id}",
            )
    return None


def _filename_hint_score(filenames: list[str]) -> dict[str, float]:
    scores: dict[str, float] = {}
    haystack = " ".join(name.lower().replace(".", "_").replace("-", "_") for name in filenames)
    for pack_id, keywords in _FILENAME_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in haystack)
        if hits:
            scores[pack_id] = float(hits)
    return scores


def _read_text_preview(path: Path, max_bytes: int = 6 * 1024) -> str:
    """Best-effort text preview for filename + lightweight content scoring.

    Returns empty string for binary files we can't preview cheaply.
    """
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".csv"}:
        try:
            return path.read_text(encoding="utf-8", errors="ignore")[:max_bytes]
        except Exception:
            return ""
    return ""


def _content_score(project_dir: Path) -> dict[str, float]:
    """Score each pack against the artifact filenames + textual content.

    For PDFs we'd ideally parse the first page text, but that's
    expensive at routing time.  Instead we score:
    - filenames (cheap, almost always informative)
    - any plain-text / markdown / csv files we can read directly

    Each pack's ``device_aliases`` keys + each ``entity_types[].aliases``
    entry contribute to the haystack-needle scoring.

    We require an ``artifacts/`` subdirectory to exist; if it doesn't,
    we return empty scores so callers fall through to default_pack
    instead of accidentally scoring against project metadata files.
    """
    scores: dict[str, float] = {}
    pack_ids = list(_SERVICE_LINE_SYNONYMS.keys())
    # Cheap pass — filenames.  Restricted to ``artifacts/`` so we
    # never accidentally score against ``labels/`` gold standards or
    # documentation in the project root.
    artifact_dir = project_dir / "artifacts"
    if not artifact_dir.is_dir():
        return scores
    filenames: list[str] = []
    text_blobs: list[str] = []
    skip_names = {
        "source_notes.md",
        "readme.md",
        "license",
        "license.md",
        "license.txt",
        "structured.json",  # parser-emitted derived artifact
        "structured.md",
        "orbitbrief.input.json",
        "orbitbrief.input.md",
    }
    skip_substrings = ("gold_standard", ".gold.", "_gold.", "_review.", ".review.")
    skip_dir_suffixes = (".derived",)
    skip_dir_names = {".orbitbrief", ".cache", ".git", ".github", "labels", "node_modules", "__pycache__"}
    for path in artifact_dir.rglob("*"):
        if not path.is_file():
            continue
        # Skip files inside parser-managed derived / output dirs
        try:
            rel_parts = path.relative_to(artifact_dir).parts
        except ValueError:
            rel_parts = (path.name,)
        if any(part.endswith(skip_dir_suffixes) for part in rel_parts[:-1]):
            continue
        if any(part.lower() in skip_dir_names for part in rel_parts[:-1]):
            continue
        name_lower = path.name.lower()
        if name_lower in skip_names:
            continue
        if any(token in name_lower for token in skip_substrings):
            continue
        filenames.append(path.name)
        text_preview = _read_text_preview(path)
        if text_preview:
            text_blobs.append(text_preview)
    if not filenames:
        return scores

    # Filename keyword hits get weighted higher than content hits
    # (filenames are deliberate signals, content can have incidental
    # word collisions).  We also skip wide reference packs in the
    # filename pass — same rationale as the content pass below.
    filename_scores = _filename_hint_score(filenames)
    for pack_id, sc in filename_scores.items():
        try:
            pack = load_domain_pack(pack_id)
            service_lines = [sl.lower() for sl in (pack.service_lines or [])]
            if pack.reference_ontology_path:
                continue
            if not service_lines or service_lines == ["default"]:
                continue
        except Exception:  # pragma: no cover
            pass
        scores[pack_id] = scores.get(pack_id, 0.0) + sc * 2.0

    if text_blobs:
        haystack = "\n".join(text_blobs).lower()
        for pack_id in pack_ids:
            try:
                pack = load_domain_pack(pack_id)
            except Exception:  # pragma: no cover
                continue
            # Skip wide reference packs that intentionally cover many
            # service lines — they'd dominate scoring otherwise.  A
            # pack is "wide" if it declares only the default service
            # line or carries a reference_ontology_path.
            service_lines = [sl.lower() for sl in (pack.service_lines or [])]
            if pack.reference_ontology_path:
                continue
            if not service_lines or service_lines == ["default"]:
                continue
            hits = 0
            for surfaces in (pack.device_aliases or {}).values():
                for surface in surfaces:
                    if surface and surface.lower() in haystack:
                        hits += 1
            for entity in pack.entity_types or []:
                for alias in entity.aliases or []:
                    if alias.lower() in haystack:
                        hits += 1
            if hits:
                scores[pack_id] = scores.get(pack_id, 0.0) + float(hits)
    return scores


def _route_from_content(project_dir: Path) -> RoutingDecision | None:
    scores = _content_score(project_dir)
    if not scores:
        return None
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    best_id, best_score = ranked[0]
    runner_score = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = best_score - runner_score
    # Require both a positive score and a measurable margin to commit
    # to a non-default pack.  Otherwise fall through to default_pack.
    if best_score < 2.0 or margin < 1.0:
        return None
    confidence = min(0.85, 0.55 + 0.05 * best_score)
    return RoutingDecision(
        pack_id=best_id,
        source="content",
        confidence=confidence,
        rationale=(
            f"content scoring picked {best_id} (score {best_score:.1f}, "
            f"margin {margin:.1f} over runner-up)"
        ),
        alternatives=[(pid, score) for pid, score in ranked[1:6]],
    )


def auto_route_pack(
    project_dir: Path,
    *,
    explicit: str | Path | None = None,
) -> tuple[DomainPack, RoutingDecision]:
    """Pick the best DomainPack for ``project_dir``.

    ``explicit`` (if given) overrides all auto-detection — it's whatever
    the user passed via ``--domain-pack``.  Returns the loaded pack
    plus the decision rationale for telemetry.
    """
    if explicit is not None:
        # User-supplied pack id wins.  Validate by loading; on failure
        # fall through to default_pack.
        try:
            pack = load_domain_pack(explicit)
            pack_path = _candidate_pack_path(explicit)
            return pack, RoutingDecision(
                pack_id=pack.pack_id,
                source="explicit",
                confidence=1.0,
                rationale=f"user override --domain-pack={explicit} (resolved to {pack_path.name})",
            )
        except Exception as exc:  # pragma: no cover
            return load_domain_pack(None), RoutingDecision(
                pack_id=DEFAULT_PACK_ID,
                source="default",
                confidence=0.30,
                rationale=f"failed to load explicit pack {explicit!r}: {exc}; using default_pack",
            )

    for resolver in (_route_from_project_yaml, _route_from_source_notes, _route_from_content):
        decision = resolver(project_dir)
        if decision is None:
            continue
        try:
            pack = load_domain_pack(decision.pack_id)
        except Exception:  # pragma: no cover — bad pack id, fall through
            continue
        return pack, decision

    return load_domain_pack(None), RoutingDecision(
        pack_id=DEFAULT_PACK_ID,
        source="default",
        confidence=0.50,
        rationale="no project.yaml / SOURCE_NOTES service line / content scoring signal; using default_pack",
    )


__all__ = ["auto_route_pack", "RoutingDecision"]
