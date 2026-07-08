"""Drop stale duplicate HubSpot email artifacts from compile manifests.

When ingest re-writes an hs-email row with a new multipart .eml (higher
``emlBuilderVersion``), the old plain-text blob can remain as a separate
attachment row. Manifest builders must keep only the best row per HubSpot
email id so OCR runs on inline CID PDFs, not a 2KB text stub.
"""

from __future__ import annotations

import re
from typing import Any

_HS_EMAIL_EXTERNAL_RE = re.compile(r"^hs-email:(\d+)$", re.I)
_HS_EMAIL_FILENAME_RE = re.compile(r"-hs-email-(\d+)\.eml$", re.I)


def _hubspot_email_id(artifact: dict[str, Any]) -> str | None:
    external = str(artifact.get("external_id") or "").strip()
    match = _HS_EMAIL_EXTERNAL_RE.match(external)
    if match:
        return match.group(1)
    filename = str(artifact.get("filename") or "").replace("\\", "/").split("/")[-1]
    match = _HS_EMAIL_FILENAME_RE.search(filename)
    if match:
        return match.group(1)
    return None


def _metadata_dict(artifact: dict[str, Any]) -> dict[str, Any]:
    meta = artifact.get("metadata")
    return meta if isinstance(meta, dict) else {}


def _eml_builder_version(artifact: dict[str, Any]) -> int:
    raw = _metadata_dict(artifact).get("emlBuilderVersion")
    try:
        return int(str(raw or "0").strip() or "0")
    except ValueError:
        return 0


def _size_bytes(artifact: dict[str, Any]) -> int:
    try:
        return max(0, int(artifact.get("size_bytes") or 0))
    except (TypeError, ValueError):
        return 0


def _timestamp_ms(artifact: dict[str, Any]) -> int:
    for key in ("updated_at", "created_at", "ingestedAt"):
        raw = artifact.get(key) or _metadata_dict(artifact).get(key)
        if not raw:
            continue
        try:
            return int(__import__("datetime").datetime.fromisoformat(str(raw).replace("Z", "+00:00")).timestamp() * 1000)
        except ValueError:
            continue
    return 0


def _multipart_score(artifact: dict[str, Any]) -> int:
    meta = _metadata_dict(artifact)
    score = 0
    if meta.get("hasBodyHtml"):
        score += 2
    inline_parts = meta.get("inlineImageParts")
    try:
        if int(inline_parts or 0) > 0:
            score += 4
    except (TypeError, ValueError):
        pass
    attachment_ids = meta.get("attachmentIds")
    if isinstance(attachment_ids, list) and attachment_ids:
        score += 1
    mime = str(artifact.get("mime_type") or "").lower()
    if mime == "message/rfc822" and _size_bytes(artifact) > 4096:
        score += 1
    return score


def email_artifact_preference_score(artifact: dict[str, Any]) -> tuple[int, int, int, int]:
    """Higher tuple wins when choosing among duplicate hs-email artifacts."""
    return (
        _eml_builder_version(artifact),
        _multipart_score(artifact),
        _size_bytes(artifact),
        _timestamp_ms(artifact),
    )


def dedupe_manifest_email_artifacts(artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the newest multipart hs-email artifact; drop stale plain-text duplicates."""
    if not artifacts:
        return []

    winners: dict[str, dict[str, Any]] = {}
    passthrough: list[dict[str, Any]] = []

    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        email_id = _hubspot_email_id(artifact)
        if not email_id:
            passthrough.append(artifact)
            continue
        current = winners.get(email_id)
        if current is None or email_artifact_preference_score(artifact) > email_artifact_preference_score(current):
            winners[email_id] = artifact

    out = passthrough + list(winners.values())
    # Preserve stable ordering for non-email rows; append deduped emails sorted by id.
    out.sort(
        key=lambda row: (
            1 if _hubspot_email_id(row) else 0,
            str(row.get("filename") or ""),
            str(row.get("attachment_id") or ""),
        ),
    )
    return out


__all__ = [
    "dedupe_manifest_email_artifacts",
    "email_artifact_preference_score",
]
