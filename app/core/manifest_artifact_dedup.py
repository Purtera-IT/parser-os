"""Drop stale duplicate HubSpot email artifacts from compile manifests.

When ingest re-writes an hs-email row with a new multipart .eml (higher
``emlBuilderVersion``), the old plain-text blob can remain as a separate
attachment row. Manifest builders must keep only the best row per HubSpot
email id so OCR runs on inline CID PDFs, not a 2KB text stub.

Also: when a compile manifest still points at a plain-text stub but a larger
multipart sibling already exists under ``deals/<deal>/artifacts/*/``, rewrite
the stub row to that sibling so CID OCR can run without a HubSpot re-ingest.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable
from urllib.parse import unquote, urlparse

_HS_EMAIL_EXTERNAL_RE = re.compile(r"^hs-email:(\d+)$", re.I)
_HS_EMAIL_FILENAME_RE = re.compile(r"-hs-email-(\d+)\.eml$", re.I)
# Plain-text HubSpot stubs are typically ~2KB; multipart with inline PNGs is >>8KB.
_STUB_EML_SIZE_BYTES = 8192

log = logging.getLogger(__name__)


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


def _is_stub_hs_email(artifact: dict[str, Any]) -> bool:
    """True when the row looks like a plain-text hs-email stub (no inline MIME)."""
    if not _hubspot_email_id(artifact):
        return False
    if _size_bytes(artifact) >= _STUB_EML_SIZE_BYTES:
        return False
    meta = _metadata_dict(artifact)
    try:
        if int(meta.get("inlineImageParts") or 0) > 0:
            return False
    except (TypeError, ValueError):
        pass
    if _eml_builder_version(artifact) >= 2 and _size_bytes(artifact) >= _STUB_EML_SIZE_BYTES:
        return False
    return True


def _artifact_blob_url(artifact: dict[str, Any]) -> str:
    return str(artifact.get("blob_url") or artifact.get("url") or "").strip()


def _parse_artifact_blob_url(blob_url: str) -> tuple[str, str] | None:
    """Return ``(container, blob_path)`` or None when unparseable."""
    if not blob_url:
        return None
    parsed = urlparse(blob_url)
    parts = parsed.path.lstrip("/").split("/", 1)
    if len(parts) != 2:
        return None
    return unquote(parts[0]), unquote(parts[1])


def _deal_artifacts_prefix(blob_path: str) -> str | None:
    """``deals/<deal_id>/artifacts/<sha>/<file>`` → ``deals/<deal_id>/artifacts/``."""
    parts = blob_path.replace("\\", "/").split("/")
    if len(parts) < 4 or parts[0] != "deals" or parts[2] != "artifacts":
        return None
    return "/".join(parts[:3]) + "/"


def _sibling_blob_url(container: str, blob_path: str, *, account_host: str | None) -> str:
    if account_host:
        return f"https://{account_host}/{container}/{blob_path}"
    return f"{container}/{blob_path}"


def upgrade_stub_hs_email_artifacts_from_siblings(
    artifacts: list[dict[str, Any]],
    *,
    list_blobs: Callable[[str, str], list[tuple[str, int]]],
    account_host: str | None = None,
) -> list[dict[str, Any]]:
    """Rewrite stub hs-email rows to a larger multipart sibling already in blob storage.

    ``list_blobs(container, prefix)`` must return ``[(blob_path, size_bytes), ...]``.
    Used by the worker so stale recompile bases (2KB plain-text .eml) still OCR
    inline CID images when a v3 multipart sibling was ingested earlier.
    """
    if not artifacts:
        return []

    out: list[dict[str, Any]] = []
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        row = dict(artifact)
        if not _is_stub_hs_email(row):
            out.append(row)
            continue

        blob_url = _artifact_blob_url(row)
        parsed = _parse_artifact_blob_url(blob_url)
        if not parsed:
            out.append(row)
            continue
        container, blob_path = parsed
        prefix = _deal_artifacts_prefix(blob_path)
        filename = str(row.get("filename") or "").replace("\\", "/").split("/")[-1]
        if not prefix or not filename:
            out.append(row)
            continue

        try:
            siblings = list_blobs(container, prefix)
        except Exception as exc:  # pragma: no cover — storage failures stay on stub
            log.warning("stub hs-email sibling list failed for %s: %s", filename, exc)
            out.append(row)
            continue

        best_path = ""
        best_size = _size_bytes(row)
        for sib_path, sib_size in siblings:
            sib_name = sib_path.replace("\\", "/").split("/")[-1]
            if sib_name != filename:
                continue
            if int(sib_size or 0) <= best_size:
                continue
            best_size = int(sib_size or 0)
            best_path = sib_path

        if not best_path or best_size <= _size_bytes(row):
            out.append(row)
            continue

        sha = best_path.replace("\\", "/").split("/")[-2] if "/" in best_path else ""
        meta = dict(_metadata_dict(row))
        meta.setdefault("upgradedFromStub", True)
        meta.setdefault("stubSizeBytes", _size_bytes(row))
        if best_size >= _STUB_EML_SIZE_BYTES:
            meta.setdefault("inlineImageParts", meta.get("inlineImageParts") or 1)
            meta.setdefault("emlBuilderVersion", meta.get("emlBuilderVersion") or "3")
            meta.setdefault("hasBodyHtml", True)
        row["blob_url"] = _sibling_blob_url(container, best_path, account_host=account_host)
        row["size_bytes"] = best_size
        if sha and len(sha) >= 32:
            row["content_sha256"] = sha
        row["metadata"] = meta
        email_id = _hubspot_email_id(row)
        if email_id and not row.get("external_id"):
            row["external_id"] = f"hs-email:{email_id}"
        log.info(
            "upgraded stub hs-email %s %sB → %sB (%s)",
            filename,
            meta.get("stubSizeBytes"),
            best_size,
            best_path,
        )
        out.append(row)

    return dedupe_manifest_email_artifacts(out)


__all__ = [
    "dedupe_manifest_email_artifacts",
    "email_artifact_preference_score",
    "upgrade_stub_hs_email_artifacts_from_siblings",
]
