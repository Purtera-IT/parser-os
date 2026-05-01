from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass

ALLOWED_EXTENSIONS = {".xlsx", ".csv", ".txt", ".md", ".eml", ".docx", ".vtt", ".srt", ".json"}
DEFAULT_MAX_UPLOAD_BYTES = 25 * 1024 * 1024
_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class SafeUpload:
    sanitized_filename: str
    extension: str
    sha256: str
    size_bytes: int
    storage_filename: str


def max_upload_bytes() -> int:
    raw = os.getenv("PURTERA_MAX_UPLOAD_BYTES", str(DEFAULT_MAX_UPLOAD_BYTES))
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_UPLOAD_BYTES
    return max(1, value)


def sanitize_filename(filename: str) -> str:
    name = filename.strip()
    if not name:
        raise ValueError("Filename is required.")
    if "/" in name or "\\" in name or ".." in name:
        raise ValueError("Path traversal is not allowed in filenames.")
    normalized = _SAFE_FILENAME_RE.sub("_", name).strip("._")
    if not normalized:
        raise ValueError("Filename is invalid after sanitization.")
    return normalized


def validate_upload(filename: str, content: bytes) -> SafeUpload:
    safe_name = sanitize_filename(filename)
    extension = os.path.splitext(safe_name)[1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise ValueError(f"Unsupported file extension '{extension}'. Allowed extensions: {allowed}")

    size_bytes = len(content)
    if size_bytes == 0:
        raise ValueError("Uploaded file is empty.")
    max_bytes = max_upload_bytes()
    if size_bytes > max_bytes:
        raise ValueError(f"Uploaded file exceeds max size of {max_bytes} bytes.")

    digest = hashlib.sha256(content).hexdigest()
    storage_filename = f"{digest}{extension}"
    return SafeUpload(
        sanitized_filename=safe_name,
        extension=extension,
        sha256=digest,
        size_bytes=size_bytes,
        storage_filename=storage_filename,
    )
