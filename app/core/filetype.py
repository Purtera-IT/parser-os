"""Identify a file by its contents, because extensions lie.

Routing on ``path.suffix`` alone is correct for most files and silently wrong
for the ones that matter. Measured against the parser registry:

    a .docx renamed contract.pdf   ->  routed to the PDF parser
    an .eml saved as "message"     ->  routed to no parser at all

Both are ordinary: mail clients strip extensions from attachments, scanners
and DMS exports rename to a document number, and users retype extensions by
hand. The file itself is never ambiguous -- a DOCX is a ZIP with a ``word/``
directory in it, and no amount of renaming changes that.

``puremagic`` supplies the binary signature table (pure Python, no libmagic
system dependency, which matters because the worker image is slim). It cannot
tell the three OOXML formats apart, since they are all ZIPs, so the container
is opened and its members inspected; nor does it cover the text formats, whose
first bytes are their own unambiguous declaration (``WEBVTT``, ``BEGIN:VCALENDAR``).

Everything here reports what a file *is*. Deciding whether that beats the
extension is the registry's call, not this module's.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

try:  # pragma: no cover - exercised by whichever environment lacks it
    import puremagic as _puremagic
except Exception:  # pragma: no cover
    _puremagic = None  # type: ignore[assignment]

#: Members that identify an OOXML container. Checked as prefixes because the
#: leading entry may be "[Content_Types].xml" or "_rels/".
_OOXML_MARKERS: tuple[tuple[str, str], ...] = (
    ("word/", ".docx"),
    ("xl/", ".xlsx"),
    ("ppt/", ".pptx"),
    ("visio/", ".vsdx"),
)

#: OpenDocument declares itself in an uncompressed "mimetype" member.
_ODF_MIMETYPES: dict[str, str] = {
    "application/vnd.oasis.opendocument.text": ".odt",
    "application/vnd.oasis.opendocument.spreadsheet": ".ods",
    "application/vnd.oasis.opendocument.presentation": ".odp",
}

#: Text formats whose opening bytes are a literal declaration.
_TEXT_SIGNATURES: tuple[tuple[str, str], ...] = (
    ("WEBVTT", ".vtt"),
    ("BEGIN:VCALENDAR", ".ics"),
    ("{\rtf", ".rtf"),
    ("%PDF-", ".pdf"),
)

#: RFC 5322 headers that, appearing at the very top, mean this is a message.
_RFC822_HEADERS = ("from:", "received:", "return-path:", "message-id:", "subject:", "to:")


def _sniff_zip(path: Path) -> str | None:
    """Distinguish the ZIP-based formats by what is inside the container."""
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            if "mimetype" in names:
                try:
                    mt = zf.read("mimetype").decode("ascii", "replace").strip()
                    if mt in _ODF_MIMETYPES:
                        return _ODF_MIMETYPES[mt]
                except (KeyError, OSError):
                    pass
            for name in names:
                for marker, ext in _OOXML_MARKERS:
                    if name.startswith(marker):
                        return ext
    except (zipfile.BadZipFile, OSError):
        return None
    return ".zip"  # a real archive, just not an Office one


def _sniff_text(head: bytes) -> str | None:
    """Identify the text formats that announce themselves in their first line."""
    try:
        text = head.decode("utf-8-sig", errors="replace")
    except Exception:  # pragma: no cover - decode with replace does not raise
        return None
    stripped = text.lstrip()
    for sig, ext in _TEXT_SIGNATURES:
        if stripped.startswith(sig):
            return ext
    lowered = stripped.lower()
    if lowered.startswith("<!doctype html") or lowered.startswith("<html"):
        return ".html"
    if lowered.startswith("mime-version:") or any(
        lowered.startswith(h) for h in _RFC822_HEADERS
    ):
        # An mbox begins with the "From " separator line, not a header.
        if lowered.startswith("from ") and not lowered.startswith("from:"):
            return ".mbox"
        return ".eml"
    if stripped.startswith("From ") and "\n" in stripped:
        return ".mbox"
    return None


def sniff(path: Path) -> str | None:
    """Return the canonical extension this file's *contents* indicate.

    None means "no confident opinion" -- an unmarked text file, an empty file,
    or a format outside the tables above. Callers keep the extension then.
    """
    p = Path(path)
    try:
        with p.open("rb") as fh:
            head = fh.read(4096)
    except OSError:
        return None
    if not head:
        return None

    if head[:4] == b"PK\x03\x04":
        return _sniff_zip(p)
    if head[:5] == b"%PDF-":
        return ".pdf"

    text_guess = _sniff_text(head)
    if text_guess:
        return text_guess

    if _puremagic is not None:
        try:
            for match in _puremagic.magic_string(head):
                ext = (getattr(match, "extension", "") or "").lower()
                if ext:
                    return ext if ext.startswith(".") else f".{ext}"
        except Exception:
            return None
    return None
