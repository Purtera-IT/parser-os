"""Decode a text file without deleting the bytes it cannot read.

``read_text(encoding="utf-8", errors="ignore")`` appeared in forty-odd places.
It never raises, which is why it spread, and that is exactly the problem: the
failure mode is a character quietly vanishing, so nothing upstream ever learns
that a document was misread.

Measured on files a Windows user actually produces:

    cp1252 CSV   "Bjorn Andre" with accents  ->  "Bjrn Andr"
                 the contact name corrupted; curly quotes and em dashes gone
    UTF-16 txt   "PART-77-K298"  ->  "P\\x00A\\x00R\\x00T\\x00-..."
                 every second byte is a NUL, so the whole file is garbage

UTF-16 is what Notepad writes when you choose "Unicode", and what several CRM
and telecom exports emit. Under ``errors="ignore"`` such a file parses into
nonsense rather than failing, which produces confident, wrong atoms.

Order of attempts: the BOMs first, since they are unambiguous; then strict
UTF-8, which is right for most files and costs nothing; then
``charset-normalizer``; then cp1252 and latin-1, which cannot fail on any byte
sequence. The last resort still decodes every byte to *something* -- the point
is that no byte is ever silently dropped.
"""

from __future__ import annotations

import pathlib

try:  # pragma: no cover - exercised by whichever environment lacks it
    import charset_normalizer as _cn
except Exception:  # pragma: no cover
    _cn = None  # type: ignore[assignment]

#: Bytes inspected when sniffing an encoding. Detection quality plateaus well
#: before this; reading a 40 MB export in full to identify it is waste.
_SNIFF_BYTES = 65_536

#: Byte-order marks, longest first so UTF-32 is not misread as UTF-16.
#: These are the BOM-*consuming* codec names, not the -le/-be ones: decoding
#: with "utf-16-le" turns the mark itself into a leading U+FEFF, which then
#: travels into the atom text as an invisible character.
_BOMS: tuple[tuple[bytes, str], ...] = (
    (b"\xff\xfe\x00\x00", "utf-32"),
    (b"\x00\x00\xfe\xff", "utf-32"),
    (b"\xef\xbb\xbf", "utf-8-sig"),
    (b"\xff\xfe", "utf-16"),
    (b"\xfe\xff", "utf-16"),
)


def _newlines(text: str) -> str:
    """Universal-newline translation, matching ``open()`` in text mode.

    ``Path.read_text()`` does this; decoding bytes by hand does not. Every
    caller replaced here was reading through text mode, so without it a CRLF
    document arrives carrying carriage returns it never had before, and every
    ``line.strip()`` and ``== "WEBVTT"`` comparison downstream shifts under it.
    """
    if "\r" not in text:
        return text
    return text.replace("\r\n", "\n").replace("\r", "\n")


def decode_bytes(data: bytes) -> str:
    """Decode ``data`` to text, choosing an encoding rather than dropping bytes."""
    if not data:
        return ""

    for bom, enc in _BOMS:
        if data.startswith(bom):
            try:
                return _newlines(data.decode(enc))
            except (UnicodeDecodeError, LookupError):
                break  # declared a BOM and then lied; fall through to sniffing

    try:
        return _newlines(data.decode("utf-8"))
    except UnicodeDecodeError:
        pass

    if _cn is not None:
        try:
            best = _cn.from_bytes(data[:_SNIFF_BYTES]).best()
            if best is not None and best.encoding:
                return _newlines(data.decode(best.encoding, errors="replace"))
        except Exception:  # pragma: no cover - detection must never raise
            pass

    # cp1252 is the single most likely origin for a non-UTF-8 business
    # document, and latin-1 maps every possible byte, so this cannot fail.
    for enc in ("cp1252", "latin-1"):
        try:
            return _newlines(data.decode(enc))
        except (UnicodeDecodeError, LookupError):
            continue
    return _newlines(data.decode("utf-8", errors="replace"))


def read_text(path: str | pathlib.Path, *, max_bytes: int | None = None) -> str:
    """Read a text file, detecting its encoding.

    ``max_bytes`` caps the *bytes read*, for callers that only need a head in
    order to sniff a format. Truncation happens before decoding, so a
    multi-byte character split across the boundary becomes a replacement
    character rather than corrupting what follows it.

    A missing or unreadable file returns ``""``, matching what the
    ``errors="ignore"`` calls being replaced did on failure.
    """
    p = pathlib.Path(path)
    try:
        if max_bytes is not None:
            with p.open("rb") as fh:
                data = fh.read(max_bytes)
        else:
            data = p.read_bytes()
    except OSError:
        return ""
    return decode_bytes(data)
