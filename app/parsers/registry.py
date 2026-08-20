from __future__ import annotations

from app.core.textio import read_text

from pathlib import Path

from app.core.filetype import sniff
from app.core.normalizers import normalize_text
from app.core.schemas import ArtifactType, ParserMatch
from app.domain.schemas import DomainPack
from app.parsers.base import ArtifactParser

_REGISTERED: list[ArtifactParser] = []
_DEFAULTS_REGISTERED = False
MATCH_THRESHOLD = 0.5


def register_parser(parser: ArtifactParser) -> None:
    if any(existing.capability.parser_name == parser.capability.parser_name for existing in _REGISTERED):
        return
    _REGISTERED.append(parser)


def _ensure_defaults() -> None:
    global _DEFAULTS_REGISTERED
    if _DEFAULTS_REGISTERED:
        return
    from app.parsers.docx_parser import DocxParser
    from app.parsers.email_parser import EmailParser
    from app.parsers.image_parser import ImageParser
    from app.parsers.json_parser import JsonParser
    from app.parsers.markdown_parser import MarkdownParser
    from app.parsers.orbitbrief_pdf import OrbitBriefPdfParser
    from app.parsers.pptx_parser import PptxParser
    from app.parsers.quote_parser import QuoteParser
    from app.parsers.hubspot_note_parser import HubspotNoteParser
    from app.parsers.transcript_parser import TranscriptParser
    from app.parsers.universal_parsers import (
        HtmlParser, IcsParser, MboxParser, RtfParser, ZipParser,
    )
    from app.parsers._universal_extras import (
        MsgParser, OdtParser, OdsParser, VsdxParser, MppParser,
    )
    from app.parsers.xlsx_parser import XlsxParser

    for parser in [
        MarkdownParser(),
        XlsxParser(),
        QuoteParser(),
        EmailParser(),
        HubspotNoteParser(),
        TranscriptParser(),
        JsonParser(),
        DocxParser(),
        PptxParser(),
        ImageParser(),
        HtmlParser(),
        MboxParser(),
        RtfParser(),
        IcsParser(),
        ZipParser(),
        MsgParser(),
        OdtParser(),
        OdsParser(),
        VsdxParser(),
        MppParser(),
        OrbitBriefPdfParser(),
    ]:
        register_parser(parser)
    _DEFAULTS_REGISTERED = True


def get_registered_parsers() -> list[ArtifactParser]:
    _ensure_defaults()
    return list(_REGISTERED)


def _artifact_type_for_suffix(suffix: str) -> ArtifactType:
    suffix = (suffix or "").lower()
    if suffix == ".xlsx":
        return ArtifactType.xlsx
    if suffix == ".csv":
        return ArtifactType.csv
    if suffix == ".docx":
        return ArtifactType.docx
    if suffix == ".eml":
        return ArtifactType.email
    if suffix in {".vtt", ".srt"}:
        return ArtifactType.transcript
    if suffix == ".pdf":
        return ArtifactType.pdf
    if suffix == ".pptx":
        return ArtifactType.pptx
    if suffix in {".png", ".jpg", ".jpeg", ".heic", ".heif", ".webp", ".tiff", ".tif", ".bmp"}:
        return ArtifactType.image
    if suffix in {".html", ".htm", ".xhtml"}:
        return ArtifactType.html
    if suffix == ".mbox":
        return ArtifactType.mbox
    if suffix == ".rtf":
        return ArtifactType.rtf
    if suffix in {".ics", ".ical"}:
        return ArtifactType.ics
    if suffix == ".zip":
        return ArtifactType.zip_archive
    if suffix == ".msg":
        return ArtifactType.msg
    if suffix == ".odt":
        return ArtifactType.odt
    if suffix == ".ods":
        return ArtifactType.ods
    if suffix in {".vsdx", ".vsd"}:
        return ArtifactType.vsdx
    if suffix == ".mpp":
        return ArtifactType.mpp
    if suffix in {".json", ".jsonl"}:
        return ArtifactType.json
    return ArtifactType.txt


def _artifact_type_for_path(path: Path) -> ArtifactType:
    """Route on what the file *is*, falling back to what it is called.

    The extension is right almost always and is far cheaper, so it is tried
    first and kept unless the bytes disagree. Two cases where they do:

      * no usable extension -- an email attachment saved as "message", a DMS
        export named for its document number. The suffix table returns ``txt``
        and the file reaches a parser that cannot read it, or none at all.
      * a wrong extension -- a .docx renamed contract.pdf goes to the PDF
        parser, which cannot open a ZIP.

    A magic-number match is decisive in a way a filename is not, so when the
    sniffer has an opinion *and* that opinion is a format we parse, it wins.
    Silence from the sniffer (a CSV has no signature) leaves the suffix alone.
    """
    by_suffix = _artifact_type_for_suffix(path.suffix)
    sniffed = sniff(path)
    if not sniffed:
        return by_suffix
    by_content = _artifact_type_for_suffix(sniffed)
    if by_content is ArtifactType.txt:
        return by_suffix  # the sniffer saw something we have no parser for
    if by_content is not by_suffix:
        return by_content
    return by_suffix


def _deterministic_tie_break(
    path: Path,
    sample_text: str,
    candidates: list[tuple[ArtifactParser, ParserMatch]],
) -> tuple[ArtifactParser, ParserMatch]:
    name = path.name.lower()
    lowered = normalize_text(sample_text)
    by_name = {match.parser_name: (parser, match) for parser, match in candidates}
    if "hubspot_note" in by_name and (
        "-hs-note-" in name or "hubspot note:" in lowered
    ):
        return by_name["hubspot_note"]
    if {"email", "transcript"}.issubset(by_name):
        email_markers = ("from:" in lowered and "sent:" in lowered) or (" wrote:" in lowered)
        meeting_markers = ("decisions:" in lowered) or ("open questions:" in lowered) or ("[00:" in lowered)
        if email_markers and not meeting_markers:
            return by_name["email"]
        if meeting_markers and "hubspot note:" not in lowered:
            return by_name["transcript"]
    if {"quote", "xlsx"}.issubset(by_name):
        from app.parsers.spreadsheet_route_signals import resolve_quote_vs_xlsx_tie

        choice, tie_reasons = resolve_quote_vs_xlsx_tie(path)
        parser, match = by_name[choice]
        merged = list(match.reasons) + [f"router:{r}" for r in tie_reasons]
        return parser, match.model_copy(update={"reasons": merged})
    ranked = sorted(candidates, key=lambda row: row[0].capability.parser_name)
    return ranked[0]


# ── magic-byte fallback ──────────────────────────────────────────────
# Suffix-based routing drops a real document when it has no extension (an
# extensionless %PDF export) or a wrong one. Sniff the leading bytes so content,
# not the filename, decides — a file is never silently dropped for lacking a suffix.
_SNIFF_TYPE_TO_CLASS = {
    ArtifactType.pdf: "OrbitBriefPdfParser",
    ArtifactType.xlsx: "XlsxParser",
    ArtifactType.docx: "DocxParser",
    ArtifactType.pptx: "PptxParser",
    ArtifactType.zip_archive: "ZipParser",
    ArtifactType.rtf: "RtfParser",
    ArtifactType.email: "EmailParser",
    ArtifactType.transcript: "TranscriptParser",
    ArtifactType.html: "HtmlParser",
    ArtifactType.mbox: "MboxParser",
    ArtifactType.ics: "IcsParser",
    ArtifactType.odt: "OdtParser",
    ArtifactType.ods: "OdsParser",
    ArtifactType.vsdx: "VsdxParser",
}


def _ooxml_or_zip(path: Path) -> ArtifactType:
    """A PK\\x03\\x04 file is a zip container — peek members to tell an OOXML
    document (docx/xlsx/pptx) from a plain archive."""
    import zipfile
    try:
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
    except Exception:
        return ArtifactType.zip_archive
    if any(n.startswith("word/") for n in names):
        return ArtifactType.docx
    if any(n.startswith("xl/") for n in names):
        return ArtifactType.xlsx
    if any(n.startswith("ppt/") for n in names):
        return ArtifactType.pptx
    return ArtifactType.zip_archive


def _sniff_parser(path: Path) -> tuple[ArtifactParser | None, ArtifactType | None]:
    # Previously PDF/OOXML/RTF only, which left every text-based format --
    # .eml, .vtt, .html, .ics -- unroutable without its extension, even though
    # each one opens with a literal self-declaration of what it is.
    sniffed = sniff(path)
    if not sniffed:
        return None, None
    atype = _artifact_type_for_suffix(sniffed)
    if atype is ArtifactType.txt:
        return None, None
    cls_name = _SNIFF_TYPE_TO_CLASS.get(atype)
    for parser in get_registered_parsers():
        if cls_name and type(parser).__name__ == cls_name:
            return parser, atype
    return None, None


def _content_override(
    path: Path,
    sniffed_ext: str | None,
    match: ParserMatch,
    winner: ArtifactParser,
) -> tuple[ArtifactParser, ParserMatch] | None:
    """Prefer the file's own signature when it contradicts its name.

    Fires only when the sniffer is confident -- a magic number or a literal
    format declaration -- *and* names a format that disagrees with the parser
    the suffix selected. A .docx renamed contract.pdf otherwise reaches the
    PDF parser, which cannot open a ZIP.

    Deliberately narrow: silence from the sniffer, or agreement between the
    two, changes nothing, so this can never re-route a correctly routed file.
    """
    if not sniffed_ext:
        return None
    by_content = _artifact_type_for_suffix(sniffed_ext)
    if by_content is ArtifactType.txt:
        return None
    # Compare against the SUFFIX, not against the parser that won. Several
    # parsers legitimately handle one format -- a vendor quote and a generic
    # workbook are both .xlsx, and the quote parser wins on content confidence.
    # Testing against match.artifact_type treated that specialisation as a
    # disagreement and demoted every quote to the generic xlsx parser.
    # The only thing this is entitled to correct is a filename that lied.
    if by_content is _artifact_type_for_suffix(path.suffix):
        return None
    cls_name = _SNIFF_TYPE_TO_CLASS.get(by_content)
    if not cls_name:
        return None
    for parser in get_registered_parsers():
        if type(parser).__name__ == cls_name:
            if parser is winner or type(parser) is type(winner):
                # Already the right parser. Several parsers sniff their own
                # magic bytes, so the winner may have reached the same answer
                # by itself -- overriding then changes nothing except the
                # reason string, which callers and tests read.
                return None
            return parser, ParserMatch(
                parser_name=parser.capability.parser_name,
                confidence=max(match.confidence, 0.6),
                reasons=[f"content_overrides_suffix:{sniffed_ext}"],
                artifact_type=by_content,
            )
    return None


def choose_parser(
    path: Path,
    domain_pack: DomainPack | None = None,
) -> tuple[ArtifactParser | None, ParserMatch, list[ParserMatch]]:
    sample_text: str | None = None
    # Content first: an .eml attachment saved as "message" has no suffix to
    # match on, so without this the text parsers never see its body at all.
    _sniffed_ext = sniff(path)
    _effective_suffix = (_sniffed_ext or path.suffix).lower()
    if _effective_suffix in {".txt", ".md", ".eml", ".json", ".jsonl", ".csv",
                             ".vtt", ".srt", ".html", ".mbox", ".ics"}:
        try:
            sample_text = read_text(path)[:4000]
        except Exception:
            sample_text = ""
    parsers = get_registered_parsers()
    matches: list[tuple[ArtifactParser, ParserMatch]] = []
    for parser in parsers:
        match = parser.match(path, sample_text, domain_pack)
        matches.append((parser, match))

    sorted_matches = sorted(
        [match for _, match in matches],
        key=lambda row: (-row.confidence, row.parser_name),
    )
    viable = [(parser, match) for parser, match in matches if match.confidence >= MATCH_THRESHOLD]
    if not viable:
        # Suffix-based routing found nothing — fall back to magic-byte sniffing
        # so an extensionless or mis-named real document isn't dropped.
        sniffed, atype = _sniff_parser(path)
        if sniffed is not None and atype is not None:
            return (
                sniffed,
                ParserMatch(
                    parser_name=sniffed.capability.parser_name,
                    confidence=0.5,
                    reasons=["magic_byte_sniff"],
                    artifact_type=atype,
                ),
                sorted_matches,
            )
        return (
            None,
            ParserMatch(
                parser_name="none",
                confidence=0.0,
                reasons=["no_parser_over_threshold"],
                artifact_type=_artifact_type_for_path(path),
            ),
            sorted_matches,
        )
    max_confidence = max(match.confidence for _, match in viable)
    top = [(parser, match) for parser, match in viable if abs(match.confidence - max_confidence) < 1e-9]
    if len(top) == 1:
        parser, match = top[0]
        override = _content_override(path, _sniffed_ext, match, parser)
        if override is not None:
            return override[0], override[1], sorted_matches
        return parser, match, sorted_matches
    parser, match = _deterministic_tie_break(path, sample_text or "", top)
    return parser, match, sorted_matches
