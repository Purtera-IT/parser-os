from __future__ import annotations

from pathlib import Path

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


def _artifact_type_for_path(path: Path) -> ArtifactType:
    suffix = path.suffix.lower()
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
    try:
        with open(path, "rb") as fh:
            head = fh.read(8)
    except Exception:
        return None, None
    if head.startswith(b"%PDF"):
        atype: ArtifactType | None = ArtifactType.pdf
    elif head.startswith(b"PK\x03\x04"):
        atype = _ooxml_or_zip(path)
    elif head[:5].lower().startswith(b"{\\rtf"):
        atype = ArtifactType.rtf
    else:
        atype = None
    if atype is None:
        return None, None
    cls_name = _SNIFF_TYPE_TO_CLASS.get(atype)
    for parser in get_registered_parsers():
        if cls_name and type(parser).__name__ == cls_name:
            return parser, atype
    return None, None


def choose_parser(
    path: Path,
    domain_pack: DomainPack | None = None,
) -> tuple[ArtifactParser | None, ParserMatch, list[ParserMatch]]:
    sample_text: str | None = None
    if path.suffix.lower() in {".txt", ".md", ".eml", ".json", ".jsonl", ".csv", ".vtt", ".srt"}:
        try:
            sample_text = path.read_text(encoding="utf-8", errors="ignore")[:4000]
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
        return parser, match, sorted_matches
    parser, match = _deterministic_tie_break(path, sample_text or "", top)
    return parser, match, sorted_matches
