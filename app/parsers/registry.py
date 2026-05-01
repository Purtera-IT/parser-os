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
    from app.parsers.quote_parser import QuoteParser
    from app.parsers.transcript_parser import TranscriptParser
    from app.parsers.xlsx_parser import XlsxParser

    for parser in [XlsxParser(), QuoteParser(), EmailParser(), TranscriptParser(), DocxParser()]:
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
    return ArtifactType.txt


def _deterministic_tie_break(
    path: Path,
    sample_text: str,
    candidates: list[tuple[ArtifactParser, ParserMatch]],
) -> tuple[ArtifactParser, ParserMatch]:
    name = path.name.lower()
    lowered = normalize_text(sample_text)
    by_name = {match.parser_name: (parser, match) for parser, match in candidates}
    if {"email", "transcript"}.issubset(by_name):
        email_markers = ("from:" in lowered and "sent:" in lowered) or (" wrote:" in lowered)
        meeting_markers = ("decisions:" in lowered) or ("open questions:" in lowered) or ("[00:" in lowered)
        if email_markers and not meeting_markers:
            return by_name["email"]
        if meeting_markers:
            return by_name["transcript"]
    if {"quote", "xlsx"}.issubset(by_name):
        from app.parsers.spreadsheet_route_signals import resolve_quote_vs_xlsx_tie

        choice, tie_reasons = resolve_quote_vs_xlsx_tie(path)
        parser, match = by_name[choice]
        merged = list(match.reasons) + [f"router:{r}" for r in tie_reasons]
        return parser, match.model_copy(update={"reasons": merged})
    ranked = sorted(candidates, key=lambda row: row[0].capability.parser_name)
    return ranked[0]


def choose_parser(
    path: Path,
    domain_pack: DomainPack | None = None,
) -> tuple[ArtifactParser | None, ParserMatch, list[ParserMatch]]:
    sample_text: str | None = None
    if path.suffix.lower() in {".txt", ".md", ".eml", ".json", ".csv", ".vtt", ".srt"}:
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
