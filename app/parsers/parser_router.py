from __future__ import annotations

from pathlib import Path

from app.core.ids import stable_id
from app.core.schemas import EvidenceAtom, ParserCapability, ParserMatch
from app.domain.schemas import DomainPack
from app.parsers.base import ArtifactParser
from app.parsers.registry import choose_parser as registry_choose_parser
from app.parsers.registry import get_registered_parsers


def choose_parser(
    path: Path,
    domain_pack: DomainPack | None = None,
) -> tuple[ArtifactParser | None, ParserMatch, list[ParserMatch]]:
    return registry_choose_parser(path=path, domain_pack=domain_pack)


def parser_capabilities() -> list[ParserCapability]:
    return sorted([parser.capability for parser in get_registered_parsers()], key=lambda row: row.parser_name)


def parse_artifact(
    project_id: str,
    artifact_id: str,
    path: Path,
    domain_pack: DomainPack | None = None,
) -> list[EvidenceAtom]:
    parser, _, _ = choose_parser(path=path, domain_pack=domain_pack)
    if parser is None:
        return []
    return parser.parse_artifact(project_id=project_id, artifact_id=artifact_id, path=path, domain_pack=domain_pack)


def parse_artifact_file(path: Path) -> list[EvidenceAtom]:
    """Compatibility wrapper for older call sites."""
    artifact_id = stable_id("art", str(path))
    return parse_artifact(project_id="unknown_project", artifact_id=artifact_id, path=path, domain_pack=None)
