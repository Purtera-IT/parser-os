from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from app.core.schemas import (
    ArtifactType,
    EvidenceAtom,
    ParserCapability,
    ParserMatch,
    ParserOutput,
)
from app.domain.schemas import DomainPack

class ArtifactParser(ABC):
    parser_name: str
    parser_version: str = "unknown"
    capability: ParserCapability

    def match(self, path: Path, sample_text: str | None, domain_pack: DomainPack | None) -> ParserMatch:
        suffix = path.suffix.lower()
        confidence = 0.0
        reasons: list[str] = []
        if suffix in self.capability.supported_extensions:
            confidence = 0.6
            reasons.append(f"extension:{suffix}")
        artifact_type = (
            self.capability.supported_artifact_types[0]
            if self.capability.supported_artifact_types
            else ArtifactType.txt
        )
        return ParserMatch(
            parser_name=self.capability.parser_name,
            confidence=confidence,
            reasons=reasons,
            artifact_type=artifact_type,
        )

    def parse_artifact(
        self,
        project_id: str,
        artifact_id: str,
        path: Path,
        domain_pack: DomainPack | None = None,
    ) -> list[EvidenceAtom] | ParserOutput:
        """Legacy parser entry-point.

        Older parsers historically returned a flat ``list[EvidenceAtom]``
        from this method; some (PDF, post-v3) return a ``ParserOutput``
        envelope.  Prefer overriding :meth:`parse_artifact_full` going
        forward — Parser OS's compiler uses that as the canonical entry
        point and surfaces ``derived_files`` to the cache + envelope.
        """
        del project_id, artifact_id
        del domain_pack
        parsed = self.parse(path)
        if isinstance(parsed, ParserOutput):
            return parsed
        return ParserOutput(atoms=list(parsed))

    def parse_artifact_full(
        self,
        project_id: str,
        artifact_id: str,
        path: Path,
        domain_pack: DomainPack | None = None,
    ) -> ParserOutput:
        """Canonical parser entry-point.

        Always returns a :class:`ParserOutput` so the compiler can
        forward ``candidates``, ``warnings``, and especially
        ``derived_files`` (parser-emitted side files like
        ``structured.json`` / ``structured.md``) to the cache, the
        OrbitBrief envelope, and source-replay verifiers.

        The default implementation defers to :meth:`parse_artifact` and
        wraps a bare list of atoms.  Subclasses should override **either**
        ``parse_artifact_full`` (preferred) or ``parse_artifact`` —
        whichever is most natural for the parser.
        """
        result = self.parse_artifact(
            project_id=project_id,
            artifact_id=artifact_id,
            path=path,
            domain_pack=domain_pack,
        )
        if isinstance(result, ParserOutput):
            return result
        return ParserOutput(atoms=list(result))

    @abstractmethod
    def parse(self, artifact_path: Path) -> list[Any]:
        raise NotImplementedError


class BaseParser(ArtifactParser):
    pass
