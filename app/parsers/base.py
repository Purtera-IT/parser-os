from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from app.core.schemas import ArtifactType, ParserCapability, ParserMatch, ParserOutput
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
    ) -> ParserOutput:
        """Preferred parser interface for Parser OS."""
        del project_id, artifact_id
        del domain_pack
        parsed = self.parse(path)
        if isinstance(parsed, ParserOutput):
            return parsed
        return ParserOutput(atoms=list(parsed))

    @abstractmethod
    def parse(self, artifact_path: Path) -> list[Any]:
        raise NotImplementedError


class BaseParser(ArtifactParser):
    pass
