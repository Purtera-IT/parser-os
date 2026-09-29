from __future__ import annotations

import threading
import weakref
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


class PerThreadState:
    """A parser attribute whose value belongs to the thread that set it.

    ``registry._REGISTERED`` holds exactly ONE instance of each parser and
    ``choose_parser`` hands that same object to every caller, so a parser that
    parks per-DOCUMENT state on ``self`` is sharing that state with every other
    document. Serially that is invisible -- each parse finishes before the next
    one starts -- but ``_prefetch_parses`` runs a thread pool, and then two
    documents are writing the same attribute at the same time.

    Two ways that went wrong in the live corpus:

    * ``DocxParser._table_lead_in`` is the prose introducing a table. Parsed
      alongside another document, 010237's SLA table picked up a lead-in that
      the serial parse did not give it -- the table's meaning came from
      whichever document happened to be interleaved.
    * ``XlsxParser._block_detection_failures`` accumulates during a parse and
      is drained and reset at the end. A second workbook resetting it first
      discards the first workbook's pending failures, so a sheet whose block
      detection crashed is never reported and simply reads as empty -- the one
      failure mode ``_note_block_failure`` exists to prevent.

    Storage is per thread and per instance. By default an unset attribute
    still raises ``AttributeError``, so the ``getattr(self, name, default)``
    reads throughout the parsers keep their exact present behaviour. Pass a
    ``factory`` for state that was previously a class-level ``= []`` default
    and is read directly: each thread then gets its own fresh value instead of
    sharing one mutable list with every other document.

    Single-threaded parsing is unchanged; concurrent parsing stops crossing
    documents.
    """

    __slots__ = ("_name", "_local", "_factory")

    def __init__(self, factory: Any = None) -> None:
        self._local = threading.local()
        self._factory = factory

    def __set_name__(self, owner: type, name: str) -> None:
        self._name = name

    def _slots(self, obj: Any) -> dict:
        table = getattr(self._local, "table", None)
        if table is None:
            # Weak, so a parser that is not a registry singleton is still
            # collectable and this never becomes a per-thread leak.
            table = weakref.WeakKeyDictionary()
            self._local.table = table
        slots = table.get(obj)
        if slots is None:
            slots = {}
            table[obj] = slots
        return slots

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        if obj is None:
            return self
        slots = self._slots(obj)
        if self._name not in slots:
            if self._factory is None:
                raise AttributeError(self._name)
            slots[self._name] = self._factory()
        return slots[self._name]

    def __set__(self, obj: Any, value: Any) -> None:
        self._slots(obj)[self._name] = value

    def __delete__(self, obj: Any) -> None:
        try:
            del self._slots(obj)[self._name]
        except KeyError:
            raise AttributeError(self._name) from None


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
