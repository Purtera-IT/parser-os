"""The parser of last resort: a file nobody can read is still a file.

When no parser claims an artifact the router returns None and the compile
records `skipped_no_parser`. The file then exists in the manifest and nowhere
else -- not in the envelope, not in PM_HANDOFF, not in the Deal Kit -- so the
only way to learn it was in the intake is to go and look at the manifest.

Measured across 461 dev envelopes, that silence covers:

    .dwg    2   AutoCAD drawings (now read by `dwg_parser`)
    .json   6   historical; the JSON parser claims these at 0.55 now
    .xls    2   legacy Excel, pre-2007
    .rpmsg  2   RMS-encrypted Outlook mail, unreadable by design
    .doc    1   legacy Word
    .gif    1   an image the image parser's extension list omitted
    .tsv    1   tab-separated values

Some of those want a real parser and some genuinely cannot be read -- an
`.rpmsg` is encrypted and no amount of work here will open it. The point is
that the two look identical from outside, and both look like the file was never
sent. A marker costs one atom and turns "nothing" into "this arrived and
nobody could read it", which is a thing a PM can act on.

This parser never competes. It is consulted only where the router had already
decided to return None, so a real parser at any confidence still wins.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from app.core.ids import stable_id
from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    EvidenceAtom,
    ParserCapability,
    ParserMatch,
    ReviewStatus,
    SourceRef,
)
from app.domain.schemas import DomainPack
from app.parsers.base import BaseParser

#: What we can say about a format beyond its extension. Only entries where the
#: reason is worth a PM's attention -- "encrypted" is actionable, "unknown" is
#: not.
NUL = bytes([0])

#: A line shorter than this is not a statement. Matches multitask_table's floor.
MIN_LINE = 8
#: The fallback is not a real parser; it should not flood a deal.
MAX_LINES = 400


def _as_text(path: Path) -> str:
    """The file's contents when it is text, else "". Binary decides itself:
    a NUL byte in the first block is the oldest and most reliable signal."""
    try:
        head = path.read_bytes()[:8192]
    except OSError:
        return ""
    if NUL in head:
        return ""
    try:
        body = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        try:
            body = path.read_text(encoding="latin-1")
        except (OSError, UnicodeDecodeError):
            return ""
    printable = sum(1 for ch in body[:4000] if ch.isprintable() or ch.isspace())
    return body if printable >= 0.9 * max(len(body[:4000]), 1) else ""


_KNOWN = {
    ".rpmsg": "an RMS-encrypted Outlook message; it cannot be opened without "
              "the sender's rights policy, so ask for an unprotected copy",
    ".doc": "legacy Word (pre-2007). Ask for a .docx",
    ".xls": "legacy Excel (pre-2007). Ask for a .xlsx",
    ".tsv": "tab-separated values. Ask for a .csv or .xlsx",
    ".zipx": "an extended ZIP archive",
    ".7z": "a 7-Zip archive",
    ".rar": "a RAR archive",
    ".dmg": "a macOS disk image",
    ".exe": "an executable, which should not be in a deal's documents at all",
}


def describe(path: Path) -> str:
    suffix = path.suffix.lower()
    known = _KNOWN.get(suffix)
    if known:
        return known
    if not suffix:
        return "a file with no extension, and its contents match no known format"
    return f"a {suffix.lstrip('.')} file, which no parser here claims"


class UnreadParser(BaseParser):
    parser_name = "unread"
    parser_version = "unread_parser_v1"
    capability = ParserCapability(
        parser_name=parser_name,
        parser_version=parser_version,
        supported_extensions=[],
        supported_artifact_types=[ArtifactType.txt],
        emitted_atom_types=[AtomType.open_question],
        supported_domain_packs=["*"],
        requires_binary=False,
        supports_source_replay=False,
    )

    def match(self, path: Path, sample_text: str | None,
              domain_pack: DomainPack | None) -> ParserMatch:
        """Never claims anything. The router reaches for this by name, at the
        point where it would otherwise return None, so it cannot outrank a
        parser that actually understands the file."""
        del path, sample_text, domain_pack
        return ParserMatch(parser_name=self.parser_name, confidence=0.0,
                           reasons=[], artifact_type=ArtifactType.txt)

    def parse(self, artifact_path: Path) -> list[Any]:
        return self.parse_artifact("unknown_project",
                                   stable_id("art", str(artifact_path)),
                                   artifact_path)

    def parse_artifact(self, project_id: str, artifact_id: str, path: Path,
                       domain_pack: DomainPack | None = None) -> list[EvidenceAtom]:
        del domain_pack
        try:
            size = path.stat().st_size
        except OSError:
            size = 0

        # A file that decodes as text is READABLE, whatever its extension, and
        # marking it unread would be a lie. No parser claims a plain .txt with
        # no structure -- not one, at any confidence -- so an unstructured text
        # file was losing its whole contents, which is worse than the binary
        # case this parser was written for. Read it.
        body = _as_text(path)
        if body:
            return self._lines(project_id, artifact_id, path, body, size)

        text = (f"[Unread file] {path.name} — {describe(path)}. "
                f"{size:,} bytes. Nothing in it reached this deal, so whatever "
                f"it says is missing from the brief and the SOW.")
        source_ref = SourceRef(
            id=stable_id("src", artifact_id, "unread", path.name),
            artifact_id=artifact_id,
            artifact_type=ArtifactType.txt,
            filename=path.name,
            locator={"kind": "unread_artifact", "suffix": path.suffix.lower()},
            extraction_method="unread_marker",
            parser_version=self.parser_version,
        )
        return [EvidenceAtom(
            id=stable_id("atm", project_id, artifact_id, "unread", path.name),
            project_id=project_id,
            artifact_id=artifact_id,
            atom_type=AtomType.open_question,
            raw_text=text,
            normalized_text=text.lower(),
            value={"kind": "unread_marker", "suffix": path.suffix.lower(),
                   "size_bytes": size},
            entity_keys=[],
            source_refs=[source_ref],
            authority_class=AuthorityClass.customer_current_authored,
            confidence=0.8,
            review_status=ReviewStatus.needs_review,
            parser_version=self.parser_version,
        )]

    def _lines(self, project_id: str, artifact_id: str, path: Path,
               body: str, size: int) -> list[EvidenceAtom]:
        """A readable file nobody specialised claimed, read plainly.

        One atom per non-empty line, capped: this is the fallback, not a real
        parser, and its job is that the words exist somewhere rather than that
        they are well modelled. A specialised parser claiming the file later
        supersedes this entirely.
        """
        seen: set[str] = set()
        out: list[EvidenceAtom] = []
        for index, raw in enumerate(body.splitlines()):
            line = " ".join(raw.split())
            if len(line) < MIN_LINE or line.lower() in seen:
                continue
            seen.add(line.lower())
            out.append(self._atom(
                project_id, artifact_id, path, line, AtomType.scope_item,
                {"kind": "plain_text_line", "line": index + 1,
                 "suffix": path.suffix.lower()},
                locator={"kind": "plain_text", "line": index + 1},
                suffix=f"{index}"))
            if len(out) >= MAX_LINES:
                break
        if not out:
            return []
        marker = (f"[Read as plain text] {path.name} — {describe(path)}, so it "
                  f"was read line by line. {len(out)} lines, {size:,} bytes. "
                  f"A parser that understands this format would do better.")
        out.insert(0, self._atom(
            project_id, artifact_id, path, marker, AtomType.deal_metadata,
            {"kind": "plain_text_marker", "line_count": len(out),
             "size_bytes": size, "suffix": path.suffix.lower()},
            locator={"kind": "plain_text"}, suffix="marker"))
        return out

    def _atom(self, project_id: str, artifact_id: str, path: Path, text: str,
              atom_type: AtomType, value: dict[str, Any], *,
              locator: dict[str, Any], suffix: str) -> EvidenceAtom:
        source_ref = SourceRef(
            id=stable_id("src", artifact_id, "unread", suffix),
            artifact_id=artifact_id,
            artifact_type=ArtifactType.txt,
            filename=path.name,
            locator=locator,
            extraction_method="unread_plain_text",
            parser_version=self.parser_version,
        )
        return EvidenceAtom(
            id=stable_id("atm", project_id, artifact_id, "unread", suffix),
            project_id=project_id,
            artifact_id=artifact_id,
            atom_type=atom_type,
            raw_text=text,
            normalized_text=text.lower(),
            value=value,
            entity_keys=[],
            source_refs=[source_ref],
            authority_class=AuthorityClass.customer_current_authored,
            confidence=0.6,
            review_status=ReviewStatus.needs_review,
            parser_version=self.parser_version,
        )
