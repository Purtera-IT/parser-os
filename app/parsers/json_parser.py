"""Structured-JSON parser — intake requests, manifests, API exports, configs.

JSON is the single most common *uploaded* artifact shape in real deals (customer
portal intake payloads like ``INTAKE_REQUEST.json``, ``case_manifest.json``,
URL/record lists, API exports). Before this parser those files either produced
zero atoms (the transcript parser claimed ``.json`` but only understood
``{utterances:[...]}``) or fell back to ``txt`` and lost all structure.

Design goals (robust-first):
  * NEVER crash a compile. Every node is walked under try/except; a malformed or
    pathological file degrades to a single fallback atom carrying the raw text,
    never an exception.
  * NEVER steal meeting-transcript JSON. ``match`` defers (confidence 0.0) when
    the document is transcript-shaped so ``TranscriptParser`` still wins.
  * Lossless-ish flattening. Every scalar leaf becomes one atom whose text is
    ``dotted.key.path: value`` with a JSON-Pointer locator, so source-replay can
    re-find the exact value and packetization sees real key/value content.
  * Bounded. Hard caps on atoms, depth, nodes, and string length keep a 50 MB
    or deeply-recursive document from blowing up memory or the atom store.

Handles: nested objects, flat objects, arrays of scalars, arrays of objects,
mixed/heterogeneous arrays, top-level scalars, JSON Lines (``.jsonl``), UTF-8
BOM, and malformed JSON (raw fallback). Everything is deterministic.
"""
from __future__ import annotations

import json
import os
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
    ParserOutput,
    ReviewStatus,
    SourceRef,
)
from app.domain.schemas import DomainPack
from app.parsers.base import BaseParser


def _int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return default


# Bounds — overridable via env for pathological inputs.
def _max_atoms() -> int:
    return _int_env("SOWSMITH_JSON_MAX_ATOMS", 5000)


def _max_depth() -> int:
    return _int_env("SOWSMITH_JSON_MAX_DEPTH", 40)


def _max_nodes() -> int:
    return _int_env("SOWSMITH_JSON_MAX_NODES", 200000)


_MAX_STR = 2000  # per-value text cap (chars)
_MAX_RAW_FALLBACK = 4000  # raw-text fallback cap for malformed files

# Keys that signal a meeting-transcript JSON shape -> defer to TranscriptParser.
_TRANSCRIPT_KEYS = ('"utterances"', '"segments"', '"transcript"')


class JsonParser(BaseParser):
    """Flatten structured JSON / JSONL into key/value evidence atoms."""

    parser_name = "json"
    parser_version = "json_parser_v1"
    capability = ParserCapability(
        parser_name=parser_name,
        parser_version=parser_version,
        supported_extensions=[".json", ".jsonl"],
        supported_artifact_types=[ArtifactType.json],
        emitted_atom_types=[
            AtomType.scope_item,
            AtomType.constraint,
            AtomType.exclusion,
            AtomType.assumption,
            AtomType.open_question,
            AtomType.project_metadata,
        ],
        supported_domain_packs=["*"],
        requires_binary=False,
        supports_source_replay=True,
    )

    # ── routing ──────────────────────────────────────────────────────

    def match(
        self,
        path: Path,
        sample_text: str | None,
        domain_pack: DomainPack | None,
    ) -> ParserMatch:
        del domain_pack
        suffix = path.suffix.lower()
        if suffix not in {".json", ".jsonl"}:
            return ParserMatch(
                parser_name=self.parser_name, confidence=0.0, reasons=[],
                artifact_type=ArtifactType.json,
            )

        sample = (sample_text or "").lstrip("\ufeff \t\r\n")

        if suffix == ".jsonl":
            first = next((ln for ln in sample.splitlines() if ln.strip()), "")
            conf = 0.9 if first.strip().startswith(("{", "[")) else 0.6
            return ParserMatch(
                parser_name=self.parser_name, confidence=conf,
                reasons=["jsonl_extension"], artifact_type=ArtifactType.json,
            )

        # .json — structural sniff on the sample (may be truncated for big files).
        looks_json = sample.startswith(("{", "["))
        if looks_json:
            head = sample[:4000]
            # Defer transcript-shaped payloads to the transcript parser (0.8).
            if any(k in head for k in _TRANSCRIPT_KEYS) or (
                '"speaker"' in head and '"text"' in head
            ):
                # A DEFERRAL, NOT A WITHDRAWAL. This returned 0.0 -- handing the
                # file to TranscriptParser and assuming it would be caught. It
                # is not always caught, and then nobody reads the file:
                #
                #   * ``head`` is the first 4000 chars of a truncated sample,
                #     so a large .json does not parse as JSON here
                #   * TranscriptParser's .json branch requires json.loads() to
                #     SUCCEED, and on a truncated sample it raises
                #   * both parsers score 0.00 and the artifact routes to NONE
                #
                # Measured on the local corpus: 2062 of 2500 artifacts routed
                # to NONE, essentially all of them valid .json that parse fine
                # in full. They contain the key ``"segments"`` -- which here
                # means document segments, not diarised speech, and is an
                # ordinary business word besides (network segments, customer
                # segments, cable segments).
                #
                # 0.55 sits BELOW TranscriptParser's 0.8, so a real transcript
                # payload still goes there, and ABOVE MATCH_THRESHOLD, so when
                # the transcript parser cannot take it the file is still read
                # rather than dropped. A parser may step aside; it may not
                # leave the artifact with nobody holding it.
                return ParserMatch(
                    parser_name=self.parser_name, confidence=0.55,
                    reasons=["defer_to_transcript"], artifact_type=ArtifactType.json,
                )
            return ParserMatch(
                parser_name=self.parser_name, confidence=0.9,
                reasons=["structured_json"], artifact_type=ArtifactType.json,
            )
        # .json extension but content doesn't look like JSON — still claim it
        # (low) so we can emit a raw-fallback atom instead of losing the file.
        return ParserMatch(
            parser_name=self.parser_name, confidence=0.55,
            reasons=["json_extension_unverified"], artifact_type=ArtifactType.json,
        )

    # ── entry points ─────────────────────────────────────────────────

    def parse(self, artifact_path: Path) -> list[Any]:
        artifact_id = stable_id("art", str(artifact_path))
        return self.parse_artifact_full(
            project_id="unknown_project", artifact_id=artifact_id, path=artifact_path,
        ).atoms

    def parse_artifact(
        self, project_id: str, artifact_id: str, path: Path,
        domain_pack: DomainPack | None = None,
    ) -> list[EvidenceAtom]:
        return self.parse_artifact_full(
            project_id=project_id, artifact_id=artifact_id, path=path,
            domain_pack=domain_pack,
        ).atoms

    def parse_artifact_full(
        self, *, project_id: str, artifact_id: str, path: Path,
        domain_pack: DomainPack | None = None,
    ) -> ParserOutput:
        del domain_pack
        filename = path.name
        warnings: list[str] = []

        try:
            # newline="": keep CRLF as written, so offsets on the locator count
            # the same characters the viewer renders (universal-newline mode
            # dropped one per line and every span drifted).
            with open(path, encoding="utf-8-sig", errors="replace", newline="") as fh:
                raw = fh.read()
        except Exception as exc:  # unreadable file — never crash the compile
            return ParserOutput(
                atoms=[], warnings=[f"json_parser: unreadable file: {exc!r}"],
            )

        data, parse_warn, is_jsonl = self._load(raw, path.suffix.lower())
        if parse_warn:
            warnings.append(parse_warn)

        if data is _UNPARSEABLE:
            atom = self._raw_fallback_atom(project_id, artifact_id, filename, raw)
            return ParserOutput(atoms=[atom] if atom else [], warnings=warnings)

        ctx = _WalkCtx(
            project_id=project_id, artifact_id=artifact_id, filename=filename,
            parser_version=self.parser_version,
            max_atoms=_max_atoms(), max_depth=_max_depth(), max_nodes=_max_nodes(),
        )
        # Where each value sits in the file the viewer shows. The atom's text
        # is a flattened key path ("equipment.items[1].quantity: 1") that never
        # appears in the file, so the source pane cannot find it by text, and
        # with no position on the locator the labeling walk fell back to
        # envelope (hash-tie) order. Spans are measured against ``raw`` -- the
        # file as read, BOM stripped, which is what a browser decodes.
        ctx.spans = _value_spans(raw, jsonl=is_jsonl)
        ctx.line_starts = _line_starts(raw)
        # JSONL: each line is an independent record -> index them at the root.
        if is_jsonl and isinstance(data, list):
            for i, rec in enumerate(data):
                self._walk(rec, [f"line{i + 1}"], ctx)
        else:
            self._walk(data, [], ctx)

        if ctx.truncated:
            warnings.append(
                f"json_parser: output truncated at {ctx.max_atoms} atoms "
                f"(set SOWSMITH_JSON_MAX_ATOMS higher to capture more)"
            )

        if not ctx.atoms:
            # Valid JSON but no scalar leaves (e.g. {} / [] / all-null) — emit a
            # single marker so the file is never silently dropped.
            marker = self._empty_marker_atom(project_id, artifact_id, filename, data)
            if marker:
                ctx.atoms.append(marker)

        return ParserOutput(atoms=ctx.atoms, warnings=warnings)

    # ── loading ──────────────────────────────────────────────────────

    def _load(self, raw: str, suffix: str) -> tuple[Any, str | None, bool]:
        """Return (data, warning, is_jsonl). data is ``_UNPARSEABLE`` on failure."""
        stripped = raw.strip()
        if not stripped:
            return _UNPARSEABLE, "json_parser: empty file", False

        # Straight JSON first (covers .json and single-object .jsonl).
        if suffix != ".jsonl":
            try:
                return json.loads(stripped), None, False
            except Exception:
                pass

        # JSON Lines: parse each non-blank line; tolerate occasional bad lines.
        records: list[Any] = []
        bad = 0
        lines = [ln for ln in stripped.splitlines() if ln.strip()]
        for ln in lines:
            try:
                records.append(json.loads(ln))
            except Exception:
                bad += 1
        if records and bad <= len(lines) // 2:
            warn = (
                f"json_parser: {bad} malformed JSONL line(s) skipped"
                if bad else None
            )
            return records, warn, True

        # Last resort for .jsonl that is actually one multi-line JSON doc.
        if suffix == ".jsonl":
            try:
                return json.loads(stripped), None, False
            except Exception:
                pass

        return _UNPARSEABLE, "json_parser: content is not valid JSON or JSONL", False

    # ── walking ──────────────────────────────────────────────────────

    def _walk(self, node: Any, path: list[str], ctx: "_WalkCtx") -> None:
        if ctx.truncated:
            return
        ctx.nodes += 1
        if ctx.nodes > ctx.max_nodes or len(path) > ctx.max_depth:
            return
        try:
            if isinstance(node, dict):
                for key, val in node.items():
                    if ctx.truncated:
                        return
                    self._walk(val, path + [str(key)], ctx)
            elif isinstance(node, (list, tuple)):
                for i, val in enumerate(node):
                    if ctx.truncated:
                        return
                    self._walk(val, path + [f"[{i}]"], ctx)
            else:
                self._emit_leaf(node, path, ctx)
        except Exception:
            # A single bad node must never abort the whole document.
            return

    def _emit_leaf(self, value: Any, path: list[str], ctx: "_WalkCtx") -> None:
        if value is None:
            return
        if isinstance(value, str):
            sval = value.strip()
            if not sval:
                return
            vtype = "string"
        elif isinstance(value, bool):
            sval = "true" if value else "false"
            vtype = "boolean"
        elif isinstance(value, (int, float)):
            sval = repr(value) if isinstance(value, float) else str(value)
            vtype = "number"
        else:
            sval = str(value)
            vtype = "other"

        if len(sval) > _MAX_STR:
            sval = sval[:_MAX_STR] + "…"

        label = _dotted(path) or "(root)"
        text = f"{label}: {sval}" if path else sval
        if not text.strip():
            return

        if len(ctx.atoms) >= ctx.max_atoms:
            ctx.truncated = True
            return

        pointer = _pointer(path)
        identity = {
            "kind": "json_value",
            "key_path": label,
            "json_pointer": pointer,
        }
        locator = dict(identity)
        span = (ctx.spans or {}).get(pointer)
        if span is not None:
            member_start, value_start, value_end = span
            locator.update({
                "line_start": _line_of(ctx.line_starts, member_start),
                "line_end": _line_of(ctx.line_starts, max(value_start, value_end - 1)),
                # Offsets into the file text (code points, BOM stripped):
                # char_start..char_end is the value itself (a string's quotes
                # included); member_char_start is where its key begins.
                "char_start": value_start,
                "char_end": value_end,
                "member_char_start": member_start,
            })
        atom = _make_json_atom(
            project_id=ctx.project_id, artifact_id=ctx.artifact_id,
            filename=ctx.filename, parser_version=ctx.parser_version,
            text=text,
            locator=locator,
            id_locator=identity,
            value_extra={
                "key_path": label,
                "json_pointer": _pointer(path),
                "value": value if vtype != "other" else sval,
                "value_type": vtype,
            },
        )
        if atom is not None:
            ctx.atoms.append(atom)

    # ── fallbacks ────────────────────────────────────────────────────

    def _raw_fallback_atom(
        self, project_id: str, artifact_id: str, filename: str, raw: str,
    ) -> EvidenceAtom | None:
        text = raw.strip()[:_MAX_RAW_FALLBACK]
        if not text:
            return None
        return _make_json_atom(
            project_id=project_id, artifact_id=artifact_id, filename=filename,
            parser_version=self.parser_version,
            text=f"(unparsed JSON) {text}",
            locator={"kind": "json_raw_fallback", "key_path": "(raw)"},
            value_extra={"value_type": "raw", "parse_ok": False},
            confidence=0.4,
        )

    def _empty_marker_atom(
        self, project_id: str, artifact_id: str, filename: str, data: Any,
    ) -> EvidenceAtom | None:
        shape = type(data).__name__
        return _make_json_atom(
            project_id=project_id, artifact_id=artifact_id, filename=filename,
            parser_version=self.parser_version,
            text=f"JSON document contained no scalar values (shape: {shape})",
            locator={"kind": "json_empty", "key_path": "(empty)"},
            value_extra={"value_type": "empty", "shape": shape},
            confidence=0.4,
        )


# ── module helpers ───────────────────────────────────────────────────

_UNPARSEABLE = object()


class _WalkCtx:
    __slots__ = (
        "project_id", "artifact_id", "filename", "parser_version",
        "max_atoms", "max_depth", "max_nodes", "atoms", "nodes", "truncated",
        "spans", "line_starts",
    )

    def __init__(
        self, *, project_id: str, artifact_id: str, filename: str,
        parser_version: str, max_atoms: int, max_depth: int, max_nodes: int,
    ) -> None:
        self.project_id = project_id
        self.artifact_id = artifact_id
        self.filename = filename
        self.parser_version = parser_version
        self.max_atoms = max_atoms
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.atoms: list[EvidenceAtom] = []
        self.nodes = 0
        self.truncated = False
        self.spans: dict[str, tuple[int, int, int]] = {}
        self.line_starts: list[int] = [0]


def _dotted(path: list[str]) -> str:
    """Human-readable key path: dict keys joined by '.', list indices as '[i]'."""
    out = ""
    for part in path:
        if part.startswith("[") and part.endswith("]"):
            out += part
        else:
            out = f"{out}.{part}" if out else part
    return out


def _pointer(path: list[str]) -> str:
    """RFC-6901 JSON Pointer for exact source-replay re-location."""
    out = []
    for part in path:
        if part.startswith("[") and part.endswith("]"):
            out.append(part[1:-1])
        else:
            out.append(part.replace("~", "~0").replace("/", "~1"))
    return "/" + "/".join(out) if out else ""


# ── source positions ─────────────────────────────────────────────────

_WS = " \t\r\n"
_NUM_RE = None  # compiled lazily
_STR_RE = None


def _line_starts(text: str) -> list[int]:
    starts = [0]
    find = text.find
    i = find("\n")
    while i != -1:
        starts.append(i + 1)
        i = find("\n", i + 1)
    return starts


def _line_of(starts: list[int], offset: int) -> int:
    """1-based line number holding ``offset``."""
    import bisect
    return bisect.bisect_right(starts, max(0, offset))


class _SpanError(Exception):
    pass


def _value_spans(text: str, *, jsonl: bool) -> dict[str, tuple[int, int, int]]:
    """``json_pointer -> (member_start, value_start, value_end)`` for every value.

    A position-tracking scan of the same text ``json.loads`` read, keyed by the
    pointer ``_walk`` builds, so a leaf finds its own occurrence even when the
    same ``"quantity": 1`` appears in every item. JSONL records are keyed
    ``/line<n>`` exactly as the walk names them. Never raises: a text the scan
    cannot follow yields whatever it mapped before giving up (possibly nothing),
    and the atom simply carries no position.
    """
    import re
    global _NUM_RE, _STR_RE
    if _NUM_RE is None:
        _NUM_RE = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][-+]?\d+)?")
        _STR_RE = re.compile(r'"[^"\\]*(?:\\.[^"\\]*)*"', re.S)
    out: dict[str, tuple[int, int, int]] = {}
    n = len(text)

    def skip(i: int) -> int:
        while i < n and text[i] in _WS:
            i += 1
        return i

    def string_end(i: int) -> int:  # text[i] == '"'; returns index past closing quote
        m = _STR_RE.match(text, i)
        if not m:
            raise _SpanError("unterminated string")
        return m.end()

    def value(i: int, ptr: str, member_start: int, depth: int) -> int:
        if depth > 500:
            raise _SpanError("too deep")
        i = skip(i)
        if i >= n:
            raise _SpanError("eof")
        c = text[i]
        start = i
        if c == "{":
            i = skip(i + 1)
            if i < n and text[i] == "}":
                end = i + 1
            else:
                while True:
                    i = skip(i)
                    if i >= n or text[i] != '"':
                        raise _SpanError("key expected")
                    kstart = i
                    kend = string_end(i)
                    key = json.loads(text[kstart:kend])
                    i = skip(kend)
                    if i >= n or text[i] != ":":
                        raise _SpanError("colon expected")
                    seg = str(key).replace("~", "~0").replace("/", "~1")
                    i = value(i + 1, f"{ptr}/{seg}", kstart, depth + 1)
                    i = skip(i)
                    if i < n and text[i] == ",":
                        i += 1
                        continue
                    if i < n and text[i] == "}":
                        end = i + 1
                        break
                    raise _SpanError("bad object")
        elif c == "[":
            i = skip(i + 1)
            if i < n and text[i] == "]":
                end = i + 1
            else:
                k = 0
                while True:
                    i = skip(i)
                    i = value(i, f"{ptr}/{k}", i, depth + 1)
                    k += 1
                    i = skip(i)
                    if i < n and text[i] == ",":
                        i += 1
                        continue
                    if i < n and text[i] == "]":
                        end = i + 1
                        break
                    raise _SpanError("bad array")
        elif c == '"':
            end = string_end(i)
        elif text.startswith("true", i):
            end = i + 4
        elif text.startswith("false", i):
            end = i + 5
        elif text.startswith("null", i):
            end = i + 4
        else:
            m = _NUM_RE.match(text, i)
            if not m or m.end() == i:
                # NaN / Infinity / -Infinity: json.loads accepts them.
                for lit in ("NaN", "Infinity", "-Infinity"):
                    if text.startswith(lit, i):
                        end = i + len(lit)
                        break
                else:
                    raise _SpanError("bad value")
            else:
                end = m.end()
        # Later duplicates win, as they do in json.loads.
        out[ptr] = (member_start, start, end)
        return end

    try:
        if jsonl:
            k = 0
            pos = 0
            for line in text.splitlines(keepends=True):
                body = line.strip()
                if body:
                    try:
                        json.loads(body)
                    except Exception:
                        pos += len(line)
                        continue
                    k += 1
                    value(pos, f"/line{k}", skip(pos), 0)
                pos += len(line)
        else:
            value(0, "", skip(0), 0)
    except (_SpanError, RecursionError, ValueError):
        pass
    except Exception:
        pass
    return out


# Same lightweight classifier the universal parsers use, so JSON atoms land in
# the same packet buckets (e.g. "lift_required: yes" -> constraint).
def _classify_json(text: str) -> AtomType:
    from app.parsers.universal_parsers import _classify
    return _classify(text)


def _make_json_atom(
    *,
    project_id: str,
    artifact_id: str,
    filename: str,
    parser_version: str,
    text: str,
    locator: dict[str, Any],
    value_extra: dict[str, Any],
    confidence: float = 0.85,
    id_locator: dict[str, Any] | None = None,
) -> EvidenceAtom | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        atom_type = _classify_json(text)
    except Exception:
        atom_type = AtomType.scope_item
    # Ids hash the identity of the value (its pointer), not its position, so
    # adding line/char offsets did not re-key atoms already labeled or held.
    ident = str(id_locator if id_locator is not None else locator)
    src = SourceRef(
        id=stable_id("src", artifact_id, ident, text[:80]),
        artifact_id=artifact_id,
        artifact_type=ArtifactType.json,
        filename=filename,
        locator=locator,
        extraction_method="json_flatten",
        parser_version=parser_version,
    )
    return EvidenceAtom(
        id=stable_id("atm", project_id, artifact_id, text[:120], ident),
        project_id=project_id,
        artifact_id=artifact_id,
        atom_type=atom_type,
        raw_text=text,
        normalized_text=text.lower(),
        value=value_extra,
        entity_keys=[],
        source_refs=[src],
        authority_class=AuthorityClass.customer_current_authored,
        confidence=confidence,
        review_status=ReviewStatus.auto_accepted,
        parser_version=parser_version,
    )
