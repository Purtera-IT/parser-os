from __future__ import annotations

from datetime import datetime, timezone

from app.core.textio import read_text

import json
import re
from pathlib import Path
from typing import Any

from app.domain import get_active_domain_pack
from app.core.address_parse import US_STATES, find_us_addresses_in_text
from app.core.ids import stable_id
from app.core.normalizers import (
    detect_speaker,
    detect_section,
    extract_meeting_entities,
    fold_standalone_speaker_lines,
    normalize_text,
    normalize_transcript_text,
    parse_timestamp,
    split_transcript_segments,
)
from app.core.segments import ArtifactSegment
from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    EvidenceAtom,
    ParserOutput,
    ReviewStatus,
    SourceRef,
    ParserCapability,
    ParserMatch,
)
from app.parsers.base import BaseParser
from app.parsers.segmenters import segment_transcript
from app.parsers.structured_projection import (
    derived_files_for,
    make_bullet_list,
    make_page,
    make_paragraph,
    make_section,
    make_structured_document,
    stamp_section_and_block_ids,
)
from app.domain.schemas import DomainPack

STRUCTURED_SCHEMA_TRANSCRIPT = "orbitbrief.transcript.structured.v1"

#: W3C voice span, e.g. ``<v.loud Cliff Creech>text</v>``. Only the
#: dependency-free fallback needs this -- webvtt-py handles it natively.
_VTT_VOICE_SPAN_RE = re.compile(r"<v(?:\.[^\s>]+)*\s+([^>]+)>(.*?)(?:</v>|$)", re.I | re.S)

DECISION_RE = re.compile(
    r"\b(decision:|decided|agreed|confirmed|approved|we will|the plan is|final decision)\b",
    re.I,
)
ACTION_RE = re.compile(
    r"\b(action item|ai:|todo|owner:|customer to|purtera to|customer will|purtera will|follow up|send|confirm|provide)\b",
    re.I,
)
QUESTION_RE = re.compile(r"\?|open question|tbd|need to confirm|confirm whether|unknown|pending", re.I)
CONSTRAINT_RE = re.compile(
    r"\b(access window|escort required|escort access|badge required|loading dock|parking|after hours|weekdays|weekends|site access|security requirement|staging|approval gate)\b",
    re.I,
)
EXCLUSION_RE = re.compile(
    r"\b(exclude|excluded|removed from scope|out of scope|not in scope|do not include|customer will not proceed with)\b",
    re.I,
)
SCOPE_RE = re.compile(
    r"\b(install|deploy|replace|remove|survey|rack|configure|camera|ap|switch|reader|device|rollout)\b",
    re.I,
)
CUSTOMER_DIRECTIVE_RE = re.compile(
    r"\b(please remove|please add|we approve|do not proceed|hold off|go ahead)\b",
    re.I,
)
QUANTITY_RE = re.compile(
    r"\b(add|remove|reduce to|set to|additionally add|may add)?\s*(\d+)\s*(more\s+)?(ip cameras?|cameras?|aps?|access points?|devices?)\b",
    re.I,
)

SCOPE_IMPACTING_TYPES = {
    AtomType.scope_item,
    AtomType.exclusion,
    AtomType.customer_instruction,
    AtomType.decision,
    AtomType.meeting_commitment,
    AtomType.quantity,
}



_ULID_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_ULID_RE = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$")


def _ulid_iso(ulid: str) -> str | None:
    """The creation time a ULID encodes, as ISO-8601 UTC, when the id is one."""
    u = (ulid or "").strip().upper()
    if not _ULID_RE.match(u):
        return None
    ms = 0
    for ch in u[:10]:
        ms = ms * 32 + _ULID_ALPHABET.index(ch)
    try:
        dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None
    if not (2000 <= dt.year <= 2100):
        return None
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _iso_date(value: Any) -> str | None:
    """A JSON date field (ISO string or epoch ms) as ISO-8601 UTC, or None."""
    if value is None or value == "":
        return None
    try:
        if isinstance(value, (int, float)):
            ms = float(value)
            dt = datetime.fromtimestamp(ms / 1000 if ms > 1e11 else ms, tz=timezone.utc)
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        text = str(value).strip()
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return None

class TranscriptParser(BaseParser):
    parser_name = "transcript"
    parser_version = "transcript_parser_v1"
    capability = ParserCapability(
        parser_name=parser_name,
        parser_version=parser_version,
        supported_extensions=[".txt", ".md", ".vtt", ".srt", ".json"],
        supported_artifact_types=[ArtifactType.transcript, ArtifactType.txt],
        emitted_atom_types=[
            AtomType.decision,
            AtomType.meeting_commitment,
            AtomType.action_item,
            AtomType.open_question,
            AtomType.constraint,
            AtomType.exclusion,
            AtomType.scope_item,
            AtomType.customer_instruction,
            AtomType.quantity,
        ],
        supported_domain_packs=["*"],
        requires_binary=False,
        supports_source_replay=True,
    )

    def match(self, path: Path, sample_text: str | None, domain_pack: DomainPack | None) -> ParserMatch:
        del domain_pack
        suffix = path.suffix.lower()
        text = sample_text or ""
        lowered = normalize_text(text)
        reasons: list[str] = []
        confidence = 0.0
        artifact_type = ArtifactType.transcript if suffix in {".vtt", ".srt", ".json"} else ArtifactType.txt
        if suffix in {".vtt", ".srt"}:
            confidence = 0.95
            reasons.append(f"caption_extension:{suffix}")
        elif suffix == ".json" and text:
            # A transcript is recognised by SHAPE, not by parsing whole.
            #
            # This required json.loads() to succeed on `sample_text` -- which is
            # a truncated head of the file. A Fireflies transcript runs ~77 KB,
            # so the sample is cut mid-array, json.loads raises, and this scored
            # 0.0. JsonParser's deliberate 0.55 deferral then won by default and
            # flattened the call into key/value atoms: 1,315 of them on one deal,
            # typed `scope_item`, including `utterances[25].speaker: Trent
            # Torrence`. A speaker's name became a scope item, and 147,132 atoms
            # corpus-wide -- 35% of all evidence -- were conversational
            # fragments wearing the authority of extracted scope.
            #
            # Both parsers already agree on the signature; only this side
            # insisted on proof it cannot have from a truncated sample.
            head = text[:8000]
            # "segments" alone is NOT a transcript signal. It is an ordinary
            # business word -- network segments, cable segments, customer
            # segments -- and claiming every file containing it was how an
            # intake manifest ended up here. Diarised speech is `utterances`,
            # or per-item speaker AND text.
            shaped = '"utterances"' in head or ('"speaker"' in head and '"text"' in head)
            parsed_whole = False
            try:
                parsed_whole = isinstance(json.loads(text), (dict, list))
            except Exception:
                parsed_whole = False
            # Shape is required either way. This branch used to fire on ANY
            # .json that parsed, so a `case_manifest.json` mentioning
            # "segments" -- network segments, cable segments, an ordinary
            # business word -- was claimed as a transcript at 0.8 and taken
            # from the parser that could actually read it.
            if parsed_whole and shaped:
                confidence = 0.8
                reasons.append("json_transcript_candidate")
            elif shaped:
                # Above JsonParser's 0.55 deferral, below a clean parse, because
                # a shape read off a truncated head is the weaker claim.
                confidence = 0.7
                reasons.append("json_transcript_shape_truncated_sample")
        elif suffix in {".txt", ".md"}:
            if "open questions:" in lowered or "decisions:" in lowered:
                confidence = 0.9
                reasons.append("meeting_sections_detected")
            else:
                # Per-line check: ``detect_speaker`` / ``parse_timestamp``
                # were anchored on a leading-letter for safety, and the
                # bare ``.+$`` end-of-string anchor inside detect_speaker
                # fails when the full document has more than one line.
                # Walk the first ~40 lines and accept on any single hit.
                # A TIMESTAMP is unambiguous transcript evidence. A bare
                # "Name:" line is not -- ``detect_speaker`` matches any
                # "Label: value", which is what business documents are made
                # of. Accepting one such line in the first forty and stopping
                # meant "School District Contact:" on page one of a Request
                # for Proposals claimed the whole document at 0.82.
                #
                # Measured across 19 real .txt files: eight RFPs, SOWs, specs
                # and addenda were taken this way, while the two files that
                # actually ARE meeting notes have a speaker-line density of
                # 0.0% and qualify through ``meeting_sections_detected``
                # above. The highest densities in the corpus -- 37% to 43% --
                # are customer emails. The signal was never measuring
                # transcript-ness.
                #
                # So: timestamps qualify on sight; speaker labels qualify only
                # when they are how the document is BUILT, which is what a
                # transcript is. The threshold sits above every business
                # document measured (RFP 3.2-7.6%, SOW 1.7%, specs 1.3%,
                # addendum 10.2%, Q&A 15.2%) and above email headers, which
                # are short files where a few header lines dominate -- those
                # are claimed by EmailParser on its own evidence anyway.
                speaker_or_ts = False
                scan = [ln for ln in text.splitlines()[:400] if ln.strip()]
                if any(parse_timestamp(ln) is not None for ln in scan):
                    speaker_or_ts = True
                elif len(scan) >= 8:
                    turns = sum(1 for ln in scan if detect_speaker(ln) is not None)
                    if turns / len(scan) >= 0.50:
                        speaker_or_ts = True
                if not speaker_or_ts:
                    # Otter, Rev and Zoom put the speaker on its OWN line, so
                    # the colon density above reads 0% and a real transcript
                    # landed on the prose floor -- read, but with every
                    # utterance unattributed. Ask the folder, which is the
                    # same function the parser uses to canonicalise the file,
                    # so this decision and that rewrite can never disagree.
                    _folded, fold_stats = fold_standalone_speaker_lines(text)
                    if fold_stats["qualifies"]:
                        speaker_or_ts = True
                        reasons.append(
                            "standalone_speaker_lines("
                            f"{fold_stats['folds']} turns, "
                            f"{fold_stats['distinct_speakers']} speakers)"
                        )
                if not speaker_or_ts:
                    # PDF/meeting exports often use ``Name [mm:ss]`` mid-paragraph.
                    if re.search(
                        r"[A-Z][A-Za-z.'\-]*(?:\s+[A-Z][A-Za-z0-9.'\-]*){0,4}\s*"
                        r"\[\d{1,2}:\d{2}(?::\d{2})?\]",
                        text[:4000],
                    ):
                        speaker_or_ts = True
                if speaker_or_ts:
                    confidence = 0.82
                    reasons.append("speaker_or_timestamp_markers")
        # Filename/title cue. This was 0.78 -- above MATCH_THRESHOLD -- and
        # applied to ANY suffix, so the name alone both created a claim and
        # created it for file types this parser does not support.
        #
        # Swept across 2500 real artifacts, renaming each without touching a
        # byte of content:
        #
        #   .json  NONE(0.00)        -> named "meeting_transcript" -> 0.78   2062
        #   .xlsx  XlsxParser(0.58)  -> named "meeting_transcript" -> 0.78     82
        #
        # The second is the bad one: a spreadsheet handed to this parser, which
        # reads it as text -- and .xlsx is not in supported_extensions at all.
        # "Q3_transcript_summary.xlsx" is an ordinary filename, so this needs
        # no adversary to happen.
        #
        # The same sweep answered whether the prior is load-bearing: of 2500
        # real artifacts, ZERO change parser when their name is neutralised.
        # Nothing relies on it. Timestamps, speaker density and the
        # own-line-speaker fold claim real transcripts on their own evidence.
        #
        # So: restricted to the extensions this parser actually supports, and
        # scored below MATCH_THRESHOLD, where a prior belongs. Kept in the
        # reasons so routing stays explainable.
        if suffix in {".txt", ".md", ".vtt", ".srt", ".json"} and (
            "transcript" in path.name.lower().replace("_", " ").replace("-", " ")
        ):
            confidence = max(confidence, 0.45)
            reasons.append("filename_transcript")
        return ParserMatch(
            parser_name=self.parser_name,
            confidence=confidence,
            reasons=reasons,
            artifact_type=artifact_type,
        )

    def parse(self, artifact_path: Path) -> list[Any]:
        artifact_id = stable_id("art", str(artifact_path))
        return self.parse_artifact(project_id="unknown_project", artifact_id=artifact_id, path=artifact_path)

    def segment_artifact(self, project_id: str, artifact_id: str, path: Path) -> list[ArtifactSegment]:
        return segment_transcript(
            project_id=project_id,
            artifact_id=artifact_id,
            path=path,
            parser_version=self.parser_version,
        )

    def parse_artifact(
        self,
        project_id: str,
        artifact_id: str,
        path: Path,
        domain_pack: DomainPack | None = None,
    ) -> list[EvidenceAtom]:
        return self.parse_artifact_full(
            project_id=project_id,
            artifact_id=artifact_id,
            path=path,
            domain_pack=domain_pack,
        ).atoms

    def parse_artifact_full(
        self,
        project_id: str,
        artifact_id: str,
        path: Path,
        domain_pack: DomainPack | None = None,
    ) -> ParserOutput:
        del domain_pack
        segments = self._segments_from_path(path)
        atoms: list[EvidenceAtom] = []
        header = self._call_header_atom(project_id=project_id, artifact_id=artifact_id, path=path)
        if header is not None:
            atoms.append(header)
        for segment in segments:
            atoms.extend(
                self._atoms_from_segment(
                    project_id=project_id,
                    artifact_id=artifact_id,
                    filename=path.name,
                    segment=segment,
                )
            )
        structured_doc = self._build_structured_doc(filename=path.name, segments=segments)
        stamp_section_and_block_ids(structured_doc, artifact_seed=artifact_id)
        return ParserOutput(
            atoms=atoms,
            derived_files=derived_files_for(artifact_path=path, structured_doc=structured_doc),
        )

    def _call_header_atom(self, *, project_id: str, artifact_id: str, path: Path) -> EvidenceAtom | None:
        """One provenance record for the call itself: title, date, participants.

        A JSON transcript states its own date when the exporter wrote one;
        a Fireflies id is a ULID, whose first ten characters are the
        creation time, so a call is dated even when the exporter did not
        say (live 010300: the Carl Painter call sorted after every document
        because nothing dated it). ``document_date`` is what the envelope
        reads as the document's own date.
        """
        if path.suffix.lower() != ".json":
            return None
        try:
            payload = json.loads(read_text(path))
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        title = " ".join(str(payload.get("title") or "").split())
        date_iso = _iso_date(payload.get("date")) or _ulid_iso(str(payload.get("id") or ""))
        participants = [str(x).strip() for x in (payload.get("participants") or []) if str(x).strip()]
        if not (title or date_iso or participants):
            return None
        parts = []
        if title:
            parts.append(f"Call: {title}")
        if date_iso:
            parts.append(f"Date: {date_iso}")
        if participants:
            parts.append("Participants: " + ", ".join(participants))
        text = " | ".join(parts)
        value: dict[str, Any] = {"kind": "transcript_header", "text": text, "title": title or None}
        if date_iso:
            value["document_date"] = date_iso
        if participants:
            value["participants"] = participants
        return EvidenceAtom(
            id=stable_id("atm", project_id, artifact_id, "transcript_header", text),
            project_id=project_id,
            artifact_id=artifact_id,
            atom_type=AtomType.deal_metadata,
            raw_text=text,
            normalized_text=normalize_text(text),
            value=value,
            authority_class=AuthorityClass.meeting_note,
            confidence=0.95,
            review_status=ReviewStatus.auto_accepted,
            entity_keys=[],
            parser_version=self.parser_version,
            source_refs=[
                SourceRef(
                    id=stable_id("src", artifact_id, "transcript_header"),
                    artifact_id=artifact_id,
                    artifact_type=ArtifactType.transcript,
                    filename=path.name,
                    locator={"line_start": 0, "line_end": 0, "kind": "transcript_header"},
                    extraction_method="transcript_metadata",
                    parser_version=self.parser_version,
                )
            ],
        )

    def _build_structured_doc(
        self,
        *,
        filename: str,
        segments: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Render a transcript as a single page with one section per topic
        (Decisions, Action Items, Open Questions, Constraints, Discussion).
        Each utterance becomes a bullet item carrying its speaker and
        timestamp in plain text so an LLM can quote it directly.
        """
        bucket_order = [
            "Decisions",
            "Action Items",
            "Open Questions",
            "Constraints",
            "Discussion",
        ]
        buckets: dict[str, list[dict[str, Any]]] = {label: [] for label in bucket_order}

        for segment in segments:
            text = str(segment.get("text", "")).strip()
            if not text:
                continue
            speaker = segment.get("speaker") or "Unknown"
            timestamp = segment.get("timestamp_start")
            stamp = f" [{timestamp}]" if timestamp else ""
            bullet_text = f"**{speaker}**{stamp}: {text}"
            section = (segment.get("section") or "").strip().lower()
            target = "Discussion"
            if "decision" in section:
                target = "Decisions"
            elif "action" in section:
                target = "Action Items"
            elif "question" in section:
                target = "Open Questions"
            elif "constraint" in section:
                target = "Constraints"
            else:
                lowered = text.lower()
                if DECISION_RE.search(text):
                    target = "Decisions"
                elif ACTION_RE.search(text):
                    target = "Action Items"
                elif QUESTION_RE.search(text):
                    target = "Open Questions"
                elif CONSTRAINT_RE.search(text):
                    target = "Constraints"
                del lowered
            buckets[target].append({"text": bullet_text, "children": []})

        sections: list[dict[str, Any]] = []
        for label in bucket_order:
            items = buckets[label]
            if not items:
                continue
            sections.append(
                make_section(
                    heading=label,
                    level=2,
                    blocks=[make_bullet_list(items=items)],
                )
            )
        if not sections:
            sections.append(
                make_section(
                    heading="Transcript",
                    level=2,
                    blocks=[make_paragraph("(empty transcript)")],
                )
            )
        page = make_page(page=0, title=filename, sections=sections)
        return make_structured_document(
            schema_version=STRUCTURED_SCHEMA_TRANSCRIPT,
            filename=filename,
            artifact_type=ArtifactType.transcript.value,
            title=filename,
            metadata=[f"utterance_count: {len(segments)}"],
            pages=[page],
        )

    def _segments_from_path(self, path: Path) -> list[dict[str, Any]]:
        suffix = path.suffix.lower()
        raw = read_text(path)
        if suffix == ".json":
            return self._segments_from_json(raw)
        if suffix == ".vtt":
            return self._segments_from_text(self._clean_vtt(raw))
        if suffix == ".srt":
            return self._segments_from_text(self._clean_srt(raw))
        return self._segments_from_text(raw)

    def _segments_from_json(self, raw_text: str) -> list[dict[str, Any]]:
        payload = json.loads(raw_text)
        items: list[dict[str, Any]]
        if isinstance(payload, dict):
            items = payload.get("utterances") or payload.get("segments") or []
        elif isinstance(payload, list):
            items = payload
        else:
            return []

        segments: list[dict[str, Any]] = []
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            speaker = item.get("speaker")
            timestamp_start = item.get("start") or item.get("timestamp")
            segments.append(
                {
                    "utterance_index": idx,
                    "line_start": idx + 1,
                    "line_end": idx + 1,
                    "speaker": speaker,
                    "timestamp_start": str(timestamp_start) if timestamp_start is not None else None,
                    "timestamp_end": str(item.get("end")) if item.get("end") is not None else None,
                    "section": item.get("section"),
                    "text": text,
                }
            )
        return segments

    def _segments_from_text(self, raw_text: str) -> list[dict[str, Any]]:
        text = normalize_transcript_text(raw_text)
        return split_transcript_segments(text)

    def _clean_vtt(self, raw_text: str) -> str:
        """Reduce a WebVTT file to ``Speaker: text`` lines.

        The previous version dropped the ``WEBVTT`` banner, the timing lines
        and blank lines, and kept everything else verbatim. Three things went
        wrong with that, all visible on a file Teams would produce:

        * ``<v Cliff Creech>`` is the W3C voice span -- the standard way every
          major platform names a speaker -- and it survived into the atom as
          literal markup, with the speaker recorded as ``None``.
        * a cue *identifier* is an optional line before the timing line, and
          it is usually just ``1``, ``2``, ``3``. Those became atoms whose
          entire text was a digit. (``_clean_srt`` directly below already
          skipped these; VTT never got the same treatment.)
        * ``NOTE`` comments, ``STYLE`` and ``REGION`` blocks and the other cue
          payload tags (``<c>``, ``<i>``, inline timestamps) were all kept as
          if they were speech.

        ``webvtt-py`` implements the spec, so the parsing is delegated rather
        than re-derived: it separates identifier, timing and payload, strips
        cue tags, and exposes the voice span as ``caption.voice``. The old
        line filter remains as the fallback for a file the library refuses, so
        a malformed transcript degrades instead of raising.
        """
        try:
            import webvtt
        except Exception:  # pragma: no cover - dependency-free fallback
            return self._clean_vtt_fallback(raw_text)
        try:
            captions = list(webvtt.from_string(raw_text))
        except Exception:
            # Malformed, or not actually VTT -- keep the old behaviour.
            return self._clean_vtt_fallback(raw_text)
        if not captions:
            return self._clean_vtt_fallback(raw_text)

        lines: list[str] = []
        for caption in captions:
            text = " ".join((caption.text or "").split())
            if not text:
                continue
            speaker = " ".join((caption.voice or "").split())
            lines.append(f"{speaker}: {text}" if speaker else text)
        return "\n".join(lines)

    def _clean_vtt_fallback(self, raw_text: str) -> str:
        """The pre-library line filter, plus the cue-identifier skip it lacked."""
        lines = []
        for line in raw_text.splitlines():
            stripped = line.strip()
            if stripped.upper() == "WEBVTT":
                continue
            if stripped.upper().startswith(("NOTE", "STYLE", "REGION")):
                continue
            if "-->" in line:
                continue
            if not stripped:
                continue
            if stripped.isdigit():  # a cue identifier, not speech
                continue
            match = _VTT_VOICE_SPAN_RE.search(line)
            if match:
                speaker = " ".join(match.group(1).split())
                said = " ".join(match.group(2).split())
                line = f"{speaker}: {said}" if said else speaker
            lines.append(line)
        return "\n".join(lines)

    def _clean_srt(self, raw_text: str) -> str:
        lines = []
        for line in raw_text.splitlines():
            stripped = line.strip()
            if stripped.isdigit():
                continue
            if "-->" in stripped:
                continue
            if not stripped:
                continue
            lines.append(line)
        return "\n".join(lines)

    def _speaker_role(self, speaker: str | None, text: str) -> str:
        source = normalize_text(f"{speaker or ''} {text}")
        if any(token in source for token in ("customer", "client")):
            return "customer"
        if any(token in source for token in ("purtera", "pm", "project manager", "coordinator")):
            return "internal"
        if speaker and "@" in speaker:
            email = speaker.split("<")[-1].strip("> ").lower()
            domain = email.split("@")[-1] if "@" in email else ""
            if domain and "purtera" not in domain:
                return "customer"
            if "purtera" in domain:
                return "internal"
        return "unknown"

    def _base_source_ref(self, artifact_id: str, filename: str, segment: dict[str, Any], speaker_role: str) -> SourceRef:
        section = segment.get("section")
        locator: dict[str, Any] = {
            "line_start": segment["line_start"],
            "line_end": segment["line_end"],
            "speaker": segment.get("speaker"),
            "speaker_role": speaker_role,
            "timestamp_start": segment.get("timestamp_start"),
            "timestamp_end": segment.get("timestamp_end"),
            "section": section,
            "utterance_index": segment["utterance_index"],
        }
        if section:
            locator["section_path"] = [str(section)]
        return SourceRef(
            id=stable_id("src", artifact_id, segment["utterance_index"], segment["line_start"], segment["text"]),
            artifact_id=artifact_id,
            artifact_type=ArtifactType.transcript,
            filename=filename,
            locator=locator,
            extraction_method="transcript_rule_engine",
            parser_version=self.parser_version,
        )

    def _atoms_from_segment(
        self,
        project_id: str,
        artifact_id: str,
        filename: str,
        segment: dict[str, Any],
    ) -> list[EvidenceAtom]:
        text = str(segment.get("text", "")).strip()
        if not text:
            return []
        lowered = normalize_text(text)
        pack = get_active_domain_pack()
        speaker_role = self._speaker_role(segment.get("speaker"), text)
        source_ref = self._base_source_ref(artifact_id, filename, segment, speaker_role)
        entity_keys = extract_meeting_entities(text)

        # A question is an ask, never a commitment. Live 010300: "Is the
        # children dentistry done after hours?" was emitted twice, once as an
        # open_question and once as a CONSTRAINT, so the brief carried a
        # question mark as a rule the crew had to work to. Asking whether
        # something is so does not make it so, whatever words the sentence
        # shares with a real constraint.
        _is_question = text.rstrip().endswith("?")

        atom_types: list[AtomType] = []
        if DECISION_RE.search(text):
            atom_types.append(AtomType.decision)
            if "we will" in lowered:
                atom_types.append(AtomType.meeting_commitment)
        if ACTION_RE.search(text) or any(
            re.search(rf"\b{re.escape(normalize_text(alias))}\b", lowered)
            for aliases in pack.action_aliases.values()
            for alias in aliases
        ):
            atom_types.append(AtomType.action_item)
        if QUESTION_RE.search(text):
            atom_types.append(AtomType.open_question)
        if CONSTRAINT_RE.search(text) or any(
            re.search(rf"\b{re.escape(normalize_text(pattern))}\b", lowered)
            for patterns in pack.constraint_patterns.values()
            for pattern in patterns
        ):
            if not _is_question:
                atom_types.append(AtomType.constraint)
        if EXCLUSION_RE.search(text) or any(
            re.search(rf"\b{re.escape(normalize_text(pattern))}\b", lowered)
            for pattern in pack.exclusion_patterns
        ):
            atom_types.append(AtomType.exclusion)
        if SCOPE_RE.search(text):
            atom_types.append(AtomType.scope_item)
        if speaker_role == "customer" and (
            CUSTOMER_DIRECTIVE_RE.search(text)
            or any(
                re.search(rf"\b{re.escape(normalize_text(pattern))}\b", lowered)
                for pattern in pack.customer_instruction_patterns
            )
        ):
            atom_types.append(AtomType.customer_instruction)
        if QUANTITY_RE.search(text):
            atom_types.append(AtomType.quantity)

        # section-driven typing for note bullets
        section = (segment.get("section") or "").lower()
        if section == "decisions" and AtomType.decision not in atom_types:
            atom_types.append(AtomType.decision)
        if section == "action Items".lower() and AtomType.action_item not in atom_types:
            atom_types.append(AtomType.action_item)
        if section == "open Questions".lower() and AtomType.open_question not in atom_types:
            atom_types.append(AtomType.open_question)

        atoms: list[EvidenceAtom] = []
        try:
            from app.core.vendor_site_ban import is_purtera_vendor_address

            banned_vendor_addr = is_purtera_vendor_address(text=text)
        except Exception:
            banned_vendor_addr = False
        if not banned_vendor_addr:
            for parsed in find_us_addresses_in_text(text):
                if (
                    not parsed.city
                    or not parsed.state
                    or parsed.state not in US_STATES
                    or not parsed.street_address
                ):
                    continue
                slug = re.sub(
                    r"[^a-z0-9]+",
                    "_",
                    f"{parsed.city}_{parsed.state}_{parsed.zip or parsed.street_address}".lower(),
                ).strip("_")
                display = f"{parsed.street_address}, {parsed.city}, {parsed.state} {parsed.zip or ''}".strip()
                site_keys = list(dict.fromkeys([*entity_keys, f"site:{slug}"]))
                aliases = list(dict.fromkeys(parsed.aliases))
                names = list(dict.fromkeys([display, parsed.city, *aliases]))
                atoms.append(
                    EvidenceAtom(
                        id=stable_id("atm", project_id, artifact_id, "transcript_note_physical_site", slug),
                        project_id=project_id,
                        artifact_id=artifact_id,
                        atom_type=AtomType.physical_site,
                        raw_text=display,
                        normalized_text=normalize_text(display),
                        value={
                            "kind": "physical_site",
                            "id": slug,
                            "site_id": slug,
                            "name": display,
                            "names": names,
                            "aliases": aliases,
                            "street_address": parsed.street_address,
                            "address": parsed.street_address,
                            "city": parsed.city,
                            "state": parsed.state,
                            "zip": parsed.zip,
                            "inferred": True,
                            "source_context": text[:600],
                        },
                        entity_keys=site_keys,
                        source_refs=[source_ref],
                        authority_class=AuthorityClass.meeting_note,
                        confidence=0.72,
                        review_status=ReviewStatus.needs_review,
                        review_flags=["transcript_note_physical_site"],
                        parser_version=self.parser_version,
                    )
                )
        deduped_types: list[AtomType] = []
        for atom_type in atom_types:
            if atom_type not in deduped_types:
                deduped_types.append(atom_type)

        # Coverage floor. Every branch above is a *pattern* -- a keyword, an
        # alias, a question mark. When none fires the loop below runs zero
        # times and the utterance is gone: no atom, no receipt, nothing that
        # records it was ever said. Measured on a ten-turn call written in
        # ordinary language, five turns vanished, and they were the wrong
        # five: the scope ("forty sites before end of Q3"), the access
        # constraint ("dock is only open until two"), the exclusion ("not
        # paying for the mid-turn jumpers") and the part number all went,
        # while "Understood", "Noted" and "Good" survived on their keywords.
        #
        # So the utterance is kept untyped instead. Typing it is a judgement
        # that belongs downstream where it can be learned and corrected;
        # deciding it was never spoken is not a judgement this layer is
        # entitled to make.
        if not deduped_types:
            deduped_types.append(AtomType.raw_utterance)

        for atom_type in deduped_types:
            value: dict[str, Any] = {"text": text}
            review_status = ReviewStatus.auto_accepted
            review_flags: list[str] = []
            confidence = 0.78

            if atom_type == AtomType.raw_utterance:
                # Deliberately the lowest confidence any transcript atom
                # carries, so it never outranks a typed one covering the same
                # words and never reads as an assertion about the deal.
                confidence = 0.40
                review_flags.append("unclassified_utterance")

            if atom_type == AtomType.quantity:
                match = QUANTITY_RE.search(text)
                if match:
                    op = (match.group(1) or "").strip().lower() or None
                    quantity = int(match.group(2))
                    item = (match.group(4) or "").strip()
                    value.update(
                        {
                            "quantity": quantity,
                            "unit": "count",
                            "item": item,
                            "operation": op,
                        }
                    )
            if atom_type == AtomType.action_item:
                owner = "customer" if "customer to" in lowered else ("purtera" if "purtera to" in lowered else speaker_role)
                value.update({"owner": owner, "action": text})
                if any(token in lowered for token in ("scope", "add", "remove", "price", "cost", "commercial")):
                    review_status = ReviewStatus.needs_review
            if atom_type == AtomType.constraint:
                value.update({"constraint_type": "access", "raw_constraint": text})
            if atom_type == AtomType.open_question:
                review_status = ReviewStatus.needs_review
                review_flags.append("missing_information_candidate")
                confidence = 0.74
            if atom_type == AtomType.exclusion:
                review_status = ReviewStatus.needs_review
                review_flags.extend(["verbal_commitment_requires_confirmation", "exclusion_present"])
            if atom_type in {AtomType.scope_item, AtomType.decision, AtomType.meeting_commitment, AtomType.quantity}:
                review_status = ReviewStatus.needs_review
                if "verbal_commitment_requires_confirmation" not in review_flags:
                    review_flags.append("verbal_commitment_requires_confirmation")
            if atom_type == AtomType.customer_instruction:
                review_status = ReviewStatus.needs_review
                review_flags.extend(["customer_spoken_instruction", "verbal_commitment_requires_confirmation"])

            atom = EvidenceAtom(
                id=stable_id(
                    "atm",
                    project_id,
                    artifact_id,
                    segment["utterance_index"],
                    atom_type.value,
                    text,
                ),
                project_id=project_id,
                artifact_id=artifact_id,
                atom_type=atom_type,
                raw_text=text,
                normalized_text=text.strip(),
                value=value,
                entity_keys=entity_keys,
                source_refs=[source_ref],
                authority_class=AuthorityClass.meeting_note,
                confidence=confidence,
                review_status=review_status,
                review_flags=sorted(set(review_flags)),
                parser_version=self.parser_version,
            )
            atoms.append(atom)
        return atoms
