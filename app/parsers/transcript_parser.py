from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.domain import get_active_domain_pack
from app.core.ids import stable_id
from app.core.normalizers import (
    detect_speaker,
    detect_section,
    extract_meeting_entities,
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
    ReviewStatus,
    SourceRef,
    ParserCapability,
    ParserMatch,
)
from app.parsers.base import BaseParser
from app.parsers.segmenters import segment_transcript
from app.domain.schemas import DomainPack

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
            try:
                if isinstance(json.loads(text), (dict, list)):
                    confidence = 0.8
                    reasons.append("json_transcript_candidate")
            except Exception:
                confidence = 0.0
        elif suffix in {".txt", ".md"}:
            if "open questions:" in lowered or "decisions:" in lowered:
                confidence = 0.9
                reasons.append("meeting_sections_detected")
            elif parse_timestamp(text) is not None or detect_speaker(text) is not None:
                confidence = 0.82
                reasons.append("speaker_or_timestamp_markers")
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
        del domain_pack
        segments = self._segments_from_path(path)
        atoms: list[EvidenceAtom] = []
        for segment in segments:
            atoms.extend(
                self._atoms_from_segment(
                    project_id=project_id,
                    artifact_id=artifact_id,
                    filename=path.name,
                    segment=segment,
                )
            )
        return atoms

    def _segments_from_path(self, path: Path) -> list[dict[str, Any]]:
        suffix = path.suffix.lower()
        raw = path.read_text(encoding="utf-8", errors="ignore")
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
        lines = []
        for line in raw_text.splitlines():
            if line.strip().upper() == "WEBVTT":
                continue
            if "-->" in line:
                continue
            if not line.strip():
                continue
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
        return SourceRef(
            id=stable_id("src", artifact_id, segment["utterance_index"], segment["line_start"], segment["text"]),
            artifact_id=artifact_id,
            artifact_type=ArtifactType.transcript,
            filename=filename,
            locator={
                "line_start": segment["line_start"],
                "line_end": segment["line_end"],
                "speaker": segment.get("speaker"),
                "speaker_role": speaker_role,
                "timestamp_start": segment.get("timestamp_start"),
                "timestamp_end": segment.get("timestamp_end"),
                "section": segment.get("section"),
                "utterance_index": segment["utterance_index"],
            },
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
        deduped_types: list[AtomType] = []
        for atom_type in atom_types:
            if atom_type not in deduped_types:
                deduped_types.append(atom_type)

        for atom_type in deduped_types:
            value: dict[str, Any] = {"text": text}
            review_status = ReviewStatus.auto_accepted
            review_flags: list[str] = []
            confidence = 0.78

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
