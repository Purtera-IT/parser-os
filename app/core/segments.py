from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.core.ids import stable_id
from app.core.normalizers import normalize_text
from app.core.schemas import ArtifactType, SourceRef

SegmentType = Literal[
    "spreadsheet_row",
    "spreadsheet_cell",
    "email_message",
    "email_line",
    "docx_paragraph",
    "docx_table_row",
    "docx_tracked_deletion",
    "transcript_utterance",
    "transcript_section",
    "quote_line_item",
    "text_block",
]


class ArtifactSegment(BaseModel):
    id: str
    project_id: str
    artifact_id: str
    artifact_type: ArtifactType
    segment_type: SegmentType
    text: str
    normalized_text: str
    locator: dict[str, Any] = Field(default_factory=dict)
    source_ref: SourceRef
    speaker: str | None = None
    speaker_role: str | None = None
    section: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def make_segment(
    *,
    project_id: str,
    artifact_id: str,
    artifact_type: ArtifactType,
    segment_type: SegmentType,
    text: str,
    locator: dict[str, Any],
    source_ref: SourceRef,
    speaker: str | None = None,
    speaker_role: str | None = None,
    section: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ArtifactSegment:
    normalized_text = normalize_text(text)
    segment_id = stable_id(
        "seg",
        project_id,
        artifact_id,
        segment_type,
        normalized_text,
        locator,
        source_ref.id,
    )
    return ArtifactSegment(
        id=segment_id,
        project_id=project_id,
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        segment_type=segment_type,
        text=text,
        normalized_text=normalized_text,
        locator=dict(locator),
        source_ref=source_ref,
        speaker=speaker,
        speaker_role=speaker_role,
        section=section,
        metadata=dict(metadata or {}),
    )
