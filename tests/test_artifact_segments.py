from __future__ import annotations

from pathlib import Path

from app.core.ids import stable_id
from app.parsers.docx_parser import DocxParser
from app.parsers.email_parser import EmailParser
from app.parsers.quote_parser import QuoteParser
from app.parsers.segmenters import (
    segment_docx,
    segment_email,
    segment_quote,
    segment_text,
    segment_transcript,
    segment_xlsx,
)
from app.parsers.transcript_parser import TranscriptParser
from app.parsers.xlsx_parser import XlsxParser


def test_xlsx_rows_become_spreadsheet_row_segments(demo_project: Path) -> None:
    path = demo_project / "site_list.xlsx"
    artifact_id = stable_id("art", "proj", path.name)
    segments = segment_xlsx(project_id="proj", artifact_id=artifact_id, path=path)
    assert segments
    assert all(segment.segment_type == "spreadsheet_row" for segment in segments)


def test_email_blocks_become_email_message_and_email_line_segments(demo_project: Path) -> None:
    path = demo_project / "customer_email.txt"
    artifact_id = stable_id("art", "proj", path.name)
    segments = segment_email(project_id="proj", artifact_id=artifact_id, path=path)
    assert any(segment.segment_type == "email_message" for segment in segments)
    assert any(segment.segment_type == "email_line" for segment in segments)
    assert any(segment.metadata.get("quoted") for segment in segments if segment.segment_type == "email_message")


def test_transcript_lines_become_transcript_utterance_segments(demo_project: Path) -> None:
    path = demo_project / "kickoff_transcript.txt"
    artifact_id = stable_id("art", "proj", path.name)
    segments = segment_transcript(project_id="proj", artifact_id=artifact_id, path=path)
    utterances = [segment for segment in segments if segment.segment_type == "transcript_utterance"]
    assert utterances
    assert all(segment.source_ref.locator.get("line_start") is not None for segment in utterances)


def test_docx_paragraphs_become_docx_paragraph_segments(demo_project: Path) -> None:
    path = demo_project / "sow_draft.docx"
    artifact_id = stable_id("art", "proj", path.name)
    segments = segment_docx(project_id="proj", artifact_id=artifact_id, path=path)
    paragraph_segments = [segment for segment in segments if segment.segment_type == "docx_paragraph"]
    assert paragraph_segments


def test_docx_tracked_deletion_segment_exists(demo_project: Path) -> None:
    path = demo_project / "sow_draft.docx"
    artifact_id = stable_id("art", "proj", path.name)
    segments = segment_docx(project_id="proj", artifact_id=artifact_id, path=path)
    tracked = [segment for segment in segments if segment.segment_type == "docx_tracked_deletion"]
    assert tracked
    assert any("install av displays in conference rooms" in segment.normalized_text for segment in tracked)


def test_every_segment_has_source_ref(demo_project: Path) -> None:
    files = [
        demo_project / "site_list.xlsx",
        demo_project / "customer_email.txt",
        demo_project / "kickoff_transcript.txt",
        demo_project / "vendor_quote.xlsx",
    ]
    all_segments = []
    for file in files:
        artifact_id = stable_id("art", "proj", file.name)
        if file.name == "site_list.xlsx":
            all_segments.extend(segment_xlsx(project_id="proj", artifact_id=artifact_id, path=file))
        elif file.name == "customer_email.txt":
            all_segments.extend(segment_email(project_id="proj", artifact_id=artifact_id, path=file))
        elif file.name == "kickoff_transcript.txt":
            all_segments.extend(segment_transcript(project_id="proj", artifact_id=artifact_id, path=file))
        else:
            all_segments.extend(segment_quote(project_id="proj", artifact_id=artifact_id, path=file))
    assert all_segments
    assert all(segment.source_ref is not None for segment in all_segments)


def test_segment_ids_are_deterministic(demo_project: Path) -> None:
    path = demo_project / "customer_email.txt"
    artifact_id = stable_id("art", "proj", path.name)
    first = segment_email(project_id="proj", artifact_id=artifact_id, path=path)
    second = segment_email(project_id="proj", artifact_id=artifact_id, path=path)
    assert [segment.id for segment in first] == [segment.id for segment in second]


def test_parser_outputs_stable_after_segmenter_refactor(demo_project: Path, tmp_path: Path) -> None:
    project_id = "proj"
    parsers = [
        (XlsxParser(), demo_project / "site_list.xlsx"),
        (QuoteParser(), demo_project / "vendor_quote.xlsx"),
        (EmailParser(), demo_project / "customer_email.txt"),
        (TranscriptParser(), demo_project / "kickoff_transcript.txt"),
        (DocxParser(), demo_project / "sow_draft.docx"),
    ]
    for parser, path in parsers:
        artifact_id = stable_id("art", project_id, path.name)
        first_atoms = parser.parse_artifact(project_id=project_id, artifact_id=artifact_id, path=path)
        second_atoms = parser.parse_artifact(project_id=project_id, artifact_id=artifact_id, path=path)
        assert [atom.id for atom in first_atoms] == [atom.id for atom in second_atoms]
        assert [atom.raw_text for atom in first_atoms] == [atom.raw_text for atom in second_atoms]

    text_file = tmp_path / "notes.txt"
    text_file.write_text("Block one\n\nBlock two", encoding="utf-8")
    text_segments = segment_text(project_id="proj", artifact_id="art_notes", path=text_file)
    assert any(segment.segment_type == "text_block" for segment in text_segments)
