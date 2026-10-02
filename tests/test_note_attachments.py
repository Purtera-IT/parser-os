"""HubSpot notes that only point at an attached file are not documents.

Deal 010246: every HubSpot note on the deal carried a file, and each became its
own document whose whole content was the export header and the word "Note",
while the file it carried arrived as a separate, unrelated document. The note
says nothing; who attached the file and when belongs on the file.

A note with real text keeps its document and is linked both ways to the files
it carried.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.core.compiler import _iter_artifacts, compile_project
from app.core.orbitbrief_envelope import build_orbitbrief_envelope

NOTE_ID = "114999000001"
FILE_ID = "218000000001"
NOTE_AT = "2026-09-10T14:02:11.415Z"
FILE_AT = "2026-09-10T14:02:11.022Z"  # HubSpot writes both in one request
ATTACHMENT = "010246 Site Survey.csv"
CSV = "Site,Address,APs\nStore 12,100 Main St Springfield IL,4\nStore 14,200 Oak Ave Peoria IL,6\n"


def _note_text(title: str, body: str) -> str:
    return (
        f"HubSpot Note: {title}\nHubSpot Note ID: {NOTE_ID}\nDate: {NOTE_AT}\n"
        f"Author: Tanner Norris\nAuthor-Email: tanner@purtera-it.com\n\n{body}"
    )


def _project(tmp_path: Path, *, title: str, body: str, explicit_ids: bool) -> tuple[Path, str]:
    project = tmp_path / "deal"
    project.mkdir()
    note_name = f"010246-hs-note-{NOTE_ID}-{title}.txt"
    (project / note_name).write_text(_note_text(title, body), encoding="utf-8")
    (project / ATTACHMENT).write_text(CSV, encoding="utf-8")
    note_md = {
        "title": title,
        "author": "Tanner Norris",
        "authorEmail": "tanner@purtera-it.com",
        "mirroredFrom": "hubspot",
        "hubspotNoteId": NOTE_ID,
        "hubspotNoteUpdatedAt": NOTE_AT,
    }
    file_row = {
        "filename": ATTACHMENT,
        "source": "hubspot",
        "metadata": {"mirroredFrom": "hubspot", "hubspotFileUpdatedAt": FILE_AT},
    }
    if explicit_ids:
        # What the files carry once Purpulse records the link: HubSpot's own
        # hs_attachment_ids on the note, and the file's id on the file.
        note_md["attachmentIds"] = [FILE_ID]
        file_row["hubspot_file_id"] = FILE_ID
        # A timestamp that would NOT pair them -- the ids decide, not the clock.
        file_row["metadata"]["hubspotFileUpdatedAt"] = "2026-09-12T08:00:00Z"
    manifest = {
        "artifacts": [
            {"filename": note_name, "source": "other", "external_id": f"hs-note:{NOTE_ID}", "metadata": note_md},
            file_row,
        ]
    }
    (project / ".parser_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return project, note_name


def _compile(project: Path):
    result = compile_project(project_dir=project, project_id="note_attach", use_cache=False)
    envelope = build_orbitbrief_envelope(project_dir=project, compile_result=result)
    return result, {d["filename"]: d for d in envelope["documents"]}


def _note_meta_atoms(result):
    return [
        a for a in result.atoms
        if isinstance(a.value, dict) and a.value.get("kind") == "hubspot_note_meta"
    ]


def _assert_folded(project: Path, note_name: str) -> None:
    assert note_name not in {p.name for p in _iter_artifacts(project)}
    result, docs = _compile(project)
    assert note_name not in docs, "a note that only says 'Note' must not be a document"
    assert not _note_meta_atoms(result), "no header atom for a pointer note"
    assert ATTACHMENT in docs
    carried_by = docs[ATTACHMENT]["hubspot_note"]
    assert carried_by["hubspot_note_id"] == NOTE_ID
    assert carried_by["author"] == "Tanner Norris"
    assert carried_by["author_email"] == "tanner@purtera-it.com"
    assert carried_by["created_at"] == NOTE_AT
    assert carried_by["note_folded"] is True
    assert carried_by["note_filename"] is None


def test_placeholder_note_with_attachment_is_folded_into_the_file(tmp_path: Path) -> None:
    project, note_name = _project(tmp_path, title="Note", body="", explicit_ids=False)
    _assert_folded(project, note_name)
    _r, docs = _compile(project)
    assert docs[ATTACHMENT]["hubspot_note"]["link"] == "arrived_with"


def test_placeholder_note_linked_by_hubspot_attachment_ids(tmp_path: Path) -> None:
    project, note_name = _project(tmp_path, title="Note", body="", explicit_ids=True)
    _assert_folded(project, note_name)
    _r, docs = _compile(project)
    assert docs[ATTACHMENT]["hubspot_note"]["link"] == "hubspot_attachment_ids"


def test_note_with_real_text_keeps_its_document_and_links_its_attachment(tmp_path: Path) -> None:
    body = "Customer sent the site survey. Please confirm AP counts before quoting."
    project, note_name = _project(tmp_path, title="Site survey", body=body, explicit_ids=False)
    result, docs = _compile(project)
    assert note_name in docs
    carried = docs[note_name]["note_attachments"]
    assert carried["hubspot_note_id"] == NOTE_ID
    assert [a["filename"] for a in carried["attachments"]] == [ATTACHMENT]
    carried_by = docs[ATTACHMENT]["hubspot_note"]
    assert carried_by["hubspot_note_id"] == NOTE_ID
    assert carried_by["note_filename"] == note_name
    assert carried_by["note_folded"] is False
    meta = _note_meta_atoms(result)
    assert meta and meta[0].value["attachments"] == [ATTACHMENT]


def test_note_with_real_text_records_explicit_attachment_ids(tmp_path: Path) -> None:
    body = "Customer sent the site survey. Please confirm AP counts before quoting."
    project, note_name = _project(tmp_path, title="Site survey", body=body, explicit_ids=True)
    result, docs = _compile(project)
    carried = docs[note_name]["note_attachments"]
    assert carried["attachment_ids"] == [FILE_ID]
    assert carried["attachments"][0]["hubspot_file_id"] == FILE_ID
    assert docs[ATTACHMENT]["hubspot_note"]["link"] == "hubspot_attachment_ids"
    assert _note_meta_atoms(result)[0].value["attachment_ids"] == [FILE_ID]


def test_placeholder_note_with_no_attachment_keeps_its_document(tmp_path: Path) -> None:
    # No file arrived with it: nothing to carry its provenance, so it stays.
    project, note_name = _project(tmp_path, title="Note", body="", explicit_ids=False)
    manifest = json.loads((project / ".parser_manifest.json").read_text())
    manifest["artifacts"][1]["metadata"]["hubspotFileUpdatedAt"] = "2026-09-20T09:00:00Z"
    (project / ".parser_manifest.json").write_text(json.dumps(manifest))
    assert note_name in {p.name for p in _iter_artifacts(project)}
    _r, docs = _compile(project)
    assert note_name in docs
    assert "hubspot_note" not in docs[ATTACHMENT]


def test_burst_of_placeholder_notes_all_fold(tmp_path: Path) -> None:
    # 010127: several "Note" notes and their files posted within seconds. Time
    # cannot say which note holds which file, but none of the notes is content.
    project, first = _project(tmp_path, title="Note", body="", explicit_ids=False)
    manifest = json.loads((project / ".parser_manifest.json").read_text())
    second = f"010246-hs-note-114999000002-Note.txt"
    (project / second).write_text(
        _note_text("Note", "").replace(NOTE_ID, "114999000002").replace("14:02:11.415", "14:02:12.900"),
        encoding="utf-8",
    )
    manifest["artifacts"].append({
        "filename": second, "external_id": "hs-note:114999000002",
        "metadata": {"title": "Note", "hubspotNoteId": "114999000002", "hubspotNoteUpdatedAt": "2026-09-10T14:02:12.900Z"},
    })
    (project / ".parser_manifest.json").write_text(json.dumps(manifest))
    names = {p.name for p in _iter_artifacts(project)}
    assert first not in names and second not in names
