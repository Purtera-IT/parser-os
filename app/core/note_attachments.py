"""Link HubSpot notes to the files attached to them.

In HubSpot a file reaches a deal by being attached to an engagement -- most
often a note whose only content is the default title "Note". Purpulse ingests
the two separately: the note as ``{deal}-hs-note-{noteId}-Note.txt`` (an
export header and the word "Note"), the file under its own HubSpot name. On
deal 010246 every note was one of these, so the compile carried a "Note"
document per attachment, each saying nothing, and the attachments themselves
carried no trace of who attached them or when.

This module decides, from the manifest sidecar Purpulse writes next to the
files (``.parser_manifest.json``), which note carried which file:

* **Explicit** -- the note's ``metadata.attachmentIds`` (HubSpot's own
  ``hs_attachment_ids``) matched against each file's ``hubspot_file_id``
  (top-level, or ``metadata.hubspotFileId``). Authoritative when present; a
  note that carries the key is never linked any other way.
* **arrived_with** -- legacy rows carry neither id. HubSpot writes the file and
  the note that holds it in the same request, so the note's timestamp
  (``metadata.hubspotNoteUpdatedAt``, which is its ``hs_timestamp``) sits
  within a second or two of the file's ``metadata.hubspotFileUpdatedAt``.
  Measured on the 17 OxBlue deals (platform-infra oxblue-migration export):
  34 of 35 "Note" notes had a file within 3 s; of the 21 notes with real
  text, the 17 portal-intake notes sat 0.1-0.2 s from the intake JSON they
  carry, and the other four had no file within 4 minutes. Each file goes to the NEAREST note in
  the window; the link records how it was made and the gap in seconds, so a
  reader can weigh it.

From the links:

* A note whose only content is a placeholder title ("Note", or nothing) and
  which carried at least one file is **folded**: it is not a document of its
  own. :func:`app.core.compiler._iter_artifacts` skips it, so it mints no
  header atom and no document, and its author / date / id travel on the
  attached file instead (``hubspot_note`` on that envelope document).
* A note with real text keeps its document; it lists its files under
  ``note_attachments`` and each file names it under ``hubspot_note``.

Nothing here reads file contents except the notes themselves, and nothing
raises: an unreadable sidecar means no links, which is today's behaviour.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

PARSER_MANIFEST_SIDECAR = ".parser_manifest.json"

#: How far apart a note and a file may be and still be the same HubSpot
#: request. Live gaps were 0.0-2.9 s; a human writing a second note takes far
#: longer than this.
ARRIVED_WITH_WINDOW_SECONDS = 10.0

_HS_NOTE_FILENAME_RE = re.compile(r"-hs-note-(\d+)?", re.I)
_HS_EMAIL_RE = re.compile(r"-hs-email-", re.I)


@dataclass
class NoteAttachmentLinks:
    #: Manifest filenames of notes that are only a pointer at their files.
    folded: set[str] = field(default_factory=set)
    #: Note filename -> what it carried.
    notes: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Attachment filename -> the note that carried it.
    attachments: dict[str, dict[str, Any]] = field(default_factory=dict)


def _when(value: Any) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else None


def _norm(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def _md(art: dict[str, Any]) -> dict[str, Any]:
    md = art.get("metadata")
    return md if isinstance(md, dict) else {}


def _load_artifacts(project_dir: Path) -> list[dict[str, Any]]:
    path = project_dir / PARSER_MANIFEST_SIDECAR
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    arts = data.get("artifacts") if isinstance(data, dict) else None
    return [a for a in (arts or []) if isinstance(a, dict) and str(a.get("filename") or "").strip()]


def _is_note(art: dict[str, Any]) -> bool:
    ext = str(art.get("external_id") or "").strip().lower()
    return ext.startswith("hs-note:") or bool(_HS_NOTE_FILENAME_RE.search(str(art.get("filename") or "")))


def _is_email(art: dict[str, Any]) -> bool:
    ext = str(art.get("external_id") or "").strip().lower()
    name = str(art.get("filename") or "")
    return ext.startswith("hs-email:") or bool(_HS_EMAIL_RE.search(name)) or name.lower().endswith(".eml")


def _note_id(art: dict[str, Any], parsed: dict[str, Any]) -> str:
    ext = str(art.get("external_id") or "").strip()
    if ext.lower().startswith("hs-note:") and ext.split(":", 1)[1].strip():
        return ext.split(":", 1)[1].strip()
    md_id = str(_md(art).get("hubspotNoteId") or "").strip()
    if md_id:
        return md_id
    if parsed.get("note_id"):
        return str(parsed["note_id"])
    m = _HS_NOTE_FILENAME_RE.search(str(art.get("filename") or ""))
    return (m.group(1) or "") if m else ""


def _file_id(art: dict[str, Any]) -> str:
    return str(art.get("hubspot_file_id") or _md(art).get("hubspotFileId") or "").strip()


def is_placeholder_only_note(parsed: dict[str, Any]) -> bool:
    """True when a parsed note says nothing: its title is the CRM default (or
    empty) and its body is that title again (or empty)."""
    from app.parsers.hubspot_note_parser import _is_placeholder_note_title

    title = _norm(parsed.get("title") or "")
    body = _norm(parsed.get("body") or "")
    if title and not _is_placeholder_note_title(title):
        return False
    return not body or body == title or _is_placeholder_note_title(body)


def _read_note(project_dir: Path, filename: str) -> dict[str, Any] | None:
    from app.core.textio import read_text
    from app.parsers.hubspot_note_parser import parse_hubspot_note_text

    path = project_dir / filename.lstrip("/\\")
    if not path.is_file():
        return None
    try:
        return parse_hubspot_note_text(read_text(path))
    except Exception:  # pragma: no cover - an unreadable note is just unlinked
        return None


def note_attachment_links(project_dir: Path | str | None) -> NoteAttachmentLinks:
    """Which note carried which file, and which notes are only a pointer."""
    out = NoteAttachmentLinks()
    if not project_dir:
        return out
    try:
        return _build(Path(project_dir), out)
    except Exception:  # pragma: no cover - never fail a compile over links
        return NoteAttachmentLinks()


def _build(project_dir: Path, out: NoteAttachmentLinks) -> NoteAttachmentLinks:
    arts = _load_artifacts(project_dir)
    if not arts:
        return out

    notes: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    for art in arts:
        name = str(art["filename"]).strip()
        if _is_note(art):
            parsed = _read_note(project_dir, name)
            if parsed is None:
                continue
            md = _md(art)
            explicit = md.get("attachmentIds")
            notes.append({
                "filename": name,
                "parsed": parsed,
                "note_id": _note_id(art, parsed),
                "at": _when(md.get("hubspotNoteUpdatedAt")) or _when(parsed.get("date_raw"))
                or _when(art.get("authored_at")),
                "explicit_ids": (
                    [str(x).strip() for x in explicit if str(x).strip()]
                    if isinstance(explicit, list) else None
                ),
                "author": str(parsed.get("author") or md.get("author") or "").strip(),
                "author_email": str(parsed.get("author_email") or md.get("authorEmail") or "").strip(),
                "date": str(parsed.get("date_raw") or md.get("hubspotNoteUpdatedAt") or "").strip(),
                "title": str(parsed.get("title") or md.get("title") or "").strip(),
                "placeholder": is_placeholder_only_note(parsed),
            })
        elif not _is_email(art):
            files.append({
                "filename": name,
                "file_id": _file_id(art),
                "at": _when(_md(art).get("hubspotFileUpdatedAt")),
            })
    if not notes or not files:
        # A pointer note whose file is not in this run (an as-of cut, a failed
        # mirror) still says nothing -- but with no file here there is nowhere
        # to carry its provenance, so it keeps its document.
        return out

    # note filename -> [(file, how, gap_seconds)]
    carried: dict[str, list[tuple[dict[str, Any], str, float | None]]] = {n["filename"]: [] for n in notes}

    by_id = {f["file_id"]: f for f in files if f["file_id"]}
    explicitly_linked: set[str] = set()
    for n in notes:
        for fid in n["explicit_ids"] or []:
            f = by_id.get(fid)
            if f is not None:
                carried[n["filename"]].append((f, "hubspot_attachment_ids", None))
                explicitly_linked.add(f["filename"])

    legacy = [n for n in notes if n["explicit_ids"] is None and n["at"] is not None]
    # Notes with a file inside the window, whether or not that file ends up
    # attributed to them. Several "Note" notes posted in one burst (010127:
    # six notes and six files inside 3 s) cannot be told apart by time, but
    # every one of them is a pointer at a file; none is content.
    arrived_with: set[str] = set()
    for f in files:
        if f["filename"] in explicitly_linked or f["at"] is None or not legacy:
            continue
        gaps = sorted(
            (abs((f["at"] - n["at"]).total_seconds()), n["note_id"], n["filename"]) for n in legacy
        )
        arrived_with.update(name for gap, _nid, name in gaps if gap <= ARRIVED_WITH_WINDOW_SECONDS)
        gap, _nid, note_name = gaps[0]
        if gap <= ARRIVED_WITH_WINDOW_SECONDS:
            carried[note_name].append((f, "arrived_with", round(gap, 3)))

    for n in notes:
        links = carried[n["filename"]]
        near_file = bool(links) or n["filename"] in arrived_with
        if not near_file and not n["explicit_ids"]:
            continue
        folded = n["placeholder"] and near_file
        if folded:
            out.folded.add(n["filename"])
        else:
            out.notes[n["filename"]] = {
                "hubspot_note_id": n["note_id"],
                "attachment_ids": list(n["explicit_ids"] or [f["file_id"] for f, _h, _g in links if f["file_id"]]),
                "attachments": [
                    {"filename": f["filename"], "hubspot_file_id": f["file_id"] or None, "link": how,
                     **({"gap_seconds": gap} if gap is not None else {})}
                    for f, how, gap in links
                ],
            }
        for f, how, gap in links:
            out.attachments.setdefault(f["filename"], {
                "hubspot_note_id": n["note_id"],
                "author": n["author"] or None,
                "author_email": n["author_email"] or None,
                "created_at": n["date"] or None,
                "title": n["title"] or None,
                # The note's own document, when it has one; a folded note lives
                # only here.
                "note_filename": None if folded else n["filename"],
                "note_folded": folded,
                "link": how,
                **({"gap_seconds": gap} if gap is not None else {}),
            })
    return out


__all__ = [
    "ARRIVED_WITH_WINDOW_SECONDS",
    "NoteAttachmentLinks",
    "is_placeholder_only_note",
    "note_attachment_links",
]
