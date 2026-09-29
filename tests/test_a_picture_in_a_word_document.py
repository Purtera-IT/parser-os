"""A picture in a .docx is still a picture.

The Kronos DX installation guide in deal 010264 carries 31 illustrated
steps -- they ARE the procedure a technician follows at ~1,400 clocks; the
text around them is narration. Every one of them reached the deal as

    [Image awaiting OCR / vision / OLE extraction] media/image16.png ...
    A vision or embedded-object pass is required to recover its content.

typed `open_question`, so 31 of the PM's 154 questions were the parser's
own notes to itself. The vision pass those markers ask for exists and runs
on every compile. It skipped all of them, for three separate reasons:
`region_ref` had to start with "page", there had to be a `saved_path`, and
the filename had to end ".pdf". The .docx parser set none of the three and
never wrote the image out at all.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

from app.core.pdf_image_vision import _iter_image_markers
from app.parsers.docx_parser import DocxParser


def _docx_with_image(tmp_path: Path, png: bytes) -> Path:
    p = tmp_path / "guide.docx"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("word/document.xml", "<w:document/>")
        z.writestr("word/media/image1.png", png)
    return p


def test_a_docx_picture_is_written_out_so_vision_can_see_it(tmp_path,
                                                            monkeypatch):
    monkeypatch.setenv("SOWSMITH_IMAGE_DIR", str(tmp_path / "imgs"))
    png = b"\x89PNG\r\n\x1a\n" + b"x" * 4000
    atoms = DocxParser()._emit_embedded_media_markers(
        project_id="p", artifact_id="a", filename="guide.docx",
        path=_docx_with_image(tmp_path, png))
    markers = [a for a in atoms if (a.value or {}).get("kind") == "image_marker"]
    assert len(markers) == 1
    saved = (markers[0].value or {}).get("saved_path")
    assert saved, "without a saved_path the vision stage skips the marker"
    assert Path(saved).read_bytes() == png

    # ...and the stage now yields it. This is the whole fix: before, the
    # same marker matched none of the three conditions and was silently
    # dropped, leaving an open_question nobody could answer.
    assert len(list(_iter_image_markers(atoms))) == 1


def test_an_unreadable_picture_degrades_to_the_old_marker(tmp_path,
                                                          monkeypatch):
    """A parser that cannot write the file still emits the marker. The
    region is never allowed to vanish silently -- that guarantee predates
    this change and must survive it."""
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("a file, so mkdir beneath it fails")
    monkeypatch.setenv("SOWSMITH_IMAGE_DIR", str(blocker / "imgs"))
    atoms = DocxParser()._emit_embedded_media_markers(
        project_id="p", artifact_id="a", filename="guide.docx",
        path=_docx_with_image(tmp_path, b"\x89PNG" + b"y" * 100))
    markers = [a for a in atoms if (a.value or {}).get("kind") == "image_marker"]
    assert len(markers) == 1, "the marker survives an unwritable image dir"
    assert (markers[0].value or {}).get("saved_path") is None
    assert not list(_iter_image_markers(atoms))
