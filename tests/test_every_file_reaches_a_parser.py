"""No artifact may route to nobody.

When the router returned None the compile recorded `skipped_no_parser`, and the
file then existed in the manifest and nowhere else -- not the envelope, not
PM_HANDOFF, not the Deal Kit. The only way to learn it had arrived was to go
and look at the manifest.

Measured across 461 dev envelopes:

    .dwg    2   AutoCAD drawings           -> dwg_parser
    .json   6   historical                 -> JsonParser claims these at 0.55
    .xls    2   legacy Excel               -> unread marker
    .rpmsg  2   RMS-encrypted Outlook mail -> unread marker
    .doc    1   legacy Word                -> unread marker
    .gif    1   an image                   -> ImageParser, extension was missing
    .tsv    1   tab-separated values       -> unread marker
    .pptx   6   failed_parse, "python-pptx is required" -- the parser shipped
                and its dependency was never declared

Some of these deserve a real parser and some cannot be read at all: an `.rpmsg`
is encrypted by design. The point is that from outside those two look
identical, and both look like the file was never sent.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.parsers.registry import choose_parser
from app.parsers.unread_parser import UnreadParser, describe

#: (filename, first bytes) -> the parser that must take it.
ROUTES = [
    ("plan.dwg", b"AC1032" + b"\x00" * 600, "dwg"),
    ("plan.dxf", b"  0\nSECTION\n" + b"x" * 600, "dwg"),
    ("anim.gif", b"GIF89a" + b"x" * 600, "image"),
    ("photo.png", b"\x89PNG\r\n\x1a\n" + b"x" * 600, "image"),
    ("legacy.doc", b"\xd0\xcf\x11\xe0" + b"x" * 600, "unread"),
    ("book.xls", b"\xd0\xcf\x11\xe0" + b"x" * 600, "unread"),
    ("protected.rpmsg", b"x" * 600, "unread"),
    ("sites.tsv", b"a\tb\nc\td\n", "unread"),
    ("mystery.xyz", b"x" * 600, "unread"),
]


@pytest.mark.parametrize("name,body,expected", ROUTES)
def test_every_artifact_reaches_a_parser(tmp_path: Path, name, body, expected):
    path = tmp_path / name
    path.write_bytes(body)
    parser, match, _ = choose_parser(path)
    assert parser is not None, f"{name} routed to nobody"
    assert parser.capability.parser_name == expected, match.reasons


def test_nothing_routes_to_none(tmp_path: Path):
    """The invariant, stated once: whatever arrives, somebody holds it."""
    for name in ("a.rpmsg", "b.7z", "c", "d.dmg", "e.wpd", "f.pages"):
        path = tmp_path / name
        path.write_bytes(b"x" * 100)
        parser, _, _ = choose_parser(path)
        assert parser is not None, f"{name} routed to nobody"


def test_the_marker_says_what_arrived_and_why_it_is_missing(tmp_path: Path):
    path = tmp_path / "Signed SOW.rpmsg"
    path.write_bytes(b"x" * 4211)
    atom = UnreadParser().parse(path)[0]
    assert "Signed SOW.rpmsg" in atom.raw_text
    assert "encrypted" in atom.raw_text
    assert "4,211 bytes" in atom.raw_text
    assert atom.atom_type.value == "open_question"
    assert atom.value["suffix"] == ".rpmsg"


def test_it_names_a_fix_where_there_is_one():
    """A PM can act on "ask for a .docx". They cannot act on "unsupported"."""
    assert "docx" in describe(Path("x.doc"))
    assert "xlsx" in describe(Path("x.xls"))
    assert "unprotected copy" in describe(Path("x.rpmsg"))


def test_an_unknown_extension_still_gets_an_honest_sentence():
    assert ".wpd" in describe(Path("x.wpd")) or "wpd" in describe(Path("x.wpd"))
    assert "no extension" in describe(Path("noext"))


def test_the_marker_parser_never_competes(tmp_path: Path):
    """It is reached only where the router had already decided on None, so a
    real parser at any confidence beats it."""
    path = tmp_path / "anything.pdf"
    path.write_bytes(b"%PDF-1.4\n" + b"x" * 600)
    assert UnreadParser().match(path, None, None).confidence == 0.0


def test_a_cad_drawing_is_typed_as_a_picture():
    """It fell through to ArtifactType.txt while dwg_parser declared `image` on
    its source refs, and the marker atom never reached the envelope. The image
    parsers' markers do reach it -- 133 atoms across 31 deals -- and this is the
    difference between them."""
    from app.core.manifest import _artifact_type_for_path
    from app.parsers.registry import _artifact_type_for_suffix

    assert _artifact_type_for_path(Path("plan.dwg")).value == "image"
    assert _artifact_type_for_suffix(".dwg").value == "image"
    assert _artifact_type_for_suffix(".dxf").value == "image"


def test_pptx_has_its_dependency_declared():
    """Six decks across the dev corpus come back failed_parse with
    "python-pptx is required for the PPTX parser". The parser shipped; the
    dependency was never declared."""
    import tomllib

    text = Path("pyproject.toml").read_bytes()
    deps = tomllib.loads(text.decode())["project"]["dependencies"]
    assert any(d.startswith("python-pptx") for d in deps), deps
