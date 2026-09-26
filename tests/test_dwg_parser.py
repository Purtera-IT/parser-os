"""A drawing must arrive, whether or not anything here can read it.

Deal 010180 ships 7 Penn Plaza's 12th floor as a PDF and twice as the AutoCAD
source, and both `.dwg` files come back `skipped_no_parser`: nothing extracted,
nothing shown, and the previewer saying "DWG files can't be previewed inline".

The CAD route is also strictly better than the picture of it. The same room
label through each:

    WOMEN'S RESTROOM        from the DWG, whole, at (500, 40) on A-ROOM-IDEN
    WOM RESTR: EN'S ROOM    from the PDF, torn in half and filed as a
                            site_access_restriction

because a plan's text is positioned labels, and a table detector reading the
walls as a grid splits each one at its first space.
"""
from __future__ import annotations

import base64
import struct
from pathlib import Path

import pytest

from app.parsers.dwg_parser import (
    DwgParser, is_apparatus, preview_png, release_of, text_entities,
)

ezdxf = pytest.importorskip("ezdxf")


@pytest.fixture
def plan(tmp_path: Path) -> Path:
    """A floor plan the shape of the one on 010180."""
    doc = ezdxf.new("R2018")
    msp = doc.modelspace()
    for layer in ("A-ROOM-IDEN", "A-WALL", "DIMS", "TITLE"):
        doc.layers.add(layer)
    for name, x, y in (("IT CLOSET", 120, 40), ("BOARD ROOM", 300, 180),
                       ("RECEPTION", 40, 200), ("EXECUTIVE OFFICE", 420, 90),
                       ("WOMEN'S RESTROOM", 500, 40), ("HUDDLE ROOM", 250, 60)):
        msp.add_text(name, dxfattribs={"layer": "A-ROOM-IDEN"}).set_placement((x, y))
    msp.add_text("7 PENN PLAZA - 12TH FLOOR",
                 dxfattribs={"layer": "TITLE"}).set_placement((10, 600))
    msp.add_text('SCALE 1/8" = 1\'-0"',
                 dxfattribs={"layer": "DIMS"}).set_placement((10, 580))
    out = tmp_path / "SP-6.dxf"
    doc.saveas(out)
    return out


def _png(width: int = 4, height: int = 3) -> bytes:
    """A real, minimal PNG -- chunk-walkable to IEND."""
    def chunk(kind: bytes, body: bytes) -> bytes:
        import zlib
        return (struct.pack(">I", len(body)) + kind + body
                + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF))

    import zlib
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + b"\xff\x00\x00" * width for _ in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


# --------------------------------------------------------------------------
# The label survives whole
# --------------------------------------------------------------------------

def test_a_room_label_arrives_in_one_piece(plan: Path):
    """The PDF route produced "WOM RESTR: EN'S ROOM" for this same string."""
    found = {e["text"] for e in text_entities(plan)}
    assert "WOMEN'S RESTROOM" in found
    assert "IT CLOSET" in found


def test_a_label_knows_where_it_is(plan: Path):
    """The point of reading CAD rather than a picture of it: a room a cable
    has to reach, not a string."""
    closet = next(e for e in text_entities(plan) if e["text"] == "IT CLOSET")
    assert (closet["x"], closet["y"]) == (120.0, 40.0)
    assert closet["layer"] == "A-ROOM-IDEN"


def test_the_sheet_is_separated_from_the_building():
    assert is_apparatus("DIMS") is True
    assert is_apparatus("TITLE") is True
    assert is_apparatus("A-ROOM-IDEN") is False
    assert is_apparatus("A-WALL") is False


def test_only_the_building_becomes_atoms(plan: Path):
    out = DwgParser().parse(plan)
    labels = [a for a in out.atoms if a.atom_type.value == "site_attribute"]
    assert {a.raw_text for a in labels} == {
        "IT CLOSET", "BOARD ROOM", "RECEPTION", "EXECUTIVE OFFICE",
        "WOMEN'S RESTROOM", "HUDDLE ROOM",
    }
    assert out.atoms[0].atom_type.value == "deal_metadata"
    assert "6 labels recovered" in out.atoms[0].raw_text


# --------------------------------------------------------------------------
# The preview, which costs nothing and needs no CAD library
# --------------------------------------------------------------------------

def test_the_embedded_preview_is_found_by_signature():
    """Not by the header's image table. On 010180's SP-6-1.dwg that table
    claims 63 images at offsets past the end of the file, so trusting its
    count is how this crashes on the second drawing anyone supplies."""
    png = _png()
    data = b"AC1032" + b"\x00" * 400 + png + b"\x00" * 100
    assert preview_png(data) == png


def test_a_drawing_with_no_preview_says_so_rather_than_guessing():
    assert preview_png(b"AC1032" + b"\x00" * 5000) is None


def test_a_truncated_png_is_not_a_preview():
    """A signature is not a picture; it has to walk to its own IEND."""
    assert preview_png(b"AC1032" + b"\x89PNG\r\n\x1a\n" + b"\x00" * 40) is None


def test_the_preview_travels_as_a_derived_file(tmp_path: Path):
    png = _png(8, 6)
    dwg = tmp_path / "SP-6.dwg"
    dwg.write_bytes(b"AC1032" + b"\x00" * 400 + png)
    out = DwgParser().parse(dwg)
    derived = [d for d in out.derived_files if d.relative_path.endswith(".preview.json")]
    assert len(derived) == 1
    body = derived[0].content_json
    assert (body["width"], body["height"]) == (8, 6)
    assert base64.b64decode(body["base64"]) == png


# --------------------------------------------------------------------------
# A drawing nobody can convert is still a drawing
# --------------------------------------------------------------------------

def test_an_unconvertible_drawing_is_visible_rather_than_skipped(tmp_path: Path):
    """The whole complaint: today a .dwg is `skipped_no_parser`, so the PM
    never learns the plan was in the intake."""
    dwg = tmp_path / "SP-6.dwg"
    dwg.write_bytes(b"AC1032" + b"\x00" * 2000)
    out = DwgParser().parse(dwg)
    assert len(out.atoms) == 1
    marker = out.atoms[0]
    assert marker.atom_type.value == "open_question"
    assert "AutoCAD 2018" in marker.raw_text
    assert marker.value["has_preview"] is False


def test_the_release_is_named():
    assert release_of(b"AC1032xxxx")[1] == "AutoCAD 2018"
    assert release_of(b"AC1015xxxx")[1] == "AutoCAD 2000"
    assert release_of(b"NOPE00xxxx")[1] == "unknown release"


def test_the_parser_claims_cad_and_nothing_else():
    parser = DwgParser()
    assert parser.match(Path("a.dwg"), None, None).confidence == 0.95
    assert parser.match(Path("a.dxf"), None, None).confidence == 0.95
    assert parser.match(Path("a.eml"), None, None).confidence == 0.0


def test_it_is_registered():
    from app.parsers.registry import get_registered_parsers

    assert any(p.capability.parser_name == "dwg" for p in get_registered_parsers())
