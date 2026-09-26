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


# --------------------------------------------------------------------------
# A schedule is laid out, not tabulated
# --------------------------------------------------------------------------

def test_a_schedule_row_is_one_fact():
    """7 Penn Plaza's program table puts its labels at one x and its counts at
    another. Entity by entity it is 26 loose strings, and 106 of nothing is not
    a quantity -- 106 workstations at two Cat6A drops each is the 212 the quote
    bills for."""
    from app.parsers.dwg_parser import rows_from_entities

    rows = rows_from_entities([
        {"text": "5'-0\" WORKSTATIONS", "layer": "TEMPLATE TEXT", "x": 0.8,
         "y": 7.2, "kind": "MTEXT", "height": 0.11},
        {"text": "106", "layer": "TEMPLATE TEXT", "x": 4.1, "y": 7.2,
         "kind": "MTEXT", "height": 0.11},
    ])
    assert [r["text"] for r in rows] == ['5\'-0" WORKSTATIONS 106']


def test_a_far_away_panel_is_not_part_of_the_row():
    """"KEY PLAN" sits on the same line as "IT CLOSET 1" and 16 units away, and
    joined naively the row read "IT CLOSET 1 KEY PLAN"."""
    from app.parsers.dwg_parser import rows_from_entities

    rows = rows_from_entities([
        {"text": "IT CLOSET", "layer": "T", "x": 0.8, "y": 5.1, "kind": "MTEXT",
         "height": 0.11},
        {"text": "1", "layer": "T", "x": 4.1, "y": 5.1, "kind": "MTEXT",
         "height": 0.11},
        {"text": "KEY PLAN", "layer": "T", "x": 20.3, "y": 5.1, "kind": "MTEXT",
         "height": 0.11},
    ])
    assert sorted(r["text"] for r in rows) == ["IT CLOSET 1", "KEY PLAN"]


def test_a_title_block_prints_itself_twice():
    """Template and filled-in instance sit a hair apart, so "PROGRAM SUMMARY"
    arrives twice -- and once the row is split on the wide gap, the two copies
    land in different runs, so the de-duplication has to span the whole line."""
    from app.parsers.dwg_parser import rows_from_entities

    rows = rows_from_entities([
        {"text": "PROGRAM SUMMARY", "layer": "T", "x": 20.0, "y": 4.3,
         "kind": "MTEXT", "height": 0.11},
        {"text": "PROGRAM SUMMARY", "layer": "T", "x": 20.05, "y": 4.3,
         "kind": "MTEXT", "height": 0.11},
    ])
    assert [r["text"] for r in rows] == ["PROGRAM SUMMARY"]


def test_a_wrapped_room_tag_is_one_room():
    """The plan prints "WOMEN'S" over "RESTROOM"; read line by line the deal
    learns it has a room called RESTROOM."""
    from app.parsers.dwg_parser import rows_from_entities

    rows = rows_from_entities([
        {"text": "WOMEN'S", "layer": "ROOM-TAG", "x": 100.0, "y": 50.0,
         "kind": "TEXT", "height": 4.0},
        {"text": "RESTROOM", "layer": "ROOM-TAG", "x": 100.5, "y": 44.0,
         "kind": "TEXT", "height": 4.0},
    ])
    assert [r["text"] for r in rows] == ["WOMEN'S RESTROOM"]


def test_two_rooms_near_each_other_stay_two_rooms():
    """COAT and STORAGE are a few feet apart on the same plan. A window loose
    enough to catch every wrapped label swallowed these into one room, so the
    window is tight and some wrapped labels stay split. A split label is
    visible and a labeler can join it; a merged one invents a room."""
    from app.parsers.dwg_parser import rows_from_entities

    rows = rows_from_entities([
        {"text": "COAT", "layer": "ROOM-TAG", "x": 100.0, "y": 90.0,
         "kind": "TEXT", "height": 4.0},
        {"text": "STORAGE", "layer": "ROOM-TAG", "x": 126.0, "y": 34.0,
         "kind": "TEXT", "height": 4.0},
    ])
    assert sorted(r["text"] for r in rows) == ["COAT", "STORAGE"]


# --------------------------------------------------------------------------
# The drawing is a template somebody else's job was drawn on
# --------------------------------------------------------------------------

def test_another_tenants_study_is_not_this_deals_address():
    """BR Design's SP-6 for 7 Penn Plaza still prints "PRELIMINARY SPACE STUDY:
    JOELE FRANK" and "622 THIRD AVE | 36TH FLOOR" in its title block, beside
    its own unfilled placeholders. Read as facts they put the job at the wrong
    address for the wrong client."""
    from app.parsers.dwg_parser import is_template_leftover

    assert is_template_leftover("STREET ADDRESS | XX FLOOR") is True
    assert is_template_leftover("DATE: XX.XX.22") is True
    assert is_template_leftover("PROJECT NO: 29009") is True
    assert is_template_leftover("SCALE: 1/16\" = 1' | DRAWN BY: RA") is True


def test_an_xref_into_another_project_is_not_a_fact():
    r"""G:\69401 - Elise AI\ARCH\... names a client this deal has never heard
    of."""
    from app.parsers.dwg_parser import is_template_leftover

    assert is_template_leftover(r"G:\69401 - Elise AI\ARCH\FROM OTHERS\PDF\x.pdf") is True


def test_the_architects_masthead_is_not_a_fact_about_the_building():
    from app.parsers.dwg_parser import is_template_leftover

    assert is_template_leftover("BR DESIGN ASSOCIATES, LLC 630 NINTH AVENUE") is True
    assert is_template_leftover("NOTHING BEATS 72 YEARS OF STABILITY") is True


def test_a_real_room_is_not_a_leftover():
    from app.parsers.dwg_parser import is_template_leftover

    assert is_template_leftover("IT CLOSET 1") is False
    assert is_template_leftover("7 PENN PLAZA") is False
    assert is_template_leftover("Floor 12 | Suite 1200 | 12,154 RSF") is False
