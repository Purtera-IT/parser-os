"""Three things a drawing says that were being thrown away.

The CAD parse on 010180 recovered 65 labels and lost the three facts that
change what a quote means:

    EXECUTIVE OFFICE 2   the schedule's count, collapsed into the bare room tag
    RECEPTION 01         the same
    OPTION A             the sheet is one of two layouts
    SCALE: 1/16" = 1'    the only thing turning the sheet into distances

None was a converter problem. Each was a filter that is right in general and
wrong on a drawing's program summary.
"""
from __future__ import annotations

import pytest

from app.parsers.dwg_parser import is_template_leftover, names_a_layout_option


# ---------------------------------------------------------------- the scale

def test_a_scale_that_states_a_ratio_is_a_fact():
    """Distance is what the site walk exists to establish on a cabling job, and
    the ratio is the only thing that turns this sheet into distances."""
    assert not is_template_leftover('SCALE: 1/16" = 1\' | DRAWN BY: RA')
    assert not is_template_leftover('SCALE: 1/8" = 1\'-0"')
    assert not is_template_leftover("Scale 1:100")


def test_a_scale_that_states_none_is_still_stationery():
    """The marker was not wrong, only too broad. "NTS" is title-block
    furniture and so is the paper size."""
    assert is_template_leftover("SCALE: NTS | DRAWN BY:")
    assert is_template_leftover('1/8" SCALE: 24 X 36 PAPER SIZE')


def test_the_rest_of_the_stationery_is_untouched():
    for line in ("PROJECT NO: XXXXX", "STREET ADDRESS | XX FLOOR",
                 "PRELIMINARY SPACE STUDY | TENANT NAME", "APPROVAL:",
                 "BR DESIGN ASSOCIATES, LLC  630 NINTH AVENUE"):
        assert is_template_leftover(line), line


# --------------------------------------------------------------- the option

def test_an_option_label_is_recognised():
    """010180's sheet carries these on DEFPOINTS -- AutoCAD's non-plotting
    layer, which the apparatus filter drops as construction marks. If the sheet
    is Option A then its room schedule is a proposal, and 106 workstations is a
    proposal rather than a count: a different thing to price."""
    assert names_a_layout_option("OPTION A")
    assert names_a_layout_option("OFFICE OPTION A")
    assert names_a_layout_option("OFFICE OPTION B")


def test_it_does_not_fire_on_ordinary_words():
    """A rule that catches "optional extras" would drag furniture notes back in
    with it."""
    for line in ("RECEPTION", "DM-DEMO RED HIDDEN", "optional extras",
                 "OPTIONS", "MECH. RM"):
        assert not names_a_layout_option(line), line


# --------------------------------------------------------------- the counts
#
# These go through `cross_type_dedup_atoms` rather than a helper, because the
# fix is not a new rule about counts -- it is giving a drawing the structural
# identity the dedup key already scopes tables by. The layer is the table, a
# baseline is the row.


class _Ref:
    def __init__(self, layer, y):
        # Exactly what `DwgParser._make_atom` writes -- no injected sheet/row.
        self.locator = {"kind": "cad_drawing", "layer": layer, "x": 0.0, "y": y}


class _CadAtom:
    def __init__(self, atom_type, text, layer="TEMPLATE TEXT", y=0.0):
        self.atom_type = atom_type
        self.raw_text = self.text = text
        self.confidence = 0.85
        self.artifact_id = "art:sp6"
        self.source_refs = [_Ref(layer, y)]
        self.receipts = []
        self.entity_keys = []
        self.review_flags = []


def test_two_schedule_rows_are_not_one_row():
    """The cross-type key strips quantities, so "PRIVATE OFFICE 01" and
    "PRIVATE OFFICE 2" reduce to the same text. On live 010180 that collapsed a
    schedule that listed both into a schedule that listed one. Different
    baselines are different rows."""
    from app.core.semantic_dedup import cross_type_dedup_atoms

    out = cross_type_dedup_atoms([
        _CadAtom("quantity", "PRIVATE OFFICE 01", y=140.0),
        _CadAtom("quantity", "PRIVATE OFFICE 2", y=132.5),
    ])
    assert {a.raw_text for a in out} == {"PRIVATE OFFICE 01", "PRIVATE OFFICE 2"}


def test_a_counted_row_survives_a_bare_room_tag():
    """The row that says HOW MANY sits on the program-summary layer; the tag
    that says WHICH is placed on the plan. They are different cells, so the
    count is not discarded in favour of the label."""
    from app.core.semantic_dedup import cross_type_dedup_atoms

    out = cross_type_dedup_atoms([
        _CadAtom("quantity", "EXECUTIVE OFFICE 2", layer="TEMPLATE TEXT", y=120.0),
        _CadAtom("site_room_mix", "EXECUTIVE OFFICE", layer="ROOM-TAG", y=64.25),
    ])
    assert len(out) == 2


def test_two_tags_of_the_same_room_both_survive():
    """Two tags naming the same room in different places are two rooms.

    SP-6 does this with EXECUTIVE OFFICE, HUDDLE, PRIVATE OFFICE and RESTROOM,
    each tagged twice on ROOM-TAG at different baselines. Keyed on text alone
    the second tag vanished, and a plan that shows two of something read as
    showing one -- so a schedule row could never be checked against its tags.

    (An earlier version of this docstring said SP-6 carried two PANTRY tags.
    It carries one. Counted across all four layouts, the second "PANTRY" was
    the schedule ROW "PANTRY 1", not a tag.)"""
    from app.core.semantic_dedup import cross_type_dedup_atoms

    out = cross_type_dedup_atoms([
        _CadAtom("site_room_mix", "PANTRY", layer="ROOM-TAG", y=88.0),
        _CadAtom("site_room_mix", "PANTRY", layer="ROOM-TAG", y=41.75),
    ])
    assert len(out) == 2


def test_one_entity_typed_twice_still_collapses():
    """The narrowing has to stay narrow. Two types emitted off the SAME text
    entity share a layer and a baseline, so they are one cell and still fold."""
    from app.core.semantic_dedup import cross_type_dedup_atoms

    out = cross_type_dedup_atoms([
        _CadAtom("scope_item", "IT CLOSET 1", y=104.0),
        _CadAtom("quantity", "IT CLOSET 1", y=104.0),
    ])
    assert len(out) == 1


# ------------------------------------------------- the render is looked at

def test_the_render_is_sized_for_a_screen():
    """ezdxf's auto-sized page describes the sheet in REAL-WORLD units, and for
    SP-6 that is `width="27.1mm"` -- about 102 px. An `<img>` takes its
    intrinsic size from those attributes and a `max-width` rule only caps a
    picture, never grows one, so the floor plan would have arrived as an
    unreadable hundred-pixel thumbnail with every line in it technically
    present."""
    from app.parsers.dwg_parser import _RENDER_WIDTH_PX, _sized_for_a_screen

    tiny = ('<svg xmlns="http://www.w3.org/2000/svg" width="27.1mm" '
            'height="21.1mm" viewBox="0 0 1000000 778598"><g/></svg>')
    out = _sized_for_a_screen(tiny)
    assert f'width="{_RENDER_WIDTH_PX}"' in out
    assert 'height="1557"' in out          # the viewBox's aspect, preserved
    assert 'viewBox="0 0 1000000 778598"' in out


def test_a_render_without_a_viewbox_is_left_alone():
    """Resizing by an aspect ratio that is not there would invent one."""
    from app.parsers.dwg_parser import _sized_for_a_screen

    plain = '<svg xmlns="http://www.w3.org/2000/svg" width="10mm" height="8mm"/>'
    assert _sized_for_a_screen(plain) == plain
