"""The drawing is content, and the legend on it assigns money to companies."""
from __future__ import annotations

import pytest

from app.core import linked_picture_ink as ink
from app.core import linked_picture_vision as lpv


class FakeImage:
    """An image whose pixels are whatever the test says they are.

    Boxes are (x0, y0, x1, y1) -> a single RGB colour filling that box.
    """

    def __init__(self, boxes, size=(400, 400)):
        self._boxes = boxes
        self._size = size

    def crop(self, box):
        for area, rgb in self._boxes:
            if (box[0], box[1], box[2], box[3]) == area:
                return _Pixels(rgb, 40)
        # Anything not painted is paper.
        return _Pixels((255, 255, 255), 40)

    def convert(self, _mode):
        return self


class _Pixels:
    def __init__(self, rgb, n):
        self._rgb, self._n = rgb, n

    def getdata(self):
        return [self._rgb] * self._n


def poly(x0, y0, x1, y1):
    return [x0, y0, x1, y0, x1, y1, x0, y1]


ORANGE = (240, 150, 40)
PURPLE = (110, 100, 220)
BLACK = (20, 20, 20)


# ── the fetch refuses to be pointed at our own network ──────────────


@pytest.mark.parametrize("url", [
    "http://example.com/a.png",          # not https
    "https://localhost/a.png",
    "https://127.0.0.1/a.png",
    "https://169.254.169.254/latest/meta-data/",   # cloud metadata
    "https://10.0.0.5/a.png",
    "https://192.168.1.10/a.png",
])
def test_a_document_cannot_point_the_parser_at_our_own_network(url):
    """The URL comes out of a file a stranger wrote and the fetch runs inside
    our network. Every one of these is a link an attacker would like us to
    follow, and none of them is a vendor drawing."""
    with pytest.raises(lpv.UnsafeURL):
        lpv._assert_public(url)


def test_a_bad_link_costs_the_deal_nothing():
    assert lpv.fetch_picture("https://127.0.0.1/a.png") is None
    assert lpv.fetch_picture("not a url at all") is None


# ── colour is measured, never asked for ─────────────────────────────


def test_the_legend_calibrates_itself_from_its_own_ink():
    """Nothing hard-codes "orange". A legend entry is printed in the colour it
    defines, so a sheet keyed in any two colours reads the same way."""
    lines = [
        {"content": "- Huzzard supplied Components", "polygon": poly(0, 0, 10, 10)},
        {"content": "- Installer supplied Components", "polygon": poly(0, 20, 10, 30)},
        {"content": "Barcode Door Access Control Diagram", "polygon": poly(0, 40, 10, 50)},
    ]
    image = FakeImage([((0, 0, 11, 11), ORANGE), ((0, 20, 11, 31), PURPLE)])
    refs = ink.legend_reference_hues(image, lines)
    assert sorted(refs.values()) == ["Huzzard supplied Components",
                                     "Installer supplied Components"]
    # The title is not a legend entry and contributes no reference.
    assert len(refs) == 2


def test_a_label_in_no_legend_colour_gets_no_supplier():
    """The title block is printed in brand blue, which is near the installer's
    violet but is not it. Guessing here would put the sheet's own title on
    somebody's bill of materials."""
    image = FakeImage([((0, 0, 11, 11), (45, 95, 210))])  # the sheet's brand blue, hue ~215
    refs = {30: "Huzzard supplied Components", 240: "Installer supplied Components"}
    assert ink.classify(image, poly(0, 0, 10, 10), refs) is None


def test_black_ink_is_not_a_legend_colour():
    image = FakeImage([((0, 0, 11, 11), BLACK)])
    assert ink.ink_hue(image, poly(0, 0, 10, 10)) is None


def test_two_legend_entries_in_one_colour_are_both_dropped():
    """An ambiguous key is worse than no key: it would assign parts to a
    company on a coin flip."""
    lines = [
        {"content": "Huzzard supplied Components", "polygon": poly(0, 0, 10, 10)},
        {"content": "Installer supplied Components", "polygon": poly(0, 20, 10, 30)},
    ]
    image = FakeImage([((0, 0, 11, 11), ORANGE), ((0, 20, 11, 31), ORANGE)])
    assert ink.legend_reference_hues(image, lines) == {}


# ── a label torn off its other half is worthless ────────────────────


def test_a_label_wrapped_onto_two_rows_is_one_label():
    """"Serial / Com Cable" is printed on two rows. Two atoms saying the sheet
    shows "Serial /" and "Com Cable" tell a PM nothing."""
    lines = [
        {"content": "Serial /", "polygon": poly(0, 0, 40, 12)},
        {"content": "Com Cable", "polygon": poly(0, 13, 40, 25)},
    ]
    groups = lpv.merge_wrapped_labels(lines, [30, 30])
    assert groups == [[0, 1]]


def test_two_callouts_stacked_in_the_same_colour_stay_apart():
    """"Couplers" and "Adapter Cable" are also stacked and also orange. On the
    real sheet they sit 0.55 of a line-height apart and a wrap sits 0.18."""
    lines = [
        {"content": "Couplers", "polygon": poly(0, 0, 40, 11)},
        {"content": "Adapter Cable", "polygon": poly(0, 17, 40, 28)},
    ]
    assert lpv.merge_wrapped_labels(lines, [30, 30]) == [[0], [1]]


def test_labels_in_different_colours_never_merge():
    lines = [
        {"content": "Mag Lock Cable", "polygon": poly(0, 0, 40, 12)},
        {"content": "Local Extender", "polygon": poly(0, 13, 40, 25)},
    ]
    assert lpv.merge_wrapped_labels(lines, [240, 30]) == [[0], [1]]


@pytest.mark.parametrize("junk", ["-", "1", "E", ".", "  "])
def test_ocr_noise_is_not_a_line_item(junk):
    """Arrowheads and leader dashes OCR as single characters. Each one that
    got through would be a part on a bill of materials that does not exist."""
    assert not lpv._is_a_label(junk)


def test_a_leader_dash_is_not_part_of_the_name():
    assert lpv._label_text("- Adapter Cable") == "Adapter Cable"


def test_a_list_heading_keeps_its_items():
    """"Supported Software Includes:" with nothing under it, and two product
    names with nothing above them, are three atoms that each say nothing."""
    assert lpv._note_text("Supported Software Includes: . ABC Ignite • Peak Pro") == (
        "Supported Software Includes: ABC Ignite, Peak Pro")


# ── what the sheet ends up saying ───────────────────────────────────


def test_a_component_says_who_the_legend_assigns_it_to():
    read = {
        "is_drawing": True, "title": "Barcode Door Access Control Diagram",
        "drawing_ref": "BPW061725 Rev1", "vendor": "Huzzard",
        "legend": ["Huzzard supplied Components", "Installer supplied Components"],
        "components": [
            {"label": "Relay", "means": "Installer supplied Components"},
            {"label": "Couplers", "means": "Huzzard supplied Components"},
        ],
        "notes": [], "connections": [],
    }
    said = {x["kind"]: x for x in lpv.statements(read)}
    assert "BPW061725 Rev1" in said["title"]["text"]

    # The colour is the HEADING and the part is the LINE, so the drawing's
    # supply list renders in the same shape as the email's own -- which is
    # what makes the four disagreements between them visible at a glance.
    parts = {x["text"]: x["lead"] for x in lpv.statements(read) if x["kind"] == "component"}
    assert parts == {"Relay": "Installer supplied Components:",
                     "Couplers": "Huzzard supplied Components:"}

    # A legend entry announces how many parts its colour covers.
    legend = [x for x in lpv.statements(read) if x["kind"] == "legend"]
    opens = {r["key"] for x in legend for r in x["reads"]}
    assert opens == {"opens_block"}


def test_a_part_on_a_vendor_drawing_is_not_a_line_on_our_bill():
    """On 010288 the email carries ten supply lines and eight of them are drawn
    on this sheet too. As bom_line these would sit beside the email's under
    different wording, and anything summing a bill of materials counts them
    twice. The atom says what it is -- a statement about a vendor's drawing --
    and a PM can promote it."""
    from app.core.schemas import AtomType

    read = {"is_drawing": True, "components": [
        {"label": "Relay", "means": "Installer supplied Components"}]}
    said = lpv.statements(read)[0]
    kind, atom_type = said["kind"], said["type"]
    assert kind == "component"
    assert atom_type is not AtomType.bom_line
    assert atom_type is AtomType.deal_metadata


def test_a_component_in_no_legend_colour_says_so_rather_than_guessing():
    read = {"is_drawing": True, "components": [{"label": "Door", "means": ""}]}
    said = lpv.statements(read)[0]
    assert "does not say who supplies it" in said["text"]
    # With no colour there is no heading to sit under, and none is invented.
    assert said["lead"] == ""


def test_a_drawing_atom_does_not_claim_to_be_a_link_to_a_picture():
    """`image_url` on an atom is a RENDER CONTRACT downstream: the labeler
    replaces the card's text with "Linked image (below)" and draws the picture
    under it. That is right for "Diagram: https://..." and wrong for a reading
    taken OUT of a picture -- claiming it made all 22 of 010288's drawing atoms
    display as "Linked image (below)", the analysis thrown away and the same
    drawing pasted 23 times down one note. Provenance goes in the locator."""
    src = type("A", (), {"id": "atm_src", "project_id": "p", "artifact_id": "art",
                         "source_refs": []})()
    atom = lpv._emit(source=src, url="https://v.example/d.png", fact_kind="component",
                     text="The drawing shows Relay, in Installer supplied Components.",
                     atom_type=__import__("app.core.schemas", fromlist=["AtomType"]).AtomType.deal_metadata,
                     confidence=0.64)
    assert "image_url" not in atom.value, "the card would discard this atom's text"
    assert "media_type" not in atom.value
    assert atom.value["read_from_image"] == "https://v.example/d.png"
    # Which picture it came from is still recorded, where provenance belongs.
    assert atom.source_refs[0].locator["image_url"] == "https://v.example/d.png"


def test_topology_is_not_emitted_by_default(monkeypatch):
    """The words come from OCR and the colours from the pixels. Which line runs
    to which is neither -- it is the model tracing wires, and a wrong
    connection reads exactly like a right one."""
    read = {"is_drawing": True, "connections": [{"from": "Relay", "to": "PC", "via": "USB"}]}
    monkeypatch.delenv("SOWSMITH_LINKED_PICTURE_TOPOLOGY", raising=False)
    assert lpv.statements(read) == []
    monkeypatch.setenv("SOWSMITH_LINKED_PICTURE_TOPOLOGY", "1")
    assert "connects Relay to PC via USB" in lpv.statements(read)[0]["text"]


def test_a_reading_sits_where_the_drawing_sits():
    """An atom belongs where its sender put it.

    These were emitted with a locator holding only the image URL, so every key
    the labeller sorts by -- page, block, line -- scored zero and all 22 sorted
    ABOVE Alec's opening sentence. A reader met twenty-two facts about a
    picture before the message that sent it.
    """
    from app.core.schemas import AtomType

    class Ref:
        filename = "x.eml"
        locator = {"message_index": 0, "line_start": 45, "line_end": 45,
                   "sender": "alec@cdw.com", "quoted": False}

    class Src:
        id, project_id, artifact_id = "atm_src", "p", "art"
        source_refs = [Ref()]

    made = [
        lpv._emit(source=Src(), url="https://v/d.png", fact_kind=k, text=t,
                  atom_type=AtomType.deal_metadata, confidence=0.6, ordinal=i)
        for i, (k, t) in enumerate([("title", "The drawing is X."),
                                    ("legend", "Legend A."),
                                    ("component", "Shows Relay.")])
    ]
    lines = [a.source_refs[0].locator["line_start"] for a in made]
    # Below the line that pointed at the drawing...
    assert all(x > 45 for x in lines)
    # ...above whatever the sender wrote next...
    assert all(x < 46 for x in lines)
    # ...and in the order the sheet reads.
    assert lines == sorted(lines)
    # The message they arrived in is theirs too, so they thread correctly.
    assert made[0].source_refs[0].locator["message_index"] == 0
    assert made[0].source_refs[0].locator["sender"] == "alec@cdw.com"


def test_a_reading_with_no_position_to_inherit_does_not_invent_one():
    from app.core.schemas import AtomType

    class Src:
        id, project_id, artifact_id = "atm_src", "p", "art"
        source_refs = []

    atom = lpv._emit(source=Src(), url="https://v/d.png", fact_kind="component",
                     text="Shows Relay.", atom_type=AtomType.deal_metadata,
                     confidence=0.6, ordinal=3)
    assert "line_start" not in atom.source_refs[0].locator


def test_the_whole_stage_is_a_no_op_when_the_flag_is_off(monkeypatch):
    monkeypatch.delenv("SOWSMITH_LINKED_PICTURE_VISION", raising=False)
    assert lpv.atoms_from_linked_pictures([object()]) == []


def test_a_part_and_its_colour_do_not_collide_with_the_email_s_own_line():
    """"Relay" is a line in Alec's supply list AND a label on his vendor's
    drawing. A label key is deal + file + page + text, and both now live in
    the same file -- so the sheet supplies the page. Without it the two are
    one card and half the comparison disappears."""
    from app.core.schemas import AtomType

    class Ref:
        filename = "ask.eml"
        locator = {"message_index": 0, "line_start": 45}

    class Src:
        id, project_id, artifact_id = "s", "p", "art"
        source_refs = [Ref()]

    atom = lpv._emit(source=Src(), url="https://v/d.png", fact_kind="component",
                     text="Relay", atom_type=AtomType.deal_metadata, confidence=0.6,
                     ordinal=0, lead="Installer supplied Components:",
                     sheet="BPW061725 Rev1")
    loc = atom.source_refs[0].locator
    assert loc["sheet"] == "BPW061725 Rev1"
    assert loc["lead_in"] == ["Installer supplied Components:"]
    assert loc["section_path"] == ["Installer supplied Components"]
    # `page` stays untouched -- the walk sorts on it, and position came from
    # the line that pointed at the drawing.
    assert "page" not in loc
