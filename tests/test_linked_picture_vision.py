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
    said = {kind: text for kind, text, _t in lpv.statements(read)}
    assert "BPW061725 Rev1" in said["title"]
    texts = [t for _k, t, _ in lpv.statements(read)]
    assert "The drawing shows Relay, in the colour its legend calls Installer supplied Components." in texts
    assert "The drawing shows Couplers, in the colour its legend calls Huzzard supplied Components." in texts


def test_a_component_in_no_legend_colour_says_so_rather_than_guessing():
    read = {"is_drawing": True, "components": [{"label": "Door", "means": ""}]}
    text = lpv.statements(read)[0][1]
    assert "does not say who supplies it" in text


def test_topology_is_not_emitted_by_default(monkeypatch):
    """The words come from OCR and the colours from the pixels. Which line runs
    to which is neither -- it is the model tracing wires, and a wrong
    connection reads exactly like a right one."""
    read = {"is_drawing": True, "connections": [{"from": "Relay", "to": "PC", "via": "USB"}]}
    monkeypatch.delenv("SOWSMITH_LINKED_PICTURE_TOPOLOGY", raising=False)
    assert lpv.statements(read) == []
    monkeypatch.setenv("SOWSMITH_LINKED_PICTURE_TOPOLOGY", "1")
    assert "connects Relay to PC via USB" in lpv.statements(read)[0][1]


def test_the_whole_stage_is_a_no_op_when_the_flag_is_off(monkeypatch):
    monkeypatch.delenv("SOWSMITH_LINKED_PICTURE_VISION", raising=False)
    assert lpv.atoms_from_linked_pictures([object()]) == []
