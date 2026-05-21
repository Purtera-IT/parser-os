"""Unit tests for the page-kind classifier (Marriott DD routing fix).

The classifier routes each PDF page to the right downstream pipeline.
Pins the boundary cases the Marriott DD revealed:

* T0.01 with FOUR legends (Structured Cabling + Intrusion +
  Access Control + CCTV) must classify as LEGEND_TABLE.
* T1.04 floor plan (lots of vector strokes + short callouts +
  sheet number) must classify as SCHEMATIC_DRAWING.
* T0.00 cover with responsibility matrix + prose specs must
  classify as SPEC_PROSE or SCHEDULE_BOM, NOT a schematic page.
* Single-line dimensional callouts in a legend region must not
  hijack the locator into emitting a fake legend.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


# Lightweight stand-in for TextBlock — the classifier accepts duck-typed
# objects with .text and .bbox attributes. Production code passes the
# real ``TextBlock`` from the locator; tests pass these stubs.
@dataclass
class _Block:
    text: str
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 100.0, 20.0)


# Lightweight stand-in for a PyMuPDF page. Only the attributes the
# classifier reads (`.rect` + `.get_drawings()`) need to be present.
class _Rect:
    def __init__(self, w: float, h: float) -> None:
        self.width = w
        self.height = h
        self.x0 = 0.0
        self.y0 = 0.0
        self.x1 = w
        self.y1 = h


class _Page:
    def __init__(
        self,
        *,
        width: float = 2160.0,
        height: float = 3240.0,
        drawings: Sequence[dict] | None = None,
    ) -> None:
        self.rect = _Rect(width, height)
        self._drawings = list(drawings or [])

    def get_drawings(self) -> list[dict]:
        return self._drawings


# ── Imports under test ────────────────────────────────────────────


def _import_classifier():
    from orbitbrief_page_os.segmentation.schematic.page_kind_classifier import (
        LEGEND_TABLE,
        SCHEMATIC_DRAWING,
        SCHEDULE_BOM,
        SPEC_PROSE,
        COVER_TITLE,
        UNKNOWN,
        classify_page_kind,
    )
    return {
        "LEGEND_TABLE": LEGEND_TABLE,
        "SCHEMATIC_DRAWING": SCHEMATIC_DRAWING,
        "SCHEDULE_BOM": SCHEDULE_BOM,
        "SPEC_PROSE": SPEC_PROSE,
        "COVER_TITLE": COVER_TITLE,
        "UNKNOWN": UNKNOWN,
        "classify_page_kind": classify_page_kind,
    }


def _import_legend_filters():
    from orbitbrief_page_os.segmentation.schematic.legend_parser import (
        _looks_dimensional,
        _all_entries_look_dimensional,
    )
    return _looks_dimensional, _all_entries_look_dimensional


# ── Per-kind classification cases ─────────────────────────────────


def test_legend_table_with_four_headers_classifies_as_legend_table() -> None:
    """The exact Marriott T0.01 shape."""
    api = _import_classifier()
    blocks: list[_Block] = []
    # Four legend headers (short)
    for header in [
        "STRUCTURED CABLING SYMBOLS LEGEND",
        "INTRUSION DETECTION SYMBOLS LEGEND",
        "ACCESS CONTROL AND INTERCOM SYMBOLS LEGEND",
        "CCTV SYMBOLS LEGEND",
    ]:
        blocks.append(_Block(text=header))
    # Many short symbol/label rows under each
    for i in range(40):
        blocks.append(_Block(text=f"SYM{i} = LABEL{i}"))
    page = _Page(drawings=[{"rect": _Rect(10, 5)}])  # very low stroke density
    result = api["classify_page_kind"](
        page_index=1, page=page, blocks=blocks
    )
    assert result.kind == api["LEGEND_TABLE"], result.rationale
    assert result.signals["legend_headers"] >= 1


def test_spec_prose_page_classifies_as_spec_prose() -> None:
    """Page 0 of Marriott — STRUCTURED CABLING SYSTEM SCOPE/OVERVIEW paragraphs."""
    api = _import_classifier()
    blocks = [
        _Block(
            text=(
                "THIS PROJECT INVOLVES THE INSTALLATION OF INFRASTRUCTURE FOR THE VOICE, "
                "DATA, MATV, CCTV, ACCESS CONTROL AND INTRUSION DETECTION SYSTEMS. "
                "THE DESIGN OF THESE INFRASTRUCTURE SYSTEMS IS BASED UPON CURRENT "
                "INDUSTRY STANDARDS, SUPPORTING AN OPEN SYSTEM ARCHITECTURE."
            )
        ),
        _Block(
            text=(
                "FIBER OPTIC AND CAT-6 UTP INFRASTRUCTURE TO SUPPORT DATA, VOICE, AND "
                "INTERNET SERVICES THROUGHOUT THE FACILITY. OUTLETS WILL BE CONFIGURED "
                "PER FLOOR PLANS AND CORRESPONDING OUTLET CONFIGURATION DETAILS."
            )
        ),
    ]
    page = _Page(drawings=[])
    result = api["classify_page_kind"](page_index=0, page=page, blocks=blocks)
    assert result.kind == api["SPEC_PROSE"], result.rationale


def test_schedule_bom_with_specifications_list_header() -> None:
    """Page 2 of Marriott — COPPER COMPONENTS SPECIFICATIONS LIST."""
    api = _import_classifier()
    blocks = [
        _Block(text="COPPER COMPONENTS SPECIFICATIONS LIST"),
        _Block(text="DESCRIPTION"),
        _Block(text="MANUFACTURER"),
        _Block(text="PART NUMBER"),
        _Block(text="COMMENTS"),
    ]
    # Add ~30 short rows to look tabular
    for i in range(30):
        blocks.append(_Block(text=f"row {i}"))
    page = _Page(drawings=[])
    result = api["classify_page_kind"](page_index=2, page=page, blocks=blocks)
    assert result.kind == api["SCHEDULE_BOM"], result.rationale


def test_schematic_drawing_with_strokes_and_short_callouts() -> None:
    """Page 4 of Marriott — T1.01 floor plan."""
    api = _import_classifier()
    # Short callouts
    blocks = [_Block(text=tok) for tok in (
        "T1.01", "LOWER LOBBY", "CR", "TV", "WN", "1", "2", "3", "4", "5",
        "F", "E", "D", "A", "B", "C",
    )]
    # Lots of vector strokes covering the page
    drawings = [
        {"rect": _Rect(500, 800)},
        {"rect": _Rect(600, 700)},
        {"rect": _Rect(800, 900)},
    ]
    page = _Page(drawings=drawings)
    result = api["classify_page_kind"](page_index=4, page=page, blocks=blocks)
    assert result.kind == api["SCHEMATIC_DRAWING"], (
        f"got kind={result.kind} rationale={result.rationale} signals={result.signals}"
    )


def test_unknown_for_empty_page() -> None:
    """An empty page falls through to UNKNOWN so the conservative full
    flow runs (preserves existing behavior)."""
    api = _import_classifier()
    page = _Page(drawings=[])
    result = api["classify_page_kind"](page_index=99, page=page, blocks=[])
    assert result.kind == api["UNKNOWN"]


# ── Dimensional-label filter ──────────────────────────────────────


def test_dimensional_filter_catches_marriott_conduit_callout() -> None:
    _looks, _all_dim = _import_legend_filters()
    assert _looks('1-1/4"∅') is True
    assert _looks('7-8') is True
    assert _looks('2"') is True
    assert _looks('50 mm') is True
    assert _looks("3/4∅") is True


def test_dimensional_filter_keeps_real_legend_labels() -> None:
    _looks, _all_dim = _import_legend_filters()
    assert _looks("CARD READER") is False
    assert _looks("PTZ CAMERA") is False
    assert _looks("FIRE ALARM PULL STATION") is False
    assert _looks("WALL-MOUNTED NETWORK OUTLET") is False


def test_all_dimensional_check_rejects_pure_unit_legend() -> None:
    """The original Marriott failure mode: a 'legend' where every
    entry is dimensional text. Must return True so the caller drops it."""
    _looks, _all_dim = _import_legend_filters()

    class E:
        def __init__(self, label):
            self.label_text = label
    entries = [E('1-1/4"∅'), E('2"'), E('3/4∅')]
    assert _all_dim(entries) is True


def test_all_dimensional_check_keeps_mixed_legend() -> None:
    """A legend with at least one real symbol/label row passes."""
    _looks, _all_dim = _import_legend_filters()

    class E:
        def __init__(self, label):
            self.label_text = label
    entries = [E('1-1/4"∅'), E("CARD READER"), E("PTZ CAMERA")]
    assert _all_dim(entries) is False
