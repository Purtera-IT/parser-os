"""The decode contract: what every format hands upward, and nothing more."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class Locator:
    """How to find this content in the source again.

    Every field is optional because formats differ in what they can honestly
    offer: a spreadsheet knows sheet/row/column and has no bbox, a PDF knows
    page and bbox, an email knows a character range in the decoded body.
    A locator states what that format actually knows rather than inventing
    coordinates to satisfy a schema.

    ``bbox`` is the upgrade worth calling out. A byte range proves a claim came
    from a file; a page and a rectangle proves it and can be SHOWN -- the
    reviewer sees the highlighted region instead of being told an offset. The
    schematic replay path already renders a page, crops to a bbox and hashes
    the pixels, so bbox verification is an existing capability being extended
    to a second producer, not a new one.

    ``EvidenceReceipt.locator`` is ``dict[str, Any]`` already, so carrying this
    needs no schema migration -- ``as_dict()`` drops straight in.
    """

    page: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    char_start: int | None = None
    char_end: int | None = None
    sheet: str | None = None
    row: int | None = None
    col: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Only the fields this locator actually knows."""
        out: dict[str, Any] = {}
        for k in ("page", "bbox", "char_start", "char_end", "sheet", "row", "col"):
            v = getattr(self, k)
            if v is not None:
                out[k] = list(v) if k == "bbox" else v
        out.update(self.extra)
        return out


@dataclass(frozen=True)
class Block:
    """One run of text as the document presents it.

    ``kind`` is layout, not meaning: "heading" says the document typeset it as
    a heading, not that it introduces a scope section. The moment a decoder
    wants to say "requirement" or "site" it has crossed out of this package.
    """

    text: str
    kind: str = "paragraph"  # paragraph | heading | list_item | caption | cell
    locator: Locator = field(default_factory=Locator)
    order: int = 0


@dataclass(frozen=True)
class Table:
    """A grid, as rows of cell strings.

    Deliberately dumb: no header detection, no column typing, no roster
    recognition. Those are judgments, they belong above the seam, and they are
    the readouts that need to learn from pooled corrections.
    """

    rows: list[list[str]]
    locator: Locator = field(default_factory=Locator)
    order: int = 0

    @property
    def cell_count(self) -> int:
        return sum(1 for r in self.rows for c in r if str(c or "").strip())


@dataclass(frozen=True)
class Figure:
    """An embedded image, and where it sits."""

    locator: Locator = field(default_factory=Locator)
    image_ref: str | None = None
    caption: str | None = None
    order: int = 0


@dataclass(frozen=True)
class DecodedDoc:
    """Everything one decoder could honestly read out of one file."""

    path: Path
    blocks: list[Block] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    figures: list[Figure] = field(default_factory=list)
    page_count: int = 0
    #: Which implementation produced each part, e.g. {"tables": "doc_intel",
    #: "text": "fitz"}. Recorded because the choice is made per page and per
    #: document, and a receipt should say which reader it came from.
    backends: dict[str, str] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n".join(b.text for b in self.blocks if b.text)

    @property
    def cell_count(self) -> int:
        return sum(t.cell_count for t in self.tables)


@runtime_checkable
class Decoder(Protocol):
    """Bytes to structure. Implementations must not raise.

    A decoder that cannot read a file returns an empty ``DecodedDoc`` rather
    than throwing, because one unreadable artifact must never fail a compile
    that has twenty others to get through.
    """

    name: str

    def can_decode(self, path: Path) -> bool: ...

    def decode(self, path: Path) -> DecodedDoc: ...
