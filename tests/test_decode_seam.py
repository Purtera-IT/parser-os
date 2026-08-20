"""The decode seam: structure out, judgment absent.

These pin the two properties that make the seam worth having -- that a decoder
reports what a format says without deciding what it means, and that the reader
choice is made per component on what was actually recovered.
"""
from __future__ import annotations

import pathlib

import pytest

from app.parsers.decode import Block, DecodedDoc, Figure, Locator, Table


class TestLocator:
    def test_reports_only_what_the_format_knows(self):
        # A spreadsheet has no bbox and a PDF has no sheet name. A locator
        # states what it knows rather than padding a fixed schema with nulls.
        assert Locator(page=3, bbox=(1, 2, 3, 4)).as_dict() == {
            "page": 3, "bbox": [1, 2, 3, 4],
        }
        assert Locator(sheet="Sites", row=7, col=2).as_dict() == {
            "sheet": "Sites", "row": 7, "col": 2,
        }
        assert Locator().as_dict() == {}

    def test_drops_into_an_evidence_receipt_unchanged(self):
        # EvidenceReceipt.locator is dict[str, Any], so bbox provenance needs
        # no schema migration -- this is the whole reason as_dict exists.
        d = Locator(page=2, bbox=(10.0, 20.0, 30.0, 40.0),
                    extra={"reader": "doc_intel"}).as_dict()
        assert d["page"] == 2 and d["bbox"] == [10.0, 20.0, 30.0, 40.0]
        assert d["reader"] == "doc_intel"


class TestDecodedDoc:
    def test_counts_only_non_empty_cells(self):
        t = Table(rows=[["a", "", "b"], ["", "", ""], ["c", "d", ""]])
        assert t.cell_count == 4
        assert DecodedDoc(path=__import__("pathlib").Path("x"), tables=[t]).cell_count == 4

    def test_text_is_the_blocks_in_order(self):
        doc = DecodedDoc(
            path=__import__("pathlib").Path("x"),
            blocks=[Block(text="one", order=0), Block(text="two", order=1)],
        )
        assert doc.text == "one\ntwo"


def test_a_decoder_emits_layout_kinds_only():
    """A decoder may say "heading"; it may never say "requirement".

    Asserted against what a decoder actually emits rather than what its
    docstring claims. The moment a decoder assigns meaning, that judgment stops
    being poolable -- it becomes one more format-specific copy of a decision
    that should be a single readout learning from every format's corrections.
    """
    from app.parsers.decode.pdf import PdfDecoder

    LAYOUT = {"paragraph", "heading", "list_item", "caption", "cell"}
    TAXONOMY = {"requirement", "scope_item", "physical_site", "bom_line",
                "service_line", "deliverable", "open_question"}

    pdf = pathlib.Path(__file__).resolve().parents[1] / "tests" / "fixtures"
    sample = next(iter(sorted(pdf.rglob("*.pdf"))), None)
    if sample is None:
        pytest.skip("no PDF fixture in this checkout")

    doc = PdfDecoder().decode(sample)
    kinds = {b.kind for b in doc.blocks} | {"cell"}
    assert kinds <= LAYOUT, f"decoder emitted non-layout kinds: {kinds - LAYOUT}"
    assert not (kinds & TAXONOMY)


def test_decoder_protocol_is_minimal():
    """Three questions and no others: says what, tabular what, found where."""
    from app.parsers.decode.base import Decoder

    callables = {m for m in dir(Decoder) if not m.startswith("_")}
    assert callables == {"can_decode", "decode"}
    # `name` is declared as an annotation, so it identifies the implementation
    # without adding a fourth thing a decoder has to do.
    assert "name" in getattr(Decoder, "__annotations__", {})


def test_a_trivial_decoder_satisfies_the_protocol():
    """The contract has to be cheap enough that a new format is one small class."""
    from app.parsers.decode.base import Decoder

    class Nothing:
        name = "nothing"

        def can_decode(self, path):
            return False

        def decode(self, path):
            return DecodedDoc(path=path)

    assert isinstance(Nothing(), Decoder)
