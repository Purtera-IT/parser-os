"""Decoding: bytes in, structure out. No judgment.

The parser layer currently does three jobs in one place, per format:

    decode      bytes  -> blocks, tables, figures, locators
    segment     blocks -> atoms carrying receipts
    interpret   atoms  -> types, facets, roles

Fusing them is why ``orbitbrief_pdf.py`` is 10,072 lines and ``xlsx_parser.py``
is 4,997, and why "is this table a site roster?" exists once in
``site_roster_extractor.looks_like_site_roster`` for PDFs and again in
``sheet_classifier.classify_sheet`` for spreadsheets. Two implementations of
one judgment, neither able to learn from the other's corrections.

That last point is the reason this package exists, and it is not cosmetic. The
architecture calls for one shared span encoder trained on *every correction
from every task, pooled* -- ten tasks with a few hundred labels each are ten
badly-fit models, while the same labels routed into one representation are a
few thousand examples. Pooling is impossible while the same judgment is
duplicated across five format-specific modules with five different shapes.

So: **decoding is commodity and never learns; interpretation is ours and
always learns.** This package is strictly the first half. Nothing in here may
decide what a block MEANS -- only what it says, where it sits, and how to find
it again.

A decoder answers three questions and no others:

    what does this document say?      -> blocks, in reading order
    what is tabular?                  -> tables, as rows of cells
    where did each piece come from?   -> a Locator per item

Adding a format should mean writing one decoder, not another thousand lines of
extraction tangled with typing rules.
"""

from app.parsers.decode.base import (
    Block,
    DecodedDoc,
    Figure,
    Locator,
    Table,
)

__all__ = ["Block", "DecodedDoc", "Figure", "Locator", "Table"]
