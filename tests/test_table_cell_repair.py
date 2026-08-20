"""The guards on PDF table-cell repair.

``table.extract()`` and ``page.get_text(clip=cell)`` read the same rectangle and
disagree. Neither is ground truth, so the repair only substitutes the clipped
text under a condition a repair can satisfy and a corruption cannot. These tests
pin that condition — every case below is a real disagreement measured across the
stored corpus, not a hypothetical.
"""
from __future__ import annotations

from app.parsers.orbitbrief_pdf import _same_but_for_underscores, _same_glyphs


class TestSameGlyphs:
    """Transposition: extract() emits the right characters in the wrong order."""

    def test_transposed_glyphs_are_a_repair(self):
        # Xtra Lease K298: 9 of 54 cells came back like this.
        assert _same_glyphs("Initail document", "Initial document")
        assert _same_glyphs("order of executoin", "order of execution")

    def test_a_dropped_character_is_not_a_repair(self):
        # A clip boundary slicing the leading "f" off "fication:". Rejecting
        # this is the whole reason the guard is not "just trust the clip".
        assert not _same_glyphs("ication:", "fication:")

    def test_a_neighbouring_cell_bleeding_in_is_not_a_repair(self):
        # APS_fiber_RFP p2: the clip pulled two stray glyphs off the next cell.
        assert not _same_glyphs(
            "APPENDIX B - Letter of Transmittal Form",
            "g p APPENDIX B - Letter of Transmittal Form",
        )

    def test_a_different_word_is_never_a_repair(self):
        assert not _same_glyphs("install 4 cameras", "install 7 cameras")


class TestSameButForUnderscores:
    """Underscore loss: the case the anagram guard could not see."""

    def test_a_dropped_underscore_in_an_email_is_a_repair(self):
        # APS_fiber_RFP p12. "russell r@aps.edu" is not a deliverable address,
        # and nothing downstream can tell that it was mangled.
        assert _same_but_for_underscores(
            "Send to russell r@aps.edu", "Send to russell_r@aps.edu"
        )

    def test_a_relocated_underscore_is_a_repair(self):
        # The commoner shape: pushed to the end, a space left behind.
        assert _same_but_for_underscores("quantity conflict _", "quantity_conflict")
        assert _same_but_for_underscores("material:cat6 utp _", "material:cat6_utp")

    def test_it_does_not_license_a_changed_character(self):
        assert not _same_but_for_underscores("part_A100", "part_A700")

    def test_it_does_not_license_a_bleeding_neighbour(self):
        assert not _same_but_for_underscores(
            "APPENDIX B - Letter of Transmittal Form",
            "g p APPENDIX B - Letter of Transmittal Form",
        )

    def test_it_does_not_license_a_sliced_glyph(self):
        assert not _same_but_for_underscores("ication:", "fication:")


def test_the_two_guards_cover_different_failures():
    """Neither guard subsumes the other, which is why both are applied."""
    # anagram only: same characters, reordered, no underscore involved
    assert _same_glyphs("Initail", "Initial")
    assert not _same_but_for_underscores("Initail", "Initial")

    # underscore only: a character is genuinely absent from one side
    assert _same_but_for_underscores("russell r@aps.edu", "russell_r@aps.edu")
    assert not _same_glyphs("russell r@aps.edu", "russell_r@aps.edu")
