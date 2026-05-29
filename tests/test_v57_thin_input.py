"""v57: thin-input regression tests for Notes.pdf-style discovery notes.

Covers:

* ``_split_form_qa_blob`` — splits free-form Q&A transcripts (no Q1./A1.
  markers) into one chunk per question.
* ``_decode_html_entities`` — strips ``&nbsp;`` runs that the PDF text
  extractor preserves verbatim.
* ``_is_negated_match`` / ``_emit_devices`` — suppresses ``device:storage``
  from "not via thumb drive" hallucinations.
"""

from __future__ import annotations

from app.parsers.orbitbrief_pdf import (
    _decode_html_entities,
    _split_form_qa_blob,
)
from app.core.entity_extraction import _emit_devices, _is_negated_match


# The exact text shape from the Yonah Sapir Notes.pdf
NOTES_PDF_TEXT = (
    "City/state for this location? – location Santa Fe, NM 87506 "
    "What size TVs? – Do you have the model #?   "
    "LG part 65UN570H0UD – 65\" "
    "Would you be providing the mounting gear? –  DCW to provide "
    "Will everything be there when techs arrive? Yes "
    "Will techs need to perform an inventory count?   Yes "
    "Property has 23 dwellings,  approx 8 have second story  "
    "( walking stairs,  no elevator ) "
    "Programming is pretty easy,  but not via thumb drive,  "
    "several categories and menus within TV to manually set the IP addressing. "
    "Approx time to configure and test each unit,  approx 15 mins "
    "All boxes and old TVs remain at  the local IT managers care. "
    " ( nothing has to be done with them)"
)


def test_decode_html_entities_strips_nbsp():
    assert _decode_html_entities("a b") == "a b"
    assert _decode_html_entities("a&nbsp;&nbsp;b") == "a b"
    assert _decode_html_entities("a&#160;b") == "a b"
    assert _decode_html_entities("") == ""
    assert _decode_html_entities("plain text") == "plain text"


def test_form_qa_splitter_short_circuits_on_few_question_marks():
    text = "Only one question? With its answer."
    assert _split_form_qa_blob(text) == ["Only one question? With its answer."]


def test_form_qa_splitter_short_circuits_on_q1_a1_markers():
    text = "Q1. What time? A1. 5pm. Q2. Where? A2. Lobby."
    # Should defer to _split_qa_blob — return as-is (no double-split).
    result = _split_form_qa_blob(text)
    assert len(result) == 1
    assert "Q1." in result[0]


def test_form_qa_splitter_on_notes_pdf_text():
    chunks = _split_form_qa_blob(NOTES_PDF_TEXT)
    # We expect at least 6 distinct chunks — one per question + the
    # trailing declarative facts about dwellings/programming/timing.
    assert len(chunks) >= 6, f"only got {len(chunks)} chunks: {chunks}"
    # Each major fact should land in its own chunk.
    joined = " || ".join(chunks)
    assert "Santa Fe" in joined
    assert "65UN570H0UD" in joined
    assert "DCW to provide" in joined
    assert "23 dwellings" in joined
    assert "15 mins" in joined
    # No chunk should be the entire blob (the bug we're fixing).
    for chunk in chunks:
        assert len(chunk) < len(NOTES_PDF_TEXT) * 0.8, (
            f"chunk too large — splitter didn't trigger: {chunk[:120]}..."
        )


def test_form_qa_splitter_decodes_nbsp():
    chunks = _split_form_qa_blob(NOTES_PDF_TEXT)
    for chunk in chunks:
        assert " " not in chunk, f"nbsp leaked into chunk: {chunk}"
        assert "&nbsp;" not in chunk


def test_is_negated_match_detects_thumb_drive_negation():
    text = "programming is pretty easy, but not via thumb drive, several"
    idx = text.find("thumb drive")
    assert idx > 0
    assert _is_negated_match(text, idx) is True


def test_is_negated_match_skips_unrelated_negation_far_back():
    # Negation 80 chars back — outside the ~40 char window.
    text = (
        "we do not store backups locally. instead the operations team "
        "manages thumb drive rotation manually."
    )
    idx = text.find("thumb drive")
    assert _is_negated_match(text, idx) is False


def test_is_negated_match_respects_override():
    text = "we ship hdd and also no tape backup, plus thumb drive"
    idx = text.find("thumb drive")
    # "plus" is an override AFTER "no", so the negation is overridden.
    assert _is_negated_match(text, idx) is False


def test_emit_devices_suppresses_negated_storage():
    # Minimal alias index — just the universal "thumb drive" alias.
    alias_index = {"thumb drive": "storage"}
    text = "programming is pretty easy, but not via thumb drive, several menus"
    keys = _emit_devices(text.lower(), alias_index, pack=None)
    assert "device:storage" not in keys, (
        f"expected negation guard to suppress storage, got: {keys}"
    )


def test_emit_devices_still_fires_on_positive_mentions():
    alias_index = {"thumb drive": "storage", "ssd": "storage"}
    text = "we'll ship 50 thumb drives with the kit, plus 20 ssds for rack 3"
    keys = _emit_devices(text.lower(), alias_index, pack=None)
    assert "device:storage" in keys
