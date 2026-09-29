# -*- coding: utf-8 -*-
"""The replay normaliser's fast path must give the same answer, always.

``_replay_norm`` strips Unicode combining marks so "café" matches "cafe" and
"M\xa0Smith" matches "M Smith" -- a lot of "failed" replay receipts in the
corpus were only NFKD drift between the parser's text and the re-read source.

It is also the hottest thing in the compile's second phase. Profiled on one
live deal the combining-mark generator ran **42,451,772 times** -- 17.6 of
``source_replay``'s 41.7 seconds, with ``unicodedata.combining`` alone
accounting for 42.3 million calls -- because replay asks about the same
strings relentlessly: every atom against every candidate line.

It is a pure function of its input, so it is now cached, and ASCII skips the
Unicode work entirely: NFKD leaves ASCII alone and ASCII carries no combining
marks, so both steps are provably no-ops there. Measured on 15,195 real atoms:
9.80s to 4.62s, with byte-identical receipts.

These tests exist so that "provably" stays true.
"""
from __future__ import annotations

import random
import unicodedata

import pytest

from app.core.normalizers import normalize_text
from app.core.source_replay import _replay_norm


def _reference(text: str) -> str:
    """Exactly what the function did before the fast path."""
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return normalize_text(text)


#: The behaviours the docstring promises, and the shapes that live in the
#: corpus: accents, non-breaking spaces, ligatures, en dashes, CJK, symbols.
CASES = [
    "café",
    "M\xa0Smith",
    "naïve",
    "Ünïcödé",
    "é",                       # combining acute, not precomposed
    "SOW – Premise Wiring, Bldg. 704",
    "Cat6 UTP drop",
    "RJ45 / Data Jack",
    "ﬁ ligature",
    "Ω ohm",
    "北京",
    "Ⅻ roman",
    "①②③",
    "  spaced  ",
    "",
]


@pytest.mark.parametrize("text", CASES)
def test_the_fast_path_agrees_with_the_reference(text: str) -> None:
    assert _replay_norm(text) == _reference(text)


def test_the_promises_in_the_docstring_still_hold() -> None:
    assert _replay_norm("café") == _replay_norm("cafe")
    assert _replay_norm("M\xa0Smith") == _replay_norm("M Smith")


def test_none_and_empty_are_the_same_as_before() -> None:
    assert _replay_norm(None) == _reference(None)
    assert _replay_norm("") == _reference("")


def test_it_agrees_on_random_unicode() -> None:
    """The hand-picked cases are the ones we thought of. This covers the rest."""
    rnd = random.Random(7)
    for _ in range(4000):
        text = "".join(chr(rnd.randint(32, 0x2FFF)) for _ in range(rnd.randint(0, 40)))
        assert _replay_norm(text) == _reference(text), repr(text)


def test_the_cache_cannot_hand_back_another_string_s_answer() -> None:
    """A cache keyed on the wrong thing is worse than no cache."""
    seen: dict[str, str] = {}
    for text in CASES * 3:
        got = _replay_norm(text)
        if text in seen:
            assert got == seen[text]
        seen[text] = got
        assert got == _reference(text)
