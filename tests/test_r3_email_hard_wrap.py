"""A hard-wrapped email sentence is one sentence, and two sentences are two atoms (010003).

A plain-text client wraps at ~72 columns: "But we are waiting for tv to
arrive at their office (it is with the shipping" / "carrier now). I also
need to keep my eye on the delivery status." Read line by line, the first
half of the dependency was one atom and "carrier now)." another, glued to
nothing. Wrapped lines are rejoined, then cut at their real sentence ends:
the wait is one atom, "keep my eye on the delivery" its own.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.parsers.email_parser import EmailParser, _expand_lines_to_sentences

WAIT = "But we are waiting for tv to arrive at their office (it is with the shipping carrier now)."
EYE = "I also need to keep my eye on the delivery status."


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    monkeypatch.setenv("SOWSMITH_DISABLE_LLM", "1")


def _texts(tmp_path: Path, body: str) -> list[str]:
    p = tmp_path / "m.eml"
    p.write_text("From: Patrick Kelly <patrick.kelly@purtera-it.com>\nTo: Sarah <sarah@acme.com>\n"
                 "Subject: TVs\nDate: Mon, 10 Aug 2026 10:00:00 -0400\n"
                 "Content-Type: text/plain; charset=utf-8\n\n" + body, encoding="utf-8")
    return [a.raw_text for a in EmailParser().parse_artifact_full(project_id="p", artifact_id="a", path=p).atoms]


def test_a_wrapped_sentence_is_rejoined_and_split_at_its_sentence_ends(tmp_path: Path) -> None:
    texts = _texts(tmp_path, "Hi Sarah,\n\nBut we are waiting for tv to arrive at their office (it is with the shipping\n"
                             "carrier now). I also need to keep my eye on the delivery status.\n\nThanks,\nPatrick\n")
    assert WAIT in texts and EYE in texts, texts
    assert "carrier now)." not in texts


def test_quoted_wrap_keeps_its_marker_and_short_lines_never_join() -> None:
    out = _expand_lines_to_sentences(
        ["> But we are waiting for tv to arrive at their office (it is with the shipping",
         "> carrier now). I also need to keep my eye on the delivery status.",
         "Thanks,", "patrick kelly"], 10)
    assert [(n, t) for n, _, t in out] == [(10, "> " + WAIT), (10, "> " + EYE), (12, "Thanks,"), (13, "patrick kelly")]
    # a line that ends its sentence is not wrapped, however long
    out = _expand_lines_to_sentences(
        ["The install crew arrives Monday morning at the loading dock entrance.", "see you then"], 1)
    assert len(out) == 2
