# -*- coding: utf-8 -*-
"""``split_sentences`` must return the same split on every thread.

``pysbd.Segmenter`` parks its input on the instance and reads it back to
locate each sentence, so one shared segmenter lets a second thread's document
erase the text the first is still searching. The sentences it cannot find are
dropped silently, which turned a paragraph's three atoms into one and moved
atom text -- and therefore ``label_key`` -- from run to run.
"""
from concurrent.futures import ThreadPoolExecutor

from app.core.sentences import split_sentences

# Distinct texts, so a thread that reads another's buffer finds none of its own
# sentences.
CASES = [
    "Part No. 77-K298 ships from St. Louis. It arrives Monday. Please confirm.",
    "As a follow up, AZ would like that built out. Costs may vary by location. "
    "Is this something you can get back to me on?",
    "Dr. Alvarez signed off on Sept. 14. The J. Smith quote is attached. "
    "We need it by 5 p.m.",
    "Install 12 APs at the St. Paul site. Each drop runs to the IDF. "
    "Total is 18 hours.",
]


def test_abbreviations_stay_intact() -> None:
    """Guards the rest of this file: a naive-regex fallback fails here.

    If the soft dependency is missing the split degrades to the old regex,
    which severs "St. Louis" -- and a degraded splitter is stateless, so it
    would pass the stability tests below for the wrong reason.
    """
    pieces = split_sentences(CASES[0])
    assert any("St. Louis" in p for p in pieces), pieces


def test_split_is_stable_under_threads() -> None:
    serial = [split_sentences(t) for t in CASES]
    assert all(len(p) > 1 for p in serial), serial

    # Enough repeats and enough width that an unsynchronised segmenter is
    # overwhelmingly likely to interleave at least once.
    work = [CASES[i % len(CASES)] for i in range(400)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        got = list(pool.map(split_sentences, work))

    for i, pieces in enumerate(got):
        want = serial[i % len(CASES)]
        assert pieces == want, (
            "thread pool changed the split of %r:\n  serial: %r\n  pooled: %r"
            % (work[i], want, pieces)
        )


def test_every_sentence_survives() -> None:
    """A dropped sentence is the specific failure: nothing may go missing."""
    work = list(CASES) * 100
    with ThreadPoolExecutor(max_workers=8) as pool:
        got = list(pool.map(split_sentences, work))
    for text, pieces in zip(work, got):
        assert "".join(pieces).replace(" ", "") == text.replace(" ", ""), (
            "segmentation lost text from %r: %r" % (text, pieces)
        )
