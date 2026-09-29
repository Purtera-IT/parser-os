# -*- coding: utf-8 -*-
"""PyMuPDF has ONE global context, so PDFs parse one at a time.

Separate ``fitz.Document`` objects in separate threads still share that
context and corrupt each other's extraction -- silently. Live 01491cca's
``Attachment+1_B704+Fitness+Center+P.W.+SOW.pdf`` on PyMuPDF 1.27.2.3, same
file, same bytes:

    serially     156 atoms, four runs out of four
    4 threads    190, 153, 157, 157

That is a statement of work losing or inventing up to 24% of its content
depending on what else happened to be parsing, and moving every ``label_key``
with it.
"""
import threading
from concurrent.futures import ThreadPoolExecutor

from app.parsers._pdf_lock import PDF_PARSE_LOCK
from app.parsers.orbitbrief_pdf import OrbitBriefPdfParser


def test_the_lock_is_reentrant() -> None:
    """A PDF parse re-enters through nested helpers."""
    with PDF_PARSE_LOCK:
        with PDF_PARSE_LOCK:
            assert True


def test_parse_artifact_holds_the_lock(monkeypatch) -> None:
    """The lock must cover the WHOLE parse, not just the fitz calls.

    OCR runs inside the fitz scope -- ``ocr_pdf_page`` needs the open page --
    so a lock that only wrapped document access would leave the pages exposed.
    """
    parser = OrbitBriefPdfParser()
    held: list[bool] = []

    def fake(self, *a, **kw):
        # PDF_PARSE_LOCK is an RLock: acquire(blocking=False) from the holding
        # thread succeeds, so ask whether ANOTHER thread could take it.
        taken = []

        def probe() -> None:
            got = PDF_PARSE_LOCK.acquire(blocking=False)
            taken.append(got)
            if got:
                PDF_PARSE_LOCK.release()

        t = threading.Thread(target=probe)
        t.start()
        t.join()
        held.append(not taken[0])
        return "parsed"

    monkeypatch.setattr(OrbitBriefPdfParser, "_parse_artifact_locked", fake)
    out = parser.parse_artifact("p", "a", __import__("pathlib").Path("x.pdf"))
    assert out == "parsed"
    assert held == [True], "parse_artifact ran without holding PDF_PARSE_LOCK"


def test_only_one_pdf_parses_at_a_time(monkeypatch) -> None:
    """Two threads must never be inside a PDF parse together."""
    parser = OrbitBriefPdfParser()
    inside = 0
    peak = 0
    guard = threading.Lock()

    def fake(self, *a, **kw):
        nonlocal inside, peak
        with guard:
            inside += 1
            peak = max(peak, inside)
        # Long enough that overlap would be certain if it were possible.
        threading.Event().wait(0.02)
        with guard:
            inside -= 1
        return "parsed"

    monkeypatch.setattr(OrbitBriefPdfParser, "_parse_artifact_locked", fake)
    from pathlib import Path

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(
            lambda i: parser.parse_artifact("p", "a", Path("x%d.pdf" % i)),
            range(16),
        ))
    assert peak == 1, "%d PDF parses overlapped; MuPDF's context is shared" % peak
