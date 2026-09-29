"""One OCR per distinct image, even with eight parsers running.

`parse_artifacts` runs SOWSMITH_PARSE_WORKERS threads (default 4). A plain
check-then-fill cache lets every thread miss on the same signature logo before
any of them fills it, so the cache would turn 16 billed Document Intelligence
calls into 8 rather than into 1 -- on the single largest line of the dev bill.
"""

from __future__ import annotations

import threading
import time

from app.parsers import _ocr_chain as oc


def _install_slow_backend(monkeypatch, delay: float = 0.2):
    calls: list[int] = []

    def _slow(image_bytes, notes):
        calls.append(len(image_bytes))
        time.sleep(delay)          # wide enough for the herd to form
        return {"text": "OCR", "backend": "fake", "confidence": 1.0, "notes": list(notes)}

    monkeypatch.setattr(oc, "_ocr_image_bytes_uncached", _slow)
    oc._OCR_CACHE.clear()
    oc._OCR_KEY_LOCKS.clear()
    return calls


def test_one_ocr_per_image_when_many_threads_want_the_same_one(monkeypatch):
    calls = _install_slow_backend(monkeypatch)
    logo = b"L" * 9000
    out: list[dict] = []

    threads = [threading.Thread(target=lambda: out.append(oc.ocr_image_bytes(logo)))
               for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(calls) == 1, f"{len(calls)} threads paid for the same picture"
    assert len(out) == 8
    assert all(o["text"] == "OCR" for o in out)


def test_different_images_are_not_serialised(monkeypatch):
    """The lock is per key: a shared one would stall the whole parse pool."""
    calls = _install_slow_backend(monkeypatch, delay=0.2)
    a, b = b"A" * 9000, b"B" * 9000

    started = time.time()
    threads = [threading.Thread(target=oc.ocr_image_bytes, args=(a,)),
               threading.Thread(target=oc.ocr_image_bytes, args=(b,))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.time() - started

    assert len(calls) == 2
    # Serialised would be ~0.4s; concurrent ~0.2s. 0.35 leaves room for a slow
    # machine without letting a true serialisation pass.
    assert elapsed < 0.35, f"distinct images serialised: {elapsed:.2f}s"


def test_a_caller_cannot_poison_the_cache_for_others(monkeypatch):
    _install_slow_backend(monkeypatch, delay=0)
    img = b"X" * 9000
    first = oc.ocr_image_bytes(img)
    first["notes"].append("CALLER SCRIBBLE")
    first["text"] = "MUTATED"

    second = oc.ocr_image_bytes(img)
    assert second["text"] == "OCR"
    assert not any("SCRIBBLE" in n for n in second["notes"])
