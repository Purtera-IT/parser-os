r"""Sentence segmentation that survives abbreviations.

The regex this replaces -- ``(?<=[.!?])\s+(?=[A-Z0-9"'])`` -- splits on any
period followed by a capital, which in procurement prose is wrong constantly.
On one paragraph of ordinary SOW text it found ten sentences where there are
six, and severed a part number from its own label::

    | Part No.
    | 77-K298 ships from St.
    | Louis.

``pysbd`` is the Golden Rules segmenter: a rule set built against a published
test suite of exactly these cases (titles, initials, geographic abbreviations,
enumerations, decimals). It is pure Python, has no model to download, and
costs microseconds.

Soft dependency, as with ``rapidfuzz`` in ``entity_resolution``: absent the
library the old regex still runs, so segmentation degrades rather than fails.
"""

from __future__ import annotations

import re
import threading

try:  # pragma: no cover - exercised by whichever environment lacks it
    import pysbd as _pysbd
except Exception:  # pragma: no cover
    _pysbd = None  # type: ignore[assignment]

#: The previous behaviour, kept as the fallback so a missing dependency is a
#: quality regression and never an exception.
_NAIVE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")

#: ONE SEGMENTER PER THREAD, never one per process.
#:
#: ``pysbd.Segmenter.segment`` is not reentrant. It parks the text on the
#: instance -- ``self.original_text = text`` -- and then reads it back in
#: ``sentences_with_char_spans`` to locate each sentence it just produced::
#:
#:     re.finditer(re.escape(sent), self.original_text)
#:
#: With ``parse_artifacts`` running a thread pool, a second thread overwrites
#: ``original_text`` while the first is still searching it. The first thread
#: then looks for ITS sentences inside the OTHER thread's document, finds
#: nothing, and the inner loop appends nothing -- so the sentence is dropped
#: from the returned list with no exception and no log line.
#:
#: The caller, ``_expand_lines_to_sentences``, keeps the line whole unless it
#: gets back two or more substantial pieces, so a dropped sentence turns a
#: paragraph's three atoms into one. Measured on live 010094:
#:
#:     serial  192 atoms   "As a follow up, AZ would like to see that attached
#:                          built out." | "As they know that costs may vary by
#:                          location..." | "Is this something you may be able
#:                          to get back to me?"
#:     4-way   190 atoms   ...all three joined into a single atom
#:
#: Atom text is three-quarters of ``label_key``, so this silently detached gold
#: labels at a rate that depended on thread interleaving -- the same deal
#: parsed twice did not produce the same atoms. Deal totals across one corpus
#: went 1298 / 1294 / 1289 at widths 1 / 4 / 8.
#:
#: A lock would serialise every split across the pool. A segmenter is cheap to
#: build and each thread builds at most one, so thread-local state costs a
#: handful of constructions per compile and restores "same input, same atoms".
_local = threading.local()


def _get_segmenter():
    """This thread's reusable segmenter. Construction dominates a split."""
    if _pysbd is None:
        return None
    seg = getattr(_local, "segmenter", None)
    if seg is None:
        # clean=False keeps the text byte-identical to the input, which the
        # callers that build character offsets depend on.
        seg = _pysbd.Segmenter(language="en", clean=False)
        _local.segmenter = seg
    return seg


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences, keeping abbreviations intact."""
    if not text or not text.strip():
        return []
    seg = _get_segmenter()
    if seg is None:  # pragma: no cover - dependency-free fallback
        return [s for s in _NAIVE_SPLIT.split(text) if s.strip()]
    try:
        return [s for s in seg.segment(text) if s and s.strip()]
    except Exception:  # pragma: no cover - never fail a parse over segmentation
        return [s for s in _NAIVE_SPLIT.split(text) if s.strip()]


def count_sentences(text: str) -> int:
    """Sentence count, used as the denominator in coverage ratios."""
    return len(split_sentences(text))


#: An inline list separator: a hyphen or en dash with a space on each side.
#: "4-8 hours" and "on-call" have no spaces, so they never match.
_INLINE_DASH_SEP = re.compile(r"\s+[-–]\s+")

#: A list item opens with a capital, a digit or a bracket ("24/7 on-call").
_ITEM_START = re.compile(r"^[A-Z0-9(\"']")


def split_inline_dash_list(text: str, *, min_items: int = 3, max_item_words: int = 20) -> list[str]:
    """A paragraph written as an inline " - " list, as ``[lead_in, item, ...]``.

    Live 000132: a HubSpot note pasted a bullet list flattened onto one line --
    "Maintenance and support of ... virtualization - Support for physical
    network components (LAN, WLAN) - ... - 24/7 on-call availability" -- and
    the whole scope became one atom. Returns ``[]`` (do not split) unless
    there are at least ``min_items`` " - " separators and every segment after
    the lead-in reads as a list item: it opens with a capital or a digit, is
    short, and holds no sentence break. Prose with a single dash ("the router
    - if it arrives - goes in rack 2") or a range ("4-8 hours") is untouched.
    The lead-in may be empty when the text itself starts with "- ".
    """
    s = " ".join(str(text or "").split())
    if not s:
        return []
    lead_dash = bool(re.match(r"^[-–]\s+", s))
    if lead_dash:
        s = re.sub(r"^[-–]\s+", "", s)
    parts = [p.strip() for p in _INLINE_DASH_SEP.split(s)]
    if lead_dash:
        lead, items = "", parts
    else:
        lead, items = parts[0], parts[1:]
    if len(items) < min_items or any(not p for p in items):
        return []
    for item in items:
        if not _ITEM_START.match(item):
            return []
        if len(item.split()) > max_item_words:
            return []
        if re.search(r"[.!?]\s+[A-Z]", item):
            return []
    return [lead, *items]
