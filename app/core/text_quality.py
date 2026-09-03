"""Is this text readable prose, or OCR debris?

Live 010300 (2026-09-03): the image-only Teams PSOW went through both a
vision paraphrase (readable) and a tesseract layer (not). Sixteen tesseract
lines -- "‘Tes aks wilenur tht projetcompen mee egutements…", "Psat iy Wa ns
ache A VIA", "44 Marware and materats ae ot nclded" -- reached the envelope as
scope_items, pricing_assumptions and vendor_line_items. A PM cannot act on
them and a head cannot learn from them.

The judge is the LANGUAGE, not the domain: a lowercase token counts as a word
when it is in an English wordlist (shipped compressed beside this module);
capitalised tokens (names, products), all-caps abbreviations (SOW, PoE, D4C)
and tokens with digits are never held against a line. OCR debris is made of
lowercase tokens that are not words. Shape rules (a vowel, no long consonant
run) are the fallback when the wordlist cannot be loaded.
"""
from __future__ import annotations

import gzip
import re
from functools import lru_cache
from pathlib import Path

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'’\-]*")
_VOWEL_RE = re.compile(r"[aeiouyAEIOUY]")
_CONSONANT_RUN_RE = re.compile(r"[bcdfghjklmnpqrstvwxzBCDFGHJKLMNPQRSTVWXZ]{5,}")
_WORDLIST = Path(__file__).with_name("data") / "english_words.txt.gz"


@lru_cache(maxsize=1)
def _words() -> frozenset[str]:
    try:
        with gzip.open(_WORDLIST, "rt", encoding="utf-8") as fh:
            return frozenset(w.strip() for w in fh if w.strip())
    except Exception:
        return frozenset()


def _shape_ok(tok: str) -> bool:
    if not _VOWEL_RE.search(tok) or _CONSONANT_RUN_RE.search(tok):
        return False
    vowels = len(_VOWEL_RE.findall(tok))
    return not (len(tok) >= 6 and vowels / len(tok) > 0.7)


def _is_word(tok: str) -> bool:
    core = tok.strip("'’-").lower()
    if len(core) < 3:
        return True
    words = _words()
    if words:
        if core in words:
            return True
        # inflections the base list lacks: plurals, past, progressive, adverbs
        for suf in ("s", "es", "ed", "d", "ing", "ly", "er", "ers", "ies", "ied"):
            if core.endswith(suf) and len(core) - len(suf) >= 3:
                stem = core[: -len(suf)]
                if stem in words or (suf in ("ing", "ed", "er", "ers") and stem + "e" in words) or (suf in ("ies", "ied") and stem + "y" in words):
                    return True
        # simple possessive / hyphen halves
        if core.endswith("'s") and core[:-2] in words:
            return True
        parts = [p for p in core.split("-") if p]
        if len(parts) > 1 and all(p in words or len(p) < 3 for p in parts):
            return True
        return False
    return _shape_ok(core)


def _judged(text: str) -> list[str]:
    """Lowercase alphabetic tokens of length >= 3: the only ones that can be debris."""
    out: list[str] = []
    for t in _TOKEN_RE.findall(text or ""):
        core = t.strip("'’-")
        if len(core) < 3 or not core[0].islower():
            continue  # capitalised = name/product/heading; never held against the line
        out.append(t)
    return out


def readability(text: str) -> float:
    """Share (by letters) of judged tokens that are words. 1.0 when nothing to judge."""
    toks = _judged(text)
    if not toks:
        return 1.0
    good = sum(len(t) for t in toks if _is_word(t))
    total = sum(len(t) for t in toks)
    return good / total if total else 1.0


def is_unreadable(text: str, *, threshold: float = 0.55, min_tokens: int = 4) -> bool:
    """True when the lowercase words of the text are mostly not words.

    Needs at least ``min_tokens`` judged tokens: a short line is never called
    debris on the strength of one odd token.
    """
    toks = _judged(text)
    # One very long lowercase non-word ("tonmnuinunvionetenatucnnihnapusstsnse")
    # is debris on its own: no language has an 18-letter word that is not one.
    if any(len(t.strip("'’-")) >= 18 and not _is_word(t) for t in toks):
        return True
    if len(toks) < min_tokens:
        return False
    return readability(text) < threshold


__all__ = ["readability", "is_unreadable"]
