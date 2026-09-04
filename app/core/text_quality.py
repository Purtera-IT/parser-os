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


#: The two-letter words worth believing. The full dictionary also lists "te",
#: "ae", "oe", "ao" (musical notes, Scots, digraphs), which is exactly what OCR
#: debris is made of, so two-letter tokens are only words from this closed set.
_SHORT_WORDS = frozenset({
    "a", "i", "am", "an", "as", "at", "be", "by", "do", "go", "he", "hi", "if", "in", "is",
    "it", "me", "my", "no", "of", "ok", "on", "or", "so", "to", "up", "us", "we", "pm",
    "id", "hr", "sr", "jr", "mr", "ms", "dr", "st", "rd", "ne", "nw", "se", "sw", "ip",
    "pc", "tv", "ac", "dc", "re", "vs", "ex", "et", "al", "ca", "co", "la", "de", "ft",
})


@lru_cache(maxsize=1)
def _words() -> frozenset[str]:
    try:
        with gzip.open(_WORDLIST, "rt", encoding="utf-8") as fh:
            return frozenset(w.strip() for w in fh if len(w.strip()) >= 3) | _SHORT_WORDS
    except Exception:
        return frozenset()


def _shape_ok(tok: str) -> bool:
    if not _VOWEL_RE.search(tok) or _CONSONANT_RUN_RE.search(tok):
        return False
    vowels = len(_VOWEL_RE.findall(tok))
    return not (len(tok) >= 6 and vowels / len(tok) > 0.7)


def _is_word(tok: str) -> bool:
    core = tok.strip("'’-").lower()
    words = _words()
    if len(core) < 3:
        return core in words if words else True
    if len(core) < 3:
        return True
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


_NAME_ALLOWANCE = 2  # capitalised non-words a line may carry as names before they count


def _plain(text: str) -> str:
    """Text with URLs, emails and product codes removed: they are not language."""
    return re.sub(r"\S*://\S*|\S+@\S+|www\.\S+|\S*\d\S*", " ", text or "")


def _judged(text: str) -> list[str]:
    """Tokens that can be debris: every lowercase alphabetic token of length
    >= 3, plus capitalised tokens beyond an allowance of two that are not
    words either (a line has a name or two; "Dut Su nee, es ae a Stoeger see
    ene Scoala ny Syne tae" has six). All-caps abbreviations and tokens with
    digits are never judged."""
    lower: list[str] = []
    caps_unknown: list[str] = []
    for t in _TOKEN_RE.findall(_plain(text)):
        core = t.strip("'’-")
        if len(core) < 3:
            continue
        if core[0].islower():
            lower.append(t)
        elif core.isupper():
            continue  # SOW, PSOW, CDW: abbreviation
        elif not _is_word(t):
            caps_unknown.append(t)
    # One or two capitalised non-words are names; three or more are debris and
    # every one of them is judged.
    extra = caps_unknown if len(caps_unknown) > _NAME_ALLOWANCE else []
    return lower + extra


def readability(text: str) -> float:
    """Share (by letters) of judged tokens that are words. 1.0 when nothing to judge."""
    toks = _judged(text)
    if not toks:
        return 1.0
    good = sum(len(t) for t in toks if _is_word(t))
    total = sum(len(t) for t in toks)
    return good / total if total else 1.0


def _compound_of_words(token: str) -> bool:
    """"Customer-Designated", "Buyer-formatted", "time-and-materials": a token
    joined by hyphens or apostrophes is as readable as its parts. Live 010300:
    the 19-letter "Customer-Designated" tripped the 18-letter-debris rule and
    the whole Services Fees clause was dropped as unreadable."""
    parts = [p for p in re.split(r"[-'’]", token) if p]
    return len(parts) >= 2 and all(len(p) <= 2 or _is_word(p) for p in parts)


def is_unreadable(text: str, *, threshold: float = 0.55, min_tokens: int = 4) -> bool:
    """True when the lowercase words of the text are mostly not words.

    Needs at least ``min_tokens`` judged tokens: a short line is never called
    debris on the strength of one odd token.
    """
    toks = _judged(text)
    # One very long lowercase non-word ("tonmnuinunvionetenatucnnihnapusstsnse")
    # is debris on its own: no language has an 18-letter word that is not one.
    if any(len(t.strip("'’-")) >= 18 and not _is_word(t) and not _compound_of_words(t) for t in toks):
        return True
    # Whole-line view: when fewer than a third of ALL alphabetic tokens are
    # words -- capitalised or not, abbreviations included -- the line is
    # debris whatever its case pattern ("IC Tes Pia a OE SPR Seep a france").
    all_toks = [t for t in _TOKEN_RE.findall(_plain(text)) if len(t.strip("'’-")) >= 2]
    if len(all_toks) >= 4:
        words = _words()

        def _known_strict(t: str) -> bool:
            core = t.strip("'’-")
            if core.isupper() and 2 <= len(core) <= 6:
                return True  # SOW, PSOW, CDW, TEAMS: abbreviations are language
            core = core.lower()
            if words:
                return core in words or (len(core) >= 3 and _is_word(t))
            return _shape_ok(core)

        # Evidence FOR the line: tokens of four or more letters that are words
        # (short dictionary oddities like "oe" or "pia" prove nothing).
        # Evidence AGAINST: tokens of three or more letters that are not.
        known = sum(1 for t in all_toks if len(t.strip("'’-")) >= 4 and _known_strict(t))
        unknown = sum(1 for t in all_toks if not _known_strict(t))
        if unknown >= 2 and known / len(all_toks) < 0.35:
            return True
    # A line whose capitalised tokens are mostly not words either ("IC Tes Pia
    # a OE SPR Seep a france") has fewer lowercase tokens to judge, but the
    # excess capitalised non-words are the judgement.
    caps_excess = sum(1 for t in toks if t.strip("'’-")[:1].isupper())
    if caps_excess and len(toks) >= 2:
        return readability(text) < threshold
    if len(toks) < min_tokens:
        return False
    return readability(text) < threshold


__all__ = ["readability", "is_unreadable"]
