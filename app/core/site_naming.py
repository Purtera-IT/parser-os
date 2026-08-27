"""Give every site row a name a PM can read.

A site row earns a ``facility_name`` only when a ``physical_site`` atom
anchors it — the roster extractor read it out of a table, so the name came
along with the row. Sites the deal only ever mentions in prose never get
that atom. They still become rows, because the entity emitters correctly
recognise them as sites, but the emitter slugifies the phrase to build the
key and throws the phrase away. What reaches the PM is the identifier::

    site:prudential_center_office        ← what the dossier shows
    "Prudential Center office"           ← what the document said

Both name the same place; only one of them is readable. This module
recovers the second from evidence the compile already holds, in two tiers:

  Tier 1 — the document's own words.
    Scan the atom text for a contiguous run of tokens whose slug is
    exactly the site's slug. ``site:wind_creek_atmore`` is named by the
    sentence that produced it: "Wind Creek Atmore is soliciting
    proposals...". This is a projection of the source, not a guess — the
    phrase is the one the emitter slugified.

  Tier 2 — an identity string the compile already established.
    Sites that arrive through alias collapse have no surface phrase of
    their own: ``site:azs_1`` is what survived merging
    ``site:maricopa_county_iron_mountain_data_centers_azs_1_scottsdale``.
    Look through the names, aliases and document cells for the shortest
    string whose slug contains the site's slug at a token boundary AND
    which one of the row's own aliases corroborates. For AZS-1 that is
    the workbook cell "Iron Mountain Data Centers AZS-1 - Scottsdale".

Recovering the document's words is not the same as producing a readable
name, so a third pass reads what the first two recovered — and what the
roster extractor anchored, via ``site_facility_head``:

  Readability — the same words, in the form a PM reads them.
    A document writes a number twice for precision ("Building one hundred
    (100)"); a name that inherits both forms is gibberish
    ("building eight hundred 800"), so the spelled half goes and the
    digits stay, because the digits are what is on the door. A name that
    is only a number ("900") gets the type word the source prints beside
    it ("Building: 900 |") — and if no evidence prints one, keeps the
    number and raises ``site_name_is_bare_identifier`` rather than
    inventing "Building". Two rows landing on one name inside a deal are
    split only by a token their own identity evidence supplies, or else
    all of them are flagged ``site_name_duplicate_within_deal``; they are
    never merged and never given a suffix we made up.

Never invents a name. A site the documents describe only by city and ZIP
("Atmore, AL 36502") has no facility name to recover, and stays unnamed —
which is correct, and is why tier 2 rejects candidates that merely echo
the key back with the underscores taken out. Every name that leaves this
module is a projection of the evidence: the readability pass only ever
deletes tokens, re-cases them, or copies one forward from the source or
the identifier, so ``unsupported_name_tokens`` comes back empty for
anything it produced.

Only ever fills blanks. A row that already carries a ``facility_name`` is
not read, not scored, and not touched; a deal where every site is anchored
does no work at all.

Pure functions, no I/O, no LLM.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Mapping, Sequence

# A token is a word, keeping the internal hyphen/apostrophe that
# ``AZS-1`` and ``Levy's`` depend on — the slugifier collapses those to
# ``_`` either way, so matching has to see them as one token to line up
# with the key.
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")

# A token that carries a word, as opposed to a bare number or ordinal.
_WORD_RE = re.compile(r"[A-Za-z]{2,}")

# Longest site slug we will look for as a phrase. Beyond this a "site
# name" is a sentence fragment, and scanning every window that long over
# every atom costs more than the name is worth.
_MAX_PHRASE_TOKENS = 8

# Bounds on a string that can serve as a display name. Shorter than this
# is an abbreviation, longer is a paragraph.
_MIN_NAME_LEN = 3
_MAX_NAME_LEN = 120

# How deep to walk a document's structured projection looking for cells.
_MAX_WALK_DEPTH = 12

# Where one clause ends and the next begins. Pipes and bullets separate the
# fields of a flattened table row; the sentence enders separate prose. A
# comma does not break a clause — "Prudential Center office in Boston, MA"
# is one phrase.
_CLAUSE_BREAK_RE = re.compile(
    r"[|;:\u2022\u00B7\n\r\t]+"      # field and list separators
    r"|(?<=[^\W\d_])[.!?](?=\s|$)"   # a sentence ender after a letter
    r"|\s[-\u2010-\u2015]\s"         # a spaced dash, which sets off an aside
)

# ── Number words ─────────────────────────────────────────────────────────
# A document that writes a quantity twice — "Building one hundred (100)" —
# is being careful, not naming two things. Recovered names inherit both
# forms and read as gibberish ("building eight hundred 800"), so one form
# has to go: the digits, because that is what is stencilled on the door.
_NUM_UNITS: dict[str, int] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_BIG_SCALES: dict[str, int] = {
    "thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000,
}
_NUM_JOINERS = frozenset({"and"})
_NUM_VOCAB = frozenset(_NUM_UNITS) | frozenset(_BIG_SCALES) | {"hundred"} | _NUM_JOINERS

# Hyphens ("twenty-one"), NBSP and plain space all join one spelled number.
_NUM_SEP = r"[\s\u00A0\u202F\u2010-\u2015-]"
_NUM_WORD_ALT = "|".join(sorted(_NUM_VOCAB, key=len, reverse=True))
_SPELLED_RUN = rf"(?<![A-Za-z])(?:{_NUM_WORD_ALT})(?:{_NUM_SEP}+(?:{_NUM_WORD_ALT}))*(?![A-Za-z])"
_DIGIT_RUN = r"\d{1,3}(?:,\d{3})+|\d+"
_GAP = r"[\s\u00A0\u202F]*"

# "one hundred (100)" / "one hundred [100]"
_WORDS_THEN_PAREN_RE = re.compile(
    # The closing bracket is optional: a name recovered as a token run
    # ("Building one hundred (100") ends on the digits, and the bracket it
    # opened is left dangling.
    rf"(?P<words>{_SPELLED_RUN}){_GAP}[(\[]{_GAP}(?P<digits>{_DIGIT_RUN})(?:{_GAP}[)\]])?",
    re.IGNORECASE,
)
# "100 (one hundred)"
_PAREN_THEN_WORDS_RE = re.compile(
    rf"(?P<digits>{_DIGIT_RUN}){_GAP}[(\[]{_GAP}(?P<words>{_SPELLED_RUN}){_GAP}[)\]]",
    re.IGNORECASE,
)
# "eight hundred 800" — bare adjacency, which is how a slugified identifier
# ("BUILDING-EIGHT-HUNDRED-800") reads once the hyphens come out.
_WORDS_THEN_DIGITS_RE = re.compile(
    rf"(?P<words>{_SPELLED_RUN}){_NUM_SEP}+(?P<digits>{_DIGIT_RUN})(?!\d)",
    re.IGNORECASE,
)
# "800 eight hundred"
_DIGITS_THEN_WORDS_RE = re.compile(
    rf"(?P<digits>{_DIGIT_RUN}){_NUM_SEP}+(?P<words>{_SPELLED_RUN})",
    re.IGNORECASE,
)

# ── Type words ───────────────────────────────────────────────────────────
# Place nouns a document puts in front of a bare number. These are the only
# words this module will ever pull forward to name a number, and it pulls
# them from the evidence — never from this list. The list decides *whether*
# a word in the source is a type word, not *which* word to use.
_TYPE_WORDS = frozenset({
    "annex", "arena", "barn", "bay", "berth", "block", "branch", "building",
    "bldg", "campus", "center", "centre", "clinic", "complex", "court",
    "depot", "dock", "facility", "floor", "garage", "gate", "hall",
    "hangar", "house", "lab", "laboratory", "library", "lot", "mall",
    "office", "pad", "park", "pavilion", "pier", "plant", "plaza", "room",
    "rm", "school", "shop", "silo", "site", "stadium", "store", "studio",
    "ste", "suite", "terminal", "tower", "unit", "warehouse", "wing",
    "yard",
})
_TYPE_WORD_ALT = "|".join(sorted(_TYPE_WORDS, key=len, reverse=True))

# What may sit between a type word and its number: a colon, a hash, a dash,
# a dot, whitespace. Anything else and the two are not a pair.
_TYPE_GAP = r"[\s\u00A0\u202F:#.,\u2010-\u2015-]{0,4}"

_ALPHA_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\u2019]*")

# Review flags this module raises. They are how an unnameable site stays
# visibly unnamed instead of quietly ugly.
SITE_NAME_BARE_IDENTIFIER = "site_name_is_bare_identifier"
SITE_NAME_DUPLICATE = "site_name_duplicate_within_deal"
SITE_NAME_NUMBER_COLLAPSED = "site_name_number_word_collapsed"
SITE_NAME_DISAMBIGUATED = "site_name_disambiguated"


def _slug(s: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s or "").lower()).strip("_")


def _casedness(s: str) -> int:
    """How much original casing a string preserves.

    ``Iron Mountain Data Centers AZS-1 - Scottsdale`` and ``iron mountain
    data centers azs 1 scottsdale`` slugify identically; the first is the
    one the document actually printed.
    """
    return sum(1 for ch in s if ch.isupper())


def _atom_text(atom: Any) -> str:
    for field in ("raw_text", "text"):
        v = getattr(atom, field, None)
        if isinstance(v, str) and v:
            return v
    if isinstance(atom, Mapping):
        for field in ("raw_text", "text"):
            v = atom.get(field)
            if isinstance(v, str) and v:
                return v
    return ""


def _atom_value(atom: Any) -> Mapping[str, Any]:
    for field in ("value", "structured"):
        v = getattr(atom, field, None)
        if isinstance(v, Mapping):
            return v
    if isinstance(atom, Mapping):
        for field in ("value", "structured"):
            v = atom.get(field)
            if isinstance(v, Mapping):
                return v
    return {}


def _is_name_shaped(s: Any) -> bool:
    return (
        isinstance(s, str)
        and _MIN_NAME_LEN <= len(s.strip()) <= _MAX_NAME_LEN
        and any(c.isalpha() for c in s)
    )


def _walk_strings(obj: Any, depth: int = 0) -> Iterator[str]:
    """Yield every name-shaped string in a nested projection."""
    if depth > _MAX_WALK_DEPTH:
        return
    if isinstance(obj, Mapping):
        for v in obj.values():
            yield from _walk_strings(v, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk_strings(v, depth + 1)
    elif _is_name_shaped(obj):
        yield obj.strip()


def _clause_spans(text: str) -> Iterator[tuple[int, int]]:
    """Yield the ``(start, end)`` of each clause in ``text``.

    A name is a phrase inside one clause. It is never a run that begins in
    one field of a pipe-delimited row and ends in the next, or that steps
    over a sentence break — "…nine hundred thirty eight (938) | Remaining:
    two thousand…" is two facts, and any "name" stitched across that bar
    is an artefact of the scan window, not something the document said.
    """
    last = 0
    for m in _CLAUSE_BREAK_RE.finditer(text):
        if m.start() > last:
            yield last, m.start()
        last = m.end()
    if last < len(text):
        yield last, len(text)


def _scan_phrases(text: str, wanted: set[str], max_tokens: int) -> Iterator[tuple[str, str]]:
    """Yield ``(slug, phrase)`` for every token run in ``text`` whose slug
    is one of ``wanted``.

    A run must carry at least two worded tokens. Anything less is the key
    spelled differently — "AZS-1" or "HC 238" for ``site:azs_1`` and
    ``site:hc_238`` — and handing that back as the display name would look
    like a fix while telling the PM nothing they could not already read
    off the identifier.

    Runs are bounded to a single clause: see :func:`_clause_spans`.
    """
    for lo, hi in _clause_spans(text):
        clause = text[lo:hi]
        tokens = list(_TOKEN_RE.finditer(clause))
        worded = [bool(_WORD_RE.search(t.group(0))) for t in tokens]
        for i in range(len(tokens)):
            # j is exclusive, and starts at i+2 so the shortest window is
            # two tokens wide.
            for j in range(i + 2, min(len(tokens), i + max_tokens) + 1):
                if sum(worded[i:j]) < 2:
                    continue
                phrase = clause[tokens[i].start():tokens[j - 1].end()]
                slug = _slug(phrase)
                if slug in wanted:
                    yield slug, phrase


def _better(candidate: str, current: str | None) -> bool:
    if current is None:
        return True
    return _casedness(candidate) > _casedness(current)


def _identity_pool(
    atoms: Iterable[Any], documents: Iterable[Mapping[str, Any]] | None
) -> dict[str, str]:
    """slug → the best-cased string the compile has for that slug.

    Draws on what the compile already established about identity: the
    name/address fields and alias lists of every atom, plus the cells of
    each document's structured projection — which is where a workbook's
    own spelling of a site survives when no atom captured it.
    """
    pool: dict[str, str] = {}

    def offer(s: Any) -> None:
        if not _is_name_shaped(s):
            return
        s = s.strip()
        slug = _slug(s)
        if slug and _better(s, pool.get(slug)):
            pool[slug] = s

    for atom in atoms:
        value = _atom_value(atom)
        for field in ("name", "facility_name", "street_address", "address"):
            offer(value.get(field))
        for field in ("names", "aliases", "alternative_names"):
            listed = value.get(field)
            if isinstance(listed, (list, tuple)):
                for item in listed:
                    offer(item)

    for doc in documents or ():
        structured = doc.get("structured") if isinstance(doc, Mapping) else None
        for s in _walk_strings(structured):
            offer(s)

    return pool


def _alias_slugs(row: Mapping[str, Any]) -> set[str]:
    out: set[str] = set()
    for alias in row.get("aliases") or ():
        if not isinstance(alias, str):
            continue
        body = alias[len("site:"):] if alias.startswith("site:") else alias
        slug = _slug(body)
        if slug:
            out.add(slug)
    return out


# ═══════════════════════════════════════════════════════════════════════
# Name quality — what the PM actually reads
# ═══════════════════════════════════════════════════════════════════════
#
# Recovering the document's words is only half the job. The words come
# back exactly as the document wrote them, and documents write things
# people do not read aloud: a number twice over for legal precision, a
# building keyed by nothing but its number, the same building named twice
# under two identifiers. Three deterministic passes, none of which may
# ever introduce a token the evidence does not already carry:
#
#   1. Collapse a number written twice ("eight hundred 800" → "800").
#   2. Give a bare number the type word the source puts next to it
#      ("Building: 900 |" → "Building 900"), or, if the source supplies
#      none, keep the identifier and flag it. Never manufacture one.
#   3. Flag names that collide inside one deal, disambiguating only from
#      identifier or locality evidence — never from a suffix we invent.


@dataclass(frozen=True)
class SiteNameDecision:
    """A display name and everything a reviewer needs to audit it."""

    name: str
    flags: tuple[str, ...] = ()
    changed: bool = False

    @property
    def is_named(self) -> bool:
        """False when the "name" is still nothing but an identifier."""
        return SITE_NAME_BARE_IDENTIFIER not in self.flags


def _nfkc(s: str) -> str:
    """Fold the dress: NBSP, fullwidth digits, curly quotes, ligatures.

    ``Building 100`` and ``Building 100`` are the same name; only one
    of them survives a naive space split.
    """
    return unicodedata.normalize("NFKC", s)


def _tidy(s: str) -> str:
    """Collapse whitespace and shed the separators a field split leaves."""
    s = _nfkc(str(s or "")).replace(" ", " ")
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"^[\s|:;,\-‐-―]+", "", s)
    s = re.sub(r"[\s|:;,\-‐-―]+$", "", s)
    # An empty bracket pair is what a collapse leaves behind.
    s = re.sub(r"[(\[]\s*[)\]]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _spelled_value(run: str) -> int | None:
    """The integer a spelled-out number run denotes, or None.

    ``"two thousand nine hundred eighty seven"`` → 2987. Returns None for
    a run that is not a well-formed number, so a false positive on the
    vocabulary ("one" in "one-off") can never trigger a collapse.
    """
    words = [w for w in re.split(_NUM_SEP + r"+", _nfkc(run).strip().lower()) if w]
    if not words:
        return None
    total = current = 0
    seen = False
    for word in words:
        if word in _NUM_JOINERS:
            continue
        if word in _NUM_UNITS:
            current += _NUM_UNITS[word]
        elif word == "hundred":
            current = (current or 1) * 100
        elif word in _BIG_SCALES:
            total += (current or 1) * _BIG_SCALES[word]
            current = 0
        else:
            return None
        seen = True
    return total + current if seen else None


def _digit_value(digits: str) -> int | None:
    try:
        return int(digits.replace(",", ""))
    except ValueError:
        return None


def collapse_number_word_duplication(text: str) -> str:
    """Drop the spelled-out half of a number the text writes twice.

    "Building one hundred (100) Complete" → "Building 100 Complete";
    "building eight hundred 800" → "building 800". The digits win because
    they are what is stencilled on the door and printed on the badge
    reader — the PM matches "Building 800", not "Building Eight Hundred".

    Only ever deletes. A number written once, in either form, is left
    exactly as the document wrote it, so this can never introduce a token
    the source did not carry.
    """
    if not text:
        return text

    def rewrite(match: re.Match[str]) -> str:
        spelled = _spelled_value(match.group("words"))
        digits = match.group("digits")
        if spelled is None or spelled != _digit_value(digits):
            return match.group(0)
        return digits

    out = text
    # Parenthesised forms first, so the brackets go with the words they
    # wrapped and "Building one hundred (100)" does not leave "Building
    # 100 ()" behind.
    for pattern in (
        _WORDS_THEN_PAREN_RE,
        _PAREN_THEN_WORDS_RE,
        _WORDS_THEN_DIGITS_RE,
        _DIGITS_THEN_WORDS_RE,
    ):
        out = pattern.sub(rewrite, out)
    return out


def _case_score(token: str) -> int:
    """How well a token's casing reads in a name.

    A header shouts ("BUILDING"); the body prints ("Building"). Short
    all-caps tokens are acronyms ("HQ", "MDF") and score as printed.
    """
    if not token:
        return 0
    if token.isupper() and len(token) > 4:
        return 0
    if token[:1].isupper():
        return 2
    return 1


def _recase_from_source(name: str, source: str) -> str:
    """Adopt the source's own casing for each word of ``name``.

    The identifier round-trip lowercases everything it touches, so
    ``BUILDING-100`` comes back as "building 100" while the document
    plainly printed "Building 100". Swapping in the document's spelling
    is a projection of the source, not a style rule — and it only ever
    changes case, never which words are present.
    """
    if not name or not source:
        return name
    variants: dict[str, str] = {}

    def offer(key: str, tok: str) -> None:
        best = variants.get(key)
        if best is None or _case_score(tok) > _case_score(best):
            variants[key] = tok

    for m in _ALPHA_TOKEN_RE.finditer(_nfkc(source)):
        tok = m.group(0)
        offer(tok.casefold(), tok)
        # "Location: All Buildings" prints the same word the name spells
        # in the singular. The casing is the document's either way.
        folded = tok.casefold()
        for suffix in ("es", "s"):
            if len(folded) > len(suffix) + 1 and folded.endswith(suffix):
                variants.setdefault(folded[: -len(suffix)], tok[: -len(suffix)])
                break

    def swap(m: re.Match[str]) -> str:
        tok = m.group(0)
        best = variants.get(tok.casefold())
        if best is None or best == tok:
            return tok
        return best if _case_score(best) > _case_score(tok) else tok

    return _ALPHA_TOKEN_RE.sub(swap, name)


def _is_bare_identifier(name: str) -> bool:
    """True when the "name" carries no word at all — "900", "12-A"."""
    return bool(name) and not _WORD_RE.search(name)


def _digit_core(name: str) -> str:
    """The digit run a bare identifier is built around."""
    digits = re.findall(r"\d+", name)
    return digits[0] if len(digits) == 1 else ""


def _type_word_for(digits: str, source: str) -> str:
    """The place noun the source prints immediately before ``digits``.

    "Building: 900 |" hands back "Building" — the document's own word, in
    the document's own casing. A source that never puts a type word next
    to the number hands back nothing, and the caller must not invent one.
    """
    if not digits or not source:
        return ""
    pattern = re.compile(
        rf"(?<![A-Za-z])(?P<word>{_TYPE_WORD_ALT})(?P<plural>e?s)?"
        rf"(?![A-Za-z]){_TYPE_GAP}[(\[]?\s*{re.escape(digits)}(?!\d)",
        re.IGNORECASE,
    )
    match = pattern.search(_nfkc(source))
    if not match:
        return ""
    word = match.group("word")
    # "Buildings 100, 200 and 600" names Building 100, not Buildings 100.
    if match.group("plural") and word.casefold() in _TYPE_WORDS:
        return word
    return word


def clean_site_name(
    name: Any,
    *,
    source_text: str = "",
    site_id: str = "",
) -> SiteNameDecision:
    """Turn one recovered name into one a PM can read.

    ``source_text`` is the atom's own text and ``site_id`` the identifier
    the compile established. Both are evidence; between them they are the
    *only* place a word in the returned name may come from. The function
    deletes, re-cases and copies forward — it never composes.

    A name that survives with no word in it at all (a bare "900" whose
    documents never once wrote a type word beside it) keeps the
    identifier and carries ``site_name_is_bare_identifier``, so the row
    reads as unnamed rather than as a name that happens to look like a
    number.
    """
    original = _tidy(name)
    if not original:
        return SiteNameDecision("", (), False)

    source = _nfkc(str(source_text or ""))
    flags: list[str] = []

    # 1 — a number written twice is one number.
    collapsed = _tidy(collapse_number_word_duplication(original))
    if collapsed and collapsed != original:
        flags.append(SITE_NAME_NUMBER_COLLAPSED)
    working = collapsed or original

    # 2 — the document's own spelling of the words we kept.
    working = _tidy(_recase_from_source(working, source)) or working

    # 3 — a bare number is not a name. Recover the type word the evidence
    # prints beside it, or say so.
    if _is_bare_identifier(working):
        digits = _digit_core(working)
        # The atom's text first: that is the document speaking. The
        # identifier second: it is what the compile established, and
        # "BUILDING-900" is still evidence, not a guess.
        type_word = _type_word_for(digits, source)
        if type_word and _case_score(type_word) == 0:
            # Lifted out of a shouted header. The word is the evidence's,
            # the SHOUTING is the header's.
            type_word = type_word[:1].upper() + type_word[1:].lower()
        if not type_word:
            type_word = _type_word_for(
                digits, re.sub(r"[_\-‐-―]+", " ", str(site_id or ""))
            )
            # An identifier is stored flattened ("BUILDING-900"); same
            # rule, and the flattening loses the case either way.
            if type_word and (type_word.isupper() or type_word.islower()):
                type_word = type_word[:1].upper() + type_word[1:].lower()
        if type_word:
            working = f"{type_word} {digits}"
            # A type word lifted out of an identifier still reads as the
            # document's word if the document printed it anywhere.
            working = _tidy(_recase_from_source(working, source)) or working
        else:
            flags.append(SITE_NAME_BARE_IDENTIFIER)

    working = _tidy(working) or original
    return SiteNameDecision(working, tuple(flags), working != original)


# ── Duplicate names inside one deal ──────────────────────────────────────
# Two rows that end up with one name are not automatically one site, and
# merging them would lose a row the roster asserted. Disambiguation may
# only draw on identity evidence — the site's own identifier, its city,
# an explicit qualifier — never on the prose the atom happens to sit in.
# A milestone line's words ("Wave", "September") distinguish nothing; they
# would read as a name and mean a sentence.

_GENERIC_QUALIFIERS = frozenset({
    "the", "and", "of", "at", "in", "for", "a", "an", "site", "sites",
    "location", "locations", "all", "new", "old", "main", "n", "s", "e",
    "w", "no", "nbr", "num", "number", "id",
}) | _TYPE_WORDS


def _identity_tokens(entry: Mapping[str, Any]) -> list[str]:
    """Tokens of the identity evidence a site may be qualified by.

    Its identifier, its city, its state, an explicit qualifier — the
    fields that say *which* site this is. Deliberately not the atom text.
    """
    out: list[str] = []
    for key in ("site_id", "city", "state", "qualifier", "code"):
        raw = entry.get(key)
        if not isinstance(raw, str) or not raw:
            continue
        spaced = re.sub(r"[_\-‐-―]+", " ", _nfkc(raw))
        found = [m.group(0) for m in _TOKEN_RE.finditer(spaced)]
        if key == "site_id" and len(found) < 2:
            # A one-token identifier ("s1") has no part to spare: handing
            # it back as a qualifier is the key echoed with a bracket
            # round it, which tier 2 already refuses as a name.
            continue
        out.extend(found)
    return out


def _qualifier_display(token: str, source: str) -> str:
    """The token as the document prints it, else conventionally cased."""
    recased = _recase_from_source(token, source)
    if recased != token:
        return recased
    if token.isalpha() and (token.islower() or token.isupper()) and len(token) > 3:
        return token[:1].upper() + token[1:].lower()
    return token


def resolve_site_names(
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, SiteNameDecision]:
    """Clean every site name in one compile, then settle the collisions.

    ``entries`` are mappings carrying ``site_id`` and ``name``, optionally
    ``source_text``, ``city``, ``state``, ``qualifier``, ``code``. Returns
    ``{site_id: SiteNameDecision}``.

    Order-invariant (the result does not depend on the order the rows
    arrive in) and idempotent (feeding the returned names back in changes
    nothing). Where two rows collide, either *every* member of the
    collision earns a distinguishing token that its own identity evidence
    supplies, or *none* does and all of them are flagged — a half-
    disambiguated group is worse than an honest one, because the member
    left bare looks like the canonical one.
    """
    decided: dict[str, SiteNameDecision] = {}
    evidence: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        site_id = str(entry.get("site_id") or "").strip()
        if not site_id or site_id in decided:
            continue
        decided[site_id] = clean_site_name(
            entry.get("name"),
            source_text=str(entry.get("source_text") or ""),
            site_id=site_id,
        )
        evidence[site_id] = entry

    groups: dict[str, list[str]] = {}
    for site_id, decision in decided.items():
        if not decision.name:
            continue
        groups.setdefault(decision.name.casefold(), []).append(site_id)

    for _key, members in groups.items():
        if len(members) < 2:
            continue
        members = sorted(members)  # order-invariance
        tokens = {
            sid: {t.casefold() for t in _identity_tokens(evidence[sid])}
            for sid in members
        }
        taken = {
            t for sid in members for t in
            {w.casefold() for w in _TOKEN_RE.findall(decided[sid].name)}
        }
        chosen: dict[str, str] = {}
        for sid in members:
            others: set[str] = set()
            for other in members:
                if other != sid:
                    others |= tokens[other]
            unique = sorted(
                t for t in tokens[sid]
                if t not in others
                and t not in taken
                and t not in _GENERIC_QUALIFIERS
                and len(t) > 1
            )
            if unique:
                # Shortest first, then alphabetical: deterministic, and a
                # short token reads as a qualifier rather than a sentence.
                chosen[sid] = min(unique, key=lambda t: (len(t), t))
        if len(chosen) == len(members):
            for sid in members:
                base = decided[sid]
                token = _qualifier_display(
                    chosen[sid], str(evidence[sid].get("source_text") or "")
                )
                decided[sid] = SiteNameDecision(
                    f"{base.name} ({token})",
                    tuple(dict.fromkeys((*base.flags, SITE_NAME_DISAMBIGUATED))),
                    True,
                )
        else:
            for sid in members:
                base = decided[sid]
                decided[sid] = SiteNameDecision(
                    base.name,
                    tuple(dict.fromkeys((*base.flags, SITE_NAME_DUPLICATE))),
                    base.changed,
                )
    return decided


def unsupported_name_tokens(name: str, *sources: str) -> set[str]:
    """Tokens of ``name`` that none of ``sources`` contains.

    The projection invariant, made checkable. A name this module produces
    must be a rearrangement of words the evidence already holds; anything
    this returns is a word we manufactured, which is the one thing the
    module must never do.
    """
    have: set[str] = set()
    for source in sources:
        spaced = re.sub(r"[_\-‐-―]+", " ", _nfkc(str(source or "")))
        have |= {m.group(0).casefold() for m in _TOKEN_RE.finditer(spaced)}
        # A hyphenated source token also supports its parts, since the
        # slugifier splits them: "AZS-1" supports "AZS" and "1".
        for part in re.findall(r"[A-Za-z0-9]+", spaced):
            have.add(part.casefold())
    return {
        m.group(0).casefold()
        for m in _TOKEN_RE.finditer(_nfkc(name))
        if m.group(0).casefold() not in have
    }


def _contains_at_boundary(haystack: str, needle: str) -> bool:
    return re.search(rf"(?:^|_){re.escape(needle)}(?:_|$)", haystack) is not None


def recover_site_display_names(
    *,
    sites: Sequence[Mapping[str, Any]],
    atoms: Sequence[Any],
    documents: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, str]:
    """Return ``{site_key: display_name}`` for site rows that have no name.

    ``sites`` are ``site_readiness`` rows. Rows that already carry a
    ``facility_name`` are skipped entirely — this only fills blanks, and
    a deal whose sites are all roster-anchored returns ``{}`` without
    reading a single atom.
    """
    targets: dict[str, Mapping[str, Any]] = {}
    for row in sites:
        if not isinstance(row, Mapping) or row.get("facility_name"):
            continue
        key = row.get("site") or row.get("site_key") or ""
        if not isinstance(key, str) or not key.startswith("site:"):
            continue
        slug = key[len("site:"):]
        if slug:
            targets[slug] = row
    if not targets:
        return {}

    resolved: dict[str, str] = {}
    # slug → the text the winning phrase came out of, so the quality pass
    # can check its work against the same evidence the phrase came from.
    context: dict[str, str] = {}

    # ── Tier 1: the phrase the emitter slugified, recovered from the
    # text it was slugified from.
    wanted = {s for s in targets if 2 <= len(s.split("_")) <= _MAX_PHRASE_TOKENS}
    if wanted:
        best: dict[str, str] = {}
        for atom in atoms:
            text = _atom_text(atom)
            if not text:
                continue
            for slug, phrase in _scan_phrases(text, wanted, _MAX_PHRASE_TOKENS):
                if _better(phrase, best.get(slug)):
                    best[slug] = phrase
                    context[slug] = text
        resolved.update(best)

    # ── Tier 2: for sites that arrived through alias collapse and so have
    # no surface phrase of their own, the established identity string that
    # one of the row's aliases corroborates.
    unresolved = {s: row for s, row in targets.items() if s not in resolved}
    if unresolved:
        pool = _identity_pool(atoms, documents)
        for slug, row in unresolved.items():
            aliases = _alias_slugs(row)
            candidates: list[tuple[int, str]] = []
            for pool_slug, phrase in pool.items():
                if pool_slug == slug:
                    # The key with its underscores taken out. Not a name.
                    continue
                if not _contains_at_boundary(pool_slug, slug):
                    continue
                corroborated = any(
                    pool_slug == a or pool_slug in a or a in pool_slug
                    for a in aliases
                )
                if not corroborated:
                    continue
                candidates.append((len(pool_slug), phrase))
            if candidates:
                # Shortest wins: the longer strings are this site fused
                # with its neighbours ("Maricopa County Iron Mountain Data
                # Centers Azs 1 Scottsdale" names two sites, not one).
                candidates.sort()
                resolved[slug] = candidates[0][1]
                context[slug] = candidates[0][1]

    # A recovered phrase is the document's words, which is not the same as
    # words a PM can read: "Building one hundred (100)" is verbatim and
    # still unreadable. Clean each one against the evidence it came from.
    # ``clean_site_name`` only ever deletes and re-cases, so a name that
    # was a projection of the source before this line still is after it.
    out: dict[str, str] = {}
    for slug, name in resolved.items():
        decision = clean_site_name(
            name, source_text=context.get(slug, name), site_id=slug,
        )
        if decision.name:
            out[f"site:{slug}"] = decision.name
    return out


__all__ = [
    "SITE_NAME_BARE_IDENTIFIER",
    "SITE_NAME_DISAMBIGUATED",
    "SITE_NAME_DUPLICATE",
    "SITE_NAME_NUMBER_COLLAPSED",
    "SiteNameDecision",
    "clean_site_name",
    "collapse_number_word_duplication",
    "recover_site_display_names",
    "resolve_site_names",
    "unsupported_name_tokens",
]
