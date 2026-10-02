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


#: A piece that ends on an ellipsis -- three or more dots, or "\u2026" -- with
#: no whitespace after it.
_GLUED_ELLIPSIS_END_RE = re.compile(r"(?:\.{3,}|\u2026)$")


def _rejoin_glued_ellipsis(pieces: list[str]) -> list[str]:
    """Undo a split at an ellipsis that runs straight into the next word.

    pysbd ends a sentence at "...." even when a letter follows with no space:
    live 000132's "NFL is back tonight....preseason but I'll take it." came
    back as "NFL is back tonight...." and "preseason but I'll take it.", two
    atoms for one trailing-off thought. With ``clean=False`` a piece keeps the
    whitespace that followed it, so a piece ending exactly on the dots was
    glued to the next word in the source; rejoin those. "He left... Then came
    back." keeps its boundary: there the piece ends in a space.
    """
    out: list[str] = []
    for piece in pieces:
        if out and _GLUED_ELLIPSIS_END_RE.search(out[-1]) and piece[:1].isalpha():
            out[-1] = out[-1] + piece
        else:
            out.append(piece)
    return out


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences, keeping abbreviations intact."""
    if not text or not text.strip():
        return []
    seg = _get_segmenter()
    if seg is None:  # pragma: no cover - dependency-free fallback
        return [s for s in _NAIVE_SPLIT.split(text) if s.strip()]
    try:
        return _rejoin_glued_ellipsis([s for s in seg.segment(text) if s and s.strip()])
    except Exception:  # pragma: no cover - never fail a parse over segmentation
        return [s for s in _NAIVE_SPLIT.split(text) if s.strip()]


def count_sentences(text: str) -> int:
    """Sentence count, used as the denominator in coverage ratios."""
    return len(split_sentences(text))


#: A list marker at the start of a line: a bullet glyph ("-", "*", "•", "◦",
#: the Word Symbol-font bullet U+F0B7, en/em dash), Word's "o" sub-bullet, or
#: an ordinal ("1.", "1)", "(a)", "a)"), always followed by whitespace and then
#: text. "-5", "-based" and "1.5" have no space after the sign, so they never
#: match; "o" counts only before a capital, a digit or a bracket.
_LIST_MARKER_RE = re.compile(
    r"^[^\S\n]*(?P<marker>[-*•◦▪●‣·–—]|o(?=[^\S\n]+[A-Z0-9(])|\(?\d{1,3}[.)]|\(?[A-Za-z]\))"
    r"[^\S\n]+(?=\S)"
)


def strip_list_marker(text: str) -> tuple[str, str]:
    """``(marker, item)`` for a list line, ``("", text)`` for anything else.

    Live 000132: one HubSpot note typed its scope one "- " bullet per line and
    another flattened the same list onto one " - " line. The flattened items
    came out bare and the per-line ones kept "- ", so the same bullet had two
    texts, two keys and two highlights. Every list path strips the marker the
    same way, here; the item text is what follows it on the line.
    """
    s = str(text or "")
    m = _LIST_MARKER_RE.match(s)
    if not m:
        return "", s
    return m.group("marker"), s[m.end():]


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


# ── sentence kinds: banter, talk, work ─────────────────────────────────────
#
# One email paragraph can hold a cheer, a dependency and a housekeeping note:
#
#     "Woohoo! Let's go Sarah! Famous words of D Khaled...Another one! We are
#      all good over here. Just need PO from you/customer and we can start
#      scheduling and getting the ball rolling on install. Also, just adding
#      the opportunity number on subject line for tracking purposes."
#
# As one atom the cheer buries the one fact (the PO gates scheduling). Split
# by KIND, consecutive sentences of the same kind staying together, so the
# cheer is one chatter atom and the dependency its own atom. A paragraph whose
# sentences are all one kind is never split here.

#: Anything that says something about the job, a decision or a request.
_WORK_CUE_RE = re.compile(
    r"[\d$?]|\b(?:need|needs|needed|require[sd]?|requirement|please|quote[sd]?|install\w*|"
    r"schedul\w*|po|p\.o\.|purchase order|order\w*|ship\w*|deliver\w*|send|sent|confirm\w*|"
    r"price|pricing|cost\w*|budget|invoice|contract|sow|scope|site|floor|room|suite|building|"
    r"cable|cabling|drop|drops|rack|door|camera|switch|license|licence|access|deadline|due|"
    r"start\w*|finish\w*|complete\w*|approv\w*|accept\w*|sign\w*|proceed|go ahead|go for it|"
    r"do it|option|decid\w*|cancel\w*|move forward|attach\w*|onsite|on-site|tech\w*|"
    r"labor|labour|hardware|equipment|material\w*|survey|walkthrough|walk-through|"
    r"waiting|wait|pending|carrier|transit|tracking|arriv\w*|backorder\w*|depend\w*|blocked)\b",
    re.I,
)
#: A first-person undertaking is never banter.
#: ("we'?ll" would read "well" as a promise, so the apostrophe is required.)
_PROMISE_RE = re.compile(
    r"\b(?:i|we)\s*(?:['’]ll|will|can|shall)\s+\w+|\b(?:i|we)['’]ll\b|"
    r"\b(?:i|we)\s+(?:am|are)\s+going\s+to\b|\blet me\s+\w+",
    re.I,
)
#: Social markers that make a sentence banter whatever its punctuation.
_BANTER_MARKER_RE = re.compile(
    r"\b(?:woo+hoo+|yay|hooray|wow|congrat\w*|let'?s go\b|all good|"
    r"hope (?:you|all|everyone|your|this)|happy (?:friday|monday|holidays?|new year|4th)|"
    r"have a (?:great|good|nice|wonderful)|enjoy (?:the|your)|cheers|haha+|lol|"
    r"good (?:morning|afternoon|evening)|how are you|how'?s it going|nice to meet|"
    r"great to (?:meet|hear|see)|pleasure)\b",
    re.I,
)


#: A holiday's name is not a figure. "Hope you had a great 4th of July!" read
#: as work because the "4" in "4th" is a digit, and the pleasantry came out of
#: the email as deal_metadata context instead of chatter. Only the holiday
#: phrase is removed before the work-cue test; any other digit still counts.
_HOLIDAY_RE = re.compile(
    r"\b(?:(?:the\s+)?(?:4th|fourth)(?:\s+of\s+july)?|july\s+(?:4th|4|fourth)|"
    r"new\s+year'?s?(?:\s+(?:day|eve))?|christmas|xmas|thanksgiving|easter|hanukkah|"
    r"memorial\s+day|labou?r\s+day|independence\s+day|veterans\s+day|"
    r"president'?s'?\s+day|mlk\s+day|juneteenth|halloween)\b",
    re.I,
)
_HOLIDAY_CONTEXT_RE = re.compile(
    r"\b(?:hope|happy|enjoy|have\s+a|had\s+a|great|good|nice|wonderful|merry|"
    r"weekend|holiday|break)\b",
    re.I,
)


#: ", once they are delivered we will ..." -- a trigger clause that carries
#: its own main clause, spliced onto a sentence with a comma.
_TRIGGER_SPLICE_RE = re.compile(
    r",\s+(?=(?:and\s+)?(?:once|as\s+soon\s+as|when|after)\s+"
    r"[^,.;!?]*?\b(?:we|i|they|you)(?:\s+(?:will|can|would|should|shall)|'ll)\b)",
    re.I,
)
_HEAD_CLAUSE_RE = re.compile(
    r"^(?:we|i|they|you|he|she|it|the\s+\w+)(?:'re|'m|'ve|\s+(?:are|am|is|were|was|have|has|had|"
    r"will|would|can|still|just|currently)\b)",
    re.I,
)


def split_trigger_clause(sentence: str) -> list[str]:
    """``[head, trigger]`` for a comma splice whose second half is a trigger
    with its own main clause, else ``[sentence]``.

    "We are waiting for the TVs to arrive, once they are delivered we will
    schedule the install." is a dependency AND the commitment it gates -- two
    statements (010003). "We will install once the TVs arrive" has no comma
    and no second subject, and stays whole.

    Each side is then split at its compound-clause boundaries
    (:func:`split_compound_clauses`), so the callers -- the email and note
    splitters -- get one piece per statement from one call.
    """
    t = str(sentence or "").strip()
    m = _TRIGGER_SPLICE_RE.search(t)
    if not m:
        return split_compound_clauses(t) if t else []
    head, tail = t[: m.start()].strip(), t[m.end():].strip()
    if len(head.split()) < 4 or len(tail.split()) < 4 or not _HEAD_CLAUSE_RE.match(head):
        return split_compound_clauses(t)
    return [*split_compound_clauses(head), *split_compound_clauses(tail[:1].upper() + tail[1:])]


#: Where one sentence joins two statements that each stand on their own:
#:
#:   ", so I ..."      -- a fact, then what the writer does about it
#:   ") so I ..."      -- the same, after a parenthetical aside, no comma
#:   " so I/we ..."    -- the same, no comma, first person only
#:   "; it ..."        -- two clauses a semicolon holds side by side
#:   " and I also ..." -- a fact, then a separate undertaking
#:
#: Each boundary requires a PRONOUN SUBJECT right after it, so the right-hand
#: side is a clause with its own subject. That is what keeps lists ("Rack A;
#: Rack B; the IDF"), addresses and "cats and dogs" whole, and "so that",
#: "so far" and "and so on" whole (no subject after the "so").
#:
#: Live 010003 wrote the TV sentence with NO comma: "... (it is with the
#: shipping carrier now) so I also need to keep my eye on the delivery
#: status." -- the comma-only boundary never fired on the real mail.
_COMPOUND_BOUNDARY_RE = re.compile(
    r",\s+(?=so\s+(?:i|we|you|they|he|she|it)\s+\w)"
    r"|(?<=\))\s+(?=so\s+(?:i|we|you|they|he|she|it)\s+\w)"
    r"|(?<![,)])\s+(?=so\s+(?:i|we)\s+\w)"
    r"|;\s+(?=(?:i|we|you|they|he|she|it)(?:['’]\w+)?\s+\w)"
    r"|,?\s+and\s+(?=(?:i|we)\s+also\s+\w)",
    re.I,
)
_COMPOUND_MIN_WORDS = 4
#: A word right before " so <subject>" that makes the "so" part of a phrase
#: rather than the start of a new clause.
_SO_NOT_A_BOUNDARY_RE = re.compile(r"\b(?:and|or|but|not|just|even|only|if)$", re.I)


def split_compound_clauses(sentence: str) -> list[str]:
    """One sentence holding several statements, as one piece per statement.

    Live 010003: "But we are waiting for tv to arrive at their office (it is
    with the shipping carrier now), so I also need to keep my eye on the
    delivery status." is the delivery fact AND the writer's own note that they
    are watching it -- two statements, and as one atom the dependency was
    typed by the remark. Split at ", so <subject>", "; <subject>" and
    " and I also"; each piece is the source text verbatim (the connector "so"
    stays on its clause), so every piece is still findable in the document.

    Conservative: a boundary inside parentheses never splits, and both sides
    must hold at least four words. Anything else returns ``[sentence]``.
    """
    t = str(sentence or "").strip()
    if not t:
        return []
    pieces: list[str] = []
    start = 0
    for m in _COMPOUND_BOUNDARY_RE.finditer(t):
        before = t[: m.start()]
        if before.count("(") != before.count(")"):
            continue  # inside a parenthetical aside
        if _SO_NOT_A_BOUNDARY_RE.search(before):
            continue  # "and so I", "not so we": the "so" is not a connector here
        head, tail = t[start: m.start()], t[m.end():]
        if len(head.split()) < _COMPOUND_MIN_WORDS or len(tail.split()) < _COMPOUND_MIN_WORDS:
            continue
        pieces.append(head.strip().rstrip(",;").strip())
        start = m.end()
    if not pieces:
        return [t]
    pieces.append(t[start:].strip())
    return pieces


#: A ", so <subject>" clause boundary inside one sentence.
_SO_CLAUSE_RE = re.compile(r",\s+so\s+(?=(?:i|we|you|they|it|he|she)\b)", re.I)


def sentence_kind(text: str) -> str:
    """``banter`` (a cheer, a pleasantry: no claim at all), ``talk``
    (relationship or pipeline talk, :func:`app.core.deal_chatter.is_chatter`)
    or ``work`` (anything else).

    Conservative: a digit, a question, a request word, a product word or a
    first-person promise makes a sentence ``work`` whatever it sounds like.
    """
    t = " ".join(str(text or "").split())
    if not t:
        return "work"
    # A greeting is judged as a greeting and the rest on its own (the same
    # rule as deal_chatter.is_chatter): "Hi Bob!" is banter, "Hi Bob, we need
    # 40 drops" is work.
    from app.core.greetings import starts_with_greeting, strip_leading_greeting

    if starts_with_greeting(t):
        rest = strip_leading_greeting(t)
        if not rest or not re.search(r"[A-Za-z0-9]", rest):
            return "banter"
        t = rest
    # The holiday is dropped only inside a pleasantry ("Hope you had a great
    # 4th of July!"); "install before the 4th of July" keeps its date.
    cue_text = _HOLIDAY_RE.sub(" ", t) if _HOLIDAY_CONTEXT_RE.search(t) else t
    if not _WORK_CUE_RE.search(cue_text) and not _PROMISE_RE.search(t):
        if _BANTER_MARKER_RE.search(t) or (t.endswith("!") and len(t.split()) <= 8):
            return "banter"
    try:
        from app.core.deal_chatter import is_chatter

        if is_chatter(t):
            return "talk"
    except Exception:  # pragma: no cover - never fail a parse over this
        pass
    return "work"


def split_by_kind(text: str, *, min_work_chars: int = 12) -> list[str]:
    """``text`` as runs of same-kind sentences, or ``[]`` (keep it whole).

    Splits only when the sentences are of more than one kind; consecutive
    sentences of one kind stay together. A ``work`` run shorter than
    ``min_work_chars`` is a fragment, so the paragraph stays whole. An
    ellipsis inside a sentence ("D Khaled...Another one!") is not a break.
    """
    s = str(text or "").strip()
    if not s:
        return []
    pieces: list[str] = []
    for sent in (p.strip() for p in split_sentences(s) if p.strip()):
        # "<a fact>, so I also need to <process talk>" is two statements in
        # one sentence; split at the ", so" boundary only when its two sides
        # are of different kinds, so a plain "..., so we need 40 drops" stays.
        m = _SO_CLAUSE_RE.search(sent)
        if m:
            head, tail = sent[:m.start()].strip() + ",", sent[m.start() + 1:].strip()
            if len(head) >= min_work_chars and sentence_kind(head) != sentence_kind(tail):
                pieces.extend([head, tail])
                continue
        pieces.append(sent)
    if len(pieces) < 2:
        return []
    kinds = [sentence_kind(p) for p in pieces]
    if len(set(kinds)) < 2:
        return []
    runs: list[tuple[str, list[str]]] = []
    for piece, kind in zip(pieces, kinds):
        if runs and runs[-1][0] == kind:
            runs[-1][1].append(piece)
        else:
            runs.append((kind, [piece]))
    out = [" ".join(ps) for _, ps in runs]
    if any(k == "work" and len(t) < min_work_chars for (k, _), t in zip(runs, out)):
        return []
    return out
