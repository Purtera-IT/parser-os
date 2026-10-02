"""Read the body text out of an email, keeping the tables intact.

Lived in two places — ``email_parser`` and ``segmenters`` — as byte-identical
copies, so a fix to one silently left the other reading mail the old way. One
copy now, imported by both.
"""
from __future__ import annotations

from app.core.textio import decode_html_entities, read_text

import re
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup


#: Separates cells of one table row in the flattened text stream. Chosen
#: because it survives normalize_text, does not occur in ordinary prose, and
#: reads as a row to both the downstream splitters and a human in source replay.
_CELL_SEP = " | "


#: A cell that is only a number — the signal that a column carries data rather
#: than layout. Allows thousands separators, decimals, currency and percent.
_NUMERIC_CELL_RE = re.compile(r"^[$€£]?\s*[\d,]+(?:\.\d+)?\s*%?$")

#: Contact-block vocabulary. Mail clients lay signatures out in <table>, so the
#: shape alone cannot tell a signature from a price list — the words can.
_CONTACT_CELL_RE = re.compile(
    r"(?:\+?\d[\d\s().\-]{7,}\d)"                     # a phone number
    r"|[\w.+-]+@[\w-]+\.[\w.]+"                        # an address
    r"|\bwww\.|https?://"                              # a link
    r"|\b(?:mobile|office|direct|cell|tel|fax|linkedin|twitter|facebook"
    r"|instagram|youtube|hq|headquarters|suite|ste\.)\b"
    r"|\b(?:manager|director|engineer|executive|president|founder|cto|ceo"
    r"|coo|cfo|vp|specialist|consultant|architect|coordinator|analyst"
    r"|representative|associate|principal|partner|owner)\b",
    re.IGNORECASE,
)

#: Legal / marketing boilerplate that trails a message.
_BOILERPLATE_RE = re.compile(
    r"\b(?:unsubscribe|confidentiality notice|this e-?mail (?:and any|is intended)"
    r"|intended solely for|privileged and confidential|if you are not the intended"
    r"|delete this message|views expressed|do not reply to this"
    r"|update your preferences|manage your preferences|sales communications)\b",
    re.IGNORECASE,
)


def _looks_like_layout_table(rows: list[list[str]]) -> bool:
    """True when a ``<table>`` is holding a signature or disclaimer, not data.

    Outlook and every marketing platform lay contact blocks out in tables, so
    reading tables faithfully means reading signatures faithfully too. Left
    alone this is the dominant output: across 45 deals, 61% of the rows
    recovered from email were contact blocks and only 2% carried a numeric
    cell, and because a signature repeats on every message in a thread just
    22% of them were even distinct.

    A data table earns its rows by having a column of numbers — counts, prices,
    quantities. A signature never does; what it has is phone numbers,
    addresses, job titles and links. So: no numeric cell anywhere and contact
    vocabulary in a meaningful share of cells means layout, not data. Any
    boilerplate phrase settles it outright.
    """
    cells = [c for row in rows for c in row if c]
    if not cells:
        return True
    if any(_BOILERPLATE_RE.search(c) for c in cells):
        return True
    # A numeric cell is the tell for real data. One is enough — a price list
    # with a single total still deserves its rows.
    if any(_NUMERIC_CELL_RE.match(c) for c in cells):
        return False
    contact = sum(1 for c in cells if _CONTACT_CELL_RE.search(c))
    return contact * 3 >= len(cells)


#: Tags that mean "this text continues", not "new line". `get_text` puts a
#: separator at EVERY element boundary, so each of these splits a sentence
#: wherever a mail client wrapped a word in markup.
_INLINE_TAGS = (
    "a", "b", "strong", "i", "em", "u", "sup", "sub", "small", "span",
    "font", "abbr", "code", "mark", "s", "strike", "big", "tt", "label",
)


def _unwrap_inline_in_place(soup: BeautifulSoup) -> int:
    """Dissolve inline markup so a sentence stays one line.

    `soup.get_text(separator="\\n")` inserts a newline at every element
    boundary, which is right for `<p>` and `<tr>` and wrong for everything a
    mail client uses to style part of a sentence. Measured on deal 010180's
    call notes, one message alone:

        <b>Layout & Technical Requirements</b>: The team discussed...
            -> "Layout & Technical Requirements" + ": The team discussed..."
               a heading asserting nothing, and a sentence with no subject

        Suite 1316, 3<sup>rd</sup> Floor
            -> "Suite 1316, 3" + "rd" + "Floor"

        <a href="mailto:...">LS Srinivas</a> &lt;ls.srinivas@flextrade.com&gt;
            -> "LS Srinivas <" + "ls.srinivas@flextrade.com" + ">"

    41 atoms across 26 deals start on a bare colon from the first shape alone,
    and the halves do not merely read badly -- they do not survive. Two of
    010180's four never reached the envelope: the entire Security & Access
    Control discussion, and the sentence carrying "two Ethernet connections per
    workstation", which is the multiplier its $110,108 is built on.

    Unwrapping keeps the text and drops the tag, so the separator has nothing
    to break on. Block-level tags are untouched.
    """
    unwrapped = 0
    for name in _INLINE_TAGS:
        for tag in soup.find_all(name):
            tag.unwrap()
            unwrapped += 1
    # `unwrap` alone changes nothing: it drops the tag and leaves the text as
    # separate strings, and `get_text(separator=...)` joins EVERY string, not
    # every element. `smooth` merges the adjacent strings back into one, which
    # is the half that actually repairs the sentence.
    soup.smooth()
    _collapse_source_newlines_in_place(soup)
    return unwrapped


#: Where a newline is real and must survive: preformatted text.
_PREFORMATTED = ("pre", "textarea", "plaintext")


def _collapse_source_newlines_in_place(soup: BeautifulSoup) -> None:
    """A newline INSIDE a text node is whitespace, not a line break.

    Unwrapping the inline tags is only half the repair, because a mail client
    also breaks its source markup wherever it likes. Outlook sent 010180 this:

        Are there any specific days the week of the 5
        <sup>th</sup>&nbsp;they want to do the walkthrough?

    The `<sup>` is unwrapped and smoothed back into one string -- and that
    string still contains the newline the markup had before the tag. The
    separator then splits on it anyway, and the deal got

        "Are there any specific days the week of the 5"   (scope_item)
        "th they want to do the walkthrough?"             (open_question)

    Two fragments, neither of which asks anything, from a question that is
    still open: nobody has said which days the customer wants.

    HTML has always said this newline is whitespace -- it is only a line break
    in `<pre>`. Collapsing it here means block tags remain the only thing that
    starts a new line, which is what the separator was meant to express.
    """
    for node in list(soup.find_all(string=True)):
        if "\n" not in node and "\r" not in node:
            continue
        if any(p.name in _PREFORMATTED for p in node.parents if p.name):
            continue
        collapsed = re.sub(r"\s+", " ", str(node))
        if collapsed != str(node):
            node.replace_with(collapsed)
    _rejoin_ordinals_in_place(soup)


#: "5" and "th" that a `<sup>` split, once the tag is gone. Anchored on a digit
#: so only an ordinal suffix is ever joined.
_SPLIT_ORDINAL = re.compile(r"(?<=\d)\s+(st|nd|rd|th)\b", re.I)


def _rejoin_ordinals_in_place(soup: BeautifulSoup) -> None:
    """Put the "th" back on the "5".

    Collapsing the source newline keeps the sentence whole, which is the part
    that matters. It still leaves the space that stood before the `<sup>`:

        the week of the 5 th they want to do the walkthrough?

    The author wrote "5th" -- the message's own text/plain alternative says so
    -- and a PM reading "the 5 th" sees a parse artifact and trusts the atom
    less. Only ever joins a suffix that follows a digit.
    """
    for node in list(soup.find_all(string=True)):
        text = str(node)
        if not _SPLIT_ORDINAL.search(text):
            continue
        if any(p.name in _PREFORMATTED for p in node.parents if p.name):
            continue
        node.replace_with(_SPLIT_ORDINAL.sub(r"\1", text))


def _flatten_tables_in_place(soup: BeautifulSoup) -> int:
    """Rewrite each data ``<table>`` as one line per ``<tr>``, cells joined by ``|``.

    ``soup.get_text(separator="\\n")`` emits every cell on its own line, which
    silently destroys the only thing a table means: which cell belongs to which
    row. A site-count table arrives as ``Arkansas``, ``327``, ``Idaho``, ``42``
    with nothing tying a state to its number, and no later stage can rebuild the
    pairing — ``table_rollup`` runs on the flattened text and is a no-op.

    Rewriting the element before the text pass keeps the row intact
    (``Arkansas | 327 | Arkansas | 312``) for every consumer downstream.

    Layout tables are left alone rather than dropped: their text still reaches
    the parser exactly as it did before any of this, one cell per line. Nothing
    is lost, it just stops being asserted as a row.

    Returns the number of tables rewritten.
    """
    count = 0
    for table in soup.find_all("table"):
        rows: list[list[str]] = []
        for row in table.find_all("tr"):
            cells = [
                c.get_text(separator=" ", strip=True)
                for c in row.find_all(["th", "td"])
            ]
            # Outlook pads tables with empty spacer cells; dropping them keeps
            # the row readable without shifting the columns that carry data.
            rows.append([c for c in cells if c])
        if not any(rows):
            continue
        if _looks_like_layout_table(rows):
            continue
        lines = [_CELL_SEP.join(r) for r in rows if r]
        if lines:
            table.replace_with("\n" + "\n".join(lines) + "\n")
            count += 1
    return count


def _is_body_part(part: Any) -> bool:
    """True for a part that is message body, not an attached file.

    An attached .html file or a forwarded .eml would otherwise be mistaken for
    the body and silently replace it.
    """
    if part.get_filename():
        return False
    disposition = (part.get("Content-Disposition") or "").strip().lower()
    return not disposition.startswith("attachment")


def _body_parts_by_type(msg: Any, content_type: str) -> list[Any]:
    """Every inline body part of ``content_type``, in document order.

    Deliberately walks the tree instead of using ``msg.get_body()``.
    ``get_body`` is RFC-correct and that is exactly the problem here: inside a
    ``multipart/related`` it returns the *root* part — the first one, or
    whichever ``start`` names — because in a related set the siblings are
    resources belonging to the root, not alternatives to it. So when a message
    is shaped

        multipart/mixed
          multipart/related
            text/plain      <- root, and all get_body can ever return
            text/html       <- a sibling, so preferencelist=("html",) is blind
            image/png ...

    asking for the HTML alternative returns None even though the HTML is
    sitting right there. The correct shape nests a ``multipart/alternative``
    around the two text parts; senders that skip it are technically malformed,
    but they are the overwhelming majority of what actually arrives, so the
    reader has to cope rather than take the structure at its word.
    """
    return [
        p for p in msg.walk()
        if p.get_content_type() == content_type
        and not p.is_multipart()
        and _is_body_part(p)
    ]


#: A line that is nothing but a URL, and a short line that introduces one.
#: Outlook writes "Diagram:" and the link as separate paragraphs, sometimes
#: with several empty ones between; the label is then a label with no value and
#: the URL is a URL with no claim, and BOTH are correctly discarded as not
#: statements. On deal 010288 that is how the only pointer to the job's wiring
#: diagram vanished out of the email that delivered it -- it survived solely
#: because somebody pasted the same link into a note six days later, on one
#: line, where it parsed.
#: A line holding one link and nothing else. Outlook's plain-text part
#: writes a link as the display URL followed by its real target in angle
#: brackets -- "https://huzzard.com/a.png<https://urldefense.com/v3/__...>"
#: -- which is still a line that is only a link. The display URL is the
#: one kept; the wrapper is peeled downstream anyway.
_ONE_URL = r"<?(https?://[^\s<>\"']+)>?"
_BARE_URL_LINE_RE = re.compile(
    r"^\s*" + _ONE_URL + r"\s*(?:<\s*https?://[^\s<>\"']+\s*>)?\s*$", re.I)
_LABEL_LINE_RE = re.compile(r"^\s*([A-Za-z][\w \-/&()]{0,38}):\s*$")
#: How far a value may sit from its label. Outlook put five empty paragraphs
#: between the two on 010288; beyond a short gap they are two separate things.
_MAX_BLANK_GAP = 8

# ---------------------------------------------------------------------------
# The meeting-invite block
#
# Teams, Zoom and Webex append a fixed block of join details to a message, and
# nothing in it is about the job. Left in, it parses: measured across 461 dev
# envelopes, 87 atoms on 38 deals came out of one, and the types are the
# problem rather than the count --
#
#   42  scope_item   "Dial in by phone", "Reset dial-in PIN", "Meeting options"
#   22  raw_utterance
#   14  deal_metadata
#    7  constraint   "Find a local number"
#    2  site_access_restriction
#
# So a cabling job at 7 Penn Plaza had a Teams passcode filed as a restriction
# on getting into the site, six deals were told to "Dial in by phone" as scope,
# and "Need help?" became an open_question -- Orbit asking the PM the invite's
# own rhetorical question.
#
# It is also the one place a message carries live credentials. A bridge
# passcode and a dial-in PIN are not facts about a deal and have no business in
# an artifact, a brief or a training corpus.
# ---------------------------------------------------------------------------

#: A line that only a join block says. `Passcode:` is deliberately NOT here --
#: on its own it could be a door code, which is a real site fact. It counts
#: only once a platform marker has opened a block.
_INVITE_OPENS_RE = re.compile(
    r"^\s*(?:"
    r"microsoft\s+teams\s+(?:meeting|need\s+help)"
    r"|join\s+(?:zoom\s+meeting|microsoft\s+teams\s+meeting"
    r"|on\s+a\s+video\s+conferencing\s+device|the\s+meeting\s+now)"
    r"|dial\s+in\s+by\s+phone"
    r"|________+\s*microsoft\s+teams"
    r")\s*[<>|]*\s*$", re.I)

#: Lines that continue a block once one is open.
#: One thing a join block says. Matched against each `|`-separated SEGMENT of
#: a line, not the whole line, because the block is laid out as a table: after
#: inline markup is unwrapped the row flattener joins its cells, so "Need help?"
#: and "System reference" arrive as one line and a label arrives with its value
#: ("Passcode: jz7o5CE9") rather than on a line of its own. The first version
#: of this anchored on the whole line and stopped matching the moment the text
#: extraction improved -- which put a bridge passcode back into deal 010180.
_INVITE_SEGMENT_RE = re.compile(
    r"^\s*(?:"
    r"(?:meeting\s+id|passcode|phone\s+conference\s+id|tenant\s+key|video\s+id"
    r"|conference\s+id|access\s+code|webinar\s+id|for\s+organizers|where|when"
    r"|join|dial-?in)\s*:.*"
    r"|need\s+help\s*\??"
    r"|find\s+a\s+local\s+number"
    r"|reset\s+dial-?in\s+pin"
    r"|meeting\s+options"
    r"|system\s+reference"
    r"|more\s+info"
    r"|one\s+tap\s+mobile"
    r"|join\s+on\s+a\s+video\s+conferencing\s+device"
    r"|dial\s+in\s+by\s+phone"
    r"|microsoft\s+teams\s+meeting"
    r"|united\s+states(?:,\s*\w[\w .'-]*)?"
    # A dial-in number, optionally followed by where it rings: after the
    # unwrap "+1 847-371-3000,,25104158#" and "United States, Libertyville"
    # arrive on one line.
    r"|\+?\d[\d\s().,-]{7,}\#?(?:\s+[A-Z][\w .'-]*(?:,\s*[\w .'-]+)?)?"
    r"|[\d][\d\s]{5,}\#?"
    r"|[A-Za-z0-9]{6,12}"
    r"|[\w.+-]+@\w[\w.-]*\.\w+"
    r"|https?://\S*(?:teams\.microsoft|zoom\.us|webex|meet\.google|gotomeet)\S*"
    r"|[_=-]{10,}"
    r")\s*[<>]?\s*$", re.I)


def _is_invite_line(line: str) -> bool:
    """True when every cell on the line is a join detail."""
    parts = [p.strip() for p in line.split("|")]
    parts = [p for p in parts if p]
    if not parts:
        # A line of nothing but separators is layout the block drew, not
        # content -- it must not end the run.
        return bool(line.strip())
    return all(_INVITE_SEGMENT_RE.match(p) for p in parts)


#: A block must say at least this many invite-only things before we believe it.
#: One mention in prose ("I'll send a Teams meeting") is not a block.
_MIN_INVITE_LINES = 3


def strip_meeting_invite(text: str) -> str:
    """Remove Teams/Zoom/Webex join blocks, keeping everything a person wrote.

    A block opens on a platform marker and runs while the lines keep looking
    like join details. The first line of real prose closes it, so an agenda or
    a question written under the invite survives.
    """
    lines = text.split("\n")
    drop: set[int] = set()
    i = 0
    while i < len(lines):
        if not _INVITE_OPENS_RE.match(lines[i]):
            i += 1
            continue
        j = i
        hits = 0
        while j < len(lines):
            line = lines[j]
            if not line.strip():
                j += 1
                continue
            if _is_invite_line(line) or _INVITE_OPENS_RE.match(line):
                hits += 1
                j += 1
                continue
            break
        if hits >= _MIN_INVITE_LINES:
            drop.update(range(i, j))
            # The rule Outlook draws immediately above the block belongs to it.
            k = i - 1
            while k >= 0 and not lines[k].strip():
                k -= 1
            if k >= 0 and re.fullmatch(r"\s*[_=-]{10,}\s*", lines[k]):
                drop.add(k)
        i = max(j, i + 1)
    return "\n".join(line for n, line in enumerate(lines) if n not in drop)


def rejoin_label_and_value(text: str) -> str:
    """Pull a lone URL back up onto the label line that introduces it.

    Only fires on the exact shape that loses information: a SHORT line ending
    in a colon and nothing else, then blank lines, then a line holding one URL
    and nothing else. Prose is untouched -- a sentence that happens to end in a
    colon is longer than the label pattern allows, and a URL with any words
    beside it is already a statement and is left alone.
    """
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        label = _LABEL_LINE_RE.match(lines[i])
        if label:
            j = i + 1
            while j < len(lines) and not lines[j].strip() and j - i <= _MAX_BLANK_GAP:
                j += 1
            url = _BARE_URL_LINE_RE.match(lines[j]) if j < len(lines) else None
            if url:
                out.append(f"{label.group(1)}: {url.group(1)}")
                i = j + 1
                continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


#: A heading that lost its colon to the markup. Short, no terminal punctuation,
#: and not already a sentence -- the shape of a `<b>` run, not of prose.
_BARE_HEADING_RE = re.compile(r"^\s*([A-Z][\w &/,'()-]{2,60})\s*$")
#: ...and the body that follows it, which the separator left starting on a colon.
_ORPHANED_BODY_RE = re.compile(r"^\s*:\s*(\S.*)$")


def rejoin_split_heading(text: str) -> str:
    """Put `<b>Heading</b>: body` back together.

    BeautifulSoup's ``get_text(separator="\\n")`` inserts a newline between an
    element and the text after it, so a bold heading and the sentence it
    introduces arrive as two lines:

        Layout & Technical Requirements
        : The team discussed the office layout, including 106 workstations...

    Both halves are then worthless. The heading is a noun phrase that asserts
    nothing and gets typed `scope_item` -- so "Security & Access Control"
    becomes work the job includes -- and the body is a sentence with no
    subject, starting on a bare colon.

    Worse, the body is then an orphan, and on deal 010180 two of the four did
    not survive the run to the envelope at all: the whole Security & Access
    Control discussion, and the sentence carrying "Cat 6A cabling, two Ethernet
    connections per workstation", which is the multiplier the deal's $110,108
    is built on. Rejoined, each is one statement under its own heading, and the
    heading becomes the lead_in it always was.

    This is the mirror of `rejoin_label_and_value`, which handles `Label:` on
    one line and its value on the next. Same defect, opposite side of the
    colon.
    """
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        heading = _BARE_HEADING_RE.match(lines[i])
        if heading:
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            body = _ORPHANED_BODY_RE.match(lines[j]) if j < len(lines) else None
            if body:
                out.append(f"{heading.group(1).strip()}: {body.group(1).strip()}")
                i = j + 1
                continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


#: A call-notes heading: a short Title-Case noun phrase, no verb, the kind a
#: notetaker writes over a paragraph. "Layout & Technical Requirements",
#: "Sequence & Coordination", "Security & Access Control".
_NOTES_HEADING = re.compile(
    r"^(?P<head>[A-Z][A-Za-z]*(?:[ ](?:&|and|of|/|[A-Z][A-Za-z]*)){1,5})"
    r":[ 	]+(?P<body>\S.*)$")
#: The offset into a call recording a notetaker leaves on the end of an entry
#: ("…for conference rooms. 2:01"). It is not a time the job cares about,
#: and left on it becomes part of the last sentence's atom.
_CALL_OFFSET = re.compile(r"[\s ]*\d{1,2}:\d{2}[\s ]*$")
#: The same offset anywhere in the text, for counting them.
_CALL_OFFSET_ANY = re.compile(r"[\s\u00a0]\d{1,2}:\d{2}[\s\u00a0]*$", re.M)
#: End of sentence: a full stop, ? or !, followed by space and a capital.
_SENTENCE_END = re.compile(r"(?<=[.!?])[\s ]+(?=[A-Z0-9])")


#: A heading's body is PROSE: it ends like a sentence and it is long enough to
#: be one. Six words, because "CAD drawings and plans were shared for review."
#: is eight and nothing shorter has turned up as a notes sentence.
_MIN_PROSE_WORDS = 6


def _is_prose(body: str) -> bool:
    """Is what follows the colon a paragraph, or a value?"""
    text = (body or "").strip()
    if not text.endswith((".", "!", "?")):
        return False
    return len(text.split()) >= _MIN_PROSE_WORDS


def reads_as_call_notes(text: str) -> bool:
    """Does this body carry recording offsets -- is it notes from a call?

    The offsets themselves are not facts about the job: "2:01" is a position in
    a recording, and left on the end of a sentence it becomes part of that
    sentence's atom. But they are evidence ABOUT the document, and dropping
    them silently threw that away.

    A document that is somebody's notes from a call is weaker evidence than a
    signed SOW -- nobody agreed to it, it is one person's summary -- and it
    says a recording exists that the deal does not hold, which is the `chase`
    reading. Two or more offsets, because one stray time of day is a time of
    day.
    """
    return len(_CALL_OFFSET_ANY.findall(text or "")) >= 2


def split_notes_entries(text: str) -> str:
    """One heading, one paragraph, several facts -- give each fact its own line.

    A call-notes email arrives as one line per section:

        Notes below:
        Layout & Technical Requirements: The team discussed the office layout,
        including 106 workstations, conference rooms, phone rooms, IT room, and
        pantry. CAD drawings and plans were shared for review. The setup will
        require Cat 6A cabling, two Ethernet connections per workstation, and AV
        work for conference rooms. 2:01

    Left whole that is ONE atom, and it is three different facts of three
    different types: a room mix, a document exchange, and a cabling
    specification. Typed as any one of them it is wrong about the other two, and
    a head asked to learn from it learns that a room count and a Cat 6A spec are
    the same kind of thing. The heading is also swallowed into the text, so the
    thing that says WHICH section a fact belongs to is not available as context.

    So: the heading goes onto its own line, where the parser already reads a
    short `Label:` line as the intro its items are read with, and each sentence
    of the body becomes a line of its own. The recording offset on the end is
    dropped -- it times a moment in a call, not anything about the job.

    Conservative: only fires on a heading-shaped label with a multi-sentence
    body, so "Passcode: jz7o5CE9" and "Total workstation drops: 212." are left
    exactly as they are.
    """
    out: list[str] = []
    for line in (text or "").splitlines():
        match = _NOTES_HEADING.match(line.strip())
        if match is None:
            out.append(line)
            continue
        body = _CALL_OFFSET.sub("", match.group("body")).strip()
        if not _is_prose(body):
            # A label with a VALUE, not a heading over a paragraph:
            # "Meeting ID: 228 859 003 479 315", "Direct: (732) 982-0189",
            # "Total workstation drops: 212." Splitting those would tear a
            # label off its value, which is the defect this file already
            # exists to undo.
            out.append(line)
            continue
        sentences = [s.strip() for s in _SENTENCE_END.split(body) if s.strip()]
        if not sentences:
            out.append(line)
            continue
        # The heading goes on its own line even when the body is ONE sentence:
        # it is the pointer that says which section the fact belongs to, and a
        # fact with one sibling deserves it as much as a fact with three.
        out.append(f"{match.group('head')}:")
        out.extend(sentences)
    return "\n".join(out)


def _extract_email_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".eml":
        raw = path.read_bytes()
        msg = BytesParser(policy=policy.default).parsebytes(raw)
        # Prefer the HTML rendering when the sender put a table in it.
        #
        # Mail clients ship the same message twice. Outlook's plain-text
        # alternative has already flattened every table to one value per line
        # before the file reaches us, while the HTML still has <tr>/<td>. For a
        # message carrying a table the HTML is strictly the richer source.
        # Prose-only mail keeps taking the plain part, which is cleaner (no
        # style noise, no tracking markup).
        content = None
        html_parts = _body_parts_by_type(msg, "text/html")
        for part in html_parts:
            try:
                candidate = part.get_content()
            except Exception:
                continue
            if "<table" in candidate.lower():
                content = candidate
                break
        if content is None:
            plain_parts = _body_parts_by_type(msg, "text/plain")
            if plain_parts:
                content = plain_parts[0].get_content()
            elif html_parts:
                content = html_parts[0].get_content()
            else:
                content = raw.decode("utf-8", errors="ignore")
    else:
        content = read_text(path)
    # HubSpot hands email bodies over HTML-escaped, sometimes twice ("server
    # &amp;amp; virtualization", live 000132): decode to the text the author
    # wrote -- after BeautifulSoup's own single decode for markup, before it
    # for plain text -- so every line split and char offset downstream is
    # computed on what the viewer shows.
    if "<html" in content.lower() or "<table" in content.lower():
        soup = BeautifulSoup(content, "html.parser")
        # Inline markup first: a table cell's contents must be whole before the
        # table is flattened, or the cell is split and then joined by "|".
        _unwrap_inline_in_place(soup)
        _flatten_tables_in_place(soup)
        return split_notes_entries(strip_meeting_invite(rejoin_split_heading(
            rejoin_label_and_value(decode_html_entities(soup.get_text(separator="\n", strip=True))))))
    return split_notes_entries(strip_meeting_invite(rejoin_split_heading(
        rejoin_label_and_value(decode_html_entities(content)))))


