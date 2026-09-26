"""Read the body text out of an email, keeping the tables intact.

Lived in two places — ``email_parser`` and ``segmenters`` — as byte-identical
copies, so a fix to one silently left the other reading mail the old way. One
copy now, imported by both.
"""
from __future__ import annotations

from app.core.textio import read_text

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
_INVITE_CONTINUES_RE = re.compile(
    r"^\s*(?:"
    r"(?:meeting\s+id|passcode|phone\s+conference\s+id|tenant\s+key|video\s+id"
    r"|conference\s+id|access\s+code|webinar\s+id|for\s+organizers)\s*:?"
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
    r"|\+?\d[\d\s().,-]{7,}\#?"                      # a dial-in number
    r"|[\d][\d\s]{5,}\#?"                            # a bare meeting id
    r"|[A-Za-z0-9]{6,12}"                            # a bare passcode token
    r"|[\w.+-]+@\w[\w.-]*\.\w+"                      # tenant key address
    # "Join: https://teams.microsoft.com/meet/..." -- the label may lead.
    r"|(?:join|link|url)?\s*:?\s*https?://\S*"
    r"(?:teams\.microsoft|zoom\.us|webex|meet\.google|gotomeet)\S*"
    r"|[_=-]{10,}"                                   # the rule Outlook draws
    r"|\|"
    r")\s*[<>|]*\s*$", re.I)

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
            if _INVITE_CONTINUES_RE.match(line) or _INVITE_OPENS_RE.match(line):
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
    if "<html" in content.lower() or "<table" in content.lower():
        soup = BeautifulSoup(content, "html.parser")
        _flatten_tables_in_place(soup)
        return strip_meeting_invite(
            rejoin_label_and_value(soup.get_text(separator="\n", strip=True)))
    return strip_meeting_invite(rejoin_label_and_value(content))


