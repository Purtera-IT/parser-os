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
        return soup.get_text(separator="\n", strip=True)
    return content


