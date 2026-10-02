"""What did the parser NOT read?

A recall miss leaves nothing behind: a paragraph that produced no atom makes
no card, no flag, no row. The only way anyone found out was a PM reading the
source next to the output and saying "out of that whole paragraph you have one
atom" -- which is not a process, it is luck.

So every text artifact is diffed against its own atoms: each line of the
source is CLAIMED (some atom quotes it), SUPPRESSED (an atom existed and a
gate dropped it, reason attached) or UNREAD (nothing ever came of it). The
unclaimed lines ride in the envelope, the labeler shows them greyed under the
message they belong to, and a PM promotes one in a click instead of hunting.

Deliberately blind to nothing: mail headers, quote markers and signature
chrome are listed too, marked ``chrome``, so the PM can see the parser's
judgement rather than trust it.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

#: Text we can read back and diff.
TEXT_SUFFIXES = {".eml", ".txt", ".md", ".msg", ".html", ".htm"}
#: A PDF is read back line by line off its text layer. Its line breaks are a
#: rendering, so a line is claimed by the atoms that between them carry its
#: words, not by one atom that merely touches it (010003: "f. Complete
#: billing tasks" and the PO's "CDW PO's are not transferrable." were never
#: atoms and never showed as unread).
PDF_SUFFIXES = {".pdf"}
_MAX_PDF_PAGES = 200

_CHROME_RE = re.compile(
    r"^\s*(?:"
    r"(?:from|to|cc|bcc|sent|date|subject|importance|reply-to|message-id)\s*:|"
    r">+|_{5,}|-{5,}|={5,}|\*{3,}|"
    r"(?:cell|mobile|office|direct|tel|phone|p|o|m|f|fax|email|e)\s*:\s*\S|"
    r"\[cid:|\[Image|<https?://|https?://\S+$|"
    r"(?:thanks|thank you|regards|best|sincerely|cheers)[,!.]?\s*$|"
    r"(?:the )?contents of this (?:e-?mail|message) are intended|"
    r"this (?:e-?mail|message) (?:and any attachments |)(?:is|are|may be) (?:confidential|privileged)|"
    r"if you (?:are not|have received) (?:the|this)|"
    r"team inbox\s*:|"
    r"hubspot note(?: id)?\s*:|note_id=|"
    # a line that is ONLY an address, a domain, a wrapped link or a phone
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?:<[^>]*>)?\s*$|"
    r"[A-Za-z0-9.-]+\.(?:com|net|org|io|co)(?:<[^>]*>)?\s*$|"
    r"[+(]?\d[\d\s().-]{8,}\s*$"
    r")",
    re.I,
)

#: Legal / vendor footer boilerplate, matched ANYWHERE in the line: a mail
#: gateway or a vendor's signature appends the same paragraph to every
#: message ("CDW Trust Center", "This email ... intended solely for ...",
#: "Privacy Policy | Unsubscribe"). It is nobody's statement about the deal.
_FOOTER_RE = re.compile(
    r"\btrust center\b|\bprivacy (?:policy|statement|notice)\b|\bunsubscribe\b|"
    r"\ball rights reserved\b|\bterms (?:of|and) (?:use|conditions|service)\b|"
    r"\bconfidential(?:ity)? (?:notice|statement)\b|"
    r"\b(?:intended|addressed) (?:solely|only|exclusively) for\b|"
    r"\bmay contain (?:confidential|privileged|proprietary)\b|"
    r"\b(?:is|are) (?:strictly )?prohibited\b|"
    r"\bnotify the sender\b|\bdelete (?:this|the) (?:e-?mail|message)\b|"
    r"\bexternal (?:sender|email)\b.*\bcaution\b|\bcaution\b.*\bexternal (?:sender|email)\b|"
    r"\bthis (?:message|e-?mail) (?:is|was) (?:from|sent from) an external\b|"
    r"\bplease consider the environment\b|\bdo not reply to this\b|"
    r"(?:^|\s)(?:©|\(c\))\s*(?:19|20)\d\d\b",
    re.I,
)

#: The opening of a quoted reply: everything below it is a COPY of an older
#: message. Outlook's "From: / Sent:" block is recognised separately (it needs
#: a look ahead).
_QUOTE_OPENER_RE = re.compile(
    r"^\s*(?:-{2,}\s*(?:original|forwarded) message\s*-{2,}|"
    r"begin forwarded message\s*:|"
    r"on\b.{4,200}\bwrote\s*:)\s*$",
    re.I,
)
_HDR_FROM_RE = re.compile(r"^(?:from)\s*:", re.I)
_HDR_NEXT_RE = re.compile(r"^(?:sent|date|to|subject|cc)\s*:", re.I)


def _header_norm(line: str) -> str:
    """A line as a header row reads it: quote marks, markdown / HTML bold and
    stray bullets around the label removed ("> *From:* X", "**Sent:** Y",
    "<b>To:</b> Z" all read as the bare label)."""
    s = re.sub(r"^(?:>\s?)+", "", str(line or "").strip()).strip()
    s = re.sub(r"</?(?:b|strong|span|font|p|div)[^>]*>", "", s, flags=re.I)
    s = re.sub(r"^[*_]{1,2}([A-Za-z][A-Za-z -]*:)[*_]{1,2}\s*", r"\1 ", s)
    return s.strip()


def _quoted_from(lines: list[str]) -> int | None:
    """Index of the first line of quoted history, or ``None``.

    A Gmail / Apple "On ... wrote:", an "-----Original Message-----", or an
    Outlook header block: a ``From:`` row with a ``Sent:``/``Date:``/``To:``/
    ``Subject:`` row in the next few lines, after some content (a .txt export
    starts with the message's OWN header block, which is not a quote).
    """
    seen_content = False
    for i, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue
        if _QUOTE_OPENER_RE.match(line):
            return i
        h = _header_norm(line)
        if _HDR_FROM_RE.match(h) and seen_content:
            ahead = [_header_norm(x) for x in lines[i + 1:i + 7] if x.strip()]
            if any(_HDR_NEXT_RE.match(x) for x in ahead):
                return i
        if not _HDR_NEXT_RE.match(h) and not _HDR_FROM_RE.match(h):
            seen_content = True
    return None


#: Outlook writes an inline image as "[A screenshot of a phone   AI-generated
#: content may be incorrect., Picture, Picture]". The picture is real content
#: nobody read -- 010289's door diagram arrived exactly this way.
_IMAGE_PLACEHOLDER_RE = re.compile(r"^\s*\[[^\]]*(?:picture|image|screenshot|logo|cid:)[^\]]*\]?\s*$", re.I)

#: A line shorter than this cannot carry a fact on its own.
_MIN_CHARS = 12


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).strip()


def _html_to_text(html: str) -> str:
    """An HTML-only mail body as lines: block tags break, every tag goes. Read
    raw, each ``<td style=...>`` row was an "unread" line."""
    import html as _html

    t = re.sub(r"(?is)<(script|style|head)\b.*?</\1>", " ", str(html or ""))
    t = re.sub(r"(?i)<br\s*/?>|</(?:p|div|tr|li|h\d|table|blockquote)>", "\n", t)
    t = re.sub(r"(?s)<[^>]+>", "", t)
    t = _html.unescape(t).replace("\xa0", " ")
    return "\n".join(re.sub(r"[ \t]+", " ", x).strip() for x in t.splitlines())


def _read_pdf_lines(path: Path) -> list[tuple[int | None, str]]:
    try:
        import fitz  # PyMuPDF
    except Exception:
        return []
    out: list[tuple[int | None, str]] = []
    try:
        with fitz.open(str(path)) as doc:
            for pno in range(min(doc.page_count, _MAX_PDF_PAGES)):
                for ln in (doc.load_page(pno).get_text("text") or "").splitlines():
                    out.append((pno + 1, ln))
    except Exception:
        return []
    return out


def _read_lines(path: Path) -> list[tuple[int | None, str]]:
    """(page, line) pairs: page is None for a text artifact."""
    if path.suffix.lower() in PDF_SUFFIXES:
        return _read_pdf_lines(path)
    return [(None, ln) for ln in _read_text(path).splitlines()]


def _is_claimed(n: str, claimed: list[str], joined: str = "") -> bool:
    """Is the normalized line ``n`` carried by the atoms?

    Claimed when one atom's text contains the whole line, or when atoms whose
    text sits INSIDE the line together cover all of it but an enumerator or
    a stray word ("a." before "Coordinate resources..."). One short atom
    inside a long line no longer claims the line: "Line 1 | ... | $4,500.00
    CDW PO's are not transferrable." is not read because "4 500 00" was.
    """
    if any(n in c for c in claimed):
        return True
    # A wrapped line is a fragment of an atom; a line that runs across a
    # sentence break ("... by Friday. Also the riser") is the tail of one
    # atom and the head of the next, so it is also read against the atoms
    # laid end to end.
    if len(n) >= _MIN_CHARS and n in joined:
        return True
    covered = bytearray(len(n))
    for c in claimed:
        if len(c) < 3 or len(c) >= len(n):
            continue
        start = n.find(c)
        while start != -1:
            end = start + len(c)
            # whole words only
            if (start == 0 or n[start - 1] == " ") and (end == len(n) or n[end] == " "):
                for k in range(start, end):
                    covered[k] = 1
            start = n.find(c, start + 1)
    residue = "".join(ch for ch, cv in zip(n, covered) if not cv and ch != " ")
    return len(residue) <= max(2, int(0.1 * len(n.replace(" ", ""))))


#: Words a splitter drops between the clauses it cuts a sentence into ("X,
#: Y and Z" -> X / Y / Z). Uncovered, they are glue, not an unread fact.
_GLUE_WORDS = frozenset("a an and or but nor the also plus then as well so to".split())
#: A run of words an atom shares with a line counts as that atom's piece of
#: the line only when it is a phrase (three words, a fact's length): "of the"
#: or "the customer" shared by chance is not a quote.
_MIN_PIECE_WORDS = 3


def _pieces_claimed(n: str, own: list[str]) -> bool:
    """Is the long normalized line ``n`` carried by this document's atoms
    taken together?

    A long line the parser cut into clauses (000132's HubSpot note line 9,
    "Maintenance and support...", eleven atoms) is not inside any one atom,
    and the atoms are not always inside it either: a clause keeps its label
    ("Maintenance and support: ...") or runs on into the next line. So each
    atom contributes the runs of words it shares with the line, and the line
    is claimed when those runs between them cover nearly all of it -- the
    splitter's dropped "and"s aside. One short piece never claims a long
    line: whatever it leaves uncovered is the residue.
    """
    words = n.split()
    if len(n) < _MIN_CHARS or len(words) < 4:
        return False
    vocab = set(words)
    covered = [False] * len(words)
    for c in own:
        cw = c.split()
        if not cw or not (vocab & set(cw)):
            continue
        for blk in SequenceMatcher(None, words, cw, autojunk=False).get_matching_blocks():
            if not blk.size:
                continue
            piece = words[blk.a:blk.a + blk.size]
            whole = blk.size == len(cw) and len(c) >= 3
            if whole or (blk.size >= _MIN_PIECE_WORDS and len(" ".join(piece)) >= _MIN_CHARS):
                covered[blk.a:blk.a + blk.size] = [True] * blk.size
    total = sum(len(w) for w in words)
    residue = sum(len(w) for w, cv in zip(words, covered) if not cv and w not in _GLUE_WORDS)
    return residue <= max(2, int(0.1 * total))


def _read_text(path: Path) -> str:
    try:
        raw = path.read_bytes()
    except Exception:
        return ""
    if path.suffix.lower() in {".eml", ".msg"}:
        try:
            import email
            from email import policy

            msg = email.message_from_bytes(raw, policy=policy.default)
            body = msg.get_body(preferencelist=("plain", "html"))
            if body is not None:
                content = body.get_content()
                if body.get_content_type() == "text/html":
                    content = _html_to_text(content)
                return content
        except Exception:
            pass
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            text = raw.decode(enc)
        except Exception:
            continue
        if path.suffix.lower() in {".html", ".htm"}:
            text = _html_to_text(text)
        return text
    return ""


def _atom_texts(atoms: list[Any], artifact_id: str | None) -> list[str]:
    """Every string an atom quotes. ``artifact_id=None`` means every atom of
    the deal, whichever document it came from."""
    out = []
    for a in atoms:
        if artifact_id is not None and str(getattr(a, "artifact_id", "") or "") != artifact_id:
            continue
        for t in (getattr(a, "raw_text", ""), getattr(a, "normalized_text", "")):
            n = _norm(t)
            if n:
                out.append(n)
        v = getattr(a, "value", None)
        if isinstance(v, dict):
            for k in ("quote", "text", "value"):
                n = _norm(v.get(k) or "")
                if n:
                    out.append(n)
            # A label consumed into its items ("Provided by us:") was read --
            # it lives on every item as the lead-in.
            for k in ("list_label", "parent_field", "field_name"):
                n = _norm(v.get(k) or "")
                if n:
                    out.append(n)
            for lead in v.get("lead_in") or []:
                n = _norm(lead)
                if n:
                    out.append(n)
        locs = [getattr(a, "locator", None)]
        locs += [getattr(r, "locator", None) for r in (getattr(a, "source_refs", None) or [])[:1]]
        for loc in locs:
            if not isinstance(loc, dict):
                continue
            # A heading the parser read as structure (the section an atom
            # sits under) was read, not missed.
            for lead in list(loc.get("lead_in") or []) + list(loc.get("section_path") or []):
                n = _norm(lead)
                if n:
                    out.append(n)
    return out


def _sentences_claimed(line: str, claimed: list[str], joined: str) -> bool:
    parts = [_norm(p) for p in re.split(r"(?<=[.!?;])\s+", line)]
    parts = [p for p in parts if p]
    if len(parts) < 2:
        return False
    return all(len(p) < _MIN_CHARS or _is_claimed(p, claimed, joined) for p in parts)


def _source_lines(path: Path) -> list[tuple[int, int | None, str, bool]]:
    """``(line_no, page, stripped_line, quoted)`` for every non-blank line;
    page is None for a text artifact."""
    return _lines_of(_read_lines(path))


def _lines_of(pairs: list[tuple[int | None, str]]) -> list[tuple[int, int | None, str, bool]]:
    raw_lines = [ln for _, ln in pairs]
    q_from = _quoted_from(raw_lines)
    out = []
    for i, (page, raw) in enumerate(pairs, start=1):
        line = raw.strip()
        if not _norm(line):
            continue
        quoted = (q_from is not None and i - 1 >= q_from) or line.startswith(">")
        out.append((i, page, line, quoted))
    return out


def coverage_for_artifact(
    path: Path,
    artifact_id: str,
    kept: list[Any],
    suppressed: list[Any] | None = None,
    claimed_anywhere: list[str] | None = None,
    owner_of: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Line-by-line: what became an atom, what was dropped, what was never read.

    ``owner_of`` maps a folded line to the artifact whose copy of it is the one
    that counts (see :func:`_line_owners`). A quoted copy anywhere else -- a
    reply quoting the message it answers -- is listed as ``copy``: still
    visible, never counted as unread a second time.
    """
    source_lines = _source_lines(path)
    if not source_lines:
        return {}
    # A signature, a quoted history, a repeated ask: read ONCE for the deal on
    # purpose. Looking only at this artifact's atoms, every later copy reads as
    # a miss -- live 010289 reported 84, nearly all of them "Account
    # Executive" and a phone number in mail number seven.
    own = _atom_texts(kept, artifact_id)
    claimed = own + list(claimed_anywhere or [])
    joined = " ".join(own) + " \x00 " + " ".join(claimed_anywhere or [])
    dropped = _atom_texts(list(suppressed or []), artifact_id)
    lines: list[dict[str, Any]] = []
    n_claimed = 0
    counted_here: set[str] = set()
    for i, page, line, quoted in source_lines:
        n = _norm(line)
        if (_is_claimed(n, claimed, joined) or _sentences_claimed(line, claimed, joined)
                or _pieces_claimed(n, own)):
            n_claimed += 1
            continue
        hdr = _header_norm(line)
        if _IMAGE_PLACEHOLDER_RE.match(line):
            # A picture sat in this mail and nothing read it. Not chrome, not
            # prose we missed: its own answer ("go look at the image").
            state = "image"
        elif (_CHROME_RE.match(line) or _CHROME_RE.match(hdr) or _FOOTER_RE.search(line)
              or len(line) < _MIN_CHARS):
            state = "chrome"
        else:
            state = "unread"
        # A dropped atom SHORTER than the line marks it only when it is long
        # enough to mean something: a gate can drop a bare "Thanks," and a
        # "thanks" inside a real unread sentence ("thanks -- we need 40
        # drops") must stay `unread`.
        if any(n in d or (d in n and (len(d) >= _MIN_CHARS or d == n)) for d in dropped):
            state = "suppressed"
        if state == "unread":
            owner = (owner_of or {}).get(n)
            if quoted and ((owner and owner != artifact_id) or n in counted_here):
                # A QUOTED line whose counted copy is elsewhere: the message
                # the reply answers, or the first reply that quoted a message
                # the deal never got as a file, or this mail's own words
                # quoted back further down. A line a mail WROTE is always its
                # own miss, even when another mail wrote it too.
                state = "copy"
            else:
                counted_here.add(n)
        row = {"line": i, "text": line[:400], "state": state}
        if quoted:
            row["quoted"] = True
        if page is not None:
            row["page"] = page
        lines.append(row)
    total = n_claimed + len(lines)
    return {
        "artifact_id": artifact_id,
        "lines_total": total,
        "lines_claimed": n_claimed,
        "unclaimed": lines[:200],
        "unread_count": sum(1 for x in lines if x["state"] == "unread"),
        "image_count": sum(1 for x in lines if x["state"] == "image"),
        "copy_count": sum(1 for x in lines if x["state"] == "copy"),
    }


def _line_owners(texts: dict[str, str]) -> dict[str, str]:
    """Folded line -> the one artifact whose copy of it counts.

    A line a message WROTE (above its quoted history) beats a line it only
    QUOTES, so the original mail owns its words and every reply quoting it
    holds a copy. Between equals, the first artifact read wins. A line that
    exists only as a quote (its message never reached the deal as a file)
    still has exactly one owner, so it is counted -- once.
    """
    owner: dict[str, str] = {}
    quoted_owner: dict[str, str] = {}
    for aid, text in texts.items():
        for _i, _page, line, quoted in _lines_of([(None, ln) for ln in text.splitlines()]):
            n = _norm(line)
            if len(n) < _MIN_CHARS:
                continue
            (quoted_owner if quoted else owner).setdefault(n, aid)
    for n, aid in quoted_owner.items():
        owner.setdefault(n, aid)
    return owner


def build_text_coverage(
    artifact_paths: dict[str, Path],
    atoms: list[Any],
    suppressed: list[Any] | None = None,
) -> list[dict[str, Any]]:
    """One row per readable text artifact. Never raises: coverage reporting
    must not be able to fail a compile."""
    out: list[dict[str, Any]] = []
    # Long enough to be a fact rather than a coincidence, and to keep this
    # deal-wide pass from swallowing a real miss.
    everywhere = [t for t in _atom_texts(atoms, None) if len(t) >= 18]
    paths: dict[str, Path] = {}
    texts: dict[str, str] = {}
    for artifact_id, path in (artifact_paths or {}).items():
        try:
            p = Path(path)
            if p.suffix.lower() not in TEXT_SUFFIXES | PDF_SUFFIXES:
                continue
            paths[str(artifact_id)] = p
            texts[str(artifact_id)] = "\n".join(ln for _, ln in _read_lines(p))
        except Exception:
            continue
    try:
        owners = _line_owners(texts)
    except Exception:
        owners = {}
    for artifact_id, p in paths.items():
        try:
            row = coverage_for_artifact(p, artifact_id, atoms, suppressed, everywhere, owners)
            if row:
                out.append(row)
        except Exception:
            continue
    return out


__all__ = ["build_text_coverage", "coverage_for_artifact", "PDF_SUFFIXES", "TEXT_SUFFIXES"]
