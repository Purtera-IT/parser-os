"""People from signature blocks and "Name Title" list lines, by shape.

Live 010300 (2026-09-03): the deal's ten stakeholder records had no email or
phone between them. Carl Painter's signature carried a title, direct, mobile,
fax and email on five consecutive lines; the parser typed the name line and
the phone lines separately and never joined them. A bullet list of the CDW
team ("Rhonda Sharp Professional Services Manager") was read as a five-word
name.

Nothing here is a vocabulary. A signature is a CLUSTER: a name-shaped line with
an email-shaped or phone-shaped line within a few lines of it. The lines in
between that are neither are the title and organisation. A "Name Title" list
line is a name-shape prefix followed by more capitalised words.
"""
from __future__ import annotations

import re
from typing import Any

from app.parsers.contact_property_block import _NAME_SHAPE
from app.parsers.value_shapes import classify_value

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(?:\+?1[\s.\-]?)?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}")
_LINK_RE = re.compile(r"https?://|www\.|<mailto:|\[cid:", re.I)
_LABEL_VALUE_RE = re.compile(r"^\s*([A-Za-z][A-Za-z ./&-]{0,24}?)\s*[:：]\s*(.+?)\s*$")
_SEP_RE = re.compile(r"\s*[|•·]\s*")
_MAX_GAP = 4
# Mirrors email_parser._LOCATION_LABEL_RE (kept local: email_parser imports us).
_LOCATION_LABEL_RE = re.compile(
    r"^(?:site\s+)?(?:location|address|site|job\s*site|site\s+address|ship\s*to|"
    r"install(?:ation)?\s+(?:address|location|site))\s*:\s*$",
    re.IGNORECASE,
)
_TITLE_MAX_WORDS = 8


def _clean(line: str) -> str:
    # Collapse whitespace runs, which also folds the NON-BREAKING SPACE
    # Outlook puts between a first and last name. It survived every check
    # here -- str.split() treats U+00A0 as whitespace and the \s class
    # matches it -- so it never broke the name SHAPE, it just rode along
    # into the stored value. Live 010237 filed the names of Tim Penney
    # and Zach Burdick with U+00A0 in them, which dedup then correctly
    # folded with the plain-space spellings while keeping the
    # non-breaking one, so every later match on "Tim Penney" missed a
    # person the deal had already identified.
    return re.sub(r"\s+", " ", re.sub(r"<[^>]*>", "", line)).strip(" \t\r\n;,-–—*")


def _name_from(line: str) -> str | None:
    core = _clean(line)
    if not core:
        return None
    # "Carl Painter| Sr. Account Manager| carlpai@cdw.com| CDW" -- take the
    # first cell when the line is cell-separated; the other cells may well
    # be the email and phone, that is the point.
    first = _SEP_RE.split(core)[0].strip()
    if not first or len(first) > 60 or _EMAIL_RE.search(first) or _PHONE_RE.search(first):
        return None
    # A salutation ("Hi Hiran,") ends with a comma; a signature name does not.
    if line.rstrip().endswith((",", ":")):
        return None
    toks = first.split()
    if 2 <= len(toks) <= 3 and _NAME_SHAPE.match(first):
        return first
    return None


#: How an email stops talking. Not a job title, however title-shaped.
_SIGNOFF_RE = re.compile(
    r"^(thanks?|thank\s+you|thx|many\s+thanks|much\s+appreciated|appreciate\s+it|"
    r"(kind(est)?|best|warm(est)?)\s+regards?|regards?|cheers|sincerely|"
    r"best\s+wishes|all\s+the\s+best|yours\s+(truly|sincerely|faithfully)|"
    r"talk\s+soon|speak\s+soon|looking\s+forward|respectfully|"
    r"sent\s+from\s+my\s+\w+)\s*[.!]?$",
    re.IGNORECASE,
)


def _is_titleish(line: str) -> bool:
    core = _clean(line)
    if not core or _EMAIL_RE.search(core) or _PHONE_RE.search(core) or _LINK_RE.search(core):
        return False
    words = core.split()
    if not (1 <= len(words) <= _TITLE_MAX_WORDS):
        return False
    # A sign-off ("Thanks," / "Best regards,") ends with a comma; a title does
    # not. One-word lines are not titles unless they are an all-caps org.
    if line.rstrip().endswith((",", ":")):
        return False
    # ...but only when the writer typed the comma. "Thank you" on its own line
    # is two words with a leading capital and passes every test below, so live
    # 010237 filed a person whose job title was "Thank you" -- and, because the
    # cluster began at the line ABOVE the sign-off, that person also took Chase
    # Smith's email and phone and was named after a document heading. How an
    # email stops talking is a closed set of phrases, worth naming rather than
    # guessing at from shape.
    if _SIGNOFF_RE.match(core):
        return False
    if len(words) == 1 and not core.isupper():
        return False
    if core.endswith((".", "?", "!")) and len(words) > 3:
        return False
    caps = sum(1 for w in words if w[:1].isupper())
    return caps >= max(1, len(words) // 2)


def _phones_from(line: str) -> list[tuple[str, str]]:
    """(label, phone) pairs on one line: "Direct:847-968-9740 | Mobile: 847-363-7372"."""
    out: list[tuple[str, str]] = []
    core = _clean(line)
    for cell in _SEP_RE.split(core):
        m = _PHONE_RE.search(cell)
        if not m:
            continue
        label = cell[: m.start()].strip(" :：").strip()
        out.append((label, m.group(0).strip()))
    return out


def signature_line_indexes(lines: list[str]) -> set[int]:
    """Indexes of every line that belongs to a signature cluster.

    The cluster is the same one ``people_from_signature_lines`` reads; the
    title, phone, email, link and org lines inside it are contact chrome and
    must not be typed as deal content. Live 010300: "Account Executive" and
    "Executive Vice President of Sales" became stakeholder records of their
    own once the name lines around them were (correctly) skipped.
    """
    out: set[int] = set()
    for rec in people_from_signature_lines(lines):
        span = rec.get("_span")
        if span:
            out.update(range(span[0], span[1]))
    return out


def people_from_signature_lines(lines: list[str]) -> list[dict[str, Any]]:
    """Person records from the signature clusters in ``lines``.

    A cluster starts at a name-shaped line and extends while the following
    lines (up to ``_MAX_GAP`` non-matching lines) are title-ish, email, phone,
    or link lines. A cluster counts only if it has an email OR a phone: a bare
    name followed by prose is not a signature.
    """
    people: list[dict[str, Any]] = []
    n = len(lines)
    i = 0
    while i < n:
        name = _name_from(lines[i])
        if not name:
            i += 1
            continue
        # A name under "Location:" / "Address:" is a place, and the lines
        # after it are its street and phone, not a person's contact card.
        # Live 010289: "Nesfield Performance Bethesda / 7832 Wisconsin Ave ..."
        # became a stakeholder.
        prev = next((_clean(lines[k]) for k in range(i - 1, -1, -1) if _clean(lines[k])), "")
        if _LOCATION_LABEL_RE.match(prev):
            i += 1
            continue
        rec: dict[str, Any] = {"kind": "person", "name": name}
        titles: list[str] = []
        phones: dict[str, str] = {}
        # Inline cells on the name line: "Carl Painter| Sr. Account Manager| carlpai@cdw.com| CDW"
        cells = _SEP_RE.split(_clean(lines[i]))[1:]
        for c in cells:
            c = c.strip()
            if not c:
                continue
            if _EMAIL_RE.search(c):
                rec.setdefault("email", _EMAIL_RE.search(c).group(0).lower())
            elif _PHONE_RE.search(c):
                for lab, ph in _phones_from(c):
                    phones[lab or "phone"] = ph
            elif _is_titleish(c):
                titles.append(c)
        j = i + 1
        gap = 0
        last_hit = i
        #: Non-blank lines examined since the name. A signature is routinely
        #: double-spaced ("Patrick Kelly", "", "Account Executive"), so
        #: "the line after the name" has to be counted in CONTENT lines --
        #: measuring it in j made the blank line push the title out of the
        #: exemption below and the title became the person.
        content_seen = 0
        while j < n and gap < _MAX_GAP:
            raw = lines[j]
            core = _clean(raw)
            if not core:
                gap += 1
                j += 1
                continue
            content_seen += 1
            if _name_from(raw) and j > i + 1 and (rec.get("email") or phones):
                break  # next person's signature
            if (_name_from(raw) and content_seen > 1
                    and not (rec.get("email") or phones or titles)):
                # A contact card belongs to the NEAREST name above it. This
                # cluster has collected nothing -- no title, no address, no
                # number -- so the line it started on was not a signature, and
                # letting it run on hands the next person's details to it.
                # That is how "WIFI Example", a heading, came to own Chase
                # Smith's address and direct line on live 010237. The line
                # The first CONTENT line under a name is exempt: that is
                # where a job title lives, "Sr. Account Manager" is
                # name-shaped, and a signature is routinely double-spaced.
                break
            em = _EMAIL_RE.search(core)
            ph = _phones_from(raw)
            if em and not rec.get("email"):
                rec["email"] = em.group(0).lower()
                gap = 0
            elif ph:
                for lab, val in ph:
                    phones[lab or "phone"] = val
                gap = 0
            elif _LINK_RE.search(core):
                gap = 0
            elif _is_titleish(core) and not titles and len(core.split()) <= _TITLE_MAX_WORDS:
                titles.append(core)
                gap = 0
            elif _is_titleish(core) and titles and len(core.split()) <= 3 and core.isupper():
                rec.setdefault("organization", core)
                gap = 0
            else:
                gap += 1
            if gap == 0:
                last_hit = j
            j += 1
        if rec.get("email") or phones:
            rec["_span"] = (i, last_hit + 1)
            if titles:
                rec["role"] = titles[0]
            if phones:
                # Prefer a labelled "Direct"/"Office" line, then anything.
                ordered = sorted(phones.items(), key=lambda kv: (0 if re.search(r"direct|office|desk|work", kv[0], re.I) else 1 if kv[0] == "phone" else 2))
                rec["phone"] = ordered[0][1]
                if len(phones) > 1:
                    rec["phones"] = {k or "phone": v for k, v in phones.items()}
            if not rec.get("organization") and rec.get("email"):
                dom = rec["email"].split("@", 1)[1]
                rec["organization_domain"] = dom
            people.append(rec)
            i = j
        else:
            i += 1
    return people


def name_and_role_from_list_line(line: str) -> dict[str, str] | None:
    """"Rhonda Sharp Professional Services Manager" -> name + role.

    A bullet whose first two tokens are a name shape and whose remaining
    tokens are capitalised title words (3 to 8 in total) is a person WITH a
    role, not a five-word name. Shape only.
    """
    core = _clean(line)
    if not core or _EMAIL_RE.search(core) or _PHONE_RE.search(core) or _LINK_RE.search(core):
        return None
    if re.search(r"[.,;:()]", core):
        return None
    toks = core.split()
    if not (4 <= len(toks) <= 8):
        return None
    head = " ".join(toks[:2])
    if not _NAME_SHAPE.match(head):
        return None
    rest = toks[2:]
    if sum(1 for w in rest if w[:1].isupper()) < max(1, len(rest) - 1):
        return None
    return {"kind": "person", "name": head, "role": " ".join(rest)}


__all__ = ["people_from_signature_lines", "name_and_role_from_list_line"]
