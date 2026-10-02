"""Shapes of the parts of a signed SOW that are not scope, shared by the PDF
and DOCX parsers.

Three things on a signed statement of work are routinely mistaken for scope:

* the **exclusions list**. Bullets under "Out of Scope" / "Exclusions" are
  negative scope; their verbs ("Chromebook imaging", "install ...") made the
  lexical typer call them scope_item, and the shortest ones were hidden as
  chatter. The heading the author wrote is the authority.
* the **signature block** ("DocuSigned by:", the signer's name, "Title:",
  "Date:", the company name over it). It names the parties who sign; it is
  never work.
* the **e-signature envelope stamp** ("Docusign Envelope ID: 3F2A...") the
  platform prints at the top of every page. It is furniture; it stays an atom
  (labels on it train the reject head) but flagged chatter.

Everything here is shape and general vocabulary, never a deal or a name.
"""
from __future__ import annotations

import re
from typing import Iterable

# ── exclusions ────────────────────────────────────────────────────────────

_NUM_PREFIX = r"(?:(?:section\s+)?\d{1,2}(?:\.\d{1,2})*[.)]?\s+|[A-Z][.)]\s+)?"
_EXCLUSION_CORE = (
    r"(?:out[\s-]+of[\s-]+scope|not\s+in\s+scope|exclusions?|excluded|"
    r"not\s+included|what(?:'|’)?s\s+not\s+included)"
)
_EXCLUSION_WORDS = r"(?:scope|services?|work|items?|tasks?|activities|project|deliverables?|and|of|the)"
#: A heading that names an exclusions section: "OUT OF SCOPE", "Out of Scope:",
#: "7. Exclusions", "Scope Exclusions", "Services Not Included",
#: "Out-of-Scope Items". The whole line must be the label.
EXCLUSION_HEADING_RE = re.compile(
    rf"^\s*{_NUM_PREFIX}(?:{_EXCLUSION_WORDS}\s+){{0,3}}{_EXCLUSION_CORE}"
    rf"(?:\s+{_EXCLUSION_WORDS}){{0,3}}\s*:?\s*$",
    re.I,
)


#: The lead-in sentence that opens an exclusions list in place of a heading:
#: "The following are not included in this SOW:", "The following items are
#: excluded from the scope of work", "Not included in this quote:" (010003).
EXCLUSION_LEADIN_RE = re.compile(
    r"^\s*(?:the\s+following(?:\s+(?:items|services|tasks|activities|work|are\s+items))?\s+"
    r"(?:are|is|will\s+be)\s+|)"
    r"(?:not\s+included|excluded|out\s+of\s+scope|not\s+in\s+scope|not\s+part\s+of)"
    r"(?:\s+(?:in|from|of|under)\s+(?:this|the)\s+(?:sow|statement\s+of\s+work|scope(?:\s+of\s+work)?|"
    r"project|quote|proposal|agreement|engagement|services?))?\s*:?\s*$",
    re.I,
)


def is_exclusion_heading(text: str) -> bool:
    """True when ``text`` is (only) the label of an exclusions section, or
    the lead-in sentence that opens one ("The following are not included in
    this SOW:").

    "Inclusions and Exclusions" names both halves, so it is not."""
    t = " ".join(str(text or "").split())
    if t and len(t) <= 90 and EXCLUSION_LEADIN_RE.match(t):
        return True
    if not t or len(t) > 60:
        return False
    if re.search(r"\binclu(?:sions?|ded)\b", t, re.I) and not re.search(r"\bnot\s+included\b", t, re.I):
        return False
    return bool(EXCLUSION_HEADING_RE.match(t))


def under_exclusion_heading(section_path: Iterable[str] | None) -> bool:
    """True when the nearest heading in ``section_path`` that names a section
    type is an exclusions heading."""
    for heading in reversed([str(h or "") for h in (section_path or [])]):
        if is_exclusion_heading(heading):
            return True
        if is_sow_section_label(heading):
            return False
    return False


#: The usual names of the other sections of a SOW. Recognised as headings only
#: to END an exclusions section, so a fee table or a signature block printed
#: after the exclusions list is not read as more exclusions.
_SOW_SECTION_LABEL_RE = re.compile(
    rf"^\s*{_NUM_PREFIX}(?:"
    r"(?:project\s+|services?\s+|professional\s+services\s+)?(?:fees?|pricing|price|costs?|investment|rates?)(?:\s+(?:and|&)\s+\w+)?|"
    r"payment(?:\s+(?:terms|schedule))?|invoic(?:e|ing)|billing|"
    r"(?:project\s+)?assumptions?(?:\s+(?:and|&)\s+dependencies)?|dependencies|"
    r"(?:project\s+)?deliverables?|acceptance(?:\s+criteria)?|"
    # Any party's responsibilities ("PurTera Responsibilities", "Customer
    # Responsibilities") ends an exclusions section: 010087 typed the PMO
    # duties under "PURTERA RESPONSIBILITIES" as exclusions.
    r"(?:[a-z][\w.&'-]*\s+){0,2}responsibilities|"
    r"(?:in\s+)?scope(?:\s+of\s+(?:work|services))?|scope\s+(?:summary|overview)|"
    r"(?:project\s+)?(?:schedule|timeline|milestones?)|"
    r"change\s+(?:orders?|management|requests?)|"
    r"terms(?:\s+(?:and|&)\s+conditions)?|general\s+terms|term|"
    r"signatures?|authori[sz]ed\s+signatures?|approvals?|"
    r"materials?|equipment|hardware|bill\s+of\s+materials"
    r")\s*:?\s*$",
    re.I,
)


def is_sow_section_label(text: str) -> bool:
    t = " ".join(str(text or "").split())
    return bool(t) and len(t) <= 60 and bool(_SOW_SECTION_LABEL_RE.match(t))


# ── signature block ───────────────────────────────────────────────────────

SIGNATURE_HEADING_RE = re.compile(
    rf"^\s*{_NUM_PREFIX}(?:"
    r"signatures?|signature\s+page|signature\s+block|authori[sz]ed\s+signatures?|"
    r"acceptance\s+(?:and|&)\s+signatures?|"
    r"(?:accepted|agreed)\s+(?:and|&)\s+(?:accepted|agreed)"
    r")(?:\s+(?:by|of\s+the\s+parties))?\s*:?\s*$"
    r"|^\s*in\s+witness\s+whereof\b[^.]{0,80}$",
    re.I,
)
#: e-signature badge furniture: "DocuSigned by:", "Signed by:", the signature
#: hash under the drawn name ("4A1B2C3D4E5F4A6"), "Signature:", "By: ____".
_ESIGN_FURNITURE_RE = re.compile(
    r"^\s*(?:"
    r"(?:docu\s*signed|e-?signed|digitally\s+signed|electronically\s+signed|signed)\s+by\s*:?\s*$|"
    r"(?:docusign|adobe\s+sign|dropbox\s+sign|hellosign|pandadoc|signnow)\b[^.]{0,40}$|"
    r"[0-9A-F]{12,40}\.{0,3}\s*$|"
    r"(?:signature|sign(?:ed)?|by)\s*:?\s*[_.\s]*$|"
    r"_{3,}\s*$"
    r")",
    re.I,
)
#: A line that opens an e-signature badge: the badge is a signature block
#: wherever it sits.
_ESIGN_MARKER_RE = re.compile(
    r"^\s*(?:docu\s*signed|e-?signed|digitally\s+signed|electronically\s+signed|signed)\s+by\b",
    re.I,
)
_SIG_LABEL_RE = re.compile(
    r"^\s*(?:name|printed\s+name|print\s+name|title|date|date\s+signed|signature|by|company|"
    r"organi[sz]ation|customer|client|provider|seller|vendor|partner|signer|signed)\s*:",
    re.I,
)
_DATE_RE = re.compile(
    r"^\s*(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}|"
    r"[A-Z][a-z]+\.?\s+\d{1,2},\s+\d{4})(?:\s*\|\s*\d{1,2}:\d{2}.*)?\s*$"
)


def is_signature_heading(text: str) -> bool:
    t = " ".join(str(text or "").split())
    return bool(t) and len(t) <= 90 and bool(SIGNATURE_HEADING_RE.match(t))


def is_esign_marker(line: str) -> bool:
    return bool(_ESIGN_MARKER_RE.match(str(line or "")))


def is_esign_furniture(line: str) -> bool:
    """The badge's own chrome (no party, no person): kept as chatter."""
    return bool(_ESIGN_FURNITURE_RE.match(str(line or "")))


def is_signature_label_line(line: str) -> bool:
    """A labelled signature row ("Name: ...", "Title: ...", "Date: ...") or
    an e-signature badge line: positive evidence of a signature block."""
    t = str(line or "")
    return bool(_SIG_LABEL_RE.match(t) or is_esign_marker(t))


def is_signature_line(line: str) -> bool:
    """A line with the shape of a signature-block entry: a label row
    ("Name: ...", "Title: ...", "Date: ..."), a date, badge furniture, or a
    short line with no sentence in it (a signer's name, a title, a company)."""
    t = " ".join(str(line or "").split())
    if not t:
        return False
    if is_esign_furniture(t) or is_esign_marker(t) or _SIG_LABEL_RE.match(t) or _DATE_RE.match(t):
        return True
    if len(t) > 60 or re.search(r"[.!?;]\s*$", t) or len(t.split()) > 8:
        return False
    return not re.search(r"\$\s?\d|\d{3,}", t)


# ── e-signature envelope stamp ────────────────────────────────────────────

#: "Docusign Envelope ID: 3F2A9C1E-1B2C-4D5E-9F00-ABCDEF123456" (and the other
#: platforms' page stamps). Matched at the start of a line; whatever follows
#: on the same line is the page's own text.
_STAMP_BODY = (
    r"docu\s*sign\s+envelope\s+id\s*[:#]?\s*[0-9A-F]{8}(?:-?[0-9A-F]{4}){3}-?[0-9A-F]{12}|"
    r"(?:adobe(?:\s+acrobat)?\s+sign|echosign|dropbox\s+sign|hellosign|pandadoc|signnow)\s+"
    r"(?:transaction|document|envelope|signature|agreement)\s+(?:id|number|no\.?)\s*[:#]?\s*[A-Za-z0-9_-]{8,}"
)
DOC_STAMP_RE = re.compile(r"^\s*(?:" + _STAMP_BODY + r")", re.I)
#: The same stamp anywhere on a line: the layout can set it after the page's
#: own text ("Signatures Docusign Envelope ID: ...") or a footer after it.
_DOC_STAMP_ANY_RE = re.compile(r"(?<![A-Za-z0-9])(?:" + _STAMP_BODY + r")", re.I)
#: The stamp's label alone, its id wrapped onto the next line.
_STAMP_LABEL_RE = re.compile(
    r"^\s*(?:docu\s*sign\s+envelope\s+id|(?:adobe(?:\s+acrobat)?\s+sign|echosign|dropbox\s+sign|"
    r"hellosign|pandadoc|signnow)\s+(?:transaction|document|envelope|signature|agreement)\s+"
    r"(?:id|number|no\.?))\s*[:#]?\s*$", re.I)
_STAMP_ID_RE = re.compile(r"^\s*(?:[0-9A-F]{8}(?:-?[0-9A-F]{4}){3}-?[0-9A-F]{12}|[A-Za-z0-9_-]{12,})\s*$", re.I)
#: What a page stamp line also carries: a page number.
_PAGE_FURNITURE_RE = re.compile(r"^\s*(?:page\s*\d+(?:\s*(?:of|/)\s*\d+)?|\d{1,3}(?:\s*(?:of|/)\s*\d{1,3})?)\s*$", re.I)

DOC_STAMP_RULE = "doc_stamp"


def split_doc_stamp(line: str) -> tuple[str, str] | None:
    """``(stamp, rest)`` when ``line`` carries an e-signature page stamp:
    at its start, or after / before the page's own text on the same line.
    ``rest`` is the line without the stamp (and without a page number that
    only rode along with it)."""
    line = str(line or "")
    m = DOC_STAMP_RE.match(line) or _DOC_STAMP_ANY_RE.search(line)
    if not m:
        return None
    rest = " ".join(x for x in (line[:m.start()].strip(), line[m.end():].strip()) if x)
    if _PAGE_FURNITURE_RE.match(rest):
        rest = ""
    return m.group(0).strip(), rest


def join_wrapped_stamps(lines: list[str]) -> list[str]:
    """Put a stamp whose id wrapped onto the next line back on one line
    ("Docusign Envelope ID:" / "3F2A9C1E-...")."""
    out: list[str] = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if _STAMP_LABEL_RE.match(ln or "") and i + 1 < len(lines) and _STAMP_ID_RE.match(lines[i + 1] or ""):
            out.append(f"{ln.strip()} {lines[i + 1].strip()}")
            i += 2
            continue
        out.append(ln)
        i += 1
    return out


def is_doc_stamp(text: str) -> bool:
    """The whole text is an e-signature page stamp (a page number aside)."""
    t = " ".join(str(text or "").split())
    s = split_doc_stamp(t)
    if s is None:
        joined = join_wrapped_stamps(str(text or "").splitlines())
        s = split_doc_stamp(" ".join(joined)) if len(joined) == 1 else None
    return bool(s) and not s[1]


__all__ = [
    "DOC_STAMP_RULE",
    "is_doc_stamp",
    "is_esign_furniture",
    "is_esign_marker",
    "is_exclusion_heading",
    "is_signature_heading",
    "is_signature_label_line",
    "is_signature_line",
    "is_sow_section_label",
    "join_wrapped_stamps",
    "split_doc_stamp",
    "under_exclusion_heading",
]
