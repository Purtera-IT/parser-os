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


def is_exclusion_heading(text: str) -> bool:
    """True when ``text`` is (only) the label of an exclusions section.

    "Inclusions and Exclusions" names both halves, so it is not."""
    t = " ".join(str(text or "").split())
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
    r"(?:customer|client|seller|provider|partner|vendor)\s+responsibilities|responsibilities|"
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
DOC_STAMP_RE = re.compile(
    r"^\s*(?:"
    r"docu\s*sign\s+envelope\s+id\s*[:#]?\s*[0-9A-F]{8}(?:-?[0-9A-F]{4}){3}-?[0-9A-F]{12}|"
    r"(?:adobe\s+sign|dropbox\s+sign|hellosign|pandadoc|signnow)\s+"
    r"(?:transaction|document|envelope|signature)\s+(?:id|number|no\.?)\s*[:#]?\s*[A-Za-z0-9_-]{8,}"
    r")",
    re.I,
)

DOC_STAMP_RULE = "doc_stamp"


def split_doc_stamp(line: str) -> tuple[str, str] | None:
    """``(stamp, rest)`` when ``line`` opens with an e-signature page stamp."""
    m = DOC_STAMP_RE.match(str(line or ""))
    if not m:
        return None
    return m.group(0).strip(), str(line)[m.end():].strip()


def is_doc_stamp(text: str) -> bool:
    """The whole text is an e-signature page stamp."""
    s = split_doc_stamp(" ".join(str(text or "").split()))
    return bool(s) and not s[1]


__all__ = [
    "DOC_STAMP_RULE",
    "is_doc_stamp",
    "is_esign_furniture",
    "is_esign_marker",
    "is_exclusion_heading",
    "is_signature_heading",
    "is_signature_line",
    "is_sow_section_label",
    "split_doc_stamp",
    "under_exclusion_heading",
]
