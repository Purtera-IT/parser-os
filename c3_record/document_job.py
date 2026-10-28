"""Which documents are another job: the one set-aside rule for training's views and the parser's compile.

A deal's CRM record can carry another job's documents (a programme customer's other work orders). The
parser's ``document_job_scope`` judges each conversation (an email thread with the files that came with
it) ``this_deal`` or ``other_job`` and sets aside the other jobs; the labeling page's ``document_job``
card is a person's verdict on one document. Both read the verdict through ``set_aside``:

1. only ``other_job`` sets anything aside, and only when it is sure: a person's verdict (a Document card,
   a PM's taught lesson) always is; a model's or the base's when its confidence reaches the floor;
2. a drawing is never set aside on a model's word alone (a drawing is titled by the building, not the
   job); a person's verdict still sets it aside;
3. a document whose own text names this deal -- its project name without reference numbers, or a site
   address the CRM gives -- is this deal's, whatever the verdict ("VC Links, 15733 US-224 Findlay").

What is set aside is never a model input: the compile sends it to the suppression ledger, and training's
views (runE views.py) drop the document from both the quoting and the planning view. Pure Python.
"""
from __future__ import annotations

import re
from pathlib import PurePath
from typing import Any, Iterable

OTHER_JOB = "other_job"
THIS_DEAL = "this_deal"

#: A drawing is titled by the BUILDING, never by the job.
DRAWING_SUFFIXES = (".dwg", ".dxf", ".rvt", ".skp",
                    ".png", ".jpg", ".jpeg", ".gif", ".tif", ".tiff", ".bmp", ".heic")
#: A drawing sheet number ("SP-6", "A-101", "E1.2") in the filename.
_SHEET_NUMBER_RE = re.compile(r"(?:^|[_\s-])[A-Z]{1,2}-?\d{1,3}(?:\.\d+)?(?:-\d+)?(?=[_\s.-]|$)")
#: A request, ticket or programme number: "010329", "OX-0036", "#70598001".
_REF_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9])#?(?:[A-Za-z]{1,4}[-_ ]?)?\d{3,}(?![A-Za-z0-9])")
_REF_SEPARATORS_RE = re.compile(r"(?:\s*[-_#:|]\s*){2,}")
#: CRM fields that name where the job is, when the manifest carries them.
CRM_SITE_KEYS = ("site_address", "address", "street_address", "deal_address", "site")


def is_drawing(filenames: Iterable[Any]) -> bool:
    """Every file is a drawing: an image or CAD file, or a PDF named by a sheet number."""
    names = [str(f or "") for f in filenames if f]
    if not names:
        return False
    return all(
        n.lower().endswith(DRAWING_SUFFIXES)
        or (n.lower().endswith(".pdf") and bool(_SHEET_NUMBER_RE.search(PurePath(n).stem.upper())))
        for n in names
    )


def strip_refs(text: Any) -> str:
    """A title or deal name without its reference numbers (filing, not work)."""
    s = _REF_TOKEN_RE.sub(" ", str(text or ""))
    s = _REF_SEPARATORS_RE.sub(" - ", s)
    s = " ".join(s.split()).strip(" -_#:|")
    return s or " ".join(str(text or "").split())


def words(text: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text or "").lower()))


def deal_anchors(deal_name: Any, crm: dict[str, Any] | None = None) -> list[str]:
    """Phrases that, written in a document's own text, tie it to this deal: the deal name without its
    numbers and any site address the CRM gives. One short word is no anchor."""
    out: list[str] = []
    for raw in [strip_refs(deal_name)] + [str((crm or {}).get(k) or "") for k in CRM_SITE_KEYS]:
        w = words(raw)
        if w and (len(w.split()) >= 2 or len(w) >= 8) and w not in out:
            out.append(w)
    return out


def names_deal(texts: Iterable[Any], anchors: list[str]) -> str:
    """The anchor the documents' own text names, or ''. Read from the lines, never the filename."""
    if not anchors:
        return ""
    body = " " + " ".join(words(t) for t in texts) + " "
    for anchor in anchors:
        if f" {anchor} " in body:
            return anchor
    return ""


def set_aside(verdict: Any, *, taught: bool, confidence: float = 0.0, floor: float = 0.8,
              drawing: bool = False, named: str = "") -> bool:
    """True when a document (or a conversation) is set aside as another job (rules 1-3 above).

    ``taught``: the verdict is a person's (a Document card on the labeling page, a PM's lesson in the
    store); ``drawing``: every file is a drawing (``is_drawing``); ``named``: what ``names_deal`` found."""
    if str(verdict or "") != OTHER_JOB:
        return False
    if not taught and float(confidence or 0.0) < floor:
        return False
    if drawing and not taught:
        return False
    return not named


__all__ = ["OTHER_JOB", "THIS_DEAL", "DRAWING_SUFFIXES", "CRM_SITE_KEYS", "is_drawing", "strip_refs", "words",
           "deal_anchors", "names_deal", "set_aside"]
