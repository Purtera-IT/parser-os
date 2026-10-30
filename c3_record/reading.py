"""Reading order, document kind and arrival order: one rule for both sides.

The training export (``ml/c3/export.py``, from the labeling page's export) and
the parser's compile (``app/core/c3_type_base.py``, from the compile's atoms)
each describe the same deal with different fields. These functions take the
inputs both can produce, so a line gets the same ``order``, its document the
same ``kind`` and the same filled-in arrival, however the deal was produced.
Pure Python, no dependencies.

Documents, in reading order (``document_rank``): by authored time (a
document with none reads after every dated one), ties by the document's place
in the compile manifest (``pos``). The labeling page lists documents as a time
story in which an undated document keeps its manifest place, so its own
document list order is that place for the page export.

Lines inside a document (``reading_order``): the labeling page's own order
(atom-labeling.js buildWalk: message, page, block, line, sentence, column,
envelope order), except that a workbook's sheets are read one after another,
each in row order, instead of row 1 of every sheet, then row 2. A sheet comes
in the order its first row reads. The page export gives each line its place in
that order (``position``); the compile computes the same key from the atom's
locator (``page_position``).

A document's kind (``document_kind``): an email, note or meeting is that event;
a file is its lifecycle type (the document lifecycle table, looked up by the
file's bytes) when the base trained on that type, else ``deal_kit`` /
``estimating_calc`` / ``site_list`` by its name, else ``file``.
"""
from __future__ import annotations

import math
import re
from typing import Any, Iterable, Sequence

#: Every document kind the base trained on: the lowercased lifecycle types
#: (document_lifecycle/taxonomy.py) the training files carry, plus the three
#: event kinds and ``file``, training's own bucket for a document of no known
#: type. Anything else reads as ``file``, so the model never sees a tag it
#: never trained on.
TRAINED_KINDS = frozenset({
    "email", "meeting", "note", "file", "sow_draft", "deal_kit", "site_survey",
    "estimating_calc", "sow_signed", "site_list", "runbook", "cost_breakdown", "w9",
    "purchase_order", "rfq", "rate_card", "engineering_design", "install_instructions",
    "credit_application", "sow", "intake_request", "checklist", "quote", "bom",
    "vendor_datasheet", "floor_plan", "case_manifest", "sow_template", "nda", "crm_note",
})
_EVENTS = ("email", "note", "meeting")
_CALC_NAME = re.compile(r"estimat\w*calc")
#: A file named as a site list ("Site List Project Bastion.xlsx") is one.
_SITE_LIST_NAME = re.compile(r"site(list|roster)|locationlist|storelist")
#: atom-labeling.js isEmailDoc.
_EMAIL_TYPES = frozenset({"email", "msg", "mbox"})
_EMAIL_FILE = re.compile(r"\.(eml|msg|mbox)$", re.I)


def file_kind(filename: Any, lifecycle_type: Any = None) -> str:
    """A file's kind: its lifecycle type when the base trained on it (any
    other type is ``file``); with no type, ``deal_kit``, ``estimating_calc``
    or ``site_list`` by the file's name; else ``file``."""
    kind = str(lifecycle_type or "").strip().lower()
    if kind:
        return kind if kind in TRAINED_KINDS else "file"
    name = re.sub(r"[^a-z]", "", str(filename or "").lower())
    if "dealkit" in name:
        return "deal_kit"
    if _CALC_NAME.search(name):
        return "estimating_calc"
    if _SITE_LIST_NAME.search(name):
        return "site_list"
    return "file"


#: Our own estimating calculator by what it says: a spreadsheet that lays out
#: internal economics -- a margin (GM, gross margin, margin %, markup) next to
#: a cost build. A customer's or a vendor's price sheet states prices, never
#: our margin. 010245's five "Pricing Sheet ADJ" workbooks ("LABOR GM",
#: "UNIT: COST", "materials inflated by 10% for bidding") were named neither
#: "Deal Kit" nor "estimating calc" and fed 1,846 of 2,402 brief lines.
_SHEET_FILE = re.compile(r"\.(xlsx|xlsm|xls|xlsb|ods|csv)$", re.I)
_MARGIN_WORDS = re.compile(r"\bGM\b|\bgross\s+margin|\bmargin\s*%|%\s*margin|\bmark-?up\b|\bGP\s*%", re.I)
_COST_WORDS = re.compile(r"\b(?:unit|total|labor|labour|material|extended|our)\s+cost\b|\bcost\s*(?:\||:|$)|"
                         r"\bcost\s+(?:per|each|ea)\b", re.I)
#: Lines of each kind a workbook must have: one stray "markup" in a vendor
#: quote is no calculator.
CALC_CONTENT_MIN = 3


def estimating_calc_by_content(texts: Iterable[Any], filename: Any = "", *, spreadsheet: bool | None = None) -> bool:
    """True when a spreadsheet's own lines show it is our estimating calc
    (``file_kind`` would say ``estimating_calc`` had it been named one): at
    least ``CALC_CONTENT_MIN`` lines naming a margin and as many naming a cost.
    ``spreadsheet``: the caller's own reading of the file type; else the name."""
    sheet = spreadsheet if spreadsheet is not None else bool(_SHEET_FILE.search(str(filename or "").strip()))
    if not sheet:
        return False
    margin = cost = 0
    for t in texts:
        s = str(t or "")
        margin += bool(_MARGIN_WORDS.search(s))
        cost += bool(_COST_WORDS.search(s))
        if margin >= CALC_CONTENT_MIN and cost >= CALC_CONTENT_MIN:
            return True
    return False


def document_kind(event_kind: Any, lifecycle_type: Any = None, filename: Any = "") -> str:
    """``email`` / ``note`` / ``meeting`` when the document is that event,
    else ``file_kind``."""
    ev = str(event_kind or "").strip().lower()
    return ev if ev in _EVENTS else file_kind(filename, lifecycle_type)


def is_email_file(artifact_type: Any, filename: Any) -> bool:
    """The labeling page's email document (atom-labeling.js isEmailDoc)."""
    return (str(artifact_type or "").strip().lower() in _EMAIL_TYPES
            or bool(_EMAIL_FILE.search(str(filename or ""))))


_HS_KIND = re.compile(r"-hs-(note|meeting|call|email)-", re.I)


def is_note_meta(value: Any) -> bool:
    """HubSpot's own record of a note (atom-labeling.js noteMetaOf)."""
    v = value if isinstance(value, dict) else {}
    return v.get("kind") == "hubspot_note_meta" or bool(v.get("hubspot_note_id") and v.get("source") == "hubspot_note")


def event_kind(artifact_type: Any, filename: Any, values: Iterable[Any] = ()) -> str:
    """``email``, ``note`` or ``meeting`` when a document is that event, else
    '' (a file): the labeling page's rule (atom-labeling.js docKindOf), from
    the document's type, its file name and its own atoms' values."""
    if is_email_file(artifact_type, filename):
        return "email"
    if any(is_note_meta(v) for v in values):
        return "note"
    m = _HS_KIND.search(str(filename or ""))
    if m:
        k = m.group(1).lower()
        return k if k in ("email", "note") else "meeting"
    return "meeting" if str(artifact_type or "").strip().lower() == "transcript" else ""


def _epoch(t: Any) -> float | None:
    from .sender import epoch

    return epoch(t)


def document_rank(documents: Iterable[dict[str, Any]]) -> dict[str, int]:
    """Each document's place in reading order: by ``authored_at`` (undated
    last), ties by ``pos`` (the manifest place; the list index when absent)."""
    docs = [d for d in documents if isinstance(d, dict)]

    def key(nd: tuple[int, dict[str, Any]]) -> tuple[float, float]:
        n, d = nd
        t = _epoch(d.get("authored_at"))
        pos = d.get("pos")
        return (t if t is not None else math.inf, float(pos) if isinstance(pos, (int, float)) else float(n))

    out: dict[str, int] = {}
    for r, (_, d) in enumerate(sorted(enumerate(docs), key=key)):
        out.setdefault(str(d.get("doc_id") or ""), r)
    return out


def document_arrivals(documents: Iterable[dict[str, Any]], event_kinds: Sequence[str] = _EVENTS
                      ) -> dict[str, Any]:
    """Arrival per document: its ``sent_at``; else, for an event, its
    ``authored_at``; else the arrival of the document before it in reading
    order (``document_rank``); else (nothing before it: no evidence of when it
    arrived) the LATEST time in the deal, any arrival or authored time, so an
    undated, uncarried document never enters a quoting read early (leak-safe;
    the Platform as_of cut's ``documentArrivals`` is the same rule). Times are
    returned as given."""
    docs = [d for d in documents if isinstance(d, dict)]
    rank = document_rank(docs)
    docs.sort(key=lambda d: rank.get(str(d.get("doc_id") or ""), len(rank)))
    known = [d.get("sent_at") or (d.get("authored_at") if str(d.get("kind") or "") in event_kinds else None)
             for d in docs]
    latest, latest_t = None, None
    for t in known + [d.get("authored_at") for d in docs]:
        e = _epoch(t) if t else None
        if e is not None and (latest_t is None or e > latest_t):
            latest, latest_t = t, e
    prev = None
    out: dict[str, Any] = {}
    for d, t in zip(docs, known):
        prev = t or prev
        at = prev or latest
        if d.get("doc_id") and at:
            out[str(d["doc_id"])] = at
    return out


# --------------------------------------------------------------- lines

_MISSING = object()


def _js_num(v: Any, fallback: Any) -> Any:
    """JavaScript's ``Number(v)`` with the page's ``num(v, fallback)``:
    a missing key is the fallback, ``null`` and '' are 0."""
    if v is _MISSING:
        return fallback
    if v is None:
        return 0
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return v if math.isfinite(v) else fallback
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return 0
        try:
            x = float(s)
        except ValueError:
            return fallback
        return x if math.isfinite(x) else fallback
    return fallback


_ENTITY = re.compile(r"&(amp|lt|gt|quot|#39|nbsp);")
_ENTITIES = {"amp": "&", "lt": "<", "gt": ">", "quot": '"', "#39": "'", "nbsp": " "}


def _fold(s: Any) -> str:
    s = _ENTITY.sub(lambda m: _ENTITIES[m.group(1)], str(s or "")).lower()
    s = re.sub(r"^[\s\-*•–—]+", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _offset_in_container(locator: dict[str, Any], value: dict[str, Any], text: str) -> int | None:
    for c in (locator.get("line_text"), value.get("line"), value.get("source_line"), value.get("paragraph")):
        if not isinstance(c, str) or not c:
            continue
        needle = _fold(text)
        if not needle:
            return None
        hay = _fold(c)
        at = hay.find(needle)
        if at >= 0:
            return at
        head = needle[:40]
        at2 = hay.find(head) if len(head) >= 12 else -1
        if at2 >= 0:
            return at2
    return None


def message_index(locator: dict[str, Any], value: dict[str, Any]) -> int | None:
    """The message of an email file a line was written in (0: the file's own
    message; 1, 2, ...: older messages it quotes), as the page reads it."""
    et = value.get("email_thread") if isinstance(value.get("email_thread"), dict) else None
    m = et.get("message") if et and isinstance(et.get("message"), dict) else None
    if m and m.get("index") is not None:
        index = m["index"]
    elif locator.get("message_index") is not None:
        index = locator["message_index"]
    else:
        index = value.get("message_index", _MISSING)
    if index is None or index == "" or index is _MISSING:
        return None
    n = _js_num(index, None)
    return None if n is None else int(n) if float(n).is_integer() else n


def page_position(locator: Any, value: Any, text: Any, *, email: bool) -> tuple[Any, ...]:
    """A parser atom's place inside its document, as the labeling page sorts
    it (atom-labeling.js buildWalk rowOf / rows.sort): message (an email
    file's deepest quote first), page, block, line, sentence, column.
    ``email``: the document is an email file (``is_email_file``)."""
    loc = locator if isinstance(locator, dict) else {}
    v = value if isinstance(value, dict) else {}
    g = lambda k: loc.get(k, _MISSING)  # noqa: E731
    msg = -max(0, message_index(loc, v) or 0) if email else 0
    at = _offset_in_container(loc, v, str(text or "").strip())
    return (msg,
            _js_num(g("page"), _js_num(g("row"), 0)),
            _js_num(g("block_index"), _js_num(g("row_index"), _js_num(g("row"), 0))),
            _js_num(g("line_start"), _js_num(g("qa_chunk_index"), 0)),
            _js_num(g("sentence_index"), 0),
            _js_num(g("char_start"), at if at is not None else 0))


def page_group(locator: Any) -> str | None:
    """The sheet a line sits on: the page export's ``page`` (the locator's
    page, else its sheet) when that is a sheet name."""
    loc = locator if isinstance(locator, dict) else {}
    page = loc.get("page") if loc.get("page") is not None else loc.get("sheet")
    return page if isinstance(page, str) else None


def reading_order(lines: Sequence[dict[str, Any]], documents: Iterable[dict[str, Any]]) -> list[int]:
    """Each line's rank in the deal's reading order.

    ``lines``, in the compile's (envelope's) order: ``{"doc_id", "within",
    "group"}``. ``within`` is the line's place inside its document, the same
    order on both sides (the page export's ``position``, or ``page_position``
    from a parser atom); ``group`` is its sheet (the page export's ``page``
    when it is a string, or ``page_group``), else None.
    ``documents``: ``{"doc_id", "authored_at", "pos"}`` (``document_rank``).

    Documents in reading order (one the list lacks reads after them all, by
    its first line); inside a document, its sheets one after another, each
    placed by its first line; then ``within``; then the compile's order."""
    rank = document_rank(documents)
    first: dict[str, int] = {}
    for i, ln in enumerate(lines):
        first.setdefault(str(ln.get("doc_id") or ""), i)

    def doc_key(d: str) -> tuple[int, int]:
        return (0, rank[d]) if d in rank else (1, first[d])

    starts: dict[tuple[str, Any], tuple[Any, int]] = {}
    for i, ln in enumerate(lines):
        k = (str(ln.get("doc_id") or ""), ln.get("group"))
        here = (ln.get("within"), i)
        if k not in starts or here < starts[k]:
            starts[k] = here
    by_doc: dict[str, list[tuple[Any, int, Any]]] = {}
    for (d, grp), st in starts.items():
        by_doc.setdefault(d, []).append((st[0], st[1], grp))
    group_rank: dict[tuple[str, Any], int] = {}
    for d, gs in by_doc.items():
        for r, (_, _, grp) in enumerate(sorted(gs, key=lambda x: (x[0], x[1]))):
            group_rank[(d, grp)] = r

    def key(i: int) -> tuple[Any, ...]:
        ln = lines[i]
        d = str(ln.get("doc_id") or "")
        return (doc_key(d), group_rank[(d, ln.get("group"))], ln.get("within"), i)

    out = [0] * len(lines)
    for r, i in enumerate(sorted(range(len(lines)), key=key)):
        out[i] = r
    return out


__all__ = ["TRAINED_KINDS", "file_kind", "document_kind", "is_email_file", "is_note_meta", "event_kind",
           "document_rank",
           "document_arrivals", "message_index", "page_position", "page_group", "reading_order"]
