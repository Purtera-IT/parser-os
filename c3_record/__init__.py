"""The C3 deal record: what the base reads, and the one way to build it.

A deal is documents and lines. Whoever holds the deal -- the labeling page's
export (``ml/c3/export.py``), the parser's compile
(``app/core/c3_type_base.py``), or another company's system -- describes it in
these two shapes and calls ``build_rows``; the base then reads the same deal
however it was produced. That is the whole contract, and the only place its
rules live. Pure Python, no dependencies: the parser's worker ships it
without the model code.

Document (one per source file, email, note or meeting), in arrival order:
    doc_id       stable id of the source
    kind         what it is: ``email``, ``meeting``, ``note`` for events; a
                 file's type (``sow_draft``, ``site_survey``, ...), else ``file``
    sent_at      when it reached the deal (the email or note that carried it,
                 or the message itself), if known
    authored_at  when it was written or held, if known
    origin       who brought it: ``{direction, sender_email, sender_name, via}``
                 (a file's is the email or note that carried it), if known
    pos          its place in the compile manifest (the page export: its
                 place in the page's document list), if known
    role         the lifecycle's role sentence (the page's ``docRole``), if known

Line (one per atom), in the order it appears in its source:
    key, text, doc_id, section, speaker_role, speaker_side, atom_id, and when
    known said_by, message_author (an email line's message), speaker_name (a
    meeting line's speaker), order (``reading_order``: the same rank from the
    page export and the compile), table_row (``table_geometry``)

Only the lines the parser kept are lines: a suppressed copy, mail chrome, a
Dropped card's copy and a hand-added Missed line are never sent by the
compile, so they are never model input in training either.

Each row also gets speaker_org, doc_direction and stage_at; the sender and
stage rules are ``c3_record.sender``.

When a line entered the deal is what the base's time order, its foresight
mask and the direction of its causal links read, so every line gets one, and
it is its DOCUMENT's arrival, the same for every line of a document:

1. the document's ``sent_at``;
2. else, for an event (meeting, note, email), its ``authored_at``: a meeting
   enters the deal when it happens;
3. else the arrival of the document before it in reading order
   (``c3_record.reading.document_rank``: authored time, ties by manifest
   place ``pos``), the one order both the page export and the compile have
   (a file's ``authored_at`` is when someone wrote it, often years earlier,
   and says nothing about when the deal saw it; it only places the file);
4. else (no document, or nothing before it) the latest time in the deal.

A line's own timestamp (one message inside a thread, one utterance in a
call) is never its arrival: training never saw one, and reading a thread in
message order instead of arrival order moves the base's answers.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

from .reading import (TRAINED_KINDS, document_arrivals, document_kind, document_rank,  # noqa: F401
                      event_kind, file_kind, is_email_file, page_group, page_position, reading_order)
from .sender import SENDER_INPUTS, annotate, doc_direction, is_ours, milestones, sender, stage_at  # noqa: F401

#: Document kinds whose ``authored_at`` is the moment they entered the deal.
EVENT_KINDS = ("meeting", "note", "email")

#: Speaker values the base was trained on; anything else reads as ''.
ROLES = frozenset({"internal", "reseller", "vendor", "unknown"})
SIDES = frozenset({"ours", "theirs", "partner", "vendor"})


#: Where a line sits in its table, as the model reads it (``ml.c3.data._geometry``).
GEOMETRY_KEYS = ("bbox", "table_row", "table_col", "row_span", "col_span")
#: "<table id> r<row>"; a row the parser stamped as a float ("r3.0") is the same row.
_TABLE_REF = re.compile(r"^(.*?)\s+r(\d+)(?:\.0+)?$")


def table_geometry(table_ref: Any) -> dict[str, int]:
    """``{"table_row": n}`` for a table-row line, else ``{}``.

    ``table_ref`` is the one string both sides already stamp on a table row:
    the labeling page's ``tableRef`` and the parser's ``_atom_table_ref``
    (``"<table id> r<row>"``). The row counts from 1, so the first row of a
    table is not read as "not in a table" (0)."""
    m = _TABLE_REF.match(str(table_ref or "").strip())
    return {"table_row": int(m.group(2)) + 1} if m else {}


#: Fold stages whose copies the deal's read holds besides the compile's kept
#: lines (the parser's ``head_fold.PRE_HEAD_FOLDS``: ``c3_type_base.also_read``
#: hands every folded copy of these stages to the heads, chrome-classed ones
#: too). A stage's drop (no survivor) is never read.
ALSO_READ_STAGES = frozenset({"quoted_history_dedup", "quoted_chatter_dedup", "pasted_note_dedup",
                              "duplicate_atom_collapse", "pre_classify_dedup"})


def read_with_deal(suppressed: Any) -> bool:
    """True when a line is a model input: a kept line (no ``suppressed``), or
    a copy a pre-read fold set aside. ``suppressed`` is the labeling page's
    marker (``{kind: copy|chrome|dropped, stage}``) or an envelope ledger row
    (``{kind: fold|drop, stage}``)."""
    if not suppressed:
        return True
    if not isinstance(suppressed, dict):
        return False
    kind = str(suppressed.get("kind") or "").lower()
    return str(suppressed.get("stage") or "") in ALSO_READ_STAGES and kind not in ("drop", "dropped")


def is_deal_kit(doc_kind: Any) -> bool:
    """Our own pricing workbook, however the document kind is spelled. The
    base is never scored on, and never acts on, a Deal Kit line."""
    return "dealkit" in re.sub(r"[^a-z]", "", str(doc_kind or "").lower())


def document_times(documents: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Arrival per document (rules 1-3), filled forward in reading order
    (``reading.document_arrivals``), whatever order the list is in.

    Each document is ``{"doc_id", "kind", "sent_at", "authored_at"}`` and,
    when known, ``pos`` (its manifest place; the list index otherwise); the
    times are returned as given (ISO strings or numbers), never converted."""
    return document_arrivals(documents, EVENT_KINDS)


def speaker(role: Any, side: Any) -> tuple[str, str]:
    """A speaker the base knows, else ''. A side without a role, or a role it
    never saw, is a line that speaks for nobody it was taught."""
    r, s = str(role or "").strip().lower(), str(side or "").strip().lower()
    return (r if r in ROLES else ""), (s if s in SIDES else "")


def build_rows(lines: Iterable[dict[str, Any]], documents: Iterable[dict[str, Any]],
               *, latest_fallback: Any = None) -> list[dict[str, Any]]:
    """The base's rows: each line with its document's arrival and kind.

    ``order`` is the line's position in ``lines``. A line whose document has
    no arrival takes the latest arrival in the deal (rule 4); with no arrival
    anywhere, ``latest_fallback`` (export.py refuses that deal instead)."""
    docs = [d for d in documents if isinstance(d, dict)]
    times = document_times(docs)
    kinds = {str(d.get("doc_id")): str(d.get("kind") or "").strip().lower() for d in docs}
    latest = max(times.values(), key=_sort_key, default=latest_fallback)
    rows = []
    for i, ln in enumerate(lines):
        doc = str(ln.get("doc_id") or "")
        role, side = speaker(ln.get("speaker_role"), ln.get("speaker_side"))
        row = {
            "key": str(ln.get("key") or ""), "text": str(ln.get("text") or ""),
            "entered_at": times.get(doc, latest), "doc_id": doc,
            "doc_kind": kinds.get(doc, str(ln.get("doc_kind") or "")),
            "section": str(ln.get("section") or ""), "order": float(ln.get("order", i)),
            "speaker_role": role, "speaker_side": side, "atom_id": str(ln.get("atom_id") or ""),
        }
        # Who sent it and where the deal stood (c3_record.sender): a speaker
        # given outright stands as the line's own.
        said = ln.get("said_by") or ({"side": side, "role": role} if side else None)
        row.update({k: ln.get(k) for k in SENDER_INPUTS if ln.get(k)}, said_by=said)
        row.update({k: ln[k] for k in GEOMETRY_KEYS if ln.get(k) not in (None, "", [])})
        rows.append(row)
    annotate(rows, [{"doc_id": str(d.get("doc_id") or ""), "kind": kinds.get(str(d.get("doc_id")), ""),
                     "arrival": times.get(str(d.get("doc_id"))), "origin": d.get("origin"),
                     "role": d.get("role")} for d in docs])
    return rows


def _sort_key(t: Any) -> tuple[int, Any]:
    """Numbers and ISO strings both compare within their own kind."""
    return (0, float(t)) if isinstance(t, (int, float)) else (1, str(t))


__all__ = ["EVENT_KINDS", "ROLES", "SIDES", "GEOMETRY_KEYS", "TRAINED_KINDS", "is_deal_kit", "document_times",
           "speaker", "build_rows", "table_geometry", "annotate", "sender", "doc_direction", "is_ours",
           "milestones", "stage_at", "event_kind", "file_kind", "document_kind", "is_email_file", "document_rank",
           "document_arrivals", "page_position", "page_group", "reading_order", "ALSO_READ_STAGES",
           "read_with_deal"]
