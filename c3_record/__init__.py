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
    role         the lifecycle's role sentence (the page's ``docRole``), if known
    filename     the source file's name, if known

Line (one per atom), in the order it appears in its source:
    key, text, doc_id, section, speaker_role, speaker_side, atom_id, and when
    known said_by / said_to, message_author (an email line's message) and
    speaker_name (a meeting line's speaker)

Each row also gets speaker_org, doc_direction and stage_at, and its
speaker_side / speaker_role are re-read from all of the above; the sender
and stage rules are ``c3_record.sender``, the same rules the training export
wrote them by. A field that cannot be derived is ''.

When a line entered the deal is what the base's time order, its foresight
mask and the direction of its causal links read, so every line gets one, and
it is its DOCUMENT's arrival, the same for every line of a document:

1. the document's ``sent_at``;
2. else, for an event (meeting, note, email), its ``authored_at``: a meeting
   enters the deal when it happens;
3. else the arrival of the document before it in arrival order (a file's
   ``authored_at`` is when someone wrote it, often years earlier, and says
   nothing about when the deal saw it);
4. else (no document, or nothing before it) the latest time in the deal.

A line's own timestamp (one message inside a thread, one utterance in a
call) is never its arrival: training never saw one, and reading a thread in
message order instead of arrival order moves the base's answers.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

from .sender import (SENDER_INPUTS, annotate, crm_marks, crm_stage_at, doc_direction, is_ours,  # noqa: F401
                     milestones, sender, stage_at)

#: Document kinds whose ``authored_at`` is the moment they entered the deal.
EVENT_KINDS = ("meeting", "note", "email")

#: Speaker values the base was trained on; anything else reads as ''.
ROLES = frozenset({"internal", "reseller", "vendor", "unknown"})
SIDES = frozenset({"ours", "theirs", "partner", "vendor"})

#: A table row's place (c3_record.context.table_fields; build.py's (d)); table_header follows from them.
TABLE_KEYS = ("table_id", "table_row_index", "sheet")
#: Extraction facts of a line, sent as given when set: its depth in quoted reply
#: history (c3_record.context.quoted_depth) and the line it copies (the parser's duplicate_of).
LINE_FACT_KEYS = ("quoted_depth", "copy_of")
#: The sender and stage fields every row carries (``c3_record.sender.annotate``).
SENDER_FIELDS = ("speaker_side", "speaker_role", "speaker_org", "doc_direction", "stage_at")


def is_deal_kit(doc_kind: Any) -> bool:
    """Our own pricing workbook, however the document kind is spelled. The
    base is never scored on, and never acts on, a Deal Kit line."""
    return "dealkit" in re.sub(r"[^a-z]", "", str(doc_kind or "").lower())


def document_times(documents: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Arrival per document (rules 1-3), filled forward in arrival order.

    Each document is ``{"doc_id", "kind", "sent_at", "authored_at"}``; the
    times are returned as given (ISO strings or numbers), never converted."""
    docs = [d for d in documents if isinstance(d, dict)]
    known = [d.get("sent_at") or (d.get("authored_at") if str(d.get("kind") or "") in EVENT_KINDS else None)
             for d in docs]
    prev = next((t for t in known if t), None)
    out: dict[str, Any] = {}
    for d, t in zip(docs, known):
        prev = t or prev
        if d.get("doc_id") and prev:
            out[str(d["doc_id"])] = prev
    return out


def speaker(role: Any, side: Any) -> tuple[str, str]:
    """A speaker the base knows, else ''. A side without a role, or a role it
    never saw, is a line that speaks for nobody it was taught."""
    r, s = str(role or "").strip().lower(), str(side or "").strip().lower()
    return (r if r in ROLES else ""), (s if s in SIDES else "")


def build_rows(lines: Iterable[dict[str, Any]], documents: Iterable[dict[str, Any]],
               *, latest_fallback: Any = None, timeline: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """The base's rows: each line with its document's arrival and kind.

    ``order`` is the line's position in ``lines``. A line whose document has
    no arrival takes the latest arrival in the deal (rule 4); with no arrival
    anywhere, ``latest_fallback`` (export.py refuses that deal instead).
    ``timeline``: the deal's ``crm.stage_timeline`` (``sender.crm_marks``)."""
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
        # Where the line sits (c3_record.context): its table and sheet, as the training build stamps them.
        row.update({k: ln[k] for k in TABLE_KEYS if ln.get(k) not in (None, "", [])})
        row.update({k: ln[k] for k in LINE_FACT_KEYS if ln.get(k)})
        rows.append(row)
    try:
        annotate(rows, [{"doc_id": str(d.get("doc_id") or ""), "kind": kinds.get(str(d.get("doc_id")), ""),
                         "arrival": times.get(str(d.get("doc_id"))), "origin": d.get("origin"),
                         "role": d.get("role"), "filename": d.get("filename"), "sent_at": d.get("sent_at")}
                        for d in docs], timeline=timeline)
    except Exception:  # noqa: BLE001 - a sender that cannot be read is '', never a failed read
        for r in rows:
            r.setdefault("speaker_org", "")
            r.setdefault("doc_direction", "")
            r.setdefault("stage_at", "")
    for r in rows:
        for k in SENDER_INPUTS:
            r.pop(k, None)
        for k in SENDER_FIELDS:
            r[k] = str(r.get(k) or "")
    return rows


def _sort_key(t: Any) -> tuple[int, Any]:
    """Numbers and ISO strings both compare within their own kind."""
    return (0, float(t)) if isinstance(t, (int, float)) else (1, str(t))


__all__ = ["EVENT_KINDS", "ROLES", "SIDES", "SENDER_FIELDS", "is_deal_kit", "document_times", "speaker",
           "build_rows", "annotate", "sender", "doc_direction", "is_ours", "milestones", "stage_at", "crm_marks",
           "crm_stage_at"]
