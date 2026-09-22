"""A question and its answer are one fact.

Live 010289 asked three questions and got three answers, and the parser filed
six atoms: "Has the door been installed with the lock?" as an open question,
"Defer to client -- my understanding is client was working with installer" as
metadata, and nothing joining them. The PM then sees an open question that is
not open, and an answer with no question.

Three shapes, all of them in that one deal:

* inline -- "How many doors - 1 external access point [front door]"
* the next line -- a question, then the reply typed under it
* across the thread -- "Where is this site located?" answered two mails later

A pair never deletes anything: the question is retyped ``answered_question``
and carries the answer, the answer atom keeps its own type (it is usually
scope in its own right) and points back. A cross-message pair is a proposal --
it lands ``needs_review`` for a PM to confirm, because guessing which sentence
of a reply answers which question is exactly the judgement a head should
learn.
"""
from __future__ import annotations

import re
from typing import Any

#: "How many doors - 1 external access point" -- the answer typed after a dash
#: on the same line. Not an em-dash sentence ("the kit -- which we supply --").
_INLINE_ANSWER_RE = re.compile(r"^(?P<q>[^?]{6,120}?)\s+[-–—:]\s+(?P<a>\S.{2,200})$")

#: Words that make a line a question even without a mark.
_QUESTION_LEAD_RE = re.compile(
    r"^\s*(?:who|what|when|where|which|why|how|do|does|did|is|are|was|were|can|could|"
    r"will|would|should|have|has|any|please confirm|let me know|lmk)\b",
    re.I,
)

_FOLLOW_UP_LIMIT = 2


def _type(atom: Any) -> str:
    at = getattr(atom, "atom_type", None)
    return at.value if hasattr(at, "value") else str(at or "")


def _value(atom: Any) -> dict:
    v = getattr(atom, "value", None)
    return v if isinstance(v, dict) else {}


def _text(atom: Any) -> str:
    return " ".join(str(getattr(atom, "raw_text", "") or "").split())


def _is_question(atom: Any) -> bool:
    if _type(atom) in {"open_question", "internal_question"}:
        return True
    t = _text(atom)
    return t.endswith("?") and bool(_QUESTION_LEAD_RE.match(t))


#: Who someone IS never answers a question about the work. A signature block
#: sits at the end of every mail, so without this the first "answer" to any
#: question is the sender's own contact card (live 010289: "Where is this site
#: located?" -> "Alec Burns | Senior Client Executive").
_IDENTITY_TYPES = {"stakeholder", "signatory", "entity"}


def _looks_like_answer(atom: Any) -> bool:
    """A statement that could answer something: not another question, not a
    list item, not chatter, not a signature, and it says something."""
    if _is_question(atom) or _text(atom).endswith("?"):
        return False
    if _type(atom) in _IDENTITY_TYPES:
        return False
    v = _value(atom)
    if v.get("chatter") or v.get("list_item") or str(v.get("kind") or "") in {"person", "email_header", "quoted_message_header"}:
        return False
    return len(_text(atom).split()) >= 2


def _pair(question: Any, answer_text: str, *, source: str, answer_atom: Any | None = None) -> None:
    """Join them in place: the question becomes answered, the answer points back."""
    from app.core.schemas import ReviewStatus

    qv = _value(question)
    qv["question"] = qv.get("question") or _text(question)
    qv["answer"] = answer_text
    qv["answer_source"] = source
    if answer_atom is not None:
        qv["answer_atom_id"] = str(getattr(answer_atom, "id", "") or "")
    question.value = qv
    # ``answered_question`` is a label-space type, not a prod enum member, so
    # the atom keeps open_question and carries the parser's read for the
    # labeler to confirm.
    qv["answered"] = True
    qv["proposed_type"] = "answered_question"
    flags = list(getattr(question, "review_flags", None) or [])
    for f in ("answered_question", f"answer_from:{source}"):
        if f not in flags:
            flags.append(f)
    question.review_flags = flags
    if source == "cross_message":
        try:
            question.review_status = ReviewStatus.needs_review
        except Exception:
            pass
    if answer_atom is not None:
        av = _value(answer_atom)
        av["answers_question_id"] = str(getattr(question, "id", "") or "")
        av["answers_question"] = qv["question"]
        answer_atom.value = av


def pair_questions_with_answers(atoms: list[Any]) -> int:
    """Join questions to their answers in place. Returns the number paired."""
    paired = 0
    by_artifact: dict[str, list[Any]] = {}
    for a in atoms:
        by_artifact.setdefault(str(getattr(a, "artifact_id", "") or ""), []).append(a)

    #: Only prose the author typed can be paired. A question the compiler
    #: MINTED about its own processing ("Referenced inline equipment image
    #: could not be read") has no answer in the text and paired with whatever
    #: sat next to it.
    body_kinds = {"email_body_line", "hubspot_note_body", "note_field", "note_field_item", "email_context"}

    def _is_body(a: Any) -> bool:
        return str(_value(a).get("kind") or "") in body_kinds

    def _pos_of(a: Any) -> tuple | None:
        v = _value(a)
        loc = getattr(a, "locator", None)
        loc = loc if isinstance(loc, dict) else {}
        msg = v.get("message_index", loc.get("message_index"))
        line = v.get("line", v.get("line_start", loc.get("line_start")))
        if not isinstance(line, int):
            return None
        return (int(msg) if isinstance(msg, int) else 0, line)

    def _where(a: Any) -> tuple:
        """Where the line sits in the document. Later stages reorder the atom
        list, so "the next line" has to mean the next line of the SOURCE --
        pairing on list order answered the wrong question (010289)."""
        v = _value(a)
        loc = getattr(a, "locator", None)
        loc = loc if isinstance(loc, dict) else {}
        msg = v.get("message_index", loc.get("message_index"))
        line = v.get("line", v.get("line_start", loc.get("line_start")))
        return (int(msg) if isinstance(msg, int) else 0, int(line) if isinstance(line, int) else 0)

    for group in by_artifact.values():
        group = sorted(group, key=_where)
        for i, atom in enumerate(group):
            if not _is_question(atom) or _value(atom).get("answered"):
                continue
            text = _text(atom)

            # 1) the answer is on the same line, after a dash or colon
            m = _INLINE_ANSWER_RE.match(text) if _is_body(atom) else None
            if m and not m.group("a").endswith("?"):
                _pair(atom, m.group("a").strip(), source="same_line")
                _value(atom)["question"] = m.group("q").strip().rstrip("-:").strip()
                paired += 1
                continue

            # 2) the next line or two of the SAME message is the reply. Both
            # sides must know where they sit: without a line number "next" is
            # a guess, and a guess pairs a question with a sign-off.
            qpos = _pos_of(atom)
            if qpos is None or not _is_body(atom):
                continue
            for j in range(i + 1, min(i + 1 + _FOLLOW_UP_LIMIT, len(group))):
                cand = group[j]
                if _is_question(cand):
                    break
                cpos = _pos_of(cand)
                if cpos is None or cpos[0] != qpos[0] or not (0 < cpos[1] - qpos[1] <= 4):
                    continue
                if _looks_like_answer(cand):
                    _pair(atom, _text(cand), source="next_line", answer_atom=cand)
                    paired += 1
                    break
    return paired


def pair_across_thread(atoms: list[Any]) -> int:
    """A question answered by a LATER message in the same thread. Proposed, not
    asserted: it lands needs_review for a PM to confirm."""
    threads: dict[str, list[Any]] = {}
    for a in atoms:
        et = _value(a).get("email_thread")
        if isinstance(et, dict) and et.get("thread_id"):
            threads.setdefault(str(et["thread_id"]), []).append(a)

    paired = 0
    for group in threads.values():
        def _pos(a: Any) -> tuple:
            et = _value(a).get("email_thread") or {}
            return (int(et.get("thread_index") or 0), str(getattr(a, "id", "")))

        ordered = sorted(group, key=_pos)
        for i, atom in enumerate(ordered):
            if not _is_question(atom) or _value(atom).get("answered"):
                continue
            here = _pos(atom)[0]
            asker = str((_value(atom).get("email_thread") or {}).get("sender") or "").lower()
            for cand in ordered[i + 1:]:
                cet = _value(cand).get("email_thread") or {}
                # Only the DIRECT reply, and only from the other person. Any
                # looser rule guesses: the first statement of message six is
                # not the answer to a question asked in message two.
                if int(cet.get("thread_index") or 0) != here + 1:
                    continue
                if str(cet.get("sender") or "").lower() == asker:
                    continue
                if _value(cand).get("quoted") or not _looks_like_answer(cand):
                    continue
                _pair(atom, _text(cand), source="cross_message", answer_atom=cand)
                paired += 1
                break
    return paired


__all__ = ["pair_questions_with_answers", "pair_across_thread"]
