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


#: A question expects a KIND of answer. "Where" wants a place, "how many"
#: wants a number. Live 010289 proposed an email recipient line as the
#: location and "Will be in touch." as whether the lock was installed.
_EXPECTS = (
    ("where|location|address|site", {"physical_site"}, re.compile(r"\d{2,}\s+\w+|\b[A-Z]{2}\s+\d{5}\b")),
    ("how many|how much|quantity|number of", {"quantity", "bom_line"}, re.compile(r"\b\d+\b")),
    ("when|what date|how soon|lead time", {"deadline", "milestone_phase", "lead_time_constraint"},
     re.compile(r"\b(?:mon|tue|wed|thu|fri|sat|sun|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)|\d{1,2}/\d{1,2}|\bweeks?\b|\bdays?\b", re.I)),
    ("who|contact", {"stakeholder", "signatory"}, re.compile(r"@|\b[A-Z][a-z]+\s+[A-Z][a-z]+\b")),
)

#: Words that carry no topic.
_STOP = frozenset("""a an the is are was were be been do does did we you they it this that these those
of to in on at for with from by and or if as what when where which who whom how any some our your their
have has had will would can could should please know let me my i us them he she""".split())


#: '"Albert Arzate" <albert@rd-systems.com>' -- a recipient, not an answer.
_ADDRESS_LINE_RE = re.compile(r"^[\"'<]?[\w .,-]+[\"'>]?\s*<[^>]+@[^>]+>\s*$")


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in _STOP}


def _answers_this_question(question: Any, cand: Any) -> bool:
    """Does the candidate actually fit the question? A reply's first sentence
    is usually not the answer -- it is "Thanks" or a recipient line."""
    q = _text(question).lower()
    ctype = _type(cand)
    ctext = _text(cand)
    # a line that is only an address, a name-and-address, or a header is chrome
    if _ADDRESS_LINE_RE.match(ctext.strip()):
        return False
    for pattern, types, shape in _EXPECTS:
        if re.search(rf"\b(?:{pattern})\b", q):
            return ctype in types or bool(shape.search(ctext))
    # no expectation we can name: demand the answer talk about the question
    return bool(_content_words(q) & _content_words(ctext))


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


def pair_within_one_line(atoms: list[Any]) -> int:
    """Pair a line that asks k times and then answers k times, in order.

    The parser can split one source line into several atoms. When it does, the
    inline rule has nothing left to match and the next-line rule refuses to look
    sideways, so a line like

        "Has the door been installed with the lock? Do we know the type of
        lock? - Defer to client ... They are intending to use a maglock."

    yields two questions nobody ever answers. Live 010288.

    Only the unambiguous shape is taken: every question first, every answer
    after, and the same number of each. Anything interleaved, lopsided or
    chattier than that is left for a human -- guessing which half of a line
    answers which question is the judgement a head should learn, not a rule.
    """
    paired = 0
    body_kinds = {"email_body_line", "hubspot_note_body", "note_field",
                  "note_field_item", "email_context"}
    groups: dict[tuple, list[Any]] = {}
    for a in atoms:
        v = _value(a)
        if str(v.get("kind") or "") not in body_kinds:
            continue
        loc = getattr(a, "locator", None)
        loc = loc if isinstance(loc, dict) else {}
        line = v.get("line", v.get("line_start", loc.get("line_start")))
        if not isinstance(line, int):
            continue
        msg = v.get("message_index", loc.get("message_index"))
        key = (str(getattr(a, "artifact_id", "") or ""),
               int(msg) if isinstance(msg, int) else 0, line)
        groups.setdefault(key, []).append(a)

    for group in groups.values():
        if len(group) < 4:          # one Q and one A on a line is rule 1's job
            continue
        flags = [_is_question(a) for a in group]
        qs = [a for a, q in zip(group, flags) if q]
        rest = [a for a, q in zip(group, flags) if not q]
        if len(qs) < 2 or len(qs) != len(rest):
            continue
        # every question before every answer -- "Q Q A A", not "Q A Q A"
        if any(flags[i] for i in range(len(flags))[len(qs):]):
            continue
        if not all(_looks_like_answer(a) for a in rest):
            continue
        if any(_value(q).get("answered") for q in qs):
            continue
        for q, a in zip(qs, rest):
            _pair(q, _text(a), source="same_line_split", answer_atom=a)
            _absorb(q, a)
            paired += 1
    return paired


def _absorb(question: Any, answer: Any) -> None:
    """Fold the answer into the question so the pair is ONE atom.

    The unsplit form of this shape -- "How many doors - 1 external access point
    [front door]" -- is a single atom, and a reader sees one row saying one
    thing. Leaving the answer standing as well would put the same words on the
    card twice: once inside the question and once as a loose line with nothing
    visibly tying it back.

    The answer is marked rather than dropped here; the compiler moves it to the
    suppression ledger, so it stays auditable the way every other removal on
    this deal does.
    """
    qt, at = _text(question), _text(answer)
    merged = f"{qt} - {at}"
    for attr in ("raw_text", "normalized_text"):
        if getattr(question, attr, None) is not None:
            setattr(question, attr, merged)
    # The answer's keys describe the fact, and the fact now lives here.
    try:
        qk = list(getattr(question, "entity_keys", None) or [])
        for k in list(getattr(answer, "entity_keys", None) or []):
            if k not in qk:
                qk.append(k)
        question.entity_keys = qk
    except Exception:
        pass
    qv = _value(question)
    qv["merged_answer_atom_id"] = str(getattr(answer, "id", "") or "")
    question.value = qv
    av = _value(answer)
    av["absorbed_into"] = str(getattr(question, "id", "") or "")
    av["absorbed_reason"] = "answer merged into its question on the same line"
    answer.value = av


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
                if not _answers_this_question(atom, cand):
                    continue
                _pair(atom, _text(cand), source="cross_message", answer_atom=cand)
                paired += 1
                break
    return paired


__all__ = ["pair_questions_with_answers", "pair_across_thread", "pair_within_one_line"]
