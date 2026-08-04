"""PM answers → evidence atoms.

When a PM answers an open question in the brief, they are not just silencing a
card — they are stating deal truth that usually exists nowhere in the documents
("the customer confirmed they patch the drywall"). Until this module, that text
went into a transient string used to gate the next question pass and was then
discarded: it never became an atom, so it never reached the packets, the
reconciliation pass, or the SOW.

Turning the answer into a real :class:`EvidenceAtom` buys both halves of the
loop at once:

* **This compile** — the answer is evidence like any other, so the SOW is built
  from it, it can be cited, and (at ``pm_confirmed`` authority, rank 95) it
  *governs* over a stale document line instead of becoming one more competing
  claim.
* **Next compile** — it is a human-authored, known-correct example carrying the
  same fields every head consumes, so it is gold for the type/facet heads
  rather than a special case bolted onto the side.

Pure and side-effect-free: no I/O, deterministic ids, safe to unit-test.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable

from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    EvidenceAtom,
    ReviewStatus,
    SourceRef,
)

# A human typed it and stood behind it. High, but not 1.0 — a PM can still be
# wrong, and leaving headroom keeps calibration honest.
PM_ANSWER_CONFIDENCE = 0.95

PARSER_NAME = "pm_answer"
PARSER_VERSION = "pm_answer_v1"

# Answers that resolve nothing. These still settle the question card (that is
# the feedback ledger's job) but must never enter evidence as fact.
_EMPTY_ANSWER = re.compile(
    r"^(n/?a|none|no|yes|ok|okay|tbd|tba|unknown|\?+|-+|\.+)$",
    re.IGNORECASE,
)

_MIN_ANSWER_CHARS = 3


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _atom_id(project_id: str, key: str, answer: str) -> str:
    h = hashlib.sha1(f"{project_id}|{key}|{answer}".encode("utf-8")).hexdigest()[:16]
    return f"atm_pm_{h}"


def is_substantive_answer(answer: str) -> bool:
    """True when the answer carries information worth entering as evidence.

    A bare "yes" is a real reply to the PM's card but is meaningless stripped of
    its question, and an atom is read on its own downstream — so it is settled
    in the ledger and kept out of evidence.
    """
    a = _norm(answer)
    if len(a) < _MIN_ANSWER_CHARS:
        return False
    return not _EMPTY_ANSWER.match(a)


def build_claim(question: str, answer: str) -> str:
    """Join the two verbatim strings into a self-contained claim.

    Deliberately a join, never a paraphrase. An atom is read without its
    surrounding card, so a lone "the customer does" is useless to the SOW and to
    an embedding — but rewriting the pair into a fluent sentence would be
    inventing wording nobody approved. Labelling both halves keeps it faithful
    and self-contained.
    """
    q = _norm(question).rstrip("?")
    a = _norm(answer)
    if not q:
        return a
    return f"{q}? PM answer: {a}"


def pm_answer_to_atom(
    *,
    project_id: str,
    question: str,
    answer: str,
    rule_id: str = "",
    deal_id: str = "",
    answered_at: str = "",
    actor: str = "",
    artifact_id: str = "",
) -> EvidenceAtom | None:
    """One answered question → one atom, or ``None`` if it carries no facts."""
    if not is_substantive_answer(answer):
        return None
    answer_n = _norm(answer)
    claim = build_claim(question, answer_n)
    key = rule_id or _norm(question) or answer_n
    art_id = artifact_id or f"art_pm_answers_{(deal_id or project_id)[:24]}"
    atom_id = _atom_id(project_id, key, answer_n)

    source = SourceRef(
        id=f"src_{atom_id[4:]}",
        artifact_id=art_id,
        artifact_type=ArtifactType.pm_answer,
        filename="pm_answers.jsonl",
        # Enough to walk back to the exact card the PM answered.
        locator={
            "rule_id": rule_id,
            "question": _norm(question),
            "answered_at": answered_at,
            "actor": actor,
        },
        extraction_method="pm_answer",
        parser_version=PARSER_VERSION,
        parser=PARSER_NAME,
        path="pm_answers.jsonl",
    )

    return EvidenceAtom(
        id=atom_id,
        atom_id=atom_id,
        project_id=project_id,
        artifact_id=art_id,
        # A PM resolving an open question IS a decision — and it keeps these out
        # of `open_question`, which is what the brief re-raises as gaps.
        atom_type=AtomType.decision,
        raw_text=claim,
        normalized_text=claim.lower(),
        value={
            "question": _norm(question),
            "answer": answer_n,
            "rule_id": rule_id,
            "kind": "pm_answer",
        },
        source_refs=[source],
        authority_class=AuthorityClass.pm_confirmed,
        confidence=PM_ANSWER_CONFIDENCE,
        # A human authored it; it does not go back in the review queue.
        review_status=ReviewStatus.approved,
        review_flags=["pm_authored"],
        parser_version=PARSER_VERSION,
        claim=claim,
        normalized_claim=claim.lower(),
    )


def pm_answers_to_atoms(
    events: Iterable[Any],
    *,
    project_id: str,
) -> list[EvidenceAtom]:
    """Convert ``answered`` feedback events into atoms, newest answer winning.

    ``events`` are mappings (the JSONL ledger rows) or anything with the same
    attributes. Rows that are not ``answered``, or whose answer carries no
    facts, are skipped. De-duplicated per rule so re-answering a question
    replaces the earlier atom rather than stacking a contradiction.
    """
    by_key: dict[str, EvidenceAtom] = {}
    for ev in events:
        get = ev.get if isinstance(ev, dict) else lambda k, d=None: getattr(ev, k, d)
        if str(get("action", "") or "") != "answered":
            continue
        answer = str(get("edited_text", "") or "")
        question = str(get("question_text", "") or "")
        rule_id = str(get("rule_id", "") or "")
        atom = pm_answer_to_atom(
            project_id=project_id,
            question=question,
            answer=answer,
            rule_id=rule_id,
            deal_id=str(get("deal_id", "") or ""),
            answered_at=str(get("created_at", "") or ""),
            actor=str(get("actor", "") or ""),
        )
        if atom is None:
            continue
        # Ledger is append-only and chronological, so a later answer to the same
        # question is the PM's current position.
        by_key[rule_id or question or atom.id] = atom
    return list(by_key.values())
