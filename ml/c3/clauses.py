"""Split a written explanation into clauses, each with the job it does.

A full WHY or rule card is not one fact. A paragraph like

    "Each clinic gets four 65-inch displays. Because a 65-inch panel is a
    two-person lift, every mount needs two technicians, so the crew doubles
    and install hours go up, unless the customer's own staff lift the panels.
    Matches the reseller quote: 8 displays. 2 sites x 4 = 8 displays."

carries a statement, a reason, a consequence, an exception, a piece of
evidence and an arithmetic check. Squeezing it into one vector loses which
part is the condition and which the exception. ``split_clauses`` cuts it at
sentence boundaries and at discourse connectives and gives every clause a
role:

=============  ===========================================  =====================================
role           cue words (lower-case, at clause start)      what it becomes (operators.py)
=============  ===========================================  =====================================
condition      if, when, whenever, once, for any, any line   a region the rule applies in (AND)
exception      unless, except, other than, but not           a region cut out of it (OR)
cause          because, since, given that, due to            the fold: what matters, and so what
                                                             does not
consequence    so, therefore, which means, as a result, or   the move toward an answer and the
               a change verb (adds, cuts, doubles, needs)    claims graded at close
evidence       matches, per, according to, the SOW says      where to look (topic)
quantity       an arithmetic chain "a x b = c"               an executable program (notes.py)
statement      anything else                                 what the line is (topic)
=============  ===========================================  =====================================

It is deterministic and has no model in it, so a person can read exactly how
their paragraph was cut (``python -m ml.c3.clauses "text"``).
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass

from .notes import extract_programs

CONDITION, EXCEPTION, CAUSE, CONSEQUENCE, EVIDENCE, QUANTITY, STATEMENT = (
    "condition", "exception", "cause", "consequence", "evidence", "quantity", "statement")
ROLES = (CONDITION, EXCEPTION, CAUSE, CONSEQUENCE, EVIDENCE, QUANTITY, STATEMENT)

_CUES: list[tuple[str, str]] = [
    (EXCEPTION, r"unless|except(?: when| if| for)?|other than|but not|excluding"),
    (CAUSE, r"because|since|given that|due to"),
    (CONSEQUENCE, r"so that|so|therefore|thus|which means|this means|that means|as a result|"
                  r"which (?:adds|cuts|changes|sets|moves|raises|lowers|doubles|removes|needs)"),
    (CONDITION, r"if|when|whenever|once|for any|any line|for every|where"),
    (EVIDENCE, r"matches|per|according to|the sow says|the quote says|as stated in|confirmed by"),
]
_CHANGE_VERBS = re.compile(
    r"\b(adds?|cuts?|changes?|doubles?|halves?|removes?|raises?|lowers?|increases?|decreases?|"
    r"needs?|requires?|replaces?|supersedes?|reprices?|goes up|goes down)\b", re.I)
#: Split points inside a sentence: before a cue word that follows a comma,
#: or before a strong cue anywhere.
_SPLIT = re.compile(
    r",\s+(?=(?:unless|except|because|since|so|therefore|which means|as a result|if|when|"
    r"whenever|once|but not|other than|given that|due to)\b)"
    r"|\s+(?=(?:unless|because|so that|which means|as a result)\b)", re.I)
_SENT = re.compile(r"(?<=[.!?;])\s+|\n+")


@dataclass(frozen=True)
class Clause:
    text: str
    role: str
    cue: str
    sentence: int


def _role(text: str) -> tuple[str, str]:
    low = text.lower().lstrip(" ,;:-")
    if extract_programs(text):
        return QUANTITY, "="
    for role, pat in _CUES:
        m = re.match(rf"(?:{pat})\b", low)
        if m:
            return role, m.group(0)
    if _CHANGE_VERBS.search(low):
        return CONSEQUENCE, "verb"
    return STATEMENT, ""


def split_clauses(text: str) -> list[Clause]:
    """Every clause of ``text``, in order, with its role. Nothing is dropped:
    the clause texts joined back together cover every word of the input."""
    out: list[Clause] = []
    for si, sent in enumerate(s for s in _SENT.split(str(text or "")) if s.strip()):
        for part in _SPLIT.split(sent):
            part = part.strip(" ,")
            if not part:
                continue
            role, cue = _role(part)
            # "Because X, Y" / "When X, Y": the main clause after the comma is
            # the "then" part, so it is split off and read as a consequence
            # unless it carries its own role.
            if role in (CONDITION, CAUSE) and ", " in part:
                head, rest = part.split(", ", 1)
                r_role, r_cue = _role(rest)
                if r_role == STATEMENT:
                    r_role, r_cue = CONSEQUENCE, "then"
                out.append(Clause(head.strip(), role, cue, si))
                out.append(Clause(rest.strip(), r_role, r_cue, si))
                continue
            out.append(Clause(part, role, cue, si))
    return out


def main() -> None:
    for c in split_clauses(" ".join(sys.argv[1:]) or sys.stdin.read()):
        print(f"{c.role:<12} {c.cue:<14} {c.text}")


if __name__ == "__main__":
    main()
