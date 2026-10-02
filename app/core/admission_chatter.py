"""Lines a parser's admission regex refused, kept as chatter atoms.

"Hi Trent,", "Thank you,", the name under a sign-off: a regex is right that
they are not deal content, but a regex is not the judge -- the admission head
is, and it learns only from what a labeler sees. A line cut with a bare
``continue`` never reaches the labeling page, so the head never gets the
negative. So a refused line is a REAL, kept atom, flagged exactly the way
:func:`app.core.deal_chatter.mark_chatter` flags relationship talk
(``chatter`` flag plus a ``small_talk`` read), and carrying why the regex
refused it (``value["admission_regex"] = <reason>``).

The compiler holds these atoms OUT of every stage between parse and
packetizing and puts them back only for coverage and the result, so no head
-- scope, pricing, sites, packets, dedup, threading -- ever reads one. See
``compile_project`` (``held_chatter``).
"""
from __future__ import annotations

from typing import Any

from app.core.deal_chatter import CHATTER_FLAG

#: Marks an atom that exists only because an admission regex refused its line.
ADMISSION_REGEX_FLAG = "admission_regex"


def mark_admission_chatter(atom: Any, reason: str) -> Any:
    """Stamp ``atom`` as a regex-refused line: chatter, with the reason."""
    val = dict(getattr(atom, "value", None) or {})
    val[CHATTER_FLAG] = True
    val[ADMISSION_REGEX_FLAG] = reason
    val.setdefault("provenance", f"{ADMISSION_REGEX_FLAG}: {reason}")
    reads = list(val.get("reads") or [])
    if not any(r.get("key") == "small_talk" for r in reads):
        # The same prediction mark_chatter leaves: a guess a labeler confirms
        # or drops, never a verdict.
        reads.append({
            "key": "small_talk", "value": True,
            "why": f"{ADMISSION_REGEX_FLAG}: {reason}",
            "confidence": 0.5, "source": "rule",
        })
    val["reads"] = reads
    atom.value = val
    flags = list(getattr(atom, "review_flags", None) or [])
    for f in (CHATTER_FLAG, ADMISSION_REGEX_FLAG):
        if f not in flags:
            flags.append(f)
    atom.review_flags = flags
    return atom


def is_admission_chatter(atom: Any) -> bool:
    return ADMISSION_REGEX_FLAG in (getattr(atom, "review_flags", None) or [])


__all__ = ["ADMISSION_REGEX_FLAG", "is_admission_chatter", "mark_admission_chatter"]
