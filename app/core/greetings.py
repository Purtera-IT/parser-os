"""A greeting at the start of a line, and what the line says after it.

"Hello, we already have wall mounts, and I believe parking is not free"
(deal 010003) opens with a greeting and then states two facts. Every place
that judged lines by their opening word -- the note parser's caption check,
the transcript turn-role classifier, the small-talk prediction -- read the
"Hello," and filed the facts as chatter. The greeting is judged as a greeting;
the rest of the line is judged on its own.
"""
from __future__ import annotations

import re

#: "Hello,", "Hi Trent,", "Good morning team -", "Dear all:". One optional
#: addressee word after the greeting (a name or "all"/"team"/...).
LEADING_GREETING_RE = re.compile(
    r"^\s*(?:hi|hey|hello|hiya|dear|greetings|good\s+(?:morning|afternoon|evening))\b"
    r"(?:\s+(?:all|team|everyone|guys|folks|there|[A-Z][a-z]+))?\s*[,!.:\-]*\s*",
    re.I,
)


def starts_with_greeting(text: str) -> bool:
    return bool(LEADING_GREETING_RE.match(str(text or "")))


def strip_leading_greeting(text: str) -> str:
    """The line without the greeting it opened with; unchanged when it has none."""
    t = str(text or "")
    m = LEADING_GREETING_RE.match(t)
    return t[m.end():].strip() if m else t.strip()


__all__ = ["LEADING_GREETING_RE", "starts_with_greeting", "strip_leading_greeting"]
