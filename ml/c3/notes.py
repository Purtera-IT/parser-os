"""The note: the universal WHY, the company's line, and the arithmetic in it.

``split_note`` follows the note grammar exactly as
``app/learning/human_labels.split_note`` does (tests/test_c3_model.py checks
the two agree), but lives here so this package never imports ``app``.

``mask_verdict`` hides the words that would let the rationale encoder read
the answer off the WHY ("superseded", "reject"...). The rationale latent is
pulled toward the masked WHY, so it learns the *reason*, not the verdict.

``extract_programs`` reads arithmetic a labeler wrote ("4 sites x 52 days x
$920 = $191,360") into a typed program that executes to the stated result.
Only programs that reproduce their own stated number are kept: that is the
program head's weak supervision.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

EXCLUDE_NOTE_PREFIX = "EXCLUDE_FROM_TRAINING"

#: Verdict words and policy names hidden before a WHY is encoded.
VERDICT_WORDS = (
    "reject", "rejected", "keep", "kept", "ignore", "ignored", "superseded",
    "supersedes", "noise", "purtera", "deal kit", "hubspot", "atlas", "portal", "gantt",
)
_VERDICT_RE = re.compile(r"\b(" + "|".join(re.escape(w) for w in VERDICT_WORDS) + r")\b", re.I)
MASK = "[MASK]"


def split_note(note: str, company: str = "purtera") -> tuple[str, str]:
    """(universal WHY, company policy) from one stored note."""
    text = str(note or "").strip()
    head = text.lstrip()
    if head.lstrip("[").upper().startswith(EXCLUDE_NOTE_PREFIX):
        end = head.find("]") if head.startswith("[") else -1
        if end >= 0:
            text = head[end + 1:]
        else:
            text = head.split("\n", 1)[1] if "\n" in head else ""
    marker = f"[{company}]"
    universal: list[str] = []
    policy: list[str] = []
    into = universal
    for line in text.splitlines():
        if into is universal and line.lstrip().lower().startswith(marker):
            into = policy
            line = line.lstrip()[len(marker):]
        into.append(line)
    return "\n".join(universal).strip(), "\n".join(policy).strip()


#: Parser remarks (SHOULD SPLIT, SHOULD MERGE, fixed or open parse problems,
#: atom/column/page notes) go on an optional last line tagged ``[parser]``
#: (user decision, Oct 3). They are about our tooling, not the deal, and a
#: model taught to reproduce them learns to talk about the parser, so
#: training drops that line and anything after it.
PARSER_TAG = "[parser]"


def drop_meta(text: str) -> str:
    """``text`` up to, not including, its ``[parser]`` line."""
    keep = []
    for line in str(text or "").splitlines():
        if line.lstrip().lower().startswith(PARSER_TAG):
            break
        keep.append(line)
    return "\n".join(keep).strip()


def mask_verdict(text: str) -> str:
    return _VERDICT_RE.sub(MASK, str(text or ""))


# ---------------------------------------------------------------- programs

_NUM = r"\$?\d[\d,]*(?:\.\d+)?"
_OP = r"(?:x|×|\*|\+|-|/)"
_TERM = rf"({_NUM})(?:\s*[A-Za-z\"'/-]+(?:\s+[A-Za-z-]+)?)?"
_CHAIN = re.compile(rf"{_TERM}(?:\s*{_OP}\s*{_TERM})+\s*=\s*({_NUM})")
_OPS = {"x": "mul", "×": "mul", "*": "mul", "+": "add", "-": "sub", "/": "div"}


def _num(s: str) -> float:
    return float(s.replace("$", "").replace(",", ""))


@dataclass(frozen=True)
class Program:
    """A left-to-right chain over stated numbers: ``((a op b) op c) ...``.

    ``units`` keeps the word after each operand ("sites", "days"), which is
    what lets a trained head bind operands to belief slots later.
    """
    operands: tuple[float, ...]
    ops: tuple[str, ...]
    units: tuple[str, ...]
    stated: float
    text: str

    def execute(self) -> float:
        acc = self.operands[0]
        for op, b in zip(self.ops, self.operands[1:]):
            acc = {"mul": acc * b, "add": acc + b, "sub": acc - b,
                   "div": acc / b if b else float("nan")}[op]
        return acc

    @property
    def exact(self) -> bool:
        return abs(self.execute() - self.stated) <= max(0.005 * abs(self.stated), 0.01)


def extract_programs(why: str) -> list[Program]:
    out: list[Program] = []
    for m in _CHAIN.finditer(str(why or "")):
        span = m.group(0)
        left = span.split("=")[0]
        nums = re.findall(_NUM, left)
        ops = re.findall(rf"(?<=[\s\d\"a-z])({_OP})(?=\s*\$?\d)", left)
        units = tuple(
            (re.match(rf"{re.escape(n)}\s*([A-Za-z\"'/-]+)?", left[left.find(n):]) or [None, None])[1] or ""
            for n in nums)
        units = tuple("" if u.lower() in ("x", "×") else u for u in units)
        if len(nums) < 2 or len(ops) != len(nums) - 1:
            continue
        p = Program(operands=tuple(_num(n) for n in nums),
                    ops=tuple(_OPS[o] for o in ops), units=units,
                    stated=_num(m.groups()[-1]), text=span)
        if p.exact:
            out.append(p)
    return out
