"""Check every teacher page of a run against the teacher's window, on CPU.

Loads only the teacher's tokenizer, builds every part the way the run does,
and reads each page the teacher would: the direct, company and reasoned line
pages, the WHY and value pages, the flip pages, the cards and the grounded
pages. It reports what was fitted (context or notes dropped), what is left
out (cards, grounding), and anything that would still overflow, which is
the one thing that stops a run. Exit status 1 when anything would overflow.

    python -m ml.c3.budget_check --teacher Qwen/Qwen2.5-7B-Instruct --max-len 32768 <deal files>
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter

import torch

from .brain import Brain, _tail, company_field_notes, notes_from_deal, teacher_cards, told_why
from .data import Batch
from .lm import LMBase
from .losses import supposed_twins
from .schema import Schema, load_schema


class TokenizerOnly(LMBase):
    """The teacher's tokenizer and window, without its weights."""

    def __init__(self, name: str, max_len: int):
        super().__init__()
        from transformers import AutoTokenizer  # optional dependency

        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.dim, self.max_len = 8, max_len
        self._anchor = torch.nn.Parameter(torch.zeros(1))

    def token_count(self, text: str) -> int:
        return len(self.tokenizer(str(text), truncation=False)["input_ids"])

    def encode(self, text: str, first: bool) -> list[int]:
        return self.tokenizer(str(text), add_special_tokens=first, truncation=False)["input_ids"]


def check_part(brain: Brain, batch: Batch, counts: Counter) -> list[str]:
    """Tally one part; return a line per page that would still overflow."""
    bad: list[str] = []
    why = told_why(batch, brain.schema)
    notes, cnotes = notes_from_deal(batch)
    cfield = company_field_notes(batch, brain.schema)

    def page(i: int, ns: list[str], tail: str, what: str) -> None:
        counts["pages"] += 1
        full = brain._page(batch, i, max(0, i - brain.context_lines), list(ns))
        text = brain.page(batch, i, ns, tail)
        if text != full:
            counts["pages_fitted"] += 1
        if not brain.fits(text + tail):
            bad.append(f"{batch.deal_id} line {i} {what}: {brain.lm.token_count(text + tail)} tokens")

    for i in range(len(batch)):
        page(i, notes[i], _tail(None, i), "direct")
        page(i, notes[i] + cnotes[i], _tail(None, i), "company")
        if why[i]:
            page(i, notes[i], _tail(why, i), "reasoned")
            page(i, notes[i] + cnotes[i] + cfield[i], _tail(why, i), "reasoned company")
            page(i, notes[i], "Why: " + why[i], "WHY")
            counts["grounding"] += 1
            if not any(brain.fits(brain.grounded_page(batch, i, w)[0], brain.GROUND_TOKENS)
                       for w in (12, 8, 4, 2, 1, 0)):
                counts["grounding_left_out"] += 1
    for item in batch.literals:
        if item.line >= 0 and str(item.text or "").strip():
            counts["values"] += 1
            if not brain.fits(batch.texts[item.line] + "\nValue: " + str(item.text)):
                bad.append(f"{batch.deal_id} line {item.line} value")
    for f in supposed_twins(batch)[0]:
        tail = (f"Why: {why[f.line]}\n" if why[f.line] else "") + f"Suppose instead: {f.condition}\nAnswer:"
        page(f.line, notes[f.line], tail, "flip")
    counts["cards"] += len(batch.judged)
    counts["cards_left_out"] += len(batch.judged) - len(teacher_cards(brain, batch))
    return bad


def check(batches: list[Batch], schema: Schema, lm: LMBase, context_lines: int = 6) -> tuple[Counter, list[str]]:
    brain = Brain(schema, lm, context_lines=context_lines)
    counts: Counter = Counter()
    bad: list[str] = []
    for b in batches:
        bad += check_part(brain, b, counts)
        counts["parts"] += 1
    return counts, bad


def main() -> None:
    from .loop import load_split  # noqa: PLC0415

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--teacher", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--max-len", type=int, default=32768)
    p.add_argument("--max-atoms", type=int, default=8192)
    p.add_argument("--context-lines", type=int, default=6)
    p.add_argument("deals", nargs="+")
    args = p.parse_args()
    schema = load_schema()
    batches, _ = load_split(args.deals, schema, args.max_atoms)
    counts, bad = check(batches, schema, TokenizerOnly(args.teacher, args.max_len), args.context_lines)
    print("budget", " ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    for line in bad[:50]:
        print("overflow", line)
    print("overflow_total", len(bad))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
