"""Reasons are predictions; the deal's ending grades them (architecture v5).

Every WHY already says what a line changes ("adds install time per display",
"cuts the site count from 3 to 2"). v5 makes that claim explicit and
checkable:

* each line gets a **change vector**: for hours, crew, sites, price, tasks and
  schedule, does it push down, leave alone, or push up? (``model.changes``,
  trained from the proposed ``changes`` reading);
* each compiled explanation carries the **change it claims**
  (``ReasonCompiler.claims``), trained to match the change vector of the line
  its WHY was written for, so the compiler learns to read consequences out of
  a sentence;
* when a deal closes, ``realized_changes`` reads what actually moved between
  the quote and the close, and ``ReliabilityLedger`` grades every explanation
  that fired on that deal: did what it claimed happen? Its track record then
  scales how strongly it acts on the next deal (``Explanation.reliability``).

So an explanation is not just trusted because someone wrote it. A rule card
that keeps predicting overruns that never come loses weight on its own, and
one that keeps being right gains it. That is the consequential-semantics
thesis (v4) applied to the reasons themselves.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import torch
from torch.nn import functional as F

from .data import Batch, realized_changes
from .explain import ExplanationBank
from .model import CHANGE_SLOTS, DIRECTIONS, C3Output


def change_targets(batch: Batch) -> torch.Tensor:
    """[N, slots] direction index per line, -100 where the label is silent."""
    t = torch.full((len(batch), len(CHANGE_SLOTS)), -100, dtype=torch.long)
    for i, ch in enumerate(batch.changes or []):
        for slot, d in (ch or {}).items():
            if slot in CHANGE_SLOTS and d in DIRECTIONS:
                t[i, CHANGE_SLOTS.index(slot)] = DIRECTIONS.index(d)
    return t


def changes_loss(out: C3Output, batch: Batch) -> torch.Tensor:
    t = change_targets(batch).to(out.r.device)
    if out.changes is None or not (t != -100).any():
        return out.r.new_zeros(())
    return F.cross_entropy(out.changes.reshape(-1, 3), t.reshape(-1), ignore_index=-100)


def claims_loss(out: C3Output, batch: Batch, bank: ExplanationBank) -> torch.Tensor:
    """A WHY's compiled claim should match the change vector of its own line."""
    t = change_targets(batch).to(out.r.device)
    total, terms = out.r.new_zeros(()), 0
    for layer, claims in out.reason_claims.items():
        for col, bi in enumerate(out.reason_index[layer]):
            line = bank.items[bi].line
            if line < 0 or not (t[line] != -100).any():
                continue
            total = total + F.cross_entropy(claims[col], t[line], ignore_index=-100)
            terms += 1
    return total / max(terms, 1)


def applicability(out: C3Output, layer_index: list[int]) -> torch.Tensor | None:
    """[N, M] how strongly each explanation acted on each line, over all questions."""
    gs = [g for g in out.gates.values() if g.shape[1] == len(layer_index)]
    return torch.stack(gs).amax(0) if gs else None


def rule_link_loss(out: C3Output, batch: Batch, bank: ExplanationBank) -> torch.Tensor:
    """follows_rule -> the rule's operator should act on the line (1);
    exception_to -> it should not (0). Supervises the rule's region directly."""
    total, terms = out.r.new_zeros(()), 0
    for layer, idx in out.reason_index.items():
        g = applicability(out, idx)
        if g is None:
            continue
        col = {bank.items[bi].ref: c for c, bi in enumerate(idx) if bank.items[bi].ref}
        for line, rid, sign in batch.rule_links:
            if rid in col:
                p = g[line, col[rid]].clamp(1e-5, 1 - 1e-5)
                total = total - (torch.log(p) if sign > 0 else torch.log1p(-p))
                terms += 1
    return total / max(terms, 1)


@dataclass
class ReliabilityLedger:
    """Per-explanation track record against closed deals, Beta(1, 1) prior."""
    agree: dict[str, float] = field(default_factory=dict)
    seen: dict[str, float] = field(default_factory=dict)
    fire_threshold: float = 0.5

    def reliability(self, ref: str) -> float:
        return (self.agree.get(ref, 0.0) + 1.0) / (self.seen.get(ref, 0.0) + 2.0)

    @torch.no_grad()
    def grade(self, out: C3Output, batch: Batch, bank: ExplanationBank) -> dict[str, float]:
        """Grade every explanation that fired on this (closed) deal. Returns
        the per-ref agreement on this deal, for inspection."""
        realized = realized_changes(batch.outcome)
        if not realized:
            return {}
        graded = {}
        for layer, idx in out.reason_index.items():
            g = applicability(out, idx)
            if g is None:
                continue
            claims = out.reason_claims[layer].argmax(-1)          # [M, slots]
            for col, bi in enumerate(idx):
                e = bank.items[bi]
                if not e.ref or float(g[:, col].max()) < self.fire_threshold:
                    continue
                hits, n = 0.0, 0.0
                for slot, d in realized.items():
                    if slot not in CHANGE_SLOTS:
                        continue
                    claim = DIRECTIONS[int(claims[col, CHANGE_SLOTS.index(slot)])]
                    if claim == "none":
                        continue                   # it claimed nothing about this slot
                    n += 1
                    hits += float(claim == d)
                if n:
                    self.agree[e.ref] = self.agree.get(e.ref, 0.0) + hits
                    self.seen[e.ref] = self.seen.get(e.ref, 0.0) + n
                    graded[e.ref] = hits / n
        return graded

    def apply(self, bank: ExplanationBank) -> ExplanationBank:
        return ExplanationBank([dataclasses.replace(e, reliability=self.reliability(e.ref))
                                if e.ref else e for e in bank.items])
