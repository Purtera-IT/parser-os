"""Long-deal parts that keep every evidence edge and teach every label once.

Dense attention is refused above the atom budget, and the source is never
truncated. A deal over the budget becomes three kinds of part:

* **Windows.** Overlapping spans of the deal in time order. Each line's own
  labels train in exactly one window (its home: the first window it falls
  in); in the overlap it is context only. A link with both ends in a window
  trains there. A labeled line whose link leaves its window is listed in
  ``Batch.link_elsewhere`` so the window never teaches it "no link".
* **Edge packs.** Links that no window holds whole, packed together into a
  few windows of their endpoints and a few neighbours each, in time order.
  Labels are stripped there (they already trained at home); every link with
  both ends in a pack trains, so a pack never teaches a gold target as a
  "not this one". Pair and group cards that fit no window ride in packs too.
* **The digest.** One window for the whole-deal cards and the deal's saved
  answers: the deal's commercial documents (Deal Kit, SOW, quote, PO)
  first, then labeled lines, latest first, up to the budget, in time order.
  Unanchored questions that read the deal are taught here too. Its labels
  and links are stripped: they train in the windows and packs.

A card whose subject is not in the deal stays uncovered.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

from .data import Atom, Batch, DealExample, _is_a_person, _Lines, excluded, featurize, is_deal_kit, judgment_subject
from .schema import DEAL, Schema

#: Lines either side of a link's endpoint that come with it into a pack.
PACK_CONTEXT = 8

_COMMERCIAL = re.compile(r"sow|quote|proposal|purchase_?order|contract|change_?order|estimate|pricing")


@dataclass
class LongDealPlan:
    windows: list[Batch]
    #: Edge packs: links no window holds whole, plus cards that fit no window.
    bridges: list[Batch]
    covered_edges: int
    uncovered_judgments: int
    complete: bool
    notes: list[str] = field(default_factory=list)
    #: The whole-deal part (empty when the deal has no whole-deal card or answer).
    digest: list[Batch] = field(default_factory=list)

    @property
    def parts(self) -> list[Batch]:
        return [*self.windows, *self.bridges, *self.digest]


def _context_only(atom: Atom) -> Atom:
    """A line present for what it says, whose labels train in another part."""
    return replace(atom, label=None, history_only=False, row_weight=None)


def _part(deal: DealExample, atoms: list[Atom], edges: list, judgments: list, schema: Schema,
          max_atoms: int, *, answers: list | None = None, elsewhere=()) -> Batch:
    part = replace(deal, atoms=atoms, edges=edges, judgments=judgments,
                   source_supervision={**deal.source_supervision, "judgments": judgments,
                                       "deal_answers": list(answers or [])})
    return featurize(part, schema, max_atoms=max_atoms, elsewhere=elsewhere)


def _inside(edges: list, keys: set[str]) -> list:
    return [e for e in edges if e[0] in keys and e[1] in keys]


def _digest_rank(atom: Atom, i: int) -> tuple:
    kind = str(atom.doc_kind or "").lower()
    commercial = is_deal_kit(kind) or bool(_COMMERCIAL.search(kind))
    return (not commercial, atom.label is None, -i)


def plan_long_deal(deal: DealExample, schema: Schema, *, max_atoms: int = 4096,
                   overlap: int = 32, context: int = PACK_CONTEXT,
                   always_digest: bool = False) -> LongDealPlan:
    """``always_digest``: build the digest even with nothing saved to teach on
    it, so a prediction can read the whole-deal answers of a new deal."""
    if max_atoms < 2:
        raise ValueError("Atom budget must leave room for an evidence pair")
    if overlap < 0 or overlap >= max_atoms:
        raise ValueError("Window overlap must be smaller than the atom budget")
    atoms = list(deal.atoms)
    n = len(atoms)
    if n <= max_atoms:
        return LongDealPlan([featurize(deal, schema, max_atoms=max_atoms)], [],
                            len(deal.edges), 0, True)
    step = max(1, max_atoms - overlap)
    spans: list[tuple[int, int]] = []
    start = 0
    while start < n:
        end = min(n, start + max_atoms)
        spans.append((start, end))
        if end == n:
            break
        start += step
    home = [0] * n
    w = 0
    for i in range(n):
        while i >= spans[w][1]:
            w += 1
        home[i] = w
    pos = {a.key: i for i, a in enumerate(atoms)}
    # featurize drops a link that touches a set-aside row or leaves the deal;
    # it trains nothing, so it is not planned.
    aside = {a.key for a in atoms if excluded(a.label)}
    edges = [e for e in deal.edges if e[0] in pos and e[1] in pos and not {e[0], e[1]} & aside]
    dropped_edges = len(deal.edges) - len(edges)

    lines = _Lines(atoms)
    kept = n - len(aside)
    owned: list[list[dict]] = [[] for _ in spans]
    loose: list[tuple[tuple[int, ...], dict]] = []
    whole: list[dict] = []
    uncovered = 0
    spread = 0
    for judgment in deal.judgments:
        if not isinstance(judgment, dict) or excluded(judgment) or not _is_a_person(judgment.get("labeler")):
            continue                              # featurize trains neither
        head = str(judgment.get("head") or "")
        opp = schema.by_key().get(f"jdg:{head}")
        size = opp.size if opp is not None else "line"
        if size == DEAL:
            whole.append(judgment)
            continue
        idx = judgment_subject(judgment, size, lines)
        if idx is None and size == "line" and str(judgment.get("note") or "").strip():
            # A line card whose line is not in this file's input (a Dropped
            # card on a line the compile set aside): one window, in turn,
            # carries it, and its note alone trains (data.unplaced_notes).
            owned[spread % len(spans)].append(judgment)
            spread += 1
            continue
        if not idx:
            uncovered += 1
            continue
        first = spans[home[min(idx)]]
        if all(first[0] <= i < first[1] for i in idx):
            owned[home[min(idx)]].append(judgment)
            continue
        owners = [w for w, (b, e) in enumerate(spans) if all(b <= i < e for i in idx)]
        if owners:
            owned[owners[0]].append(judgment)
        elif len(idx) == kept:
            whole.append(judgment)                # an unanchored question reads the deal
        elif len(idx) <= max_atoms:
            loose.append((tuple(sorted(idx)), judgment))
        else:
            uncovered += 1

    covered: set[tuple[str, str, str]] = set()
    windows: list[Batch] = []
    for w, ((begin, end), judgments) in enumerate(zip(spans, owned)):
        subset = [a if home[begin + k] == w else _context_only(a) for k, a in enumerate(atoms[begin:end])]
        keys = {a.key for a in subset}
        inside = _inside(edges, keys)
        covered.update(inside)
        away = {(e[0], e[2]) for e in edges if e[0] in keys and home[pos[e[0]]] == w and e[1] not in keys}
        windows.append(_part(deal, subset, inside, judgments, schema, max_atoms, elsewhere=away))

    items: list[tuple[tuple[int, ...], tuple | None, dict | None]] = [
        (tuple(sorted((pos[e[0]], pos[e[1]]))), e, None) for e in edges if e not in covered]
    items += [(idx, None, j) for idx, j in loose]
    items.sort(key=lambda it: it[0])
    packs: list[Batch] = []
    chosen: set[int] = set()
    pack_judgments: list[dict] = []

    def flush() -> None:
        if not chosen:
            return
        subset = [_context_only(atoms[i]) for i in sorted(chosen)]
        inside = _inside(edges, {a.key for a in subset})
        covered.update(inside)
        packs.append(_part(deal, subset, inside, list(pack_judgments), schema, max_atoms))
        chosen.clear()
        pack_judgments.clear()

    for idx, _edge, judgment in items:
        if len(chosen) + sum(i not in chosen for i in idx) > max_atoms:
            flush()
        chosen.update(idx)
        for k in range(1, context + 1):
            for i in idx:
                for j in (i - k, i + k):
                    if len(chosen) >= max_atoms:
                        break
                    if 0 <= j < n:
                        chosen.add(j)
        if judgment is not None:
            pack_judgments.append(judgment)
    flush()

    digest: list[Batch] = []
    answers = list(deal.source_supervision.get("deal_answers") or [])
    if whole or answers or always_digest:
        ranked = sorted((i for i, a in enumerate(atoms) if not excluded(a.label)),
                        key=lambda i: _digest_rank(atoms[i], i))[:max_atoms]
        subset = [_context_only(atoms[i]) for i in sorted(ranked)]
        # Links train in the windows and packs; the digest teaches none.
        digest.append(_part(deal, subset, [], whole, schema, max_atoms, answers=answers))

    missing = [e for e in edges if e not in covered]
    if missing:
        raise RuntimeError(f"{len(missing)} evidence edges were not placed")
    notes = []
    if uncovered:
        notes.append(f"{uncovered} cards name no line of this deal")
    if dropped_edges:
        notes.append(f"{dropped_edges} links touch a set-aside or missing line and train nothing")
    if whole:
        notes.append(f"{len(whole)} whole-deal cards read a digest of {len(ranked)} lines")
    return LongDealPlan(windows, packs, len(covered), uncovered, uncovered == 0, notes, digest)
