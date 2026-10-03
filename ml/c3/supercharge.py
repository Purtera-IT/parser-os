"""Heads supercharged by the long labels (architecture v6).

The product is the task heads (model.C3Model): one per labeling opportunity,
no language model and no text generation at run time. The long WHY
paragraphs are used where they pay the most: **in training, as privileged
information**. A teacher reads them; the heads learn from the teacher; the
teacher is then thrown away.

Teacher (training only): ``brain.Brain`` reads each line with its WHY on the
page, so its answers and its internal picture of the line are informed by
the expert's reasoning. It can be built on a pretrained encoder or LM so it
already knows the language; the heads never depend on it.

What the heads get from it, per labeled line with a WHY:

==================  =====================================================  ==================
term                what it does                                           why it saves labels
==================  =====================================================  ==================
``teach_heads``     every head matches the teacher's answer on that line,  one paragraph
                    including heads the human left blank (a WHY that says  teaches many heads,
                    "the crew doubles" also tells the changes head and     not just the one
                    the crew reading)                                       field it was for
``teach_geometry``  lines the teacher sees as alike *for the reason* are   the reason, not the
                    alike in the heads' latent (relational distillation    surface words, sets
                    of the pairwise similarities)                          who is near whom
``teach_unlabeled`` unlabeled lines get the teacher's answer with the      labels spread to
                    deal's other WHYs on its page                          the lines nobody
                                                                           labeled
``teach_pointers``  which other lines the WHY rests on (the SOW line it    the heads learn
                    checks, the email that sets the crew), as the teacher  where to look, not
                    reads it, becomes the heads' pointer head              only what to answer
``teach_words``     which words of the line carry the reason (qty, size,   the heads learn
                    model number) and which the WHY passes over (a price   which facts count;
                    it calls irrelevant), as a weight per word             the rest is noise
==================  =====================================================  ==================

The last two are read from the teacher, not parsed: the teacher's
likelihood of the WHY is traced back to the page (``Brain.grounding``), so
"QTY 4 = 4 units of mount work" puts weight on the 4 and the QTY column, and
"the prices are irrelevant" leaves the price with almost none, because that
is what a model that predicts the paragraph has to rely on.

Plus what v4/v5 already do with the WHY in the heads' own space
(``why_align``, ``why_sufficiency`` in losses.py).

The measured claim is label efficiency: the same heads, the same labeled
lines, with and without these terms (efficiency.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .data import IGNORE, Batch
from .model import C3Model, C3Output


@dataclass
class TeachWeights:
    heads: float = 1.0
    geometry: float = 0.5
    unlabeled: float = 0.3
    pointers: float = 0.5
    words: float = 0.5
    temperature: float = 2.0


@dataclass
class TeacherView:
    """The teacher's read of one deal, detached: what the heads learn from."""
    told: dict[str, torch.Tensor]        # answers with each line's own WHY on the page [N, A]
    noted: dict[str, torch.Tensor]       # answers with the deal's other WHYs as notes [N, A]
    embedding: torch.Tensor              # [N, dim] the told page, mean-pooled
    has_why: torch.Tensor                # [N] bool
    labeled: torch.Tensor                # [N] bool
    pointers: torch.Tensor | None = None  # [N, N+1] what each WHY rests on (last = itself)
    words: list[torch.Tensor | None] | None = None  # per line: weight on each of its words


@torch.no_grad()
def teacher_view(teacher, batch: Batch, company: str | None = None,
                 grounding: bool = True) -> TeacherView:
    from .brain import company_field_notes, notes_from_deal, told_why  # noqa: PLC0415

    was = teacher.training
    teacher.eval()
    desc = teacher.describe()
    notes, cnotes = notes_from_deal(batch)
    why = told_why(batch, teacher.schema)
    told = teacher(batch, company=company, why=why, desc=desc,
                   company_notes=company_field_notes(batch, teacher.schema))
    noted = teacher(batch, company=company, notes=notes, company_notes=cnotes, desc=desc)
    m = told.mask.unsqueeze(-1).to(told.hidden.dtype)
    emb = (told.hidden * m).sum(1) / m.sum(1).clamp(min=1)
    pointers, words = teacher.grounding(batch) if grounding else (None, None)
    teacher.train(was)
    dev = emb.device
    if pointers is not None:
        # Where a labeler named the lines a decision came from (hint_refs),
        # their answer replaces the teacher's reading.
        for i, js in enumerate(batch.hint_lines or []):
            if js:
                pointers[i] = 0.0
                pointers[i, js] = 1.0 / len(js)
    return TeacherView(told.logits, noted.logits, emb,
                       torch.tensor([bool(t) for t in why], device=dev),
                       torch.tensor(batch.labeled, device=dev), pointers, words)


def _kl(student: torch.Tensor, teacher: torch.Tensor, t: float) -> torch.Tensor:
    return F.kl_div((student / t).log_softmax(-1), (teacher / t).log_softmax(-1),
                    log_target=True, reduction="batchmean") * t * t


def supercharge_loss(model: C3Model, out: C3Output, view: TeacherView,
                     w: TeachWeights | None = None,
                     texts: list[str] | None = None) -> dict[str, torch.Tensor]:
    """The three teaching terms for one deal. ``out`` is the heads' own
    forward pass (no teacher inside it); ``view`` is detached."""
    w = w or TeachWeights()
    zero = out.r.new_zeros(())
    keys = [k for k in view.told if k in out.logits]
    parts = {"teach_heads": zero, "teach_geometry": zero, "teach_unlabeled": zero,
             "teach_pointers": zero, "teach_words": zero}
    if not keys:
        return parts
    why = view.has_why
    if why.any():
        parts["teach_heads"] = torch.stack(
            [_kl(out.logits[k][why], view.told[k][why], w.temperature) for k in keys]).mean()
    if int(why.sum()) > 2:
        s = F.normalize(out.r[why], dim=-1)
        t = F.normalize(view.embedding[why], dim=-1)
        eye = torch.eye(s.shape[0], dtype=torch.bool, device=s.device)
        ss = (s @ s.T / 0.1).masked_fill(eye, -1e4)
        tt = (t @ t.T / 0.1).masked_fill(eye, -1e4)
        parts["teach_geometry"] = F.kl_div(ss.log_softmax(-1), tt.log_softmax(-1),
                                           log_target=True, reduction="batchmean")
    rest = ~view.labeled
    if rest.any():
        parts["teach_unlabeled"] = torch.stack(
            [_kl(out.logits[k][rest], view.noted[k][rest], w.temperature) for k in keys]).mean()
    if view.pointers is not None and out.pointers is not None and why.any():
        # The heads see only earlier lines: the teacher's weight on later
        # lines is dropped and the rest renormalized.
        allowed = torch.isfinite(out.pointers)
        t = view.pointers.masked_fill(~allowed, 0.0)[why]
        keep = t.sum(-1) > 0
        if keep.any():
            t = t[keep] / t[keep].sum(-1, keepdim=True)
            s = out.pointers[why][keep].clamp(min=-1e4)
            parts["teach_pointers"] = -(t * s).sum(-1).mean() + (t * t.clamp(min=1e-9).log()).sum(-1).mean()
    if view.words is not None and texts is not None:
        mine = model.word_weights(out, texts)
        kls = [F.kl_div(m, t.clamp(min=1e-9).log(), log_target=True, reduction="sum")
               for m, t in zip(mine, view.words) if m is not None and t is not None
               and m.shape == t.shape]
        if kls:
            parts["teach_words"] = torch.stack(kls).mean()
    return parts


def weighted(parts: dict[str, torch.Tensor], w: TeachWeights | None = None) -> torch.Tensor:
    w = w or TeachWeights()
    return (w.heads * parts["teach_heads"] + w.geometry * parts["teach_geometry"]
            + w.unlabeled * parts["teach_unlabeled"] + w.pointers * parts["teach_pointers"]
            + w.words * parts["teach_words"])


def mask_labels(batch: Batch, keep: set[int]) -> Batch:
    """The same deal with only lines in ``keep`` labeled: targets, numbers,
    WHYs, company lines and changes of every other line are removed."""
    import dataclasses

    def col(v):
        return [x if i in keep else IGNORE for i, x in enumerate(v)]

    return dataclasses.replace(
        batch,
        targets={k: col(v) for k, v in batch.targets.items()},
        numbers_target={k: [x if i in keep else None for i, x in enumerate(v)]
                        for k, v in batch.numbers_target.items()},
        edges={r: [(s, d) for s, d in ps if s in keep] for r, ps in batch.edges.items()},
        labeled=[bool(x) and i in keep for i, x in enumerate(batch.labeled)],
        why=[x if i in keep else None for i, x in enumerate(batch.why)],
        policy_note=[x if i in keep else None for i, x in enumerate(batch.policy_note)],
        rule_links=[x for x in batch.rule_links if x[0] in keep],
        changes=[x if i in keep else None for i, x in enumerate(batch.changes)],
        field_notes=[x if i in keep else {} for i, x in enumerate(batch.field_notes)],
        hint_lines=[x if i in keep else [] for i, x in enumerate(batch.hint_lines)],
        entities=[x if i in keep else [] for i, x in enumerate(batch.entities)],
        # Pair verdicts follow their lines; a group's or the deal's verdict is
        # not a line label and stays.
        judged=[j for j in batch.judged if j.size != "pair" or set(j.lines) <= keep],
        negatives={k: [x if i in keep else [] for i, x in enumerate(v)] for k, v in batch.negatives.items()},
    )
