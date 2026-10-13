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
``teach_heads``     every class head matches the teacher's answer on     one paragraph
                    that line, including heads the human left blank (a     teaches many heads,
                    WHY that says "the crew doubles" also tells the        not just the one
                    changes head and the crew reading)                     field it was for
``teach_numbers``   every numeric head matches the teacher's log1p        hours, term slots
                    reading on that line                                   and kit hours move
                                                                           with the same WHY
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
``teach_flip``      each sentence of a WHY, supposed: the teacher reads    the heads learn
                    the page with it assumed and answers (it decides       where the boundary
                    whether the sentence names a different case, no        sits next to each
                    word lists); the heads, with the line moved a little   line, and to hold
                    by the sentence, match that answer                     still when nothing
                                                                           that matters moved
``teach_rewrites``  the teacher writes each labeled line again: in other   the meaning lock:
                    words with the reason kept, and changed as a WHY       the encoder itself
                    sentence says; it reads both. The heads, on the real   learns that wording
                    rewritten text, hold their answer on the first and     is not meaning, so
                    take the teacher's reading on the second. A reworded   a keyword shortcut
                    line the teacher itself reads differently is dropped   stops paying in
                                                                           training
==================  =====================================================  ==================

``near_miss_loss`` lives in losses.py because it needs no teacher: lines
that read alike but were labeled differently must each prefer their own
answer by a margin. The teacher learns the supposition pass it uses for
``teach_flip`` from look-alike lines (losses.supposed_twins): line i's page
with line j supposed must answer j's label.

``teach_pointers`` and ``teach_words`` are read from the teacher, not parsed: the teacher's
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

import dataclasses
from dataclasses import dataclass, field

import torch
from torch.nn import functional as F

from .data import IGNORE, Batch, judgment_identity, number_tokens
from .model import C3Model, C3Output


@dataclass
class TeachWeights:
    heads: float = 1.0
    judgments: float = 1.0
    numbers: float = 0.2
    geometry: float = 0.5
    # Opt in after teacher calibration/warmup, not from an untrained teacher.
    unlabeled: float = 0.0
    unlabeled_min_confidence: float = 0.9
    pointers: float = 0.5
    words: float = 0.5
    flips: float = 0.5
    flip_size: float = 0.05     # the sentence moves the line only a little
    rewrites: float = 0.5
    temperature: float = 2.0

    @classmethod
    def for_base(cls) -> "TeachWeights":
        """First usable version.

        The person's mark is the lesson wherever they made one. The paragraph
        fills only the questions they left blank. Nothing is copied onto lines
        nobody labeled, and the helper does not rewrite lines or imagine a
        different sentence. Card verdicts stay the person's.
        """
        return cls(flips=0.0, rewrites=0.0, unlabeled=0.0, judgments=0.0)


@dataclass(frozen=True)
class Rewrite:
    """Line ``line`` written again by the teacher (Brain.rewrite). ``same``:
    other words, the same reason; otherwise changed as a WHY sentence says."""
    line: int
    text: str
    same: bool


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
    flipped: dict[str, torch.Tensor] | None = None  # supposition pass, row n = batch.flips[n] [F, A]
    rewrites: list[Rewrite] = field(default_factory=list)   # kept after the round trip
    rewritten: dict[str, torch.Tensor] | None = None  # target per rewrite, row n = rewrites[n] [R, A]
    card_logits: dict[str, torch.Tensor] = field(default_factory=dict)
    card_subjects: dict[str, tuple] = field(default_factory=dict)
    told_numbers: dict[str, torch.Tensor] = field(default_factory=dict)


@torch.no_grad()
def teacher_view(teacher, batch: Batch, company: str | None = None,
                 grounding: bool = True, rewrites: list[Rewrite] | None = None,
                 graph=None, ground_rows: list[int] | None = None) -> TeacherView:
    from .brain import company_field_notes, notes_from_deal, teacher_cards, told_why  # noqa: PLC0415

    was = teacher.training
    teacher.eval()
    desc = teacher.describe()
    notes, cnotes = notes_from_deal(batch)
    why = told_why(batch, teacher.schema)
    told = teacher(batch, company=company, why=why, desc=desc, graph=graph,
                   company_notes=company_field_notes(batch, teacher.schema))
    noted = teacher(batch, company=company, notes=notes, company_notes=cnotes, desc=desc,
                    graph=graph)
    emb = told.pooled
    pointers, words = teacher.grounding(batch, rows=ground_rows) if grounding else (None, None)
    flipped = (teacher.flip_read(batch, why, batch.flips, notes, desc, graph).logits
               if batch.flips else None)
    kept, rewritten = read_rewrites(teacher, batch, rewrites or [], why, told.logits, desc)
    readable = dataclasses.replace(batch, judged=teacher_cards(teacher, batch))
    card_logits = teacher.card_logits(readable, with_why=True, desc=desc, graph=graph)
    grouped = {}
    for j in readable.judged:
        grouped.setdefault(j.key, []).append(j)
    card_subjects = {key: tuple(judgment_identity(batch, j) for j in js)
                     for key, js in grouped.items()}
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
                       torch.tensor(batch.labeled, device=dev), pointers, words, flipped,
                       kept, rewritten, card_logits, card_subjects, told.numbers)


def with_texts(batch: Batch, repl: dict[int, str]) -> Batch:
    """The same deal with some lines' text replaced."""
    texts, numbers = list(batch.texts), list(batch.numbers)
    for i, t in repl.items():
        texts[i], numbers[i] = t, number_tokens(t)
    return dataclasses.replace(batch, texts=texts, numbers=numbers)


@torch.no_grad()
def rewrite_lines(teacher, batch: Batch, max_new: int = 80) -> list[Rewrite]:
    """Two rewrites per labeled line with a WHY, both written by the teacher:
    one in other words with the reason kept, one changed as the line's first
    supposed sentence says. Training only, once per deal; costs one short
    generation per rewrite. Empty or unchanged text is dropped."""
    from .brain import told_why  # noqa: PLC0415

    why = told_why(batch, teacher.schema)
    first = {}
    for f in batch.flips:
        first.setdefault(f.line, f.condition)
    out: list[Rewrite] = []
    for i, w in enumerate(why):
        if not w or not batch.labeled[i]:
            continue
        for change in [None] + ([first[i]] if i in first else []):
            t = teacher.rewrite(batch, i, w, change, max_new)
            if t and t.strip() != batch.texts[i].strip():
                out.append(Rewrite(i, t, change is None))
    return out


def read_rewrites(teacher, batch: Batch, rewrites: list[Rewrite], why: list[str | None],
                  told: dict[str, torch.Tensor], desc
                  ) -> tuple[list[Rewrite], dict[str, torch.Tensor] | None]:
    """The teacher reads each rewritten line in its deal, with the WHY.

    A reworded line ("same") must read as the original did on the type: if
    the teacher's own reading moves, the rewrite changed the meaning and is
    dropped (the round trip). Its target is the original reading, so the
    heads learn to hold. A changed line's target is whatever the teacher
    reads: it decides what the change did."""
    if not rewrites:
        return [], None
    pages = [teacher.page(with_texts(batch, {r.line: r.text}), r.line)
             + (f"Why: {why[r.line]}\n" if why[r.line] else "") + "Answer:" for r in rewrites]
    read, _, _, _ = teacher.read(pages, "universal", desc)
    keys = [k for k in read.logits if k in told]
    if not keys:
        return [], None
    keep: list[int] = []
    for n, r in enumerate(rewrites):
        k = "col:label_type" if "col:label_type" in keys else keys[0]
        multiple = teacher.schema.by_key()[k].multiple
        agrees = (torch.equal(read.logits[k][n] > 0, told[k][r.line] > 0) if multiple
                  else int(read.logits[k][n].argmax()) == int(told[k][r.line].argmax()))
        if not r.same or agrees:
            keep.append(n)
    if not keep:
        return [], None
    kept = [rewrites[n] for n in keep]
    target = {k: torch.stack([told[k][r.line] if r.same else read.logits[k][n]
                              for n, r in zip(keep, kept)]) for k in keys}
    return kept, target


def teach_rewrites(model: C3Model, batch: Batch, view: TeacherView, t: float) -> torch.Tensor:
    """``teach_rewrites``: the heads run on the deal with lines replaced by
    their rewrites (real text through the whole trunk, not a latent move)
    and match each rewrite's target. Each forward replaces at most one
    rewrite per line."""
    zero = next(model.parameters()).new_zeros(())
    if not view.rewrites or view.rewritten is None:
        return zero
    rounds: list[list[int]] = []
    for n, r in enumerate(view.rewrites):
        for g in rounds:
            if all(view.rewrites[m].line != r.line for m in g):
                g.append(n)
                break
        else:
            rounds.append([n])
    terms = []
    for g in rounds:
        b = with_texts(batch, {view.rewrites[n].line: view.rewrites[n].text for n in g})
        out = model(b.inputs())
        lines = [view.rewrites[n].line for n in g]
        terms += [_kl(out.logits[k][lines], view.rewritten[k][g], t, model.schema.by_key()[k].multiple)
                  for k in view.rewritten if k in out.logits]
    return torch.stack(terms).mean() if terms else zero


def rows_the_paragraph_may_teach(batch: Batch | None, key: str, has_why: torch.Tensor) -> torch.Tensor:
    """Lines where a paragraph may teach ``key``.

    A saved mark already teaches that cell. The paragraph teaches the same
    line's blank questions, which is how one note covers many heads.
    """
    if batch is None:
        return has_why
    if key in batch.multi_targets:
        unmarked = [choice is None for choice in batch.multi_targets[key]]
    else:
        gold = batch.targets.get(key)
        if not gold:
            return has_why
        unmarked = [y == IGNORE for y in gold]
    if len(unmarked) != int(has_why.shape[0]):
        return has_why
    return has_why & torch.tensor(unmarked, device=has_why.device)


def _bernoulli_entropy(p: torch.Tensor) -> torch.Tensor:
    """H(p) per element, in float32. F.binary_cross_entropy(p, p) is the same
    number but is refused under CUDA autocast."""
    p = p.float()
    return -(p * p.log() + (1 - p) * (1 - p).log())


def _kl(student: torch.Tensor, teacher: torch.Tensor, t: float, multiple: bool = False) -> torch.Tensor:
    if multiple:
        p = (teacher.float() / t).sigmoid().detach().clamp(1e-7, 1 - 1e-7)
        return (F.binary_cross_entropy_with_logits(student.float() / t, p)
                - _bernoulli_entropy(p).mean()) * t * t
    return F.kl_div((student / t).log_softmax(-1), (teacher.detach() / t).log_softmax(-1),
                    log_target=True, reduction="batchmean") * t * t


def supercharge_loss(model: C3Model, out: C3Output, view: TeacherView,
                     w: TeachWeights | None = None,
                     texts: list[str] | None = None,
                     batch: Batch | None = None, desc=None,
                     unlabeled_keys: set[str] | None = None) -> dict[str, torch.Tensor]:
    """The teaching terms for one deal. ``out`` is the heads' own forward pass
    (no teacher inside it); ``view`` is detached. ``batch`` (with its flips)
    turns on ``teach_flip``."""
    w = w or TeachWeights()
    zero = out.r.new_zeros(())
    keys = [k for k in view.told if k in out.logits]
    parts = {"teach_heads": zero, "teach_geometry": zero, "teach_unlabeled": zero,
             "teach_pointers": zero, "teach_words": zero, "teach_flip": zero, "teach_flip_size": zero,
             "teach_rewrites": zero, "teach_judgments": zero, "teach_numbers": zero}
    if batch is not None and view.flipped is not None and batch.flips:
        parts["teach_flip"], parts["teach_flip_size"] = teach_flip(model, out, view, batch,
                                                                   w.temperature, desc)
    if batch is not None and view.rewrites:
        parts["teach_rewrites"] = teach_rewrites(model, batch, view, w.temperature)
    if batch is not None and view.card_logits:
        grouped = {}
        for j in batch.judged:
            grouped.setdefault(j.key, []).append(j)
        terms = []
        for key, teacher_logits in view.card_logits.items():
            # The view holds the cards the teacher could read whole.
            subjects = view.card_subjects.get(key, ())
            read = set(subjects)
            js = [j for j in grouped.get(key, []) if judgment_identity(batch, j) in read]
            identity = tuple(judgment_identity(batch, j) for j in js)
            if identity != subjects or teacher_logits.shape[0] != len(subjects):
                raise ValueError("Teacher card subjects do not match this batch")
            o = model.schema.by_key()[key]
            if not o.universal and out.conduct_x is None:
                continue
            logits = model.judge(out, key, [j.lines for j in js], desc, queries=[j.query for j in js])
            weights = logits.new_tensor([j.why_weight for j in js])
            losses = torch.stack([_kl(logits[i:i+1], teacher_logits[i:i+1], w.temperature, o.multiple)
                                  for i in range(len(js))])
            terms.append((losses * weights).sum() / weights.sum().clamp(min=1e-6))
        parts["teach_judgments"] = torch.stack(terms).mean() if terms else zero
    if not keys:
        return parts
    why = view.has_why
    if why.any():
        taught = []
        for key in keys:
            rows = rows_the_paragraph_may_teach(batch, key, why)
            if rows.any():
                taught.append(_kl(out.logits[key][rows], view.told[key][rows], w.temperature,
                                  model.schema.by_key()[key].multiple))
        if taught:
            parts["teach_heads"] = torch.stack(taught).mean()
        number_terms = []
        for key, teacher_num in view.told_numbers.items():
            student = out.numbers.get(key)
            if student is None or student.shape != teacher_num.shape:
                continue
            rows = why
            vals = batch.numbers_target.get(key) if batch is not None else None
            if vals is not None and len(vals) == int(why.shape[0]):
                rows = why & torch.tensor([v is None for v in vals], device=why.device)
            if rows.any():
                number_terms.append(F.smooth_l1_loss(student[rows], teacher_num.detach()[rows]))
        if number_terms:
            parts["teach_numbers"] = torch.stack(number_terms).mean()
    if int(why.sum()) > 2:
        s = F.normalize(out.r[why], dim=-1)
        t = F.normalize(view.embedding[why], dim=-1)
        eye = torch.eye(s.shape[0], dtype=torch.bool, device=s.device)
        ss = (s @ s.T / 0.1).masked_fill(eye, -1e4)
        tt = (t @ t.T / 0.1).masked_fill(eye, -1e4)
        parts["teach_geometry"] = F.kl_div(ss.log_softmax(-1), tt.log_softmax(-1),
                                           log_target=True, reduction="batchmean")
    rest = ~view.labeled
    if rest.any() and w.unlabeled > 0:
        terms = []
        if not 0.5 < w.unlabeled_min_confidence <= 1.0:
            raise ValueError("Unlabeled confidence threshold must be in (0.5, 1]")
        for k in keys:
            if unlabeled_keys is not None and k not in unlabeled_keys:
                continue
            multiple = model.schema.by_key()[k].multiple
            teacher = view.noted[k].detach()
            p = teacher.sigmoid() if multiple else teacher.softmax(-1)
            confidence = torch.maximum(p, 1 - p).amin(-1) if multiple else p.amax(-1)
            eligible = rest & (confidence >= w.unlabeled_min_confidence)
            if eligible.any():
                terms.append(_kl(out.logits[k][eligible], teacher[eligible], w.temperature, multiple))
        if terms:
            parts["teach_unlabeled"] = torch.stack(terms).mean()
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


def teach_flip(model: C3Model, out: C3Output, view: TeacherView, batch: Batch,
               t: float, desc=None) -> tuple[torch.Tensor, torch.Tensor]:
    """``teach_flip``: for each supposed sentence, the heads on the line moved
    by the sentence's text (``flip_proj``, training only) match the teacher's
    reading of the page with that sentence assumed, on every universal head.
    Where the teacher's answer changes, the heads must change with a small
    move, so the boundary sits right beside the line, along what the
    sentence says; where it does not, they learn to hold. Returns (the KL,
    the move's size relative to the line, which stays small)."""
    from .losses import moved_heads  # noqa: PLC0415 (cycle)

    zero = out.r.new_zeros(())
    fl = list(batch.flips)
    move = model.flip_proj(model.text([f.condition for f in fl]))
    sub = moved_heads(model, out, [f.line for f in fl], move,
                      desc if desc is not None else model.describe())
    rows = [f.line for f in fl]
    size = (move.pow(2).sum(-1) / out.r[rows].detach().pow(2).sum(-1).clamp(min=1e-6)).mean()
    keys = [k for k in view.flipped if k in sub.logits]
    if not keys:
        return zero, size
    # A supposition from a draft WHY nobody saved counts DRAFT_WHY_WEIGHT.
    from .losses import why_row_weights  # noqa: PLC0415 (cycle)
    ww = why_row_weights(batch, out.r.device)[rows]

    def kl_rows(s: torch.Tensor, tch: torch.Tensor, multiple: bool) -> torch.Tensor:
        if multiple:
            p = (tch.float() / t).sigmoid().detach().clamp(1e-7, 1 - 1e-7)
            per = (F.binary_cross_entropy_with_logits(s.float() / t, p, reduction="none")
                   - _bernoulli_entropy(p)).mean(-1) * t * t
            return (per * ww).sum() / ww.sum()
        per = F.kl_div((s / t).log_softmax(-1), (tch / t).log_softmax(-1),
                       log_target=True, reduction="none").sum(-1) * t * t
        return (per * ww).sum() / ww.sum()

    return torch.stack([kl_rows(sub.logits[k], view.flipped[k], model.schema.by_key()[k].multiple)
                        for k in keys]).mean(), size


def weighted(parts: dict[str, torch.Tensor], w: TeachWeights | None = None) -> torch.Tensor:
    w = w or TeachWeights()
    return (w.heads * parts["teach_heads"] + w.geometry * parts["teach_geometry"]
            + w.unlabeled * parts["teach_unlabeled"] + w.pointers * parts["teach_pointers"]
            + w.words * parts["teach_words"] + w.flips * parts.get("teach_flip", 0.0)
            + w.flip_size * parts.get("teach_flip_size", 0.0)
            + w.rewrites * parts.get("teach_rewrites", 0.0)
            + w.judgments * parts.get("teach_judgments", 0.0)
            + w.numbers * parts.get("teach_numbers", 0.0))


def mask_labels(batch: Batch, keep: set[int]) -> Batch:
    """The same deal with only lines in ``keep`` labeled: targets, numbers,
    WHYs, company lines and changes of every other line are removed."""
    import dataclasses

    def col(v):
        return [x if i in keep else IGNORE for i, x in enumerate(v)]

    return dataclasses.replace(
        batch,
        targets={k: col(v) for k, v in batch.targets.items()},
        multi_targets={k: [x if i in keep else None for i, x in enumerate(v)]
                       for k, v in batch.multi_targets.items()},
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
        flips=[f for f in batch.flips if f.line in keep],
        near_misses=[p for p in batch.near_misses if p[0] in keep and p[1] in keep],
        twins=[p for p in batch.twins if p[0] in keep and p[1] in keep],
    )
