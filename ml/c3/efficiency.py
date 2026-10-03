"""Label efficiency: do the long labels make each label count for more?

The claim v6 makes is measurable: train the same heads on the same labeled
lines twice, once from the labels alone and once with the explanations as
privileged teaching (supercharge.py), and compare on held-out lines. Repeat
at several label budgets and seeds; the gap between the two curves, and how
many labels the labels-only arm needs to catch up, is the answer.

==========  ===========================================================
arm         what it trains on
==========  ===========================================================
labels      the k labeled lines' fields; no WHY anywhere (why terms off)
explained   the same k lines' fields and WHYs: the teacher reads them,
            the heads learn from the labels, their WHYs and the teacher
==========  ===========================================================

Both arms start from the same initial weights and see the same k lines.
Held-out lines are never labeled in either arm.

On the synthetic fixture this only proves the harness runs: 14 invented
lines say nothing about real label efficiency. The real run belongs to the
training thread, on the labeled deals, with a pretrained teacher.

    python -m ml.c3.efficiency --deal ml/c3/fixtures/synthetic_deal.json \\
        --budgets 2,4,6 --seeds 2 --steps 30
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Callable

import torch

from .brain import Brain
from .data import IGNORE, Batch, DealExample, featurize
from .lm import TinyCausalLM
from .losses import BrainWeights, LossWeights, brain_loss, c3_loss
from .model import C3Config, C3Model
from .schema import Schema, load_schema
from .supercharge import TeachWeights, mask_labels, supercharge_loss, teacher_view, weighted

SMALL = C3Config(d_text=64, d=64, n_layers=1, n_heads=4, d_r=32, d_q=16, d_head=32,
                 residual_dim=8, box_dim=8, n_policy_atoms=4, policy_rank=4)
NO_WHY = dict(why_align=0.0, why_sufficiency=0.0, why_echo=0.0)


@dataclass
class Row:
    budget: int
    seed: int
    arm: str
    accuracy: float
    n_eval: int
    keep: tuple[int, ...]
    near_miss: float = float("nan")   # held-out near-miss pairs with both lines right
    n_pairs: int = 0
    twin: float = float("nan")        # held-out twins given the same answers
    n_twins: int = 0


def heads_accuracy(model: C3Model, batch: Batch, eval_ids: list[int]) -> tuple[float, int]:
    """Mean argmax accuracy over every (held-out line, universal head) pair
    with a gold answer."""
    model.eval()
    with torch.no_grad():
        out = model(batch.inputs())
    hit = total = 0
    for o in model.schema.select(layer="universal"):
        if o.key not in out.logits:
            continue
        pred = out.logits[o.key].argmax(-1)
        for i in eval_ids:
            y = batch.targets[o.key][i]
            if y != IGNORE:
                hit += int(pred[i] == y)
                total += 1
    return (hit / total if total else float("nan")), total


def near_miss_accuracy(model: C3Model, batch: Batch, eval_ids: list[int]) -> tuple[float, int]:
    """The small-difference test: of the near-miss pairs (lines that read
    almost alike but answer differently) with both lines held out, the share
    where the heads get both right. Averaging the two scores at most half."""
    held = set(eval_ids)
    pairs = [p for p in batch.near_misses if p[0] in held and p[1] in held]
    if not pairs:
        return float("nan"), 0
    model.eval()
    with torch.no_grad():
        out = model(batch.inputs())
    hit = 0
    for i, j, key in pairs:
        pred = out.logits[key].argmax(-1)
        hit += int(pred[i] == batch.targets[key][i] and pred[j] == batch.targets[key][j])
    return hit / len(pairs), len(pairs)


def twin_consistency(model: C3Model, batch: Batch, eval_ids: list[int]) -> tuple[float, int]:
    """The keyword test: of the twins (lines that read almost alike and were
    labeled the same) with both lines held out, the share where the heads
    give both lines the same answer on every universal line head either has
    gold for. A model that keys on surface words splits twins whose wording
    differs where the answer does not care (55 inch vs 65 inch); one that
    reads the line does not. No word list picks the twins: the labels do."""
    held = set(eval_ids)
    pairs = [p for p in batch.twins if p[0] in held and p[1] in held]
    if not pairs:
        return float("nan"), 0
    model.eval()
    with torch.no_grad():
        out = model(batch.inputs())
    keys = [o.key for o in model.schema.select(layer="universal") if o.key in out.logits]
    hit = 0
    for i, j in pairs:
        ks = [k for k in keys if k in batch.targets
              and IGNORE not in (batch.targets[k][i], batch.targets[k][j])]
        hit += int(all(out.logits[k][i].argmax() == out.logits[k][j].argmax() for k in ks))
    return hit / len(pairs), len(pairs)


def _train_heads(schema: Schema, batch: Batch, seed: int, steps: int, cfg: C3Config,
                 teacher: Brain | None, lr: float) -> C3Model:
    torch.manual_seed(seed)
    model = C3Model(schema, cfg)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    w = LossWeights(clause_use=0.0, **({} if teacher is not None else NO_WHY))
    view = teacher_view(teacher, batch, batch.company or None) if teacher is not None else None
    for _ in range(steps):
        model.train()
        loss, _ = c3_loss(model, batch, w)
        if view is not None:
            out = model(batch.inputs(), company=batch.company or None)
            loss = loss + weighted(supercharge_loss(model, out, view, texts=batch.texts,
                                                    batch=batch), TeachWeights())
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.ema_update()
    return model


def _train_teacher(schema: Schema, batch: Batch, seed: int, steps: int, lr: float,
                   make_lm: Callable[[], torch.nn.Module]) -> Brain:
    torch.manual_seed(seed)
    teacher = Brain(schema, make_lm(), context_lines=2)
    opt = torch.optim.Adam(teacher.parameters(), lr=lr)
    for _ in range(steps):
        teacher.train()
        loss, _ = brain_loss(teacher, batch, BrainWeights(consolidate=0.0))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return teacher


def run_curve(deal: DealExample, schema: Schema | None = None, budgets=(2, 4, 6), seeds=(0,),
              steps: int = 30, teacher_steps: int = 10, cfg: C3Config = SMALL,
              lr: float = 3e-3,
              make_lm: Callable[[], torch.nn.Module] = lambda: TinyCausalLM(32, 1, 2, 1024),
              ) -> list[Row]:
    schema = schema or load_schema()
    full = featurize(deal, schema)
    labeled = [i for i, x in enumerate(full.labeled) if x]
    rows: list[Row] = []
    for seed in seeds:
        rng = random.Random(seed)
        ids = labeled[:]
        rng.shuffle(ids)
        eval_ids = sorted(ids[: max(1, len(ids) // 3)])
        pool = ids[len(eval_ids):]
        for k in budgets:
            if k > len(pool):
                continue
            keep = set(rng.sample(pool, k))
            train = mask_labels(full, keep)
            plain = mask_labels(full, keep)
            plain.why = [None] * len(plain)
            plain.flips = []          # suppositions come from WHYs: the labels-only arm has none
            m_lab = _train_heads(schema, plain, seed, steps, cfg, None, lr)
            teacher = _train_teacher(schema, train, seed, teacher_steps, lr, make_lm)
            m_exp = _train_heads(schema, train, seed, steps, cfg, teacher, lr)
            for arm, m in (("labels", m_lab), ("explained", m_exp)):
                acc, n = heads_accuracy(m, full, eval_ids)
                nm, n_nm = near_miss_accuracy(m, full, eval_ids)
                tw, n_tw = twin_consistency(m, full, eval_ids)
                rows.append(Row(k, seed, arm, acc, n, tuple(sorted(keep)), nm, n_nm, tw, n_tw))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--deal", required=True)
    ap.add_argument("--budgets", default="2,4,6")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--teacher-steps", type=int, default=10)
    a = ap.parse_args()
    rows = run_curve(DealExample.load(a.deal), budgets=[int(x) for x in a.budgets.split(",")],
                     seeds=range(a.seeds), steps=a.steps, teacher_steps=a.teacher_steps)
    print(f"{'labels':>6}  {'seed':>4}  {'arm':<9}  {'accuracy':>8}  {'pairs':>5}  {'near-miss':>9}  {'n':>3}  {'twins':>5}  {'n':>3}")
    for r in rows:
        print(f"{r.budget:>6}  {r.seed:>4}  {r.arm:<9}  {r.accuracy:>8.3f}  {r.n_eval:>5}"
              f"  {r.near_miss:>9.3f}  {r.n_pairs:>3}  {r.twin:>5.3f}  {r.n_twins:>3}")


if __name__ == "__main__":
    main()
