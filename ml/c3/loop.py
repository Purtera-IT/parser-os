"""Base training schedule for one model on a locked train / holdout split.

The release entrypoint stays ``ml.c3.gates.train``, which still refuses.
This loop is the base experiment: fixed steps, the holdout scored once at
the end, never used to stop, tune, or set a threshold.

Order, one shared model, universal questions and the company layer in the
same pass (the company layer runs because each deal names its company):

1. Teacher alone. It learns to repeat saved marks and write the paragraphs.
   The student does not step.
2. Student on saved marks, link gate shut: line readings, then links and
   abstain, then pair and group cards, then numbers, then whole-deal answers.
3. The same decision loss opens the link gate, so a link the model draws
   can change a classification.
4. The teacher's paragraph fills blank cells only.
5. The teacher is shown the deal graph the student just built, and a small
   projection learns to read it. The language-model body stays frozen.
6. A light pass on the "what if this sentence" boundaries.
7. High-confidence fill only on questions that appeared many times.

Rewrites stay off: the teacher does not invent a new line. ``READY_FOR_TRAINING``
is not changed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any
import time
import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import torch

from .brain import Brain
from .coverage import READY_FOR_TRAINING
from .data import IGNORE, Batch, DealExample
from .gates import save_checkpoint
from .holdout import LABEL_HOLDOUT, refuse_fixed_holdout
from .lm import TinyCausalLM
from .losses import BrainWeights, LossWeights, brain_loss, c3_loss, teacher_chunks, teacher_context
from .longdeal import plan_long_deal
from .model import C3Config, C3Model
from .readiness import digest
from .schema import GROUP, NUMBER, PAIR, RELATION, Schema, load_schema
from .supercharge import TeachWeights, supercharge_loss, teacher_view, weighted

SMALL = C3Config(d_text=64, d=64, n_layers=1, n_heads=4, d_r=32, d_q=16, d_head=32,
                 residual_dim=8, box_dim=8, n_policy_atoms=4, policy_rank=4)

STAGES = ("teacher", "lines", "links", "cards", "numbers", "deal", "open_links",
         "bind", "teach", "flips", "spread")
# A question seen fewer times than this is never copied onto unlabeled lines.
MIN_MARKS = 8


@dataclass(frozen=True)
class Budget:
    """Fixed step counts. One step walks every training deal once."""

    teacher: int = 20
    lines: int = 20
    links: int = 20
    cards: int = 20
    numbers: int = 10
    deal: int = 10
    open_links: int = 20
    teach: int = 20
    bind: int = 0
    flips: int = 0
    spread: int = 0

    def steps(self, stage: str) -> int:
        return int(getattr(self, stage))


def with_stage_steps(budget: Budget, spec: str) -> Budget:
    """``lines=30,links=0``: the named stages' step counts replaced."""
    changes = {}
    for part in filter(None, (x.strip() for x in str(spec or "").split(","))):
        stage, _, n = part.partition("=")
        if stage not in Budget.__dataclass_fields__ or not n.strip().isdigit():
            raise SystemExit(f"--stage-steps: cannot read {part!r}")
        changes[stage] = int(n)
    return replace(budget, **changes)


@dataclass
class BaseRun:
    train_ids: tuple[str, ...]
    holdout_ids: tuple[str, ...]
    stages: tuple[str, ...]
    heads_seen: dict[str, int]
    holdout_scores: dict[str, dict[str, dict[str, float]]]
    holdout_used_for_tuning: bool = False
    unlabeled_enabled: bool = False
    ready_for_training: bool = field(default_factory=lambda: READY_FOR_TRAINING)
    stage_loss: dict[str, float] = field(default_factory=dict)
    #: One-shot scores on the validation deals after each student stage,
    #: pooled over their parts. This is what chooses steps and thresholds.
    validation_scores: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    validation_ids: tuple[str, ...] = ()


def source_deal_id(deal_id: str) -> str:
    """Window ids are ``<deal>#c3part<n>``. The source deal is the part before that."""
    head, sep, _ = deal_id.partition("#c3part")
    # A view of a deal ("010003@pre_quote", "010003@planning") is that deal.
    return (head if sep else deal_id).partition("@")[0]


def lock_split(train: list[Batch], holdout: list[Batch],
               need_holdout: bool = True) -> tuple[list[Batch], list[Batch]]:
    """Refuse a holdout deal that is also a training deal, including its windows,
    and any fixed holdout (``holdout.FIXED_HOLDOUT``) in training. A run that
    does not score the holdout may leave it out; the fixed holdout still
    cannot train."""
    if not train or (need_holdout and not holdout):
        raise ValueError("The base needs both a training set and a holdout")
    train_ids = [b.deal_id for b in train]
    hold_ids = [b.deal_id for b in holdout]
    if len(train_ids) != len(set(train_ids)) or len(hold_ids) != len(set(hold_ids)):
        raise ValueError("Duplicate deal id in a split")
    refuse_fixed_holdout(train_ids)
    shared = {source_deal_id(i) for i in train_ids} & {source_deal_id(i) for i in hold_ids}
    if shared:
        raise ValueError("Holdout deal is in the training set: " + ", ".join(sorted(shared)))
    return train, holdout


def lock_validation(train: list[Batch], holdout: list[Batch], validation: list[Batch]) -> None:
    """Validation deals are training deals' peers: never trained, never a holdout."""
    if not validation:
        return
    ids = {source_deal_id(b.deal_id) for b in validation}
    refuse_fixed_holdout(ids)
    clash = ids & ({source_deal_id(b.deal_id) for b in train} | {source_deal_id(b.deal_id) for b in holdout})
    if clash:
        raise ValueError("Validation deal is also in training or the holdout: " + ", ".join(sorted(clash)))


def part_scales(train: list[Batch]) -> dict[str, float]:
    """Loss scale per part: a deal cut into k parts counts sqrt(k), not k.

    One long deal would otherwise outvote the small ones by its length alone.
    """
    parts: dict[str, int] = {}
    for b in train:
        src = source_deal_id(b.deal_id)
        parts[src] = parts.get(src, 0) + 1
    return {b.deal_id: parts[source_deal_id(b.deal_id)] ** -0.5 for b in train}


def pool_scores(rows: list[dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    """Exact match pooled over parts or deals, weighted by how many were scored."""
    hit: dict[str, float] = {}
    n: dict[str, float] = {}
    for row in rows:
        for key, v in row.items():
            hit[key] = hit.get(key, 0.0) + v["exact"] * v["n"]
            n[key] = n.get(key, 0.0) + v["n"]
    return {key: {"exact": hit[key] / n[key], "n": n[key]} for key in n if n[key]}


def majority_baseline(train: list[Batch], validation: list[Batch]) -> dict[str, float]:
    """Exact match on the validation deals of always answering each
    single-answer question's most common training answer: the floor a
    trained head has to clear."""
    counts: dict[str, dict[int, int]] = {}
    for b in train:
        for key, ys in b.targets.items():
            for y in ys:
                if y != IGNORE:
                    counts.setdefault(key, {})[y] = counts.setdefault(key, {}).get(y, 0) + 1
    hit: dict[str, int] = {}
    n: dict[str, int] = {}
    for b in validation:
        for key, ys in b.targets.items():
            if key not in counts:
                continue
            top = max(counts[key], key=counts[key].get)
            for y in ys:
                if y != IGNORE:
                    hit[key] = hit.get(key, 0) + int(y == top)
                    n[key] = n.get(key, 0) + 1
    return {key: hit[key] / n[key] for key in n}


def heads_seen(train: list[Batch]) -> dict[str, int]:
    """How many saved marks each question has on the training deals."""
    counts: dict[str, int] = {}
    for batch in train:
        for key, ys in batch.targets.items():
            n = sum(y != IGNORE for y in ys)
            if n:
                counts[key] = counts.get(key, 0) + n
        for key, rows in batch.multi_targets.items():
            n = sum(row is not None for row in rows)
            if n:
                counts[key] = counts.get(key, 0) + n
        for item in batch.judged:
            counts[item.key] = counts.get(item.key, 0) + 1
        for rel, edges in batch.edges.items():
            if edges:
                key = f"rel:{rel}"
                counts[key] = counts.get(key, 0) + len(edges)
        if "declared_site_count" in batch.deal_numbers:
            counts["deal:declared_site_count"] = counts.get("deal:declared_site_count", 0) + 1
    return counts


def frequent_heads(counts: dict[str, int], minimum: int = MIN_MARKS) -> set[str]:
    """Questions seen often enough that a confident blank may be filled."""
    return {key for key, n in counts.items() if n >= minimum and not key.startswith("rel:")}


def gold_weights(stage: str) -> tuple[LossWeights, tuple[str, ...] | None]:
    """Cumulative saved-mark weights. Later stages keep the earlier lessons.

    Card sizes: pair and group cards before whole-deal cards. None means
    every card size.
    """
    weights = LossWeights(relations=0.0, governs=0.0, judgments=0.0, judgment_rationale=0.0,
                          numbers=0.0, literals=0.0, deal_numbers=0.0)
    if stage == "lines":
        return weights, None
    weights = replace(weights, relations=0.5, governs=0.5)
    if stage == "links":
        return weights, None
    weights = replace(weights, judgments=1.0, judgment_rationale=0.2)
    if stage == "cards":
        return weights, (PAIR, GROUP)
    weights = replace(weights, numbers=0.2, literals=0.4)
    if stage == "numbers":
        return weights, (PAIR, GROUP)
    weights = replace(weights, deal_numbers=0.2)
    return weights, None


def _freeze_link_gate(model: C3Model, trainable: bool) -> None:
    model.link_mix.weight.requires_grad_(trainable)
    model.link_mix.bias.requires_grad_(trainable)


# A 7B teacher in bf16 fits beside ModernBERT-large and the student on an H100.
H100_TEACHER = "Qwen/Qwen2.5-7B-Instruct"


#: A progress line every this many parts, so a long stage is never silent.
PROGRESS_EVERY = int(os.environ.get("C3_PROGRESS_EVERY", "25"))
# The live teacher's memory caps. A call is one pass through the 7B body; a
# piece is what one backward holds (checkpointed: about 0.2 MB a token).
TEACHER_CALL_TOKENS = 32768
# Qwen2.5's own window. Most pages are far shorter; the longest saved lines
# and card evidence need it, and nothing is cut to fit.
TEACHER_MAX_LEN = 32768
TEACHER_PIECE_TOKENS = 131072
# What one teacher step teaches of a part: lines, cards, flips. A fresh
# sample each step (seeded); a part of 8,192 lines would otherwise take hours.
TEACHER_CAPS = (256, 128, 32)
# Lines per part whose WHY the teacher's lesson view grounds (a forward and
# backward through the 7B each); the rest of the view still covers every line.
VIEW_GROUND_ROWS = 128
# Parts per lesson stage that get a teacher view (and so a lesson): a view
# is ~4 teacher passes over every line of a part, ~20 minutes at 8,192 lines.
VIEW_PARTS = 6
# Heads every check also reads on validation, beside line type (a lesson
# stage cost run A's task tier; a gate can name ``valid_jdg_task_tier``).
WATCH_CHECK = ("jdg:task_tier",)
# Parts per bind step.
BIND_PARTS = 16
PIECE_PROGRESS_EVERY = 8
# Heads the base must get right before anything is built on it: what a line
# is, and the sites, quantities and tasks it states. Printed on every
# validation line.
WATCH_HEADS = ("label_type", "site", "qty", "count", "task")


_LAST_PART: dict[str, float] = {}


def _progress(stage: str, done: int, total: int, started: float) -> None:
    now = time.time()
    last = now - _LAST_PART.get(stage, started) if done > 1 else now - started
    _LAST_PART[stage] = now
    if done % PROGRESS_EVERY == 0 or done == total:
        rate = (now - started) / max(done, 1)
        print(f"progress {stage} {done}/{total} parts {rate:.1f}s/part last {last:.1f}s "
              f"eta {rate * (total - done) / 60:.0f} min", flush=True)


def _finite(loss: torch.Tensor, stage: str) -> None:
    if not torch.isfinite(loss.detach()):
        raise RuntimeError(f"Non-finite loss during {stage}")


def _arm_teacher_memory(teacher: Brain) -> None:
    """Checkpoint the pretrained teacher so a long page fits beside the student.

    Checkpointing only runs while the body is in train mode, and only if the
    token embeddings request grad. Dropout stays off so the frozen body does
    not jitter.
    """
    inner = getattr(teacher.lm, "model", None)
    if inner is None:
        return
    config = getattr(inner, "config", None)
    if config is not None:
        config.use_cache = False
    enable = getattr(inner, "gradient_checkpointing_enable", None)
    if enable is not None:
        enable()
    hook = getattr(inner, "enable_input_require_grads", None)
    if hook is not None:
        hook()
    _steady_teacher(teacher)


def _steady_teacher(teacher: Brain) -> None:
    """Train mode for checkpointing, dropout off so the frozen body stays still."""
    inner = getattr(teacher.lm, "model", None)
    if inner is None:
        return
    inner.train()
    for module in inner.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()
    config = getattr(inner, "config", None)
    if config is not None:
        config.use_cache = False


def _teacher_part(teacher: Brain, batch: Batch, weights, *, device: str, scale: float,
                  size: int, budget: int, caps: tuple[int, int, int], rng: random.Random | None,
                  graph: torch.Tensor | None = None, label: str = "teacher") -> torch.Tensor:
    """One part's teacher loss, backward done, piece by piece.

    The schema descriptions are embedded once: each piece reads detached
    copies, their gradients add up, and one backward through the embedding
    pass at the end gives the exact gradient a single pass would."""
    started = time.time()
    with _autocast(device):
        full = teacher.describe()
    keys = list(full)
    leaves = {k: tuple(t.detach().requires_grad_(t.requires_grad) for t in full[k]) for k in keys}
    context = teacher_context(teacher, batch)
    pieces = teacher_chunks(teacher, batch, size, budget, caps=caps, rng=rng)
    loss, tokens = None, 0
    for n, part in enumerate(pieces, 1):
        with _autocast(device):
            piece, _ = brain_loss(teacher, batch, weights, graph=graph, rows=part.rows,
                                  judged=part.cards, flips=part.flips, extras=part.extras,
                                  desc=leaves, context=context)
            piece = piece * (scale * part.scale)
        _finite(piece, label)
        if piece.requires_grad:
            piece.backward()
        loss = piece.detach() if loss is None else loss + piece.detach()
        tokens += part.tokens
        if n % PIECE_PROGRESS_EVERY == 0 or n == len(pieces):
            print(f"progress {label} {batch.deal_id} piece {n}/{len(pieces)} "
                  f"{tokens} tok {time.time() - started:.0f}s loss {float(loss) / max(scale, 1e-9):.4f}",
                  flush=True)
    outs, grads = [], []
    for k in keys:
        for t, leaf in zip(full[k], leaves[k]):
            if leaf.grad is not None and t.requires_grad:
                outs.append(t)
                grads.append(leaf.grad)
    if outs:
        torch.autograd.backward(outs, grads)
    return loss


def teacher_state(teacher: Brain) -> dict[str, torch.Tensor]:
    """What training changed on the teacher: its low-rank slots, readers,
    paragraph memory and graph projection. The pretrained body is left out
    (it never moves, and it is 15 GB)."""
    return {name: p.detach().cpu().clone() for name, p in teacher.named_parameters()
            if not name.startswith("lm.model.") or name.endswith((".A", ".B"))}


def save_teacher(path: str, teacher: Brain, stage: str) -> None:
    torch.save({"stage": stage, "state": teacher_state(teacher)}, path)
    print(f"saved teacher {stage} {path}", flush=True)


def load_teacher(path: str, teacher: Brain) -> str:
    """Put a saved teacher's trained weights back. Every saved tensor must
    land on the teacher; returns the stage it was saved after."""
    saved = torch.load(path, map_location="cpu")
    own = dict(teacher.named_parameters())
    unknown = [k for k in saved["state"] if k not in own]
    if unknown:
        raise ValueError(f"Saved teacher does not match this teacher: {unknown[:3]}")
    with torch.no_grad():
        for k, v in saved["state"].items():
            own[k].copy_(v.to(own[k].device, own[k].dtype))
    return saved["stage"]


def _optimizer(params, lr: float, adamw: bool) -> torch.optim.Optimizer:
    live = [p for p in params if p.requires_grad]
    if not live:
        raise ValueError("No trainable parameters")
    if adamw:
        return torch.optim.AdamW(live, lr=lr, weight_decay=0.01)
    return torch.optim.Adam(live, lr=lr)


# Tests turn this on to run the CPU path under the pod's bf16 autocast.
CPU_AUTOCAST = False


def half_dtype() -> torch.dtype:
    """The half precision the encoder and autocast use on a GPU:
    ``C3_HALF_DTYPE`` = bf16 (default, what training used) | fp16. A T4 (and
    any GPU before Ampere) has no bf16 arithmetic, so it reads in fp16; the
    trained weights are the same, only how they are computed changes."""
    v = os.environ.get("C3_HALF_DTYPE", "").strip().lower()
    return torch.float16 if v in ("fp16", "float16", "half") else torch.bfloat16


def to_read_precision(model: Any) -> Any:
    """After the checkpoint loads (its encoder check reads the bf16 weights
    it was trained on), move the line encoder to ``half_dtype()`` when that
    is not bf16. Weights are unchanged; only their number format is."""
    want = half_dtype()
    body = getattr(getattr(model, "text", None), "model", None)
    if body is not None and want != torch.bfloat16:
        body.to(want)
    return model


def _autocast(device: str):
    if device.startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=half_dtype())
    if device.startswith("mps"):
        # The half-precision encoder feeds float32 layers; Metal refuses a
        # mixed matmul outside autocast.
        return torch.autocast(device_type="mps", dtype=half_dtype())
    if CPU_AUTOCAST:
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    return nullcontext()


def hold_line_encoder(model: C3Model) -> bool:
    """Freeze a pretrained line encoder. A hashed test encoder has nothing to hold."""
    body = getattr(model.text, "model", None)
    if body is None:
        return False
    for p in body.parameters():
        p.requires_grad_(False)
    body.eval()
    model._frozen_desc = None
    return True


def release_encoder_top(model: C3Model, layers: int) -> int:
    """Let the line encoder's top ``layers`` blocks (and its final norm) learn.

    The rest stays frozen. Returns how many blocks were released. Whether this
    beats a frozen encoder is a validation question, not a default.
    """
    body = getattr(model.text, "model", None)
    if body is None or layers <= 0:
        return 0
    blocks = getattr(body, "layers", None)
    if blocks is None:
        blocks = getattr(getattr(body, "encoder", None), "layer", None)
    if blocks is None:
        raise RuntimeError("Cannot find the line encoder's blocks to release")
    for block in list(blocks)[-layers:]:
        for p in block.parameters():
            p.requires_grad_(True)
    for name in ("final_norm", "norm"):
        norm = getattr(body, name, None)
        if norm is not None:
            for p in norm.parameters():
                p.requires_grad_(True)
    # The released blocks keep every line's activations for backward; at
    # 8,192 lines that is the largest thing on the card. Recompute instead.
    enable = getattr(body, "gradient_checkpointing_enable", None)
    if enable is not None:
        enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model._frozen_desc = None
    return min(layers, len(blocks))


def _student_optimizer(model: C3Model, lr: float, adamw: bool,
                       encoder_lr: float | None) -> torch.optim.Optimizer:
    """One group for the deal model and heads; a smaller rate for any released encoder blocks."""
    text = {id(p) for p in model.text.parameters()}
    released = [p for p in model.parameters() if p.requires_grad and id(p) in text]
    if not released or encoder_lr is None:
        return _optimizer(model.parameters(), lr, adamw)
    rest = [p for p in model.parameters() if p.requires_grad and id(p) not in text]
    groups = [{"params": rest, "lr": lr}, {"params": released, "lr": encoder_lr}]
    if adamw:
        return torch.optim.AdamW(groups, lr=lr, weight_decay=0.01)
    return torch.optim.Adam(groups, lr=lr)


def hold_teacher_body(teacher: Brain) -> None:
    """Keep the pretrained language model. Train its low-rank slots and the readers."""
    for p in teacher.lm.parameters():
        p.requires_grad_(False)
    for slot in teacher.lm.adapters():
        if slot.A is not None and slot.B is not None:
            slot.A.requires_grad_(True)
            slot.B.requires_grad_(True)


def _lesson(stage: str, counts: dict[str, int]) -> tuple[TeachWeights, set[str] | None]:
    """Blank-fill first. Boundaries next. Frequent questions may fill a blank line last."""
    weights = TeachWeights.for_base()
    if stage == "flips":
        return replace(weights, flips=0.2), None
    if stage == "spread":
        return replace(weights, unlabeled=0.35), frequent_heads(counts)
    return weights, None


def _deal_graph(model: C3Model, teacher: Brain, batch: Batch, device: str):
    if teacher.graph is None:
        return None
    model.eval()
    with torch.no_grad(), _autocast(device):
        return model(batch.oneshot(), company=batch.company or None).r.detach()


def _bind_graph(teacher: Brain, model: C3Model, train: list[Batch], steps: int,
                device: str, lr: float, adamw: bool, rng: random.Random,
                scale: dict[str, float] | None = None, chunk: int = 0,
                piece_tokens: int = 0, caps: tuple[int, int, int] = (0, 0, 0),
                parts: int = 0) -> None:
    """Teach the frozen language model to read the student's deal graph.

    Only the small projection trains. The pretrained body does not move.
    """
    if teacher.graph is None or steps <= 0:
        return
    for p in teacher.parameters():
        p.requires_grad_(False)
    for p in teacher.graph.parameters():
        p.requires_grad_(True)
    opt = _optimizer(teacher.graph.parameters(), lr, adamw)
    weights = BrainWeights(consolidate=0.0, flip=0.0)
    losses: list[float] = []
    for _ in range(steps):
        teacher.train()
        _steady_teacher(teacher)
        # Binding needs the graph read on a fair sample of parts, not on all.
        order = rng.sample(train, parts) if parts and len(train) > parts else train[:]
        rng.shuffle(order)
        for batch in order:
            graph = _deal_graph(model, teacher, batch, device)
            opt.zero_grad()
            loss = _teacher_part(teacher, batch, weights, device=device,
                                 scale=(scale or {}).get(batch.deal_id, 1.0), size=chunk,
                                 budget=piece_tokens, caps=caps, rng=rng if any(caps) else None,
                                 graph=graph, label="bind")
            torch.nn.utils.clip_grad_norm_([p for p in teacher.graph.parameters() if p.grad is not None], 1.0)
            opt.step()
            losses.append(float(loss.detach()))
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    if losses:
        print(f"stage bind loss {sum(losses) / len(losses):.4f}", flush=True)


def _student_step(model: C3Model, opt: torch.optim.Optimizer, batch: Batch,
                  weights: LossWeights, sizes: tuple[str, ...] | None,
                  teacher: Brain | None, view, device: str,
                  lesson: TeachWeights | None = None,
                  unlabeled_keys: set[str] | None = None,
                  scale: float = 1.0, shares: dict[str, list[float]] | None = None) -> torch.Tensor:
    model.train()
    with _autocast(device):
        kept: dict = {}
        loss, parts = c3_loss(model, batch, weights, judgment_sizes=sizes, keep=kept)
        if shares is not None:
            # Each term as it enters the total, so a stage shows what it trained on.
            for k, v in parts.items():
                shares.setdefault(k, []).append(getattr(weights, k) * v)
        if teacher is not None and view is not None:
            # The lesson reads the same forward pass as the gold terms: a
            # second trunk pass would double what backward holds.
            loss = loss + weighted(
                supercharge_loss(model, kept["out"], view, texts=batch.texts, batch=batch,
                                 desc=kept["desc"], unlabeled_keys=unlabeled_keys),
                lesson or TeachWeights.for_base())
        loss = loss * scale
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.grad is not None], 1.0)
    opt.step()
    model.ema_update()
    return loss.detach()


def _score_oneshot(model: C3Model, batch: Batch, device: str = "cpu") -> dict[str, dict[str, float]]:
    """Exact match on labeled rows from atoms only. Stage and geography are null.

    All of it runs under the same no-grad autocast: the cards and links are
    read from the pass's bf16 outputs by float32 layers."""
    model.eval()
    with torch.no_grad(), _autocast(device):
        return _score_oneshot_inner(model, batch)


def _score_oneshot_inner(model: C3Model, batch: Batch) -> dict[str, dict[str, float]]:
    out = model(batch.oneshot(), company=batch.company or None)
    scores: dict[str, dict[str, float]] = {}
    for key, logits in out.logits.items():
        if key in batch.multi_targets:
            gold = batch.multi_targets[key]
            hit = total = 0
            for i, chosen in enumerate(gold):
                if chosen is None:
                    continue
                predicted = tuple(int(j) for j in (logits[i] > 0).nonzero().flatten().tolist())
                hit += int(predicted == chosen)
                total += 1
        else:
            gold_row = batch.targets.get(key)
            if not gold_row:
                continue
            pred = logits.argmax(-1)
            hit = total = 0
            for i, y in enumerate(gold_row):
                if y == IGNORE:
                    continue
                hit += int(int(pred[i]) == y)
                total += 1
        if total:
            scores[key] = {"exact": hit / total, "n": float(total)}
    judged: dict[str, list] = {}
    for item in batch.judged:
        judged.setdefault(item.key, []).append(item)
    for key, items in judged.items():
        if key in out.logits or not hasattr(model, "judge"):
            continue
        try:
            logits = model.judge(out, key, [item.lines for item in items],
                                 queries=[item.query for item in items])
        except ValueError:
            continue
        opp = model.schema.by_key().get(key)
        if opp is not None and opp.multiple:
            hit = 0
            for i, item in enumerate(items):
                predicted = tuple(int(j) for j in (logits[i] > 0).nonzero().flatten().tolist())
                hit += int(predicted == tuple(item.answers or ()))
        else:
            pred = logits.argmax(-1)
            hit = sum(int(int(pred[i]) == item.answer) for i, item in enumerate(items))
        scores[key] = {"exact": hit / len(items), "n": float(len(items))}
    labeled = {i for i, on in enumerate(batch.labeled) if on}
    for rel, scores_row in out.relations.items():
        none = out.link_none.get(rel)
        gold: dict[int, list[int]] = {}
        for src, dst in batch.edges.get(rel, []):
            gold.setdefault(src, []).append(dst)
        hit = total = 0
        n = int(scores_row.shape[0])
        away = set(batch.link_elsewhere.get(rel, ()))
        for src in labeled:
            if src >= n or (src in away and src not in gold):
                continue
            row = scores_row[src]
            if none is not None:
                row = torch.cat([row, none[src].view(1)])
            if not torch.isfinite(row).any():
                continue
            pick = int(row.argmax())
            dsts = gold.get(src)
            if dsts:
                hit += int(pick in dsts)
                total += 1
            elif none is not None:
                hit += int(pick == n)
                total += 1
        if total:
            scores[f"rel:{rel}"] = {"exact": hit / total, "n": float(total)}
    return scores


TYPE_KEY = "col:label_type"
# Documents the live parser never reads (old manual Deal Kits and their
# estimating calcs): left out of the type checks, as in type_compare.
CHECK_SKIP_DOC_KINDS = ("deal_kit", "estimating_calc")


def type_check(model: C3Model, batches: list[Batch], device: str = "cpu", *,
               key: str = TYPE_KEY, focus: str = "task", sure: float = 0.5) -> dict[str, float]:
    """Line type on labeled lines, the way the promote path will use it.

    ``exact``: top answer. ``typed_exact``: best named type, ``_keep`` set
    aside. ``focus_recall`` / ``focus_precision``: for ``focus`` (task).
    ``routed_exact``: the base where it is at least ``sure``, the parser's
    classifier elsewhere, on lines the parser typed; ``routed_share`` is the
    share the base takes. ``parser_exact`` is the classifier alone on those
    lines, and ``margin`` the base's exact minus the classifier's there.
    Counts and rates only."""
    opp = model.schema.by_key()[key]
    names = [a.value for a in opp.answers]
    t, keep = names.index(focus) if focus in names else -1, names.index("_keep") if "_keep" in names else -1
    n = hit = typed = real = said = both = 0
    rn = rhit = phit = took = bhit = 0
    preal = pboth = bboth = 0
    model.eval()
    with torch.no_grad(), _autocast(device):
        for b in batches:
            gold = b.targets.get(key)
            if not gold or all(y == IGNORE for y in gold):
                continue
            logits = model(b.oneshot(), company=b.company or None).logits[key].float()
            prob = logits.softmax(-1)
            p, pred = prob.max(-1)
            named = prob.clone()
            if keep >= 0:
                named[:, keep] = -1
            named = named.argmax(-1)
            parser = b.parser_targets.get(key) or [IGNORE] * len(gold)
            for i, y in enumerate(gold):
                if y == IGNORE or str(b.doc_kind[i] or "").lower() in CHECK_SKIP_DOC_KINDS:
                    continue
                g = int(pred[i])
                n += 1
                hit += g == y
                typed += int(named[i]) == y
                real += y == t
                said += g == t
                both += g == y == t
                if parser[i] != IGNORE:
                    rn += 1
                    phit += parser[i] == y
                    bhit += g == y
                    sure_here = float(p[i]) >= sure
                    took += sure_here
                    rhit += (g if sure_here else parser[i]) == y
                    preal += y == t
                    pboth += parser[i] == y == t
                    bboth += g == y == t
    return {"n": float(n), "exact": hit / n if n else 0.0, "typed_exact": typed / n if n else 0.0,
            "focus_recall": both / real if real else 0.0,
            "focus_precision": both / said if said else 0.0,
            "routed_n": float(rn), "routed_exact": rhit / rn if rn else 0.0,
            "routed_share": took / rn if rn else 0.0, "parser_exact": phit / rn if rn else 0.0,
            # The base alone against the classifier on the lines both typed:
            # above 0 is what full replacement needs.
            "margin": (bhit - phit) / rn if rn else 0.0,
            # Task recall on those same lines, the base's and the classifier's.
            "pair_focus_recall": bboth / preal if preal else 0.0,
            "parser_focus_recall": pboth / preal if preal else 0.0}


def parse_gates(specs: list[str]) -> list[tuple[str, str, str, float]]:
    """``lines.valid_exact>=0.42`` or ``lines@10.fit_exact>=0.5``: after that
    stage (or that pass of it), the check must clear the bar or the run stops.
    ``fit_*`` reads the training-deal sample, ``valid_*`` the validation deals."""
    out = []
    for spec in specs:
        for op in (">=", "<="):
            if op in spec:
                left, value = spec.split(op, 1)
                at, metric = left.strip().rsplit(".", 1)
                out.append((at, metric, op, float(value)))
                break
        else:
            raise ValueError(f"A gate needs >= or <=: {spec}")
    return out


def gate_failures(gates: list[tuple[str, str, str, float]], at: str,
                  checks: dict[str, float]) -> list[str]:
    bad = []
    for where, metric, op, bar in gates:
        if where != at:
            continue
        if metric not in checks:
            bad.append(f"{metric} was not measured")
            continue
        got = checks[metric]
        if (got < bar) if op == ">=" else (got > bar):
            bad.append(f"{metric}={got:.3f} needs {op}{bar}")
    return bad


def run_base(train: list[Batch], holdout: list[Batch], schema: Schema | None = None,
             budget: Budget | None = None, cfg: C3Config = SMALL, lr: float = 3e-3,
             seed: int = 0, warmup_passes: float = 0.0,
             teacher_lr: float | None = None,
             context_lines: int = 2,
             make_lm=lambda: TinyCausalLM(32, 1, 2, 4096),
             checkpoint: str | None = None,
             model: C3Model | None = None,
             device: str = "cpu",
             adamw: bool = False,
             freeze_teacher: bool = False,
             page_batch: int = 0,
             read_chunk: int = 0,
             teacher_rows_per_step: int = 0,
             teacher_piece_tokens: int = 0,
             teacher_caps: tuple[int, int, int] = (0, 0, 0),
             view_ground_rows: int = 0,
             view_parts: int = 0, lesson_gold: int = 0,
             teacher_from: str | None = None, teacher_continue: bool = False,
             call_tokens: int = 0,
             deal_graph: bool = False,
             validation: list[Batch] | None = None,
             balance: bool = True,
             encoder_lr: float | None = None,
             score_holdout: bool = True,
             head_weight: dict[str, float] | None = None,
             loss_weight: dict[str, float] | None = None,
             free_heads: tuple[str, ...] = (),
             fit_parts: int = 8,
             check_every: int = 0,
             gates: list[tuple[str, str, str, float]] | None = None) -> tuple[C3Model, BaseRun]:
    """Train on ``train`` only. Score ``holdout`` once, with atoms only.

    The language model is the teacher. Tests pass the tiny stand-in on CPU.
    The H100 path passes a pretrained causal LM, bf16, and ``adamw``. This
    function does not certify a release and does not call ``gates.train``.
    """
    if READY_FOR_TRAINING:
        raise RuntimeError("This loop must not be the path that flips the release flag")
    train, holdout = lock_split(train, holdout, need_holdout=score_holdout)
    validation = list(validation or [])
    lock_validation(train, holdout, validation)
    scale = part_scales(train) if balance else {b.deal_id: 1.0 for b in train}
    schema = schema or load_schema()
    budget = budget or Budget()
    torch.manual_seed(seed)
    rng = random.Random(seed)
    # A run where no stage reads the teacher (no teacher passes, bind or
    # lesson, and none loaded) builds the tiny stand-in, not the 7B model:
    # nothing would read it, and it would hold 15 GB of the GPU.
    needs_teacher = bool(teacher_from) or budget.teacher > 0 or budget.bind > 0 or any(
        budget.steps(s) > 0 for s in ("teach", "flips", "spread"))
    lm = make_lm() if needs_teacher else TinyCausalLM(32, 1, 2, 4096)
    if not needs_teacher:
        print("no stage reads the teacher; the language model is not loaded", flush=True)
    teacher = Brain(schema, lm, context_lines=context_lines, page_batch=page_batch,
                    read_chunk=read_chunk, call_tokens=call_tokens, d_graph=cfg.d_r if deal_graph else None)
    if freeze_teacher:
        hold_teacher_body(teacher)
        _arm_teacher_memory(teacher)
    # The graph projection has nothing to read until the student exists.
    if teacher.graph is not None:
        for p in teacher.graph.parameters():
            p.requires_grad_(False)
    teacher.to(device)
    teacher_opt = _optimizer(teacher.parameters(), teacher_lr if teacher_lr is not None else lr, adamw)
    teacher_updates = max(1, budget.teacher * len(train))
    teacher_sched = (torch.optim.lr_scheduler.CosineAnnealingLR(teacher_opt, T_max=teacher_updates)
                     if adamw else None)
    if adamw and teacher_from and teacher_continue:
        # A trained teacher resumes on a fresh Adam: ramp it up as the student does.
        t_warm = max(1, len(train) // 4)
        teacher_sched = torch.optim.lr_scheduler.LambdaLR(
            teacher_opt, lambda s: min(1.0, (s + 1) / t_warm)
            * 0.5 * (1 + math.cos(math.pi * min(s, teacher_updates) / teacher_updates)))
    stage_loss: dict[str, float] = {}

    # The teacher trains when the first stage that reads it comes up (bind, or
    # a lesson): the gold-only student stages do not need it, so their
    # validation scores arrive within the first hour, not after the teacher.
    def train_teacher() -> None:
        if teacher_from:
            print(f"teacher loaded from {teacher_from} (after {load_teacher(teacher_from, teacher)})",
                  flush=True)
            if not teacher_continue:
                for p in teacher.parameters():
                    p.requires_grad_(False)
                teacher.eval()
                return
        teacher_w = BrainWeights(consolidate=0.0, flip=0.0)
        teacher_losses: list[float] = []
        started, done = time.time(), 0
        for _ in range(budget.teacher):
            teacher.train()
            if freeze_teacher:
                _steady_teacher(teacher)
            order = train[:]
            rng.shuffle(order)
            for batch in order:
                # One part's loss, a piece at a time: each piece's graph is freed
                # by its own backward, so memory holds one piece, not the part.
                teacher_opt.zero_grad()
                loss = _teacher_part(teacher, batch, teacher_w, device=device, scale=scale[batch.deal_id],
                                     size=teacher_rows_per_step, budget=teacher_piece_tokens,
                                     caps=teacher_caps, rng=rng if any(teacher_caps) else None)
                torch.nn.utils.clip_grad_norm_([p for p in teacher.parameters() if p.grad is not None], 1.0)
                teacher_opt.step()
                if teacher_sched is not None:
                    teacher_sched.step()
                teacher_losses.append(float(loss.detach()))
                done += 1
                _progress("teacher", done, budget.teacher * len(train), started)
        if teacher_losses:
            stage_loss["teacher"] = sum(teacher_losses) / len(teacher_losses)
            print(f"stage teacher loss {stage_loss['teacher']:.4f}", flush=True)
        for p in teacher.parameters():
            p.requires_grad_(False)
        teacher.eval()
        if checkpoint:
            save_teacher(f"{checkpoint}.teacher", teacher, "teacher")

    model = model or C3Model(schema, cfg)
    for key in free_heads:
        if key == "all":
            for o in schema.opportunities:
                if o.universal and o.kind not in (RELATION, NUMBER) and len(o.answers) > 1:
                    model.add_free_head(o.key)
        else:
            model.add_free_head(key)
    model.to(device)
    _freeze_link_gate(model, False)
    gates = list(gates or [])
    # The fit check reads a fixed sample of training parts: a base that cannot
    # type the lines it trained on is under-fit, whatever validation says.
    fit_pool = (sorted(random.Random(seed + 1).sample(train, fit_parts), key=lambda b: b.deal_id)
                if fit_parts and len(train) > fit_parts else (train if fit_parts else []))
    checks: dict[str, dict[str, float]] = {}
    stopped = ""

    def check(at: str) -> list[str]:
        row: dict[str, float] = {}
        if TYPE_KEY in schema.by_key():
            if fit_pool:
                row.update({f"fit_{k}": v for k, v in type_check(model, fit_pool, device).items()})
            if validation:
                row.update({f"valid_{k}": v for k, v in type_check(model, validation, device).items()})
        if validation and WATCH_CHECK:
            pooled = pool_scores([_score_oneshot(model, b, device) for b in validation])
            for key in WATCH_CHECK:
                if key in pooled:
                    row[f"valid_{key.replace(':', '_')}"] = pooled[key]["exact"]
        checks[at] = row
        if row:
            print(f"check {at} " + " ".join(f"{k}={v:.3f}" for k, v in row.items()
                                            if not k.endswith(("_n", "n_"))), flush=True)
        if checkpoint:
            Path(f"{checkpoint}.checks.json").write_text(json.dumps(checks, indent=1, sort_keys=True),
                                                         encoding="utf-8")
        bad = gate_failures(gates, at, row)
        if bad:
            print(f"GATE FAILED at {at}: " + "; ".join(bad), flush=True)
        return bad
    opt = _student_optimizer(model, lr, adamw, encoder_lr)
    lesson_stages = {"teach", "flips", "spread"}
    student_stages = []
    for stage in ("lines", "links", "cards", "numbers", "deal", "open_links", "bind", "teach", "flips", "spread"):
        if stage == "bind":
            if budget.bind > 0 and teacher.graph is not None:
                student_stages.append(stage)
            continue
        if stage in lesson_stages and (budget.steps(stage) == 0 or budget.teacher == 0):
            continue
        student_stages.append(stage)
    student_updates = sum(budget.steps(s) * len(train) for s in student_stages if s != "bind")
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, student_updates))
             if adamw else None)
    # A run that starts from trained heads ramps the rate up: a fresh Adam
    # moves every weight by the full rate on its first steps, which in run C's
    # smoke undid the fitted line-type heads in four updates.
    warm = int(warmup_passes * len(train))
    if adamw and warm > 0:
        total = max(1, student_updates)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    seen = heads_seen(train)
    validation_scores: dict[str, dict[str, dict[str, float]]] = {}
    views = None
    completed = 0
    schema_sha = digest([asdict(o) for o in schema.opportunities]) if checkpoint else ""
    baseline = majority_baseline(train, validation) if validation else {}
    if baseline:
        print(f"validation baseline single_heads={len(baseline)} "
              f"majority={sum(baseline.values()) / len(baseline):.4f}", flush=True)
    # Where the run starts, read the same way as every later check: a start
    # the checks already disagree with (a different part size, say) shows
    # here, before any step moves it.
    if fit_pool or validation:
        check("start")
    teacher_ready = False
    for stage in student_stages:
        fresh = False
        if not teacher_ready and (stage == "bind" or stage in lesson_stages):
            train_teacher()
            teacher_ready = True
        if stage == "open_links":
            _freeze_link_gate(model, True)
            opt.add_param_group({
                "params": [model.link_mix.weight, model.link_mix.bias],
                "lr": opt.param_groups[0]["lr"],
                "initial_lr": opt.param_groups[0].get("initial_lr", lr),
                "weight_decay": 0.0,
            })
        if stage == "bind":
            _bind_graph(teacher, model, train, budget.bind, device, lr, adamw, rng, scale,
                        chunk=teacher_rows_per_step, piece_tokens=teacher_piece_tokens,
                        caps=teacher_caps, parts=view_parts and BIND_PARTS)
            if checkpoint:
                save_teacher(f"{checkpoint}.teacher", teacher, "bind")
            continue
        weights, sizes = gold_weights("deal" if stage in ("open_links",) or stage in lesson_stages else stage)
        if head_weight:
            weights = replace(weights, head=dict(head_weight))
        if loss_weight:
            # Only terms the stage already trains: a 0 in gold_weights stays 0.
            weights = replace(weights, **{k: v for k, v in loss_weight.items() if getattr(weights, k)})
        use_teacher = stage in lesson_stages
        lesson, allowed = _lesson(stage, seen) if use_teacher else (None, None)
        steps = budget.steps(stage)
        stage_losses: list[float] = []
        shares: dict[str, list[float]] = {}
        started, done = time.time(), 0
        for pass_i in range(steps):
            # The teacher's view is read once per lesson stage (it follows the
            # student's deal graph from stage to stage, not step to step).
            if use_teacher and (views is None or (teacher.graph is not None and not fresh)):
                views, fresh = {}, True
                seen_at = time.time()
                # A lesson reads the teacher on a seeded sample of parts: one
                # view is ~4 teacher passes over every line of a part.
                pool = (sorted(rng.sample(train, view_parts), key=lambda b: b.deal_id)
                        if view_parts and len(train) > view_parts else train)
                for k, batch in enumerate(pool, 1):
                    graph = _deal_graph(model, teacher, batch, device)
                    ground = None
                    if view_ground_rows:
                        told = [i for i in range(len(batch)) if batch.why[i]]
                        ground = (sorted(rng.sample(told, view_ground_rows))
                                  if len(told) > view_ground_rows else told)
                    with _autocast(device):
                        views[batch.deal_id] = teacher_view(
                            teacher, batch, batch.company or None, graph=graph, ground_rows=ground)
                    print(f"progress view {stage} {k}/{len(pool)} parts "
                          f"{time.time() - seen_at:.0f}s", flush=True)
                # Part times below are the student's, not the views'.
                started = time.time()
                _LAST_PART.pop(stage, None)
            # A lesson trains on the parts it has a view of; the gold-only
            # stages already walked every part.
            order = [b for b in train if b.deal_id in views] if use_teacher and views else train[:]
            # Gold steps on parts without a view keep a lesson from pulling
            # the base toward the few parts it can see (run A's teach stage
            # read 6 of 71 parts ten times and lost task tier, 1.000 to .656).
            if use_teacher and views and lesson_gold:
                rest = [b for b in train if b.deal_id not in views]
                order += rng.sample(rest, min(len(rest), lesson_gold * len(order)))
            rng.shuffle(order)
            for batch in order:
                taught = use_teacher and bool(views) and batch.deal_id in views
                step_loss = _student_step(model, opt, batch, weights, sizes,
                                         teacher if taught else None,
                                         views.get(batch.deal_id) if taught else None,
                                         device, lesson if taught else None,
                                         allowed if taught else None, scale[batch.deal_id], shares)
                _finite(step_loss, stage)
                stage_losses.append(float(step_loss.detach()))
                completed += 1
                done += 1
                _progress(stage, done, steps * len(train), started)
                if sched is not None:
                    sched.step()
            passed = pass_i + 1
            if check_every and passed % check_every == 0 and passed < steps:
                at = f"{stage}@{passed}"
                if checkpoint:
                    save_checkpoint(f"{checkpoint}.{at}", model, step=completed, schema_sha256=schema_sha)
                if check(at):
                    stopped = at
                    break
        if stage_losses:
            stage_loss[stage] = sum(stage_losses) / len(stage_losses)
            print(f"stage {stage} loss {stage_loss[stage]:.4f}", flush=True)
            means = {k: sum(v) / len(v) for k, v in shares.items() if v}
            top = sorted(((k, m) for k, m in means.items() if abs(m) > 1e-4), key=lambda kv: -abs(kv[1]))
            print(f"stage {stage} terms " + " ".join(f"{k}={m:.3f}" for k, m in top), flush=True)
        if validation and steps:
            pooled = pool_scores([_score_oneshot(model, b, device) for b in validation])
            validation_scores[stage] = pooled
            macro = sum(v["exact"] for v in pooled.values()) / max(1, len(pooled))
            same = [k for k in baseline if k in pooled]
            mine = sum(pooled[k]["exact"] for k in same) / max(1, len(same))
            floor = sum(baseline[k] for k in same) / max(1, len(same))
            beat = sum(pooled[k]["exact"] > baseline[k] for k in same)
            print(f"validation {stage} heads={len(pooled)} macro_exact={macro:.4f} "
                  f"single_heads={len(same)} macro={mine:.4f} majority={floor:.4f} "
                  f"beat_majority={beat}/{len(same)}", flush=True)
            watched = [k for k in sorted(pooled) if any(w in k for w in WATCH_HEADS)]
            if watched:
                print(f"validation {stage} watch " + " ".join(
                    f"{k}={pooled[k]['exact']:.3f}"
                    + (f"/maj{baseline[k]:.3f}" if k in baseline else "") + f"(n={int(pooled[k]['n'])})"
                    for k in watched), flush=True)
        if steps and not stopped and check(stage):
            stopped = stage
        # A stage that took no step would only copy the last file (3.4 GB on
        # the live model), so it saves nothing.
        if checkpoint and completed and steps:
            save_checkpoint(checkpoint, model, step=completed, schema_sha256=schema_sha)
            # Every stage keeps its own copy, so a later stage that makes
            # validation worse costs a file pick, not a rerun.
            save_checkpoint(f"{checkpoint}.{stage}", model, step=completed, schema_sha256=schema_sha)
            if validation_scores:
                Path(f"{checkpoint}.validation.json").write_text(
                    json.dumps(validation_scores, indent=1, sort_keys=True), encoding="utf-8")
        if stopped:
            print(f"run stopped at {stopped}: a gate failed; later stages did not run", flush=True)
            break
    if not teacher_ready and not stopped:
        train_teacher()
    if checkpoint and budget.teacher and teacher_ready:
        save_teacher(f"{checkpoint}.teacher", teacher, "end")
    if checkpoint and completed:
        save_checkpoint(checkpoint, model, step=completed, schema_sha256=schema_sha)
    # The live runs leave the holdout to `ml.c3.predict --score`, once, on the
    # run the validation deals picked.
    scores = ({batch.deal_id: _score_oneshot(model, batch, device) for batch in holdout}
              if score_holdout else {})
    return model, BaseRun(
        train_ids=tuple(b.deal_id for b in train),
        holdout_ids=tuple(b.deal_id for b in holdout),
        stages=STAGES,
        heads_seen=seen,
        holdout_scores=scores,
        unlabeled_enabled=budget.spread > 0,
        stage_loss=stage_loss,
        validation_scores=validation_scores,
        validation_ids=tuple(sorted({source_deal_id(b.deal_id) for b in validation})),
    )


def training_batches(deal: DealExample, schema: Schema, *, max_atoms: int = 4096) -> tuple[list[Batch], list[str]]:
    """One batch when the deal fits. Otherwise windows, edge packs and a digest, never a truncated deal."""
    overlap = min(256, max(0, max_atoms // 4))
    plan = plan_long_deal(deal, schema, max_atoms=max_atoms, overlap=overlap)
    parts = plan.parts
    notes = [f"{deal.deal_id}: {note}" for note in plan.notes]
    notes += [f"{deal.deal_id}: {note}" for part in parts for note in part.notes]
    if len(parts) == 1:
        return parts, notes
    return [replace(part, deal_id=f"{deal.deal_id}#c3part{i}") for i, part in enumerate(parts)], notes


def load_deal(path: str | Path) -> DealExample:
    """A joined deal file, or a labeling export that still has ``labels`` and ``atoms``."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("A training file must be one JSON object")
    from .export import is_page_export

    if is_page_export(payload):
        raise ValueError("This is the labeling page's export. Convert it first: "
                         "python -m ml.c3.export --page <file> --deal-id <deal> --output <training file>")
    labels = payload.get("labels")
    atoms = payload.get("atoms")
    if isinstance(labels, list) and isinstance(atoms, list):
        deal_id = str(payload.get("deal_id") or "").strip()
        if not deal_id:
            raise ValueError("A labeling export needs a deal_id")
        deal = DealExample.from_training_blob(payload, atoms, deal_id=deal_id,
                                              company=str(payload.get("company") or "purtera"))
        policy = payload.get("company_policy")
        if policy:
            deal.company_policy = str(policy)
        return deal
    if "atoms" not in payload or "deal_id" not in payload:
        raise ValueError("A training file needs atoms and a deal_id, or labels plus atoms")
    return DealExample.from_dict(payload)


def load_split(paths: list[str], schema: Schema, max_atoms: int = 4096) -> tuple[list[Batch], list[str]]:
    batches: list[Batch] = []
    notes: list[str] = []
    for path in paths:
        parts, part_notes = training_batches(load_deal(path), schema, max_atoms=max_atoms)
        batches.extend(parts)
        notes.extend(part_notes)
    return batches, notes


def require_live_teacher(teacher: str, tiny: bool) -> None:
    """The CPU stand-in is opt-in. A real base names its pretrained teacher."""
    if tiny:
        return
    if not str(teacher).strip():
        raise ValueError(
            "Pass --teacher with a pretrained causal LM. "
            "The tiny language model is only for a CPU check (--tiny).")


def live_student(schema: Schema, encoder_name: str) -> C3Model:
    """Design widths, a frozen pretrained line encoder, and asked layers.

    The encoder stays frozen so seventeen deals learn the deal graph on top
    of language features instead of rewriting them.
    """
    from .text import HFEncoder

    cfg = C3Config.design()
    # The longest saved lines run past 9,000 characters (about 2,300 tokens);
    # ModernBERT reads 8,192.
    text = HFEncoder(encoder_name, max_length=8192, dtype=torch.bfloat16)
    if int(text.dim) != cfg.d_text:
        raise ValueError(f"Encoder width {text.dim} does not match the design width {cfg.d_text}")
    model = C3Model(schema, cfg, text=text)
    if not hold_line_encoder(model):
        raise RuntimeError("The live encoder did not freeze")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Base loop. Does not release a model.")
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--holdout", nargs="*", default=[],
                        help="The labeling holdout. Required with --score-holdout; without it the "
                             "files may be left out (the fixed holdout still cannot train)")
    parser.add_argument("--validation", nargs="*", default=[],
                        help="Deals scored after every stage to choose steps; never trained")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--teacher-from", default="",
                        help="A saved <checkpoint>.teacher: skip the teacher stage and use it")
    parser.add_argument("--teacher", default=H100_TEACHER,
                        help="Pretrained causal LM for the training-only teacher")
    parser.add_argument("--encoder", default="answerdotai/ModernBERT-large")
    parser.add_argument("--tiny", action="store_true", help="CPU stand-in. Not the base.")
    parser.add_argument("--max-atoms", type=int, default=4096,
                        help="Lines per window. A bigger GPU takes more, so fewer deals are cut")
    parser.add_argument("--train-encoder-layers", type=int, default=0,
                        help="Release this many top blocks of the line encoder (0 keeps it frozen)")
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--lr", type=float, default=3e-4, help="the student's rate (live path)")
    parser.add_argument("--teacher-continue", action="store_true",
                        help="train a --teacher-from teacher for --teacher-steps more passes")
    parser.add_argument("--view-parts", type=int, default=VIEW_PARTS,
                        help="parts per lesson stage that get a teacher view")
    parser.add_argument("--lesson-gold", type=int, default=0,
                        help="gold-only steps on unviewed parts per lesson step")
    parser.add_argument("--warmup-passes", type=float, default=0.0,
                        help="ramp the student's rate up over this many passes")
    parser.add_argument("--limit-parts", type=int, default=0,
                        help="Preflight only: train on the N largest parts, to measure memory and "
                             "time per part without paying for a whole pass")
    parser.add_argument("--teacher-steps", type=int, default=None,
                        help="Teacher (and bind) steps, when they should differ from --steps; "
                             "a teacher step on a large part costs far more than a student step")
    parser.add_argument("--extra-lessons", action="store_true",
                        help="Also run the flips and spread lesson stages (each reads the teacher's "
                             "view again)")
    parser.add_argument("--smoke", action="store_true",
                        help="Minutes, not hours: every stage on two average-size parts with small "
                             "teacher samples, lines twice (the second pass is the warm cost), "
                             "validation on one part, teacher saved. For catching GPU-only errors")
    parser.add_argument("--init-from", default="",
                        help="Start the student from this checkpoint instead of from scratch")
    parser.add_argument("--stage-steps", default="",
                        help="Per-stage step overrides, e.g. lines=30,links=0,teach=0; a stage "
                             "with 0 steps neither trains nor scores")
    parser.add_argument("--score-holdout", action="store_true",
                        help="Score the holdout here. Off for the live runs: the winner is scored "
                             "once with `python -m ml.c3.predict --score`")
    parser.add_argument("--head-weight", action="append", default=[],
                        help="A question's share of the heads loss, e.g. col:label_type=20 (repeat)")
    parser.add_argument("--loss-weight", action="append", default=[],
                        help="Scale a loss term in every stage that trains it, e.g. why_align=0.1 (repeat)")
    parser.add_argument("--free-head", action="append", default=[],
                        help="Give this question one free weight row per answer, e.g. col:label_type, "
                             "or all for every universal class question")
    parser.add_argument("--check-every", type=int, default=0,
                        help="Also check line type (and save a copy) every N passes inside a stage")
    parser.add_argument("--fit-parts", type=int, default=8,
                        help="Training parts the fit check reads (0 skips it)")
    parser.add_argument("--gate", action="append", default=[],
                        help="Stop the run when a check misses its bar, e.g. lines@5.fit_exact>=0.5 "
                             "or lines.valid_exact>=0.45 (repeat)")
    args = parser.parse_args()
    head_weight = {}
    for spec in args.head_weight:
        key, _, w = spec.rpartition("=")
        if not key:
            raise SystemExit(f"--head-weight: cannot read {spec!r}")
        head_weight[key] = float(w)
    loss_weight = {}
    for spec in args.loss_weight:
        key, _, w = spec.partition("=")
        if key not in LossWeights.__dataclass_fields__ or key in ("space", "head"):
            raise SystemExit(f"--loss-weight: no loss term {key!r}")
        loss_weight[key] = float(w)
    unknown = [k for k in list(head_weight) + args.free_head
               if k != "all" and k not in load_schema().by_key()]
    if unknown:
        raise SystemExit("Unknown question: " + ", ".join(unknown))
    gates = parse_gates(args.gate)
    checks = dict(head_weight=head_weight, loss_weight=loss_weight, free_heads=tuple(args.free_head),
                  fit_parts=args.fit_parts, check_every=args.check_every, gates=gates)
    require_live_teacher("" if args.tiny else args.teacher, args.tiny)
    schema = load_schema()
    teacher_steps = args.steps if args.teacher_steps is None else args.teacher_steps
    budget = Budget(teacher=teacher_steps, lines=args.steps, links=args.steps, cards=args.steps,
                    numbers=max(1, args.steps // 2), deal=max(1, args.steps // 2),
                    open_links=args.steps, teach=args.steps,
                    bind=0 if args.tiny else max(1, teacher_steps // 4),
                    flips=0 if args.tiny or not args.extra_lessons else max(1, args.steps // 4),
                    spread=0 if args.tiny or not args.extra_lessons else max(1, args.steps // 4))
    budget = with_stage_steps(budget, args.stage_steps)
    train, train_notes = load_split(args.train, schema, args.max_atoms)
    holdout, hold_notes = load_split(args.holdout, schema, args.max_atoms)
    validation, val_notes = load_split(args.validation, schema, args.max_atoms)
    caps, ground, view_parts = TEACHER_CAPS, VIEW_GROUND_ROWS, args.view_parts
    if args.smoke:
        mean = sum(len(b) for b in train) / len(train)
        near = sorted(train, key=lambda b: abs(len(b) - mean))
        # C3_SMOKE_LARGEST=1: the largest part and one average part, to
        # measure the memory peak the full run will reach.
        train = ([max(train, key=len), near[0]] if os.environ.get("C3_SMOKE_LARGEST") == "1"
                 else near[:2])
        validation = sorted(validation, key=len)[:1]
        # Every stage the run itself trains, once; a stage the run turned
        # off (``--stage-steps teach=0``, ``--teacher-steps 0``) stays off.
        full = budget
        budget = Budget(teacher=1, lines=2, links=1, cards=1, numbers=1, deal=1, open_links=1,
                        teach=1, bind=1)
        budget = replace(budget, **{f: 0 for f in asdict(budget) if getattr(full, f) == 0})
        caps, ground, view_parts = (16, 8, 4), 4, 1
        global PROGRESS_EVERY, BIND_PARTS
        PROGRESS_EVERY, BIND_PARTS = 1, 1
        print("smoke_parts", " ".join(f"{b.deal_id}:{len(b)}" for b in train), flush=True)
    if args.limit_parts:
        train = sorted(train, key=len, reverse=True)[:args.limit_parts]
        print("preflight_parts", " ".join(f"{b.deal_id}:{len(b)}" for b in train), flush=True)
    print("parts", f"train={len(train)} validation={len(validation)} holdout={len(holdout)}", flush=True)
    for note in train_notes + hold_notes + val_notes:
        print("note", note)
    held = {source_deal_id(b.deal_id) for b in holdout}
    if not args.tiny and (args.score_holdout or args.holdout) and not set(LABEL_HOLDOUT) <= held:
        raise SystemExit("The base scores the whole labeling holdout. Missing: "
                         + ", ".join(i for i in LABEL_HOLDOUT if i not in held))
    if args.tiny:
        print("model", "tiny")
        # Real deals have hundreds of lines: the stand-in reads them 8 pages at a time.
        _, result = run_base(train, holdout, schema, budget, checkpoint=args.checkpoint or None,
                             validation=validation, page_batch=8,
                             score_holdout=args.score_holdout or bool(holdout), **checks)
    else:
        from .lm import HFCausalLM

        if not torch.cuda.is_available():
            raise SystemExit("This base run is for the H100. CUDA is not available. --tiny is the CPU check.")
        torch.set_float32_matmul_precision("high")
        print("model", "h100", args.encoder, args.teacher)
        student = live_student(schema, args.encoder).to("cuda")
        if args.init_from:
            from .gates import load_checkpoint

            step = load_checkpoint(args.init_from, student,
                                   schema_sha256=digest([asdict(o) for o in schema.opportunities]))
            print("init_from", args.init_from, "step", step, flush=True)
        released = release_encoder_top(student, args.train_encoder_layers)
        print("encoder_blocks_training", released, flush=True)
        _, result = run_base(
            train, holdout, schema, budget, cfg=student.cfg,
            make_lm=lambda: HFCausalLM(
                args.teacher, targets=("q_proj", "k_proj", "v_proj", "o_proj"),
                max_len=TEACHER_MAX_LEN, trainable_rank=32, dtype=torch.bfloat16),
            checkpoint=args.checkpoint or None, model=student, device="cuda",
            lr=args.lr, warmup_passes=args.warmup_passes, teacher_lr=1e-4, context_lines=6,
            adamw=True, freeze_teacher=True, page_batch=8, read_chunk=16,
            call_tokens=TEACHER_CALL_TOKENS, teacher_rows_per_step=32,
            teacher_piece_tokens=TEACHER_PIECE_TOKENS, teacher_caps=caps,
            view_ground_rows=ground, view_parts=view_parts, lesson_gold=args.lesson_gold, deal_graph=True,
            validation=validation, encoder_lr=args.encoder_lr if released else None,
            score_holdout=args.score_holdout, teacher_from=args.teacher_from or None,
            teacher_continue=args.teacher_continue, **checks)
    print("train", " ".join(result.train_ids))
    print("holdout", " ".join(result.holdout_ids))
    print("validation", " ".join(result.validation_ids) or "(none)")
    print("holdout_used_for_tuning", result.holdout_used_for_tuning)
    print("ready_for_training", result.ready_for_training)
    measured = []
    for deal_id, scores in result.holdout_scores.items():
        print("deal", deal_id)
        for key, row in sorted(scores.items()):
            print(f"  {key} exact={row['exact']:.3f} n={int(row['n'])}")
            measured.append(key)
    silent = sorted(k for k in result.heads_seen if k not in measured)
    print("not_measured", " ".join(silent) if silent else "(none)")


if __name__ == "__main__":
    main()
