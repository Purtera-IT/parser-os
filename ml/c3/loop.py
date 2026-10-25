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
import hashlib
import json
import math
import os
from typing import Any
import time
import random
import re
import sys
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import torch

from .brain import Brain
from .coverage import READY_FOR_TRAINING
from .data import IGNORE, Batch, DealExample
from .gates import load_checkpoint, save_checkpoint
from .holdout import LABEL_HOLDOUT, SELECTION, refuse_fixed_holdout
from .lm import TinyCausalLM
from .losses import BrainWeights, LossWeights, brain_loss, c3_loss, teacher_chunks, teacher_context
from .longdeal import plan_long_deal
from .pick import PICK_HEADS, best_at, read_floors
from .model import C3Config, C3Model
from .readiness import digest
from .graph import graph_parameters, parse_graph
from .schema import GROUP, LINE, NUMBER, PAIR, RELATION, Schema, load_schema, shape_digest
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
    selected_as_test = {source_deal_id(i) for i in hold_ids} & SELECTION
    if selected_as_test:
        raise ValueError("Selection deals cannot be reused as holdout tests: " + ", ".join(sorted(selected_as_test)))
    shared = {source_deal_id(i) for i in train_ids} & {source_deal_id(i) for i in hold_ids}
    if shared:
        raise ValueError("Holdout deal is in the training set: " + ", ".join(sorted(shared)))
    return train, holdout


def lock_validation(train: list[Batch], holdout: list[Batch], validation: list[Batch]) -> None:
    """Validation deals are training deals' peers: never trained, never a holdout."""
    if not validation:
        return
    ids = {source_deal_id(b.deal_id) for b in validation}
    refuse_fixed_holdout(ids, allow_selection=True)
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


def training_scales(train: list[Batch], *, balance: bool = True, cap: float | None = None) -> dict[str, float]:
    """Compose the explicit deal multiplier with the existing part balance.

    Both are outside row/head normalization. All views and windows from a
    source deal must carry the same multiplier; a partial rebuild is refused.
    """
    from .data import _deal_loss_scale

    seen: dict[str, float] = {}
    for batch in train:
        value = _deal_loss_scale(batch.deal_loss_scale)
        source = source_deal_id(batch.deal_id)
        if source in seen and seen[source] != value:
            raise ValueError(f"Inconsistent deal_loss_scale for source deal {source}")
        seen[source] = value
    base = part_scales(train) if balance else {b.deal_id: 1.0 for b in train}
    scaled = {b.deal_id: base[b.deal_id] * b.deal_loss_scale for b in train}
    if cap is not None:
        from .balance import cap_scales

        mass: dict[str, float] = {}
        for batch in train:
            source = source_deal_id(batch.deal_id)
            mass[source] = mass.get(source, 0.0) + scaled[batch.deal_id]
        factors = cap_scales(mass, cap)
        scaled = {b.deal_id: scaled[b.deal_id] * factors[source_deal_id(b.deal_id)] for b in train}
    return scaled


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
    full = LossWeights()
    weights = LossWeights(relations=0.0, governs=0.0, judgments=0.0, judgment_rationale=0.0,
                          numbers=0.0, literals=0.0, deal_numbers=0.0,
                          link_why_contrast=0.0, link_why_alone=0.0,
                          derived_cards=0.0, table_kind=0.0, coherence=0.0, node_pointers=0.0)
    if stage == "lines":
        return weights, None
    # The link notes train with the links (link_why.py), and so do the table
    # nodes and their pointers (graph.py): the gate opens at this stage.
    weights = replace(weights, relations=0.5, governs=0.5, link_why_contrast=full.link_why_contrast,
                      link_why_alone=full.link_why_alone, table_kind=full.table_kind,
                      coherence=full.coherence, node_pointers=full.node_pointers)
    if stage == "links":
        return weights, None
    weights = replace(weights, judgments=1.0, judgment_rationale=0.2, derived_cards=full.derived_cards)
    # Queried line judgments (the Questions head and its fields) train with
    # the cards, not only in the tail.
    if stage == "cards":
        return weights, (PAIR, GROUP, LINE)
    weights = replace(weights, numbers=0.2, literals=0.4)
    if stage == "numbers":
        return weights, (PAIR, GROUP, LINE)
    weights = replace(weights, deal_numbers=0.2)
    return weights, None


def _freeze_link_gate(model: C3Model, trainable: bool) -> None:
    """The gate group (graph.graph_parameters): link_mix, the graph's
    zero-init maps and deal_number. Shut through the lines stage."""
    for p in _gate_params(model):
        p.requires_grad_(trainable)


def _gate_params(model: C3Model) -> list:
    return graph_parameters(model) + list(model.deal_number.parameters())


#: The stage the gate group opens at, and its warmup (passes) to a constant rate.
GATE_STAGE = "links"
GATE_WARMUP_PASSES = 1.0
#: The main groups' rate floor (share of lr) through the deal stage: the deal
#: heads train at a normal rate, not the cosine tail (fix 8).
DEAL_RATE = 0.3


def rate_lambdas(stage_steps: list[tuple[str, int]], parts: int, warm: int,
                 gate_warm_passes: float = GATE_WARMUP_PASSES, deal_rate: float = DEAL_RATE):
    """(main, gate) LambdaLR multipliers over the student's updates.

    main: warmup then cosine to 0 over every student update, floored at
    ``deal_rate`` through the deal stage. gate: 0 until ``GATE_STAGE``, then a
    linear warmup over ``gate_warm_passes`` passes, then 1 (constant)."""
    total = max(1, sum(n for _, n in stage_steps))
    spans, at = {}, 0
    for st, n in stage_steps:
        spans[st] = (at, at + n)
        at += n
    gate_at = spans.get(GATE_STAGE, (total, total))[0]
    deal = spans.get("deal", (0, 0))
    gate_warm = max(1, int(gate_warm_passes * parts))

    def main(s: int) -> float:
        f = 0.5 * (1 + math.cos(math.pi * min(s, total) / total))
        if deal[0] <= s < deal[1]:
            f = max(f, deal_rate)
        return (min(1.0, (s + 1) / warm) if warm > 0 else 1.0) * f

    def gate(s: int) -> float:
        if s < gate_at:
            return 0.0
        return min(1.0, (s - gate_at + 1) / gate_warm)

    return main, gate


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
# head_fit holds each of these to its min_watch bar.
WATCH_CHECK = ("jdg:task_tier",)
# What a loop check stores: the watched heads, the pick's gated heads
# (pick.PICK_HEADS: admission, supplier, about, line type, the SOW heads,
# co_action, sites, task tier, hours) and the support links; each row also
# stores ``valid_<key>_n`` and the step it was read at.
CHECK_HEADS = tuple(dict.fromkeys(WATCH_CHECK + PICK_HEADS + ("rel:supports",)))
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
    # Bind-only whole-body recomputation is automatically bypassed by the LM
    # whenever its adapters are trainable or gradients are disabled.
    teacher.lm.checkpoint_frozen_hidden = True
    first = next(inner.parameters(), None)
    if first is not None and first.device.type == "mps":
        # Calls contain complete pages; microbatching changes no target,
        # context window, piece sampling, or objective weight.
        teacher.page_batch = 1
        teacher.read_chunk = 1
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
                       encoder_lr: float | None, trunk_lr_scale: float = 1.0,
                       trunk_decay: float | None = None) -> torch.optim.Optimizer:
    """One group for the deal model and heads; a smaller rate for any released
    encoder blocks; and the gate group (``_gate_params``) last, in the
    optimizer from the start (shut by requires_grad and a 0 rate until the
    links stage), so no group is ever added mid-run: torch >= 2.7's scheduler
    refuses a group it was not built with.

    ``trunk_lr_scale`` != 1 puts the shared trunk (anchor.TRUNK_PREFIXES: the
    deal graph's blocks, lift and norm) in its own group at ``lr x scale``,
    second, with weight decay ``trunk_decay`` when given (0 when the trunk is
    pulled toward INIT instead of toward zero)."""
    from .anchor import trunk_ids  # noqa: PLC0415

    text = {id(p) for p in model.text.parameters()}
    gate = _gate_params(model)
    gate_ids = {id(p) for p in gate}
    released = [p for p in model.parameters() if p.requires_grad and id(p) in text]
    rest = [p for p in model.parameters() if p.requires_grad and id(p) not in text and id(p) not in gate_ids]
    if not rest:
        raise ValueError("No trainable parameters")
    trunk: list = []
    if trunk_lr_scale != 1.0:
        tid = trunk_ids(model)
        trunk = [p for p in rest if id(p) in tid]
        rest = [p for p in rest if id(p) not in tid]
    groups = [{"params": rest, "lr": lr}]
    if trunk:
        g = {"params": trunk, "lr": lr * trunk_lr_scale, "trunk": True}
        if trunk_decay is not None:
            g["weight_decay"] = trunk_decay
        groups.append(g)
    if released and encoder_lr is not None:
        groups.append({"params": released, "lr": encoder_lr})
    elif released:
        groups[0]["params"] = rest + released
    groups.append({"params": gate, "lr": lr, "weight_decay": 0.0, "gate": True})
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
                  scale: float = 1.0, shares: dict[str, list[float]] | None = None,
                  anchor=None) -> torch.Tensor:
    if getattr(batch, "task_profile", None) is not None and (teacher is not None or view is not None):
        raise ValueError("teacher/lesson execution is unsupported for isolated auxiliary tasks")
    model.train()
    with _autocast(device):
        kept: dict = {}
        loss, parts = c3_loss(model, batch, weights, judgment_sizes=sizes, keep=kept)
        if shares is not None:
            # Each term as it enters the total, so a stage shows what it trained on.
            for k, v in parts.items():
                shares.setdefault(k, []).append(getattr(weights, k) * v)
        if anchor is not None and getattr(batch, "task_profile", None) is None:
            # anchor.py: KL to INIT's predictions on this part's unlabeled rows.
            kl = anchor.loss(kept["out"], batch)
            if kl is not None:
                loss = loss + anchor.kl * kl
                if shares is not None:
                    shares.setdefault("anchor_kl", []).append(anchor.kl * float(kl.detach()))
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
    if anchor is not None and anchor.pull > 0 and getattr(batch, "task_profile", None) is None:
        # anchor.py: the decoupled L2 pull to INIT, at the trunk group's schedule.
        g = next((g for g in opt.param_groups if g.get("trunk")), opt.param_groups[0])
        anchor.pull_step(model, g["lr"] / max(g.get("initial_lr", g["lr"]), 1e-30))
    if getattr(batch, "task_profile", None) is None:
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
    auxiliary = getattr(batch, "task_profile", None) is not None
    if auxiliary:
        from .tasks import validate_batch
        validate_batch(batch)
    elif getattr(batch, "task_input_digest", ""):
        raise ValueError("materialized task profile cannot be removed")
    out = model(batch.oneshot(), company=None if auxiliary else batch.company or None)
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
    if auxiliary:
        return scores  # no invented no-link/card/forecast evaluation targets
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
        ignored = set(batch.link_ignored.get(rel, ()))
        for src in labeled:
            if src >= n or src in ignored or (src in away and src not in gold):
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


def configure_training_view_tag(model: C3Model, parts: list[Batch], enabled: bool | None,
                                *, resumed: bool = False) -> None:
    """Opt in after INIT loading; a resumed run must retain its saved input contract."""
    saved = bool(model.cfg.view_tag)
    if resumed and enabled is not None and enabled != saved:
        raise ValueError("view_tag differs from the resumed checkpoint; start a new run")
    active = saved if enabled is None else enabled
    if active:
        from .tasks import validate_batch
        for b in parts:
            if getattr(b, "task_profile", None) is not None:
                validate_batch(b)
        bad = [b.deal_id for b in parts if getattr(b, "task_profile", None) is None and (len(b.view) != len(b)
               or any(v not in ("quoting", "planning") for v in b.view))]
        if bad:
            raise ValueError("view_tag training requires rebuilt per-line view tags: " + ", ".join(bad))
    # SMALL is a shared default; this training choice must not alter another model.
    model.cfg = replace(model.cfg, view_tag=active)
    model.encoder.cfg = model.cfg


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
             deal_cap: float | None = None,
             head_cap: dict[str, float] | None = None,
             loss_weight: dict[str, float] | None = None,
             free_heads: tuple[str, ...] = (),
             fit_parts: int = 8,
             check_every: int = 0,
             gates: list[tuple[str, str, str, float]] | None = None,
             ramp: dict[str, float] | None = None,
             resume: bool = False,
             keep_checkpoints: int = 0,
             pick_floors: str | None = None,
             graph: dict[str, Any] | None = None,
             max_atoms: int | None = None,
             deal_rate: float = DEAL_RATE,
             trunk_lr_scale: float = 1.0, trunk_pull: float = 0.0,
             anchor_kl: float = 0.0, anchor_heads: tuple[str, ...] = (),
             view_tag: bool | None = None) -> tuple[C3Model, BaseRun]:
    """Train on ``train`` only. Score ``holdout`` once, with atoms only.

    The language model is the teacher. Tests pass the tiny stand-in on CPU.
    The H100 path passes a pretrained causal LM, bf16, and ``adamw``. This
    function does not certify a release and does not call ``gates.train``.

    ``ramp``: loss term -> passes. Each named term's weight grows linearly
    from 0 to its stage weight over that many passes, counted from the first
    step it is on (a fresh encoder top does not meet every new term at once).

    ``resume``: carry on from ``<checkpoint>.resume.pt`` (written at every
    saved checkpoint: the model file, the optimizer, the schedule, the step,
    the stage and pass, the checks and the random states).

    ``graph``: graph.py's cross-line flags (``graph.parse_graph``), turned on
    before the optimizer is built; saved with every checkpoint. ``max_atoms``
    is recorded in each checkpoint (the training window).

    ``trunk_lr_scale``, ``trunk_pull``, ``anchor_kl``, ``anchor_heads``: hold
    the shared trunk near the start (anchor.py): its own rate, a decoupled L2
    pull to its start weights, and a KL to the start's predictions on the
    unlabeled rows of the anchor heads, read once before the first step.

    ``keep_checkpoints``: keep at most this many saved checkpoints on disk
    (0 keeps all), plus the best one by the pick (pick.best_at against
    ``pick_floors``); ``<checkpoint>`` itself is then a link to the latest.
    """
    if READY_FOR_TRAINING:
        raise RuntimeError("This loop must not be the path that flips the release flag")
    train, holdout = lock_split(train, holdout, need_holdout=score_holdout)
    validation = list(validation or [])
    lock_validation(train, holdout, validation)
    from .balance import validate_balance_inputs

    validate_balance_inputs(train, deal_cap=deal_cap, head_caps=head_cap)
    scale = training_scales(train, balance=balance, cap=deal_cap)
    schema = schema or load_schema()
    from .balance import head_cap_scales

    head_scales = head_cap_scales(train, schema, scale, head_cap or {}, head_weight)
    # A repeated run/resume on the same Batch objects recomputes, never compounds.
    for batch in train:
        batch.head_loss_scale = head_scales[batch.deal_id]
    if deal_cap is not None or head_cap:
        print(f"balance: deal_cap={deal_cap} head_caps={head_cap or {}} "
              "on actual training parts; coefficient mass, not gradient norms", flush=True)
    budget = budget or Budget()
    from .tasks import validate_batch
    for batch in train + holdout + validation:
        if getattr(batch, "task_profile", None) is not None:
            validate_batch(batch)
        elif getattr(batch, "task_input_digest", ""):
            raise ValueError("materialized task profile cannot be removed")
    auxiliary_run = any(getattr(b, "task_profile", None) is not None for b in train)
    if auxiliary_run:
        if resume:
            raise ValueError("auxiliary resume is unsupported; use a separate fresh-optimizer candidate pass")
        if teacher_from or any(getattr(budget, k) > 0 for k in ("teacher", "bind", "teach", "flips", "spread")):
            raise ValueError("isolated auxiliary tasks require teacher/bind/lesson stages disabled")
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
        if checkpoint and not auxiliary_run:
            save_teacher(f"{checkpoint}.teacher", teacher, "teacher")

    model = model or C3Model(schema, cfg)
    for key in free_heads:
        if key == "all":
            for o in schema.opportunities:
                if o.universal and o.kind not in (RELATION, NUMBER) and len(o.answers) > 1:
                    model.add_free_head(o.key)
        else:
            model.add_free_head(key)
    if graph is not None:
        model.enable_graph(graph)
    model.to(device)
    _freeze_link_gate(model, False)
    gates = list(gates or [])
    resume_path = f"{checkpoint}.resume.pt" if checkpoint else ""
    state = None
    if resume and resume_path and Path(resume_path).exists():
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        # The saved EMA hindsight copy loads only into a model that has one.
        model.build_hindsight_teacher()
        model.to(device)
        load_checkpoint(state["model"], model,
                        schema_sha256=digest([asdict(o) for o in schema.opportunities]))
        print(f"resume from {state['model']} at {state['stage']} pass {state['passed']} "
              f"step {state['completed']}", flush=True)
    elif resume:
        print(f"resume: no {resume_path or '<checkpoint>.resume.pt'}; starting fresh", flush=True)
    configure_training_view_tag(model, train + validation, view_tag, resumed=state is not None)
    print(f"view_tag={model.cfg.view_tag}", flush=True)
    # The fit check reads a fixed sample of training parts: a base that cannot
    # type the lines it trained on is under-fit, whatever validation says.
    fit_pool = (sorted(random.Random(seed + 1).sample(train, fit_parts), key=lambda b: b.deal_id)
                if fit_parts and len(train) > fit_parts else (train if fit_parts else []))
    checks: dict[str, dict[str, float]] = dict(state["checks"]) if state else {}
    stopped = ""

    def check(at: str) -> list[str]:
        row: dict[str, float] = {}
        if TYPE_KEY in schema.by_key():
            if fit_pool:
                row.update({f"fit_{k}": v for k, v in type_check(model, fit_pool, device).items()})
            if validation:
                row.update({f"valid_{k}": v for k, v in type_check(model, validation, device).items()})
        if validation and CHECK_HEADS:
            pooled = pool_scores([_score_oneshot(model, b, device) for b in validation])
            for key in CHECK_HEADS:
                if key in pooled:
                    row[f"valid_{key.replace(':', '_')}"] = pooled[key]["exact"]
                    row[f"valid_{key.replace(':', '_')}_n"] = pooled[key]["n"]
        if row:
            row["at_step"] = float(completed)
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
    opt = _student_optimizer(model, lr, adamw, encoder_lr, trunk_lr_scale,
                             0.0 if trunk_pull > 0 else None)
    anchor = None
    if trunk_pull > 0 or (anchor_kl > 0 and anchor_heads):
        from .anchor import TrunkAnchor  # noqa: PLC0415

        missing = [k for k in anchor_heads if k not in schema.by_key()]
        if missing:
            raise ValueError("anchor heads not in the schema: " + ", ".join(missing))
        anchor = TrunkAnchor(model, pull=trunk_pull, kl=anchor_kl, heads=tuple(anchor_heads))
        anchor_path = f"{checkpoint}.anchor.pt" if checkpoint else ""
        if state is not None and anchor_path and anchor.load(anchor_path, device):
            print(f"anchor: INIT trunk and predictions from {anchor_path}", flush=True)
        else:
            if state is not None:
                print(f"anchor: no {anchor_path or '<checkpoint>.anchor.pt'} to resume from; anchoring "
                      "to the resumed weights", flush=True)
            rows = anchor.cache(model, train, device, _autocast)
            if anchor_path:
                anchor.save(anchor_path)
            print(f"anchor: trunk lr x{trunk_lr_scale:g}, pull {trunk_pull:g}/step to INIT; KL {anchor_kl:g} "
                  f"on {rows} unlabeled rows of {len(train)} parts ({','.join(anchor_heads) or 'no heads'})",
                  flush=True)
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
    # A run that starts from trained heads ramps the rate up: a fresh Adam
    # moves every weight by the full rate on its first steps, which in run C's
    # smoke undid the fitted line-type heads in four updates. One multiplier
    # per group: cosine for the deal model (floored through the deal stage),
    # 0-then-constant for the gate group (fix 1).
    warm = int(warmup_passes * len(train))
    main_rate, gate_rate = rate_lambdas([(st, budget.steps(st) * len(train)) for st in student_stages
                                         if st != "bind"], len(train), warm, deal_rate=deal_rate)
    sched = (torch.optim.lr_scheduler.LambdaLR(
        opt, [gate_rate if g.get("gate") else main_rate for g in opt.param_groups]) if adamw else None)
    seen = heads_seen(train)
    validation_scores: dict[str, dict[str, dict[str, float]]] = dict(state["validation_scores"]) if state else {}
    views = None
    completed = int(state["completed"]) if state else 0
    ramp = dict(ramp or {})
    first_on: dict[str, int] = dict(state.get("first_on", {})) if state else {}
    saved: list[str] = list(state.get("saved", [])) if state else []
    floors = read_floors(pick_floors) if pick_floors else {}
    if state:
        stage_loss.update(state.get("stage_loss", {}))
        rng.setstate(state["rng"])
        if state.get("global_rng") is not None:
            random.setstate(state["global_rng"])  # layout_lock's rewrite pick
        torch.set_rng_state(state["torch_rng"])
        if state.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    resume_at = (state["stage"], int(state["passed"])) if state else None
    pending_opt = state

    def keep(path: str, stage: str, passed: int) -> None:
        """After a saved checkpoint: the resume file, then the rotation."""
        if path not in saved:
            saved.append(path)
        if keep_checkpoints:
            best = best_at(checks, floors, exists=lambda at: f"{checkpoint}.{at}" in saved)
            best_path = f"{checkpoint}.{best}" if best else ""
            for old in [p for p in saved[:-1] if p != best_path]:
                if len([p for p in saved if p != best_path]) <= keep_checkpoints:
                    break
                if os.path.exists(old):
                    os.remove(old)
                saved.remove(old)
                print(f"checkpoint dropped {old} (keeping {keep_checkpoints} + best "
                      f"{best_path or 'none'})", flush=True)
            link(path)
        tmp = f"{resume_path}.tmp"
        torch.save({"model": path, "stage": stage, "passed": passed, "completed": completed,
                    "opt": opt.state_dict(), "sched": sched.state_dict() if sched is not None else None,
                    "checks": checks, "rng": rng.getstate(), "global_rng": random.getstate(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    "stage_loss": stage_loss, "validation_scores": validation_scores,
                    "first_on": first_on, "saved": saved}, tmp)
        os.replace(tmp, resume_path)

    def link(path: str) -> None:
        """With a rotation, ``<checkpoint>`` itself is a link to the latest file, not a copy."""
        if os.path.lexists(checkpoint):
            os.remove(checkpoint)
        os.symlink(os.path.basename(path), checkpoint)
    schema_sha = digest([asdict(o) for o in schema.opportunities]) if checkpoint else ""
    schema_shape = shape_digest(schema) if checkpoint else ""
    baseline = majority_baseline(train, validation) if validation else {}
    if baseline:
        print(f"validation baseline single_heads={len(baseline)} "
              f"majority={sum(baseline.values()) / len(baseline):.4f}", flush=True)
    # Where the run starts, read the same way as every later check: a start
    # the checks already disagree with (a different part size, say) shows
    # here, before any step moves it.
    if (fit_pool or validation) and not state:
        check("start")
    teacher_ready = False
    order_of = {s: i for i, s in enumerate(student_stages)}
    for stage in student_stages:
        fresh = False
        start_pass = 0
        if resume_at is not None:
            if resume_at[0] not in order_of:
                raise SystemExit(f"resume: stage {resume_at[0]} is not in this run's stages")
            before = order_of[stage] < order_of[resume_at[0]]
            done_here = stage == resume_at[0] and resume_at[1] >= budget.steps(stage)
            if before or done_here:
                # Already trained: only the setup a later stage relies on.
                if stage == GATE_STAGE:
                    _freeze_link_gate(model, True)
                continue
            if stage == resume_at[0]:
                start_pass = resume_at[1]
        if not teacher_ready and (stage == "bind" or stage in lesson_stages):
            train_teacher()
            teacher_ready = True
        if stage == GATE_STAGE:
            # The gate group was in the optimizer from the start (fix 1).
            _freeze_link_gate(model, True)
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
        if pending_opt is not None:
            # The optimizer's groups match the saved run's: none is ever added.
            opt.load_state_dict(pending_opt["opt"])
            if sched is not None and pending_opt.get("sched") is not None:
                sched.load_state_dict(pending_opt["sched"])
            pending_opt = None
        if steps:
            print(f"stage {stage} weights " + " ".join(
                f"{k}={getattr(weights, k):g}" for k in LossWeights.__dataclass_fields__
                if k not in ("space", "head")) + (f" head={weights.head}" if weights.head else "")
                + (" ramp " + " ".join(f"{k}:{v:g}" for k, v in ramp.items()) if ramp else ""), flush=True)
        stage_losses: list[float] = []
        shares: dict[str, list[float]] = {}
        started, done = time.time(), 0
        for pass_i in range(start_pass, steps):
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
                step_w = weights
                if ramp:
                    grow = {}
                    for k, passes in ramp.items():
                        if not getattr(weights, k):
                            continue
                        first_on.setdefault(k, completed)
                        f = min(1.0, (completed - first_on[k] + 1) / max(1.0, passes * len(train)))
                        if f < 1.0:
                            grow[k] = getattr(weights, k) * f
                    if grow:
                        step_w = replace(weights, **grow)
                step_loss = _student_step(model, opt, batch, step_w, sizes,
                                         teacher if taught else None,
                                         views.get(batch.deal_id) if taught else None,
                                         device, lesson if taught else None,
                                         allowed if taught else None, scale[batch.deal_id], shares,
                                         anchor=anchor)
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
                    save_checkpoint(f"{checkpoint}.{at}", model, step=completed, schema_sha256=schema_sha,
                                    shape_sha256=schema_shape, max_atoms=max_atoms)
                bad = check(at)
                if checkpoint:
                    keep(f"{checkpoint}.{at}", stage, passed)
                if bad:
                    stopped = at
                    break
        if stage_losses:
            stage_loss[stage] = sum(stage_losses) / len(stage_losses)
            print(f"stage {stage} loss {stage_loss[stage]:.4f}", flush=True)
            if anchor is not None and anchor.pull > 0:
                print(f"stage {stage} trunk_drift {anchor.drift(model):.4f} (||p - INIT|| / ||INIT||)", flush=True)
            means = {k: sum(v) / len(v) for k, v in shares.items() if v}
            top = sorted(((k, m) for k, m in means.items() if abs(m) > 1e-4), key=lambda kv: -abs(kv[1]))
            print(f"stage {stage} terms " + " ".join(f"{k}={m:.3f}" for k, m in top), flush=True)
            # A term the stage weights on that never moved the total: no data for it here.
            silent = sorted({k for k in shares if getattr(weights, k, 0) and abs(means.get(k, 0.0)) <= 1e-8}
                            | {k for k in ("layout_lock", "why_contrast", "why_swap", "why_alone", "keyword_dropout",
                                         "link_why_contrast", "link_why_alone")
                               if getattr(weights, k, 0) and k not in shares})
            if silent:
                print(f"stage {stage} silent " + " ".join(silent), flush=True)
            if torch.cuda.is_available():
                print(f"stage {stage} peak_gpu_gib allocated={torch.cuda.max_memory_allocated() / 2**30:.1f} "
                      f"reserved={torch.cuda.max_memory_reserved() / 2**30:.1f}", flush=True)
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
        if checkpoint and completed and steps and not (stopped and stopped != stage):
            # Every stage keeps its own copy, so a later stage that makes
            # validation worse costs a file pick, not a rerun.
            if not keep_checkpoints:
                save_checkpoint(checkpoint, model, step=completed, schema_sha256=schema_sha,
                                shape_sha256=schema_shape, max_atoms=max_atoms)
            save_checkpoint(f"{checkpoint}.{stage}", model, step=completed, schema_sha256=schema_sha,
                            shape_sha256=schema_shape, max_atoms=max_atoms)
            keep(f"{checkpoint}.{stage}", stage, steps)
            if validation_scores:
                Path(f"{checkpoint}.validation.json").write_text(
                    json.dumps(validation_scores, indent=1, sort_keys=True), encoding="utf-8")
        if stopped:
            print(f"run stopped at {stopped}: a gate failed; later stages did not run", flush=True)
            break
    if not teacher_ready and not stopped and not auxiliary_run:
        train_teacher()
    if checkpoint and budget.teacher and teacher_ready:
        save_teacher(f"{checkpoint}.teacher", teacher, "end")
    if checkpoint and completed:
        if keep_checkpoints and saved:
            link(saved[-1])
        else:
            save_checkpoint(checkpoint, model, step=completed, schema_sha256=schema_sha,
                            shape_sha256=schema_shape, max_atoms=max_atoms)
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


#: Marks a draft row the owner approved through ``--accept-drafts``.
APPROVED_DRAFT = "[owner-approved draft]"


def read_draft_approvals(path: str | Path) -> dict[str, str]:
    """The owner's approval manifest: ``{"files": [{"file": "<name>.json", "sha256": "<of its bytes>"}]}``.
    Returns file name -> sha256. Only these files' draft rows train."""
    m = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = m.get("files") if isinstance(m, dict) else None
    if not isinstance(rows, list):
        raise ValueError("A draft approval manifest needs a files list")
    out = {}
    for r in rows:
        if not isinstance(r, dict) or not r.get("file") or not r.get("sha256"):
            raise ValueError("Each approved draft needs its file name and sha256")
        # Keyed by the file's path relative to the manifest's folder (runE/approval_views.json:
        # "train_presow_approved/010050@pre_quote.json"), never the bare name: two views share a deal code.
        out[Path(str(r["file"])).as_posix().lstrip("./")] = str(r["sha256"]).lower()
    return out


def approved_sha(accepted: dict[str, str] | None, path: str | Path) -> str | None:
    """The approved sha256 for ``path``: the manifest key that is ``path`` or its trailing part."""
    if not accepted:
        return None
    p = Path(path).resolve().as_posix()
    hits = [k for k in accepted if p == k or p.endswith("/" + k)]
    if len(hits) > 1:
        raise ValueError(f"{path} matches more than one approval entry: {hits}")
    return accepted[hits[0]] if hits else None


def accept_drafts(payload: dict[str, Any]) -> int:
    """Turn a file's assistant-drafted rows into rows the loader trains (the
    file is on the owner's approval manifest). The row keeps who drafted it."""
    from .data import NOT_A_PERSON, _is_a_person  # noqa: PLC0415

    n = 0
    for part in ("labels", "judgments", "deal_answers", "links", "missed_history"):
        for row in payload.get(part) or []:
            if not isinstance(row, dict) or not row.get("labeler") or _is_a_person(row.get("labeler")):
                continue
            who = str(row["labeler"])
            for mark in NOT_A_PERSON:
                who = re.sub(re.escape(mark), "", who, flags=re.I)
            row["labeler"] = f"{who.strip()} {APPROVED_DRAFT}".strip()
            n += 1
        # A draft that edits a saved row carries "supersedes": the saved row's id
        # (runE merge/protect_saved.py). Approved, the draft replaces it.
        gone = {str(r["supersedes"]) for r in payload.get(part) or []
                if isinstance(r, dict) and r.get("supersedes")}
        if gone and isinstance(payload.get(part), list):
            payload[part] = [r for r in payload[part] if not (isinstance(r, dict) and str(r.get("id")) in gone
                                                              and not r.get("supersedes"))]
    return n


def load_deal(path: str | Path, accepted: dict[str, str] | None = None, *,
              task_consumer: str = "legacy") -> DealExample:
    """A joined deal file, or a labeling export that still has ``labels`` and ``atoms``.

    ``accepted`` (``--accept-drafts``): file name -> sha256 the owner approved.
    A listed file's draft rows train; a listed file whose bytes changed since
    the approval is refused. Any other file's draft rows are skipped as always."""
    raw = Path(path).read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("A training file must be one JSON object")
    if task_consumer not in ("legacy", "bounded_auxiliary"):
        raise ValueError("unknown task consumer")
    contract = payload.get("supervision_holds")
    auxiliary = isinstance(contract, dict) and contract.get("version") == 2
    if auxiliary != (task_consumer == "bounded_auxiliary"):
        raise ValueError("task consumer mismatch: v2 requires explicit bounded_auxiliary; legacy/cache/postfit refuses it")
    name = Path(path).name
    want = approved_sha(accepted, path)
    if want is not None:
        if auxiliary:
            raise ValueError("draft promotion is unsupported in the bounded auxiliary loader")
        if hashlib.sha256(raw).hexdigest() != want:
            raise ValueError(f"{name} changed since the owner approved its drafts")
        accept_drafts(payload)
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
        if policy and deal.task_profile is None:
            deal.company_policy = str(policy)
        return deal
    if "atoms" not in payload or "deal_id" not in payload:
        raise ValueError("A training file needs atoms and a deal_id, or labels plus atoms")
    return DealExample.from_dict(payload)


def load_split(paths: list[str], schema: Schema, max_atoms: int = 4096,
               accepted: dict[str, str] | None = None, *,
               task_consumer: str = "legacy") -> tuple[list[Batch], list[str]]:
    batches: list[Batch] = []
    notes: list[str] = []
    for path in paths:
        parts, part_notes = training_batches(load_deal(path, accepted, task_consumer=task_consumer), schema, max_atoms=max_atoms)
        batches.extend(parts)
        notes.extend(part_notes)
    return batches, notes


def without_heads(schema: Schema, keys: str) -> Schema:
    """The schema without the opportunities ``keys`` (comma separated) names:
    the shape a checkpoint saved before those heads were added has."""
    drop = {k.strip() for k in str(keys or "").split(",") if k.strip()}
    unknown = drop - {o.key for o in schema.opportunities}
    if unknown:
        raise SystemExit("--allow-new-heads: no such question " + ", ".join(sorted(unknown)))
    return replace(schema, opportunities=[o for o in schema.opportunities if o.key not in drop]) if drop else schema


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
    parser.add_argument("--task-consumer", choices=("legacy", "bounded_auxiliary"), default="legacy",
                        help="Explicit separate auxiliary pass; legacy/cache/postfit consumers refuse v2")
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
    parser.add_argument("--view-tag", action=argparse.BooleanOptionalAction, default=None,
                        help="Train with per-line view tags after loading INIT; default inherits the checkpoint. "
                             "A resume must match its saved flag; rebuilt view files are required")
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
    parser.add_argument("--allow-text-change", action="store_true",
                        help="With --init-from: accept a checkpoint whose schema digest differs when only "
                             "description texts changed (same opportunity keys, order and answer values); "
                             "refused otherwise. Re-run the start gate: step-0 scores move with the texts")
    parser.add_argument("--init-schema-ref", default="",
                        help="With --allow-text-change, for a checkpoint saved before shape digests: the git "
                             "commit it was trained on; its schema digest must equal the checkpoint's")
    parser.add_argument("--allow-new-heads", default="",
                        help="With --allow-text-change: opportunity keys this schema adds over the checkpoint's "
                             "(comma separated, e.g. jdg:hours:per,jdg:hours:qty). The rest must keep the "
                             "checkpoint's shape; the added heads start from their descriptions")
    parser.add_argument("--accept-drafts", default="",
                        help="The owner's approval manifest of draft training files "
                             '({"files": [{"file", "sha256"}]}): only those files\' assistant drafts train')
    parser.add_argument("--stage-steps", default="",
                        help="Per-stage step overrides, e.g. lines=30,links=0,teach=0; a stage "
                             "with 0 steps neither trains nor scores")
    parser.add_argument("--score-holdout", action="store_true",
                        help="Score the holdout here. Off for the live runs: the winner is scored "
                             "once with `python -m ml.c3.predict --score`")
    parser.add_argument("--deal-cap", type=float, default=None,
                        help="Cap source-deal loss coefficient mass over actual training parts (all views)")
    parser.add_argument("--head-cap", action="append", default=[],
                        help="Cap a head's direct CE/BCE coefficient mass per source deal, key=fraction; "
                             "infeasible sparse caps warn and keep learning")
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
    parser.add_argument("--ramp", action="append", default=[],
                        help="term=passes: grow that loss term's weight from 0 over this many passes from "
                             "the first step it is on, e.g. why_contrast=2 (repeat). Off under --smoke")
    parser.add_argument("--resume", action="store_true",
                        help="Carry on from <checkpoint>.resume.pt (optimizer, schedule, step, stage, "
                             "checks); a fresh start when there is none")
    parser.add_argument("--keep-checkpoints", type=int, default=0,
                        help="Keep at most N saved checkpoints on disk plus the best by the pick (0 keeps all)")
    parser.add_argument("--pick-floors", default="",
                        help="The live model's per-head scores (headscore scores.json shape): what the "
                             "rotation's best-by-pick reads")
    parser.add_argument("--graph", default="",
                        help="Cross-line reasoning flags (ml/c3/graph.py): rune, or key=value,... over the "
                             "defaults (mask=index|arrival|view, cut_dropout, hop_both, hop2, hop_to_h, "
                             "pointer_edge, hop_near_miss, nodes, card_slots, lattice_doc). Saved in every checkpoint")
    parser.add_argument("--validation-max-atoms", type=int, default=0,
                        help="Lines per validation window (default --max-atoms). Run E scores at serve's "
                             "window (8192), so the checks and the pick read what serve will read")
    parser.add_argument("--deal-rate", type=float, default=DEAL_RATE,
                        help="The main groups' rate floor (share of --lr) through the deal stage")
    parser.add_argument("--trunk-lr-scale", type=float, default=1.0,
                        help="The shared trunk (deal graph blocks, lift, ln) at lr x this (ml/c3/anchor.py)")
    parser.add_argument("--trunk-pull", type=float, default=0.0,
                        help="Decoupled L2 pull of the trunk to the start weights: the share of the drift "
                             "removed per step at the full rate (0 = off)")
    parser.add_argument("--anchor-kl", type=float, default=0.0,
                        help="Weight of KL(start || model) on the anchor heads' unlabeled rows (0 = off)")
    parser.add_argument("--anchor-heads", default="",
                        help="Comma-separated heads the KL anchor holds; none may be a new head")
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
    head_cap = {}
    for spec in args.head_cap:
        key, sep, value = spec.rpartition("=")
        if not sep or not key:
            raise SystemExit(f"--head-cap: cannot read {spec!r}")
        head_cap[key] = float(value)
    loss_weight = {}
    for spec in args.loss_weight:
        key, _, w = spec.partition("=")
        if key not in LossWeights.__dataclass_fields__ or key in ("space", "head"):
            raise SystemExit(f"--loss-weight: no loss term {key!r}")
        loss_weight[key] = float(w)
    unknown = [k for k in list(head_cap) + list(head_weight) + [f.removeprefix("r:") for f in args.free_head]
               if k != "all" and k not in load_schema().by_key()]
    if unknown:
        raise SystemExit("Unknown question: " + ", ".join(unknown))
    gates = parse_gates(args.gate)
    ramp = {}
    for spec in args.ramp:
        key, _, n = spec.partition("=")
        if key not in LossWeights.__dataclass_fields__ or key in ("space", "head"):
            raise SystemExit(f"--ramp: no loss term {key!r}")
        ramp[key] = float(n)
    if args.smoke and ramp:
        # The smoke measures the peak with every term at its full weight.
        print("smoke: ramp off, every term at its full weight", flush=True)
        ramp = {}
    checks = dict(deal_cap=args.deal_cap, head_cap=head_cap, head_weight=head_weight,
                  loss_weight=loss_weight, free_heads=tuple(args.free_head),
                  fit_parts=args.fit_parts, check_every=args.check_every, gates=gates, ramp=ramp,
                  resume=args.resume, keep_checkpoints=args.keep_checkpoints,
                  pick_floors=args.pick_floors or None,
                  graph=parse_graph(args.graph) if args.graph else None,
                  max_atoms=args.max_atoms, deal_rate=args.deal_rate,
                  trunk_lr_scale=args.trunk_lr_scale, trunk_pull=args.trunk_pull,
                  anchor_kl=args.anchor_kl,
                  anchor_heads=tuple(k for k in args.anchor_heads.split(",") if k),
                  view_tag=args.view_tag)
    fresh_heads = set(filter(None, (k.strip() for k in args.allow_new_heads.split(","))))
    if set(checks["anchor_heads"]) & fresh_heads:
        raise SystemExit("--anchor-heads: INIT has no prediction for a new head: "
                         + ", ".join(sorted(set(checks["anchor_heads"]) & fresh_heads)))
    if checks["graph"] is not None:
        print("graph " + " ".join(f"{k}={v}" for k, v in checks["graph"].items()), flush=True)
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
    accepted = read_draft_approvals(args.accept_drafts) if args.accept_drafts else None
    if accepted:
        print("accept_drafts", len(accepted), "approved files", flush=True)
    train, train_notes = load_split(args.train, schema, args.max_atoms, accepted, task_consumer=args.task_consumer)
    from .balance import validate_balance_inputs

    validate_balance_inputs(train, deal_cap=args.deal_cap, head_caps=head_cap)
    holdout, hold_notes = load_split(args.holdout, schema, args.max_atoms, task_consumer=args.task_consumer)
    validation, val_notes = load_split(args.validation, schema, args.validation_max_atoms or args.max_atoms,
                                     task_consumer=args.task_consumer)
    if args.validation_max_atoms:
        print(f"validation window {args.validation_max_atoms} lines (training {args.max_atoms})", flush=True)
    caps, ground, view_parts = TEACHER_CAPS, VIEW_GROUND_ROWS, args.view_parts
    if args.smoke:
        mean = sum(len(b) for b in train) / len(train)
        near = sorted(train, key=lambda b: abs(len(b) - mean))
        # C3_SMOKE_LARGEST=1: the largest part that carries layout rewrites
        # (every term fires there: the lock's forward is the third graph) and
        # the largest part overall, to measure the peak the full run reaches.
        if os.environ.get("C3_SMOKE_LARGEST") == "1":
            locked = [b for b in train if b.layout_rewrites and any(b.why)]
            pick_parts = ([max(locked, key=len)] if locked else []) + [max(train, key=len)]
            pick_parts = list({id(b): b for b in pick_parts}.values())
            if len(pick_parts) < 2:
                pick_parts.append(next(b for b in near if b is not pick_parts[0]))
            if not locked:
                print("smoke: no part carries layout rewrites; layout_lock cannot fire", flush=True)
            train = pick_parts
        else:
            train = near[:2]
        # The validation part with the most labeled lines, so the smoke's checks score something.
        validation = sorted(validation, key=lambda b: -sum(b.labeled))[:1]
        # Every stage the run itself trains, once; a stage the run turned
        # off (``--stage-steps teach=0``, ``--teacher-steps 0``) stays off.
        full = budget
        budget = Budget(teacher=1, lines=2, links=1, cards=1, numbers=1, deal=1, open_links=1,
                        teach=1, bind=1)
        budget = replace(budget, **{f: 0 for f in asdict(budget) if getattr(full, f) == 0})
        caps, ground, view_parts = (16, 8, 4), 4, 1
        global PROGRESS_EVERY, BIND_PARTS
        PROGRESS_EVERY, BIND_PARTS = 1, 1
        print("smoke_parts", " ".join(f"{b.deal_id}:{len(b)} rewrites={len(b.layout_rewrites)}"
                                      for b in train), flush=True)
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

        # Pilot (Mac): C3_DEVICE=mps runs the full student on Apple silicon with
        # the encoder frozen (--train-encoder-layers 0); the teacher stays off.
        dev = "cuda"
        if not torch.cuda.is_available():
            if os.environ.get("C3_DEVICE") == "mps" and torch.backends.mps.is_available():
                dev = "mps"
            else:
                raise SystemExit("This base run is for the H100. CUDA is not available. --tiny is the CPU check.")
        torch.set_float32_matmul_precision("high")
        print("model", "h100" if dev == "cuda" else dev, args.encoder, args.teacher)
        student = live_student(schema, args.encoder).to(dev)
        if args.init_from:
            from .gates import load_checkpoint

            known = {}
            if args.allow_text_change and args.init_schema_ref:
                from .gates import schema_digests_at

                full, shape = schema_digests_at(args.init_schema_ref)
                known[full] = shape
            step = load_checkpoint(args.init_from, student,
                                   schema_sha256=digest([asdict(o) for o in schema.opportunities]),
                                   allow_text_change=args.allow_text_change,
                                   shape_sha256=shape_digest(without_heads(schema, args.allow_new_heads)),
                                   known_shapes=known)
            print("init_from", args.init_from, "step", step, flush=True)
        released = release_encoder_top(student, args.train_encoder_layers)
        print("encoder_blocks_training", released, flush=True)
        _, result = run_base(
            train, holdout, schema, budget, cfg=student.cfg,
            make_lm=lambda: HFCausalLM(
                args.teacher, targets=("q_proj", "k_proj", "v_proj", "o_proj"),
                max_len=TEACHER_MAX_LEN, trainable_rank=32, dtype=torch.bfloat16),
            checkpoint=args.checkpoint or None, model=student, device=dev,
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
    if args.smoke:
        if torch.cuda.is_available():
            print(f"smoke peak_gpu_gib allocated={torch.cuda.max_memory_allocated() / 2**30:.1f} "
                  f"reserved={torch.cuda.max_memory_reserved() / 2**30:.1f}", flush=True)
        import resource  # noqa: PLC0415

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(f"smoke peak_rss_gib {rss / (2**30 if sys.platform == 'darwin' else 2**20):.2f}", flush=True)


if __name__ == "__main__":
    main()
