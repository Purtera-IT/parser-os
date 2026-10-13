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
from .schema import GROUP, PAIR, Schema, load_schema
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
    return head if sep else deal_id


def lock_split(train: list[Batch], holdout: list[Batch]) -> tuple[list[Batch], list[Batch]]:
    """Refuse a holdout deal that is also a training deal, including its windows,
    and any fixed holdout (``holdout.FIXED_HOLDOUT``) in training."""
    if not train or not holdout:
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
PROGRESS_EVERY = 25
# The live teacher's memory caps. A call is one pass through the 7B body; a
# piece is what one backward holds (checkpointed: about 0.2 MB a token).
TEACHER_CALL_TOKENS = 32768
TEACHER_PIECE_TOKENS = 131072


def _progress(stage: str, done: int, total: int, started: float) -> None:
    if done % PROGRESS_EVERY == 0 or done == total:
        rate = (time.time() - started) / max(done, 1)
        print(f"progress {stage} {done}/{total} parts {rate:.1f}s/part "
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


def _optimizer(params, lr: float, adamw: bool) -> torch.optim.Optimizer:
    live = [p for p in params if p.requires_grad]
    if not live:
        raise ValueError("No trainable parameters")
    if adamw:
        return torch.optim.AdamW(live, lr=lr, weight_decay=0.01)
    return torch.optim.Adam(live, lr=lr)


def _autocast(device: str):
    if device.startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
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
                piece_tokens: int = 0) -> None:
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
        order = train[:]
        rng.shuffle(order)
        for batch in order:
            graph = _deal_graph(model, teacher, batch, device)
            opt.zero_grad()
            context = teacher_context(teacher, batch)
            loss = None
            for part in teacher_chunks(teacher, batch, chunk, piece_tokens):
                with _autocast(device):
                    piece, _ = brain_loss(teacher, batch, weights, graph=graph, rows=part.rows,
                                          judged=part.cards, flips=part.flips, extras=part.extras,
                                          context=context)
                    piece = piece * ((scale or {}).get(batch.deal_id, 1.0) * part.scale)
                _finite(piece, "bind")
                if piece.requires_grad:
                    piece.backward()
                loss = piece.detach() if loss is None else loss + piece.detach()
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
                  scale: float = 1.0) -> torch.Tensor:
    model.train()
    with _autocast(device):
        loss, _ = c3_loss(model, batch, weights, judgment_sizes=sizes)
        if teacher is not None and view is not None:
            out = model(batch.inputs(), company=batch.company or None)
            loss = loss + weighted(
                supercharge_loss(model, out, view, texts=batch.texts, batch=batch,
                                 unlabeled_keys=unlabeled_keys),
                lesson or TeachWeights.for_base())
        loss = loss * scale
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.grad is not None], 1.0)
    opt.step()
    model.ema_update()
    return loss.detach()


def _score_oneshot(model: C3Model, batch: Batch, device: str = "cpu") -> dict[str, dict[str, float]]:
    """Exact match on labeled rows from atoms only. Stage and geography are null."""
    model.eval()
    with torch.no_grad(), _autocast(device):
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


def run_base(train: list[Batch], holdout: list[Batch], schema: Schema | None = None,
             budget: Budget | None = None, cfg: C3Config = SMALL, lr: float = 3e-3,
             seed: int = 0,
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
             call_tokens: int = 0,
             deal_graph: bool = False,
             validation: list[Batch] | None = None,
             balance: bool = True,
             encoder_lr: float | None = None,
             score_holdout: bool = True) -> tuple[C3Model, BaseRun]:
    """Train on ``train`` only. Score ``holdout`` once, with atoms only.

    The language model is the teacher. Tests pass the tiny stand-in on CPU.
    The H100 path passes a pretrained causal LM, bf16, and ``adamw``. This
    function does not certify a release and does not call ``gates.train``.
    """
    if READY_FOR_TRAINING:
        raise RuntimeError("This loop must not be the path that flips the release flag")
    train, holdout = lock_split(train, holdout)
    validation = list(validation or [])
    lock_validation(train, holdout, validation)
    scale = part_scales(train) if balance else {b.deal_id: 1.0 for b in train}
    schema = schema or load_schema()
    budget = budget or Budget()
    torch.manual_seed(seed)
    rng = random.Random(seed)
    teacher = Brain(schema, make_lm(), context_lines=context_lines, page_batch=page_batch,
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
    teacher_w = BrainWeights(consolidate=0.0, flip=0.0)
    stage_loss: dict[str, float] = {}
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
            # The schema descriptions are re-read per piece for the same reason.
            teacher_opt.zero_grad()
            context = teacher_context(teacher, batch)
            loss = None
            for part in teacher_chunks(teacher, batch, teacher_rows_per_step, teacher_piece_tokens):
                with _autocast(device):
                    piece, _ = brain_loss(teacher, batch, teacher_w, rows=part.rows, judged=part.cards,
                                          flips=part.flips, extras=part.extras, context=context)
                    piece = piece * (scale[batch.deal_id] * part.scale)
                _finite(piece, "teacher")
                if piece.requires_grad:
                    piece.backward()
                loss = piece.detach() if loss is None else loss + piece.detach()
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

    model = model or C3Model(schema, cfg)
    model.to(device)
    _freeze_link_gate(model, False)
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
    seen = heads_seen(train)
    validation_scores: dict[str, dict[str, dict[str, float]]] = {}
    views = None
    completed = 0
    schema_sha = digest([asdict(o) for o in schema.opportunities]) if checkpoint else ""
    for stage in student_stages:
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
                        chunk=teacher_rows_per_step, piece_tokens=teacher_piece_tokens)
            continue
        weights, sizes = gold_weights("deal" if stage in ("open_links",) or stage in lesson_stages else stage)
        use_teacher = stage in lesson_stages
        lesson, allowed = _lesson(stage, seen) if use_teacher else (None, None)
        steps = budget.steps(stage)
        stage_losses: list[float] = []
        started, done = time.time(), 0
        for _ in range(steps):
            if use_teacher and (views is None or teacher.graph is not None):
                views = {}
                for batch in train:
                    graph = _deal_graph(model, teacher, batch, device)
                    views[batch.deal_id] = teacher_view(
                        teacher, batch, batch.company or None, graph=graph)
            order = train[:]
            rng.shuffle(order)
            for batch in order:
                step_loss = _student_step(model, opt, batch, weights, sizes,
                                         teacher if use_teacher else None,
                                         views.get(batch.deal_id) if use_teacher and views else None,
                                         device, lesson, allowed, scale[batch.deal_id])
                _finite(step_loss, stage)
                stage_losses.append(float(step_loss.detach()))
                completed += 1
                done += 1
                _progress(stage, done, steps * len(train), started)
                if sched is not None:
                    sched.step()
        if stage_losses:
            stage_loss[stage] = sum(stage_losses) / len(stage_losses)
            print(f"stage {stage} loss {stage_loss[stage]:.4f}", flush=True)
        if validation:
            pooled = pool_scores([_score_oneshot(model, b, device) for b in validation])
            validation_scores[stage] = pooled
            macro = sum(v["exact"] for v in pooled.values()) / max(1, len(pooled))
            print(f"validation {stage} heads={len(pooled)} macro_exact={macro:.4f}", flush=True)
        if checkpoint and completed:
            save_checkpoint(checkpoint, model, step=completed, schema_sha256=schema_sha)
            # Every stage keeps its own copy, so a later stage that makes
            # validation worse costs a file pick, not a rerun.
            save_checkpoint(f"{checkpoint}.{stage}", model, step=completed, schema_sha256=schema_sha)
            if validation_scores:
                Path(f"{checkpoint}.validation.json").write_text(
                    json.dumps(validation_scores, indent=1, sort_keys=True), encoding="utf-8")
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
    parser.add_argument("--holdout", nargs="+", required=True)
    parser.add_argument("--validation", nargs="*", default=[],
                        help="Deals scored after every stage to choose steps; never trained")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--teacher", default=H100_TEACHER,
                        help="Pretrained causal LM for the training-only teacher")
    parser.add_argument("--encoder", default="answerdotai/ModernBERT-large")
    parser.add_argument("--tiny", action="store_true", help="CPU stand-in. Not the base.")
    parser.add_argument("--max-atoms", type=int, default=4096,
                        help="Lines per window. A bigger GPU takes more, so fewer deals are cut")
    parser.add_argument("--train-encoder-layers", type=int, default=0,
                        help="Release this many top blocks of the line encoder (0 keeps it frozen)")
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--limit-parts", type=int, default=0,
                        help="Preflight only: train on the N largest parts, to measure memory and "
                             "time per part without paying for a whole pass")
    parser.add_argument("--score-holdout", action="store_true",
                        help="Score the holdout here. Off for the live runs: the winner is scored "
                             "once with `python -m ml.c3.predict --score`")
    args = parser.parse_args()
    require_live_teacher("" if args.tiny else args.teacher, args.tiny)
    schema = load_schema()
    budget = Budget(teacher=args.steps, lines=args.steps, links=args.steps, cards=args.steps,
                    numbers=max(1, args.steps // 2), deal=max(1, args.steps // 2),
                    open_links=args.steps, teach=args.steps,
                    bind=0 if args.tiny else max(1, args.steps // 4),
                    flips=0 if args.tiny else max(1, args.steps // 4),
                    spread=0 if args.tiny else max(1, args.steps // 4))
    train, train_notes = load_split(args.train, schema, args.max_atoms)
    holdout, hold_notes = load_split(args.holdout, schema, args.max_atoms)
    validation, val_notes = load_split(args.validation, schema, args.max_atoms)
    if args.limit_parts:
        train = sorted(train, key=len, reverse=True)[:args.limit_parts]
        print("preflight_parts", " ".join(f"{b.deal_id}:{len(b)}" for b in train), flush=True)
    print("parts", f"train={len(train)} validation={len(validation)} holdout={len(holdout)}", flush=True)
    for note in train_notes + hold_notes + val_notes:
        print("note", note)
    held = {source_deal_id(b.deal_id) for b in holdout}
    if not args.tiny and not set(LABEL_HOLDOUT) <= held:
        raise SystemExit("The base scores the whole labeling holdout. Missing: "
                         + ", ".join(i for i in LABEL_HOLDOUT if i not in held))
    if args.tiny:
        print("model", "tiny")
        # Real deals have hundreds of lines: the stand-in reads them 8 pages at a time.
        _, result = run_base(train, holdout, schema, budget, checkpoint=args.checkpoint or None,
                             validation=validation, page_batch=8)
    else:
        from .lm import HFCausalLM

        if not torch.cuda.is_available():
            raise SystemExit("This base run is for the H100. CUDA is not available. --tiny is the CPU check.")
        torch.set_float32_matmul_precision("high")
        print("model", "h100", args.encoder, args.teacher)
        student = live_student(schema, args.encoder).to("cuda")
        released = release_encoder_top(student, args.train_encoder_layers)
        print("encoder_blocks_training", released, flush=True)
        _, result = run_base(
            train, holdout, schema, budget, cfg=student.cfg,
            make_lm=lambda: HFCausalLM(
                args.teacher, targets=("q_proj", "k_proj", "v_proj", "o_proj"),
                max_len=8192, trainable_rank=32, dtype=torch.bfloat16),
            checkpoint=args.checkpoint or None, model=student, device="cuda",
            lr=3e-4, teacher_lr=1e-4, context_lines=6,
            adamw=True, freeze_teacher=True, page_batch=8, read_chunk=16,
            call_tokens=TEACHER_CALL_TOKENS, teacher_rows_per_step=32,
            teacher_piece_tokens=TEACHER_PIECE_TOKENS, deal_graph=True,
            validation=validation, encoder_lr=args.encoder_lr if released else None,
            score_holdout=args.score_holdout)
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
