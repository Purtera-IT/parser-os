"""Calibration, checkpoints, per-head scores and the refusal to train.

Nothing here starts a customer training run. ``train`` always refuses while
``READY_FOR_TRAINING`` is false. Checkpoints store weights, step and hashes,
never source text. Holdout rows are rejected by the calibration and threshold
fitters.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from .coverage import READY_FOR_TRAINING, release_blockers
from .data import IGNORE, Batch
from .model import C3Model


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, *, split: str) -> float:
    """One temperature on validation logits. The holdout is not a fitting set."""
    if split != "validation":
        raise ValueError("Temperature is fit on the validation split only")
    if logits.ndim != 2 or labels.ndim != 1 or logits.shape[0] != labels.shape[0]:
        raise ValueError("Temperature fit expects [N, C] logits and [N] labels")
    log_t = torch.nn.Parameter(torch.zeros((), device=logits.device))
    opt = torch.optim.LBFGS([log_t], lr=0.25, max_iter=15)

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp().clamp(min=1e-2), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp().clamp(min=1e-2))


def confidence_threshold(probabilities: torch.Tensor, correct: torch.Tensor, *,
                         split: str, minimum_coverage: float = 0.5) -> float:
    """Highest threshold whose validation coverage stays at least ``minimum_coverage``.

    Predictions below the threshold abstain. This is not a production bar.
    """
    if split != "validation":
        raise ValueError("Thresholds are fit on the validation split only")
    if probabilities.numel() == 0:
        raise ValueError("No validation scores")
    best = 0.5
    order = torch.sort(probabilities).values
    for value in order.tolist():
        keep = probabilities >= value
        if float(keep.float().mean()) < minimum_coverage:
            break
        best = float(value)
    if not bool(correct.any()):
        return best
    return best


def abstain(probabilities: torch.Tensor, threshold: float) -> torch.Tensor:
    """True where the score is below the threshold and must not be served."""
    return probabilities < threshold


def per_head_scores(model: C3Model, batch: Batch) -> dict[str, dict[str, float]]:
    """Exact-match rates for labeled rows only. An untrained model is not accurate."""
    model.eval()
    with torch.no_grad():
        out = model(batch.inputs(), company=batch.company or None)
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
        if key not in out.logits and hasattr(model, "judge"):
            try:
                logits = model.judge(out, key, [item.lines for item in items],
                                     queries=[item.query for item in items])
            except ValueError:
                continue
            pred = logits.argmax(-1)
            hit = sum(int(int(pred[i]) == item.answer) for i, item in enumerate(items))
            scores[key] = {"exact": hit / len(items), "n": float(len(items))}
    return scores


def save_checkpoint(path: str | Path, model: C3Model, *, step: int, schema_sha256: str,
                    torch_version: str | None = None) -> None:
    payload = {
        "step": int(step),
        "schema_sha256": schema_sha256,
        # str(): torch.__version__ is a TorchVersion, which weights_only loads refuse.
        "torch": str(torch_version or torch.__version__),
        "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }
    if any(isinstance(v, str) and len(v) > 80 for v in payload.values()):
        raise ValueError("Checkpoint metadata must stay free of source text")
    torch.save(payload, path)


def load_checkpoint(path: str | Path, model: C3Model, *, schema_sha256: str) -> int:
    # Checkpoints saved before the str() above carry a TorchVersion.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_sha256") != schema_sha256:
        raise ValueError("Checkpoint schema hash does not match this model")
    state = payload["state"]
    # The EMA hindsight copy (teacher, teacher_proj) is built lazily during
    # training; a model that has not built it (predict) reads without it.
    for name in ("teacher", "teacher_proj"):
        if getattr(model, name, None) is None:
            state = {k: v for k, v in state.items() if not k.startswith(name + ".")}
    model.load_state_dict(state)
    return int(payload["step"])


def certify_page_budget(texts: list[str], max_tokens: int, count) -> None:
    if any(count(text) > max_tokens for text in texts):
        raise ValueError("Page exceeds the pretrained context budget; do not truncate it")


def export_shadow(path: str | Path, model: C3Model, *, schema_sha256: str,
                  thresholds: dict[str, float], contract_ready: bool,
                  pretrained_budget_certified: bool, rules_evaluated: bool,
                  holdout_touched: bool) -> dict[str, Any]:
    """Write nothing and return the blockers. Shadow export is refused."""
    blockers = list(release_blockers())
    if not contract_ready:
        blockers.append("contract_not_ready")
    if not pretrained_budget_certified:
        blockers.append("pretrained_budget_not_certified")
    if not rules_evaluated:
        blockers.append("rules_evaluation_missing")
    if holdout_touched:
        blockers.append("holdout_was_touched")
    if blockers or not READY_FOR_TRAINING:
        raise RuntimeError("Shadow export refused: " + ", ".join(blockers))
    save_checkpoint(path, model, step=0, schema_sha256=schema_sha256)
    return {"thresholds": thresholds}


def train(*_args: Any, **_kwargs: Any) -> None:
    """Customer training entrypoint. It does not run."""
    raise RuntimeError(
        "READY_FOR_TRAINING is false. Refusing to start a training run until the "
        "frozen audited corpus, untouched holdout, separate Rules evaluation and "
        "pretrained budget certification are verified. Passing unit tests is not "
        "learned accuracy. Blockers: " + ", ".join(release_blockers()))
