"""Calibration, checkpoints, per-head scores and the refusal to train.

Nothing here starts a customer training run. ``train`` always refuses while
``READY_FOR_TRAINING`` is false. Checkpoints store weights, step and hashes,
never source text. Holdout rows are rejected by the calibration and threshold
fitters.
"""
from __future__ import annotations

import hashlib
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


def body_prefixes(model: C3Model) -> tuple[str, ...]:
    """State-dict prefixes of every pretrained body the model holds (the
    shared line encoder appears under several names, and under ``teacher.``
    once the EMA copy is built)."""
    found = {f"{name}.{m.pretrained_body}." if name else f"{m.pretrained_body}."
             for name, m in model.named_modules(remove_duplicate=False)
             if getattr(m, "pretrained_body", None)}
    found |= {"teacher." + p[len("encoder."):] for p in found if p.startswith("encoder.")}
    return tuple(sorted(found))


def _split_body(key: str, prefixes: tuple[str, ...]) -> str | None:
    for p in prefixes:
        if key.startswith(p):
            return key[len(p):]
    return None


def body_digest(tensors: dict[str, torch.Tensor]) -> str:
    """sha256 over the body's tensors by name, bytes exactly as stored."""
    h = hashlib.sha256()
    for name in sorted(tensors):
        t = tensors[name].detach().cpu().contiguous().reshape(-1)
        h.update(name.encode() + b"\0" + str(t.dtype).encode() + b"\0")
        h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def slim_state(state: dict[str, torch.Tensor], model: C3Model) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Leave out the pretrained body where it is exactly the hub's weights.

    ``model`` holds the body as freshly downloaded. A saved tensor that differs
    from it (a trained layer) stays in the checkpoint. Returns the slim state
    and what the reader needs to check its own download against."""
    prefixes = body_prefixes(model)
    fresh = {k: v for k, v in model.state_dict().items() if _split_body(k, prefixes) is not None}
    fresh = {_split_body(k, prefixes): v for k, v in fresh.items()}
    slim, dropped = {}, {}
    for k, v in state.items():
        rest = _split_body(k, prefixes)
        if rest is not None and rest in fresh and fresh[rest].shape == v.shape \
                and fresh[rest].dtype == v.dtype and torch.equal(fresh[rest].cpu(), v.cpu()):
            dropped[rest] = v
        else:
            slim[k] = v
    if not dropped:
        return dict(state), {}
    return slim, {"sha256": body_digest(dropped), "tensors": len(dropped)}


def save_checkpoint(path: str | Path, model: C3Model, *, step: int, schema_sha256: str,
                    torch_version: str | None = None, slim: bool = False,
                    shape_sha256: str | None = None, max_atoms: int | None = None) -> None:
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    body: dict[str, Any] = {}
    if slim:
        state, body = slim_state(state, model)
    payload = {
        "step": int(step),
        "schema_sha256": schema_sha256,
        # str(): torch.__version__ is a TorchVersion, which weights_only loads refuse.
        "torch": str(torch_version or torch.__version__),
        "state": state,
    }
    if shape_sha256:
        # The schema's (key, answer values) digest: lets a later run whose
        # registries changed only text start from this one (allow_text_change).
        payload["shape_sha256"] = shape_sha256
    if body:
        payload["pretrained_body"] = body
    graph = getattr(model, "graph", None)
    from .graph import GRAPH_DEFAULTS  # noqa: PLC0415

    if graph and dict(graph) != GRAPH_DEFAULTS:
        # The cross-line flags (graph.py): load_checkpoint builds the same
        # graph, so serve reads with the mask and hops the run trained with.
        payload["graph"] = dict(graph)
    if getattr(getattr(model, "cfg", None), "view_tag", False):
        # The view:<quoting|planning> input tag (c3_record.view): serve hashes it only for a base trained on it.
        payload["view_tag"] = True
    if max_atoms:
        # The training window, for the record (the pick scores at serve's window).
        payload["max_atoms"] = int(max_atoms)
    if any(isinstance(v, str) and len(v) > 80 for v in payload.values()):
        raise ValueError("Checkpoint metadata must stay free of source text")
    torch.save(payload, path)


def load_checkpoint(path: str | Path, model: C3Model, *, schema_sha256: str,
                    allow_text_change: bool = False, shape_sha256: str | None = None,
                    known_shapes: dict[str, str] | None = None) -> int:
    """Load weights saved under the same schema digest.

    ``allow_text_change`` (``loop.py --init-from --allow-text-change`` only;
    never serving) accepts a different schema digest when the opportunity keys,
    their order and every answer-value list are identical, so only description
    texts differ: ``shape_sha256`` is this schema's ``schema.shape_digest`` and
    must equal the checkpoint's. A checkpoint saved before shape digests were
    stored is matched through ``known_shapes`` (its schema digest -> shape
    digest, from ``schema_digests_at`` on the commit it was trained on)."""
    # Checkpoints saved before the str() above carry a TorchVersion.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_sha256") != schema_sha256:
        if not allow_text_change:
            raise ValueError("Checkpoint schema hash does not match this model")
        if not shape_sha256:
            raise ValueError("allow_text_change needs this schema's shape digest")
        stored = payload.get("shape_sha256") or (known_shapes or {}).get(str(payload.get("schema_sha256")))
        if not stored:
            raise ValueError("Checkpoint schema hash does not match this model, and the checkpoint carries "
                             "no shape digest: name the commit it was trained on (--init-schema-ref)")
        if stored != shape_sha256:
            raise ValueError("Checkpoint schema differs in more than text: the opportunity keys, their order "
                             "or an answer-value list changed")
    state = payload["state"]
    if hasattr(model, "cfg"):
        model.cfg.view_tag = bool(payload.get("view_tag"))
    if payload.get("graph") and hasattr(model, "enable_graph"):
        # Built before the weights load, as the free heads are below.
        model.enable_graph(payload["graph"])
    # The EMA hindsight copy (teacher, teacher_proj) is built lazily during
    # training; a model that has not built it (predict) reads without it.
    for name in ("teacher", "teacher_proj"):
        if getattr(model, name, None) is None:
            state = {k: v for k, v in state.items() if not k.startswith(name + ".")}
    # A free head in the checkpoint is built before it loads; one the model
    # has and the checkpoint lacks starts at zero (reads as the described head).
    # An r-only row (free_rows.py) is built as one first.
    from .free_rows import build_from_state  # noqa: PLC0415

    build_from_state(model, state)
    for k in state:
        if k.startswith("free_heads.") and k.endswith(".weight"):
            model.add_free_head(k[len("free_heads."):-len(".weight")])
    missing, unexpected = model.load_state_dict(state, strict=False)
    # company_why_proj (model.py) trains only; an older checkpoint leaves it at init.
    missing = [k for k in missing if not k.startswith(("free_heads.", "company_why_proj."))]
    # A slim checkpoint left the pretrained body out: the model's own download
    # stands in, and it must be byte for byte the body the run trained on.
    body = payload.get("pretrained_body")
    if body:
        prefixes = body_prefixes(model)
        own = model.state_dict()
        filled = {_split_body(k, prefixes): own[k] for k in missing if _split_body(k, prefixes) is not None}
        if len(filled) != int(body["tensors"]) or body_digest(filled) != body["sha256"]:
            raise RuntimeError("This model's pretrained encoder is not the one the slim checkpoint was "
                               "saved against (different download or revision)")
        missing = [k for k in missing if _split_body(k, prefixes) is None]
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint does not match the model: missing {missing[:5]} "
                           f"unexpected {unexpected[:5]}")
    # The WHY-retrieval bank: off unless C3_WHY_BANK asks and its gate passed (why_bank.py).
    from .why_bank import maybe_attach  # noqa: PLC0415

    maybe_attach(model, path)
    return int(payload["step"])


def checkpoint_meta(path: str | Path) -> dict[str, Any]:
    """A checkpoint's metadata without its weights' meaning: step, digests,
    graph flags, training window."""
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        payload = torch.load(path, map_location="cpu", weights_only=True)
    return {k: v for k, v in payload.items() if k != "state"}


_DIGESTS_AT = """
import hashlib, json, sys
from dataclasses import asdict
sys.path.insert(0, sys.argv[1])
from ml.c3.schema import load_schema
s = load_schema()
d = lambda v: hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":"),
                                        ensure_ascii=False).encode()).hexdigest()
print(d([asdict(o) for o in s.opportunities]), d([(o.key, [a.value for a in o.answers]) for o in s.opportunities]))
"""


def schema_digests_at(ref: str, repo: str | Path | None = None) -> tuple[str, str]:
    """(schema digest, shape digest) of the schema at git commit ``ref``: its own
    registries and its own schema code, read from a scratch copy of that commit
    in a separate interpreter. A pre-shape checkpoint is matched only when the
    first equals the digest it was saved under.

    ``ref`` may instead be a saved shape file (``new_heads.py --write-shape``,
    ml/c3/runE/init_shape.json): its two digests, for a copy with no git history."""
    if str(ref).endswith(".json") and Path(ref).is_file():
        import json  # noqa: PLC0415

        blob = json.loads(Path(ref).read_text(encoding="utf-8"))
        return str(blob["schema_sha256"]), str(blob["shape_sha256"])
    import subprocess
    import sys
    import tarfile
    import tempfile

    repo = Path(repo) if repo else Path(__file__).resolve().parents[2]
    paths = ["ml/__init__.py", "ml/c3", "app/core/label_heads.json", "app/core/atom_types.json",
             "app/core/judgment_reasons.json"]
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "ref.tar"
        with open(archive, "wb") as fh:
            subprocess.run(["git", "-C", str(repo), "archive", ref, *paths], stdout=fh, check=True)
        root = Path(tmp) / "tree"
        with tarfile.open(archive) as tar:
            tar.extractall(root, filter="data")
        out = subprocess.run([sys.executable, "-I", "-c", _DIGESTS_AT, str(root)], cwd=tmp,
                             capture_output=True, text=True, check=True).stdout.split()
    return out[0], out[1]


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
