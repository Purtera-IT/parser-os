"""A WHY-retrieval bank at serve, behind a flag, OFF by default (arch_review §4-5 rank 5).

The review's experiment: a nearest-neighbour vote of a line's r against the
training WHYs in ``why_proj`` space (``why_r``) was the most layout-agnostic
line-type reading measured, and mixed half and half with the model it fixed
010358's layout flips at flat holdout accuracy. It was also imprecise: the
site WHYs were 31% of that bank. So:

* **build** (``python -m ml.c3.why_bank build``): the training WHYs that name
  one answer of ``KEY`` (a WHY text written under two answers is left out),
  each ``why_proj``-encoded once and normalized; every item weighs
  ``prior[c] / items[c]``, so each answer's total vote is its share of the
  *corpus* labels (all labeled lines, not only those with a WHY), never its
  share of the WHYs. Over ``cap`` items, each answer keeps a prior-proportional
  sample. Saved next to the checkpoint as ``<ck>.why_bank.pt``: vectors,
  answer indices and weights only, never text.
* **read**: ``p_why(c) ∝ Σ_topk w_m exp(cos(r_i, v_m) / τ)`` over the items of
  answer c among the ``k`` nearest; the logits become
  ``log((1 - mix) p_model + mix p_why)``. A forward hook on the model, so every
  reader (predict, serve, gate_lib) gets it once it is attached.
* **flag**: ``gates.load_checkpoint`` calls ``maybe_attach``. ``C3_WHY_BANK``
  unset or ``0``: nothing (the default). ``1``: attach only when
  ``<ck>.why_bank.decision.json`` says ``enable`` (the gate experiment passed).
  ``force``: attach regardless (the gate experiment's own "on" read).
* **decide** (``python -m ml.c3.why_bank decide off/items.json on/items.json``):
  the gate experiment. Two swap_gate reads of one checkpoint, bank off and
  forced on. The bank is enabled only when the swap gate's line-type flip rate
  under the held-out perturbations drops AND line-type accuracy on the
  holdouts (the golden and holdout-B deals, clean) does not drop
  (``min_gain`` 0). Writes the decision file.

The as-designed bank reader (explain.ExplanationReader + compiled operators)
stays off: untrained, it cut 010358 site recall from 89 to 21 of 101, and the
compiler's word read crashes past 256 tokens (operators.py:155).
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import torch
from torch.nn import functional as F

from .data import IGNORE

KEY = "col:label_type"
ENV = "C3_WHY_BANK"
K, TAU, MIX = 16, 0.05, 0.5


def bank_path(ck: str | Path) -> Path:
    return Path(f"{ck}.why_bank.pt")


def decision_path(ck: str | Path) -> Path:
    return Path(f"{ck}.why_bank.decision.json")


@torch.no_grad()
def build_bank(model, batches, *, key: str = KEY, cap: int = 20000, chunk: int = 256,
               seed: int = 0) -> dict:
    """The bank (module docstring) from featurized training parts."""
    prior: Counter = Counter()
    seen: dict[str, set[int]] = {}
    for b in batches:
        y = b.targets.get(key)
        if y is None:
            continue
        for i, c in enumerate(y):
            if c == IGNORE:
                continue
            prior[c] += 1
            t = " ".join(str(b.why[i] or "").split())
            if t:
                seen.setdefault(t, set()).add(c)
    items = [(t, next(iter(cs))) for t, cs in seen.items() if len(cs) == 1]
    total = sum(prior.values())
    by: dict[int, list[str]] = {}
    for t, c in items:
        by.setdefault(c, []).append(t)
    g = torch.Generator().manual_seed(seed)
    if len(items) > cap:
        keep = []
        for c, ts in by.items():
            n = max(1, round(cap * prior[c] / total))
            order = torch.randperm(len(ts), generator=g)[:n].tolist()
            keep += [(ts[k], c) for k in order]
        items = keep
    count = Counter(c for _, c in items)
    model.eval()
    vecs = []
    texts = [t for t, _ in items]
    for a in range(0, len(texts), chunk):
        vecs.append(F.normalize(model.why_proj(model.aux_text(texts[a:a + chunk])).float(), dim=-1).cpu())
    labels = torch.tensor([c for _, c in items], dtype=torch.long)
    weights = torch.tensor([prior[c] / total / count[c] for _, c in items], dtype=torch.float32)
    n_ans = len(model.schema.by_key()[key].answers)
    return {"key": key, "vectors": (torch.cat(vecs) if vecs else torch.zeros(0, model.cfg.d_r)).half(),
            "labels": labels, "weights": weights, "answers": n_ans, "k": K, "tau": TAU, "mix": MIX,
            "prior": torch.tensor([prior.get(c, 0) / max(total, 1) for c in range(n_ans)])}


def why_probs(bank: dict, r: torch.Tensor) -> torch.Tensor:
    """[N, A] p_why per line from its r."""
    v = bank["vectors"].to(r.device, torch.float32)
    n_ans = int(bank["answers"])
    if v.shape[0] == 0:
        return torch.full((r.shape[0], n_ans), 1.0 / n_ans, device=r.device)
    sim = F.normalize(r.float(), dim=-1) @ v.T                      # [N, M]
    k = min(int(bank["k"]), v.shape[0])
    top, idx = sim.topk(k, dim=-1)
    w = bank["weights"].to(r.device)[idx] * torch.exp((top - top[:, :1]) / float(bank["tau"]))
    p = torch.zeros(r.shape[0], n_ans, device=r.device).scatter_add_(1, bank["labels"].to(r.device)[idx], w)
    return p / p.sum(-1, keepdim=True).clamp(min=1e-12)


def mix_logits(bank: dict, logits: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    m = float(bank["mix"])
    p = (1 - m) * logits.float().softmax(-1) + m * why_probs(bank, r)
    return p.clamp(min=1e-12).log().to(logits.dtype)


def attach(model, bank: dict):
    """Mix the bank into every forward's ``bank['key']`` logits. Returns the hook handle."""
    def hook(_mod, _inp, out):
        lg = out.logits.get(bank["key"])
        if lg is not None:
            out.logits[bank["key"]] = mix_logits(bank, lg, out.r)
        return out

    model.why_bank = bank
    return model.register_forward_hook(hook)


def load_bank(path: str | Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def maybe_attach(model, ck: str | Path):
    """The serve flag (module docstring). Returns the hook handle or None."""
    flag = os.environ.get(ENV, "").strip().lower()
    if flag in ("", "0", "off", "false") or getattr(model, "why_bank", None) is not None:
        return None
    path = bank_path(ck)
    if not path.exists():
        if flag == "force":
            raise FileNotFoundError(f"{ENV}=force: no bank {path}")
        return None
    if flag != "force":
        dec = decision_path(ck)
        if not dec.exists() or not json.loads(dec.read_text(encoding="utf-8")).get("enable"):
            print(f"why_bank: {ENV}={flag} but the gate experiment has not enabled the bank; off", flush=True)
            return None
    print(f"why_bank: on ({path.name}, {ENV}={flag})", flush=True)
    return attach(model, load_bank(path))


# ---------------------------------------------------------------- the gate experiment

PERTURBED = ("keywords", "entities", "layout")
HOLDOUT_ROLES = ("golden", "holdout2")


def _rates(items: dict, key: str = KEY) -> dict:
    """Clean line-type accuracy on the holdouts, and the flip rate (perturbed
    answer differs from the clean one) per perturbation."""
    hit = n = 0
    flips: Counter = Counter()
    seen: Counter = Counter()
    for _deal, rec in items["deals"].items():
        if rec.get("role") not in HOLDOUT_ROLES:
            continue
        clean = {k: v for k, v in rec["kinds"].get("clean", {}).items() if k.split("|")[1:2] == [key]}
        for k, (p, g, _doc) in clean.items():
            if g is not None:
                hit += int(p == g)
                n += 1
        for kind in PERTURBED:
            other = rec["kinds"].get(kind) or {}
            for k, (p0, _g, _doc) in clean.items():
                if k in other:
                    seen[kind] += 1
                    flips[kind] += int(other[k][0] != p0)
    flip_all = sum(flips.values()) / max(1, sum(seen.values()))
    return {"accuracy": hit / n if n else float("nan"), "n": n, "flip_rate": flip_all,
            "flips": {k: flips[k] / max(1, seen[k]) for k in PERTURBED if seen[k]}}


def gate_decision(off: dict, on: dict, *, min_gain: float = 0.0) -> dict:
    a, b = _rates(off), _rates(on)
    enable = bool(b["n"]) and b["flip_rate"] < a["flip_rate"] and b["accuracy"] >= a["accuracy"] + min_gain
    return {"enable": enable, "off": a, "on": b, "min_gain": min_gain,
            "rule": "flip rate lower and holdout line-type accuracy not lower (by min_gain)"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--checkpoint", required=True)
    b.add_argument("--train", nargs="+", required=True)
    b.add_argument("--cap", type=int, default=20000)
    b.add_argument("--max-atoms", type=int, default=4096)
    b.add_argument("--encoder", default="answerdotai/ModernBERT-large")
    d = sub.add_parser("decide")
    d.add_argument("off")
    d.add_argument("on")
    d.add_argument("--checkpoint", required=True)
    d.add_argument("--min-gain", type=float, default=0.0)
    a = ap.parse_args()
    if a.cmd == "decide":
        res = gate_decision(json.load(open(a.off)), json.load(open(a.on)), min_gain=a.min_gain)
        decision_path(a.checkpoint).write_text(json.dumps(res, indent=1), encoding="utf-8")
        print("why_bank decide " + json.dumps(res), flush=True)
        return
    from dataclasses import asdict  # noqa: PLC0415

    from .gates import load_checkpoint  # noqa: PLC0415
    from .loop import SMALL, live_student, load_split  # noqa: PLC0415
    from .model import C3Model  # noqa: PLC0415
    from .readiness import digest  # noqa: PLC0415
    from .schema import load_schema  # noqa: PLC0415

    schema = load_schema()
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        tiny = "text.bag.weight" in torch.load(a.checkpoint, map_location="cpu", weights_only=True)["state"]
    model = C3Model(schema, SMALL) if tiny else live_student(schema, a.encoder)
    os.environ.pop(ENV, None)
    load_checkpoint(a.checkpoint, model, schema_sha256=digest([asdict(o) for o in schema.opportunities]))
    parts, _ = load_split(a.train, schema, a.max_atoms)
    bank = build_bank(model, parts, cap=a.cap)
    torch.save(bank, bank_path(a.checkpoint))
    per = Counter(bank["labels"].tolist())
    print(f"why_bank built {len(bank['labels'])} items over {len(per)} answers -> {bank_path(a.checkpoint)}",
          flush=True)


if __name__ == "__main__":
    main()
