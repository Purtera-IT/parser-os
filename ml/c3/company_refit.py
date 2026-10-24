"""Pretend a new company: refit only the company layer on a frozen base (arch_review §8 c4).

The product claim is a cheap company layer on a fixed base: a new company's
decisions (co_action, co_reason, the hours rules) fit on its own labels while
every universal reading stays exactly as the base gives it. This check
measures that claim on Purtera's own data:

1. read the probe deal (010043 at the pick) through the base: every universal
   logit, plus every base parameter;
2. hold out ``hold`` training deals' company labels; refit ONLY the company
   layer (``COMPANY_PARTS``: the conduct block, its head, the company free rows,
   the company slot input, the company card head, the policy codes, the company
   WHY projection) on the other deals' company labels, base frozen;
3. read the probe again: the universal outputs must be bit-identical and every
   base parameter unchanged (``base_identical``);
4. score co_action on the held-out deals before and after (``co_action_before``
   / ``co_action_after``) and count the labels the refit used.

A device whose reads are not repeatable (two reads of the same base differ)
is reported (``repeatable`` False) and judged on the parameters alone.
Counts and rates only, never line text.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from .data import IGNORE, Batch

#: Module / parameter name prefixes that make up the company layer.
COMPANY_PARTS = ("conduct_base.", "conduct_head.", "atom_u", "atom_v", "atom_gate", "atom_theta", "alpha",
                 "alpha_from_text.", "company_slot_in.", "reader_company.", "compiler_company.",
                 "judgment_head_company.", "pool_score_company.", "pool_in_company.", "company_why_proj.")
CO_KEY = "read:co_action"


def _device(model) -> str:
    return str(next(model.parameters()).device)


def _reading(model):
    """Every forward here under the loop's autocast for the model's device:
    on CUDA / MPS the bf16 encoder feeds fp32 layers only inside autocast."""
    from .loop import _autocast  # noqa: PLC0415

    return _autocast(_device(model))


def dtype_safe(model) -> None:
    """On the CPU (no autocast) the frozen bf16 line encoder's vectors are cast
    to float32 before the fp32 deal graph reads them; a mixed bf16 x fp32
    matmul raised there. The encoder is frozen, so nothing else changes."""
    text = model.text
    if getattr(text, "_c3_float_out", False):
        return
    fwd = text.forward

    def forward(texts, _fwd=fwd):
        return _fwd(texts).float()
    text.forward = forward
    if hasattr(text, "fixed"):
        fixed = text.fixed

        def fixed_(texts, _f=fixed):
            return _f(texts).float()
        text.fixed = fixed_
    text._c3_float_out = True


def company_param_names(model) -> list[str]:
    by = model.schema.by_key()
    names = []
    for n, _ in model.named_parameters():
        if n.startswith(COMPANY_PARTS):
            names.append(n)
        elif n.startswith("free_heads."):
            key = n[len("free_heads."):].rsplit(".", 1)[0]
            if key in by and not by[key].universal:
                names.append(n)
    return names


@torch.no_grad()
def universal_read(model, batch: Batch) -> dict[str, torch.Tensor]:
    model.eval()
    with _reading(model):
        out = model(batch.oneshot(), company=batch.company or None)
    keep = {o.key for o in model.schema.opportunities if o.universal}
    got = {k: v.detach().cpu().clone() for k, v in out.logits.items() if k in keep}
    got.update({f"num:{k}": v.detach().cpu().clone() for k, v in out.numbers.items() if k in keep})
    got["r"] = out.r.detach().cpu().clone()
    return got


@torch.no_grad()
def co_exact(model, batches: list[Batch], key: str = CO_KEY) -> tuple[float, int]:
    model.eval()
    hit = n = 0
    for b in batches:
        y = b.targets.get(key)
        if y is None or not any(v != IGNORE for v in y):
            continue
        with _reading(model):
            out = model(b.oneshot(), company=b.company or None)
        pred = out.logits[key].argmax(-1).tolist()
        for p, t in zip(pred, y):
            if t != IGNORE:
                hit += int(p == t)
                n += 1
    return (hit / n if n else float("nan")), n


def _same(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def refit_company(model, parts: list[Batch], *, passes: int = 2, lr: float = 1e-3) -> int:
    """Train the company layer only (company head CE) on ``parts``; returns the labels used."""
    from .losses import LossWeights, head_losses  # noqa: PLC0415

    names = set(company_param_names(model))
    params = []
    for n, p in model.named_parameters():
        p.requires_grad_(n in names)
        if n in names:
            params.append(p)
    opt = torch.optim.Adam(params, lr=lr)
    w = LossWeights()
    used = 0
    keys = [o.key for o in model.schema.select(layer="company")]
    for b in parts:
        used += sum(sum(v != IGNORE for v in b.targets.get(k, [])) for k in keys)
    for _ in range(passes):
        for b in parts:
            if not b.company:
                continue
            model.train()
            with _reading(model):
                out = model(b.inputs(), company=b.company)
                co, _ = head_losses(model, out, b, w, "company")
            if not co.requires_grad:
                continue
            opt.zero_grad()
            co.backward()
            opt.step()
    for p in model.parameters():
        p.requires_grad_(False)
    return used


def pretend_new_company(model, train: list[Batch], probe: Batch, *, hold: int = 3, passes: int = 2,
                        lr: float = 1e-3) -> dict:
    """The check (module docstring). Moves the company layer's weights: run it
    on a copy of the model, or reload after."""
    held = sorted(train, key=lambda b: b.deal_id)[:hold]
    rest = [b for b in train if b not in held]
    base_names = [n for n, _ in model.named_parameters() if n not in set(company_param_names(model))]
    before_params = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if n in set(base_names)}
    first = universal_read(model, probe)
    again = universal_read(model, probe)
    repeatable = _same(first, again)
    co_before, n_held = co_exact(model, held)
    used = refit_company(model, rest, passes=passes, lr=lr)
    after = universal_read(model, probe)
    after_params = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if n in set(base_names)}
    params_same = _same(before_params, after_params)
    outputs_same = _same(first, after)
    co_after, _ = co_exact(model, held)
    return {"base_identical": params_same and (outputs_same or not repeatable),
            "params_identical": params_same, "outputs_identical": outputs_same, "repeatable": repeatable,
            "held_deals": len(held), "held_co_action_n": n_held, "refit_labels": used,
            "co_action_before": co_before, "co_action_after": co_after,
            "company_params": len(company_param_names(model)), "base_params": len(base_names)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--train", nargs="+", required=True, help="training files; the first --hold by deal id are held")
    ap.add_argument("--probe", nargs="+", required=True, help="the validation files (010043): base must not move")
    ap.add_argument("--hold", type=int, default=3)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--max-atoms", type=int, default=4096)
    ap.add_argument("--encoder", default="answerdotai/ModernBERT-large")
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="auto", help="auto: cuda when there is one, else cpu")
    a = ap.parse_args()
    from .gates import load_checkpoint  # noqa: PLC0415
    from .loop import SMALL, live_student, load_split  # noqa: PLC0415
    from .model import C3Model  # noqa: PLC0415
    from .readiness import digest  # noqa: PLC0415
    from .schema import load_schema  # noqa: PLC0415

    schema = load_schema()
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        payload = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    tiny = "text.bag.weight" in payload["state"]
    del payload
    model = C3Model(schema, SMALL) if tiny else live_student(schema, a.encoder)
    load_checkpoint(a.checkpoint, model, schema_sha256=digest([asdict(o) for o in schema.opportunities]))
    dev = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)
    if dev == "cpu":
        dtype_safe(model)
    for p in model.parameters():
        p.requires_grad_(False)
    train, _ = load_split(a.train, schema, a.max_atoms)
    probe, _ = load_split(a.probe, schema, a.max_atoms)
    probe = max(probe, key=lambda b: sum(b.labeled))
    res = pretend_new_company(model, train, probe, hold=a.hold, passes=a.passes)
    print("company_refit " + " ".join(f"{k}={v}" for k, v in res.items()), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    if not res["base_identical"]:
        raise SystemExit("company_refit: the base moved when only the company layer was refit")


if __name__ == "__main__":
    main()
