"""Two pre-launch checks on the smoke's checkpoint (pod/runE.sh SMOKE).

``bypass``: a generalization-critical head must decide from the reasoning
path r, not from the raw deal-graph state h. Today the only direct read of h
by a head is the free head (model.add_free_head: ``Linear([x, h])``). For each
critical head the check zeroes the h half of that input and measures

    h-path share = ||logits - logits(h half zeroed)|| / ||logits - row mean||

over the part's lines (a row's mean is removed: a shift every answer shares
decides nothing). It fails when a head's share is above ``--max-h-share``
(0.10). A head with no free head, or whose free head reads no h, scores 0.

It reads the run's configuration after the h ramp: the run's free rows
(``--free-head``, as runE.sh passes them: ``r:<key>`` rows for the critical
heads) and their share of h at ``1 - --free-h-out`` (runE.sh ramps
``free_h_out`` 0 -> 1, so 0 after the ramp). The smoke itself trains with the
ramp off; whatever ``h_keep`` its checkpoint holds, the check sets the
post-ramp one, the configuration every later read uses.

``portability``: one company-layer refit step (only the company parameters
train: the policy codes and atoms, the conduct heads, every ``*_company``
module) must leave every base output (each universal logit and number)
bit-identical, and must move at least one company output (or the step proved
nothing).

    python -m ml.c3.smoke_checks --tiny --checkpoint smokeE.pt --train <files> --out smoke_checks.md
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import torch

from .data import Batch

#: The heads that must generalize through the WHY (owner, 2026-10-23).
CRITICAL = ("col:label_type", "col:admission", "col:supplier", "col:about", "read:site", "jdg:site",
            "jdg:site_role", "read:visit_count", "read:sow_section", "read:sow_element",
            "read:task_owner_role")
#: Parameters a company-layer refit trains (model.freeze_for_new_company plus
#: the company heads): names starting alpha / atom_, or naming conduct or company.
COMPANY_PARAM = re.compile(r"^(alpha|atom_)|conduct|company")


def _forward(model, batch: Batch, device: str, seed: int = 0):
    from .loop import _autocast  # noqa: PLC0415

    torch.manual_seed(seed)
    random.seed(seed)
    with torch.no_grad(), _autocast(device):
        return model(batch.inputs(), company=batch.company or None, desc=model.describe())


def h_path_shares(model, batch: Batch, keys=CRITICAL, device: str = "cpu") -> dict[str, float | None]:
    """Key -> h-path share on ``batch`` (None: the head has no line logits here)."""
    model.eval()
    d = model.cfg.d
    zero = {"on": False}

    def pre(mod, inputs):
        if not zero["on"]:
            return None
        x = inputs[0]
        if x.shape[-1] <= model.cfg.d_head:       # a free head that reads no h
            return None
        x = x.clone()
        x[..., -d:] = 0
        return (x,) + tuple(inputs[1:])

    hooks = [m.register_forward_pre_hook(pre) for m in model.free_heads.values()]
    try:
        full = _forward(model, batch, device)
        zero["on"] = True
        cut = _forward(model, batch, device)
    finally:
        for h in hooks:
            h.remove()
    out: dict[str, float | None] = {}
    for k in keys:
        if k not in full.logits:
            out[k] = None
            continue
        a, b = full.logits[k].float(), cut.logits[k].float()
        centred = a - a.mean(-1, keepdim=True)
        den = float(centred.norm())
        out[k] = float((a - b).norm()) / den if den else 0.0
    return out


def portability(model, batch: Batch, device: str = "cpu", lr: float = 0.1) -> dict:
    """One company-layer refit step; the base outputs before and after."""
    from .losses import c3_loss  # noqa: PLC0415
    from .loop import _autocast  # noqa: PLC0415

    schema = model.schema
    universal = {o.key for o in schema.opportunities if o.layer == "universal"}
    company = {o.key for o in schema.opportunities if o.layer != "universal"}
    model.eval()
    before = _forward(model, batch, device)
    again = _forward(model, batch, device)
    repeatable = all(torch.equal(before.logits[k], again.logits[k]) for k in before.logits)
    names = [n for n, _ in model.named_parameters() if COMPANY_PARAM.search(n)]
    for n, p in model.named_parameters():
        p.requires_grad_(bool(COMPANY_PARAM.search(n)))
    params = [p for n, p in model.named_parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr)
    torch.manual_seed(0)
    random.seed(0)
    with _autocast(device):
        loss, parts = c3_loss(model, batch)
    opt.zero_grad()
    loss.backward()
    moved_params = sum(1 for p in params if p.grad is not None and bool(p.grad.abs().sum() > 0))
    opt.step()
    after = _forward(model, batch, device)
    changed_base = sorted(k for k in before.logits if k in universal
                          and not torch.equal(before.logits[k], after.logits[k]))
    changed_base += sorted(f"{k} (number)" for k in before.numbers if k in universal
                           and not torch.equal(before.numbers[k], after.numbers[k]))
    moved_company = sorted(k for k in before.logits if k in company
                           and not torch.equal(before.logits[k], after.logits[k]))
    return {"company_params": len(names), "params_with_grad": moved_params,
            "base_outputs": sum(1 for k in before.logits if k in universal),
            "base_changed": changed_base, "company_moved": moved_company,
            "repeatable": repeatable, "conduct_loss": parts.get("conduct")}


def pick_part(files: list[str], approve: list[str], schema, max_atoms: int, n_files: int = 4) -> Batch:
    """The part with the most labeled lines among the smallest training files."""
    import os  # noqa: PLC0415

    from .signal_audit import payload_parts, read_payload  # noqa: PLC0415

    best = None
    prefixes = [str(Path(x).expanduser().resolve()) for x in approve]
    for f in sorted(files, key=os.path.getsize)[:n_files]:
        payload = read_payload(f, any(str(Path(f).resolve()).startswith(x) for x in prefixes))
        if payload.get("skip") or not payload.get("atoms"):
            continue
        for b in payload_parts(payload, schema, max_atoms):
            if best is None or sum(b.labeled) > sum(best.labeled):
                best = b
    if best is None:
        raise SystemExit("no training part to check")
    return best


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint", default="", help="the smoke's checkpoint (default: a fresh model)")
    p.add_argument("--train", nargs="+", required=True)
    p.add_argument("--approve", nargs="*", default=[])
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--encoder", default="answerdotai/ModernBERT-large")
    p.add_argument("--allow-new-heads", default="")
    p.add_argument("--init-schema-ref", default="")
    p.add_argument("--train-encoder-layers", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-atoms", type=int, default=512)
    p.add_argument("--max-h-share", type=float, default=0.10)
    p.add_argument("--free-head", action="append", default=[],
                   help="the run's free rows after --free-head all (runE.sh why2_free: r:<key>, company rows)")
    p.add_argument("--free-h-out", type=float, default=1.0,
                   help="free_h_out after the ramp (runE.sh: 1, so r-only rows read no h)")
    p.add_argument("--heads", default=",".join(CRITICAL))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--signal-json", default="", help="signal_audit --json, summarized in the report")
    p.add_argument("--out", default="")
    p.add_argument("--json", default="")
    args = p.parse_args(argv)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    from .schema import load_schema  # noqa: PLC0415
    from .signal_audit import build_model  # noqa: PLC0415

    schema = load_schema()
    args.init_from = args.checkpoint
    model = build_model(schema, args)
    from .free_rows import r_only_keys, set_h_keep  # noqa: PLC0415

    set_h_keep(model, 1.0 - args.free_h_out)
    r_only = r_only_keys(model)
    batch = pick_part(args.train, args.approve, schema, args.max_atoms)
    shares = h_path_shares(model, batch, tuple(k for k in args.heads.split(",") if k), args.device)
    port = portability(model, batch, args.device)

    fails = [f"{k}: h-path share {v:.3f} > {args.max_h_share}" for k, v in shares.items()
             if v is not None and v > args.max_h_share]
    if port["base_changed"]:
        fails.append(f"company refit moved {len(port['base_changed'])} base outputs: "
                     + ", ".join(port["base_changed"][:8]))
    if not port["params_with_grad"] or not port["company_moved"]:
        fails.append("company refit moved no company output: the step proved nothing")
    if not port["repeatable"]:
        fails.append("two identical forwards differ: bit-identity cannot be checked on this device")

    lines = [f"# Smoke checks ({'tiny' if args.tiny else 'live'}; checkpoint {args.checkpoint or 'fresh'}; "
             f"part {batch.deal_id}, {len(batch)} lines)", "",
             f"## Bypass: h-path share of each critical head (fail above {args.max_h_share:.0%})", "",
             f"Post-ramp configuration: r-only free rows on {', '.join(r_only) or 'none'}, "
             f"their h share {1.0 - args.free_h_out:g} (free_h_out {args.free_h_out:g}).", "",
             "| head | h-path share | |", "|---|---:|---|"]
    for k, v in shares.items():
        lines.append(f"| {k} | {'-' if v is None else f'{100 * v:.2f}%'} | "
                     f"{'**FAIL**' if v is not None and v > args.max_h_share else ''} |")
    lines += ["", "## Portability: one company-layer refit step", "",
              f"- company parameters trained: {port['company_params']} ({port['params_with_grad']} got a gradient)",
              f"- base outputs bit-identical: {port['base_outputs'] - len(port['base_changed'])}"
              f"/{port['base_outputs']} logits"
              + (f"; moved: {', '.join(port['base_changed'][:12])}" if port["base_changed"] else ""),
              f"- company outputs moved: {len(port['company_moved'])}",
              f"- forward repeatable: {port['repeatable']}"]
    if args.signal_json and Path(args.signal_json).exists():
        sj = json.loads(Path(args.signal_json).read_text())
        total = sum(r["words"] for r in sj.get("coverage", []))
        cov = sum(r["covered"] for r in sj.get("coverage", []))
        lines += ["", "## Signal audit", "",
                  f"- signals failing the gate: {len(sj.get('failing', []))} "
                  f"({' '.join(sj.get('failing', [])[:30])})",
                  f"- reasoning words reaching a loss: {cov}/{total} ({100 * cov / max(total, 1):.1f}%)"]
        for c in sj.get("coverage_failing", []):
            lines.append(f"- FAIL {c}")
    lines += ["", "**GO**" if not fails else "**NO-GO**: " + "; ".join(fails)]
    text = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    if args.json:
        Path(args.json).write_text(json.dumps({"h_share": shares, "portability": port, "fails": fails},
                                              indent=1, default=str))
    if fails:
        print("smoke checks FAILED: " + "; ".join(fails), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
