"""Train only the free answer heads on a saved run, on a Mac, no pod.

    python -m ml.c3.head_fit --checkpoint runA.pt.open_links --train train/*.json \
        --validation 010087.json 010149.json --out runA.heads.pt

The trunk and the encoder stay as the checkpoint has them. Every part is read
once (atoms only, as at run time) and each head's input and each line's
deal-graph state are kept for the labeled lines; the free heads
(``C3Model.add_free_head``) then train on those for many epochs in minutes.
Line type is checked on the validation deals every few epochs (Deal Kit and
estimating-calc lines left out), and the best epoch is saved as a normal
checkpoint that ``ml.c3.predict`` reads. An epoch counts only when the watched
heads (task tier) hold ``--min-watch`` and the other heads' mean exact has not
dropped below where the checkpoint started, so a refit that buys line type by
breaking another head is never saved. Validation picks the epoch; the holdout
is never read. Counts and rates only, never line text.
"""
from __future__ import annotations

import argparse
import random
import time
from dataclasses import asdict

import torch
import torch.nn.functional as F

from .data import IGNORE
from .loop import CHECK_SKIP_DOC_KINDS, TYPE_KEY, WATCH_CHECK, _autocast, live_student, load_split
from .readiness import digest
from .schema import NUMBER, RELATION, load_schema


def _cache(model, parts, keys, device):
    """Per key: head input x, line state h, described logits, targets, row
    weights, parser picks and a scored mask, for the labeled rows only."""
    grabbed: dict[str, torch.Tensor] = {}
    hooks = [head.register_forward_hook(lambda m, a, o, s=space: grabbed.__setitem__(s, a[0].detach()))
             for space, head in model.heads.items()]
    space_of = {k: model.schema.by_key()[k].space for k in keys}
    rows: dict[str, dict[str, list]] = {k: {n: [] for n in ("x", "h", "base", "y", "w", "parser", "scored")}
                                         for k in keys}
    started = time.time()
    try:
        for i, b in enumerate(parts, 1):
            grabbed.clear()
            with torch.no_grad(), _autocast(device):
                out = model(b.oneshot(), company=b.company or None)
            h = out.h.float().cpu()
            w = torch.tensor(b.weights if len(b.weights) == len(b) else [1.0] * len(b))
            scored = torch.tensor([str(k or "").lower() not in CHECK_SKIP_DOC_KINDS for k in b.doc_kind])
            for k in keys:
                if k not in out.logits:
                    continue
                if k in b.multi_targets:
                    sel = [j for j, v in enumerate(b.multi_targets[k]) if v is not None]
                    if not sel:
                        continue
                    a = len(model.schema.by_key()[k].answers)
                    y = torch.zeros(len(sel), a)
                    for r, j in enumerate(sel):
                        y[r, list(b.multi_targets[k][j])] = 1
                else:
                    sel = [j for j, v in enumerate(b.targets.get(k, [])) if v != IGNORE]
                    if not sel:
                        continue
                    y = torch.tensor([b.targets[k][j] for j in sel])
                ix = torch.tensor(sel)
                parser = b.parser_targets.get(k) or [IGNORE] * len(b)
                x = grabbed.get(space_of[k]) if space_of[k] in grabbed else out.conduct_x  # company heads read conduct
                rows[k]["x"].append(x.float().cpu()[ix].half())
                rows[k]["h"].append(h[ix].half())
                rows[k]["base"].append(out.logits[k].float().cpu()[ix])
                rows[k]["y"].append(y)
                rows[k]["w"].append(w[ix])
                rows[k]["parser"].append(torch.tensor([parser[j] for j in sel]))
                rows[k]["scored"].append(scored[ix])
            print(f"read {i}/{len(parts)} parts {time.time() - started:.0f}s", flush=True)
    finally:
        for hk in hooks:
            hk.remove()
    return {k: {n: torch.cat(v) for n, v in r.items()} for k, r in rows.items() if r["y"]}


def _logits(model, key, c, idx=None):
    sel = (lambda t: t) if idx is None else (lambda t: t[idx])
    z = torch.cat([sel(c["x"]).float(), sel(c["h"]).float()], -1)
    return sel(c["base"]) + model.free_heads[key](z)


def type_scores(model, c, focus: int, keep: int, sure: float = 0.5) -> dict[str, float]:
    with torch.no_grad():
        prob = _logits(model, TYPE_KEY, c).softmax(-1)
    m = c["scored"]
    y, prob, parser = c["y"][m], prob[m], c["parser"][m]
    p, pred = prob.max(-1)
    named = prob.clone()
    if keep >= 0:
        named[:, keep] = -1
    named = named.argmax(-1)
    real, said = y == focus, pred == focus
    has = parser != IGNORE
    routed = torch.where(p >= sure, pred, parser)
    n = max(1, len(y))
    return {"n": float(len(y)), "exact": float((pred == y).sum()) / n,
            "typed_exact": float((named == y).sum()) / n,
            "task_recall": float((said & real).sum()) / max(1, int(real.sum())),
            "task_precision": float((said & real).sum()) / max(1, int(said.sum())),
            "routed_exact": float((routed[has] == y[has]).sum()) / max(1, int(has.sum())),
            "routed_share": float((p[has] >= sure).sum()) / max(1, int(has.sum())),
            "parser_exact": float((parser[has] == y[has]).sum()) / max(1, int(has.sum()))}


def head_scores(model, va) -> dict[str, float]:
    """Validation exact per single-answer head (multi-answer heads left out)."""
    out = {}
    with torch.no_grad():
        for k, c in va.items():
            if k == TYPE_KEY or c["y"].dim() != 1 or k not in model.free_heads:
                continue
            out[k] = float((_logits(model, k, c).argmax(-1) == c["y"]).float().mean())
    return out


def _guard(heads: dict[str, float], start_macro: float | None, min_watch: float) -> tuple[bool, float, str]:
    """Whether an epoch may be saved: every watched head at ``min_watch`` and
    the heads' mean exact not below the checkpoint's own."""
    macro = sum(heads.values()) / max(1, len(heads))
    low = [f"{k}={heads[k]:.3f}" for k in WATCH_CHECK if k in heads and heads[k] < min_watch]
    if start_macro is not None and macro < start_macro - 1e-9:
        low.append(f"macro={macro:.4f}<{start_macro:.4f}")
    return not low, macro, " ".join(low)


def fit_heads(model, tr, va, keys, *, epochs: int = 60, check_every: int = 5, lr: float = 1e-3,
              weight_decay: float = 1e-2, type_weight: float = 5.0, batch: int = 4096,
              min_watch: float = 0.9):
    """Train the free heads on the cached rows. Returns the best validation
    line-type exact among the epochs that pass the head guard, and the free
    heads' state at that epoch (None when no guarded epoch beat where it
    started, or the start failed the guard and no epoch passed it)."""
    schema = model.schema
    opt = torch.optim.AdamW([q for k in keys for q in model.free_heads[k].parameters()],
                            lr=lr, weight_decay=weight_decay)
    names = [a.value for a in schema.by_key()[TYPE_KEY].answers]
    focus, keep = names.index("task"), names.index("_keep") if "_keep" in names else -1
    multi = {k for k in keys if tr[k]["y"].dim() == 2}
    before = type_scores(model, va[TYPE_KEY], focus, keep)
    fit0 = type_scores(model, tr[TYPE_KEY], focus, keep)
    heads0 = head_scores(model, va)
    ok0, macro0, why0 = _guard(heads0, None, min_watch)
    watch0 = " ".join(f"valid_{k.replace(':', '_')}={heads0[k]:.3f}" for k in WATCH_CHECK if k in heads0)
    print("check epoch 0 " + " ".join(f"valid_{k}={v:.3f}" for k, v in before.items() if k != "n")
          + f" fit_exact={fit0['exact']:.3f} {watch0} valid_macro={macro0:.4f}"
          + ("" if ok0 else f" guard_failed {why0}"), flush=True)
    # A start that fails the guard is no bar to beat: any guarded epoch wins.
    best, best_state = (before["exact"] if ok0 else -1.0), None
    for epoch in range(1, epochs + 1):
        orders = {k: torch.randperm(len(tr[k]["y"])) for k in keys}
        steps = max((len(o) + batch - 1) // batch for o in orders.values())
        total = 0.0
        for s in range(steps):
            loss = torch.zeros(())
            for k in keys:
                idx = orders[k][s * batch:(s + 1) * batch]
                if not len(idx):
                    continue
                lg, y, w = _logits(model, k, tr[k], idx), tr[k]["y"][idx], tr[k]["w"][idx]
                if k in multi:
                    per = F.binary_cross_entropy_with_logits(lg, y, reduction="none").mean(-1)
                else:
                    per = F.cross_entropy(lg, y, reduction="none")
                loss = loss + (type_weight if k == TYPE_KEY else 1.0) * (per * w).sum() / w.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
        if epoch % check_every == 0 or epoch == epochs:
            v = type_scores(model, va[TYPE_KEY], focus, keep)
            fit = type_scores(model, tr[TYPE_KEY], focus, keep)
            print(f"check epoch {epoch} loss {total / steps:.3f} "
                  + " ".join(f"valid_{k}={x:.3f}" for k, x in v.items() if k != "n")
                  + f" fit_exact={fit['exact']:.3f} fit_task_recall={fit['task_recall']:.3f}", flush=True)
            heads = head_scores(model, va)
            ok, macro, why = _guard(heads, macro0, min_watch)
            print(f"check epoch {epoch} " + " ".join(
                f"valid_{k.replace(':', '_')}={heads[k]:.3f}" for k in WATCH_CHECK if k in heads)
                + f" valid_macro={macro:.4f}" + ("" if ok else f" guard_failed {why}"), flush=True)
            if ok and v["exact"] > best:
                best = v["exact"]
                best_state = {k: {n: t.detach().clone() for n, t in model.free_heads[k].state_dict().items()}
                              for k in keys}
                print(f"best epoch {epoch} valid_exact={best:.3f}", flush=True)
    return best, best_state


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--train", nargs="+", required=True)
    p.add_argument("--validation", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--check-every", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--type-weight", type=float, default=5.0, help="line type's weight among the heads")
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--min-watch", type=float, default=0.9,
                   help="lowest validation exact a watched head (task tier) may have in a saved epoch")
    p.add_argument("--max-atoms", type=int, default=8192)
    p.add_argument("--encoder", default="answerdotai/ModernBERT-large")
    p.add_argument("--device", default="auto")
    p.add_argument("--only-keys", nargs="*", default=[],
                   help="refit only these heads (e.g. read:sow_section); every other head keeps the checkpoint's")
    p.add_argument("--extra-keys", nargs="*", default=[],
                   help="company-layer heads to refit beside the universal ones (e.g. read:co_action)")
    args = p.parse_args()
    from .gates import load_checkpoint, save_checkpoint

    dev = args.device
    if dev == "auto":
        dev = ("cuda" if torch.cuda.is_available()
               else "mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(0)
    rng = random.Random(0)
    schema = load_schema()
    sha = digest([asdict(o) for o in schema.opportunities])
    model = live_student(schema, args.encoder).to(dev)
    step = load_checkpoint(args.checkpoint, model, schema_sha256=sha)
    model.eval()
    keys = [o.key for o in schema.select(layer="universal")
            if o.kind not in (RELATION, NUMBER) and len(o.answers) > 1]
    keys += [k for k in args.extra_keys if k in schema.by_key() and k not in keys]
    if args.only_keys:
        keys = [k for k in args.only_keys if k in schema.by_key()]
    train, _ = load_split(args.train, schema, args.max_atoms)
    val, _ = load_split(args.validation, schema, args.max_atoms)
    print(f"parts train={len(train)} validation={len(val)}", flush=True)
    tr = _cache(model, train, keys, dev)
    va = _cache(model, val, keys, dev)
    keys = [k for k in keys if k in tr]
    # The heads train on the CPU copy: they are small, and the cache is there.
    model.to("cpu")
    for k in keys:
        model.add_free_head(k)
    print("heads " + " ".join(f"{k}:{len(tr[k]['y'])}" for k in keys), flush=True)
    best, best_state = fit_heads(model, tr, va, keys, epochs=args.epochs, check_every=args.check_every,
                                 lr=args.lr, weight_decay=args.weight_decay,
                                 type_weight=args.type_weight, batch=args.batch, min_watch=args.min_watch)
    if best_state is None:
        print("no epoch passed the head guard and beat the checkpoint on validation; nothing saved",
              flush=True)
        return
    for k, st in best_state.items():
        model.free_heads[k].load_state_dict(st)
    save_checkpoint(args.out, model, step=step, schema_sha256=sha)
    print(f"saved {args.out} (valid_exact {best:.3f})", flush=True)


if __name__ == "__main__":
    main()
