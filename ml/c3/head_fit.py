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
import gc
import hashlib
import json
from importlib.metadata import version
import random
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .data import IGNORE
from . import free_rows
from .loop import CHECK_SKIP_DOC_KINDS, TYPE_KEY, WATCH_CHECK, _autocast, live_student, load_split
from .loop import lock_split, lock_validation
from .readiness import digest
from .schema import NUMBER, RELATION, load_schema


def _atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def _cache(model, parts, keys, device, cache_dir=None):
    """Per key: head input x, line state h, described logits, targets, row
    weights, parser picks and a scored mask, for the labeled rows only."""
    grabbed: dict[str, torch.Tensor] = {}
    rationale: dict[str, torch.Tensor] = {}
    hooks = [head.register_forward_hook(lambda m, a, o, s=space: grabbed.__setitem__(s, a[0].detach()))
             for space, head in model.heads.items()]
    hooks += [head.register_forward_hook(lambda m, a, o, s=space: rationale.__setitem__(s, o.detach()))
              for space, head in model.space_in.items()]
    space_of = {k: model.schema.by_key()[k].space for k in keys}
    rows: dict[str, dict[str, list]] = {k: {n: [] for n in ("x", "xr", "h", "base", "y", "w", "parser", "scored")}
                                         for k in keys}
    started = time.time()
    try:
        for i, b in enumerate(parts, 1):
            path = Path(cache_dir) / f"part-{i:04d}.pt" if cache_dir else None
            if path is not None and path.exists():
                chunk = torch.load(path, map_location="cpu", weights_only=True)
                for k, r in chunk.items():
                    for n, t in r.items():
                        rows[k][n].append(t)
                print(f"reuse {i}/{len(parts)} parts {time.time() - started:.0f}s", flush=True)
                continue
            grabbed.clear()
            rationale.clear()
            print(f"reading {i}/{len(parts)} lines={len(b)}", flush=True)
            with torch.no_grad(), _autocast(device):
                out = model(b.oneshot(), company=b.company or None)
            h = out.h.float().cpu()
            w = torch.tensor(b.weights if len(b.weights) == len(b) else [1.0] * len(b))
            scored = torch.tensor([str(k or "").lower() not in CHECK_SKIP_DOC_KINDS for k in b.doc_kind])
            chunk = {}
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
                # Total inference logits already include the existing free row.
                base = out.logits[k].float().cpu()[ix]
                if k in out.free_logits:
                    base = base - out.free_logits[k].float().cpu()[ix]
                c = dict(x=grabbed[space_of[k]].float().cpu()[ix],
                         xr=rationale[space_of[k]].float().cpu()[ix], h=h[ix], base=base,
                         y=y, w=w[ix], parser=torch.tensor([parser[j] for j in sel]), scored=scored[ix])
                if k in model.free_heads:
                    # Fail before saving incompatible features or training a row.
                    with torch.no_grad():
                        dev = next(model.free_heads[k].parameters()).device
                        got = _logits(model, k, {n: t.to(dev) for n, t in c.items()}).cpu()
                    torch.testing.assert_close(got, out.logits[k].float().cpu()[ix], rtol=1e-5, atol=1e-5)
                chunk[k] = c
                for n, t in c.items():
                    rows[k][n].append(t)
            if path is not None:
                _atomic_save(chunk, path)
            if hasattr(model, "save_text_cache"):
                model.save_text_cache()
            # Do not keep the previous part's dense graph alive during the next forward.
            del out, h
            grabbed.clear()
            rationale.clear()
            print(f"read {i}/{len(parts)} parts {time.time() - started:.0f}s", flush=True)
    finally:
        for hk in hooks:
            hk.remove()
    return {k: {n: torch.cat(v) for n, v in r.items()} for k, r in rows.items() if r["y"]}


def _file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _cache_run(args):
    """Invalidate feature and optimizer caches on any input/code/config change."""
    if not args.cache_dir:
        return None
    files = [Path(args.checkpoint), Path(args.init_shape)] if args.init_shape else [Path(args.checkpoint)]
    files += list(Path(__file__).parent.glob("*.py"))
    encoder = Path(args.encoder)
    if not encoder.is_dir():
        raise ValueError("--cache-dir requires a local encoder for a reproducible cache identity")
    files += [p for p in encoder.rglob("*") if p.is_file()]
    files += [Path(p) for p in args.train + args.validation]
    config = vars(args).copy()
    for key in ("cache_dir", "out", "text_cache"):
        config.pop(key)
    manifest = {"version": 1, "config": config,
                "runtime": {"torch": str(torch.__version__), "transformers": version("transformers")},
                "files": {str(p.resolve()): _file_sha(p) for p in sorted(set(files))}}
    root = Path(args.cache_dir) / digest(manifest)
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"cache run {root}", flush=True)
    return root


def _cached_split(model, paths, schema, max_atoms, keys, device, cache_dir):
    """Load one deal's dense masks at a time; keep only compact labeled features."""
    chunks = []
    for i, path in enumerate(paths):
        parts, _ = load_split([path], schema, max_atoms)
        part_dir = Path(cache_dir) / f"deal-{i:04d}" if cache_dir else None
        print(f"deal {i + 1}/{len(paths)} parts={len(parts)}", flush=True)
        chunks.append(_cache(model, parts, keys, device, part_dir))
        del parts
        gc.collect()
    present = {k for chunk in chunks for k in chunk}
    return {k: {n: torch.cat([chunk[k][n] for chunk in chunks if k in chunk])
                for n in next(chunk[k] for chunk in chunks if k in chunk)} for k in present}


def _split_ids(paths, schema, max_atoms):
    ids = []
    for path in paths:
        parts, _ = load_split([path], schema, max_atoms)
        ids.extend(SimpleNamespace(deal_id=b.deal_id) for b in parts)
        del parts
    return ids


def _logits(model, key, c, idx=None):
    sel = (lambda t: t) if idx is None else (lambda t: t[idx])
    return sel(c["base"]) + free_rows.universal_free(
        model.free_heads[key], sel(c["xr"]).float(), sel(c["x"]).float(), sel(c["h"]).float())


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
              min_watch: float = 0.9, progress_path=None):
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
    first_epoch = 1
    if progress_path is not None and Path(progress_path).exists():
        saved = torch.load(progress_path, map_location="cpu", weights_only=True)
        for k, st in saved["current"].items():
            model.free_heads[k].load_state_dict(st)
        opt.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["rng"])
        best, best_state = saved["best"], saved["best_state"]
        first_epoch = saved["epoch"] + 1
        print(f"resume heads epoch {first_epoch}", flush=True)
    for epoch in range(first_epoch, epochs + 1):
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
        if progress_path is not None:
            _atomic_save(dict(epoch=epoch, current={k: model.free_heads[k].state_dict() for k in keys},
                              optimizer=opt.state_dict(), rng=torch.get_rng_state(), best=best,
                              best_state=best_state), progress_path)
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
    p.add_argument("--init-shape", default="",
                   help="the checkpoint predates this schema's texts (desc_gate.py, pilot A M2): load it across "
                        "the text change and the heads added since (ml/c3/runE/init_shape.json), and refit on "
                        "this code's texts")
    p.add_argument("--keys", default="", help="comma-separated heads to refit (default every universal class head)")
    p.add_argument("--text-cache", default="", help="line vectors by sha1 of the text (desc_gate._text_cache)")
    p.add_argument("--cache-dir", default="", help="fingerprinted per-part features and resumable head optimizer")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--graph-set", default="",
                   help="graph flags over the checkpoint's (graph.parse_graph keys): fit under the run's graph, "
                        "e.g. rune,lattice_doc=1")
    args = p.parse_args()
    cache_run = _cache_run(args)
    if cache_run is not None and not args.text_cache:
        args.text_cache = str(cache_run / "encoder-text.pt")
    if args.threads:
        torch.set_num_threads(args.threads)
    from .gates import load_checkpoint, save_checkpoint

    dev = args.device
    if dev == "auto":
        dev = ("cuda" if torch.cuda.is_available()
               else "mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(0)
    rng = random.Random(0)
    schema = load_schema()
    sha = digest([asdict(o) for o in schema.opportunities])
    if args.init_shape:
        from .desc_gate import load_start  # noqa: PLC0415

        model, _ = load_start(args.checkpoint, args.encoder, args.init_shape, dev, args.text_cache)
        step = 0
    else:
        model = live_student(schema, args.encoder).to(dev)
        step = load_checkpoint(args.checkpoint, model, schema_sha256=sha)
    if args.graph_set:
        from .graph import GRAPH_DEFAULTS, parse_graph  # noqa: PLC0415

        over = parse_graph(args.graph_set)
        model.enable_graph({**model.graph, **{k: v for k, v in over.items() if v != GRAPH_DEFAULTS[k]}})
        model.to(dev)
        print("graph " + " ".join(f"{k}={v}" for k, v in model.graph.items()), flush=True)
    model.eval()
    keys = [o.key for o in schema.select(layer="universal")
            if o.kind not in (RELATION, NUMBER) and len(o.answers) > 1]
    if args.keys:
        want = [k for k in args.keys.split(",") if k]
        keys = [k for k in want if k in keys]
        missing = sorted(set(want) - set(keys))
        if missing:
            raise SystemExit("--keys: not a universal class head: " + ", ".join(missing))
    train = _split_ids(args.train, schema, args.max_atoms)
    val = _split_ids(args.validation, schema, args.max_atoms)
    lock_split(train, [], need_holdout=False)
    lock_validation(train, [], val)
    print(f"parts train={len(train)} validation={len(val)}", flush=True)
    del train, val
    gc.collect()
    tr = _cached_split(model, args.train, schema, args.max_atoms, keys, dev,
                       cache_run / "train" if cache_run else None)
    # Guard watched heads even when --keys only selects other rows for training.
    validation_keys = list(dict.fromkeys(keys + list(WATCH_CHECK)))
    va = _cached_split(model, args.validation, schema, args.max_atoms, validation_keys, dev,
                       cache_run / "validation" if cache_run else None)
    if hasattr(model, "save_text_cache"):
        model.save_text_cache()
    keys = [k for k in keys if k in tr]
    # The heads train on the CPU copy: they are small, and the cache is there.
    model.to("cpu")
    for k in keys:
        model.add_free_head(k)
    print("heads " + " ".join(f"{k}:{len(tr[k]['y'])}" for k in keys), flush=True)
    best, best_state = fit_heads(model, tr, va, keys, epochs=args.epochs, check_every=args.check_every,
                                 lr=args.lr, weight_decay=args.weight_decay,
                                 type_weight=args.type_weight, batch=args.batch, min_watch=args.min_watch,
                                 progress_path=cache_run / "heads-progress.pt" if cache_run else None)
    if best_state is None:
        print("no epoch passed the head guard and beat the checkpoint on validation; nothing saved",
              flush=True)
        return
    for k, st in best_state.items():
        model.free_heads[k].load_state_dict(st)
    from .schema import shape_digest

    shape = shape_digest(schema)
    if args.init_shape:
        # A refit start for a run (runE.sh INIT=<out> INIT_MD5=skip): the shape is INIT's, so
        # new_heads.py still derives the heads added since and loop --init-from accepts it.
        import json  # noqa: PLC0415

        from .loop import without_heads  # noqa: PLC0415
        from .new_heads import derive  # noqa: PLC0415

        old = json.loads(open(args.init_shape, encoding="utf-8").read())["shape"]
        shape = shape_digest(without_heads(schema, ",".join(derive(schema, [(k, list(v)) for k, v in old]))))
    save_checkpoint(args.out, model, step=step, schema_sha256=sha, shape_sha256=shape)
    print(f"saved {args.out} (valid_exact {best:.3f})", flush=True)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------- locked refit (opt-in)
# A targeted refit that can only move what its labels ask for (refit owner, 2026-10-29):
#   (a) frozen base: trunk, lattice and every other head stay as loaded; each target head's free row trains as
#       base + delta (low-rank when rank > 0), so no other head's input or output can move;
#   (b) consistency: every line of every training deal that is not a new label is held to the base's own answer
#       distribution (KL, base predictions cached once);
#   (c) replay: a fixed seeded sample of approved labels, at its own weight, every step;
#   (d) accept: the heads whose saved weights differ from the base (only the targets may), and the lines whose
#       answer flips on unseen deals, with before/after p.
# head_fit's own fit_heads is untouched; overnight_1028/refit/targeted_locked.py (targeted_O.sh LOCKED=1) drives this.

def line_rows(model, parts, keys, device):
    """Per key: EVERY line of the parts (labeled or not): head input x / xr / h (float16), the base logits without
    the existing free row (float32), target y (IGNORE when unlabeled), row weight w, and the line text. Company keys
    read the conduct layer's line (x = conduct_x, xr / h empty), as company_refit does."""
    by = model.schema.by_key()
    grabbed: dict[str, torch.Tensor] = {}
    rationale: dict[str, torch.Tensor] = {}
    hooks = [head.register_forward_hook(lambda m, a, o, s=space: grabbed.__setitem__(s, a[0].detach()))
             for space, head in model.heads.items()]
    hooks += [head.register_forward_hook(lambda m, a, o, s=space: rationale.__setitem__(s, o.detach()))
              for space, head in model.space_in.items()]
    rows = {k: {n: [] for n in ("x", "xr", "h", "base", "y", "w")} for k in keys}
    texts: dict[str, list[str]] = {k: [] for k in keys}
    try:
        for b in parts:
            grabbed.clear()
            rationale.clear()
            with torch.no_grad(), _autocast(device):
                out = model(b.oneshot(), company=b.company or None)
            h = out.h.float().cpu()
            w = torch.tensor(b.weights if len(b.weights) == len(b) else [1.0] * len(b))
            for k in keys:
                if k not in out.logits:
                    continue
                if k in b.multi_targets:
                    raise ValueError(f"{k}: the locked refit takes single-answer heads only")
                y = torch.tensor(b.targets.get(k) or [IGNORE] * len(b))
                base = out.logits[k].float().cpu()
                if k in out.free_logits:
                    base = base - out.free_logits[k].float().cpu()
                if by[k].universal:
                    x, xr, hh = grabbed[by[k].space].cpu(), rationale[by[k].space].cpu(), h
                else:
                    x = out.conduct_x.float().cpu()
                    xr, hh = x[:, :0], h[:, :0]
                for n, t in (("x", x), ("xr", xr), ("h", hh)):
                    rows[k][n].append(t.half())
                rows[k]["base"].append(base)
                rows[k]["y"].append(y)
                rows[k]["w"].append(w)
                texts[k] += list(b.texts)
            if hasattr(model, "save_text_cache"):
                model.save_text_cache()
            del out, h
    finally:
        for hk in hooks:
            hk.remove()
    got = {k: {n: torch.cat(v) for n, v in r.items()} for k, r in rows.items() if r["y"]}
    return got, {k: texts[k] for k in got}


def cached_line_rows(model, paths, schema, max_atoms, keys, device, cache_of=None):
    """line_rows per file, kept under ``cache_of(path)`` (a directory, or None for no cache). Returns
    (rows per key, texts per key, file index per row per key)."""
    out: dict[str, dict[str, list]] = {}
    txt: dict[str, list[str]] = {}
    src: dict[str, list[int]] = {}
    for i, path in enumerate(paths):
        d = Path(cache_of(path)) if cache_of else None
        if d is not None and (d / "lines.pt").exists():
            got, tx = torch.load(d / "lines.pt", map_location="cpu", weights_only=True)
        else:
            parts, _ = load_split([path], schema, max_atoms)
            got, tx = line_rows(model, parts, keys, device)
            del parts
            gc.collect()
            if d is not None:
                _atomic_save((got, tx), d / "lines.pt")
        for k, r in got.items():
            for n, t in r.items():
                out.setdefault(k, {}).setdefault(n, []).append(t)
            txt.setdefault(k, []).extend(tx[k])
            src.setdefault(k, []).extend([i] * len(r["y"]))
        print(f"lines {i + 1}/{len(paths)} {Path(path).name} " + " ".join(f"{k}:{len(r['y'])}" for k, r in got.items()),
              flush=True)
    return ({k: {n: torch.cat(v) for n, v in r.items()} for k, r in out.items()}, txt,
            {k: torch.tensor(v) for k, v in src.items()})


class Lock:
    """Frozen-base deltas on the target free rows. Every parameter of a target row is read as base + delta: a
    weight matrix's delta is B @ A (B zero, so the start is exactly the base) when ``rank`` > 0, else a full
    zero matrix; a bias's delta is a zero vector. Weight decay acts on the deltas, so it pulls toward the base,
    never toward zero. Nothing else in the model has requires_grad."""

    def __init__(self, model, keys, rank: int = 0, seed: int = 0):
        for q in model.parameters():
            q.requires_grad_(False)
        g = torch.Generator().manual_seed(seed)
        self.keys, self.rank = list(keys), rank
        self.base = {k: {n: p.detach().clone() for n, p in model.free_heads[k].named_parameters()} for k in keys}
        self.delta: dict[str, dict[str, list[torch.Tensor]]] = {}
        for k in keys:
            self.delta[k] = {}
            for n, p in self.base[k].items():
                if p.dim() == 2 and rank > 0:
                    a = (torch.randn(rank, p.shape[1], generator=g) / p.shape[1] ** 0.5).requires_grad_()
                    self.delta[k][n] = [torch.zeros(p.shape[0], rank, requires_grad=True), a]
                else:
                    self.delta[k][n] = [torch.zeros_like(p, requires_grad=True)]

    def parameters(self):
        return [t for k in self.keys for ts in self.delta[k].values() for t in ts]

    def params(self, k):
        return {n: p + (d[0] @ d[1] if len(d) == 2 else d[0]) for n, p, d in
                ((n, self.base[k][n], self.delta[k][n]) for n in self.base[k])}

    def logits(self, model, k, c, idx=None, params=None):
        """base logits + this key's free row read at base + delta, or at ``params`` (head_fit._logits / company
        row maths)."""
        from torch.func import functional_call  # noqa: PLC0415

        sel = (lambda t: t) if idx is None else (lambda t: t[idx])
        row = model.free_heads[k]
        x = sel(c["x"]).float()
        if c["xr"].shape[-1] == 0:          # company row: reads the conduct line only
            z = x
        else:
            xr, h = sel(c["xr"]).float(), sel(c["h"]).float()
            if isinstance(row, free_rows.RFreeRow):
                keep = row.h_keep.to(x.dtype)
                x = xr + keep * (x - xr)
            z = torch.cat([x, h], -1)
        params = self.params(k) if params is None else params
        return sel(c["base"]).float() + functional_call(row, params, (z,), strict=False)

    def state(self):
        with torch.no_grad():
            return {k: {n: t.detach().clone() for n, t in self.params(k).items()} for k in self.keys}

    def commit(self, model, state=None):
        """Write base + delta (or a saved ``state``) into the model's free rows, for save_checkpoint."""
        state = state or self.state()
        with torch.no_grad():
            for k, st in state.items():
                for n, p in model.free_heads[k].named_parameters():
                    p.copy_(st[n])


def _margin(lp, a):
    """How far answer ``a`` leads the best other answer (log-prob / logit units)."""
    top = lp.gather(-1, a[:, None]).squeeze(-1)
    return top - lp.scatter(-1, a[:, None], float("-inf")).max(-1).values


def _sample(c, n, gen):
    return torch.randperm(len(c["y"]), generator=gen)[:n]


def locked_fit(model, lock: Lock, new, cons, replay, va, *, new_weight=30.0, cons_weight=1.0, replay_weight=1.0,
               hold_weight=0.0, hold_margin=0.5, type_weight=5.0, epochs=40, steps=10, batch=2048, lr=1e-3,
               weight_decay=1e-2, check_every=2, max_change=0.03, max_flip=0.01, tolerance=0.0, on_check=None):
    """Train ``lock``'s deltas. Per key and step: new rows (cross-entropy x new_weight x row weight, all of them),
    a batch of consistency rows (KL(base || candidate), x cons_weight) and a batch of replay rows (cross-entropy
    on the saved label, x replay_weight), normalised by their total weight. ``hold_weight`` adds a flip hinge on the
    same consistency batch: the base's answer must stay ahead of every other answer by min(its base margin,
    ``hold_margin``) (KL alone barely charges for flipping a near-tie line). A ``cw`` column on ``cons`` weighs
    each line's KL and hinge (the driver's --guard-types: lines whose base answer is a guarded type). ``cons`` / ``replay`` rows carry the
    base log-probs in ``teach`` (computed here, at delta 0). Every ``check_every`` epochs: validation exact and the
    share of validation rows whose answer changed, the share of consistency rows that flipped, and the new-row fit.
    An epoch is eligible when no head's validation exact falls below the base's (minus ``tolerance``), no head
    changes more than ``max_change`` of its validation rows and no more than ``max_flip`` of its consistency rows.
    Returns (best eligible (epoch, fit) or None, its state, log); ``on_check(epoch, info)`` is called at checks."""
    keys = lock.keys
    with torch.no_grad():
        for k in keys:
            for c in (cons.get(k), replay.get(k)):
                if c is not None:
                    c["teach"] = lock.logits(model, k, c).log_softmax(-1)
            if k in cons:
                cons[k]["a0"] = cons[k]["teach"].argmax(-1)
                cons[k]["hold"] = _margin(cons[k]["teach"], cons[k]["a0"]).clamp(max=hold_margin)
            if k in va:
                va[k]["start"] = lock.logits(model, k, va[k]).argmax(-1)
    gen = torch.Generator().manual_seed(0)
    opt = torch.optim.AdamW(lock.parameters(), lr=lr, weight_decay=weight_decay)

    def check():
        ex, ch, fl, nf = {}, {}, {}, {}
        with torch.no_grad():
            for k in keys:
                if k in va:
                    pred = lock.logits(model, k, va[k]).argmax(-1)
                    ex[k] = float((pred == va[k]["y"]).float().mean())
                    ch[k] = float((pred != va[k]["start"]).float().mean())
                if k in cons:
                    fl[k] = float((lock.logits(model, k, cons[k]).argmax(-1) != cons[k]["teach"].argmax(-1))
                                  .float().mean())
                nf[k] = float((lock.logits(model, k, new[k]).argmax(-1) == new[k]["y"]).float().mean())
        return ex, ch, fl, nf

    ex0, _, _, nf0 = check()
    print("locked epoch 0 valid " + " ".join(f"{k}={v:.4f}" for k, v in ex0.items()) + " | new fit "
          + " ".join(f"{k}={v:.3f}" for k, v in nf0.items()), flush=True)
    best, best_state, log = None, None, []
    for ep in range(1, epochs + 1):
        for _ in range(steps):
            loss = torch.zeros(())
            for k in keys:
                wn = new[k]["w"] * new_weight
                num = (F.cross_entropy(lock.logits(model, k, new[k]), new[k]["y"], reduction="none") * wn).sum()
                den = wn.sum()
                if k in cons and cons_weight > 0:
                    i = _sample(cons[k], batch, gen)
                    lp = lock.logits(model, k, cons[k], i).log_softmax(-1)
                    cw = cons[k]["cw"][i] if "cw" in cons[k] else torch.ones(len(i))
                    num = num + cons_weight * (cw * F.kl_div(lp, cons[k]["teach"][i], log_target=True,
                                                             reduction="none").sum(-1)).sum()
                    den = den + cons_weight * cw.sum()
                    if hold_weight > 0:
                        miss = F.relu(cons[k]["hold"][i] - _margin(lp, cons[k]["a0"][i]))
                        num = num + hold_weight * (cw * miss).sum()
                        den = den + hold_weight * cw.sum()
                if k in replay and replay_weight > 0:
                    i = _sample(replay[k], batch, gen)
                    num = num + replay_weight * F.cross_entropy(lock.logits(model, k, replay[k], i),
                                                                replay[k]["y"][i], reduction="none").sum()
                    den = den + replay_weight * len(i)
                loss = loss + (type_weight if k == TYPE_KEY else 1.0) * num / den
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ep % check_every == 0 or ep == epochs:
            ex, ch, fl, nf = check()
            drop = [k for k in ex if ex[k] < ex0[k] - tolerance - 1e-9]
            big = [k for k in ch if ch[k] > max_change]
            flip = [k for k in fl if fl[k] > max_flip]
            ok = not drop and not big and not flip
            fit = sum(nf.values()) / len(nf)
            last = float(loss.detach())
            log.append(dict(epoch=ep, loss=last, valid=ex, change=ch, cons_flip=fl, newfit=nf, ok=ok))
            print(f"locked epoch {ep} loss {last:.4f} valid " + " ".join(f"{k}={v:.4f}" for k, v in ex.items())
                  + " | change " + " ".join(f"{k}={v:.3%}" for k, v in ch.items())
                  + " | cons flip " + " ".join(f"{k}={v:.3%}" for k, v in fl.items())
                  + " | new fit " + " ".join(f"{k}={v:.3f}" for k, v in nf.items())
                  + ("" if ok else f" | NOT eligible drop={drop} change>{max_change}={big} flip>{max_flip}={flip}"),
                  flush=True)
            if on_check is not None:
                on_check(ep, log[-1])
            if ok and (best is None or fit > best[1] + 1e-9):
                best, best_state = (ep, fit), lock.state()
                print(f"locked best epoch {ep} new fit {fit:.3f}", flush=True)
    return best, best_state, log


def changed_heads(base_ck, cand_ck) -> dict[str, list[str]]:
    """Every state tensor that differs between two checkpoints, grouped by head (free_heads.<key> -> <key>; any
    other module path is reported under its first two components). A locked refit must change only its targets."""
    def state(p):
        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
            return torch.load(p, map_location="cpu", weights_only=True, mmap=True)["state"]
    a, b = state(base_ck), state(cand_ck)
    out: dict[str, list[str]] = {}
    for n in sorted(set(a) | set(b)):
        if n in a and n in b and a[n].shape == b[n].shape and torch.equal(a[n], b[n]):
            continue
        head = n[len("free_heads."):].rsplit(".", 1)[0] if n.startswith("free_heads.") else ".".join(n.split(".")[:2])
        out.setdefault(head, []).append(n if n in a and n in b else f"{n} ({'added' if n in b else 'removed'})")
    return out


def flip_report(model, lock: Lock, rows, texts, src, files, answers, state=None, top=200):
    """Lines of unseen deals whose answer flips between the base (delta 0) and ``state`` (default: the lock's
    current deltas), per key: count, before -> after pairs, and the lines (file, text, before/after answer and p)."""
    out = {}
    for k in lock.keys:
        if k not in rows:
            continue
        with torch.no_grad():
            p0 = lock.logits(model, k, rows[k], params=lock.base[k]).softmax(-1)
            p1 = lock.logits(model, k, rows[k], params=state[k] if state else None).softmax(-1)
        a0, a1 = p0.argmax(-1), p1.argmax(-1)
        idx = (a0 != a1).nonzero().flatten().tolist()
        pairs: dict[str, int] = {}
        lines = []
        for i in idx:
            b, c = answers[k][int(a0[i])], answers[k][int(a1[i])]
            pairs[f"{b} -> {c}"] = pairs.get(f"{b} -> {c}", 0) + 1
            lines.append(dict(file=Path(files[int(src[k][i])]).name, text=(texts[k][i] or "")[:200], before=b,
                              p_before=round(float(p0[i, a0[i]]), 3), after=c, p_after=round(float(p1[i, a1[i]]), 3)))
        lines.sort(key=lambda r: -r["p_after"])
        out[k] = dict(lines_read=len(a0), flipped=len(idx), share=len(idx) / max(1, len(a0)),
                      pairs=dict(sorted(pairs.items(), key=lambda kv: -kv[1])), lines=lines[:top])
    return out

