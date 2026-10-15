"""Which line types the base can take from the LLM, one type at a time.

    python -m ml.c3.route_report --checkpoint runC.pt.open_links \\
        --deal 010043.json --out runC.routes.json [--sure 0.7] [--bar 0.85]

For every line-type answer, on the labeled lines of the given deals (Deal Kit
and estimating calc left out, as at run time):

* ``base_n`` / ``base_precision``: lines the base gives that answer at
  ``--sure`` or more, and how many of those the labels agree with. These are
  the lines the live layer would take from the LLM.
* ``classifier_precision`` / ``classifier_recall``: the parser's classifier
  on the same lines, for that answer.
* ``base_recall``: share of the answer's labeled lines the base takes, sure.
* ``acts``: the answer is one the live layer may act on at all (a value-light
  AtomType, or ``_keep``); the rest always reach the LLM, which writes values.
* ``replace``: ``acts``, at least ``--min-n`` sure lines, base precision at
  ``--bar`` and no worse than the classifier's for that answer.

The ``replace`` answers are the ``SOWSMITH_C3_TYPES`` list, printed last with
what it would take from the LLM and how often it would be wrong (misroute).
Pick the list on validation deals. A holdout deal is read only with
``--confirm`` and a frozen ``--types`` list, once per checkpoint
(``<checkpoint>.routed``), so the holdout checks the list and never picks it.
Counts and rates only, never line text.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from .data import IGNORE
from .holdout import LABEL_HOLDOUT, deal_number
from .loop import CHECK_SKIP_DOC_KINDS, TYPE_KEY, _autocast, live_student, load_deal, training_batches
from .readiness import digest
from .schema import load_schema

#: What the live layer may assign (app/core/typed_atom_classifier.py's
#: value-light set; a test keeps the two equal) plus ``_keep``.
VALUE_LIGHT = frozenset({
    "requirement", "exclusion", "contract_term", "deal_metadata",
    "acceptance_criterion", "task", "change_order_rule", "constraint",
    "dependency", "mitigation", "compliance_rule", "submission_req",
    "addendum_qa",
})
ACTABLE = VALUE_LIGHT | {"_keep"}


def count(model, batches, device: str, sure: float) -> tuple[list[str], list[tuple[int, int, int, float]]]:
    """(answer names, one (gold, base, parser, p) row per scored line)."""
    names = [a.value for a in model.schema.by_key()[TYPE_KEY].answers]
    rows = []
    model.eval()
    with torch.no_grad(), _autocast(device):
        for b in batches:
            gold = b.targets.get(TYPE_KEY)
            if not gold or all(y == IGNORE for y in gold):
                continue
            prob = model(b.oneshot(), company=b.company or None).logits[TYPE_KEY].float().softmax(-1)
            p, pred = prob.max(-1)
            parser = b.parser_targets.get(TYPE_KEY) or [IGNORE] * len(gold)
            for i, y in enumerate(gold):
                if y == IGNORE or str(b.doc_kind[i] or "").lower() in CHECK_SKIP_DOC_KINDS:
                    continue
                rows.append((int(y), int(pred[i]), int(parser[i]), float(p[i])))
    return names, rows


def table(names: list[str], rows, *, sure: float, bar: float, min_n: int) -> dict[str, dict]:
    out = {}
    for a, name in enumerate(names):
        gold_n = sum(g == a for g, _, _, _ in rows)
        took = [(g, b) for g, b, _, p in rows if b == a and p >= sure]
        said = [g for g, _, c, _ in rows if c == a]
        if not gold_n and not took and not said:
            continue
        base_hit = sum(g == a for g, _ in took)
        cls_hit = sum(g == a for g in said)
        base_prec = base_hit / len(took) if took else 0.0
        cls_prec = cls_hit / len(said) if said else 0.0
        acts = name in ACTABLE
        out[name] = {
            "gold_n": gold_n, "base_n": len(took), "base_precision": round(base_prec, 4),
            "base_recall": round(base_hit / gold_n, 4) if gold_n else 0.0,
            "classifier_n": len(said), "classifier_precision": round(cls_prec, 4),
            "classifier_recall": round(cls_hit / gold_n, 4) if gold_n else 0.0,
            "acts": acts,
            "replace": bool(acts and len(took) >= min_n and base_prec >= bar and base_prec >= cls_prec),
        }
    return out


def routed(names: list[str], rows, types: set[str], sure: float) -> dict[str, float]:
    """What a ``SOWSMITH_C3_TYPES`` list does on these lines: the share the
    base takes from the LLM, how many of those are wrong, and line type
    exact with the base on the list and the classifier everywhere else."""
    idx = {names.index(t) for t in types if t in names}
    n = max(1, len(rows))
    took = [(g, b) for g, b, _, p in rows if b in idx and p >= sure]
    wrong = sum(g != b for g, b in took)
    final = [b if (b in idx and p >= sure) else c for g, b, c, p in rows]
    cls_only = sum(g == c for g, _, c, _ in rows)
    return {"lines": len(rows), "taken": len(took), "taken_share": round(len(took) / n, 4),
            "misroute": wrong, "misroute_rate": round(wrong / max(1, len(took)), 4),
            "routed_exact": round(sum(f == r[0] for f, r in zip(final, rows)) / n, 4),
            "classifier_exact": round(cls_only / n, 4)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--deal", nargs="+", required=True, help="training-format deal files")
    p.add_argument("--out", required=True)
    p.add_argument("--sure", type=float, default=0.7, help="the live SOWSMITH_C3_SURE")
    p.add_argument("--bar", type=float, default=0.85, help="lowest base precision for a type it takes")
    p.add_argument("--min-n", type=int, default=20, help="fewest sure lines a type needs to be judged")
    p.add_argument("--types", help="a frozen comma list to check instead of picking one")
    p.add_argument("--confirm", action="store_true", help="read the holdout, once, to check --types")
    p.add_argument("--max-atoms", type=int, default=8192)
    p.add_argument("--encoder", default="answerdotai/ModernBERT-large")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    from .gates import load_checkpoint

    deals = [load_deal(path) for path in args.deal]
    held = {deal_number(d.deal_id) for d in deals} & set(LABEL_HOLDOUT)
    mark = Path(str(args.checkpoint) + ".routed")
    if held:
        if not (args.confirm and args.types):
            raise SystemExit("A holdout deal is read only with --confirm and a frozen --types list")
        if mark.exists():
            raise SystemExit(f"This checkpoint's list was already checked on the holdout ({mark})")
        if held != set(LABEL_HOLDOUT):
            raise SystemExit("Check the whole labeling holdout at once")
    dev = args.device
    if dev == "auto":
        dev = ("cuda" if torch.cuda.is_available()
               else "mps" if torch.backends.mps.is_available() else "cpu")
    schema = load_schema()
    model = live_student(schema, args.encoder).to(dev)
    load_checkpoint(args.checkpoint, model, schema_sha256=digest([asdict(o) for o in schema.opportunities]))
    parts = [b for d in deals for b in training_batches(d, schema, max_atoms=args.max_atoms)[0]]
    names, rows = count(model, parts, dev, args.sure)
    per = table(names, rows, sure=args.sure, bar=args.bar, min_n=args.min_n)
    types = ({t.strip() for t in args.types.split(",") if t.strip()} if args.types
             else {k for k, v in per.items() if v["replace"]})
    result = {"checkpoint": Path(args.checkpoint).name, "deals": sorted(d.deal_id for d in deals),
              "sure": args.sure, "bar": args.bar, "min_n": args.min_n, "per_type": per,
              "types": sorted(types), "routed": routed(names, rows, types, args.sure)}
    Path(args.out).write_text(json.dumps(result, indent=1, sort_keys=True), encoding="utf-8")
    for k, v in sorted(per.items(), key=lambda kv: -kv[1]["base_n"]):
        print(f"{k:24s} gold={v['gold_n']:5d} base={v['base_n']:5d} prec={v['base_precision']:.3f} "
              f"rec={v['base_recall']:.3f} cls_prec={v['classifier_precision']:.3f} "
              f"cls_rec={v['classifier_recall']:.3f}" + (" REPLACE" if v["replace"] else "")
              + ("" if v["acts"] else " (llm only)"), flush=True)
    r = result["routed"]
    print(f"SOWSMITH_C3_TYPES={','.join(sorted(types))}")
    print(f"takes {r['taken']}/{r['lines']} ({r['taken_share']:.1%}) misroute {r['misroute']} "
          f"({r['misroute_rate']:.1%}) routed_exact={r['routed_exact']:.3f} "
          f"classifier_exact={r['classifier_exact']:.3f}", flush=True)
    if held:
        mark.write_text(json.dumps({"deals": sorted(held), "types": sorted(types)}), encoding="utf-8")


if __name__ == "__main__":
    main()
