"""Can the base take the document-job judge's place, and at what certainty?

    python -m ml.c3.document_route_report --checkpoint runC.pt.open_links \\
        --deal 010043.json --out runC.docroutes.json [--sure 0.8]

The parser's ``document_job_scope`` stage asks an LLM, per conversation,
whether a document is about this deal's work (``this_deal``) or another job
(``other_job``); only a confident ``other_job`` removes anything. The base
answers the same question per document (``jdg:document_job``, asked of all of
a document's lines, as training asked it). For each certainty in ``--grid``
this reports, on the labeled documents of the given deals:

* ``taken``: documents the base answers at that certainty or more;
* per answer, how many it gave and how many the labels agree with;
* ``wrong_other_job``: sure ``other_job`` calls the labels say are this deal's.
  These are the costly mistakes: a whole document set aside.

``--sure`` is the certainty the live stage will use. Pick it on validation
deals. A holdout deal is read only with ``--confirm`` and a frozen ``--sure``,
once per checkpoint (``<checkpoint>.doc_routed``), so the holdout checks the
number and never picks it. Counts and rates only, never document text.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from .holdout import LABEL_HOLDOUT, deal_number
from .loop import _autocast, live_student, load_deal, training_batches
from .readiness import digest
from .schema import load_schema

KEY = "jdg:document_job"
GRID = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)


def collect(model, batches, key: str = KEY) -> tuple[list[str], list[tuple[int, int, float]]]:
    """(answer names, [(gold, predicted, p)]) for every labeled document."""
    opp = model.schema.by_key()[key]
    names = [a.value for a in opp.answers]
    rows: list[tuple[int, int, float]] = []
    model.eval()
    with torch.no_grad(), _autocast(next(model.parameters()).device.type):
        for b in batches:
            items = [j for j in b.judged if j.key == key]
            if not items:
                continue
            out = model(b.oneshot(), company=b.company or None)
            logits = model.judge(out, key, [j.lines for j in items],
                                 queries=[j.query for j in items]).float()
            p, pred = logits.softmax(-1).max(-1)
            for j, pj, yj in zip(items, p.tolist(), pred.tolist()):
                rows.append((int(j.answer), int(yj), float(pj)))
    return names, rows


def table(names: list[str], rows, sure: float) -> dict:
    took = [(g, b) for g, b, p in rows if p >= sure]
    per = {}
    for a, name in enumerate(names):
        said = [g for g, b in took if b == a]
        gold_n = sum(g == a for g, _, _ in rows)
        if not said and not gold_n:
            continue
        per[name] = {"gold": gold_n, "said": len(said), "right": sum(g == a for g in said),
                     "precision": round(sum(g == a for g in said) / len(said), 4) if said else None}
    other = names.index("other_job") if "other_job" in names else -1
    this = names.index("this_deal") if "this_deal" in names else -1
    return {"sure": sure, "documents": len(rows), "taken": len(took),
            "taken_share": round(len(took) / max(1, len(rows)), 4),
            "exact_taken": round(sum(g == b for g, b in took) / max(1, len(took)), 4),
            "wrong_other_job": sum(1 for g, b in took if b == other and g == this),
            "answers": per}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--deal", nargs="+", required=True, help="training-format deal files")
    p.add_argument("--out", required=True)
    p.add_argument("--sure", type=float, default=0.8, help="the live stage's certainty")
    p.add_argument("--confirm", action="store_true", help="read the holdout, once, at a frozen --sure")
    p.add_argument("--max-atoms", type=int, default=8192)
    p.add_argument("--encoder", default="answerdotai/ModernBERT-large")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    from .gates import load_checkpoint

    deals = [load_deal(path) for path in args.deal]
    held = {deal_number(d.deal_id) for d in deals} & set(LABEL_HOLDOUT)
    mark = Path(str(args.checkpoint) + ".doc_routed")
    if held:
        if not args.confirm:
            raise SystemExit("A holdout deal is read only with --confirm and a frozen --sure")
        if mark.exists():
            raise SystemExit(f"This checkpoint's certainty was already checked on the holdout ({mark})")
        if held != set(LABEL_HOLDOUT):
            raise SystemExit("Check the whole labeling holdout at once")
    dev = args.device
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    schema = load_schema()
    model = live_student(schema, args.encoder).to(dev)
    load_checkpoint(args.checkpoint, model, schema_sha256=digest([asdict(o) for o in schema.opportunities]))
    batches = [b for d in deals for b in training_batches(d, schema, max_atoms=args.max_atoms)[0]]
    names, rows = collect(model, batches)
    grid = (args.sure,) if held else tuple(sorted(set(GRID) | {args.sure}))
    report = {"checkpoint": str(args.checkpoint), "deals": [d.deal_id for d in deals],
              "holdout": bool(held), "grid": [table(names, rows, s) for s in grid]}
    Path(args.out).write_text(json.dumps(report, indent=1))
    if held:
        mark.write_text(json.dumps({"sure": args.sure, "deals": sorted(held)}))
    for t in report["grid"]:
        a = t["answers"]
        print(f"sure>={t['sure']:.2f}: takes {t['taken']}/{t['documents']} docs "
              f"exact {t['exact_taken']:.3f} wrong_other_job={t['wrong_other_job']} | "
              + " ".join(f"{k}:{v['right']}/{v['said']}" for k, v in a.items()))


if __name__ == "__main__":
    main()
