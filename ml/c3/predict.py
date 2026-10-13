"""The base's read of a deal, as one file the next models use.

    python -m ml.c3.predict --checkpoint runA.pt --deal train/010172.json \\
        --output reads/010172.json [--max-atoms 8192] [--tiny]

One file per deal, labeled or not (a Deal Kit deal nobody labeled reads the
same way). It holds the deal's own words, so like the training files it
lives outside the repository:

* ``lines``: per line, its words, document, section and arrival time, then
  every reading the base makes (answer and probability;
  a multi-select reading lists every answer over 0.5; a number is the
  stated value). A line's readings come from its home window, as in
  training.
* ``links``: per relation, each line's best target when it beats "no link"
  and reaches ``--link-min``, with the probability. ``governs`` keeps each
  parent's children over 0.5, best first, at most ``--governs-top``.
* ``deal``: the whole-deal answers, read on the deal's digest (or the whole
  deal when it fits one window).
* ``unsure``: lines whose type is under ``--unsure`` and links between
  ``--link-min`` and 0.5: the next labeling queue.

The Deal Kit, the SOW step and a prompted LLM read this file, so what they
train on is what they will see live. ``--score`` instead scores the saved
labels of a holdout deal, once: it writes ``<checkpoint>.scored`` and refuses
a second time, so the holdout cannot become a tuning set.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .data import Batch, DealExample
from .gates import load_checkpoint
from .holdout import LABEL_HOLDOUT, deal_number
from .loop import SMALL, _autocast, _score_oneshot, live_student, load_deal, pool_scores
from .longdeal import plan_long_deal
from .model import C3Model
from .readiness import digest
from .schema import DEAL, LINE, NUMBER, RELATION, Schema, load_schema


def deal_parts(deal: DealExample, schema: Schema, max_atoms: int) -> tuple[list[Batch], Batch]:
    """The windows a deal is read in, and the part its whole-deal answers come from."""
    if len(deal.atoms) <= max_atoms:
        plan = plan_long_deal(deal, schema, max_atoms=max_atoms)
        return plan.windows, plan.windows[0]
    plan = plan_long_deal(deal, schema, max_atoms=max_atoms, overlap=min(256, max_atoms // 4),
                          always_digest=True)
    return plan.windows, plan.digest[0]


def _answer(opp, i: int) -> str:
    return opp.answers[i].value if 0 <= i < len(opp.answers) else str(i)


def _line_readings(model: C3Model, out, batch: Batch, row: int, schema: Schema) -> dict[str, Any]:
    by_key = schema.by_key()
    reads: dict[str, Any] = {}
    for key, logits in out.logits.items():
        opp = by_key.get(key)
        if opp is None or opp.size != LINE:
            continue
        lg = logits[row].float()
        if opp.multiple:
            p = lg.sigmoid()
            reads[key] = {"answers": [_answer(opp, int(j)) for j in (p > 0.5).nonzero().flatten().tolist()],
                          "p": [round(float(x), 4) for x in p.tolist()]}
        else:
            p = lg.softmax(-1)
            j = int(p.argmax())
            reads[key] = {"answer": _answer(opp, j), "p": round(float(p[j]), 4)}
    for key, values in out.numbers.items():
        opp = by_key.get(key)
        if opp is not None and opp.kind == NUMBER:
            reads.setdefault(key, {})["value"] = round(float(torch.expm1(values[row].float())), 3)
    return reads


def _links(out, batch: Batch, rows: list[int], link_min: float, governs_top: int
           ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    links, unsure = [], []
    keys = batch.keys
    for rel, scores in out.relations.items():
        none = out.link_none.get(rel)
        for i in rows:
            row = scores[i].float()
            if none is not None:
                row = torch.cat([row, none[i].float().view(1)])
            if not torch.isfinite(row).any():
                continue
            p = row.softmax(-1)
            j = int(p.argmax())
            if none is not None and j == len(keys):
                continue
            if float(p[j]) < link_min:
                continue
            link = {"src": keys[i], "dst": keys[j], "relation": rel, "p": round(float(p[j]), 4)}
            (links if float(p[j]) >= 0.5 else unsure).append(link)
    if out.governs is not None:
        g = out.governs.float().exp()
        for i in rows:
            kids = [(float(g[i, j]), j) for j in range(len(keys)) if j != i and float(g[i, j]) > 0.5]
            for p, j in sorted(kids, reverse=True)[:governs_top]:
                links.append({"src": keys[i], "dst": keys[j], "relation": "governs", "p": round(p, 4)})
    return links, unsure


def _deal_answers(model: C3Model, out, batch: Batch, schema: Schema) -> dict[str, Any]:
    answers: dict[str, Any] = {}
    every = tuple(range(len(batch)))
    for opp in schema.opportunities:
        if opp.size != DEAL or opp.kind == RELATION or opp.key.endswith(":reason"):
            continue
        try:
            logits = model.judge(out, opp.key, [every], queries=[""])[0].float()
        except (ValueError, KeyError):
            continue
        p = logits.softmax(-1)
        j = int(p.argmax())
        answers[opp.key] = {"answer": _answer(opp, j), "p": round(float(p[j]), 4)}
    if hasattr(model, "deal_number"):
        count = torch.expm1(model.deal_number(out.r.mean(0)).float().squeeze(-1))
        answers["deal:declared_site_count"] = {"value": round(float(count), 2)}
    return answers


def read_deal(model: C3Model, deal: DealExample, schema: Schema, **kwargs) -> dict[str, Any]:
    """``_read_deal`` with every read, including the cards and links taken
    from the pass's outputs, under one no-grad autocast."""
    with torch.no_grad(), _autocast(kwargs.get("device", "cpu")):
        return _read_deal(model, deal, schema, **kwargs)


def _read_deal(model: C3Model, deal: DealExample, schema: Schema, *, max_atoms: int = 4096,
              device: str = "cpu", link_min: float = 0.3, unsure: float = 0.6,
              governs_top: int = 5) -> dict[str, Any]:
    model.eval()
    windows, whole = deal_parts(deal, schema, max_atoms)
    lines: dict[str, dict[str, Any]] = {}
    links: list[dict[str, Any]] = []
    shaky: list[dict[str, Any]] = []
    atoms = {a.key: a for a in deal.atoms}
    for batch in windows:
        with _autocast(device):
            out = model(batch.oneshot(), company=batch.company or None)
        home = [i for i, k in enumerate(batch.keys) if k not in lines]
        for i in home:
            a = atoms[batch.keys[i]]
            lines[a.key] = {"atom_id": a.atom_id, "text": a.text, "doc_id": a.doc_id,
                            "doc_kind": a.doc_kind, "section": a.section, "entered_at": a.entered_at,
                            "reads": _line_readings(model, out, batch, i, schema)}
        found, maybe = _links(out, batch, home, link_min, governs_top)
        links += found
        shaky += maybe
    with _autocast(device):
        out = model(whole.oneshot(), company=whole.company or None)
    answers = _deal_answers(model, out, whole, schema)
    low = sorted(k for k, r in lines.items()
                 if r["reads"].get("col:label_type", {}).get("p", 1.0) < unsure)
    return {"deal_id": deal.deal_id, "lines": lines, "links": links, "deal": answers,
            "unsure": {"lines": low, "links": shaky}}


def kit_from_read(kit: dict[str, Any], read: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
    """A Deal Kit file whose atoms carry the base's types and links, not the saved ones.

    Live, the kit only ever sees the base's read, so it trains on it too. A
    kit line whose atom the base typed for another section keeps the line but
    loses the atom: that disagreement is counted, and it is the base's to fix.
    """
    from .kit import section_for

    lines = read.get("lines") or {}
    out_atoms, stats = [], {"atoms": 0, "retyped": 0, "not_read": 0, "lines_unhosted": 0}
    for raw in kit.get("atoms") or []:
        if not isinstance(raw, dict):
            continue
        key = str(raw.get("key") or raw.get("label_key") or "").strip()
        got = lines.get(key)
        row = {k: v for k, v in raw.items() if k not in ("links",)}
        stats["atoms"] += 1
        if got is None:
            stats["not_read"] += 1
        else:
            kind = (got["reads"].get("col:label_type") or {}).get("answer")
            saved = str(raw.get("type") or raw.get("label_type") or raw.get("atom_type") or "")
            if kind and kind != saved:
                stats["retyped"] += 1
            for name in ("label_type", "atom_type"):
                row.pop(name, None)
            row["type"] = kind or saved
            row["admission"] = (got["reads"].get("col:admission") or {}).get("answer")
        out_atoms.append(row)
    sections = {str(a.get("key") or a.get("label_key")): section_for(str(a.get("type") or ""))
                for a in out_atoms}
    kit_lines = []
    for raw in kit.get("lines") or []:
        row = dict(raw) if isinstance(raw, dict) else raw
        if isinstance(row, dict) and row.get("atom_key") and \
                sections.get(str(row["atom_key"])) not in (None, row.get("section")):
            row["atom_key"] = None
            stats["lines_unhosted"] += 1
        kit_lines.append(row)
    keys = {str(a.get("key") or a.get("label_key")) for a in out_atoms}
    links = [{"src": k["src"], "dst": k["dst"], "relation": k["relation"]}
             for k in read.get("links") or [] if k["src"] in keys and k["dst"] in keys]
    doc = {k: v for k, v in kit.items() if k not in ("atoms", "lines", "links", "edges")}
    return {**doc, "atoms": out_atoms, "lines": kit_lines, "links": links}, stats


def score_holdout(model: C3Model, deal: DealExample, schema: Schema, *, max_atoms: int,
                  device: str) -> dict[str, dict[str, float]]:
    from .loop import training_batches

    parts, _ = training_batches(deal, schema, max_atoms=max_atoms)
    return pool_scores([_score_oneshot(model, b, device) for b in parts])


def _model(schema: Schema, args) -> tuple[C3Model, str]:
    if args.tiny:
        return C3Model(schema, SMALL), "cpu"
    if not torch.cuda.is_available():
        raise SystemExit("Reading with the live base needs CUDA. --tiny is the CPU check.")
    return live_student(schema, args.encoder).to("cuda"), "cuda"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--deal", nargs="+", required=True, help="training-format deal files")
    p.add_argument("--output", help="output file (one deal) or directory (several)")
    p.add_argument("--max-atoms", type=int, default=4096)
    p.add_argument("--encoder", default="answerdotai/ModernBERT-large")
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--link-min", type=float, default=0.3)
    p.add_argument("--unsure", type=float, default=0.6)
    p.add_argument("--governs-top", type=int, default=5)
    p.add_argument("--score", action="store_true", help="score the holdout's saved labels, once")
    p.add_argument("--kit", help="a Deal Kit file for this deal: write it again with the base's atoms")
    p.add_argument("--kit-output", help="where the rewritten Deal Kit file goes")
    args = p.parse_args()
    schema = load_schema()
    model, device = _model(schema, args)
    step = load_checkpoint(args.checkpoint, model,
                           schema_sha256=digest([asdict(o) for o in schema.opportunities]))
    print("checkpoint", args.checkpoint, "step", step)
    deals = [load_deal(path) for path in args.deal]
    if args.score:
        mark = Path(str(args.checkpoint) + ".scored")
        if mark.exists():
            raise SystemExit(f"This checkpoint was already scored on the holdout ({mark})")
        ids = {deal_number(d.deal_id) for d in deals}
        if not set(LABEL_HOLDOUT) <= ids:
            raise SystemExit("Score the whole labeling holdout at once: "
                             + ", ".join(i for i in LABEL_HOLDOUT if i not in ids))
        rows = []
        for deal in deals:
            scores = score_holdout(model, deal, schema, max_atoms=args.max_atoms, device=device)
            rows.append(scores)
            print("deal", deal.deal_id)
            for key, row in sorted(scores.items()):
                print(f"  {key} exact={row['exact']:.3f} n={int(row['n'])}")
        pooled = pool_scores(rows)
        print("holdout")
        for key, row in sorted(pooled.items()):
            print(f"  {key} exact={row['exact']:.3f} n={int(row['n'])}")
        mark.write_text(json.dumps({"deals": sorted(ids), "heads": len(pooled)}), encoding="utf-8")
        return
    if not args.output:
        raise SystemExit("--output is required unless --score")
    out = Path(args.output)
    for deal in deals:
        doc = read_deal(model, deal, schema, max_atoms=args.max_atoms, device=device,
                        link_min=args.link_min, unsure=args.unsure, governs_top=args.governs_top)
        target = out / f"{deal_number(deal.deal_id)}.json" if len(deals) > 1 or out.is_dir() else out
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(doc), encoding="utf-8")
        print("wrote", target, "lines", len(doc["lines"]), "links", len(doc["links"]),
              "unsure", len(doc["unsure"]["lines"]))
        if args.kit:
            if len(deals) != 1 or not args.kit_output:
                raise SystemExit("--kit takes one --deal and a --kit-output")
            kit, stats = kit_from_read(json.loads(Path(args.kit).read_text(encoding="utf-8")), doc)
            Path(args.kit_output).write_text(json.dumps(kit), encoding="utf-8")
            print("kit", args.kit_output, json.dumps(stats))


if __name__ == "__main__":
    main()
