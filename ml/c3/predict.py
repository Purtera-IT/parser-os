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
import time
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


def deal_parts(deal: DealExample, schema: Schema, max_atoms: int, *, lines_only: bool = False
               ) -> tuple[list[Batch], Batch | None]:
    """The windows a deal is read in, and the part its whole-deal answers come
    from (None with ``lines_only``, which reads no whole-deal answers)."""
    if len(deal.atoms) <= max_atoms:
        plan = plan_long_deal(deal, schema, max_atoms=max_atoms)
        return plan.windows, plan.windows[0]
    plan = plan_long_deal(deal, schema, max_atoms=max_atoms, overlap=min(256, max_atoms // 4),
                          always_digest=not lines_only)
    return plan.windows, (None if lines_only else plan.digest[0])


def _read_autocast(device: str):
    """The live encoder is bf16 and the layers above it float32, so a read on
    any device runs under autocast, as training on CUDA did."""
    if device.startswith("cpu"):
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    return _autocast(device)


def _answer(opp, i: int) -> str:
    return opp.answers[i].value if 0 <= i < len(opp.answers) else str(i)


def _readings(out, rows: list[int], schema: Schema,
              heads: frozenset[str] | None = None) -> list[dict[str, Any]]:
    """Every line reading for ``rows`` (or only ``heads``), one dict per row.

    Each head's logits for the rows move to the CPU once and are read there:
    element by element on the device, a 2,000-line window costs a device sync
    per number."""
    by_key = schema.by_key()
    reads: list[dict[str, Any]] = [{} for _ in rows]
    idx = torch.tensor(rows, dtype=torch.long)
    for key, logits in out.logits.items():
        opp = by_key.get(key)
        if opp is None or opp.size != LINE or (heads is not None and key not in heads):
            continue
        lg = logits[idx.to(logits.device)].float().cpu()
        if opp.multiple:
            p = lg.sigmoid()
            on = (p > 0.5).tolist()
            for r, (pr, hit) in enumerate(zip(p.tolist(), on)):
                reads[r][key] = {"answers": [_answer(opp, j) for j, h in enumerate(hit) if h],
                                 "p": [round(x, 4) for x in pr]}
        else:
            prob = lg.softmax(-1)
            top, j = prob.max(-1)
            for r, (pj, jj) in enumerate(zip(top.tolist(), j.tolist())):
                reads[r][key] = {"answer": _answer(opp, jj), "p": round(pj, 4)}
            if key == "col:label_type":
                # "_keep" says keep the line, not what it is: ``typed`` is the
                # best named type, for a consumer that needs one.
                keep = [k for k, a in enumerate(opp.answers) if a.value == "_keep"]
                if keep:
                    prob[:, keep] = 0.0
                    tp, tj = prob.max(-1)
                    for r, (pj, jj) in enumerate(zip(tp.tolist(), tj.tolist())):
                        reads[r][key]["typed"] = _answer(opp, jj)
                        reads[r][key]["typed_p"] = round(pj, 4)
    for key, values in out.numbers.items():
        opp = by_key.get(key)
        if opp is not None and opp.kind == NUMBER and (heads is None or key in heads):
            v = torch.expm1(values[idx.to(values.device)].float().cpu()).tolist()
            for r, x in enumerate(v):
                reads[r].setdefault(key, {})["value"] = round(x, 3)
    return reads


def _links(out, batch: Batch, rows: list[int], link_min: float, governs_top: int
           ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    links, unsure = [], []
    keys = batch.keys
    n = len(keys)
    idx = torch.tensor(rows, dtype=torch.long)
    for rel, scores in out.relations.items():
        none = out.link_none.get(rel)
        sc = scores[idx.to(scores.device)].float().cpu()
        if none is not None:
            sc = torch.cat([sc, none[idx.to(none.device)].float().cpu().view(-1, 1)], 1)
        live = torch.isfinite(sc).any(-1)
        p, j = sc.softmax(-1).max(-1)
        for i, ok, pj, jj in zip(rows, live.tolist(), p.tolist(), j.tolist()):
            if not ok or (none is not None and jj == n) or pj < link_min:
                continue
            link = {"src": keys[i], "dst": keys[jj], "relation": rel, "p": round(pj, 4)}
            (links if pj >= 0.5 else unsure).append(link)
    if out.governs is not None and governs_top > 0:
        g = out.governs[idx.to(out.governs.device)].float().cpu().exp()
        g[torch.arange(len(rows)), idx] = 0.0
        top, j = g.topk(min(governs_top, n), -1)
        for i, ps, js in zip(rows, top.tolist(), j.tolist()):
            for pj, jj in zip(ps, js):
                if pj > 0.5:
                    links.append({"src": keys[i], "dst": keys[jj], "relation": "governs",
                                  "p": round(pj, 4)})
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
    with torch.no_grad(), _read_autocast(kwargs.get("device", "cpu")):
        return _read_deal(model, deal, schema, **kwargs)


def _read_deal(model: C3Model, deal: DealExample, schema: Schema, *, max_atoms: int = 8192,
              device: str = "cpu", link_min: float = 0.3, unsure: float = 0.6,
              governs_top: int = 5, lines_only: bool = False,
              deadline: float | None = None,
              heads: tuple[str, ...] | None = None,
              document_heads: tuple[str, ...] = (),
              questions: tuple[str, ...] = (),
              site_pairs: bool = False) -> dict[str, Any]:
    """``site_pairs``: the site-pair head (``jdg:site``) asked of every pair
    of lines in one window that the type head reads as ``physical_site``
    (``_judge_site_pairs``), under ``site_pairs``: ``[[key_a, key_b, answer,
    p]]``.

    ``questions``: question texts asked of the whole deal through the
    Questions head (``jdg:gap``, unanchored, as training asks a checklist or
    generated question): ``{"questions": [{"text", "answer", "p_valid"}]}``,
    where valid means this deal needs it asked and nothing answers it yet.

    ``lines_only`` reads each line and skips the links and the whole-deal
    answers (the live type layer needs only the lines). ``heads`` keeps only
    those line readings (the type layer needs ``col:label_type``); the
    model's pass is the same, so the answers are too. ``deadline`` is a
    ``time.monotonic()`` past which the read stops with ``TimeoutError``,
    checked before each window.

    ``document_heads`` are group judgments asked of each whole document
    (``jdg:document_job``), from the same pass and as training asked them: the
    subject is every line of the document (data.py ``_Lines.document``). A
    document split across windows is judged in the window holding most of
    its lines. They come back under ``documents``: ``{doc_id: {key:
    {"answer", "p"}}}``."""
    model.eval()
    windows, whole = deal_parts(deal, schema, max_atoms, lines_only=lines_only and not questions)
    lines: dict[str, dict[str, Any]] = {}
    links: list[dict[str, Any]] = []
    shaky: list[dict[str, Any]] = []
    documents: dict[str, dict[str, Any]] = {}
    covered: dict[str, int] = {}
    pairs: list[list[Any]] = []
    judged: set[tuple[str, str]] = set()
    atoms = {a.key: a for a in deal.atoms}
    for batch in windows:
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError(f"read passed its deadline at {len(lines)}/{len(deal.atoms)} lines")
        with _read_autocast(device):
            out = model(batch.oneshot(), company=batch.company or None)
        home = [i for i, k in enumerate(batch.keys) if k not in lines]
        for i, reads in zip(home, _readings(out, home, schema,
                                            frozenset(heads) if heads is not None else None)):
            a = atoms[batch.keys[i]]
            lines[a.key] = {"atom_id": a.atom_id, "text": a.text, "doc_id": a.doc_id,
                            "doc_kind": a.doc_kind, "section": a.section, "entered_at": a.entered_at,
                            "reads": reads}
        if document_heads:
            _judge_documents(model, out, batch, atoms, schema, document_heads, documents, covered)
        if site_pairs:
            _judge_site_pairs(model, out, batch, lines, schema, pairs, judged)
        if lines_only:
            continue
        found, maybe = _links(out, batch, home, link_min, governs_top)
        links += found
        shaky += maybe
    answers: dict[str, Any] = {}
    asked: list[dict[str, Any]] = []
    if not lines_only or questions:
        with _read_autocast(device):
            out = model(whole.oneshot(), company=whole.company or None)
        if not lines_only:
            answers = _deal_answers(model, out, whole, schema)
        if questions:
            asked = _ask_questions(model, out, whole, schema, questions)
    low = sorted(k for k, r in lines.items()
                 if r["reads"].get("col:label_type", {}).get("p", 1.0) < unsure)
    got = {"deal_id": deal.deal_id, "lines": lines, "links": links, "deal": answers,
           "documents": documents, "questions": asked, "unsure": {"lines": low, "links": shaky}}
    if site_pairs:
        got["site_pairs"] = pairs
    return got


QUESTION_KEY = "jdg:gap"


def _query_features(model: C3Model, texts: list[str]):
    """judgment_query(aux_text(t)) per text, cached on the model."""
    cache = model.__dict__.setdefault("_question_cache", {})
    todo = [t for t in dict.fromkeys(texts) if t not in cache]
    if todo:
        for t, f in zip(todo, model.judgment_query(model.aux_text(todo))):
            cache[t] = f.detach()
    return torch.stack([cache[t] for t in texts])


def _ask_questions(model: C3Model, out, batch: Batch, schema: Schema, questions: tuple[str, ...],
                   chunk: int = 64) -> list[dict[str, Any]]:
    """Each question asked of the whole deal (every line not excluded is its
    subject, data.judgment_subject's unanchored case)."""
    opp = schema.by_key()[QUESTION_KEY]
    every = tuple(range(len(batch)))
    valid = [a.value for a in opp.answers].index("valid")
    rows: list[dict[str, Any]] = []
    # The deal is pooled once; each question's text features are encoded once
    # per server and kept (the catalog barely changes): judge() re-encoded all
    # 154 per compile, 20 s of a 24 s read on 010043.
    pooled = model.judgment_features(out, QUESTION_KEY, [every], queries=[""])
    desc = model.describe()
    refit = getattr(model, "question_refit", None)
    for at in range(0, len(questions), chunk):
        part = list(questions[at:at + chunk])
        qf = _query_features(model, part)
        features = pooled.expand(len(part), -1) + qf.to(pooled.dtype)
        logits, _ = model.judgment_head(features, desc[QUESTION_KEY][0].unsqueeze(0), [desc[QUESTION_KEY][1]])
        logits = logits[0].float()
        if refit is not None:
            # The Mac-trained add-on (question_fit.py) on the frozen head.
            logits = logits + refit(pooled.float().expand(len(part), -1), qf.float()).float()
        p = logits.softmax(-1)
        cal = getattr(refit, "calibration", None) if refit is not None else None
        if cal:
            # Held-out temperature on valid-vs-rest (question_fit): real probabilities.
            pv = p[:, valid].clamp(1e-6, 1 - 1e-6)
            pv = torch.sigmoid(torch.log(pv / (1 - pv)) * cal["scale"] + cal["shift"])
        else:
            pv = p[:, valid]
        for q, pq, v in zip(part, p.tolist(), pv.tolist()):
            rows.append({"text": q, "answer": "valid" if v >= 0.5 else "invalid", "p_valid": round(v, 4)})
    return rows


def _judge_documents(model: C3Model, out, batch: Batch, atoms: dict[str, Any], schema: Schema,
                     keys: tuple[str, ...], documents: dict[str, dict[str, Any]],
                     covered: dict[str, int]) -> None:
    """Each document's group judgments from this window, kept where the window
    holds more of the document's lines than any window before it."""
    rows: dict[str, list[int]] = {}
    for i, k in enumerate(batch.keys):
        doc = atoms[k].doc_id
        if doc:
            rows.setdefault(doc, []).append(i)
    take = [d for d, r in rows.items() if len(r) > covered.get(d, 0)]
    if not take or not hasattr(model, "judge"):
        return
    by_key = schema.by_key()
    for key in keys:
        opp = by_key.get(key)
        if opp is None:
            continue
        subjects, queries = [tuple(rows[d]) for d in take], ["" for _ in take]
        try:
            logits = model.judge(out, key, subjects, queries=queries).float()
            # A Mac-trained add-on (judgment_fit.py) on the frozen head, as training scored it.
            refit = (getattr(model, "judgment_refits", None) or {}).get(key)
            if refit is not None:
                feats = model.judgment_features(out, key, subjects, queries).float()
                logits = logits + refit(feats.to(next(refit.parameters()).device)).float().to(logits.device)
        except (ValueError, KeyError):
            continue
        p = logits.softmax(-1)
        top, j = p.max(-1)
        for d, pj, jj in zip(take, top.tolist(), j.tolist()):
            documents.setdefault(d, {})[key] = {"answer": _answer(opp, jj), "p": round(pj, 4)}
    for d in take:
        covered[d] = len(rows[d])


SITE_PAIR_KEY = "jdg:site"
#: Site lines per window the pair head is asked about (every pair among them):
#: 300 lines is 44,850 pairs. Past it, the first 300 in reading order.
SITE_PAIR_MAX_LINES = 300
_PAIR_CHUNK = 4096


def _judge_site_pairs(model: C3Model, out, batch: Batch, lines: dict[str, dict[str, Any]], schema: Schema,
                      pairs: list[list[Any]], judged: set[tuple[str, str]]) -> None:
    """The site-pair head (``jdg:site``: same_site / distinct_site) on every
    pair of lines in this window that the type head reads as
    ``physical_site``, as training asks it (the pair's two lines, in reading
    order). A pair already judged in an earlier window is not asked again.
    The Deal Kit reads the answers so it never merges two rows the head calls
    different sites."""
    opp = schema.by_key().get(SITE_PAIR_KEY)
    if opp is None or not hasattr(model, "judge"):
        return
    site = [i for i, k in enumerate(batch.keys)
            if ((lines.get(k) or {}).get("reads", {}).get("col:label_type") or {}).get("answer") == "physical_site"]
    site = site[:SITE_PAIR_MAX_LINES]
    todo = [(i, j) for n, i in enumerate(site) for j in site[n + 1:]
            if (batch.keys[i], batch.keys[j]) not in judged]
    refit = (getattr(model, "judgment_refits", None) or {}).get(SITE_PAIR_KEY)
    for at in range(0, len(todo), _PAIR_CHUNK):
        chunk = todo[at:at + _PAIR_CHUNK]
        subjects, queries = [tuple(x) for x in chunk], ["" for _ in chunk]
        try:
            logits = model.judge(out, SITE_PAIR_KEY, subjects, queries=queries).float()
            if refit is not None:
                feats = model.judgment_features(out, SITE_PAIR_KEY, subjects, queries).float()
                logits = logits + refit(feats.to(next(refit.parameters()).device)).float().to(logits.device)
        except (ValueError, KeyError):
            return
        p = logits.softmax(-1)
        top, j = p.max(-1)
        for (a, b), pj, jj in zip(chunk, top.tolist(), j.tolist()):
            ka, kb = batch.keys[a], batch.keys[b]
            judged.add((ka, kb))
            answer = _answer(opp, jj)
            if answer != "_absent":
                pairs.append([ka, kb, answer, round(pj, 4)])


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


def _print_type(head: str, row: dict[str, float]) -> None:
    print(head + " " + " ".join(f"{k}={int(v)}" if k in ("n", "routed_n") else f"{k}={v:.3f}"
                                for k, v in row.items()), flush=True)


def _model(schema: Schema, args) -> tuple[C3Model, str]:
    if args.tiny:
        return C3Model(schema, SMALL), "cpu"
    device = args.device
    if device == "auto":
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available() else "cpu")
    # The base is a 400M-parameter reader: a GPU is faster, not required.
    return live_student(schema, args.encoder).to(device), device


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--deal", nargs="+", required=True, help="training-format deal files")
    p.add_argument("--output", help="output file (one deal) or directory (several)")
    p.add_argument("--max-atoms", type=int, default=8192)
    p.add_argument("--encoder", default="answerdotai/ModernBERT-large")
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--device", default="auto", help="cuda, mps or cpu (auto picks the first present)")
    p.add_argument("--link-min", type=float, default=0.3)
    p.add_argument("--unsure", type=float, default=0.6)
    p.add_argument("--governs-top", type=int, default=5)
    p.add_argument("--score", action="store_true", help="score the holdout's saved labels, once")
    p.add_argument("--train", nargs="*", default=[],
                   help="with --score: the training deals, for each head's majority-answer floor")
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
        from .loop import majority_baseline, training_batches, type_check

        rows, every = [], []
        for deal in deals:
            scores = score_holdout(model, deal, schema, max_atoms=args.max_atoms, device=device)
            rows.append(scores)
            print("deal", deal.deal_id)
            for key, row in sorted(scores.items()):
                print(f"  {key} exact={row['exact']:.3f} n={int(row['n'])}")
            # Line type against the parser's classifier on the same lines, as
            # the run's checks read it (Deal Kit and estimating calc left out).
            parts, _ = training_batches(deal, schema, max_atoms=args.max_atoms)
            every += parts
            _print_type(f"  type {deal.deal_id}", type_check(model, parts, device))
        pooled = pool_scores(rows)
        # Every head against always giving its most common training answer:
        # a head that does not clear that floor has learned nothing to keep.
        floor = {}
        if args.train:
            seen = [b for path in args.train
                    for b in training_batches(load_deal(path), schema, max_atoms=args.max_atoms)[0]]
            floor = majority_baseline(seen, every)
        print("holdout")
        for key, row in sorted(pooled.items()):
            print(f"  {key} exact={row['exact']:.3f} n={int(row['n'])}"
                  + (f" majority={floor[key]:.3f}" if key in floor else ""))
        same = [k for k in pooled if k in floor]
        if same:
            print(f"  beat_majority {sum(pooled[k]['exact'] > floor[k] for k in same)}/{len(same)} "
                  f"macro={sum(pooled[k]['exact'] for k in same) / len(same):.4f} "
                  f"majority={sum(floor[k] for k in same) / len(same):.4f}")
        _print_type("  type holdout", type_check(model, every, device))
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
