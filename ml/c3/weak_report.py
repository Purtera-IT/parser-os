"""Where the base is weak, and what to label next.

    python -m ml.c3.weak_report --reads reads/*.json [--routes routes.json ...] \\
        [--shadow-log compile1.log ...] --out weak.md [--queue 50]

Three kinds of evidence, any of them optional, one report:

* ``--reads``: ``ml.c3.predict`` reads of any deals, labeled or not. Per head,
  the share of lines the base is unsure on (top answer under ``--unsure``);
  per deal, how many lines it is unsure of the line type; and the label
  queue: the least-sure lines per deal, by line key, with the base's guess.
* ``--routes``: ``ml.c3.route_report`` outputs. Types the base cannot take
  yet, and why: too few sure lines (label more of that type) or precision
  under the bar or under the classifier (label its confusions).
* ``--shadow-log``: compile logs from a dev compile in shadow. The
  ``c3_type_shadow`` lines, pooled: agreement per answer and the commonest
  disagreements (base answer -> compile type), and the layer's time.

Line keys, counts and rates only, never line text: the labeler opens a key on
the labeling page.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

TYPE = "col:label_type"


def from_reads(reads: list[dict[str, Any]], unsure: float, queue: int) -> dict[str, Any]:
    heads: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    deals, picks = {}, []
    for r in reads:
        lines = r.get("lines") or {}
        low_type = Counter()
        scored = []
        for key, line in lines.items():
            for head, got in (line.get("reads") or {}).items():
                p = got.get("p") if isinstance(got, dict) else None
                if p is None:
                    continue
                heads[head][0] += 1
                heads[head][1] += p < unsure
            t = (line.get("reads") or {}).get(TYPE) or {}
            if t.get("p") is not None:
                scored.append((float(t["p"]), key, str(t.get("answer"))))
                if t["p"] < unsure:
                    low_type[str(t.get("answer"))] += 1
        scored.sort()
        deals[r.get("deal_id", "?")] = {"lines": len(lines), "unsure_type": sum(low_type.values()),
                                        "unsure_by_guess": dict(low_type.most_common(8))}
        picks += [{"deal": r.get("deal_id", "?"), "key": k, "guess": a, "p": round(p, 3)}
                  for p, k, a in scored[:queue] if p < unsure]
    weak = sorted(((h, b / n, n) for h, (n, b) in heads.items() if n), key=lambda x: -x[1])
    return {"heads": [{"head": h, "unsure_share": round(s, 4), "n": n} for h, s, n in weak],
            "deals": deals, "queue": sorted(picks, key=lambda x: x["p"])[:queue * max(1, len(reads))]}


def from_routes(routes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in routes:
        min_n, bar, keep_bar = r.get("min_n", 20), r.get("bar", 0.85), r.get("keep_bar", 0.97)
        for t, v in sorted((r.get("per_type") or {}).items()):
            if not v.get("acts") or v.get("replace"):
                continue
            need = keep_bar if t == "_keep" else bar
            if v["base_n"] < min_n:
                why = f"only {v['base_n']} sure lines (needs {min_n}): label more {t} lines"
            elif v["base_precision"] < need:
                why = f"precision {v['base_precision']:.3f} under {need}: label its confusions"
            else:
                why = (f"classifier better ({v['classifier_precision']:.3f} vs "
                       f"{v['base_precision']:.3f}): label its confusions")
            out.append({"checkpoint": r.get("checkpoint"), "type": t, "gold_n": v["gold_n"], "why": why})
    return out


def from_shadow(texts: list[str]) -> dict[str, Any]:
    agree: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    pairs: Counter = Counter()
    c3_ms = llm_ms = 0.0
    compiles = 0
    for text in texts:
        for raw in text.splitlines():
            raw = raw.strip()
            if not raw.startswith("{"):
                continue
            try:
                ev = json.loads(raw)
            except ValueError:
                continue
            if ev.get("event") == "c3_type_shadow":
                compiles += 1
                for a, s in (ev.get("by_answer") or {}).items():
                    agree[a][0] += s.get("agree", 0)
                    agree[a][1] += s.get("disagree", 0)
                pairs.update(ev.get("disagree_pairs") or {})
            elif ev.get("event") == "typed_atom_deflect":
                c3_ms += float((ev.get("layer_ms") or {}).get("c3", 0.0))
                llm_ms += float(ev.get("llm_ms") or 0.0)
    return {"compiles": compiles, "c3_ms": round(c3_ms, 1), "llm_ms": round(llm_ms, 1),
            "agreement": {a: {"agree": g, "disagree": d, "rate": round(g / max(1, g + d), 4)}
                          for a, (g, d) in sorted(agree.items())},
            "top_disagreements": dict(pairs.most_common(15))}


def markdown(rep: dict[str, Any]) -> str:
    out = ["# Where the base is weak", ""]
    if rep.get("routes"):
        out += ["## Types the base can't take yet", "", "| type | labeled | what to do |", "|---|---|---|"]
        out += [f"| {r['type']} | {r['gold_n']} | {r['why']} |" for r in rep["routes"]] + [""]
    s = rep.get("shadow")
    if s and s["compiles"]:
        out += ["## Dev shadow", "",
                f"{s['compiles']} compiles. Base time {s['c3_ms'] / 1000:.1f}s, LLM time {s['llm_ms'] / 1000:.1f}s.",
                "", "| answer | agree | disagree | rate |", "|---|---|---|---|"]
        out += [f"| {a} | {v['agree']} | {v['disagree']} | {v['rate']:.3f} |" for a, v in s["agreement"].items()]
        out += ["", "Commonest disagreements (base -> compile):", ""]
        out += [f"- {k}: {n}" for k, n in s["top_disagreements"].items()] + [""]
    r = rep.get("reads")
    if r:
        out += ["## Heads least sure on these deals", "", "| head | unsure share | lines |", "|---|---|---|"]
        out += [f"| {h['head']} | {h['unsure_share']:.3f} | {h['n']} |" for h in r["heads"][:15]] + [""]
        out += ["## Deals", "", "| deal | lines | unsure type | unsure by guess |", "|---|---|---|---|"]
        out += [f"| {d} | {v['lines']} | {v['unsure_type']} | "
                + ", ".join(f"{k} {n}" for k, n in v["unsure_by_guess"].items()) + " |"
                for d, v in sorted(r["deals"].items(), key=lambda kv: -kv[1]["unsure_type"])] + [""]
        out += ["## Label queue (least sure first)", "", "| deal | line key | base guess | p |", "|---|---|---|---|"]
        out += [f"| {q['deal']} | {q['key']} | {q['guess']} | {q['p']:.3f} |" for q in r["queue"]] + [""]
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--reads", nargs="*", default=[])
    p.add_argument("--routes", nargs="*", default=[])
    p.add_argument("--shadow-log", nargs="*", default=[])
    p.add_argument("--unsure", type=float, default=0.6)
    p.add_argument("--queue", type=int, default=50, help="least-sure lines per deal in the queue")
    p.add_argument("--out", required=True, help="markdown report; a .json beside it holds the same")
    args = p.parse_args()
    load = lambda f: json.loads(Path(f).read_text(encoding="utf-8"))  # noqa: E731
    rep: dict[str, Any] = {}
    if args.reads:
        rep["reads"] = from_reads([load(f) for f in args.reads], args.unsure, args.queue)
    if args.routes:
        rep["routes"] = from_routes([load(f) for f in args.routes])
    if args.shadow_log:
        rep["shadow"] = from_shadow([Path(f).read_text(encoding="utf-8", errors="ignore")
                                     for f in args.shadow_log])
    out = Path(args.out)
    out.write_text(markdown(rep), encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(rep, indent=1, sort_keys=True), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
