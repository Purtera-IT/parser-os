"""Refit judgment heads (site pairs, site role, facility, ...) from labels, on a
Mac, in minutes, no GPU, and say honestly how they compare with the parser.

    python -m ml.c3.judgment_fit --deals 'train/*.json' \\
        --keys jdg:site jdg:site_role jdg:facility jdg:geo_mention --out site_refit.pt

Each deal is read once through the same path training uses
(``loop.training_batches``), so every labeled card gets the subject lines and
source query training gave it. Per card three things are cached: the frozen
head's judgment features, its logits, and the label; and the parser's own
answer on that card (the ``parser_value`` the labeling page recorded: what the
rules or the LLM said). A zero-init add-on per head is trained on top of the
frozen logits.

Report, per head:
* ``base``    the frozen head on these deals (it trained on them: flattering);
* ``parser``  the rules / LLM answer the card recorded;
* ``refit``   leave-one-deal-out: each deal scored by an add-on trained on the
              others, so it is on deals the add-on never saw.
Counts and rates only, never line text.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class JudgmentRefit(nn.Module):
    def __init__(self, d: int, answers: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, answers))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def _parser_values(path: str) -> dict[str, str]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for j in d.get("judgments") or []:
        key = str(j.get("target_key") or "")
        pv = j.get("parser_value")
        if key and pv not in (None, ""):
            out[f"jdg:{j.get('head')}|{key}"] = str(pv)
    return out


@torch.no_grad()
def cache_deal(model, schema, device, path: str, keys: tuple[str, ...], max_atoms: int):
    from .loop import load_deal, training_batches
    from .predict import _read_autocast

    deal = load_deal(path)
    parts, _ = training_batches(deal, schema, max_atoms=max_atoms)
    parser = _parser_values(path)
    rows: dict[str, list] = collections.defaultdict(list)
    seen: set[tuple] = set()
    for part in parts:
        items = [it for it in part.judged if it.key in keys and it.answers is None]
        if not items:
            continue
        with _read_autocast(device):
            out = model(part.oneshot(), company=part.company or None)
            for key in keys:
                mine = [it for it in items if it.key == key]
                mine = [it for it in mine if (key, it.subject_key or it.lines) not in seen]
                if not mine:
                    continue
                for it in mine:
                    seen.add((key, it.subject_key or it.lines))
                lines = [it.lines for it in mine]
                queries = [it.query for it in mine]
                feats = model.judgment_features(out, key, lines, queries).float()
                logits = model.judge(out, key, lines, queries=queries).float()
                for it, f, lg in zip(mine, feats, logits):
                    rows[key].append({"f": f.cpu(), "base": lg.cpu(), "y": it.answer,
                                      "parser": parser.get(f"{key}|{it.subject_key}")})
    return rows


def train(rows, answers: int, *, epochs: int = 300, lr: float = 3e-3, wd: float = 1e-2) -> JudgmentRefit:
    torch.manual_seed(0)
    f = torch.stack([r["f"] for r in rows])
    base = torch.stack([r["base"] for r in rows])
    y = torch.tensor([r["y"] for r in rows])
    m = JudgmentRefit(f.shape[-1], answers)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(base + m(f), y).backward()
        opt.step()
    return m.eval()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--deals", nargs="+", required=True)
    ap.add_argument("--keys", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-atoms", type=int, default=4096)
    a = ap.parse_args()

    from .serve import _load

    model, schema, device = _load()
    keys = tuple(a.keys)
    files = sorted({f for g in a.deals for f in glob.glob(g)} | {f for f in a.deals if Path(f).is_file()})
    started = time.time()
    per_deal: dict[str, dict[str, list]] = {}
    for f in files:
        rows = cache_deal(model, schema, device, f, keys, a.max_atoms)
        if any(rows.values()):
            per_deal[Path(f).stem] = rows
            print(f"read {Path(f).stem}: " + ", ".join(f"{k}={len(v)}" for k, v in rows.items()),
                  f"{time.time() - started:.0f}s", flush=True)
    report, saved = {}, {}
    for key in keys:
        names = [x.value for x in schema.by_key()[key].answers]
        deals = {d: r[key] for d, r in per_deal.items() if r.get(key)}
        if not deals:
            continue
        hit = collections.Counter()
        n = 0
        conf = collections.Counter()
        for held, rows in deals.items():
            rest = [r for d, rs in deals.items() if d != held for r in rs]
            m = train(rest, len(names)) if rest else None
            with torch.no_grad():
                base = torch.stack([r["base"] for r in rows])
                fit = base + (m(torch.stack([r["f"] for r in rows])) if m is not None else 0)
            for r, b, ft in zip(rows, base.argmax(-1).tolist(), fit.argmax(-1).tolist()):
                n += 1
                hit["base"] += int(b == r["y"])
                hit["refit"] += int(ft == r["y"])
                if r["parser"] is not None:
                    hit["parser_n"] += 1
                    hit["parser"] += int(r["parser"] == names[r["y"]])
                    hit["refit_on_parser_cards"] += int(ft == r["y"])
                    hit["base_on_parser_cards"] += int(b == r["y"])
                conf[(names[r["y"]], names[ft])] += 1
        report[key] = {
            "cards": n, "deals": len(deals),
            "label_mix": dict(collections.Counter(names[r["y"]] for rs in deals.values() for r in rs)),
            "base_exact": round(hit["base"] / n, 3),
            "refit_lodo_exact": round(hit["refit"] / n, 3),
            "parser_exact": round(hit["parser"] / max(1, hit["parser_n"]), 3) if hit["parser_n"] else None,
            "on_cards_the_parser_answered": {
                "n": hit["parser_n"],
                "parser": round(hit["parser"] / max(1, hit["parser_n"]), 3),
                "base": round(hit["base_on_parser_cards"] / max(1, hit["parser_n"]), 3),
                "refit_lodo": round(hit["refit_on_parser_cards"] / max(1, hit["parser_n"]), 3)},
            "refit_confusion": {f"{t}->{p}": c for (t, p), c in conf.most_common()},
        }
        final = train([r for rs in deals.values() for r in rs], len(names))
        saved[key] = {"state": final.state_dict(), "d": final.net[0].in_features, "answers": names}
    torch.save({"heads": saved, "report": report}, a.out)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
