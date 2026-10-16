"""Same site or not: a site-pair matcher trained on a Mac from the labeled
site cards, scored leave-one-deal-out against what the parser said.

    python -m ml.c3.site_pair_fit --deals 'train/*.json' --out site_pair.pt

Why a separate fit: the base's pair judgment never learned site matching (45%
right even on the deals it trained on, below always-"distinct"), and the
parser says "distinct" on nearly every pair (1 of 64 labeled same-site pairs
caught), so duplicate sites survive. Same-or-different site is mostly an
address comparison, which the base's text encoder plus a few address signals
can learn from a few hundred cards.

Each site is described by its card name, aliases and evidence line. Features:
the base encoder's embeddings of both sides (product and difference, so the
order of the pair does not matter) and address signals (street number, ZIP,
street-word and name-word overlap, same city). Counts and rates only.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_ZIP = re.compile(r"\b(\d{5})(?:-\d{4})?\b")
_NUM = re.compile(r"\b(\d{1,6})\s+[A-Za-z]")
_WORD = re.compile(r"[a-z0-9]+")
_STOP = {"the", "of", "and", "suite", "ste", "floor", "fl", "building", "bldg", "unit", "facility",
         "site", "address", "phone", "id", "inc", "llc", "corp", "co"}


def site_text(side: dict) -> str:
    ev = side.get("evidence") if isinstance(side.get("evidence"), dict) else {}
    parts = [str(side.get("name") or "")] + [str(a) for a in side.get("aliases") or []] + [str(ev.get("text") or "")]
    return " | ".join(p for p in parts if p)[:600]


def _words(s: str) -> set[str]:
    return {w for w in _WORD.findall(s.lower()) if w not in _STOP and len(w) > 1}


def signals(a: str, b: str) -> list[float]:
    za, zb = set(_ZIP.findall(a)), set(_ZIP.findall(b))
    na, nb = set(_NUM.findall(a)), set(_NUM.findall(b))
    wa, wb = _words(a), _words(b)
    jac = len(wa & wb) / max(1, len(wa | wb))
    return [float(bool(za & zb)), float(bool(za) and bool(zb) and not (za & zb)),
            float(bool(na & nb)), float(bool(na) and bool(nb) and not (na & nb)), jac,
            float(a.strip().lower() == b.strip().lower())]


def cards(path: str) -> list[tuple[str, str, int]]:
    out = []
    for j in json.loads(Path(path).read_text(encoding="utf-8")).get("judgments") or []:
        t = j.get("target") if isinstance(j.get("target"), dict) else {}
        if j.get("head") == "site" and j.get("verdict") in ("same_site", "distinct_site") \
                and isinstance(t.get("a"), dict) and isinstance(t.get("b"), dict):
            out.append((site_text(t["a"]), site_text(t["b"]), int(j["verdict"] == "same_site")))
    return out


class SitePair(nn.Module):
    def __init__(self, d: int, n_sig: int, hidden: int = 64):
        super().__init__()
        self.proj = nn.Linear(d, hidden)
        self.out = nn.Sequential(nn.Linear(2 * hidden + n_sig, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, ea, eb, sig):
        pa, pb = self.proj(ea), self.proj(eb)
        return self.out(torch.cat([pa * pb, (pa - pb).abs(), sig], -1)).squeeze(-1)


def train(rows, *, epochs: int = 400, lr: float = 3e-3, wd: float = 1e-2) -> SitePair:
    torch.manual_seed(0)
    ea = torch.stack([r[0] for r in rows]); eb = torch.stack([r[1] for r in rows])
    sig = torch.stack([r[2] for r in rows]); y = torch.tensor([float(r[3]) for r in rows])
    m = SitePair(ea.shape[-1], sig.shape[-1])
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    pos = y.mean().clamp(0.05, 0.95)
    for _ in range(epochs):
        opt.zero_grad()
        # Both orders, so the pair is symmetric.
        loss = (F.binary_cross_entropy_with_logits(m(ea, eb, sig), y, pos_weight=(1 - pos) / pos)
                + F.binary_cross_entropy_with_logits(m(eb, ea, sig), y, pos_weight=(1 - pos) / pos))
        loss.backward()
        opt.step()
    return m.eval()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--deals", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from .predict import _read_autocast
    from .serve import _load

    model, _, device = _load()
    files = sorted({f for g in a.deals for f in glob.glob(g)} | {f for f in a.deals if Path(f).is_file()})
    by_deal = {Path(f).stem: cards(f) for f in files}
    by_deal = {d: c for d, c in by_deal.items() if c}
    texts = sorted({t for c in by_deal.values() for x in c for t in x[:2]})
    with torch.no_grad(), _read_autocast(device):
        emb = torch.cat([model.aux_text(texts[i:i + 64]).float().cpu() for i in range(0, len(texts), 64)])
    E = dict(zip(texts, emb))
    rows = {d: [(E[x[0]], E[x[1]], torch.tensor(signals(x[0], x[1])), x[2]) for x in c] for d, c in by_deal.items()}
    tp = fp = fn = tn = 0
    per = {}
    for held, rs in rows.items():
        rest = [r for d, x in rows.items() if d != held for r in x]
        m = train(rest)
        with torch.no_grad():
            p = torch.sigmoid(m(torch.stack([r[0] for r in rs]), torch.stack([r[1] for r in rs]),
                                torch.stack([r[2] for r in rs])))
        said = (p >= 0.5).tolist()
        y = [r[3] for r in rs]
        dtp = sum(1 for s, t in zip(said, y) if s and t); dfp = sum(1 for s, t in zip(said, y) if s and not t)
        dfn = sum(1 for s, t in zip(said, y) if not s and t); dtn = len(y) - dtp - dfp - dfn
        tp += dtp; fp += dfp; fn += dfn; tn += dtn
        per[held] = {"pairs": len(y), "same": sum(y), "caught": dtp, "false_merge": dfp}
    n = tp + fp + fn + tn
    report = {"pairs": n, "deals": len(rows), "same_site": tp + fn,
              "refit_lodo": {"right": round((tp + tn) / n, 3), "same_caught": f"{tp}/{tp + fn}",
                             "false_merges": fp, "merge_precision": round(tp / max(1, tp + fp), 3)},
              "parser": {"right": round((tn + fp + 0) / n, 3) if False else None},
              "per_deal": per}
    final = train([r for x in rows.values() for r in x])
    torch.save({"state": final.state_dict(), "d": emb.shape[-1], "n_sig": 6, "report": report}, a.out)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
