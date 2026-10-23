"""Refit the Questions head from new verdicts, on a Mac, in minutes, no GPU.

    python -m ml.c3.question_fit --deals 'train/*.json' 010043.json \\
        [--feedback verdicts.jsonl] --out question_refit.pt

The base model stays frozen. Each deal is read once (the whole-deal pass the
brief already makes) and, per labeled question, three things are cached: the
deal's pooled Questions-head features, the question's text features, and the
frozen head's own logits. A small add-on is trained on top of the frozen
logits (zero at start, so before any training it answers exactly as the base
does). It sees the deal and the question together, which is how a verdict like
"this job needs a lift-access question, that one doesn't" becomes learnable
from a few hundred labels.

Labels: every unanchored Questions verdict in the deal files (``judgments``
with head ``gap`` and target kind headstart / checklist / generated_gap /
catalog, valid or invalid), plus ``--feedback`` rows
``{"deal_file": path, "text": question, "verdict": "valid"|"invalid"}`` (a
PM's keep or dismiss on a brief question).

Validation is leave-one-deal-out: each deal is scored by an add-on trained on
the others, so every reported number is on a deal the add-on never saw. The
saved add-on is trained on all deals. Counts and rates only, never line text.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

QUESTION_KEY = "jdg:gap"
UNANCHORED = ("headstart", "checklist", "generated_gap", "catalog")


class QuestionRefit(nn.Module):
    """Logit correction for the Questions head from (deal, question)."""

    def __init__(self, d: int, answers: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3 * d, hidden), nn.GELU(), nn.Linear(hidden, answers))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, deal: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([deal, query, deal * query], -1))


def file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stale_refit(refit_path: str, checkpoint_path: str) -> str:
    """Why an overlay must not be used with this base checkpoint ("" when it
    may): it carries no base md5 (fitted before the guard), or it was fitted
    on another file. The swap gate and serve both refuse a stale overlay."""
    saved = torch.load(refit_path, map_location="cpu", weights_only=False)
    base = saved.get("base_md5")
    if not base:
        return "the overlay records no base md5: refit it on this checkpoint"
    got = file_md5(checkpoint_path)
    if base != got:
        return f"fitted on base {base[:8]}, serving {got[:8]}: refit it on this checkpoint"
    return ""


def load_refit(path: str, device: str = "cpu") -> QuestionRefit:
    """The saved add-on, ready to correct the live Questions head."""
    saved = torch.load(path, map_location="cpu", weights_only=False)
    m = QuestionRefit(saved["d"], len(saved["answers"]))
    m.load_state_dict(saved["state"])
    m.bar = (saved.get("summary") or {}).get("bar")
    m.calibration = (saved.get("summary") or {}).get("calibration")
    return m.to(device).eval()


def _head_wording() -> dict[str, str]:
    """Card wording -> the wording the Questions head reads (c3_record catalog):
    a catalog card shows `display`, the head is asked `text`."""
    try:
        from c3_record.questions_catalog import catalog

        return {q.get("display") or q["text"]: q["text"] for q in catalog()}
    except Exception:
        return {}


def labels_of(path: str) -> list[tuple[str, str]]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    wording = _head_wording()
    out = []
    for j in d.get("judgments") or []:
        t = j.get("target") if isinstance(j.get("target"), dict) else {}
        if j.get("head") == "gap" and t.get("kind") in UNANCHORED and j.get("verdict") in ("valid", "invalid"):
            # A catalog card shows clean wording; `query` is what the head reads.
            text = str(t.get("query") or t.get("text") or "").strip()
            text = wording.get(text, text)
            if text:
                out.append((text, j["verdict"]))
    return out


@torch.no_grad()
def cache_deal(model, schema, device, path: str, questions: list[str], max_atoms: int):
    """(deal features [d], query features [Q, d], frozen logits [Q, A])."""
    from .loop import load_deal
    from .predict import _read_autocast, deal_parts

    model.eval()
    deal = load_deal(path)
    _, whole = deal_parts(deal, schema, max_atoms)
    every = tuple(range(len(whole)))
    # The same precision context the live read uses (predict.read_deal).
    with _read_autocast(device):
        out = model(whole.oneshot(), company=whole.company or None)
        pooled = model.judgment_features(out, QUESTION_KEY, [every], queries=[""]).float()[0]
        q = model.judgment_query(model.aux_text(questions)).float()
        frozen = torch.cat([model.judge(out, QUESTION_KEY, [every] * len(questions[i:i + 64]),
                                        queries=questions[i:i + 64]).float()
                            for i in range(0, len(questions), 64)])
    return pooled.cpu(), q.cpu(), frozen.cpu()


def _auc(p: list[float], y: list[int]) -> float | None:
    pos = [a for a, t in zip(p, y) if t]
    neg = [a for a, t in zip(p, y) if not t]
    if not pos or not neg:
        return None
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


def train(rows, answers: int, valid: int, *, epochs: int = 300, lr: float = 3e-3, wd: float = 1e-2,
          seed: int = 0) -> QuestionRefit:
    torch.manual_seed(seed)
    random.seed(seed)
    deal = torch.stack([r[0] for r in rows])
    query = torch.stack([r[1] for r in rows])
    frozen = torch.stack([r[2] for r in rows])
    y = torch.tensor([r[3] for r in rows])
    m = QuestionRefit(deal.shape[-1], answers)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    # Plain cross-entropy: the scores stay real probabilities, so how many
    # questions a deal gets is the head's call (a reweighted loss inflated
    # every score and asked 56 universal questions on 010043).
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(frozen + m(deal, query), y)
        loss.backward()
        opt.step()
    return m.eval()


def check_main(argv: list[str]) -> int:
    """``--check <refit.pt> --checkpoint <base.pt>``: exit 1 (and say why) when
    the overlay was not fitted on that base. For the swap gate."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", required=True)
    ap.add_argument("--checkpoint", required=True)
    a = ap.parse_args(argv)
    why = stale_refit(a.check, a.checkpoint)
    print(f"question refit {'STALE: ' + why if why else 'fitted on this base'}")
    return 1 if why else 0


def main() -> None:
    import sys

    if "--check" in sys.argv[1:]:
        raise SystemExit(check_main(sys.argv[1:]))
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--deals", nargs="+", required=True)
    ap.add_argument("--feedback", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-atoms", type=int, default=8192)
    ap.add_argument("--precision", type=float, default=0.68,
                    help="held-out precision the chosen bar must reach")
    a = ap.parse_args()

    from .serve import _load

    model, schema, device = _load()
    opp = schema.by_key()[QUESTION_KEY]
    names = [x.value for x in opp.answers]
    valid, invalid = names.index("valid"), names.index("invalid")
    files = sorted({f for g in a.deals for f in glob.glob(g)} | {f for f in a.deals if Path(f).is_file()})
    labeled = {f: labels_of(f) for f in files}
    if a.feedback:
        for line in Path(a.feedback).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                labeled.setdefault(r["deal_file"], []).append((r["text"].strip(), r["verdict"]))
    started = time.time()
    by_deal: dict[str, list] = {}
    for f, labs in labeled.items():
        if not labs:
            continue
        pooled, q, frozen = cache_deal(model, schema, device, f, [t for t, _ in labs], a.max_atoms)
        by_deal[f] = [(pooled, q[i], frozen[i], valid if v == "valid" else invalid)
                      for i, (_, v) in enumerate(labs)]
        print(f"read {Path(f).stem}: {len(labs)} verdicts, {time.time() - started:.0f}s", flush=True)

    # Leave one deal out: every score below is on a deal the add-on never saw.
    report, all_base, all_fit, all_y = {}, [], [], []
    for held in by_deal:
        rest = [r for f, rs in by_deal.items() if f != held for r in rs]
        m = train(rest, len(names), valid)
        rows = by_deal[held]
        with torch.no_grad():
            base = torch.stack([r[2] for r in rows]).softmax(-1)[:, valid].tolist()
            fit = (torch.stack([r[2] for r in rows]) + m(torch.stack([r[0] for r in rows]),
                                                        torch.stack([r[1] for r in rows]))).softmax(-1)[:, valid].tolist()
        y = [int(r[3] == valid) for r in rows]
        all_base += base; all_fit += fit; all_y += y
        report[Path(held).stem] = {"n": len(y), "valid": sum(y), "auc_base": _auc(base, y), "auc_refit": _auc(fit, y)}
    # Calibrate on the held-out scores: one temperature on the valid-vs-rest
    # log-odds, so a score of 0.7 is right about 70% of the time.
    import math
    lo = torch.tensor([math.log(max(p, 1e-6) / max(1 - p, 1e-6)) for p in all_fit])
    yt = torch.tensor(all_y, dtype=torch.float)
    temp = torch.zeros(2, requires_grad=True)  # log-scale, shift
    opt_t = torch.optim.LBFGS([temp], max_iter=200)

    def _closure():
        opt_t.zero_grad()
        loss = F.binary_cross_entropy_with_logits(lo * temp[0].exp() + temp[1], yt)
        loss.backward()
        return loss
    opt_t.step(_closure)
    scale, shift = float(temp[0].exp()), float(temp[1])
    all_fit = torch.sigmoid(lo * scale + shift).tolist()
    # The bar the brief asks at, from held-out scores: the lowest bar whose
    # held-out precision is at least --precision (more questions, still right).
    bars = {}
    for bar in [x / 100 for x in range(5, 100, 5)]:
        said = [y for p, y in zip(all_fit, all_y) if p >= bar]
        bars[bar] = {"asked": len(said), "precision": round(sum(said) / max(1, len(said)), 3),
                     "recall": round(sum(said) / max(1, sum(all_y)), 3)}
    good = [b for b, v in bars.items() if v["precision"] >= a.precision and v["asked"] >= 10]
    summary_bar = 0.5  # calibrated: ask when the head says more likely needed than not
    summary = {"deals": len(by_deal), "verdicts": len(all_y), "valid": sum(all_y),
               "bar": summary_bar, "bars": bars, "calibration": {"scale": scale, "shift": shift},
               "auc_base_pooled": _auc(all_base, all_y), "auc_refit_pooled": _auc(all_fit, all_y),
               "per_deal": report, "read_s": round(time.time() - started)}
    final = train([r for rs in by_deal.values() for r in rs], len(names), valid)
    torch.save({"state": final.state_dict(), "d": final.net[0].in_features // 3, "answers": names,
                "key": QUESTION_KEY, "summary": summary,
                # The base this overlay corrects (stale_refit): serve and the
                # swap gate refuse it on any other checkpoint.
                "base_md5": file_md5(os.environ["SOWSMITH_C3_CHECKPOINT"].strip())}, a.out)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("per_deal",)}, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
