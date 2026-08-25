"""GO #5 -- train ONE backbone over the multitask table, emit per-task candidates.

The bet (see _BETS_LEDGER.md): one encoder, many heads. Every head that serves
through ``ContrastiveTypeKNN.load_promoted`` shares an embedding space, so a PM
correction harvested for one head moves the geometry under all of them.

One command on the GPU box::

    python -m app.learning.multitask_table          # freshen the table
    python runpod_detector/train_multitask_backbone.py

CPU proof of the loop (tiny slice, tiny steps, bge-small)::

    python runpod_detector/train_multitask_backbone.py --smoke

Design decisions, so the box run does not re-litigate them:

* **The split column governs. Nothing re-splits.** The table was split BY DEAL
  at assembly; re-splitting here would leak deals across the boundary and the
  number printed would be memorisation.
* **PM rows are eval-only.** ``teacher='pm'`` rows never enter training -- they
  are the only correctness signal we have, and a model that trained on them
  can only report agreement with itself.
* **No task prefix in the text.** Serving embeds the atom's bare decide-text;
  a ``[task]`` prefix trained in but never served is silent train/serve skew.
  Tasks are separated in the LOSS instead: SupCon labels are namespaced
  ``task::label``, so ``exclusion`` under atom_type and ``exclusion`` under a
  future task are different classes to the loss while the text pipeline stays
  byte-identical to production.
* **Epoch 0 is the baseline.** The frozen encoder is evaluated before any
  training step; every later epoch is reported as a delta against it, and the
  checkpoint that persists is the BEST epoch, never the last.
* **The promotion metric is the GO #5 metric:** value_light_answered on the
  atom-type holdout at >=0.95 accuracy-when-answering. The base encoder
  answers 7. Moving that number is the point; overall accuracy is context.
* **Ready is not promotion.** Candidates land in ``_backbone_candidate/<task>/``
  in the exact ``load_promoted`` layout; pointing the env flag at one stays a
  human act, and the meta carries the full eval + repr-version mix + caveats.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path

import sys

import numpy as np

# run as a file from anywhere: the repo root is the script's parent's parent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.learning.build_atom_store import VALUE_LIGHT, _BAD_LABELS

_STRICT_TAU = 0.98
_SIM_FLOOR = 0.55
_K = 15


# -- data (importable without torch; the trainer imports lazily) -------------

def load_rows(table_db: Path, tasks: tuple[str, ...] | None = None) -> dict:
    """Rows per task, honouring the stored split; PM rows diverted to eval."""
    conn = sqlite3.connect(table_db)
    sel = ("select task, text, label, deal_id, split, teacher, repr_version "
           "from multitask_rows")
    out: dict[str, dict[str, list]] = {}
    for task, text, label, deal, split, teacher, ver in conn.execute(sel):
        if tasks and task not in tasks:
            continue
        if not str(text or "").strip() or str(label) in _BAD_LABELS:
            continue
        slot = out.setdefault(task, {"train": [], "holdout": [], "pm": []})
        row = (str(text), str(label), str(deal or ""), int(ver or 0))
        if str(teacher) == "pm":
            slot["pm"].append(row)          # eval-only, whatever its split says
        elif split == "holdout":
            slot["holdout"].append(row)
        else:
            slot["train"].append(row)
    conn.close()
    return out


def namespaced(task: str, label: str) -> str:
    """Loss-space label: tasks share the encoder, never a class."""
    return f"{task}::{label}"


def knn_eval(embed, X, y, rows) -> dict:
    """The build_atom_store verdict, same discipline: strict tau, value-light split."""
    if not rows:
        return {"n": 0}
    Q = embed([t for t, *_ in rows])
    overall = {"answered": 0, "right": 0}
    light = {"answered": 0, "right": 0, "eligible": 0}
    answered_mask = []
    correct_mask = []
    for (text, lab, *_), q in zip(rows, Q):
        if lab in VALUE_LIGHT:
            light["eligible"] += 1
        sims = X @ q
        nb = np.argpartition(-sims, min(_K, len(sims)) - 1)[:_K]
        ans, right = False, False
        if float(sims[nb].max()) >= _SIM_FLOOR:
            votes: dict[str, float] = defaultdict(float)
            for j in nb:
                votes[str(y[j])] += max(float(sims[j]), 0.0)
            total = sum(votes.values()) or 1.0
            ranked = sorted(votes.items(), key=lambda kv: -kv[1])
            margin = (ranked[0][1]
                      - (ranked[1][1] if len(ranked) > 1 else 0.0)) / total
            if margin >= _STRICT_TAU:
                ans, pred = True, ranked[0][0]
                right = pred == lab
                overall["answered"] += 1
                overall["right"] += int(right)
                if pred in VALUE_LIGHT:
                    light["answered"] += 1
                    light["right"] += int(right)
        answered_mask.append(ans)
        correct_mask.append(right)
    return {
        "n": len(rows),
        "overall_answered": overall["answered"],
        "overall_accuracy": round(overall["right"] / overall["answered"], 4)
        if overall["answered"] else 0.0,
        "value_light_answered": light["answered"],
        "value_light_accuracy": round(light["right"] / light["answered"], 4)
        if light["answered"] else 0.0,
        "value_light_eligible": light["eligible"],
        "_answered": answered_mask,
        "_correct": correct_mask,
    }


def paired_vs_baseline(base_eval: dict, cand_eval: dict) -> dict:
    """McNemar over the shared holdout; p=None when no discordant pairs."""
    from app.eval.router_eval import _mcnemar_p
    b_ans, c_ans = base_eval.get("_answered", []), cand_eval.get("_answered", [])
    b_ok, c_ok = base_eval.get("_correct", []), cand_eval.get("_correct", [])
    wins = losses = 0
    for ba, ca, bo, co in zip(b_ans, c_ans, b_ok, c_ok):
        # abstention is never a hit: an unanswered row scores as wrong here,
        # so a candidate that ANSWERS MORE at the same accuracy earns wins.
        bscore, cscore = (ba and bo), (ca and co)
        if cscore and not bscore:
            wins += 1
        elif bscore and not cscore:
            losses += 1
    p = _mcnemar_p(wins, losses) if (wins + losses) else None
    return {"wins": wins, "losses": losses, "p": p}


# -- the trainer (GPU path; --smoke proves it on CPU) ------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="_multitask_table.db")
    ap.add_argument("--out", default="_backbone_candidate")
    ap.add_argument("--encoder", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--temp", type=float, default=0.07)
    ap.add_argument("--tasks", default="atom_type")
    ap.add_argument("--min-repr", type=int, default=0,
                    help="train only on rows with repr_version >= N "
                         "(2 = decide-text v2)")
    ap.add_argument("--smoke", action="store_true",
                    help="CPU proof: bge-small, ~400 rows, 1 short epoch")
    args = ap.parse_args()

    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    # transformers report_to="all" probes every logging integration; a broken
    # tensorflow install turns that probe into a crash. We log to stdout only.
    try:  # pragma: no cover - environment armor, not logic
        from transformers.integrations import integration_utils as _iu
        _iu.is_tensorboard_available = lambda: False
        _iu.is_wandb_available = lambda: False
        _iu.is_mlflow_available = lambda: False
    except Exception:
        pass

    from sentence_transformers import InputExample, SentenceTransformer
    from sentence_transformers.datasets import SentenceLabelDataset

    class SupConLoss(nn.Module):
        """Supervised contrastive loss over L2-normalized sentence embeddings.
        (Same formulation as train_contrastive_encoder_gpu -- the house loss.)"""

        def __init__(self, model, temperature=0.07):
            super().__init__(); self.model = model; self.t = temperature

        def forward(self, sentence_features, labels):
            z = self.model(sentence_features[0])["sentence_embedding"]
            z = torch.nn.functional.normalize(z, dim=1)
            B = z.shape[0]
            sim = (z @ z.T) / self.t
            sim = sim - sim.max(dim=1, keepdim=True)[0].detach()
            self_mask = torch.eye(B, device=z.device)
            lab = labels.view(-1, 1)
            pos = (lab == lab.T).float() - self_mask
            exp = torch.exp(sim) * (1 - self_mask)
            log_prob = sim - torch.log(exp.sum(1, keepdim=True) + 1e-12)
            pcount = pos.sum(1)
            mean_lp = (pos * log_prob).sum(1) / torch.clamp(pcount, min=1.0)
            valid = (pcount > 0).float()
            return -(mean_lp * valid).sum() / torch.clamp(valid.sum(), min=1.0)

    if args.smoke:
        args.encoder = "BAAI/bge-small-en-v1.5"
        args.epochs, args.batch = 1, 16

    tasks = tuple(t.strip() for t in args.tasks.split(",") if t.strip())
    data = load_rows(Path(args.table), tasks)
    if not data:
        raise SystemExit(f"no rows for tasks {tasks} in {args.table}")

    for task, slot in data.items():
        if args.min_repr:
            before = len(slot["train"])
            slot["train"] = [r for r in slot["train"] if r[3] >= args.min_repr]
            print(f"[{task}] --min-repr {args.min_repr}: "
                  f"train {before} -> {len(slot['train'])}")
        if args.smoke:
            slot["train"] = slot["train"][:400]
            slot["holdout"] = slot["holdout"][:120]
        vmix = Counter(f"v{r[3]}" for r in slot["train"])
        print(f"[{task}] train {len(slot['train'])} "
              f"| holdout {len(slot['holdout'])} "
              f"| pm-gold {len(slot['pm'])} | repr {dict(vmix)}")
        if vmix.get("v0", 0) > 0.5 * max(len(slot["train"]), 1):
            print(f"[{task}] WARNING: majority v0 bare text; serving is v2 "
                  "decide-text. The candidate meta will say so.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"encoder {args.encoder} on {device}")
    model = SentenceTransformer(args.encoder, device=device)

    def embed(texts):
        return np.asarray(model.encode(
            texts, batch_size=256, show_progress_bar=False,
            normalize_embeddings=True, convert_to_numpy=True), dtype=np.float32)

    def eval_all() -> dict:
        per = {}
        for task, slot in data.items():
            if not slot["train"] or not slot["holdout"]:
                per[task] = {"refused": "empty train or holdout side"}
                continue
            X = embed([t for t, *_ in slot["train"]])
            y = np.array([lab for _, lab, *_ in slot["train"]])
            per[task] = knn_eval(embed, X, y, slot["holdout"])
            if slot["pm"]:
                pm = knn_eval(embed, X, y, slot["pm"])
                per[task]["pm_gold"] = {k: v for k, v in pm.items()
                                        if not k.startswith("_")}
        return per

    def promotion_metric(per: dict) -> tuple[int, float]:
        at = per.get("atom_type", {})
        answered = at.get("value_light_answered", 0)
        acc = at.get("value_light_accuracy", 0.0)
        # answered counts ONLY while accuracy holds the 0.95 line
        return (answered if acc >= 0.95 else 0,
                at.get("overall_accuracy", 0.0))

    # loss-space examples: namespaced labels, bare production text
    lab2i: dict[str, int] = {}
    examples = []
    for task, slot in data.items():
        for text, lab, *_ in slot["train"]:
            ns = namespaced(task, lab)
            examples.append(InputExample(
                texts=[text], label=lab2i.setdefault(ns, len(lab2i))))
    print(f"{len(examples)} training examples across {len(lab2i)} loss classes")

    t0 = time.time()
    print("=== epoch 0 (frozen -- this IS the baseline) ===")
    baseline = eval_all()
    best_eval, best_metric, best_epoch = baseline, promotion_metric(baseline), 0
    at0 = baseline.get("atom_type", {})
    print(json.dumps({k: v for k, v in at0.items()
                      if not k.startswith("_")}, indent=2))

    ds = SentenceLabelDataset(examples, samples_per_label=2)
    loader = DataLoader(ds, batch_size=args.batch, drop_last=True)
    loss = SupConLoss(model, temperature=args.temp)

    out_root = Path(args.out)
    for ep in range(1, args.epochs + 1):
        model.fit(train_objectives=[(loader, loss)], epochs=1,
                  warmup_steps=int(0.06 * len(loader)),
                  show_progress_bar=not args.smoke,
                  optimizer_params={"lr": args.lr})
        per = eval_all()
        metric = promotion_metric(per)
        at = per.get("atom_type", {})
        paired = paired_vs_baseline(baseline.get("atom_type", {}), at)
        tag = "NEW BEST" if metric > best_metric else "below best"
        print(f"epoch {ep} | value_light {at.get('value_light_answered', 0)}"
              f"@{at.get('value_light_accuracy', 0.0):.3f}"
              f" (baseline {at0.get('value_light_answered', 0)}"
              f"@{at0.get('value_light_accuracy', 0.0):.3f})"
              f" | overall {at.get('overall_answered', 0)}"
              f"@{at.get('overall_accuracy', 0.0):.3f}"
              f" | vs-baseline wins {paired['wins']} losses {paired['losses']}"
              f" p={paired['p']} | {tag}", flush=True)
        if metric > best_metric:
            best_eval, best_metric, best_epoch = per, metric, ep
            _write_candidates(model, data, per, baseline, args, out_root, ep, t0)

    if best_epoch == 0:
        print("no epoch beat the frozen baseline -- NOTHING WRITTEN. "
              "The base encoder remains the incumbent; that is the honest "
              "result.")
    else:
        print(f"best epoch {best_epoch} persisted -> {out_root}/<task>/ "
              f"(store.npz, knn_meta.json, best/)")


def _write_candidates(model, data, per, baseline, args, out_root: Path,
                      epoch: int, t0: float) -> None:
    """Every task gets a load_promoted-layout dir sharing THIS encoder."""

    def embed(texts):
        return np.asarray(model.encode(
            texts, batch_size=256, show_progress_bar=False,
            normalize_embeddings=True, convert_to_numpy=True), dtype=np.float32)

    for task, slot in data.items():
        if not slot["train"]:
            continue
        d = out_root / task
        d.mkdir(parents=True, exist_ok=True)
        X = embed([t for t, *_ in slot["train"]])
        y = np.array([lab for _, lab, *_ in slot["train"]])
        texts = np.array([t for t, *_ in slot["train"]])
        np.savez_compressed(d / "store.npz", emb=X, y=y, text=texts)
        model.save(str(d / "best"))
        ev = {k: v for k, v in per.get(task, {}).items()
              if not k.startswith("_")}
        base = {k: v for k, v in baseline.get(task, {}).items()
                if not k.startswith("_")}
        vmix = Counter(f"v{r[3]}" for r in slot["train"])
        meta = {
            "mode": "typed",
            "k": _K, "sim_floor": _SIM_FLOOR, "tau": _STRICT_TAU,
            "encoder": args.encoder,
            "backbone": {"shared": True, "epoch": epoch,
                         "tasks": sorted(data.keys())},
            "value_light": sorted(VALUE_LIGHT),
            "n_train": len(slot["train"]),
            "n_holdout": len(slot["holdout"]),
            "eval": ev,
            "baseline_eval": base,
            "repr_versions": dict(vmix),
            "caveats": [
                "labels are TEACHER labels; accuracy here is agreement, not "
                "correctness (pm_gold in eval is the only correctness signal)",
                "shared-backbone candidate: promoting one task's dir moves "
                "the embedding space for that task only; promote together "
                "or not at all",
            ],
            "ready": bool(
                ev.get("value_light_answered", 0) >= 20
                and ev.get("value_light_accuracy", 0.0) >= 0.95
                and ev.get("value_light_answered", 0)
                > base.get("value_light_answered", 0)),
            "built_seconds": round(time.time() - t0, 1),
        }
        (d / "knn_meta.json").write_text(json.dumps(meta, indent=2),
                                         encoding="utf-8")


if __name__ == "__main__":
    main()
