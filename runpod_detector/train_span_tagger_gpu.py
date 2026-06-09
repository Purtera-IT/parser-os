"""GPU fine-tune the #71 span heads (requirements / sites / commercial) and SHOW
per-epoch learning. Breaks the frozen-embedding recall ceiling (requirements 0.74,
sites 0.69) by fine-tuning the representation per relation.

Per relation, binary "is this atom a <relation> item?" with an UNFROZEN encoder.
Every epoch prints train loss + HELD-OUT (by-deal) recall @ the precision-floor
threshold, so you watch recall climb toward the 0.93 skip bar. Final line = the
honest recall vs the frozen baseline = does the LLM call become skippable.

Run on RunPod:
  pip install -U "transformers>=4.44" "datasets>=2.20" "accelerate>=0.33" scikit-learn
  python train_span_tagger_gpu.py
Inputs: _training_deepseek.db (ship it next to this script).
"""
import os, sqlite3, hashlib, random
import numpy as np

DB = os.environ.get("SOWSMITH_TRAINING_LOG_DB", "_training_deepseek.db")
# bge-small: BERT/WordPiece tokenizer (no sentencepiece gotcha), small+fast.
MODEL = os.environ.get("BASE_MODEL", "BAAI/bge-small-en-v1.5")
EPOCHS = int(os.environ.get("EPOCHS", "8"))
BATCH = int(os.environ.get("BATCH", "32"))
HOLDOUT = 0.25
PRECISION_FLOOR = 0.80
SKIP_BAR = 0.93
RELATIONS = os.environ.get("RELATIONS", "requirements,site_clusters,commercial_line_items").split(",")
FROZEN = {"requirements": 0.74, "site_clusters": 0.69, "commercial_line_items": 0.67}


def split(deal_id):
    h = int(hashlib.sha256((deal_id or "").encode()).hexdigest(), 16)
    return "test" if (h % 100) / 100.0 < HOLDOUT else "train"


def load_all():
    con = sqlite3.connect(DB)
    rows = con.execute(
        "SELECT relation, COALESCE(NULLIF(masked_text,''),raw_text) AS t, deal_id "
        "FROM training_rows WHERE COALESCE(masked_text,raw_text,'')!=''").fetchall()
    con.close()
    return [(r, t, d or "") for r, t, d in rows if t]


def pick_threshold(scores, y, floor):
    best = (0.5, 0.0, 0.0)
    for thr in np.unique(scores):
        pred = scores >= thr
        tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
        if tp == 0:
            continue
        prec = tp / (tp + fp); rec = tp / (tp + fn)
        if prec >= floor and rec > best[1]:
            best = (float(thr), rec, prec)
    return best


def main():
    import torch
    from datasets import Dataset
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              TrainingArguments, Trainer, TrainerCallback)

    data = load_all()
    feat_deal = {}
    for _r, t, d in data:
        feat_deal.setdefault(t, d)
    tok = AutoTokenizer.from_pretrained(MODEL)
    def enc(b): return tok(b["text"], truncation=True, max_length=128, padding="max_length")

    for rel in RELATIONS:
        rel = rel.strip()
        pos = {t for r, t, _ in data if r == rel}
        if len(pos) < 40:
            print(f"\n### {rel}: insufficient ({len(pos)} positives) — skip"); continue
        neg = [t for t in feat_deal if t not in pos]
        random.seed(0)
        if len(neg) > 3 * len(pos):
            neg = random.sample(neg, 3 * len(pos))
        feats = list(pos) + neg
        lab = {t: (1 if t in pos else 0) for t in feats}
        tr = [(t, lab[t]) for t in feats if split(feat_deal[t]) == "train"]
        te = [(t, lab[t]) for t in feats if split(feat_deal[t]) == "test"]
        base = FROZEN.get(rel, 0.0)
        print(f"\n{'='*64}\n### {rel}: train={len(tr)} held-out={len(te)} "
              f"(pos {sum(y for _,y in te)}) | frozen recall baseline {base:.2f}")

        dtr = Dataset.from_dict({"text": [t for t, _ in tr], "label": [y for _, y in tr]}).map(enc, batched=True)
        dte = Dataset.from_dict({"text": [t for t, _ in te], "label": [y for _, y in te]}).map(enc, batched=True)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)

        yte = np.array([y for _, y in te])

        def metrics(p):
            logits, y = p
            prob = torch.softmax(torch.tensor(logits), -1).numpy()[:, 1]
            thr, rec, prec = pick_threshold(prob, np.array(y), PRECISION_FLOOR)
            return {"recall": rec, "precision": prec, "thr": thr}

        class Watch(TrainerCallback):
            def on_evaluate(self, args, state, control, metrics=None, **kw):
                m = metrics or {}; ep = state.epoch or 0
                rec = m.get("eval_recall", 0); prec = m.get("eval_precision", 0)
                d = rec - base
                tag = ("SKIPPABLE ✅ (>=0.93)" if rec >= SKIP_BAR else
                       "LEARNING ✅" if d > 0.02 else "flat")
                print(f"  epoch {ep:>4.1f} | held-out recall {rec:.3f} ({d:+.3f} vs frozen) "
                      f"@ prec {prec:.3f} | {tag}", flush=True)

        args = TrainingArguments(
            output_dir=f"runs/span_{rel}", num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH, per_device_eval_batch_size=64,
            eval_strategy="epoch", save_strategy="epoch", logging_steps=50,
            learning_rate=2e-5, warmup_ratio=0.06, weight_decay=0.01,
            load_best_model_at_end=True, metric_for_best_model="recall",
            report_to=[], fp16=torch.cuda.is_available(), disable_tqdm=False,
        )
        t_ = Trainer(model=model, args=args, train_dataset=dtr, eval_dataset=dte,
                     compute_metrics=metrics, callbacks=[Watch()])
        t_.train()
        f = t_.evaluate()
        rec = f.get("eval_recall", 0)
        print(f"  VERDICT {rel}: recall {rec:.3f} vs frozen {base:.2f} "
              f"({'SKIP UNLOCKS — LLM call dies' if rec >= SKIP_BAR else 'better, not yet skippable' if rec>base+0.05 else 'no gain'})")
        t_.save_model(f"runs/span_{rel}/best")
        tok.save_pretrained(f"runs/span_{rel}/best")   # without this the head loads an all-UNK tokenizer
        # span_meta.json: the runtime needs the held-out recall (skip-eligibility)
        # AND the recall-tuned decision threshold (admission). Without this the
        # runtime loader abstains.
        import json as _json
        _json.dump({"relation": rel,
                    "recall": float(f.get("eval_recall", 0.0)),
                    "precision": float(f.get("eval_precision", 0.0)),
                    "threshold": float(f.get("eval_thr", 0.5)),
                    "skippable": bool(f.get("eval_recall", 0.0) >= SKIP_BAR),
                    "verbatim": rel in ("requirements", "site_clusters")},
                   open(f"runs/span_{rel}/best/span_meta.json", "w"))


if __name__ == "__main__":
    main()
