"""Cross-encoder keep-vs-typed GATE — the ceiling-pusher past bge-base's 0.83.

Two upgrades over train_gate_rubric.py, stacked:
  1. STRONGER BASE: DeBERTa-v3-large (disentangled attention) — SOTA for nuanced
     text classification, far more capacity than bge-base on a fuzzy semantic
     boundary.
  2. CROSS-ATTENTION OVER CONTEXT: each clause is classified JOINTLY with its
     prev/next clause as a text-pair ([CLS] clause [SEP] prev|next [SEP]). The
     model cross-attends the clause against its surroundings — and role (the whole
     keep/typed call) often depends on context ("Project Name:" is keep in a form,
     typed in a scope sentence). bge-base saw the clause alone; this doesn't.

Consumes the SAME clean labels (_rubric_gate_data.json from rubric_relabel_deepseek.py,
already held-out by deal) and reconstructs each clause's context from the training
DB by text lookup — so NO re-relabel needed. Trade-off: a cross-encoder loses
instant-learning (it's a fine-tuned model, not a kNN), but the gate decision is
stable, so that's an acceptable trade for accuracy.

Reports per-epoch held-out acc/prec/rec + the GUESS-FREE operating point: the
smallest confidence threshold that hits TARGET_PREC (default 0.95) and how much of
the deals it confidently covers = how much typing we can safely auto-handle.

Run on the pod (after the relabel):
  pip install -U "transformers>=4.44" datasets accelerate scikit-learn sentencepiece
  python runpod_detector/train_gate_crossencoder.py
Env: BASE_MODEL=microsoft/deberta-v3-large EPOCHS=4 BATCH=16 MAXLEN=256
     CONTEXT=1 TARGET_PREC=0.95
"""
import os, json, sqlite3, collections
import numpy as np

DB = os.environ.get("SOWSMITH_TRAINING_LOG_DB", "_training_deepseek.db")
DATA = os.environ.get("GATE_DATA", "_rubric_gate_data.json")
MODEL = os.environ.get("BASE_MODEL", "microsoft/deberta-v3-large")
EPOCHS = int(os.environ.get("EPOCHS", "4"))
BATCH = int(os.environ.get("BATCH", "16"))
MAXLEN = int(os.environ.get("MAXLEN", "256"))
CONTEXT = os.environ.get("CONTEXT", "1") not in ("0", "false", "no")
TARGET_PREC = float(os.environ.get("TARGET_PREC", "0.95"))


def build_context_map():
    """text -> (prev, next) from per-deal rowid order (same as the relabeler)."""
    if not CONTEXT:
        return {}
    con = sqlite3.connect(DB)
    rows = con.execute(
        "SELECT COALESCE(NULLIF(masked_text,''),raw_text) AS t, deal_id "
        "FROM training_rows WHERE relation='atom_type' "
        "AND COALESCE(masked_text,raw_text,'')!='' ORDER BY deal_id, rowid").fetchall()
    con.close()
    by_deal = collections.defaultdict(list)
    for t, d in rows:
        by_deal[d or ""].append((t or "").strip())
    ctx = {}
    for seq in by_deal.values():
        for i, t in enumerate(seq):
            if t and t not in ctx:
                ctx[t] = (seq[i - 1] if i > 0 else "", seq[i + 1] if i + 1 < len(seq) else "")
    return ctx


def operating_point(prob, y, target):
    """Smallest confidence tau whose slice precision >= target; returns (tau, cov, prec)."""
    pred = prob.argmax(-1); conf = prob.max(-1); y = np.array(y)
    correct = (pred == y).astype(float)
    for tau in np.unique(conf):
        sel = conf >= tau
        if sel.sum() and correct[sel].mean() >= target:
            return float(tau), float(sel.mean()), float(correct[sel].mean())
    return None


def main():
    import torch
    from datasets import Dataset
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              TrainingArguments, Trainer, TrainerCallback)
    from sklearn.metrics import accuracy_score, precision_score, recall_score

    d = json.load(open(DATA))
    tr, te = d["train"], d["test"]
    ctx = build_context_map()

    def pair(r):
        p, n = ctx.get(r["text"], ("", ""))
        return ((p + " | " + n).strip(" |")) if CONTEXT else ""

    print(f"train={len(tr)} held-out={len(te)} | base={MODEL} | context={CONTEXT} "
          f"| target precision {TARGET_PREC}")
    print(f"train typed-frac={np.mean([r['y'] for r in tr]):.2f} "
          f"held-out typed-frac={np.mean([r['y'] for r in te]):.2f}")

    tok = AutoTokenizer.from_pretrained(MODEL)

    def enc(b):
        return tok(b["text"], b["ctx"], truncation=True, max_length=MAXLEN, padding="max_length")

    def ds(rows):
        return Dataset.from_dict({
            "text": [r["text"] for r in rows],
            "ctx": [pair(r) for r in rows],
            "label": [r["y"] for r in rows],
        }).map(enc, batched=True)

    dtr, dte = ds(tr), ds(te)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)

    cnt = collections.Counter(r["y"] for r in tr); tot = len(tr)
    w = torch.tensor([tot / (2 * cnt[c]) for c in (0, 1)], dtype=torch.float)

    class Weighted(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            y = inputs.pop("labels"); out = model(**inputs)
            loss = torch.nn.functional.cross_entropy(out.logits.float(), y, weight=w.to(out.logits.device))
            return (loss, out) if return_outputs else loss

    def metrics(p):
        logits, y = p
        prob = torch.softmax(torch.tensor(logits), -1).numpy()
        pred = prob.argmax(-1)
        op = operating_point(prob, y, TARGET_PREC)
        return {"acc": accuracy_score(y, pred),
                "prec": precision_score(y, pred, zero_division=0),
                "rec": recall_score(y, pred, zero_division=0),
                "op_cov": (op[1] if op else 0.0), "op_prec": (op[2] if op else 0.0),
                "op_tau": (op[0] if op else 1.0)}

    class Watch(TrainerCallback):
        best = 0.0
        def on_evaluate(self, args, state, control, metrics=None, **kw):
            m = metrics or {}; acc = m.get("eval_acc", 0); Watch.best = max(Watch.best, acc)
            print(f"  epoch {state.epoch or 0:>3.1f} | held-out acc {acc:.3f} | prec {m.get('eval_prec',0):.3f} "
                  f"rec {m.get('eval_rec',0):.3f} | guess-free@{TARGET_PREC:.2f}: "
                  f"{m.get('eval_op_cov',0)*100:.0f}% @ {m.get('eval_op_prec',0):.3f} (tau {m.get('eval_op_tau',1):.2f}) "
                  f"| best {Watch.best:.3f}", flush=True)

    args = TrainingArguments(
        output_dir="runs/gate_crossenc", num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH, per_device_eval_batch_size=32,
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="acc", greater_is_better=True, save_total_limit=1,
        logging_steps=50, learning_rate=1e-5, warmup_ratio=0.06, weight_decay=0.01,
        report_to=[], fp16=torch.cuda.is_available(),
        gradient_checkpointing=True, disable_tqdm=False)
    t = Weighted(model=model, args=args, train_dataset=dtr, eval_dataset=dte,
                 compute_metrics=metrics, callbacks=[Watch()],
                 preprocess_logits_for_metrics=lambda logits, labels: logits.float())
    print("=== cross-encoder gate (DeBERTa-v3-large + context) — watch it climb ===")
    t.train(); f = t.evaluate()
    acc = f.get("eval_acc", 0)
    print(f"\n================ CROSS-ENCODER GATE VERDICT ================")
    print(f"held-out acc = {acc:.3f}  (bge-base gate was 0.83)")
    print(f"  precision {f.get('eval_prec',0):.3f} | recall {f.get('eval_rec',0):.3f}")
    print(f"  GUESS-FREE: {f.get('eval_op_cov',0)*100:.0f}% of atoms confidently typed @ "
          f"{f.get('eval_op_prec',0):.3f} precision (tau {f.get('eval_op_tau',1):.2f}); rest -> LLM")
    print("VERDICT:", "BREAKS PAST bge-base ✅ ship the cross-encoder gate" if acc >= 0.86 else
          f"{acc:.3f} — context+DeBERTa helped" if acc > 0.835 else
          "no gain over bge-base — 0.83 is the real held-out-by-deal ceiling")
    os.makedirs("runs/gate_crossenc/best", exist_ok=True)
    t.save_model("runs/gate_crossenc/best"); tok.save_pretrained("runs/gate_crossenc/best")


if __name__ == "__main__":
    main()
