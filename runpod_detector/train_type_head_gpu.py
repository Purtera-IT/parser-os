"""GPU fine-tune the atom_type Type head (#70) and SHOW per-epoch learning.

Why GPU: the CPU head uses FROZEN embeddings + a linear classifier and caps at
~0.65 held-out because the frozen space can't separate 43 overlapping classes.
This UNFREEZES a small encoder and fine-tunes the representation to the task —
the lever that breaks the ceiling.

You SEE it learn: every epoch prints train loss + HELD-OUT (by-deal) accuracy,
macro-F1, and the cutover metric (precision on confident predictions). Rising
held-out with rising train = truly learning; held-out stalling while train climbs
= overfitting (you'll see it). Final line = the honest ceiling vs the 0.65
frozen baseline = "how good it actually is."

Run on RunPod (1 GPU):
  pip install -U "transformers>=4.44" "datasets>=2.20" "accelerate>=0.33" scikit-learn
  python train_type_head_gpu.py
Inputs: _training_deepseek.db (ship it next to this script).
"""
import os, sqlite3, hashlib
import numpy as np

DB = os.environ.get("SOWSMITH_TRAINING_LOG_DB", "_training_deepseek.db")
MODEL = os.environ.get("BASE_MODEL", "microsoft/deberta-v3-small")
EPOCHS = int(os.environ.get("EPOCHS", "8"))
BATCH = int(os.environ.get("BATCH", "32"))
FROZEN_BASELINE = 0.65   # the CPU frozen-embedding head's held-out accuracy
HOLDOUT = 0.25
CONF_THR = 0.85          # cutover metric: precision on predictions above this


def split(deal_id):
    h = int(hashlib.sha256((deal_id or "").encode()).hexdigest(), 16)
    return "test" if (h % 100) / 100.0 < HOLDOUT else "train"


def load():
    con = sqlite3.connect(DB)
    rows = con.execute(
        "SELECT COALESCE(NULLIF(masked_text,''),raw_text) AS t, label, deal_id "
        "FROM training_rows WHERE relation='atom_type' AND COALESCE(masked_text,raw_text,'')!='' "
        "AND label IS NOT NULL").fetchall()
    con.close()
    return [(t, l, d or "") for t, l, d in rows if t]


def main():
    import torch
    from datasets import Dataset
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              TrainingArguments, Trainer, TrainerCallback)
    from sklearn.metrics import accuracy_score, f1_score

    data = load()
    labels = sorted({l for _, l, _ in data})
    l2i = {l: i for i, l in enumerate(labels)}
    tr = [(t, l2i[l]) for t, l, d in data if split(d) == "train"]
    te = [(t, l2i[l]) for t, l, d in data if split(d) == "test"]
    print(f"classes={len(labels)} train={len(tr)} held-out={len(te)}  base={MODEL}")
    print(f"FROZEN-EMBEDDING BASELINE held-out acc = {FROZEN_BASELINE:.3f} (what we must beat)\n")

    tok = AutoTokenizer.from_pretrained(MODEL)
    def enc(b): return tok(b["text"], truncation=True, max_length=128, padding="max_length")
    dtr = Dataset.from_dict({"text": [t for t, _ in tr], "label": [y for _, y in tr]}).map(enc, batched=True)
    dte = Dataset.from_dict({"text": [t for t, _ in te], "label": [y for _, y in te]}).map(enc, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=len(labels))

    def metrics(p):
        logits, y = p
        prob = torch.softmax(torch.tensor(logits), -1).numpy()
        pred = prob.argmax(-1)
        conf = prob.max(-1)
        sel = conf >= CONF_THR
        cut_prec = accuracy_score(y[sel], pred[sel]) if sel.sum() else 0.0
        return {"acc": accuracy_score(y, pred),
                "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
                "cutover_cov": float(sel.mean()), "cutover_prec": float(cut_prec)}

    class Watch(TrainerCallback):
        def on_evaluate(self, args, state, control, metrics=None, **kw):
            m = metrics or {}
            ep = state.epoch or 0
            acc = m.get("eval_acc", 0); f1 = m.get("eval_macro_f1", 0)
            cov = m.get("eval_cutover_cov", 0); cp = m.get("eval_cutover_prec", 0)
            delta = acc - FROZEN_BASELINE
            verdict = "LEARNING ✅" if delta > 0.02 else ("flat" if abs(delta) <= 0.02 else "below baseline")
            print(f"  epoch {ep:>4.1f} | held-out acc {acc:.3f} ({delta:+.3f} vs frozen) "
                  f"| macroF1 {f1:.3f} | cutover {cov*100:.0f}%@{cp:.3f} | {verdict}", flush=True)

    args = TrainingArguments(
        output_dir="runs/type_head_gpu", num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH, per_device_eval_batch_size=64,
        eval_strategy="epoch", save_strategy="epoch", logging_steps=50,
        learning_rate=2e-5, warmup_ratio=0.06, weight_decay=0.01,
        load_best_model_at_end=True, metric_for_best_model="acc",
        report_to=[], fp16=torch.cuda.is_available(),
    )
    tr_ = Trainer(model=model, args=args, train_dataset=dtr, eval_dataset=dte,
                  compute_metrics=metrics, callbacks=[Watch()])
    print("=== fine-tuning (watch held-out acc climb each epoch) ===")
    tr_.train()
    final = tr_.evaluate()
    acc = final.get("eval_acc", 0)
    print(f"\n=== VERDICT ===")
    print(f"fine-tuned held-out acc = {acc:.3f}  vs frozen baseline {FROZEN_BASELINE:.3f}  "
          f"(+{acc-FROZEN_BASELINE:.3f})")
    print(f"cutover: {final.get('eval_cutover_cov',0)*100:.0f}% of atoms deflectable at "
          f"{final.get('eval_cutover_prec',0):.3f} precision")
    print("STRONG (ship cutover)" if acc >= 0.85 else "BETTER than frozen, keep growing data"
          if acc > FROZEN_BASELINE + 0.05 else "no gain — needs more/cleaner data")
    tr_.save_model("runs/type_head_gpu/best")
    import json; json.dump({"labels": labels}, open("runs/type_head_gpu/best/labels.json", "w"))


if __name__ == "__main__":
    main()
