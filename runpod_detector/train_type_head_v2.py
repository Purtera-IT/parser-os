"""Type-head v2 — break the 0.65 ceiling by attacking WHERE the error is.

Diagnosis (measured on the v1 single-stage run): held-out acc 0.573, and **70% of
errors touch the `_keep` boundary** (44% real-types dumped into _keep, 26% the
reverse) — the teacher draws the noise<->typed line inconsistently, and one flat
41-way head can't learn a boundary the labels don't draw. Plus the model is biased
to _keep (0.80 acc on _keep rows vs 0.31 on typed rows).

v2 changes:
  1. TWO-STAGE. Stage A = binary "is this atom worth typing?" (_keep vs typed),
     class-weighted to fix the keep-bias. Stage B = fine-grained type, trained
     ONLY on typed rows (no _keep noise). Decouples the noisy boundary from the
     clean type confusion.
  2. CLASS WEIGHTING on both stages (inverse-frequency) — counter the 54% _keep
     majority and the long-tail types.
  3. Configurable encoder + optional LoRA (USE_LORA=1) so the SAME code runs
     bge-small full-FT on a Mac or LoRA-on-a-7-8B-encoder on an A100. The v1 run
     lost to the frozen baseline because bge-small (33M) can't beat frozen
     qwen3-embedding:8b (8B) features — a bigger encoder is the real lever.
  4. Saves tokenizer with each stage (v1 didn't → heads loaded an all-UNK
     tokenizer and degenerated).

Eval is honest: held-out BY DEAL (never-trained deals). Reports end-to-end acc
(route through A then B), each stage alone, and the confident-slice cutover,
all vs the 0.65 frozen baseline and the 0.573 v1 single-stage.

Run:  python runpod_detector/train_type_head_v2.py
Env:  BASE_MODEL (default BAAI/bge-small-en-v1.5), USE_LORA=0/1, EPOCHS=6,
      BATCH=32, HOLDOUT=0.25, CONF_THR=0.85
"""
import os, sqlite3, hashlib, json, collections
import numpy as np

DB = os.environ.get("SOWSMITH_TRAINING_LOG_DB", "_training_deepseek.db")
MODEL = os.environ.get("BASE_MODEL", "BAAI/bge-small-en-v1.5")
USE_LORA = os.environ.get("USE_LORA", "0") == "1"
EPOCHS = int(os.environ.get("EPOCHS", "6"))
BATCH = int(os.environ.get("BATCH", "32"))
HOLDOUT = float(os.environ.get("HOLDOUT", "0.25"))
CONF_THR = float(os.environ.get("CONF_THR", "0.85"))
FROZEN_BASELINE = 0.65
V1_SINGLE_STAGE = 0.573
OUT = os.environ.get("OUT_DIR", "runs/type_head_v2")


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


def lora_targets(model):
    mt = getattr(model.config, "model_type", "")
    if mt in ("bert", "roberta", "xlm-roberta"):
        return ["query", "key", "value"]
    return ["q_proj", "k_proj", "v_proj", "o_proj"]   # qwen/llama-family


def fit(texts, ys, n_labels, weights, tag, out_dir):
    """Train one stage with an 80/20 (by index) inner split for early stopping."""
    import torch
    from datasets import Dataset
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              TrainingArguments, Trainer, TrainerCallback)
    from sklearn.metrics import accuracy_score, f1_score

    tok = AutoTokenizer.from_pretrained(MODEL)
    def enc(b): return tok(b["text"], truncation=True, max_length=128, padding="max_length")

    # deterministic inner val split
    rng = np.random.RandomState(0)
    order = rng.permutation(len(texts))
    n_val = max(1, int(0.15 * len(texts)))
    val_i, tr_i = set(order[:n_val].tolist()), order[n_val:].tolist()
    dtr = Dataset.from_dict({"text": [texts[i] for i in tr_i], "label": [ys[i] for i in tr_i]}).map(enc, batched=True)
    dva = Dataset.from_dict({"text": [texts[i] for i in order[:n_val]], "label": [ys[i] for i in order[:n_val]]}).map(enc, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=n_labels)
    if USE_LORA:
        from peft import LoraConfig, get_peft_model, TaskType
        cfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32,
                         lora_dropout=0.1, target_modules=lora_targets(model))
        model = get_peft_model(model, cfg); model.print_trainable_parameters()

    w = torch.tensor(weights, dtype=torch.float)

    class Weighted(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels")
            out = model(**inputs)
            loss = torch.nn.functional.cross_entropy(out.logits, labels, weight=w.to(out.logits.device))
            return (loss, out) if return_outputs else loss

    def metrics(p):
        logits, y = p
        pred = np.asarray(logits).argmax(-1)
        return {"acc": accuracy_score(y, pred),
                "macro_f1": f1_score(y, pred, average="macro", zero_division=0)}

    class Watch(TrainerCallback):
        best = 0.0
        def on_evaluate(self, args, state, control, metrics=None, **kw):
            m = metrics or {}; acc = m.get("eval_acc", 0); f1 = m.get("eval_macro_f1", 0)
            Watch.best = max(Watch.best, acc)
            print(f"  [{tag}] epoch {state.epoch or 0:>3.1f} | val acc {acc:.3f} | macroF1 {f1:.3f} | best {Watch.best:.3f}", flush=True)

    args = TrainingArguments(
        output_dir=out_dir, num_train_epochs=EPOCHS, per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=64, eval_strategy="epoch", save_strategy="epoch",
        logging_steps=50, learning_rate=(1e-4 if USE_LORA else 2e-5), warmup_ratio=0.06,
        weight_decay=0.01, load_best_model_at_end=True, metric_for_best_model="acc",
        report_to=[], fp16=torch.cuda.is_available(), disable_tqdm=False)
    tr = Weighted(model=model, args=args, train_dataset=dtr, eval_dataset=dva,
                  compute_metrics=metrics, callbacks=[Watch()])
    print(f"\n=== training stage [{tag}]  (LoRA={USE_LORA}, base={MODEL}) ===")
    tr.train()
    os.makedirs(f"{out_dir}/best", exist_ok=True)
    tr.save_model(f"{out_dir}/best"); tok.save_pretrained(f"{out_dir}/best")
    return tr, tok


def predict(tr, tok, texts):
    import torch
    mdl = tr.model.eval(); dev = next(mdl.parameters()).device
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            enc = tok(texts[i:i+64], truncation=True, max_length=128, padding=True, return_tensors="pt").to(dev)
            pr = torch.softmax(mdl(**enc).logits, -1).cpu().numpy()
            out.append(pr)
    return np.concatenate(out, 0)


def main():
    from sklearn.metrics import accuracy_score
    data = load()
    tr = [(t, l) for t, l, d in data if split(d) == "train"]
    te = [(t, l) for t, l, d in data if split(d) == "test"]
    typed_labels = sorted({l for _, l in tr if l != "_keep"})
    t2i = {l: i for i, l in enumerate(typed_labels)}
    print(f"train={len(tr)} held-out={len(te)} | typed classes={len(typed_labels)} "
          f"| base={MODEL} LoRA={USE_LORA}")
    print(f"baselines: frozen={FROZEN_BASELINE:.3f}  v1-single-stage={V1_SINGLE_STAGE:.3f}\n")

    # ---- Stage A: _keep(0) vs typed(1), class-weighted ----
    a_txt = [t for t, _ in tr]; a_y = [0 if l == "_keep" else 1 for _, l in tr]
    cnt = collections.Counter(a_y); tot = len(a_y)
    a_w = [tot / (2 * cnt[c]) for c in (0, 1)]
    trA, tokA = fit(a_txt, a_y, 2, a_w, "A:keep-vs-typed", f"{OUT}/stageA")

    # ---- Stage B: fine-grained type on typed rows only, class-weighted ----
    b_rows = [(t, l) for t, l in tr if l != "_keep"]
    b_txt = [t for t, _ in b_rows]; b_y = [t2i[l] for _, l in b_rows]
    cntb = collections.Counter(b_y); k = len(typed_labels)
    b_w = [len(b_y) / (k * cntb.get(i, 1)) for i in range(k)]
    trB, tokB = fit(b_txt, b_y, k, b_w, "B:fine-type", f"{OUT}/stageB")

    # ---- End-to-end held-out eval ----
    te_txt = [t for t, _ in te]; te_lab = [l for _, l in te]
    pa = predict(trA, tokA, te_txt)             # P(typed)
    pb = predict(trB, tokB, te_txt)             # type dist
    a_pred = pa.argmax(-1); a_conf = pa.max(-1)
    b_pred = pb.argmax(-1); b_conf = pb.max(-1)
    i2t = {i: l for l, i in t2i.items()}

    final = []
    for j in range(len(te)):
        final.append("_keep" if a_pred[j] == 0 else i2t[b_pred[j]])
    acc = accuracy_score(te_lab, final)

    # stage diagnostics
    a_true = [0 if l == "_keep" else 1 for l in te_lab]
    accA = np.mean(a_pred == np.array(a_true))
    typed_mask = np.array(a_true) == 1
    # stage B accuracy on rows that are truly typed (routing aside)
    btrue = [t2i.get(l, -1) for l in te_lab]
    accB = np.mean([b_pred[j] == btrue[j] for j in range(len(te)) if btrue[j] >= 0])

    # confident cutover: agree-and-confident slice
    conf = np.where(a_pred == 0, a_conf, np.minimum(a_conf, b_conf))
    sel = conf >= CONF_THR
    cut_prec = accuracy_score([te_lab[j] for j in range(len(te)) if sel[j]],
                              [final[j] for j in range(len(te)) if sel[j]]) if sel.sum() else 0.0

    print(f"\n================= TYPE-HEAD v2 VERDICT =================")
    print(f"end-to-end held-out acc = {acc:.3f}")
    print(f"   vs frozen 0.650 : {acc-FROZEN_BASELINE:+.3f}   vs v1 0.573 : {acc-V1_SINGLE_STAGE:+.3f}")
    print(f"stage A (keep-vs-typed) acc = {accA:.3f}   stage B (type|typed) acc = {accB:.3f}")
    print(f"confident cutover: {sel.mean()*100:.0f}% of atoms @ {cut_prec:.3f} precision (thr {CONF_THR})")
    print("VERDICT:", "BEATS FROZEN ✅" if acc > FROZEN_BASELINE else
          "beats v1, below frozen — bigger encoder/clean labels next" if acc > V1_SINGLE_STAGE else
          "no gain")
    json.dump({"typed_labels": typed_labels}, open(f"{OUT}/stageB/best/labels.json", "w"))


if __name__ == "__main__":
    main()
