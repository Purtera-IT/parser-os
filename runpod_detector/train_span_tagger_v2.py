"""Span-tagger v2 — make the SKIP gate TRUSTWORTHY (honest threshold) + LoRA-ready.

v1 problem: it picked the operating threshold on the SAME held-out set it then
reported recall on → the 0.93 "skippable" verdict was optimistic, and on relations
with 13-24 held-out positives that's a real risk for a gate that kills a production
LLM call.

v2 changes:
  1. THREE-way BY-DEAL split (train / val / test). Pick the precision-floor
     threshold on VAL, report recall on TEST. No more grading-own-homework.
  2. Reports the held-out positive COUNT and a recall confidence band so a tiny-N
     "1.000" can't masquerade as certainty.
  3. Optional LoRA (USE_LORA=1) + configurable encoder — bge-small full-FT on a
     Mac, or LoRA-a-big-encoder on an A100.
  4. Saves the tokenizer with each head.

A relation is only declared SKIPPABLE if TEST recall >= 0.93 AND it has enough
test positives to mean it (>= MIN_TEST_POS, default 20).

Run:  python runpod_detector/train_span_tagger_v2.py
Env:  BASE_MODEL, USE_LORA, EPOCHS=8, RELATIONS, MIN_TEST_POS=20
"""
import os, sqlite3, hashlib, random
import numpy as np

DB = os.environ.get("SOWSMITH_TRAINING_LOG_DB", "_training_deepseek.db")
MODEL = os.environ.get("BASE_MODEL", "BAAI/bge-small-en-v1.5")
USE_LORA = os.environ.get("USE_LORA", "0") == "1"
EPOCHS = int(os.environ.get("EPOCHS", "8"))
BATCH = int(os.environ.get("BATCH", "32"))
PRECISION_FLOOR = 0.80
SKIP_BAR = 0.93
MIN_TEST_POS = int(os.environ.get("MIN_TEST_POS", "20"))
RELATIONS = os.environ.get("RELATIONS", "requirements,site_clusters,commercial_line_items").split(",")
FROZEN = {"requirements": 0.74, "site_clusters": 0.69, "commercial_line_items": 0.67}


def bucket(deal_id):
    """Deterministic by-deal 3-way split: 60% train / 20% val / 20% test."""
    h = int(hashlib.sha256((deal_id or "").encode()).hexdigest(), 16) % 100
    return "train" if h < 60 else "val" if h < 80 else "test"


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


def recall_at(scores, y, thr):
    pred = scores >= thr
    tp = int((pred & (y == 1)).sum()); fn = int((~pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    return rec, prec, (tp + fn)


def wilson_low(rec, n, z=1.96):
    """Lower bound of a Wilson interval — honest floor for tiny-N recall."""
    if n == 0:
        return 0.0
    p = rec
    denom = 1 + z*z/n
    centre = p + z*z/(2*n)
    margin = z*((p*(1-p)/n + z*z/(4*n*n))**0.5)
    return max(0.0, (centre - margin)/denom)


def lora_targets(model):
    mt = getattr(model.config, "model_type", "")
    return ["query", "key", "value"] if mt in ("bert", "roberta", "xlm-roberta") else ["q_proj", "k_proj", "v_proj", "o_proj"]


def main():
    import torch
    from datasets import Dataset
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              TrainingArguments, Trainer)

    data = load_all()
    feat_deal = {}
    for _r, t, d in data:
        feat_deal.setdefault(t, d)
    tok = AutoTokenizer.from_pretrained(MODEL)
    def enc(b): return tok(b["text"], truncation=True, max_length=128, padding="max_length")

    summary = []
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
        grp = {t: bucket(feat_deal[t]) for t in feats}
        tr = [(t, lab[t]) for t in feats if grp[t] == "train"]
        va = [(t, lab[t]) for t in feats if grp[t] == "val"]
        te = [(t, lab[t]) for t in feats if grp[t] == "test"]
        base = FROZEN.get(rel, 0.0)
        print(f"\n{'='*64}\n### {rel}: train={len(tr)} val={len(va)} test={len(te)} "
              f"| test-pos {sum(y for _,y in te)} | frozen {base:.2f} | LoRA={USE_LORA}")

        dtr = Dataset.from_dict({"text": [t for t, _ in tr], "label": [y for _, y in tr]}).map(enc, batched=True)
        dva = Dataset.from_dict({"text": [t for t, _ in va], "label": [y for _, y in va]}).map(enc, batched=True)

        model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
        if USE_LORA:
            from peft import LoraConfig, get_peft_model, TaskType
            cfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32,
                             lora_dropout=0.1, target_modules=lora_targets(model))
            model = get_peft_model(model, cfg); model.print_trainable_parameters()

        args = TrainingArguments(
            output_dir=f"runs/span2_{rel}", num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH, per_device_eval_batch_size=64,
            eval_strategy="epoch", save_strategy="no", logging_steps=50,
            learning_rate=(1e-4 if USE_LORA else 2e-5), warmup_ratio=0.06, weight_decay=0.01,
            report_to=[], fp16=torch.cuda.is_available(), disable_tqdm=False)
        t_ = Trainer(model=model, args=args, train_dataset=dtr, eval_dataset=dva)
        t_.train()

        def scores(rows):
            mdl = t_.model.eval(); dev = next(mdl.parameters()).device
            out = []
            with torch.no_grad():
                for i in range(0, len(rows), 64):
                    enc_ = tok([t for t, _ in rows[i:i+64]], truncation=True, max_length=128,
                               padding=True, return_tensors="pt").to(dev)
                    out.append(torch.softmax(mdl(**enc_).logits, -1)[:, 1].cpu().numpy())
            return np.concatenate(out, 0) if out else np.array([])

        # threshold from VAL, recall on TEST  (honest)
        sv, yv = scores(va), np.array([y for _, y in va])
        thr, vrec, vprec = pick_threshold(sv, yv, PRECISION_FLOOR)
        st, yt = scores(te), np.array([y for _, y in te])
        trec, tprec, npos = recall_at(st, yt, thr)
        lo = wilson_low(trec, npos)
        skippable = trec >= SKIP_BAR and npos >= MIN_TEST_POS and lo >= 0.85
        verdict = ("SKIP UNLOCKS ✅" if skippable else
                   f"recall>=bar but only {npos} test-pos (need {MIN_TEST_POS}) — grow data" if trec >= SKIP_BAR else
                   "better than frozen" if trec > base + 0.05 else "no gain")
        print(f"  thr@val={thr:.3f} (val recall {vrec:.3f}) | TEST recall {trec:.3f} "
              f"@prec {tprec:.3f} on {npos} pos | recall 95%-low {lo:.3f} | vs frozen {base:.2f}")
        print(f"  VERDICT {rel}: {verdict}")
        os.makedirs(f"runs/span2_{rel}/best", exist_ok=True)
        t_.save_model(f"runs/span2_{rel}/best"); tok.save_pretrained(f"runs/span2_{rel}/best")
        summary.append((rel, trec, npos, lo, skippable))

    print(f"\n================= SPAN v2 SUMMARY (honest) =================")
    for rel, trec, npos, lo, sk in summary:
        print(f"  {rel:24s} test-recall {trec:.3f} (95%-low {lo:.3f}, {npos} pos)  "
              f"{'SKIPPABLE' if sk else 'not yet — needs more positives'}")


if __name__ == "__main__":
    main()
