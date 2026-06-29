"""Train the PDF image-kind CPU gate from TrainingLog rows (relation pdf_image_kind).

Silver rows are logged automatically by pdf_image_vision after each VLM gate
decision. Gold rows come from PM chip corrections on the ``image`` head.

Export the log first, or point TRAINING_DB at the worker DB::

    python -c "
    from app.core.training_log import TrainingLog
    import json
    log = TrainingLog('_training_deepseek.db')
    rows = [r for r in log.rows(relation='pdf_image_kind')]
    json.dump({'rows': [{'text': r.feature_text or r.raw_text, 'label': r.label,
                         'weight': r.weight, 'split': r.split, 'teacher': r.teacher}
                        for r in rows]},
              open('_pdf_image_gate_data.json','w'), indent=2)
    print(len(rows), 'rows')
    "

Then on RunPod / GPU box::

    TRAINING_DB=_training_deepseek.db python train_pdf_image_gate.py

Packages to ``runs/pdf_image_gate/best`` — tar as ``gate_pdf_image.tgz`` and
upload to blob ml-artifacts (worker unpacks to SOWSMITH_PDF_IMAGE_GATE_DIR).
"""
from __future__ import annotations

import collections
import json
import os
import sys

EPOCHS = int(os.environ.get("EPOCHS", "8"))
BATCH = int(os.environ.get("BATCH", "16"))
MODEL = os.environ.get("BASE_MODEL", "BAAI/bge-small-en-v1.5")
CONF = float(os.environ.get("CONF", "0.85"))
DATA_PATH = os.environ.get("PDF_IMAGE_GATE_DATA", "_pdf_image_gate_data.json")
TRAINING_DB = os.environ.get("TRAINING_DB", "")


def _export_from_db(path: str) -> None:
    from app.core.training_log import TrainingLog
    from app.core.pdf_image_gate import gate_feature_text

    log = TrainingLog(TRAINING_DB)
    rows = log.rows(relation="pdf_image_kind")
    if not rows:
        print(f"No pdf_image_kind rows in {TRAINING_DB}", file=sys.stderr)
        sys.exit(1)
    out = []
    for r in rows:
        text = (r.masked_text or r.raw_text or "").strip()
        if not text or not r.label:
            continue
        out.append({
            "text": text,
            "label": r.label.strip().lower(),
            "weight": float(r.weight or 1.0),
            "split": r.split,
            "teacher": r.teacher,
        })
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"rows": out}, fh, indent=2)
    print(f"Exported {len(out)} rows -> {path}")


def _label_map(rows: list[dict]) -> dict[str, int]:
    labels = sorted({r["label"] for r in rows})
    return {lb: i for i, lb in enumerate(labels)}


def main() -> None:
    import numpy as np
    import torch
    from datasets import Dataset
    from sklearn.metrics import accuracy_score
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    if TRAINING_DB and not os.path.isfile(DATA_PATH):
        _export_from_db(DATA_PATH)
    if not os.path.isfile(DATA_PATH):
        print(f"Missing {DATA_PATH}. Set TRAINING_DB or create the JSON export.",
              file=sys.stderr)
        sys.exit(1)

    raw = json.load(open(DATA_PATH, encoding="utf-8"))["rows"]
    train = [r for r in raw if r.get("split") != "holdout"]
    test = [r for r in raw if r.get("split") == "holdout"]
    if not test:
        # hash-split fallback: last 20% as holdout
        test = raw[-max(1, len(raw) // 5):]
        train = raw[: len(raw) - len(test)]
    if len(train) < 20:
        print(f"Need >=20 train rows, have {len(train)}", file=sys.stderr)
        sys.exit(1)

    lmap = _label_map(train + test)
    print(f"labels ({len(lmap)}): {list(lmap.keys())}")
    print(f"train={len(train)} holdout={len(test)}")

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def enc(batch):
        return tok(batch["text"], truncation=True, max_length=256, padding="max_length")

    def to_ds(part):
        return Dataset.from_dict({
            "text": [r["text"] for r in part],
            "label": [lmap[r["label"]] for r in part],
        }).map(enc, batched=True)

    dtr, dte = to_ds(train), to_ds(test)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=len(lmap),
    )
    model.config.id2label = {i: lb for lb, i in lmap.items()}
    model.config.label2id = lmap
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tok.pad_token_id

    cnt = collections.Counter(lmap[r["label"]] for r in train)
    tot = len(train)
    w = torch.tensor([tot / (len(cnt) * cnt[c]) for c in range(len(lmap))], dtype=torch.float)

    class Weighted(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            y = inputs.pop("labels")
            out = model(**inputs)
            loss = torch.nn.functional.cross_entropy(
                out.logits.float(), y, weight=w.to(out.logits.device).float(),
            )
            return (loss, out) if return_outputs else loss

    def metrics(p):
        logits, y = p
        prob = torch.softmax(torch.tensor(logits), -1).numpy()
        pred = prob.argmax(-1)
        conf = prob.max(-1)
        sel = conf >= CONF
        return {
            "acc": accuracy_score(y, pred),
            "conf_cov": float(sel.mean()),
            "conf_acc": accuracy_score(np.array(y)[sel], pred[sel]) if sel.sum() else 0.0,
        }

    args = TrainingArguments(
        output_dir="runs/pdf_image_gate",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="acc",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=25,
        learning_rate=2e-5,
        warmup_ratio=0.06,
        weight_decay=0.01,
        report_to=[],
        fp16=torch.cuda.is_available(),
    )
    t = Weighted(
        model=model, args=args, train_dataset=dtr, eval_dataset=dte,
        compute_metrics=metrics,
        preprocess_logits_for_metrics=lambda logits, labels: logits.float(),
    )
    print("=== training pdf_image_kind CPU gate ===")
    t.train()
    f = t.evaluate()
    acc = f.get("eval_acc", 0)
    print(f"held-out acc={acc:.3f} | conf slice {f.get('eval_conf_cov', 0)*100:.0f}% "
          f"@ {f.get('eval_conf_acc', 0):.3f}")
    os.makedirs("runs/pdf_image_gate/best", exist_ok=True)
    t.save_model("runs/pdf_image_gate/best")
    tok.save_pretrained("runs/pdf_image_gate/best")


if __name__ == "__main__":
    main()
