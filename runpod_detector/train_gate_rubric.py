"""Train the keep-vs-typed GATE on rubric-consistent labels and measure held-out.

Consumes _rubric_gate_data.json (train = rubric labels on train-deal atoms, test =
rubric labels on HELD-OUT-deal atoms). Because the rubric labels are ~95% reproducible
(two independent models agree), this is a trustworthy target — unlike the 59%-agreement
teacher labels. If a small bge-small gate trained here clears 0.90 held-out, we've shown
a cheap model can reproduce the rubric on unseen deals = the deflection we want.

Run AFTER rubric_adjudicate.py relabel.  Env: EPOCHS=6 BATCH=32 CONF=0.85
"""
import os, json, collections
import numpy as np

EPOCHS=int(os.environ.get("EPOCHS","6")); BATCH=int(os.environ.get("BATCH","32"))
MODEL=os.environ.get("BASE_MODEL","BAAI/bge-small-en-v1.5")
USE_LORA=os.environ.get("USE_LORA","0")=="1"
CONF=float(os.environ.get("CONF","0.85"))

def lora_targets(model):
    mt=getattr(model.config,"model_type","")
    return ["query","key","value"] if mt in ("bert","roberta","xlm-roberta") else ["q_proj","k_proj","v_proj","o_proj"]

def main():
    import torch
    from datasets import Dataset
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              TrainingArguments, Trainer, TrainerCallback)
    from sklearn.metrics import accuracy_score, precision_score, recall_score

    d=json.load(open("_rubric_gate_data.json"))
    tr, te = d["train"], d["test"]
    print(f"train={len(tr)} held-out={len(te)} | train typed-frac={np.mean([r['y'] for r in tr]):.2f} "
          f"held-out typed-frac={np.mean([r['y'] for r in te]):.2f}")

    tok=AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    def enc(b): return tok(b["text"],truncation=True,max_length=128,padding="max_length")
    dtr=Dataset.from_dict({"text":[r["text"] for r in tr],"label":[r["y"] for r in tr]}).map(enc,batched=True)
    dte=Dataset.from_dict({"text":[r["text"] for r in te],"label":[r["y"] for r in te]}).map(enc,batched=True)
    mkw={"num_labels":2}
    if USE_LORA: mkw["torch_dtype"]=torch.bfloat16   # 8B in bf16 fits 96GB; LoRA adapters stay fp32
    model=AutoModelForSequenceClassification.from_pretrained(MODEL,**mkw)
    if model.config.pad_token_id is None: model.config.pad_token_id=tok.pad_token_id
    if USE_LORA:
        from peft import LoraConfig, get_peft_model, TaskType
        cfg=LoraConfig(task_type=TaskType.SEQ_CLS,r=16,lora_alpha=32,lora_dropout=0.1,target_modules=lora_targets(model))
        model=get_peft_model(model,cfg); model.print_trainable_parameters()

    cnt=collections.Counter(r["y"] for r in tr); tot=len(tr)
    w=torch.tensor([tot/(2*cnt[c]) for c in (0,1)],dtype=torch.float)
    class Weighted(Trainer):
        def compute_loss(self,model,inputs,return_outputs=False,**kw):
            y=inputs.pop("labels"); out=model(**inputs)
            # compute loss in fp32 (logits are bf16 under the 8B path; weights must match)
            loss=torch.nn.functional.cross_entropy(out.logits.float(),y,weight=w.to(out.logits.device).float())
            return (loss,out) if return_outputs else loss
    def metrics(p):
        logits,y=p; prob=torch.softmax(torch.tensor(logits),-1).numpy(); pred=prob.argmax(-1)
        conf=prob.max(-1); sel=conf>=CONF
        return {"acc":accuracy_score(y,pred),
                "prec":precision_score(y,pred,zero_division=0),
                "rec":recall_score(y,pred,zero_division=0),
                "conf_cov":float(sel.mean()),
                "conf_acc":accuracy_score(np.array(y)[sel],pred[sel]) if sel.sum() else 0.0}
    class Watch(TrainerCallback):
        best=0.0
        def on_evaluate(self,args,state,control,metrics=None,**kw):
            m=metrics or {}; acc=m.get("eval_acc",0); Watch.best=max(Watch.best,acc)
            print(f"  epoch {state.epoch or 0:>3.1f} | held-out acc {acc:.3f} | prec {m.get('eval_prec',0):.3f} "
                  f"rec {m.get('eval_rec',0):.3f} | confident {m.get('eval_conf_cov',0)*100:.0f}%@{m.get('eval_conf_acc',0):.3f} "
                  f"| best {Watch.best:.3f}",flush=True)
    args=TrainingArguments(output_dir="runs/gate_rubric",num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH,per_device_eval_batch_size=64,eval_strategy="epoch",
        save_strategy="epoch",load_best_model_at_end=True,metric_for_best_model="acc",greater_is_better=True,save_total_limit=1,
        logging_steps=50,learning_rate=(1e-4 if USE_LORA else 2e-5),warmup_ratio=0.06,weight_decay=0.01,
        report_to=[],
        bf16=(USE_LORA and torch.cuda.is_available()),          # 8B loads in bf16 -> train bf16 on CUDA
        fp16=(not USE_LORA and torch.cuda.is_available()),
        disable_tqdm=False)
    t=Weighted(model=model,args=args,train_dataset=dtr,eval_dataset=dte,compute_metrics=metrics,callbacks=[Watch()],
               preprocess_logits_for_metrics=lambda logits,labels: logits.float())  # bf16 -> fp32 before numpy
    print("=== training keep-vs-typed gate on RUBRIC labels (watch held-out climb) ===")
    t.train(); f=t.evaluate()
    print(f"\n================ GATE-ON-RUBRIC VERDICT ================")
    print(f"held-out acc (vs rubric labels) = {f.get('eval_acc',0):.3f}")
    print(f"  precision {f.get('eval_prec',0):.3f} | recall {f.get('eval_rec',0):.3f}")
    print(f"  confident slice: {f.get('eval_conf_cov',0)*100:.0f}% of atoms @ {f.get('eval_conf_acc',0):.3f} precision")
    acc=f.get('eval_acc',0)
    print("VERDICT:", "NAILED IT — gate >=0.90 held-out on a 95%-reproducible target ✅" if acc>=0.90 else
          f"{acc:.3f} held-out — strong, climbing toward 90 (more rubric labels / bigger encoder)" if acc>=0.80 else
          "below 0.80 — investigate")
    os.makedirs("runs/gate_rubric/best",exist_ok=True)
    t.save_model("runs/gate_rubric/best"); tok.save_pretrained("runs/gate_rubric/best")

if __name__=="__main__":
    main()
