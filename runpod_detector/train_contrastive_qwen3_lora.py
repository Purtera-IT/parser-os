"""UNIFIED-SPACE variant: contrastive LoRA on qwen3-embedding:8b, served by vLLM.

This is the "one embedding space for everything" path. Instead of a separate
bge encoder, we LoRA-fine-tune the SAME model the runtime already embeds with
(Qwen/Qwen3-Embedding-8B, 4096-d) using the supervised-contrastive objective, so
the type/facet kNN lives in the same space as the general decide() store.

WHY vLLM makes this work (the user's insight):
  - Ollama can't serve a custom embedding LoRA. vLLM can: `--task embed
    --enable-lora` serves the BASE model AND any number of LoRA adapters from one
    process (multi-LoRA). So the worker gets, from ONE served model:
      * base qwen3-embedding  -> the existing pinned decide() store (unchanged), and
      * base + this adapter   -> the type/facet kNN (re-sorted space).
    The pinned "never swap the embed model" constraint is respected: base output
    is identical; the adapter is an ADD-ON namespace, not a replacement.
  - Trade-off (honest): vLLM is CUDA-only, so prod embedding must run on a CUDA
    box, not the Apple-Silicon Mac Studio. See VLLM_EMBEDDING_LORA.md.

Compared to the bge build: same SupCon loss, same unified labels, same value-
invariance aug, same kNN eval — only the base model + LoRA differ. Run both and
keep whichever wins the held-out kNN verdict.

Run on RunPod A100 (80GB; use QLORA=1 for 40GB):
  pip install -U "transformers>=4.51" peft accelerate datasets scikit-learn bitsandbytes
  LABEL_MODE=unified python train_contrastive_qwen3_lora.py
Inputs: _training_deepseek.db. Output: runs/qwen3_lora_<mode>/adapter (load in vLLM).
"""
import os, sqlite3, hashlib, json, collections
import numpy as np

DB = os.environ.get("SOWSMITH_TRAINING_LOG_DB", "_training_deepseek.db")
MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3-Embedding-8B")
EPOCHS = int(os.environ.get("EPOCHS", "6"))
BATCH = int(os.environ.get("BATCH", "64"))
MAXLEN = int(os.environ.get("MAXLEN", "128"))
K = int(os.environ.get("KNN_K", "15"))
TEMP = float(os.environ.get("TEMP", "0.07"))
LR = float(os.environ.get("LR", "1e-4"))
HOLDOUT = 0.25
LABEL_MODE = os.environ.get("LABEL_MODE", "unified")
SIM_FLOOR = float(os.environ.get("SIM_FLOOR", "0.55"))
TARGET_PREC = float(os.environ.get("TARGET_PREC", "0.95"))
QLORA = os.environ.get("QLORA", "").strip().lower() in ("1", "true", "yes", "on")
GATE_BASELINE = 0.82
FACET_BASELINE = 0.846

FACET = {}
for fac, types in {
    "SITE": "physical_site site_attribute site_access_restriction site_room_mix site_infrastructure",
    "COMMERCIAL": "service_line bom_line payment_term commercial_total pricing_assumption site_budget",
    "WORK": "requirement task deliverable acceptance_criterion milestone_phase cutover_step "
            "electrical_acceptance_test site_implementation_note site_access_window exclusion integration_checkpoint",
    "COMPLIANCE": "compliance_rule compliance_classification approval_authority submission_req "
                  "change_order_rule bonding_insurance",
    "TIMING": "blackout_date_range lead_time_constraint deadline dependency",
    "META": "deal_metadata eval_criterion approval_decision signatory",
    "PARTY": "stakeholder",
}.items():
    for t in types.split():
        FACET[t] = fac


def split(deal_id):
    h = int(hashlib.sha256((deal_id or "").encode()).hexdigest(), 16)
    return "test" if (h % 100) / 100.0 < HOLDOUT else "train"


def _map(label):
    if LABEL_MODE == "gate":
        return "_keep" if label == "_keep" else "typed"
    if LABEL_MODE == "facet":
        return None if label == "_keep" else FACET.get(label)
    return "_keep" if label == "_keep" else FACET.get(label)


def load():
    con = sqlite3.connect(DB)
    rows = con.execute(
        "SELECT raw_text, COALESCE(masked_text,'') AS m, label, deal_id, COALESCE(teacher,'') AS t "
        "FROM training_rows WHERE relation='atom_type' AND label IS NOT NULL "
        "AND COALESCE(masked_text,raw_text,'')!=''").fetchall()
    con.close()
    by_text = {}
    for raw, masked, label, deal, teacher in rows:
        cls = _map(label)
        if cls is None:
            continue
        key = (raw or masked or "").strip()
        if not key:
            continue
        gold = teacher.lower() in ("pm", "human", "gold")
        prev = by_text.get(key)
        if prev and prev[2] and not gold:
            continue
        by_text[key] = (cls, deal or "", gold, raw or "", masked or "")
    return list(by_text.values())


def knn_from_emb(s_emb, s_y, t_emb, t_y, labels):
    sims = t_emb @ s_emb.T
    s_y = np.array(s_y)
    idx = np.argpartition(-sims, min(K, sims.shape[1] - 1), axis=1)[:, :K]
    preds, confs = [], []
    for i in range(t_emb.shape[0]):
        nb = idx[i]; top1 = float(sims[i, nb].max())
        votes = collections.defaultdict(float)
        for j in nb:
            votes[s_y[j]] += max(float(sims[i, j]), 0.0)
        ranked = sorted(votes.items(), key=lambda kv: -kv[1])
        total = sum(votes.values()) + 1e-9
        margin = (ranked[0][1] - (ranked[1][1] if len(ranked) > 1 else 0.0)) / total
        preds.append(ranked[0][0]); confs.append(margin if top1 >= SIM_FLOOR else 0.0)
    preds = np.array(preds); confs = np.array(confs); t_y = np.array(t_y)
    correct = (preds == t_y).astype(float)
    acc = float(correct.mean())
    op = None
    for tau in np.unique(confs):
        sel = confs >= tau
        if sel.sum() and correct[sel].mean() >= TARGET_PREC:
            op = (float(tau), float(sel.mean()), float(correct[sel].mean())); break
    rec = {c: (float((preds[t_y == c] == c).mean()) if (t_y == c).sum() else 0.0) for c in labels}
    return acc, op, rec


def main():
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel
    from peft import LoraConfig, get_peft_model

    def last_token_pool(h, mask):
        left = bool((mask[:, -1].sum() == mask.shape[0]).item())
        if left:
            return h[:, -1]
        lengths = mask.sum(dim=1) - 1
        return h[torch.arange(h.size(0), device=h.device), lengths]

    def supcon(z, labels, t=TEMP):
        z = F.normalize(z, dim=1)
        sim = (z @ z.T) / t
        sim = sim - sim.max(1, keepdim=True)[0].detach()
        B = z.size(0); self_mask = torch.eye(B, device=z.device)
        lab = labels.view(-1, 1); pos = (lab == lab.T).float() - self_mask
        exp = torch.exp(sim) * (1 - self_mask)
        logp = sim - torch.log(exp.sum(1, keepdim=True) + 1e-12)
        pc = pos.sum(1); mlp = (pos * logp).sum(1) / torch.clamp(pc, min=1.0)
        valid = (pc > 0).float()
        return -(mlp * valid).sum() / torch.clamp(valid.sum(), min=1.0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load()
    labels = sorted({d[0] for d in data})
    l2i = {c: i for i, c in enumerate(labels)}
    base = FACET_BASELINE if LABEL_MODE == "facet" else GATE_BASELINE

    tr_text, tr_lab, store_t, store_l, te_t, te_l = [], [], [], [], [], []
    dist = collections.Counter()
    for cls, deal, gold, raw, masked in data:
        canon = masked or raw
        if split(deal) == "train":
            dist[cls] += 1
            for txt in {x for x in (raw, masked) if x}:
                tr_text.append(txt); tr_lab.append(l2i[cls])
            store_t.append(canon); store_l.append(cls)
        else:
            te_t.append(canon); te_l.append(cls)

    print(f"MODE={LABEL_MODE} | classes={labels} | base={MODEL} | qlora={QLORA}")
    print(f"train atoms={len(store_t)} (aug points={len(tr_text)}) held-out={len(te_t)}")
    print(f"class balance: {dict(dist)}")
    print(f"BASELINE TO BEAT (via kNN): {base:.3f} | target precision {TARGET_PREC:.2f}\n")

    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model_kw = dict(trust_remote_code=True, torch_dtype=torch.bfloat16)
    if QLORA:
        from transformers import BitsAndBytesConfig
        model_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModel.from_pretrained(MODEL, **model_kw)
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.to(device); model.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)

    @torch.no_grad()
    def encode(texts, bs=32):
        model.eval(); out = []
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i + bs], padding=True, truncation=True,
                      max_length=MAXLEN, return_tensors="pt").to(device)
            h = model(**enc).last_hidden_state
            z = F.normalize(last_token_pool(h, enc["attention_mask"]), dim=1)
            out.append(z.float().cpu().numpy())
        return np.vstack(out)

    def report(tag):
        acc, op, rec = knn_from_emb(encode(store_t), store_l, encode(te_t), te_l, labels)
        opstr = (f"{op[1]*100:.0f}%@{op[2]:.3f}(tau{op[0]:.2f})" if op else f"none@{TARGET_PREC:.2f}")
        worst = min(rec.items(), key=lambda kv: kv[1])
        print(f"  {tag} | kNN acc {acc:.3f} ({acc-base:+.3f} vs {base:.2f}) "
              f"| guess-free {opstr} | worst {worst[0]}={worst[1]:.2f}", flush=True)
        return acc, op, rec

    print("=== epoch 0 (base qwen3-embedding, before LoRA) ===")
    best, _, _ = report("epoch  0")

    rng = np.random.default_rng(0)
    order = np.arange(len(tr_text))
    for ep in range(1, EPOCHS + 1):
        model.train(); rng.shuffle(order)
        for i in range(0, len(order), BATCH):
            bi = order[i:i + BATCH]
            if len(bi) < 4:
                continue
            enc = tok([tr_text[j] for j in bi], padding=True, truncation=True,
                      max_length=MAXLEN, return_tensors="pt").to(device)
            h = model(**enc).last_hidden_state
            z = last_token_pool(h, enc["attention_mask"])
            loss = supcon(z, torch.tensor([tr_lab[j] for j in bi], device=device))
            opt.zero_grad(); loss.backward(); opt.step()
        acc, op, rec = report(f"epoch {ep:>2}")
        best = max(best, acc)

    acc, op, rec = report("final ")
    print(f"\n=== VERDICT (qwen3-LoRA, MODE={LABEL_MODE}) ===")
    print(f"best kNN held-out acc = {best:.3f} vs baseline {base:.3f}")
    print(f"per-class recall: { {k: round(v,2) for k,v in rec.items()} }")
    if op:
        print(f"GUESS-FREE OPERATING POINT: {op[1]*100:.0f}% confident @ {op[2]:.3f} prec (tau {op[0]:.2f})")
    print("UNLOCK ✅ — unified qwen3 space beats the head; serve adapter via vLLM" if best > base else
          "matches baseline — ambiguity wall (LLM fallback)" if best > base - 0.04 else "below — clean labels")

    out = f"runs/qwen3_lora_{LABEL_MODE}/adapter"
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out); tok.save_pretrained(out)
    s_emb = encode(store_t)
    np.savez_compressed(f"runs/qwen3_lora_{LABEL_MODE}/store.npz",
                        emb=s_emb, y=np.array(store_l), text=np.array(store_t, dtype=object))
    json.dump({"labels": labels, "k": K, "mode": LABEL_MODE, "sim_floor": SIM_FLOOR,
               "target_precision": TARGET_PREC, "operating_tau": (op[0] if op else None),
               "base_model": MODEL, "serve": "vllm --task embed --enable-lora"},
              open(f"runs/qwen3_lora_{LABEL_MODE}/knn_meta.json", "w"))
    print(f"saved LoRA adapter -> {out} (load in vLLM: --lora-modules type={out})")


if __name__ == "__main__":
    main()
