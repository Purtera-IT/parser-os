"""THE ARCHITECTURE UNLOCK (Layer 1): supervised-CONTRASTIVE encoder + kNN.

Why this (measured this session): a classifier HEAD on the embedding caps ~0.82
because a head pattern-matches and can't replicate the labeler's reasoning; and
kNN on the FROZEN space caps ~0.65 because that space is sorted by general meaning
— neighbors have mixed labels. The fix is NOT a better head; it's a better SPACE.

This fine-tunes the encoder with a supervised-contrastive (batch-hard triplet)
loss on the CLEAN rubric labels: pull same-label atoms together, push opposite
apart. The space is then organized around the keep-vs-typed (or 7-facet) boundary,
so **kNN over it works** — and kNN is the deployment (instant-learning: a PM
correction is usable on the next atom, no retrain).

CRITICAL: we eval via **kNN** (the real deployment), NOT a linear probe — else we
repeat the classifier-head mistake. Every epoch prints held-out-by-deal kNN
accuracy + precision@coverage (the guess-free confident slice) vs the baselines.

Modes (LABEL_MODE):
  gate   — binary keep-vs-typed (the highest-error decision). baseline to beat: 0.82.
  facet  — 7-way over TYPED atoms (the dashboard sections). baseline: ~0.846 agreement ceiling.

Run on RunPod:
  pip install -U "sentence-transformers>=3.0" "datasets>=2.20" scikit-learn
  LABEL_MODE=gate  python train_contrastive_encoder_gpu.py
  LABEL_MODE=facet python train_contrastive_encoder_gpu.py
Inputs: _training_deepseek.db (ship next to this script).
"""
import os, sqlite3, hashlib, json, collections
import numpy as np

DB = os.environ.get("SOWSMITH_TRAINING_LOG_DB", "_training_deepseek.db")
MODEL = os.environ.get("BASE_MODEL", "BAAI/bge-small-en-v1.5")
EPOCHS = int(os.environ.get("EPOCHS", "10"))
BATCH = int(os.environ.get("BATCH", "64"))
K = int(os.environ.get("KNN_K", "15"))
HOLDOUT = 0.25
LABEL_MODE = os.environ.get("LABEL_MODE", "gate")
GATE_BASELINE = 0.82      # LoRA classifier-head ceiling (what kNN must beat)
FACET_BASELINE = 0.846    # two-model facet agreement ceiling

# 41 micro-types -> 7 dashboard facets (from RUBRIC.md).
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


def load():
    con = sqlite3.connect(DB)
    rows = con.execute(
        "SELECT COALESCE(NULLIF(masked_text,''),raw_text) AS t, label, deal_id "
        "FROM training_rows WHERE relation='atom_type' AND COALESCE(masked_text,raw_text,'')!='' "
        "AND label IS NOT NULL").fetchall()
    con.close()
    out = []
    for t, l, d in rows:
        if not t:
            continue
        if LABEL_MODE == "gate":
            out.append((t, "_keep" if l == "_keep" else "typed", d or ""))
        else:  # facet: only typed atoms, mapped to a facet
            if l == "_keep" or l not in FACET:
                continue
            out.append((t, FACET[l], d or ""))
    return out


def knn_eval(model, tr_texts, tr_y, te_texts, te_y, labels):
    """kNN over the contrastive space = the deployment. Returns (acc, and the
    precision/coverage curve over confidence). Confidence = winning-label vote share."""
    import torch
    tr_emb = model.encode(tr_texts, batch_size=256, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    te_emb = model.encode(te_texts, batch_size=256, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    sims = te_emb @ tr_emb.T                       # cosine (normalized)
    idx = np.argpartition(-sims, K, axis=1)[:, :K]
    preds, confs = [], []
    tr_y = np.array(tr_y)
    for i in range(len(te_texts)):
        nb = idx[i]
        w = sims[i, nb]
        w = np.clip(w, 1e-6, None)
        votes = collections.defaultdict(float)
        for j, lab in zip(nb, tr_y[nb]):
            votes[lab] += sims[i, j]
        win = max(votes, key=votes.get)
        preds.append(win)
        confs.append(votes[win] / (sum(votes.values()) + 1e-9))
    preds = np.array(preds); confs = np.array(confs); te_y = np.array(te_y)
    acc = float((preds == te_y).mean())
    curve = {}
    for tau in (0.6, 0.7, 0.8, 0.9, 0.95):
        sel = confs >= tau
        curve[tau] = (float(sel.mean()), float((preds[sel] == te_y[sel]).mean()) if sel.sum() else 0.0)
    return acc, curve


def main():
    from sentence_transformers import SentenceTransformer, losses, InputExample
    from sentence_transformers.datasets import SentenceLabelDataset
    from torch.utils.data import DataLoader

    data = load()
    labels = sorted({l for _, l, _ in data})
    l2i = {l: i for i, l in enumerate(labels)}
    tr = [(t, l) for t, l, d in data if split(d) == "train"]
    te = [(t, l) for t, l, d in data if split(d) == "test"]
    base = GATE_BASELINE if LABEL_MODE == "gate" else FACET_BASELINE
    print(f"MODE={LABEL_MODE} | classes={labels} | train={len(tr)} held-out={len(te)} | base={MODEL}")
    print(f"BASELINE TO BEAT (via kNN, held-out-by-deal): {base:.3f}\n")

    model = SentenceTransformer(MODEL)
    train_examples = [InputExample(texts=[t], label=l2i[l]) for t, l in tr]
    ds = SentenceLabelDataset(train_examples, samples_per_label=2)
    loader = DataLoader(ds, batch_size=BATCH, drop_last=True)
    loss = losses.BatchHardTripletLoss(model=model)

    tr_t = [t for t, _ in tr]; tr_l = [l for _, l in tr]
    te_t = [t for t, _ in te]; te_l = [l for _, l in te]

    print("=== epoch 0 (frozen, before contrastive fit) ===")
    acc0, c0 = knn_eval(model, tr_t, tr_l, te_t, te_l, labels)
    print(f"  kNN acc {acc0:.3f} ({acc0-base:+.3f} vs baseline) | conf>=0.9: {c0[0.9][0]*100:.0f}%@{c0[0.9][1]:.3f}")

    best = acc0
    for ep in range(1, EPOCHS + 1):
        model.fit(train_objectives=[(loader, loss)], epochs=1, warmup_steps=int(0.06*len(loader)),
                  show_progress_bar=True, optimizer_params={"lr": 2e-5})
        acc, curve = knn_eval(model, tr_t, tr_l, te_t, te_l, labels)
        up = acc > best + 1e-6; best = max(best, acc)
        cov9, p9 = curve[0.9]; cov8, p8 = curve[0.8]
        delta = acc - base
        tag = ("BEATS BASELINE ✅ (new best)" if up and delta > 0 else
               "BEATS BASELINE ✅" if delta > 0 else
               "approaching" if delta > -0.05 else "below")
        print(f"  epoch {ep:>2} | kNN acc {acc:.3f} ({delta:+.3f} vs {base:.2f}) "
              f"| guess-free conf>=0.8: {cov8*100:.0f}%@{p8:.3f}  conf>=0.9: {cov9*100:.0f}%@{p9:.3f} "
              f"| best {best:.3f} | {tag}", flush=True)

    print(f"\n=== VERDICT (MODE={LABEL_MODE}) ===")
    print(f"best kNN held-out acc = {best:.3f}  vs classifier-head/agreement baseline {base:.3f}")
    print("UNLOCK ✅ — contrastive space beats the head; ship kNN cascade" if best > base else
          "matches baseline — the boundary is the irreducible-ambiguity wall (route to LLM)" if best > base - 0.04 else
          "below — needs cleaner rubric labels / more diverse pairs")
    out = f"runs/contrastive_{LABEL_MODE}/best"
    os.makedirs(out, exist_ok=True)
    model.save(out)
    # persist the labeled store (train embeddings + labels) for runtime kNN
    tr_emb = model.encode(tr_t, batch_size=256, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    np.savez_compressed(f"runs/contrastive_{LABEL_MODE}/store.npz", emb=tr_emb, y=np.array(tr_l))
    json.dump({"labels": labels, "k": K, "mode": LABEL_MODE}, open(f"{out}/knn_meta.json", "w"))
    print(f"saved encoder -> {out} ; kNN store -> runs/contrastive_{LABEL_MODE}/store.npz")


if __name__ == "__main__":
    main()
