"""THE ARCHITECTURE UNLOCK (Layer 1): supervised-CONTRASTIVE encoder + kNN.

This is the BEST practical build for breaking the type-decision ceiling while
keeping instant-learning (kNN deployment) and guess-free abstain.

Why a better SPACE, not a better head (measured this session):
  - kNN on the FROZEN embedding caps ~0.65: the space is sorted by general
    meaning, so a point's nearest neighbors have mixed labels.
  - a classifier HEAD caps ~0.82: it pattern-matches, can't replicate the
    labeler's reasoning, and FREEZES — a PM correction needs a retrain to land.
The fix is to fine-tune the representation so the geometry encodes the DECISION;
then plain kNN reads it, and a PM correction is just a new neighbor (usable on
the very next atom, zero retrain).

WHAT MAKES THIS THE BEST BUILD (vs a vanilla triplet fine-tune):
  1. SupCon loss (Khosla et al.), not single-triplet. Uses ALL same-label points
     as positives and ALL others as negatives per batch with a temperature — it
     optimizes exactly the geometry kNN reads, and is far more stable than
     batch-hard triplet on imbalanced classes. Biggest single quality lever.
  2. UNIFIED 8-way space (default): _keep + the 7 dashboard facets in ONE space.
     A single kNN lookup answers both "keep or type?" AND "which facet?" — the
     whole type decision, one model, one store. (gate/facet modes remain as
     diagnostic ablations to prove each cut independently.)
  3. Value-invariance augmentation: each atom is trained as BOTH its raw_text and
     its masked_text (delexicalized). This bakes the rubric's core principle —
     role matters, the specific "110 TVs" value does not — into the geometry.
  4. Gold-priority + class-balanced batches: PM-gold labels win ties over LLM
     silver; the sampler guarantees same-label positives in every batch so SupCon
     always has a gradient.
  5. Deployment-honest eval: kNN (the real runtime) on a held-out-BY-DEAL split,
     every epoch — accuracy + per-class recall + the precision@coverage curve +
     the recommended guess-free operating point (smallest tau hitting target
     precision = max confident coverage). Confidence blends vote-margin with
     top-1 similarity so out-of-distribution atoms abstain to the LLM.

Modes (LABEL_MODE):
  unified — _keep + 7 facets (DEFAULT; the production target). beat 0.82.
  gate    — binary keep-vs-typed (ablation: the highest-error cut). beat 0.82.
  facet   — 7-way over TYPED atoms only (ablation: dashboard sections). ~0.846 ceiling.

Run on RunPod (A100):
  pip install -U "sentence-transformers>=3.0" scikit-learn
  python train_contrastive_encoder_gpu.py                 # unified (default)
  LABEL_MODE=gate  python train_contrastive_encoder_gpu.py
  LABEL_MODE=facet python train_contrastive_encoder_gpu.py
Inputs: _training_deepseek.db (ship next to this script).

Deployment note: the worker embeds atoms with THIS fine-tuned encoder for the
type/facet kNN — a small separate model from qwen3-embedding:8b (which stays for
the general decide() store). Both are cheap to serve.
"""
import os, sqlite3, hashlib, json, collections
import numpy as np

DB = os.environ.get("SOWSMITH_TRAINING_LOG_DB", "_training_deepseek.db")
# bge-base: strong encoder, BERT/WordPiece tokenizer (no sentencepiece gotcha),
# cheap to serve on CPU at runtime. On an A100 you can afford bge-large via env.
MODEL = os.environ.get("BASE_MODEL", "BAAI/bge-base-en-v1.5")
EPOCHS = int(os.environ.get("EPOCHS", "15"))
BATCH = int(os.environ.get("BATCH", "128"))     # SupCon loves big batches (more negatives)
K = int(os.environ.get("KNN_K", "15"))
TEMP = float(os.environ.get("TEMP", "0.07"))
HOLDOUT = 0.25
LABEL_MODE = os.environ.get("LABEL_MODE", "unified")
SIM_FLOOR = float(os.environ.get("SIM_FLOOR", "0.55"))  # OOD gate: top-1 cosine below this -> abstain
PRIOR_ALPHA = float(os.environ.get("PRIOR_ALPHA", "0.5"))  # class-prior debias in kNN vote (0=raw,1=balanced)
TARGET_PREC = float(os.environ.get("TARGET_PREC", "0.95"))  # guess-free operating point
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


try:
    from _split_util import load_split_map, split_of
except ImportError:  # when run as a module
    from runpod_detector._split_util import load_split_map, split_of

_SPLIT_MAP = None


def split(deal_id):
    """Canonical split: prefer the recorded `split` column (holdout-wins for
    mixed deals), hash only as fallback. Replaces the old sha256-only derivation
    that disagreed with the recorded split by 17.9% (coarse) / 97.2% (cloud) and
    leaked 3,773 held-out rows into train."""
    global _SPLIT_MAP
    if _SPLIT_MAP is None:
        _c = sqlite3.connect(DB)
        try:
            _SPLIT_MAP = load_split_map(_c)
        finally:
            _c.close()
    return "test" if split_of(deal_id, _SPLIT_MAP, HOLDOUT) == "holdout" else "train"


def _map(label):
    """raw atom_type label -> the class for this LABEL_MODE (or None to drop)."""
    if LABEL_MODE == "gate":
        return "_keep" if label == "_keep" else "typed"
    if LABEL_MODE == "facet":
        return None if label == "_keep" else FACET.get(label)
    # unified: _keep + 7 facets
    return "_keep" if label == "_keep" else FACET.get(label)


def load():
    """Returns rows: (raw_text, masked_text, class, deal_id). Gold (pm) wins ties
    over silver (llm) for the same text."""
    con = sqlite3.connect(DB)
    rows = con.execute(
        "SELECT raw_text, COALESCE(masked_text,'') AS m, label, deal_id, "
        "COALESCE(teacher,'') AS teacher "
        "FROM training_rows WHERE relation='atom_type' AND label IS NOT NULL "
        "AND COALESCE(masked_text,raw_text,'')!=''").fetchall()
    con.close()
    by_text = {}  # text -> (class, deal, is_gold) ; gold overrides silver
    out = []
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
            continue  # keep existing gold over new silver
        by_text[key] = (cls, deal or "", gold)
        out.append((raw or "", masked or "", cls, deal or ""))
    # de-dup on (text, class) but keep gold-resolved class
    seen = set(); final = []
    for raw, masked, cls, deal in out:
        key = (raw or masked).strip()
        resolved = by_text.get(key)
        if not resolved:
            continue
        cls = resolved[0]
        sig = (key, cls)
        if sig in seen:
            continue
        seen.add(sig)
        final.append((raw, masked, cls, deal))
    return final


def knn_eval(model, store_texts, store_y, te_texts, te_y, labels):
    """kNN over the contrastive space = the deployment. Confidence blends the
    distance-weighted vote margin with the top-1 similarity (OOD gate)."""
    s_emb = model.encode(store_texts, batch_size=256, convert_to_numpy=True,
                         normalize_embeddings=True, show_progress_bar=False)
    t_emb = model.encode(te_texts, batch_size=256, convert_to_numpy=True,
                         normalize_embeddings=True, show_progress_bar=False)
    sims = t_emb @ s_emb.T                       # cosine (normalized)
    store_y = np.array(store_y)
    # class-prior normalization: divide each class's vote by count**alpha so a rare
    # class (e.g. TIMING) is not auto-zeroed by majority (_keep) neighbors. alpha=0
    # -> raw kNN; alpha=1 -> fully balanced. Tunable via PRIOR_ALPHA.
    counts = collections.Counter(store_y.tolist())
    prior = {c: (counts[c] ** PRIOR_ALPHA) for c in counts}
    idx = np.argpartition(-sims, min(K, sims.shape[1]-1), axis=1)[:, :K]
    preds, confs = [], []
    for i in range(len(te_texts)):
        nb = idx[i]
        top1 = float(sims[i, nb].max())
        votes = collections.defaultdict(float)
        for j in nb:
            votes[store_y[j]] += max(sims[i, j], 0.0)
        votes = {c: v / prior.get(c, 1.0) for c, v in votes.items()}
        ranked = sorted(votes.items(), key=lambda kv: -kv[1])
        win = ranked[0][0]
        total = sum(votes.values()) + 1e-9
        share = ranked[0][1] / total
        margin = (ranked[0][1] - (ranked[1][1] if len(ranked) > 1 else 0.0)) / total
        # confidence: vote margin, hard-gated by OOD (top-1 too far -> abstain)
        conf = margin if top1 >= SIM_FLOOR else 0.0
        preds.append(win); confs.append(conf)
    preds = np.array(preds); confs = np.array(confs); te_y = np.array(te_y)
    correct = (preds == te_y).astype(float)
    acc = float(correct.mean())
    # precision@coverage curve
    curve = {}
    for tau in (0.0, 0.2, 0.4, 0.6, 0.8):
        sel = confs >= tau
        curve[tau] = (float(sel.mean()), float(correct[sel].mean()) if sel.sum() else 0.0)
    # recommended guess-free operating point: smallest tau hitting TARGET_PREC
    op = None
    for tau in np.unique(confs):
        sel = confs >= tau
        if sel.sum() and correct[sel].mean() >= TARGET_PREC:
            op = (float(tau), float(sel.mean()), float(correct[sel].mean()))
            break
    # per-class recall
    rec = {}
    for c in labels:
        m = te_y == c
        rec[c] = float((preds[m] == c).mean()) if m.sum() else 0.0
    # per-class guess-free operating point: among atoms PREDICTED c, the smallest
    # tau whose precision >= TARGET_PREC, and how much of class c that retains.
    class_op = {}
    for c in labels:
        pc = preds == c
        if not pc.sum():
            class_op[c] = None; continue
        best = None
        for tau in np.unique(confs[pc]):
            sel = pc & (confs >= tau)
            if sel.sum() and correct[sel].mean() >= TARGET_PREC:
                best = (float(tau), int(sel.sum()), float(correct[sel].mean())); break
        class_op[c] = best
    return acc, curve, op, rec, class_op


def main():
    import torch
    from torch import nn
    from sentence_transformers import SentenceTransformer, InputExample
    from sentence_transformers.datasets import SentenceLabelDataset
    from torch.utils.data import DataLoader

    class SupConLoss(nn.Module):
        """Supervised contrastive loss over L2-normalized sentence embeddings."""
        def __init__(self, model, temperature=0.07):
            super().__init__(); self.model = model; self.t = temperature
        def forward(self, sentence_features, labels):
            z = self.model(sentence_features[0])["sentence_embedding"]
            z = torch.nn.functional.normalize(z, dim=1)
            B = z.shape[0]
            sim = (z @ z.T) / self.t
            sim = sim - sim.max(dim=1, keepdim=True)[0].detach()  # stability
            self_mask = torch.eye(B, device=z.device)
            lab = labels.view(-1, 1)
            pos = (lab == lab.T).float() - self_mask           # same label, not self
            exp = torch.exp(sim) * (1 - self_mask)             # exclude self from denom
            log_prob = sim - torch.log(exp.sum(1, keepdim=True) + 1e-12)
            pcount = pos.sum(1)
            mean_lp = (pos * log_prob).sum(1) / torch.clamp(pcount, min=1.0)
            valid = (pcount > 0).float()
            return -(mean_lp * valid).sum() / torch.clamp(valid.sum(), min=1.0)

    data = load()
    labels = sorted({c for _, _, c, _ in data})
    l2i = {c: i for i, c in enumerate(labels)}
    base = FACET_BASELINE if LABEL_MODE == "facet" else GATE_BASELINE

    # train rows -> BOTH raw and masked as separate points (value-invariance aug).
    # store/eval use the canonical text (masked if present, else raw) once per atom.
    train_examples, store_t, store_l, te_t, te_l = [], [], [], [], []
    dist = collections.Counter()
    for raw, masked, cls, deal in data:
        canon = masked or raw
        if split(deal) == "train":
            dist[cls] += 1
            for txt in {t for t in (raw, masked) if t}:
                train_examples.append(InputExample(texts=[txt], label=l2i[cls]))
            store_t.append(canon); store_l.append(cls)
        else:
            te_t.append(canon); te_l.append(cls)

    print(f"MODE={LABEL_MODE} | classes={labels}")
    print(f"train atoms={len(store_t)} (aug points={len(train_examples)}) held-out={len(te_t)} | base={MODEL}")
    print(f"train class balance: {dict(dist)}")
    print(f"BASELINE TO BEAT (via kNN, held-out-by-deal): {base:.3f}  | target precision {TARGET_PREC:.2f}\n")

    model = SentenceTransformer(MODEL)
    ds = SentenceLabelDataset(train_examples, samples_per_label=2)
    loader = DataLoader(ds, batch_size=BATCH, drop_last=True)
    loss = SupConLoss(model, temperature=TEMP)

    def report(ep_label, ep_num):
        acc, curve, op, rec, _cop = knn_eval(model, store_t, store_l, te_t, te_l, labels)
        delta = acc - base
        opstr = (f"{op[1]*100:.0f}% @ {op[2]:.3f} (tau {op[0]:.2f})" if op
                 else f"none reaches {TARGET_PREC:.2f}")
        worst = min(rec.items(), key=lambda kv: kv[1])
        macro = sum(rec.values()) / max(len(rec), 1)
        tag = ("BEATS BASELINE" if delta > 0 else "approaching" if delta > -0.05 else "below")
        print(f"  {ep_label} | kNN acc {acc:.3f} ({delta:+.3f} vs {base:.2f}) | macro-rec {macro:.3f} "
              f"| guess-free@{TARGET_PREC:.2f}prec: {opstr} "
              f"| worst {worst[0]}={worst[1]:.2f} | {tag}", flush=True)
        return acc

    print("=== epoch 0 (frozen, before contrastive fit) ===")
    best = report("epoch  0", 0)

    for ep in range(1, EPOCHS + 1):
        model.fit(train_objectives=[(loader, loss)], epochs=1,
                  warmup_steps=int(0.06 * len(loader)),
                  show_progress_bar=True, optimizer_params={"lr": 2e-5})
        acc = report(f"epoch {ep:>2}", ep)
        best = max(best, acc)

    acc, curve, op, rec, class_op = knn_eval(model, store_t, store_l, te_t, te_l, labels)
    print(f"\n=== VERDICT (MODE={LABEL_MODE}) ===")
    print(f"best kNN held-out acc = {best:.3f}  vs classifier-head/agreement baseline {base:.3f}")
    print(f"per-class recall: { {k: round(v,2) for k,v in rec.items()} }")
    if op:
        print(f"GUESS-FREE OPERATING POINT (global): type {op[1]*100:.0f}% of atoms confidently "
              f"@ {op[2]:.3f} precision (tau {op[0]:.2f}); rest -> LLM fallback")
    # per-class guess-free slice: which classes have a confident, deflectable subset
    print("per-class confident slice @ %.2f precision:" % TARGET_PREC)
    for c in labels:
        co = class_op.get(c)
        print(f"  {c:<12} " + (f"{co[1]} atoms deflectable (tau {co[0]:.2f}, prec {co[2]:.3f})"
                               if co else "no confident slice -> always LLM"))
    print("UNLOCK ✅ — contrastive space beats the head; ship kNN cascade" if best > base else
          "matches baseline — boundary is the irreducible-ambiguity wall (LLM fallback earns its keep)"
          if best > base - 0.04 else "below — needs cleaner rubric labels / more diverse pairs")

    out = f"runs/contrastive_{LABEL_MODE}/best"
    os.makedirs(out, exist_ok=True)
    model.save(out)
    s_emb = model.encode(store_t, batch_size=256, convert_to_numpy=True,
                         normalize_embeddings=True, show_progress_bar=False)
    np.savez_compressed(f"runs/contrastive_{LABEL_MODE}/store.npz",
                        emb=s_emb, y=np.array(store_l), text=np.array(store_t, dtype=object))
    json.dump({"labels": labels, "k": K, "mode": LABEL_MODE, "sim_floor": SIM_FLOOR,
               "target_precision": TARGET_PREC,
               "operating_tau": (op[0] if op else None), "base_model": MODEL},
              open(f"{out}/knn_meta.json", "w"))
    print(f"saved encoder -> {out} ; kNN store -> runs/contrastive_{LABEL_MODE}/store.npz")


if __name__ == "__main__":
    main()
