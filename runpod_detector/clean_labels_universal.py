"""Universal, guess-free atom_type label cleaning via embedding confident-learning.

The noise is SEMANTIC (measured: only 0.4% of rows are exact-text conflicts, but
100% of conflicts touch _keep) — the teacher labels similar-but-different atoms
inconsistently on the _keep<->typed axis. So we clean using the data's own
geometry, with NO per-deal / per-keyword / per-model rules:

  1. Embed every atom with qwen3-embedding:8b (the same 8B encoder behind the
     0.65 baseline) via local Ollama. Cache to disk.
  2. For each TRAIN atom, take its k nearest TRAIN neighbors (cosine) and form a
     similarity-weighted label vote = its semantic neighborhood's consensus.
  3. If the neighborhood STRONGLY disagrees with the atom's label
     (consensus weight >= AGREE and the atom's own label has <= SELFMAX support),
     the label is unreliable -> DROP the row. We never relabel (that would be
     guessing); we only remove contradictory signal. Guess-free by construction.

Held-out (test deals) rows are NEVER touched, so downstream eval stays honest.
Writes a cleaned copy of the DB for the v2 trainer to consume via
SOWSMITH_TRAINING_LOG_DB.

Run:  python runpod_detector/clean_labels_universal.py
Env:  OLLAMA=http://localhost:11434  EMB_MODEL=qwen3-embedding:8b
      K=15  AGREE=0.80  SELFMAX=0.15  HOLDOUT=0.25  OUT_DB=_training_clean.db
"""
import os, sqlite3, hashlib, re, json, collections, time, urllib.request
import numpy as np

DB = os.environ.get("SOWSMITH_TRAINING_LOG_DB", "_training_deepseek.db")
OUT_DB = os.environ.get("OUT_DB", "_training_clean.db")
OLLAMA = os.environ.get("OLLAMA", "http://localhost:11434")
EMB_MODEL = os.environ.get("EMB_MODEL", "qwen3-embedding:8b")
CACHE = os.environ.get("EMB_CACHE", "_atom_emb_cache.npz")
K = int(os.environ.get("K", "15"))
AGREE = float(os.environ.get("AGREE", "0.80"))
SELFMAX = float(os.environ.get("SELFMAX", "0.15"))
HOLDOUT = float(os.environ.get("HOLDOUT", "0.25"))


def norm(t):
    return re.sub(r"\s+", " ", (t or "").strip().lower())


def split(deal_id):
    h = int(hashlib.sha256((deal_id or "").encode()).hexdigest(), 16)
    return "test" if (h % 100) / 100.0 < HOLDOUT else "train"


def embed_batch(texts):
    body = json.dumps({"model": EMB_MODEL, "input": texts}).encode()
    req = urllib.request.Request(f"{OLLAMA}/api/embed", data=body,
                                 headers={"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())["embeddings"]
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(2)


def get_embeddings(uniq):
    if os.path.exists(CACHE):
        d = np.load(CACHE, allow_pickle=True)
        if list(d["texts"]) == uniq:
            print(f"  loaded cached embeddings ({len(uniq)})")
            return d["emb"]
    print(f"  embedding {len(uniq)} unique texts via {EMB_MODEL} ...")
    embs = []
    B = 64
    t0 = time.time()
    for i in range(0, len(uniq), B):
        embs.extend(embed_batch(uniq[i:i+B]))
        if (i // B) % 10 == 0:
            print(f"    {i+B}/{len(uniq)}  ({time.time()-t0:.0f}s)", flush=True)
    emb = np.asarray(embs, dtype=np.float32)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    np.savez(CACHE, texts=np.array(uniq, dtype=object), emb=emb)
    return emb


def main():
    con = sqlite3.connect(DB)
    rows = con.execute(
        "SELECT id, COALESCE(NULLIF(masked_text,''),raw_text) AS t, label, deal_id "
        "FROM training_rows WHERE relation='atom_type' AND COALESCE(masked_text,raw_text,'')!='' "
        "AND label IS NOT NULL").fetchall()
    con.close()

    rid = [r[0] for r in rows]
    txt = [norm(r[1]) for r in rows]
    lab = [r[2] for r in rows]
    spl = [split(r[3]) for r in rows]

    uniq = sorted(set(txt))
    u2i = {t: i for i, t in enumerate(uniq)}
    emb = get_embeddings(uniq)

    # per-unique-text majority TRAIN label + train count
    train_lab = collections.defaultdict(collections.Counter)
    for t, l, s in zip(txt, lab, spl):
        if s == "train":
            train_lab[t][l] += 1
    train_uids = [u2i[t] for t in uniq if t in train_lab]
    maj = {uid: train_lab[uniq[uid]].most_common(1)[0][0] for uid in train_uids}
    cnt = {uid: sum(train_lab[uniq[uid]].values()) for uid in train_uids}

    U = emb[train_uids]                          # (n_tr_uniq, d)
    labels = sorted({maj[uid] for uid in train_uids})
    l2i = {l: i for i, l in enumerate(labels)}
    y = np.array([l2i[maj[uid]] for uid in train_uids])
    w = np.array([cnt[uid] for uid in train_uids], dtype=np.float32)

    print(f"  confident-learning over {len(train_uids)} train unique texts (k={K}) ...")
    flagged_uids = set()
    drop_axis = collections.Counter()
    BL = 512
    for s in range(0, len(train_uids), BL):
        block = U[s:s+BL]
        sims = block @ U.T                       # (bl, n)
        for bi in range(block.shape[0]):
            gi = s + bi
            sims[bi, gi] = -1                     # exclude self
            nn = np.argpartition(-sims[bi], K)[:K]
            sw = np.clip(sims[bi, nn], 0, None) * w[nn]
            if sw.sum() <= 0:
                continue
            vote = collections.defaultdict(float)
            for j, weight in zip(nn, sw):
                vote[y[j]] += weight
            tot = sum(vote.values())
            best_lbl, best_w = max(vote.items(), key=lambda kv: kv[1])
            self_w = vote.get(y[gi], 0.0) / tot
            if best_lbl != y[gi] and (best_w / tot) >= AGREE and self_w <= SELFMAX:
                flagged_uids.add(train_uids[gi])
                a, b = labels[y[gi]], labels[best_lbl]
                axis = "_keep<->typed" if "_keep" in (a, b) else "typed<->typed"
                drop_axis[axis] += 1

    flagged_txt = {uniq[uid] for uid in flagged_uids}
    drop_rows = [rid[i] for i in range(len(rows)) if spl[i] == "train" and txt[i] in flagged_txt]
    n_train = sum(1 for s in spl if s == "train")
    print(f"\nflagged unique texts : {len(flagged_uids)}")
    print(f"  by axis            : {dict(drop_axis)}")
    print(f"train rows dropped   : {len(drop_rows)} / {n_train} = {len(drop_rows)/n_train:.1%}")
    print(f"held-out rows        : UNTOUCHED (honest eval preserved)")

    # write cleaned DB
    import shutil
    shutil.copy2(DB, OUT_DB)
    con = sqlite3.connect(OUT_DB)
    con.executemany("DELETE FROM training_rows WHERE id=?", [(i,) for i in drop_rows])
    con.commit()
    left = con.execute("SELECT COUNT(*) FROM training_rows WHERE relation='atom_type'").fetchone()[0]
    con.close()
    print(f"\nwrote {OUT_DB}: atom_type rows now {left}")
    print(f"train with:  SOWSMITH_TRAINING_LOG_DB={OUT_DB} python runpod_detector/train_type_head_v2.py")


if __name__ == "__main__":
    main()
