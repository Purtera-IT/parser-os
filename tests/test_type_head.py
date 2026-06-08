"""Learnable #70 type-deflector: trains from the log, deflects confident typed
predictions, abstains on _keep/low-confidence, and the eval-gate is monotonic."""
import os, sqlite3, numpy as np
from app.core import type_head as TH


def _fake_embed(texts):
    """Deterministic, label-separable embedding: a token -> a basis direction."""
    toks = ["shall", "exclude", "neutral"]
    out = []
    for t in texts:
        v = np.zeros(8, dtype=np.float32)
        low = t.lower()
        for i, tok in enumerate(toks):
            if tok in low:
                v[i] = 3.0
        v[3:] = (hash(t) % 7) / 7.0  # mild noise
        out.append(v)
    return np.vstack(out)


def _seed_log(path):
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE training_rows (id TEXT PRIMARY KEY, relation TEXT, label TEXT,
        raw_text TEXT, masked_text TEXT, label_kind TEXT, teacher TEXT, weight REAL,
        deal_id TEXT, project_id TEXT, split TEXT, created_at REAL)""")
    rows = []
    for d in range(10):
        for i in range(12):
            rows.append((f"r{d}_{i}_req", "atom_type", "requirement",
                         f"vendor shall provide item {i}", f"shall provide item {i}",
                         "type", "llm", 1.0, f"deal{d}", "p", "train", 0.0))
            rows.append((f"r{d}_{i}_exc", "atom_type", "exclusion",
                         f"this excludes thing {i}", f"exclude thing {i}",
                         "type", "llm", 1.0, f"deal{d}", "p", "train", 0.0))
            rows.append((f"r{d}_{i}_keep", "atom_type", "_keep",
                         f"neutral filler line {i}", f"neutral filler {i}",
                         "type", "llm", 1.0, f"deal{d}", "p", "train", 0.0))
    con.executemany("INSERT INTO training_rows VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    con.commit(); con.close()


def test_head_trains_deflects_and_abstains(tmp_path):
    db = str(tmp_path / "log.db")
    _seed_log(db)
    head = TH.train_type_head(log_db=db, embed_fn=_fake_embed, threshold=0.6)
    assert head is not None
    assert head.metrics.deflect_precision >= 0.8       # learns a clean boundary
    # deflects a confident specific type
    res = head.classify("the vendor shall provide cabling")
    assert res is not None and res[0] == "requirement" and res[1] >= 0.6
    # abstains on a _keep-ish line (guess-free -> falls back to LLM)
    assert head.classify("neutral filler text here") is None


def test_eval_gate_promotes_then_is_stale(tmp_path, monkeypatch):
    db = str(tmp_path / "log.db")
    _seed_log(db)
    monkeypatch.setenv("SOWSMITH_TYPE_HEAD_DIR", str(tmp_path / "reg"))
    monkeypatch.setenv("SOWSMITH_TRAINING_LOG_DB", db)
    monkeypatch.setattr("app.core.embedding_retrieval.embed_texts", _fake_embed, raising=False)
    r1 = TH.retrain_if_stale(min_growth=0, min_precision=0.7)
    assert r1["status"] == "promoted"
    # no new rows -> staleness gate is a no-op (doesn't waste compute)
    r2 = TH.retrain_if_stale(min_growth=300, min_precision=0.7)
    assert r2["status"] == "fresh"
