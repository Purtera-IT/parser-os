"""#71 span-extractor: per-relation recall head trains, is recall-tuned, the
eval-gate is recall-monotonic, and SpanExtractorSet identifies items."""
import zlib

import numpy as np
from app.core import span_extractor as SE


def _embed(texts):
    out = []
    for t in texts:
        v = np.zeros(8, dtype=np.float32)
        if "shall" in t.lower() or "must" in t.lower():
            v[0] = 3.0          # requirement signal
# zlib.crc32, not hash(): hash() on a str is salted per process (PEP 456),
# so these components are different noise on every run. A model trained on
# them is trained partly on randomness, and the assertions below then pass
# or fail by luck. crc32 is stable across processes and machines.
        v[3:] = (zlib.crc32(t.encode("utf-8")) % 5) / 5.0
        out.append(v)
    return np.vstack(out)


def _seed(path):
    import sqlite3
    c = sqlite3.connect(path)
    c.execute("""CREATE TABLE training_rows (id TEXT PRIMARY KEY, relation TEXT, label TEXT,
        raw_text TEXT, masked_text TEXT, label_kind TEXT, teacher TEXT, weight REAL,
        deal_id TEXT, project_id TEXT, split TEXT, created_at REAL)""")
    rows = []
    for d in range(10):
        for i in range(12):
            rows.append((f"req{d}_{i}", "requirements", "x", f"vendor shall provide item d{d} n{i}",
                         f"shall provide item d{d} n{i}", "span", "llm", 1.0, f"deal{d}", "p", "train", 0.0))
            rows.append((f"oth{d}_{i}", "atom_type", "_keep", f"plain note line d{d} n{i}",
                         f"plain note d{d} n{i}", "type", "llm", 1.0, f"deal{d}", "p", "train", 0.0))
    c.executemany("INSERT INTO training_rows VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    c.commit(); c.close()


def test_span_head_trains_and_is_recall_tuned(tmp_path):
    db = str(tmp_path / "log.db"); _seed(db)
    h = SE.train_span_head("requirements", log_db=db, embed_fn=_embed)
    assert h is not None
    assert h.metrics.recall >= 0.7 and h.metrics.precision >= 0.8
    assert h.is_item("the contractor shall install conduit")[0] is True
    assert h.is_item("plain unrelated filler")[0] is False


def test_eval_gate_promotes(tmp_path, monkeypatch):
    db = str(tmp_path / "log.db"); _seed(db)
    monkeypatch.setenv("SOWSMITH_SPAN_HEAD_DIR", str(tmp_path / "reg"))
    monkeypatch.setattr("app.core.embedding_retrieval.embed_texts", _embed, raising=False)
    res = SE.retrain_span_heads(["requirements"], log_db=db, min_recall=0.5)
    assert res["requirements"]["status"] == "promoted"
    s = SE.SpanExtractorSet(["requirements"])
    assert "requirements" in s.heads
