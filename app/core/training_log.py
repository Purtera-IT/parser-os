"""Training log — the dataset the Grounded Extractor heads learn from.

Every judgment the system makes that *could* one day be served by a trained
head is recorded here as a labeled row:

* the **LLM teacher**'s extraction outputs and atom-type verdicts (silver
  labels — abundant, lower weight), and
* the **PM**'s natural-language corrections (gold labels — scarce, high
  weight, the real signal).

Nothing trains yet (that is tasks #70-72). This module is the *foundation*:
it is pure logging — **zero behavior change** to the compile path — but it is
the prerequisite for replacing the LLM, because a student head can only be as
good as the data we accumulate for it.

Three design guarantees:

* **Generalization-first.** We store the **delexicalized** text
  (:mod:`app.core.delexicalize`) as the training feature, never relying on the
  raw identity. ``raw_text`` is kept only for audit. So the dataset itself
  teaches the *rule*, not the name. The applied role-map is stored in
  ``provenance`` for reversibility.
* **Honest eval baked in.** Each row's ``split`` is derived from a hash of its
  ``deal_id`` (not random), so all rows from one deal land in the same split.
  That gives leave-one-deal-out evaluation for free: a head is only credited
  with "learning" when it works on deals whose names it never trained on.
* **Durable / shippable.** Like the feedback store, this DB is meant to ride
  along with the deploy as a warm base (point ``SOWSMITH_TRAINING_LOG_DB`` at a
  persistent file). In-memory (``:memory:``) is the test/default-off mode.

Never raises into the caller: :func:`log_rows` swallows everything (a logging
failure must never break a compile).
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from app.core.delexicalize import delexicalize

# Teacher provenance — who produced the label.
TEACHER_LLM = "llm"      # silver: the big model's output
TEACHER_PM = "pm"        # gold: a human correction (natural language)
TEACHER_STORE = "store"  # the kNN store fired (already-learned)

# Default training-row weights by teacher. PM gold dominates LLM silver so the
# head can't be drowned by abundant (and occasionally wrong) teacher labels.
_DEFAULT_WEIGHT = {TEACHER_PM: 5.0, TEACHER_LLM: 1.0, TEACHER_STORE: 2.0}

# Fraction of deals routed to the held-out split (by deal-id hash).
_HOLDOUT_FRACTION = 0.2


@dataclass
class TrainingRow:
    """One labeled example for the extractor heads."""

    relation: str          # extractor/decision key: "atom_type", "payment_terms", ...
    label: str             # the verdict / type / normalized value
    raw_text: str          # original span (AUDIT ONLY — never the sole feature)
    masked_text: str = ""  # delexicalized feature (generalization-safe)
    label_kind: str = "type"   # "type" | "span" | "norm" | "judgment"
    teacher: str = TEACHER_LLM
    weight: float = 1.0
    confidence: float = 0.0
    scope: str = "global"
    scope_key: str = ""
    deal_id: str = ""
    project_id: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)  # model, role_map, offsets, ...
    complaint_id: str | None = None
    id: str = field(default_factory=lambda: "trn_" + uuid.uuid4().hex[:16])
    created_at: float = field(default_factory=time.time)
    split: str = "train"   # "train" | "holdout" — assigned by deal-id hash

    def to_row(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["provenance"] = json.dumps(self.provenance, ensure_ascii=False)
        return d


_SCHEMA = """
CREATE TABLE IF NOT EXISTS training_rows (
    id TEXT PRIMARY KEY,
    relation TEXT NOT NULL,
    label TEXT NOT NULL,
    raw_text TEXT NOT NULL DEFAULT '',
    masked_text TEXT NOT NULL DEFAULT '',
    label_kind TEXT NOT NULL DEFAULT 'type',
    teacher TEXT NOT NULL DEFAULT 'llm',
    weight REAL NOT NULL DEFAULT 1.0,
    confidence REAL NOT NULL DEFAULT 0.0,
    scope TEXT NOT NULL DEFAULT 'global',
    scope_key TEXT NOT NULL DEFAULT '',
    deal_id TEXT NOT NULL DEFAULT '',
    project_id TEXT NOT NULL DEFAULT '',
    provenance TEXT NOT NULL DEFAULT '{}',
    complaint_id TEXT,
    created_at REAL NOT NULL DEFAULT 0,
    split TEXT NOT NULL DEFAULT 'train'
);
CREATE INDEX IF NOT EXISTS ix_training_rows_relation ON training_rows(relation);
CREATE INDEX IF NOT EXISTS ix_training_rows_split ON training_rows(split);
CREATE INDEX IF NOT EXISTS ix_training_rows_teacher ON training_rows(teacher);
"""

_COLUMNS = [
    "id", "relation", "label", "raw_text", "masked_text", "label_kind",
    "teacher", "weight", "confidence", "scope", "scope_key", "deal_id",
    "project_id", "provenance", "complaint_id", "created_at", "split",
]


def assign_split(deal_id: str, *, holdout_fraction: float = _HOLDOUT_FRACTION) -> str:
    """Deterministically route a deal to train/holdout by hashing its id.

    Hash-based (not random) so every row from a deal lands in the same split
    across runs — the precondition for leave-one-deal-out evaluation. A deal
    with no id is treated as train (can't hold out the unidentifiable).
    """
    if not deal_id:
        return "train"
    h = hashlib.sha256(deal_id.encode("utf-8")).digest()
    # First 4 bytes → [0,1)
    frac = int.from_bytes(h[:4], "big") / 0xFFFFFFFF
    return "holdout" if frac < holdout_fraction else "train"


class TrainingLog:
    """Append-only labeled-example store. See module docstring."""

    def __init__(self, db_path: str = ":memory:") -> None:
        # check_same_thread=False: written from the compiler worker threads and
        # the FastAPI PM-feedback threadpool, mirroring FeedbackStore.
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    # ── write path ──────────────────────────────────────────────────────
    def add(self, row: TrainingRow) -> None:
        self.add_many([row])

    def add_many(self, rows: list[TrainingRow]) -> int:
        """Insert rows; fills masked_text/weight/split when unset. Returns count."""
        prepared: list[dict[str, Any]] = []
        for r in rows:
            if not r.masked_text and r.raw_text:
                role_map = {}
                if isinstance(r.provenance, dict):
                    role_map = r.provenance.get("role_map") or {}
                dl = delexicalize(r.raw_text, role_map or None)
                r.masked_text = dl.masked
                if isinstance(r.provenance, dict) and dl.substitutions and "role_map" not in r.provenance:
                    r.provenance = {**r.provenance, "role_map": dl.role_map}
            if not r.weight or r.weight == 1.0:
                r.weight = _DEFAULT_WEIGHT.get(r.teacher, 1.0)
            r.split = assign_split(r.deal_id)
            prepared.append(r.to_row())
        if not prepared:
            return 0
        placeholders = ", ".join("?" for _ in _COLUMNS)
        sql = f"INSERT OR REPLACE INTO training_rows ({', '.join(_COLUMNS)}) VALUES ({placeholders})"
        with self._conn:
            self._conn.executemany(sql, [[p[c] for c in _COLUMNS] for p in prepared])
        return len(prepared)

    # ── read / inspect ──────────────────────────────────────────────────
    def count(self, *, relation: str | None = None, split: str | None = None,
              teacher: str | None = None) -> int:
        clauses, args = [], []
        if relation:
            clauses.append("relation = ?"); args.append(relation)
        if split:
            clauses.append("split = ?"); args.append(split)
        if teacher:
            clauses.append("teacher = ?"); args.append(teacher)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        cur = self._conn.execute(f"SELECT COUNT(*) FROM training_rows{where}", args)
        return int(cur.fetchone()[0])

    def rows(self, *, relation: str | None = None, split: str | None = None,
             teacher: str | None = None, limit: int | None = None) -> list[TrainingRow]:
        clauses, args = [], []
        if relation:
            clauses.append("relation = ?"); args.append(relation)
        if split:
            clauses.append("split = ?"); args.append(split)
        if teacher:
            clauses.append("teacher = ?"); args.append(teacher)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        lim = f" LIMIT {int(limit)}" if limit else ""
        cur = self._conn.execute(
            f"SELECT * FROM training_rows{where} ORDER BY created_at{lim}", args
        )
        out: list[TrainingRow] = []
        for r in cur.fetchall():
            d = dict(r)
            d["provenance"] = json.loads(d.get("provenance") or "{}")
            out.append(TrainingRow(**d))
        return out

    def summary(self) -> dict[str, Any]:
        """A glanceable census: rows by relation × teacher × split."""
        cur = self._conn.execute(
            "SELECT relation, teacher, split, COUNT(*) AS n "
            "FROM training_rows GROUP BY relation, teacher, split ORDER BY relation"
        )
        by_relation: dict[str, dict[str, int]] = {}
        total = 0
        for row in cur.fetchall():
            rel, teacher, split, n = row["relation"], row["teacher"], row["split"], int(row["n"])
            d = by_relation.setdefault(rel, {})
            d[f"{teacher}/{split}"] = d.get(f"{teacher}/{split}", 0) + n
            total += n
        return {"total": total, "by_relation": by_relation}

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass


# ── process-wide singleton + safe, env-gated logging entrypoint ─────────────
_LOG: TrainingLog | None = None


def get_training_log() -> TrainingLog | None:
    """Return the process log iff ``SOWSMITH_TRAINING_LOG_DB`` is set.

    Default-off: with no env var there is no log and :func:`log_rows` is a
    no-op, so production behavior is byte-identical until we opt in.
    """
    global _LOG
    if _LOG is not None:
        return _LOG
    db = os.environ.get("SOWSMITH_TRAINING_LOG_DB")
    if not db:
        return None
    try:
        _LOG = TrainingLog(db)
    except Exception:
        _LOG = None
    return _LOG


def set_training_log(log: TrainingLog | None) -> None:
    """Inject a log (tests / explicit wiring)."""
    global _LOG
    _LOG = log


def log_rows(rows: list[TrainingRow]) -> int:
    """Safe entrypoint for the compile path: never raises, no-op when off.

    Wire this at the LLM extractor / classifier tap points and the PM
    correction intake. Returns rows written (0 when logging is off/failed).
    """
    if not rows:
        return 0
    log = get_training_log()
    if log is None:
        return 0
    try:
        return log.add_many(rows)
    except Exception:
        return 0
