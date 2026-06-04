"""Shadow history — the metrics-over-time ledger that tells us *when* to cut over.

:mod:`app.core.shadow_eval` answers "is the student ready *right now*?" for each
relation. But readiness is not a coin flip we re-check in a vacuum — it is a
**trend**. A relation crosses the cutover bar when enough deals have accumulated
that the student generalizes to unseen names, and the only honest way to know
that moment arrived is to watch the curve: accuracy / coverage / gold-accuracy /
holdout-size plotted against the growing row count, deal after deal.

This module is that curve. Every time we shadow-evaluate (typically right after a
compile seeds new rows), we snapshot :func:`shadow_eval.evaluate_all`'s verdict —
one row per relation — into an **append-only** SQLite store. Nothing is ever
overwritten; the table *is* the time series. Reading it back gives:

* :meth:`ShadowHistory.trend` — the full chronological curve for one relation, so
  you can see accuracy climb as rows accrue.
* :meth:`ShadowHistory.latest` — the most recent snapshot per relation (the
  current dashboard).
* :meth:`ShadowHistory.first_ready` — the timestamp a relation *first* cleared
  every bar, i.e. the moment cutover (#70/#71) became defensible.

Design, deliberately mirroring :mod:`app.core.training_log`:

* **Append-only / durable.** Point ``SOWSMITH_SHADOW_HISTORY_DB`` at a persistent
  file to let the curve ride along with the deploy; ``:memory:`` is the
  test/default-off mode.
* **Env-gated, zero behavior change.** With no env var there is no store and
  :func:`record` is a no-op — the compile path is byte-identical until we opt in.
* **Never raises into the caller.** Recording a metric must never break a compile;
  :func:`record` swallows everything.

This module only *reads* shadow reports and *writes* its own ledger. It never
touches the compile path or mutates the training log.
"""

from __future__ import annotations

import os
import sqlite3
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # avoid import cost / cycles on the default-off path
    from app.core.shadow_eval import RelationReport
    from app.core.training_log import TrainingLog


_SCHEMA = """
CREATE TABLE IF NOT EXISTS shadow_snapshots (
    id TEXT PRIMARY KEY,
    snapshot_id TEXT NOT NULL,
    created_at REAL NOT NULL DEFAULT 0,
    label TEXT NOT NULL DEFAULT '',
    relation TEXT NOT NULL,
    n_rows INTEGER NOT NULL DEFAULT 0,
    n_holdout INTEGER NOT NULL DEFAULT 0,
    n_answered INTEGER NOT NULL DEFAULT 0,
    coverage REAL NOT NULL DEFAULT 0,
    accuracy REAL NOT NULL DEFAULT 0,
    gold_accuracy REAL NOT NULL DEFAULT 0,
    ready INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_shadow_snapshots_relation ON shadow_snapshots(relation, created_at);
CREATE INDEX IF NOT EXISTS ix_shadow_snapshots_sid ON shadow_snapshots(snapshot_id);
"""

_COLUMNS = [
    "id", "snapshot_id", "created_at", "label", "relation", "n_rows",
    "n_holdout", "n_answered", "coverage", "accuracy", "gold_accuracy", "ready",
]


@dataclass
class SnapshotRow:
    """One relation's metrics at one point in time (a read-back record)."""

    relation: str
    snapshot_id: str
    created_at: float
    label: str = ""
    n_rows: int = 0
    n_holdout: int = 0
    n_answered: int = 0
    coverage: float = 0.0
    accuracy: float = 0.0
    gold_accuracy: float = 0.0
    ready: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "relation": self.relation,
            "snapshot_id": self.snapshot_id,
            "created_at": self.created_at,
            "label": self.label,
            "n_rows": self.n_rows,
            "n_holdout": self.n_holdout,
            "n_answered": self.n_answered,
            "coverage": round(self.coverage, 4),
            "accuracy": round(self.accuracy, 4),
            "gold_accuracy": round(self.gold_accuracy, 4),
            "ready": self.ready,
        }


class ShadowHistory:
    """Append-only time series of per-relation shadow metrics. See module docstring."""

    def __init__(self, db_path: str = ":memory:") -> None:
        # check_same_thread=False mirrors TrainingLog / FeedbackStore: written
        # from worker threads and the FastAPI threadpool.
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    # ── write path ──────────────────────────────────────────────────────
    def snapshot(
        self,
        reports: dict[str, "RelationReport"],
        *,
        log: Optional["TrainingLog"] = None,
        label: str = "",
        created_at: float | None = None,
    ) -> str:
        """Append one snapshot — a row per relation — and return its snapshot_id.

        ``log`` (optional) is used only to capture the *total* row count per
        relation (train + holdout) at snapshot time, so the curve can be read
        against dataset growth. Reports themselves carry the holdout-only counts.
        """
        sid = "snp_" + uuid.uuid4().hex[:16]
        ts = time.time() if created_at is None else created_at
        prepared: list[list[object]] = []
        for rel, rep in reports.items():
            n_rows = 0
            if log is not None:
                try:
                    n_rows = log.count(relation=rel)
                except Exception:
                    n_rows = 0
            prepared.append([
                "snr_" + uuid.uuid4().hex[:16],
                sid,
                ts,
                label,
                rel,
                int(n_rows),
                int(rep.n_holdout),
                int(rep.n_answered),
                float(rep.coverage),
                float(rep.accuracy),
                float(rep.gold_accuracy),
                1 if rep.ready() else 0,
            ])
        if prepared:
            placeholders = ", ".join("?" for _ in _COLUMNS)
            sql = f"INSERT INTO shadow_snapshots ({', '.join(_COLUMNS)}) VALUES ({placeholders})"
            with self._conn:
                self._conn.executemany(sql, prepared)
        return sid

    # ── read / inspect ──────────────────────────────────────────────────
    def _row(self, r: sqlite3.Row) -> SnapshotRow:
        return SnapshotRow(
            relation=r["relation"],
            snapshot_id=r["snapshot_id"],
            created_at=float(r["created_at"]),
            label=r["label"],
            n_rows=int(r["n_rows"]),
            n_holdout=int(r["n_holdout"]),
            n_answered=int(r["n_answered"]),
            coverage=float(r["coverage"]),
            accuracy=float(r["accuracy"]),
            gold_accuracy=float(r["gold_accuracy"]),
            ready=bool(r["ready"]),
        )

    def relations(self) -> list[str]:
        cur = self._conn.execute(
            "SELECT DISTINCT relation FROM shadow_snapshots ORDER BY relation"
        )
        return [r[0] for r in cur.fetchall()]

    def trend(self, relation: str) -> list[SnapshotRow]:
        """The full chronological curve for one relation (oldest → newest)."""
        cur = self._conn.execute(
            "SELECT * FROM shadow_snapshots WHERE relation = ? "
            "ORDER BY created_at, snapshot_id",
            [relation],
        )
        return [self._row(r) for r in cur.fetchall()]

    def latest(self) -> dict[str, SnapshotRow]:
        """The most recent snapshot per relation — the current dashboard."""
        cur = self._conn.execute(
            "SELECT * FROM shadow_snapshots ORDER BY created_at, snapshot_id"
        )
        out: dict[str, SnapshotRow] = {}
        for r in cur.fetchall():
            row = self._row(r)
            out[row.relation] = row  # later rows overwrite → last wins
        return out

    def first_ready(self, relation: str) -> Optional[float]:
        """Timestamp the relation *first* cleared every bar, or None if never.

        This is the moment cutover became defensible — the headline the tracker
        exists to surface.
        """
        cur = self._conn.execute(
            "SELECT MIN(created_at) FROM shadow_snapshots "
            "WHERE relation = ? AND ready = 1",
            [relation],
        )
        v = cur.fetchone()[0]
        return float(v) if v is not None else None

    def snapshot_count(self) -> int:
        """Number of distinct snapshot runs recorded."""
        cur = self._conn.execute(
            "SELECT COUNT(DISTINCT snapshot_id) FROM shadow_snapshots"
        )
        return int(cur.fetchone()[0])

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass


# ── process-wide singleton + safe, env-gated recording entrypoint ───────────
_HISTORY: ShadowHistory | None = None


def get_shadow_history() -> ShadowHistory | None:
    """Return the process history iff ``SOWSMITH_SHADOW_HISTORY_DB`` is set.

    Default-off: with no env var there is no store and :func:`record` is a
    no-op, so production behavior is byte-identical until we opt in.
    """
    global _HISTORY
    if _HISTORY is not None:
        return _HISTORY
    db = os.environ.get("SOWSMITH_SHADOW_HISTORY_DB")
    if not db:
        return None
    try:
        _HISTORY = ShadowHistory(db)
    except Exception:
        _HISTORY = None
    return _HISTORY


def set_shadow_history(history: ShadowHistory | None) -> None:
    """Inject a history store (tests / explicit wiring)."""
    global _HISTORY
    _HISTORY = history


def record(
    reports: dict[str, "RelationReport"],
    *,
    log: Optional["TrainingLog"] = None,
    label: str = "",
) -> str | None:
    """Snapshot ``reports`` into the process history. No-op (returns None) when
    unconfigured. Never raises — a metrics-logging failure must not break a
    compile."""
    try:
        hist = get_shadow_history()
        if hist is None:
            return None
        return hist.snapshot(reports, log=log, label=label)
    except Exception:
        return None


__all__ = [
    "ShadowHistory",
    "SnapshotRow",
    "get_shadow_history",
    "set_shadow_history",
    "record",
]
