"""Every override, recorded against the proposal that produced it (PUR-54).

An override that is only a new number teaches nothing. Each row keeps the pair
(proposed, accepted), the inputs the proposal used, the proposal id AND version
(:func:`app.core.estimator_head.estimator_version`: code version + lesson set),
who changed it, and why -- a typed reason plus free text. One row per FIELD, so
an hours-per-visit override is never confused with a site-count fix.

Readable per deal (:meth:`OverrideLog.for_deal`) and queryable across deals
(:meth:`OverrideLog.query`). SQLite, local path or ``:memory:``; this module
never talks to a remote database. The Platform-infra table
``deal_kit_estimate_overrides`` mirrors these columns.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

#: Typed reasons. "other" requires free text.
REASON_CODES: tuple[str, ...] = (
    "wrong_site_count",
    "wrong_visit_count",
    "wrong_unit_count",
    "after_hours",
    "customer_supplies_equipment",
    "other",
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS estimate_overrides (
    id TEXT PRIMARY KEY,
    deal_id TEXT NOT NULL,
    proposal_id TEXT NOT NULL,
    proposal_version TEXT NOT NULL,
    line_index INTEGER NOT NULL DEFAULT 0,
    field TEXT NOT NULL,
    proposed_value REAL NOT NULL,
    accepted_value REAL NOT NULL,
    inputs TEXT NOT NULL DEFAULT '{}',
    reason_code TEXT NOT NULL,
    reason_text TEXT NOT NULL DEFAULT '',
    actor TEXT NOT NULL,
    lesson_id TEXT,
    key_mode TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_estimate_overrides_deal ON estimate_overrides (deal_id, created_at);
CREATE INDEX IF NOT EXISTS ix_estimate_overrides_field ON estimate_overrides (field, reason_code);
"""


@dataclass
class OverrideRecord:
    deal_id: str
    proposal_id: str
    proposal_version: str
    field: str
    proposed_value: float
    accepted_value: float
    reason_code: str
    actor: str
    line_index: int = 0
    inputs: dict[str, Any] = field(default_factory=dict)
    reason_text: str = ""
    lesson_id: str | None = None
    key_mode: str = ""
    id: str = field(default_factory=lambda: f"ovr_{uuid.uuid4().hex[:16]}")
    created_at: float = field(default_factory=time.time)


def validate(rec: OverrideRecord) -> list[str]:
    from app.core.estimator_head import FIELDS

    problems: list[str] = []
    for name in ("deal_id", "proposal_id", "proposal_version", "actor"):
        if not str(getattr(rec, name) or "").strip():
            problems.append(f"{name} is required")
    if rec.field not in FIELDS:
        problems.append(f"field must be one of {FIELDS} (a total is not correctable)")
    if rec.reason_code not in REASON_CODES:
        problems.append(f"reason_code must be one of {REASON_CODES}")
    if rec.reason_code == "other" and not rec.reason_text.strip():
        problems.append("reason_text is required when reason_code is 'other'")
    try:
        if float(rec.accepted_value) == float(rec.proposed_value):
            problems.append("accepted_value equals proposed_value: not an override")
    except (TypeError, ValueError):
        problems.append("proposed_value and accepted_value must be numbers")
    return problems


class OverrideLog:
    def __init__(self, db_path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    def record(self, rec: OverrideRecord) -> str:
        problems = validate(rec)
        if problems:
            raise ValueError("; ".join(problems))
        d = asdict(rec)
        d["inputs"] = json.dumps(rec.inputs, sort_keys=True, default=str)
        cols = ", ".join(d)
        self._conn.execute(
            f"INSERT INTO estimate_overrides ({cols}) VALUES ({', '.join(':' + k for k in d)})", d
        )
        self._conn.commit()
        return rec.id

    @staticmethod
    def _row(r: sqlite3.Row) -> OverrideRecord:
        d = dict(r)
        d["inputs"] = json.loads(d.get("inputs") or "{}")
        return OverrideRecord(**d)

    def for_deal(self, deal_id: str) -> list[OverrideRecord]:
        rows = self._conn.execute(
            "SELECT * FROM estimate_overrides WHERE deal_id = ? ORDER BY created_at, id", (deal_id,)
        ).fetchall()
        return [self._row(r) for r in rows]

    def query(
        self,
        *,
        field: str | None = None,
        reason_code: str | None = None,
        proposal_version: str | None = None,
    ) -> list[OverrideRecord]:
        where, args = [], []
        for col, val in (("field", field), ("reason_code", reason_code), ("proposal_version", proposal_version)):
            if val is not None:
                where.append(f"{col} = ?")
                args.append(val)
        sql = "SELECT * FROM estimate_overrides"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at, id"
        return [self._row(r) for r in self._conn.execute(sql, args).fetchall()]


__all__ = ["REASON_CODES", "OverrideLog", "OverrideRecord", "validate"]
