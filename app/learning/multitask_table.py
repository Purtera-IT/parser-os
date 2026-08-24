"""Assemble the one-backbone training table from every training DB.

Bet #5: type / facet / route / admission are trained today as separate
contrastive stores, which wastes the one thing this system has plenty of --
CORRELATION between tiny label sets. An atom labelled ``exclusion`` informs
routing informs facets. One task-conditioned encoder over the shared atom
representation is a 3-5x effective-data multiplier at this scale, and it puts
every head in the same embedding space -- which is what lets a feedback-store
correction apply ACROSS heads instead of to one.

This module is the CPU half: gather every labelled row from the training DBs
into one deduplicated, deal-split, per-task table, and report what a training
run would actually see -- so the GPU decision is made on counts, not vibes.
The GPU half (the trainer) consumes this table; nothing here needs a GPU.

Discipline carried over from everything else on this branch:

* the split is BY DEAL, reusing ``assign_split`` -- rows without a deal id
  keep their stored split or default to train; nothing unidentifiable can
  land in the hold-out and flatter it.
* teacher provenance is preserved per row. PM rows are the scarce metal
  (218 of 31,876 locally) and the trainer must be able to weight or hold
  them out separately -- collapsing them into the LLM rows would repeat the
  circularity mistake at training time.
* duplicate (task, text) pairs collapse to ONE row, keeping the most
  trustworthy teacher. The same atom labelled by DeepSeek twice is one fact,
  not two votes.
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

#: Teacher trust order for dedup: a human beats the pipeline beats the LLM.
_TEACHER_RANK = {"pm": 3, "human": 3, "pipeline": 2, "llm": 1, "deepseek": 1, "": 0}

#: The tasks the backbone trains on. Everything else in the DBs (edge tables,
#: span work) has its own machinery and is excluded on purpose.
DEFAULT_TASKS = (
    "atom_type",
    "atom_type_coarse",
    "facet",
    "service_routing",
    "admission",
)


@dataclass(frozen=True)
class TaskRow:
    task: str
    text: str
    label: str
    deal_id: str
    split: str          # "train" | "holdout"
    teacher: str
    source_db: str


@dataclass
class MultitaskTable:
    rows: list[TaskRow] = field(default_factory=list)
    skipped: Counter = field(default_factory=Counter)

    def per_task(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for row in self.rows:
            slot = out.setdefault(row.task, {"train": 0, "holdout": 0, "pm": 0, "classes": 0})
            slot[row.split] += 1
            if _TEACHER_RANK.get(row.teacher.lower(), 0) >= 3:
                slot["pm"] += 1
        for task, slot in out.items():
            slot["classes"] = len({r.label for r in self.rows if r.task == task})
        return out

    def summary(self) -> str:
        lines = [
            f"multitask table: {len(self.rows)} rows across "
            f"{len({r.task for r in self.rows})} tasks",
            "",
            f"  {'task':<20} {'train':>7} {'holdout':>8} {'classes':>8} {'pm-gold':>8}",
        ]
        for task, s in sorted(self.per_task().items()):
            lines.append(
                f"  {task:<20} {s['train']:>7} {s['holdout']:>8} "
                f"{s['classes']:>8} {s['pm']:>8}"
            )
        if self.skipped:
            lines += ["", "  skipped (named, not silent):"]
            for reason, n in self.skipped.most_common():
                lines.append(f"    {reason}: {n}")
        return "\n".join(lines)

    def write(self, target: Path) -> int:
        """One SQLite the GPU trainer mounts. Idempotent: replaces the table."""
        conn = sqlite3.connect(target)
        try:
            conn.execute("DROP TABLE IF EXISTS multitask_rows")
            conn.execute(
                "CREATE TABLE multitask_rows ("
                "task TEXT, text TEXT, label TEXT, deal_id TEXT, "
                "split TEXT, teacher TEXT, source_db TEXT)"
            )
            conn.executemany(
                "INSERT INTO multitask_rows VALUES (?,?,?,?,?,?,?)",
                [(r.task, r.text, r.label, r.deal_id, r.split, r.teacher, r.source_db)
                 for r in self.rows],
            )
            conn.commit()
            return len(self.rows)
        finally:
            conn.close()


def assemble(
    db_paths: Iterable[Path],
    *,
    tasks: tuple[str, ...] = DEFAULT_TASKS,
    min_text_chars: int = 8,
) -> MultitaskTable:
    """Gather, dedup, and split-preserve every labelled row for ``tasks``."""
    table = MultitaskTable()
    best: dict[tuple[str, str], TaskRow] = {}

    for db_path in db_paths:
        db_path = Path(db_path)
        if not db_path.exists():
            table.skipped[f"missing db {db_path.name}"] += 1
            continue
        try:
            conn = sqlite3.connect(db_path)
            names = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )}
            if "training_rows" not in names:
                table.skipped[f"{db_path.name}: no training_rows table"] += 1
                conn.close()
                continue
            cols = {r[1] for r in conn.execute("PRAGMA table_info(training_rows)")}
            wanted = ["relation", "label", "raw_text", "deal_id", "split", "teacher"]
            select = ", ".join(c if c in cols else "''" for c in wanted)
            cursor = conn.execute(f"SELECT {select} FROM training_rows")
            for relation, label, raw_text, deal_id, split, teacher in cursor:
                relation = str(relation or "")
                if relation not in tasks:
                    table.skipped[f"relation {relation} not a backbone task"] += 1
                    continue
                text = str(raw_text or "").strip()
                if len(text) < min_text_chars:
                    table.skipped["text too short"] += 1
                    continue
                label = str(label or "").strip()
                if not label:
                    table.skipped["empty label"] += 1
                    continue
                split = str(split or "").strip() or _fallback_split(str(deal_id or ""))
                row = TaskRow(
                    task=relation, text=text, label=label,
                    deal_id=str(deal_id or ""), split=split,
                    teacher=str(teacher or ""), source_db=db_path.name,
                )
                key = (relation, text)
                held = best.get(key)
                if held is None or (
                    _TEACHER_RANK.get(row.teacher.lower(), 0)
                    > _TEACHER_RANK.get(held.teacher.lower(), 0)
                ):
                    if held is not None:
                        table.skipped["duplicate (kept most trusted teacher)"] += 1
                    best[key] = row
                else:
                    table.skipped["duplicate (kept most trusted teacher)"] += 1
            conn.close()
        except Exception as exc:  # noqa: BLE001 - one bad DB must not sink the table
            table.skipped[f"{db_path.name}: {type(exc).__name__}"] += 1

    table.rows = list(best.values())
    return table


def _fallback_split(deal_id: str) -> str:
    try:
        from app.core.training_log import assign_split

        return assign_split(deal_id)
    except Exception:  # noqa: BLE001
        return "train"


if __name__ == "__main__":  # pragma: no cover - operator entry point
    import glob

    dbs = [Path(p) for p in glob.glob("_training_*.db")]
    table = assemble(dbs)
    print(table.summary())
    written = table.write(Path("_multitask_table.db"))
    print(f"\nwrote {written} rows -> _multitask_table.db")
