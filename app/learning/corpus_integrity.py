"""What Postgres holds, what the blob carries, what the corpus admits.

Four times in one day a field was written correctly and silently dropped one
step later:

  * `doc.judgments = []` stubbed in the mirror -- 94 verdicts written and lost;
  * `buildBlobDoc` maps explicit fields, so `consumer`, `rejected`,
    `decided_by`, `weight_tier` and `rejected_reads` reached Postgres and
    stopped;
  * the judgment `reason` the same way;
  * `DEFAULT_TASKS` never listed the judgement heads, so 892 of 1,241 rows were
    dropped one step from the model.

Every one of those produced perfect data that arrived nowhere, and nothing
complained. Counting rows would have caught none of them: the rows were all
there, with a field missing or a relation unlisted.

So this compares the three stages FIELD BY FIELD and RELATION BY RELATION, and
names what falls between them. It is cheap enough to run on every deal after
labelling, and it is the only check that fails when somebody adds a column and
forgets the mapper.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Columns that exist in Postgres for bookkeeping and are not supervision.
#: Anything NOT here must reach the blob, or it is a silent drop.
_NOT_SUPERVISION = frozenset({
    "id", "deal_id", "labeler", "labeled_at", "answered_at", "judged_at",
    "created_at", "purpose", "correction_id", "coarse",
})

#: Relations held back from the classifier on purpose. A span is an extraction
#: problem and a rationale is a generative one.
_HELD_BACK = ("evidence_span:", "reads_value:", "rationale:")


@dataclass
class IntegrityReport:
    dropped_fields: list[str] = field(default_factory=list)
    dropped_rows: list[str] = field(default_factory=list)
    unaccounted: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not (self.dropped_fields or self.dropped_rows or self.unaccounted)

    def __str__(self) -> str:
        lines = [f"counts: {self.counts}"]
        for name, items in (("fields dropped on the way to blob", self.dropped_fields),
                            ("records dropped on the way to blob", self.dropped_rows),
                            ("rows admitted nowhere", self.unaccounted)):
            lines.append(f"{name}: {len(items)}")
            lines.extend(f"    {x}" for x in items[:12])
        return "\n".join(lines)


def check_fields(pg_records: list[dict[str, Any]], blob_records: list[dict[str, Any]],
                 *, what: str) -> list[str]:
    """Every column a labeler filled must appear in the blob document.

    Field by field, because the drops were fields. A record count matches
    perfectly while `consumer` quietly never leaves the database.
    """
    out: list[str] = []
    if not pg_records:
        return out
    blob_keys: set[str] = set()
    for b in blob_records or []:
        blob_keys |= set(b or {})
    filled: set[str] = set()
    for r in pg_records:
        for k, v in (r or {}).items():
            if k in _NOT_SUPERVISION:
                continue
            if v is None or v == "" or v == [] or v == {}:
                continue
            filled.add(k)
    for k in sorted(filled - blob_keys):
        out.append(f"{what}.{k} is filled in Postgres and absent from the blob document")
    return out


def check_rows(blob_counts: dict[str, int], row_relations: dict[str, int],
               *, expect: dict[str, str]) -> list[str]:
    """Every kind of record a labeler produced must produce at least one row."""
    out: list[str] = []
    for kind, relation_prefix in expect.items():
        n = blob_counts.get(kind, 0)
        if not n:
            continue
        produced = sum(v for k, v in row_relations.items()
                       if k == relation_prefix or k.startswith(relation_prefix))
        if not produced:
            out.append(f"{n} {kind} in the blob produced no {relation_prefix}* row")
    return out


def check_admitted(row_relations: dict[str, int], backbone_tasks: tuple[str, ...]) -> list[str]:
    """A row that is neither a backbone task nor deliberately held back is lost.

    This is the check that would have caught DEFAULT_TASKS: `gap_valid` was
    emitted, mirrored and ingested, and then matched nothing -- and the only
    symptom was a skip counter nobody read.
    """
    out: list[str] = []
    for rel, n in sorted(row_relations.items()):
        if rel in backbone_tasks:
            continue
        if rel.startswith(_HELD_BACK):
            continue
        out.append(f"{n} rows of `{rel}` reach neither the backbone nor a held-back head")
    return out
