"""What Postgres holds, what the blob carries, what the corpus admits.

Seven times in one day a field was written correctly and silently dropped one
step later:

  * `doc.judgments = []` stubbed in the mirror -- 94 verdicts written and lost;
  * `buildBlobDoc` maps explicit fields, so `consumer`, `rejected`,
    `decided_by`, `weight_tier` and `rejected_reads` reached Postgres and
    stopped;
  * the judgment `reason` the same way;
  * `DEFAULT_TASKS` never listed the judgement heads, so 892 of 1,241 rows were
    dropped one step from the model;
  * every one of 159 evidence pointers was emitted as a span, and 87 of them
    name a structured field or another surface -- so a third of the span
    supervision asked a head to produce words that are not on the page it
    holds, which is how a span head learns to invent one;
  * `assemble` keyed its dedupe on (relation, text), so a labeler's second
    answer read as a contradiction and `decided_from` lost 83 of 148 rows;
  * ...and an edge's target lives in its provenance, so the six atoms one
    sentence governs were six identical rows, five of them discarded.

Every one produced perfect data that arrived nowhere, and nothing complained.
Counting rows would have caught none of them: the rows were all there, with a
field missing, a relation unlisted, a label its own prompt did not contain, or
a duplicate that was not one.

So this compares the stages FIELD BY FIELD, RELATION BY RELATION and ROW BY
ROW, and names what falls between them. It is cheap enough to run on every deal
after labelling, and it is the only check that fails when somebody adds a
column and forgets the mapper.
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
_HELD_BACK = ("evidence_span:", "evidence_doc:", "reads_value:", "rationale:")


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


def check_spans(rows: list[dict[str, Any]]) -> list[str]:
    """A span head can only point at words it is holding.

    The fifth silent loss of the day, and the largest. All 159 of 010288's
    pointers were emitted as `evidence_span`, but 56 of them name a structured
    field -- the envelope, the document type -- and 21 name another surface.
    Neither is text on the page the head is given, so a third of the span
    supervision was an instruction to produce words that are not there, which
    is exactly how a span head learns to invent one.

    Nothing counted this. The rows were present, the relations were listed, and
    the field check passed, because the loss was INSIDE a row: a label that its
    own prompt does not contain.
    """
    out: list[str] = []
    bad = [r for r in rows
           if str(r.get("relation") or "").startswith("evidence_span:")
           and _norm(r.get("label")) not in _norm(r.get("raw_text"))]
    if bad:
        shown = ", ".join(sorted({str(r.get("relation")) for r in bad})[:4])
        out.append(f"{len(bad)} evidence_span rows point at words absent from "
                   f"their own prompt ({shown})")
    return out


def _norm(s: Any) -> str:
    return " ".join(str(s or "").split()).lower()


def check_assembled(emitted: dict[str, int], assembled: dict[str, int],
                    backbone_tasks: tuple[str, ...],
                    distinct: dict[str, int] | None = None) -> list[str]:
    """A backbone relation must reach the table with every DISTINCT row it emits.

    `distinct` counts assertions rather than rows -- same relation, same text,
    same label, same target counted once. Pass it whenever rows can legitimately
    repeat, because otherwise this check reports correct de-duplication as loss:
    010180 quotes the same two email headers down a fourteen-message thread, so
    268 atom_type rows are 197 distinct assertions, and the 71 that collapse are
    not a defect. Without it this fired on six relations at once and every one
    was a false alarm.

    The seam past every other check here: the rows leave `human_labels`
    complete, the relation IS a backbone task, and the loss happens inside
    `assemble`, in a counter whose name is the one word that stops anyone
    looking -- "duplicate".

    Two of them on 010288. `decided_from` lost 83 of 148 because the key was
    (relation, text) and a second label read as a conflict, when an atom is
    genuinely decided by its own words AND by who said it. `edge_relation` lost
    5 of 80 because an edge's target lives in its provenance, so six edges out
    of one sentence were six identical rows.
    """
    out: list[str] = []
    for task in sorted(backbone_tasks):
        was = (distinct or {}).get(task, emitted.get(task, 0))
        now = assembled.get(task, 0)
        if was and now < was:
            raw = emitted.get(task, 0)
            extra = f" ({raw} rows, {was} distinct)" if raw != was else ""
            out.append(f"{task}: {was} assertions emitted, {now} reached the "
                       f"table ({was - now} lost inside assemble){extra}")
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
