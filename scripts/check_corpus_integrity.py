"""Run the integrity guard on 010288, end to end: Postgres -> blob -> rows."""
import json
import os
from collections import Counter

import psycopg2
import psycopg2.extras

from app.learning.corpus_integrity import (
    IntegrityReport, check_admitted, check_fields, check_rows,
)
from app.learning.human_labels import rows_for_deal
from app.learning.multitask_table import DEFAULT_TASKS

DEAL = "c2a3bdce-53bb-4da6-a840-a5260841685a"
HUMAN = "developer@purtera-it.com"
BLOB = (r"D:\temp\claude\C--Users-lilli-parser-os"
        r"\7a65a361-3618-4f10-89f7-8a042c1ff248\scratchpad\final_mirror2.json")

doc = json.load(open(BLOB, encoding="utf-8"))
conn = psycopg2.connect(os.environ["PG"])
cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

rep = IntegrityReport()
for table, blob_key in (("atom_labels", "labels"),
                        ("atom_label_links", "links"),
                        ("atom_label_judgments", "judgments"),
                        ("atom_label_deal_answers", "deal_answers")):
    cur.execute(f"SELECT * FROM public.{table} WHERE deal_id=%s AND labeler LIKE %s",
                (DEAL, "%"))
    pg = [dict(r) for r in cur.fetchall()]
    blob = doc.get(blob_key) or []
    rep.counts[f"{blob_key} pg"] = len(pg)
    rep.counts[f"{blob_key} blob"] = len(blob)
    if len(pg) != len(blob):
        rep.dropped_rows.append(
            f"{blob_key}: {len(pg)} in Postgres, {len(blob)} in the blob")
    rep.dropped_fields += check_fields(pg, blob, what=blob_key)

rows = rows_for_deal(doc)
rels = Counter(r["relation"] for r in rows)
rep.counts["training rows"] = len(rows)

rep.dropped_rows += check_rows(
    {k: len(doc.get(k) or []) for k in ("labels", "links", "judgments", "deal_answers")},
    rels,
    expect={"labels": "atom_type", "links": "edge_relation",
            "judgments": "gap_valid", "deal_answers": "rationale:deal"},
)
rep.unaccounted = check_admitted(rels, DEFAULT_TASKS)

print(rep)
print()
print("OK" if rep.ok else "PROBLEMS FOUND")
