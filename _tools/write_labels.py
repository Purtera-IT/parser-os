"""Write labels for one deal from a JSON pass file. Dry run unless --apply.

One pass per document, so a long labelling run is durable: whatever is written
is written, and a pass that is re-run replaces only its own rows.

The key is built from the SAME filename and page that get stored beside it. On
010288, 31 of 130 labels do not recompute their own key -- the drawing's atoms
were keyed under one filename and stored under another -- so those labels will
not re-attach on a re-compile, which is the one thing the key exists to do.

    python write_labels.py pass_01.json            # show what it would do
    python write_labels.py pass_01.json --apply
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import psycopg2
import psycopg2.extras

from app.core.label_key import label_key

#: The deal being labelled. An env var, not a constant: these scripts were
#: written for 010180 and a hardcoded id here would silently write another
#: deal's labels under this deal's rows, which no audit would catch.
DEAL = os.environ.get("DEAL") or "c79db726-323e-41f8-899d-1d8ca29a579a"
# The workspace is PER LABELER: a row written under any other name shows the
# deal as 0% labelled, however many rows exist. Live 2026-09-26: 268 labels
# were invisible until they were moved. Overridable, never guessed.
#: The registry's "not a fact worth typing" type.
KEEP_TYPE = "_keep"

MINE = os.environ.get("LABELER", "developer@purtera-it.com")
HERE = Path(__file__).parent

#: Columns on atom_labels this writer sets. Anything not here keeps its default.
_FIELDS = (
    "label_key", "deal_id", "labeler", "atom_id", "compile_id", "text",
    "filename", "doc_type", "page", "section", "lead_in", "neighbors_above",
    "neighbors_below", "table_ref", "hints", "entity_keys", "note", "purpose",
    "parser_type", "label_type", "coarse", "supplier", "about", "wants",
    "reads_set", "said_by", "said_to", "internal_only", "consumer", "rejected",
    "decided_by", "weight_tier", "rejected_reads", "hint_refs",
    "reads_kept", "reads_shown", "is_new_type", "origin",
)
_JSON_FIELDS = {"section", "lead_in", "neighbors_above", "neighbors_below",
                "hints", "entity_keys", "reads_set", "hint_refs",
                "rejected_reads", "reads_kept", "reads_shown", "said_to"}

#: Columns the table declares NOT NULL with a default. An explicit INSERT
#: naming the column overrides the default with NULL, so the defaults have to
#: be restated here or every write fails on `lead_in`.
_NOT_NULL = {
    "section": [], "lead_in": [], "neighbors_above": [], "neighbors_below": [],
    "hints": [], "entity_keys": [], "hint_refs": [], "reads_set": {},
    "rejected_reads": {}, "reads_kept": [], "reads_shown": [], "said_to": [],
    "is_new_type": False, "internal_only": False, "origin": "parser",
}


def main() -> None:
    path = HERE / sys.argv[1]
    apply = "--apply" in sys.argv
    payload = json.loads(path.read_text(encoding="utf-8"))
    walk = {a["labelKey"]: a for a in
            json.loads((HERE / os.environ.get("WALK", "walk_180.json")).read_text(encoding="utf-8"))["atoms"]}

    rows = []
    stale = []
    for item in payload["labels"]:
        atom = walk.get(item["labelKey"])
        if atom is None:
            # The atom this label was written for no longer exists. Six of
            # 010180's did after the re-compile -- the Teams-block fix removed
            # some and the money fix rewrote others -- and a label keyed to a
            # line nobody will ever parse again is not gold, it is litter.
            stale.append(item["labelKey"])
            continue
        key = label_key(DEAL, atom["filename"], atom["page"], atom["text"])
        if key != atom["labelKey"]:
            raise SystemExit(f"key drift on {item['labelKey']}: walk and writer disagree")
        row = {
            "label_key": key,
            "deal_id": DEAL,
            "labeler": MINE,
            "atom_id": atom["atomId"],
            "compile_id": payload.get("compile_id") or "",
            "text": atom["text"],
            "filename": atom["filename"],
            "page": atom["page"],
            "parser_type": atom["parserType"],
            "section": atom.get("section") or [],
            "entity_keys": atom.get("entityKeys") or [],
            "purpose": payload.get("purpose", "train"),
        }
        row.update(item.get("fields") or {})
        # The workspace styles a row red off the `rejected` COLUMN, not off the
        # type, so a `_keep` written without it renders as an ordinary row and
        # drops out of the "Rejected (n)" filter. 49 of 71 rejections on 010180
        # were invisible that way: the judgement was in the database and the
        # reviewer could not see it, which is the one thing rejecting is for.
        if str(row.get("label_type") or "") == KEEP_TYPE:
            row.setdefault("rejected", "true")
            row["rejected"] = "true"
        rows.append(row)

    print(f"{path.name}: {len(rows)} labels for {payload.get('document', '?')}")
    if stale:
        print(f"   skipped {len(stale)} stale (atom no longer exists)")
    for row in rows[:400]:
        print(f"   {row['label_key'][:14]} {str(row.get('label_type')):<22} "
              f"{row['text'][:56]}")
    if not apply:
        print("\ndry run -- pass --apply to write")
        return

    conn = psycopg2.connect(os.environ["PG"])
    cur = conn.cursor()
    cols = list(dict.fromkeys(_FIELDS))  # order-preserving, no repeats
    placeholders = ", ".join(["%s"] * len(cols))
    updates = ", ".join(f"{c}=EXCLUDED.{c}" for c in cols
                        if c not in ("label_key", "deal_id", "labeler"))
    sql = (f"INSERT INTO public.atom_labels ({', '.join(cols)}) "
           f"VALUES ({placeholders}) "
           f"ON CONFLICT (deal_id, label_key, labeler) DO UPDATE SET {updates}")
    written = 0
    for row in rows:
        values = []
        for col in cols:
            value = row.get(col)
            if value is None and col in _NOT_NULL:
                value = _NOT_NULL[col]
            if col in _JSON_FIELDS and value is not None and not isinstance(value, str):
                value = json.dumps(value, ensure_ascii=False)
            values.append(value)
        cur.execute(sql, values)
        written += 1
    conn.commit()
    cur.close()
    conn.close()
    print(f"\nwrote {written} labels as {MINE}")


if __name__ == "__main__":
    main()
