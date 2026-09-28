"""Set SOME columns on an existing label, leaving the rest alone.

`write_labels.py` rebuilds a row from the walk atom plus the pass file and
UPDATEs every column, so a pass that supplies only `entity_keys` silently blanks
the pointers, about and wants that were already there. That is a destructive
partial write dressed as an edit, and it is how a label gets quietly worse.

This one issues UPDATE ... SET only for the keys present in `fields`, and
refuses to run on a label that does not already exist.

    PG=... python patch_labels.py pass_x.json          # dry run
    PG=... python patch_labels.py pass_x.json --apply
"""
import json
import os
import sys
from pathlib import Path

import psycopg2

#: The deal being labelled. An env var, not a constant: these scripts were
#: written for 010180 and a hardcoded id here would silently write another
#: deal's labels under this deal's rows, which no audit would catch.
DEAL = os.environ.get("DEAL") or "c79db726-323e-41f8-899d-1d8ca29a579a"
#: The labeler whose rows this reads. An env var, not a constant: it was hardcoded,
#: so a second labeler auditing their own deal graded someone else's rows and saw
#: 0% forever -- or, worse, saw 100% on a deal they had not touched.
MINE = os.environ.get("LABELER", "developer@purtera-it.com")
HERE = Path(__file__).parent

_JSON_COLS = {"section", "lead_in", "neighbors_above", "neighbors_below", "hints",
              "entity_keys", "reads_set", "hint_refs", "rejected_reads", "reads_kept",
              "reads_shown", "said_to"}
#: Columns this tool will touch. Anything else is a typo, not an intention.
_ALLOWED = _JSON_COLS | {"rejected", "label_type", "weight_tier", "note", "about", "wants", "supplier",
                         "decided_by", "origin", "purpose", "coarse", "internal_only"}


def main():
    payload = json.loads((HERE / sys.argv[1]).read_text(encoding="utf-8"))
    apply = "--apply" in sys.argv
    cn = psycopg2.connect(os.environ["PG"], connect_timeout=25)
    cur = cn.cursor()

    done = skipped = 0
    for item in payload["labels"]:
        key = item["labelKey"]
        fields = {k: v for k, v in (item.get("fields") or {}).items()}
        bad = set(fields) - _ALLOWED
        if bad:
            raise SystemExit(f"unknown column(s) {bad} on {key}")
        cur.execute("SELECT 1 FROM public.atom_labels WHERE deal_id=%s AND labeler=%s "
                    "AND label_key=%s", (DEAL, MINE, key))
        if not cur.fetchone():
            print(f"   SKIP (no such label) {key}")
            skipped += 1
            continue
        # Same rule as write_labels: a `_keep` must carry `rejected`, or the
        # reviewer cannot see what was rejected.
        if str(fields.get("label_type") or "") == "_keep":
            fields["rejected"] = "true"
        cols, vals = [], []
        for col, val in fields.items():
            if col in _JSON_COLS and not isinstance(val, str):
                val = json.dumps(val, ensure_ascii=False)
            cols.append(f"{col}=%s")
            vals.append(val)
        print(f"   {key[:16]} set {', '.join(sorted(fields))}")
        if apply:
            cur.execute(f"UPDATE public.atom_labels SET {', '.join(cols)} "
                        f"WHERE deal_id=%s AND labeler=%s AND label_key=%s",
                        (*vals, DEAL, MINE, key))
        done += 1

    if apply:
        cn.commit()
        print(f"\npatched {done} label(s), skipped {skipped}")
    else:
        print(f"\ndry run -- would patch {done}, skip {skipped}")
    cur.close()
    cn.close()


if __name__ == "__main__":
    main()
