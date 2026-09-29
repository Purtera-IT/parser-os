# -*- coding: utf-8 -*-
"""Drop three numeric entity keys that name counts their atoms never state."""
import json, os, re, psycopg2

DEAL = "c065bfc4-ba3f-430d-b986-68a243666e5d"
MINE = "developer@purtera-it.com"

FIX = {
 "lbl_f69b46bfb946448e450a": ("quantity:1",
   " Count key removed. The atom says a AMS resource -- singular by grammar, not a stated number -- "
   "so quantity:1 was me reading the article as a numeral. One resource is almost certainly right "
   "and it is still not what this sentence says, and a head trained on it learns to emit counts "
   "from indefinite articles. The singular reading survives where it belongs, in the note."),
 "lbl_f4b45cdf3301d6726435": ("quantity:1",
   " Count key removed, same as on the email copy of this sentence: a AMS resource states no "
   "numeral, and inferring one from the article is exactly the fabrication the numeric invariant "
   "exists to catch."),
 "lbl_a301739e969392405764": ("quantity:2",
   " Count key removed. The atom writes the count as the word two, not as a numeral, and the "
   "invariant compares digits -- but the rule underneath is the one that matters: the RFI atom "
   "answering how many applications are integrated states 2 properly and carries the key. This "
   "atom should reach that count through a link to that answer, not by restating it, which is the "
   "same correction the room keys on 010180 needed when they referred to a room rather than "
   "naming it."),
}

cn = psycopg2.connect(os.environ["PG"], connect_timeout=30)
cur = cn.cursor()
out = []
for key, (drop, addendum) in FIX.items():
    cur.execute("SELECT entity_keys, note FROM public.atom_labels "
                "WHERE deal_id=%s AND labeler=%s AND label_key=%s", (DEAL, MINE, key))
    row = cur.fetchone()
    if not row:
        raise SystemExit(f"no such label {key}")
    ek = json.loads(row[0]) if isinstance(row[0], str) else (row[0] or [])
    out.append({"labelKey": key, "fields": {
        "entity_keys": [x for x in ek if x != drop],
        "note": (row[1] or "").rstrip() + addendum,
    }})
cur.close(); cn.close()

with open("pass_07_fix.json", "w", encoding="utf-8") as fh:
    json.dump({"document": "numeric-key corrections", "labels": out}, fh, indent=1,
              ensure_ascii=False)
print(f"pass_07_fix.json: {len(out)} corrections")
