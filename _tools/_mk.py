"""Build a pass file from (filename, atom-index) specs.

The index is into the document's LIVE atoms in walk order -- the same list the
dump prints -- so a spec is readable next to the dump and cannot silently drift
onto another atom: the builder re-prints the text it resolved, and refuses a
pointer whose own_words are not literally in that atom.
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).parent
WALK = json.loads((HERE / "walk_180.json").read_text(encoding="utf-8"))["atoms"]
LIVE = [a for a in WALK if "dedup" not in (a.get("suppressedBy") or "")]


def build(doc, spec, out, purpose="train"):
    sel = [a for a in LIVE if a["filename"] == doc]
    labels, bad = [], []
    for idx, fields in spec.items():
        atom = sel[idx]
        text = " ".join(atom["text"].split()).lower()
        for p in fields.get("hint_refs") or []:
            if p.get("hint") == "own_words":
                if " ".join(p["text"].split()).lower() not in text:
                    bad.append((idx, p["text"], atom["text"][:70]))
        labels.append({"labelKey": atom["labelKey"], "fields": fields})
    if bad:
        for i, q, t in bad:
            print(f"  POINTER NOT IN ATOM  [{i}] {q!r}  not in  {t!r}")
        raise SystemExit(f"{len(bad)} fabricated pointer(s) -- nothing written")
    (HERE / out).write_text(json.dumps(
        {"document": doc, "purpose": purpose, "labels": labels},
        indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{out}: {len(labels)} labels for {doc}")
