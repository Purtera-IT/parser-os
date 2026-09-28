# -*- coding: utf-8 -*-
"""Emit a hand-editable pass file with one stub per atom, for labelling by hand.

The pass files on 010237 were written as Python because that deal was labelled in
bulk. Labelling by hand wants the same seven fields without writing code, so this
prints the atoms of one document as JSON you edit in a text editor and then feed
straight to write_labels.py.

Each stub carries the atom's own text, section and lead-in as `_read` keys. They
start with an underscore, they are ignored by the writer, and they are there so
you are never labelling a line you cannot see in context -- which is the single
most common way a label goes wrong.

    DEAL=... python make_template.py                      # list the documents
    DEAL=... python make_template.py "NW-OP365071a" > p1.json
    # ...edit p1.json...
    DEAL=... LABELER=you@purtera-it.com python validate_pass.py p1.json
    DEAL=... LABELER=you@purtera-it.com python write_labels.py p1.json --apply
"""
import collections, json, os, sys
from pathlib import Path

HERE = Path(__file__).parent
WALK = json.loads((HERE / os.environ.get("WALK", "walk_180.json")).read_text(encoding="utf-8"))
ATOMS = [a for a in WALK["atoms"] if "dedup" not in (a.get("suppressedBy") or "")]

#: Every field the completeness audit grades, so a stub that is filled in is
#: complete by construction. `supplier` is optional -- the registry declares which
#: types even ask the question -- so it is left out rather than stubbed empty.
STUB = {
    "label_type": "", "weight_tier": "", "about": "", "wants": "",
    "hint_refs": [{"hint": "own_words", "kind": "text", "text": ""}],
    "entity_keys": [], "note": "",
}


def main() -> None:
    if len(sys.argv) < 2:
        by_doc = collections.Counter(a["filename"] for a in ATOMS)
        print(f"{len(ATOMS)} live atoms across {len(by_doc)} documents:\n", file=sys.stderr)
        for name, n in by_doc.most_common():
            print(f"   {n:4d}  {name}", file=sys.stderr)
        print("\nPass a filename (or any unique part of one) to emit its stubs.",
              file=sys.stderr)
        return

    needle = sys.argv[1]
    sel = [a for a in ATOMS if needle.lower() in a["filename"].lower()]
    if not sel:
        raise SystemExit(f"no document matching {needle!r}")
    names = {a["filename"] for a in sel}
    if len(names) > 1:
        raise SystemExit(f"{needle!r} matches {len(names)} documents: {sorted(names)}")
    doc = names.pop()

    labels = []
    for i, a in enumerate(sel):
        fields = json.loads(json.dumps(STUB))
        labels.append({
            "_i": i,
            "_text": a["text"],
            "_parser_type": a.get("parserType"),
            "_section": a.get("section") or [],
            "_suppressed_by": a.get("suppressedBy") or "",
            "_parser_entity_keys": a.get("entityKeys") or [],
            "labelKey": a["labelKey"],
            "fields": fields,
        })
    print(json.dumps({"document": doc, "purpose": "train", "labels": labels},
                     indent=1, ensure_ascii=False))
    print(f"{len(labels)} stubs for {doc}", file=sys.stderr)


if __name__ == "__main__":
    main()
