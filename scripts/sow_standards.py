#!/usr/bin/env python3.11
"""Add, list or retire PurTera SOW standard items without a deploy (app/core/standards_store.py).

    AZURE_STORAGE_CONNECTION_STRING=... python3.11 scripts/sow_standards.py list [--family F]
    ... add --family F --section S --text "Sentence." --by you@purtera-it.com [--when REGEX] [--deal 010217]
    ... bulk --file harvest_approved.json --by you@purtera-it.com     # [{family, section, text, when?, sources?}]
    ... retire --id std_... --by you@purtera-it.com

Items take effect on the next compile (the worker re-reads the overlay every 5 minutes).
Use SOWSMITH_STANDARDS_OVERLAY_FILE=/path.json to try it against a local file first.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.core import standards_store as S  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("list"); ls.add_argument("--family", default="")
    ad = sub.add_parser("add")
    for k in ("family", "section", "text", "by"):
        ad.add_argument(f"--{k}", required=True)
    ad.add_argument("--when", default=""); ad.add_argument("--unless", default=""); ad.add_argument("--deal", default="")
    ad.add_argument("--status", default="proposed", choices=["proposed", "approved"])
    bk = sub.add_parser("bulk"); bk.add_argument("--file", required=True); bk.add_argument("--by", required=True)
    bk.add_argument("--status", default="proposed", choices=["proposed", "approved"])
    ap = sub.add_parser("approve"); ap.add_argument("--id", required=True); ap.add_argument("--by", required=True)
    rt = sub.add_parser("retire"); rt.add_argument("--id", required=True); rt.add_argument("--by", required=True)
    a = p.parse_args()
    if a.cmd == "list":
        ov, _ = S.read_overlay()
        for it in ov.get("items") or []:
            if not a.family or it.get("family") == a.family:
                print(f"{it['id']} {it.get('status','active'):8} {it['family']}/{it['section']}: {S.item_text(it)}")
    elif a.cmd == "add":
        it = S.add_item(family=a.family, section=a.section, text=a.text, added_by=a.by, when=a.when,
                        unless=a.unless, source_deal=a.deal, status=a.status)
        print("added", it["id"])
    elif a.cmd == "bulk":
        have = {(f, s, S._strip(t)) for f, b in S.load().items() if isinstance(b, dict)
                for s, v in b.items() if isinstance(v, list) for t in v}
        n = 0
        for r in json.loads(Path(a.file).read_text()):
            if (r["family"], r["section"], S._strip(S.item_text(r))) in have:
                continue
            S.add_item(family=r["family"], section=r["section"], text=r["text"], added_by=a.by,
                       when=r.get("when", ""), unless=r.get("unless", ""),
                       source_deal=",".join(r.get("sources") or []), note=r.get("note", ""), status=a.status)
            n += 1
        print(f"added {n}")
    elif a.cmd == "approve":
        print("approved", S.approve_item(a.id, by=a.by)["id"])
    elif a.cmd == "retire":
        print("retired", S.retire_item(a.id, by=a.by)["id"])


if __name__ == "__main__":
    main()
