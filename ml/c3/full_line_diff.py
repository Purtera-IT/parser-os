"""Old vs new base on EVERY line of every deal: the check a refit must pass before it serves.

    python -m ml.c3.full_line_diff --old http://host:8765 --new http://host:8766 \
        --dumps '/Users/purtera/c3-serve/dump_*.json' --out /outside/diff.json \
        [--heads read:co_action col:supplier ...] [--documents jdg:document_job]

The 2026-10-18 refit scored well on held-out labeled lines and still called
nearly every line of an unseen deal scope_item (c3-refit-collapse). So this
asks both servers the same deal requests (the worker's own dumps) and compares
the top answer of the type head and each named head on every line, and of each
document head on every document. It reports, per head: lines changed, and the
answer whose share moved most (a collapse shows as one answer jumping). The
JSON output lists every changed line by key; text is written only to the
output file, never printed.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import urllib.request
from collections import Counter

DEFAULT_HEADS = ("read:co_action", "read:co_reason", "read:sow_section", "read:noise_class", "col:supplier",
                 "read:task_owner_role")
#: An answer's share of a deal's lines moving this much is a collapse, whatever the change count.
COLLAPSE_SHIFT = 0.15


def _post(url: str, body: dict, token: str, timeout: int = 900) -> dict:
    r = urllib.request.Request(url.rstrip("/") + "/read", data=json.dumps(body).encode(),
                               headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=timeout))


def _top(v):
    return v[0] if isinstance(v, (list, tuple)) and v else None


def compare(old: dict, new: dict, heads: list[str], doc_heads: list[str]) -> dict:
    """Per head: changed line keys and the biggest share shift. Pure: tested without servers."""
    out = {}
    per_line = {"type": (old.get("types") or {}, new.get("types") or {})}
    for h in heads:
        per_line[h] = ({k: (r or {}).get(h) for k, r in (old.get("reads") or {}).items()},
                       {k: (r or {}).get(h) for k, r in (new.get("reads") or {}).items()})
    for h in doc_heads:
        per_line[h] = ({k: (r or {}).get(h) for k, r in (old.get("documents") or {}).items()},
                       {k: (r or {}).get(h) for k, r in (new.get("documents") or {}).items()})
    for h, (o, n) in per_line.items():
        keys = sorted(set(o) | set(n))
        changed = [{"key": k, "old": o.get(k), "new": n.get(k)} for k in keys if _top(o.get(k)) != _top(n.get(k))]
        co = Counter(_top(o.get(k)) for k in keys)
        cn = Counter(_top(n.get(k)) for k in keys)
        tot = max(1, len(keys))
        shift = max(((abs(cn[a] - co[a]) / tot, a) for a in set(co) | set(cn)), default=(0.0, None))
        out[h] = {"lines": len(keys), "changed": len(changed), "max_shift": round(shift[0], 3),
                  "max_shift_answer": shift[1], "collapse": shift[0] >= COLLAPSE_SHIFT, "changes": changed}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--old", required=True)
    p.add_argument("--new", required=True)
    p.add_argument("--dumps", nargs="+", required=True)
    p.add_argument("--heads", nargs="*", default=list(DEFAULT_HEADS))
    p.add_argument("--documents", nargs="*", default=["jdg:document_job"])
    p.add_argument("--token", default=os.path.expanduser("~/c3-serve/token"))
    p.add_argument("--out", required=True)
    p.add_argument("--budget", type=int, default=170)
    a = p.parse_args()
    token = open(a.token).read().strip()
    report = {}
    for path in sorted(x for pat in a.dumps for x in glob.glob(pat)):
        deal = json.load(open(path))
        body = {"deal": deal, "budget_s": a.budget, "line_heads": a.heads}
        if a.documents:
            body["document_heads"] = a.documents
        old, new = _post(a.old, body, token), _post(a.new, body, token)
        cmp = compare(old, new, a.heads, a.documents)
        text = {str(x.get("key")): str(x.get("text") or "")[:200] for x in deal.get("atoms") or []}
        for h in cmp.values():
            for c in h["changes"]:
                c["text"] = text.get(c["key"], "")
        name = os.path.basename(path)
        report[name] = cmp
        flags = " ".join(f"{h}:{v['changed']}/{v['lines']}{' COLLAPSE ' + str(v['max_shift_answer']) if v['collapse'] else ''}"
                         for h, v in cmp.items())
        print(f"{name}: {flags}", flush=True)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1)
    bad = [(d, h) for d, r in report.items() for h, v in r.items() if v["collapse"]]
    print("COLLAPSE " + str(bad) if bad else "no collapse")


if __name__ == "__main__":
    main()
