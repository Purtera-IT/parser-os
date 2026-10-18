"""Score a running base against the rubric labels, by replaying each deal's
live request.   python3 eval_labels.py <port> [deal ...]

Per field: how many labeled lines the base's top answer matches, and how many
it matches sure (p >= 0.85, the compile's authority bar)."""
import glob
import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
port = sys.argv[1]
only = set(sys.argv[2:])
tok = open("/Users/purtera/c3-serve/token").read().strip()
HEADS = {"supplier": "col:supplier", "co_action": "read:co_action", "sow_section": "read:sow_section"}
tot = {f: [0, 0, 0] for f in ("label_type", *HEADS)}
for path in sorted(glob.glob(os.path.join(HERE, "labels_*.json"))):
    spec = json.load(open(path))
    if only and spec["deal"] not in only:
        continue
    deal = json.load(open(f"/Users/purtera/c3-serve/dump_{spec['uuid']}.json"))
    body = {"deal": deal, "budget_s": 170, "line_heads": list(HEADS.values())}
    req = urllib.request.Request(f"http://100.114.102.122:{port}/read", data=json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + tok, "Content-Type": "application/json"})
    res = json.load(urllib.request.urlopen(req, timeout=600))
    types, reads = res.get("types") or {}, res.get("reads") or {}
    row = {f: [0, 0, 0] for f in tot}
    for lb in spec["labels"]:
        k = lb["atom_id"]
        for f in tot:
            want = lb.get(f)
            if not want:
                continue
            got = types.get(k) if f == "label_type" else (reads.get(k) or {}).get(HEADS[f])
            row[f][0] += 1
            if got and got[0] == want:
                row[f][1] += 1
                if float(got[1] or 0) >= 0.85:
                    row[f][2] += 1
    for f in tot:
        for i in range(3):
            tot[f][i] += row[f][i]
    print(spec["deal"], " ".join(f"{f}={r[1]}/{r[0]}(sure {r[2]})" for f, r in row.items() if r[0]), flush=True)
print("ALL", " ".join(f"{f}={r[1]}/{r[0]} ({r[1] / max(1, r[0]):.0%}, sure {r[2]})" for f, r in tot.items()))
