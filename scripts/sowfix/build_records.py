"""Turn each deal's live base request (dump_<uuid>.json) plus the rubric labels
(labels_<deal>.json) into a training record the C3 loaders read.

    python3 build_records.py            -> sowfix/train/sowfix_<deal>.json

Labels carry no labeler (the loader accepts anonymous rows when no person
labeled the line) and say where they came from in `note`.
"""
import glob
import json
import re
import hashlib
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "train")
os.makedirs(OUT, exist_ok=True)
VALID_SUPPLIER = {"us", "partner", "customer", "third_party"}
REJECT_CAP = 60
VALID_CO = {"keep", "reject", "ignore"}

total = {}


def _names_place(text):
    """A whole place: a street address, a facility name, or a row with several fields."""
    t = (text or "").strip()
    if re.search(r"[A-Za-z][A-Za-z .'-]+,\s*[A-Z]{2}\b", t):
        return True  # "Tupelo, MS"
    if len(t) < 12 or not re.search(r"[A-Za-z]{3}", t):
        return False
    return bool(re.search(r"\d+\s+\w", t) or re.search(r"(hospital|medical|center|store|clinic|campus|office|warehouse)", t, re.I) or t.count("|") >= 2 or t.count("\t") >= 2 or t.count(",") >= 2)
for path in sorted(glob.glob(os.path.join(HERE, "labels_*.json"))):
    spec = json.load(open(path))
    deal, uuid = spec["deal"], spec["uuid"]
    dump = json.load(open(f"/Users/purtera/c3-serve/dump_{uuid}.json"))
    atoms = []
    by_key = {}
    for a in dump["atoms"]:
        row = {k: v for k, v in a.items() if k != "key"}
        row["label_key"] = a["key"]
        row["atom_id"] = a.get("atom_id") or a["key"]
        atoms.append(row)
        by_key[a["key"]] = row
    labels, skipped, thin = [], 0, 0
    seen = set()
    rows = spec.get("labels") or []
    bare = [lb for lb in rows if lb.get("co_action") == "reject" and not (lb.get("label_type") or lb.get("supplier") or lb.get("sow_section"))]
    if len(bare) > REJECT_CAP:  # one other-job pilot or form sheet must not outweigh the deal's real errors
        keep_bare = {id(lb) for lb in sorted(bare, key=lambda lb: hashlib.md5(lb["atom_id"].encode()).hexdigest())[:REJECT_CAP]}
        rows = [lb for lb in rows if lb not in bare or id(lb) in keep_bare]
    for lb in rows:
        key = lb.get("atom_id")
        if key not in by_key or key in seen:
            skipped += 1
            continue
        seen.add(key)
        reads = {"doc_kind": by_key[key].get("doc_kind") or ""}
        if lb.get("co_action") in VALID_CO:
            reads["co_action"] = lb["co_action"]
        if lb.get("sow_section"):
            reads["sow_section"] = lb["sow_section"]
        row = {
            "label_key": key,
            "atom_id": key,
            "text": by_key[key].get("text") or "",
            "reads_set": reads,
            "rejected": "true" if lb.get("co_action") == "reject" else None,
            "purpose": "train",
            "origin": "parser",
            "note": f"sowfix rubric {lb.get('rule', '')}: {lb.get('why', '')}"[:400],
        }
        lt = lb.get("label_type")
        if lt == "physical_site" and not _names_place(row["text"]):
            lt = None  # a bare ZIP, state, city or store-number cell is not a whole site row
            thin += 1
        if lt:
            row["label_type"] = lt
        if lb.get("supplier") in VALID_SUPPLIER:
            row["supplier"] = lb["supplier"]
        labels.append(row)
    record = {"deal_id": f"sowfix_{deal}", "company": "purtera", "source": ["sowfix_rubric"],
              "atoms": atoms, "labels": labels, "judgments": [], "links": [], "deal_answers": []}
    json.dump(record, open(os.path.join(OUT, f"sowfix_{deal}.json"), "w"))
    total[deal] = (len(labels), skipped)
    print(deal, "labels", len(labels), "skipped", skipped, "thin-site dropped", thin, "atoms", len(atoms))
print("ALL", sum(v[0] for v in total.values()))
