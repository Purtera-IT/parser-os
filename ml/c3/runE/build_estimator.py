"""Run E quoting view from the Deal Kit estimator's deals: ~/c3-local/runE/train_estimator/<deal>@pre_quote.json

    PYTHONPATH=~/c3-runE-wt python3.11 build_estimator.py [--manifests DIR]

Inputs (read-only):
  ~/deal-kit-training/c3files/<deal>.json   pre-quote atoms (run C trunk format, atoms only)
  ~/c3-local/runE/estimator_targets.json    the estimator's targets (Max's kits matched to a customer line)
  ~/deal-kit-training/task_labels.json      Max's job type per deal (-> deal:primary_service)
  --manifests DIR/<deal>/*.json             the deal's parser manifests (every document the deal ever had),
                                            for the cut check against our first outbound pricing file

Per deal: the c3file's atoms unchanged (they are the inputs, no line labels), one `hours` judgment card per
train:true target on its atom (verdict in the hours grammar -> read:co_hours_estimate, or read:labor_hours
when basis=stated, with the hours value and the hours/per/qty literal), and deal:primary_service from Max's
job type where it maps one-to-one. head_scale masks every opportunity but the quote heads, so a line with no
label teaches nothing on any other head.

Not written: crew_size / equipment_qty line reads. They need a line label, and a labeled line is taught
"no link" on every relation head (losses.relation_loss), which head_scale does not reach. The target's qty
rides in the hours literal (hours=..;per=..;qty=..).

The cards are machine drafts (why_author machine_draft, status draft in the source file). data.py trains only
person-labeled rows; the labeler below names the source so it passes that gate, on the owner's instruction
to train the train:true targets.
"""
import argparse
import collections
import datetime
import glob
import hashlib
import json
import os
import re
import sys

HOME = os.path.expanduser("~")
C3FILES = f"{HOME}/deal-kit-training/c3files"
TARGETS = f"{HOME}/c3-local/runE/estimator_targets.json"
TASKS = f"{HOME}/deal-kit-training/task_labels.json"
OUT = f"{HOME}/c3-local/runE/train_estimator"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ.get("C3_REPO", f"{HOME}/c3-runE-wt"))

from ml.c3.data import _is_a_person, _time  # noqa: E402
from ml.c3.holdout import FIXED_HOLDOUT  # noqa: E402
from ml.c3.schema import load_schema  # noqa: E402
from ml.c3.values import hours_judgment_basis, parse_hours_verdict  # noqa: E402

LABELER = "deal-kit estimator draft (Max's kits)"
VIEW = "pre_quote"
EXCLUDE = {
    "010270": "carries other deals' documents", "010328": "carries other deals' documents",
    "000051": "kit holdout", "010095": "kit holdout", "010253": "kit holdout", "010287": "kit holdout",
    "010038": "C3 holdout", "010043": "C3 validation", "010387": "C3 holdout",
    "010003": "duplicate of a Run E view deal", "010172": "duplicate of a Run E view deal",
}
QUOTE_HEADS = ("read:co_hours_estimate", "read:labor_hours", "read:crew_size", "read:equipment_qty",
               "jdg:hours:reason", "deal:primary_service")
# Max's job type -> deal:primary_service, only where it is one service (what the person labels on the 17 use).
SERVICE = {"conference_room": "audio_visual", "tv_mount": "audio_visual", "camera_mast": "security_camera",
           "camera_install": "security_camera", "ap_swap": "wireless", "ap_install": "wireless",
           "cabling_ap_install": "wireless", "cabling": "low_voltage_cabling", "cable_cleanup": "low_voltage_cabling",
           "phones": "telecom", "smart_hands": "staff_augmentation", "staff_aug": "staff_augmentation",
           "relocation": "imac", "server_relocation": "imac", "workstation_deployment": "imac",
           "rack_install": "rack_and_stack", "server_install": "rack_and_stack"}
# Our pricing / scoping paper: a file (not a message) named as ours. Our SOWs are "<deal> - <customer>
# (m.d.yy_v1)"; a partner's or customer's SOW / proposal stays an input (010300's CDW SOW).
OWN_FILE = re.compile(r"deal ?kit|\(\d{1,2}\.\d{1,2}\.\d{2}_v|purtera[^/]*(quote|sow|proposal|rom\b|estimate)"
                      r"|\brom\b|\bloe\b|\bcalc\b|rate card|cost breakdown|^quote[ _-]", re.I)
MESSAGE = re.compile(r"-hs-(note|email|meeting|call|task)-|fireflies|\.eml$|transcript", re.I)
# Our own pricing said in a note or email ("I say we quote 2 techs...") is the answer, never an input
# (deal-kit-training build_rows.py OWN_QUOTE; to_c3.py kept these lines in the c3files).
OWN_QUOTE = re.compile(r"\bI say we quote\b|\brevised ROM\b|\bROM quote\b|\battached a ROM\b", re.I)


def is_own_file(name):
    return bool(name) and not MESSAGE.search(name) and bool(OWN_FILE.search(name))


def _when(a):
    md = a.get("metadata") or {}
    for v in (a.get("authored_at"), md.get("hubspotFileUpdatedAt"), md.get("hubspotNoteUpdatedAt")):
        try:
            t = _time(v)
        except ValueError:
            t = 0.0
        if t > 0:
            return t
    try:   # "Wed Aug 12 2026 15:15:16 GMT+0000 (...)": the upload, an upper bound on arrival
        return datetime.datetime.strptime(str(a.get("created_at"))[:24], "%a %b %d %Y %H:%M:%S") \
            .replace(tzinfo=datetime.timezone.utc).timestamp()
    except ValueError:
        return 0.0


def head_scale(keys):
    return {k: 0.0 for k in keys if k not in QUOTE_HEADS}


def label_key(atom_id):
    return "lbl_" + hashlib.sha1(atom_id.encode()).hexdigest()[:20]


def first_pricing(deal, mdir):
    """(time and name of our first pricing/scoping file, every document seen), from all the deal's manifests."""
    docs = {}
    for p in glob.glob(f"{mdir}/{deal}/*.json"):
        try:
            m = json.load(open(p))
        except ValueError:
            continue
        for a in m.get("artifacts") or []:
            key = a.get("content_sha256") or a.get("filename")
            t = _when(a)
            if key not in docs or 0 < t and (docs[key][0] <= 0 or t < docs[key][0]):
                docs[key] = (t, a.get("filename") or "")
    own = sorted((t, f) for t, f in docs.values() if is_own_file(f) and t > 0)
    return (own[0] if own else (float("inf"), None)), docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", default=f"{HOME}/c3-local/runE/train_estimator_work/manifests")
    args = ap.parse_args()
    schema = load_schema()
    keys = [o.key for o in schema.opportunities]
    scale = head_scale(keys)
    service_opp = schema.by_key()["deal:primary_service"]
    assert _is_a_person(LABELER)
    src = json.load(open(TARGETS))
    T, date = src["targets"], src.get("date")
    tasks = json.load(open(TASKS))
    tasks.pop("_about", None)
    by_deal = collections.defaultdict(list)
    for r in T:
        if r.get("train") is True:
            by_deal[r["deal"]].append(r)
    os.makedirs(OUT, exist_ok=True)
    rows, problems, cut_rows = [], [], []
    per_head = collections.Counter()
    for deal in sorted(by_deal):
        if deal in EXCLUDE or deal in FIXED_HOLDOUT:
            continue
        blob = json.load(open(f"{C3FILES}/{deal}.json"))
        assert str(blob["deal_id"]) == deal
        atoms = [a for a in blob["atoms"] if not OWN_QUOTE.search(a.get("text") or "")]
        dropped_own_quote = len(blob["atoms"]) - len(atoms)
        by_id = {a["atom_id"]: a for a in atoms}
        # Cut check: every input arrived strictly before our first pricing file; none is one of ours.
        (t_quote, own_name), docs = first_pricing(deal, args.manifests)
        as_of = _time(((blob.get("source") or {}).get("cutoff") or {}).get("as_of"))
        latest = max(_time(a.get("entered_at")) for a in atoms)
        own_in = sorted({a["doc_id"] for a in atoms if is_own_file(a.get("doc_id") or "")})
        late = sum(1 for a in atoms if _time(a.get("entered_at")) >= t_quote)
        untimed = sum(1 for a in atoms if _time(a.get("entered_at")) <= 0)
        cut_rows.append((deal, len(docs), own_name, t_quote, as_of, latest, late, len(own_in), untimed,
                         dropped_own_quote))
        if not docs:
            problems.append(f"{deal}: no manifest, cut not checked")
        if late or own_in:
            problems.append(f"{deal}: {late} atoms at/after our first pricing file, {len(own_in)} own docs in inputs")
        cards = []
        for r in by_deal[deal]:
            t = r["target"]
            a = by_id.get(t["atom_id"])
            if a is None or a["label_key"] != label_key(t["atom_id"]) or a["label_key"] != t["label_key_c3files"] \
                    or not a["text"].startswith(t["text"]):     # the targets file cuts text at 300 chars
                problems.append(f"{deal}: target {r['id']} does not resolve to its atom")
                continue
            parsed = parse_hours_verdict(r["verdict"])
            basis = hours_judgment_basis(r["verdict"])
            assert parsed and basis in ("stated", "estimate"), r["verdict"]
            per_head["read:labor_hours" if basis == "stated" else "read:co_hours_estimate"] += 1
            cards.append({
                "id": r["id"], "deal_id": deal, "head": "hours", "target_key": f"atom:{a['atom_id']}",
                "text": a["text"], "parser_value": None, "verdict": r["verdict"], "note": "", "reason": "",
                "labeler": LABELER, "why_author": "machine_draft", "purpose": "train",
                "judged_at": date, "fields": {},
                "target": {"a": {"text": a["text"], "atomId": a["atom_id"], "atomType": t.get("atom_type") or "",
                                 "filename": a.get("doc_id") or "", "section": []},
                           "head": "hours", "text": a["text"], "targetKey": f"atom:{a['atom_id']}"},
            })
        if not cards:
            continue
        svc = SERVICE.get((tasks.get(deal) or {}).get("task") or "")
        answers = []
        if svc:
            assert service_opp.index(svc) is not None, svc
            answers.append({"deal_id": deal, "labeler": LABELER, "primary_service": svc, "purpose": "train"})
            per_head["deal:primary_service"] += 1
        out = {
            "deal_id": f"{deal}@{VIEW}", "company": blob.get("company") or "purtera",
            "atoms": atoms, "labels": [], "judgments": cards, "links": [], "deal_answers": answers,
            "missed_history": [], "head_scale": scale, "view": VIEW, "view_stage": "quoting",
            "view_cut": {"quote_at": t_quote if t_quote != float("inf") else None, "as_of": as_of},
            "source": [blob.get("source"), "deal-kit-training/c3files", "runE/estimator_targets.json",
                       "runE/build_estimator.py"],
        }
        with open(f"{OUT}/{deal}@{VIEW}.json", "w") as fh:
            json.dump(out, fh)
        rows.append((deal, len(atoms), len(cards), svc or "-"))
    print(f"{'deal':7} {'atoms':>5} {'cards':>5} service")
    for r in rows:
        print(f"{r[0]:7} {r[1]:5} {r[2]:5} {r[3]}")
    print(f"files {len(rows)}  cards {sum(r[2] for r in rows)}  per head {dict(per_head)}")
    print(f"head_scale: {len(scale)} of {len(keys)} opportunities at 0; trained: {', '.join(QUOTE_HEADS)}")
    print("CUT deal    docs  first_pricing_at      as_of                 last_input            late own untimed own_quote_lines_dropped  first pricing file")
    fmt = lambda t: "-" if t in (None, float("inf")) or not t else \
        datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M")
    for deal, nd, own_name, tq, ao, last, late, own_in, untimed, dq in cut_rows:
        print(f"CUT {deal} {nd:5}  {fmt(tq):20}  {fmt(ao):20}  {fmt(last):20}  {late:4} {own_in:3} {untimed:3} {dq:3}"
              f"  {(own_name or 'no pricing file found')[:60]}")
    bad = [r[0] for r in cut_rows if r[6] or r[7]]
    print(f"CUT SUMMARY {len(cut_rows)} deals: {sum(1 for r in cut_rows if r[3] != float('inf'))} with a pricing file found, "
          f"{len(bad)} with an input at/after it or an own file in the inputs {bad}; "
          f"{sum(r[9] for r in cut_rows)} own-quote lines dropped")
    print("PROBLEMS", len(problems))
    for p in problems:
        print("  ", p)


if __name__ == "__main__":
    main()
