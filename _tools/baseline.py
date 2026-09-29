# -*- coding: utf-8 -*-
"""Snapshot a labelled deal's compile, so a change can be proved not to lose quality.

The rule this enforces is the one that has been learned the hard way here: an
impact run is the gate -- same files twice, old tree against new, LOST 0 before
shipping. Tests written alongside a fix share its blind spot; gold labels do not,
because they were written before the change existed.

The measure that matters is **label re-attachment**. A `label_key` is built from
(deal, filename, page, text). It is the one thing that is supposed to survive a
re-compile, so every gold label that stops resolving is a fact the parser used to
produce and no longer does -- renamed, re-split, re-worded or dropped. That is
quality loss, and unlike an atom count it cannot be hidden by a change that loses
one atom and gains another.

    DEAL=<uuid> PG=... python baseline.py before        # capture
    ... deploy / change something, re-compile ...
    DEAL=<uuid> PG=... python baseline.py after         # capture
    DEAL=<uuid> PG=... python baseline.py --diff before after
"""
import collections, json, os, sys
from pathlib import Path

import psycopg2
from azure.storage.blob import BlobServiceClient

DEAL = os.environ["DEAL"]
HERE = Path(__file__).parent
SNAP = HERE / "_baselines"


def _container():
    conn = (HERE / ".bloburl").read_text(encoding="utf-8").strip()
    return BlobServiceClient.from_connection_string(conn).get_container_client(
        "orbitbrief-artifacts")


def capture(tag: str) -> dict:
    cc = _container()
    env = json.loads(cc.download_blob(
        f"deals/{DEAL}/orbitbrief/latest/envelope.json").readall())
    res = json.loads(cc.download_blob(
        f"deals/{DEAL}/parser-os/latest/result.json").readall())
    props = cc.get_blob_client(
        f"deals/{DEAL}/orbitbrief/latest/envelope.json").get_blob_properties()

    sys.path.insert(0, str(HERE.parent))
    from app.core.label_key import label_key

    docs = {d.get("artifact_id"): (d.get("filename") or "") for d in (env.get("documents") or [])}
    live_keys, atoms = set(), []
    for a in env.get("atoms") or []:
        loc = a.get("locator") or {}
        text = " ".join(str(a.get("text") or "").split())
        k = label_key(DEAL, docs.get(a.get("artifact_id"), ""), loc.get("page"), text)
        live_keys.add(k)
        atoms.append({"key": k, "type": a.get("atom_type"), "text": text[:120]})

    # Gold written by a PERSON. `human_labels` skips any other labeler, so rows
    # under a tool name are not gold and must not flatter this measure.
    cn = psycopg2.connect(os.environ["PG"], connect_timeout=30)
    cur = cn.cursor()
    cur.execute("""SELECT label_key, label_type, left(text, 110) FROM public.atom_labels
                    WHERE deal_id=%s AND labeler NOT ILIKE %s AND labeler LIKE %s""",
                (DEAL, "%claude%", "%@%"))
    gold = {r[0]: {"type": r[1], "text": r[2]} for r in cur.fetchall()}
    cur.execute("""SELECT from_key, to_label_key, relation FROM public.atom_label_links
                    WHERE deal_id=%s AND labeler NOT ILIKE %s AND labeler LIKE %s""",
                (DEAL, "%claude%", "%@%"))
    links = [(a, b, r) for a, b, r in cur.fetchall()]
    cur.close(); cn.close()

    attached = [k for k in gold if k in live_keys]
    lost = [k for k in gold if k not in live_keys]
    link_ok = [l for l in links if l[0] in live_keys and l[1] in live_keys]

    t = res.get("trace") or {}
    snap = {
        "tag": tag,
        "deal_id": DEAL,
        "compile_id": res.get("compile_id"),
        "envelope_written": props.last_modified.isoformat(),
        "totals": {
            "wall_s": round((t.get("total_duration_ms") or 0) / 1000, 1),
            "artifacts": t.get("artifact_count"),
            "atoms": len(atoms),
            "entities": t.get("entity_count"),
            "edges": t.get("edge_count"),
            "packets": t.get("packet_count"),
            "suppressed": len(res.get("suppressed_atoms") or []),
            "warnings": len(res.get("warnings") or []),
        },
        "gold": {
            "labels": len(gold),
            "attached": len(attached),
            "lost": len(lost),
            "links": len(links),
            "links_both_ends_live": len(link_ok),
        },
        "lost_detail": [{"key": k, **gold[k]} for k in lost][:40],
        "types": dict(collections.Counter(a["type"] for a in atoms).most_common()),
        "stages": {s["stage_name"]: round((s.get("duration_ms") or 0) / 1000, 1)
                   for s in (t.get("stages") or [])},
    }

    SNAP.mkdir(exist_ok=True)
    (SNAP / f"{DEAL[:8]}_{tag}.json").write_text(
        json.dumps(snap, indent=1, ensure_ascii=False), encoding="utf-8")

    g = snap["gold"]
    pct = 100 * g["attached"] / max(g["labels"], 1)
    print(f"[{tag}] compile {snap['compile_id']}  {snap['totals']['wall_s']}s  "
          f"{snap['totals']['atoms']} atoms")
    print(f"  GOLD RE-ATTACHED : {g['attached']}/{g['labels']}  ({pct:.1f}%)   LOST {g['lost']}")
    print(f"  links both ends  : {g['links_both_ends_live']}/{g['links']}")
    if lost:
        print("  lost labels (first 10):")
        for k in lost[:10]:
            print(f"     {gold[k]['type']:22} {gold[k]['text'][:70]}")
    print(f"  wrote {SNAP / f'{DEAL[:8]}_{tag}.json'}")
    return snap


def diff(a_tag: str, b_tag: str) -> None:
    a = json.loads((SNAP / f"{DEAL[:8]}_{a_tag}.json").read_text(encoding="utf-8"))
    b = json.loads((SNAP / f"{DEAL[:8]}_{b_tag}.json").read_text(encoding="utf-8"))
    print(f"=== {a_tag} -> {b_tag} ===")
    for k in a["totals"]:
        x, y = a["totals"][k], b["totals"][k]
        if x != y:
            d = (y - x) if isinstance(x, (int, float)) and isinstance(y, (int, float)) else ""
            print(f"  {k:12} {x} -> {y}   {f'({d:+})' if d != '' else ''}")

    ga, gb = a["gold"], b["gold"]
    print(f"\n  GOLD RE-ATTACHED {ga['attached']}/{ga['labels']} -> "
          f"{gb['attached']}/{gb['labels']}")
    regressed = gb["attached"] - ga["attached"]
    # The gate. An atom count can rise while gold detaches; only this says the
    # parser still produces the facts a human already confirmed it produced.
    if regressed < 0:
        print(f"  *** QUALITY LOST: {-regressed} gold label(s) stopped attaching. "
              f"DO NOT SHIP. ***")
        seen = {x["key"] for x in a["lost_detail"]}
        for x in b["lost_detail"]:
            if x["key"] not in seen:
                print(f"     NEWLY LOST  {x['type']:22} {x['text'][:70]}")
    elif regressed > 0:
        print(f"  gold attachment IMPROVED by {regressed}")
    else:
        print("  gold attachment unchanged -- no quality lost")

    print("\n  stage deltas (>0.5s):")
    for s in sorted(set(a["stages"]) | set(b["stages"]),
                    key=lambda s: -(b["stages"].get(s, 0) - a["stages"].get(s, 0))):
        x, y = a["stages"].get(s, 0), b["stages"].get(s, 0)
        if abs(y - x) > 0.5:
            print(f"     {s:32} {x:7.1f}s -> {y:7.1f}s  ({y - x:+.1f}s)")
    wa, wb = a["totals"]["wall_s"], b["totals"]["wall_s"]
    print(f"\n  WALL {wa}s -> {wb}s  ({wb - wa:+.1f}s, "
          f"{100 * (wb - wa) / max(wa, 1):+.0f}%)")


if __name__ == "__main__":
    if "--diff" in sys.argv:
        i = sys.argv.index("--diff")
        diff(sys.argv[i + 1], sys.argv[i + 2])
    else:
        capture(sys.argv[1] if len(sys.argv) > 1 else "snap")
