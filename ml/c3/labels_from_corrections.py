"""Training records for the MAIN super heads from the corrections queue.

    python -m ml.c3.labels_from_corrections --queue <run>/queue/queue.jsonl \
        --envelopes <export>/envelopes --out <run>/rows [--approve approvals.json]

Input: ``scripts/main_heads_queue.py``'s ``queue.jsonl`` (one PM correction
exemplar per row, with the atom it resolved to in the deal's latest envelope).
Output: one record per deal in the ``loop.load_deal`` contract (atoms + labels
+ judgments), split by deal into ``train/``, ``val/`` (picks the refit's epoch)
and ``holdout/`` (never seen by the refit; the nightly's held-out report), plus
``index.json`` naming, per head, the files that carry its labels.

What each head trains as (the shapes the labeling page already writes):

* ``col:label_type``  the line's label row ``label_type``;
* ``col:admission``   an ``admission`` judgment (alias of col:admission);
* ``col:supplier``    a ``bom_owner`` judgment (alias: we_supply -> us, customer_furnished -> customer);
* ``jdg:site``        a ``site`` pair judgment; ``jdg:site_role`` / ``jdg:facility`` line judgments;
* ``jdg:conflict``    a ``conflict`` pair judgment, which also draws ``rel:supports`` on a "supports".

Drafts. A row a person did not make (an agent, a Deal Kit publish, a model)
is written with labeler ``main-heads-loop (assistant)`` and ``why_author:
machine_draft``: ``data.select_labels`` and ``featurize`` skip it, so it never
trains as gold. The owner approves drafts by correction id in ``--approve``
(``{"approved": [{"id": "<correction id>", "by": "<who>"}]}``); an approved
draft is labeled ``<who> [owner-approved draft]`` and trains like a person's.
Where a person and a draft answer the same line, only the person's row is
written. Two people disagreeing on one line: the later correction wins and the
clash is counted, never silently merged.

No source text is written anywhere but ``--out``. Never imports ``app``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .labels_from_sow import _rows
from .schema import load_schema

DRAFT_LABELER = "main-heads-loop (assistant)"
APPROVED_DRAFT = "[owner-approved draft]"
#: Queue head -> how it is stored on the record.
LABEL_FIELD = {"type": "label_type"}
JUDGMENT_HEADS = {"admission", "bom_owner", "site", "site_role", "facility", "conflict"}
#: Dense attention budget the refit reads with (slot_fit --max-atoms).
MAX_ATOMS = 8192


def split_of(deal: str, mod: int = 5) -> str:
    """Stable by deal: 1 in ``mod`` held out, 1 in ``mod`` validation, the rest train."""
    h = int(hashlib.sha1(deal.encode()).hexdigest(), 16) % mod
    return "holdout" if h == 0 else "val" if h == 1 else "train"


def read_approvals(path: str | None) -> dict[str, str]:
    """correction id -> who approved it."""
    if not path:
        return {}
    data = json.loads(Path(path).read_text())
    out = {}
    for a in data.get("approved") or []:
        if not a.get("id") or not str(a.get("by") or "").strip():
            raise ValueError("Each approved draft needs its correction id and who approved it")
        out[str(a["id"])] = str(a["by"]).strip()
    return out


def labeler_of(row: dict[str, Any], approvals: dict[str, str]) -> tuple[str, bool]:
    """(labeler, trains): a person's own name, an approved draft, or the draft marker."""
    if row.get("by_person"):
        return str(row.get("author") or "pm"), True
    who = approvals.get(str(row.get("correction_id")))
    if who:
        return f"{who} {APPROVED_DRAFT}", True
    return DRAFT_LABELER, False


def answer_ok(schema, head: str, key: str, verdict: str) -> bool:
    """The verdict is one the head can learn (vocabulary or alias map)."""
    if head in schema.judgment_aliases:
        return bool(schema.judgment_aliases[head][1].get(verdict))
    opp = schema.by_key().get(key)
    return opp is not None and opp.index(verdict) is not None


def _subject(r: dict[str, Any]) -> tuple:
    return (r["head"], tuple(sorted(r["atoms"])))


def pick(rows: list[dict[str, Any]], approvals: dict[str, str], counts: Counter) -> list[tuple[dict[str, Any], str, bool]]:
    """One answer per (head, line or pair): a trained row (person or approved)
    beats a draft; among trained rows the latest wins and a disagreement is
    counted as a clash."""
    by: dict[tuple, list[tuple[dict[str, Any], str, bool]]] = defaultdict(list)
    for r in rows:
        lab, trains = labeler_of(r, approvals)
        by[_subject(r)].append((r, lab, trains))
    out = []
    for _, cands in by.items():
        trained = [c for c in cands if c[2]]
        pool = trained or cands
        pool.sort(key=lambda c: c[0].get("at") or 0)
        win = pool[-1]
        if len({c[0]["verdict"] for c in pool}) > 1:
            counts["clash_" + ("person" if trained else "draft")] += 1
        if trained and len(trained) < len(cands):
            counts["draft_shadowed_by_person"] += len(cands) - len(trained)
        out.append(win)
    return out


def deal_record(env: dict[str, Any], deal: str, rows: list[dict[str, Any]], approvals: dict[str, str],
                schema, counts: Counter) -> tuple[dict[str, Any] | None, Counter]:
    """One deal's training record, and per-head counts of the rows it carries."""
    # Lines the correction named that admission set aside are lines too.
    extra = {a for r in rows for a in r["atoms"]} - {str(a.get("id")) for a in env.get("atoms") or []}
    sup = [dict(s, id=str(s.get("id"))) for s in env.get("suppressed") or [] if str(s.get("id")) in extra]
    lines = _rows(dict(env, atoms=list(env.get("atoms") or []) + sup))
    have = {ln["key"]: ln for ln in lines}
    labels: dict[str, dict[str, Any]] = {}
    judgments: list[dict[str, Any]] = []
    carried: Counter = Counter()
    usable = []
    for r in rows:
        if any(a not in have for a in r["atoms"]):
            counts["atom_not_in_record"] += 1
            continue
        if not answer_ok(schema, r["head"], r["key"], r["verdict"]):
            counts[f"bad_answer:{r['key']}={r['verdict']}"] += 1
            continue
        usable.append(r)
    for r, lab, trains in pick(usable, approvals, counts):
        tag = ("person" if r.get("by_person") else "approved") if trains else "draft"
        prov = {"correction_id": r["correction_id"], "source": "pm_correction"}
        if not trains:
            prov.update(why_author="machine_draft", drafted_by=r.get("author") or "")
        if r["head"] in LABEL_FIELD:
            aid = r["atoms"][0]
            # A person's row and a draft row for one line stay separate rows:
            # select_labels keeps the person's and skips the draft.
            slot = f"{aid}|{'t' if trains else 'd'}"
            lb = labels.setdefault(slot, {"label_key": aid, "atom_id": aid, "labeler": lab,
                                          "weight_tier": "ordinary", **({} if trains else
                                                                       {"why_author": "machine_draft"})})
            lb[LABEL_FIELD[r["head"]]] = r["verdict"]
            lb.setdefault("corrections", []).append(r["correction_id"])
        elif r["head"] in JUDGMENT_HEADS:
            a = r["atoms"]
            j = {"id": r["id"], "head": r["head"], "verdict": r["verdict"], "labeler": lab, **prov}
            if r["size"] == "pair":
                j["target"] = {"a": {"atomId": a[0]}, "b": {"atomId": a[1]}}
                j["text"] = f"{have[a[0]]['text']} || {have[a[1]]['text']}"
                j["target_key"] = f"pair:{a[0]}:{a[1]}"
            else:
                j["target"] = {"a": {"atomId": a[0]}}
                j["text"] = have[a[0]]["text"]
                j["target_key"] = f"atom:{a[0]}"
            judgments.append(j)
        else:
            continue
        carried[(r["key"], tag)] += 1
        for also in r.get("also") or []:
            carried[(also, tag)] += 1
    if not labels and not judgments:
        return None, carried
    if len(lines) > MAX_ATOMS:
        counts["deal_over_attention_budget"] += 1
        return None, Counter()
    rec = {"deal_id": deal, "company": "purtera", "atoms": [dict(ln, label_key=ln["key"]) for ln in lines],
           "labels": list(labels.values()), "judgments": judgments, "links": [], "deal_answers": []}
    return rec, carried


def build(queue: list[dict[str, Any]], envelopes: str, out_dir: str, approvals: dict[str, str] | None = None,
          mod: int = 5) -> dict[str, Any]:
    approvals = approvals or {}
    schema = load_schema()
    counts: Counter = Counter()
    by_deal: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in queue:
        if r.get("skip"):
            counts["skipped_in_queue"] += 1
            continue
        by_deal[r["deal_uuid"]].append(r)
    out = Path(out_dir)
    index: dict[str, dict[str, Any]] = {}
    files = {"train": 0, "val": 0, "holdout": 0}
    for uuid, rows in sorted(by_deal.items()):
        p = Path(envelopes).expanduser() / f"{uuid}.json"
        if not p.exists():
            counts["no_envelope"] += 1
            continue
        env = json.loads(p.read_text())
        deal = next((r["deal"] for r in rows if r.get("deal")), "") or uuid
        rec, carried = deal_record(env, deal, rows, approvals, schema, counts)
        if rec is None:
            continue
        part = split_of(deal, mod)
        (out / part).mkdir(parents=True, exist_ok=True)
        path = out / part / f"{deal}.json"
        path.write_text(json.dumps(rec))
        files[part] += 1
        for (key, tag), n in carried.items():
            h = index.setdefault(key, {})
            s = h.setdefault(part, {"files": [], "person": 0, "approved": 0, "draft": 0})
            if tag != "draft" and str(path) not in s["files"]:
                s["files"].append(str(path))   # a file trains this head only if it has trained rows
            s[tag] += n
    summary = {"files": files, "heads": index, "counts": dict(counts), "approved_drafts": len(approvals)}
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.json").write_text(json.dumps(summary, indent=1))
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--queue", required=True)
    p.add_argument("--envelopes", required=True, help="folder of <deal uuid>.json")
    p.add_argument("--out", required=True)
    p.add_argument("--approve", default=None, help='owner approvals {"approved": [{"id", "by"}]}')
    p.add_argument("--split-mod", type=int, default=5)
    a = p.parse_args(argv)
    queue = [json.loads(line) for line in Path(a.queue).read_text().splitlines() if line.strip()]
    s = build(queue, a.envelopes, a.out, read_approvals(a.approve), a.split_mod)
    print(json.dumps({"files": s["files"], "counts": s["counts"],
                      "heads": {k: {part: {t: v[t] for t in ("person", "approved", "draft")} for part, v in h.items()}
                                for k, h in s["heads"].items()}}))
    if not sum(s["files"].values()):
        print("FATAL labels_from_corrections: no training record written")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
