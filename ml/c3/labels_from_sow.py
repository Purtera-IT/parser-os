"""Training records from the SOW writer's rule decisions and the PM's reviews.

    python -m ml.c3.labels_from_sow --envelopes /outside/final_*_env.json \
        [--reviews /outside/sow_reviews.jsonl] --out /outside/sow_train/

Every rule in ``app/core/c3_sow.py`` that moves, drops or re-parties a cited
line records the call in ``c3_sow.rule_decisions`` (rule, head, atom_ids,
value, was). Those calls are what the super heads have to learn before the rule
can be retired (owner rule: heads decide, the parser extracts). This turns them
into the trainer's records, one file per deal, in the ``loop.load_deal``
contract:

* a decision the PM ACCEPTED in SowSmith's "Teach the heads" queue, or a PM's
  own edit (move or delete in the Studio), is a human label at the ordinary
  tier;
* a decision the PM REJECTED is a counter-label: ``sow_keep`` → keep, a
  ``sow_section`` move → the section it came from (``was``), unless the PM
  gave the right one (``corrected``);
* an unreviewed rule decision is a silver label at the ``slight`` tier (0.3),
  so a human label always outweighs it.

Heads: ``sow_section`` → ``read:sow_section`` (base). ``sow_keep`` drop →
``read:co_action`` reject (company layer) only for the noise rules (small talk,
pitch, brochure, captions), where "not in the SOW" means "not a fact of this
job"; other drops (a duplicate, a contradiction) say nothing about the line
itself and are skipped.

Reviews file (JSONL, written from the Studio's edit log / review queue):
    {"deal", "atom_id", "head", "value", "verdict": "accepted"|"rejected"|"pm_edit",
     "corrected"?: str, "by"?: str, "at"?: str}

No source text is written anywhere but the output directory.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

from c3_record import build_rows

NOISE_RULES = ("grounding",)  # the noise filters run inside _grounding


def _rows(env: dict) -> list[dict]:
    docs = [{"doc_id": d.get("artifact_id"),
             "kind": "email" if str(d.get("artifact_type") or "").lower() in ("eml", "email", "msg")
             else ("note" if d.get("hubspot_note") else "file"),
             "sent_at": (d.get("doc_origin") or {}).get("sent_at") if isinstance(d.get("doc_origin"), dict) else None,
             "authored_at": d.get("authored_at")} for d in env.get("documents") or []]
    lines, seen = [], set()
    for i, a in enumerate(env.get("atoms") or []):
        aid = str(a.get("id") or f"atom:{i}")
        if aid in seen:
            continue
        seen.add(aid)
        sb = (a.get("structured") or {}).get("said_by") or {}
        lines.append({"key": aid, "text": str(a.get("text") or ""), "doc_id": a.get("artifact_id") or "",
                      "section": " > ".join(str(x) for x in a.get("section_path") or [] if str(x).strip()),
                      "order": float(i), "speaker_role": sb.get("role") or sb.get("role_guess") or "",
                      "speaker_side": sb.get("side") or "", "atom_id": aid})
    return build_rows(lines, docs, latest_fallback=0.0)


def _read(head: str, value: str, rule: str) -> tuple[str, str] | None:
    """The trainer's (reads key, answer) for one decision, from c3_sow.TEACHABLE."""
    from app.core.c3_sow import TEACHABLE

    spec = TEACHABLE.get(head)
    if not spec or not value:
        return None
    key, mapping = spec
    if head == "sow_keep" and value == "drop" and rule not in NOISE_RULES and rule != "pm":
        return None  # a duplicate or a contradiction says nothing about the line itself
    answer = mapping.get(value) if mapping else value
    if not answer:
        return None
    return key, answer


def _target(read: tuple[str, str]) -> dict:
    """A read: head is stored under reads_set; a col: head under its own field
    (the label row's ``supplier``), as the labeling page writes them."""
    key, answer = read
    kind, name = key.split(":", 1)
    return {name: answer, "reads_set": {}} if kind == "col" else {"reads_set": {name: answer}}


#: Real-SOW alignment sections → read:sow_section answers.
_REAL_SECTION = {"materials_customer": "customer_responsibilities", "materials_provider": "provider_responsibilities",
                 "schedule": "timeline"}


def real_sow_reviews(paths: list[str]) -> list[dict]:
    """PurTera's own first-draft SOWs, aligned line by line to the atoms they
    came from (sow_slot/sem_g*.jsonl, kind "stated"): the strongest labels we
    have — the section the PM actually put the fact in, and that it belongs in
    the SOW at all. Turned into accepted reviews so they train as gold."""
    out = []
    for path in paths:
        for line in open(path):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("kind") != "stated" or not r.get("atom_ids"):
                continue
            sec = _REAL_SECTION.get(r.get("label"), r.get("label"))
            for aid in r["atom_ids"][:1]:
                out.append({"deal": r["deal"], "atom_id": aid, "head": "sow_section", "value": sec,
                            "verdict": "accepted", "by": "real_sow"})
                out.append({"deal": r["deal"], "atom_id": aid, "head": "sow_keep", "value": "keep",
                            "verdict": "accepted", "by": "real_sow"})
    return out


def build(envelopes: list[str], reviews: list[dict], out_dir: str) -> dict[str, int]:
    by_deal: dict[str, list[dict]] = {}
    for r in reviews:
        by_deal.setdefault(str(r.get("deal")), []).append(r)
    os.makedirs(out_dir, exist_ok=True)
    counts = {"gold": 0, "silver": 0, "counter": 0, "deals": 0}
    for path in envelopes:
        env = json.loads(Path(path).read_text())
        crm = env.get("crm") or {}
        deal = str(crm.get("deal_name") or "")[:6] or Path(path).stem
        sow = env.get("c3_sow") or {}
        rows = _rows(env)
        have = {r["key"]: r["text"] for r in rows}
        labels: dict[tuple[str, str], dict] = {}
        rev = {(str(r.get("atom_id")), str(r.get("head"))): r for r in by_deal.get(deal, [])}
        # Human reviews and PM edits first: they win.
        for (aid, head), r in rev.items():
            if aid not in have:
                continue
            verdict = str(r.get("verdict") or "")
            value = str(r.get("corrected") or r.get("value") or "")
            if verdict == "rejected" and not r.get("corrected"):
                value = "keep" if head == "sow_keep" else str(r.get("was") or "")
            read = _read(head, value, "pm")
            if not read or not read[1]:
                continue
            labels[(aid, read[0])] = dict(_target(read), label_key=aid, atom_id=aid, labeler=r.get("by") or "pm@purtera-it.com",
                                          origin="labeler", text=have[aid], weight_tier="ordinary")
            counts["counter" if verdict == "rejected" else "gold"] += 1
        # Unreviewed rule decisions: silver.
        for d in sow.get("rule_decisions") or []:
            read = _read(str(d.get("head")), str(d.get("value")), str(d.get("rule")))
            if not read:
                continue
            for aid in (d.get("atom_ids") or [])[:1]:
                if aid not in have or (aid, read[0]) in labels:
                    continue
                labels[(aid, read[0])] = dict(_target(read), label_key=aid, atom_id=aid, labeler="rule:" + str(d.get("rule")),
                                              origin="labeler", text=have[aid], weight_tier="slight")
                counts["silver"] += 1
        if not labels:
            continue
        # One label row per atom: merge the reads.
        merged: dict[str, dict] = {}
        for (aid, _), lb in labels.items():
            m = merged.setdefault(aid, dict(lb, reads_set={}))
            m["reads_set"].update(lb.get("reads_set") or {})
            for k2, v2 in lb.items():
                if k2 not in ("reads_set",) and k2 not in m:
                    m[k2] = v2
            if "supplier" in lb:
                m["supplier"] = lb["supplier"]
            if lb["weight_tier"] == "ordinary":
                m["weight_tier"] = "ordinary"
        rec = {"deal_id": deal, "company": "purtera", "atoms": [dict(r, label_key=r["key"]) for r in rows],
               "labels": list(merged.values()), "judgments": [], "links": [], "deal_answers": []}
        Path(out_dir, f"{deal}.json").write_text(json.dumps(rec))
        counts["deals"] += 1
    return counts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--envelopes", nargs="+", required=True)
    p.add_argument("--reviews", default=None)
    p.add_argument("--real-sow", nargs="*", default=[], help="sem_g*.jsonl real-SOW alignments (gold)")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    paths = [x for pat in a.envelopes for x in glob.glob(pat)] or a.envelopes
    reviews = [json.loads(l) for l in open(a.reviews)] if a.reviews and os.path.exists(a.reviews) else []
    reviews += real_sow_reviews([x for pat in a.real_sow for x in glob.glob(pat)])
    print(json.dumps(build(paths, reviews, a.out)))


if __name__ == "__main__":
    main()
