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


#: A placement's party: a provider task or a customer duty (task_owner_role).
_PARTY_OF = {"scope_of_work": "technician", "provider_responsibilities": "technician",
             "customer_responsibilities": "customer"}


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
                # Who acts is the part of a placement two SOW writers agree on
                # (84% real vs graded, against 45% for the section itself).
                if sec in _PARTY_OF:
                    out.append({"deal": r["deal"], "atom_id": aid, "head": "sow_party", "value": _PARTY_OF[sec],
                                "verdict": "accepted", "by": "real_sow"})
                out.append({"deal": r["deal"], "atom_id": aid, "head": "sow_keep", "value": "keep",
                            "verdict": "accepted", "by": "real_sow"})
    return out


#: A rubric owner as col:supplier, on lines that are work or a supply (who does or supplies it).
SUPPLIER_OF_OWNER = {"technician": "us", "customer": "customer", "partner": "partner", "other": "third_party"}
SUPPLY_SECTIONS = ("scope_of_work", "customer_responsibilities", "provider_responsibilities")


def supplier_reviews(reviews: list[dict]) -> list[dict]:
    """col:supplier answers implied by rubric or judge placements: the owner of
    a work or supply line is who supplies it (010326 "They will handle rack &
    stack gear": rubric customer_responsibilities / customer, the base said us
    0.728). Same source and verdict as the placement they come from."""
    sec = {(r["deal"], r["atom_id"]): r["value"] for r in reviews if r.get("head") == "sow_section"}
    out = []
    for r in reviews:
        if r.get("head") != "sow_party" or sec.get((r["deal"], r["atom_id"])) not in SUPPLY_SECTIONS:
            continue
        sup = SUPPLIER_OF_OWNER.get(r.get("value"))
        if sup:
            out.append(dict(r, head="sow_supplier", value=sup))
    return out


def rubric_reviews(paths: list[str]) -> list[dict]:
    """Lines labeled by the SOW section rubric (docs/C3_RUBRIC_SOW_SECTION.md):
    one section and one owner per line, judged by what the line does in the
    job. Two independent labelers agreed on 93-95% of sections and 95% of
    owners (against 45% between a real SOW and a graded one), so these train as
    gold. Where two labelers disagree the line is skipped. Input: the label
    files (``<batch>.A.jsonl``, optional ``<batch>.B.jsonl``) next to the batch
    file that carries each line's deal."""
    out = []
    for path in paths:
        batch = path.rsplit(".", 2)[0] + ".jsonl"
        if path.endswith(".B.jsonl") or not os.path.exists(batch):
            continue
        deal_of = {}
        for line in open(batch):
            r = json.loads(line)
            deal_of[r["id"]] = r["deal"]
        other = path[: -len(".A.jsonl")] + ".B.jsonl"
        second = {}
        if os.path.exists(other):
            for line in open(other):
                r = json.loads(line)
                second[r["id"]] = r
        for line in open(path):
            try:
                r = json.loads(line)
            except Exception:
                continue
            aid, deal = r.get("id"), deal_of.get(r.get("id"))
            if not aid or not deal:
                continue
            b = second.get(aid)
            sec, own = r.get("section"), r.get("owner")
            if sec and (b is None or b.get("section") == sec):
                out.append({"deal": deal, "atom_id": aid, "head": "sow_section", "value": sec,
                            "verdict": "accepted", "by": "rubric"})
            if own in ("technician", "customer", "partner", "other") and (b is None or b.get("owner") == own):
                out.append({"deal": deal, "atom_id": aid, "head": "sow_party", "value": own,
                            "verdict": "accepted", "by": "rubric"})
    return out


#: Who wrote a judge label: a model applying the rubric, never a person or a PM.
JUDGE_CREATED_BY = "judge"


def judge_reviews(paths: list[str]) -> list[dict]:
    """Lines placed by the rubric judge (jdg:sow_section, app/core/sow_rubric_judge.py)
    over deals no person labeled. Model drafts, kept apart from the gold: they
    train at the ``slight`` tier, lose to any rubric, real-SOW or PM answer on
    the same line, and never enter the feedback store (they live only in these
    files), so no run-time lookup can apply them as taught answers. Input:
    ``<batch>.J.jsonl`` ({"id", "deal", "section", "owner", "model"}), deal on
    each row."""
    out = []
    for path in paths:
        for line in open(path):
            try:
                r = json.loads(line)
            except Exception:
                continue
            aid, deal, sec, own = r.get("id"), r.get("deal"), r.get("section"), r.get("owner")
            if not aid or not deal or not sec:
                continue
            by = f"{JUDGE_CREATED_BY}:{r.get('model') or 'unknown'}"
            out.append({"deal": deal, "atom_id": aid, "head": "sow_section", "value": sec,
                        "verdict": "judge", "by": by})
            if own in ("technician", "customer", "partner", "other"):
                out.append({"deal": deal, "atom_id": aid, "head": "sow_party", "value": own,
                            "verdict": "judge", "by": by})
    return out


def outcome_reviews(envelopes: list[str]) -> list[dict]:
    """Where the graded SOW finally put each cited fact (the writer's LLM call
    plus every rule after it): one sow_section answer per atom, its first list
    only. The overview is a summary of other lines, not a placement, and is
    skipped. Silver: the SOW graders rated these drafts, they are not a PM's
    word."""
    from app.core.c3_sow import _LIST_SECTION

    out = []
    for path in envelopes:
        env = json.loads(Path(path).read_text())
        deal = str((env.get("crm") or {}).get("deal_name") or "")[:6] or Path(path).stem
        seen = set()
        for lst, sec in _LIST_SECTION.items():
            if lst == "overview":
                continue
            for line in (env.get("c3_sow") or {}).get(lst) or []:
                if not isinstance(line, dict) or line.get("kind") != "fact":
                    continue
                for aid in (line.get("cites") or [])[:1]:
                    if aid not in seen:
                        seen.add(aid)
                        out.append({"deal": deal, "atom_id": aid, "head": "sow_section", "value": sec,
                                    "verdict": "outcome", "by": "sow_outcome"})
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
        rev: dict[tuple[str, str], dict] = {}
        draft = ("outcome", "judge")  # model or draft answers: any human verdict beats them
        for r in by_deal.get(deal, []):
            k = (str(r.get("atom_id")), str(r.get("head")))
            if k not in rev or (rev[k].get("verdict") in draft and r.get("verdict") not in draft):
                rev[k] = r
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
            if verdict == "outcome":
                if (aid, read[0]) in labels:
                    continue
                labels[(aid, read[0])] = dict(_target(read), label_key=aid, atom_id=aid, labeler="sow_outcome",
                                              origin="labeler", text=have[aid], weight_tier="slight")
                counts["outcome"] = counts.get("outcome", 0) + 1
                continue
            if verdict == "judge":
                if (aid, read[0]) in labels:
                    continue
                labels[(aid, read[0])] = dict(_target(read), label_key=aid, atom_id=aid, labeler=r.get("by"),
                                              created_by=JUDGE_CREATED_BY, origin="labeler", text=have[aid],
                                              weight_tier="slight")
                counts["judge"] = counts.get("judge", 0) + 1
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
    p.add_argument("--rubric", nargs="*", default=[], help="rubric label files (<batch>.A.jsonl)")
    p.add_argument("--judge", nargs="*", default=[], help="rubric-judge label files (<batch>.J.jsonl), model drafts")
    p.add_argument("--supplier-from-owner", action="store_true",
                   help="derive col:supplier from rubric/judge owners on work and supply lines")
    p.add_argument("--outcomes", action="store_true", help="add the graded SOW's placements as silver")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    paths = [x for pat in a.envelopes for x in glob.glob(pat)] or a.envelopes
    reviews = [json.loads(l) for l in open(a.reviews)] if a.reviews and os.path.exists(a.reviews) else []
    # The rubric first: the first human answer for a line wins, and the rubric
    # is the consistent one (a real SOW's placement is one writer's choice).
    reviews += rubric_reviews([x for pat in a.rubric for x in glob.glob(pat)])
    reviews += real_sow_reviews([x for pat in a.real_sow for x in glob.glob(pat)])
    reviews += judge_reviews([x for pat in a.judge for x in glob.glob(pat)])
    if a.supplier_from_owner:
        reviews += supplier_reviews(reviews)
    if a.outcomes:
        reviews += outcome_reviews(paths)
    print(json.dumps(build(paths, reviews, a.out)))


if __name__ == "__main__":
    main()
