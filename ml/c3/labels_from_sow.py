"""Training records from the PM's corrections to the SOW (SowSmith).

    python -m ml.c3.labels_from_sow --envelopes /outside/final_*_env.json \
        [--reviews /outside/sow_reviews.jsonl] --out /outside/sow_train/

Owner policy (2026-10-27): a PM correction is an approved training label, so a
label from SowSmith is ONLY a real PM correction. Every rule in
``app/core/c3_sow.py`` still records its calls in ``c3_sow.rule_decisions``
(rule, head, atom_ids, value, was), but those are the rules' output, unreviewed:
they stay report-only (``rule_retire`` and the triage read them) and never
become a row here, at any tier. Nor does the SOW writer's own placement
(``verdict: outcome``).

What trains, in the ``loop.load_deal`` contract:

* a PM's own edit in the Studio (``verdict: pm_edit``: a move, a supply-list
  or party move, a delete), or a PM's verdict on a rule's call in the Teach
  queue (``accepted`` / ``rejected``): a human label at the ordinary tier;
* a REJECTED call is a counter-label: ``sow_keep`` → the opposite answer, a
  ``sow_section`` move → the section it came from (``was``), unless the PM gave
  the right one (``corrected``);
* external gold the PM's process owns (rubric labels, PurTera's own signed
  SOWs) at the ordinary tier, and rubric-judge drafts at ``slight``.

A PM delete (``sow_keep`` drop) says the line does not belong in THIS SOW: a
duplicate, a line a template covers, a wording the PM rewrote. That is not
"not a fact of the job", so it reaches ``read:co_action`` reject only when the
PM's reason says so (``reason`` in NOT_A_JOB_FACT, the fix box's "not a job
fact"); any other delete is skipped and counted.

Heads: ``sow_section`` → ``read:sow_section``, ``sow_party`` →
``read:task_owner_role``, ``sow_supplier`` → ``col:supplier``, ``sow_keep`` →
``read:co_action`` (see above).

Reviews file (JSONL, written from the Studio's edit log / review queue):
    {"deal", "atom_id", "head", "value", "verdict": "accepted"|"rejected"|"pm_edit",
     "corrected"?: str, "was"?: str, "reason"?: str, "note"?: str, "by"?: str, "at"?: str}

No source text is written anywhere but the output directory.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path
from typing import Any

from c3_record import build_rows

from .sow_tables import LIST_SECTION, TEACHABLE

NOISE_RULES = ("grounding",)  # the noise filters run inside _grounding (rule_retire's reading)
#: A PM delete's reason that says the line is not a fact of this job (the fix
#: box's reason, or the triage's): only then is a delete a co_action reject.
NOT_A_JOB_FACT = frozenset({"not_job_fact", "not a job fact", "not_a_job_fact", "noise"})
_NOT_A_FACT_WORDS = re.compile(r"(?i)\bnot (a |part of the |part of this )?(job|deal) fact\b|\bnot (part of )?(this|the) job\b|"
                               r"\bsmall talk\b|\bsignature\b|\bmarketing\b")


def not_a_job_fact(reason: Any) -> bool:
    """The PM's reason (a code or their words) says the line is not a fact of the job."""
    r = str(reason or "").strip().lower()
    return r in NOT_A_JOB_FACT or bool(_NOT_A_FACT_WORDS.search(r))


#: Verdicts a person gave. Anything else (a rule's call, the writer's outcome)
#: is not a PM correction and never trains from here.
PM_VERDICTS = frozenset({"pm_edit", "accepted", "rejected"})


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


def _read(head: str, value: str, rule: str, reason: str = "") -> tuple[str, str] | None:
    """The trainer's (reads key, answer) for one decision, from c3_sow.TEACHABLE.
    ``rule`` "pm" is a PM's own call: its delete trains co_action only with a
    not-a-job-fact ``reason``."""
    spec = TEACHABLE.get(head)
    if not spec or not value:
        return None
    key, mapping = spec
    if head == "sow_keep" and value == "drop":
        if rule == "pm" and not not_a_job_fact(reason):
            return None  # removed from this SOW (a duplicate, a template's line), not "not a job fact"
        if rule != "pm" and rule not in NOISE_RULES:
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
    only. For reports and comparisons only: ``build`` never trains on an
    outcome (it is the writer's output, not a PM's word)."""
    out = []
    for path in envelopes:
        env = json.loads(Path(path).read_text())
        deal = str((env.get("crm") or {}).get("deal_name") or "")[:6] or Path(path).stem
        seen = set()
        for lst, sec in LIST_SECTION.items():
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
    counts = {"gold": 0, "counter": 0, "deals": 0, "skipped_not_pm": 0, "skipped_keep_drop": 0,
              "rule_decisions_report_only": 0}
    for path in envelopes:
        env = json.loads(Path(path).read_text())
        crm = env.get("crm") or {}
        deal = str(crm.get("deal_name") or "")[:6] or Path(path).stem
        sow = env.get("c3_sow") or {}
        rows = _rows(env)
        have = {r["key"]: r["text"] for r in rows}
        labels: dict[tuple[str, str], dict] = {}
        rev: dict[tuple[str, str], dict] = {}
        draft = ("judge",)  # model drafts: any human verdict beats them
        for r in by_deal.get(deal, []):
            k = (str(r.get("atom_id")), str(r.get("head")))
            if k not in rev or (rev[k].get("verdict") in draft and r.get("verdict") not in draft):
                rev[k] = r
        # Human reviews and PM edits first: they win.
        for (aid, head), r in rev.items():
            if aid not in have:
                continue
            verdict = str(r.get("verdict") or "")
            if verdict not in PM_VERDICTS and verdict != "judge":
                counts["skipped_not_pm"] += 1  # the writer's outcome, a rule's call: not a PM correction
                continue
            value = str(r.get("corrected") or r.get("value") or "")
            if verdict == "rejected" and not r.get("corrected"):
                if head == "sow_keep":
                    value = "keep" if str(r.get("value") or "drop") == "drop" else "drop"
                else:
                    value = str(r.get("was") or "")
            read = _read(head, value, "pm", f"{r.get('reason') or ''} {r.get('note') or ''}")
            if head == "sow_keep" and value == "drop" and not read:
                counts["skipped_keep_drop"] += 1
                continue
            if not read or not read[1]:
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
        # Unreviewed rule decisions are the rules' output, not a PM's: never a
        # label (owner policy, 2026-10-27). Counted so the report shows them.
        counts["rule_decisions_report_only"] += len(sow.get("rule_decisions") or [])
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
    print(json.dumps(build(paths, reviews, a.out)))


if __name__ == "__main__":
    main()
