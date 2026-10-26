"""Nightly triage of PM fixes to the heads' SOW: which kind of fix each one is.

    python scripts/sow_fix_triage.py --edits ~/c3-local/sow_fix/exports \\
        --envs ~/c3-local/heads_loop/envs/nightly --out ~/c3-local/sow_fix/runs/20261019
    # add --post to send the head labels to the DEV feedback route (dry run otherwise)

Input, per deal, SowSmith's edit export (the Studio's ``scope_process_v1``):
    {deal, deal_id, compile_id, sowEditLog: [...], sowFixNotes: [...], standardsProposals: [...]}
and that deal's envelope (``final_<deal>_env.json``), whose ``c3_sow`` carries the cites,
``rule_decisions`` and ``writer_draft``, and ``atom_trace`` (app/core/c3_sow_trace; computed here
when an older envelope lacks it).

Every PM change to a heads line lands in exactly one bucket:

* ``head``      a super head called it wrong → a label for sow_section / sow_keep, posted through
                the dev feedback route (``POST /projects/{id}/feedback/correction``) and written as
                ``reviews.jsonl`` rows ``ml.c3.labels_from_sow --reviews`` reads;
* ``parser``    the parser misread the source (the atom's type or text) → a parser label to make on
                the labeling page; never posted from here;
* ``template``  PurTera standard handling for the job family → ``standards_queue.jsonl``, for the
                owner to approve (never applied here);
* ``writer``    the writer or one of its rules got it wrong while the heads were right → a bug
                note with deal + line + trace (``writer_bugs.jsonl``);
* ``ask``       genuinely ambiguous → one question for the PM (``questions.jsonl``).

The PM's own reason (a "fix this line" note) always wins. Without one the agent decides from the
trace, defending the heads first (owner rule): a head is blamed only when its own call disagrees
with the fix, the line says what its source says, and no rule made the call. Section judgments use
docs/C3_RUBRIC_SOW_SECTION.md's first-matching-test order.

Dev only: --post refuses any feedback URL that is not the dev service.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core import c3_sow_trace  # noqa: E402

DEV_FEEDBACK = "https://parser-os-service-dev-eus2.whitehill-a3348ba5.eastus2.azurecontainerapps.io"

#: Studio list → the sow_section answer it prints under (diffHeadsSow.SOW_SECTION_OF).
SECTION_OF = {
    "overview": "overview",
    "scope": "scope_of_work",
    "customer_responsibilities": "customer_responsibilities",
    "assumptions": "assumptions",
    "out_of_scope": "out_of_scope",
    "deliverables": "deliverables",
}
SOW_SECTION_CANDIDATES = [
    "overview", "scope_of_work", "deliverables", "acceptance", "out_of_scope", "assumptions",
    "customer_responsibilities", "provider_responsibilities", "timeline", "fees", "change_control",
    "terms", "contacts", "other",
]
#: The c3_sow lists each Studio section prints from (lineTrace.C3_LISTS_OF).
C3_LISTS_OF = {
    "overview": ["overview"],
    "scope": ["schedule", "arrival_protocol", "scope", "equipment_in_scope", "provider_materials",
              "provider_tools", "customer_materials"],
    "customer_responsibilities": ["customer_responsibilities"],
    "assumptions": ["assumptions"],
    "out_of_scope": ["out_of_scope"],
    "deliverables": ["deliverables"],
}

_STOP = set("""a an the and or of to for in on at by with from as is are be been will shall
must may can any all each per its it this that these those their our your into onto upon
before after during than then so not no""".split())


def norm(s: Any) -> str:
    s = re.sub(r"[*_`#]", " ", str(s or "").lower())
    return re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", s)).strip()


def content_words(s: str) -> set[str]:
    return {w for w in norm(s).split() if w not in _STOP and len(w) > 2}


def coverage(line: str, sources: list[str]) -> float:
    """Share of the line's content words its sources contain (a paraphrase keeps most of them)."""
    words = content_words(line)
    if not words:
        return 1.0
    have: set[str] = set()
    for s in sources:
        have |= content_words(s)
    # Stems, so "mounted" covers "mount" and "visible" covers "visibility".
    stems = {w[:5] for w in have}
    return sum(1 for w in words if w in have or w[:5] in stems) / len(words)


def dice(a: str, b: str) -> float:
    """Token overlap 2|A∩B|/(|A|+|B|), the FE's tokenOverlap (diffHeadsSow.ts)."""
    ta, tb = set(norm(a).split()), set(norm(b).split())
    return 2 * len(ta & tb) / (len(ta) + len(tb)) if ta and tb else 0.0


_GENERIC_CAPS = {"provider", "customer", "customers", "technician", "partner", "reseller", "sow", "the", "a", "an",
                 "all", "any", "each", "no", "not"}


def specifics(s: str) -> set[str]:
    """What a line can add or drop as a FACT: numbers, tokens with digits (CAT6, MS120), all-caps
    names (PoE aside) and capitalised words that do not start a sentence (Neat, Plano, Bernie)."""
    out = set(re.findall(r"\b\d[\d,./:-]*\b", s))
    out |= {w.lower() for w in re.findall(r"\b[A-Za-z]+\d[\w-]*\b", s)}
    for m in re.finditer(r"\b[A-Z][A-Za-z0-9-]+\b", s):
        w = m.group(0)
        before = s[: m.start()].rstrip()
        sentence_start = not before or before[-1] in ".:;!?(\"“-•"
        if w.lower() in _GENERIC_CAPS or (sentence_start and not w.isupper()):
            continue
        out.add(w.lower())
    return out


# ── The rubric's first-matching test, as words (docs/C3_RUBRIC_SOW_SECTION.md) ─────────────────

_RUBRIC: list[tuple[str, re.Pattern[str]]] = [
    ("_absent", re.compile(r"\b(thanks|thank you|regards|signature|\$\s?\d|price|pricing|quote #|invoice)\b", re.I)),
    ("out_of_scope", re.compile(r"\b(not (in scope|included|responsible)|will not|won't|excluded|exclud\w+|out of scope|no (cabling|electrical|disposal))\b", re.I)),
    ("timeline", re.compile(r"\b(\d{1,2}/\d{1,2}(/\d{2,4})?|\d{1,2}(:\d{2})?\s?(am|pm)|business hours|after hours|weekend|monday|tuesday|wednesday|thursday|friday|per week|deadline|by end of)\b", re.I)),
    ("deliverables", re.compile(r"\b(photos?|checklist|serial numbers?|as-?builts?|sign-?off|completion report|report)\b", re.I)),
    ("customer_responsibilities", re.compile(r"^(customer|the customer|reseller|partner)\b|\b(customer (will|to|shall|must|provides?|ships?|configures?)|ship(s|ped)? (the )?equipment|escort|provide (access|credentials|badges?|parking))\b", re.I)),
    ("provider_responsibilities", re.compile(r"\b(tools?|ladders?|lifts?|vehicles?|consumables|materials provided by provider)\b", re.I)),
    ("scope_of_work", re.compile(r"^(install|mount|remove|test|cable|connect|configure|dismantle|pack|label|verify|haul|support|move|route|conceal|power|replace|assist|arrive|meet|check in|unbox|stage)\b|\b(provider will|technician will)\b", re.I)),
    ("assumptions", re.compile(r"\b(existing|assum\w+|is available|are available|can be|will remain|no (lift|asbestos)|accessible|reachable|standard working)\b", re.I)),
]


def rubric_section(text: str) -> str | None:
    for section, pat in _RUBRIC:
        if pat.search(text):
            return section
    return None


# ── Inputs ───────────────────────────────────────────────────────────────────────────────────

def load_exports(path: str) -> list[dict[str, Any]]:
    files = sorted(glob.glob(os.path.join(path, "*.json"))) if os.path.isdir(path) else [path]
    out = []
    for f in files:
        try:
            j = json.load(open(f))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  skip {f}: {exc}", file=sys.stderr)
            continue
        if isinstance(j, dict) and j.get("deal"):
            j["_file"] = f
            out.append(j)
    return out


def load_envelope(envs: str, deal: str) -> dict[str, Any] | None:
    for pat in (f"final_{deal}_env.json", f"*{deal}*env*.json"):
        hits = sorted(glob.glob(os.path.join(envs, pat)))
        if hits:
            env = json.load(open(hits[0]))
            sow = env.get("c3_sow") or {}
            if isinstance(sow, dict) and not sow.get("error") and "atom_trace" not in sow:
                c3_sow_trace.attach(env)
            return env
    return None


def latest_steps(log: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The newest step of each edit, in the order the edits were first made."""
    first: dict[str, int] = {}
    last: dict[str, dict[str, Any]] = {}
    for i, e in enumerate(log or []):
        if not isinstance(e, dict) or not e.get("id"):
            continue
        first.setdefault(e["id"], i)
        last[e["id"]] = e
    return [last[k] for k in sorted(last, key=lambda k: first[k])]


# ── The trace (the same evidence the Studio box shows; lib/sowsmith/fix/lineTrace.ts) ──────────

def find_item(sow: dict[str, Any], section: str | None, text: str) -> dict[str, Any] | None:
    n = norm(text)
    lists = C3_LISTS_OF.get(section or "", []) + [k for ks in C3_LISTS_OF.values() for k in ks]
    for k in lists:
        for it in sow.get(k) or []:
            if isinstance(it, dict) and norm(it.get("text") or it.get("item")) == n:
                return it
    return None


def build_trace(env: dict[str, Any], section: str | None, text: str, cites: list[str]) -> dict[str, Any]:
    sow = env.get("c3_sow") or {}
    if isinstance(sow, dict) and "atom_trace" not in sow and not sow.get("error"):
        c3_sow_trace.attach(env)
    item = find_item(sow, section, text)
    if not cites and item:
        cites = c3_sow_trace.cite_ids(item.get("cites"))
    words = {norm(text)}
    draft = None
    best = 0.0
    for lst, xs in (sow.get("writer_draft") or {}).items():
        for it in xs or []:
            t = str((it or {}).get("text") or "")
            if not t or it.get("kind") == "heading":
                continue
            if norm(t) == norm(text):
                draft = {"list": lst, "text": t, "fate": "same"}
                break
            shares = bool(set(c3_sow_trace.cite_ids(it.get("cites"))) & set(cites))
            sim = dice(text, t)
            # A shared cite makes a loose reword the same line; without one it must read nearly the same.
            if (shares and sim >= 0.35 or sim >= 0.6) and sim > best:
                best, draft = sim, {"list": lst, "text": t, "fate": "reworded"}
        if draft and draft["fate"] == "same":
            break
    if draft:
        words.add(norm(draft["text"]))
    rules = [d for d in sow.get("rule_decisions") or []
             if d.get("ledger") != "lost_line"
             and (set(d.get("atom_ids") or []) & set(cites) or norm(d.get("text")) in words)]
    traces = sow.get("atom_trace") or {}
    return {
        "cites": cites,
        "standard": bool(item and item.get("standard")),
        "kind": (item or {}).get("kind"),
        "atoms": {a: traces.get(a) or {"missing": True} for a in cites},
        "rules": [{k: d.get(k) for k in ("rule", "head", "value", "was", "atom_ids")} for d in rules],
        "draft": draft,
    }


def shared_lines(env: dict[str, Any], cites: list[str], text: str) -> list[str]:
    sow = env.get("c3_sow") or {}
    out = []
    for ks in C3_LISTS_OF.values():
        for k in ks:
            for it in sow.get(k) or []:
                if not isinstance(it, dict) or norm(it.get("text")) == norm(text):
                    continue
                if set(c3_sow_trace.cite_ids(it.get("cites"))) & set(cites):
                    out.append(str(it.get("text")))
    return out


def paper_names(env: dict[str, Any]) -> set[str]:
    """The parties a contract's own paper names: the reseller account, the end customer, and the
    roles the paper gives them. A line naming only these is template, whatever its capitals."""
    sow = env.get("c3_sow") or {}
    names = {"subcontractor", "subcontractors", "contractor", "statement", "work", "sow", "services", "project",
             "change", "order", "technical", "contacts", "point", "contact", "international", "corp", "inc", "llc"}
    for raw in (((sow.get("crm") or {}).get("account_name")), sow.get("end_customer")):
        names |= {w.lower() for w in re.findall(r"[A-Za-z0-9]+", str(raw or ""))}
    return names


def trace_of(env: dict[str, Any], aid: str) -> dict[str, Any]:
    """One atom's trace, also for an atom no SOW line cites (built the way c3_sow_trace builds it)."""
    t = ((env.get("c3_sow") or {}).get("atom_trace") or {}).get(aid)
    if t:
        return t
    probe = {**env, "c3_sow": {"rule_decisions": [{"atom_ids": [aid]}]}}
    return c3_sow_trace.build_atom_trace(probe).get(aid, {})


def find_atom_for(env: dict[str, Any], text: str, min_cov: float = 0.7) -> tuple[str, float] | None:
    """The one atom that states a PM-written line best (a fact the heads may have left out). An atom
    holding every specific of the line (its numbers and names) needs only half its words."""
    best: tuple[str, float] | None = None
    want = specifics(text)
    for a in env.get("atoms") or []:
        t = str(a.get("text") or "")
        if len(t) < 12 or len(t) > 800:
            continue
        cov = coverage(text, [t])
        bar = 0.5 if len(want) >= 2 and want <= specifics(t) else min_cov
        if cov >= bar and (best is None or cov > best[1]):
            best = (str(a.get("id")), round(cov, 2))
    return best


# ── Deciding ─────────────────────────────────────────────────────────────────────────────────

def _label(head: str, target: str, old: str, new: str, text: str, why: str) -> dict[str, Any]:
    return {"head": head, "target_id": target, "old_value": old, "new_value": new, "text": text, "why": why}


def classify(edit: dict[str, Any], note: dict[str, Any] | None, env: dict[str, Any],
             misread: set[str] | frozenset[str] = frozenset()) -> dict[str, Any]:
    """One PM change → {bucket, why, confidence, label?, proposal?, question?, trace}.

    ``misread``: atoms the PM marked "source misread" elsewhere in this deal."""
    kind = edit.get("kind")
    frm, to = edit.get("fromList"), edit.get("toList")
    text = str(edit.get("text") or "")
    draft_text = str(edit.get("prevText") or text)
    cites = [c for c in edit.get("cites") or [] if c]
    tr = build_trace(env, frm or to, draft_text, cites)
    cites = tr["cites"]
    atoms = tr["atoms"]
    sources = [str(a.get("text") or "") for a in atoms.values() if not a.get("missing")]
    job = str((env.get("c3_sow") or {}).get("job_family") or "unknown")
    out: dict[str, Any] = {"edit_id": edit.get("id"), "kind": kind, "from": frm, "to": to, "text": text,
                           "draft_text": draft_text if draft_text != text else None, "trace": tr,
                           "already_sent": edit.get("label") if edit.get("sent") else None}

    def head_label_for_edit(why: str) -> dict[str, Any] | None:
        if not cites:
            return None
        if kind == "move" and frm and to and SECTION_OF.get(frm) != SECTION_OF.get(to):
            return _label("sow_section", cites[0], SECTION_OF[frm], SECTION_OF[to], text, why)
        if kind == "delete":
            return _label("sow_keep", cites[0], "keep", "drop", text, why)
        return None

    def template(why: str, action: str) -> dict[str, Any]:
        return {**out, "bucket": "template", "why": why, "confidence": "high" if note else "medium",
                "proposal": {"job_family": job, "action": action, "section": to or frm, "text": text,
                             "was": draft_text if action == "reword" else None, "note": (note or {}).get("note")}}

    # 1. The PM said why: their reason wins.
    if note:
        r = note.get("reason")
        words = f" PM: \"{note['note']}\"" if note.get("note") else ""
        if r == "head":
            lab = head_label_for_edit(f"PM marked it a wrong head call.{words}")
            if lab:
                return {**out, "bucket": "head", "why": lab["why"], "confidence": "high", "label": lab}
            return {**out, "bucket": "head", "why": f"PM taught the heads in the Studio's Teach sheet.{words}",
                    "confidence": "high", "label": None}
        if r == "source":
            return {**out, "bucket": "parser", "confidence": "high",
                    "why": f"PM marked the source misread.{words}", "parser": parser_findings(atoms)}
        if r == "template":
            action = "remove" if kind == "delete" else "move" if kind == "move" else "reword" if kind in ("edit", "reword") else "add"
            return template(f"PM marked it PurTera standard handling.{words}", action)
        if r == "writer":
            return {**out, "bucket": "writer", "confidence": "high", "why": f"PM marked it a writer mistake.{words}"}

    # 2. No reason given: decide from the trace, defending the heads first.
    if kind == "add":
        hit = find_atom_for(env, text)
        if hit:
            aid, cov = hit
            t = trace_of(env, aid)
            dropped_by = [d for d in (env.get("c3_sow") or {}).get("rule_decisions") or []
                          if aid in (d.get("atom_ids") or []) and d.get("head") == "sow_keep" and d.get("value") == "drop"
                          and d.get("ledger") != "lost_line"]
            if dropped_by:
                return {**out, "bucket": "writer", "confidence": "medium",
                        "why": f"The PM added a fact atom {aid} states; rule {dropped_by[0]['rule']} dropped it after the heads kept it.",
                        "matched_atom": {"id": aid, "coverage": cov, "type": t.get("type"), "brief": t.get("brief")}}
            return {**out, "bucket": "head", "confidence": "medium",
                    "why": f"The PM added a fact atom {aid} states ({int(cov * 100)}% of its words); the heads left it out of the SOW.",
                    "label": _label("sow_keep", aid, "drop", "keep", text, "PM-added line its source states"),
                    "matched_atom": {"id": aid, "coverage": cov, "type": t.get("type"), "brief": t.get("brief")}}
        if not specifics(text):
            return template("A generic line no source states: a candidate for the job family's standards.", "add")
        paper = paper_names(env)
        if specifics(text) <= paper:
            account = str(((env.get("c3_sow") or {}).get("crm") or {}).get("account_name") or "reseller")
            r = template(f"Contract-paper wording naming only the parties ({', '.join(sorted(specifics(text)))}): "
                         f"{account}'s SOW template, not a deal fact.", "add")
            r["proposal"]["job_family"] = f"paper:{account.lower()}"
            return r
        return {**out, "bucket": "ask", "confidence": "low",
                "why": "The PM added a specific fact no atom states.",
                "question": f"Where does \"{text}\" come from? Link it to its source in the Studio (Link to source), or say it is PurTera standard."}

    if not cites:
        if tr["standard"]:
            action = {"delete": "remove", "move": "move"}.get(kind, "reword")
            return template(f"A PurTera standard line for {job} jobs the PM {'removed' if kind == 'delete' else 'changed'}.", action)
        return {**out, "bucket": "writer", "confidence": "medium",
                "why": "The writer's own uncited method line; no head read it."}

    flagged = sorted(set(cites) & set(misread))
    if flagged and kind in ("delete", "move", "edit", "reword"):
        return {**out, "bucket": "parser", "confidence": "medium", "parser": parser_findings({a: atoms[a] for a in flagged}),
                "why": f"The PM marked {', '.join(flagged)} a source misread on another line; this fix follows from the same atom."}

    calls = {a: t for a, t in atoms.items() if not t.get("missing")}
    rule_made = [d for d in tr["rules"]
                 if (kind == "move" and d["head"] == "sow_section") or (kind == "delete" and d["head"] == "sow_keep" and d["value"] == "keep")]
    cov = coverage(draft_text, sources) if sources else 0.0

    if kind in ("edit", "reword"):
        added = specifics(text) - specifics(draft_text)
        src_specs: set[str] = set()
        for s in sources:
            src_specs |= specifics(s)
        if added and added <= src_specs:
            return {**out, "bucket": "writer", "confidence": "high",
                    "why": f"The PM put back {sorted(added)}, which the cited source states and the writer dropped."}
        elsewhere = [str(x.get("id")) for x in env.get("atoms") or []
                     if added and added <= specifics(str(x.get("text") or ""))]
        if added and elsewhere:
            return {**out, "bucket": "writer", "confidence": "medium",
                    "why": f"The PM put back {sorted(added)}; atom {elsewhere[0]} states them though this line does not cite it (the writer missed it)."}
        if added:
            return {**out, "bucket": "ask", "confidence": "low",
                    "why": f"The PM added {sorted(added)}, which no cited source states.",
                    "question": f"\"{text}\": where do {', '.join(sorted(added))} come from? A missed source (link it) or PM knowledge?"}
        return {**out, "bucket": "writer", "confidence": "medium",
                "why": "A reword that adds no fact: a writer wording example, not a head label."}

    if kind == "delete":
        others = shared_lines(env, cites, draft_text)
        if rule_made:
            return {**out, "bucket": "writer", "confidence": "high",
                    "why": f"Rule {rule_made[0]['rule']} kept it over the heads; the rule is the suspect."}
        placed = {t.get("section") for t in calls.values() if t.get("section")}
        if placed and SECTION_OF.get(frm or "") not in placed:
            return {**out, "bucket": "writer", "confidence": "medium",
                    "why": f"The heads placed its source in {sorted(placed)}; the writer printed it in {frm}. The PM removed the writer's placement, not the fact."}
        if cov < 0.45:
            return {**out, "bucket": "writer", "confidence": "medium",
                    "why": f"The line says what its source does not ({int(cov * 100)}% of its words are in the cited atoms): the writer, not the head."}
        if others:
            return {**out, "bucket": "ask", "confidence": "low",
                    "why": "The atom also feeds other lines the PM kept; a keep=drop label would drop them too.",
                    "question": f"You removed \"{text}\" but kept {others[:2]} from the same source. Is the source itself not part of this job, or only this line wrong?"}
        low = [a for a, t in calls.items() if (t.get("type_p") or 1) < 0.6]
        lab = _label("sow_keep", cites[0], "keep", "drop", text,
                     "PM removed a line its source states; the keep head kept it")
        return {**out, "bucket": "head", "confidence": "medium" if not low else "low",
                "why": lab["why"] + (f" (type head unsure on {low})" if low else ""), "label": lab}

    if kind == "move":
        want = SECTION_OF.get(to or "", "")
        said = {t.get("section") for t in calls.values() if t.get("section")}
        if said and said == {want}:
            return {**out, "bucket": "writer", "confidence": "high",
                    "why": f"The heads already placed it in {want}; the writer put it in {frm}."}
        if rule_made:
            lab = _label("sow_section", cites[0], SECTION_OF.get(frm or "", ""), want, text,
                         f"rule {rule_made[0]['rule']} moved it; PM moved it to {want}")
            return {**out, "bucket": "head", "confidence": "medium",
                    "why": f"Rule {rule_made[0]['rule']} moved it to {frm}; the PM's {want} is a label for the head that should replace that rule.",
                    "label": lab, "rule_flag": rule_made[0]["rule"]}
        rub = rubric_section(text)
        lab = _label("sow_section", cites[0], SECTION_OF.get(frm or "", ""), want, text,
                     f"PM moved it to {want}" + (f"; the rubric agrees" if rub == want else ""))
        if rub == want:
            return {**out, "bucket": "head", "confidence": "high", "why": f"The rubric's first test puts it in {want}, as the PM did.", "label": lab}
        if rub and rub == SECTION_OF.get(frm or ""):
            return {**out, "bucket": "ask", "confidence": "low", "label": lab,
                    "why": f"The rubric's first test says {rub}, where it was.",
                    "question": f"\"{text}\": the section rubric reads it as {rub}. Keep your move to {want}? (A label against the rubric splits the head's training.)"}
        return {**out, "bucket": "head", "confidence": "medium", "why": f"The PM moved it to {want}; heads said {sorted(s for s in said if s) or 'nothing'}.", "label": lab}

    return {**out, "bucket": "ask", "confidence": "low", "why": f"Unknown edit kind {kind!r}.", "question": "?"}


def parser_findings(atoms: dict[str, dict[str, Any]]) -> list[str]:
    """What looks misread on the cited atoms: a merged block, an unsure or flipped type."""
    out = []
    for a, t in atoms.items():
        if t.get("missing"):
            out.append(f"{a}: not in this compile's parse")
            continue
        text = str(t.get("text") or "")
        sentences = len(re.findall(r"[.!?](\s|$)", text))
        if len(text) >= 400 or sentences >= 3:
            out.append(f"{a}: one atom holds {sentences} sentences / {len(text)} chars (segmentation)")
        if (t.get("type_p") or 1) < 0.6:
            out.append(f"{a}: type head unsure ({t.get('type')} {t.get('type_p')})")
        if t.get("base_type") and t["base_type"][0] != t.get("type"):
            out.append(f"{a}: base said {t['base_type'][0]}, head says {t.get('type')}")
    return out or ["no automatic finding; the PM's Teach-sheet label carries the fix"]


# ── Notes ↔ edits ────────────────────────────────────────────────────────────────────────────

def note_for(edit: dict[str, Any], notes: list[dict[str, Any]]) -> dict[str, Any] | None:
    hit = None
    for n in notes:  # newest wins
        if n.get("editId") and n["editId"] == edit.get("id"):
            hit = n
        elif not n.get("editId") and n.get("kind") in ("edit", "look") and edit.get("kind") in ("reword", "edit"):
            if norm(n.get("after") or n.get("text")) == norm(edit.get("text")) or norm(n.get("text")) == norm(edit.get("prevText")):
                hit = n
    return hit


def look_notes(notes: list[dict[str, Any]], edits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Notes on lines the PM did not change (`t`, then a reason): fixes in their own right."""
    used_text = {norm(e.get("text")) for e in edits} | {norm(e.get("prevText")) for e in edits}
    return [n for n in notes if n.get("kind") == "look" and norm(n.get("text")) not in used_text]


# ── Run ──────────────────────────────────────────────────────────────────────────────────────

def triage_deal(exp: dict[str, Any], env: dict[str, Any]) -> list[dict[str, Any]]:
    notes = [n for n in exp.get("sowFixNotes") or [] if isinstance(n, dict)]
    edits = latest_steps(exp.get("sowEditLog") or [])
    misread = {a for n in notes if n.get("reason") == "source" for a in (n.get("trace") or {}).get("cites") or []}
    rows = []
    for e in edits:
        n = note_for(e, notes)
        r = classify(e, n, env, misread)
        r["note"] = n
        if n:  # what the agent would have said alone: the measure of whether it can stop asking
            r["agent_alone"] = classify(e, None, env, misread - set((n.get("trace") or {}).get("cites") or []) if n.get("reason") == "source" else misread)["bucket"]
        rows.append(r)
    for n in look_notes(notes, edits):
        pseudo = {"id": n.get("id"), "kind": "look", "fromList": n.get("section"), "text": n.get("text"),
                  "cites": (n.get("trace") or {}).get("cites") or []}
        r = classify({**pseudo, "kind": "move" if n.get("to") else "look"}, n, env, misread)
        r["kind"] = "look"
        r["note"] = n
        rows.append(r)
    for r in rows:
        r["deal"] = exp["deal"]
        r["deal_id"] = exp.get("deal_id") or exp["deal"]
        r["compile_id"] = exp.get("compile_id")
    return rows


def group_questions(asks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One question per deal and shared specific term: five lines naming "Neat Center" that no
    source states are one question to the PM, not five."""
    out: list[dict[str, Any]] = []
    by_deal: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in asks:
        by_deal[r["deal"]].append(r)
    for deal, rs in by_deal.items():
        counts = Counter(t for r in rs if r["kind"] == "add" for t in specifics(r["text"]))
        done: set[str] = set()
        for term, n in counts.most_common():
            if n < 2:
                break
            grp = [r for r in rs if r["kind"] == "add" and r["edit_id"] not in done and term in specifics(r["text"])]
            if len(grp) < 2:
                continue
            done |= {r["edit_id"] for r in grp}
            out.append({"deal": deal, "edit_ids": [r["edit_id"] for r in grp], "lines": [r["text"] for r in grp],
                        "question": f"{len(grp)} lines you added name \"{term}\", which no source of this deal states. "
                                    f"Which document says it? (Link one line to it in the Studio; the rest follow.)"})
        for r in rs:
            if r["edit_id"] not in done:
                out.append({"deal": deal, "edit_ids": [r["edit_id"]], "lines": [r["text"]], "question": r["question"]})
    return out


def correction_body(r: dict[str, Any]) -> dict[str, Any]:
    lab = r["label"]
    head = lab["head"]
    cands = SOW_SECTION_CANDIDATES if head == "sow_section" else ["keep", "drop"]
    return {"head": head, "deal_id": r["deal_id"], "compile_id": r.get("compile_id") or "",
            "target_id": lab["target_id"], "text": lab["text"], "old_value": lab["old_value"],
            "new_value": lab["new_value"], "scope": "deal", "rationale": f"sow_fix_triage: {r['why']}",
            "candidates": cands, "pm": ((r.get("note") or {}).get("by") or "sow_fix_triage")}


def post(bodies: list[dict[str, Any]], base: str, token: str) -> list[dict[str, Any]]:
    import urllib.request

    if "-dev-" not in base:
        raise SystemExit(f"refusing to post to a non-dev feedback service: {base}")
    out = []
    for b in bodies:
        req = urllib.request.Request(f"{base}/projects/{b['deal_id']}/feedback/correction", data=json.dumps(b).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                out.append({"ok": True, "status": resp.status, "target_id": b["target_id"]})
        except Exception as exc:  # recorded, never fatal: the next night retries
            out.append({"ok": False, "error": str(exc)[:200], "target_id": b["target_id"]})
    return out


def report_md(rows: list[dict[str, Any]], run: dict[str, Any]) -> str:
    by = Counter(r["bucket"] for r in rows)
    L = [f"# SOW fix triage — {run['at']}", "",
         f"Mode: **{'POST (dev)' if run['post'] else 'dry run'}** · deals: {', '.join(run['deals']) or '—'} · fixes: {len(rows)}", "",
         "| bucket | n | what happens |", "|---|---|---|",
         f"| head | {by['head']} | label to the dev feedback route + reviews.jsonl for labels_from_sow |",
         f"| parser | {by['parser']} | parser label on the labeling page (not posted from here) |",
         f"| template | {by['template']} | standards_queue.jsonl, owner approves |",
         f"| writer | {by['writer']} | writer_bugs.jsonl with the trace |",
         f"| ask | {by['ask']} | questions.jsonl for the PM ({len(group_questions([r for r in rows if r['bucket'] == 'ask']))} question(s) after grouping) |", ""]
    noted = [r for r in rows if r.get("agent_alone")]
    if noted:
        agree = sum(1 for r in noted if r["agent_alone"] == r["bucket"])
        L += [f"**Agent vs PM reasons:** the agent alone agreed on {agree}/{len(noted)} fixes the PM explained"
              + (": " + "; ".join(f"“{r['text'][:50]}” PM {r['bucket']} vs agent {r['agent_alone']}" for r in noted if r['agent_alone'] != r['bucket']) if agree < len(noted) else "."), ""]
    sent = [r for r in rows if r.get("already_sent")]
    if sent:
        L += [f"Labels the Studio already sent on its own: {len(sent)} (never re-posted)", ""]
    if run.get("missing_envelopes"):
        L += [f"Skipped (no envelope): {', '.join(run['missing_envelopes'])}", ""]
    for deal in run["deals"]:
        L += [f"## {deal}", ""]
        for r in [x for x in rows if x["deal"] == deal]:
            pm = r.get("note")
            src = f"PM: {pm['reason']}" if pm else "agent"
            L.append(f"- **{r['bucket']}** ({r['confidence']}, {src}) · {r['kind']}"
                     + (f" {r['from']}→{r['to']}" if r.get("to") and r.get("from") else f" {r.get('from') or r.get('to') or ''}")
                     + f" · “{r['text'][:110]}”")
            L.append(f"  - why: {r['why']}" + (f" (agent alone: {r['agent_alone']})" if r.get("agent_alone") and r["agent_alone"] != r["bucket"] else ""))
            if r.get("label"):
                lab = r["label"]
                sent = r.get("already_sent")
                state = "already sent by the Studio" if sent and sent.get("head") == lab["head"] and sent.get("new_value") == lab["new_value"] else (
                    "posted" if run["post"] else "would post")
                L.append(f"  - label ({state}): `{lab['head']}` {lab['target_id']} {lab['old_value']} → **{lab['new_value']}**")
            elif r.get("already_sent"):
                s = r["already_sent"]
                L.append(f"  - ⚠ the Studio already sent `{s['head']}` {s['old_value']} → {s['new_value']}; this triage disagrees: counter-label needed")
            if r.get("proposal"):
                p = r["proposal"]
                L.append(f"  - standards proposal: {p['action']} in {p['job_family']} / {p['section']}" + (f" — “{p['note']}”" if p.get("note") else ""))
            if r.get("parser"):
                L += [f"  - parser: {x}" for x in r["parser"]]
            if r.get("question"):
                L.append(f"  - question for the PM: {r['question']}")
            tr = r["trace"]
            calls = []
            for a, t in tr["atoms"].items():
                if t.get("missing"):
                    calls.append(f"{a} (missing)")
                    continue
                bits = [f"type {t.get('type')} {t.get('type_p')}"]
                if t.get("section"):
                    bits.append(f"section {t['section']}")
                if t.get("keep_p") is not None:
                    bits.append(f"keep {t['keep_p']}")
                if t.get("supplier"):
                    bits.append(f"supplier {t['supplier'][0]} {t['supplier'][1]}")
                if not t.get("section") and t.get("brief"):
                    bits.append("brief " + "/".join(t["brief"]))
                calls.append(f"{a}: " + ", ".join(bits))
            if calls:
                L.append("  - heads: " + "; ".join(calls))
            if tr["rules"]:
                L.append("  - rules: " + "; ".join(f"{d['rule']} {d['head']}={d['value']} (was {d['was']})" for d in tr["rules"]))
            if tr["draft"]:
                L.append(f"  - writer draft ({tr['draft']['fate']}): “{tr['draft']['text'][:110]}”")
        L.append("")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--edits", required=True, help="a SowSmith edit export (.json) or a folder of them")
    ap.add_argument("--envs", required=True, help="folder of final_<deal>_env.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--post", action="store_true", help="send head labels to the DEV feedback route (default: dry run)")
    ap.add_argument("--feedback-url", default=os.environ.get("PARSER_OS_DEV_URL", DEV_FEEDBACK))
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    run: dict[str, Any] = {"at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), "post": a.post,
                           "deals": [], "missing_envelopes": []}
    for exp in load_exports(a.edits):
        env = load_envelope(a.envs, exp["deal"])
        if env is None:
            run["missing_envelopes"].append(exp["deal"])
            continue
        if exp.get("compile_id") and env.get("compile_id") and exp["compile_id"] != env["compile_id"]:
            print(f"  {exp['deal']}: edits were made on compile {exp['compile_id']}, envelope is {env['compile_id']}; atom ids may not match",
                  file=sys.stderr)
        run["deals"].append(exp["deal"])
        rows += triage_deal(exp, env)

    def dump(name: str, items: list[dict[str, Any]]) -> None:
        with open(out / name, "w") as f:
            for x in items:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")

    to_post = [r for r in rows if r["bucket"] == "head" and r.get("label") and r.get("confidence") != "low"
               and not (r.get("already_sent") and r["already_sent"].get("new_value") == r["label"]["new_value"])]
    seen: set[tuple[str, str, str]] = set()
    bodies = []
    for r in to_post:  # one label per atom, head and answer, whichever fix found it first
        k = (r["label"]["target_id"], r["label"]["head"], r["label"]["new_value"])
        if k not in seen:
            seen.add(k)
            bodies.append(correction_body(r))
    dump("would_post.jsonl", bodies)
    dump("reviews.jsonl", [{"deal": r["deal"], "atom_id": r["label"]["target_id"], "head": r["label"]["head"],
                            "value": r["label"]["new_value"], "verdict": "pm_edit", "was": r["label"]["old_value"],
                            "by": (r.get("note") or {}).get("by") or "sow_fix_triage", "at": run["at"]}
                           for r in rows if r["bucket"] == "head" and r.get("label") and r.get("confidence") != "low"])
    dump("standards_queue.jsonl", [{**r["proposal"], "deal": r["deal"], "status": "pending_owner"} for r in rows if r.get("proposal")])
    dump("writer_bugs.jsonl", [{k: r.get(k) for k in ("deal", "compile_id", "kind", "from", "to", "text", "draft_text", "why", "trace")}
                               | {"pm_note": (r.get("note") or {}).get("note")} for r in rows if r["bucket"] == "writer"])
    dump("parser_labels.jsonl", [{"deal": r["deal"], "atoms": list(r["trace"]["atoms"]), "findings": r.get("parser"), "text": r["text"]}
                                 for r in rows if r["bucket"] == "parser"])
    dump("questions.jsonl", group_questions([r for r in rows if r["bucket"] == "ask"]))
    if a.post and bodies:
        token = os.environ.get("PARSER_OS_DEV_BEARER", "")
        if not token:
            raise SystemExit("--post needs PARSER_OS_DEV_BEARER (az containerapp secret show -n parser-os-service-dev-eus2 ...)")
        run["posted"] = post(bodies, a.feedback_url.rstrip("/"), token)
    (out / "report.json").write_text(json.dumps({"run": run, "rows": rows}, indent=1, ensure_ascii=False))
    (out / "report.md").write_text(report_md(rows, run))
    print(f"{len(rows)} fixes · " + " · ".join(f"{k} {v}" for k, v in Counter(r['bucket'] for r in rows).most_common())
          + f" · {len(bodies)} label(s) {'posted' if a.post else 'would post'} → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
