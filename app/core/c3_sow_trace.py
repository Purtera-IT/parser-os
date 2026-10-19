"""The trace behind each SOW line, small enough to ship with ``c3_sow``.

SowSmith's "fix this line" box shows a PM how a line came to be: the source
words -> the atom(s) -> what the heads called them (type, section, keep,
supplier, with confidence) -> each c3_sow rule that moved, kept or dropped it
-> the line. ``c3_sow`` already carries the cites, ``rule_decisions`` and
``writer_draft``; the head calls and source words live in ``c3_brief`` and
``atoms``, which never reach the frontend (Core copies ``c3_sow`` only). This
module copies just those facts, for just the atoms some SOW line cites, into
``c3_sow["atom_trace"]``.

Read-only over the envelope: it decides nothing and changes no line. The
nightly fix triage (scripts/sow_fix_triage.py) reads the same trace, so the
PM and the agent judge a fix from the same evidence.
"""
from __future__ import annotations

import re
from typing import Any

ATOM_ID = re.compile(r"atm_[0-9a-f]{8,}")
#: Lists of ``c3_sow`` whose items carry ``cites``.
CITED_LISTS = (
    "overview", "schedule", "arrival_protocol", "scope", "equipment_in_scope",
    "provider_materials", "provider_tools", "customer_materials",
    "customer_responsibilities", "assumptions", "out_of_scope", "locations",
)
_TEXT_MAX = 600
_AROUND_MAX = 240
_AROUND_N = 2
_ELSEWHERE = ("also_mentioned", "kept_unsure", "questions", "job_details", "label_next")


def cite_ids(raw: Any) -> list[str]:
    """Atom ids from a ``cites`` value however the writer sent it (a list, a
    stringified list, a string)."""
    out: list[str] = []
    for x in raw if isinstance(raw, list) else [raw]:
        for m in ATOM_ID.findall(str(x or "")):
            if m not in out:
                out.append(m)
    return out


def cited_ids(sow: dict[str, Any]) -> list[str]:
    """Every atom a line of the SOW, its writer draft, its drops or its rule
    decisions names, in first-seen order."""
    seen: dict[str, None] = {}

    def add(raw: Any) -> None:
        for a in cite_ids(raw):
            seen.setdefault(a, None)

    for k in CITED_LISTS:
        for it in sow.get(k) or []:
            if isinstance(it, dict):
                add(it.get("cites"))
    for xs in (sow.get("writer_draft") or {}).values():
        for it in xs or []:
            if isinstance(it, dict):
                add(it.get("cites"))
    for k in ("dropped", "dropped_unsupported"):
        for it in sow.get(k) or []:
            if isinstance(it, dict):
                add(it.get("cites"))
    for d in sow.get("rule_decisions") or []:
        add(d.get("atom_ids"))
    return list(seen)


def _where(atom: dict[str, Any]) -> str:
    loc = atom.get("locator") or {}
    if loc.get("message_label"):
        return str(loc["message_label"])
    bits = []
    for k, name in (("sheet", "sheet"), ("page", "p."), ("row", "row"), ("line_start", "line")):
        if loc.get(k) not in (None, ""):
            bits.append(f"{name} {loc[k]}")
    return " · ".join(bits)


def _round(p: Any) -> float | None:
    try:
        return round(float(p), 3)
    except (TypeError, ValueError):
        return None


def build_atom_trace(envelope: dict[str, Any]) -> dict[str, dict[str, Any]]:
    sow = envelope.get("c3_sow") or {}
    ids = cited_ids(sow)
    if not ids:
        return {}
    atoms = [a for a in envelope.get("atoms") or [] if isinstance(a, dict)]
    by_id = {str(a.get("id")): a for a in atoms}
    files = {str(d.get("artifact_id") or d.get("id")): str(d.get("filename") or d.get("name") or "")
             for d in envelope.get("documents") or [] if isinstance(d, dict)}
    order: dict[str, list[str]] = {}
    for a in atoms:
        order.setdefault(str(a.get("artifact_id")), []).append(str(a.get("id")))
    brief: dict[str, tuple[str, dict[str, Any]]] = {}
    for section, items in ((envelope.get("c3_brief") or {}).get("sections") or {}).items():
        for it in items or []:
            if isinstance(it, dict) and it.get("id"):
                brief.setdefault(str(it["id"]), (section, it))
    # Outside the sections the brief still names atoms: kept as an aside, kept
    # unsure, asked as a question, given to the writer as a job detail.
    elsewhere: dict[str, list[str]] = {}
    unsure: dict[str, list[Any]] = {}  # label_next: a head the brief was unsure of on it
    for place in _ELSEWHERE:
        for it in (envelope.get("c3_brief") or {}).get(place) or []:
            if not isinstance(it, dict):
                continue
            ev = [x.get("id") for x in it.get("evidence") or [] if isinstance(x, dict)]
            for a in cite_ids([it.get("id"), *(it.get("cites") or []), *(it.get("atom_ids") or []), it.get("atom_id"), *ev]):
                elsewhere.setdefault(a, [])
                if place not in elsewhere[a]:
                    elsewhere[a].append(place)
                if place == "also_mentioned" and a not in brief:
                    brief[a] = (None, it)
                if place == "label_next" and it.get("head"):
                    unsure.setdefault(a, [it["head"], it.get("base_says"), _round(it.get("p"))])

    out: dict[str, dict[str, Any]] = {}
    for aid in ids:
        a = by_id.get(aid)
        if not a:
            out[aid] = {"missing": True}
            continue
        s = a.get("structured") or {}
        lab = s.get("c3_label") or {}
        base = s.get("c3_base_says") or {}
        art = str(a.get("artifact_id"))
        sib = order.get(art, [])
        i = sib.index(aid) if aid in sib else -1
        around = []
        if i >= 0:
            for j in range(max(0, i - _AROUND_N), min(len(sib), i + _AROUND_N + 1)):
                if j != i:
                    around.append({"pos": j - i, "text": str(by_id[sib[j]].get("text") or "")[:_AROUND_MAX]})
        said = s.get("said_by") or {}
        row: dict[str, Any] = {
            "text": str(a.get("text") or "")[:_TEXT_MAX],
            "file": files.get(art, ""),
            "where": _where(a),
            "type": lab.get("type") or a.get("atom_type"),
            "type_p": _round(lab.get("p") if lab.get("p") is not None else a.get("confidence")),
        }
        if lab.get("was"):
            row["type_was"] = lab["was"]
        if lab.get("act"):
            row["act"] = lab["act"]
        if base.get("type"):
            row["base_type"] = [base["type"], _round(base.get("p"))]
        if said.get("name") or said.get("company"):
            row["said_by"] = {k: said.get(k) for k in ("name", "company", "side") if said.get(k)}
        ctx = (s.get("email_thread") or {}).get("context")
        if ctx:
            row["context"] = str(ctx)[:_AROUND_MAX]
        if around:
            row["around"] = around
        if aid in brief:
            section, it = brief[aid]
            row["section"] = section
            for k in ("section_by", "by"):
                if it.get(k):
                    row["section_by" if k == "section_by" else "keep_by"] = it[k]
            if it.get("keep_p") is not None:
                row["keep_p"] = _round(it["keep_p"])
            sup = it.get("supplier")
            if isinstance(sup, list) and sup:
                row["supplier"] = [sup[0], _round(sup[1]) if len(sup) > 1 else None]
        else:
            row["section"] = None
        # None of the brief's places: the heads left it out of the brief.
        if aid in unsure:
            row["unsure"] = unsure[aid]
        places = (["sections"] if row["section"] else []) + elsewhere.get(aid, [])
        row["brief"] = places or ["left_out"]
        out[aid] = row
    return out


def attach(envelope: dict[str, Any]) -> None:
    """Put the trace on ``envelope["c3_sow"]``. Never raises into the compile."""
    sow = envelope.get("c3_sow")
    if not isinstance(sow, dict) or sow.get("error"):
        return
    try:
        sow["atom_trace"] = build_atom_trace(envelope)
    except Exception as exc:  # pragma: no cover - a trace never costs the SOW
        sow["atom_trace_error"] = type(exc).__name__
