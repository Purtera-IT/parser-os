"""The two reads of a deal, shared by training (runE/views.py) and serving (the compile, /read).

quoting    the customer's and partner's lines that arrived before our first SOW / quote / Deal Kit or
           our first priced message (``provenance.quoting_inputs``), cut earlier by a run's as_of. Our
           documents are never inputs. Quote, SOW and universal heads are asked; PM heads are not.
planning   every line before the planning end (``planning_end``: our first kickoff or scheduling message
           after the deal is won), our signed SOW and quote included. PM and universal heads are asked;
           quote and SOW heads are not (they would read the answer off our own SOW).

Every line carries ``view:<quoting|planning>`` (``tag``) and a ``stage_at`` clamped to what the view
can hold (``clamp_stage``), on both sides.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

from .sender import epoch

QUOTING = "quoting"
PLANNING = "planning"
VIEWS = (QUOTING, PLANNING)
#: Training's view names (build.py --pre-sow / --plan) -> the shared ones.
_ALIASES = {"pre_quote": QUOTING, "presow": QUOTING, "pre_sow": QUOTING, "quote": QUOTING, QUOTING: QUOTING,
            "plan": PLANNING, PLANNING: PLANNING}

QUOTE_HEADS = ("read:sow_coverage", "read:sow_fill", "read:sow_section", "read:derivation", "read:intake_gap",
               "read:hours_stated", "read:labor_hours", "read:rate", "read:rate_for", "read:billing_type",
               "read:removes_cost", "read:crew_size", "read:tech_level", "read:tech_coverage", "read:equipment_qty",
               "read:job_scale", "read:co_hours_estimate", "read:co_crew_rule", "read:co_tech_base_miles",
               "read:remote_miles", "read:train_for", "read:co_deal_kit_route", "read:sow_available_at_stage",
               # Run E SOW heads: what a pre-quote line becomes in our SOW, and where a slot's content comes from.
               # In the planning view our SOW is an input, so they would read the answer off it: masked there.
               "read:sow_element", "read:sow_transform")
QUOTE_PREFIXES = ("jdg:commercial", "jdg:term", "jdg:hours", "jdg:billing_type", "jdg:tier", "jdg:hardware_scope",
                  "jdg:materials_scope", "jdg:expenses_scope", "jdg:travel_scope", "jdg:bom_owner", "deal:",
                  "jdg:sow_slot", "jdg:exclusion_set")
PM_HEADS = ("read:task_owner_role", "read:triggered_task", "read:trigger_event", "read:fires_task",
            "read:blocked_on", "read:urgency", "read:lead_time", "read:needs_artifact", "read:chase",
            "read:needed_by", "read:commitment", "read:delivery_field", "read:start_date", "read:cadence",
            "jdg:task_tier", "jdg:task_tier:reason")
#: PM heads train at this weight in the quoting view (they are not served there).
PM_IN_QUOTE_VIEW = 0.25


def name(view: Any) -> str:
    """The shared name of a view ('' for none / unknown)."""
    return _ALIASES.get(str(view or "").strip().lower(), "")


def of_run(as_of: Any) -> str:
    """A compile's view: quoting when the run has an as_of, planning otherwise."""
    return QUOTING if str(as_of or "").strip() else PLANNING


def is_quote_head(key: str) -> bool:
    return key in QUOTE_HEADS or key.startswith(QUOTE_PREFIXES)


def is_pm_head(key: str) -> bool:
    return key in PM_HEADS


def head_scale(view: Any, keys: Iterable[str]) -> dict[str, float]:
    """Training's loss scale per head key in a view (a key not listed counts 1)."""
    keys = list(keys)
    if name(view) == QUOTING:
        return {k: PM_IN_QUOTE_VIEW for k in keys if is_pm_head(k)}
    return {k: 0.0 for k in keys if is_quote_head(k)}


def heads(view: Any, keys: Iterable[str]) -> list[str]:
    """The heads a read in this view asks: quoting drops the PM heads, planning the quote and SOW heads;
    no view asks everything."""
    v = name(view)
    if v == QUOTING:
        return [k for k in keys if not is_pm_head(k)]
    if v == PLANNING:
        return [k for k in keys if not is_quote_head(k)]
    return list(keys)


def runs_quote(view: Any) -> bool:
    """The SOW writer and the quote heads (/v1/quote) run on a quoting read (or one with no view)."""
    return name(view) != PLANNING


def tag(view: Any) -> str:
    """The per-line input tag (``view:<name>``), '' without a view."""
    v = name(view)
    return f"view:{v}" if v else ""


def clamp_stage(stage: Any, view: Any, *, crm: bool, before_cut: bool) -> str:
    """``stage_at`` inside a view. With the deal's CRM stage history (``crm``) the stage is the CRM stage at
    the line's arrival, and the quoting view holds no line past quoting (contracting -> quoted). Without one
    (the milestone rule) the quoting view is before our first priced paper or message, so every line is
    ``scoping``; in the planning view a line before the quoting cut (``before_cut``) is ``scoping`` and a
    later one keeps its milestone stage (else ``quoted``). No view: unchanged."""
    v, was = name(view), str(stage or "")
    if not v:
        return was
    if crm:
        return ("quoted" if was == "contracting" else was or "scoping") if v == QUOTING else (was or "scoping")
    return "scoping" if v == QUOTING or before_cut else (was or "quoted")


def stamp(rows: Iterable[dict[str, Any]], view: Any, *, crm: bool, cut: Any = None) -> int:
    """Clamp every row's ``stage_at`` to the view (``clamp_stage``; a row is before the cut when its
    ``entered_at`` is) and write its ``view``, in place. The one call training's views and the serve read
    make. Returns the rows whose stage changed."""
    v = name(view)
    c = epoch(cut) if cut not in (None, float("inf")) else None
    n = 0
    for r in rows:
        w = epoch(r.get("entered_at"))
        was = str(r.get("stage_at") or "")
        want = clamp_stage(was, v, crm=crm, before_cut=c is None or (w or 0.0) < c)
        if want != was:
            r["stage_at"] = want
            n += 1
        if v:
            r["view"] = v
    return n


#: Our message that starts delivery: a kickoff, techs on site or dispatched, the install scheduled for a day.
KICKOFF = re.compile(
    r"\bkick[- ]?off\s+(?:call|meeting)\b|"
    r"\b(?:techs?|technicians?|engineers?)\s+(?:are|is|will be|were)\s+(?:on[- ]?site|dispatched)\b|"
    r"\bdispatched\s+(?:a\s+|the\s+)?(?:techs?|technicians?)\b|"
    r"\b(?:install(?:ation)?|deployment|cutover|site visit|work)\s+(?:is|has been|was)\s+(?:scheduled|confirmed)\b|"
    r"\bscheduled\s+(?:for|on)\s+(?:mon|tue|wed|thu|fri|sat|sun|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|"
    r"\d{1,2}[/-]\d{1,2}|next week|tomorrow)", re.I)
#: A line that only talks about scheduling some day ("once approved, we'll schedule", "if we are unable to
#: dispatch", a question) starts nothing.
NOT_YET = re.compile(r"\?|\b(?:once|if|unless|until|whether|need|needs|unable|could you|can you|would you|please provide)\b",
                     re.I)
_OUR_MESSAGE_KINDS = ("email", "note", "crm_note")


def planning_end(rows: Iterable[dict[str, Any]], *, cut: Any = None, timeline_marks: Any = None,
                 won_at: Any = None) -> float:
    """When the planning view ends: the arrival of our first email or note that kicks off or schedules the
    work (``KICKOFF``, not ``NOT_YET``), at or after the deal is won: the CRM's first ``contracting`` mark,
    else ``won_at`` (our signed SOW / PO's arrival). A deal not yet won has no end. Observable on both sides
    (the line's text and side, its document's kind and arrival); never a label. ``inf`` when there is none
    (the view holds every line). ``cut`` (the quoting cut) is a floor: delivery never starts before it.

    A row: ``text``, ``entered_at``, ``doc_kind`` and ``speaker_side`` (``ours`` for our own line)."""
    start = None
    for when, stage in timeline_marks or ():
        if stage == "contracting":
            start = float(when)
            break
    if start is None:
        start = epoch(won_at)
    if start is None:
        return float("inf")
    c = epoch(cut) if cut not in (None, float("inf")) else None
    if c is not None:
        start = max(start, c)
    best = float("inf")
    for r in rows:
        if r.get("speaker_side") != "ours" or str(r.get("doc_kind") or "") not in _OUR_MESSAGE_KINDS:
            continue
        w = epoch(r.get("entered_at"))
        if w is None or w <= 0 or w < start or w >= best:
            continue
        text = str(r.get("text") or "")
        if KICKOFF.search(text) and not NOT_YET.search(text):
            best = w
    return best


def planning_inputs(rows: list[dict[str, Any]], end: float, *, exclude: Iterable[str] = (),
                    other_job: Iterable[str] = (), key: str = "label_key") -> list[int]:
    """Indices of the planning view's input lines: dated, arrived before ``end``, not another job's
    document, not one of ``exclude`` (our pricing workbooks' keys)."""
    ex, other = set(exclude), set(other_job)
    out = []
    for i, r in enumerate(rows):
        w = epoch(r.get("entered_at")) or 0.0
        if 0 < w < end and r.get("doc_id") not in other and r.get(key) not in ex:
            out.append(i)
    return out


__all__ = ["QUOTING", "PLANNING", "VIEWS", "QUOTE_HEADS", "QUOTE_PREFIXES", "PM_HEADS", "PM_IN_QUOTE_VIEW",
           "name", "of_run", "is_quote_head", "is_pm_head", "head_scale", "heads", "runs_quote", "tag",
           "clamp_stage", "stamp", "KICKOFF", "planning_end", "planning_inputs"]
