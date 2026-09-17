"""The hours estimator as a correctable head (PUR-13).

Before this, the hours proposal was one number computed outside the correction
system, so a PM changing it taught nothing -- and a correction to a TOTAL is
unattributable: nothing can tell whether the PM disagreed about how long a
visit takes or about how many visits there are, and those two generalise
completely differently.

Here the estimate is three named fields, each its own relation in
:data:`app.core.pm_feedback.HEAD_REGISTRY`:

    units            what the work order's count becomes once priced
    visits           how many trips the work takes
    hours_per_visit  how long one trip takes

    total_hours = hours_per_visit x visits        (derived, never corrected)

A correction lands on exactly one field. It is stored as a RATE relative to the
deal's own inputs, so it transfers across job sizes instead of copying one
deal's number onto another:

    units            -> units per stated count      (accepted units / count)
    visits           -> visits per site             (accepted visits / sites)
    hours_per_visit  -> hours per unit of work      (accepted h/visit x visits / units)

What a lesson is KEYED on is the scope decision in ``docs/LESSON_KEYS.md``:
``wording`` (today: the deal's sentence) or ``work_shape`` (the structured work,
:mod:`app.core.work_shape`), selected per call or by ``SOWSMITH_LESSON_KEY``.

Baseline rule (used when no lesson fires): one unit per stated count, one visit
per site, one hour per unit. Deliberately naive and labelled as such in every
field's ``source`` -- this module is the correction seam, not a pricing model.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from app.core import work_shape as ws
from app.core.decide import DecisionScope
from app.core.feedback_store import SCOPE_GLOBAL, Correction

ESTIMATOR_VERSION = "estimator-v1"

#: Correctable fields, in the order they resolve (hours_per_visit needs the
#: other two).
FIELDS: tuple[str, ...] = ("units", "visits", "hours_per_visit")
RELATIONS: dict[str, str] = {f: f"estimate_{f}" for f in FIELDS}
LABELS: dict[str, str] = {
    "units": "Units",
    "visits": "Number of visits",
    "hours_per_visit": "Hours per visit",
}
BASELINE_RATES: dict[str, float] = {"units": 1.0, "visits": 1.0, "hours_per_visit": 1.0}
RATE_BASIS: dict[str, str] = {
    "units": "units_per_count",
    "visits": "visits_per_site",
    "hours_per_visit": "hours_per_unit",
}
LESSON_THRESHOLD = 0.74


@dataclass
class FieldValue:
    value: float
    rate: float
    basis: str
    source: str  # "baseline" | "lesson"
    correction_id: str | None = None
    confidence: float = 0.0


@dataclass
class LineProposal:
    index: int
    work: str
    wording: str
    shape: dict[str, str]
    inputs: dict[str, Any]
    fields: dict[str, FieldValue]

    @property
    def total_hours(self) -> float:
        return round(self.fields["hours_per_visit"].value * self.fields["visits"].value, 4)


@dataclass
class Proposal:
    proposal_id: str
    version: str
    deal_id: str
    key_mode: str
    lines: list[LineProposal] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for ln, line in zip(d["lines"], self.lines):
            ln["total_hours"] = line.total_hours
        return d

    def field_value(self, line_index: int, field_name: str) -> float:
        return self.lines[line_index].fields[field_name].value


def _sha(s: str, n: int = 16) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:n]


def _num(v: Any, default: float) -> float:
    try:
        f = float(v)
        return f if f > 0 else default
    except (TypeError, ValueError):
        return default


def _lines(deal: dict) -> list[dict]:
    wo = deal.get("work_order") or {}
    return [ln for ln in (wo.get("work_lines") or []) if isinstance(ln, dict)]


def line_shape(deal: dict, line: dict) -> ws.WorkShape:
    return ws.from_work_line(
        line,
        deal.get("work_order") or {},
        billing_type=str(deal.get("billing_type") or ""),
        delivery_model=str(deal.get("delivery_model") or ""),
    )


def line_wording(deal: dict, line: dict) -> str:
    """What a wording key sees: the sentence the documents used."""
    return str(line.get("wording") or deal.get("wording") or line.get("work") or "").strip()


def _estimator_corrections(store: Any) -> list[Correction]:
    rels = set(RELATIONS.values())
    try:
        return [c for c in store.all_corrections(active_only=True) if c.relation in rels]
    except Exception:
        return []


def estimator_version(store: Any) -> str:
    """Version = code version + the lesson set it ran with, so an override can
    always be traced to exactly what produced the number it overrode."""
    sig = ",".join(sorted(f"{c.id}={c.verdict}" for c in _estimator_corrections(store))) if store is not None else ""
    return f"{ESTIMATOR_VERSION}+lessons:{_sha(sig, 8)}"


def encode_rate(rate: float) -> str:
    return f"rate={float(rate):.6g}"


def parse_rate(verdict: str) -> float | None:
    key, sep, val = str(verdict or "").partition("=")
    if key.strip() != "rate" or not sep:
        return None
    try:
        r = float(val)
    except ValueError:
        return None
    return r if r > 0 else None


def query_for(deal: dict, line: dict, field_name: str, key_mode: str) -> tuple[str, dict | None]:
    """(text, relations) the store is asked with for this field."""
    if key_mode == ws.MODE_WORK_SHAPE:
        shape = line_shape(deal, line)
        return ws.key_text(shape, field_name), {ws.REL_SHAPE: shape.as_dict()}
    return line_wording(deal, line), None


def _lookup(store: Any, deal: dict, line: dict, field_name: str, key_mode: str):
    if store is None:
        return None
    relation = RELATIONS[field_name]
    verdicts = sorted({c.verdict for c in _estimator_corrections(store) if c.relation == relation})
    if not verdicts:
        return None
    text, relations = query_for(deal, line, field_name, key_mode)
    if not text:
        return None
    return store.resolve(
        relation=relation,
        text=text,
        candidates=verdicts,
        context="",
        scope=DecisionScope(deal_id=str(deal.get("deal_id") or "")),
        instruction="",
        relations=relations,
        neural_head=False,
    )


def propose(deal: dict, store: Any = None, *, key_mode: str | None = None) -> Proposal:
    """Estimate every work line of ``deal``, field by field."""
    mode = key_mode or ws.lesson_key_mode()
    deal_id = str(deal.get("deal_id") or "")
    version = estimator_version(store)
    wo = deal.get("work_order") or {}
    out_lines: list[LineProposal] = []
    for i, line in enumerate(_lines(deal)):
        count = _num(line.get("count"), 1.0)
        sites = _num(wo.get("site_count"), 1.0)
        fields: dict[str, FieldValue] = {}
        for f in FIELDS:
            rate, source, cid, conf = BASELINE_RATES[f], "baseline", None, 0.0
            d = _lookup(store, deal, line, f, mode)
            if d is not None and d.verdict:
                r = parse_rate(d.verdict)
                if r is not None:
                    rate, source, cid, conf = r, "lesson", d.correction_id, float(d.confidence or 0.0)
            if f == "units":
                value = rate * count
            elif f == "visits":
                value = rate * sites
            else:
                units = fields["units"].value
                visits = fields["visits"].value or 1.0
                value = rate * units / visits
            fields[f] = FieldValue(
                value=round(value, 4), rate=rate, basis=RATE_BASIS[f],
                source=source, correction_id=cid, confidence=round(conf, 3),
            )
        out_lines.append(LineProposal(
            index=i,
            work=str(line.get("work") or ""),
            wording=line_wording(deal, line),
            shape=line_shape(deal, line).as_dict(),
            inputs={"count": line.get("count"), "site_count": wo.get("site_count")},
            fields=fields,
        ))
    inputs_sig = json.dumps([(ln.inputs, ln.shape, ln.wording) for ln in out_lines], sort_keys=True, default=str)
    pid = "prop_" + _sha(f"{deal_id}|{version}|{mode}|{inputs_sig}")
    return Proposal(proposal_id=pid, version=version, deal_id=deal_id, key_mode=mode, lines=out_lines)


def rate_from_accepted(proposal: Proposal, line_index: int, field_name: str, accepted: float) -> float:
    """Turn a PM's accepted value into the transferable rate for that field."""
    if field_name not in FIELDS:
        raise ValueError(f"unknown estimator field {field_name!r}; one of {FIELDS}")
    accepted = float(accepted)
    if accepted <= 0:
        raise ValueError("accepted value must be positive")
    line = proposal.lines[line_index]
    if field_name == "units":
        return accepted / _num(line.inputs.get("count"), 1.0)
    if field_name == "visits":
        return accepted / _num(line.inputs.get("site_count"), 1.0)
    units = line.fields["units"].value or 1.0
    visits = line.fields["visits"].value or 1.0
    return accepted * visits / units


def lesson_from_override(
    deal: dict,
    proposal: Proposal,
    *,
    line_index: int,
    field_name: str,
    accepted: float,
    reason_code: str,
    reason_text: str = "",
    actor: str = "pm",
    key_mode: str | None = None,
) -> Correction:
    """The lesson a single-field override teaches. Pure; nothing is stored."""
    mode = key_mode or proposal.key_mode
    line = _lines(deal)[line_index]
    rate = rate_from_accepted(proposal, line_index, field_name, accepted)
    text, _ = query_for(deal, line, field_name, mode)
    relations: dict[str, Any] = {
        "field": field_name,
        "reason_code": reason_code,
        "source_deal_id": proposal.deal_id,
        "source_proposal_id": proposal.proposal_id,
        "source_proposal_version": proposal.version,
        "key_mode": mode,
    }
    if mode == ws.MODE_WORK_SHAPE:
        relations.update(ws.lesson_relations(line_shape(deal, line), field_name))
        relations["wording"] = line_wording(deal, line)
    verdict = encode_rate(rate)
    now = time.time()
    return Correction(
        id=f"lesson_{field_name}_{_sha(f'{mode}|{text}|{verdict}', 12)}",
        relation=RELATIONS[field_name],
        verdict=verdict,
        scope=SCOPE_GLOBAL,
        exemplars=[text],
        threshold=LESSON_THRESHOLD,
        relations=relations,
        instruction=(
            f"PM {LABELS[field_name]}: {proposal.field_value(line_index, field_name):g} -> {float(accepted):g}"
            f" ({reason_code}{': ' + reason_text if reason_text else ''})"
        ),
        complaint_id=proposal.proposal_id,
        created_by=actor,
        created_at=now,
        updated_at=now,
    )


def preview_transfer(
    store: Any,
    lesson: Correction,
    corpus: list[dict],
    *,
    key_mode: str | None = None,
) -> list[dict[str, Any]]:
    """Which OTHER deals this lesson would change, before it is committed.

    Runs on an evaluation twin; the live store is never written."""
    twin = store.evaluation_twin(extra=[lesson])
    field_name = lesson.relations.get("field") or next(
        (f for f, r in RELATIONS.items() if r == lesson.relation), ""
    )
    source = lesson.relations.get("source_deal_id")
    changes: list[dict[str, Any]] = []
    for deal in corpus:
        if str(deal.get("deal_id")) == str(source):
            continue
        before = propose(deal, store, key_mode=key_mode)
        after = propose(deal, twin, key_mode=key_mode)
        for lb, la in zip(before.lines, after.lines):
            vb = lb.fields[field_name].value
            va = la.fields[field_name].value
            if abs(vb - va) > 1e-9:
                changes.append({
                    "deal_id": before.deal_id,
                    "line": lb.index,
                    "field": field_name,
                    "before": vb,
                    "after": va,
                })
    return changes


__all__ = [
    "ESTIMATOR_VERSION",
    "FIELDS",
    "RELATIONS",
    "FieldValue",
    "LineProposal",
    "Proposal",
    "estimator_version",
    "lesson_from_override",
    "preview_transfer",
    "propose",
    "rate_from_accepted",
]
