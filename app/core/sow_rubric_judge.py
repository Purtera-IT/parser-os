"""Where does a deal line go in PurTera's SOW, and who acts? (``jdg:sow_section``)

The SOW writer (``app/core/c3_sow.py``) places facts with hand-written rules.
The owner's rule is that heads decide every line and rules retire, but the
labels heads learn from were inconsistent: for the same atom PurTera's real SOW
and an A-graded draft pick the same section only 45% of the time. The rubric
(``docs/C3_RUBRIC_SOW_SECTION.md``, text below) fixed that: two independent
labelers applying it agree on 93-95% of sections and 95% of owners. No trained
encoder carries it at ~6k labels (frozen trunk 64% vs 61.5% majority, a
ModernBERT fine-tune 41%), so this head is the rubric itself, applied by a
model through the decide() chokepoint, like the document-job judge:

    STORE (a PM taught this line's section)  ->  LLM with the rubric  ->  undecided

The model is named (``SOWSMITH_SOW_JUDGE_MODEL``, default the local
qwen3:32b over the explicit ``ollama:`` route) because it is a hard
discrimination. Teacher-cache rows (the model's own earlier answers) never
decide: ``exclude_created_by=("teacher",)``.

Owner follows from the section for most sections (the rubric labels show
scope_of_work -> technician, assumptions/timeline/overview/_absent -> none);
only where the labels split (out_of_scope, customer duties, deliverables,
provider supplies) is the model asked a second question.

Undecided (no model, timeout, off-menu answer) is ``None``: callers keep their
own placement. Never raises. Live-path code: imports ``app`` only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from app.core.decide import DecisionScope, decide

RELATION = "sow_section"
OWNER_RELATION = "sow_owner"

#: Rubric order: the first matching test wins.
SECTIONS = [
    "_absent",
    "out_of_scope",
    "timeline",
    "deliverables",
    "customer_responsibilities",
    "provider_responsibilities",
    "scope_of_work",
    "assumptions",
    "overview",
]
OWNERS = ["technician", "customer", "partner", "other", "none"]

#: docs/C3_RUBRIC_SOW_SECTION.md, verbatim (tests/test_sow_rubric_judge.py keeps them equal).
RUBRIC = """# SOW section rubric (one answer per line, judged by what the line DOES in the job)

You label lines from a field-services deal (PurTera = "Provider", sends technicians onsite for a reseller/customer).
For each line pick the ONE section where its main fact belongs in PurTera's Statement of Work, and who acts.

section (pick by the test, in this order — first test that matches wins):
1. _absent — not SOW material: chatter, greeting, signature, meeting header, a question with no answer, pricing/commercial talk, pure metadata (deal ids, emails, phone numbers), another job's line.
2. out_of_scope — states something Provider will NOT do, or that is excluded / handled by someone else in place of Provider ("we will preconfigure everything" from the customer => Provider does not configure => out_of_scope is NOT right; that is customer_responsibilities. Use out_of_scope only when the line itself rules work out: "no cabling", "not in scope", "we are not approved to dispose").
3. timeline — a date, duration, work window, hours of work (business/after hours/weekend), pace (sites per week), deadline, arrival time.
4. deliverables — something handed over at the end: photos, checklist, serial numbers, as-builts, sign-off, report.
5. customer_responsibilities — something the customer, reseller or a third party DOES or SUPPLIES: ships equipment, provides mounts/cables, configures, gives access, escorts, has an onsite contact, unracks.
6. provider_responsibilities — materials, tools, consumables, lifts or vehicles PROVIDER supplies (not labor).
7. scope_of_work — work Provider's technicians perform, or a service requested from Provider (install, mount, remove, test, cable, haul away, support), including how it must be done ("no zip ties, use Velcro").
8. assumptions — a stated condition about the site or job the work relies on: existing infrastructure reused, brownfield, quantities/counts of existing things, access conditions, equipment arrives configured, power available, room details.
9. overview — a one-line summary of the whole project goal (rare; only a line that states the overall purpose).

owner (who acts or supplies): technician (Provider) | customer (end customer) | partner (the reseller, e.g. CDW/SHI, acting as itself) | other (third party vendor) | none (no actor: condition, date, chatter).

Notes: judge the line itself, not the trade or customer. A request for added work is scope_of_work. A count of devices to install is scope_of_work; a count of EXISTING things is assumptions. A customer-written list does not make Provider tasks customer duties.
"""

_SECTION_ASK = (
    "Apply this rubric to TEXT and answer its SECTION only (the first test, in order, that matches wins).\n"
    + RUBRIC
)
_OWNER_ASK = (
    "Apply this rubric to TEXT, whose section is {section}, and answer its OWNER only: who acts or supplies.\n"
    + RUBRIC
)

#: Sections whose owner the rubric labels fix (6,221 lines, 2026-10-19: >=98% one owner).
OWNER_OF = {
    "_absent": "none",
    "timeline": "none",
    "assumptions": "none",
    "overview": "none",
    "scope_of_work": "technician",
}

_DEFAULT_MODEL = "ollama:qwen3:32b"


def judge_model() -> str | None:
    """``SOWSMITH_SOW_JUDGE_MODEL``; '' or 'default' keeps decide()'s own route."""
    raw = os.environ.get("SOWSMITH_SOW_JUDGE_MODEL", _DEFAULT_MODEL).strip()
    return None if raw.lower() in ("", "default") else raw


def judge_timeout() -> int:
    try:
        return int(os.environ.get("SOWSMITH_SOW_JUDGE_TIMEOUT", "60"))
    except ValueError:
        return 60


@dataclass(frozen=True)
class Verdict:
    section: str | None
    section_conf: float
    owner: str | None
    owner_conf: float
    source: str  # store | llm | fallback | teacher (decide()'s tier for the section)


def context_for(job: str = "", under: str = "", neighbours: list[str] | None = None) -> str:
    """What the labelers saw beside a line: the job family and its heading.
    ``neighbours`` (the lines around it in its document) is optional."""
    parts = []
    if job:
        parts.append(f"Job family: {job}")
    if under:
        parts.append(f"Heading the line sat under: {under}")
    for n in neighbours or []:
        if n and n.strip():
            parts.append(f"Nearby line: {n.strip()[:200]}")
    return "\n".join(parts)


def judge(
    text: str,
    *,
    job: str = "",
    under: str = "",
    neighbours: list[str] | None = None,
    deal_id: str = "",
    model: str | None = "",
    timeout: int | None = None,
) -> Verdict:
    """One line's section and owner by the rubric. ``model=""`` takes
    ``judge_model()``; ``None`` keeps decide()'s default route."""
    if model == "":
        model = judge_model()
    timeout = timeout if timeout is not None else judge_timeout()
    text = str(text or "").strip()
    if not text:
        return Verdict(None, 0.0, None, 0.0, "fallback")
    ctx = context_for(job, under, neighbours)
    scope = DecisionScope(deal_id=deal_id)
    d = decide(RELATION, text[:600], SECTIONS, instruction=_SECTION_ASK, context=ctx[:1200], scope=scope,
               model=model, timeout=timeout, exclude_created_by=("teacher",))
    if d.verdict is None or d.source == "teacher":
        return Verdict(None, 0.0, None, 0.0, d.source)
    if d.verdict in OWNER_OF:
        return Verdict(d.verdict, d.confidence, OWNER_OF[d.verdict], d.confidence, d.source)
    o = decide(OWNER_RELATION, text[:600], OWNERS, instruction=_OWNER_ASK.format(section=d.verdict),
               context=ctx[:1200], scope=scope, model=model, timeout=timeout, exclude_created_by=("teacher",))
    owner = o.verdict if o.verdict is not None and o.source != "teacher" else None
    return Verdict(d.verdict, d.confidence, owner, o.confidence if owner else 0.0, d.source)


__all__ = ["RELATION", "OWNER_RELATION", "SECTIONS", "OWNERS", "RUBRIC", "OWNER_OF", "Verdict", "judge",
           "judge_model", "context_for"]
