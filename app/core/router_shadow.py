"""Run the head beside the base, record both, and decide who to ask.

Three jobs, and they unblock each other in this order.

**Collect.** The router's training set in the blob registry is one row of one
class, unchanged for ten days -- so the nightly retrain has been faithfully
rebuilding nothing. Every compile already routes a deal; logging that route
with the input it was made from turns each compile into a training example.
This is the whole of "collect really good data from each deal", and it costs a
row.

**Shadow.** The head must run beside the base without deciding, logging both
answers. The point is the calendar: when PM labels finally arrive, the
retrospective evaluation is already collected and can be run that afternoon,
instead of starting a clock that then has to run for months. Waiting in
parallel rather than in series.

**Queue.** PM attention is the scarcest input in this system, and a correction
is only worth asking for where it discriminates. Deals where the base and the
head AGREE teach almost nothing -- both routers already have that case. Deals
where they DISAGREE are where the decision is hard, and one tap resolves an
argument. So disagreement is the queue.

That also fixes a bias the correction set would otherwise have. A hold-out
built from errors a PM happened to notice skews toward the obvious; one built
from disagreements skews toward the *hard*, which is a far better skew for a
gate.

Nothing here decides anything. It records, and it raises a hand.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

#: The relation these rows are logged under -- matches
#: ``pm_feedback.HEAD_REGISTRY["router"]`` so a PM correction and a shadow row
#: describe the same decision and can be joined later.
ROUTER_RELATION = "service_routing"


@dataclass
class ShadowRecord:
    """What the two routers said, and whether a PM should be asked."""

    deal_id: str = ""
    base_label: str | None = None
    base_confidence: float = 0.0
    head_label: str | None = None
    head_confidence: float = 0.0
    ask_pm: bool = False
    ask_reason: str = ""
    logged: bool = False
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def agree(self) -> bool | None:
        """None when only one router answered -- that is not agreement."""
        if self.base_label is None or self.head_label is None:
            return None
        return self.base_label == self.head_label

    def as_dict(self) -> dict[str, Any]:
        return {
            "base": self.base_label,
            "base_confidence": self.base_confidence,
            "head": self.head_label,
            "head_confidence": self.head_confidence,
            "agree": self.agree,
            "ask_pm": self.ask_pm,
            "ask_reason": self.ask_reason,
            "logged": self.logged,
        }


def decide_ask(
    base_label: str | None,
    head_label: str | None,
    *,
    base_confidence: float = 0.0,
) -> tuple[bool, str]:
    """Should this deal be put in front of a PM for one tap?

    Ranked by how much a label would buy:

    * **the two routers disagree** -- the most valuable tap available. One
      correction settles which is right and lands in the hold-out as a hard case.
    * **neither answered** -- nothing routed the deal at all, so any label is
      the first evidence there is.
    * **only the base answered, and weakly** -- the head had no opinion and the
      base was unsure, which is the shape of a class nobody has taught yet.

    Deliberately NOT asked: both agreeing, or the base answering confidently
    alone. Those cost a PM the same tap and teach nearly nothing, and a queue
    that cries wolf is a queue nobody clears.
    """
    if base_label and head_label and base_label != head_label:
        return True, f"routers disagree: base={base_label} head={head_label}"
    if not base_label and not head_label:
        return True, "no router placed this deal"
    if base_label and not head_label and base_confidence < 0.5:
        return True, f"only the base answered, and weakly ({base_confidence:.2f})"
    return False, ""


def record(
    *,
    deal_id: str,
    project_id: str = "",
    base_label: str | None,
    base_confidence: float = 0.0,
    head_label: str | None = None,
    head_confidence: float = 0.0,
    provenance: dict[str, Any] | None = None,
    scope_summary: str = "",
) -> ShadowRecord:
    """Log the pair and return the queue signal. Never raises.

    The row is logged under the BASE's label, because the base is what shipped
    and its answer is the closest thing to a label available before a PM says
    otherwise. The head's answer rides in provenance rather than as the label:
    a head's own output must never become its training target, which is the
    circularity this whole effort exists to break.
    """
    prov = dict(provenance or {})
    if scope_summary and "scope_summary_sha256" not in prov:
        prov["scope_summary_sha256"] = hashlib.sha256(
            scope_summary.encode("utf-8")
        ).hexdigest()[:16]
    prov.update(
        {
            "shadow": True,
            "base_label": base_label,
            "base_confidence": round(float(base_confidence), 4),
            "head_label": head_label,
            "head_confidence": round(float(head_confidence), 4),
        }
    )

    ask, reason = decide_ask(base_label, head_label, base_confidence=base_confidence)
    rec = ShadowRecord(
        deal_id=deal_id,
        base_label=base_label,
        base_confidence=float(base_confidence),
        head_label=head_label,
        head_confidence=float(head_confidence),
        ask_pm=ask,
        ask_reason=reason,
        provenance=prov,
    )
    if ask:
        prov["ask_reason"] = reason

    # Only a routed deal is a training example. A row whose label is None
    # teaches nothing and would inflate n_train with silence -- which is
    # arguably how the registry ended up reporting one row and calling it
    # trained.
    if not base_label:
        return rec

    try:
        from app.core.training_log import TrainingRow, log_rows

        written = log_rows([
            TrainingRow(
                relation=ROUTER_RELATION,
                label=str(base_label),
                raw_text=scope_summary[:4000],
                masked_text="",
                label_kind="judgment",
                confidence=float(base_confidence),
                scope="deal",
                scope_key=deal_id,
                deal_id=deal_id,
                project_id=project_id,
                provenance=prov,
            )
        ])
        rec.logged = bool(written)
    except Exception:
        rec.logged = False
    return rec
