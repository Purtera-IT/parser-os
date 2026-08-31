"""Where a document sits in the deal's own life, and who may therefore read it.

THE IDEA
--------
A deal moves through pipeline stages on dates HubSpot records. WHEN a document
arrived relative to those dates is a fact about the deal, not a guess about the
file. That is a far better foundation than reading a filename.

WHY STAGE ALONE IS NOT ENOUGH
-----------------------------
Because stages move administratively, not when the work happens. Deal 010215
went Open->Submitted for Quoting in eight minutes and then sat in Decision
Pending for fifteen days while everything real occurred. Measured over 30 deals
with >=12 dated artifacts: the median deal puts 65% of its material in a SINGLE
stage bucket, 11 of 30 put over 70% there, and two put 100%.

So on 010215 a pure-stage rule files ``Marion County School District
Locations.docx`` -- a site list the customer sent us -- under "negotiation over
our own output", beside our own SOW. Confidently wrong is worse than unlabelled.

THE SECOND AXIS
---------------
Direction. HubSpot records whether a message was sent or received (100% coverage
corpus-wide: 1,345 outbound / 878 inbound), and notes carry author affiliation.
Stage says WHEN; direction says WHOSE. Split that same 35-item bucket by
direction and it separates 13 inbound / 13 outbound cleanly.

COMPOSITION, IN ORDER
---------------------
1. If the document has already been CLASSIFIED as our own output (label) or as
   delivery material (atlas), that wins. A Deal Kit is our output whenever it
   arrived; type is decisive for the artifacts we produce, and letting a stage
   window override it would readmit our own answer as evidence -- the exact
   failure the label class exists to prevent.
2. Otherwise stage x direction decides.
3. If the stage is unknown -- undated artifact, or no timeline -- keep whatever
   the classifier said and mark it unplaced. An unknown position is not a
   position, and inventing one is how a corpus quietly poisons itself.
"""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

# Stage labels are typed by people into HubSpot and are NOT stable strings. The
# live pipeline reads "Submitted for Quoting " (trailing space) and "Closed Won:
# 100%". Matching those literally is how this module shipped its first bug:
# `_stage_index("Submitted for Quoting")` found nothing, returned -1, and every
# stage then compared as "after quoting" -- so our own outbound mail during
# discovery was labelled produced output. Compare on a normalised form instead.
def _norm(label: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(label or "").lower())


_QUOTED = "Submitted for Quoting"
_PRE_DEAL = "Before deal created"

# Ordering is taken from the pipeline's own displayOrder, carried on each
# transition, so a renamed or reordered stage does not silently change meaning.
# Prefix match: the live label is "Closed Won: 100%".
_CLOSED_WON = "closedwon"
_TERMINAL_NEITHER = ("closedlost", "archive")


def stage_at_arrival(
    authored_at: str | None,
    timeline: Mapping[str, Any] | None,
) -> str | None:
    """Which stage the deal was in when this arrived.

    Returns ``None`` when it cannot be determined. That ``None`` is load-bearing:
    the caller must show the document as unplaced rather than file it under the
    first or last stage, either of which asserts a date we do not have.
    """
    if not authored_at or not timeline:
        return None
    transitions: Sequence[Mapping[str, Any]] = timeline.get("transitions") or ()
    created = timeline.get("created_at")
    if created and authored_at < str(created):
        return _PRE_DEAL
    if not transitions:
        return None
    if authored_at < str(transitions[0].get("ts") or ""):
        return _PRE_DEAL
    current: str | None = None
    for entry in transitions:
        ts = str(entry.get("ts") or "")
        if ts and authored_at >= ts:
            current = str(entry.get("label") or "").strip()
        else:
            break
    return current or None


def _stage_index(stage: str, timeline: Mapping[str, Any] | None) -> int:
    """Position of a stage in the pipeline, or -1 for pre-deal / unknown."""
    if stage == _PRE_DEAL:
        return -1
    target = _norm(stage)
    for entry in (timeline or {}).get("transitions") or ():
        if _norm(entry.get("label")) == target:
            return int(entry.get("order") or 0)
    return -1


def admissibility(
    *,
    stage: str | None,
    direction: str | None,
    classified_as: str | None,
    timeline: Mapping[str, Any] | None = None,
) -> tuple[str | None, str]:
    """Return ``(admissible_for, why)``.

    ``why`` is not decoration. Every one of these calls is a claim about whether
    a model may read a document, and a claim a person cannot audit is one they
    have to take on faith.
    """
    # 1. Our own output stays our own output, whenever it arrived.
    if classified_as in ("label", "atlas"):
        return classified_as, f"classified as {classified_as}; type is decisive for produced material"

    # 3. Unknown position -> keep what we had, and say it is unplaced.
    if not stage:
        return classified_as, "no stage: undated artifact or no deal timeline; left where the classifier put it"

    # 2. Stage x direction.
    norm = _norm(stage)
    if norm.startswith(_TERMINAL_NEITHER):
        return "neither", f"arrived in {stage}; after the decision, carries no scope"
    if norm.startswith(_CLOSED_WON):
        return "atlas", "arrived after close; delivery material, belongs to Atlas"
    if stage == _PRE_DEAL:
        return "evidence", "predates the deal; discovery context"

    # If the quoting stage is not in this deal's timeline at all, we cannot say
    # whether our own outbound material postdates a quote -- so do not claim it
    # does. -1 here previously made EVERY stage compare as quoted.
    quoted_idx = _stage_index(_QUOTED, timeline)
    here_idx = _stage_index(stage, timeline)
    quoted = quoted_idx >= 0 and here_idx >= quoted_idx
    if direction == "inbound":
        # The case a pure-stage rule gets wrong: the customer is still sending us
        # scope long after the stage moved on.
        return "evidence", f"inbound during {stage}; what we are quoting FROM"
    if direction == "outbound" and quoted:
        return "label", f"outbound during {stage}; material we produced after quoting"
    if direction == "outbound":
        return "evidence", f"outbound before quoting; part of establishing scope"
    if direction == "internal":
        return "evidence", f"internal note during {stage}"

    # No direction (files carry none of their own). Stage alone is too weak to
    # overrule a classifier here, so defer rather than assert.
    return classified_as, f"arrived in {stage}; no direction available, classifier left in place"


def annotate(
    lifecycle: dict[str, Any] | None,
    *,
    authored_at: str | None,
    direction: str | None,
    timeline: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build the ``deal_stage`` block written onto each envelope document."""
    stage = stage_at_arrival(authored_at, timeline)
    classified = (lifecycle or {}).get("admissible_for")
    adm, why = admissibility(
        stage=stage, direction=direction, classified_as=classified, timeline=timeline,
    )
    return {
        "stage_at_arrival": stage,
        "direction": direction,
        "admissible_for": adm,
        "why": why,
        "changed_from_classifier": bool(classified) and adm != classified,
        "unplaced": stage is None,
    }
