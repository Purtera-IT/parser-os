"""Complaint root-cause router: classify a PM complaint into the fix it needs.

Verifies the router distinguishes the three honest buckets — a learnable SEAM
correction, a deterministic GATE bug (code fix), and a never-detected region
(extractor) — plus the unlocalizable case, using only the provenance the
pipeline already records (span ledger, retained suppressed atoms, content
census). The point is the *negative* guarantee: it must NOT hand back a learnable
correction for a GATE bug or a never-detected miss, because such a correction
would never fire and the PM would re-report the same loss.
"""

from __future__ import annotations

from app.core.complaint_intake import Complaint, KIND_WRONGLY_DROPPED, KIND_WRONGLY_KEPT
from app.core.complaint_router import RootCause, route
from app.core.content_census import ContentCensus, Region, RegionKind
from app.core.span_ledger import SpanLedger, StageKind


class _Atom:
    """Minimal atom stand-in matching the attributes the router reads."""

    def __init__(self, aid, text, *, value=None, decision_provenance=None):
        self.id = aid
        self.raw_text = text
        self.normalized_text = text
        self.value = value or {}
        self.decision_provenance = decision_provenance


class _Result:
    def __init__(self, atoms=None, suppressed=None):
        self.atoms = atoms or []
        self.suppressed_atoms = suppressed or []


# ── 1. accepted atom → learnable SEAM correction ────────────────────────
def test_accepted_atom_routes_to_learnable_seam_correction():
    result = _Result(atoms=[_Atom("a1", "PurTera Solutions, 500 Main St")])
    c = Complaint(
        relation="physical_site",
        desired_verdict="not_a_site",
        text="PurTera Solutions, 500 Main St",
        kind=KIND_WRONGLY_KEPT,
    )
    v = route(c, result=result)
    assert v.root_cause is RootCause.SEAM_CORRECTION
    assert v.learnable is True
    assert v.code_fix is False
    assert v.resolution is not None  # proposed correction ready to confirm
    assert v.resolution.proposed_correction.status == "proposed"


# ── 2. seam-suppressed atom → learnable (store can re-admit) ─────────────
def test_seam_suppressed_atom_is_learnable():
    dropped = _Atom(
        "a2",
        "24 access points at the Annex",
        value={"_suppression": {"stage": "decide:scope", "reason": "low conf"}},
        decision_provenance={"source": "store", "rationale": "kNN drop"},
    )
    result = _Result(suppressed=[dropped])
    c = Complaint(
        relation="scope_item",
        desired_verdict="keep",
        text="24 access points at the Annex",
        kind=KIND_WRONGLY_DROPPED,
    )
    v = route(c, result=result)
    assert v.root_cause is RootCause.SEAM_CORRECTION
    assert v.learnable is True
    assert v.recoverable is True


# ── 3. gate-suppressed atom → GATE bug, NOT learnable ───────────────────
def test_gate_suppressed_atom_is_code_fix_not_learnable():
    dropped = _Atom(
        "a3",
        "Lookup helper: region codes",
        value={"_suppression": {"stage": "sheet_router", "reason": "sheet_role=DROP"}},
    )
    result = _Result(suppressed=[dropped])
    ledger = SpanLedger()
    ledger.record_drop(
        span_id="s3",
        stage="sheet_router",
        kind=StageKind.GATE,
        rule="sheet_role_router",
        reason="sheet_role=DROP",
        raw_text="Lookup helper: region codes",
    )
    c = Complaint(
        relation="scope_item",
        desired_verdict="keep",
        text="Lookup helper: region codes",
        kind=KIND_WRONGLY_DROPPED,
    )
    v = route(c, result=result, ledger=ledger)
    assert v.root_cause is RootCause.GATE_BUG
    assert v.learnable is False     # a store correction would never fire here
    assert v.code_fix is True
    assert v.recoverable is True    # retained marker → PM can pull it now
    assert v.resolution is None


# ── 4. never-detected (census UNCOVERED) → needs extractor ──────────────
def test_never_detected_region_routes_to_extractor():
    census = ContentCensus(artifact="a")
    census.register(
        Region(
            region_id="r1",
            artifact="a",
            kind=RegionKind.TEXT,
            location="textbox/floorplan-callout",
            text="Server room requires 3x 42U racks",
        )
    )
    census.reconcile([])  # no atoms produced → region is UNCOVERED
    c = Complaint(
        relation="scope_item",
        desired_verdict="keep",
        text="Server room requires 3x 42U racks",
        kind=KIND_WRONGLY_DROPPED,
    )
    v = route(c, result=_Result(), census=census)
    assert v.root_cause is RootCause.NEEDS_EXTRACTOR
    assert v.learnable is False
    assert v.code_fix is True
    assert v.recoverable is False
    assert "textbox/floorplan-callout" in v.fix_target


# ── 5. ledger-only GATE loss (no atom at all) → GATE bug ────────────────
def test_ledger_gate_loss_with_no_atom_is_gate_bug():
    ledger = SpanLedger()
    ledger.record_drop(
        span_id="s5",
        stage="prose_gate",
        kind=StageKind.GATE,
        rule="docx_prose_keep_drop",
        reason="looked like boilerplate",
        raw_text="All cabling must be plenum-rated per code.",
    )
    c = Complaint(
        relation="scope_item",
        desired_verdict="keep",
        text="All cabling must be plenum-rated per code.",
    )
    v = route(c, result=_Result(), ledger=ledger)
    assert v.root_cause is RootCause.GATE_BUG
    assert v.learnable is False
    assert "docx_prose_keep_drop" in v.fix_target


# ── 6. nothing matches → unlocalized (ask the PM) ───────────────────────
def test_unlocalizable_complaint():
    c = Complaint(relation="scope_item", desired_verdict="keep", text="ghost text")
    v = route(c, result=_Result())
    assert v.root_cause is RootCause.UNLOCALIZED
    assert v.learnable is False
    assert v.code_fix is False
    assert v.resolution is None


# ── 7. recurring GATE bug can ALSO be learnable (the "and if both" case) ─
def test_fix_shape_strings_are_actionable():
    # Pure seam.
    seam = _Result(atoms=[_Atom("a", "PurTera HQ")])
    v = route(
        Complaint(relation="physical_site", desired_verdict="not_a_site", text="PurTera HQ"),
        result=seam,
    )
    assert "self-healing" in v.fix_shape

    # Gate bug.
    led = SpanLedger()
    led.record_drop(
        span_id="s", stage="g", kind=StageKind.GATE, rule="r",
        reason="x", raw_text="dropped line here",
    )
    v2 = route(
        Complaint(relation="scope_item", desired_verdict="keep", text="dropped line here"),
        result=_Result(),
        ledger=led,
    )
    assert "code fix" in v2.fix_shape.lower()
