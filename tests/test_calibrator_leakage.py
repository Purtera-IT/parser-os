"""The calibrator must not be able to memorise its own labels.

Background: it was promoting nightly with Brier=0.0001 vs a 0.4957 heuristic.
That is not calibration, it is leakage — the silver label rule assigns 0 when
`review_flags` is non-empty, and `review_flag_count` was handed to the model as
a feature. The packet side had the identical hole via `contradicting_atom_count`.

This matters beyond a metric: calibrated_confidence drives the review/auto-accept
gate, so a meaningless-but-confident number can auto-accept work a PM should
have seen.
"""
from __future__ import annotations

from app.learning.calibration import build_calibration_labels
from app.learning.features import build_atom_feature_row, build_packet_feature_row
from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    EvidenceAtom,
    ReviewStatus,
    SourceRef,
)


def _atom(atom_id="atm_1", flags=None, status=ReviewStatus.auto_accepted, conf=0.9):
    return EvidenceAtom(
        id=atom_id,
        atom_id=atom_id,
        project_id="p1",
        artifact_id="art_1",
        atom_type=AtomType.scope_item,
        raw_text="Contractor to patch penetrations.",
        normalized_text="contractor to patch penetrations.",
        source_refs=[
            SourceRef(
                id="src_1",
                artifact_id="art_1",
                artifact_type=ArtifactType.docx,
                filename="scope.docx",
                locator={"page": 1},
                extraction_method="docx_rules",
                parser_version="v1",
            )
        ],
        authority_class=AuthorityClass.customer_current_authored,
        confidence=conf,
        review_status=status,
        review_flags=flags or [],
        parser_version="v1",
    )


def test_atom_features_no_longer_encode_the_silver_label():
    """`label = 0 iff review_flags` — so exposing the flag COUNT hands the model
    its own answer key."""
    row = build_atom_feature_row(_atom(flags=["low_confidence_needs_review"]))
    assert "review_flag_count" not in row


def test_the_leak_is_real_if_the_feature_comes_back():
    """Guards the reasoning, not just the deletion: a flagged atom is labelled 0
    and an unflagged confident one is labelled 1, so any feature counting flags
    separates the classes perfectly."""
    flagged = _atom("atm_bad", flags=["low_confidence_needs_review"])
    clean = _atom("atm_good", flags=[], conf=0.95)
    labels = build_calibration_labels([_Result([flagged, clean])])
    by_id = {r["atom_id"]: r["label"] for r in labels["atom_labels"]}
    assert by_id.get("atm_bad") == 0
    # If review_flag_count were a feature, len(flags) alone would predict this.
    assert len(flagged.review_flags) > 0 and len(clean.review_flags) == 0


def test_packet_features_no_longer_encode_the_packet_label():
    row = build_packet_feature_row(_Packet(), [])
    assert "contradicting_atom_count" not in row


def test_surviving_features_are_still_useful():
    """The fix must not gut the model — raw confidence is exactly what a
    calibrator SHOULD see; it just must not be the label."""
    row = build_atom_feature_row(_atom())
    for expected in ("confidence_raw", "authority_class", "atom_type", "source_ref_count"):
        assert expected in row


class _Result:
    def __init__(self, atoms):
        self.atoms = atoms
        self.packets = []


class _Cert:
    ambiguity_score = 0.0
    evidence_completeness_score = 1.0


class _Packet:
    from app.core.schemas import PacketFamily as _PF

    id = "pkt_1"
    family = _PF.scope_inclusion
    confidence = 0.9
    confidence_raw = 0.9
    governing_atom_ids: list = []
    supporting_atom_ids: list = []
    contradicting_atom_ids: list = ["atm_bad"]
    certificate = _Cert()
    risk = None
    review_flags: list = []
