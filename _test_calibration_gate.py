"""Calibrator wiring (Phase-1 #8) — the deterministic review gate + envelope
projection. No joblib/model needed: recalibrate_confidence sets a non-null
calibrated_confidence and flips low-confidence auto_accepted atoms to
needs_review (atomic with the calibration_abstain flag), and _compact_atom now
projects calibrated_confidence + review_status so consumers stop reading null.
Run: python _test_calibration_gate.py
"""
from app.core.confidence_recalibration import recalibrate_confidence
from app.core.ids import stable_id
from app.core.orbitbrief_envelope import _compact_atom
from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    EvidenceAtom,
    ReviewStatus,
    SourceRef,
)
from app.learning.calibration import default_calibrator_path


def _mk(aid, authority, text="Provider shall perform the work."):
    return EvidenceAtom(
        id=aid, project_id="p", artifact_id="art", atom_type=AtomType.scope_item,
        raw_text=text, normalized_text=text.lower(), value={"kind": "paragraph"}, entity_keys=[],
        source_refs=[SourceRef(id=stable_id("src", aid), artifact_id="art",
            artifact_type=ArtifactType.txt, filename="art.txt", locator={},
            extraction_method="t", parser_version="t")],
        receipts=[], authority_class=authority, confidence=0.9, confidence_raw=0.9,
        calibrated_confidence=None, review_status=ReviewStatus.auto_accepted,
        review_flags=[], parser_version="t")


def test_default_calibrator_path_noop_when_unset():
    import os
    os.environ.pop("SOWSMITH_CALIBRATOR_PATH", None)
    assert default_calibrator_path() is None
    os.environ["SOWSMITH_CALIBRATOR_PATH"] = "/nope/x.joblib"
    assert default_calibrator_path() is None  # missing file -> no-op
    os.environ.pop("SOWSMITH_CALIBRATOR_PATH", None)
    print("  ok: default_calibrator_path is a safe no-op when unset/missing")


def test_review_gate_and_projection():
    a = _mk("a1", AuthorityClass.quoted_old_email)  # no corroboration -> low score
    recalibrate_confidence([a], abstain_threshold=0.70)
    assert a.calibrated_confidence is not None, "calibrated_confidence must be non-null"
    if a.calibrated_confidence < 0.70:
        assert a.review_status == ReviewStatus.needs_review
        assert "calibration_abstain" in a.review_flags
    # validator invariant: the flag is present IFF needs_review
    if "calibration_abstain" in a.review_flags:
        assert a.review_status == ReviewStatus.needs_review
    proj = _compact_atom(a)
    assert proj["calibrated_confidence"] is not None
    assert proj["review_status"] is not None
    assert "confidence_raw" in proj
    print(f"  ok: gate + projection (calibrated={a.calibrated_confidence}, status={a.review_status.value})")


if __name__ == "__main__":
    test_default_calibrator_path_noop_when_unset()
    test_review_gate_and_projection()
    print("PASS _test_calibration_gate")
