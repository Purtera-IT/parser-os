"""Calibrator auto-train harvester (Phase-1 #11). Verifies build_calibration_labels
(PM gold + silver bootstrap, both classes) + brier_score. The fit itself + the
eval-gate are exercised by the nightly job (needs blob result.json); train_calibrator
is covered by tests/test_confidence_calibration.py. Run: python _test_calibrator_train.py
"""
from types import SimpleNamespace as NS

from app.learning.calibration import brier_score, build_calibration_labels


def _atom(aid, flags=None, conf=0.9, verified=False, status="auto_accepted"):
    return NS(id=aid, review_status=NS(value=status), review_flags=flags or [],
              calibrated_confidence=conf, confidence=conf,
              receipts=[NS(replay_status="verified")] if verified else [])


def _pkt(pid, contra=None, conf=0.9):
    return NS(id=pid, contradicting_atom_ids=contra or [], confidence=conf)


def test_harvester_labels_both_classes():
    res = NS(
        atoms=[_atom("a_pm"), _atom("a_flag", flags=["calibration_abstain"]),
               _atom("a_good", conf=0.9, verified=True), _atom("a_mid", conf=0.6)],
        packets=[_pkt("p_bad", contra=["x"]), _pkt("p_good", conf=0.95), _pkt("p_mid", conf=0.5)],
    )
    lab = build_calibration_labels([res], pm_corrected_atom_ids={"a_pm"})
    am = {r["atom_id"]: r["label"] for r in lab["atom_labels"]}
    pm = {r["packet_id"]: r["correct_packet"] for r in lab["reviews"]}
    assert am == {"a_pm": 0, "a_flag": 0, "a_good": 1}, am          # PM gold + silver; mid omitted
    assert pm == {"p_bad": False, "p_good": True}, pm                # contested vs clean; mid omitted
    assert len({*am.values()}) == 2 and len({*pm.values()}) == 2     # both classes -> fittable
    print("  ok: harvester yields PM-gold + silver labels, both classes")


def test_brier():
    assert abs(brier_score([0.9, 0.1], [1, 0]) - 0.01) < 1e-9
    assert brier_score([], []) == 1.0
    print("  ok: brier_score")


if __name__ == "__main__":
    test_harvester_labels_both_classes()
    test_brier()
    print("PASS _test_calibrator_train")
