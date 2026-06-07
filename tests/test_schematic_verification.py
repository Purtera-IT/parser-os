"""The 100% engine: verify + cross-check + abstain. Local, deterministic, no API.
Emit only verified items; flag everything uncertain; surface gaps."""
from __future__ import annotations

from app.core.schematic_verification import (
    CONFIDENT,
    FLAGGED,
    MISSING,
    Detection,
    LegendEntry,
    delivered_accuracy,
    verify_page,
)


def _scenario():
    legend = [
        LegendEntry("cam", "ptz camera", declared_count=3),
        LegendEntry("smk", "smoke detector", declared_count=2),
        LegendEntry("spk", "speaker", declared_count=4),
        LegendEntry("door", "card reader", declared_count=1),
    ]
    dets = [
        Detection("d1", "cam"), Detection("d2", "cam"), Detection("d3", "cam"),   # 3=3 confident
        Detection("d4", "smk"),                                                    # 1!=2 flagged
        Detection("d5", "spk", 0.4), Detection("d6", "spk"), Detection("d7", "spk"), Detection("d8", "spk"),  # low-conf flag
        Detection("d9", None, 0.5),                                                # unknown flag
        # card reader: none found -> MISSING
    ]
    return legend, dets


def test_count_match_is_confident():
    legend, dets = _scenario()
    r = verify_page(legend, dets)
    cam = next(i for i in r.items if i.detail.get("type") == "ptz camera")
    assert cam.status == CONFIDENT


def test_count_mismatch_is_flagged():
    legend, dets = _scenario()
    r = verify_page(legend, dets)
    smk = next(i for i in r.items if i.detail.get("type") == "smoke detector")
    assert smk.status == FLAGGED


def test_zero_found_is_missing_gap():
    legend, dets = _scenario()
    r = verify_page(legend, dets)
    door = next(i for i in r.items if i.detail.get("type") == "card reader")
    assert door.status == MISSING


def test_unknown_symbol_is_flagged():
    legend, dets = _scenario()
    r = verify_page(legend, dets)
    assert any(i.kind == "detection" and i.status == FLAGGED for i in r.items)


def test_no_silent_errors_everything_triaged():
    legend, dets = _scenario()
    r = verify_page(legend, dets)
    # every item is exactly one of the three states — nothing emitted unverified
    assert all(i.status in (CONFIDENT, FLAGGED, MISSING) for i in r.items)
    assert len(r.confident) + len(r.review_queue) == len(r.items)


def test_delivered_accuracy_is_100_when_queue_resolved():
    legend, dets = _scenario()
    r = verify_page(legend, dets)
    assert delivered_accuracy(r, human_resolves_queue=True) == 1.0
    assert delivered_accuracy(r, human_resolves_queue=False) < 1.0  # auto-only is honest, not 100


def test_perfect_page_all_confident_zero_queue():
    legend = [LegendEntry("cam", "camera", declared_count=2)]
    dets = [Detection("d1", "cam"), Detection("d2", "cam")]
    r = verify_page(legend, dets)
    assert len(r.review_queue) == 0
    assert r.coverage == 1.0
    assert delivered_accuracy(r, human_resolves_queue=False) == 1.0
