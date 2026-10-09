"""PUR-29: Division is proposed from the work order, spelling variants collapse
without a synonym list, and the proposer abstains when work could be two divisions.
All data is synthetic (tests/fixtures/division)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from app.core import division_proposal as dp

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/division/synthetic_labelled.json"


@pytest.fixture(scope="module")
def examples():
    return dp.load_examples(str(FIXTURE))


@pytest.fixture(scope="module")
def model(examples):
    return dp.DivisionProposer().fit(examples)


def _wo(*lines):
    return {"work_lines": [{"work": w, "object": o, "count": c, "unit": u} for w, o, c, u in lines]}


def test_fixture_is_synthetic():
    assert json.loads(FIXTURE.read_text())["synthetic"] is True


def test_spelling_variants_collapse_from_string_and_cooccurrence(model):
    m = model.label_map
    assert m["networking"] == m["network"] == "Network"
    assert m["cabling"] == m["cable"] == "Cable"
    assert m["network/lv"] == "Network/LV"  # shares a prefix, but is its own label
    assert len(set(m.values())) == 9


def test_string_similarity_alone_does_not_merge():
    feats_a = ["obj:switch"] * 5
    feats_b = ["obj:pallet"] * 5
    out = dp.cluster_labels([("Network", feats_a), ("Networking", feats_b)])
    assert out["network"] != out["networking"]


def test_similar_work_with_different_spelling_does_not_merge():
    f = ["obj:camera"] * 5
    out = dp.cluster_labels([("Camera Install", f), ("AV", f)])
    assert out["camera install"] != out["av"]


def test_proposes_from_work_signals(model):
    p = model.propose(_wo(("install and configure access points", "access point", 12, "ap"),
                          ("mount network switch", "network switch", 2, "device")))
    assert p.division == "Network"
    assert any(s.startswith("obj:") for s in p.signals)


def test_same_word_different_work_changes_division(model):
    """'install' appears everywhere; the object decides."""
    cam = model.propose(_wo(("install security cameras", "security camera", 8, "camera")))
    av = model.propose(_wo(("install displays", "display", 8, "room")))
    assert cam.division == "Camera Install" and av.division == "AV"


def test_abstains_when_work_splits_across_divisions(model):
    p = model.propose(_wo(("image laptops", "laptop", 40, "user"),
                          ("install security cameras", "security camera", 10, "camera")))
    assert p.division is None and p.reason.startswith("ambiguous")


def test_abstains_without_work_signal(model):
    p = model.propose({"work_lines": [{"work": "quote per attached", "object": "xyzzy"}]})
    assert p.division is None and p.reason == "no_work_signal"


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv(dp.FLAG, raising=False)
    assert dp.propose_division(_wo(("install cameras", "security camera", 2, "camera"))) is None


def test_enabled_without_labels_abstains(monkeypatch):
    monkeypatch.setenv(dp.FLAG, "1")
    monkeypatch.delenv(dp.LABELS_ENV, raising=False)
    assert dp.propose_division(_wo(("install cameras", "security camera", 2, "camera")))["reason"] == "no_model"


def test_enabled_with_labels(monkeypatch):
    monkeypatch.setenv(dp.FLAG, "1")
    monkeypatch.setenv(dp.LABELS_ENV, str(FIXTURE))
    out = dp.propose_division(_wo(("palletize hardware kits", "pallet", 6, "pallet")))
    assert out["division"] == "palletize" and out["abstained"] is False


def test_holdout_stratified_evaluation(examples):
    sys.path.insert(0, str(ROOT / "scripts"))
    from eval_division_proposal import evaluate, stratified_folds

    folds = stratified_folds(examples, 3, 0)
    for f in folds:  # every normalised label appears in every holdout fold
        assert {dp.normalize_label(examples[i]["division"]) for i in f} == {
            dp.normalize_label(e["division"]) for e in examples}
    r = evaluate(examples, k=3, seed=0)
    assert r["accuracy_on_answered"] >= 0.9
    assert r["abstention_rate_ambiguous"] >= 0.8
    assert r["abstention_rate_clean"] <= 0.1


def test_work_order_stage_attaches_proposal(monkeypatch):
    from app.core import work_order
    from tests.test_work_order import SCOPE, WORK_ORDER

    class D:
        verdict, source = "about_this_job", "store"

    monkeypatch.setattr("app.core.decide.decide", lambda *a, **k: D())
    monkeypatch.setattr("app.core.llm_client.complete", lambda *a, **k: json.dumps(WORK_ORDER))
    monkeypatch.setenv(dp.FLAG, "1")
    monkeypatch.setenv(dp.LABELS_ENV, str(FIXTURE))
    _, _, report = work_order.apply_work_order(list(SCOPE), project_id="p1", deal_name="d")
    assert report["summary"]["division_proposal"]["division"] == "Network"

    monkeypatch.delenv(dp.FLAG)
    _, _, report = work_order.apply_work_order(list(SCOPE), project_id="p1", deal_name="d")
    assert "division_proposal" not in report["summary"]
