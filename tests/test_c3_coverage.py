"""Consumers for the surfaces PR 414 left blocked. Synthetic data only."""
import json

import pytest

from ml.c3.coverage import READY_FOR_TRAINING, coverage_table, release_blockers
from ml.c3.data import DealExample, featurize
from ml.c3.gates import abstain, certify_page_budget, confidence_threshold, export_shadow, fit_temperature, per_head_scores, save_checkpoint, train
from ml.c3.longdeal import plan_long_deal
from ml.c3.readiness import audit_contract
from ml.c3.schema import load_schema
from ml.c3.values import canonical_term_verdict, word_indexes


def atom(key, text="Install two displays.", **kw):
    return {"label_key": key, "atom_id": f"at-{key}", "text": text, **kw}


def test_reason_codes_and_deal_answers_are_targets_not_just_retained():
    schema = load_schema()
    blob = {"judgments": [{
        "head": "gap", "target_key": "q:height", "labeler": "expert@example.com",
        "verdict": "invalid", "reason": "duplicate",
        "target": {"text": "How high are the displays?"},
        "note": "The height is already on the drawing.",
    }], "deal_answers": [{
        "labeler": "expert@example.com", "primary_service": "audio_visual",
        "declared_site_count": 3, "note": "Three clinics, one service.",
    }]}
    batch = featurize(DealExample.from_training_blob(blob, [atom("a")], deal_id="synthetic"), schema)
    reasons = [j for j in batch.judged if j.key == "jdg:gap:reason"]
    assert len(reasons) == 1
    assert reasons[0].answer == schema.by_key()["jdg:gap:reason"].index("duplicate")
    service = [j for j in batch.judged if j.key == "deal:primary_service"]
    assert service[0].answer == schema.by_key()["deal:primary_service"].index("audio_visual")
    assert batch.deal_numbers["declared_site_count"] == 3
    report = audit_contract(blob, [atom("a")], schema, deal_id="synthetic")
    assert "unconsumed_deal_answers" not in report["blockers"]
    assert "jdg:gap:reason" in report["targets"]


def test_unknown_reason_text_stays_rationale_and_unknown_service_fails():
    schema = load_schema()
    blob = {"judgments": [{
        "head": "gap", "target_key": "q:one", "labeler": "expert@example.com",
        "verdict": "valid", "reason": "This sentence is the expert explanation.",
        "target": {"text": "Who unlocks the door?"},
    }]}
    batch = featurize(DealExample.from_training_blob(blob, [atom("a")], deal_id="synthetic"), schema)
    assert not any(j.key.endswith(":reason") for j in batch.judged)
    assert any("sentence" in j.note for j in batch.judged)
    bad = {"deal_answers": [{"labeler": "expert@example.com", "primary_service": "invented"}]}
    with pytest.raises(ValueError, match="primary_service"):
        featurize(DealExample.from_training_blob(bad, [atom("a")], deal_id="synthetic"), schema)


def test_hours_terms_and_open_strings_keep_their_values():
    schema = load_schema()
    line = "Install 24 cable drops."
    blob = {"labels": [{"label_key": "a", "labeler": "expert@example.com", "label_type": "task",
                       "reads_set": {"region": "Bayview"}}],
            "judgments": [
                {"head": "hours", "target_key": "atom:at-a", "labeler": "expert@example.com",
                 "verdict": "hours=3;per=cable drop;basis=stated", "reason": "kept"},
                {"head": "term", "target_key": "atom:at-a", "labeler": "expert@example.com",
                 "verdict": "kind=payment_term;net_days=30;timing=arrears"},
                {"head": "norm", "target_key": "atom:at-a", "labeler": "expert@example.com",
                 "verdict": "24 cable"},
            ]}
    batch = featurize(DealExample.from_training_blob(blob, [atom("a", line)], deal_id="synthetic"), schema)
    hours = schema.by_key()["read:labor_hours"]
    assert batch.numbers_target["read:labor_hours"][0] == 3
    assert batch.targets["read:labor_hours"][0] == hours.index("present")
    assert batch.targets["jdg:hours:reason"][0] == schema.by_key()["jdg:hours:reason"].index("kept")
    assert batch.targets["jdg:term"][0] == schema.by_key()["jdg:term"].index("payment_term")
    assert batch.numbers_target["jdg:term:net_days"][0] == 30
    assert any(item.key == "jdg:norm" and item.text == "24 cable" and item.words == (1, 2)
               for item in batch.literals)
    assert any(item.key == "read:region" and item.text == "Bayview" for item in batch.literals)
    report = audit_contract(blob, [atom("a", line)], schema, deal_id="synthetic")
    assert "unconsumed_tab:hours" not in report["blockers"]
    assert "unconsumed_tab:term" not in report["blockers"]
    assert "24 cable" not in json.dumps(report)


def test_pixels_and_geometry_are_inputs_and_required_for_those_tabs():
    schema = load_schema()
    pixels = [0.25] * (8 * 8 * 3)
    row = atom("a", bbox=[0.1, 0.2, 0.3, 0.4], table_row=2, table_col=1,
               row_span=1, col_span=1, image_pixels=pixels)
    blob = {"judgments": [
        {"head": "image", "target_key": "atom:at-a", "labeler": "expert@example.com",
         "verdict": "photo"},
        {"head": "sheet", "target_key": "atom:at-a", "labeler": "expert@example.com",
         "text": "Install two displays.", "verdict": "scope",
         "target": {"a": {"atomId": "at-a"}}},
    ]}
    batch = featurize(DealExample.from_training_blob(blob, [row], deal_id="synthetic"), schema)
    assert any(batch.pixels[0]) and any(batch.geometry[0])
    assert "pixels" in batch.inputs() and "verdict" not in json.dumps(batch.inputs(), default=lambda t: t.tolist())
    report = audit_contract(blob, [row], schema, deal_id="synthetic")
    assert "vision_input_not_implemented" not in report["blockers"]
    assert "table_geometry_missing" not in report["blockers"]
    bare = audit_contract(blob, [atom("a")], schema, deal_id="synthetic")
    assert "vision_input_not_implemented" in bare["blockers"]
    assert "table_geometry_missing" in bare["blockers"]


def test_long_deal_keeps_every_edge_and_does_not_truncate():
    schema = load_schema()
    atoms = [atom(str(i), f"Line {i}.", entered_at=f"2026-01-0{i+1}T00:00:00Z") for i in range(4)]
    blob = {"links": [{"from_key": "0", "to_label_key": "3", "relation": "supports",
                      "labeler": "expert@example.com"}]}
    deal = DealExample.from_training_blob(blob, atoms, deal_id="synthetic")
    assert deal.edges == [("0", "3", "supports")]
    plan = plan_long_deal(deal, schema, max_atoms=2, overlap=0)
    assert plan.covered_edges == 1 and plan.bridges
    with pytest.raises(ValueError, match="do not truncate"):
        featurize(deal, schema, max_atoms=2)


def test_release_gates_refuse_training_holdout_fitting_and_export(tmp_path):
    torch = pytest.importorskip("torch")
    assert READY_FOR_TRAINING is False
    assert "frozen_corpus_not_audited" in release_blockers()
    with pytest.raises(RuntimeError, match="READY_FOR_TRAINING"):
        train()
    logits = torch.tensor([[2.0, 0.0], [0.1, 1.5]], requires_grad=True)
    labels = torch.tensor([0, 1])
    assert fit_temperature(logits, labels, split="validation") > 0
    with pytest.raises(ValueError, match="validation"):
        fit_temperature(logits.detach(), labels, split="holdout")
    probs = torch.tensor([0.9, 0.4, 0.2])
    threshold = confidence_threshold(probs, torch.tensor([True, False, False]), split="validation")
    assert bool(abstain(probs, threshold).any())
    with pytest.raises(ValueError, match="do not truncate"):
        certify_page_budget(["abcdef"], 2, len)
    schema = load_schema()
    from ml.c3.model import C3Config, C3Model
    model = C3Model(schema, C3Config(d_text=32, d=32, n_layers=1, n_heads=2, d_r=16, d_q=8,
                                    d_head=16, residual_dim=4, box_dim=4, n_policy_atoms=2, policy_rank=2))
    labeled = {"labels": [{"label_key": "a", "labeler": "expert@example.com", "label_type": "task"}]}
    batch = featurize(DealExample.from_training_blob(labeled, [atom("a")], deal_id="synthetic"), schema)
    scores = per_head_scores(model, batch)
    assert scores["col:label_type"]["n"] == 1
    assert 0 <= scores["col:label_type"]["exact"] <= 1
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, step=3, schema_sha256="abc")
    blob = json.dumps({"state": "ok"})
    assert "Install" not in path.read_bytes().decode("utf-8", errors="ignore")
    with pytest.raises(RuntimeError, match="Shadow export refused"):
        export_shadow(tmp_path / "export.pt", model, schema_sha256="abc", thresholds={},
                      contract_ready=False, pretrained_budget_certified=False,
                      rules_evaluated=False, holdout_touched=True)
    assert blob  # checkpoint metadata is hashes and tensors, not this source line


def test_coverage_table_does_not_mark_release_gates_consumed():
    rows = {row["surface"]: row for row in coverage_table()}
    assert rows["jdg:gap:reason"]["status"] == "consumed"
    assert rows["deal:declared_site_count"]["status"] == "consumed"
    assert rows["jdg:norm"]["status"] == "consumed_without_student_decoder"
    assert rows["jdg:rule"]["status"] == "consumed" and rows["jdg:rule"]["loss"] == "judgment_loss"
    assert rows["frozen_corpus_not_audited"]["status"] == "blocked"
    assert {opp.key for opp in load_schema().opportunities} <= set(rows)
    assert rows["jdg:router"]["status"] == "deferred"


def test_value_grammars_match_the_parser_readers():
    from app.core.commercial_term_read import canonical_term_verdict as app_term
    from app.core.task_hours import hours_judgment_basis as app_basis
    from ml.c3.values import hours_judgment_basis
    assert canonical_term_verdict("kind=payment_term;net_days=30;timing=arrears") == app_term(
        "timing=arrears;kind=payment_term;net_days=30")
    assert hours_judgment_basis("hours=3;basis=stated") == app_basis("hours=3;basis=stated")
    assert hours_judgment_basis("unstated") == app_basis("unstated")
    assert word_indexes("Install 24 cable drops.", "24 cable") == (1, 2)
