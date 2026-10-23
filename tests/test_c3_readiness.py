"""Synthetic regressions for data identity, rich targets and release gates."""
import json

import pytest

from ml.c3.data import DealExample, IGNORE, _Lines, featurize, select_labels
from ml.c3.readiness import audit_contract, validate_split
from ml.c3.schema import load_schema


def atom(key, text="Example work item.", **kw):
    return {"label_key": key, "atom_id": f"at-{key}", "text": text, **kw}


def label(key, **kw):
    return {"label_key": key, "labeler": "reviewer@example.com", "label_type": "task",
            "reads_set": {}, **kw}


def test_person_priority_and_negative_evidence_survive_both_export_orders():
    person = label("a", parser_type="deadline", rejected_reads={"crew_size": "Not stated."})
    mirror = label("a", labeler="tool (assistant)", label_type="deadline")
    for rows in ([person, mirror], [mirror, person]):
        deal = DealExample.from_training_blob({"labels": rows}, [atom("a")], deal_id="synthetic")
        assert deal.atoms[0].label == person
        batch = featurize(deal, load_schema())
        assert batch.negatives["col:label_type"][0]
        assert batch.targets["read:crew_size"][0] == 0


def test_conflicting_saved_answers_fail_instead_of_last_row_wins():
    with pytest.raises(ValueError, match="adjudication"):
        select_labels([label("a"), label("a", label_type="schedule")])
    assert select_labels([label("a", labeler="tool (assistant)")]) == {}


def test_repeated_text_requires_occurrence_and_document_evidence():
    deal = DealExample.from_training_blob({}, [atom("a", doc_id="first"),
                                              atom("b", doc_id="second")], deal_id="synthetic")
    lines = _Lines(deal.atoms)
    assert lines.find({"text": "Example work item."}) is None
    assert lines.find({"labelKey": "b"}) == 1
    assert lines.find({"text": "Example work item.", "artifactId": "second"}) == 1
    blob = {"judgments": [{"head": "suppression", "target_key": "sup:b@1",
                           "labeler": "reviewer@example.com", "text": "Example work item.",
                           "verdict": "correctly_dropped"}]}
    deal = DealExample.from_training_blob(blob, [atom("a"), atom("b")], deal_id="synthetic",
                                          serve_lines_only=False)
    batch = featurize(deal, load_schema())
    # The card names a copy at another position: it gets its own line, and
    # the kept line b keeps its own (empty) admission.
    copy = [a.key for a in deal.atoms].index("sup:b@1")
    assert [a.key for a in deal.atoms][:2] == ["a", "b"]
    assert batch.targets["col:admission"][copy] == 2
    assert batch.targets["col:admission"][:2] == [IGNORE, IGNORE]


def test_retired_head_collision_does_not_remove_live_task_links():
    blob = {"judgments": [{"head": "hours", "target_key": "atom:at-a",
                           "labeler": "reviewer@example.com",
                           "note": "EXCLUDE_FROM_TRAINING: retired hours."}],
            "links": [{"from_head": head, "from_key": "atom:at-a", "to_label_key": "b",
                       "relation": "supports", "labeler": "reviewer@example.com"}
                      for head in ("hours", "task_tier")]}
    deal = DealExample.from_training_blob(blob, [atom("a"), atom("b", "Other item.")],
                                         deal_id="synthetic")
    assert deal.edges == [("a", "b", "supports")]
    assert deal.link_stats["retired_card_links"] == 1


def test_missing_suppressed_occurrence_cannot_become_unique_survivor():
    blob = {"links": [{"from_head": "suppression", "from_key": "sup:missing@1",
                       "from_text": "Example work item.", "to_label_key": "b",
                       "relation": "same_as", "labeler": "reviewer@example.com"}]}
    deal = DealExample.from_training_blob(blob, [atom("a"), atom("b", "Other item.")],
                                         deal_id="synthetic")
    assert not deal.edges and deal.link_stats["unresolved"] == 1


def test_deferred_tabs_fields_and_modalities_block_readiness_without_text_output():
    secret = "PRIVATE_SOURCE_TEXT_123"
    blob = {"labels": [label("a", reads_set={"unregistered_read": secret})],
            "deal_answers": [{"answer": secret, "labeler": "reviewer@example.com"}],
            "judgments": [{"head": "norm", "target_key": "atom:at-a", "text": secret,
                           "verdict": secret, "labeler": "reviewer@example.com"}]}
    deal = DealExample.from_training_blob(blob, [atom("a")], deal_id="synthetic")
    assert deal.source_supervision["deal_answers"] == blob["deal_answers"]
    report = audit_contract(blob, [atom("a")], load_schema(), deal_id="synthetic")
    assert not report["ready"]
    assert {"unconsumed_deal_answer_field:answer",
            "unconsumed_read:unregistered_read"} <= report["blockers"].keys()
    assert "unconsumed_tab:norm" not in report["blockers"]
    assert report["classes"]["jdg:norm"] == {"consumed": 1}
    assert secret not in json.dumps(report)


def test_manifest_requires_untouched_holdout_and_source_group_separation():
    manifest = {split: [{"deal_id": split, "snapshot_sha256": "a" * 64,
                        "audit_status": "verified", "source_groups": [split]}]
                for split in ("train", "validation", "holdout")}
    manifest["holdout_used_for_tuning"] = False
    assert validate_split(manifest) == []
    manifest["holdout"][0]["source_groups"] = ["train"]
    assert "source_group_leakage" in validate_split(manifest)
    manifest["holdout_used_for_tuning"] = True
    assert "holdout_not_certified_untouched" in validate_split(manifest)


def test_cli_checks_actual_snapshot_hash_and_exits_blocked_on_mismatch(tmp_path, monkeypatch):
    from ml.c3.readiness import main

    blob, atoms = {"labels": [label("a")]}, [atom("a")]
    paths = {k: tmp_path / (k + ".json") for k in ("blob", "atoms", "manifest", "output")}
    paths["blob"].write_text(json.dumps(blob))
    paths["atoms"].write_text(json.dumps(atoms))
    manifest = {split: [{"deal_id": split, "snapshot_sha256": "a" * 64,
                        "audit_status": "verified", "source_groups": [split]}]
                for split in ("train", "validation", "holdout")}
    manifest["holdout_used_for_tuning"] = False
    paths["manifest"].write_text(json.dumps(manifest))
    argv = ["readiness", "--blob", str(paths["blob"]), "--atoms", str(paths["atoms"]),
            "--deal-id", "train", "--split-manifest", str(paths["manifest"]),
            "--output", str(paths["output"])]
    monkeypatch.setattr("sys.argv", argv)
    assert main() == 2
    assert "snapshot_not_pinned_in_manifest" in json.loads(paths["output"].read_text())["split_blockers"]
    report = audit_contract(blob, atoms, load_schema(), deal_id="train")
    manifest["train"][0]["snapshot_sha256"] = report["snapshot_sha256"]
    paths["manifest"].write_text(json.dumps(manifest))
    assert main() == 0


def test_dense_long_deal_cannot_silently_truncate():
    deal = DealExample.from_training_blob({}, [atom("a"), atom("b")], deal_id="synthetic")
    with pytest.raises(ValueError, match="do not truncate"):
        featurize(deal, load_schema(), max_atoms=1)


def test_multi_select_loss_trains_all_members_and_masks_unknowns():
    torch = pytest.importorskip("torch")
    from ml.c3.losses import answer_loss
    from ml.c3.supercharge import _kl, mask_labels

    schema = load_schema()
    deal = DealExample.from_training_blob({"labels": [label("a", hints=["table_column", "own_words"])]},
                                         [atom("a"), atom("b")], deal_id="synthetic")
    batch = featurize(deal, schema)
    key = "col:hints"
    assert batch.targets[key] == [IGNORE, IGNORE]
    opp = schema.by_key()[key]
    logits = torch.zeros(2, len(opp.answers), requires_grad=True)
    loss = answer_loss(logits, batch, key, torch.ones(2))
    loss.backward()
    for value in ("table_column", "own_words"):
        assert logits.grad[0, opp.index(value)] < 0
    assert logits.grad[0, 0] > 0
    assert torch.equal(logits.grad[1], torch.zeros_like(logits.grad[1]))
    assert mask_labels(batch, set()).multi_targets[key] == [None, None]
    # Matching independent probabilities distill to zero; no softmax competition.
    assert abs(float(_kl(logits.detach(), logits.detach(), 2.0, True))) < 1e-6


def test_excluded_fields_and_proposals_do_not_block_or_train():
    blob = {"labels": [label("a", note="EXCLUDE_FROM_TRAINING: excluded.",
                              reads_set={"unregistered_read": "not a target"})],
            "judgments": [{"head": "norm", "labeler": "tool (assistant)", "verdict": "proposal"}]}
    report = audit_contract(blob, [atom("a")], load_schema(), deal_id="synthetic")
    assert report["ready"]
    assert report["counts"]["excluded_labels"] == 1


def test_duplicate_occurrence_identifiers_are_rejected():
    with pytest.raises(ValueError, match="Duplicate source occurrence"):
        DealExample.from_training_blob({}, [atom("a"), atom("a")], deal_id="synthetic")


def test_unlabeled_distillation_requires_enablement_and_confidence():
    torch = pytest.importorskip("torch")
    from ml.c3.model import C3Config, C3Model
    from ml.c3.supercharge import TeacherView, TeachWeights, supercharge_loss

    schema = load_schema()
    cfg = C3Config(d_text=32, d=32, n_layers=1, n_heads=2, d_r=16, d_q=8,
                   d_head=16, residual_dim=4, box_dim=4, n_policy_atoms=2, policy_rank=2)
    model = C3Model(schema, cfg)
    batch = featurize(DealExample.from_training_blob({}, [atom("a")], deal_id="synthetic"), schema)
    out = model(batch.inputs())
    key = "col:label_type"
    target = torch.zeros_like(out.logits[key])
    view = TeacherView(told={key: target}, noted={key: target}, embedding=out.r.detach(),
                       has_why=torch.tensor([False]), labeled=torch.tensor([False]))
    assert supercharge_loss(model, out, view)["teach_unlabeled"] == 0
    weights = TeachWeights(unlabeled=0.3)
    assert supercharge_loss(model, out, view, weights)["teach_unlabeled"] == 0
    target[:, 0] = 40
    assert supercharge_loss(model, out, view, weights)["teach_unlabeled"] > 0


def test_teacher_and_heads_train_multi_selection_without_scalar_gold():
    torch = pytest.importorskip("torch")
    from ml.c3.brain import Brain
    from ml.c3.lm import TinyCausalLM
    from ml.c3.losses import BrainWeights, LossWeights, brain_loss, head_losses
    from ml.c3.model import C3Config, C3Model

    schema = load_schema()
    deal = DealExample.from_training_blob({"labels": [label("a", hints=["own_words", "table_column"],
        note="The quantity is stated explicitly and its column supplies the unit.")]},
        [atom("a")], deal_id="synthetic")
    batch = featurize(deal, schema)
    cfg = C3Config(d_text=32, d=32, n_layers=1, n_heads=2, d_r=16, d_q=8,
                   d_head=16, residual_dim=4, box_dim=4, n_policy_atoms=2, policy_rank=2)
    model = C3Model(schema, cfg)
    out = model(batch.inputs())
    loss, _ = head_losses(model, out, batch, LossWeights(), "universal")
    assert torch.isfinite(loss)
    loss.backward()
    teacher = Brain(schema, TinyCausalLM(32, 1, 2, 4096), context_lines=2)
    loss, parts = brain_loss(teacher, batch, BrainWeights(consolidate=0))
    assert torch.isfinite(loss) and parts["direct"] > 0 and parts["reasoned"] > 0
    loss.backward()
