"""Frozen rule preparation is person-only and never reads held-out blobs."""
import hashlib
import json

import pytest

from app.learning.rule_feedback import export_frozen_training, rows_from_judgments


def judgment(verdict="should_fire", labeler="reviewer@example.com"):
    return {"head": "rule", "labeler": labeler, "verdict": verdict,
            "target": {"rule": "example_rule", "text": "Example candidate text.", "fired": True,
                       "bestPos": 0.7, "bestNeg": 0.4, "threshold": 0.5}}


def frozen(tmp_path, rows):
    blob = tmp_path / "train.json"
    blob.write_text(json.dumps({"judgments": rows}))
    manifest = {split: [{"deal_id": split, "audit_status": "verified", "source_groups": [split],
                        "blob_path": str(tmp_path / "DO_NOT_OPEN.json"), "blob_sha256": "a" * 64}]
                for split in ("train", "validation", "holdout")}
    manifest["train"][0].update(blob_path=str(blob), blob_sha256=hashlib.sha256(blob.read_bytes()).hexdigest())
    manifest["holdout_used_for_tuning"] = False
    return manifest


def test_proposals_and_unnamed_authors_are_not_rule_gold():
    for author in ("", "tool (assistant)", "tool (bot)", "tool (model)"):
        assert rows_from_judgments([judgment(labeler=author)]) == []


def test_only_pinned_training_blobs_are_read(tmp_path):
    manifest = frozen(tmp_path, [judgment(), judgment(labeler="tool (assistant)")])
    out = tmp_path / "export.jsonl"
    assert export_frozen_training(manifest, out) == 1
    row = json.loads(out.read_text())
    assert row["deal_id"] == "train" and row["label"] == 1
    assert row["truth_source"] == "judgment"
    assert out.stat().st_mode & 0o777 == 0o600


def test_conflicts_and_changed_blobs_fail_before_replacing_artifact(tmp_path):
    out = tmp_path / "export.jsonl"
    out.write_text("previous artifact")
    manifest = frozen(tmp_path, [judgment(), judgment("should_not_fire")])
    with pytest.raises(ValueError, match="Conflicting"):
        export_frozen_training(manifest, out)
    assert out.read_text() == "previous artifact"
    manifest = frozen(tmp_path, [judgment()])
    manifest["train"][0]["blob_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="differs"):
        export_frozen_training(manifest, out)
    assert out.read_text() == "previous artifact"


def test_partition_leakage_and_unverified_deals_fail(tmp_path):
    manifest = frozen(tmp_path, [judgment()])
    manifest["holdout"][0]["source_groups"] = ["train"]
    with pytest.raises(ValueError, match="cross"):
        export_frozen_training(manifest, tmp_path / "out.jsonl")
    manifest = frozen(tmp_path, [judgment()])
    manifest["train"][0]["audit_status"] = "pending"
    with pytest.raises(ValueError, match="unverified"):
        export_frozen_training(manifest, tmp_path / "out.jsonl")


def test_eval_assignment_cannot_be_overridden_by_training_manifest(tmp_path):
    manifest = frozen(tmp_path, [{**judgment(), "purpose": "eval"}])
    with pytest.raises(ValueError, match="Evaluation-purpose"):
        export_frozen_training(manifest, tmp_path / "out.jsonl")


def test_export_is_read_by_existing_rule_trainer_without_any_training(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path

    out = tmp_path / "export.jsonl"
    export_frozen_training(frozen(tmp_path, [judgment()]), out)
    monkeypatch.setenv("SOWSMITH_RULE_LOG", str(out))
    script = Path(__file__).resolve().parents[1] / "_train_semantic_rules.py"
    spec = importlib.util.spec_from_file_location("synthetic_rule_reader", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._feedback_examples("example_rule") == [("Example candidate text.", 1)]
