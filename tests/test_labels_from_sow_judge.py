"""Rubric-judge labels are model drafts: their own source, slight tier, never over gold."""
import json

from ml.c3.labels_from_sow import JUDGE_CREATED_BY, build, judge_reviews, rubric_reviews


def _env(tmp_path):
    env = {"crm": {"deal_name": "010999 - test"},
           "documents": [{"artifact_id": "d1", "artifact_type": "eml"}],
           "atoms": [{"id": "a1", "text": "Mount the TV", "artifact_id": "d1"},
                     {"id": "a2", "text": "Customer ships the TVs", "artifact_id": "d1"},
                     {"id": "a3", "text": "Thanks all", "artifact_id": "d1"}],
           "c3_sow": {}}
    p = tmp_path / "final_010999_env.json"
    p.write_text(json.dumps(env))
    return str(p)


def test_judge_labels_are_slight_and_lose_to_rubric(tmp_path):
    env = _env(tmp_path)
    (tmp_path / "b.jsonl").write_text(json.dumps({"deal": "010999", "id": "a1", "text": "Mount the TV"}) + "\n")
    (tmp_path / "b.A.jsonl").write_text(json.dumps({"id": "a1", "section": "scope_of_work", "owner": "technician"}) + "\n")
    (tmp_path / "j.J.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"id": "a1", "deal": "010999", "section": "assumptions", "owner": "none", "model": "ollama:qwen3:32b"},
        {"id": "a2", "deal": "010999", "section": "customer_responsibilities", "owner": "customer",
         "model": "ollama:qwen3:32b"},
        {"id": "a3", "deal": "010999", "section": "_absent", "owner": "none", "model": "ollama:qwen3:32b"},
    ]) + "\n")
    # judge listed FIRST: precedence must not depend on order
    reviews = judge_reviews([str(tmp_path / "j.J.jsonl")]) + rubric_reviews([str(tmp_path / "b.A.jsonl")])
    counts = build([env], reviews, str(tmp_path / "out"))
    rec = json.loads((tmp_path / "out" / "010999.json").read_text())
    by = {lb["atom_id"]: lb for lb in rec["labels"]}
    assert by["a1"]["reads_set"]["sow_section"] == "scope_of_work"  # rubric gold wins
    assert by["a1"]["weight_tier"] == "ordinary" and by["a1"]["labeler"] == "rubric"
    assert by["a2"]["reads_set"]["sow_section"] == "customer_responsibilities"
    assert by["a2"]["weight_tier"] == "slight"
    assert by["a2"]["created_by"] == JUDGE_CREATED_BY
    assert by["a2"]["labeler"] == "judge:ollama:qwen3:32b"
    assert by["a3"]["reads_set"]["sow_section"] == "_absent"
    assert counts["judge"] >= 2 and counts["gold"] >= 1


def test_supplier_from_owner_only_on_work_and_supply_lines():
    from ml.c3.labels_from_sow import supplier_reviews
    rv = [{"deal": "d", "atom_id": "a", "head": "sow_section", "value": "customer_responsibilities", "verdict": "accepted", "by": "rubric"},
          {"deal": "d", "atom_id": "a", "head": "sow_party", "value": "customer", "verdict": "accepted", "by": "rubric"},
          {"deal": "d", "atom_id": "b", "head": "sow_section", "value": "out_of_scope", "verdict": "accepted", "by": "rubric"},
          {"deal": "d", "atom_id": "b", "head": "sow_party", "value": "technician", "verdict": "accepted", "by": "rubric"},
          {"deal": "d", "atom_id": "c", "head": "sow_section", "value": "scope_of_work", "verdict": "judge", "by": "judge:m"},
          {"deal": "d", "atom_id": "c", "head": "sow_party", "value": "technician", "verdict": "judge", "by": "judge:m"}]
    out = {(r["atom_id"]): (r["value"], r["verdict"]) for r in supplier_reviews(rv)}
    assert out == {"a": ("customer", "accepted"), "c": ("us", "judge")}
