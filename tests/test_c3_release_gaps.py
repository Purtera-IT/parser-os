"""Consumers, long-deal edges, and the refusal to train. Not learned accuracy."""
import pytest

from ml.c3.chunks import split_deal
from ml.c3.coverage import READY_FOR_TRAINING, coverage_table
from ml.c3.data import Atom, DealExample, featurize
from ml.c3.gates import fit_temperature, train
from ml.c3.schema import load_schema


def test_coverage_does_not_call_a_blocked_surface_consumed():
    rows = coverage_table()
    assert rows
    for row in rows:
        if row["status"] == "blocked":
            assert row["loss"] == "none" or "READY_FOR_TRAINING" in row["loss"] or row["target"] == "release.training_blockers" or "none" in row["loss"]
    assert any(row["surface"] == "router" and row["status"] == "blocked" for row in rows)
    assert READY_FOR_TRAINING is False


def test_train_refuses_before_any_step():
    with pytest.raises(RuntimeError, match="READY_FOR_TRAINING is false"):
        train()


def test_hours_and_deal_service_become_targets_and_eval_does_not():
    schema = load_schema()
    blob = {
        "labels": [{"label_key": "a", "labeler": "expert@example.com", "label_type": "task",
                    "text": "Install 3 cameras in 16 hours."}],
        "judgments": [{"head": "hours", "target_key": "atom:at-a", "labeler": "expert@example.com",
                       "verdict": "hours=16;per=camera;qty=3;basis=stated"}],
        "deal_answers": [
            {"labeler": "expert@example.com", "primary_service": "wireless",
             "declared_site_count": 4, "note": "Four named sites, one wireless job."},
            {"labeler": "expert@example.com", "purpose": "eval", "primary_service": "itad",
             "declared_site_count": 9},
        ],
    }
    atoms = [{"label_key": "a", "atom_id": "at-a", "text": "Install 3 cameras in 16 hours."}]
    deal = DealExample.from_training_blob(blob, atoms, deal_id="synthetic")
    batch = featurize(deal, schema)
    assert batch.numbers_target["read:labor_hours"][0] == 16
    services = [j for j in batch.judged if j.key == "deal:primary_service"]
    assert len(services) == 1
    assert schema.by_key()["deal:primary_service"].answers[services[0].answer].value == "wireless"
    assert batch.deal_numbers["declared_site_count"] == 4
    assert any(item.key == "deal:note" and "itad" not in item.text for item in batch.literals)


def test_cross_chunk_edge_is_kept():
    deal = DealExample(
        deal_id="synthetic",
        atoms=[Atom(key="a", text="Question about the ladder."),
               Atom(key="b", text="Bring two ladders.")],
        edges=[("b", "a", "answers")])
    chunks = split_deal(deal, max_atoms=1)
    assert [edge for chunk in chunks for edge in chunk.cross_edges] == [("b", "a", "answers")]
    assert all(len(chunk.deal.atoms) == 1 for chunk in chunks)


def test_temperature_refuses_the_holdout():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    labels = torch.tensor([0, 1])
    with pytest.raises(ValueError, match="validation"):
        fit_temperature(logits, labels, split="holdout")
    assert fit_temperature(logits, labels, split="validation") > 0
