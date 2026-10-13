"""Synthetic card subjects: shared evidence must not merge distinct questions."""
import pytest
import torch

from ml.c3.data import DealExample, IGNORE, featurize
from ml.c3.losses import judgment_loss, judgment_rationale_loss
from ml.c3.model import C3Config, C3Model
from ml.c3.readiness import audit_contract
from ml.c3.schema import load_schema

ATOMS = [{"label_key": "one", "atom_id": "one", "text": "Bring two ladders.",
          "doc_id": "synthetic", "entered_at": "2026-01-01T00:00:00Z"}]


def card(key, query, verdict, source=None, note=""):
    target = {"text": query}
    if source is not None:
        target["source"] = {"atomId": source}
    return {"head": "gap", "target_key": key, "target": target,
            "labeler": "expert@example.com", "verdict": verdict, "note": note}


def setup(cards):
    schema = load_schema()
    blob = {"judgments": cards}
    batch = featurize(DealExample.from_training_blob(blob, ATOMS, deal_id="synthetic"), schema)
    model = C3Model(schema, C3Config(d_text=32, d=32, n_layers=1, n_heads=2,
                      d_r=16, d_q=8, d_head=16, residual_dim=4, box_dim=4,
                      n_policy_atoms=2, policy_rank=2))
    return schema, blob, batch, model


def test_distinct_queries_share_evidence_without_losing_targets_or_notes():
    schema, blob, batch, model = setup([
        card("q:quantity", "How many ladders?", "invalid", "one", "Two are specified."),
        card("q:height", "What height must the ladders reach?", "valid", "one", "Reach is unspecified.")])
    assert len(batch.judged) == 2
    assert batch.targets["jdg:gap"] == [IGNORE]
    assert {j.note for j in batch.judged} == {"Two are specified.", "Reach is unspecified."}
    report = audit_contract(blob, ATOMS, schema, deal_id="synthetic")
    assert not any("target_conflict" in k for k in report["blockers"])
    out = model(batch.inputs())
    loss = judgment_loss(model, out, batch) + judgment_rationale_loss(model, out, batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.judgment_query.weight.grad.abs().sum() > 0
    assert model.judgment_note.weight.grad.abs().sum() > 0


def test_unanchored_query_reads_deal_but_stale_anchor_does_not():
    _, _, batch, _ = setup([card("q:open", "Who unlocks the door?", "valid"),
                           card("q:stale", "What time?", "valid", "missing")])
    assert [j.subject_key for j in batch.judged] == ["q:open"]
    assert batch.judged[0].lines == (0,)


def test_inference_features_do_not_take_verdicts_or_expert_notes():
    _, _, batch, model = setup([card("q:one", "How many ladders?", "invalid", "one", "Two are specified.")])
    model.eval()
    out = model(batch.inputs())
    original = model.judge(out, "jdg:gap", [(0,)], queries=["How many ladders?"])
    changed = model.judge(out, "jdg:gap", [(0,)], queries=["Who unlocks the door?"])
    assert not torch.allclose(original, changed)
    with pytest.raises(ValueError, match="One source query"):
        model.judge(out, "jdg:gap", [(0,)], queries=[])
    assert "why" not in batch.inputs() and "judged" not in batch.inputs()


def test_card_fields_keep_multiple_consumers_and_company_separation():
    j = card("q:height", "What height must the ladders reach?", "valid", "one")
    j["fields"] = {"deal_stage": "planning", "needed_by": ["project_manager", "delivery"],
                   "intake_gap": True}
    schema, blob, batch, model = setup([j])
    fields = {j.key: j for j in batch.judged}
    need = schema.by_key()["read:needed_by"]
    assert fields[need.key].answers == tuple(sorted([need.index("project_manager"), need.index("delivery")]))
    assert len(batch.judged) == 4
    out = model(batch.inputs())
    # Universal base can train without a company pass. Company targets skipped.
    assert torch.isfinite(judgment_loss(model, out, batch))
    with pytest.raises(ValueError, match="company question"):
        model.judge(out, need.key, [(0,)], queries=[j["target"]["text"]])
    out = model(batch.inputs(), company="synthetic-company")
    loss = judgment_loss(model, out, batch)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.judgment_head_company.parameters())
    report = audit_contract(blob, ATOMS, schema, deal_id="synthetic")
    assert "unconsumed_card_fields:gap" not in report["blockers"]


def test_unknown_card_field_fails_instead_of_disappearing():
    j = card("q:one", "Who unlocks the door?", "valid")
    j["fields"] = {"deal_stage": "invented-stage"}
    with pytest.raises(ValueError, match="Unknown question field"):
        setup([j])


def test_full_card_teacher_to_student_path_and_subject_guard():
    from ml.c3.brain import Brain
    from ml.c3.lm import TinyCausalLM
    from ml.c3.supercharge import teacher_view, supercharge_loss, weighted
    schema, _, batch, model = setup([
        card("q:quantity", "How many ladders?", "invalid", "one", "Two are specified."),
        card("q:height", "What height must the ladders reach?", "valid", "one", "Reach is unspecified.")])
    teacher = Brain(schema, TinyCausalLM(dim=32, layers=1, heads=2, max_len=2048))
    teacher.eval()
    direct = teacher.card_logits(batch)
    reasoned = teacher.card_logits(batch, with_why=True)
    assert direct["jdg:gap"].shape == (2, len(schema.by_key()["jdg:gap"].answers))
    assert not torch.allclose(direct["jdg:gap"], reasoned["jdg:gap"])
    reasoned["jdg:gap"].sum().backward()
    assert teacher.lm.tok.weight.grad.abs().sum() > 0
    view = teacher_view(teacher, batch, grounding=False)
    assert not view.card_logits["jdg:gap"].requires_grad
    out = model(batch.inputs())
    parts = supercharge_loss(model, out, view, batch=batch)
    assert torch.isfinite(parts["teach_judgments"])
    weighted(parts).backward()
    assert model.judgment_query.weight.grad.abs().sum() > 0
    from dataclasses import replace
    changed = replace(batch, judged=[replace(batch.judged[0], note="Changed expert annotation."), *batch.judged[1:]])
    with pytest.raises(ValueError, match="Teacher card subjects"):
        supercharge_loss(model, out, view, batch=changed)
    view.card_subjects["jdg:gap"] = ()
    with pytest.raises(ValueError, match="Teacher card subjects"):
        supercharge_loss(model, out, view, batch=batch)


def test_teacher_card_policy_is_only_in_company_pages():
    from dataclasses import replace
    from ml.c3.brain import Brain
    from ml.c3.lm import TinyCausalLM
    schema, _, batch, _ = setup([card("q:quantity", "How many ladders?", "invalid", "one", "Two are specified.")])
    teacher = Brain(schema, TinyCausalLM(dim=32, layers=1, heads=2, max_len=2048))
    batch = replace(batch, company_policy="PRIVATE POLICY SENTINEL")
    universal = replace(batch.judged[0], policy="PRIVATE NOTE SENTINEL")
    assert "PRIVATE" not in teacher.card_page(batch, universal, with_why=True)
    company = replace(universal, key="read:intake_gap")
    page = teacher.card_page(batch, company, with_why=True)
    assert "PRIVATE POLICY SENTINEL" in page and "PRIVATE NOTE SENTINEL" in page


def test_legacy_clause_path_is_opt_in_and_invalid_boolean_is_not_absence():
    from ml.c3.losses import LossWeights
    assert LossWeights().clause_use == 0
    j = card("q:one", "Who unlocks the door?", "valid")
    j["fields"] = {"intake_gap": "not-a-boolean"}
    with pytest.raises(ValueError, match="Unknown question field"):
        setup([j])


def test_all_teacher_tokenization_and_token_readers_reject_truncation():
    from ml.c3.lm import TinyCausalLM
    from ml.c3.text import HashingEncoder
    lm = TinyCausalLM(dim=16, layers=1, heads=2, max_len=12)
    with pytest.raises(ValueError, match="Teacher text exceeds"):
        lm.tokenize(["A full expert paragraph that cannot fit."])
    ids, mask = lm.tokenize(["Short."])
    assert int(mask.sum()) == 1 + len("Short.".encode())
    with pytest.raises(ValueError, match="Encoder paragraph exceeds"):
        HashingEncoder(16).tokens(["one two three four five"], max_len=4)
