"""entity_swap: gate 2's entity edits as a training lock (ml/c3/entity_swap.py, losses.entity_swap_loss)."""
from __future__ import annotations

import random
import re
from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import ml.c3.losses as losses  # noqa: E402
from ml.c3 import entity_swap as es  # noqa: E402
from ml.c3.data import DealExample, featurize, number_tokens  # noqa: E402
from ml.c3.loop import SMALL  # noqa: E402
from ml.c3.losses import LossWeights, c3_loss, entity_swap_loss  # noqa: E402
from ml.c3.model import C3Model  # noqa: E402
from ml.c3.schema import load_schema  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "ml" / "c3" / "fixtures" / "synthetic_deal.json"

LINES = [
    "Site Contact: John Smith (john.smith@acmecorp.com) 555-1212",
    "Install 12 cameras at Dallas store #4521 for Acme Corp, 3 hrs each, $1,250.00",
    "123 Main St | Dallas, TX 77002 | DALLAS NORTH | Qty 4",
    "Ticket INC0012345 opened by Purtera; email ops@purtera.com",
    "John Smith will confirm with Acme Corp on 10/14/2025",
    "PURCHASER BILLING INFO | DELIVER TO | INSTALLATION",
    "EXCEPT AS PROVIDED IN THIS SECTION 20, IN NO EVENT WILL LIABILITY EXCEED $5,000",
    "Ordered Amount: $1,622.00 | Line Description: Checkout LLC-2026.07-Smart Hands",
    "They need some professional services; see Professional Services Inc for the scope.",
]


def _swap(lines=LINES, seed=0, company="purtera"):
    return es.swap_texts(list(lines), company, random.Random(seed))


def test_the_swap_keeps_numbers_quantities_and_type_cues():
    out = _swap()
    no_mail = lambda t: es.EMAIL.sub(" ", t)  # noqa: E731
    for a, b in zip(LINES, out):
        # Magnitude tokens (a model input) never move; IDs and ZIPs keep length and leading zeros.
        assert number_tokens(no_mail(a)) == number_tokens(no_mail(b)), (a, b)
        # Every lowercase word stays: the line's type cues are prose, not names.
        assert set(re.findall(r"\b[a-z]{3,}\b", no_mail(a))) <= set(re.findall(r"\b[a-z]{3,}\b", no_mail(b))), (a, b)
    for cue in ("Site Contact:", "Install 12 cameras", "3 hrs each, $1,250.00", "Qty 4", "123 Main St",
                "on 10/14/2025", "IN NO EVENT", "SECTION 20", "$5,000", "LLC-2026.07", "Ordered Amount: $1,622.00"):
        assert any(cue in b for b in out), cue
    assert out[5] == LINES[5]                                  # dictionary words in caps cells stay
    assert "professional services" in out[8]                   # a common-word "brand" is not a brand
    # Ours never moves.
    assert "Purtera" in out[3] and "ops@purtera.com" in out[3]
    # Names, emails, places, IDs and brands do.
    joined = "\n".join(out)
    for gone in ("John", "john.smith", "acmecorp", "Dallas", "DALLAS NORTH", "TX 77002", "#4521",
                 "INC0012345", "Acme"):
        assert gone not in joined, gone


def test_one_entity_one_stand_in_across_the_part_and_never_the_gates_pools():
    out = _swap()
    first = re.match(r"Site Contact: (\w+) (\w+)", out[0]).groups()
    assert out[4].startswith(f"{first[0]} {first[1]} will confirm")
    city = re.search(r"at (.+?) store", out[1]).group(1)
    assert f"| {city}, " in out[2]
    brand = re.search(r"for (\w+) Corp", out[1]).group(1)
    assert f"with {brand} Corp" in out[4]
    # Same seed, same swap; another seed, other stand-ins (the heads cannot learn them).
    assert _swap(seed=0) == out and _swap(seed=1) != out
    # Disjoint from gate 2's substitutes, so the gate stays held out.
    names, _ = es._lexicon()
    assert not set(es.TRAIN_FIRST) & set(names)
    assert not set(es.TRAIN_LAST) & set(es.GATE_LAST)
    assert not set(es.TRAIN_CITIES) & set(es.GATE_CITIES)
    assert not set(es.TRAIN_STATES) & set(es.GATE_STATES)
    assert not set(es.TRAIN_COMPANIES) & set(es.GATE_COMPANIES)


def test_a_deal_without_names_reads_unchanged():
    b = featurize(DealExample.load(FIXTURE), load_schema())
    assert es.swap_texts(list(b.texts), b.company, random.Random(0)) == list(b.texts)


def _batch():
    schema = load_schema()
    b = featurize(DealExample.load(FIXTURE), schema)
    texts = list(b.texts)
    texts[0] = "Hi team, John Smith from Acme Corp here, thanks for the call!"
    texts[12] = "Bayview, 410 Harbor Rd, Dallas, TX 75201 - store #4521"
    return schema, replace(b, texts=texts, numbers=[number_tokens(t) for t in texts])


class _Invariant:
    """A model whose reading ignores the names: every forward is the first one."""

    def __init__(self, model, out):
        self.schema, self.out = model.schema, out

    def __call__(self, *a, **kw):
        return self.out


class _ReadsNames(_Invariant):
    """A model that answers differently on a renamed line: the first reading, its classes rolled there."""

    def __init__(self, model, out, texts):
        super().__init__(model, out)
        self.texts = list(texts)

    def __call__(self, inputs, **kw):
        import copy

        rows = torch.tensor([a != b for a, b in zip(self.texts, inputs["texts"])])
        moved = copy.copy(self.out)
        moved.logits = {k: torch.where(rows[:, None], v.roll(1, -1) * 4, v) if v.dim() == 2 else v
                        for k, v in self.out.logits.items()}
        return moved


def test_the_hold_is_zero_when_the_model_is_invariant_and_positive_when_it_is_not(monkeypatch):
    schema, batch = _batch()
    torch.manual_seed(0)
    model = C3Model(schema, SMALL).eval()
    w = LossWeights()
    with torch.no_grad():
        desc = model.describe()
        out = model(batch.inputs(), company=batch.company, desc=desc)
    # Only the hold (KL to the first reading): the saved-answer CE is the heads' own loss, on other texts.
    monkeypatch.setattr(losses, "head_losses", lambda *a, **k: (out.r.new_zeros(()), out.r.new_zeros(())))
    with torch.no_grad():
        same = entity_swap_loss(_Invariant(model, out), out, batch, w, rng=random.Random(0), desc=desc)
        moved = entity_swap_loss(model, out, batch, w, rng=random.Random(0), desc=desc)
        reads = entity_swap_loss(_ReadsNames(model, out, batch.texts), out, batch, w, rng=random.Random(0),
                                 desc=desc)
    assert float(same) == 0.0
    assert float(reads) > 1e-3   # 2 of 14 lines renamed; the hold averages over every labeled line
    assert float(moved) > 0.0   # a random SMALL model barely reads names (~1e-7 here), but it does
    # Nothing to rename: no forward, no loss.
    monkeypatch.setattr(es, "swap_texts", lambda texts, company, rng: list(texts))
    calls = []
    monkeypatch.setattr(model, "forward", lambda *a, **k: calls.append(1))
    assert float(entity_swap_loss(model, out, batch, w, rng=random.Random(0), desc=desc)) == 0.0 and not calls


def test_entity_swap_reuses_the_steps_descriptions(monkeypatch):
    schema, batch = _batch()
    model = C3Model(schema, SMALL)
    calls = {"n": 0}
    real = model.describe

    def counted():
        calls["n"] += 1
        return real()

    monkeypatch.setattr(model, "describe", counted)
    _, parts = c3_loss(model, batch, LossWeights(entity_swap=0.3, keyword_dropout=0.0))
    assert "entity_swap" in parts and parts["entity_swap"] > 0
    assert calls["n"] == 1


def _step(schema, batch, w, state):
    model = C3Model(schema, SMALL)
    model.load_state_dict(state)
    torch.manual_seed(7)
    total, parts = c3_loss(model, batch, w)
    total.backward()
    grads = {k: p.grad.detach().clone() for k, p in model.named_parameters() if p.grad is not None}
    return float(total.detach()), parts, grads, torch.get_rng_state()


def test_weight_zero_is_a_no_op(monkeypatch):
    schema, batch = _batch()
    torch.manual_seed(0)
    state = {k: v.clone() for k, v in C3Model(schema, SMALL).state_dict().items()}
    off = LossWeights(entity_swap=0.0)
    t0, p0, g0, r0 = _step(schema, batch, off, state)

    def never(*a, **k):
        raise AssertionError("entity_swap ran at weight 0")

    # The code path before this term existed: the function is never reached, no draw from torch's generator.
    monkeypatch.setattr(losses, "entity_swap_loss", never)
    t1, p1, g1, r1 = _step(schema, batch, off, state)
    assert "entity_swap" not in p0 and t0 == t1 and p0 == p1 and torch.equal(r0, r1)
    assert g0.keys() == g1.keys() and all(torch.equal(g0[k], g1[k]) for k in g0)
    monkeypatch.undo()
    t2, p2, _, _ = _step(schema, batch, replace(off, entity_swap=0.3), state)
    assert "entity_swap" in p2 and t2 != t0


def test_runE_sets_entity_swap_ramps_it_and_counts_its_graph():
    text = (ROOT / "ml/c3/pod/runE.sh").read_text(encoding="utf-8")
    assert "ENTITY_SWAP=${ENTITY_SWAP:-0.3}" in text
    assert 'entity_swap="$ENTITY_SWAP"' in text
    assert re.search(r"for t in [^\n]*\bentity_swap\b", text)
    assert '[[ "$ENTITY_SWAP" != 0 ]] && graphs=$((graphs + 1))' in text


def test_aux_rotate_alternates_swap_and_flip_at_twice_the_weight(monkeypatch):
    # C3_AUX_ROTATE=1: one edited copy per step, entity_swap then meaning_flip, each at twice its weight.
    schema, batch = _batch()
    torch.manual_seed(0)
    state = {k: v.clone() for k, v in C3Model(schema, SMALL).state_dict().items()}
    w = LossWeights(entity_swap=0.3, meaning_flip=0.3)
    calls = []
    monkeypatch.setattr(losses, "entity_swap_loss", lambda *a, **k: calls.append("swap") or torch.tensor(1.0))
    monkeypatch.setattr(losses, "meaning_flip_loss", lambda *a, **k: calls.append("flip") or torch.tensor(1.0))
    monkeypatch.setattr(losses, "_AUX_ROTATE", True)
    monkeypatch.setattr(losses, "_AUX_TURN", __import__("itertools").count())
    model = C3Model(schema, SMALL)
    model.load_state_dict(state)
    _, p1 = c3_loss(model, batch, w)
    _, p2 = c3_loss(model, batch, w)
    assert calls == ["swap", "flip"]
    assert p1["entity_swap"] == 2.0 and "meaning_flip" not in p1
    assert p2["meaning_flip"] == 2.0 and "entity_swap" not in p2
    monkeypatch.setattr(losses, "_AUX_ROTATE", False)
    calls.clear()
    _, p3 = c3_loss(model, batch, w)
    assert calls == ["swap", "flip"] and p3["entity_swap"] == 1.0 and p3["meaning_flip"] == 1.0


def test_answer_weights_scale_rows_of_the_listed_answer(monkeypatch):
    # C3_ANSWER_WEIGHTS: a listed answer's rows count that many times in its line head; other rows and heads unchanged.
    schema, batch = _batch()
    opp = schema.by_key()["col:label_type"]
    y = [v for v in batch.targets["col:label_type"] if v != losses.IGNORE]
    common = max(set(y), key=y.count)
    monkeypatch.setattr(losses, "_ANSWER_WEIGHTS", {"col:label_type": {opp.answers[common].value: 3.0}})
    model = C3Model(schema, SMALL)
    out = model(batch.inputs(), company=batch.company)
    rw = torch.ones(len(batch))
    scaled = losses._rows_by_answer(opp, out, batch, rw)
    for i, t in enumerate(batch.targets["col:label_type"]):
        assert float(scaled[i]) == (3.0 if t == common else 1.0)
    assert torch.equal(losses._rows_by_answer(schema.by_key()["col:supplier"], out, batch, rw), rw)
