"""Balance coefficients must survive both normalizations, without changing background supervision."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from ml.c3.balance import cap_scales, head_cap_scales
from ml.c3.data import IGNORE, DealExample, featurize
from ml.c3.loop import _student_step, training_batches, training_scales
from ml.c3.losses import LossWeights, head_losses, weighted_ce
from ml.c3.schema import CLASS, load_schema


class BatchStub(NS):
    def __len__(self):
        return len(self.weights)


def schema_stub():
    ops = [NS(key=k, kind=CLASS, space='x', layer='universal') for k in ('fd', 'other')]
    return NS(select=lambda **kw: ops, by_key=lambda: {o.key: o for o in ops})


def batch_stub(deal='a', scale=1., multiple=False, other=True):
    return BatchStub(deal_id=deal, deal_loss_scale=1., weights=[1., .1], head_scale={'fd': scale},
                     head_loss_scale={}, targets={'fd': [0, 0], 'other': [0, 0] if other else [IGNORE]*2},
                     multi_targets={'fd': [(0,), (1,)]} if multiple else {})


def head_grads(batch, *, weights=None):
    logits = {k: torch.tensor([[.1, -.1], [-.2, .2]], requires_grad=True) for k in ('fd', 'other')}
    out = NS(r=logits['fd'], logits=logits, numbers={})
    loss, _ = head_losses(NS(schema=schema_stub()), out, batch, weights or LossWeights(), 'universal')
    grads = torch.autograd.grad(loss, list(logits.values()), allow_unused=True)
    return dict(zip(logits, grads))


@pytest.mark.parametrize('multiple', [False, True])
@pytest.mark.parametrize('other', [False, True])
def test_fractional_head_scale_is_absolute_and_does_not_reallocate_other_heads(multiple, other):
    full = head_grads(batch_stub(multiple=multiple, other=other))
    scaled = head_grads(batch_stub(scale=.2, multiple=multiple, other=other))
    assert torch.allclose(scaled['fd'], full['fd']*.2)
    if other:
        assert torch.equal(scaled['other'], full['other'])


def test_runtime_head_cap_is_independent_of_view_and_head_weight():
    batch = batch_stub(scale=.5)
    weights = LossWeights(head={'fd': 20.})
    full = head_grads(batch, weights=weights)
    batch.head_loss_scale = {'fd': .2}
    small = head_grads(batch, weights=weights)
    assert torch.allclose(small['fd'], full['fd']*.2)
    assert torch.equal(small['other'], full['other'])
    batch.head_scale['fd'] = 0
    assert head_grads(batch, weights=weights)['fd'] is None


def test_part_multiplier_reaches_backward_but_background_stays_normalized(monkeypatch):
    import ml.c3.loop as loop

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.tensor([[.1, -.1], [-.2, .2]]))
        def ema_update(self):
            pass

    def loss(model, batch, w, **kw):
        value = .1*weighted_ce(model.logits, torch.tensor([0, 0]), torch.tensor(batch.weights))
        return value, {'heads': float(value.detach())}
    monkeypatch.setattr(loop, 'c3_loss', loss)
    batch = batch_stub()
    def read(scale):
        model = Model()
        opt = torch.optim.SGD(model.parameters(), lr=0.)
        _student_step(model, opt, batch, LossWeights(), None, None, None, 'cpu', scale=scale)
        return model.logits.grad.clone()
    full = read(1.)
    batch.deal_loss_scale = .2
    scale = training_scales([batch])[batch.deal_id]
    assert torch.allclose(read(scale), full*.2)
    # A uniform change of row weights cancels (the intended background mean).
    batch.weights = [.2, .02]
    assert torch.allclose(read(1.), full)
    batch.weights = [1., 1.]
    assert not torch.allclose(read(1.), full)


def test_explicit_scale_survives_load_featurize_windows_and_is_never_an_input():
    schema = load_schema()
    path = Path(__file__).resolve().parents[1]/'ml/c3/fixtures/synthetic_deal.json'
    deal = replace(DealExample.load(path), deal_loss_scale=.25)
    # Raw training-blob ingestion must not drop the new metadata.
    blob = {'deal_loss_scale': .25, 'atoms': [], 'labels': []}
    loaded = DealExample.from_training_blob(blob, [], deal_id='synthetic-empty')
    assert loaded.deal_loss_scale == .25
    batch = featurize(deal, schema)
    assert batch.deal_loss_scale == .25
    assert 'deal_loss_scale' not in batch.inputs() and 'head_loss_scale' not in batch.inputs()
    parts, _ = training_batches(deal, schema, max_atoms=8)
    assert all(p.deal_loss_scale == .25 for p in parts)
    scales = training_scales(parts)
    assert sum(scales.values()) == pytest.approx(.25*len(parts)**.5)
    assert all(v == .25 for v in training_scales(parts, balance=False).values())


def test_caps_use_actual_parts_and_group_views_and_new_sources():
    parts = [batch_stub(f'new@pre_quote#c3part{i}') for i in range(50)]
    parts += [batch_stub(f'new@planning#c3part{i}') for i in range(50)]
    parts += [batch_stub(f'other{i}@pre_quote') for i in range(9)]
    scale = training_scales(parts, cap=.15)
    total = sum(scale.values())
    assert sum(scale[b.deal_id] for b in parts if b.deal_id.startswith('new@'))/total == pytest.approx(.15)
    assert all(v>0 for v in scale.values())
    parts[0].deal_loss_scale=.2
    with pytest.raises(ValueError, match='Inconsistent'):
        training_scales(parts)


def test_head_cap_uses_real_head_divisors_and_preserves_background_gradients():
    parts=[batch_stub(f'new@planning#c3part{i}', other=False, multiple=True) for i in range(20)]
    parts += [batch_stub(f'old{i}', multiple=True) for i in range(9)]
    scales=training_scales(parts)
    factors=head_cap_scales(parts, schema_stub(), scales, {'fd': .15})
    before=head_grads(parts[0])
    for b in parts:
        b.head_loss_scale=factors[b.deal_id]
    after=head_grads(parts[0])
    assert torch.allclose(after['fd'], before['fd']*factors[parts[0].deal_id]['fd'])
    # Every stub has the same fd labels/logits/background, so scalar mass is
    # also directly observable in the sum of gradients on a shared logit.
    grads=[]
    dominant=[]
    for b in parts:
        g=head_grads(b)['fd']*scales[b.deal_id]
        grads.append(g)
        if b.deal_id.startswith('new@'):
            dominant.append(g)
    assert torch.allclose(sum(dominant), .15*sum(grads), atol=1e-7)


@pytest.mark.parametrize('count', [0, 1, 6])
def test_sparse_cap_reports_infeasible_and_never_zeros_learning(count):
    mass={str(i): i+1. for i in range(count)}
    with pytest.warns(RuntimeWarning, match='infeasible'):
        assert cap_scales(mass, .15) == dict.fromkeys(mass, 1.)


def test_feasible_boundary_and_zero_mass_do_not_erase_learning():
    mass={'a': 100., 'b': 1., 'unlabeled': 0.}
    scale=cap_scales(mass, .5)
    assert scale['unlabeled'] == 1.
    assert all(v>0 for v in scale.values())
    assert mass['a']*scale['a'] == pytest.approx(mass['b']*scale['b'])


@pytest.mark.parametrize('cap', [0., -1., float('nan'), float('inf'), 1.1])
def test_invalid_caps_fail(cap):
    with pytest.raises(ValueError): cap_scales({'a': 1.}, cap)


@pytest.mark.parametrize('value', [0., -1., float('nan'), float('inf'), 1.1])
def test_invalid_explicit_scale_fails(value):
    with pytest.raises(ValueError): training_scales([batch_stub(scale=1., deal='a')], cap=value)
    b=batch_stub()
    b.deal_loss_scale=value
    with pytest.raises(ValueError): training_scales([b])
