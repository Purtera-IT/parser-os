"""The locked refit (head_fit.Lock / locked_fit): starts exactly at the served model on every line, trains only its
target rows, and its accept report sees nothing else move."""
import importlib.util
from pathlib import Path

import pytest
import torch

from ml.c3 import free_rows, head_fit
from ml.c3.data import IGNORE, DealExample, featurize
from ml.c3.gates import save_checkpoint
from ml.c3.model import C3Model
from ml.c3.schema import load_schema

KEY = 'col:label_type'


@pytest.fixture
def case():
    spec = importlib.util.spec_from_file_location('why2_fixture', Path(__file__).with_name('test_c3_why2.py'))
    f = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(f)
    schema = load_schema()
    b = featurize(DealExample.from_training_blob({'labels': f.LABELS, 'judgments': f.JUDGMENTS},
                  f.ATOMS, deal_id='synthetic-locked', serve_lines_only=False), schema)
    torch.manual_seed(0)
    return C3Model(schema, f.ASKED).eval(), b


def _rows(model, b, keep):
    row = model.add_free_head(KEY)
    with torch.no_grad():
        row.weight.normal_(std=0.1)
        row.bias.normal_(std=0.1)
    if keep is not None:
        model.add_free_head('r:' + KEY)
        free_rows.set_h_keep(model, keep)
    rows, texts = head_fit.line_rows(model, [b], [KEY], 'cpu')
    return rows[KEY], texts[KEY]


@pytest.mark.parametrize('keep', [None, 0.0])
@pytest.mark.parametrize('rank', [0, 2])
def test_every_line_starts_at_the_served_logits(case, keep, rank):
    model, b = case
    c, texts = _rows(model, b, keep)
    assert len(c['y']) == len(b) == len(texts)          # every line, labeled or not
    assert c['y'].tolist() == list(b.targets[KEY])
    lock = head_fit.Lock(model, [KEY], rank=rank)
    with torch.no_grad():
        want = model(b.oneshot()).logits[KEY]
        torch.testing.assert_close(lock.logits(model, KEY, c), want, rtol=0, atol=5e-3)   # float16 features


@pytest.mark.parametrize('rank', [0, 2])
def test_only_the_target_row_moves_and_accept_sees_it(case, rank, tmp_path):
    model, b = case
    c, _ = _rows(model, b, None)
    other = 'col:admission'
    model.add_free_head(other)
    before = {n: t.detach().clone() for n, t in model.state_dict().items()}
    save_checkpoint(tmp_path / 'base.pt', model, step=0, schema_sha256='x')
    lab = torch.arange(len(c['y'])) % 2 == 0    # half the lines are 'new labels', the rest unlabeled
    new = {KEY: {n: t[lab] for n, t in c.items()}}
    with torch.no_grad():   # ask every new line for an answer the base does not give
        new[KEY]['y'] = (head_fit.Lock(model, [KEY]).logits(model, KEY, new[KEY]).argmax(-1) + 1) % c['base'].shape[1]
    cons = {KEY: {n: t[~lab] for n, t in c.items()}}
    va = {KEY: {n: t[lab] for n, t in c.items()}}
    lock = head_fit.Lock(model, [KEY], rank=rank)
    best, state, log = head_fit.locked_fit(model, lock, new, cons, {}, va, new_weight=30, cons_weight=1.0,
                                           epochs=6, steps=5, batch=64, lr=5e-2, check_every=2,
                                           max_change=1.0, max_flip=1.0, tolerance=1.0)
    assert log[-1]['newfit'][KEY] > 0.5
    for n, t in model.state_dict().items():          # training never wrote into the model
        assert torch.equal(t, before[n]), n
    lock.commit(model, state)
    save_checkpoint(tmp_path / 'cand.pt', model, step=0, schema_sha256='x')
    changed = head_fit.changed_heads(tmp_path / 'base.pt', tmp_path / 'cand.pt')
    assert set(changed) == {KEY}
    for n, t in model.state_dict().items():
        if not n.startswith(f'free_heads.{KEY}.'):
            assert torch.equal(t, before[n]), n
    names = {KEY: [a.value for a in model.schema.by_key()[KEY].answers]}
    rep = head_fit.flip_report(model, lock, {KEY: c}, {KEY: [''] * len(c['y'])},
                               {KEY: torch.zeros(len(c['y']), dtype=torch.long)}, ['synthetic.json'], names,
                               state=state)
    assert rep[KEY]['lines_read'] == len(c['y']) and rep[KEY]['flipped'] >= int(lab.sum()) // 2


def test_consistency_holds_unlabeled_lines(case):
    model, b = case
    c, _ = _rows(model, b, None)
    lab = torch.arange(len(c['y'])) % 2 == 0    # half the lines are 'new labels', the rest unlabeled
    new = {KEY: {n: t[lab] for n, t in c.items()}}
    with torch.no_grad():
        new[KEY]['y'] = (head_fit.Lock(model, [KEY]).logits(model, KEY, new[KEY]).argmax(-1) + 1) % c['base'].shape[1]
    cons = {KEY: {n: t[~lab] for n, t in c.items()}}
    flips = {}
    for w, hold in ((0.0, 0.0), (50.0, 0.0), (50.0, 50.0)):
        lock = head_fit.Lock(model, [KEY], rank=0)
        _, _, log = head_fit.locked_fit(model, lock, new, cons, {}, {}, new_weight=1, cons_weight=w,
                                        hold_weight=hold, epochs=4, steps=5, batch=256, lr=5e-2, check_every=4,
                                        max_flip=1.0)
        flips[w, hold] = log[-1]['cons_flip'][KEY]
    assert flips[50.0, 50.0] <= flips[50.0, 0.0] <= flips[0.0, 0.0]


def test_margin():
    lp = torch.tensor([[0.0, -1.0, -3.0], [-2.0, -0.5, -0.1]])
    torch.testing.assert_close(head_fit._margin(lp, torch.tensor([0, 1])), torch.tensor([1.0, -0.4]))
