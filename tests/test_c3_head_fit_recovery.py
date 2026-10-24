"""Cached refits must use exactly the served logits, including r-only rows."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import pytest
import torch
from ml.c3 import free_rows, head_fit
from ml.c3.data import DealExample, featurize
from ml.c3.model import C3Model
from ml.c3.schema import load_schema

@pytest.fixture
def case():
    spec = importlib.util.spec_from_file_location('why2_fixture', Path(__file__).with_name('test_c3_why2.py'))
    f = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(f)
    schema = load_schema()
    b = featurize(DealExample.from_training_blob({'labels': f.LABELS, 'judgments': f.JUDGMENTS},
        f.ATOMS, deal_id='synthetic-refit', serve_lines_only=False), schema)
    torch.manual_seed(0)
    return C3Model(schema, f.ASKED).eval(), b

@pytest.mark.parametrize('keep', [None, 1.0, 0.5, 0.0])
def test_cache_logits_match_inference_before_and_after_update_and_resume(case, keep, tmp_path, monkeypatch):
    model, b = case
    key = 'col:label_type'
    row = model.add_free_head(key)
    with torch.no_grad():
        row.weight.normal_()
        row.bias.normal_()
    if keep is not None:
        row = model.add_free_head('r:' + key)
        free_rows.set_h_keep(model, keep)
    cache = head_fit._cache(model, [b], [key], 'cpu', tmp_path)[key]
    idx = torch.tensor([j for j,y in enumerate(b.targets[key]) if y != -100])
    from ml.c3.data import IGNORE
    idx = torch.tensor([j for j,y in enumerate(b.targets[key]) if y != IGNORE])
    for _ in range(2):
        with torch.no_grad():
            expected = model(b.oneshot()).logits[key][idx]
            got = head_fit._logits(model, key, cache)
            torch.testing.assert_close(got, expected, rtol=0, atol=2e-6)
            row.weight.add_(0.01)
    def refuse_forward(*args, **kwargs):
        raise AssertionError('resumed part should not run the model')
    monkeypatch.setattr(model, 'forward', refuse_forward)
    resumed = head_fit._cache(model, [b], [key], 'cpu', tmp_path)[key]
    for name in cache:
        torch.testing.assert_close(cache[name], resumed[name], rtol=0, atol=0)

def test_optimizer_resume_is_identical_to_uninterrupted(case, tmp_path, monkeypatch):
    import copy
    model, b = case
    key = 'col:label_type'
    model.add_free_head('r:' + key)
    free_rows.set_h_keep(model, 0)
    cache = head_fit._cache(model, [b], [key], 'cpu')
    uninterrupted = copy.deepcopy(model)
    kwargs = dict(epochs=2, check_every=1, batch=2)
    torch.manual_seed(17)
    head_fit.fit_heads(uninterrupted, cache, cache, [key], **kwargs)
    original = head_fit._atomic_save
    def interrupt(value, path):
        original(value, path)
        raise InterruptedError('simulated interruption after durable epoch')
    torch.manual_seed(17)
    with monkeypatch.context() as patch:
        patch.setattr(head_fit, '_atomic_save', interrupt)
        with pytest.raises(InterruptedError):
            head_fit.fit_heads(model, cache, cache, [key], progress_path=tmp_path/'progress.pt', **kwargs)
    head_fit.fit_heads(model, cache, cache, [key], progress_path=tmp_path/'progress.pt', **kwargs)
    for name, value in model.free_heads[key].state_dict().items():
        torch.testing.assert_close(value, uninterrupted.free_heads[key].state_dict()[name], rtol=0, atol=0)

def test_cache_namespace_changes_with_encoder_or_input_content(tmp_path):
    from argparse import Namespace
    enc = tmp_path/'encoder'
    enc.mkdir()
    (enc/'config.json').write_text('{}')
    ck = tmp_path/'checkpoint.pt'
    ck.write_bytes(b'checkpoint')
    data = tmp_path/'train.json'
    data.write_text('{}')
    args = Namespace(cache_dir=str(tmp_path/'cache'), encoder=str(enc), checkpoint=str(ck),
        init_shape='', train=[str(data)], validation=[], out='a.pt', text_cache='')
    first = head_fit._cache_run(args)
    args.out = 'b.pt'
    assert head_fit._cache_run(args) == first
    (enc/'config.json').write_text('{"changed":true}')
    second = head_fit._cache_run(args)
    assert second != first
    data.write_text('{"changed":true}')
    assert head_fit._cache_run(args) != second
