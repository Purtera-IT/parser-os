"""Runtime balance caps may only consume rebuilt inputs with the matching policy."""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ml.c3.balance import check_balance_files, validate_balance_inputs, validate_balance_metadata
from ml.c3.data import DealExample, featurize
import ml.c3.loop as loop
from ml.c3.schema import load_schema

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT/'ml/c3/fixtures/synthetic_deal.json'
HEAD = 'read:feeds_decision'
CAPS = {'deal_cap': .15, 'head_caps': {HEAD: .15}}


@pytest.fixture(scope='module')
def schema():
    return load_schema()


@pytest.fixture
def batch(schema):
    return featurize(DealExample.load(FIXTURE), schema)


@pytest.mark.parametrize('version,config', [
    (None, None), (0, CAPS), (2, CAPS), ('1', CAPS), (True, CAPS),
    (1, None), (1, []), (1, {}),
    (1, {'deal_cap': .2, 'head_caps': {HEAD: .15}}),
    (1, {'deal_cap': .15, 'head_caps': {HEAD: .2}}),
    (1, {'deal_cap': .15, 'head_caps': {}}),
    (1, {'deal_cap': .15, 'head_caps': {HEAD: .15, 'other': .15}}),
    (1, {'deal_cap': '.15', 'head_caps': {HEAD: .15}}),
    (1, {**CAPS, 'unsupported': True}),
])
def test_caps_reject_legacy_or_mismatched_metadata_before_model_init(batch, schema, monkeypatch, version, config):
    batch.balance_version = version
    batch.balance_config = config
    def forbidden(*a, **kw):
        pytest.fail('model initialization reached before rejecting invalid balance input')
    monkeypatch.setattr(loop, 'TinyCausalLM', forbidden)
    with pytest.raises(ValueError, match='balance_version|balance_config'):
        loop.run_base([batch], [], schema, score_holdout=False, deal_cap=.15, head_cap={HEAD: .15})


@pytest.mark.parametrize('deal_cap,head_caps', [(None, {HEAD: .15}), (.15, {}), (.15, {HEAD: .15})])
def test_single_or_combined_active_caps_require_the_exact_configuration(batch, deal_cap, head_caps):
    with pytest.raises(ValueError, match='balance_version'):
        validate_balance_inputs([batch], deal_cap=deal_cap, head_caps=head_caps)
    batch.balance_version=1
    batch.balance_config={'deal_cap': deal_cap, 'head_caps': head_caps}
    validate_balance_inputs([batch], deal_cap=deal_cap, head_caps=head_caps)


def test_correct_metadata_survives_both_loaders_and_windows_and_stays_out_of_model_inputs(schema):
    raw=json.loads(FIXTURE.read_text())
    raw.update(balance_version=1, balance_config=CAPS)
    deal=DealExample.from_dict(raw)
    parts,_=loop.training_batches(deal, schema, max_atoms=8)
    assert len(parts)>1
    validate_balance_inputs(parts, deal_cap=.15, head_caps={HEAD: .15})
    for batch in parts:
        assert batch.balance_version == 1 and batch.balance_config == CAPS
        assert 'balance_version' not in batch.inputs() and 'balance_config' not in batch.inputs()
    loaded=DealExample.from_training_blob({'balance_version':1, 'balance_config':CAPS}, [], deal_id='empty')
    assert loaded.balance_version==1 and loaded.balance_config==CAPS
    # A stale planning view must fail even when another view was rebuilt.
    parts[-1]=replace(parts[-1], balance_version=None)
    with pytest.raises(ValueError, match='balance_version'):
        validate_balance_inputs(parts, deal_cap=.15, head_caps={HEAD: .15})


def test_no_caps_preserve_direct_run_base_legacy_contract(batch, schema, monkeypatch):
    class ReachedModel(Exception):
        pass
    def reached(*a, **kw):
        raise ReachedModel
    monkeypatch.setattr(loop, 'TinyCausalLM', reached)
    with pytest.raises(ReachedModel):
        loop.run_base([batch], [], schema, score_holdout=False,
                      budget=loop.Budget(teacher=0), deal_cap=None, head_cap={})
    # Uncapped validation must not invent constraints on metadata from other callers.
    validate_balance_metadata('legacy', {'old': True})


def test_matching_contract_reaches_model_initialization(batch, schema, monkeypatch):
    class ReachedModel(Exception):
        pass
    batch.balance_version=1
    batch.balance_config={'deal_cap':1., 'head_caps':{}}
    def reached(*a, **kw):
        raise ReachedModel
    monkeypatch.setattr(loop, 'TinyCausalLM', reached)
    with pytest.raises(ReachedModel):
        loop.run_base([batch], [], schema, score_holdout=False,
                      budget=loop.Budget(teacher=0), deal_cap=1.)


def test_loop_cli_checks_all_loaded_parts_before_smoke_or_limit_selection(batch, monkeypatch):
    good=replace(batch, balance_version=1, balance_config=CAPS)
    stale=replace(batch, deal_id='stale@planning')
    monkeypatch.setattr(loop, 'load_split', lambda *a, **kw: ([good, stale], []))
    monkeypatch.setattr(sys, 'argv', ['loop', '--tiny', '--train', 'selected.json', '--smoke',
                                    '--limit-parts', '1', '--deal-cap', '.15', '--head-cap', f'{HEAD}=.15'])
    with pytest.raises(ValueError, match='stale@planning.*balance_version'):
        loop.main()


def test_preflight_checks_every_actual_selected_file_and_exits_nonzero(tmp_path):
    good=tmp_path/'rebuilt.json'
    stale=tmp_path/'legacy.json'
    good.write_text(json.dumps({'balance_version':1, 'balance_config':CAPS}))
    stale.write_text('{}')
    args=[sys.executable, '-m', 'ml.c3.balance', '--deal-cap', '.15', '--head-cap', f'{HEAD}=.15', '--train']
    env={**os.environ, 'PYTHONDONTWRITEBYTECODE':'1'}
    passed=subprocess.run([*args, str(good)], cwd=ROOT, env=env, capture_output=True, text=True)
    assert passed.returncode==0, passed.stderr
    failed=subprocess.run([*args, str(good), str(stale)], cwd=ROOT, env=env, capture_output=True, text=True)
    assert failed.returncode!=0 and str(stale) in failed.stderr and 'balance_version=1' in failed.stderr
    stale.write_text(json.dumps({'balance_version':1, 'balance_config':{**CAPS,'deal_cap':.2}}))
    with pytest.raises(ValueError, match='balance_config'):
        check_balance_files([good, stale], deal_cap=.15, head_caps={HEAD:.15})
    check_balance_files([stale])  # no caps, no rebuild requirement


def test_runE_uses_identical_caps_for_early_file_preflight_and_training():
    script=(ROOT/'ml/c3/pod/runE.sh').read_text()
    preflight='python -m ml.c3.balance --train "${train_files[@]}" "${balance_args[@]}"'
    assert script.index('done <<< "$kept"') < script.index(preflight)
    assert script.index(preflight) < script.index('mib=$(nvidia-smi')
    assert script.index(preflight) < script.index('python -m ml.c3.desc_gate')
    args=script[script.index('args=(--max-atoms'):script.index('graph_args=()')]
    assert '"${balance_args[@]}"' in args
