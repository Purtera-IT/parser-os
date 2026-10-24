"""The owner's explicit main split supersedes a stale default, never the test bar."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ml.c3.holdout import (APPROVED_CONTRACT_SHA256, FIXED_HOLDOUT, HOLDOUT_B,
                          load_split_contract, refuse_fixed_holdout, verify_holdout_manifest)
from ml.c3.loop import lock_split, lock_validation

FIXTURES=Path(__file__).parent/'fixtures/c3_split'


def parts(*ids):return [SimpleNamespace(deal_id=i) for i in ids]


def test_known_split_fixture_pins_actual_roles_and_training_release():
    c=load_split_contract(FIXTURES/'approved_contract.json')
    assert c['released_to_training']==['010215']
    assert set(c['reserved_holdout'])=={'000026','010269','010155','010358'}
    assert set(c['golden_holdout'])=={'010038','010387'}
    assert c['selection']==['010043']
    assert '010215' not in FIXED_HOLDOUT and '010358' in HOLDOUT_B
    assert verify_holdout_manifest(FIXTURES/'approved_holdout_manifest.json',c)==set(c['reserved_holdout']+c['pending_holdout'])
    lock_split(parts('010215'),parts('010038','010387','000026','010269','010155','010358'))
    lock_validation(parts('010215'),parts('010038','010387'),parts('010043'))
    for view in ('pre_quote','planning'):
        lock_split(parts(f'010215@{view}#c3part2'),[],need_holdout=False)


@pytest.mark.parametrize('deal',['000026','010269','010155','010358','010038','010387','010279','000051','010043'])
@pytest.mark.parametrize('suffix',['','@planning','#c3part1','@pre_quote#c3part3'])
def test_every_reserved_golden_pending_kit_and_selection_source_stays_out_of_training(deal,suffix):
    with pytest.raises(ValueError,match='fixed holdout'):
        lock_split(parts(deal+suffix),[],need_holdout=False)
    with pytest.raises(ValueError):refuse_fixed_holdout([deal+suffix])


def test_selection_is_valid_for_selection_but_not_training_or_claimed_test():
    lock_validation(parts('010215@planning'),[],parts('010043@pre_quote','010043@planning'))
    with pytest.raises(ValueError,match='Selection deals'):
        lock_split(parts('010215'),parts('010043'))
    for held in ('010038','010387','010358','010279'):
        with pytest.raises(ValueError):lock_validation(parts('010215'),[],parts(held))


def test_actual_train_validation_reuse_remains_refused_even_for_training_release():
    with pytest.raises(ValueError,match='Holdout deal is in'):
        lock_split(parts('010215@pre_quote'),parts('010215@planning#c3part1'))
    with pytest.raises(ValueError,match='Validation deal'):
        lock_validation(parts('010215@planning'),[],parts('010215@pre_quote'))


@pytest.mark.parametrize('case',['unknown_version','unknown_policy','extra_field','remove_reserved','old215','duplicate','overlap','empty','wrong_type'])
def test_unknown_or_weakened_split_contracts_fail_closed(tmp_path,case):
    c=json.loads((FIXTURES/'approved_contract.json').read_text())
    if case=='unknown_version':c['schema_version']=2
    if case=='unknown_policy':c['policy_id']='unchecked'
    if case=='extra_field':c['allow_heldout_training']=True
    if case=='remove_reserved':c['reserved_holdout'].remove('010358')
    if case=='old215':c['reserved_holdout'].append('010215')
    if case=='duplicate':c['reserved_holdout'].append('000026')
    if case=='overlap':c['selection'].append('010038')
    if case=='empty':c['reserved_holdout']=[]
    if case=='wrong_type':c['pending_holdout']='010279'
    p=tmp_path/'bad.json';p.write_text(json.dumps(c))
    with pytest.raises(ValueError):load_split_contract(p)


@pytest.mark.parametrize('case',['old_split','missing_pending','extra_source','duplicate','malformed','missing'])
def test_unknown_actual_manifest_does_not_fall_back_or_expand_training(tmp_path,case):
    c=load_split_contract(FIXTURES/'approved_contract.json')
    x=json.loads((FIXTURES/'approved_holdout_manifest.json').read_text());p=tmp_path/'actual.json'
    if case=='old_split':x['deals']=['010215','010269','000026','010155']
    if case=='missing_pending':x.pop('pending')
    if case=='extra_source':x['deals'].append('010215')
    if case=='duplicate':x['deals'].append('000026')
    if case!='missing':p.write_text('{' if case=='malformed' else json.dumps(x))
    with pytest.raises(ValueError):verify_holdout_manifest(p,c)
    env=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parents[1]),C3_HOLDOUT2=str(p))
    run=subprocess.run([sys.executable,'-c','import ml.c3.holdout'],env=env,capture_output=True,text=True)
    assert run.returncode!=0 and 'ValueError' in run.stderr


@pytest.mark.parametrize('name',['C3_HOLDOUT2','C3_SPLIT_CONTRACT'])
@pytest.mark.parametrize('value',['','/missing/explicit-split.json'])
def test_explicit_missing_policy_configuration_never_uses_a_default(name,value):
    env=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parents[1]));env[name]=value
    run=subprocess.run([sys.executable,'-c','import ml.c3.holdout'],env=env,capture_output=True,text=True)
    assert run.returncode!=0 and 'ValueError' in run.stderr


def test_known_bundled_policy_is_available_without_an_optional_local_manifest(tmp_path,monkeypatch):
    from ml.c3.holdout import _holdout_b_file
    monkeypatch.delenv('C3_HOLDOUT2',raising=False)
    monkeypatch.setattr(Path,'home',lambda:tmp_path)
    assert _holdout_b_file()=={'000026','010269','010155','010358','010279'}
