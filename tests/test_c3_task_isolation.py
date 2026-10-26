"""Adversarial closure of bounded v2 extraction tasks (all data invented)."""
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ml.c3.data import DealExample, LiteralTarget, featurize
from ml.c3.schema import load_schema
from ml.c3.tasks import digest, source_fingerprint, validate_batch
from ml.c3.losses import (LossWeights, absence_targets, c3_loss, deal_number_loss,
                          literal_span_loss)


def payload():
    atoms = [dict(label_key='local', atom_id='a', doc_id='request', doc_kind='email',
                  text='Customer states 4 labor hours, 3 displays at Harbor during business hours.',
                  order=0, stage_at='completed', view='planning'),
             dict(label_key='outcome', atom_id='b', doc_id='refusal', doc_kind='email',
                  text='Final provider refusal and estimated 900 hours.', order=1)]
    heads = ['col:label_type', 'read:labor_hours', 'read:equipment_qty', 'read:site',
             'read:work_window']
    profile = dict(task_id='source_local_literal', execute=True, as_of=None,
        context=dict(document_id='request', source_keys=['local'],
            source_fingerprints={'local': source_fingerprint(atoms[0])}, evidence_ref='review/context'),
        allow_heads=heads, target_provenance=[dict(source_key='local', head=h,
            evidence_keys=['local'], evidence_ref='review/' + h) for h in heads])
    return dict(atoms=atoms, labels=[dict(id='l', label_key='local', labeler='owner',
                    label_type='physical_site', note='Outcome-dependent reason must not train.',
                    reads_set=dict(labor_hours='4', equipment_qty='3', site='Harbor',
                        work_window='business_hours', co_hours_estimate='900', trigger_event='refusal')),
                dict(id='o', label_key='outcome', labeler='owner', label_type='noise')],
        links=[dict(id='link', from_key='local', to_label_key='outcome', relation='context', labeler='owner')],
        judgments=[dict(head='hours', target_key='local', verdict='900', note='outcome reason')],
        deal_answers=[dict(declared_site_count=90)], outcome=dict(won=1),
        supervision_holds=dict(version=2, groups=[], task_profile=profile))


def load(blob=None):
    blob = blob or payload()
    d = DealExample.from_training_blob(blob, blob['atoms'], deal_id='synthetic')
    return d, featurize(d, load_schema())


def test_physical_context_and_loss_closure_preserve_eligible_extraction():
    blob = payload()
    original = copy.deepcopy(blob)
    d, b = load(blob)
    assert blob == original
    assert b.keys == ['local'] and len(d.atoms) == 1
    assert b.texts == [blob['atoms'][0]['text']]
    assert b.view == [''] and b.stage == ['']
    assert all(not any(values) for values in b.context.values())
    assert b.numbers_target['read:labor_hours'] == [4.]
    assert b.numbers_target['read:equipment_qty'] == [3.]
    for h in ('read:site', 'col:label_type'):
        assert b.targets[h][0] >= 0
    assert b.multi_targets['read:work_window'][0] is not None
    assert b.targets['read:co_hours_estimate'] == [-100]
    assert not b.deal_numbers and not b.judged and not b.why[0] and not b.edges
    assert not b.field_notes[0] and not d.source_supervision
    assert (absence_targets(b) == -1).all()
    validate_batch(b)


def test_forbidden_context_counterfactual_does_not_change_any_input_or_target():
    a = payload()
    b = copy.deepcopy(a)
    b['atoms'][1]['text'] = 'Completely different future outcome, 99999 sites.'
    b['labels'][1]['reads_set'] = {'co_hours_estimate': '99999'}
    b['atoms'].append(dict(label_key='future', doc_id='future', text='New gold outcome'))
    _, x = load(a)
    _, y = load(b)
    assert digest(x.inputs()) == digest(y.inputs())
    assert x.targets == y.targets and x.numbers_target == y.numbers_target
    assert x.deal_id == y.deal_id


@pytest.mark.parametrize('mutation', ['unknown', 'forecast', 'as_of', 'extra', 'badhead',
                                      'crossdoc', 'fingerprint', 'promoted', 'missingprofile'])
def test_uncertified_or_malformed_profiles_fail_closed(mutation):
    blob = payload()
    p = blob['supervision_holds']['task_profile']
    if mutation == 'unknown': p['task_id'] = 'magic_future'
    if mutation == 'forecast': p.update(task_id='quote_forecast', certified=True)
    if mutation == 'as_of': p['as_of'] = '2024-01-01'
    if mutation == 'extra': p['share_context'] = True
    if mutation == 'badhead': p['allow_heads'].append('read:co_hours_estimate')
    if mutation == 'crossdoc': p['context']['document_id'] = 'refusal'
    if mutation == 'fingerprint': blob['atoms'][0]['text'] += ' changed'
    if mutation == 'promoted': blob['labels'][0]['promoted_from'] = 'outcome'
    if mutation == 'missingprofile': del blob['supervision_holds']['task_profile']
    with pytest.raises(ValueError): load(blob)


@pytest.mark.parametrize('location', ['v1', 'top', 'plural', 'direct'])
def test_v2_cannot_fall_through_legacy_loader(location):
    b = payload()
    p = b['supervision_holds']['task_profile']
    if location == 'v1': b['supervision_holds']['version'] = 1
    if location == 'top': b['task_profile'] = p; del b['supervision_holds']
    if location == 'plural': b['task_profiles'] = [p]; del b['supervision_holds']
    with pytest.raises(ValueError):
        if location == 'direct': DealExample.from_dict(b)
        else: load(b)


def test_legacy_context_is_unchanged_and_v1_still_supported():
    b = payload()
    b['supervision_holds'] = dict(version=1, groups=[])
    d, batch = load(b)
    assert d.task_profile is None and len(batch) == 2
    assert batch.targets['read:co_hours_estimate'][0] >= 0


@pytest.mark.parametrize('carrier', ['input', 'profile', 'target', 'number', 'set', 'literal',
                                   'reason', 'card', 'dealnumber', 'scale', 'absence'])
def test_post_materialization_bypasses_rejected_before_forward(carrier):
    _, b = load()
    if carrier == 'input': b.texts[0] += ' hidden future'
    if carrier == 'profile': b.task_profile['context']['evidence_ref'] = 'different'
    if carrier == 'target': b.targets['read:co_hours_estimate'][0] = 1
    if carrier == 'number': b.numbers_target['read:co_hours_estimate'][0] = 9.
    if carrier == 'set': b.multi_targets['forbidden'] = [(1,)]
    if carrier == 'literal': b.literals.append(LiteralTarget(0, 'read:co_hours_estimate', '9', (0,)))
    if carrier == 'reason': b.why[0] = 'outcome'
    if carrier == 'card': b.judged.append(object())
    if carrier == 'dealnumber': b.deal_numbers['declared_site_count'] = 99
    if carrier == 'scale': b.head_scale['read:co_hours_estimate'] = 1.
    if carrier == 'absence': b.targets['read:site'][0] = 0
    with pytest.raises(ValueError): c3_loss(None, b)


def test_read_hold_still_overrides_eligible_task_target():
    b = payload()
    b['supervision_holds']['groups'] = [dict(group='review', source_keys=['local'], read_fields=['site'])]
    _, batch = load(b)
    assert batch.targets['read:site'] == [-100]
    assert not any(t.key == 'read:site' for t in batch.literals)
    assert batch.numbers_target['read:labor_hours'] == [4.]


def test_separate_task_contexts_have_distinct_physical_inputs_and_identity():
    a = payload()
    b = copy.deepcopy(a)
    p = b['supervision_holds']['task_profile']
    p.update(task_id='source_local_noise', allow_heads=['col:label_type'])
    p['context'].update(document_id='refusal', source_keys=['outcome'],
        source_fingerprints={'outcome': source_fingerprint(b['atoms'][1])})
    p['target_provenance'] = [dict(source_key='outcome', head='col:label_type',
        evidence_keys=['outcome'], evidence_ref='review/noise')]
    _, x = load(a)
    _, y = load(b)
    assert x.keys == ['local'] and y.keys == ['outcome']
    assert x.deal_id != y.deal_id
    assert set(x.texts).isdisjoint(y.texts)


def test_teacher_and_serving_entrypoints_reject_auxiliary_contexts():
    from ml.c3.supercharge import teacher_view, supercharge_loss
    from ml.c3.losses import brain_loss
    from ml.c3.predict import read_deal
    d, b = load()
    with pytest.raises(ValueError): read_deal(None, d, load_schema())
    # Guards must run before dereferencing teacher/model.
    with pytest.raises(ValueError): teacher_view(None, b)
    with pytest.raises(ValueError): brain_loss(None, b)
    with pytest.raises(ValueError): supercharge_loss(None, None, None, batch=b)


class DirectModel:
    def __init__(self, b):
        self.schema = load_schema()
        self.free_heads = {}
        self.out = SimpleNamespace(r=torch.ones(len(b), 2, requires_grad=True),
            logits={o.key: torch.zeros(len(b), len(o.answers), requires_grad=True)
                    for o in self.schema.opportunities if o.key in b.targets},
            numbers={k: torch.zeros(len(b), requires_grad=True) for k in b.numbers_target})
        self.pointer = torch.zeros(20, requires_grad=True)
    def describe(self): return {}
    def __call__(self, inputs, **kw):
        assert kw['company'] is None and kw['bank'] is None
        return self.out
    def word_weights(self, out, texts, rows):
        return {i: self.pointer.log_softmax(0) for i in rows}


def test_direct_route_has_only_eligible_gradients_and_never_calls_other_losses(monkeypatch):
    import ml.c3.losses as losses
    _, b = load()
    model = DirectModel(b)
    def forbidden(*args, **kwargs): raise AssertionError('forbidden objective executed')
    for name in ('deal_number_loss', 'negative_loss', 'relation_loss', 'judgment_loss',
                 'hindsight_loss', 'why_losses'):
        monkeypatch.setattr(losses, name, forbidden)
    total, parts = c3_loss(model, b)
    total.backward()
    assert model.out.numbers['read:labor_hours'].grad.abs().sum() > 0
    assert model.out.numbers['read:co_hours_estimate'].grad is None
    assert model.out.logits['read:co_hours_estimate'].grad is None
    assert model.out.logits['col:label_type'].grad.abs().sum() > 0
    assert model.pointer.grad.abs().sum() > 0
    assert all(v == 0 for k,v in parts.items() if k not in ('heads', 'numbers', 'literals'))
    with pytest.raises(ValueError): c3_loss(model, b, bank=object())


def test_exact_zero_scale_bypasses_and_fractional_scaling():
    out = SimpleNamespace(r=torch.ones(1, 2))
    param = torch.nn.Parameter(torch.tensor([0., 0., 0.]))
    model = SimpleNamespace(deal_number=lambda r: param[0].reshape(1),
        word_weights=lambda *a, **kw: {0: param.log_softmax(0)})
    b = SimpleNamespace(deal_numbers={'declared_site_count': 4}, head_scale={}, texts=['4 labor hours'],
        literals=[LiteralTarget(0,'read:labor_hours','4',(0,))])
    base_num = deal_number_loss(model,out,b)
    base_lit = literal_span_loss(model,out,b)
    b.head_scale = {'deal:site_count': .2, 'read:labor_hours': .2}
    assert torch.allclose(deal_number_loss(model,out,b), .2 * base_num)
    assert torch.allclose(literal_span_loss(model,out,b), .2 * base_lit)
    b.head_scale = {'deal:site_count': 0., 'read:labor_hours': 0.}
    assert deal_number_loss(model,out,b).item() == 0
    assert literal_span_loss(model,out,b).item() == 0
    assert param.grad is None
    b.literals.append(LiteralTarget(0,'read:site','Harbor',(1,)))
    literal_span_loss(model,out,b).backward()
    assert param.grad.abs().sum() > 0


@pytest.mark.parametrize('mutation', ['text', 'append', 'profile', 'remove'])
def test_deal_context_cannot_be_rewritten_between_loader_and_features(mutation):
    from ml.c3.data import Atom
    d, _ = load()
    if mutation == 'text': d.atoms[0].text += ' hidden outcome'
    if mutation == 'append': d.atoms.append(Atom(key='extra', text='future'))
    if mutation == 'profile': d.task_profile['context']['evidence_ref'] = 'other'
    if mutation == 'remove': d.task_profile = None
    with pytest.raises(ValueError): featurize(d, load_schema())


def test_windows_and_chunks_preserve_task_boundaries():
    from ml.c3.longdeal import plan_long_deal
    from ml.c3.chunks import split_deal
    b = payload()
    p = b['supervision_holds']['task_profile']
    for i in range(4):
        a = dict(label_key=f'local{i}', atom_id=f'local{i}', doc_id='request',
                 text=f'Customer states {i+1} displays.', order=i+2)
        b['atoms'].append(a)
        b['labels'].append(dict(label_key=a['label_key'], labeler='owner', reads_set={'equipment_qty':str(i+1)}))
        p['context']['source_keys'].append(a['label_key'])
        p['context']['source_fingerprints'][a['label_key']] = source_fingerprint(a)
        p['target_provenance'].append(dict(source_key=a['label_key'], head='read:equipment_qty',
            evidence_keys=[a['label_key']], evidence_ref='review/quantity'))
    d, _ = load(b)
    parts = plan_long_deal(d, load_schema(), max_atoms=3, overlap=1).parts
    parts += [featurize(c.deal,load_schema()) for c in split_deal(d,2)]
    assert len(parts) > 2
    for part in parts:
        validate_batch(part)
        assert 'outcome' not in part.keys and len(part) <= 3
        assert part.task_profile == p


def test_actual_small_cpu_forward_backward_is_bounded(tmp_path):
    from ml.c3.model import C3Config, C3Model
    from ml.c3.loop import configure_training_view_tag
    _, b = load()
    cfg = C3Config(d_text=32, d=32, n_layers=1, n_heads=4, d_r=16, d_q=8,
                   d_head=16, residual_dim=4, box_dim=4, n_policy_atoms=2, policy_rank=2)
    model = C3Model(load_schema(),cfg)
    configure_training_view_tag(model, [b], True)
    assert model.cfg.view_tag
    from ml.c3.gates import save_checkpoint,load_checkpoint
    import hashlib
    parent=tmp_path/'certified-main.pt'
    candidate=tmp_path/'aux-candidate.pt'
    save_checkpoint(parent,model,step=11,schema_sha256='synthetic-registry')
    parent_hash=hashlib.sha256(parent.read_bytes()).hexdigest()
    fork=C3Model(load_schema(),cfg)
    assert load_checkpoint(parent,fork,schema_sha256='synthetic-registry')==11
    assert fork.cfg.view_tag
    model=fork
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    assert not optimizer.state  # no main momentum imported into auxiliary pass
    loss, logs = c3_loss(model, b)
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    assert model.q_head.weight.grad is not None  # shared/q updates are allowed, not strict parameter isolation
    optimizer.step()
    save_checkpoint(candidate,model,step=12,schema_sha256='synthetic-registry')
    assert hashlib.sha256(parent.read_bytes()).hexdigest()==parent_hash
    assert candidate.read_bytes()!=parent.read_bytes()
    restored=C3Model(load_schema(),cfg)
    assert load_checkpoint(candidate,restored,schema_sha256='synthetic-registry')==12
    assert restored.cfg.view_tag
    assert logs['conduct'] == logs['hindsight'] == logs['relations'] == 0


def test_anchor_and_teacher_context_cannot_supply_unlabeled_forecast_targets():
    from ml.c3.anchor import TrunkAnchor
    from ml.c3.losses import teacher_context
    _, b = load()
    a = object.__new__(TrunkAnchor)
    a.kl, a.heads = 1., ['col:label_type']
    a.targets = {}
    class NoForward:
        training = False
        def eval(self): pass
        def train(self, was): pass
        def __call__(self, *args, **kw): raise AssertionError('anchor must not forward')
    assert a.cache(NoForward(),[b]) == 0
    assert a.loss(None,b) is None
    with pytest.raises(ValueError): teacher_context(None,b)


def test_default_teacher_schedule_rejects_tasks_before_model_initialization():
    from ml.c3.loop import run_base
    _, b = load()
    with pytest.raises(ValueError, match='stages disabled'):
        run_base([b], [], score_holdout=False)


def test_derived_absence_is_not_a_literal_task_target():
    p = payload()
    p['labels'][0]['reads_set']['site'] = '_absent'
    _, b = load(p)
    assert b.targets['read:site'] == [-100]
    assert not any(t.key == 'read:site' for t in b.literals)
    validate_batch(b)


@pytest.mark.parametrize('bad', ['bound', 'cross_evidence', 'head_shape', 'hex', 'task_shape'])
def test_context_and_provenance_are_bounded_structural_contracts(bad):
    b = payload()
    p = b['supervision_holds']['task_profile']
    if bad == 'bound': p['context']['source_keys'] = ['x'] * 2049
    if bad == 'cross_evidence': p['target_provenance'][0]['evidence_keys'].append('outcome')
    if bad == 'head_shape': p['allow_heads'] = [{}]
    if bad == 'hex': p['context']['source_fingerprints']['local'] = 'z' * 64
    if bad == 'task_shape': p['task_id'] = {}
    with pytest.raises(ValueError): load(b)


def test_chunking_does_not_unmask_an_eligible_but_zero_scaled_head():
    from ml.c3.chunks import split_deal
    blob = payload()
    blob['head_scale'] = {'read:labor_hours': 0.}
    d, _ = load(blob)
    chunk = split_deal(d, 1)[0].deal
    b = featurize(chunk, load_schema())
    assert b.head_scale['read:labor_hours'] == 0.
    assert b.head_scale['read:co_hours_estimate'] == 0.


def test_removing_profile_does_not_enable_teacher_or_legacy_loss_fallback():
    from ml.c3.losses import teacher_context
    _, b = load()
    b.task_profile = None
    with pytest.raises(ValueError): c3_loss(None,b)
    with pytest.raises(ValueError): teacher_context(None,b)


def test_complete_loss_census_and_all_staged_objectives_cannot_reactivate(monkeypatch):
    import ml.c3.losses as losses
    from ml.c3.tasks import LOSS_TERMS, LOSS_CONFIG_FIELDS, validate_loss_census
    from ml.c3.loop import gold_weights, STAGES
    validate_loss_census(LossWeights)
    assert len(LOSS_TERMS) == 48 and len(LOSS_CONFIG_FIELDS) == 12
    class Unknown:
        __dataclass_fields__ = {**LossWeights.__dataclass_fields__, 'new_forecast_objective':None}
    with pytest.raises(ValueError): validate_loss_census(Unknown)
    _, b = load()
    model = DirectModel(b)
    def forbidden(*a,**kw): raise AssertionError('forbidden eager loss path')
    # Patch every loss-module consumer except the three direct targets.
    for name in list(vars(losses)):
        if name.endswith('_loss') and name not in ('c3_loss','literal_span_loss','answer_loss'):
            monkeypatch.setattr(losses,name,forbidden)
    for stage in STAGES:
        w, sizes = gold_weights(stage)
        _, parts = c3_loss(model,b,w,judgment_sizes=sizes)
        assert set(parts) == LOSS_TERMS
        assert all(v == 0 for k,v in parts.items() if k not in ('heads','numbers','literals'))


def test_observed_stage_is_an_explicit_source_input_not_a_gold_context_slot():
    b = payload()
    b['atoms'][0]['stage_at'] = 'scoping'
    p = b['supervision_holds']['task_profile']
    p['context']['source_fingerprints']['local'] = source_fingerprint(b['atoms'][0])
    p['context']['observed_stage_evidence'] = {'local':'review/actual-source-stage'}
    p['allow_heads'].append('read:stage_std')
    p['target_provenance'].append(dict(source_key='local',head='read:stage_std',
        evidence_keys=['local'],evidence_ref='review/stated-stage-classification'))
    b['labels'][0]['reads_set']['stage_std'] = 'scoping'
    _, batch = load(b)
    assert batch.stage == ['scoping'] and batch.view == ['']
    assert batch.targets['read:stage_std'][0] >= 0
    assert all(not any(v) for v in batch.context.values())  # no gold label inputs
    del p['context']['observed_stage_evidence']
    _, other = load(b)
    assert other.stage == ['']


def test_legacy_cache_and_postfit_loaders_refuse_auxiliary_files(tmp_path):
    import json
    from ml.c3.loop import load_deal,load_split
    b=payload(); b['deal_id']='synthetic'
    path=tmp_path/'aux.json'; path.write_text(json.dumps(b))
    with pytest.raises(ValueError,match='consumer mismatch'): load_deal(path)
    with pytest.raises(ValueError,match='consumer mismatch'): load_split([str(path)],load_schema())
    d=load_deal(path,task_consumer='bounded_auxiliary')
    assert d.task_profile
    parts,_=load_split([str(path)],load_schema(),task_consumer='bounded_auxiliary')
    assert parts and all(p.task_profile for p in parts)
    with pytest.raises(ValueError,match='draft promotion'):
        load_deal(path,{str(path):'wrong'},task_consumer='bounded_auxiliary')
    b['supervision_holds']=dict(version=1,groups=[])
    path.write_text(json.dumps(b))
    assert load_deal(path).task_profile is None
    with pytest.raises(ValueError): load_deal(path,task_consumer='bounded_auxiliary')


def test_auxiliary_step_cannot_pull_anchor_update_ema_or_move_unused_adamw_state(monkeypatch):
    import ml.c3.loop as loop
    _,b=load()
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shared=torch.nn.Parameter(torch.tensor(1.))
            self.held=torch.nn.Parameter(torch.tensor(2.))
        def ema_update(self): raise AssertionError('auxiliary EMA update')
    m=Model()
    opt=torch.optim.AdamW(m.parameters(),lr=.1,weight_decay=.1)
    # Prime actual optimizer momentum for the excluded scalar.
    (m.held.square()+m.shared.square()).backward(); opt.step(); opt.zero_grad(set_to_none=True)
    held=m.held.detach().clone(); shared=m.shared.detach().clone()
    anchor=SimpleNamespace(pull=1.,pull_step=lambda *a: (_ for _ in ()).throw(AssertionError('anchor pull')))
    monkeypatch.setattr(loop,'c3_loss',lambda *a,**kw:(m.shared.square(),{}))
    loop._student_step(m,opt,b,LossWeights(),None,None,None,'cpu',anchor=anchor)
    assert torch.equal(m.held,held) and not torch.equal(m.shared,shared)
    assert m.held.grad is None


def test_auxiliary_evaluation_uses_direct_context_and_never_scores_no_link():
    from ml.c3.loop import _score_oneshot_inner
    _,b=load()
    model=DirectModel(b)
    def forward(inputs,**kw):
        assert kw['company'] is None
        return model.out
    class Eval:
        def __call__(self,*a,**kw): return forward(*a,**kw)
    # No .relations/.judge supplied: a generic relation/card evaluator would fail.
    scores=_score_oneshot_inner(Eval(),b)
    assert scores and not any(k.startswith(('rel:','jdg:')) for k in scores)
    b._table_facts={'future':'cached-outcome'}
    with pytest.raises(ValueError): c3_loss(None,b)


def test_effective_read_hold_and_numeric_identity_cannot_be_reintroduced():
    b=payload()
    b['supervision_holds']['groups']=[dict(group='hold',source_keys=['local'],read_fields=['site'])]
    _,part=load(b)
    part.targets['read:site'][0]=1
    with pytest.raises(ValueError,match='supervision/holds'): c3_loss(None,part)
    _,part=load()
    part.numbers_target['read:labor_hours'][0]=999.
    with pytest.raises(ValueError,match='supervision/holds'): c3_loss(None,part)


def test_question_and_headfit_cache_execution_refuses_v2_before_forward(tmp_path):
    import json
    from ml.c3.question_fit import cache_deal
    from ml.c3.head_fit import _cached_split
    b=payload(); b['deal_id']='synthetic'
    b['judgments']=[dict(head='gap',target={'kind':'deal','query':'Outcome sentinel?'},verdict='valid')]
    p=tmp_path/'aux.json'; p.write_text(json.dumps(b))
    class NoForward:
        def eval(self): pass
        def __call__(self,*a,**kw): raise AssertionError('postfit must not forward')
    with pytest.raises(ValueError,match='consumer mismatch'):
        cache_deal(NoForward(),load_schema(),'cpu',str(p),['Outcome sentinel?'],10)
    with pytest.raises(ValueError,match='consumer mismatch'):
        _cached_split(None,[str(p)],load_schema(),10,['col:label_type'],'cpu',None)


def test_mixed_run_refusal_never_rewrites_certified_teacher_schedule():
    from ml.c3.loop import run_base,Budget
    _,aux=load()
    raw=payload(); raw['supervision_holds']=dict(version=1,groups=[])
    main,_=load(raw)
    main=featurize(main,load_schema())
    main.deal_id='different-main'
    budget=Budget(teacher=3,teach=4,bind=2,flips=1,spread=1)
    with pytest.raises(ValueError,match='stages disabled'):
        run_base([main,aux],[],budget=budget,score_holdout=False)
    assert (budget.teacher,budget.teach,budget.bind,budget.flips,budget.spread)==(3,4,2,1,1)
    assert any(main.why)
    with pytest.raises(ValueError,match='resume is unsupported'):
        run_base([aux],[],budget=Budget(teacher=0,teach=0),score_holdout=False,resume=True)


@pytest.mark.parametrize('marker',['tier','note','consumer','skip','exclude_read'])
def test_raw_source_exclusion_always_precedes_sanitization(marker):
    b=payload(); row=b['labels'][0]
    if marker=='tier': row['weight_tier']='exclude'
    if marker=='note': row['note']='EXCLUDE_FROM_TRAINING: protected owner decision'
    if marker=='consumer': row['consumer']='ignore'
    if marker=='skip': row['reads_set']['skip']='true'
    if marker=='exclude_read': row['reads_set']['exclude_from_training']='true'
    original=copy.deepcopy(b)
    _,part=load(b)
    assert part.targets['read:site']==[-100]
    assert part.numbers_target['read:equipment_qty']==[None]
    assert not part.literals and not part.why[0]
    assert len(part)==1 and b==original


def test_source_conflicting_saved_labels_are_not_hidden_by_note_sanitization():
    b=payload()
    other=copy.deepcopy(b['labels'][0]); other['id']='conflict'; other['note']='conflicting owner reasoning'
    b['labels'].append(other)
    with pytest.raises(ValueError,match='Conflicting saved'): load(b)


def test_source_target_cannot_change_between_materializing_loader_and_features():
    d,_=load()
    d.atoms[0].label['reads_set']['labor_hours']='999'
    with pytest.raises(ValueError,match='source supervision changed'): featurize(d,load_schema())


@pytest.mark.parametrize('tier,weight',[('load_bearing',3.),('slight',.3)])
def test_source_row_weight_is_not_flattened_by_projection(tier,weight):
    b=payload(); b['labels'][0]['weight_tier']=tier
    _,part=load(b)
    assert part.weights==[weight]


def test_actual_auxiliary_numbers_run_does_not_emit_teacher_sidecar(tmp_path,monkeypatch):
    import ml.c3.loop as loop
    from ml.c3.model import C3Config
    _,b=load()
    cfg=C3Config(d_text=32,d=32,n_layers=1,n_heads=4,d_r=16,d_q=8,
                 d_head=16,residual_dim=4,box_dim=4,n_policy_atoms=2,policy_rank=2)
    budget=loop.Budget(**{**{k:0 for k in loop.Budget.__dataclass_fields__},'numbers':1})
    def forbidden(*a,**kw): raise AssertionError('auxiliary teacher execution/artifact')
    monkeypatch.setattr(loop,'save_teacher',forbidden)
    monkeypatch.setattr(loop,'_teacher_part',forbidden)
    measured=[]
    original=loop.c3_loss
    def record(*args,**kwargs):
        loss,logs=original(*args,**kwargs)
        measured.append((args[2].numbers,args[2].literals,logs['numbers'],logs['literals']))
        return loss,logs
    monkeypatch.setattr(loop,'c3_loss',record)
    checkpoint=tmp_path/'aux.pt'
    _,result=loop.run_base([b],[],load_schema(),budget,cfg=cfg,checkpoint=str(checkpoint),
                           score_holdout=False,fit_parts=0,balance=False,adamw=True)
    assert checkpoint.exists() and not Path(str(checkpoint)+'.teacher').exists()
    assert 'teacher' not in result.stage_loss and result.stage_loss['numbers']>0
    assert measured and all(nw>0 and lw>0 and n>0 and lit>0 for nw,lw,n,lit in measured)
    line_weights,_=loop.gold_weights('lines')
    assert line_weights.numbers==line_weights.literals==0


def test_legacy_teacher_finalization_is_unchanged(tmp_path,monkeypatch):
    import ml.c3.loop as loop
    from ml.c3.model import C3Config
    raw=payload(); raw['supervision_holds']=dict(version=1,groups=[])
    _,b=load(raw)
    cfg=C3Config(d_text=32,d=32,n_layers=1,n_heads=4,d_r=16,d_q=8,
                 d_head=16,residual_dim=4,box_dim=4,n_policy_atoms=2,policy_rank=2)
    writes=[]
    monkeypatch.setattr(loop,'save_teacher',lambda path,*a,**kw:writes.append(path))
    budget=loop.Budget(**{k:0 for k in loop.Budget.__dataclass_fields__})
    checkpoint=tmp_path/'main.pt'
    loop.run_base([b],[],load_schema(),budget,cfg=cfg,checkpoint=str(checkpoint),
                  score_holdout=False,fit_parts=0,balance=False)
    assert writes==[str(checkpoint)+'.teacher']
