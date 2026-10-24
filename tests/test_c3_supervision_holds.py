"""Held uncertainty cannot become a positive, a no-link negative, or a note target."""
import copy
from types import SimpleNamespace

import pytest
import torch

from ml.c3.data import DealExample, featurize
from ml.c3.holds import mask_training_blob
from ml.c3.link_why import link_note_rows
from ml.c3.longdeal import plan_long_deal
from ml.c3.loop import accept_drafts
from ml.c3.losses import relation_loss
from ml.c3.schema import load_schema


@pytest.fixture
def schema():
    return load_schema()


def payload():
    atoms = [dict(label_key=f"k{i}", atom_id=f"a{i}", text=f"Site {i}",
                  doc_id="d", doc_kind="email", order=i) for i in range(5)]
    labels = [dict(id=f"label{i}", label_key=f"k{i}", label_type="physical_site",
                   labeler="owner", note=f"Verified site {i}.") for i in range(5)]
    links = [dict(id="saved", from_key="k0", to_label_key="k1", relation="same_as",
                  labeler="owner", note="uncertain saved copy"),
             dict(id="draft", supersedes="saved", from_key="k0", to_label_key="k2",
                  relation="same_as", labeler="bot (assistant)", note="uncertain draft copy"),
             dict(id="good", from_key="k3", to_label_key="k4", relation="same_as",
                  labeler="owner", note="verified edge")]
    return dict(deal_id="synthetic-holds", atoms=atoms, labels=labels, links=links,
                judgments=[], supervision_holds=dict(version=1, groups=[dict(group="g",
                    relation="same_as", source_keys=["k0"], link_ids=["saved","draft"],
                    judgment_targets=[], note_label_keys=["k0"])]))


@pytest.mark.parametrize("approved", [False, True])
def test_saved_and_draft_targets_are_ignored_without_losing_sites(schema, approved):
    blob = payload()
    original = copy.deepcopy(blob)
    if approved:
        accept_drafts(blob)
    before_load = copy.deepcopy(blob)
    deal = DealExample.from_training_blob(blob, blob["atoms"], deal_id=blob["deal_id"])
    batch = featurize(deal, schema)
    assert blob == before_load  # loader may not edit any source row
    assert len(batch.texts) == len(original["atoms"])
    assert batch.targets["col:label_type"][0] >= 0
    assert batch.link_ignored == {"same_as": [0]}
    assert batch.link_elsewhere["same_as"] == [0]
    assert (0,1) not in batch.edges.get("same_as", [])
    assert (0,2) not in batch.edges.get("same_as", [])
    assert (3,4) in batch.edges["same_as"]
    assert not batch.why[0]
    assert not any(i == 0 and rel == "same_as" for i,j,rel,t in batch.edge_notes)
    assert link_note_rows(SimpleNamespace(schema=schema), batch) == [(3,4,"same_as","verified edge")]


def test_no_negative_gradient_at_held_source(schema):
    b = payload()
    batch = featurize(DealExample.from_training_blob(b,b["atoms"],deal_id=b["deal_id"]), schema)
    scores = torch.zeros((5,5), requires_grad=True)
    none = torch.zeros(5, requires_grad=True)
    out = SimpleNamespace(r=torch.zeros(5,2), relations={"same_as": scores}, link_none={"same_as":none})
    relation_loss(out,batch).backward()
    assert torch.equal(scores.grad[0], torch.zeros(5))
    assert none.grad[0] == 0
    assert scores.grad[3].abs().sum() > 0
    assert none.grad[1] != 0  # an unheld labeled source still learns no-link


def test_card_judgment_and_note_are_not_derived_back(schema):
    b = payload()
    b["links"][0].update(from_key="atom:a0", from_head="suppression")
    b["judgments"] = [dict(id="card", head="suppression", target_key="atom:a0",
                            target={"atomId":"a0"}, verdict="duplicate", labeler="owner",
                            note="uncertain card note")]
    g = b["supervision_holds"]["groups"][0]
    g.update(source_keys=["atom:a0"], judgment_targets=["atom:a0"])
    clean, held = mask_training_blob(b,b["atoms"])
    assert ("k0","same_as") in held
    assert not clean["judgments"]
    assert b["judgments"]  # original remains present
    batch = featurize(DealExample.from_training_blob(b,b["atoms"],deal_id=b["deal_id"]),schema)
    assert not any("uncertain card note" in str(x) for x in batch.judged)
    assert not any("uncertain card note" in str(x) for x in batch.unplaced_notes)


def test_hold_survives_windows_and_bridge_parts(schema):
    b = payload()
    deal = DealExample.from_training_blob(b,b["atoms"],deal_id=b["deal_id"])
    plan = plan_long_deal(deal,schema,max_atoms=3,overlap=1)
    for batch in plan.parts:
        if "k0" in batch.keys:
            i = batch.keys.index("k0")
            assert i in batch.link_ignored["same_as"]
            assert not any(src == i for src,dst in batch.edges.get("same_as",[]))


def test_bad_hold_version_refuses_loading():
    b = payload()
    b["supervision_holds"]["version"] = 99
    with pytest.raises(ValueError, match="version"):
        DealExample.from_training_blob(b,b["atoms"],deal_id=b["deal_id"])
    with pytest.raises(ValueError, match="labeling-blob loader"):
        DealExample.from_dict(dict(deal_id='direct',atoms=[],supervision_holds={'version':1,'groups':[]}))


def test_different_hold_groups_do_not_cross_mask_relations(schema):
    b = payload()
    b['links'].append(dict(id='context',from_key='k1',to_label_key='k4',
                          relation='context',labeler='owner',note='held context'))
    b['supervision_holds']['groups'].append(dict(group='other',relation='context',
        source_keys=['k1'],link_ids=['context'],judgment_targets=[],note_label_keys=[]))
    batch = featurize(DealExample.from_training_blob(b,b['atoms'],deal_id=b['deal_id']),schema)
    assert batch.link_ignored == {'same_as':[0], 'context':[1]}
    assert 'supervision_holds' not in batch.inputs()
    assert 'link_ignored' not in batch.inputs()
