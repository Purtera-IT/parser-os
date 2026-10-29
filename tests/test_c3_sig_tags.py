"""Signature / contact-block facts (data.signature_tags) and the opt-in encoder.sig_in input: off by default (no
module, no change), zero at start when enabled, and carried by the checkpoint so serve builds the same input."""
import importlib.util
from pathlib import Path

import pytest
import torch

from ml.c3.data import Atom, DealExample, featurize, signature_tags
from ml.c3.gates import load_checkpoint, save_checkpoint
from ml.c3.model import C3Model
from ml.c3.schema import load_schema


def _atom(i, text, doc="d1", kind="email", org=""):
    return Atom(key=f"k{i}", atom_id=f"a{i}", text=text, doc_id=doc, doc_kind=kind, order=i, speaker_org=org)


def test_signature_facts():
    atoms = [_atom(0, "We will install the cameras at 12 Oak Street, Austin, TX 78701 next week."),
             _atom(1, "Thanks,"), _atom(2, "Jane Roe"), _atom(3, "Acme Corp", org="Acme Corp"),
             _atom(4, "12 Oak Street, Austin, TX 78701"), _atom(5, "Office: 512-555-0100"),
             _atom(6, "From: Bob <bob@example.com>"),
             _atom(7, "Location: 12 Oak Street, Austin, TX 78701", doc="m1", kind="meeting"),
             _atom(8, "12 Oak Street, Austin, TX 78701", doc="d2")]
    t = signature_tags(atoms)
    assert t[0] == []
    assert "sig:after_close" in t[2] and "sig:after_close" in t[4]
    assert "sig:contact" in t[4] and "sig:before_from" in t[4]
    assert "sig:sender_org" in t[3]
    assert "sig:invite_loc" in t[7]
    assert t[8] == ["sig:elsewhere"]          # the same words as a signature line, in a plain document


@pytest.fixture
def case():
    spec = importlib.util.spec_from_file_location('why2_fixture', Path(__file__).with_name('test_c3_why2.py'))
    f = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(f)
    schema = load_schema()
    b = featurize(DealExample.from_training_blob({'labels': f.LABELS, 'judgments': f.JUDGMENTS},
                  f.ATOMS, deal_id='synthetic-sig', serve_lines_only=False), schema)
    torch.manual_seed(0)
    return C3Model(schema, f.ASKED).eval(), b


def test_off_by_default_and_zero_when_enabled(case, tmp_path):
    model, b = case
    assert getattr(model.encoder, "sig_in", None) is None
    inp = b.oneshot()
    inp["sig_tags"] = [["sig:contact", "sig:after_close"]] * len(b)
    with torch.no_grad():
        before = model(inp).logits["col:label_type"]
        model.enable_sig_tags()
        same = model(inp).logits["col:label_type"]
        torch.testing.assert_close(same, before, rtol=0, atol=0)
        model.encoder.sig_in.weight.normal_()
        moved = model(inp).logits["col:label_type"]
    assert not torch.allclose(moved, before)
    save_checkpoint(tmp_path / "sig.pt", model, step=0, schema_sha256="x")
    assert torch.load(tmp_path / "sig.pt", weights_only=False)["sig_tags"] is True


def test_checkpoint_round_trip(case, tmp_path):
    model, b = case
    schema = model.schema
    plain = tmp_path / "plain.pt"
    save_checkpoint(plain, model, step=0, schema_sha256="x")
    assert "sig_tags" not in torch.load(plain, weights_only=False)
    assert not any(k.startswith("encoder.sig_in") for k in torch.load(plain, weights_only=False)["state"])
    model.enable_sig_tags()
    with torch.no_grad():
        model.encoder.sig_in.weight.normal_()
    path = tmp_path / "sig.pt"
    save_checkpoint(path, model, step=0, schema_sha256="x")
    import copy
    other = copy.deepcopy(model)
    other.encoder.sig_in = None
    other.cfg.sig_tags = False
    load_checkpoint(path, other, schema_sha256="x")
    assert other.cfg.sig_tags and torch.equal(other.encoder.sig_in.weight, model.encoder.sig_in.weight)
