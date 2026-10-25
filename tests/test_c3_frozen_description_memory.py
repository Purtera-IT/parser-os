"""Frozen description embeddings must not retain input-hook-only graphs."""
import copy

import torch

from ml.c3.brain import Brain
from ml.c3.lm import TinyCausalLM
from ml.c3.schema import load_schema


def make_brain():
    torch.manual_seed(4)
    return Brain(load_schema(), TinyCausalLM(dim=16, layers=1, heads=2,
                                           max_len=2048), d_graph=16).eval()


def test_frozen_lm_descriptions_ignore_input_grad_hook_but_reader_trains():
    brain = make_brain()
    for p in brain.lm.parameters():
        p.requires_grad_(False)
    handle = brain.lm.tok.register_forward_hook(
        lambda _module, _args, output: output.requires_grad_(True)
    )
    try:
        desc = brain.describe()
        assert all(not t.requires_grad for pair in desc.values() for t in pair)
        question, answers = desc['col:label_type']
        # Downstream trainable modules still receive gradients from constants.
        reader = torch.nn.Linear(question.numel(), 3)
        reader(question).square().sum().backward()
        assert reader.weight.grad is not None
        assert reader.weight.grad.abs().sum() > 0
        assert answers.numel() > 0
        # Real graph-prefix binding still differentiates through the frozen LM.
        output, _, _, _ = brain.read(
            ["The customer site is in Lyon."], "universal", desc,
            graph=torch.randn(1, 16),
        )
        output.logits['col:label_type'].square().sum().backward()
        assert all(p.grad is not None and p.grad.isfinite().all()
                   for p in brain.graph.parameters())
        assert sum(p.grad.abs().sum() for p in brain.graph.parameters()) > 0
        assert brain.lm.tok.weight.grad is None
    finally:
        handle.remove()


def test_trainable_lm_description_outputs_and_gradients_are_unchanged():
    brain = make_brain()
    old = copy.deepcopy(brain)
    current = brain.describe()['col:label_type']
    # Reference the pre-change embedding route, including its original slices.
    raw = old.lm.embed(old.schema.texts())
    index = next(i for i, o in enumerate(old.schema.opportunities)
                 if o.key == 'col:label_type')
    offset = sum(1 + len(o.answers) for o in old.schema.opportunities[:index])
    op = old.schema.opportunities[index]
    prior = raw[offset], raw[offset + 1:offset + 1 + len(op.answers)]
    for a, b in zip(current, prior):
        torch.testing.assert_close(a, b)
        assert a.requires_grad
    sum(t.square().sum() for t in current).backward()
    sum(t.square().sum() for t in prior).backward()
    torch.testing.assert_close(brain.lm.tok.weight.grad, old.lm.tok.weight.grad)


def test_outer_no_grad_is_respected_even_with_trainable_lm():
    brain = make_brain()
    with torch.no_grad():
        desc = brain.describe()
    assert all(not t.requires_grad for pair in desc.values() for t in pair)
