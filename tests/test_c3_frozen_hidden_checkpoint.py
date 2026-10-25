"""Bind-only page recomputation preserves the actual Qwen prefix gradient."""
import copy
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from ml.c3.lm import HFCausalLM


def make_lm():
    torch.manual_seed(8)
    lm = HFCausalLM.__new__(HFCausalLM)
    torch.nn.Module.__init__(lm)
    lm.model = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=43, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=128, attention_dropout=0.0, use_cache=False,
    )).train()
    lm.model.gradient_checkpointing_enable()
    lm.model.enable_input_require_grads()
    for p in lm.parameters():
        p.requires_grad_(False)
    return lm


def inputs():
    ids = torch.tensor([[3, 4, 5, 6], [7, 8, 0, 0]])
    return ids, ids != 0


def test_full_qwen_hidden_values_and_graph_prefix_gradients_match():
    lm = make_lm()
    prior = copy.deepcopy(lm)
    lm.checkpoint_frozen_hidden = True
    prefix = torch.randn(2, 2, 16, requires_grad=True)
    reference = prefix.detach().clone().requires_grad_()
    ids, mask = inputs()
    actual = lm.hidden(ids, mask, prefix=prefix)
    expected = prior.hidden(ids, mask, prefix=reference)
    torch.testing.assert_close(actual, expected)
    weights = torch.randn_like(actual)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    torch.testing.assert_close(prefix.grad, reference.grad, rtol=1e-5, atol=1e-6)
    assert prefix.grad.isfinite().all() and prefix.grad.abs().sum() > 0
    assert all(p.grad is None for p in lm.parameters())


def test_trainable_lm_bypasses_extra_checkpoint_and_keeps_gradients(monkeypatch):
    lm = make_lm()
    lm.model.gradient_checkpointing_disable()
    lm.checkpoint_frozen_hidden = True
    lm.model.get_input_embeddings().weight.requires_grad_(True)
    def refuse(*args, **kwargs):
        raise AssertionError('extra checkpoint must be disabled for trainable LM')
    monkeypatch.setattr('torch.utils.checkpoint.checkpoint', refuse)
    ids, mask = inputs()
    lm.hidden(ids, mask, prefix=torch.randn(2, 2, 16)).square().sum().backward()
    assert lm.model.get_input_embeddings().weight.grad is not None


def test_no_grad_inference_does_not_enable_recomputation(monkeypatch):
    lm = make_lm()
    lm.checkpoint_frozen_hidden = True
    def refuse(*args, **kwargs):
        raise AssertionError('no-grad inference must not checkpoint')
    monkeypatch.setattr('torch.utils.checkpoint.checkpoint', refuse)
    ids, mask = inputs()
    with torch.no_grad():
        result = lm.hidden(ids, mask, prefix=torch.randn(2, 2, 16))
    assert not result.requires_grad


def test_recomputation_reduces_saved_activation_bytes_for_repeated_pages():
    def retained(recompute):
        lm = make_lm()
        lm.checkpoint_frozen_hidden = recompute
        ids, mask = inputs()
        saved = []
        def pack(tensor):
            saved.append(tensor.numel() * tensor.element_size())
            return tensor
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            prefix = torch.randn(2, 2, 16, requires_grad=True)
            pages = [lm.hidden(ids, mask, prefix=prefix) for _ in range(3)]
        sum(page.sum() for page in pages).backward()
        return sum(saved)
    assert retained(True) < retained(False)
