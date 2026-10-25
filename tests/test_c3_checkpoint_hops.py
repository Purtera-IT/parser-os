"""C3_CHECKPOINT_HOPS recomputes the cross-line hops in backward: loss and every gradient must be unchanged."""
import torch

from ml.c3 import model as M
from ml.c3.graph import parse_graph
from ml.c3.losses import LossWeights, c3_loss
from ml.c3.model import C3Config, C3Model
from ml.c3.schema import load_schema


def _batch(schema, n):
    from ml.c3.data import DealExample, featurize
    from tests.test_c3_graph import _site_table   # a labeled site table with a pointer line (links, nodes, cards)
    return featurize(DealExample.from_dict(_site_table(n_rows=n, labeled=40)), schema)


def _run(flag, schema, batch):
    torch.manual_seed(0)
    cfg = C3Config(d_text=32, d=32, n_layers=1, n_heads=4, d_r=16, d_q=8, d_head=16, residual_dim=4, box_dim=4,
                   n_policy_atoms=2, policy_rank=2)
    model = C3Model(schema, cfg)
    model.enable_graph(parse_graph("rune,lattice_doc=1"))
    for p in model.parameters():          # zero-init hop maps would hide a wrong gradient
        torch.nn.init.normal_(p, std=0.02) if p.dim() > 1 else None
    old = M.CHECKPOINT_HOPS
    M.CHECKPOINT_HOPS = flag
    try:
        torch.manual_seed(1)
        loss, _ = c3_loss(model, batch, LossWeights())
        loss.backward()
    finally:
        M.CHECKPOINT_HOPS = old
    return loss.detach(), {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


def test_checkpointed_hops_match_exactly():
    schema = load_schema()
    batch = _batch(schema, M.CHECKPOINT_LINES + 64)
    l0, g0 = _run(False, schema, batch)
    l1, g1 = _run(True, schema, batch)
    torch.testing.assert_close(l1, l0, rtol=0, atol=0)
    assert g0.keys() == g1.keys() and any(n.startswith("rel_msg") or n.startswith("link_mix") for n in g0)
    for n in g0:
        torch.testing.assert_close(g1[n], g0[n], rtol=1e-6, atol=1e-7, msg=n)
