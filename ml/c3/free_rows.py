"""Free rows that read r, not the line's raw deal-graph state h (run E, arch_review §5 rank 1).

Run C's free heads read ``[x; h]``: ``h`` carries the line's own text features
straight through the residual stream, so a ×20 line-type loss could be paid by a
linear read of the surface and r (where the WHY terms act) was bypassed. For the
generalization-critical heads (``R_ONLY_KEYS``: line type, admission, supplier,
about and the site heads) a free row becomes an ``RFreeRow``:

* ``weight`` / ``bias``: one row per answer on ``space_in(r)``, the head's r
  input before the asked slot (size d_head), copied from the old row's ``x``
  columns so the start stays close;
* ``h_weight``: the old row's ``h`` columns, times ``h_keep``. ``h_keep`` is a
  saved buffer the loss sets every step from ``LossWeights.free_h_out``
  (``h_keep = 1 - free_h_out``); runE.sh ramps ``free_h_out`` 0 -> 1 over two
  passes, so h is ramped out rather than cut. At ``h_keep = 0`` (the end of the
  ramp, and every later read) the row reads r only and h gets no gradient.

The row's input is the same ``[x; h]`` the old row took (head_fit reads it that
way); the row splits it.

Company free rows (``add_free_head`` on a company key) read the conduct
layer's line ``y`` only, as serve's did, so the company layer can be refit on
a frozen base (company_refit.py).

``company_grad``: the conduct block reads r (and q, and the company slot)
through a gradient scale ``model.company_r_grad`` (1.0 = unchanged; runE.sh
0.1), so Purtera's co_action / co_reason decisions shape the shared trunk ten
times less while the conduct layer itself learns at the full rate.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

#: ``add_free_head("r:<key>")`` makes ``<key>``'s row an RFreeRow.
R_ONLY = "r:"
#: The heads whose free rows read r only (runE.sh passes ``--free-head r:<key>``).
R_ONLY_KEYS = ("col:label_type", "col:admission", "col:supplier", "col:about",
               "read:site", "jdg:site_role", "jdg:site_role:reason")


class RFreeRow(nn.Module):
    """One free row per answer on the head's r input; h only while ``h_keep`` > 0."""

    def __init__(self, d_head: int, d: int, answers: int):
        super().__init__()
        self.d_head = d_head
        self.weight = nn.Parameter(torch.zeros(answers, d_head))
        self.bias = nn.Parameter(torch.zeros(answers))
        self.h_weight = nn.Parameter(torch.zeros(answers, d))
        self.register_buffer("h_keep", torch.ones(()))

    @property
    def in_features(self) -> int:
        return self.d_head + self.h_weight.shape[1]

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x, h = z[..., :self.d_head], z[..., self.d_head:]
        out = F.linear(x, self.weight.to(x.dtype), self.bias.to(x.dtype))
        return out + self.h_keep.to(x.dtype) * F.linear(h, self.h_weight.to(h.dtype))


def make_r_only(model, key: str) -> RFreeRow:
    """``key``'s free row as an RFreeRow: from its plain row's columns when it
    has one (x -> weight, h -> h_weight, bias), else zero (reads as before)."""
    cur = model.free_heads[key] if key in model.free_heads else None
    if isinstance(cur, RFreeRow):
        return cur
    opp = model.schema.by_key()[key]
    if not opp.universal:
        raise ValueError(f"{key}: an r-only row is for a universal head")
    c = model.cfg
    row = RFreeRow(c.d_head, c.d, len(opp.answers))
    if cur is not None:
        with torch.no_grad():
            row.weight.copy_(cur.weight[:, :c.d_head])
            row.h_weight.copy_(cur.weight[:, c.d_head:])
            row.bias.copy_(cur.bias)
    ref = next(model.heads.parameters())
    model.free_heads[key] = row.to(device=ref.device)
    return row


def universal_free(row: nn.Module, x_r: torch.Tensor, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """A universal free row's logits: a plain row reads the head input and h.
    An RFreeRow reads the head's r input (``x_r``, before the asked slot); the
    slot's share ramps out with h (``x_r + h_keep (x - x_r)``), so at
    ``h_keep = 1`` it reads exactly what the plain row it came from read."""
    if isinstance(row, RFreeRow):
        keep = row.h_keep.to(x.dtype)
        base = x_r if x is x_r else x_r + keep * (x - x_r)
    else:
        base = x
    return row(torch.cat([base, h.to(base.dtype)], -1))


def set_h_keep(model, keep: float) -> None:
    """Every RFreeRow's share of h (1 = the old row, 0 = r only)."""
    keep = min(1.0, max(0.0, float(keep)))
    for row in model.free_heads.values():
        if isinstance(row, RFreeRow):
            row.h_keep.fill_(keep)


def r_only_keys(model) -> list[str]:
    return [k for k, row in model.free_heads.items() if isinstance(row, RFreeRow)]


class _GradScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, s):
        ctx.s = s
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g * ctx.s, None


def company_grad(model, x: torch.Tensor) -> torch.Tensor:
    """``x`` as the conduct block reads it: the same value, its gradient into
    the shared trunk times ``model.company_r_grad`` (1.0: unchanged)."""
    s = float(getattr(model, "company_r_grad", 1.0))
    if s == 1.0 or not torch.is_grad_enabled() or not x.requires_grad:
        return x
    if s == 0.0:
        return x.detach()
    return _GradScale.apply(x, s)


def build_from_state(model, state: dict) -> None:
    """Before a checkpoint loads: an r-only row it holds (``free_heads.<key>.h_keep``)
    is built as one, so its weights load into the right shape."""
    for k in state:
        if k.startswith("free_heads.") and k.endswith(".h_keep"):
            make_r_only(model, k[len("free_heads."):-len(".h_keep")])


__all__ = ["R_ONLY", "R_ONLY_KEYS", "RFreeRow", "build_from_state", "company_grad", "make_r_only",
           "r_only_keys", "set_h_keep", "universal_free"]
