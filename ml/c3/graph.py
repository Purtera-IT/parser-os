"""Run E's cross-line reasoning (arch/arch_graph_review.md, fixes 2-7).

Everything here is behind ``C3Model.graph`` flags. With the defaults the model
reads exactly as before (the start checkpoint reproduces); every module this
file adds starts at zero, so turning a flag on changes nothing until training
opens it. The flags travel in the checkpoint (``gates.save_checkpoint``), so
serve builds the same graph the run trained.

* **Visibility** (``mask``). Who may read whom, the same matrix for the deal
  graph, the asked lattice, the link hop, the pointers and the nodes:

  - ``index``: the old lower triangle over the sorted lines.
  - ``arrival``: j is visible to i when j entered no later than i
    (``times``), so a document that arrives whole sees itself.
  - ``view``: everything in the view. A training view is already cut at its
    stage (``view_cut``) and serve re-reads the whole deal as of compile time,
    so the cut, not line order, is the leak boundary: a later email naming
    "the attached list is our 145 stores" can change how the earlier rows are
    typed. In training, ``cut_dropout`` reads a share of steps as of a random
    earlier compile (line i sees j when t_j <= max(t*, t_i)); the hindsight
    loss then teaches q_i to predict the line's reading once the later lines
    arrive, which is what "earlier lines are revised by later ones" means.

* **Hop** (``hop_both``, ``hop2``, ``hop_to_h``, ``pointer_edge``,
  ``hop_near_miss``). Every proposed relation passes its message both ways,
  each direction through its own zero-init map (``rel_msg``); governs is read
  child -> parent (the transpose of the box scores) with a fixed "no parent"
  column, and the parent reads its rows normalised by their mass + 1. The
  trained pointer head is one more edge (``rests_on``). near_miss twins answer
  differently by construction, so they send nothing. A second hop re-reads
  the same link distributions on the updated lines (row -> header -> sibling),
  and ``link_mix_h`` carries the whole hop into the h the free heads read.

* **Nodes** (``nodes``). One node per table (same document and section) and
  one per document (an attached file is one document). Each node pools its
  lines with the card pool (``pool_score``/``pool_in``, the roster/sheet card
  features), and every line reads its table's and its document's node back:
  the table's decision conditions each row, and a row with strong evidence of
  its own still decides. A line can point at another document's node
  ("see the attached site list"): it reads what is inside, and the node reads
  what the pointer says, which every row of it then reads.

* **Card slots** (``card_slots``): the card pool reads r plus the lattice slot
  of its question's space.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

#: Defaults: the graph the start checkpoint was trained with.
GRAPH_DEFAULTS: dict[str, Any] = {
    "mask": "index", "cut_dropout": 0.0, "hop_both": False, "hop2": False, "hop_to_h": False,
    "pointer_edge": False, "hop_near_miss": True, "nodes": False, "card_slots": False,
}
#: Run E's graph (pod/runE.sh turns it on with ``--graph rune``).
RUNE_GRAPH: dict[str, Any] = {
    "mask": "view", "cut_dropout": 0.3, "hop_both": True, "hop2": True, "hop_to_h": True,
    "pointer_edge": True, "hop_near_miss": False, "nodes": True, "card_slots": True,
}
MASKS = ("index", "arrival", "view")
#: The "no parent" logit of the governs hop (log 0.5): a row whose best
#: parent is less likely than even reads nothing.
GOVERNS_NONE = -0.6931471805599453


def parse_graph(spec: str | None) -> dict[str, Any]:
    """``rune`` | ``off`` | ``key=value,...`` (over the defaults; ``rune,mask=arrival`` too)."""
    out = dict(GRAPH_DEFAULTS)
    for part in filter(None, (p.strip() for p in str(spec or "").split(","))):
        if part == "rune":
            out.update(RUNE_GRAPH)
            continue
        if part == "off":
            out = dict(GRAPH_DEFAULTS)
            continue
        k, _, v = part.partition("=")
        if k not in GRAPH_DEFAULTS:
            raise ValueError(f"--graph: no flag {k!r}")
        d = GRAPH_DEFAULTS[k]
        if isinstance(d, bool):
            out[k] = v.strip().lower() in ("1", "true", "yes", "on")
        elif isinstance(d, float):
            out[k] = float(v)
        else:
            out[k] = v.strip()
    if out["mask"] not in MASKS:
        raise ValueError(f"--graph: mask must be one of {MASKS}")
    return out


def _zero(lin: nn.Linear) -> nn.Linear:
    nn.init.zeros_(lin.weight)
    if lin.bias is not None:
        nn.init.zeros_(lin.bias)
    return lin


def hop_relations(model) -> list[str]:
    from .schema import RELATION  # noqa: PLC0415

    rels = [o.field for o in model.schema.select(kind=RELATION) if o.field != "governs"]
    return rels + ["governs", "rests_on"]


def enable_graph(model, flags: dict[str, Any] | None) -> dict[str, Any]:
    """Set the model's graph flags and build what they need (zero-init).
    Idempotent; a module already built keeps its weights."""
    flags = {**GRAPH_DEFAULTS, **dict(flags or {})}
    if flags["mask"] not in MASKS:
        raise ValueError(f"graph mask {flags['mask']!r}")
    c = model.cfg
    ref = next(model.parameters())
    dev = ref.device

    def build(name: str, make) -> None:
        if not hasattr(model, name) or getattr(model, name) is None:
            setattr(model, name, make().to(dev))

    if flags["hop_to_h"]:
        build("link_mix_h", lambda: _zero(nn.Linear(c.d_r, c.d)))
    if flags["hop_both"]:
        build("rel_msg", lambda: nn.ModuleDict({f"{r}__{d}": _zero(nn.Linear(c.d_r, c.d_r, bias=False))
                                                for r in hop_relations(model) for d in ("out", "back")}))
    if flags["nodes"]:
        build("node_table", lambda: _zero(nn.Linear(c.d_head, c.d_r)))
        build("node_doc", lambda: _zero(nn.Linear(c.d_head, c.d_r)))
        build("node_q", lambda: nn.Linear(c.d_r, c.d_head))
        build("node_k", lambda: nn.Linear(c.d_head, c.d_head))
        build("node_val", lambda: nn.Linear(c.d_r, c.d_head))
        build("node_read", lambda: _zero(nn.Linear(c.d_head, c.d_r)))
        build("node_back", lambda: _zero(nn.Linear(c.d_head, c.d_r)))
        n_types = len(model.schema.by_key()["col:label_type"].answers) if "col:label_type" in \
            model.schema.by_key() else 1
        build("node_kind", lambda: nn.Linear(c.d_head, n_types))
    if flags["card_slots"] and c.asked_layers:
        build("pool_slot", lambda: _zero(nn.Linear(c.d, c.d_r)))
    model.graph = flags
    return flags


def graph_parameters(model) -> list[nn.Parameter]:
    """The gate group (loop._student_optimizer): link_mix and every module this
    file adds. All start at zero (or only matter through one that does) and
    open at the links stage at a constant rate."""
    ps = list(model.link_mix.parameters())
    for name in ("link_mix_h", "rel_msg", "node_table", "node_doc", "node_q", "node_k", "node_val",
                 "node_read", "node_back", "node_kind", "pool_slot"):
        m = getattr(model, name, None)
        if m is not None:
            ps += list(m.parameters())
    return ps


# ---------------------------------------------------------------- visibility
def visibility(times, n: int, dev, mode: str, cut: float | None = None) -> torch.Tensor:
    """[N, N] bool, allowed[i, j]: line i may read line j (diagonal on)."""
    if mode == "index":
        return torch.tril(torch.ones(n, n, dtype=torch.bool, device=dev))
    if mode == "view" and cut is None:
        return torch.ones(n, n, dtype=torch.bool, device=dev)
    # float64 on the CPU: float32 rounds arrival times together (MPS has no float64).
    t = torch.tensor(list(times)[:n], dtype=torch.float64)
    if mode == "arrival":
        allowed = t.view(1, -1) <= t.view(-1, 1)
    else:
        seen = torch.maximum(t, torch.tensor(float(cut), dtype=torch.float64))
        allowed = t.view(1, -1) <= seen.view(-1, 1)
    return (allowed | torch.eye(n, dtype=torch.bool)).to(dev)


def training_cut(times, p: float) -> float | None:
    """With probability p, a random earlier compile time (one of the view's
    arrival times before its last), else None (the whole view)."""
    if p <= 0 or torch.rand(()) >= p:
        return None
    ts = sorted(set(float(x) for x in times))
    if len(ts) < 2:
        return None
    return ts[int(torch.randint(0, len(ts) - 1, ()))]


# ---------------------------------------------------------------- link hop
def masked_softmax(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Softmax over the allowed entries; a row with none allowed is all zero.
    Finite in backward too: a softmax over a row of -inf returns NaN gradients
    even under nan_to_num, which poisoned every weight in the smoke."""
    any_ = mask.any(-1, keepdim=True)
    x = x.masked_fill(~mask, float("-inf"))
    x = torch.where(any_, x, torch.zeros_like(x))
    return x.softmax(-1) * any_


def link_weights(proposed: dict[str, torch.Tensor], nones: dict[str, torch.Tensor],
                 off: torch.Tensor) -> dict[str, torch.Tensor]:
    """Row i's distribution over the lines j it links to, with its abstain
    mass dropped: {relation: [N, N]}."""
    n = off.shape[0]
    out = {}
    for name, scores in proposed.items():
        ok = torch.isfinite(scores) & off
        logits = scores.float()
        none = nones.get(name)
        if none is None and name == "governs":
            none = logits.new_full((n,), GOVERNS_NONE)
        if none is not None:
            none = none.float().view(-1, 1)
            logits = torch.cat([logits, none], -1)
            ok = torch.cat([ok, torch.isfinite(none)], -1)
        out[name] = masked_softmax(logits, ok)[:, :n]
    return out


def hop(model, r: torch.Tensor, weights: dict[str, torch.Tensor], off: torch.Tensor) -> torch.Tensor:
    """One hop both ways. i reads what it links to; j reads the lines that
    link to it (where j may read them), normalised by their mass + 1, so a
    header with hundreds of rows reads their mean, not their sum."""
    back_ok = off.transpose(0, 1).to(torch.float32)
    rf = r.float()
    msgs, extra = [], 0
    for name, w in weights.items():
        fwd = w @ rf
        wb = w * back_ok
        bwd = (wb.transpose(0, 1) @ rf) / (wb.sum(0) + 1.0).unsqueeze(1)
        msgs += [fwd, bwd]
        if getattr(model, "rel_msg", None) is not None:
            extra = extra + model.rel_msg[f"{name}__out"](fwd.to(r.dtype)) \
                + model.rel_msg[f"{name}__back"](bwd.to(r.dtype))
    if not msgs:
        return r
    return r + model.link_mix(torch.stack(msgs).mean(0).to(r.dtype)) + extra


# ---------------------------------------------------------------- nodes
def _pool(score: torch.Tensor, member: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    w = masked_softmax(score.view(1, -1).float().expand(member.shape[0], -1), member)
    return w @ x.float()


def node_pass(model, out, r: torch.Tensor, inp: dict[str, Any], allowed: torch.Tensor) -> torch.Tensor:
    """Table and document nodes, read and written both ways; pointers from a
    line to another document's node. Fills ``out.nodes`` for the losses and
    serve; returns the updated r."""
    from .model import pair_tensor  # noqa: PLC0415 (cycle)

    n, dev = r.shape[0], r.device
    same_doc = pair_tensor(inp["same_doc"], dev) > 0
    same_sec = pair_tensor(inp["same_section"], dev) > 0
    table = same_sec & allowed
    doc = same_doc & allowed
    has_table = same_sec.any(1)
    has_doc = same_doc.any(1)
    s = model.pool_score(r).squeeze(-1)
    ft = model.pool_in(_pool(s, table, r).to(r.dtype))          # [N, d_head] the card's features
    fd = model.pool_in(_pool(s, doc, r).to(r.dtype))
    ft = ft * has_table.unsqueeze(1)
    fd = fd * has_doc.unsqueeze(1)
    node = torch.where(has_table.unsqueeze(1), ft, fd)          # the node a line belongs to
    # A node is represented once, by its first line: a pointer chooses among
    # nodes, not among their lines.
    group = torch.where(has_table.unsqueeze(1), same_sec, same_doc)
    idx = torch.arange(n, device=dev)
    rep = ~(group & (idx.view(1, -1) < idx.view(-1, 1))).any(1) & (has_table | has_doc)
    # Pointer: line i -> a node of another document it may read; abstain 0.
    ok = rep.view(1, -1) & ~same_doc & allowed & ~torch.eye(n, dtype=torch.bool, device=dev)
    sc = (model.node_q(r) @ model.node_k(node).transpose(0, 1)).float() / model.node_q.out_features ** 0.5
    sc = sc.masked_fill(~ok, float("-inf"))
    logp = torch.cat([sc, sc.new_zeros(n, 1)], -1).log_softmax(-1)  # last column: points at nothing
    w = logp[:, :n].exp()
    # The node reads what points at it; every member then reads its node.
    # A member reads the pointers at its node that it may itself read (the
    # mask holds here too: under index / arrival no later line leaks back).
    val = model.node_val(r).float()
    member = group & rep.view(1, -1)                                          # [line, its rep]
    c = (member.float() @ w.transpose(0, 1)) * allowed.float()                # [member, pointing line]
    told = (c @ val) / (c.sum(1) + 1.0).unsqueeze(1)
    r = (r + model.node_table(ft) + model.node_doc(fd) + model.node_read((w @ node.float()).to(r.dtype))
         + model.node_back(told.to(r.dtype)))
    out.nodes = {"table": ft, "doc": fd, "has_table": has_table, "rep": rep, "pointer": logp,
                 "kind": model.node_kind(ft)}
    return r


__all__ = ["GRAPH_DEFAULTS", "RUNE_GRAPH", "enable_graph", "graph_parameters", "hop", "link_weights",
           "node_pass", "parse_graph", "training_cut", "visibility"]
