"""C3: Content, Consequence, Conduct (docs/C3_HEADS.md), as code.

Forward pass for one deal (one sequence of atoms in time order)::

  atoms ─ AtomEncoder (text + doc kind + role/side + number magnitudes + time)
        ─ DealGraph, FORESIGHT mask (line i attends only to lines that entered
          no later than it) + structural biases (same document, same section)
        ─ h_i
           ├─ CONTENT      z_c = P_c(h_i)            company adversary (gradient reversal)
           ├─ CONSEQUENCE  q_i ~ mixture(h_i)        trained toward the EMA hindsight
           │                                         encoder that sees the whole deal
           ├─ CONTEXT      stage / geo slots, dropped to null at random
           └─ RATIONALE    r_i = R(z_c, E[q_i], ctx) + small penalized residual from z_c
                │
                ├─ universal described heads, one per space (folds.DescribedHead)
                ├─ relations: described pair scorer + boxes for `governs`
                ├─ absence head: per slot, "empty now, filled later" from q_i
                └─ CONDUCT: Σ_m α_k,m · A_m(r_i, q_i)  → company described head
                            α_k sparse per company, optionally seeded from the
                            company's own written rules

Universal outputs never read the company code, so they are bit-identical for
any company (tests/test_c3_model.py checks this).

Not built here, by design: the rationale writer (decodes text from r_i), the
neural program head (notes.extract_programs gives its weak targets), and the
value-of-information question engine (``sample_consequence`` is its input).
They are listed in ml/c3/README.md with what each needs first.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .explain import ExplanationBank, ExplanationReader
from .folds import DescribedHead, DescribedRelation
from .operators import ReasonCompiler, apply_reasons
from .schema import NUMBER, PAIR, RELATION, Schema
from .text import HashingEncoder, TagEmbedding

#: The claim slots the belief tracker and the absence head work over, with the
#: readings that fill each one. Universal: any services company tracks these.
CLAIM_SLOTS: dict[str, tuple[str, ...]] = {
    "sites": ("site", "location"),
    "quantity": ("equipment_qty",),
    "display_size": ("display_size",),
    "hours": ("labor_hours", "hours_stated"),
    "crew": ("crew_size", "tech_level"),
    "rate": ("rate", "rate_for"),
    "start_date": ("start_date",),
    "billing": ("billing_type",),
    "materials": ("material",),
    "cadence": ("cadence", "lead_time"),
}

#: Relations whose source must enter the deal no earlier than its target
#: (the answer comes after the question; a derived total after its inputs).
CAUSAL_RELATIONS = {"answers": 1, "derived_from": 1, "contradicts": 1}

#: What a line can change, per the WHY grammar ("what it changes in hours,
#: crew, sites, tasks or price"), plus the schedule. Directions: down, none, up.
CHANGE_SLOTS = ("hours", "crew", "sites", "price", "tasks", "schedule")
DIRECTIONS = ("down", "none", "up")


@dataclass
class C3Config:
    d_text: int = 256        # text encoder width (1024 for ModernBERT-large)
    d: int = 256             # deal graph width
    n_layers: int = 2        # 6 in the design
    n_heads: int = 4
    d_r: int = 128           # rationale latent
    d_q: int = 64            # consequence latent
    q_components: int = 4    # mixture components of p(q | past)
    d_head: int = 128        # width of each space's head input
    residual_dim: int = 64   # the bypass around r_i, penalized
    n_folds: int = 4
    n_policy_atoms: int = 8  # grow to 24 as profiles arrive (v3 risk 4)
    policy_rank: int = 8
    box_dim: int = 32
    dropout: float = 0.1
    context_dropout: float = 0.4
    ema: float = 0.996
    #: v5 explanation bank on by default in c3_loss (reader votes, compiled
    #: operators, clause slots). Off: v6 reads explanations with the
    #: language-model brain (brain.py); v5 stays as an ablation baseline.
    legacy_reasons: bool = False

    @staticmethod
    def design() -> "C3Config":
        """The sizes the design docs name (needs a GPU and HFEncoder)."""
        return C3Config(d_text=1024, d=1024, n_layers=6, n_heads=16, d_r=256, d_q=256,
                        q_components=8, d_head=256, n_policy_atoms=24)


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class GraphBlock(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.h = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(dropout)
        # Structural biases, one per attention head: same document, same section.
        self.w_doc = nn.Parameter(torch.zeros(heads))
        self.w_sec = nn.Parameter(torch.zeros(heads))

    def forward(self, x: torch.Tensor, allowed: torch.Tensor,
                same_doc: torch.Tensor, same_sec: torch.Tensor) -> torch.Tensor:
        n, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(n, 3, self.h, d // self.h).permute(1, 2, 0, 3)
        bias = (self.w_doc.view(-1, 1, 1) * same_doc + self.w_sec.view(-1, 1, 1) * same_sec)
        bias = bias.masked_fill(~allowed, float("-inf"))
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        x = x + self.drop(self.out(a.transpose(0, 1).reshape(n, d)))
        return x + self.drop(self.mlp(self.ln2(x)))


class DealEncoder(nn.Module):
    """Atom encoder + deal graph transformer. Copied as the EMA hindsight teacher."""

    def __init__(self, cfg: C3Config, text: nn.Module):
        super().__init__()
        self.text = text
        self.lift = nn.Linear(cfg.d_text, cfg.d)
        self.tags = TagEmbedding(cfg.d)
        self.nums = TagEmbedding(cfg.d)
        self.time = nn.Linear(2, cfg.d)
        self.ln = nn.LayerNorm(cfg.d)
        self.blocks = nn.ModuleList(GraphBlock(cfg.d, cfg.n_heads, cfg.dropout)
                                    for _ in range(cfg.n_layers))

    def embed_atoms(self, inp: dict[str, Any]) -> torch.Tensor:
        dev = self.lift.weight.device
        tags = [[f"doc:{k}", f"role:{r}", f"side:{s}"]
                for k, r, s in zip(inp["doc_kind"], inp["role"], inp["side"])]
        t = torch.tensor(inp["times"], dtype=torch.float32, device=dev)
        days = (t - t.min()).clamp(min=0) / 86400.0
        rank = torch.arange(len(t), device=dev, dtype=torch.float32) / max(1, len(t) - 1)
        time = torch.stack([torch.log1p(days), rank], -1)
        return self.ln(self.lift(self.text(inp["texts"])) + self.tags(tags)
                       + self.nums(inp["numbers"]) + self.time(time))

    def forward(self, inp: dict[str, Any], *, hindsight: bool = False) -> torch.Tensor:
        x = self.embed_atoms(inp)
        n, dev = x.shape[0], x.device
        same_doc = torch.tensor(inp["same_doc"], dtype=torch.float32, device=dev)
        same_sec = torch.tensor(inp["same_section"], dtype=torch.float32, device=dev)
        # Atoms arrive sorted by (entered_at, order): the foresight mask is
        # "j entered no later than i", which with ties in order is j <= i.
        allowed = (torch.ones(n, n, dtype=torch.bool, device=dev) if hindsight
                   else torch.tril(torch.ones(n, n, dtype=torch.bool, device=dev)))
        for blk in self.blocks:
            x = blk(x, allowed, same_doc, same_sec)
        return x


@dataclass
class C3Output:
    h: torch.Tensor
    z_c: torch.Tensor
    q_mu: torch.Tensor          # [N, K, d_q]
    q_logvar: torch.Tensor      # [N, K, d_q]
    q_logit: torch.Tensor       # [N, K]
    r: torch.Tensor
    residual: torch.Tensor
    logits: dict[str, torch.Tensor] = field(default_factory=dict)      # opportunity -> [N, A]
    numbers: dict[str, torch.Tensor] = field(default_factory=dict)     # opportunity -> [N]
    relations: dict[str, torch.Tensor] = field(default_factory=dict)   # relation -> [N, N]
    governs: torch.Tensor | None = None                                # [N, N] log P(j ⊂ i)
    absence: torch.Tensor | None = None                                # [N, slots] logits
    company_logits: torch.Tensor | None = None                         # adversary
    alpha: torch.Tensor | None = None
    explained: dict[str, torch.Tensor] = field(default_factory=dict)   # opportunity -> votes [N, A]
    attention: dict[str, torch.Tensor] = field(default_factory=dict)   # opportunity -> [N, M+1]
    gates: dict[str, torch.Tensor] = field(default_factory=dict)       # opportunity -> [N, M] (operators)
    reason_claims: dict[str, torch.Tensor] = field(default_factory=dict)  # layer -> [M, slots, 3]
    reason_index: dict[str, list[int]] = field(default_factory=dict)      # layer -> bank indices
    changes: torch.Tensor | None = None                                # [N, slots, 3] what each line changes
    pointers: torch.Tensor | None = None                               # [N, N+1] log p(reason rests on j); last = itself
    conduct_x: torch.Tensor | None = None                              # [N, d_head] the company layer's lines

    @property
    def q_mean(self) -> torch.Tensor:
        w = self.q_logit.softmax(-1).unsqueeze(-1)
        return (w * self.q_mu).sum(1)


class C3Model(nn.Module):
    def __init__(self, schema: Schema, cfg: C3Config | None = None,
                 text: nn.Module | None = None, companies: tuple[str, ...] = ("purtera",)):
        super().__init__()
        self.schema, self.cfg = schema, cfg or C3Config()
        c = self.cfg
        self.text = text or HashingEncoder(c.d_text)
        self.encoder = DealEncoder(c, self.text)
        self.teacher = None                    # built lazily by hindsight_targets()
        self.teacher_proj = None

        self.p_c = nn.Sequential(nn.Linear(c.d, c.d), nn.GELU(), nn.Linear(c.d, c.d))
        self.q_head = nn.Linear(c.d, c.q_components * (2 * c.d_q + 1))
        self.q_target = nn.Linear(c.d, c.d_q)  # projection of the hindsight state (EMA copy)
        self.ctx = TagEmbedding(c.d_r)
        self.ctx_null = nn.Parameter(torch.zeros(c.d_r))
        self.rationale = nn.Sequential(nn.Linear(c.d + c.d_q + c.d_r, c.d_r), nn.GELU(),
                                       nn.Linear(c.d_r, c.d_r))
        self.res_down = nn.Linear(c.d, c.residual_dim)
        self.res_up = nn.Linear(c.residual_dim, c.d_r, bias=False)
        self.why_proj = nn.Linear(c.d_text, c.d_r)       # the WHY, into rationale space

        spaces = sorted({o.space for o in schema.opportunities if o.universal})
        self.space_in = nn.ModuleDict({s: nn.Linear(c.d_r + (c.d_q if s == "consequence" else 0),
                                                    c.d_head) for s in spaces})
        self.heads = nn.ModuleDict({s: DescribedHead(c.d_text, c.d_head, c.n_folds) for s in spaces})
        self.relation_head = DescribedRelation(c.d_text, c.d_head, c.n_folds)
        self.rel_in = nn.Linear(c.d_r, c.d_head)
        self.box = nn.Linear(c.d_r, 2 * c.box_dim)
        self.absence = nn.Linear(c.d_q + c.d_r, len(CLAIM_SLOTS))

        # Conduct: the policy genome.
        m, k = c.n_policy_atoms, c.policy_rank
        d_in = c.d_r + c.d_q
        self.atom_u = nn.Parameter(torch.randn(m, d_in, k) * d_in ** -0.5)
        self.atom_v = nn.Parameter(torch.randn(m, k, c.d_head) * k ** -0.5)
        self.atom_gate = nn.Parameter(torch.randn(m, d_in) * d_in ** -0.5)
        self.atom_theta = nn.Parameter(torch.zeros(m))   # the readable constant per atom
        self.conduct_base = nn.Linear(d_in, c.d_head)
        self.conduct_head = DescribedHead(c.d_text, c.d_head, c.n_folds)
        self.alpha_from_text = nn.Linear(c.d_text, m)
        # Readers of written explanations (explain.py): one per layer, so a
        # company's rules can never reach a universal head.
        self.reader_universal = ExplanationReader(c.d_text, c.d_head)
        self.reader_company = ExplanationReader(c.d_text, c.d_head)
        # Reasons compiled into operators (operators.py), one compiler per
        # layer so company text never trains what the base uses.
        self.compiler_universal = ReasonCompiler(c.d_text, c.d_head, n_change_slots=len(CHANGE_SLOTS))
        self.compiler_company = ReasonCompiler(c.d_text, c.d_head, n_change_slots=len(CHANGE_SLOTS))
        # What a line changes (consequence.py), and r -> text space so the model
        # can voice its own predicted reason for the question engine (ask.py).
        self.changes_head = nn.Linear(c.d_q + c.d_r, len(CHANGE_SLOTS) * 3)
        self.r_to_text = nn.Linear(c.d_r, c.d_text)
        # What a line's reason rests on (v6, supercharge.py): which earlier
        # lines it points at, and which of its own words carry it. Taught
        # from the teacher's reading of the WHYs; run without them.
        self.cite_q = nn.Linear(c.d_r, c.d_head)
        self.cite_k = nn.Linear(c.d_r, c.d_head)
        self.cite_self = nn.Parameter(torch.zeros(1))
        self.word_r = nn.Linear(c.d_r, c.d_head)
        self.word_w = nn.Linear(c.d_text, c.d_head)
        # Judgment tabs asked about more than one line (judgment_heads.json):
        # two lines read as one symmetric pair, a document / table / sheet or
        # the whole deal read by attention pooling over its lines. Universal
        # questions pool r; the company's (project tier) pools the conduct
        # layer's lines, so the base never reads company state.
        self.pair_in = nn.Linear(3 * c.d_r, c.d_head)
        self.pool_score = nn.Linear(c.d_r, 1)
        self.pool_in = nn.Linear(c.d_r, c.d_head)
        self.pool_score_company = nn.Linear(c.d_head, 1)
        self.pool_in_company = nn.Linear(c.d_head, c.d_head)
        self.judgment_head = DescribedHead(c.d_text, c.d_head, c.n_folds)
        self.judgment_head_company = DescribedHead(c.d_text, c.d_head, c.n_folds)
        self.companies = list(companies)
        # Starting codes are small and dense so every atom gets a gradient; the
        # L1 term then makes them sparse. A zero code would starve the atoms.
        self.alpha = nn.Parameter(torch.full((len(companies), m), 0.1))
        self.adversary = nn.Sequential(nn.Linear(c.d, c.d // 2), nn.GELU(),
                                       nn.Linear(c.d // 2, max(2, len(companies))))

    # ------------------------------------------------------------ schema text
    def describe(self) -> dict[str, Any]:
        """Encode every opportunity and answer description (re-run each step:
        the encoder trains, so the descriptions move with it)."""
        opps = [o for o in self.schema.opportunities]
        texts = self.schema.texts()
        emb = self.text(texts)
        out, i = {}, 0
        for o in opps:
            out[o.key] = (emb[i], emb[i + 1:i + 1 + len(o.answers)])
            i += 1 + len(o.answers)
        return out

    def add_company(self, name: str, policy_text: str = "") -> int:
        """A new company: one sparse code, optionally seeded from its written rules."""
        if name in self.companies:
            return self.companies.index(name)
        with torch.no_grad():
            seed = torch.zeros(1, self.cfg.n_policy_atoms, device=self.alpha.device)
            if policy_text:
                seed = self.alpha_from_text(self.text([policy_text]))
            self.alpha = nn.Parameter(torch.cat([self.alpha.data, seed]))
        self.companies.append(name)
        return len(self.companies) - 1

    def freeze_for_new_company(self) -> None:
        """Everything frozen except the company codes (v3 section 2.3)."""
        for p in self.parameters():
            p.requires_grad_(False)
        self.alpha.requires_grad_(True)

    # ------------------------------------------------------------ forward
    def forward(self, inp: dict[str, Any], company: str | None = None,
                desc: dict[str, Any] | None = None, adv_lambda: float = 1.0,
                bank: ExplanationBank | None = None,
                bank_emb: torch.Tensor | None = None) -> C3Output:
        c = self.cfg
        h = self.encoder(inp)
        n = h.shape[0]
        z_c = self.p_c(h)
        q = self.q_head(h).view(n, c.q_components, 2 * c.d_q + 1)
        q_mu, q_logvar, q_logit = q[..., :c.d_q], q[..., c.d_q:2 * c.d_q].clamp(-6, 4), q[..., -1]
        q_mean = (q_logit.softmax(-1).unsqueeze(-1) * q_mu).sum(1)

        ctx = self._context(inp.get("context") or {}, n, h.device)
        residual = self.res_down(z_c)
        r = self.rationale(torch.cat([z_c, q_mean, ctx], -1)) + self.res_up(residual)
        out = C3Output(h=h, z_c=z_c, q_mu=q_mu, q_logvar=q_logvar, q_logit=q_logit,
                       r=r, residual=residual)

        desc = desc if desc is not None else self.describe()
        if bank_emb is None and bank is not None and len(bank):
            bank_emb = self.text([e.text for e in bank.items])
        self.universal_heads(out, r, q_mean, desc, bank, bank_emb)

        # Links: described pair scores, boxes for governs.
        rels = [o for o in self.schema.select(kind=RELATION) if o.field != "governs"]
        if rels:
            x = self.rel_in(r)
            scores = self.relation_head(x, torch.stack([desc[o.key][0] for o in rels]))
            t = torch.arange(n, device=h.device)
            for o, s in zip(rels, scores):
                if CAUSAL_RELATIONS.get(o.field):
                    s = s.masked_fill(t.view(-1, 1) < t.view(1, -1), float("-inf"))
                out.relations[o.field] = s.masked_fill(torch.eye(n, dtype=torch.bool,
                                                                 device=h.device), float("-inf"))
        out.governs = self._box_containment(r)
        out.absence = self.absence(torch.cat([q_mean, r], -1))
        out.changes = self.changes_head(torch.cat([q_mean, r], -1)).view(n, len(CHANGE_SLOTS), 3)
        out.company_logits = self.adversary(GradReverse.apply(z_c, adv_lambda))
        out.pointers = self._pointers(r)

        if company is not None:
            self.conduct(out, company, q_mean, desc, bank, bank_emb)
        return out

    def universal_heads(self, out: C3Output, r: torch.Tensor, q_mean: torch.Tensor,
                        desc: dict[str, Any], bank: ExplanationBank | None = None,
                        bank_emb: torch.Tensor | None = None) -> None:
        """Every universal opportunity, read from r (and q for consequence).

        Separate so the WHY-sufficiency loss can run the same heads with the
        encoded WHY standing in for r."""
        for space, head in self.heads.items():
            opps = [o for o in self.schema.select(layer="universal", space=space)
                    if o.kind != RELATION]
            if not opps:
                continue
            x = r if space != "consequence" else torch.cat([r, q_mean], -1)
            x = self.space_in[space](x)
            logits, nums = head(x, torch.stack([desc[o.key][0] for o in opps]),
                                [desc[o.key][1] for o in opps])
            for j, (o, lg) in enumerate(zip(opps, logits)):
                out.logits[o.key] = lg
                if o.kind == NUMBER:
                    out.numbers[o.key] = nums[:, j]
            self._read(out, x, opps, desc, bank, bank_emb, "universal", "",
                       self.reader_universal, self.compiler_universal, head)

    def _read(self, out: C3Output, x: torch.Tensor, opps: list, desc: dict[str, Any],
              bank: ExplanationBank | None, bank_emb: torch.Tensor | None,
              layer: str, company: str, reader: ExplanationReader,
              compiler: ReasonCompiler, head: DescribedHead) -> None:
        """Let this layer's explanations act on each head: compiled operators
        move the lines before the head reads them, and the reader's votes are
        added on top. With no explanations in the layer, nothing changes."""
        if bank is None or bank_emb is None:
            return
        idx = bank.select(layer, company)
        if not idx:
            return
        exps = [bank.items[i] for i in idx]
        emb = bank_emb[idx]
        ops = compiler.compile(emb, texts=[e.text for e in exps], encoder=self.text)
        out.reason_claims[layer] = ops["claims"]
        out.reason_index[layer] = idx
        for o in opps:
            answers = [a.value for a in o.answers]
            votes, att = reader(x, desc[o.key][0], desc[o.key][1], o.key, answers, emb, exps)
            moved, g = apply_reasons(compiler, head, x, desc[o.key][0], desc[o.key][1],
                                     o.key, answers, ops, exps)
            out.explained[o.key] = votes
            out.attention[o.key] = att
            out.gates[o.key] = g
            out.logits[o.key] = moved + votes

    def conduct(self, out: C3Output, company: str, q_mean: torch.Tensor,
                desc: dict[str, Any], bank: ExplanationBank | None = None,
                bank_emb: torch.Tensor | None = None) -> None:
        k = self.companies.index(company) if company in self.companies else self.add_company(company)
        alpha = self.alpha[k]
        x = torch.cat([out.r, q_mean], -1)
        # A_m(x) = gate_m(x) · (x U_m) V_m ; gate_m = sigmoid(<g_m, x> - θ_m)
        gate = torch.sigmoid(x @ self.atom_gate.T - self.atom_theta)            # [N, M]
        low = torch.einsum("nd,mdk->nmk", x, self.atom_u)
        shift = torch.einsum("nmk,mkh->nmh", low, self.atom_v)                 # [N, M, H]
        y = self.conduct_base(x) + (alpha.view(1, -1, 1) * gate.unsqueeze(-1) * shift).sum(1)
        opps = self.schema.select(layer="company")
        opps = [o for o in opps if o.kind != RELATION]
        logits, nums = self.conduct_head(y, torch.stack([desc[o.key][0] for o in opps]),
                                         [desc[o.key][1] for o in opps])
        for j, (o, lg) in enumerate(zip(opps, logits)):
            out.logits[o.key] = lg
            if o.kind == NUMBER:
                out.numbers[o.key] = nums[:, j]
        self._read(out, y, opps, desc, bank, bank_emb, "company", company,
                   self.reader_company, self.compiler_company, self.conduct_head)
        out.alpha = alpha
        out.conduct_x = y

    def judge(self, out: C3Output, key: str, subjects: list[tuple[int, ...]],
              desc: dict[str, Any] | None = None) -> torch.Tensor:
        """Logits [len(subjects), A] for a judgment asked about two lines, a
        group of lines or the whole deal. A pair is (i, j); a group or the deal
        is its lines. A company question needs ``model(..., company=...)``."""
        o = self.schema.by_key()[key]
        desc = desc if desc is not None else self.describe()
        if o.universal:
            x, score, proj, head = out.r, self.pool_score, self.pool_in, self.judgment_head
        else:
            if out.conduct_x is None:
                raise ValueError(f"{key} is a company question: run the model with company=")
            x, score, proj, head = out.conduct_x, self.pool_score_company, self.pool_in_company, \
                self.judgment_head_company
        rows = []
        for lines in subjects:
            idx = torch.tensor(lines, device=x.device, dtype=torch.long)
            if o.size == PAIR:
                a, b = x[idx[0]], x[idx[1]]
                rows.append(self.pair_in(torch.cat([a + b, (a - b).abs(), a * b], -1)))
            else:
                xs = x[idx]
                w = score(xs).softmax(0)
                rows.append(proj((w * xs).sum(0)))
        logits, _ = head(torch.stack(rows), desc[key][0].unsqueeze(0), [desc[key][1]])
        return logits[0]

    def _context(self, context: dict[str, list], n: int, dev) -> torch.Tensor:
        """Stage and geo slots. Each is a value or null; in training a value is
        dropped to null with probability context_dropout, so the model never
        depends on it."""
        groups = []
        for i in range(n):
            g = []
            for slot, values in context.items():
                v = values[i] if i < len(values) else None
                if v and not (self.training and torch.rand(()) < self.cfg.context_dropout):
                    g.append(f"{slot}:{v}")
            groups.append(g)
        emb = self.ctx(groups)
        empty = torch.tensor([not g for g in groups], device=dev).unsqueeze(-1)
        return torch.where(empty, self.ctx_null.expand(n, -1), emb)

    def _pointers(self, r: torch.Tensor) -> torch.Tensor:
        """log p(line i's reason rests on earlier line j), plus a last column
        for "on the line itself". Foresight: only j < i."""
        n = r.shape[0]
        s = self.cite_q(r) @ self.cite_k(r).T / self.cite_q.out_features ** 0.5
        later = torch.triu(torch.ones(n, n, dtype=torch.bool, device=r.device))
        s = s.masked_fill(later, float("-inf"))
        return torch.cat([s, self.cite_self.expand(n, 1)], -1).log_softmax(-1)

    def word_weights(self, out: "C3Output", texts: list[str]) -> list[torch.Tensor | None]:
        """log p(word w of line i carries its reason), over text.word_spans."""
        from .text import word_spans  # noqa: PLC0415

        res: list[torch.Tensor | None] = []
        q = self.word_r(out.r)
        for i, t in enumerate(texts):
            ws = [w for w, _, _ in word_spans(t)]
            if not ws:
                res.append(None)
                continue
            k = self.word_w(self.text(ws))
            res.append((k @ q[i] / k.shape[-1] ** 0.5).log_softmax(-1))
        return res

    def _box_containment(self, r: torch.Tensor) -> torch.Tensor:
        """log P(box_j ⊂ box_i): a parent's box contains its children's, so
        governs is transitive by construction (v3 section 2.5)."""
        c, o = self.box(r).chunk(2, -1)
        lo, hi = c - F.softplus(o), c + F.softplus(o)
        inter = F.softplus(torch.minimum(hi.unsqueeze(1), hi.unsqueeze(0))
                           - torch.maximum(lo.unsqueeze(1), lo.unsqueeze(0)))
        child = F.softplus(hi - lo).unsqueeze(0)
        score = (torch.log(inter + 1e-9) - torch.log(child + 1e-9)).sum(-1)   # [i parent, j child]
        return score.masked_fill(torch.eye(r.shape[0], dtype=torch.bool, device=r.device),
                                 float("-inf"))

    # ------------------------------------------------------------ hindsight
    @torch.no_grad()
    def hindsight_targets(self, inp: dict[str, Any]) -> torch.Tensor:
        """The EMA teacher reads the WHOLE deal; its latent per line is what
        q_i (which saw only the past) is trained to predict."""
        if self.teacher is None:
            self.teacher = copy.deepcopy(self.encoder).requires_grad_(False)
            self.teacher_proj = copy.deepcopy(self.q_target).requires_grad_(False)
        self.teacher.eval()
        return self.teacher_proj(self.teacher(inp, hindsight=True))

    @torch.no_grad()
    def ema_update(self) -> None:
        if self.teacher is None:
            return
        m = self.cfg.ema
        for t, s in zip(self.teacher.parameters(), self.encoder.parameters()):
            t.mul_(m).add_(s.detach(), alpha=1 - m)
        for t, s in zip(self.teacher_proj.parameters(), self.q_target.parameters()):
            t.mul_(m).add_(s.detach(), alpha=1 - m)

    @torch.no_grad()
    def sample_consequence(self, out: C3Output, k: int = 32) -> torch.Tensor:
        """K samples of q_i per line [K, N, d_q]: the question engine's input."""
        comp = torch.distributions.Categorical(logits=out.q_logit).sample((k,))     # [K, N]
        idx = comp.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, out.q_mu.shape[-1])
        mu = out.q_mu.unsqueeze(0).expand(k, -1, -1, -1).gather(2, idx).squeeze(2)
        lv = out.q_logvar.unsqueeze(0).expand(k, -1, -1, -1).gather(2, idx).squeeze(2)
        return mu + torch.randn_like(mu) * (0.5 * lv).exp()

    def parameter_report(self) -> dict[str, int]:
        groups = {"text": self.text, "deal_graph": self.encoder.blocks,
                  "content": self.p_c, "consequence": self.q_head,
                  "rationale": self.rationale, "universal_heads": self.heads,
                  "links": nn.ModuleList([self.relation_head, self.rel_in, self.box]),
                  "conduct": nn.ModuleList([self.conduct_base, self.conduct_head,
                                            self.alpha_from_text])}
        out = {k: sum(p.numel() for p in m.parameters()) for k, m in groups.items()}
        out["policy_atoms"] = sum(p.numel() for p in (self.atom_u, self.atom_v,
                                                      self.atom_gate, self.atom_theta))
        out["total"] = sum(p.numel() for p in self.parameters())
        return out


def log_gauss_mixture(x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor,
                      logit: torch.Tensor) -> torch.Tensor:
    """log p(x) under a diagonal Gaussian mixture. x [N, d]; mu, logvar [N, K, d]."""
    x = x.unsqueeze(1)
    comp = -0.5 * (((x - mu) ** 2) / logvar.exp() + logvar + math.log(2 * math.pi)).sum(-1)
    return torch.logsumexp(logit.log_softmax(-1) + comp, -1)
