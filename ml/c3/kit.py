"""Deal kit training. Smaller than the atom model, and it starts where that model stops.

The atom model reads the deal and says what each line is. This model never
re-decides that. It sees the atoms from before the kit was attached — the
same cut a live quote has — and uses each atom in the section its type
already earned:

* a ``physical_site`` atom can become a site on the tab, and can carry the
  PC hours and travel days the documents do not state;
* a ``task`` atom can become a labor line, and can carry the hours;
* a ``bom_line`` atom can become a hardware row, and can carry the quantity.

Every one of those atoms is a token. The live base freezes ModernBERT-large,
the same encoder the atom model holds, and trains a slot reader on top.
When one atom is decided, every other atom stays in the attention, tagged
by what it is to this one: evidence, the same fact, a lookalike, another
atom of its section, or the section it belongs to. That tag is the
explanation the tab does not write down. Attention can pick the site out of
a long deal, so a pile of tasks cannot drown it.

A section the tab filled is a contrast. The atoms on it were quoted; the
other atoms of that section were left off. A section the tab left empty
teaches nothing, so a kit with no hardware is not a lesson that hardware is
never quoted. A site the tab kept is also a positive ``job_site`` mark for
the atom model's site head. Leaving a place off the quote is not a claim
that it is not a site, and no negative mark is written for that head.

Billing has no atom. It reads each present section once, plus the other
training kits. A reason sentence is stored with the line and is not an
input. The five published kits named in ``KIT_HOLDOUT`` never train.
"""
from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .data import is_deal_kit
from .text import HashingEncoder

KIT_HOLDOUT = ("000051", "010095", "010164", "010253", "010287")

SITE_TYPES = frozenset({"physical_site"})
LABOR_TYPES = frozenset({"task"})
HARDWARE_TYPES = frozenset({"bom_line"})
SECTIONS = ("sites", "labor", "hardware", "other")
QUOTE_SECTIONS = ("sites", "labor", "hardware")
BILLING = ("fixed", "t_and_m", "milestone", "per_site")
#: A link is the reason the tab never wrote. These three stay apart so a
#: lookalike is not averaged into the evidence that explains a number.
EVIDENCE_RELATIONS = frozenset({
    "supports", "governs", "context", "derived_from", "answers",
    "blocked_by", "triggered_by",
})
SAME_RELATIONS = frozenset({"same_as"})
DIFFER_RELATIONS = frozenset({"near_miss", "contradicts"})
#: Pair tags the reader adds to attention. ``self`` is the atom being decided.
SLOTS = ("self", "evidence", "same", "differ", "sibling", "sites", "labor", "hardware", "other")
SLOT_INDEX = {name: i for i, name in enumerate(SLOTS)}
#: The live kit base. The encoder is ModernBERT-large and stays frozen.
KIT_ENCODER = "answerdotai/ModernBERT-large"
KIT_TEXT = 1024
KIT_D = 256
KIT_LAYERS = 4
KIT_HEADS = 8
_IGNORE = float("nan")


def section_for(atom_type: str) -> str:
    """Which tab section this atom type can feed. Anything else is background."""
    name = str(atom_type or "").strip().lower()
    if name in SITE_TYPES:
        return "sites"
    if name in LABOR_TYPES:
        return "labor"
    if name in HARDWARE_TYPES:
        return "hardware"
    return "other"


def _when(value: str | None) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def atom_in_context(atom: dict[str, Any], *, attached_at: str | None) -> bool:
    """An atom the kit may read: before the kit was attached, and not the kit.

    ``reader_scope.deal_kit.visible`` is the exporter's already-applied cut
    and wins when it is present. Without it, the kit's own attachment time
    is the cut. A deal-kit document is never context.
    """
    if is_deal_kit(str(atom.get("doc_kind") or "")):
        return False
    scope = atom.get("reader_scope")
    if isinstance(scope, dict) and "deal_kit" in scope:
        block = scope.get("deal_kit")
        if isinstance(block, dict) and "visible" in block:
            return bool(block.get("visible"))
    attached = _when(attached_at)
    entered = _when(atom.get("entered_at"))
    if attached is not None and entered is not None and entered >= attached:
        return False
    return True


def lock_kit_split(train_ids: Iterable[str], holdout_ids: Iterable[str] = ()) -> None:
    """The fixed five, and any caller holdout, stay out of the training ids."""
    train = {str(i) for i in train_ids}
    blocked = set(KIT_HOLDOUT) | {str(i) for i in holdout_ids}
    if train & blocked:
        raise ValueError("Holdout kit is in the training set")


@dataclass(frozen=True)
class ContextAtom:
    key: str
    text: str
    type: str
    section: str
    atom_id: str
    entered_at: str = ""


@dataclass(frozen=True)
class KitLine:
    section: str
    atom_key: str | None
    hours: float | None = None
    pc_hours: float | None = None
    travel_days: float | None = None
    qty: float | None = None
    reason: str = ""


@dataclass(frozen=True)
class DecisionRead:
    """What one atom's quote decision is allowed to see.

    ``evidence``, ``same`` and ``differ`` are the linked atoms, split by what
    the link says. ``siblings`` are the other atoms of this section. ``channels``
    is one group per other section. Together with this atom they are every
    context atom, each in one place.
    """
    key: str
    evidence: tuple[str, ...]
    same: tuple[str, ...]
    differ: tuple[str, ...]
    siblings: tuple[str, ...]
    channels: tuple[tuple[str, tuple[str, ...]], ...]


@dataclass
class KitExample:
    deal_id: str
    atoms: tuple[ContextAtom, ...]
    lines: tuple[KitLine, ...]
    billing: str | None = None
    dropped_keys: tuple[str, ...] = ()
    attached_at: str | None = None
    reasons: tuple[str, ...] = field(default_factory=tuple)
    links: tuple[tuple[str, str, str], ...] = ()

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "KitExample":
        attached = payload.get("attached_at")
        atoms = []
        for raw in payload.get("atoms") or []:
            if not isinstance(raw, dict) or not atom_in_context(raw, attached_at=attached):
                continue
            key = str(raw.get("key") or raw.get("label_key") or "").strip()
            if not key:
                raise ValueError("A context atom needs a key")
            kind = str(raw.get("type") or raw.get("label_type") or raw.get("atom_type") or "")
            atoms.append(ContextAtom(
                key=key, text=str(raw.get("text") or ""), type=kind,
                section=section_for(kind),
                atom_id=str(raw.get("atom_id") or key),
                entered_at=str(raw.get("entered_at") or ""),
            ))
        by_key = {a.key: a for a in atoms}
        lines = []
        reasons = []
        for raw in payload.get("lines") or []:
            if not isinstance(raw, dict):
                continue
            section = str(raw.get("section") or "")
            if section not in ("sites", "labor", "hardware", "materials"):
                raise ValueError(f"Unknown kit section {section!r}")
            atom_key = str(raw.get("atom_key") or "").strip() or None
            if atom_key is not None:
                host = by_key.get(atom_key)
                if host is None:
                    atom_key = None
                elif host.section != section:
                    raise ValueError(
                        f"A {section} line points at a {host.section} atom"
                    )
            reason = str(raw.get("reason") or "")
            if reason:
                reasons.append(reason)
            lines.append(KitLine(
                section=section, atom_key=atom_key,
                hours=_num(raw.get("hours")), pc_hours=_num(raw.get("pc_hours")),
                travel_days=_num(raw.get("travel_days")), qty=_num(raw.get("qty")),
                reason=reason,
            ))
        billing = str(payload.get("billing") or "").strip().lower() or None
        if billing is not None and billing not in BILLING:
            raise ValueError(f"Unknown billing {billing!r}")
        return KitExample(
            deal_id=str(payload.get("deal_id") or ""),
            atoms=tuple(atoms), lines=tuple(lines), billing=billing,
            dropped_keys=tuple(str(k) for k in (payload.get("dropped_keys") or []) if str(k)),
            attached_at=str(attached) if attached else None,
            reasons=tuple(reasons),
            links=_links(payload, {a.key for a in atoms}),
        )

    def context_texts(self) -> tuple[str, ...]:
        """Model inputs: the context atoms only. Reasons are not among them."""
        return tuple(a.text for a in self.atoms)


def _link_rows(payload: dict[str, Any]) -> list[Any]:
    rows = list(payload.get("links") or payload.get("edges") or [])
    for raw in payload.get("atoms") or []:
        if not isinstance(raw, dict):
            continue
        src = str(raw.get("key") or raw.get("label_key") or "").strip()
        for link in raw.get("links") or []:
            if isinstance(link, dict) and src:
                rows.append({**link, "src": src})
    return rows


def _links(payload: dict[str, Any], keys: set[str]) -> tuple[tuple[str, str, str], ...]:
    """Edges whose both ends survived the attachment cut. A link into the kit, or into a later note, is gone."""
    out = []
    seen = set()
    for item in _link_rows(payload):
        if not isinstance(item, dict):
            continue
        src = str(item.get("src") or item.get("from") or item.get("from_key") or "").strip()
        dst = str(item.get("dst") or item.get("to") or item.get("to_label_key") or "").strip()
        rel = str(item.get("relation") or "").strip().lower()
        if not src or not dst or src == dst or src not in keys or dst not in keys:
            continue
        rec = (src, dst, rel)
        if rec in seen:
            continue
        seen.add(rec)
        out.append(rec)
    return tuple(out)


def link_bucket(relation: str) -> str:
    """Where a linked atom sits. An unnamed relation stays with the evidence, so it is still read."""
    if relation in SAME_RELATIONS:
        return "same"
    if relation in DIFFER_RELATIONS:
        return "differ"
    return "evidence"


def other_sections(section: str) -> tuple[str, ...]:
    return tuple(s for s in SECTIONS if s != section)


def decision_reads(example: KitExample) -> tuple[DecisionRead, ...]:
    """One read per context atom. Nothing in context is skipped, and nothing is read twice."""
    by_section: dict[str, list[str]] = {s: [] for s in SECTIONS}
    for atom in example.atoms:
        by_section[atom.section].append(atom.key)
    placed = {a.key: set() for a in example.atoms}
    buckets = {a.key: {"evidence": [], "same": [], "differ": []} for a in example.atoms}
    for src, dst, rel in example.links:
        bucket = link_bucket(rel)
        for src_key, dst_key in ((src, dst), (dst, src)):
            if dst_key in placed[src_key]:
                continue
            placed[src_key].add(dst_key)
            buckets[src_key][bucket].append(dst_key)
    reads = []
    for atom in example.atoms:
        linked = placed[atom.key]
        siblings = tuple(k for k in by_section[atom.section] if k != atom.key and k not in linked)
        channels = tuple(
            (section, tuple(k for k in by_section[section] if k not in linked))
            for section in other_sections(atom.section)
        )
        group = buckets[atom.key]
        reads.append(DecisionRead(
            key=atom.key, evidence=tuple(group["evidence"]), same=tuple(group["same"]),
            differ=tuple(group["differ"]), siblings=siblings, channels=channels,
        ))
    return tuple(reads)


def slot_matrix(example: KitExample) -> list[list[str]]:
    """Role of atom ``j`` when the reader decides atom ``i``. The diagonal is ``self``.

    Every other atom has exactly one tag, so the attention sees the whole
    context and still knows what each atom is to this decision.
    """
    reads = {read.key: read for read in decision_reads(example)}
    keys = [atom.key for atom in example.atoms]
    pos = {key: i for i, key in enumerate(keys)}
    grid = [["self"] * len(keys) for _ in keys]
    for key in keys:
        read = reads[key]
        assigned: dict[str, str] = {}
        for other in read.evidence:
            assigned[other] = "evidence"
        for other in read.same:
            assigned[other] = "same"
        for other in read.differ:
            assigned[other] = "differ"
        for other in read.siblings:
            assigned[other] = "sibling"
        for section, members in read.channels:
            for other in members:
                assigned[other] = section
        row = grid[pos[key]]
        for other, slot in assigned.items():
            row[pos[other]] = slot
    return grid


def section_vote(example: KitExample) -> dict[str, float]:
    """Each section that has an atom counts once, however many atoms it holds."""
    present = [s for s in SECTIONS if any(a.section == s for a in example.atoms)]
    if not present:
        return {}
    weight = 1.0 / len(present)
    return {section: weight for section in present}


def _num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def site_judgments(example: KitExample) -> list[dict[str, Any]]:
    """Positive site-head marks for atoms the tab kept.

    Append these to the deal's judgments before the atom model trains. A
    site atom with no kit line is absent here on purpose.
    """
    kept = {ln.atom_key for ln in example.lines if ln.section == "sites" and ln.atom_key}
    by_key = {a.key: a for a in example.atoms}
    out = []
    for key in sorted(kept):
        atom = by_key.get(key)
        if atom is None or atom.section != "sites":
            continue
        out.append({
            "head": "site_role",
            "verdict": "job_site",
            "labeler": "max@purtera-it.com",
            "target_key": f"site:{atom.atom_id}",
            "target": {"site": {"evidence": {"atomId": atom.atom_id}}},
            "source": "deal_kit",
        })
    return out


def billing_history(examples: Sequence[KitExample], deal_id: str,
                    holdout_ids: Iterable[str] = ()) -> torch.Tensor:
    """Mean billing of the other training kits. This deal and the holdout stay out."""
    blocked = {str(deal_id)} | {str(i) for i in holdout_ids} | set(KIT_HOLDOUT)
    rows = []
    for ex in examples:
        if ex.deal_id in blocked or ex.billing not in BILLING:
            continue
        row = torch.zeros(len(BILLING))
        row[BILLING.index(ex.billing)] = 1.0
        rows.append(row)
    if not rows:
        return torch.zeros(len(BILLING))
    return torch.stack(rows).mean(dim=0)


class SlotBlock(nn.Module):
    """One attention layer. The pair tag is added to the score, so a lookalike
    is not read the same way as the evidence that explains a number."""

    def __init__(self, d: int, heads: int, n_slots: int, dropout: float) -> None:
        super().__init__()
        if d % heads:
            raise ValueError("Kit width must divide across heads")
        self.h = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.slot = nn.Embedding(n_slots, d)
        nn.init.normal_(self.slot.weight, std=0.02)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, role: torch.Tensor) -> torch.Tensor:
        n, d = x.shape
        dh = d // self.h
        q, k, v = self.qkv(self.ln1(x)).view(n, 3, self.h, dh).permute(1, 2, 0, 3)
        scale = dh ** -0.5
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale
        slot = self.slot(role).view(n, n, self.h, dh).permute(2, 0, 1, 3)
        scores = scores + (q.unsqueeze(2) * slot).sum(-1) * scale
        attn = self.drop(torch.softmax(scores, dim=-1))
        mixed = torch.matmul(attn, v).transpose(0, 1).reshape(n, d)
        x = x + self.drop(self.out(mixed))
        return x + self.drop(self.mlp(self.ln2(x)))


class KitModel(nn.Module):
    """Frozen language features, then a slot reader over every pre-kit atom.

    ``KitModel()`` is the CPU stand-in: a hashing encoder that trains.
    ``live_kit`` swaps in ModernBERT-large and freezes it, so the reader
    learns the tab on top of the language base. Number heads start at zero,
    so a fresh model predicts no hours until a saved tab teaches them.
    """

    def __init__(self, encoder: nn.Module | None = None, *, d: int = 32, n_layers: int = 1,
                 n_heads: int = 4, dropout: float = 0.0, freeze_encoder: bool = False) -> None:
        super().__init__()
        self.encoder = encoder if encoder is not None else HashingEncoder(dim=d)
        d_text = int(self.encoder.dim)
        self.lift = nn.Linear(d_text, d)
        self.section_emb = nn.Embedding(len(SECTIONS), d)
        nn.init.normal_(self.section_emb.weight, std=0.02)
        self.blocks = nn.ModuleList(SlotBlock(d, n_heads, len(SLOTS), dropout) for _ in range(n_layers))
        self.section_q = nn.Parameter(torch.empty(len(SECTIONS), d))
        nn.init.normal_(self.section_q, std=0.02)
        self.keep = nn.Linear(d, 1)
        self.hours = nn.Linear(d, 1)
        self.pc_hours = nn.Linear(d, 1)
        self.travel_days = nn.Linear(d, 1)
        self.qty = nn.Linear(d, 1)
        self.materials = nn.Linear(d, 2)
        self.billing = nn.Linear(d + len(BILLING), len(BILLING))
        for head in (self.hours, self.pc_hours, self.travel_days, self.qty, self.materials):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self._freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            self.encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self._freeze_encoder:
            self.encoder.eval()
        return self

    def _atoms(self, example: KitExample) -> torch.Tensor:
        texts = list(example.context_texts())
        if not texts:
            return self.lift.weight.new_zeros((0, self.lift.out_features))
        frozen = self._freeze_encoder or not any(p.requires_grad for p in self.encoder.parameters())
        with torch.no_grad() if frozen else nullcontext():
            raw = self.encoder(texts)
        h = self.lift(raw.to(dtype=self.lift.weight.dtype))
        sections = torch.tensor([SECTIONS.index(a.section) for a in example.atoms],
                                device=h.device, dtype=torch.long)
        return h + self.section_emb(sections)

    def _read(self, h: torch.Tensor, example: KitExample) -> torch.Tensor:
        grid = slot_matrix(example)
        role = torch.tensor([[SLOT_INDEX[name] for name in row] for row in grid],
                            device=h.device, dtype=torch.long)
        for block in self.blocks:
            h = block(h, role)
        return h

    def _deal_vector(self, h: torch.Tensor, example: KitExample) -> torch.Tensor:
        """One attended summary per section, then each present section counts once."""
        n, d = h.shape
        mask = torch.zeros((len(SECTIONS), n), device=h.device, dtype=torch.bool)
        for i, atom in enumerate(example.atoms):
            mask[SECTIONS.index(atom.section), i] = True
        scores = (self.section_q.to(h.dtype) @ h.T) * (d ** -0.5)
        empty = ~mask.any(dim=1)
        scores = scores.masked_fill(~mask, float("-inf")).masked_fill(empty.unsqueeze(1), 0.0)
        weights = torch.softmax(scores, dim=-1).masked_fill(empty.unsqueeze(1), 0.0)
        pooled = weights @ h
        acc = h.new_zeros(d)
        vote = section_vote(example)
        for i, section in enumerate(SECTIONS):
            if section in vote:
                acc = acc + pooled[i] * vote[section]
        return acc

    def forward(self, example: KitExample, history: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self._atoms(example)
        if h.shape[0] == 0:
            deal = self.keep.weight.new_zeros(self.keep.in_features)
            empty = deal.new_zeros(0)
            hist = history.to(device=deal.device, dtype=deal.dtype).reshape(-1)
            return {
                "keep": empty, "hours": empty, "pc_hours": empty,
                "travel_days": empty, "qty": empty,
                "materials": self.materials(deal),
                "billing": self.billing(torch.cat([deal, hist])),
            }
        h = self._read(h, example)
        deal = self._deal_vector(h, example)
        hist = history.to(device=deal.device, dtype=deal.dtype).reshape(-1)
        return {
            "keep": self.keep(h).squeeze(-1),
            "hours": self.hours(h).squeeze(-1),
            "pc_hours": self.pc_hours(h).squeeze(-1),
            "travel_days": self.travel_days(h).squeeze(-1),
            "qty": self.qty(h).squeeze(-1),
            "materials": self.materials(deal),
            "billing": self.billing(torch.cat([deal, hist])),
        }


def _targets(example: KitExample) -> dict[str, list[float]]:
    n = len(example.atoms)
    keep = [_IGNORE] * n
    hours = [_IGNORE] * n
    pc = [_IGNORE] * n
    travel = [_IGNORE] * n
    qty = [_IGNORE] * n
    index = {a.key: i for i, a in enumerate(example.atoms)}
    # A filled section is the contrast: on the tab, or left off it. An empty
    # section stays ignored, so "no hardware rows" is not a lesson.
    filled = {ln.section for ln in example.lines if ln.section in QUOTE_SECTIONS and ln.atom_key}
    for i, atom in enumerate(example.atoms):
        if atom.section in filled:
            keep[i] = 0.0
    for key in example.dropped_keys:
        i = index.get(key)
        if i is not None and example.atoms[i].section in QUOTE_SECTIONS:
            keep[i] = 0.0
    for line in example.lines:
        if line.atom_key is None:
            continue
        i = index.get(line.atom_key)
        if i is None:
            continue
        keep[i] = 1.0
        if line.section == "labor" and line.hours is not None:
            hours[i] = line.hours
        if line.section == "sites" and line.pc_hours is not None:
            pc[i] = line.pc_hours
        if line.section == "sites" and line.travel_days is not None:
            travel[i] = line.travel_days
        if line.section == "hardware" and line.qty is not None:
            qty[i] = line.qty
    return {"keep": keep, "hours": hours, "pc_hours": pc, "travel_days": travel, "qty": qty}


def _masked_l1(pred: torch.Tensor, target: list[float]) -> torch.Tensor:
    if pred.numel() == 0:
        return pred.new_zeros(())
    gold = pred.new_tensor(target)
    mask = torch.isfinite(gold)
    if not bool(mask.any()):
        return pred.new_zeros(())
    return F.smooth_l1_loss(pred[mask], torch.log1p(gold[mask].clamp(min=0)))


def kit_loss(model: KitModel, example: KitExample, history: torch.Tensor) -> torch.Tensor:
    """Saved tab against the pre-kit atoms. Reason text is not read."""
    pred = model(example, history)
    tgt = _targets(example)
    loss = pred["billing"].new_zeros(())
    keep_gold = pred["keep"].new_tensor([0.0 if v != v else v for v in tgt["keep"]]) if pred["keep"].numel() else None
    if keep_gold is not None and pred["keep"].numel():
        finite = torch.tensor([v == v for v in tgt["keep"]], device=pred["keep"].device)
        if bool(finite.any()):
            loss = loss + F.binary_cross_entropy_with_logits(pred["keep"][finite], keep_gold[finite])
    loss = loss + _masked_l1(pred["hours"], tgt["hours"])
    loss = loss + _masked_l1(pred["pc_hours"], tgt["pc_hours"])
    loss = loss + _masked_l1(pred["travel_days"], tgt["travel_days"])
    loss = loss + _masked_l1(pred["qty"], tgt["qty"])
    material_qty = [ln.qty for ln in example.lines if ln.section == "materials" and ln.qty is not None]
    if material_qty:
        mean_qty = sum(material_qty) / len(material_qty)
        loss = loss + F.binary_cross_entropy_with_logits(pred["materials"][0], pred["materials"].new_tensor(1.0))
        loss = loss + F.smooth_l1_loss(pred["materials"][1], torch.log1p(pred["materials"].new_tensor(mean_qty)))
    elif any(ln.section == "materials" for ln in example.lines):
        loss = loss + F.binary_cross_entropy_with_logits(pred["materials"][0], pred["materials"].new_tensor(1.0))
    if example.billing in BILLING:
        gold = pred["billing"].new_tensor(BILLING.index(example.billing))
        loss = loss + F.cross_entropy(pred["billing"].unsqueeze(0), gold.unsqueeze(0).long())
    return loss


def run_kit(train: Sequence[KitExample], holdout: Sequence[KitExample], *,
            steps: int = 1, lr: float = 3e-3, seed: int = 0,
            model: KitModel | None = None, device: str = "cpu") -> dict[str, Any]:
    """Fixed steps on the training kits. The holdout is scored once, after.

    The five published holdout deals are refused if they appear in ``train``.
    A passed-in model is the live base; the default is the CPU stand-in.
    """
    lock_kit_split((ex.deal_id for ex in train), (ex.deal_id for ex in holdout))
    torch.manual_seed(seed)
    if model is None:
        model = KitModel()
    model = model.to(device)
    live = [p for p in model.parameters() if p.requires_grad]
    if not live:
        raise ValueError("No trainable parameters")
    opt = torch.optim.AdamW(live, lr=lr, weight_decay=0.01)
    seen: list[str] = []
    model.train()
    cast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if str(device).startswith("cuda") else nullcontext()
    for _ in range(steps):
        for ex in train:
            seen.append(ex.deal_id)
            opt.zero_grad()
            with cast:
                loss = kit_loss(model, ex, billing_history(train, ex.deal_id, (h.deal_id for h in holdout)))
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss during kit")
            loss.backward()
            opt.step()
    model.eval()
    scores = []
    with torch.no_grad():
        for ex in holdout:
            with cast:
                pred = model(ex, billing_history(train, ex.deal_id, (h.deal_id for h in holdout)))
            tgt = _targets(ex)
            hit = total = 0
            if pred["keep"].numel():
                called = torch.sigmoid(pred["keep"]) >= 0.5
                for i, gold in enumerate(tgt["keep"]):
                    if gold != gold:
                        continue
                    total += 1
                    hit += int(bool(called[i]) == bool(gold))
            scores.append({"deal_id": ex.deal_id, "keep_hit": hit, "keep_n": total})
    return {"train_ids": tuple(seen), "holdout": scores, "ready_for_training": False, "model": model}


def prepare_kit_run(train: Sequence[KitExample], holdout: Sequence[KitExample], *,
                    live: bool) -> None:
    """A real run names its training kits and the fixed holdout. The five never train."""
    if not train:
        raise ValueError("The kit base needs a training set")
    if not holdout:
        raise ValueError("The kit base needs a holdout")
    train_ids = [ex.deal_id for ex in train]
    hold_ids = [ex.deal_id for ex in holdout]
    if len(train_ids) != len(set(train_ids)) or len(hold_ids) != len(set(hold_ids)):
        raise ValueError("Duplicate deal id in a split")
    lock_kit_split(train_ids, hold_ids)
    if live and not set(KIT_HOLDOUT) <= set(hold_ids):
        missing = ", ".join(i for i in KIT_HOLDOUT if i not in set(hold_ids))
        raise ValueError("The live kit holdout is missing a fixed kit: " + missing)


def load_kit(path: str | Path) -> KitExample:
    """One saved tab plus the atoms from before it was attached. No source text is copied into the repo."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("A kit file must be one JSON object")
    if not str(payload.get("deal_id") or "").strip():
        raise ValueError("A kit file needs a deal_id")
    return KitExample.from_dict(payload)


def live_kit(encoder_name: str = KIT_ENCODER) -> KitModel:
    """Design width, frozen ModernBERT-large, slot reader trained on top."""
    from .text import HFEncoder

    text = HFEncoder(encoder_name, max_length=1024, dtype=torch.bfloat16)
    if int(text.dim) != KIT_TEXT:
        raise ValueError(f"Encoder width {text.dim} does not match the kit width {KIT_TEXT}")
    model = KitModel(text, d=KIT_D, n_layers=KIT_LAYERS, n_heads=KIT_HEADS,
                     dropout=0.1, freeze_encoder=True)
    if any(p.requires_grad for p in model.encoder.parameters()):
        raise RuntimeError("The live encoder did not freeze")
    return model


def save_kit(path: str | Path, model: KitModel, *, step: int) -> None:
    """Weights only. A checkpoint cannot mark the release ready, and it stores no source text."""
    payload = {
        "step": int(step),
        "ready_for_training": False,
        "torch": str(torch.__version__),
        "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }
    if payload["ready_for_training"]:
        raise RuntimeError("A kit checkpoint cannot mark the release ready")
    torch.save(payload, path)


def load_kit_checkpoint(path: str | Path, model: KitModel) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("ready_for_training"):
        raise ValueError("A kit checkpoint cannot mark the release ready")
    model.load_state_dict(payload["state"])
    return int(payload["step"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Kit base. Does not release a model.")
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--holdout", nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--encoder", default=KIT_ENCODER)
    parser.add_argument("--tiny", action="store_true", help="CPU stand-in. Not the base.")
    args = parser.parse_args()
    train = [load_kit(path) for path in args.train]
    holdout = [load_kit(path) for path in args.holdout]
    prepare_kit_run(train, holdout, live=not args.tiny)
    if args.tiny:
        print("model", "tiny")
        result = run_kit(train, holdout, steps=args.steps)
    else:
        if not torch.cuda.is_available():
            raise SystemExit("This kit base is for the H100. CUDA is not available. --tiny is the CPU check.")
        torch.set_float32_matmul_precision("high")
        print("model", "h100", args.encoder)
        model = live_kit(args.encoder).to("cuda")
        result = run_kit(train, holdout, steps=args.steps, lr=3e-4, model=model, device="cuda")
    if args.checkpoint:
        save_kit(args.checkpoint, result["model"], step=args.steps)
    print("train", " ".join(dict.fromkeys(result["train_ids"])))
    print("holdout", " ".join(row["deal_id"] for row in result["holdout"]))
    print("ready_for_training", result["ready_for_training"])
    for row in result["holdout"]:
        print(f"deal {row['deal_id']} keep_hit={row['keep_hit']} n={row['keep_n']}")


if __name__ == "__main__":
    main()
