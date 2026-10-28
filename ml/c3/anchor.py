"""Hold the shared trunk near the start checkpoint (pilot A root cause, M1).

Pilot A trained every module at lr 1e-4. The deal graph (``encoder.blocks``,
``encoder.lift``, ``encoder.ln``) moved about 5% relative, which changed h for
every line of every deal, and novel deals paid for it: 010447 went from 237
sites to 16. The heads are what a run means to teach; the trunk is what every
head and every unseen deal reads. Three controls, all off by default here and
set in pod/runE.sh (rationale in ~/c3-local/runE/loss_plan.md, "Trunk anchor"):

* ``trunk_lr_scale``: the trunk's own optimizer group at ``lr x scale``
  (loop._student_optimizer). Its AdamW decay is 0 when ``trunk_pull`` is on:
  it decays toward INIT, not toward zero.
* ``trunk_pull``: an L2 pull to the INIT weights, applied decoupled (as AdamW
  applies weight decay): after every step ``p -= pull * m_t * (p - p0)``, with
  ``m_t`` the schedule's current multiplier (warmup, cosine). At the full rate
  it removes ``pull`` of the drift per step, so drift saturates at about
  (per-step move) / pull instead of growing with the run's length. Decoupled,
  so its strength does not depend on the size of the task gradients or on
  the clip.
* ``anchor_kl``: KL(INIT || model) on the anchor heads' UNLABELED rows of each
  training part. The INIT predictions are read once per part, before the first
  step (eval mode, the run's own texts and graph flags), and kept on the CPU
  (``<checkpoint>.anchor.pt`` so a resume reads the same targets). Labeled rows
  are left to the gold terms. A head INIT does not have (``--allow-new-heads``)
  has no INIT prediction to keep and is refused. Under the planning-heads
  contract (C3_PLANNING_HEADS=1, c3_record.view.PLANNING_HEADS) a planning
  head is not anchored on a planning-view part: INIT never trained it there
  (masked), so its reading there is no behaviour to keep; the same head stays
  anchored on the quoting-view parts.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

#: The shared trunk: the deal graph, its input lift and norm (not the text encoder).
TRUNK_PREFIXES = ("encoder.blocks.", "encoder.lift.", "encoder.ln.")
#: Rows kept per part and head (the first unlabeled ones by a fixed stride).
ANCHOR_ROWS = 4096


def trunk_named(model) -> list[tuple[str, torch.nn.Parameter]]:
    return [(n, p) for n, p in model.named_parameters()
            if n.startswith(TRUNK_PREFIXES) and not n.startswith("encoder.text.")]


def trunk_ids(model) -> set[int]:
    return {id(p) for _, p in trunk_named(model)}


class TrunkAnchor:
    """INIT's trunk weights (for the pull) and INIT's predictions (for the KL)."""

    def __init__(self, model, pull: float = 0.0, kl: float = 0.0, heads: tuple[str, ...] = ()):
        self.pull, self.kl, self.heads = float(pull), float(kl), tuple(heads)
        self.init = {n: p.detach().clone() for n, p in trunk_named(model)} if self.pull > 0 else {}
        self.targets: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}

    # ---------------------------------------------------------------- pull
    @torch.no_grad()
    def pull_step(self, model, rate: float = 1.0) -> None:
        """One decoupled L2 step toward INIT; ``rate`` is the schedule's multiplier."""
        if self.pull <= 0:
            return
        f = min(1.0, self.pull * float(rate))
        for n, p in trunk_named(model):
            p0 = self.init.get(n)
            if p0 is not None and p0.shape == p.shape:
                p.sub_(f * (p - p0.to(p.device, p.dtype)))

    @torch.no_grad()
    def drift(self, model) -> float:
        """||p - p0|| / ||p0|| over the trunk (0 without a pull snapshot)."""
        num = den = 0.0
        for n, p in trunk_named(model):
            p0 = self.init.get(n)
            if p0 is None:
                continue
            p0 = p0.to(p.device, torch.float32)
            num += float((p.float() - p0).pow(2).sum())
            den += float(p0.pow(2).sum())
        return (num / den) ** 0.5 if den else 0.0

    # ---------------------------------------------------------------- KL
    @torch.no_grad()
    def cache(self, model, parts, device: str = "cpu", autocast=None) -> int:
        """INIT's log-probabilities on each part's unlabeled rows. Returns rows kept."""
        if self.kl <= 0 or not self.heads:
            return 0
        was = model.training
        model.eval()
        kept = 0
        try:
            for b in parts:
                if getattr(b, "task_profile", None) is not None:
                    continue
                rows_by = {}
                free_here = _planning_free(b)
                for key in self.heads:
                    if free_here(key):
                        continue
                    gold = b.targets.get(key)
                    n = len(b)
                    free = [i for i in range(n) if gold is None or gold[i] < 0]
                    if len(free) > ANCHOR_ROWS:
                        step = len(free) / ANCHOR_ROWS
                        free = [free[int(k * step)] for k in range(ANCHOR_ROWS)]
                    if free:
                        rows_by[key] = free
                if not rows_by:
                    continue
                ctx = autocast(device) if autocast is not None else torch.no_grad()
                with ctx:
                    out = model(b.inputs(), company=b.company or None)
                got = {}
                for key, rows in rows_by.items():
                    if key not in out.logits:
                        continue
                    idx = torch.tensor(rows, dtype=torch.long)
                    lp = out.logits[key].float()[idx.to(out.logits[key].device)].log_softmax(-1)
                    got[key] = (idx, lp.to(torch.float16).cpu())
                    kept += len(rows)
                self.targets[_part_key(b)] = got
        finally:
            model.train(was)
        return kept

    def loss(self, out, batch) -> torch.Tensor | None:
        """Mean over anchor heads of KL(INIT || model) on the part's cached rows."""
        if getattr(batch, "task_profile", None) is not None or self.kl <= 0:
            return None
        got = self.targets.get(_part_key(batch))
        if not got:
            return None
        terms = []
        for key, (idx, lp0) in got.items():
            logits = out.logits.get(key)
            if logits is None:
                continue
            dev = logits.device
            lp = logits.float()[idx.to(dev)].log_softmax(-1)
            p0 = lp0.to(dev, torch.float32)
            terms.append(F.kl_div(lp, p0, log_target=True, reduction="batchmean"))
        return torch.stack(terms).mean() if terms else None

    def save(self, path: str | Path) -> None:
        """INIT's trunk and predictions, so a resume pulls toward INIT, not toward where it stopped."""
        torch.save({"heads": self.heads, "targets": self.targets,
                    "init": {n: t.cpu() for n, t in self.init.items()}}, path)

    def load(self, path: str | Path, device=None) -> bool:
        if not Path(path).exists():
            return False
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if self.kl > 0 and tuple(blob.get("heads") or ()) != self.heads:
            return False
        self.targets = blob.get("targets") or {}
        if self.pull > 0:
            if not blob.get("init"):
                return False
            self.init = {n: t.to(device) if device is not None else t for n, t in blob["init"].items()}
        return True


def _planning_free(b):
    """key -> True when the planning-heads contract trains ``key`` on this planning-view part (no anchor)."""
    from c3_record import view as cview  # noqa: PLC0415
    views = list(getattr(b, "view", None) or [])
    if not cview.planning_heads_enabled() or not views or any(cview.name(v) != cview.PLANNING for v in views):
        return lambda key: False
    return cview.is_planning_head


def _part_key(b) -> str:
    return f"{b.deal_id}|{len(b)}"


__all__ = ["ANCHOR_ROWS", "TRUNK_PREFIXES", "TrunkAnchor", "trunk_ids", "trunk_named"]
