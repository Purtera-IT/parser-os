"""Teacher-free losses for graph.py's paths (arch_graph_review.md fixes 5, 7, and
the table node). Every target comes from the saved line labels and links; no
teacher, no model-written row.

* ``citations``: the content / beliefs slots' attention (lattice citations)
  lands on the line's gold hint rows, as the pointer head does.
* ``derived_cards``: roster and sheet verdicts for tables nobody carded, from
  the gold line types of their rows (a table whose labeled rows are mostly
  physical sites is a site list; one with almost none is not). Saved cards
  win: a table with a saved verdict of that question derives nothing.
* ``table_kind``: a table whose labeled rows are one type (80%+, three or more
  rows, headers / totals / legends aside) teaches its node that type.
* ``coherence``: the same table's unlabeled rows read that type. A labeled
  row keeps its own answer (its cross-entropy is in the heads loss), so the
  exceptions (a header, a totals row, a blank template row) are taught by the
  rows that carry them.
* ``node_pointers``: a line that points into another document (a governs edge
  across documents, or hint rows covering most of another document) points
  at that document's node.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .data import IGNORE, Batch, Judged
from .model import C3Output, pair_tensor
from .schema import GROUP

TYPE_KEY = "col:label_type"
#: Rows a table may hold whatever its kind: they do not vote on it.
ROW_EXCEPTIONS = frozenset({"list_header", "legend_entry", "commercial_total", "_keep", "deal_metadata",
                            "source_caveat", "small_talk"})
SITE_TYPES = frozenset({"physical_site"})
#: Sheet role from the row type most of its labeled rows carry.
SHEET_OF_TYPE = {
    "service_line": "scope", "task": "scope", "work_scope_item": "scope", "scope_item": "scope",
    "deliverable": "scope", "requirement": "scope",
    "rate_card": "rate_card", "pricing_assumption": "rate_card",
    "bom_line": "catalog", "vendor_line_item": "catalog", "quantity": "catalog",
    "commercial_total": "financial_summary", "site_budget": "financial_summary",
    "customer_instruction": "instructions",
}
TABLE_DOC_KINDS = frozenset({"site_list", "rate_card"})
MIN_ROWS = 3
SINGLE_KIND = 0.8
ROSTER_YES, ROSTER_NO, SHEET_SHARE = 0.6, 0.1, 0.7
CITE_SPACES = ("content", "beliefs")


def _groups(batch: Batch) -> list[tuple[int, ...]]:
    """Tables: same document and section, three or more lines, mostly table
    rows (geometry) or a table document. First line first."""
    sec = pair_tensor(batch.same_section, "cpu") > 0
    seen, out = set(), []
    geo = batch.geometry or []
    for i in range(len(batch)):
        if i in seen or not bool(sec[i].any()):
            continue
        g = tuple(int(j) for j in sec[i].nonzero().flatten().tolist())
        seen.update(g)
        if len(g) < MIN_ROWS:
            continue
        rows = sum(1 for j in g if j < len(geo) and len(geo[j]) > 4 and geo[j][4] > 0)
        if rows * 2 >= len(g) or batch.doc_kind[g[0]] in TABLE_DOC_KINDS:
            out.append(g)
    return out


def table_facts(model, batch: Batch) -> dict[str, Any]:
    """Per table: its gold majority type and share, site share; cached on the batch."""
    cached = batch.__dict__.get("_table_facts")
    if cached is not None:
        return cached
    by = model.schema.by_key()
    facts: dict[str, Any] = {"tables": [], "cards": []}
    if TYPE_KEY not in by:
        batch.__dict__["_table_facts"] = facts
        return facts
    types = [a.value for a in by[TYPE_KEY].answers]
    gold = batch.targets.get(TYPE_KEY) or [IGNORE] * len(batch)
    site = batch.targets.get("read:site")
    site_yes = by["read:site"].index("present") if "read:site" in by and site else None
    saved = {(j.key, tuple(sorted(j.lines))) for j in batch.judged}
    for g in _groups(batch):
        voted = [j for j in g if gold[j] != IGNORE and types[gold[j]] not in ROW_EXCEPTIONS]
        if len(voted) < MIN_ROWS:
            continue
        counts: dict[int, int] = {}
        for j in voted:
            counts[gold[j]] = counts.get(gold[j], 0) + 1
        top = max(counts, key=counts.get)
        share = counts[top] / len(voted)
        sites = sum(1 for j in voted if types[gold[j]] in SITE_TYPES
                    or (site_yes is not None and site[j] == site_yes))
        facts["tables"].append({"lines": g, "type": top, "share": share, "site_share": sites / len(voted)})
        key = tuple(sorted(g))
        if "jdg:roster" in by and ("jdg:roster", key) not in saved:
            s = sites / len(voted)
            verdict = "site_roster" if s >= ROSTER_YES else "not_site_roster" if s <= ROSTER_NO else None
            if verdict:
                facts["cards"].append(Judged("jdg:roster", GROUP, g, by["jdg:roster"].index(verdict)))
        role = SHEET_OF_TYPE.get(types[top])
        if role and share >= SHEET_SHARE and "jdg:sheet" in by and ("jdg:sheet", key) not in saved:
            facts["cards"].append(Judged("jdg:sheet", GROUP, g, by["jdg:sheet"].index(role)))
    batch.__dict__["_table_facts"] = facts
    return facts


def citation_loss(model, out: C3Output, batch: Batch) -> torch.Tensor:
    if out.citations is None or model.lattice is None or not batch.hint_lines:
        return out.r.new_zeros(())
    spaces = [model.lattice.spaces.index(s) for s in CITE_SPACES if s in model.lattice.spaces]
    n = out.citations.shape[0]
    rows, cols = [], []
    for i, js in enumerate(batch.hint_lines):
        js = [j for j in js if j != i and j < n]
        if i < n and js:
            rows.append(i)
            cols.append(js)
    if not rows or not spaces:
        return out.r.new_zeros(())
    mask = torch.zeros(len(rows), n, device=out.citations.device)
    for k, js in enumerate(cols):
        mask[k, js] = 1.0
    c = out.citations[rows][:, spaces].float()                      # [R, S, N]
    mass = (c * mask.unsqueeze(1)).sum(-1).clamp(min=1e-6)
    return -mass.log().mean()


def derived_card_loss(model, out: C3Output, batch: Batch, desc, sizes) -> torch.Tensor:
    cards = [j for j in table_facts(model, batch)["cards"] if sizes is None or j.size in sizes]
    if not cards:
        return out.r.new_zeros(())
    total, terms = out.r.new_zeros(()), 0
    for key in sorted({j.key for j in cards}):
        js = [j for j in cards if j.key == key]
        logits = model.judge(out, key, [j.lines for j in js], desc, queries=["" for _ in js])
        total = total + F.cross_entropy(logits.float(), torch.tensor([j.answer for j in js], device=logits.device))
        terms += 1
    return total / max(terms, 1)


def table_losses(model, out: C3Output, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
    zero = out.r.new_zeros(())
    tables = [t for t in table_facts(model, batch)["tables"] if t["share"] >= SINGLE_KIND]
    if not tables:
        return zero, zero
    kind = zero
    if out.nodes is not None:
        reps = [t["lines"][0] for t in tables]
        y = torch.tensor([t["type"] for t in tables], device=out.r.device)
        kind = F.cross_entropy(out.nodes["kind"][reps].float(), y)
    coh = zero
    lg = out.logits.get(TYPE_KEY)
    gold = batch.targets.get(TYPE_KEY) or []
    if lg is not None and gold:
        rows, ys = [], []
        for t in tables:
            for j in t["lines"]:
                if gold[j] == IGNORE and not batch.labeled[j]:
                    rows.append(j)
                    ys.append(t["type"])
        if rows:
            coh = F.cross_entropy(lg[rows].float(), torch.tensor(ys, device=lg.device))
    return kind, coh


def node_pointer_targets(batch: Batch) -> list[tuple[int, list[int]]]:
    """(line, representative lines of the nodes it points at) from governs
    edges across documents and hint rows covering most of another document.
    Representatives follow graph.node_pass: a node's first line."""
    cached = batch.__dict__.get("_node_ptr")
    if cached is not None:
        return cached
    doc = pair_tensor(batch.same_doc, "cpu") > 0
    sec = pair_tensor(batch.same_section, "cpu") > 0
    n = len(batch)

    def rep(j: int) -> int:
        g = sec[j] if bool(sec[j].any()) else doc[j]
        return int(g.nonzero()[0]) if bool(g.any()) else -1

    found: dict[int, set[int]] = {}
    for p, c in batch.edges.get("governs", []):
        if p < n and c < n and not bool(doc[p, c]) and rep(c) >= 0:
            found.setdefault(p, set()).add(rep(c))
    for i, js in enumerate(batch.hint_lines or []):
        other = [j for j in js if j < n and not bool(doc[i, j])]
        if not other:
            continue
        for d in {int(doc[j].nonzero()[0]) for j in other if bool(doc[j].any())}:
            members = doc[d].nonzero().flatten().tolist()
            if sum(1 for j in other if bool(doc[d, j])) * 2 >= len(members):
                found.setdefault(i, set()).update(rep(j) for j in members if rep(j) >= 0)
    out = [(i, sorted(r)) for i, r in sorted(found.items())]
    batch.__dict__["_node_ptr"] = out
    return out


def node_pointer_loss(out: C3Output, batch: Batch) -> torch.Tensor:
    if out.nodes is None:
        return out.r.new_zeros(())
    logp = out.nodes["pointer"]
    lses = [torch.logsumexp(logp[i, js], 0) for i, js in node_pointer_targets(batch) if js]
    if not lses:
        return out.r.new_zeros(())
    lse = torch.stack(lses)
    ok = torch.isfinite(lse)
    return -(torch.where(ok, lse, torch.zeros_like(lse))).sum() / ok.sum().clamp(min=1)


def graph_losses(model, out: C3Output, batch: Batch, desc, w, sizes) -> dict[str, torch.Tensor]:
    parts: dict[str, torch.Tensor] = {}
    if w.citations:
        parts["citations"] = citation_loss(model, out, batch)
    if w.derived_cards:
        parts["derived_cards"] = derived_card_loss(model, out, batch, desc, sizes)
    if w.table_kind or w.coherence:
        parts["table_kind"], parts["coherence"] = table_losses(model, out, batch)
    if w.node_pointers:
        parts["node_pointers"] = node_pointer_loss(out, batch)
    return parts


__all__ = ["graph_losses", "node_pointer_targets", "table_facts"]
