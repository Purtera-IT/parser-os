"""How a deal gets into the model: one JSON contract, split into inputs and targets.

The contract (``DealExample.from_dict``; fixtures/synthetic_deal.json is a
complete example)::

    {
      "deal_id": "...", "company": "acme_av",
      "company_policy": "optional: the company's rules in its own words",
      "atoms": [
        {"key": "a1", "text": "...", "entered_at": "2026-05-01T14:02:00Z",
         "doc_id": "email-3", "doc_kind": "email", "section": "body",
         "order": 0, "speaker_role": "customer", "speaker_side": "customer",
         "label": {"label_type": "...", "about": "deal", "reads_set": {...},
                   "rejected": null, "note": "WHY...\\n[acme_av] keep: ..."}}
      ],
      "edges": [{"src": "a7", "dst": "a3", "relation": "answers"}],
      "outcome": {"final_hours": 416, "final_price": 191360}
    }

``from_training_blob`` adapts the real training blob (labels keyed by
``label_key``) by joining it to the atom rows on that key.

The split this module enforces is the one docs/C3_HEADS.md section 5 states:
**inputs** are the line, where it sits, when it entered, who said it (role and
side, never a name). **Every label field is a target.** The only exception is
the two context slots (stage, geography), which are passed as *optional*
inputs with dropout, so the model uses them when present and works without.
``Batch.inputs()`` is all ``C3Model.forward`` ever receives; targets reach the
losses only.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .notes import drop_meta, mask_verdict, split_note
from .schema import ABSENT, BINARY, CLASS, NUMBER, PRESENCE, RELATION, Schema

IGNORE = -100

#: Readings fed as optional context slots (and also predicted as heads).
CONTEXT_SLOTS = ("stage_std", "location_tier")


@dataclass
class Atom:
    key: str
    text: str
    entered_at: float = 0.0
    doc_id: str = ""
    doc_kind: str = ""
    section: str = ""
    order: int = 0
    speaker_role: str = ""
    speaker_side: str = ""
    label: dict[str, Any] | None = None
    atom_id: str = ""


@dataclass
class DealExample:
    deal_id: str
    atoms: list[Atom]
    company: str = ""
    company_policy: str = ""
    edges: list[tuple[str, str, str]] = field(default_factory=list)
    outcome: dict[str, float] = field(default_factory=dict)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "DealExample":
        atoms = [Atom(key=str(a["key"]), text=str(a.get("text", "")),
                      entered_at=_time(a.get("entered_at")),
                      doc_id=str(a.get("doc_id", "")), doc_kind=str(a.get("doc_kind", "")),
                      section=str(a.get("section", "")), order=int(a.get("order", i)),
                      speaker_role=str(a.get("speaker_role", "")),
                      speaker_side=str(a.get("speaker_side", "")), label=a.get("label"),
                      atom_id=str(a.get("atom_id") or a.get("id") or ""))
                 for i, a in enumerate(d["atoms"])]
        # Time order is the deal's order: ties keep document order.
        atoms.sort(key=lambda a: (a.entered_at, a.order))
        return DealExample(
            deal_id=str(d["deal_id"]), atoms=atoms, company=str(d.get("company", "")),
            company_policy=str(d.get("company_policy", "")),
            edges=[(str(e["src"]), str(e["dst"]), str(e["relation"])) for e in d.get("edges", [])],
            outcome={k: float(v) for k, v in (d.get("outcome") or {}).items()})

    @staticmethod
    def load(path: str | Path) -> "DealExample":
        return DealExample.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @staticmethod
    def from_training_blob(blob: dict[str, Any], atoms: list[dict[str, Any]], *,
                           deal_id: str, company: str = "purtera") -> "DealExample":
        """Join the training blob's labels onto atom rows by ``label_key``.

        ``atoms`` are the parser's atom rows for the deal (text, source date,
        document, section, speaker); each must carry the ``label_key`` the
        labeling page uses. Relations come from each label's ``links`` list
        when the blob has one (``{"relation", "to"}``).
        """
        labels = {lb["label_key"]: lb for lb in blob.get("labels", [])}
        rows, edges = [], []
        for a in atoms:
            row = dict(a)
            row.setdefault("key", a.get("label_key"))
            lb = labels.get(a.get("label_key"))
            if lb is not None:
                row["label"] = lb
                for link in lb.get("links", []) or []:
                    edges.append({"src": row["key"], "dst": link["to"], "relation": link["relation"]})
            rows.append(row)
        # Missed rows: text the parser made no atom of, highlighted by a
        # labeler (origin "labeler"). They have no atom row, so they become
        # lines from their own text, or the base never learns what the parser
        # misses.
        seen = {a.get("label_key") for a in atoms}
        for lb in blob.get("labels", []):
            if lb.get("label_key") in seen or not str(lb.get("text") or "").strip():
                continue
            if str(lb.get("origin") or "").strip().lower() != "labeler":
                continue
            rows.append({"key": lb["label_key"], "text": lb["text"], "label": lb,
                         "atom_id": lb.get("atom_id") or "",
                         "entered_at": lb.get("entered_at") or lb.get("source_date"),
                         "doc_id": lb.get("doc_id", ""), "doc_kind": lb.get("doc_kind", ""),
                         "section": lb.get("section", ""), "order": lb.get("order", len(rows)),
                         "speaker_role": lb.get("speaker_role", ""),
                         "speaker_side": lb.get("speaker_side", "")})
            for link in lb.get("links", []) or []:
                edges.append({"src": lb["label_key"], "dst": link["to"], "relation": link["relation"]})
        return DealExample.from_dict({"deal_id": deal_id, "company": company,
                                      "atoms": rows, "edges": edges})


def _time(v: Any) -> float:
    if v is None or v == "":
        return 0.0
    if isinstance(v, (int, float)):
        return float(v)
    return datetime.fromisoformat(str(v).replace("Z", "+00:00")).timestamp()


# --------------------------------------------------------------- featurize

_NUMBER = re.compile(r"(\$)?(\d[\d,]*(?:\.\d+)?)\s*(\"|in\b|inch|hours?|hrs?|days?|sites?|techs?|%)?", re.I)


def number_tokens(text: str) -> list[str]:
    """Magnitude tokens: the order of magnitude and unit of every number.

    "4 sites x 52 days x $920" -> ["n0:site", "n1:day", "n2:$"]. Encoders
    read digits badly; magnitude plus unit is what the program and the
    claim heads need, and it is the same for any company.
    """
    out = []
    for m in _NUMBER.finditer(text or ""):
        try:
            val = float(m.group(2).replace(",", ""))
        except ValueError:
            continue
        mag = 0 if val < 1 else min(9, int(len(str(int(val)))) - 1)
        unit = "$" if m.group(1) else (m.group(3) or "").lower().rstrip("s").replace("inch", '"').replace("in", '"')
        out.append(f"n{mag}:{unit}")
    return out


@dataclass
class Batch:
    """One deal, featurized. Inputs and targets are kept in separate fields."""
    deal_id: str
    company: str
    company_policy: str
    # inputs
    texts: list[str]
    numbers: list[list[str]]
    doc_kind: list[str]
    role: list[str]
    side: list[str]
    times: list[float]
    same_doc: list[list[bool]]
    same_section: list[list[bool]]
    context: dict[str, list[str | None]]
    # targets
    targets: dict[str, list[int]]               # opportunity key -> class index or IGNORE
    numbers_target: dict[str, list[float | None]]
    edges: dict[str, list[tuple[int, int]]]     # relation -> (src, dst) index pairs
    labeled: list[bool]
    why: list[str | None]                        # universal WHY, verdict words masked
    policy_note: list[str | None]                # the company's line
    outcome: dict[str, float]
    #: Proposed labeling fields (v5 section 6); empty until the card has them.
    rule_links: list[tuple[int, str, int]] = field(default_factory=list)  # (line, rule id, +1 follows / -1 exception)
    changes: list[dict[str, str] | None] = field(default_factory=list)    # line -> {slot: down|none|up}
    #: The rest of the labeling (heads-readthrough.md): per-field notes by
    #: opportunity, row weight from weight_tier, the lines a decision came
    #: from (hint_refs), and the entities a line names (entity_keys).
    field_notes: list[dict[str, str]] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    hint_lines: list[list[int]] = field(default_factory=list)
    entities: list[list[str]] = field(default_factory=list)

    def inputs(self) -> dict[str, Any]:
        return {"texts": self.texts, "numbers": self.numbers, "doc_kind": self.doc_kind,
                "role": self.role, "side": self.side, "times": self.times,
                "same_doc": self.same_doc, "same_section": self.same_section,
                "context": self.context}

    def __len__(self) -> int:
        return len(self.texts)


def _field_value(label: dict[str, Any], source: str, name: str) -> Any:
    if source == "column":
        return label.get(name)
    return (label.get("reads_set") or {}).get(name)


#: Row weights by ``weight_tier`` (same values as app.learning.human_labels).
TIER_WEIGHT = {"load_bearing": 3.0, "ordinary": 1.0, "slight": 0.3}
EXCLUDE_NOTE_PREFIX = "EXCLUDE_FROM_TRAINING"
_TRUE = ("true", "1", "yes", "t")


def excluded(label: dict[str, Any] | None) -> bool:
    """An old manual Deal Kit row, or one set aside: it never trains a head
    (same test as app.learning.human_labels._marked_excluded, plus ``skip``).
    The line stays in the deal as context for the others."""
    if not label:
        return False
    note = str(label.get("note") or "").lstrip().lstrip("[").upper()
    reads = label.get("reads_set") if isinstance(label.get("reads_set"), dict) else {}
    return (note.startswith(EXCLUDE_NOTE_PREFIX)
            or str(label.get("weight_tier") or "").strip().lower() == "exclude"
            or str(label.get("consumer") or "").strip().lower() == "ignore"
            or str(reads.get("exclude_from_training")).strip().lower() in _TRUE
            or str(reads.get("skip")).strip().lower() in _TRUE)


_NOT_FACT = ("_keep", "small_talk")
_FLAG_FALSE = ("", "false", "f", "0", "no", "none")


def _rejected_flag(label: dict[str, Any]) -> bool:
    """The card's reject: "this atom is not a fact" (wreckage, boilerplate, a
    fragment). Older rows hold a type name here instead (the type the labeler
    ruled out), which is not a flag."""
    v = str(label.get("rejected") or "").strip().lower()
    return v in ("true", "t", "1", "yes")


def _derived(label: dict[str, Any]) -> dict[str, Any]:
    """Fields the card records implicitly, made explicit for the heads:

    * ``admission``: drop for a reject or a not-a-fact type (a hand-added
      ``_keep`` aside), keep for a hand-added line or a real type; the same
      rule as app.learning.human_labels.
    * known negatives: a reading the parser proposed and the labeler removed
      (``reads_shown`` minus ``reads_set``) or one the labeler considered and
      ruled out (``rejected_reads``) is taught as absent, not left unknown.
    """
    reads = dict(label.get("reads_set") or {}) if isinstance(label.get("reads_set"), dict) else {}
    typ = str(label.get("label_type") or "").strip()
    origin = str(label.get("origin") or "").strip().lower()
    out = dict(label)
    if _rejected_flag(label) or (typ in _NOT_FACT and not (typ == "_keep" and origin == "labeler")):
        out["admission"] = "drop"
    elif origin == "labeler" or (typ and typ not in _NOT_FACT):
        out["admission"] = "keep"
    removed = {str(k) for k in (label.get("reads_shown") or [])} - set(reads)
    removed |= {str(k) for k in (label.get("rejected_reads") or {})}
    for k in removed:
        reads.setdefault(k, ABSENT)
    out["reads_set"] = reads
    return out


def field_note_targets(schema: Schema) -> dict[str, str]:
    """Free-text ``<field>_note`` readings -> the opportunity they explain.
    ``sow_coverage_note`` -> read:sow_coverage, ``address_note`` ->
    read:address_level (the one reading in the registry that starts with it)."""
    fields = {o.field: o.key for o in schema.opportunities if o.source == "read"}
    out = {}
    for name in schema.note_fields:
        base = name[: -len("_note")]
        if base in fields:
            out[name] = fields[base]
        else:
            hits = [k for f, k in fields.items() if f.startswith(base + "_")]
            if len(hits) == 1:
                out[name] = hits[0]
    return out


def featurize(deal: DealExample, schema: Schema, *, absent_is_negative: bool = False) -> Batch:
    """Split one deal into model inputs and per-opportunity targets.

    ``absent_is_negative``: whether a reading missing from a labeled card is
    a negative example. Default False, matching multitask_table: a missing
    reading is unknown unless a human removed it, so it is not taught as
    "absent". Columns (label_type, about...) are always taught when present.
    """
    import dataclasses

    atoms = [dataclasses.replace(a, label=None) if excluded(a.label)
             else dataclasses.replace(a, label=_derived(a.label)) if a.label else a
             for a in deal.atoms]
    n = len(atoms)
    index = {a.key: i for i, a in enumerate(atoms)}
    company = deal.company or "purtera"

    targets: dict[str, list[int]] = {}
    numbers_target: dict[str, list[float | None]] = {}
    for opp in schema.opportunities:
        if opp.kind == RELATION:
            continue
        col = [IGNORE] * n
        nums: list[float | None] = [None] * n
        for i, a in enumerate(atoms):
            if not a.label:
                continue
            v = _field_value(a.label, opp.source, opp.field)
            if v in (None, "", []) and not absent_is_negative:
                continue
            if isinstance(v, list):          # train_for: a set; teach its first member
                v = v[0] if v else None
            ix = opp.index(v)
            if ix is not None:
                col[i] = ix
            if opp.kind == NUMBER and v not in (None, ""):
                try:
                    nums[i] = float(str(v).replace(",", "").split()[0])
                except ValueError:
                    pass
        targets[opp.key] = col
        if opp.kind == NUMBER:
            numbers_target[opp.key] = nums

    edges: dict[str, list[tuple[int, int]]] = {}
    for src, dst, rel in deal.edges:
        if src in index and dst in index:
            edges.setdefault(rel, []).append((index[src], index[dst]))

    why: list[str | None] = []
    policy: list[str | None] = []
    for a in atoms:
        note = (a.label or {}).get("note") or ""
        u, p = split_note(note, company)
        u, p = drop_meta(u), drop_meta(p)
        why.append(mask_verdict(u) if u else None)
        policy.append(p or None)

    rule_links: list[tuple[int, str, int]] = []
    changes: list[dict[str, str] | None] = []
    for i, a in enumerate(atoms):
        lb = a.label or {}
        reads = lb.get("reads_set") or {}
        for rid in _as_list(lb.get("follows_rules", reads.get("follows_rule"))):
            rule_links.append((i, rid, 1))
        for rid in _as_list(lb.get("exception_to", reads.get("exception_to"))):
            rule_links.append((i, rid, -1))
        changes.append(parse_changes(reads.get("changes")))

    # Per-field notes: the reasoning behind one field, kept per line by the
    # opportunity it explains (the teacher reads it as that head's WHY).
    by_note = field_note_targets(schema)
    opp_layer = {o.key: o.layer for o in schema.opportunities}
    field_notes: list[dict[str, str]] = []
    weights: list[float] = []
    hint_lines: list[list[int]] = []
    entities: list[list[str]] = []
    by_atom = {a.atom_id: i for i, a in enumerate(atoms) if a.atom_id}
    for a in atoms:
        lb = a.label or {}
        reads = lb.get("reads_set") if isinstance(lb.get("reads_set"), dict) else {}
        notes = {}
        for name, key in by_note.items():
            t = drop_meta(str(reads.get(name) or ""))
            if t:
                notes[key] = mask_verdict(t) if opp_layer.get(key) == "universal" else t
        for key, why_not in (lb.get("rejected_reads") or {}).items():
            t = drop_meta(str(why_not or ""))
            k = f"read:{key}"
            if t and k in opp_layer:
                notes.setdefault(k, f"not {key}: {mask_verdict(t) if opp_layer[k] == 'universal' else t}")
        field_notes.append(notes)
        weights.append(TIER_WEIGHT.get(str(lb.get("weight_tier") or "").strip().lower(), 1.0))
        refs = [r for r in (lb.get("hint_refs") or []) if isinstance(r, dict)]
        hint_lines.append(sorted({by_atom.get(str(r.get("atomId")), index.get(str(r.get("atomId"))))
                                  for r in refs} - {None}))
        entities.append(_as_list(lb.get("entity_keys")))

    context = {slot: [((a.label or {}).get("reads_set") or {}).get(slot) for a in atoms]
               for slot in CONTEXT_SLOTS}
    return Batch(
        deal_id=deal.deal_id, company=company, company_policy=deal.company_policy,
        texts=[a.text for a in atoms], numbers=[number_tokens(a.text) for a in atoms],
        doc_kind=[a.doc_kind for a in atoms], role=[a.speaker_role for a in atoms],
        side=[a.speaker_side for a in atoms], times=[a.entered_at for a in atoms],
        same_doc=[[bool(a.doc_id) and a.doc_id == b.doc_id for b in atoms] for a in atoms],
        same_section=[[bool(a.section) and a.doc_id == b.doc_id and a.section == b.section
                       for b in atoms] for a in atoms],
        context=context, targets=targets, numbers_target=numbers_target, edges=edges,
        labeled=[a.label is not None for a in atoms], why=why, policy_note=policy,
        outcome=dict(deal.outcome), rule_links=rule_links, changes=changes,
        field_notes=field_notes, weights=weights, hint_lines=hint_lines, entities=entities)


def _as_list(v: Any) -> list[str]:
    if v in (None, ""):
        return []
    if isinstance(v, str):
        return [p.strip() for p in v.split(",") if p.strip()]
    return [str(x) for x in v]


def parse_changes(v: Any) -> dict[str, str] | None:
    """``"hours:up, price:up, sites:down"`` (or a dict) -> {slot: direction}."""
    if v in (None, ""):
        return None
    items = v.items() if isinstance(v, dict) else (p.split(":", 1) for p in _as_list(v) if ":" in p)
    out = {str(k).strip(): str(d).strip() for k, d in items}
    return out or None


def realized_changes(outcome: dict[str, float], tol: float = 0.02) -> dict[str, str]:
    """What actually moved between the quote and the close: for each slot with
    ``quoted_<slot>`` and ``final_<slot>``, up / down / none."""
    out = {}
    for k, final in outcome.items():
        if not k.startswith("final_"):
            continue
        slot = k[len("final_"):]
        quoted = outcome.get(f"quoted_{slot}")
        if quoted is None:
            continue
        rel = (final - quoted) / max(abs(quoted), 1e-9)
        out[slot] = "up" if rel > tol else "down" if rel < -tol else "none"
    return out


__all__ = ["Atom", "Batch", "DealExample", "featurize", "number_tokens", "IGNORE",
           "CONTEXT_SLOTS", "ABSENT", "BINARY", "CLASS", "NUMBER", "PRESENCE"]
