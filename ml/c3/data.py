"""How a deal gets into the model: one JSON contract, split into inputs and targets.

The contract (``DealExample.from_dict``; fixtures/synthetic_deal.json is a
complete example)::

    {
      "deal_id": "...", "company": "acme_av",
      "company_policy": "optional: the company's rules in its own words",
      "atoms": [
        {"key": "a1", "text": "...", "entered_at": "2026-05-01T14:02:00Z",
         "doc_id": "email-3", "doc_kind": "email", "section": "body",
         "order": 0, "speaker_role": "unknown", "speaker_side": "theirs",
         "speaker_org": "external", "doc_direction": "inbound", "stage_at": "scoping",
         "label": {"label_type": "...", "about": "deal", "reads_set": {...},
                   "rejected": null, "note": "WHY...\\n[acme_av] keep: ..."}}
      ],
      "edges": [{"src": "a7", "dst": "a3", "relation": "answers"}],
      "outcome": {"final_hours": 416, "final_price": 191360}
    }

``from_training_blob`` adapts the real training blob (labels keyed by
``label_key``) by joining it to the atom rows on that key.

The split this module enforces is the one docs/C3_HEADS.md section 5 states:
**inputs** are the line, where it sits, when it entered, who said it (role,
side and organisation kind, never a name), its document's direction and the
deal's stage when it arrived (``c3_record.sender``). **Every label field is a target.** The only exception is
the two context slots (stage, geography), which are passed as *optional*
inputs with dropout, so the model uses them when present and works without.
``Batch.inputs()`` is all ``C3Model.forward`` ever receives; targets reach the
losses only. ``Batch.oneshot()`` is the serving contract: the same atoms with
stage and geography nulled. Links, notes and verdicts are never inputs. The
heads predict them.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import torch

from .notes import drop_meta, judgment_note, mask_verdict, sentences, split_note, without_opener
from .schema import (ABSENT, BINARY, CLASS, DEAL, GROUP, LINE, NUMBER, PAIR, PRESENCE,
                     RELATION, Schema)
from .values import (DEAL_SERVICES, HOURS_BASES, LITERAL_HEADS, hours_per, TERM_ENUMS, TERM_LINE_TYPES,
                     TERM_NUMBERS, UNSTATED, canonical_commercial, canonical_term_verdict,
                     hours_canonical, hours_judgment_basis, judged_atom_type,
                     parse_commercial_verdict, parse_hours_verdict, parse_term_verdict,
                     site_count, word_indexes)

IGNORE = -100

#: Readings fed as optional context slots (and also predicted as heads).
CONTEXT_SLOTS = ("stage_std", "location_tier")
PIXEL_DIM = 8 * 8 * 3
GEO_DIM = 8


def _pixels(raw: Any) -> tuple[float, ...]:
    if raw in (None, "", []):
        return tuple(0.0 for _ in range(PIXEL_DIM))
    try:
        values = tuple(float(v) for v in raw)
    except (TypeError, ValueError):
        raise ValueError("Image pixels must be an 8x8x3 grid of numbers") from None
    if len(values) != PIXEL_DIM:
        raise ValueError("Image pixels must be an 8x8x3 grid; do not resize by truncation")
    return values


def _geometry(row: dict[str, Any]) -> tuple[float, ...]:
    bbox = row.get("bbox")
    if bbox is None:
        box = (0.0, 0.0, 0.0, 0.0)
    else:
        try:
            box = tuple(float(v) for v in bbox)
        except (TypeError, ValueError):
            raise ValueError("bbox must be four unit-page numbers") from None
        if len(box) != 4 or any(v < 0 or v > 1 for v in box):
            raise ValueError("bbox must be four unit-page numbers")
    keys = ("table_row", "table_col", "row_span", "col_span")
    if not any(k in row and row.get(k) not in (None, "") for k in keys):
        table = (0.0, 0.0, 0.0, 0.0)
    else:
        try:
            table = tuple(float(row.get(k) or 0) for k in keys)
        except (TypeError, ValueError):
            raise ValueError("table geometry must be numeric") from None
    return box + table


def _geometry_present(row: dict[str, Any]) -> bool:
    if not isinstance(row, dict):
        return False
    if row.get("bbox") not in (None, "", []):
        return True
    return any(k in row and row.get(k) not in (None, "")
               for k in ("table_row", "table_col", "row_span", "col_span"))


@dataclass(frozen=True)
class LiteralTarget:
    """Exact saved string. ``words`` is set only for a unique span in the line."""
    line: int
    key: str
    text: str
    words: tuple[int, ...] | None = None


@dataclass
class Atom:
    key: str
    text: str
    entered_at: float = 0.0
    doc_id: str = ""
    doc_kind: str = ""
    section: str = ""
    order: float = 0
    speaker_role: str = ""
    speaker_side: str = ""
    #: Who sent it and where the deal stood (c3_record.sender): the sender's
    #: organisation kind, its document's direction, the deal's stage when it
    #: arrived. Inputs, like the speaker.
    speaker_org: str = ""
    doc_direction: str = ""
    stage_at: str = ""
    label: dict[str, Any] | None = None
    atom_id: str = ""
    pixels: tuple[float, ...] = ()
    geometry: tuple[float, ...] = ()
    pixel_present: bool = False
    geometry_present: bool = False
    history_only: bool = False
    #: A Dropped card's own occurrence: a copy the parser folded into an
    #: earlier line, or a line it set aside. It carries that card's verdict
    #: and nothing else, so the line that survived keeps its own label.
    dropped_copy: bool = False
    #: Row weight for a line that carries no label of its own (a dropped
    #: copy); a labeled row takes its weight_tier instead.
    row_weight: float | None = None
    #: Training-only multiplier on this line's loss (run E's data build):
    #: below 1 for a repeated copy of a line learned elsewhere, or for a deal
    #: capped to its share of the learned lines. Scales the row weight and
    #: the WHY weight; never an input.
    loss_weight: float = 1.0
    #: Training-only: the same row in other layouts (header-tagged, bare
    #: values, tabs, reversed cells). The meaning lock (losses.layout_lock)
    #: reads the deal with the line rewritten and holds its saved answers.
    layout_rewrites: tuple[str, ...] = ()


@dataclass
class DealExample:
    deal_id: str
    atoms: list[Atom]
    company: str = ""
    company_policy: str = ""
    edges: list[tuple[str, str, str]] = field(default_factory=list)
    outcome: dict[str, float] = field(default_factory=dict)
    #: The judgment tabs' verdicts as the training blob stores them (head,
    #: target, text, verdict, reason, note, labeler); resolved in featurize.
    judgments: list[dict[str, Any]] = field(default_factory=list)
    #: Why a labeler drew a link, by (src key, dst key, relation).
    edge_notes: dict[tuple[str, str, str], str] = field(default_factory=dict)
    #: Links from a line to a whole document ("the file this line announces"):
    #: (src key, document id).
    doc_pointers: list[tuple[str, str]] = field(default_factory=list)
    #: How the blob's evidence links fared (``_blob_links``): edges made,
    #: document pointers, self-links and unresolved links dropped, links
    #: anchored on a judgment card's line, card links with no line.
    link_stats: dict[str, int] = field(default_factory=dict)
    # Lossless training-only source packet. Never returned by Batch.inputs().
    source_supervision: dict[str, Any] = field(default_factory=dict)
    #: Training only: a view of the deal ("pre_quote", "planning") and the
    #: loss scale per opportunity key in that view (0 masks the head there).
    view: str = ""
    head_scale: dict[str, float] = field(default_factory=dict)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "DealExample":
        atoms = [Atom(key=str(a["key"]), text=str(a.get("text", "")),
                      entered_at=_time(a.get("entered_at")),
                      doc_id=str(a.get("doc_id") or a.get("artifact_id") or ""),
                      doc_kind=str(a.get("doc_kind", "")),
                      section=str(a.get("section", "")), order=float(a.get("order", i)),
                      speaker_role=str(a.get("speaker_role", "")),
                      speaker_side=str(a.get("speaker_side", "")),
                      speaker_org=str(a.get("speaker_org") or ""),
                      doc_direction=str(a.get("doc_direction") or ""),
                      stage_at=str(a.get("stage_at") or ""), label=a.get("label"),
                      atom_id=str(a.get("atom_id") or a.get("id") or ""),
                      pixels=tuple(_pixels(a.get("image_pixels"))),
                      geometry=tuple(_geometry(a)),
                      pixel_present=a.get("image_pixels") not in (None, "", []),
                      geometry_present=_geometry_present(a),
                      history_only=bool(a.get("history_only")),
                      dropped_copy=bool(a.get("dropped_copy")),
                      row_weight=(float(a["row_weight"]) if a.get("row_weight") is not None else None),
                      loss_weight=(float(a["loss_weight"]) if a.get("loss_weight") is not None else 1.0),
                      layout_rewrites=tuple(str(t) for t in a.get("layout_rewrites") or () if str(t).strip()))
                 for i, a in enumerate(d["atoms"])]
        # Time order is the deal's order: ties keep document order.
        atoms.sort(key=lambda a: (a.entered_at, a.order))
        keys = [a.key for a in atoms]
        ids = [a.atom_id for a in atoms if a.atom_id]
        if len(set(keys)) != len(keys) or len(set(ids)) != len(ids):
            raise ValueError("Duplicate source occurrence identifiers in training input")
        return DealExample(
            deal_id=str(d["deal_id"]), atoms=atoms, company=str(d.get("company", "")),
            company_policy=str(d.get("company_policy", "")),
            edges=[(str(e["src"]), str(e["dst"]), str(e["relation"])) for e in d.get("edges", [])],
            outcome={k: float(v) for k, v in (d.get("outcome") or {}).items()},
            judgments=[j for j in d.get("judgments") or [] if isinstance(j, dict)],
            edge_notes={(str(e["src"]), str(e["dst"]), str(e["relation"])): str(e["note"])
                        for e in d.get("edges", []) if e.get("note")},
            doc_pointers=[(str(p[0]), str(p[1])) for p in d.get("doc_pointers", [])],
            view=str(d.get("view") or ""),
            head_scale={str(k): float(v) for k, v in (d.get("head_scale") or {}).items()},
            source_supervision=d.get("source_supervision") or {
                "judgments": d.get("judgments") or [],
                "deal_answers": d.get("deal_answers") or [],
                "missed_history": d.get("missed_history") or []})

    @staticmethod
    def load(path: str | Path) -> "DealExample":
        return DealExample.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @staticmethod
    def from_training_blob(blob: dict[str, Any], atoms: list[dict[str, Any]], *,
                           deal_id: str, company: str = "purtera",
                           serve_lines_only: bool = True) -> "DealExample":
        """Join the training blob's labels onto atom rows by ``label_key``.

        ``atoms`` are the parser's atom rows for the deal (text, source date,
        document, section, speaker); each must carry the ``label_key`` the
        labeling page uses. Relations come from each label's ``links`` list
        when the blob has one (``{"relation", "to"}``).

        ``serve_lines_only`` (default): the deal's lines are its atom rows and
        nothing else, the lines the parser's compile sends the base. A
        labeler-added Missed row, a ``missed_history`` row and a Dropped
        card's copy (``dropped_copy_rows``) are text the compile never sends;
        as model input they changed every other line's context and rank
        (model ``embed_atoms``: index / line count) away from serving. Their
        targets stay in ``source_supervision``; they train nothing until the
        compile sends such lines. False adds them as lines, as before.
        """
        labels = select_labels(blob.get("labels", []))
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
        for lb in (() if serve_lines_only else labels.values()):
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
                         "speaker_side": lb.get("speaker_side", ""),
                         **{k: lb.get(k, "") for k in ("speaker_org", "doc_direction", "stage_at")}})
            for link in lb.get("links", []) or []:
                edges.append({"src": lb["label_key"], "dst": link["to"], "relation": link["relation"]})
        for n, lb in enumerate(() if serve_lines_only else blob.get("missed_history") or []):
            if not isinstance(lb, dict) or not _is_a_person(lb.get("labeler")):
                continue
            if str(lb.get("origin") or "labeler").strip().lower() != "labeler":
                continue
            text = str(lb.get("text") or "").strip()
            reads = lb.get("reads_set") if isinstance(lb.get("reads_set"), dict) else {}
            cause = str(reads.get(MISS_CAUSE) or lb.get(MISS_CAUSE) or "").strip()
            if not text or not cause:
                raise ValueError("missed_history row is missing its text or miss_cause")
            # The recovered line trains the cause only. The live atom carries the fact.
            rows.append({"key": f"hist:{n}:{lb.get('label_key') or n}", "text": text,
                         "label": {"label_key": f"hist:{n}:{lb.get('label_key') or n}",
                                   "labeler": lb.get("labeler"), "origin": "labeler",
                                   "reads_set": {MISS_CAUSE: cause}},
                         "entered_at": lb.get("entered_at") or lb.get("source_date"),
                         "doc_id": lb.get("doc_id", ""), "doc_kind": lb.get("doc_kind", ""),
                         "section": lb.get("section", ""), "order": lb.get("order", len(rows)),
                         "history_only": True})
        if not serve_lines_only:
            rows.extend(dropped_copy_rows(blob.get("judgments") or [], rows))
        link_stats: dict[str, int] = {}
        linked, doc_pointers = _blob_links(blob.get("links") or [], rows,
                                           judgments=blob.get("judgments") or [], stats=link_stats)
        edges += linked
        deal = DealExample.from_dict({"deal_id": deal_id, "company": company,
                                      "atoms": rows, "edges": edges, "doc_pointers": doc_pointers,
                                      "judgments": blob.get("judgments") or [],
                                      "view": blob.get("view"), "head_scale": blob.get("head_scale")})
        deal.link_stats = link_stats
        # Keep deferred fields and tab targets inspectable instead of claiming
        # that storing a row means the model has a loss for it.
        import copy
        deal.source_supervision = copy.deepcopy({
            "labels": list(labels.values()), "judgments": blob.get("judgments") or [],
            "links": blob.get("links") or [], "deal_answers": blob.get("deal_answers") or [],
            "missed_history": blob.get("missed_history") or [],
        })
        return deal


#: A dropped copy the parser was right to fold away teaches less than one it
#: should have kept: tens of thousands of folded signatures and repeats would
#: otherwise be most of the admission head's examples.
DROPPED_COPY_WEIGHT = {"correctly_dropped": 0.3, "should_have_been_kept": 1.0}


def _sup_is_copy(key: str, target: Any) -> bool:
    """A Dropped card names a copy of a surviving line (a fold, or a second
    occurrence at another position, ``sup:<key>@...``), or the set-aside
    line itself (``sup:<key>``, no survivor): only the second is that atom."""
    t = target if isinstance(target, dict) else {}
    return "@" in key or bool(t.get("isFold")) or bool(t.get("survivor"))


def dropped_copy_rows(judgments: Iterable[Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One line per Dropped card whose subject is a copy, not its survivor.

    The copy shares its survivor's ``label_key`` (same file, page and text), so
    resolving the card by that key would write "correctly dropped" onto the
    line that was kept, and two copies of one line onto each other. Each copy
    gets its own row, keyed by the card's full ``target_key``, placed just
    after its survivor in the same document and section. It carries no label:
    its card's verdict is its only target."""
    by_key = {str(r.get("key")): r for r in rows if r.get("key")}
    place = {str(r.get("key")): i for i, r in enumerate(rows) if r.get("key")}
    out: list[dict[str, Any]] = []
    made: set[str] = set()
    for j in judgments:
        if not isinstance(j, dict) or str(j.get("head") or "") != "suppression":
            continue
        if not _is_a_person(j.get("labeler")) or excluded(j):
            continue
        key = str(j.get("target_key") or "")
        if not key.startswith("sup:") or key in made or key in by_key:
            continue
        target = j.get("target") if isinstance(j.get("target"), dict) else {}
        if not _sup_is_copy(key, target):
            continue
        survivor = by_key.get(key[len("sup:"):].split("@", 1)[0])
        text = str(target.get("text") or j.get("text") or "").strip()
        if survivor is None or not text:
            continue                     # resolves to nothing; featurize leaves the card unplaced
        made.add(key)
        verdict = str(j.get("verdict") or "").strip()
        out.append({"key": key, "text": text, "atom_id": "",
                    "entered_at": survivor.get("entered_at"), "doc_id": survivor.get("doc_id", ""),
                    "doc_kind": survivor.get("doc_kind", ""), "section": survivor.get("section", ""),
                    "order": float(survivor.get("order", place[str(survivor.get("key"))])) + 0.5,
                    "speaker_role": survivor.get("speaker_role", ""),
                    "speaker_side": survivor.get("speaker_side", ""),
                    **{k: survivor.get(k, "") for k in ("speaker_org", "doc_direction", "stage_at")},
                    "dropped_copy": True, "row_weight": DROPPED_COPY_WEIGHT.get(verdict, 1.0)})
    return out


#: A relation the model reads only from the later line back to the earlier
#: one (model.CAUSAL_RELATIONS) whose direction a labeler does not mean:
#: either line's card can draw a contradiction.
_LATER_FIRST = ("contradicts",)
#: A link note that says nothing (the card's default).
_EMPTY_LINK_NOTES = ("drawn while labelling the whole deal", "drawn while labelling", "")
#: Judgment cards a link can be drawn on whose key names no line: the link
#: starts from the card's subject line (a conflict or site pair's a side, a
#: question's source line, a site's first mention).
_CARD_HEADS = ("conflict", "site", "gap", "site_role")


def _card_refs(target: Any) -> list[dict[str, Any]]:
    """The lines a judgment card stands on, as its target stores them: its a
    side (or a question's source, a site's first mention), then its b side."""
    t = target if isinstance(target, dict) else {}
    refs = []
    for ref in (t.get("a") or t.get("source") or t.get("site"), t.get("b")):
        ref = ref if isinstance(ref, dict) else {}
        if isinstance(ref.get("evidence"), dict):
            ref = ref["evidence"]             # a site: its first mention
        refs.append(ref)
    return refs


def _blob_links(links: list[Any], rows: list[dict[str, Any]], *,
                judgments: Iterable[Any] = (), stats: dict[str, int] | None = None
                ) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """The evidence links the labeling page stores at the top of the training
    blob (``doc.links``: from_key, from_text, from_head, to_label_key,
    to_atom_id, to_text, relation, note, labeler), as edges between lines.

    The from side is the card the link was drawn on: its label_key; a Dropped
    card's ``sup:<label_key>@...`` key names that exact copy and a Places
    card's ``site:<atom_id>`` (and an hours or task-tier card's
    ``atom:<atom_id>``) that atom (a dropped copy's text equals its
    original's, so text would find the wrong line); a conflict, site pair,
    Questions or site-role card (``judgments``) starts from its subject line,
    the a side (the b side when the link points at the a side itself);
    otherwise its text is used. The to side is
    resolved by label_key, then atom id, then text. An ``answers`` link always
    runs answer -> question (a Questions card draws it the other way round),
    and a contradiction runs later line -> earlier.
    Model-written drafts never train. The labeler's note goes with the edge.
    A link whose ends are not in this parse (an older parse's keys) is dropped.
    A link to a whole document (to_kind text, an artifact and no line) is
    returned as a document pointer. Text matching ignores which artifact a
    line came from, so links drawn on an earlier parse of the same file still
    resolve. ``stats`` counts what happened to each person's link."""
    stats = stats if stats is not None else {}
    for c in ("edges", "doc_pointers", "self_links", "unresolved", "card_anchored", "card_no_line"):
        stats.setdefault(c, 0)
    judgments = list(judgments)
    retired_cards = {(str(j.get("labeler")), str(j.get("head")), str(j.get("target_key")))
                     for j in judgments if isinstance(j, dict) and excluded(j)}
    cards = {(str(j.get("head")), str(j.get("target_key"))): j.get("target") for j in judgments
             if isinstance(j, dict) and _is_a_person(j.get("labeler")) and j.get("target_key")
             and str(j.get("head") or "") in _CARD_HEADS}
    by_key = {str(r.get("key")): r for r in rows if r.get("key")}
    by_id = {str(r.get("atom_id") or r.get("id")): r for r in rows if r.get("atom_id") or r.get("id")}
    card_text = {str(j.get("target_key")): str(j.get("text") or "").split("||")[0].strip()
                 for j in judgments if isinstance(j, dict) and _is_a_person(j.get("labeler"))
                 and j.get("target_key") and str(j.get("text") or "").split("||")[0].strip()}
    sup_copies = {str(j.get("target_key")): _sup_is_copy(str(j.get("target_key") or ""), j.get("target"))
                  for j in judgments if isinstance(j, dict) and str(j.get("head") or "") == "suppression"}
    by_text: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_text.setdefault(_norm_text(r.get("text")), []).append(r)

    norm_rows: list[str] = []

    def find(key: Any, atom_id: Any, text: Any) -> dict[str, Any] | None:
        exact = by_key.get(str(key or "")) or by_id.get(str(atom_id or ""))
        if exact is not None:
            return exact
        matches = by_text.get(_norm_text(text), []) if str(text or "").strip() else []
        # A repeated signature or suppressed copy is not its surviving original.
        if len(matches) == 1:
            return matches[0]
        if matches or not str(text or "").strip():
            return None
        if not norm_rows:
            norm_rows.extend(_norm_text(r.get("text")) for r in rows)
        i = _unique_prefix(norm_rows, text)
        return rows[i] if i is not None else None

    def find_from(k: dict[str, Any], to: dict[str, Any] | None) -> tuple[dict[str, Any] | None, bool]:
        """The line a link starts from, and whether its key is a judgment card's."""
        key = str(k.get("from_key") or "")
        card_key = (str(k.get("from_head") or ""), key)
        r = None
        if key.startswith("sup:"):
            # The Dropped card's own row (dropped_copy_rows), or the set-aside
            # line itself. A copy with no row of its own must not become its
            # survivor.
            r = by_key.get(key)
            if r is None:
                base = by_key.get(key[len("sup:"):].split("@", 1)[0])
                r = base if base is not None and not sup_copies.get(key) else None
            return r, False
        elif key.startswith(("site:", "atom:")):
            r = by_id.get(key.split(":", 1)[1])
        elif card_key in cards:
            for ref in _card_refs(cards[card_key]):
                r = find(None, ref.get("atomId") or ref.get("atom_id"), ref.get("text"))
                if r is not None and r is not to:
                    stats["card_anchored"] += 1
                    return r, True
            r = None
        r = r or find(key, k.get("from_atom_id"), k.get("from_text"))
        if r is None and key in card_text:
            # The link keeps only the start of its card's text; the card
            # itself (a rule card, for one) keeps the whole line. A line
            # repeated in several files is the one in the file the link
            # points into, when only one copy is there.
            full = card_text[key]
            r = find(None, None, full)
            if r is None:
                same = [x for x in by_text.get(_norm_text(full), [])
                        if to is not None and x.get("doc_id") == to.get("doc_id")]
                r = same[0] if len(same) == 1 else None
        return r, card_key in cards

    out, doc_pointers = [], []
    for k in links:
        if not isinstance(k, dict) or not _is_a_person(k.get("labeler")):
            continue
        if (str(k.get("labeler")), str(k.get("from_head")), str(k.get("from_key"))) in retired_cards:
            stats["retired_card_links"] = stats.get("retired_card_links", 0) + 1
            continue
        rel = str(k.get("relation") or "")
        b = find(k.get("to_label_key"), k.get("to_atom_id"), k.get("to_text"))
        a, on_card = find_from(k, b)
        if a is not None and b is None and k.get("to_artifact_id") and not k.get("to_label_key") \
                and not k.get("to_atom_id") and str(k.get("to_kind") or "") == "text":
            doc_pointers.append((str(a["key"]), str(k["to_artifact_id"])))
            stats["doc_pointers"] += 1
            continue
        if a is None and on_card:
            stats["card_no_line"] += 1          # e.g. a generated question with no source line
        if not rel or a is None or b is None:
            stats["unresolved"] += 1
            continue
        if a is b:
            stats["self_links"] += 1
            continue
        if rel == "answers" and str(k.get("from_head") or "") == "gap":
            a, b = b, a                              # question card: question -> answer
        if rel in _LATER_FIRST and (_time(a.get("entered_at")), a.get("order", 0)) < \
                (_time(b.get("entered_at")), b.get("order", 0)):
            a, b = b, a
        note = str(k.get("note") or "").strip()     # line breaks kept: the [company] line
        edge = {"src": a["key"], "dst": b["key"], "relation": rel}
        if " ".join(note.split()).lower() not in _EMPTY_LINK_NOTES:
            edge["note"] = note
        out.append(edge)
        stats["edges"] += 1
    return out, doc_pointers


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
    same_doc: Any                                # [N, N] bool tensor (or nested lists)
    same_section: Any
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
    #: How much each line's WHY counts (``why_weight``): 1.0, or
    #: ``DRAFT_WHY_WEIGHT`` for a draft nobody saved.
    why_weights: list[float] = field(default_factory=list)
    hint_lines: list[list[int]] = field(default_factory=list)
    entities: list[list[str]] = field(default_factory=list)
    #: Judgment-tab verdicts about two lines, a group of lines or the whole
    #: deal (line verdicts are in ``targets``).
    judged: list["Judged"] = field(default_factory=list)
    #: Answers known to be wrong for a line, by opportunity: the parser's type
    #: when a person picked another, or a type an older row ruled out.
    negatives: dict[str, list[list[int]]] = field(default_factory=dict)
    #: Suppositions: each sentence of a labeled line's WHY, for the teacher
    #: to read with that sentence assumed (training only).
    flips: list["FlipTarget"] = field(default_factory=list)
    #: Labeled lines that read almost alike but got different answers:
    #: (i, j, opportunity), i < j.
    near_misses: list[tuple[int, int, str]] = field(default_factory=list)
    #: Labeled lines that read almost alike and got the same answers: (i, j),
    #: i < j. A difference in wording that the answer ignores.
    twins: list[tuple[int, int]] = field(default_factory=list)
    # None is unknown; a tuple is the complete saved selection. Kept out of
    # scalar targets so CE/argmax cannot quietly teach only its first answer.
    multi_targets: dict[str, list[tuple[int, ...] | None]] = field(default_factory=dict)
    #: Exact open strings (presence readings, normalized values, hour grammars).
    literals: list[LiteralTarget] = field(default_factory=list)
    #: Deal-level numbers. ``declared_site_count`` is the deal answer's site count.
    deal_numbers: dict[str, float] = field(default_factory=dict)
    pixels: list[tuple[float, ...]] = field(default_factory=list)
    geometry: list[tuple[float, ...]] = field(default_factory=list)
    #: Lines of a long deal's window whose saved link of this relation points
    #: at a line outside the window (taught in another part): the window must
    #: not teach them "no link".
    link_elsewhere: dict[str, list[int]] = field(default_factory=dict)
    #: Saved answers that disagree and were not taught, for the labeler.
    notes: list[str] = field(default_factory=list)
    #: Each row's line key (label_key, or a Dropped card's own key), so a
    #: prediction can be written back to the line it is about. Never an input.
    keys: list[str] = field(default_factory=list)
    #: What the parser's classifier said, by opportunity (class index or
    #: IGNORE). Never an input or a target: the training checks score the
    #: base-where-sure, parser-elsewhere split with it.
    parser_targets: dict[str, list[int]] = field(default_factory=dict)
    #: Training only: (line, text) for a labeled line in another layout
    #: (Atom.layout_rewrites), for the meaning lock. Never an input.
    layout_rewrites: list[tuple[int, str]] = field(default_factory=list)
    #: Training only: (src, dst, relation, note) for a saved link with a note,
    #: the note's universal part with verdict words masked, for the
    #: teacher-free link losses (link_why.py). Never an input.
    edge_notes: list[tuple[int, int, str, str]] = field(default_factory=list)
    #: Training only: loss scale per opportunity key for this deal view
    #: (DealExample.head_scale); a key not listed counts 1.
    head_scale: dict[str, float] = field(default_factory=dict)
    #: Inputs: the sender's organisation kind, the document's direction and
    #: the deal's stage when the line arrived (c3_record.sender), per line.
    org: list[str] = field(default_factory=list)
    direction: list[str] = field(default_factory=list)
    stage: list[str] = field(default_factory=list)

    def inputs(self) -> dict[str, Any]:
        return {"texts": self.texts, "numbers": self.numbers, "doc_kind": self.doc_kind,
                "role": self.role, "side": self.side, "times": self.times,
                "org": self.org, "direction": self.direction, "stage": self.stage,
                "same_doc": self.same_doc, "same_section": self.same_section,
                "context": self.context, "pixels": self.pixels, "geometry": self.geometry}

    def oneshot(self) -> dict[str, Any]:
        """Atoms only. Stage and geography are null. Links and labels stay targets."""
        raw = self.inputs()
        raw["context"] = {slot: [None] * len(self) for slot in self.context}
        return raw

    def __len__(self) -> int:
        return len(self.texts)


@dataclass(frozen=True)
class Judged:
    """A distinct card subject, including source-conditioned Questions."""
    key: str                     # "jdg:conflict"
    size: str                    # PAIR | GROUP | DEAL
    lines: tuple[int, ...]       # the two lines, the group's lines, or every line
    answer: int                  # class index
    note: str = ""               # the labeler's reason and note, for the teacher
    why_weight: float = 1.0      # DRAFT_WHY_WEIGHT when a model wrote the note
    policy: str = ""             # the note's [<company>] part: the company layer only
    query: str = ""              # source question, never its verdict or expert WHY
    subject_key: str = ""        # distinct card identity even on the same source line
    answers: tuple[int, ...] | None = None # set-valued card fields
    cites: tuple[int, ...] = ()  # lines the card links to (a filled SOW slot's source line)


def judgment_identity(batch: Batch, judgment: Judged) -> tuple:
    """Bind teacher targets to source, question and privileged annotation."""
    return (judgment.subject_key, judgment.key, judgment.lines, judgment.query,
            judgment.answer, judgment.answers, judgment.note, judgment.policy,
            judgment.why_weight, judgment.cites, tuple(batch.texts[i] for i in judgment.lines),
            batch.company, batch.company_policy)


@dataclass(frozen=True)
class FlipTarget:
    """A supposition about line ``line``: read it as if ``condition`` held.
    No answer is attached: the teacher's reading supplies it."""
    line: int
    condition: str


def _words(t: str) -> str:
    return " " + re.sub(r"[^a-z0-9]+", " ", str(t).lower().replace("_", " ")).strip() + " "


def suppositions(why_raw: list[str], per_line: int = 6, min_words: int = 4) -> list[FlipTarget]:
    """Every sentence of every labeled line's WHY, as a supposition.

    Which sentences describe a different case ("if the customer supplied the
    mounts, this would be a customer task") and which only restate the rule
    is not decided here, by any word: the teacher reads the page with each
    one assumed and answers. A sentence that changes nothing teaches the
    heads to hold their answer; one that changes it teaches where the
    boundary is. Sentences under ``min_words`` words are skipped; each line
    keeps its first ``per_line``."""
    out: list[FlipTarget] = []
    for i, why in enumerate(why_raw):
        sents = [t for t in sentences(why) if len(t.split()) >= min_words]
        out += [FlipTarget(i, t) for t in sents[:per_line]]
    return out


def _shingles(t: str, n: int = 3) -> set[str]:
    w = _words(t).strip()
    return {w[k:k + n] for k in range(max(1, len(w) - n + 1))}


def look_alikes(texts: list[str], targets: dict[str, list[int]], labeled: list[bool],
                schema: Schema, threshold: float = 0.6, per_line: int = 3,
                common: int = 200, linked: Iterable[tuple[int, int]] = ()
                ) -> tuple[list[tuple[int, int, str]], list[tuple[int, int]]]:
    """Pairs of labeled lines whose text is nearly the same (character
    trigram Jaccard >= ``threshold``), split two ways:

    * near misses: gold answers differ on a line-size universal opportunity
      (the type first), as (i, j, opportunity);
    * twins: both have a gold type and every opportunity both answered
      agrees, as (i, j): the wording differs where the answer does not care.

    ``linked``: pairs a labeler joined with a ``near_miss`` link. They are
    near misses whatever their wording, which covers lines alike in meaning
    but not in characters, when both are labeled and an answer differs.

    Each line keeps its ``per_line`` closest partners of each kind.
    Candidates come from an inverted index over trigrams, skipping trigrams
    in more than ``common`` lines, so a 2,000-line deal stays cheap."""
    idx = [i for i, x in enumerate(labeled) if x and texts[i].strip()]
    sh = {i: _shingles(texts[i]) for i in idx}
    inv: dict[str, list[int]] = {}
    for i in idx:
        for g in sh[i]:
            inv.setdefault(g, []).append(i)
    keys = ["col:label_type"] + [o.key for o in schema.select(layer="universal")
                                 if o.size == LINE and o.kind in (CLASS, PRESENCE)
                                 and o.key != "col:label_type" and o.key in targets]
    keys = [k for k in keys if k in targets]
    near: dict[int, list[tuple[float, int, str]]] = {}
    same: dict[int, list[tuple[float, int]]] = {}
    for i in idx:
        shared: dict[int, int] = {}
        for g in sh[i]:
            js = inv[g]
            if len(js) > common:
                continue
            for j in js:
                if j > i:
                    shared[j] = shared.get(j, 0) + 1
        for j, c in shared.items():
            jac = c / (len(sh[i]) + len(sh[j]) - c)
            if jac < threshold:
                continue
            both = [k for k in keys if IGNORE not in (targets[k][i], targets[k][j])]
            key = next((k for k in both if targets[k][i] != targets[k][j]), None)
            if key:
                near.setdefault(i, []).append((jac, j, key))
                near.setdefault(j, []).append((jac, i, key))
            elif "col:label_type" in both and texts[i].strip() != texts[j].strip():
                same.setdefault(i, []).append((jac, j))
                same.setdefault(j, []).append((jac, i))
    pairs: set[tuple[int, int, str]] = set()
    for i, cands in near.items():
        for _, j, key in sorted(cands, reverse=True)[:per_line]:
            pairs.add((min(i, j), max(i, j), key))
    for i, j in linked:
        if i == j or not (labeled[i] and labeled[j]):
            continue
        both = [k for k in keys if IGNORE not in (targets[k][i], targets[k][j])]
        key = next((k for k in both if targets[k][i] != targets[k][j]), None)
        if key:
            pairs.add((min(i, j), max(i, j), key))
    twins: set[tuple[int, int]] = set()
    for i, cands in same.items():
        for _, j in sorted(cands, reverse=True)[:per_line]:
            twins.add((min(i, j), max(i, j)))
    return sorted(pairs), sorted(twins)


def find_near_misses(texts: list[str], targets: dict[str, list[int]], labeled: list[bool],
                     schema: Schema, **kw: Any) -> list[tuple[int, int, str]]:
    """The near-miss half of ``look_alikes``."""
    return look_alikes(texts, targets, labeled, schema, **kw)[0]


def _field_value(label: dict[str, Any], source: str, name: str) -> Any:
    if source == "column":
        return label.get(name)
    return (label.get("reads_set") or {}).get(name)


#: How much a WHY counts when a model drafted it and no person has saved it
#: yet (``why_author: machine_draft``), against 1.0 for a WHY a person wrote,
#: edited or accepted. Drafts still teach (the user's call, 2026-10-04), but
#: less: every term that reads the WHY (alignment, sufficiency, echo,
#: suppositions) scales the line by this. Answers are not affected.
DRAFT_WHY_WEIGHT = 0.5


def why_weight(label: dict[str, Any] | None) -> float:
    """``DRAFT_WHY_WEIGHT`` for a WHY nobody has saved, else 1.0. An atom label
    keeps ``why_author`` in ``reads_set``; a judgment row has it as a field."""
    label = label or {}
    reads = label.get("reads_set")
    author = str((reads if isinstance(reads, dict) else {}).get("why_author")
                 or label.get("why_author") or "").strip().lower()
    return DRAFT_WHY_WEIGHT if author == "machine_draft" else 1.0


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
    # An accepted draft's note opens with "Accepted [in bulk] from <x>'s
    # proposal:", which pushed the marker off the start: read past it.
    note = without_opener(str(label.get("note") or "").lstrip()).lstrip().lstrip("[").upper()
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


_POLICY = ("reject", "ignore")
DEAL_KIT_ROUTE = "co_deal_kit_route"


def policy_reject(label: dict[str, Any] | None) -> bool:
    """Our own keep/reject decision (co_action reject or ignore): a real fact
    the company drops on purpose. It is Purtera's signal and never the base's;
    the deal threads set ``rejected`` in the same statement, so the flag alone
    does not mean "not a fact"."""
    reads = (label or {}).get("reads_set")
    act = reads.get("co_action") if isinstance(reads, dict) else None
    return str(act or (label or {}).get("co_action") or "").strip().lower() in _POLICY


from c3_record import is_deal_kit  # noqa: E402  (one definition, shared with the parser's worker)


#: Why the parser lost a hand-added line (app.learning.human_labels.MISS_CAUSE).
MISS_CAUSE = "miss_cause"


def _derived(label: dict[str, Any], universal_reads: frozenset[str] = frozenset(),
             deal_kit: bool = False) -> dict[str, Any]:
    """Fields the card records implicitly, made explicit for the heads:

    * ``admission`` (base): drop for a line that is not a fact (the card's
      reject with no company decision, a noise class, a not-a-fact type; a
      hand-added ``_keep`` aside), keep for a hand-added line or a real type.
      A company reject (co_action reject or ignore) is a real fact we drop on
      purpose: admission keeps its type's answer and the drop trains only the
      Purtera layer through co_action.
    * known negatives: a reading the parser proposed and the labeler removed
      (``reads_shown`` minus ``reads_set``) or one the labeler considered and
      ruled out (``rejected_reads``) is taught as absent, not left unknown.
      On a company reject a removed base reading may be our rule, so only
      company readings are taught absent there.
    * ``miss_cause`` (base): trains only on a hand-added line; dropped from
      any other row, so the cause head never reads a parser atom as a miss.
    * Deal Kit routing: which parser a Deal Kit line trains is our own rule,
      so on a Deal Kit line ``train_for`` moves to the Purtera layer's
      ``co_deal_kit_route`` and the base ``train_for`` head never sees it.
    """
    reads = dict(label.get("reads_set") or {}) if isinstance(label.get("reads_set"), dict) else {}
    typ = str(label.get("label_type") or "").strip()
    origin = str(label.get("origin") or "").strip().lower()
    policy = policy_reject(label)
    out = dict(label)
    real = bool(typ) and typ not in _NOT_FACT
    if policy:
        if real:
            out["admission"] = "keep"
    elif (_rejected_flag(label) or reads.get("noise_class")
          or (typ in _NOT_FACT and not (typ == "_keep" and origin == "labeler"))):
        out["admission"] = "drop"
    elif origin == "labeler" or real:
        out["admission"] = "keep"
    removed = {str(k) for k in (label.get("reads_shown") or [])} - set(reads)
    removed |= {str(k) for k in (label.get("rejected_reads") or {})}
    if origin != "labeler":
        # Why the parser lost a line: asked only of a hand-added (Missed)
        # line. A parser atom was not missed, so it teaches neither a cause
        # nor its absence.
        reads.pop(MISS_CAUSE, None)
        removed.discard(MISS_CAUSE)
    for k in removed:
        if not (policy and k in universal_reads):
            reads.setdefault(k, ABSENT)
    if deal_kit and reads.get("train_for") not in (None, "", []):
        reads[DEAL_KIT_ROUTE] = reads.pop("train_for")
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


#: Labelers that are not people: drafts written for a person to accept
#: (same markers as app.learning.human_labels.NOT_A_PERSON).
NOT_A_PERSON = ("(assistant)", "(bot)", "(model)")


def _is_a_person(labeler: Any) -> bool:
    v = str(labeler or "").strip().lower()
    return bool(v) and not any(m in v for m in NOT_A_PERSON)


def select_labels(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Person-first labels, independent of export order.

    Conflicting saved rows require adjudication. A timestamp or assistant
    mirror must never silently choose a person's training answer. Anonymous
    legacy exports are supported only when no named person exists.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, dict) or not row.get("label_key"):
            raise ValueError("Every training label requires a label_key")
        grouped.setdefault(str(row["label_key"]), []).append(row)
    selected = {}
    metadata = {"id", "labeler", "created_at", "updated_at", "saved_at"}
    for key, candidates in grouped.items():
        people = [r for r in candidates if _is_a_person(r.get("labeler"))]
        eligible = people or [r for r in candidates if not r.get("labeler")]
        if not eligible:
            continue  # assistant proposals are not accepted ground truth
        signatures = {json.dumps({k: v for k, v in r.items() if k not in metadata},
                                 sort_keys=True, default=str) for r in eligible}
        if len(signatures) > 1:
            raise ValueError(f"Conflicting saved labels for {key}; adjudication required")
        selected[key] = min(eligible, key=lambda r: str(r.get("id") or ""))
    return selected


_LEADING_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def saved_number(v: Any) -> float | None:
    """A saved numeric value as one number: "228", "1,200", "55in" (the
    leading number), "4-6" or "4 to 6" (the middle of the range). None when
    no number leads the value."""
    import math

    text = str(v).replace(",", "").strip().lower()
    m = re.match(r"\s*(\d+(?:\.\d+)?)\s*(?:-|–|to)\s*(\d+(?:\.\d+)?)\b", text)
    if m:
        number = (float(m.group(1)) + float(m.group(2))) / 2
    else:
        m = _LEADING_NUMBER.match(text)
        if not m:
            return None
        number = float(m.group(0))
    return number if math.isfinite(number) and number >= 0 else None


def _pair_mask(ids: list[Any]) -> torch.Tensor:
    """[N, N] bool: rows i and j carry the same non-empty id. Built by
    broadcasting: a Python N x N list is half a gigabyte and seconds to
    convert on an 8,192-line window."""
    codes: dict[Any, int] = {}
    v = torch.tensor([-1 if x is None else codes.setdefault(x, len(codes)) for x in ids], dtype=torch.long)
    return (v.view(-1, 1) == v.view(1, -1)) & (v >= 0).view(-1, 1)


def _norm_text(text: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text or "").lower()))


#: Shortest cut-off text (in characters, normalized) matched by its start.
_PREFIX_MIN = 40


def _cut_prefix(text: Any) -> str:
    """A stored text cut off mid-word (the page keeps the first ~120
    characters of a link's from_text): its whole words, normalized."""
    words = _norm_text(text).split()
    return " ".join(words[:-1]) if len(words) > 1 else ""


def _unique_prefix(texts: list[str], text: Any) -> int | None:
    """The one line whose text starts with a cut-off ``text``, or None."""
    head = _cut_prefix(text)
    if len(head) < _PREFIX_MIN:
        return None
    hits = [i for i, t in enumerate(texts) if t.startswith(head)]
    return hits[0] if len(hits) == 1 else None


def _by_ref_text(by_text: dict[str, int], text: Any) -> int | None:
    """The line a pointer's text names. The page sometimes prefixes the text
    with where it came from ("<file tail>: <the line>"), so the part after
    the first ": " is tried too; a heading path ("A > B") names its last
    heading."""
    t = str(text or "").strip()
    if not t:
        return None
    j = by_text.get(_norm_text(t))
    if j is None and ": " in t:
        j = by_text.get(_norm_text(t.split(": ", 1)[1]))
    if j is None and re.search(r"\s[>›]\s", t):           # a heading path: its last heading
        j = by_text.get(_norm_text(re.split(r"\s[>›]\s", t)[-1]))
    return j


class _Lines:
    """Find a deal's lines from what a judgment stores about its subject: an
    atom id (rehashed by every re-parse) or, failing that, the atom's text."""

    def __init__(self, atoms: list[Atom]):
        self.atoms = atoms
        self.by_id = {a.atom_id: i for i, a in enumerate(atoms) if a.atom_id}
        self.by_key = {a.key: i for i, a in enumerate(atoms)}
        self.by_text: dict[str, list[int]] = {}
        self._norm: list[str] | None = None
        for i, a in enumerate(atoms):
            self.by_text.setdefault(_norm_text(a.text), []).append(i)

    def find(self, ref: Any, text: Any = "") -> int | None:
        ref = ref if isinstance(ref, dict) else {}
        if isinstance(ref.get("evidence"), dict):
            ref = ref["evidence"]                 # a site card: its first mention
        i = self.by_key.get(str(ref.get("label_key") or ref.get("labelKey") or ""))
        if i is None:
            i = self.by_id.get(str(ref.get("atomId") or ref.get("atom_id") or ""))
        if i is None:
            matches = self.by_text.get(_norm_text(ref.get("text") or text), [])
            doc = ref.get("artifactId") or ref.get("artifact_id") or ref.get("doc_id")
            if doc:
                matches = [j for j in matches if self.atoms[j].doc_id == str(doc)]
            i = matches[0] if len(matches) == 1 else None
            if i is None and not matches and not doc:
                if self._norm is None:
                    self._norm = [_norm_text(a.text) for a in self.atoms]
                i = _unique_prefix(self._norm, ref.get("text") or text)
        return i

    def group(self, i: int) -> tuple[int, ...]:
        """The lines of line i's table or sheet: same document and section."""
        a = self.atoms[i]
        return tuple(j for j, b in enumerate(self.atoms)
                     if b.doc_id == a.doc_id and (not a.section or b.section == a.section))

    def document(self, artifact_id: str) -> tuple[int, ...]:
        return tuple(j for j, b in enumerate(self.atoms) if artifact_id and b.doc_id == artifact_id)


def _judged_lines(j: dict[str, Any], size: str, lines: _Lines) -> tuple[int, ...] | None:
    """The lines a verdict is about, or None when they are not in this deal."""
    t = j.get("target") if isinstance(j.get("target"), dict) else {}
    parts = [p.strip() for p in str(j.get("text") or "").split("||")]
    if size == DEAL:
        return tuple(range(len(lines.atoms))) or None
    if size == PAIR:
        a, b = lines.find(t.get("a"), parts[0]), lines.find(t.get("b"), parts[-1] if len(parts) > 1 else "")
        return (a, b) if a is not None and b is not None and a != b else None
    if size == GROUP:
        doc = t.get("document") if isinstance(t.get("document"), dict) else None
        if doc is not None:
            return lines.document(str(doc.get("artifactId") or "")) or None
        i = lines.find(t.get("a"))
        return lines.group(i) if i is not None else None
    key = str(j.get("target_key") or "")
    if key.startswith("sup:"):
        # Suppression keys name an occurrence: the card's own copy row, or the
        # set-aside line itself. Never the survivor of a copy, never equal text.
        i = lines.by_key.get(key)
        if i is None and not _sup_is_copy(key, t):
            i = lines.by_key.get(key[4:].split("@", 1)[0])
        return (i,) if i is not None else None
    ref = (t.get("a") or t.get("source") or t.get("site")
           or {"atomId": key.split(":", 1)[1] if key.startswith(("atom:", "site:")) else ""})
    i = lines.find(ref, parts[-1])
    return (i,) if i is not None else None


#: Cards whose subject is named by a query text: Questions (the question) and
#: SOW slots (the section and the kind of fact the slot holds).
QUERY_HEADS = ("gap", "question_answered", "sow_slot", "end_customer", "exclusion_set")
#: Cards that link to the lines that decide them (target.source): the pointer loss.
CITE_HEADS = ("sow_slot", "end_customer")


def judgment_query(j: dict[str, Any]) -> str:
    """Only Questions and SOW slots have a separate source query; no verdict
    or note enters it. A SOW slot's query is its ``target.query`` ("SOW
    section: timeline. Slot: start date.")."""
    head = j.get("head")
    if head not in QUERY_HEADS:
        return ""
    t = j.get("target") if isinstance(j.get("target"), dict) else {}
    if head in ("sow_slot", "end_customer", "exclusion_set"):
        return str(t.get("query") or "").strip()
    return str(t.get("text") or j.get("text") or "").strip()


def judgment_cites(j: dict[str, Any], lines: "_Lines") -> tuple[int, ...]:
    """The lines a card links to as its evidence: a filled SOW slot's
    ``target.source`` (the earlier line that states the slot's fact). Lines
    set aside, or not in this part of the deal, are not cited."""
    if j.get("head") not in CITE_HEADS:
        return ()
    t = j.get("target") if isinstance(j.get("target"), dict) else {}
    refs = t.get("source")
    refs = refs if isinstance(refs, list) else [refs] if isinstance(refs, dict) else []
    out = []
    for ref in refs:
        i = lines.find(ref)
        if i is not None and not excluded(lines.atoms[i].label) and i not in out:
            out.append(i)
    return tuple(out)


def question_field_targets(j: dict[str, Any], schema: Schema) -> list[tuple[str, tuple[int, ...]]]:
    """Registry-backed card fields only; blank means unknown, never negative."""
    if not judgment_query(j):
        return []
    out = []
    fields = j.get("fields") or {}
    if not isinstance(fields, dict):
        raise ValueError("Question fields must be a mapping")
    for name, value in fields.items():
        if value in (None, "", []):
            continue
        opp = schema.of_field("read", name)
        if opp is None:
            raise ValueError(f"Unconsumed question field: {name}")
        if opp.multiple and not isinstance(value, (str, list, tuple)):
            raise ValueError(f"Question answer set must be a list or string: {name}")
        values = _as_list(value) if opp.multiple else [value]
        if not opp.multiple and isinstance(value, list):
            raise ValueError(f"Question field is not set-valued: {name}")
        if opp.kind == BINARY and any(not isinstance(v, bool) and str(v).lower() not in ("true", "false") for v in values):
            raise ValueError(f"Unknown question field value: {name}")
        indexes = [opp.index(v) for v in values]
        if any(i is None for i in indexes):
            raise ValueError(f"Unknown question field value: {name}")
        out.append((opp.key, tuple(sorted(set(indexes)))))
    return out


def judgment_subject(j: dict[str, Any], size: str, lines: _Lines) -> tuple[int, ...] | None:
    if size == DEAL and judgment_query(j):
        # A queried whole-deal card (a SOW slot) reads the deal's lines that
        # are not set aside, like an unanchored question.
        return tuple(i for i, a in enumerate(lines.atoms) if not excluded(a.label)) or None
    idx = _judged_lines(j, size, lines)
    if idx is not None or not judgment_query(j):
        return idx
    t = j.get("target") if isinstance(j.get("target"), dict) else {}
    # An unanchored question can read the deal. A stale named anchor cannot.
    if any(t.get(k) for k in ("a", "source", "site")):
        return None
    if str(j.get("target_key") or "").startswith(("atom:", "site:", "sup:")):
        return None
    return tuple(i for i, a in enumerate(lines.atoms) if not excluded(a.label)) or None


def _open_literal(opp, value: Any) -> str | None:
    """The saved string when the class head only records presence or a bare number."""
    if opp.kind in (CLASS, BINARY, RELATION) or opp.multiple or isinstance(value, list):
        return None
    if value in (None, "", ABSENT):
        return None
    text = str(value).strip()
    if not text or text == ABSENT:
        return None
    if opp.kind == NUMBER:
        if re.fullmatch(r"\d+(?:\.\d+)?", text.replace(",", "")):
            return None
        return text
    if opp.kind == PRESENCE and text.lower() not in ("present", "true", "false"):
        return text
    return None


def _set_class(targets: dict[str, list[int]], key: str, i: int, ix: int | None) -> None:
    if ix is None or key not in targets:
        raise ValueError(f"Missing class target for {key}")
    current = targets[key][i]
    if current not in (IGNORE, ix):
        raise ValueError(f"Conflicting saved target for {key}")
    targets[key][i] = ix


def _set_number(numbers: dict[str, list[float | None]], key: str, i: int, number: float) -> None:
    if key not in numbers:
        raise ValueError(f"Missing number target for {key}")
    current = numbers[key][i]
    if current is not None and current != number:
        raise ValueError(f"Conflicting numeric target for {key}")
    numbers[key][i] = number


def _card_note(j: dict[str, Any], company: str, schema: Schema) -> tuple[str, str, str]:
    """A listed reason code is a class target and stays out of the WHY.

    judgment_note already drops a short code and keeps a sentence. Only a code
    on this head's closed list is returned for the reason head.
    """
    surface = str(j.get("head") or "")
    raw = str(j.get("reason") or "").strip()
    code = raw if raw in schema.reason_codes.get(surface, ()) else ""
    universal, policy = judgment_note(raw, str(j.get("note") or ""), company)
    return universal, policy, code


def _apply_reason(schema: Schema, surface: str, code: str, lines: tuple[int, ...], query: str,
                  targets: dict[str, list[int]], judged: list[Judged], note: str, policy: str,
                  weight: float, subject: str) -> None:
    opp = schema.by_key().get(f"jdg:{surface}:reason")
    if opp is None:
        return
    ix = opp.index(code)
    if ix is None:
        raise ValueError(f"Unknown reason code for {surface}")
    if query or opp.size != LINE:
        judged.append(Judged(key=opp.key, size=opp.size, lines=lines, answer=ix,
                             note=mask_verdict(note) if opp.universal else note,
                             why_weight=weight, policy=policy, query=query, subject_key=subject))
        return
    _set_class(targets, opp.key, lines[0], ix)


def _remember_literal(literals: list[LiteralTarget], line: str, i: int, key: str, text: str) -> None:
    literals.append(LiteralTarget(i, key, text, word_indexes(line, text)))


def _consume_extract(head: str, verdict: str, j: dict[str, Any], i: int, atoms: list[Atom],
                     targets: dict[str, list[int]], numbers: dict[str, list[float | None]],
                     literals: list[LiteralTarget], by_key: dict) -> bool:
    line = atoms[i].text
    if head == "hours":
        basis = hours_judgment_basis(verdict)
        if basis is None:
            return False
        if basis in (UNSTATED, "stated"):
            key = "read:labor_hours"
        elif basis in ("", *HOURS_BASES):
            key = "read:co_hours_estimate"
        else:
            return False
        opp = by_key[key]
        if basis == UNSTATED:
            _set_class(targets, key, i, opp.index(ABSENT))
            return True
        parsed = parse_hours_verdict(verdict)
        if not parsed:
            return False
        _set_class(targets, key, i, opp.index("present"))
        _set_number(numbers, key, i, float(parsed["hours"]))
        _remember_literal(literals, line, i, key, hours_canonical(parsed))
        # The unit and count the hours multiply by, as heads of their own.
        per = hours_per(parsed.get("per"))
        if per and "jdg:hours:per" in by_key:
            _set_class(targets, "jdg:hours:per", i, by_key["jdg:hours:per"].index(per))
            if parsed.get("qty") is not None and "jdg:hours:qty" in by_key:
                _set_class(targets, "jdg:hours:qty", i, by_key["jdg:hours:qty"].index("present"))
                _set_number(numbers, "jdg:hours:qty", i, float(parsed["qty"]))
        return True
    if head == "term":
        parsed = parse_term_verdict(verdict)
        canonical = canonical_term_verdict(verdict)
        if not parsed or not canonical:
            return False
        _set_class(targets, "jdg:term", i, by_key["jdg:term"].index(parsed["kind"]))
        for key, value in parsed.items():
            if key == "kind":
                continue
            slot = by_key[f"jdg:term:{key}"]
            if slot.kind == NUMBER:
                _set_number(numbers, slot.key, i, float(value))
                _set_class(targets, slot.key, i, slot.index("present"))
            else:
                _set_class(targets, slot.key, i, slot.index(value))
        _remember_literal(literals, line, i, "jdg:term", canonical)
        return True
    if head == "commercial":
        if judged_atom_type(j) in TERM_LINE_TYPES:
            raise ValueError("commercial verdict on a term line: re-answer under term")
        parsed = parse_commercial_verdict(verdict)
        if not parsed:
            return False
        if "billing" in parsed:
            _set_class(targets, "jdg:commercial", i, by_key["jdg:commercial"].index(parsed["billing"]))
        for key in ("pm_hours", "pc_hours", "travel_days"):
            if key in parsed:
                slot = by_key[f"jdg:commercial:{key}"]
                _set_number(numbers, slot.key, i, float(parsed[key]))
                _set_class(targets, slot.key, i, slot.index("present"))
        _remember_literal(literals, line, i, "jdg:commercial", canonical_commercial(parsed))
        return True
    if head in LITERAL_HEADS and str(verdict).strip():
        _remember_literal(literals, line, i, f"jdg:{head}", str(verdict).strip())
        return True
    return False


_DEAL_ANSWER_KEYS = {"labeler", "primary_service", "declared_site_count", "note",
                      "purpose", "answered_at", "deal_id"}


def _consume_deal_answers(deal: DealExample, schema: Schema, judged: list[Judged],
                          numbers: dict[str, float], literals: list[LiteralTarget],
                          n: int, company: str) -> None:
    if n == 0:
        return
    service = schema.by_key()["deal:primary_service"]
    counted = schema.by_key().get("deal:site_count")
    absent_count = False
    chosen: int | None = None
    notes: list[str] = []
    for ans in (deal.source_supervision.get("deal_answers") or []):
        if not isinstance(ans, dict):
            continue
        if sorted(set(ans) - _DEAL_ANSWER_KEYS - {"deal_id"}):
            raise ValueError("Unconsumed deal answer field")
        if not _is_a_person(ans.get("labeler")):
            continue
        if str(ans.get("purpose") or "train") in ("eval", "holdout"):
            continue
        unknown = sorted(set(ans) - _DEAL_ANSWER_KEYS)
        if unknown:
            raise ValueError("Unconsumed deal answer field")
        raw = str(ans.get("primary_service") or "").strip()
        if raw:
            ix = service.index(raw)
            if ix is None:
                raise ValueError("Unknown deal primary_service; it is not a training target")
            if chosen not in (None, ix):
                raise ValueError("Conflicting deal primary_service")
            chosen = ix
        if str(ans.get("declared_site_count") or "").strip() == ABSENT:
            # The deal states no site count (a confirmed absence, not a blank).
            if numbers.get("declared_site_count") is not None:
                raise ValueError("Conflicting declared_site_count")
            absent_count = True
        elif "declared_site_count" in ans and ans.get("declared_site_count") not in (None, ""):
            count = site_count(ans.get("declared_site_count"))
            if count is None:
                raise ValueError("Invalid declared_site_count")
            if numbers.get("declared_site_count") not in (None, count):
                raise ValueError("Conflicting declared_site_count")
            numbers["declared_site_count"] = count
        note, _policy = split_note(str(ans.get("note") or ""), company)
        note = mask_verdict(drop_meta(note))
        if note:
            notes.append(note)
            literals.append(LiteralTarget(-1, "deal:note", note, None))
    if chosen is not None:
        judged.append(Judged(key="deal:primary_service", size=DEAL, lines=tuple(range(n)),
                             answer=chosen, note="\n".join(notes), subject_key="deal:primary_service"))
    if counted is not None and (absent_count or "declared_site_count" in numbers):
        if absent_count and "declared_site_count" in numbers:
            raise ValueError("Conflicting declared_site_count")
        judged.append(Judged(key="deal:site_count", size=DEAL, lines=tuple(range(n)),
                             answer=counted.index(ABSENT if absent_count else "present"),
                             note="\n".join(notes), subject_key="deal:site_count"))


def _history_atom(atom: Atom) -> Atom:
    """A recovered miss trains its cause and nothing else."""
    import dataclasses

    reads = (atom.label or {}).get("reads_set") if isinstance((atom.label or {}).get("reads_set"), dict) else {}
    cause = str(reads.get(MISS_CAUSE) or "").strip()
    return dataclasses.replace(atom, label={
        "labeler": (atom.label or {}).get("labeler"),
        "origin": "labeler",
        "reads_set": {MISS_CAUSE: cause},
    })


def featurize(deal: DealExample, schema: Schema, *, absent_is_negative: bool = False,
              max_atoms: int | None = 4096,
              elsewhere: Iterable[tuple[str, str]] = ()) -> Batch:
    """Split one deal into model inputs and per-opportunity targets.

    ``elsewhere``: (line key, relation) pairs whose saved link leaves this
    part of a long deal; those lines are not taught "no link" here.

    ``absent_is_negative``: whether a reading missing from a labeled card is
    a negative example. Default False, matching multitask_table: a missing
    reading is unknown unless a human removed it, so it is not taught as
    "absent". Columns (label_type, about...) are always taught when present.
    """
    import dataclasses

    if max_atoms is not None and len(deal.atoms) > max_atoms:
        raise ValueError("Deal exceeds dense attention budget; relation-preserving chunking "
                         "is required before training (do not truncate the source)")

    from .vocab import canonical_keys

    universal_reads = frozenset(o.field for o in schema.opportunities
                                if o.source == "read" and o.universal)
    atoms = [_history_atom(a) if a.history_only
             else dataclasses.replace(a, label=None) if excluded(a.label)
             else dataclasses.replace(a, label=_derived(a.label, universal_reads, is_deal_kit(a.doc_kind)))
             if a.label else a
             for a in deal.atoms]
    n = len(atoms)
    index = {a.key: i for i, a in enumerate(atoms)}
    company = deal.company or "purtera"

    targets: dict[str, list[int]] = {}
    numbers_target: dict[str, list[float | None]] = {}
    multi_targets: dict[str, list[tuple[int, ...] | None]] = {}
    literals: list[LiteralTarget] = []
    deal_numbers: dict[str, float] = {}
    value_notes: list[str] = []
    for opp in schema.opportunities:
        if opp.kind == RELATION or opp.size != LINE:
            continue
        col = [IGNORE] * n
        if opp.source == "judgment":
            targets[opp.key] = col               # filled from the judgment tabs below
            if opp.kind == NUMBER:
                numbers_target[opp.key] = [None] * n
            continue
        nums: list[float | None] = [None] * n
        selections: list[tuple[int, ...] | None] = [None] * n
        for i, a in enumerate(atoms):
            if not a.label:
                continue
            if a.history_only and opp.field != MISS_CAUSE:
                continue
            v = _field_value(a.label, opp.source, opp.field)
            if v in (None, "", []) and not absent_is_negative:
                continue
            if opp.multiple:
                values = v if isinstance(v, list) else [v]
                indexes = [opp.index(value) for value in values]
                if any(ix is None for ix in indexes):
                    # A value the question doesn't offer (a date typed into
                    # needed_by) teaches nothing; the labeler sees a note.
                    value_notes.append(f"{opp.key} on {a.key} has a value it doesn't offer; "
                                       "that value is not taught")
                    indexes = [ix for ix in indexes if ix is not None]
                    if not indexes:
                        continue
                selections[i] = tuple(sorted(set(indexes)))
                continue
            if isinstance(v, list):
                raise ValueError(f"Unexpected multi-value target for {opp.key} on {a.key}")
            ix = opp.index(v)
            if a.history_only and opp.field == MISS_CAUSE and ix is None:
                raise ValueError("Unknown miss_cause")
            if ix is not None:
                col[i] = ix
            literal = _open_literal(opp, v)
            if literal:
                literals.append(LiteralTarget(i, opp.key, literal, word_indexes(a.text, literal)))
            elif ix is None and v not in (None, "", ABSENT):
                value_notes.append(f"{opp.key} on {a.key} has a value it doesn't offer; "
                                   "that value is not taught")
            if opp.kind == NUMBER and v not in (None, "", ABSENT):
                number = saved_number(v)
                if number is None:
                    value_notes.append(f"{opp.key} on {a.key} is not a number; only that the line "
                                       "states it is taught")
                else:
                    nums[i] = number
        targets[opp.key] = col
        if opp.multiple:
            multi_targets[opp.key] = selections
        if opp.kind == NUMBER:
            numbers_target[opp.key] = nums

    edges: dict[str, list[tuple[int, int]]] = {}
    edges_ok = {o.field for o in schema.select(kind=RELATION)}
    # A link touching a row set aside trains nothing (as in app.learning.human_labels).
    aside = {a.key for a in deal.atoms if excluded(a.label)}
    for src, dst, rel in deal.edges:
        if src in index and dst in index and not {src, dst} & aside:
            pair = (index[src], index[dst])
            if pair not in edges.setdefault(rel, []):
                edges[rel].append(pair)

    why: list[str | None] = []
    why_raw: list[str] = []
    policy: list[str | None] = []
    for a in atoms:
        note = (a.label or {}).get("note") or ""
        u, p = split_note(note, company)
        u, p = drop_meta(u), drop_meta(p)
        why.append(mask_verdict(u) if u else None)
        why_raw.append(u if a.label is not None and not excluded(a.label) else "")
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
    heads_by_field = {o.field: o.key for o in schema.opportunities if o.source in ("read", "column")}
    unlisted_notes = {name: heads_by_field[name[:-len("_note")]]
                      for a in atoms if a.label and isinstance(a.label.get("reads_set"), dict)
                      for name in a.label["reads_set"]
                      if name.endswith("_note") and name not in by_note
                      and name[:-len("_note")] in heads_by_field}
    field_notes: list[dict[str, str]] = []
    weights: list[float] = []
    hint_lines: list[list[int]] = []
    entities: list[list[str]] = []
    by_atom = {a.atom_id: i for i, a in enumerate(atoms) if a.atom_id}
    by_text: dict[str, int] = {}
    repeated_text: set[str] = set()
    for i, a in enumerate(atoms):
        text_key = _norm_text(a.text)
        if text_key in by_text:
            repeated_text.add(text_key)
        by_text.setdefault(text_key, i)
    for text_key in repeated_text:
        by_text.pop(text_key)
    for i, a in enumerate(atoms):
        lb = a.label or {}
        reads = lb.get("reads_set") if isinstance(lb.get("reads_set"), dict) else {}
        company_reject = policy_reject(lb)
        notes = {}
        for name, key in by_note.items():
            t = drop_meta(str(reads.get(name) or ""))
            if t:
                notes[key] = mask_verdict(t) if opp_layer.get(key) == "universal" else t
        for name, value in reads.items():
            # A ``<field>_note`` the registry doesn't list yet (rate_note,
            # wants_note) still explains its field when the field is a head.
            key = unlisted_notes.get(name)
            t = drop_meta(str(value or "")) if key else ""
            if t and key not in notes:
                notes[key] = mask_verdict(t) if opp_layer.get(key) == "universal" else t
        for key, why_not in (lb.get("rejected_reads") or {}).items():
            t = drop_meta(str(why_not or ""))
            k = f"read:{key}"
            if not t or k not in opp_layer:
                continue
            if company_reject and opp_layer[k] == "universal":
                # Ruled out under our own reject: the reason is ours, so the
                # company pass reads it and the base pass never does.
                policy[i] = " ".join(x for x in (policy[i], f"not {key}: {t}") if x)
            else:
                notes.setdefault(k, f"not {key}: {mask_verdict(t) if opp_layer[k] == 'universal' else t}")
        field_notes.append(notes)
        weights.append(a.loss_weight * (
            a.row_weight if a.row_weight is not None and not a.label
            else TIER_WEIGHT.get(str(lb.get("weight_tier") or "").strip().lower(), 1.0)))
        refs = [r for r in (lb.get("hint_refs") or []) if isinstance(r, dict)]
        # A ref names a line by atom id, or (on most real rows) only by the
        # text the labeler pointed at; a heading points at its heading line
        # when the deal has one (a list header is an atom). Its own words and
        # a document type are not another line, so they point nowhere.
        found = set()
        for r in refs:
            j = by_atom.get(str(r.get("atomId")), index.get(str(r.get("atomId"))))
            if j is None and str(r.get("hint") or "") != "own_words" \
                    and str(r.get("kind") or "") != "doc_type":
                j = _by_ref_text(by_text, r.get("text"))
            if j is not None and j != i:
                found.add(j)
        hint_lines.append(sorted(found))
        # One spelling per entity (app/core/label_vocab.json): org:x and party:x pull together.
        entities.append(canonical_keys(_as_list(lb.get("entity_keys"))))

    # The parser's guess, where a person picked another type, is a type the
    # line is known not to be ("Parser's right" is the agreement, already the
    # target). Older rows store the ruled-out type in `rejected` instead.
    by_key_all = schema.by_key()
    type_opp = by_key_all.get("col:label_type")
    negatives: dict[str, list[list[int]]] = {}
    parser_targets: dict[str, list[int]] = {}
    if type_opp is not None:
        said = [type_opp.index(t) if (t := str((a.label or {}).get("parser_type") or "")) else None
                for a in atoms]
        if any(ix is not None for ix in said):
            parser_targets[type_opp.key] = [IGNORE if ix is None else ix for ix in said]
        col = [[] for _ in range(n)]
        for i, a in enumerate(atoms):
            lb = a.label
            if not lb or not _is_a_person(lb.get("labeler") or "person"):
                continue
            typ = str(lb.get("label_type") or "")
            ruled = {str(lb.get("parser_type") or "")}
            if not _rejected_flag(lb):
                ruled.add(str(lb.get("rejected") or "").strip())
            for t in ruled - {"", typ}:
                ix = type_opp.index(t)
                if ix is not None and t != ABSENT:
                    col[i].append(ix)
        if any(col):
            negatives[type_opp.key] = col

    # Judgment tabs. A verdict about one line is that line's target (an alias
    # fills an existing question only where the card left it blank); a verdict
    # about two lines, a group or the deal is kept whole. The labeler's reason
    # and note go with it, for the teacher.
    by_key = by_key_all
    judged: list[Judged] = []
    notes: list[str] = []
    lines = _Lines(atoms)
    for j in deal.judgments:
        if not _is_a_person(j.get("labeler")) or excluded(j):
            continue                               # a verdict itself set aside
        surface = str(j.get("head") or "")
        head, verdict = surface, str(j.get("verdict") or "").strip()
        if head in schema.judgment_aliases:
            key, vmap = schema.judgment_aliases[head]
            verdict = vmap.get(verdict, "")
        else:
            key = f"jdg:{head}"
        opp = by_key.get(key)
        note, pol, reason = _card_note(j, company, schema)
        query = judgment_query(j)
        subject = str(j.get("target_key") or j.get("id") or query or key)
        placed: tuple[int, ...] | None = None
        extract = head in ("hours", "term", "commercial") or head in LITERAL_HEADS
        multi_ix: tuple[int, ...] | None = None
        if opp is not None and opp.multiple and not extract:
            raw = j.get("verdict")
            vals = raw if isinstance(raw, list) else [v for v in str(raw or "").split("|") if v.strip()]
            ixs = [opp.index(str(v).strip()) for v in vals]
            if vals and all(i is not None for i in ixs):
                multi_ix = tuple(sorted(set(ixs)))
                verdict = str(vals[0]).strip()
            else:
                verdict = ""
        if opp is not None and verdict and not extract:
            ix = opp.index(verdict)
            idx = judgment_subject(j, opp.size, lines)
            if idx is not None and opp.size == DEAL and query:
                # featurize's lines carry no label where one is set aside:
                # the deal's own labels say which lines a whole-deal card skips.
                idx = tuple(i for i in idx if not excluded(deal.atoms[i].label)) or None
            kept = ix is not None and idx is not None and not (
                opp.size == PAIR and any(excluded(deal.atoms[i].label) for i in idx))
            if kept and query and any(excluded(deal.atoms[i].label) for i in idx):
                kept = False
            if kept:
                placed = idx
                if query:
                    judged.append(Judged(key=key, size=opp.size, lines=idx, answer=ix, answers=multi_ix,
                                         note=mask_verdict(note) if opp.universal else note,
                                         why_weight=why_weight(j), policy=pol, query=query,
                                         subject_key=subject,
                                         cites=tuple(i for i in judgment_cites(j, lines) if i in idx)))
                    for field_key, answers in question_field_targets(j, schema):
                        field_opp = by_key[field_key]
                        judged.append(Judged(
                            key=field_key, size=LINE, lines=idx, answer=answers[0],
                            answers=answers if field_opp.multiple else None,
                            note=mask_verdict(note) if field_opp.universal else " ".join(x for x in (note, pol) if x),
                            why_weight=why_weight(j), query=query, subject_key=subject))
                elif opp.size == LINE:
                    i = idx[0]
                    if not excluded(deal.atoms[i].label) and targets[key][i] in (IGNORE, ix):
                        targets[key][i] = ix
                        if note:
                            field_notes[i].setdefault(key, mask_verdict(note) if opp.universal else note)
                        if pol:
                            policy[i] = " ".join(x for x in (policy[i], pol) if x)
                    elif not excluded(deal.atoms[i].label) and targets[key][i] != ix:
                        # The line's own card already answers this question
                        # and the judgment tab disagrees: the line card wins
                        # (an alias only fills a blank), and the clash is
                        # reported for the labeler to settle.
                        notes.append(f"{surface} card on {deal.atoms[i].key} disagrees with the line's {key}")
                else:
                    if pol and opp.size == PAIR:
                        for i in idx:
                            policy[i] = " ".join(x for x in (policy[i], pol) if x)
                    judged.append(Judged(key=key, size=opp.size, lines=idx, answer=ix, answers=multi_ix,
                                         note=mask_verdict(note) if opp.universal else note,
                                         why_weight=why_weight(j), policy=pol))
                    if head == "conflict" and verdict in ("contradicts", "supports") and verdict in edges_ok:
                        pair = (max(idx), min(idx))
                        if pair not in edges.setdefault(verdict, []):
                            edges[verdict].append(pair)
                    if head == "site" and verdict == "same_site":
                        for i in idx:
                            entities[i] = [*entities[i], f"site:judged:{j.get('target_key') or idx}"]
        elif verdict and extract:
            idx = judgment_subject(j, LINE, lines)
            if extract:
                if idx is None or any(excluded(deal.atoms[i].label) for i in idx):
                    raise ValueError(f"Unresolved {head} evidence")
                if not _consume_extract(head, verdict, j, idx[0], atoms, targets, numbers_target,
                                        literals, by_key):
                    raise ValueError(f"Unparsed {head} verdict")
                placed = idx
        if placed is None and reason:
            size = opp.size if opp is not None else LINE
            idx = judgment_subject(j, size, lines)
            if idx is not None and not any(excluded(deal.atoms[i].label) for i in idx):
                placed = idx
        if placed is not None and reason:
            _apply_reason(schema, surface, reason, placed, query, targets, judged, note, pol,
                          why_weight(j), subject)
    _consume_deal_answers(deal, schema, judged, deal_numbers, literals, n, company)

    # A line that points at a whole document points at every line of it.
    for src, doc in deal.doc_pointers:
        i = index.get(src)
        if i is not None and src not in aside:
            hint_lines[i] = sorted(set(hint_lines[i]) | {j for j, a in enumerate(atoms)
                                                         if a.doc_id == doc and j != i})

    edge_notes: list[tuple[int, int, str, str]] = []
    # Why a link was drawn: the teacher reads it as the WHY of that relation
    # on the line the link starts from (a [purtera] part goes to the company).
    for (src, dst, rel), note in deal.edge_notes.items():
        i, k = index.get(src), f"rel:{rel}"
        if i is None or dst not in index or k not in by_key or {src, dst} & aside:
            continue
        u, p = split_note(note, company)
        u, p = drop_meta(u), drop_meta(p)
        if u and (i, index[dst]) in edges.get(rel, ()):
            edge_notes.append((i, index[dst], rel, mask_verdict(u)))
        if u:
            prev = field_notes[i].get(k)
            field_notes[i][k] = f"{prev} {mask_verdict(u)}" if prev else mask_verdict(u)
        if p:
            policy[i] = " ".join(x for x in (policy[i], p) if x)

    context = {slot: [_context_value(((a.label or {}).get("reads_set") or {}).get(slot)) for a in atoms]
               for slot in CONTEXT_SLOTS}
    alike = look_alikes([a.text for a in atoms], targets,
                        [a.label is not None and not excluded(a.label) for a in atoms], schema,
                        linked=edges.get("near_miss", []))
    return Batch(
        deal_id=deal.deal_id, company=company, company_policy=deal.company_policy,
        texts=[a.text for a in atoms], numbers=[number_tokens(a.text) for a in atoms],
        doc_kind=[a.doc_kind for a in atoms], role=[a.speaker_role for a in atoms],
        side=[a.speaker_side for a in atoms], times=[a.entered_at for a in atoms],
        org=[a.speaker_org for a in atoms], direction=[a.doc_direction for a in atoms],
        stage=[a.stage_at for a in atoms],
        same_doc=_pair_mask([a.doc_id or None for a in atoms]),
        same_section=_pair_mask([(a.doc_id, a.section) if a.section else None for a in atoms]),
        context=context, targets=targets, numbers_target=numbers_target, edges=edges,
        labeled=[a.label is not None for a in atoms], why=why, policy_note=policy,
        outcome=dict(deal.outcome), rule_links=rule_links, changes=changes,
        field_notes=field_notes, weights=weights, why_weights=[why_weight(a.label) * a.loss_weight for a in atoms],
        hint_lines=hint_lines, entities=entities,
        judged=judged, negatives=negatives,
        flips=suppositions(why_raw), near_misses=alike[0], twins=alike[1],
        multi_targets=multi_targets, literals=literals, deal_numbers=deal_numbers,
        pixels=[a.pixels or _pixels(None) for a in atoms],
        geometry=[a.geometry or _geometry({}) for a in atoms],
        link_elsewhere=_elsewhere_rows(elsewhere, index), notes=value_notes + notes,
        keys=[a.key for a in atoms], parser_targets=parser_targets, head_scale=dict(deal.head_scale),
        layout_rewrites=[(i, t) for i, a in enumerate(atoms) if a.label is not None and not a.history_only
                         for t in a.layout_rewrites if t.strip() != a.text.strip()],
        edge_notes=edge_notes)


def _elsewhere_rows(pairs: Iterable[tuple[str, str]], index: dict[str, int]) -> dict[str, list[int]]:
    out: dict[str, set[int]] = {}
    for key, rel in pairs:
        if key in index:
            out.setdefault(rel, set()).add(index[key])
    return {rel: sorted(rows) for rel, rows in out.items()}


def _context_value(v: Any) -> str | None:
    """A context slot is one token: a multi-site line's tiers join into one."""
    if isinstance(v, (list, tuple)):
        v = "+".join(sorted({str(x) for x in v if str(x).strip()}))
    return str(v) if v not in (None, "") else None


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


__all__ = ["Atom", "Batch", "DRAFT_WHY_WEIGHT", "DealExample", "FlipTarget", "featurize", "find_near_misses", "look_alikes", "suppositions", "number_tokens", "IGNORE",
           "CONTEXT_SLOTS", "ABSENT", "BINARY", "CLASS", "NUMBER", "PRESENCE"]
