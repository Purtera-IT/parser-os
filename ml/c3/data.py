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
from typing import Any, Iterable

from .notes import drop_meta, judgment_note, mask_verdict, sentences, split_note, without_opener
from .schema import (ABSENT, BINARY, CLASS, DEAL, GROUP, LINE, NUMBER, PAIR, PRESENCE,
                     RELATION, Schema)

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
    #: Which of the blob's label and judgment rows the loader kept
    #: (``_one_row_per_key``): person rows, twins a person superseded, keys
    #: only a twin answered, duplicates; and deal answers no head reads.
    label_stats: dict[str, int] = field(default_factory=dict)
    judgment_stats: dict[str, int] = field(default_factory=dict)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "DealExample":
        atoms = [Atom(key=str(a["key"]), text=str(a.get("text", "")),
                      entered_at=_time(a.get("entered_at")),
                      doc_id=str(a.get("doc_id") or a.get("artifact_id") or ""),
                      doc_kind=str(a.get("doc_kind", "")),
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
            outcome={k: float(v) for k, v in (d.get("outcome") or {}).items()},
            judgments=[j for j in d.get("judgments") or [] if isinstance(j, dict)],
            edge_notes={(str(e["src"]), str(e["dst"]), str(e["relation"])): str(e["note"])
                        for e in d.get("edges", []) if e.get("note")},
            doc_pointers=[(str(p[0]), str(p[1])) for p in d.get("doc_pointers", [])])

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
        # One row per key, the person's (``_one_row_per_key``): labels by
        # label_key (an assistant twin only where no person labeled the line),
        # judgments by (head, target_key) and links by (from, to, relation),
        # where a model's draft never trains.
        label_stats: dict[str, int] = {}
        judgment_stats: dict[str, int] = {}
        link_stats: dict[str, int] = {}
        label_rows = _one_row_per_key(blob.get("labels") or [], lambda r: r.get("label_key"),
                                      when="labeled_at", twins=True, stats=label_stats)
        judgments = _one_row_per_key(blob.get("judgments") or [],
                                     lambda r: (str(r.get("head") or ""), str(r.get("target_key") or "")),
                                     when="judged_at", twins=False, stats=judgment_stats)
        link_rows: dict[str, int] = {}
        blob_links = _one_row_per_key(blob.get("links") or [], _link_key, when="created_at",
                                      twins=False, stats=link_rows)
        # The deal-level answers (primary service, declared site count) have no
        # head in this model: app.learning.human_labels trains them (deal_gold,
        # rationale:deal). Counted, so a deal's skipped answers are visible.
        judgment_stats["deal_answers_untrained"] = sum(
            1 for a in blob.get("deal_answers") or [] if isinstance(a, dict) and _is_a_person(a.get("labeler")))
        labels = {lb["label_key"]: lb for lb in label_rows}
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
        for lb in label_rows:
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
        # Cards: every judgment's subject, a draft's too (it is the same card).
        linked, doc_pointers = _blob_links(blob_links, rows, judgments=blob.get("judgments") or [],
                                           labels=labels, stats=link_stats)
        edges += linked
        link_stats.update({k: link_rows[k] for k in ("twin_superseded", "twin_only_skipped", "duplicate")})
        deal = DealExample.from_dict({"deal_id": deal_id, "company": company,
                                      "atoms": rows, "edges": edges, "doc_pointers": doc_pointers,
                                      "judgments": judgments})
        deal.link_stats = link_stats
        deal.label_stats = label_stats
        deal.judgment_stats = judgment_stats
        return deal


#: A relation the model reads only from the later line back to the earlier
#: one (model.CAUSAL_RELATIONS) whose direction a labeler does not mean:
#: either line's card can draw a contradiction.
_LATER_FIRST = ("contradicts",)
#: A link note that says nothing (the card's default).
_EMPTY_LINK_NOTES = ("drawn while labelling the whole deal", "drawn while labelling", "")
#: Judgment cards a link can be drawn on whose key names no line: the link
#: starts from the card's subject line (a conflict or site pair's a side, a
#: question's source line, a site's first mention, the line a parser rule
#: fired on).
_CARD_HEADS = ("conflict", "site", "gap", "site_role", "rule")


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


def _row_file(r: dict[str, Any]) -> str:
    return str(r.get("filename") or (r.get("label") or {}).get("filename") or "")


def _blob_links(links: list[Any], rows: list[dict[str, Any]], *,
                judgments: Iterable[Any] = (), labels: dict[str, Any] | None = None,
                stats: dict[str, int] | None = None
                ) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """The evidence links the labeling page stores at the top of the training
    blob (``doc.links``: from_key, from_text, from_head, to_label_key,
    to_atom_id, to_filename, to_artifact_id, to_text, relation, note,
    labeler), as edges between lines.

    The from side is the card the link was drawn on: its label_key; a Dropped
    card's ``sup:<label_key>@...`` key names that exact copy and a Places
    card's ``site:<atom_id>`` (and an hours or task-tier card's
    ``atom:<atom_id>``) that atom (a dropped copy's text equals its
    original's, so text would find the wrong line); a conflict, site pair,
    Questions, site-role or rule card (``judgments``) starts from its subject
    line, the a side (the b side when the link points at the a side itself).
    The to side is resolved by label_key, then atom id.

    A key this parse does not have is never matched by its text across the
    deal: the same words elsewhere are another line, and a link drawn on a
    line that is gone would land on it. It is healed only when that is safe:
    the link (to_filename / to_artifact_id), the old key's label row
    (``labels``: filename) or the card's line (filename, artifactId) names a
    file, and exactly one line of that file has the text. Otherwise the link
    is dropped and counted (``stale_key_dropped``). An end that carries no key
    at all (a highlighted span the parser made no atom of, to_kind text) is
    found by its text, as it always was.

    An ``answers`` link always runs answer -> question (a Questions card draws
    it the other way round), and a contradiction runs later line -> earlier.
    Model-written drafts never train. The labeler's note goes with the edge.
    A link to a whole document (to_kind text, an artifact and no line) is
    returned as a document pointer. ``stats`` counts what happened to each
    person's link."""
    stats = stats if stats is not None else {}
    for c in ("edges", "doc_pointers", "self_links", "unresolved", "card_anchored", "card_no_line",
              "stale_key_dropped", "healed_in_file", "text_span"):
        stats.setdefault(c, 0)
    labels = labels or {}
    targets = {str(j.get("target_key")): j.get("target") for j in judgments
               if isinstance(j, dict) and j.get("target_key") and isinstance(j.get("target"), dict)}
    cards = {str(j.get("target_key")) for j in judgments
             if isinstance(j, dict) and j.get("target_key") and str(j.get("head") or "") in _CARD_HEADS}
    by_key = {str(r.get("key")): r for r in rows if r.get("key")}
    by_id = {str(r.get("atom_id") or r.get("id")): r for r in rows if r.get("atom_id") or r.get("id")}
    by_text: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_text.setdefault(_norm_text(r.get("text")), []).append(r)

    def heal(text: Any, filename: Any, artifact_id: Any) -> dict[str, Any] | None:
        """The one line of the named file with this text, or None."""
        t, fn, art = _norm_text(text), str(filename or ""), str(artifact_id or "")
        if not t or not (fn or art):
            return None
        hits = [r for r in by_text.get(t, ())
                if (fn and _row_file(r) == fn) or (art and str(r.get("doc_id") or "") == art)]
        return hits[0] if len(hits) == 1 else None

    def stale(key: str, text: Any, filename: Any = "", artifact_id: Any = "") -> tuple[dict[str, Any] | None, str]:
        """An end whose key this parse lacks: healed inside its file, or dropped."""
        lb = labels.get(key) if isinstance(labels.get(key), dict) else {}
        r = heal(text or lb.get("text"), filename or lb.get("filename"), artifact_id)
        return r, ("healed" if r is not None else "stale")

    def ref_line(ref: dict[str, Any]) -> dict[str, Any] | None:
        """A card's line: its atom id, else its text inside its own file."""
        r = by_id.get(str(ref.get("atomId") or ref.get("atom_id") or ""))
        return r if r is not None else heal(ref.get("text"), ref.get("filename"),
                                            ref.get("artifactId") or ref.get("artifact_id"))

    def find_to(k: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
        key, aid = str(k.get("to_label_key") or ""), str(k.get("to_atom_id") or "")
        r = by_key.get(key) or by_id.get(aid)
        if r is not None:
            return r, "key"
        if key or aid:
            return stale(key, k.get("to_text"), k.get("to_filename"), k.get("to_artifact_id"))
        hits = by_text.get(_norm_text(k.get("to_text"))) if str(k.get("to_text") or "").strip() else None
        return (hits[0], "text") if hits else (None, "none")

    def find_from(k: dict[str, Any], to: dict[str, Any] | None
                  ) -> tuple[dict[str, Any] | None, bool, str]:
        """The line a link starts from, whether its key is a judgment card's,
        and how it was found (key, card, healed, stale, text, none)."""
        key = str(k.get("from_key") or "")
        if key.startswith("sup:"):
            lk = key[len("sup:"):].split("@", 1)[0]
            r = by_key.get(lk)
            if r is not None:
                return r, False, "key"
            r, how = stale(lk, k.get("from_text"))
            return r, False, how
        if key.startswith(("site:", "atom:")):
            r = by_id.get(key.split(":", 1)[1])
            if r is not None:
                return r, False, "key"
            for ref in _card_refs(targets.get(key)):   # the atom is gone: its card's line
                r = ref_line(ref)
                if r is not None:
                    return r, False, "healed"
            return None, False, "stale"
        if key in cards:
            for ref in _card_refs(targets.get(key)):
                r = ref_line(ref)
                if r is not None and r is not to:
                    stats["card_anchored"] += 1
                    return r, True, "card"
        r = by_key.get(key) or by_id.get(str(k.get("from_atom_id") or ""))
        if r is not None:
            return r, key in cards, "key"
        if key:
            r, how = stale(key, k.get("from_text"))
            return r, key in cards, how
        hits = by_text.get(_norm_text(k.get("from_text"))) if str(k.get("from_text") or "").strip() else None
        return (hits[0], False, "text") if hits else (None, False, "none")

    out, doc_pointers = [], []
    for k in links:
        if not isinstance(k, dict) or not _is_a_person(k.get("labeler")):
            continue
        rel = str(k.get("relation") or "")
        b, how_b = find_to(k)
        a, on_card, how_a = find_from(k, b)
        if a is not None and b is None and k.get("to_artifact_id") and not k.get("to_label_key") \
                and not k.get("to_atom_id") and str(k.get("to_kind") or "") == "text":
            doc_pointers.append((str(a["key"]), str(k["to_artifact_id"])))
            stats["doc_pointers"] += 1
            continue
        if a is None and on_card:
            stats["card_no_line"] += 1          # e.g. a generated question with no source line
        if not rel or a is None or b is None:
            if "stale" in (how_a, how_b):
                stats["stale_key_dropped"] += 1
            stats["unresolved"] += 1
            continue
        if a is b:
            stats["self_links"] += 1
            continue
        stats["healed_in_file"] += "healed" in (how_a, how_b)
        stats["text_span"] += "text" in (how_a, how_b)
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
    #: What happened to the judgment-tab rows: the loader's row counts
    #: (``DealExample.judgment_stats``) plus, per verdict, trained:<size>,
    #: no_head / no_head:<head>, verdict_outside_set, no_line, set_aside, draft.
    judgment_stats: dict[str, int] = field(default_factory=dict)

    def inputs(self) -> dict[str, Any]:
        return {"texts": self.texts, "numbers": self.numbers, "doc_kind": self.doc_kind,
                "role": self.role, "side": self.side, "times": self.times,
                "same_doc": self.same_doc, "same_section": self.same_section,
                "context": self.context}

    def __len__(self) -> int:
        return len(self.texts)


@dataclass(frozen=True)
class Judged:
    """One judgment-tab verdict about more than one line."""
    key: str                     # "jdg:conflict"
    size: str                    # PAIR | GROUP | DEAL
    lines: tuple[int, ...]       # the two lines, the group's lines, or every line
    answer: int                  # class index
    note: str = ""               # the labeler's reason and note, for the teacher
    why_weight: float = 1.0      # DRAFT_WHY_WEIGHT when a model wrote the note
    policy: str = ""             # the note's [<company>] part: the company layer only


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


def is_deal_kit(doc_kind: str) -> bool:
    """Our own pricing workbook, however the document kind is spelled."""
    return "dealkit" in re.sub(r"[^a-z]", "", str(doc_kind or "").lower())


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
#: (same markers as app.learning.human_labels.NOT_A_PERSON), and the coding
#: assistant's own name however it is suffixed ("claude-code", "claude-code
#: (assistant)").
NOT_A_PERSON = ("(assistant)", "(bot)", "(model)")
NOT_A_PERSON_PREFIX = ("claude-code",)


def _is_a_person(labeler: Any) -> bool:
    v = str(labeler or "").strip().lower()
    return bool(v) and not any(m in v for m in NOT_A_PERSON) and not v.startswith(NOT_A_PERSON_PREFIX)


def _one_row_per_key(rows: Iterable[Any], key: Any, *, when: str = "", twins: bool,
                     stats: dict[str, int] | None = None) -> list[dict[str, Any]]:
    """The person's row for each key, which is the row that trains.

    The page keeps a person's row and the assistant's twin of it (its draft,
    saved under the same key) side by side, and the mirror writes them in time
    order, so a plain "last row wins" let the twin overwrite the person on
    most keys. Here a person's row always wins (the latest one when several
    people answered); a twin stands in only when no person answered that key
    and ``twins`` allows it (labels: a draft line still trains as before).
    Judgments and links pass ``twins=False``: a model's draft verdict or link
    never trains. Rows whose key is empty are kept as they are. ``when`` names
    the row's timestamp field. ``stats`` counts person rows kept, twins a
    person superseded, keys only a twin answered (used or skipped) and
    duplicate person rows."""
    stats = stats if stats is not None else {}
    for c in ("person", "twin_superseded", "twin_used", "twin_only_skipped", "duplicate"):
        stats.setdefault(c, 0)
    groups: dict[Any, list[tuple[int, dict[str, Any]]]] = {}
    loose: list[tuple[int, dict[str, Any]]] = []
    for i, r in enumerate(rows):
        if not isinstance(r, dict):
            continue
        k = key(r)
        if k in (None, "", ()) or (isinstance(k, tuple) and not all(k[:2])):
            loose.append((i, r))
        else:
            groups.setdefault(k, []).append((i, r))
    keep: list[tuple[int, dict[str, Any]]] = []
    for i, r in loose:
        if _is_a_person(r.get("labeler")):
            stats["person"] += 1
            keep.append((i, r))
        elif twins:
            stats["twin_used"] += 1
            keep.append((i, r))
        else:
            stats["twin_only_skipped"] += 1
    for group in groups.values():
        people = [(i, r) for i, r in group if _is_a_person(r.get("labeler"))]
        drafts = len(group) - len(people)
        latest = (lambda g: max(g, key=lambda x: (str(x[1].get(when) or "") if when else "", x[0])))
        if people:
            keep.append(latest(people))
            stats["person"] += 1
            stats["duplicate"] += len(people) - 1
            stats["twin_superseded"] += drafts
        elif twins:
            keep.append(latest(group))
            stats["twin_used"] += 1
            stats["duplicate"] += drafts - 1
        else:
            stats["twin_only_skipped"] += drafts
    return [r for _, r in sorted(keep, key=lambda x: x[0])]


def _link_key(k: dict[str, Any]) -> tuple[str, ...]:
    """One link: where it starts, what it points at, and how."""
    to = k.get("to_label_key") or k.get("to_atom_id") or k.get("to_artifact_id") or _norm_text(k.get("to_text"))
    return (str(k.get("from_key") or ""), str(to or ""), str(k.get("relation") or ""))


def _norm_text(text: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text or "").lower()))


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
        self.by_text: dict[str, int] = {}
        for i, a in enumerate(atoms):
            self.by_text.setdefault(_norm_text(a.text), i)

    def find(self, ref: Any, text: Any = "") -> int | None:
        ref = ref if isinstance(ref, dict) else {}
        if isinstance(ref.get("evidence"), dict):
            ref = ref["evidence"]                 # a site card: its first mention
        i = self.by_id.get(str(ref.get("atomId") or ref.get("atom_id") or ""))
        if i is None:
            i = self.by_text.get(_norm_text(ref.get("text") or text)) if (ref.get("text") or text) else None
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
    ref = (t.get("a") or t.get("source") or t.get("site")
           or {"atomId": key.split(":", 1)[1] if key.startswith(("atom:", "site:")) else ""})
    i = lines.find(ref, parts[-1])
    return (i,) if i is not None else None


def featurize(deal: DealExample, schema: Schema, *, absent_is_negative: bool = False) -> Batch:
    """Split one deal into model inputs and per-opportunity targets.

    ``absent_is_negative``: whether a reading missing from a labeled card is
    a negative example. Default False, matching multitask_table: a missing
    reading is unknown unless a human removed it, so it is not taught as
    "absent". Columns (label_type, about...) are always taught when present.
    """
    import dataclasses

    from .vocab import canonical_keys

    universal_reads = frozenset(o.field for o in schema.opportunities
                                if o.source == "read" and o.universal)
    atoms = [dataclasses.replace(a, label=None) if excluded(a.label)
             else dataclasses.replace(a, label=_derived(a.label, universal_reads, is_deal_kit(a.doc_kind)))
             if a.label else a
             for a in deal.atoms]
    n = len(atoms)
    index = {a.key: i for i, a in enumerate(atoms)}
    company = deal.company or "purtera"

    targets: dict[str, list[int]] = {}
    numbers_target: dict[str, list[float | None]] = {}
    for opp in schema.opportunities:
        if opp.kind == RELATION or opp.size != LINE:
            continue
        col = [IGNORE] * n
        if opp.source == "judgment":
            targets[opp.key] = col               # filled from the judgment tabs below
            continue
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
    field_notes: list[dict[str, str]] = []
    weights: list[float] = []
    hint_lines: list[list[int]] = []
    entities: list[list[str]] = []
    by_atom = {a.atom_id: i for i, a in enumerate(atoms) if a.atom_id}
    by_text: dict[str, int] = {}
    for i, a in enumerate(atoms):
        by_text.setdefault(_norm_text(a.text), i)
    for i, a in enumerate(atoms):
        lb = a.label or {}
        reads = lb.get("reads_set") if isinstance(lb.get("reads_set"), dict) else {}
        company_reject = policy_reject(lb)
        notes = {}
        for name, key in by_note.items():
            t = drop_meta(str(reads.get(name) or ""))
            if t:
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
        weights.append(TIER_WEIGHT.get(str(lb.get("weight_tier") or "").strip().lower(), 1.0))
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
    if type_opp is not None:
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
    #
    # What happened to each verdict is counted in ``judgment_stats`` (on top
    # of the loader's row counts, ``DealExample.judgment_stats``). A head with
    # no question in this model is skipped and counted as ``no_head:<head>``:
    # judgment_heads.json ``not_trained`` says why for each (``rule``: a parser
    # rule fires before atoms exist, so the model never sees its input, though
    # a link drawn on a rule card still trains from the card's line; ``norm``,
    # ``hours``, ``commercial``, ``term``: the answer is a value, not a class).
    # Deal-size heads that do have a question (billing_type, the *_scope heads,
    # tier) train as ``Judged`` over every line.
    by_key = by_key_all
    judged: list[Judged] = []
    lines = _Lines(atoms)
    jstats: dict[str, int] = dict(deal.judgment_stats)

    def count(what: str) -> None:
        jstats[what] = jstats.get(what, 0) + 1

    for j in deal.judgments:
        head, verdict = str(j.get("head") or ""), str(j.get("verdict") or "").strip()
        if not _is_a_person(j.get("labeler")):
            count("draft")                         # a model's verdict never trains
            continue
        if excluded(j):
            count("set_aside")                     # a verdict itself set aside
            continue
        if head in schema.judgment_aliases:
            key, vmap = schema.judgment_aliases[head]
            verdict = vmap.get(verdict, "")
        else:
            key = f"jdg:{head}"
        opp = by_key.get(key)
        if opp is None:
            count("no_head")
            count(f"no_head:{head}")
            continue
        ix = opp.index(verdict) if verdict else None
        if ix is None:
            count("verdict_outside_set")
            continue
        idx = _judged_lines(j, opp.size, lines)
        if idx is None:
            count("no_line")
            continue
        if opp.size == PAIR and any(excluded(deal.atoms[i].label) for i in idx):
            count("set_aside")
            continue                               # a pair touching a row set aside, as links
        # The note splits as an atom's or a link's does: the universal WHY is
        # the teacher's, a [purtera] line goes to the company layer only.
        note, pol = judgment_note(j.get("reason"), j.get("note"), company)
        if opp.size == LINE:
            i = idx[0]
            if excluded(deal.atoms[i].label):
                count("set_aside")
                continue                           # set aside: trains nothing
            count(f"trained:{opp.size}")
            if targets[key][i] == IGNORE:
                targets[key][i] = ix
                if note:
                    field_notes[i].setdefault(key, mask_verdict(note) if opp.universal else note)
                if pol:
                    policy[i] = " ".join(x for x in (policy[i], pol) if x)
            continue
        if pol and opp.size == PAIR:
            # About both lines, as a link's company part is about its line.
            for i in idx:
                policy[i] = " ".join(x for x in (policy[i], pol) if x)
        # A group or deal verdict keeps its company part on itself: copied to
        # every line of a sheet or deal it would repeat one rule hundreds of times.
        count(f"trained:{opp.size}")
        judged.append(Judged(key=key, size=opp.size, lines=idx, answer=ix,
                             note=mask_verdict(note) if opp.universal else note,
                             why_weight=why_weight(j), policy=pol))
        if head == "conflict" and verdict in ("contradicts", "supports") and verdict in edges_ok:
            pair = (max(idx), min(idx))            # the later line points back
            if pair not in edges.setdefault(verdict, []):
                edges[verdict].append(pair)
        if head == "site" and verdict == "same_site":
            for i in idx:
                entities[i] = [*entities[i], f"site:judged:{j.get('target_key') or idx}"]

    # A line that points at a whole document points at every line of it.
    for src, doc in deal.doc_pointers:
        i = index.get(src)
        if i is not None and src not in aside:
            hint_lines[i] = sorted(set(hint_lines[i]) | {j for j, a in enumerate(atoms)
                                                         if a.doc_id == doc and j != i})

    # Why a link was drawn: the teacher reads it as the WHY of that relation
    # on the line the link starts from (a [purtera] part goes to the company).
    for (src, dst, rel), note in deal.edge_notes.items():
        i, k = index.get(src), f"rel:{rel}"
        if i is None or dst not in index or k not in by_key or {src, dst} & aside:
            continue
        u, p = split_note(note, company)
        u, p = drop_meta(u), drop_meta(p)
        if u:
            prev = field_notes[i].get(k)
            field_notes[i][k] = f"{prev} {mask_verdict(u)}" if prev else mask_verdict(u)
        if p:
            policy[i] = " ".join(x for x in (policy[i], p) if x)

    context = {slot: [((a.label or {}).get("reads_set") or {}).get(slot) for a in atoms]
               for slot in CONTEXT_SLOTS}
    alike = look_alikes([a.text for a in atoms], targets,
                        [a.label is not None and not excluded(a.label) for a in atoms], schema,
                        linked=edges.get("near_miss", []))
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
        field_notes=field_notes, weights=weights, why_weights=[why_weight(a.label) for a in atoms],
        hint_lines=hint_lines, entities=entities,
        judged=judged, negatives=negatives, judgment_stats=jstats,
        flips=suppositions(why_raw), near_misses=alike[0], twins=alike[1])


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
