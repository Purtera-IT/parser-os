"""Human atom labels (purpulse atom labeler) -> training rows.

The labeler writes one JSON per deal to blob
``orbitbrief-artifacts/_labeling/labels/<deal_id>.json`` (Platform-infra
``shared/atom-labeling.js``). Each label carries what the labeler SAW, not
just what they chose: section, intro line, the atoms above and below, the
document type, and the hint chips naming which of those told them the type.
That context is what the heads were missing (only 3.7% of training rows
carried any), so it rides into every row's provenance.

Rows go to ``_training_human.db`` in the same ``training_rows`` shape the
multitask builder globs (``_training_*.db``), with ``teacher="human"``, which
outranks every other teacher on dedup. Labels from an assignment marked
``purpose="eval"`` force ``split="holdout"`` for the whole deal: the locked
eval set must never leak into training.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from app.core.atom_type_registry import KEEP, coarse_of, facet_of, load_registry
from app.learning.label_context import context_note, context_text
from app.learning.label_features import features_for
from app.learning.span_ranker import best_locatable, pointer_kind

#: Same representation as typed_atom_classifier.DECIDE_TEXT_VERSION (v2).
DECIDE_TEXT_VERSION = 2
HUMAN_TEACHER = "human"

_COLUMNS = (
    "relation", "label", "raw_text", "masked_text", "label_kind", "teacher",
    "weight", "confidence", "scope", "scope_key", "deal_id", "project_id",
    "provenance", "created_at", "split",
)


def _as_list(v: Any) -> list[str]:
    if isinstance(v, str):
        return [v] if v else []
    return [str(x) for x in (v or []) if x]


def decide_text(label: dict[str, Any]) -> str:
    """The v2 decide-text the heads are served, rebuilt from what the labeler saw."""
    text = " ".join(str(label.get("text") or "").split())
    table_ref = str(label.get("table_ref") or "").strip()
    section = " > ".join(_as_list(label.get("section")))[:200]
    lead_in = " › ".join(_as_list(label.get("lead_in")))[:300]
    if table_ref:
        text = f"{text} [table: {table_ref}]"
    if section:
        text = f"{text} [section: {section}]"
    if lead_in:
        text = f"{text} [intro: {lead_in}]"
    return text


#: A reading whose answer is one of a fixed set is a classification target.
#: One whose value is a phrase ("what was promised") is not -- the learnable
#: thing there is whether the atom carries one at all, and the phrase rides in
#: provenance for a span head that does not exist yet.
def _closed_read_values() -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for r in load_registry().get("reads") or []:
        vals = str(r.get("values") or "")
        if "|" in vals:
            out[str(r.get("key"))] = {v.strip() for v in vals.split("|") if v.strip()}
        elif vals in {"true", "true/false"}:
            out[str(r.get("key"))] = {"true", "false"}
    return out


CLOSED_READS = _closed_read_values()
PRESENT = "present"
#: A reading the parser proposed and a human took off. The only negative we
#: can state without assuming: a chip nobody ticked may simply not have been
#: considered, but a chip the parser ticked and the labeler cleared is a
#: decision. Without it a presence task has one class and cannot train.
ABSENT = "absent"


@dataclass
class IngestReport:
    deals: int = 0
    labels: int = 0
    rows: int = 0
    deal_answers: int = 0
    skipped: dict[str, int] = field(default_factory=dict)

    def skip(self, why: str) -> None:
        self.skipped[why] = self.skipped.get(why, 0) + 1



#: A labeler name may carry a parenthesised marker saying it is not a person.
#: Offline zip exports label with a filename and meeting exports with a
#: person's name, so "not an email" cannot be the test -- the marker is.
NOT_A_PERSON = ("(assistant)", "(bot)", "(model)")


def _is_a_person(labeler: Any) -> bool:
    """Gold comes from a person. A draft written for one to accept is not."""
    v = str(labeler or "").strip().lower()
    return not any(m in v for m in NOT_A_PERSON)


#: The reading that marks THE line a deal is about -- "4 TVs install in
#: CheckOut New York office." -- the one a router reads service, quantity, site
#: and deal type from. There is one per deal, which is what makes it learnable
#: from a single tick: once a person has picked it, every other line they
#: labeled on that deal is a line they decided was NOT it.
DEAL_SUMMARY = "deal_summary"


def _marks_summary(lb: dict[str, Any]) -> bool:
    reads = lb.get("reads_set")
    if not isinstance(reads, dict) or DEAL_SUMMARY not in reads:
        return False
    v = reads[DEAL_SUMMARY]
    return v is True or str(v or "").strip().lower() == "true"


def _with_one_deal_summary(labels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """At most one `deal_summary` per deal per labeler, and the rest as negatives.

    The page saves each card on its own, so a labeler who changes their mind
    leaves two lines marked. The latest mark wins (``labeled_at``, then file
    order) and an earlier one becomes `false`: it was considered and moved off.
    Every other label by a labeler who marked a summary on this deal gets
    `false` too. Without that a one-per-deal reading has one class and no head
    can learn it; with it, each deal is one positive against all of its lines.
    A labeler who marked nothing on the deal teaches nothing either way.
    """
    latest: dict[str, tuple[str, int]] = {}
    for i, lb in enumerate(labels):
        if _is_a_person(lb.get("labeler")) and _marks_summary(lb):
            who = str(lb.get("labeler") or "")
            at = (str(lb.get("labeled_at") or ""), i)
            if who not in latest or at >= latest[who]:
                latest[who] = at
    if not latest:
        return labels
    out = []
    for i, lb in enumerate(labels):
        who = str(lb.get("labeler") or "")
        if who not in latest or not _is_a_person(lb.get("labeler")):
            out.append(lb)
            continue
        reads = dict(lb.get("reads_set") or {}) if isinstance(lb.get("reads_set"), dict) else {}
        reads[DEAL_SUMMARY] = True if latest[who][1] == i else "false"
        out.append({**lb, "reads_set": reads})
    return out


def rows_for_deal(doc: dict[str, Any], report: IngestReport | None = None) -> list[dict[str, Any]]:
    from app.core.training_log import assign_split

    report = report if report is not None else IngestReport()
    deal_id = str(doc.get("deal_id") or "").strip()
    labels = [lb for lb in doc.get("labels") or [] if isinstance(lb, dict)]
    labels = _with_one_deal_summary(labels)
    if not deal_id or not (labels or doc.get("judgments") or doc.get("links")):
        report.skip("deal file without deal_id or labels")
        return []
    is_eval = doc.get("purpose") == "eval" or any(lb.get("purpose") == "eval" for lb in labels)
    split = "holdout" if is_eval else assign_split(deal_id)
    out: list[dict[str, Any]] = []
    for lb in labels:
        if not _is_a_person(lb.get("labeler")):
            # A draft I wrote for a labeler to accept or replace is on the
            # card on purpose -- and it is not gold. Ingesting it as
            # teacher="human" would train the heads on the assistant's own
            # answers and call them a person's.
            report.skip("labeler is not a person")
            continue
        fine = str(lb.get("label_type") or "").strip()
        if not fine:
            report.skip("label without label_type")
            continue
        report.labels += 1
        # The labeler stores the exact string the heads are served (computed
        # server-side from the envelope, pinned to _atom_decide_text by shared
        # vectors). Rebuilding from fields is the fallback for older rows. A
        # label with no context at all (offline zip exports) is bare text:
        # version 0, so a trainer never mistakes it for v2.
        stored = " ".join(str(lb.get("decide_text") or "").split())
        text = stored or decide_text(lb)
        has_context = bool(stored) or any(lb.get(k) for k in ("section", "lead_in", "table_ref"))
        version = DECIDE_TEXT_VERSION if has_context else 0
        if len(text) < 3:
            report.skip("text too short")
            continue
        coarse = str(lb.get("coarse") or "").strip() or coarse_of(fine)
        facet = KEEP if fine == KEEP else facet_of(fine)
        prov = {
            "decide_text_version": version,
            "source": lb.get("source") or "purpulse_atom_labeler",
            "correction_id": lb.get("correction_id"),
            "label_key": lb.get("label_key"),
            "atom_id": lb.get("atom_id"),
            "compile_id": lb.get("compile_id"),
            "parser_type": lb.get("parser_type"),
            "is_new_type": bool(lb.get("is_new_type")),
            "hints": _as_list(lb.get("hints")),
            # The exact thing behind each chip: the words, the heading, the
            # atom above. Nothing trains on these yet -- they are the span
            # supervision a span head will need, and throwing them away now
            # means labeling this deal twice later.
            "hint_refs": [r for r in (lb.get("hint_refs") or []) if isinstance(r, dict)][:20],
            # Facts a sentence encoder cannot see. Measured: the two
            # `blocked_on` answers sit at cosine 0.9967 as text, and the best
            # of six phrasings reached 0.9925. As numbers they come apart.
            "features": features_for(lb),
            "doc_type": lb.get("doc_type"),
            "filename": lb.get("filename"),
            "page": lb.get("page"),
            "neighbors_above": _as_list(lb.get("neighbors_above"))[:3],
            "neighbors_below": _as_list(lb.get("neighbors_below"))[:3],
            "entity_keys": _as_list(lb.get("entity_keys")),
            "note": lb.get("note") or "",
            "labeler": lb.get("labeler") or "",
            "purpose": lb.get("purpose") or "train",
        }
        base = {
            "raw_text": text, "masked_text": text, "teacher": HUMAN_TEACHER,
            "weight": _row_weight(lb), "confidence": 1.0, "scope": "deal", "scope_key": deal_id,
            "deal_id": deal_id, "project_id": deal_id,
            "created_at": lb.get("labeled_at") or "", "split": split,
            "provenance": json.dumps(prov, ensure_ascii=False),
        }
        out.append({**base, "relation": "atom_type", "label": fine, "label_kind": "type"})
        if coarse:
            out.append({**base, "relation": "atom_type_coarse", "label": coarse, "label_kind": "type"})
        if facet:
            out.append({**base, "relation": "facet", "label": facet, "label_kind": "facet"})
        else:
            report.skip("no facet (proposed type not in registry yet)")
        out.extend(_axis_rows(lb, base, prov, report))
    out.extend(_judgment_rows(doc, deal_id, split, report))
    out.extend(_question_rows(doc, labels, deal_id, split, report))
    out.extend(_link_rows(doc, deal_id, split, report))
    out.extend(deal_rationale_rows(doc, deal_id, split))
    report.rows += len(out)
    return out



#: What a row is worth to a head. `retrain.py` has always passed `weight` into
#: NeuralHead.fit as sample_weight; every row ever written carried 1.0, so the
#: mechanism was wired to a constant.
#:
#: load_bearing -- get it wrong and the quote, the scope or the site is wrong.
#: slight       -- true, and nothing downstream turns on it.
_TIER_WEIGHT = {"load_bearing": 3.0, "ordinary": 1.0, "slight": 0.3}


#: Label types that say "this line should never have been an atom". The
#: admission head reads them as `drop`; the type head still learns the class.
ADMISSION_DROP_TYPES = frozenset({KEEP, "small_talk"})

#: Values of the `rejected` column that are a flag, not a type name.
_REJECT_FLAGS = frozenset({"true", "t", "1", "yes", "false", "f", "0", "no"})


def _row_weight(lb: dict[str, Any]) -> float:
    return _TIER_WEIGHT.get(str(lb.get("weight_tier") or "").strip().lower(), 1.0)



#: Notes the card writes on a labeler's behalf. They say when something was
#: drawn, never why, and a rationale target built from one teaches the model to
#: produce filler.
_EMPTY_NOTES = frozenset({
    "drawn while labelling the whole deal",
    "drawn while labelling",
    "",
})


def _rationale_row(kind: str, prompt: str, note: str, base: dict[str, Any],
                   prov: dict[str, Any], weight: float = 1.0) -> dict[str, Any]:
    """One (what was in front of me) -> (what I argued) pair.

    The label is a paragraph, which no classifier can use and every generative
    head can. Kept under its own relation so the backbone builder skips it: a
    task it does not list is a task it ignores, which is exactly the behaviour
    wanted here.
    """
    prompt = " ".join(str(prompt or "").split())
    return {
        **base,
        "relation": f"rationale:{kind}",
        "label": str(note or "").strip(),
        "raw_text": prompt,
        "masked_text": prompt,
        "label_kind": "rationale",
        "weight": weight,
        "provenance": json.dumps(prov, ensure_ascii=False),
    }


def _norm(s: Any) -> str:
    return " ".join(str(s or "").split()).lower()


def _axis_row(relation: str, label_value: str, lb: dict[str, Any], base: dict[str, Any],
              prov: dict[str, Any], kind: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """One row, served the context THIS head needs and nothing else."""
    text = context_text(relation, lb)
    return {
        **base,
        "raw_text": text,
        "masked_text": text,
        "relation": relation,
        "label": label_value,
        "label_kind": kind,
        "provenance": json.dumps({**prov, **context_note(relation), **(extra or {})}, ensure_ascii=False),
    }


def _axis_rows(lb: dict[str, Any], base: dict[str, Any], prov: dict[str, Any],
               report: IngestReport) -> list[dict[str, Any]]:
    """`about`, `wants` and every reading the labeler set.

    These are three quarters of what a person records on a card and none of
    them reached a head before: the labeler answered them, the database kept
    them, and the training rows stopped at the type.
    """
    rows: list[dict[str, Any]] = []
    for axis in ("about", "wants", "consumer", "weight_tier", "decided_by"):
        v = str(lb.get(axis) or "").strip()
        if v:
            rows.append(_axis_row(axis, v, lb, base, prov, "judgment"))

    # The contrast, as its own row. A trainer joins these to the atom_type rows
    # on `label_key` in the provenance and gets (anchor, positive, negative):
    # "Relay under 'Provided by us' is a bom_line and specifically NOT the
    # deal_metadata its twin on the sheet is." Left empty where nothing else was
    # ever in the running -- a false contrast teaches a boundary that is not there.
    # The spans the labeler pointed at. 56 of 65 notes on 010288 quote the
    # deciding words verbatim, and hint_refs already hold them structured, each
    # tagged with the hint that says WHY it mattered. A quotation is not prose:
    # it is a pointer at the part of the document that settled the label, and a
    # head trained to produce it can run at inference -- pointing at evidence
    # needs the document, not the note. It is also the only kind of answer a
    # person can audit at a glance.
    #
    # Three destinations, not one. Measured on 010288: of 159 pointers, 82 are
    # words in the atom or the context printed beside it, 21 name another atom,
    # and 56 are the envelope or the document type. All 159 were being emitted
    # as `evidence_span`, so a third of the span supervision asked a head to
    # locate text that is nowhere on the page it is holding -- which is how a
    # span head learns to invent one.
    #
    #   evidence_span  the ranker can find these.
    #   evidence_doc   real words on another surface: retrieval, not extraction.
    #   decided_from   WHICH context field settled it. Free on every pointer,
    #                  learnable from all 159, and the answer to "why did you
    #                  say that" that a person can read.
    seen_fields: set[str] = set()
    for ref in (lb.get("hint_refs") or []):
        if not isinstance(ref, dict):
            continue
        span = " ".join(str(ref.get("text") or "").split())
        hint = str(ref.get("hint") or "").strip()
        if len(span) < 8 or not hint:
            continue
        where = {"span_kind": ref.get("kind"),
                 "span_atom_id": ref.get("atomId"),
                 "span_filename": ref.get("filename")}
        kind = pointer_kind(ref, lb)
        if kind == "span":
            # A pointer is usually a selection and occasionally a paraphrase.
            # "the two headings in dispute: ..." names words that ARE on the
            # page, in a sentence the labeler rewrote -- and a span head given
            # the rewrite is being taught to produce text its prompt does not
            # contain. Snap those to the words as written; leave an exact
            # selection exactly as the labeler made it, since it is the more
            # precise of the two.
            prompt = context_text(f"evidence_span:{hint}", lb)
            if _norm(span) not in _norm(prompt):
                snapped = best_locatable(span, lb, prompt)
                if snapped is None:
                    rows.append(_axis_row("rationale:evidence", span, lb, base,
                                          prov, "generative", where))
                    continue
                where = {**where, "span_as_written_by_labeler": span}
                span = snapped
            rows.append(_axis_row(f"evidence_span:{hint}", span, lb, base, prov,
                                  "span", where))
        elif kind == "other_atom":
            rows.append(_axis_row(f"evidence_doc:{hint}", span, lb, base, prov,
                                  "retrieval", where))
        elif kind == "prose":
            # The hint says "section" and the words are the labeler's own --
            # "the two headings in dispute: ...". Argument, not a selection, so
            # it goes where the argument goes rather than teaching a span head
            # to produce text that is not on the page.
            rows.append(_axis_row("rationale:evidence", span, lb, base, prov,
                                  "generative", where))
        elif kind == "field":
            # `decided_from` above records WHICH field settled it, and that is
            # the learnable part. But the rendering is not only a field: 56 of
            # 010288's pointers are these, and among their 12 distinct texts
            # are "purtera-it.com (internal, ours) -- internal only, never
            # leaves our org" and "'us' in this list is the reseller, not us".
            # That is an argument about the envelope, and keeping only the
            # word `who_said_it` throws it away.
            rows.append(_axis_row("rationale:evidence", span, lb, base, prov,
                                  "generative", where))
        if hint not in seen_fields:
            seen_fields.add(hint)
            rows.append(_axis_row("decided_from", hint, lb, base, prov, "axis", {}))

    # The argument itself, as a target. The prompt is what the labeler was
    # looking at; the label is what they concluded and why.
    note = str(lb.get("note") or "").strip()
    if len(note) >= 40:
        chose = " | ".join(x for x in (
            f"type={lb.get('label_type')}",
            f"about={lb.get('about')}" if lb.get("about") else "",
            f"wants={lb.get('wants')}" if lb.get("wants") else "",
            f"supplier={lb.get('supplier')}" if lb.get("supplier") else "",
        ) if x)
        rows.append(_rationale_row(
            "atom", f"{context_text('atom_type', lb)}\nCHOSE: {chose}",
            note, base, prov, _row_weight(lb)))

    # ADMISSION: should this text have been an atom at all?
    #
    # The one axis a labeler cannot teach by judging what is on screen. An atom
    # exists only where the parser admitted it, so every ordinary label is a
    # positive and the boundary has one side. Two rows fix that, and both come
    # from things a labeler already does:
    #
    #   origin=labeler  they highlighted text the parser made no atom of, so
    #                   the admission decision was WRONG to skip it.
    #   _keep           the parser made an atom and a person said it is not a
    #                   fact -- boilerplate, a header, table scaffolding --
    #                   so it was wrong to admit it.
    #
    # Measured 2026-09-27 before this existed: 65 rule decisions joined to
    # labels across two deals, 65 positives, 0 negatives. `origin` was written
    # on every row and read by nothing.
    # The vocabulary is the one HEAD_REGISTRY["admission"] already declares and
    # PM corrections already write: ("keep", "drop"). Note the collision of
    # names -- the label TYPE `_keep` means "not a fact worth typing", and the
    # admission verdict `keep` means "this is work this deal quotes". A `_keep`
    # label is therefore admission `drop`.
    #
    # `small_talk` is the same verdict with a name: "Hi Trent,", "Hope you had
    # a great 4th of July!", "Thank you,". The atom-types registry says it
    # "carries no fact about the work", so it is an admission `drop` -- and,
    # unlike `_keep`, even when the labeler highlighted it by hand. Greetings
    # and sign-offs are cut by a regex before they become atoms, so the only
    # way a person can show the admission head one is to highlight it and say
    # "small talk"; reading that as "the parser missed a fact" would teach the
    # exact opposite.
    origin = str(lb.get("origin") or "").strip().lower()
    label_type = str(lb.get("label_type") or "").strip()
    if label_type in ADMISSION_DROP_TYPES and (label_type != KEEP or origin != "labeler"):
        rows.append(_axis_row("admission", "drop", lb, base, prov, "judgment",
                              {"parser_admitted_a_non_fact": origin != "labeler",
                               "rejected_as": label_type, "origin": origin or "parser"}))
    elif origin == "labeler":
        rows.append(_axis_row("admission", "keep", lb, base, prov, "judgment",
                              {"parser_missed": True, "origin": "labeler"}))

    # `rejected` is two things in one column: the labeling page and
    # write_labels.py store the FLAG "true" on a reject, while older rows hold
    # the type the labeler ruled out. Only the second is a contrastive pair; a
    # flag here minted `rejected="true"` rows -- a class called "true".
    rejected = str(lb.get("rejected") or "").strip()
    if rejected.lower() in _REJECT_FLAGS:
        rejected = ""
    if rejected and rejected != str(lb.get("label_type") or "").strip():
        rows.append(_axis_row("rejected", rejected, lb, base, prov, "judgment",
                              {"chosen": lb.get("label_type"),
                               "contrastive_pair": True}))

    reads = lb.get("reads_set")
    if not isinstance(reads, dict):
        return rows
    # Readings the labeler CONSIDERED and ruled out. The reading that nearly
    # fitted is the best negative there is -- "blocked_on is for a conditional
    # whose gate is a person; this one's gate is an outcome" teaches the
    # boundary in a way no positive example can, and until now there was
    # nowhere to put it, so it lived in prose and taught nothing.
    for key, why in (lb.get("rejected_reads") or {}).items():
        rows.append(_axis_row(f"reads:{key}", ABSENT, lb, base, prov, "judgment",
                              {"considered_and_rejected": True,
                               "why_not": str(why or "")}))

    shown = {str(k) for k in (lb.get("reads_shown") or [])}
    for key in sorted(shown - {str(k) for k in reads}):
        rows.append(_axis_row(f"reads:{key}", ABSENT, lb, base, prov, "judgment",
                              {"parser_proposed": True, "removed_by_human": True}))
    for key, value in reads.items():
        key = str(key)
        relation = f"reads:{key}"
        closed = CLOSED_READS.get(key)
        if closed is not None:
            v = "true" if value is True else str(value or "").strip().lower()
            if v not in closed:
                report.skip(f"reading {key} outside its values")
                continue
            rows.append(_axis_row(relation, v, lb, base, prov, "judgment",
                                  {"parser_proposed": key in shown}))
        else:
            # A phrase is not a class, so the PRESENT row teaches only that the
            # atom carries one -- "is there expansion?", which is the weaker
            # half of the reading's own question. The phrase is the answer, and
            # it is a span of the atom, so it goes out as one: `expansion`
            # carries WHAT would repeat -- "lead to many more of the same
            # opportunity" -- and a head can now be asked to produce it.
            phrase = "" if value is True else str(value or "").strip()
            rows.append(_axis_row(relation, PRESENT, lb, base, prov, "judgment",
                                  {"value": phrase, "parser_proposed": key in shown}))
            if len(phrase) >= 8:
                rows.append(_axis_row(f"reads_value:{key}", phrase, lb, base, prov,
                                      "span", {"parser_proposed": key in shown}))
    return rows


#: A labeler's evidence link -> the edge relation it teaches. "answers" is
#: support for a question; "context" is not an edge claim, so it trains nothing.
#: `governs` is the announcement -> detail edge ("Here are the details for the
#: small job" over the ten supply lines under it). It is the one structural
#: relation a person can draw instantly and no rule gets right, so it is the
#: cheapest gold in the labeler.
#: Every relation a labeler can draw is its own edge class.  Two of these were
#: wrong: `answers` was folded into `supports`, and `context` was dropped
#: entirely -- so a deal could teach that one atom backs another up, but never
#: that an atom CLOSES a question or merely sits behind it.  That is the
#: distinction a reply has to carry: "they are intending to use a maglock"
#: answers "do we know the type of lock?", while "I am not sure if it is
#: already installed" is only context for "has the door been installed?".
#: Collapsing the two teaches a head to close a question on any reply at all.
#: The head builds its prototypes from whatever classes the rows contain, so a
#: class with too few examples is simply not learned yet, not an error.
_LINK_TO_EDGE = {
    "governs": "governs",
    "supports": "supports",
    "answers": "answers",
    "contradicts": "contradicts",
    "same_as": "same_as",
    "context": "context",
    "blocked_by": "blocked_by",
    "triggered_by": "triggered_by",
}


def _link_rows(doc: dict[str, Any], deal_id: str, split: str, report: IngestReport) -> list[dict[str, Any]]:
    """Evidence links the labeler drew (item on screen -> another atom or
    highlighted text) as human edge rows: the relation-edge head has no gold,
    and these are exactly the cross-document relations it must learn."""
    rows: list[dict[str, Any]] = []
    for k in doc.get("links") or []:
        if not isinstance(k, dict):
            continue
        if not _is_a_person(k.get("labeler")):
            report.skip("labeler is not a person")
            continue
        label = _LINK_TO_EDGE.get(str(k.get("relation") or ""))
        a = " ".join(str(k.get("from_text") or "").split())
        b = " ".join(str(k.get("to_text") or "").split())
        if label == "answers":
            # Always "answer || question", whichever card drew it, so the
            # reverse row below is the same pair the other way round and the
            # two never contradict each other across cards.
            a, b = _answers_pair(k)
        if not label or len(a) < 3 or len(b) < 3:
            report.skip("link without an edge relation or text")
            continue
        text = f"{a} || {b}"
        prov = {
            "source": "purpulse_atom_labeler",
            "kind": "evidence_link",
            "relation": k.get("relation"),
            "from_head": k.get("from_head"),
            "from_key": k.get("from_key"),
            "to_kind": k.get("to_kind"),
            "to_atom_id": k.get("to_atom_id"),
            "to_filename": k.get("to_filename"),
            "to_page": k.get("to_page"),
            "labeler": k.get("labeler") or "",
            "purpose": k.get("purpose") or "train",
        }
        # 22,650 characters across eighty edges on 010288, and the row said
        # "contradicts" without a word about why THESE two. The edge head has
        # no gold anywhere; it was getting the thinnest version of the richest
        # reasoning on the deal.
        knote = str(k.get("note") or "").strip()
        # A length floor cannot tell "this line cannot be read without it" (35
        # characters, an argument) from "drawn while labelling the whole deal"
        # (36, the card's default). Name the boilerplate instead.
        if len(knote) >= 24 and knote.lower() not in _EMPTY_NOTES:
            rows.append(_rationale_row(
                "edge", f"{text}\nRELATION: {label}", knote, {
                    "teacher": HUMAN_TEACHER, "confidence": 1.0, "scope": "deal",
                    "scope_key": deal_id, "deal_id": deal_id, "project_id": deal_id,
                    "created_at": k.get("created_at") or "", "split": split,
                }, prov))
        rows.append({
            "relation": "edge_relation", "label": label, "raw_text": text, "masked_text": text,
            "label_kind": "judgment", "teacher": HUMAN_TEACHER, "weight": 1.0, "confidence": 1.0,
            "scope": "deal", "scope_key": deal_id, "deal_id": deal_id, "project_id": deal_id,
            "created_at": k.get("created_at") or "", "split": split,
            "provenance": json.dumps(prov, ensure_ascii=False),
        })
        if label == "answers":
            # The same edge read from the question. A head shown only
            # "answer || question" learns to recognise an answer; shown the
            # pair both ways it also learns, from the question, what closed it.
            back = f"{b} || {a}"
            rows.append({
                "relation": "edge_relation", "label": ANSWERED_BY, "raw_text": back, "masked_text": back,
                "label_kind": "judgment", "teacher": HUMAN_TEACHER, "weight": 1.0, "confidence": 1.0,
                "scope": "deal", "scope_key": deal_id, "deal_id": deal_id, "project_id": deal_id,
                "created_at": k.get("created_at") or "", "split": split,
                "provenance": json.dumps({**prov, "reverse_of": "answers"}, ensure_ascii=False),
            })
    return rows


#: The reverse of `answers`: question || answer. Not a relation a labeler
#: draws -- every `answers` link emits it.
ANSWERED_BY = "answered_by"


def _answers_pair(k: dict[str, Any]) -> tuple[str, str]:
    """(answer, question) for an `answers` link, whichever card drew it.

    The Questions card IS the question, so its link runs question (from) ->
    answering atom (to). On an atom card the labeler is on the answer and
    points at the question it answers: answer (from) -> question (to).
    Platform-infra ``atom-labeling-routes.answersPair`` reads them the same way.
    """
    a = " ".join(str(k.get("from_text") or "").split())
    b = " ".join(str(k.get("to_text") or "").split())
    if str(k.get("from_head") or "") == "gap":
        return b, a
    return a, b


#: What the Questions card records beside valid / invalid, as heads of their
#: own. Labelers first put these on the question's ATOM card as readings
#: (reads_set), because the card had nowhere to put them; the Questions card
#: now stores them on the gap judgment (`fields`). The judgment wins, and the
#: atom reading is the fallback for a question the card never answered.
QUESTION_FIELDS: tuple[str, ...] = ("intake_gap", "needed_by", "deal_stage")
_QUESTION_FIELD_VALUES: dict[str, set[str]] = {
    "intake_gap": {"true", "false"},
    "needed_by": {"project_manager", "atlas", "portal"},
    "deal_stage": {"quoting", "planning", "delivery", "closeout"},
}


def _question_values(field_name: str, raw: Any) -> list[str] | None:
    """The class labels one stored answer makes. None when it is outside the set.

    `needed_by` is multi-label: one row per consumer, the way every multi-label
    task in the table is written (several labels on one text, one teacher).
    A reading may have been saved as a list or as "a | b"; both are read.
    """
    allowed = CLOSED_READS.get(field_name) or _QUESTION_FIELD_VALUES[field_name]
    if field_name == "intake_gap":
        v = "true" if raw is True else "false" if raw is False else str(raw or "").strip().lower()
        return [v] if v in allowed else None
    if field_name == "needed_by":
        parts = raw if isinstance(raw, list) else str(raw or "").replace(",", "|").split("|")
        vals = [str(p).strip().lower() for p in parts if str(p).strip()]
        if not vals or any(v not in allowed for v in vals):
            return None
        return sorted(set(vals), key=vals.index)
    v = str(raw or "").strip().lower()
    return [v] if v in allowed else None


def _question_rows(doc: dict[str, Any], labels: list[dict[str, Any]], deal_id: str,
                   split: str, report: IngestReport) -> list[dict[str, Any]]:
    """`question:intake_gap`, `question:needed_by`, `question:deal_stage`.

    From the Questions card's judgment first. An atom whose question the card
    answered for a field gives no fallback row for that field -- matched on the
    question's source atom, or on its words when the atom id moved with a
    re-parse -- so one question never trains on two answers.
    """
    rows: list[dict[str, Any]] = []
    covered: dict[str, set[str]] = {f: set() for f in QUESTION_FIELDS}

    def row(field_name: str, value: str, text: str, created: Any, prov: dict[str, Any]) -> dict[str, Any]:
        return {
            "relation": f"question:{field_name}", "label": value, "raw_text": text, "masked_text": text,
            "label_kind": "judgment", "teacher": HUMAN_TEACHER, "weight": 1.0, "confidence": 1.0,
            "scope": "deal", "scope_key": deal_id, "deal_id": deal_id, "project_id": deal_id,
            "created_at": created or "", "split": split,
            "provenance": json.dumps(prov, ensure_ascii=False),
        }

    for j in doc.get("judgments") or []:
        if not isinstance(j, dict) or str(j.get("head") or "") != "gap":
            continue
        fields = j.get("fields")
        if not isinstance(fields, dict) or not fields or not _is_a_person(j.get("labeler")):
            continue
        text = " ".join(str(j.get("text") or "").split())
        if len(text) < 3:
            continue
        target = j.get("target") if isinstance(j.get("target"), dict) else {}
        source = target.get("source") if isinstance(target.get("source"), dict) else {}
        keys = {f"text:{_norm(text)}"} | ({f"atom:{source['atomId']}"} if source.get("atomId") else set())
        prov = {
            "source": "purpulse_atom_labeler", "kind": "questions_tab",
            "head": "gap", "target_key": j.get("target_key"), "verdict": j.get("verdict"),
            "labeler": j.get("labeler") or "", "purpose": j.get("purpose") or "train",
        }
        for f in QUESTION_FIELDS:
            if f not in fields:
                continue
            vals = _question_values(f, fields[f])
            if vals is None:
                report.skip(f"question {f} outside its values")
                continue
            covered[f] |= keys
            rows.extend(row(f, v, text, j.get("judged_at"), prov) for v in vals)

    for lb in labels:
        reads = lb.get("reads_set")
        if not isinstance(reads, dict) or not _is_a_person(lb.get("labeler")):
            continue
        text = " ".join(str(lb.get("text") or "").split())
        if len(text) < 3:
            continue
        keys = {f"text:{_norm(text)}"} | ({f"atom:{lb['atom_id']}"} if lb.get("atom_id") else set())
        prov = {
            "source": "purpulse_atom_labeler", "kind": "atom_reads_fallback",
            "label_key": lb.get("label_key"), "atom_id": lb.get("atom_id"),
            "labeler": lb.get("labeler") or "", "purpose": lb.get("purpose") or "train",
        }
        for f in QUESTION_FIELDS:
            if f not in reads:
                continue
            if keys & covered[f]:
                report.skip(f"question {f}: the Questions card answered it")
                continue
            vals = _question_values(f, reads[f])
            if vals is None:
                report.skip(f"question {f} outside its values")
                continue
            rows.extend(row(f, v, text, lb.get("labeled_at"), prov) for v in vals)
    return rows


#: Judgment heads that ask an EXISTING head's question in different words.
#:
#: The Dropped stage asks "was this atom right to throw away?", which is the
#: `admission` head's question exactly -- same decision, same two outcomes --
#: asked about an atom the compile suppressed rather than one it kept. It had no
#: registry row of its own, so the answers reached nothing; mapped onto
#: `admission` they join the gold that head already learns from.
#:
#: Note the direction. `should_have_been_kept` means the suppression was WRONG,
#: so the atom should have been admitted: admission `keep`. This is the same
#: collision of names the atom-label mapping above documents, and getting it
#: backwards would teach the head to drop exactly what a person rescued.
_JUDGMENT_HEAD_ALIASES: dict[str, tuple[str, dict[str, str]]] = {
    "suppression": ("admission", {
        "should_have_been_kept": "keep",
        "correctly_dropped": "drop",
    }),
}


def _translate_judgment(head: str, verdict: str) -> tuple[str, str]:
    """Map a judgment surface onto the head whose question it asks.

    Returns the pair unchanged when there is nothing to translate, so an
    unregistered head still reaches the skip that names it.
    """
    alias = _JUDGMENT_HEAD_ALIASES.get(head)
    if alias is None:
        return head, verdict
    target, verdicts = alias
    return target, verdicts.get(verdict, verdict)


def _judgment_rows(doc: dict[str, Any], deal_id: str, split: str, report: IngestReport) -> list[dict[str, Any]]:
    """Conflict / site / site_role / gap / document_job verdicts -> one row each,
    under the relation that head decides (pm_feedback.HEAD_REGISTRY), so the
    edge, site, gap and document heads get human gold -- they had none."""
    from app.core.pm_feedback import HEAD_REGISTRY

    rows: list[dict[str, Any]] = []
    for j in doc.get("judgments") or []:
        if not isinstance(j, dict):
            continue
        if not _is_a_person(j.get("labeler")):
            report.skip("labeler is not a person")
            continue
        head = str(j.get("head") or "")
        verdict = str(j.get("verdict") or "").strip()
        head, verdict = _translate_judgment(head, verdict)
        spec = HEAD_REGISTRY.get(head)
        text = " ".join(str(j.get("text") or "").split())
        # THREE DIFFERENT FAILURES SHARED ONE MESSAGE, AND THE WORST OF THEM
        # WAS INVISIBLE.
        #
        # "judgment without a known head, verdict or text" was reported for an
        # unregistered head, an empty verdict and a two-character text alike.
        # The first is not a malformed row -- it is a whole LABELLING SURFACE
        # that teaches nothing, and the corpus report could not say so.
        #
        # Measured 2026-10-01: the labelling workspace offers a Rules stage and
        # a Dropped stage, and neither `rule` nor `suppression` was in
        # HEAD_REGISTRY. So every judgment made on either was stored in
        # Postgres, refused by /feedback/correction with `422 unknown head`,
        # and skipped here -- teaching nothing by either route, silently.
        if spec is None:
            # `rule` is not a miss. Nothing DECIDES a rule at compile time --
            # SemanticRule thresholds are fitted offline into
            # models/semantic_rule_thresholds.json and loaded at construction --
            # so it has no decide() relation and belongs in no registry. Its
            # verdicts are read by `rule_feedback.rows_from_judgments`, which
            # turns each one into the row `_train_semantic_rules` already reads.
            # Saying "reaches no head" about it would be wrong in the other
            # direction.
            if head == "rule":
                report.skip("rule judgments train offline via rule_feedback, not through a head")
            else:
                report.skip(
                    f"judgment head {head!r} is not in HEAD_REGISTRY: this is a "
                    f"labelling surface whose verdicts reach no head"
                )
            continue
        if not verdict or len(text) < 3:
            report.skip(f"judgment on {head!r} has no verdict or too little text")
            continue
        if spec.candidates and verdict not in spec.candidates:
            report.skip(f"judgment verdict outside {j.get('head')}'s classes")
            continue
        prov = {
            "source": "purpulse_atom_labeler",
            "head": j.get("head"),
            "target_key": j.get("target_key"),
            "parser_value": j.get("parser_value"),
            "compile_id": j.get("compile_id"),
            "note": j.get("note") or "",
            "labeler": j.get("labeler") or "",
            "purpose": j.get("purpose") or "train",
        }
        # The verdict, and separately WHY. A head told "invalid" sixty times
        # with no reason learns to distrust a tone of voice; told that this one
        # is answered, that one a duplicate and forty-five the right question
        # about the wrong size of job, it can learn the rule -- and that the
        # same question is VALID on a rollout.
        reason = str(j.get("reason") or "").strip()
        if reason:
            rows.append({
                "relation": f"{spec.relation}_reason", "label": reason,
                "raw_text": text, "masked_text": text,
                "label_kind": "judgment", "teacher": HUMAN_TEACHER,
                "weight": 1.0, "confidence": 1.0, "scope": "deal",
                "scope_key": deal_id, "deal_id": deal_id, "project_id": deal_id,
                "created_at": j.get("judged_at") or "", "split": split,
                "provenance": json.dumps({**prov, "verdict": verdict}, ensure_ascii=False),
            })
        jnote = str(j.get("note") or "").strip()
        if len(jnote) >= 40:
            rows.append(_rationale_row(
                str(j.get("head") or "judgment"),
                f"{text}\nVERDICT: {verdict}" + (f" ({reason})" if reason else ""),
                jnote, {
                    "teacher": HUMAN_TEACHER, "confidence": 1.0, "scope": "deal",
                    "scope_key": deal_id, "deal_id": deal_id, "project_id": deal_id,
                    "created_at": j.get("judged_at") or "", "split": split,
                }, prov))
        rows.append({
            "relation": spec.relation, "label": verdict, "raw_text": text, "masked_text": text,
            "label_kind": "judgment", "teacher": HUMAN_TEACHER, "weight": 1.0, "confidence": 1.0,
            "scope": "deal", "scope_key": deal_id, "deal_id": deal_id, "project_id": deal_id,
            "created_at": j.get("judged_at") or "", "split": split,
            "provenance": json.dumps(prov, ensure_ascii=False),
        })
    return rows


def _site_count(v: Any) -> int | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return int(v)
    if isinstance(v, str) and v.strip().isdigit():
        return int(v.strip())
    return None


def write_db(docs: Iterable[dict[str, Any]], target: Path) -> IngestReport:
    """Replace ``target``'s tables with the rows for ``docs``. Re-ingest is a rebuild."""
    report = IngestReport()
    conn = sqlite3.connect(target)
    try:
        conn.execute("DROP TABLE IF EXISTS training_rows")
        conn.execute("DROP TABLE IF EXISTS deal_gold")
        cols = ", ".join(
            f"{c} {'REAL' if c in ('weight', 'confidence') else 'TEXT'}" for c in _COLUMNS
        )
        conn.execute(f"CREATE TABLE training_rows (id INTEGER PRIMARY KEY AUTOINCREMENT, {cols})")
        conn.execute(
            "CREATE TABLE deal_gold (deal_id TEXT, labeler TEXT, primary_service TEXT, "
            "declared_site_count INTEGER, note TEXT, purpose TEXT, answered_at TEXT)"
        )
        insert = (
            f"INSERT INTO training_rows ({', '.join(_COLUMNS)}) "
            f"VALUES ({', '.join('?' for _ in _COLUMNS)})"
        )
        for doc in docs:
            report.deals += 1
            rows = rows_for_deal(doc, report)
            conn.executemany(insert, [tuple(r.get(c) for c in _COLUMNS) for r in rows])
            for ans in doc.get("deal_answers") or []:
                if not isinstance(ans, dict):
                    continue
                conn.execute(
                    "INSERT INTO deal_gold VALUES (?,?,?,?,?,?,?)",
                    (doc.get("deal_id"), ans.get("labeler") or "", ans.get("primary_service") or "",
                     _site_count(ans.get("declared_site_count")), ans.get("note") or "",
                     ans.get("purpose") or "train", ans.get("answered_at") or ""),
                )
                report.deal_answers += 1
        conn.commit()
    finally:
        conn.close()
    return report


def deal_rationale_rows(doc: dict[str, Any], deal_id: str, split: str) -> list[dict[str, Any]]:
    """The deal-level answer, argued. "What the parser got wrong across the
    whole deal" is the only place a labeler writes about the deal rather than
    about a line, and it reached `deal_gold`, which the backbone builder does
    not read."""
    rows: list[dict[str, Any]] = []
    for ans in doc.get("deal_answers") or []:
        if not isinstance(ans, dict) or not _is_a_person(ans.get("labeler")):
            continue
        note = str(ans.get("note") or "").strip()
        if len(note) < 40:
            continue
        prompt = (f"DEAL {deal_id}\n"
                  f"PRIMARY SERVICE: {ans.get('primary_service')}\n"
                  f"SITES: {ans.get('declared_site_count')}")
        rows.append(_rationale_row("deal", prompt, note, {
            "teacher": HUMAN_TEACHER, "confidence": 1.0, "scope": "deal",
            "scope_key": deal_id, "deal_id": deal_id, "project_id": deal_id,
            "created_at": ans.get("answered_at") or "", "split": split,
        }, {"source": "purpulse_atom_labeler", "head": "router",
            "labeler": ans.get("labeler") or ""}))
    return rows


def docs_from_gold_export(payload: dict[str, Any], *, labeler: str = "") -> list[dict[str, Any]]:
    """Offline zip-labeler exports (``gold_labels*.json``: an ``atoms`` array of
    ``{deal, doc, atom, label, parser_guess, note, ...}``) -> one deal doc each,
    in the blob shape ``rows_for_deal`` reads. These carry no context, so their
    rows are version 0 (bare text) and marked ``source=offline_gold_labeler``."""
    by_deal: dict[str, list[dict[str, Any]]] = {}
    for row in payload.get("atoms") or []:
        if not isinstance(row, dict):
            continue
        lab = str(row.get("label") or "").strip()
        body = str(row.get("atom") or row.get("body") or "").strip()
        deal = str(row.get("deal") or "").strip()
        if not (lab and body and deal):
            continue
        by_deal.setdefault(deal, []).append({
            "label_key": None,
            "text": body,
            "filename": row.get("doc"),
            "page": row.get("page"),
            "label_type": lab,
            "coarse": row.get("coarse") or None,
            "parser_type": row.get("parser_guess") or row.get("g") or "",
            "note": row.get("note") or "",
            "labeler": labeler,
            "source": "offline_gold_labeler",
            "purpose": "train",
        })
    return [{"deal_id": d, "labels": labels} for d, labels in by_deal.items()]
