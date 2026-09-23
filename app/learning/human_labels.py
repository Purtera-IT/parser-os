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


def rows_for_deal(doc: dict[str, Any], report: IngestReport | None = None) -> list[dict[str, Any]]:
    from app.core.training_log import assign_split

    report = report if report is not None else IngestReport()
    deal_id = str(doc.get("deal_id") or "").strip()
    labels = [lb for lb in doc.get("labels") or [] if isinstance(lb, dict)]
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
            "weight": 1.0, "confidence": 1.0, "scope": "deal", "scope_key": deal_id,
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
    out.extend(_link_rows(doc, deal_id, split, report))
    report.rows += len(out)
    return out


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
    for axis in ("about", "wants"):
        v = str(lb.get(axis) or "").strip()
        if v:
            rows.append(_axis_row(axis, v, lb, base, prov, "judgment"))

    reads = lb.get("reads_set")
    if not isinstance(reads, dict):
        return rows
    shown = {str(k) for k in (lb.get("reads_shown") or [])}
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
            # A phrase is not a class. What is learnable today is that the
            # atom carries one; the phrase itself waits for a span head.
            phrase = "" if value is True else str(value or "").strip()
            rows.append(_axis_row(relation, PRESENT, lb, base, prov, "judgment",
                                  {"value": phrase, "parser_proposed": key in shown}))
    return rows


#: A labeler's evidence link -> the edge relation it teaches. "answers" is
#: support for a question; "context" is not an edge claim, so it trains nothing.
#: `governs` is the announcement -> detail edge ("Here are the details for the
#: small job" over the ten supply lines under it). It is the one structural
#: relation a person can draw instantly and no rule gets right, so it is the
#: cheapest gold in the labeler.
_LINK_TO_EDGE = {
    "governs": "governs",
    "supports": "supports",
    "answers": "supports",
    "contradicts": "contradicts",
    "same_as": "same_as",
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
        rows.append({
            "relation": "edge_relation", "label": label, "raw_text": text, "masked_text": text,
            "label_kind": "judgment", "teacher": HUMAN_TEACHER, "weight": 1.0, "confidence": 1.0,
            "scope": "deal", "scope_key": deal_id, "deal_id": deal_id, "project_id": deal_id,
            "created_at": k.get("created_at") or "", "split": split,
            "provenance": json.dumps(prov, ensure_ascii=False),
        })
    return rows


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
        spec = HEAD_REGISTRY.get(str(j.get("head") or ""))
        verdict = str(j.get("verdict") or "").strip()
        text = " ".join(str(j.get("text") or "").split())
        if spec is None or not verdict or len(text) < 3:
            report.skip("judgment without a known head, verdict or text")
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
