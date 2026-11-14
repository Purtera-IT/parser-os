"""Before Deal Kit answers → PM-confirmed evidence atoms the C3 heads read.

On the brief's Review step the PM answers eight yes/no questions the documents
rarely settle (partner or direct, crews at several sites at once, union or
prevailing wage, customer paper, the customer's portal, prerequisites, nights
or weekends, and whether we have done the scope before). Platform-infra stores
them and, on every save, rewrites

    ``<container>/deals/<deal_id>/orbitbrief/latest/pre_dealkit_answers.json``

with the deal's latest answer per question, each as ONE fixed PM statement
(``statement``: "PM confirmed: some of the work will be done nights or
weekends. Why: mall is open days.") or ``null`` for a question about us rather
than the job (``known_scope``).

This module only extracts: one statement, one atom, verbatim. It decides
nothing about type, section or price. The compiler adds these atoms BEFORE the
deal's one C3 base read (``document_job_scope`` / ``c3_base_read``), so the
super heads read them like every other line -- type, admission, keep, SOW
section, supplier and the quote line heads -- and the brief, SOW and quote
follow from what the heads say. The quoting view keeps them in (they are the
PM's pre-pricing inputs, not our paper): ``c3_type_base._view_rows`` admits
them and marks it in the rule-fire ledger as ``view.pm_pre_dealkit_in``.

Same contract as :mod:`app.core.pm_answer_blob`:

* **Gated** on ``SOWSMITH_FEEDBACK_BLOB`` (the PM-answer switch), plus its own
  kill switch ``SOWSMITH_PRE_DEALKIT_ATOMS=0``.
* **Offline-safe**: any failure yields ``[]``; a compile never depends on it.
* **Cut-aware**: a run cut at ``as_of`` admits only answers given by then.
* **Deterministic**: ids hash from (project, question, statement), so the same
  answer recompiles to the same atom and a changed answer to a new one.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Iterable

from app.core import rule_ledger
from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    EvidenceAtom,
    ReviewStatus,
    SourceRef,
)

BLOB_NAME = "pre_dealkit_answers.json"
SCHEMA = "purpulse.pre-dealkit-answers.v1"
#: The source name the atoms carry. Never the blob's name: a file named with "dealkit" is read as our
#: own Deal Kit (c3_record.file_kind), which the base never reads and the brief drops as
#: old_manual_deal_kit -- the answers would never reach a head.
SOURCE_NAME = "pm_answers_before_quote.json"
PARSER_NAME = "pre_dealkit"
PARSER_VERSION = "pre_dealkit_v1"
#: The review flag every one of these atoms carries. The labeling page hides
#: flagged atoms: the statement is our wording, never a training line.
FLAG = "pm_pre_dealkit"
ATOM_PREFIX = "atm_pdk_"
#: A PM stated it on a fixed question; same headroom as a PM answer.
CONFIDENCE = 0.95

#: The starting type of each answer, by question and answer (owner 2026-11-14: route by the kind of
#: question). The question fixes what kind of fact the PM is stating, so the line starts where the SOW
#: reads that kind of fact (c3_brief.SECTION_BY_TYPE): nights or weekends is a work window (timeline),
#: union / prevailing wage a compliance term, the contract's paper a contract term, the customer's portal
#: a requirement of the work, a customer prerequisite a customer responsibility, crew concurrency and the
#: "no" answers an assumption the price rests on, the channel the deal's own metadata. It is a STARTING
#: type: the C3 base reads these lines and retypes every one it is sure of (apply_authority), and each
#: assignment is a fire of ``pre_dealkit.type_by_question`` in the rule-fire ledger.
TYPE_BY_QUESTION: dict[str, tuple[AtomType, AtomType]] = {
    # key: (type when yes, type when no)
    "through_partner": (AtomType.project_metadata, AtomType.project_metadata),
    "concurrent_crews": (AtomType.assumption, AtomType.assumption),
    "regulated_labor": (AtomType.compliance_rule, AtomType.pricing_assumption),
    "customer_paper": (AtomType.contract_term, AtomType.contract_term),
    "customer_systems": (AtomType.requirement, AtomType.assumption),
    "prerequisites": (AtomType.dependency, AtomType.assumption),
    "after_hours": (AtomType.site_access_window, AtomType.site_access_window),
}


#: The work window a nights/weekends answer states, in the work-window head's own answers.
WORK_WINDOW_BY_ANSWER = {"yes": "after_hours", "no": "business_hours"}


def work_window(atom: Any) -> str | None:
    """The work window the PM's nights/weekends answer states for its own line, else None."""
    if not is_pre_dealkit_atom(atom):
        return None
    v = getattr(atom, "value", None) if not isinstance(atom, dict) else atom.get("value")
    if not isinstance(v, dict) or v.get("key") != "after_hours":
        return None
    return WORK_WINDOW_BY_ANSWER.get(str(v.get("answer") or ""))


def enabled() -> bool:
    if os.getenv("SOWSMITH_PRE_DEALKIT_ATOMS", "1").strip().lower() in {"0", "false", "no", "off"}:
        return False
    from app.core.pm_answer_blob import _enabled

    return _enabled()


def is_pre_dealkit_atom(atom: Any) -> bool:
    """True for an atom this module made (by flag, else by id)."""
    flags = getattr(atom, "review_flags", None)
    if flags is None and isinstance(atom, dict):
        flags = atom.get("review_flags")
    if FLAG in [str(f) for f in (flags or [])]:
        return True
    aid = getattr(atom, "id", None) if not isinstance(atom, dict) else atom.get("id")
    return str(aid or "").startswith(ATOM_PREFIX)


def _norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def parse_document(raw: Any) -> list[dict[str, Any]]:
    """The answer rows of a ``pre_dealkit_answers.json`` document; ``[]`` when it is not one."""
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="replace")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if not isinstance(raw, dict) or raw.get("schema") != SCHEMA:
        return []
    return [r for r in raw.get("answers") or [] if isinstance(r, dict)]


def answer_to_atom(row: dict[str, Any], *, project_id: str, deal_id: str = "") -> EvidenceAtom | None:
    """One answered question → one atom, or ``None`` (no statement: a question about us, or unanswered)."""
    statement = _norm(row.get("statement"))
    key = _norm(row.get("key"))
    if not statement or not key or not isinstance(row.get("answer"), bool):
        return None
    h = hashlib.sha1(f"{project_id}|{key}|{statement}".encode("utf-8")).hexdigest()[:16]
    yes_no = TYPE_BY_QUESTION.get(key)
    atom_type = (yes_no[0] if row["answer"] else yes_no[1]) if yes_no else AtomType.decision
    atom_id = f"{ATOM_PREFIX}{h}"
    art_id = f"art_pm_answers_{(deal_id or project_id)[:24]}"
    answered_at = _norm(row.get("answered_at"))
    question = _norm(row.get("question"))
    source = SourceRef(
        id=f"src_pdk_{h}",
        artifact_id=art_id,
        artifact_type=ArtifactType.pm_answer,
        filename=SOURCE_NAME,
        # The time is the answer's, so the view and an as_of cut place it where it was said.
        locator={
            "key": key,
            "question": question,
            "answered_at": answered_at,
            "sent_at": answered_at,
            "actor": _norm(row.get("answered_by")),
            "kind": "pre_dealkit",
        },
        extraction_method="pre_dealkit",
        parser_version=PARSER_VERSION,
        parser=PARSER_NAME,
        path=SOURCE_NAME,
    )
    return EvidenceAtom(
        id=atom_id,
        atom_id=atom_id,
        project_id=project_id,
        artifact_id=art_id,
        # A starting type only (TYPE_BY_QUESTION): the C3 base types every
        # line it is sure of (apply_authority), and these atoms are in its read.
        atom_type=atom_type,
        raw_text=statement,
        normalized_text=statement.lower(),
        value={
            "key": key,
            "question": question,
            "answer": "yes" if row["answer"] else "no",
            "why": _norm(row.get("why")),
            "kind": "pre_dealkit",
            "said_by": {"side": "ours", "role": "pm"},
        },
        source_refs=[source],
        entity_keys=[],
        authority_class=AuthorityClass.pm_confirmed,
        confidence=CONFIDENCE,
        review_status=ReviewStatus.approved,
        review_flags=["pm_authored", FLAG],
        parser_version=PARSER_VERSION,
        claim=statement,
        normalized_claim=statement.lower(),
    )


def answers_to_atoms(rows: Iterable[dict[str, Any]], *, project_id: str, deal_id: str = "",
                     as_of: str | None = None) -> list[EvidenceAtom]:
    """Atoms for every answer with a statement, one per question, cut at ``as_of`` when given."""
    cut = str(as_of)[:19] if as_of else None
    by_key: dict[str, EvidenceAtom] = {}
    for row in rows:
        if cut and str(row.get("answered_at") or "")[:19] > cut:
            continue
        atom = answer_to_atom(row, project_id=project_id, deal_id=deal_id)
        if atom is not None:
            by_key[str(row.get("key"))] = atom
    for atom in by_key.values():
        t = getattr(atom.atom_type, "value", atom.atom_type)
        # No head has read the line yet: a no_head fire. The base's own answer for it shows in the
        # brief's head reads; where the base was sure, it replaced this type.
        rule_ledger.fire("pre_dealkit.type_by_question", atom.id, None, str(t),
                         f"{atom.value.get('key')}={atom.value.get('answer')} starts as {t}")
    return list(by_key.values())


def load_document(deal_id: str) -> Any:
    """The deal's answers document from blob, or ``None``."""
    if not deal_id:
        return None
    from app.core.pm_answer_blob import _container_client

    cc = _container_client()
    if cc is None:
        return None
    try:
        return cc.download_blob(f"deals/{deal_id}/orbitbrief/latest/{BLOB_NAME}").readall()
    except Exception:
        return None


def load_pre_dealkit_atoms(*, project_id: str, deal_id: str = "", as_of: str | None = None) -> list[EvidenceAtom]:
    """The deal's Before Deal Kit answers as atoms. ``[]`` when off, absent or unreadable."""
    try:
        if not enabled():
            return []
        rows = parse_document(load_document(deal_id or project_id))
        return answers_to_atoms(rows, project_id=project_id, deal_id=deal_id or project_id, as_of=as_of)
    except Exception:
        return []
