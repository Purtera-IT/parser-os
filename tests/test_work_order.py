"""A job is not stated in one sentence, so it cannot be recovered one span at a time.

These cover the reassembly stage: which documents it reads, what it mints from them,
and — the part that actually decides whether the stage is worth anything — whether
what it mints survives the gates that run immediately afterwards.

Shapes are taken from deal 000020 (Binghamton), where the work order lives in a
single note ("install 11 access points") beside a partner-relationship email that
says nothing about the job.
"""
from __future__ import annotations

import pytest

from app.core import work_order
from app.core.schemas import (
    AtomType,
    AuthorityClass,
    EvidenceAtom,
    ReviewStatus,
    SourceRef,
)


def _ref(artifact_id: str, filename: str, **locator):
    return SourceRef(
        id=f"sr-{artifact_id}",
        artifact_id=artifact_id,
        artifact_type="email",
        filename=filename,
        locator=locator or {"line": 1},
        extraction_method="email_parser",
        parser_version="1",
    )


def _atom(
    text,
    *,
    artifact_id="art-scope",
    filename="scope.eml",
    atom_type=AtomType.scope_item,
    authority=AuthorityClass.customer_current_authored,
    value=None,
    atom_id=None,
):
    return EvidenceAtom(
        id=atom_id or f"a-{abs(hash(text)) % 10**9}",
        project_id="p1",
        artifact_id=artifact_id,
        atom_type=atom_type,
        raw_text=text,
        normalized_text=text.lower(),
        authority_class=authority,
        confidence=0.9,
        review_status=ReviewStatus.auto_accepted,
        parser_version="1",
        source_refs=[_ref(artifact_id, filename)],
        value=value or {},
    )


SCOPE = [
    _atom("Install 11 access points across the Binghamton campus"),
    _atom("Ceiling mount, existing cabling is already in place"),
    _atom("Work has to happen after 6pm once the building is clear"),
]
CHATTER = [
    _atom(
        "Great catching up on the partnership roadmap and our Q3 service launches",
        artifact_id="art-chat",
        filename="catchup.eml",
    ),
    _atom(
        "We should talk about co-marketing and headcount next quarter",
        artifact_id="art-chat",
        filename="catchup.eml",
    ),
]

WORK_ORDER = {
    "work_lines": [
        {
            "work": "install and ceiling-mount access points",
            "object": "access point",
            "count": 11,
            "unit": "ap",
        }
    ],
    "site_count": 1,
    "after_hours": True,
    "no_onsite_hands": False,
    "customer_supplies_equipment": False,
    "one_line_summary": "Install 11 access points after hours at one Binghamton site",
}


class _Decision:
    def __init__(self, verdict, source="store"):
        self.verdict = verdict
        self.source = source


@pytest.fixture
def judge(monkeypatch):
    """Route relevance through a recorded fake decide(), and hand back the calls."""
    calls = []

    def fake_decide(relation, text, candidates, **kw):
        calls.append({"relation": relation, "text": text, **kw})
        verdict = (
            "relationship_or_other"
            if "catchup" in text or "partnership" in text
            else "about_this_job"
        )
        return _Decision(verdict)

    monkeypatch.setattr("app.core.decide.decide", fake_decide)
    return calls


@pytest.fixture
def extractor(monkeypatch):
    import json

    monkeypatch.setattr(
        "app.core.llm_client.complete",
        lambda *a, **k: json.dumps(WORK_ORDER),
    )


# ── what it reads ────────────────────────────────────────────────────


def test_the_relationship_email_is_set_aside(judge, extractor):
    atoms, minted, report = work_order.apply_work_order(
        list(SCOPE + CHATTER), project_id="p1", deal_name="Binghamton wireless refresh"
    )
    assert minted == 1
    assert report["kept_docs"] == ["scope.eml"]
    assert report["dropped_docs"] == ["catchup.eml (store)"]


def test_the_judged_text_is_the_document_not_the_deal(judge, extractor):
    """A lesson keyed on the deal name matches every document in the deal and
    mutes the seam. The deal belongs in context; the document is what is judged."""
    work_order.apply_work_order(
        list(SCOPE), project_id="p1", deal_name="Binghamton wireless refresh"
    )
    call = judge[0]
    assert call["text"].startswith("DOCUMENT: scope.eml")
    assert "Binghamton wireless refresh" not in call["text"]
    assert call["context"] == "DEAL: Binghamton wireless refresh"
    assert call["exclude_created_by"] == ("teacher",)


def test_abstention_keeps_the_document(monkeypatch, extractor):
    """Dropping a relevant document costs a work line; keeping an irrelevant one
    costs a few tokens. Abstain toward keeping."""
    monkeypatch.setattr("app.core.decide.decide", lambda *a, **k: _Decision(None, ""))
    _, minted, report = work_order.apply_work_order(
        list(SCOPE + CHATTER), project_id="p1", deal_name="d"
    )
    assert report["dropped_docs"] == []
    assert minted == 1


def test_a_judge_that_raises_does_not_break_the_compile(monkeypatch, extractor):
    def boom(*a, **k):
        raise RuntimeError("store is down")

    monkeypatch.setattr("app.core.decide.decide", boom)
    _, minted, _ = work_order.apply_work_order(
        list(SCOPE), project_id="p1", deal_name="d"
    )
    assert minted == 1


# ── what it mints ────────────────────────────────────────────────────


def _minted(atoms):
    return [
        a
        for a in atoms
        if (getattr(a, "value", None) or {}).get("backfill_reason") == "work_order"
    ]


def test_provenance_is_inherited_never_invented(judge, extractor):
    """cap_authority_to_source caps any atom whose refs name no real artifact.
    The minted task must point at the document it was summarised from."""
    atoms, _, _ = work_order.apply_work_order(
        list(SCOPE + CHATTER), project_id="p1", deal_name="d"
    )
    task = _minted(atoms)[0]
    assert task.artifact_id == "art-scope"
    assert [r.filename for r in task.source_refs] == ["scope.eml"]
    assert task.value["backfilled_from_atom_id"] in {a.id for a in SCOPE}


def test_a_summary_is_a_machine_reading_until_a_pm_says_otherwise(judge, extractor):
    atoms, _, _ = work_order.apply_work_order(
        list(SCOPE), project_id="p1", deal_name="d"
    )
    task = _minted(atoms)[0]
    assert task.atom_type == AtomType.task
    assert task.authority_class == AuthorityClass.machine_extractor
    assert task.review_status == ReviewStatus.needs_review
    assert "work_order" in task.review_flags


def test_the_count_lives_in_value_not_in_a_quantity_entity_key(judge, extractor):
    """scrub_nondeliverable_quantity_keys strips ``quantity:`` keys off any atom,
    so a count parked in an entity key would not survive the next stage."""
    atoms, _, _ = work_order.apply_work_order(
        list(SCOPE), project_id="p1", deal_name="d"
    )
    task = _minted(atoms)[0]
    assert task.value["count"] == 11
    assert task.value["unit"] == "ap"
    assert not [k for k in task.value if str(k).startswith("quantity:")]


def test_the_deal_level_facts_ride_with_the_line(judge, extractor):
    """Sodexo priced 8h at each of 10 sites. A work line without its site count
    cannot be turned into hours."""
    atoms, _, report = work_order.apply_work_order(
        list(SCOPE), project_id="p1", deal_name="d"
    )
    task = _minted(atoms)[0]
    assert task.value["site_count"] == 1
    assert task.value["after_hours"] is True
    assert report["summary"]["one_line_summary"].startswith("Install 11 access points")


def test_an_existing_task_is_not_minted_twice(judge, extractor):
    prior = _atom(
        "install and ceiling-mount access points", atom_type=AtomType.task
    )
    atoms, minted, _ = work_order.apply_work_order(
        list(SCOPE) + [prior], project_id="p1", deal_name="d"
    )
    assert minted == 0


def test_a_job_the_documents_do_not_state_mints_nothing(judge, monkeypatch):
    monkeypatch.setattr(
        "app.core.llm_client.complete", lambda *a, **k: '{"work_lines": []}'
    )
    atoms, minted, _ = work_order.apply_work_order(
        list(SCOPE), project_id="p1", deal_name="d"
    )
    assert minted == 0
    assert len(atoms) == len(SCOPE)


def test_unparseable_model_output_mints_nothing(judge, monkeypatch):
    monkeypatch.setattr("app.core.llm_client.complete", lambda *a, **k: "sorry, no")
    _, minted, _ = work_order.apply_work_order(
        list(SCOPE), project_id="p1", deal_name="d"
    )
    assert minted == 0


@pytest.mark.parametrize(
    "work", ["", "  ", "do", "42", "x" * 400, None]
)
def test_a_line_that_is_not_a_statement_of_work_is_refused(judge, monkeypatch, work):
    import json

    monkeypatch.setattr(
        "app.core.llm_client.complete",
        lambda *a, **k: json.dumps({"work_lines": [{"work": work, "count": 3}]}),
    )
    _, minted, _ = work_order.apply_work_order(
        list(SCOPE), project_id="p1", deal_name="d"
    )
    assert minted == 0


def test_support_is_the_sentence_the_line_was_summarised_from(judge, extractor):
    """Overlap picks provenance; authority breaks ties."""
    pick = work_order.pick_support(
        "install and ceiling-mount access points", list(SCOPE + CHATTER)
    )
    assert "access points" in pick.raw_text


# ── whether it survives what runs next ───────────────────────────────


def test_the_minted_task_survives_atom_type_sanity(judge, extractor):
    """The stage that runs immediately after this one. It re-types tasks back to
    scope_item when they carry Include-list polarity, floors claim types on
    serialized text, and caps authority when no real artifact backs the atom.
    A minted work line must come through all three still a task."""
    from app.core.atom_type_sanity import apply_type_sanity

    # The supporting atom carries Include polarity; the summary must not inherit it.
    scope = list(SCOPE)
    scope[0] = _atom(
        "Install 11 access points across the Binghamton campus",
        value={"list_section": "include", "kind": "email_body_line"},
    )
    atoms, _, _ = work_order.apply_work_order(
        scope, project_id="p1", deal_name="d"
    )
    task = _minted(atoms)[0]
    assert "list_section" not in task.value

    out, _, _ = apply_type_sanity(atoms, project_id="p1", artifact_ids={"art-scope"})
    survivor = [a for a in out if a.id == task.id]
    assert len(survivor) == 1
    assert survivor[0].atom_type == AtomType.task
    assert survivor[0].value["count"] == 11


def test_span_admission_leaves_the_minted_task_alone(judge, extractor):
    """Admission re-types only the weak retained types; a task is not one of them,
    so a minted work line is not a candidate for re-typing."""
    from app.core.span_admission import WEAK_ATOM_TYPES

    assert "task" not in WEAK_ATOM_TYPES


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("SOWSMITH_WORK_ORDER", raising=False)
    assert work_order.enabled() is False
    monkeypatch.setenv("SOWSMITH_WORK_ORDER", "1")
    assert work_order.enabled() is True


def test_the_include_list_demotion_is_real_and_the_exemption_is_what_saves_us():
    """Both defenses are load-bearing and neither is assumed.

    A task carrying Include polarity IS demoted to scope_item — that pass is not
    hypothetical. A minted work line survives it twice over: it drops the polarity
    it inherited, and it carries the quote-line exemption if it ever failed to.
    """
    from app.core.atom_type_sanity import demote_email_include_list_microtasks

    bare = _atom(
        "Okta integration",
        atom_type=AtomType.task,
        value={"list_section": "include", "kind": "email_body_line"},
    )
    exempt = _atom(
        "Okta integration",
        atom_type=AtomType.task,
        value={
            "list_section": "include",
            "kind": "email_body_line",
            "is_quote_line": True,
            "backfill_reason": "work_order",
        },
    )
    assert demote_email_include_list_microtasks([bare, exempt]) == 1
    assert bare.atom_type == AtomType.scope_item
    assert exempt.atom_type == AtomType.task


def test_an_invented_source_ref_would_have_been_capped(judge, extractor):
    """Why the stage deepcopies instead of constructing: an atom whose refs name
    no artifact in the compile is capped on the very next pass."""
    from app.core.atom_type_sanity import cap_authority_to_source

    invented = _atom(
        "install and ceiling-mount access points",
        artifact_id="synthetic-work-order",
        atom_type=AtomType.task,
        authority=AuthorityClass.customer_current_authored,
    )
    invented.source_refs = []
    assert cap_authority_to_source([invented], artifact_ids={"art-scope"}) == 1
    assert invented.authority_class == AuthorityClass.machine_extractor

    atoms, _, _ = work_order.apply_work_order(
        list(SCOPE), project_id="p1", deal_name="d"
    )
    real = _minted(atoms)[0]
    before = real.authority_class
    assert cap_authority_to_source([real], artifact_ids={"art-scope"}) == 0
    assert real.authority_class == before
