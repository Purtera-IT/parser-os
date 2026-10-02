"""A survey is outstanding only when somebody asks for one.

Live 000132 was staged "awaiting site survey" from job steps -- "On-site visit
once per week", "one coordinated site visit", "complete an onsite walk-through
before beginning the removal" -- and the derived atom was pinned to whichever
atom came first in the deal, an unrelated artifact.
"""
from __future__ import annotations

from pathlib import Path

from app.core.deal_state import read_deal_state


class _A:
    def __init__(self, text, artifact_id="art_x"):
        self.raw_text = text
        self.artifact_id = artifact_id


JOB_STEPS = [
    "On-site visit once per week (4-8 hours as needed)",
    "All work will be completed in one coordinated site visit.",
    "Technician will complete an onsite walk-through before beginning the removal.",
    "Walk the site with the customer at the end of the day.",
]


def test_job_steps_are_not_a_pending_survey():
    st = read_deal_state([_A(t) for t in JOB_STEPS])
    assert st.get("next_step") is None
    assert st.get("stage") is None


def test_a_requested_survey_is_outstanding_and_names_its_evidence():
    ask = _A("Customer wants us to schedule a site survey before we can quote.", "art_mail")
    st = read_deal_state([_A(t) for t in JOB_STEPS] + [ask])
    assert st.get("stage").value == "awaiting site survey"
    assert st.get("next_step").evidence_atoms == [ask]
    assert st.get("stage").evidence_atoms == [ask]


def test_a_completed_survey_is_not_outstanding():
    st = read_deal_state([_A("We completed the site survey on Tuesday.")])
    assert st.get("stage") is None


def test_compile_pins_the_deal_state_to_the_artifact_that_asked(tmp_path: Path):
    from app.core.compiler import compile_project

    (tmp_path / "a-job-steps.txt").write_text(
        "HubSpot Note: Removal plan\nHubSpot Note ID: 1\nDate: 2026-05-01T10:00:00Z\nAuthor: Trent\n\n"
        "Technician will complete an onsite walk-through before beginning the removal.\n"
        "All work will be completed in one coordinated site visit.\n",
        encoding="utf-8",
    )
    (tmp_path / "b-ask.eml").write_text(
        "From: Customer <c@customer.com>\nTo: t@purtera-it.com\nSubject: Survey\n"
        "Date: Mon, 1 Jun 2026 10:00:00 -0400\n\n"
        "Hi Trent,\n\nWe would like you to perform a site survey so you can give us a firm quote.\n\nThanks\n",
        encoding="utf-8",
    )
    r = compile_project(tmp_path, project_id="p", allow_errors=True, use_cache=False)
    states = [a for a in r.atoms if a.atom_type.value == "deal_state"]
    assert states, "a requested survey is outstanding"
    by_file = {a.artifact_id: (a.source_refs[0].filename if a.source_refs else "") for a in r.atoms}
    for st in states:
        assert by_file.get(st.artifact_id) == "b-ask.eml", st.raw_text


def test_compile_job_steps_alone_derive_no_survey_state(tmp_path: Path):
    from app.core.compiler import compile_project

    (tmp_path / "steps.txt").write_text(
        "HubSpot Note: Removal plan\nHubSpot Note ID: 1\nDate: 2026-05-01T10:00:00Z\nAuthor: Trent\n\n"
        + "\n".join(JOB_STEPS) + "\n",
        encoding="utf-8",
    )
    r = compile_project(tmp_path, project_id="p", allow_errors=True, use_cache=False)
    assert not [a for a in r.atoms if a.atom_type.value == "deal_state"]
