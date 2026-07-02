from __future__ import annotations

from app.core.schemas import AtomType
from app.core.task_atom_backfill import backfill_quote_task_atoms, should_backfill_task


class _Atom:
    def __init__(self, atom_type, text, artifact_id="art", kind=None):
        self.id = f"atm_{abs(hash(text))}"
        self.project_id = "p"
        self.artifact_id = artifact_id
        self.atom_type = atom_type
        self.raw_text = text
        self.normalized_text = text.lower()
        self.value = {"kind": kind or atom_type, "text": text}
        self.entity_keys = []
        self.source_refs = []
        self.receipts = []
        self.review_flags = []


def test_backfills_hubspot_note_quote_tasks() -> None:
    atoms = [
        _Atom("scope_item", "*   Badge/access control setup", kind="email_body_line"),
        _Atom("scope_item", "*   UID Enterprise setup", kind="email_body_line"),
        _Atom("scope_item", "*   Okta integration", kind="email_body_line"),
        _Atom("scope_item", "*   Camera configuration", kind="email_body_line"),
    ]
    out, count = backfill_quote_task_atoms(atoms, project_id="gecko")
    assert count == 4
    tasks = [a for a in out if a.atom_type == AtomType.task]
    assert [t.value["is_quote_line"] for t in tasks] == [True, True, True, True]
    assert any("Okta integration" in t.raw_text for t in tasks)


def test_backfills_question_shaped_ubiquiti_install_request() -> None:
    atoms = [
        _Atom(
            "open_question",
            "Do you have resources that can do a Ubiquiti install for some switches, routers, badge reader, cameras, and APs?",
        )
    ]
    out, count = backfill_quote_task_atoms(atoms, project_id="gecko")
    assert count == 1
    task = [a for a in out if a.atom_type == AtomType.task][0]
    assert task.raw_text == "Ubiquiti configuration / install support"
    assert task.value["task_tier"] == "parent"


def test_does_not_backfill_excluded_network_buildout() -> None:
    assert not should_backfill_task("Network build out does not need to be built into this.")
    out, count = backfill_quote_task_atoms(
        [_Atom("scope_item", "*   General firewall/network configuration")],
        project_id="gecko",
    )
    assert count == 0
    assert len(out) == 1


def test_does_not_backfill_email_header_or_narrative_fragments() -> None:
    bad = [
        "From: patrick@purtera-it.com | To: etroci@nmcms.com | Subject: 010058 - Ubiquiti Configuration Gecko Robotics",
        "Customer indicated networking configuration is largely handled internally; primary need is access-control configuration and identity integration.",
        "areas within the office, we have a lot of like OKTA groups already set up for that and we would just need to. It'd be",
        "Okta integration is considered a significant requirement by the customer.",
        "PurTera agreed to investigate Okta integration, validate feasibility, and prepare a statement of work and quote.",
        "Badge access, camera configuration, and UID Enterprise onboarding are the primary focus areas.",
    ]
    atoms = [_Atom("scope_item", text, kind="paragraph") for text in bad]
    out, count = backfill_quote_task_atoms(atoms, project_id="gecko")
    assert count == 0
    assert [a for a in out if a.atom_type == AtomType.task] == []


def test_normalizes_knowledge_transfer_label() -> None:
    atoms = [_Atom("scope_item", "*   Knowledge transfer / walking him through the setup", kind="email_body_line")]
    out, count = backfill_quote_task_atoms(atoms, project_id="gecko")
    assert count == 1
    task = [a for a in out if a.atom_type == AtomType.task][0]
    assert task.raw_text == "Knowledge transfer / guided handoff"
