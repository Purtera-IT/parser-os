"""The compile's read is one of the two training views (S1-S7), and its readings are never stale.

Hermetic: made-up lines, the service is a stub."""
from __future__ import annotations

import pytest

import app.core.c3_type_base as c3
from app.core.schemas import ArtifactType, AuthorityClass, EvidenceAtom, ReviewStatus, SourceRef


def _make(i, text, artifact_id, kind, locator=None):
    ref = SourceRef(id=f"src{i}", artifact_id=artifact_id, artifact_type=kind, filename=f"{artifact_id}.file",
                    locator=locator or {}, extraction_method="test", parser_version="test")
    return EvidenceAtom(id=f"atm{i}", project_id="deal-test", artifact_id=artifact_id, atom_type="scope_item",
                        raw_text=text, normalized_text=text.lower(), value={}, source_refs=[ref],
                        authority_class=AuthorityClass.machine_extractor, confidence=0.5,
                        review_status=ReviewStatus.auto_accepted, parser_version="test")


BUYER = {"via": "email", "direction": "customer_inbound", "sender_email": "pat@buyer-example.com",
         "sender_name": "Pat", "sent_at": "2026-03-01T00:00:00Z"}


def _deal():
    atoms = [_make(1, "We need 12 cameras at the plant.", "mail", ArtifactType.email),
             _make(2, "Provider will install 12 cameras.", "sow", ArtifactType.docx),
             _make(3, "Total price $9,600 per site.", "sow", ArtifactType.docx),
             _make(4, "Can you also cover the warehouse?", "mail2", ArtifactType.email),
             _make(5, "Labor | $95/hr", "kit", ArtifactType.xlsx),
             _make(6, "Kickoff call is booked; the install is scheduled for Monday.", "mail3", ArtifactType.email),
             _make(7, "Thanks, see you on site.", "mail4", ArtifactType.email)]
    ours = {"via": "note", "direction": "internal", "sender_email": "rep@purtera-it.com"}
    docs = [{"doc_id": "mail", "kind": "email", "sent_at": "2026-03-01T00:00:00Z", "origin": BUYER},
            {"doc_id": "kit", "kind": "deal_kit", "sent_at": "2026-03-02T00:00:00Z", "origin": ours},
            {"doc_id": "sow", "kind": "sow_draft", "sent_at": "2026-03-03T00:00:00Z", "origin": ours,
             "role": "SOW provided by PurTera-IT"},
            {"doc_id": "mail2", "kind": "email", "sent_at": "2026-03-05T00:00:00Z", "origin": BUYER},
            {"doc_id": "won", "kind": "sow_signed", "sent_at": "2026-03-08T00:00:00Z", "origin": ours},
            {"doc_id": "mail3", "kind": "email", "sent_at": "2026-03-10T00:00:00Z",
             "origin": {"via": "email", "direction": "purtera_outbound", "sender_email": "rep@purtera-it.com",
                        "sent_at": "2026-03-10T00:00:00Z"}},
            {"doc_id": "mail4", "kind": "email", "sent_at": "2026-03-12T00:00:00Z", "origin": BUYER}]
    atoms.append(_make(8, "Signed.", "won", ArtifactType.pdf))
    return atoms, docs


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in ("SOWSMITH_C3_VIEW", "SOWSMITH_C3_BRIEF", "SOWSMITH_C3_URL"):
        monkeypatch.delenv(k, raising=False)
    yield
    c3.set_deal(None)


def test_a_quoting_read_holds_only_their_lines_before_our_first_pricing():
    atoms, docs = _deal()
    c3.set_deal(docs, {}, as_of="2026-03-20T00:00:00Z")
    assert c3.view_name() == "quoting"
    _, rows, keys = c3._rows(atoms, {}, documents=docs)
    assert [r["atom_id"] for r in rows] == ["atm1"]           # the Deal Kit cut at Mar 2
    assert rows[0]["view"] == "quoting" and rows[0]["stage_at"] == "scoping"
    assert len(keys) == len(atoms)                             # the compile's atoms are unchanged
    assert c3._STATUS["view"]["cut_why"].startswith("our deal_kit")


def test_the_as_of_cuts_a_quoting_read_earlier():
    atoms, docs = _deal()
    c3.set_deal(docs, {}, as_of="2026-02-01T00:00:00Z")
    _, rows, _ = c3._rows(atoms, {}, documents=docs)
    assert rows == []


def test_a_planning_read_ends_at_our_kickoff_after_the_deal_is_won():
    atoms, docs = _deal()
    c3.set_deal(docs, {})
    assert c3.view_name() == "planning"
    _, rows, _ = c3._rows(atoms, {}, documents=docs)
    got = [r["atom_id"] for r in rows]
    assert "atm6" not in got and "atm7" not in got             # the kickoff and after
    assert "atm5" not in got                                   # never a Deal Kit line
    assert {"atm1", "atm2", "atm3", "atm4", "atm8"} <= set(got)  # our SOW and the signed one are inputs here
    assert {r["view"] for r in rows} == {"planning"}


def test_another_jobs_documents_are_in_neither_read():
    atoms, docs = _deal()
    c3.set_deal(docs, {}, as_of="2026-03-20T00:00:00Z")
    c3.set_other_job({"mail"})
    _, rows, _ = c3._rows(atoms, {}, documents=docs)
    assert rows == []


def test_view_off_reads_every_line(monkeypatch):
    monkeypatch.setenv("SOWSMITH_C3_VIEW", "0")
    atoms, docs = _deal()
    c3.set_deal(docs, {}, as_of="2026-03-20T00:00:00Z")
    _, rows, _ = c3._rows(atoms, {}, documents=docs)
    assert c3.view_name() == "" and len(rows) == len(atoms)


def test_each_read_asks_its_views_heads(monkeypatch):
    monkeypatch.setenv("SOWSMITH_C3_BRIEF", "1")
    atoms, docs = _deal()
    c3.set_deal(docs, {})
    body = c3._brief_body({})
    assert body["view"] == "planning" and "read:sow_section" not in body["line_heads"]
    assert "read:task_owner_role" in body["line_heads"] and body["table_heads"] == ["jdg:roster", "jdg:sheet"]
    c3.set_deal(docs, {}, as_of="2026-03-20T00:00:00Z")
    body = c3._brief_body({})
    assert "read:task_owner_role" not in body["line_heads"] and "read:needed_by" not in body["line_heads"]
    assert "read:sow_section" in body["line_heads"]


def _service(monkeypatch, answers):
    calls = []

    def request(path, body, timeout):
        if path == "/health":
            return {"ready": True}
        calls.append(body)
        got = answers(len(calls), body)
        if isinstance(got, Exception):
            raise got
        return got

    monkeypatch.setenv("SOWSMITH_C3_URL", "http://c3.test")
    monkeypatch.setattr(c3, "_request", request)
    return calls


def _types(body):
    return {a["key"]: ["scope_item", 0.99] for a in body["deal"]["atoms"]}


def test_a_reading_is_reused_only_for_the_same_read(monkeypatch):
    monkeypatch.setenv("SOWSMITH_C3_VIEW", "0")
    calls = _service(monkeypatch, lambda n, b: {"types": _types(b), "links": [{"n": n}], "questions": [n],
                                                 "tables": {"t": {"jdg:roster": ["site_roster", 0.9]}}})
    atoms, docs = _deal()
    c3.set_deal(docs, {})
    c3.classify(atoms)
    c3.classify(atoms)
    assert len(calls) == 1                                     # the same deal: read once
    c3.classify(atoms[:-1])
    assert len(calls) == 2                                     # a subset is another read
    assert c3._DEAL["links"] == [{"n": 2}] and c3._DEAL["questions"] == [2]   # replaced, never extended
    assert c3.full_read()["tables"] == {"t": {"jdg:roster": ["site_roster", 0.9]}}


def test_documents_are_cached_by_what_was_read(monkeypatch):
    monkeypatch.setenv("SOWSMITH_C3_VIEW", "0")
    calls = _service(monkeypatch, lambda n, b: {"types": _types(b), "documents": {"mail": {"jdg:document_job":
                                                                                          ["this_deal", 0.9]}}})
    atoms, docs = _deal()
    c3.set_deal(docs, {})
    c3.documents(atoms)
    c3.documents(atoms)
    assert len(calls) == 1
    atoms[0].raw_text = "We need 14 cameras at the plant."
    c3.documents(atoms)
    assert len(calls) == 2


def test_a_changed_deal_is_read_again_and_a_failed_reread_keeps_nothing_stale(monkeypatch):
    monkeypatch.setenv("SOWSMITH_C3_VIEW", "0")
    monkeypatch.setattr(c3, "authority_enabled", lambda: True)
    state = {"fail": False}

    def answers(n, b):
        if state["fail"]:
            return RuntimeError("down")
        return {"types": {k: [f"t{n}", 0.99] for k in _types(b)}, "documents": {}}

    _service(monkeypatch, answers)
    atoms, docs = _deal()
    c3.set_deal(docs, {})
    c3.documents(atoms)
    assert c3.reread_if_new(atoms) == 0                        # nothing changed
    atoms[1].raw_text = "Provider will install 14 cameras."
    assert c3.reread_if_new(atoms) > 0
    assert {v[0] for v in c3._DEAL["types"].values()} == {"t2"}   # replaced, not merged
    state["fail"] = True
    atoms[1].raw_text = "Provider will install 16 cameras."
    c3.reread_if_new(atoms)
    assert c3._DEAL["types"] == {}                             # no stale reading survives a failed re-read


def test_other_job_by_head_uses_the_shared_set_aside(monkeypatch):
    atoms, docs = _deal()
    monkeypatch.setattr(c3, "documents", lambda a: {"mail2": {c3.DOCUMENT_JOB_KEY: ["other_job", 0.99]},
                                                     "mail": {c3.DOCUMENT_JOB_KEY: ["other_job", 0.5]}})
    assert c3.other_job_by_head(atoms, floor=0.9) == {"mail2"}
    # A document whose own lines name the deal stays.
    assert c3.other_job_by_head(atoms, deal_name="the warehouse expansion", floor=0.9) == {"mail2"}
    atoms[3].raw_text = "Can you also cover the warehouse expansion?"
    assert c3.other_job_by_head(atoms, deal_name="Warehouse Expansion 2026", floor=0.9) == set()


def test_the_service_answers_only_the_views_heads():
    from ml.c3 import serve

    seen = {}

    def read(m, d, s, **kw):
        seen["heads"] = kw["heads"]
        seen["views"] = [a.view for a in d.atoms]
        return {"lines": {}, "deal": {"deal:primary_service": {"answer": "x", "p": 0.9},
                                      "jdg:task_tier": {"answer": "t1", "p": 0.9}}}

    base = serve.Base(load=lambda: ("m", object(), "cpu"), read=read)
    base.model = ("m", object(), "cpu")
    base._loaded.set()
    payload = {"deal_id": "d", "atoms": [{"key": "a", "text": "We need 4 APs.", "entered_at": 1.0}]}
    out = base.read(payload, 5.0, line_heads=("read:sow_section", "read:task_owner_role"), full=True,
                    view="planning")
    assert "read:sow_section" not in seen["heads"] and "read:task_owner_role" in seen["heads"]
    assert seen["views"] == ["planning"] and set(out["deal"]) == {"jdg:task_tier"}
    out = base.read(payload, 5.0, line_heads=("read:sow_section", "read:task_owner_role"), full=True,
                    view="quoting")
    assert "read:task_owner_role" not in seen["heads"] and set(out["deal"]) == {"deal:primary_service"}


def test_a_deal_with_no_time_anywhere_reads_every_line():
    atoms, _ = _deal()
    c3.set_deal([], {}, as_of="2026-03-20T00:00:00Z")
    _, rows, _ = c3._rows(atoms, {}, documents=[])
    assert len(rows) == len(atoms) and c3._STATUS["view"]["undated"]
