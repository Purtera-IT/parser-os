"""PurTera-made deliverables are outputs, not evidence.

Purpulse stamps each manifest artifact it judges ours (the SA's Deal Kit or pricing sheet, our proposal, our SOW,
the SOW PurPulse generated) with ``artifact_role: "purtera_deliverable"`` and a reason (Platform-infra
shared/purtera-deliverable.js); a person can also mark a file the customer's (``"evidence"``). The compile reads
the role as an owner ruling (c3_record.provenance.manifest_ruling): ours is out of the quoting view and the quote
and kept by the delivery read. On 010454 (2026-11-09) the Deal Kit's prefilled tasks were copied from the SA's
own Deal Kit sheet, which the compile had read as the customer's evidence.
"""
from __future__ import annotations

import json

import pytest

from app.core import c3_type_base as c3
from app.core.orbitbrief_envelope import PARSER_MANIFEST_SIDECAR, _load_manifest_provenance
from c3_record import provenance as prov

DAY = 86400.0
T0 = 1_780_000_000.0


@pytest.fixture(autouse=True)
def _fresh_deal():
    """_view_rows writes the compile's module state; leave it as it was for the next test."""
    saved = dict(c3._DEAL)
    yield
    c3._DEAL.clear()
    c3._DEAL.update(saved)


def _doc(direction="", kind="", message=False, filename="", sender="", ruling=None, role=""):
    return {"direction": direction, "sender": sender, "role": role, "kind": kind, "carrier": None,
            "carrier_at": 0.0, "message": message, "filename": filename, "ruling": ruling}


def _atom(key, doc, text, at, kind="", side=""):
    return {"label_key": key, "atom_id": key, "doc_id": doc, "text": text, "entered_at": at, "doc_kind": kind,
            "speaker_side": side, "section": ""}


def _deal(ruling):
    """The customer's ask, then a workbook nothing reads as paper (no kind, no price), then the customer's
    follow-up: only the manifest's ruling says the workbook is ours."""
    atoms = [
        _atom("c1", "ask", "We need 40 access points installed at 3 sites.", T0, kind="email"),
        _atom("w1", "sheet", "Task: mount access point", T0 + DAY),
        _atom("w2", "sheet", "Task: run cable to IDF", T0 + DAY),
        _atom("c2", "late", "Can you also cover the warehouse?", T0 + 3 * DAY, kind="email"),
    ]
    docinfo = {"ask": _doc("inbound", "email", True, sender="buyer@example.com"),
               "sheet": _doc("", "", filename="Sodexo Deal Kit.xlsx", ruling=ruling),
               "late": _doc("inbound", "email", True, sender="buyer@example.com")}
    return atoms, docinfo


def _inputs(P, atoms):
    return [atoms[i]["label_key"] for i in prov.quoting_inputs(P, atoms, exclude=prov.pricing_workbook_keys(P, atoms))]


def test_manifest_ruling_maps_roles():
    assert prov.manifest_ruling("purtera_deliverable", "heuristic") == ("ours", "heuristic")
    assert prov.manifest_ruling("PURTERA_DELIVERABLE", None) == ("ours", "manifest")
    assert prov.manifest_ruling("evidence", "pm_marked") == ("theirs", "pm_marked")
    assert prov.manifest_ruling(None) is None and prov.manifest_ruling("something_else") is None


def test_without_a_role_the_workbook_reads_as_the_customers_evidence():
    atoms, docinfo = _deal(None)
    P = prov.decide(atoms, docinfo, {})
    assert not P.doc["sheet"]["ours"]
    assert _inputs(P, atoms) == ["c1", "w1", "w2", "c2"]


def test_our_deliverable_is_out_of_the_quoting_view_like_an_owner_ruling():
    atoms, docinfo = _deal(prov.manifest_ruling("purtera_deliverable", "heuristic"))
    P = prov.decide(atoms, docinfo, {})
    assert P.doc["sheet"]["ours"] and P.doc["sheet"]["side"] == "ours_side"
    assert "manifest ruling (heuristic)" in P.doc["sheet"]["why"]
    # Ours, and its arrival is our first paper: the same cut an owner ruling sets.
    assert P.cut == T0 + DAY
    assert _inputs(P, atoms) == ["c1"]


def test_a_name_guess_gives_way_to_paper_that_names_another_provider():
    atoms, docinfo = _deal(prov.manifest_ruling("purtera_deliverable", "heuristic"))
    docinfo["sheet"]["third_party"] = "CDW"
    P = prov.decide(atoms, docinfo, {})
    assert not P.doc["sheet"]["ours"]
    # A person's mark does not give way.
    atoms, docinfo = _deal(prov.manifest_ruling("purtera_deliverable", "pm_marked"))
    docinfo["sheet"]["third_party"] = "CDW"
    P = prov.decide(atoms, docinfo, {})
    assert P.doc["sheet"]["ours"]


def test_a_person_marking_our_looking_paper_theirs_keeps_it_an_input():
    atoms = [_atom("s1", "sow", "Provider will install 40 access points.", T0, kind="sow_draft", side="ours"),
             _atom("s2", "sow", "Price: $12,000 per site.", T0, kind="sow_draft", side="ours")]
    info = {"sow": _doc("purtera_outbound", "sow_draft", role="SOW provided by PurTera-IT")}
    assert prov.decide(atoms, info, {}).doc["sow"]["ours"]
    info["sow"]["ruling"] = prov.manifest_ruling("evidence", "pm_marked")
    P = prov.decide(atoms, info, {})
    assert not P.doc["sow"]["ours"] and P.doc["sow"]["side"] == "theirs"
    assert prov.is_input(P, atoms[0])


def test_the_owners_rulings_win_over_the_manifest():
    d = "art_4e0b98f29cdd4175"            # RULINGS: theirs
    atoms = [_atom("x1", d, "Kiosk install steps", T0, kind="sow_draft", side="ours")]
    info = {d: _doc("purtera_outbound", "sow_draft", ruling=prov.manifest_ruling("purtera_deliverable", "pm_marked"))}
    P = prov.decide(atoms, info, {})
    assert not P.doc[d]["ours"] and P.doc[d]["why"].startswith("owner ruling")


def test_another_jobs_document_is_never_ruled_ours():
    atoms, docinfo = _deal(prov.manifest_ruling("purtera_deliverable", "purpulse_generated"))
    P = prov.decide(atoms, docinfo, {}, other_job={"sheet": "document_job other_job"})
    assert not P.doc["sheet"]["ours"] and P.cut == float("inf")


def test_the_manifest_role_is_read_from_the_sidecar(tmp_path):
    manifest = {"artifacts": [
        {"filename": "010454 Deal Kit.xlsx", "artifact_role": "purtera_deliverable", "artifact_role_reason": "heuristic"},
        {"filename": "Kiosk SOW.pdf", "artifact_role": "evidence", "artifact_role_reason": "pm_marked"},
        {"filename": "site-list.xlsx"},
        {"filename": "odd.pdf", "artifact_role": "nonsense", "artifact_role_reason": "heuristic"},
    ]}
    (tmp_path / PARSER_MANIFEST_SIDECAR).write_text(json.dumps(manifest), encoding="utf-8")
    p = _load_manifest_provenance(tmp_path)
    assert p["010454 Deal Kit.xlsx"]["artifact_role"] == "purtera_deliverable"
    assert p["010454 Deal Kit.xlsx"]["artifact_role_reason"] == "heuristic"
    assert p["Kiosk SOW.pdf"]["artifact_role"] == "evidence"
    assert p["site-list.xlsx"]["artifact_role"] is None and p["site-list.xlsx"]["artifact_role_reason"] is None
    assert p["odd.pdf"]["artifact_role"] is None


def _row(key, doc, text, at, kind=""):
    return {"key": key, "atom_id": key, "doc_id": doc, "text": text, "entered_at": at, "doc_kind": kind,
            "speaker_side": "", "section": "", "order": 0.0}


def _compile_docs(role=None, reason=None):
    sheet = {"doc_id": "sheet", "kind": "file", "filename": "Sodexo Deal Kit.xlsx", "role": "", "origin": {}}
    if role:
        sheet.update(artifact_role=role, artifact_role_reason=reason)
    return [{"doc_id": "ask", "kind": "email", "filename": "010454-hs-email-1.eml", "role": "",
             "origin": {"direction": "customer_inbound", "sender_email": "buyer@example.com"}},
            sheet,
            {"doc_id": "late", "kind": "email", "filename": "010454-hs-email-2.eml", "role": "",
             "origin": {"direction": "customer_inbound", "sender_email": "buyer@example.com"}}]


def _rows():
    return [_row("c1", "ask", "We need 40 access points installed at 3 sites.", T0, "email"),
            _row("w1", "sheet", "Task: mount access point", T0 + DAY),
            _row("w2", "sheet", "Task: run cable to IDF", T0 + DAY),
            _row("c2", "late", "Can you also cover the warehouse?", T0 + 3 * DAY, "email")]


def test_the_compiles_quoting_read_leaves_our_deliverable_out(monkeypatch):
    monkeypatch.setitem(c3._DEAL, "view", "quoting")
    monkeypatch.setitem(c3._DEAL, "other_job", {})
    monkeypatch.delenv("SOWSMITH_C3_OURS_OUT", raising=False)
    kept = c3._view_rows(_rows(), _compile_docs(), {}, {})
    assert [r["key"] for r in kept] == ["c1", "w1", "w2", "c2"]
    kept = c3._view_rows(_rows(), _compile_docs("purtera_deliverable", "heuristic"), {}, {})
    assert [r["key"] for r in kept] == ["c1"]
    out = c3.out_of_quoting_view()
    assert out["w1"] == "ours" and out["w2"] == "ours" and out["c2"] == "after_cut"
    # SOWSMITH_C3_OURS_OUT=0 keeps our lines in the brief and the quote (the heads' read still leaves them out).
    monkeypatch.setenv("SOWSMITH_C3_OURS_OUT", "0")
    assert "w1" not in c3.out_of_quoting_view() and "c2" in c3.out_of_quoting_view()


def test_the_delivery_read_keeps_our_deliverable(monkeypatch):
    monkeypatch.setitem(c3._DEAL, "view", "planning")
    monkeypatch.setitem(c3._DEAL, "other_job", {})
    monkeypatch.setattr(c3, "delivery_read", lambda: True)
    kept = c3._view_rows(_rows(), _compile_docs("purtera_deliverable", "purpulse_generated"), {}, {})
    assert {"w1", "w2"} <= {r["key"] for r in kept}
    # ...while the quote, made on the same compile, still may not read it.
    assert c3._DEAL["quote_out_why"]["w1"] == "ours"
