"""doc_origin: who brought each document in, and which way it went."""
from __future__ import annotations

import json

from app.core.doc_origin import INBOUND, INTERNAL, OUTBOUND, annotate_doc_origin, origin_for


def _carried(**kw):
    base = {"kind": "email", "id": "e1", "filename": "010300-hs-email-e1.eml", "self": False,
            "sender_name": None, "sender_email": None, "sent_at": None, "recipients": [], "direction": None}
    base.update(kw)
    return base


def test_file_sent_by_us_reads_as_our_output():
    doc = {"filename": "SOW.pdf", "artifact_type": "pdf",
           "carried_by": _carried(sender_name="Patrick Kelly", sender_email="patrick@purtera-it.com",
                                  sent_at="2026-06-09T15:00:00Z", recipients=["sarah.halpern@cdw.com"],
                                  direction="purtera_outbound")}
    o = origin_for(doc, {})
    assert o["direction"] == OUTBOUND
    assert o["label"] == "Sent by PurTera (Patrick Kelly) to customer · Jun 9 email"
    assert o["sender_email"] == "patrick@purtera-it.com" and o["via"] == "email"


def test_file_from_a_reseller_reads_as_received_from_customer():
    doc = {"filename": "BOM.xlsx", "artifact_type": "xlsx",
           "carried_by": _carried(sender_email="sarah.halpern@cdw.com", sent_at="2026-06-04T15:00:00Z",
                                  recipients=["patrick@purtera-it.com"], direction="customer_inbound")}
    # The name comes from the carrying message's own MIME From when it is here.
    carrier = {"filename": "010300-hs-email-e1.eml", "artifact_type": "email",
               "email_thread": {"sender": "Sarah Halpern <sarah.halpern@cdw.com>"}}
    o = origin_for(doc, {carrier["filename"]: carrier})
    assert o["direction"] == INBOUND
    assert o["label"] == "Received from customer (Sarah Halpern, CDW) · Jun 4 email"


def test_customer_material_forwarded_by_our_staff_is_the_customers():
    carrier = {"filename": "010300-hs-email-e1.eml", "artifact_type": "email",
               "email_thread": {"sender": "Trent Torrence <t@purtera-it.com>"},
               "originated_by": "Bernie Donnelly <bernie.donnelly@sodexo.com>"}
    doc = {"filename": "Site list.pdf", "artifact_type": "pdf",
           "carried_by": _carried(sender_email="t@purtera-it.com", sent_at="2026-08-12T18:00:51Z",
                                  recipients=["patrick@purtera-it.com"], direction="internal")}
    o = origin_for(doc, {carrier["filename"]: carrier})
    assert o["direction"] == INBOUND
    assert o["sender_email"] == "bernie.donnelly@sodexo.com"
    assert o["forwarded_by"] == "Trent Torrence"
    assert o["label"].startswith("Received from customer (Bernie Donnelly, Sodexo) · forwarded by Trent Torrence")


def test_a_file_nothing_carried_is_an_internal_upload():
    o = origin_for({"filename": "kit.xlsx", "artifact_type": "xlsx"}, {})
    assert o == {**o, "direction": INTERNAL, "via": "upload", "label": "Uploaded internally"}


def test_file_attached_to_a_hubspot_note_names_the_author():
    doc = {"filename": "psow.pdf", "artifact_type": "pdf",
           "hubspot_note": {"author": "Trent Torrence", "author_email": "t@purtera-it.com",
                            "created_at": "2026-06-02T10:00:00Z", "hubspot_note_id": "77"}}
    o = origin_for(doc, {})
    assert o["direction"] == INTERNAL and o["via"] == "note"
    assert o["label"] == "Uploaded internally by Trent Torrence · Jun 2 note"


def test_an_email_document_is_its_own_origin_and_names_come_from_the_roster():
    doc = {"filename": "010300-hs-email-9.eml", "artifact_type": "email",
           "authored_at": "2026-06-04T09:00:00Z", "email_thread": {"sender": "sarah.halpern@cdw.com"}}
    env = {"deal_roster": {"people": [{"email": "sarah.halpern@cdw.com", "name": "Sarah Halpern"}]}}
    annotate_doc_origin([doc], env)
    assert doc["doc_origin"]["label"] == "Received from customer (Sarah Halpern, CDW) · Jun 4 email"


def test_every_atom_carries_its_documents_origin():
    docs = [{"artifact_id": "a1", "filename": "SOW.pdf", "artifact_type": "pdf",
             "carried_by": _carried(sender_email="patrick@purtera-it.com", recipients=["x@cust.com"],
                                    sent_at="2026-06-09T15:00:00Z")},
            {"artifact_id": "a2", "filename": "kit.xlsx", "artifact_type": "xlsx"}]
    env = {"atoms": [{"id": "1", "artifact_id": "a1"}, {"id": "2", "artifact_id": "a2"}]}
    assert annotate_doc_origin(docs, env) == 2
    assert env["atoms"][0]["doc_origin"]["direction"] == OUTBOUND
    assert env["atoms"][0]["doc_origin"]["sender_email"] == "patrick@purtera-it.com"
    assert env["atoms"][1]["doc_origin"] == {
        "direction": INTERNAL, "label": "Uploaded internally", "sender_name": None,
        "sender_email": None, "sent_at": None, "via": "upload"}


def test_manifest_carried_by_reaches_the_envelope_documents_and_atoms(tmp_path):
    from app.core.compiler import compile_project
    from app.core.orbitbrief_envelope import build_orbitbrief_envelope

    project = tmp_path / "deal"
    project.mkdir()
    (project / "010300 Proposal.txt").write_text(
        "Statement of Work for VC Links.\n\n"
        "Site address: 15733 US-224, Findlay, OH.\n\n"
        "Purtera will install and terminate 24 Cat6 drops in the warehouse.\n", encoding="utf-8")
    manifest = {
        "context": {"crm": {"deal_name": "010300-VC Links"}},
        "artifacts": [{
            "filename": "010300 Proposal.txt",
            "carried_by": {"kind": "email", "id": "e1", "filename": "010300-hs-email-e1.eml", "self": False,
                           "sender_name": "Patrick Kelly", "sender_email": "patrick@purtera-it.com",
                           "sent_at": "2026-06-09T15:00:00Z", "recipients": ["sarah.halpern@cdw.com"],
                           "direction": "purtera_outbound"},
        }],
    }
    (project / ".parser_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    result = compile_project(project_dir=project, project_id="010300", use_cache=False)
    env = build_orbitbrief_envelope(project_dir=project, compile_result=result)
    (doc,) = [d for d in env["documents"] if d["filename"] == "010300 Proposal.txt"]
    assert doc["doc_origin"]["label"] == "Sent by PurTera (Patrick Kelly) to customer · Jun 9 email"
    own = [a for a in env["atoms"] if a.get("artifact_id") == doc["artifact_id"]]
    assert own and all(a["doc_origin"]["direction"] == OUTBOUND for a in own)
