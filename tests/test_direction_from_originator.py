"""A file gets the side of whoever originated it, not the name it happens to carry.

A file carries no direction of its own -- only email and notes do -- so with
none available the classifier falls back to the name. On deal 010215 that read
"SOW Smarthands Marion County SD ..." and returned ``label``: material we
produced. Those ten documents are the CUSTOMER's, one per school, sent by Bernie
Donnelly at Sodexo and forwarded in by Trent.

``admissibility`` makes type decisive over stage on purpose, so our own output
can never be readmitted as evidence. The consequence here is that a Deal Kit
model would be denied the exact ten documents the real Deal Kit was built from,
while the stage gate looked like it was working.
"""
from __future__ import annotations

from app.core.orbitbrief_envelope import _direction_from_originator

TIMELINE = {
    "created_at": "2026-08-12T18:03:46Z",
    "transitions": [
        {"ts": "2026-08-12T18:03:46Z", "label": "Open- Awaiting Scope", "order": 1},
        {"ts": "2026-08-12T18:11:39Z", "label": "Submitted for Quoting", "order": 2},
        {"ts": "2026-08-13T15:53:00Z", "label": "Decision Pending", "order": 3},
    ],
}


def _doc(**over):
    d = {
        "filename": "SOW Smarthands Marion County SD Marion High School.docx",
        "direction": None,
        "delivered_by": "Donnelly, Bernie <Bernie.Donnelly@sodexo.com>",
        "lifecycle": {"type": "SOW_DRAFT", "admissible_for": "label"},
        "deal_stage": {
            "stage_at_arrival": "Open- Awaiting Scope",
            "admissible_for": "label",
            "why": "classified as label; type is decisive for produced material",
        },
    }
    d.update(over)
    return d


def test_a_customer_document_stops_being_called_ours():
    docs = [_doc()]
    assert _direction_from_originator(docs, TIMELINE) == 1
    assert docs[0]["direction"] == "inbound"
    assert docs[0]["deal_stage"]["admissible_for"] == "evidence"
    assert docs[0]["lifecycle"]["admissible_for"] == "evidence"


def test_the_reason_names_who_it_came_from():
    # Every one of these is a claim about whether a model may read a document,
    # and a claim a person cannot audit is one they have to take on faith.
    docs = [_doc()]
    _direction_from_originator(docs, TIMELINE)
    why = docs[0]["deal_stage"]["why"]
    assert "sodexo.com" in why
    assert "originated" in why
    assert docs[0]["direction_source"] == "originator of the delivering message"


def test_an_internal_originator_is_left_alone():
    # We may be forwarding someone else's material -- which is exactly this
    # deal -- so calling an internally-forwarded file "ours" repeats the error
    # in the other direction.
    docs = [_doc(delivered_by="Trent Torrence <t@purtera-it.com>")]
    assert _direction_from_originator(docs, TIMELINE) == 0
    assert docs[0]["direction"] is None
    assert docs[0]["deal_stage"]["admissible_for"] == "label"


def test_a_document_that_already_knows_its_side_is_untouched():
    docs = [_doc(direction="outbound")]
    assert _direction_from_originator(docs, TIMELINE) == 0
    assert docs[0]["deal_stage"]["admissible_for"] == "label"


def test_a_file_with_no_delivering_message_is_left_unattributed():
    # An unattributed document must not end up looking attributed.
    docs = [_doc(delivered_by=None)]
    assert _direction_from_originator(docs, TIMELINE) == 0
    assert docs[0]["direction"] is None


def test_a_bare_name_with_no_address_asserts_nothing():
    docs = [_doc(delivered_by="Bernie Donnelly")]
    assert _direction_from_originator(docs, TIMELINE) == 0


def test_hubspot_direction_is_never_the_source():
    """The flag is deal-relative, not authorship.

    The message carrying these SOWs was marked INCOMING by HubSpot while being
    sent from our own address. Reading the flag would call a customer document
    ours whenever we forwarded it in.
    """
    import inspect

    src = inspect.getsource(_direction_from_originator)
    assert "delivered_by" in src
    assert "INTERNAL_EMAIL_DOMAINS" in src
