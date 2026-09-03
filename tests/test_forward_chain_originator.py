"""Whose document is this? Read the chain from the bottom, and only this chain.

Deal 010215: the customer's ten SOWs arrived inside a double forward
(Bernie at Sodexo -> Quinton at CDW -> Trent, who forwarded it into HubSpot).
Attributing them to Trent files them as material we produced, and
`admissibility` makes type decisive over stage on purpose -- so they became
unreadable by deal_kit, sow and orbitbrief. 431 of 489 atoms.
"""

from __future__ import annotations

from app.core.orbitbrief_envelope import _originating_sender, _resolve_delivered_by


class _Ref:
    def __init__(self, artifact_id, index, sender):
        self.artifact_id = artifact_id
        self.locator = {"message_index": index, "sender": sender}


class _Atom:
    def __init__(self, *refs):
        self.source_refs = list(refs)


_FWD = "art_forward"
_REPLY = "art_reply"


def test_the_originator_is_the_bottom_of_the_chain():
    atoms = [
        _Atom(_Ref(_FWD, 0, "t@purtera-it.com")),
        _Atom(_Ref(_FWD, 1, "Quinton James <quinton.james@cdw.com>")),
        _Atom(_Ref(_FWD, 2, "Donnelly, Bernie <Bernie.Donnelly@sodexo.com>")),
    ]
    assert _originating_sender(atoms, artifact_id=_FWD) == (
        "Donnelly, Bernie <Bernie.Donnelly@sodexo.com>"
    )


def test_another_message_s_chain_cannot_answer_for_this_one():
    """Dedup merges atoms across documents and the winner keeps the losers'
    refs. message_index counts from the top of whichever message it came from,
    so an index from a REPLY is not comparable to one from this forward."""
    atoms = [
        _Atom(_Ref(_FWD, 0, "t@purtera-it.com")),
        _Atom(_Ref(_FWD, 1, "Quinton James <quinton.james@cdw.com>")),
        _Atom(_Ref(_FWD, 2, "Donnelly, Bernie <Bernie.Donnelly@sodexo.com>")),
        # Leaked in from the reply: index 2 there, a different chain entirely.
        _Atom(_Ref(_REPLY, 2, "Trent Torrence <t@purtera-it.com>")),
        _Atom(_Ref(_REPLY, 3, "Someone Else <else@example.com>")),
    ]
    assert _originating_sender(atoms, artifact_id=_FWD) == (
        "Donnelly, Bernie <Bernie.Donnelly@sodexo.com>"
    ), "a foreign ref must not be able to speak for this document"


def test_unscoped_still_works_for_a_single_document():
    atoms = [_Atom(_Ref(_FWD, 0, "a@x.com")), _Atom(_Ref(_FWD, 1, "b@y.com"))]
    assert _originating_sender(atoms) == "b@y.com"


def test_delivered_by_joins_on_the_iso_stamp_not_the_rfc_header():
    """email_thread.date is the raw RFC 2822 header; the delivered stamp is ISO.
    Comparing them matched nothing, so every file fell to signature-scraping --
    which returns the FORWARDER, the opposite of what the join is for."""
    documents = [
        {
            "filename": "fwd.eml",
            "authored_at": "2026-08-12T18:00:51.058Z",
            "email_thread": {
                "sender": "t@purtera-it.com",
                "date": "Wed, 12 Aug 2026 18:00:51 +0000",
            },
            "originated_by": "Donnelly, Bernie <Bernie.Donnelly@sodexo.com>",
        },
        {
            "filename": "SOW Some School.docx",
            "lifecycle": {
                "delivered": [
                    {
                        "kind": "email",
                        "ts": "2026-08-12T18:00:51.058Z",
                        "text": "Fw: … Trent Torrence t@purtera-it.com 404.771.3490",
                    }
                ]
            },
        },
    ]
    _resolve_delivered_by(documents)
    sow = documents[1]
    assert sow["delivered_by"] == "Donnelly, Bernie <Bernie.Donnelly@sodexo.com>"
    assert sow["delivered_by_source"] == "delivering message"
    assert sow["forwarded_by"] == "t@purtera-it.com"
