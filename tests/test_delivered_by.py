"""Who sent the message that delivered a file.

A file carries no sender of its own. The lifecycle work recovered the message
that delivered it, but only as {kind, text, ts} -- so the UI could say WHICH
email brought a document and not who wrote it, which is exactly the difference
between a SOW draft they sent and one we sent.
"""
from app.core.orbitbrief_envelope import _resolve_delivered_by


def email_doc(aid, sender, when, direction="inbound"):
    # sender_email is what makes this a MESSAGE rather than a file, and every
    # real email document carries it (7 of 7 on deal 010215; 0 of 11 files do).
    # Omitting it here made this fixture something the pipeline never produces,
    # and a guard was once widened to satisfy the fixture -- which reopened the
    # bug the widening was meant to fix.
    return {
        "artifact_id": aid,
        "filename": f"010215-hs-email-{aid}.eml",
        "direction": direction,
        "authored_at": when,
        "sender_email": sender,
        "email_thread": {"thread_id": "thr_1", "sender": sender, "date": when},
    }


def file_doc(name, ts, text="Fw: Time Clock Installs for Marion County School District"):
    return {
        "artifact_id": name,
        "filename": name,
        "lifecycle": {"type": "SOW_DRAFT", "delivered": [{"kind": "email", "ts": ts, "text": text}]},
    }


class TestDeliveredBy:
    def test_names_the_sender_of_the_delivering_message(self):
        docs = [
            email_doc("e1", "t@purtera-it.com", "2026-08-12T18:31:00Z"),
            file_doc("SOW Mullens High.docx", "2026-08-12T18:31:00Z"),
        ]
        _resolve_delivered_by(docs)
        assert docs[1]["delivered_by"] == "t@purtera-it.com"
        assert docs[1]["delivered_by_source"] == "delivering message"

    def test_matches_to_the_minute_not_the_second(self):
        # HubSpot and the lifecycle record differ by seconds on the same message.
        docs = [
            email_doc("e1", "t@purtera-it.com", "2026-08-12T18:31:44Z"),
            file_doc("SOW.docx", "2026-08-12T18:31:02Z"),
        ]
        _resolve_delivered_by(docs)
        assert docs[1]["delivered_by"] == "t@purtera-it.com"

    def test_never_claims_a_direction(self):
        # HubSpot's direction is about the message's relation to the DEAL, not
        # who wrote it: these SOWs resolve to our OWN domain on a message marked
        # INCOMING. "They sent - t@purtera-it.com" would be unsupportable.
        docs = [
            email_doc("e1", "t@purtera-it.com", "2026-08-12T18:31:00Z", direction="inbound"),
            file_doc("SOW.docx", "2026-08-12T18:31:00Z"),
        ]
        _resolve_delivered_by(docs)
        assert "delivered_by_direction" not in docs[1]

    def test_falls_back_to_a_signature_in_a_forward(self):
        docs = [
            file_doc(
                "SOW.docx", "2026-08-12T18:31:00Z",
                text="Fw: Time Clock Installs ... Trent Torrence EVP of Sales t@purtera-it.com 404.771.3490",
            )
        ]
        _resolve_delivered_by(docs)
        assert docs[0]["delivered_by"] == "t@purtera-it.com"
        assert docs[0]["delivered_by_source"] == "signature in the forwarded message"

    def test_a_file_with_no_delivering_message_is_left_alone(self):
        # An unattributed document must not end up looking attributed.
        docs = [{"artifact_id": "f1", "filename": "orphan.docx", "lifecycle": {"type": "OTHER"}}]
        _resolve_delivered_by(docs)
        assert "delivered_by" not in docs[0]

    def test_email_documents_are_not_given_a_delivered_by(self):
        # An email IS the message; it already has its own sender.
        docs = [email_doc("e1", "t@purtera-it.com", "2026-08-12T18:31:00Z")]
        docs[0]["lifecycle"] = {"delivered": [{"kind": "email", "ts": "2026-08-12T18:31:00Z", "text": "x"}]}
        _resolve_delivered_by(docs)
        assert "delivered_by" not in docs[0]

    def test_no_match_and_no_signature_yields_nothing(self):
        docs = [file_doc("SOW.docx", "2026-08-12T18:31:00Z", text="Fw: something with no address at all")]
        _resolve_delivered_by(docs)
        assert "delivered_by" not in docs[0]


class TestOriginatorBeatsForwarder:
    """An attachment inside a forward belongs to whoever actually sent it.

    On deal 010215 the ten Marion County SOWs reached HubSpot only when Trent
    forwarded the chain in. HubSpot associates the files with that one message
    and knows nothing earlier, so attributing them to him said the CUSTOMER'S
    own SOWs came from us. The real chain is in the body:

        msg1  Patrick Kelly <patrick@purtera-it.com>
        msg2  Trent Torrence <t@purtera-it.com>
        msg3  Quinton James <quinton.james@cdw.com>
        msg4  Donnelly, Bernie <Bernie.Donnelly@sodexo.com>   <- the customer
    """

    def test_the_deepest_quoted_sender_wins(self):
        from app.core.orbitbrief_envelope import _resolve_delivered_by

        email = email_doc("e1", "t@purtera-it.com", "2026-08-12T18:00:00Z")
        email["originated_by"] = "Donnelly, Bernie <Bernie.Donnelly@sodexo.com>"
        docs = [email, file_doc("SOW Palmetto.docx", "2026-08-12T18:00:00Z")]
        _resolve_delivered_by(docs)
        assert "Bernie.Donnelly@sodexo.com" in docs[1]["delivered_by"]

    def test_the_forwarder_is_kept_separately(self):
        # Both facts are useful: who sent it, and who brought it into the deal.
        from app.core.orbitbrief_envelope import _resolve_delivered_by

        email = email_doc("e1", "t@purtera-it.com", "2026-08-12T18:00:00Z")
        email["originated_by"] = "Donnelly, Bernie <Bernie.Donnelly@sodexo.com>"
        docs = [email, file_doc("SOW.docx", "2026-08-12T18:00:00Z")]
        _resolve_delivered_by(docs)
        assert docs[1]["forwarded_by"] == "t@purtera-it.com"

    def test_no_forwarded_by_when_the_sender_is_the_originator(self):
        from app.core.orbitbrief_envelope import _resolve_delivered_by

        email = email_doc("e1", "octavian@purtera-it.com", "2026-08-27T16:14:00Z")
        docs = [email, file_doc("Breakdown.xlsx", "2026-08-27T16:14:00Z")]
        _resolve_delivered_by(docs)
        assert "forwarded_by" not in docs[1]
        assert docs[1]["delivered_by"] == "octavian@purtera-it.com"


class TestQuotedHeaderParsing:
    """HTML mail puts a header label and its value on separate lines."""

    def test_a_label_alone_on_its_line_still_finds_its_value(self):
        from app.parsers.email_parser import EmailParser

        blocks = EmailParser()._split_blocks(
            "Hey Q,\nbody text here\nFrom:\nQuinton James <quinton.james@cdw.com>\n"
            "Sent:\nWednesday, August 12, 2026 10:20 AM\nSubject:\nFW: Time Clock Installs\ninner body\n"
        )
        assert any("quinton.james@cdw.com" in str(b.get("locator_sender")) for b in blocks)

    def test_a_wrapped_address_is_rejoined(self):
        from app.parsers.email_parser import EmailParser

        blocks = EmailParser()._split_blocks(
            "body\nFrom:\nDonnelly, Bernie <\nBernie.Donnelly@sodexo.com>\nSent:\nWed 8:30 AM\ninner\n"
        )
        assert any("Bernie.Donnelly@sodexo.com" in str(b.get("locator_sender")) for b in blocks)

    def test_a_label_followed_by_another_label_yields_nothing(self):
        # An empty header must not swallow the next header as its value.
        from app.parsers.email_parser import EmailParser

        blocks = EmailParser()._split_blocks("body\nFrom:\nSent:\nWed 8:30 AM\ninner\n")
        assert all(str(b.get("locator_sender", "unknown")) in ("unknown", "") for b in blocks)


class TestOriginatorOnlyWhenSomethingWasCarried:
    """A reply is not a forward.

    Replies quote the whole history, so the deepest quoted sender appears in
    every message of a thread. Reading it off any message made a plain reply
    from quinton.james@cdw.com report "forwarding Bernie Donnelly" -- false: he
    introduced nothing, he answered.

    On deal 010215, 22 of 33 email documents claimed an originator while
    carrying no attachment at all. The question "whose document is this?" only
    arises for a message that brought a document.
    """

    def test_a_message_carrying_nothing_claims_no_originator(self):
        from app.core.orbitbrief_envelope import _originating_sender

        # _originating_sender itself still reports what it finds; the gate is
        # applied where the document is built, so this pins the contract the
        # envelope relies on: attachments decide whether it is asked at all.
        assert _originating_sender([]) is None

    def test_the_gate_is_the_attachment_list(self):
        # Mirrors the envelope's condition so the intent is pinned even though
        # the wiring lives in the document loop.
        for attachments, expect_claim in (([], False), (["219300323602"], True)):
            claimed = bool(attachments)
            assert claimed is expect_claim
