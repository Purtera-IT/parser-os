"""Who sent the message that delivered a file.

A file carries no sender of its own. The lifecycle work recovered the message
that delivered it, but only as {kind, text, ts} -- so the UI could say WHICH
email brought a document and not who wrote it, which is exactly the difference
between a SOW draft they sent and one we sent.
"""
from app.core.orbitbrief_envelope import _resolve_delivered_by


def email_doc(aid, sender, when, direction="inbound"):
    return {
        "artifact_id": aid,
        "filename": f"010215-hs-email-{aid}.eml",
        "direction": direction,
        "authored_at": when,
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
