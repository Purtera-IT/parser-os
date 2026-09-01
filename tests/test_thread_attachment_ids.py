"""A thread message must carry WHICH files it brought, not just how many.

Deal 010215 showed a thread badged "22 attachments". The Aug 12 forward's
eleven were real -- ten Marion County SOW documents and a 4.3 MB Kronos install
guide. The Aug 20 forward's eleven resolved to zero rows anywhere: never
mirrored, not deleted.

A count cannot be checked. With the ids present the reader can be taken to the
documents, and the ones that never arrived can say so instead of inflating a
number.
"""
from __future__ import annotations

from app.core.orbitbrief_envelope import _thread_index


def _doc(artifact_id: str, sender: str, attachment_ids: list[str], thread_id: str = "t1") -> dict:
    return {
        "artifact_id": artifact_id,
        "sender_email": sender,
        "originated_by": None,
        "direction": "INCOMING_EMAIL",
        "attachment_ids": attachment_ids,
        "email_thread": {
            "thread_id": thread_id,
            "subject": "Time Clock Installs for Marion County School District",
            "sender": sender,
            "date": "2026-08-12T18:00:51Z",
        },
    }


def test_a_message_carries_the_ids_of_what_it_brought():
    docs = [_doc("a1", "t@purtera-it.com", ["219296620707", "219297008906"])]
    threads = _thread_index(docs)
    msg = threads[0]["messages"][0]
    assert msg["attachment_ids"] == ["219296620707", "219297008906"]
    assert msg["attachment_count"] == 2


def test_the_count_still_matches_the_ids():
    # These are rendered next to each other. A count that disagrees with the
    # list it labels is worse than either alone.
    docs = [_doc("a1", "t@purtera-it.com", ["1", "2", "3"])]
    msg = _thread_index(docs)[0]["messages"][0]
    assert msg["attachment_count"] == len(msg["attachment_ids"])


def test_a_message_with_no_attachments_carries_an_empty_list():
    # Not None: the UI iterates it, and a null would read as "unknown" when the
    # honest answer is "none".
    docs = [_doc("a1", "quinton.james@cdw.com", [])]
    msg = _thread_index(docs)[0]["messages"][0]
    assert msg["attachment_ids"] == []
    assert msg["attachment_count"] == 0


def test_blank_and_null_ids_are_dropped():
    docs = [_doc("a1", "t@purtera-it.com", ["219296620707", "", None])]
    msg = _thread_index(docs)[0]["messages"][0]
    assert msg["attachment_ids"] == ["219296620707"]


def test_ids_are_strings_so_a_numeric_id_still_matches_a_file_row():
    # hubspot_file_id is text in the database. A numeric id here would never
    # join, and the badge would report every attachment as missing.
    docs = [_doc("a1", "t@purtera-it.com", [219296620707])]
    msg = _thread_index(docs)[0]["messages"][0]
    assert msg["attachment_ids"] == ["219296620707"]


def test_two_forwards_in_one_thread_keep_their_own_sets():
    # The whole point: the thread total was 22, split 11 real and 11 that never
    # arrived. Merging them loses which message is which.
    docs = [
        _doc("a1", "t@purtera-it.com", ["219296620707", "219297008906"]),
        _doc("a2", "patrick@purtera-it.com", ["219920670588"]),
    ]
    msgs = {m["artifact_id"]: m for m in _thread_index(docs)[0]["messages"]}
    assert msgs["a1"]["attachment_ids"] == ["219296620707", "219297008906"]
    assert msgs["a2"]["attachment_ids"] == ["219920670588"]
    assert _thread_index(docs)[0]["attachments_carried"] == 3
