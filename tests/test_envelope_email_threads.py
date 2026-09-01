"""A deal's email is not 33 files, it is 6 conversations.

email_threading.py already groups messages by RFC 5322 headers and stamps every
ATOM -- and nowhere else. So a reader above atom level saw only filenames
carrying HubSpot ids, and could not tell which of them were one conversation.

Fixtures are the real threads on deal 010215 (compile 2bb2bc69).
"""
from app.core.orbitbrief_envelope import _thread_index


def doc(aid, tid, subj_norm, sender, date):
    return {
        "artifact_id": aid,
        "email_thread": {
            "thread_id": tid, "subject_norm": subj_norm,
            "sender": sender, "date": date, "thread_index": 0, "thread_size": 2,
        },
    }


MAIN = "010215 time clock installs for marion county school district"
STRIPPED = "time clock installs for marion county school district"


class TestThreadIndex:
    def test_groups_files_into_conversations(self):
        out = _thread_index([
            doc("a1", "thr_1", MAIN, "nick@cdw.com", "2026-08-12T14:00:00Z"),
            doc("a2", "thr_1", MAIN, "patrick@purtera-it.com", "2026-08-13T14:27:14Z"),
            doc("a3", "thr_2", "sodexo - psow 202017", "octavian@purtera-it.com", "2026-08-27T16:14:19Z"),
        ])
        assert len(out) == 2
        by_id = {t["thread_id"]: t for t in out}
        assert by_id["thr_1"]["message_count"] == 2
        assert by_id["thr_1"]["artifact_ids"] == ["a1", "a2"]

    def test_names_a_thread_from_its_subject(self):
        out = _thread_index([doc("a1", "thr_1", MAIN, "n@cdw.com", "2026-08-12T14:00:00Z")])
        assert out[0]["name"] == MAIN

    def test_prefers_the_subject_carrying_the_deal_number(self):
        # Same frequency, so the longer name wins -- "010215 time clock..." is
        # more useful than the same subject with the prefix stripped off.
        out = _thread_index([
            doc("a1", "thr_1", STRIPPED, "n@cdw.com", "2026-08-12T14:00:00Z"),
            doc("a2", "thr_1", MAIN, "p@purtera-it.com", "2026-08-13T14:00:00Z"),
        ])
        assert out[0]["name"] == MAIN
        assert out[0]["subject_variants"] == sorted([MAIN, STRIPPED])

    def test_reports_a_conversation_split_by_subject_drift(self):
        # On 010215 the same conversation is two threads: someone replied with
        # the deal-number prefix stripped and the References chain did not
        # bridge it. Flag it; do NOT merge -- a prefix rule over-fires, and a
        # wrong merge is harder to notice than a reported suspicion.
        out = _thread_index([
            doc("a1", "thr_1", MAIN, "n@cdw.com", "2026-08-12T14:00:00Z"),
            doc("a2", "thr_2", STRIPPED, "p@purtera-it.com", "2026-08-31T14:00:00Z"),
        ])
        by_id = {t["thread_id"]: t for t in out}
        assert by_id["thr_1"]["looks_split_with"] == ["thr_2"]
        assert by_id["thr_2"]["looks_split_with"] == ["thr_1"]
        assert len(out) == 2, "flagged, not merged"

    def test_unrelated_threads_are_not_flagged(self):
        out = _thread_index([
            doc("a1", "thr_1", MAIN, "n@cdw.com", "2026-08-12T14:00:00Z"),
            doc("a2", "thr_2", "copy of purchase order po-00034965", "p@x.com", "2026-08-27T14:00:00Z"),
        ])
        assert all(t["looks_split_with"] == [] for t in out)

    def test_orders_by_most_recent_activity(self):
        out = _thread_index([
            doc("a1", "thr_old", "older thread", "n@cdw.com", "2026-08-12T14:00:00Z"),
            doc("a2", "thr_new", "newer thread", "p@x.com", "2026-08-31T14:00:00Z"),
        ])
        assert [t["thread_id"] for t in out] == ["thr_new", "thr_old"]

    def test_records_the_span_and_who_took_part(self):
        out = _thread_index([
            doc("a1", "thr_1", MAIN, "nick@cdw.com", "2026-08-12T14:00:00Z"),
            doc("a2", "thr_1", MAIN, "patrick@purtera-it.com", "2026-08-21T09:00:00Z"),
        ])
        assert out[0]["first_message_at"].startswith("2026-08-12")
        assert out[0]["last_message_at"].startswith("2026-08-21")
        assert set(out[0]["participants"]) == {"nick@cdw.com", "patrick@purtera-it.com"}

    def test_documents_without_a_thread_are_ignored(self):
        assert _thread_index([{"artifact_id": "f1"}, {"artifact_id": "f2", "email_thread": None}]) == []
