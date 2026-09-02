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


class TestAtomThreadContext:
    """An atom needs the conversation, not just its own message.

    Threading already answers "what is this a reply to" via the per-atom gist.
    It does not answer "what conversation is this, who is in it, and is this the
    last word" -- which is what a reader needs to weigh a single line like
    "Yes, approved, go ahead with 36".
    """

    def _setup(self):
        from app.core.orbitbrief_envelope import _enrich_atom_threads

        threads = _thread_index([
            doc("a1", "thr_1", MAIN, "nick@cdw.com", "2026-08-12T14:00:00Z"),
            doc("a2", "thr_1", MAIN, "patrick@purtera-it.com", "2026-08-21T09:00:00Z"),
        ])
        atoms = [
            {"id": "atm_1", "structured": {"email_thread": {"thread_id": "thr_1", "date": "2026-08-12T14:00:00Z"}}},
            {"id": "atm_2", "structured": {"email_thread": {"thread_id": "thr_1", "date": "2026-08-21T09:00:00Z"}}},
        ]
        _enrich_atom_threads(atoms, threads)
        return atoms

    def test_an_atom_learns_the_conversation_name_and_participants(self):
        a = self._setup()[0]["structured"]["email_thread"]
        assert a["thread_name"] == MAIN
        assert set(a["participants"]) == {"nick@cdw.com", "patrick@purtera-it.com"}
        assert a["thread_first_message_at"].startswith("2026-08-12")
        assert a["thread_last_message_at"].startswith("2026-08-21")

    def test_it_knows_whether_it_is_the_last_word(self):
        # An approval later revised reads very differently from one nobody
        # answered.
        atoms = self._setup()
        assert atoms[0]["structured"]["email_thread"]["is_latest_in_thread"] is False
        assert atoms[1]["structured"]["email_thread"]["is_latest_in_thread"] is True

    def test_no_summary_is_invented(self):
        # A generated gist of twenty messages would be an unfalsifiable claim
        # sitting in the evidence set. Only deterministic facts are added.
        a = self._setup()[0]["structured"]["email_thread"]
        assert not any("summary" in k for k in a)

    def test_an_unthreaded_atom_is_left_alone(self):
        from app.core.orbitbrief_envelope import _enrich_atom_threads

        atoms = [{"id": "x", "structured": {"kind": "table_row"}}, {"id": "y"}]
        before = [dict(a) for a in atoms]
        _enrich_atom_threads(atoms, _thread_index([doc("a1", "thr_1", MAIN, "n@cdw.com", "2026-08-12T14:00:00Z")]))
        assert atoms == before

    def test_a_split_conversation_is_carried_to_the_atom(self):
        from app.core.orbitbrief_envelope import _enrich_atom_threads

        threads = _thread_index([
            doc("a1", "thr_1", MAIN, "n@cdw.com", "2026-08-12T14:00:00Z"),
            doc("a2", "thr_2", STRIPPED, "p@x.com", "2026-08-31T14:00:00Z"),
        ])
        atoms = [{"id": "atm_1", "structured": {"email_thread": {"thread_id": "thr_1", "date": "2026-08-12T14:00:00Z"}}}]
        _enrich_atom_threads(atoms, threads)
        assert atoms[0]["structured"]["email_thread"]["thread_looks_split_with"] == ["thr_2"]


class TestPerMessageDetail:
    """A thread must say which message carried what.

    "This thread discussed the SOWs" and "THIS message is where the SOWs came
    from" are different facts, and only the second tells you whose documents
    they are. On deal 010215 exactly one message carries all eleven Marion
    County attachments -- and its originator is the customer, not the forwarder.
    """

    def _docs(self):
        a = doc("a1", "thr_1", MAIN, "t@purtera-it.com", "2026-08-12T18:00:00Z")
        a["sender_email"] = "t@purtera-it.com"
        a["originated_by"] = "Donnelly, Bernie <Bernie.Donnelly@sodexo.com>"
        a["attachment_ids"] = [str(i) for i in range(11)]
        a["direction"] = "inbound"
        b = doc("a2", "thr_1", MAIN, "patrick@purtera-it.com", "2026-08-12T18:11:00Z")
        b["sender_email"] = "patrick@purtera-it.com"
        b["attachment_ids"] = []
        return [a, b]

    def test_each_message_reports_its_sender_and_attachment_count(self):
        t = _thread_index(self._docs())[0]
        msgs = t["messages"]
        assert [m["attachment_count"] for m in msgs] == [11, 0]
        assert msgs[0]["sender"] == "t@purtera-it.com"

    def test_a_message_reports_who_the_chain_started_with(self):
        t = _thread_index(self._docs())[0]
        assert "Bernie.Donnelly@sodexo.com" in t["messages"][0]["originated_by"]

    def test_messages_are_in_time_order(self):
        t = _thread_index(self._docs())[0]
        assert [m["date"] for m in t["messages"]] == sorted(m["date"] for m in t["messages"])

    def test_the_thread_totals_what_it_carried(self):
        assert _thread_index(self._docs())[0]["attachments_carried"] == 11

    def test_a_thread_with_no_attachments_reports_zero(self):
        docs = [doc("a1", "thr_2", "quick sync", "n@cdw.com", "2026-08-12T14:00:00Z")]
        assert _thread_index(docs)[0]["attachments_carried"] == 0
