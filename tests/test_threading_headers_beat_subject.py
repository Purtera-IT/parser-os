"""RFC headers must beat a subject line, and a subject must not overrule them.

The subject fallback was not a fallback. It unioned every subject match
unconditionally, so two different conversations sharing a line — "Site Survey"
on two deals — merged even when their headers said otherwise, and headers could
never win.
"""
from __future__ import annotations

from app.core.email_threading import thread_emails
from app.core.schemas import AtomType, AuthorityClass, EvidenceAtom, ReviewStatus, SourceRef
from app.core.schemas import ArtifactType


def _header_atom(artifact_id: str, *, msg_id: str, in_reply_to: str | None, subject_norm: str):
    meta = {
        "message_id": msg_id,
        "in_reply_to": in_reply_to,
        "references": [],
        "subject_norm": subject_norm,
        "sender": "a@b.com",
        "date_raw": "2026-08-12T08:00:00Z",
    }
    return EvidenceAtom(
        id=f"atm_{artifact_id}",
        project_id="p",
        artifact_id=artifact_id,
        atom_type=AtomType.deal_metadata,
        raw_text="From: a@b.com | Subject: x",
        normalized_text="from a b com subject x",
        value={"kind": "email_header", "email_thread_meta": meta},
        source_refs=[
            SourceRef(
                id=f"src_{artifact_id}",
                artifact_id=artifact_id,
                artifact_type=ArtifactType.email,
                filename=f"{artifact_id}.eml",
                locator={"kind": "email_header"},
                extraction_method="email_headers",
                parser_version="test",
            )
        ],
        authority_class=AuthorityClass.machine_extractor,
        confidence=0.5,
        review_status=ReviewStatus.auto_accepted,
        parser_version="test",
    )


def _threads(atoms):
    thread_emails(atoms)
    out = {}
    for a in atoms:
        blk = (a.value or {}).get("email_thread")
        if blk:
            out.setdefault(a.artifact_id, blk.get("thread_id"))
    return out


def test_a_reply_joins_its_parent_even_when_the_subject_was_edited():
    atoms = [
        _header_atom("root", msg_id="<m1@x>", in_reply_to=None, subject_norm="time clock installs"),
        _header_atom("reply", msg_id="<m2@x>", in_reply_to="<m1@x>", subject_norm="marion county scheduling"),
    ]
    t = _threads(atoms)
    assert len(set(t.values())) == 1, f"headers must join a renamed reply: {t}"


def test_two_conversations_sharing_a_subject_stay_apart():
    # "Site Survey" on two different deals. Subject alone merges them.
    atoms = [
        _header_atom("a_root", msg_id="<a1@x>", in_reply_to=None, subject_norm="site survey"),
        _header_atom("a_reply", msg_id="<a2@x>", in_reply_to="<a1@x>", subject_norm="site survey"),
        _header_atom("b_root", msg_id="<b1@x>", in_reply_to=None, subject_norm="site survey"),
        _header_atom("b_reply", msg_id="<b2@x>", in_reply_to="<b1@x>", subject_norm="site survey"),
    ]
    t = _threads(atoms)
    assert t["a_root"] == t["a_reply"]
    assert t["b_root"] == t["b_reply"]
    assert t["a_root"] != t["b_root"], f"different conversations merged on subject: {t}"


def test_the_safety_net_still_works_when_there_are_no_headers():
    # The case the fallback exists for: a HubSpot export that stripped the
    # References chain. Without headers, subject is all there is.
    atoms = [
        _header_atom("one", msg_id="", in_reply_to=None, subject_norm="network refresh sow"),
        _header_atom("two", msg_id="", in_reply_to=None, subject_norm="network refresh sow"),
    ]
    t = _threads(atoms)
    assert len(set(t.values())) == 1, f"subject must still reunite header-less mail: {t}"


def test_a_header_linked_message_is_not_dragged_by_a_stray_subject_match():
    atoms = [
        _header_atom("root", msg_id="<m1@x>", in_reply_to=None, subject_norm="site survey"),
        _header_atom("reply", msg_id="<m2@x>", in_reply_to="<m1@x>", subject_norm="site survey"),
        _header_atom("stray", msg_id="", in_reply_to=None, subject_norm="site survey"),
    ]
    t = _threads(atoms)
    assert t["root"] == t["reply"]
    assert t["stray"] != t["root"], "a header-less stray must not join a header-threaded pair"
