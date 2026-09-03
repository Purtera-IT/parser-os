"""Deal 010215: the email delivering all 11 SOWs was severed from the six
message discussion that produced it.

HubSpot does not give each reply the previous email's Message-ID in
References -- it stamps every message in a conversation with the SAME
synthetic anchor:

    References: <hs-thread-e4151440260dfa08e0b65f5a536b7826@hubspot.invalid>

identical across every message, matching no individual email's own
Message-ID. Grouping only resolved a reference token by looking it up as
ANOTHER message's Message-ID (true RFC 5322 chain semantics), so this token
never resolved to anything, header_linked was never set for these six
messages, and grouping fell through to the subject fallback -- which also
failed, because "Fw: Time Clock Installs..." and "RE: 010215 Time Clock
Installs..." do not normalise to the same subject.

The fix generalises: union any two artifacts that cite the SAME token in
In-Reply-To/References, whether or not that token also happens to resolve to
another artifact's own Message-ID. No HubSpot-specific code.
"""
from __future__ import annotations

from app.core.email_threading import thread_emails
from app.core.schemas import ArtifactType, AtomType, AuthorityClass, EvidenceAtom, ReviewStatus, SourceRef


def _header_atom(artifact_id, *, msg_id, references, subject_norm):
    meta = {
        "message_id": msg_id,
        "in_reply_to": None,
        "references": references,
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


def test_a_shared_synthetic_thread_anchor_unions_the_conversation():
    anchor = "<hs-thread-e4151440260dfa08e0b65f5a536b7826@hubspot.invalid>"
    delivering = _header_atom(
        "fwd", msg_id="<hs-email-114767132205@hubspot.invalid>",
        references=[anchor], subject_norm="time clock installs for marion county school district",
    )
    reply = _header_atom(
        "reply", msg_id="<hs-email-114722896209@hubspot.invalid>",
        references=[anchor], subject_norm="010215 time clock installs for marion county school district",
    )
    threads = _threads([delivering, reply])
    assert threads["fwd"] == threads["reply"], "same References anchor, must be one conversation"


def test_two_different_anchors_stay_separate():
    a = _header_atom("a", msg_id="<a@x>", references=["<hs-thread-AAA@hubspot.invalid>"], subject_norm="s1")
    b = _header_atom("b", msg_id="<b@x>", references=["<hs-thread-BBB@hubspot.invalid>"], subject_norm="s2")
    threads = _threads([a, b])
    assert threads["a"] != threads["b"]


def test_a_real_ancestor_message_id_still_works_unchanged():
    """The generalisation must not disturb ordinary RFC 5322 chains."""
    root = _header_atom("root", msg_id="<root@x>", references=[], subject_norm="s")
    reply = _header_atom("reply", msg_id="<reply@x>", references=["<root@x>"], subject_norm="s edited")
    threads = _threads([root, reply])
    assert threads["root"] == threads["reply"]


def test_three_messages_sharing_one_anchor_all_join():
    anchor = "<hs-thread-XYZ@hubspot.invalid>"
    atoms = [
        _header_atom(f"m{i}", msg_id=f"<m{i}@x>", references=[anchor], subject_norm=f"subj {i}")
        for i in range(3)
    ]
    threads = _threads(atoms)
    assert len(set(threads.values())) == 1
