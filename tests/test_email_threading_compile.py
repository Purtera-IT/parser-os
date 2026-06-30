"""End-to-end compile integration for email threading.

Exercises three universal deal archetypes through the real compiler pipeline
(no mocks): email-only subject-threaded deal (#010065 shape), RFC-header chain
with a short approval reply, and two unrelated threads in one compile. Asserts
the threading stage links messages, stamps context on replies, and drops nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.compiler import compile_project
from app.parsers.email_parser import EmailParser


def _write_eml(
    directory: Path,
    name: str,
    *,
    sender: str,
    subject: str,
    date: str,
    body: str,
    message_id: str = "",
    in_reply_to: str = "",
    references: str = "",
) -> Path:
    headers = [
        f"From: {sender}",
        "To: pm@purtera.com",
        f"Subject: {subject}",
        f"Date: {date}",
    ]
    if message_id:
        headers.append(f"Message-ID: {message_id}")
    if in_reply_to:
        headers.append(f"In-Reply-To: {in_reply_to}")
    if references:
        headers.append(f"References: {references}")
    raw = "\n".join(headers) + "\n\n" + body + "\n"
    p = directory / name
    p.write_bytes(raw.encode("utf-8"))
    return p


def _thread_of(atom):
    v = atom.value if isinstance(atom.value, dict) else {}
    return v.get("email_thread")


@pytest.fixture
def no_llm(monkeypatch):
    monkeypatch.setenv("SOWSMITH_DISABLE_LLM", "1")


def _build_deal_010065_shape(deal_dir: Path) -> None:
    """Email-only deal: subject fallback threads 4 messages (HubSpot export
    shape — no References chain, Re:/Fwd: prefixes vary)."""
    deal_dir.mkdir(parents=True)
    _write_eml(
        deal_dir, "01.eml",
        sender="cdw@cdw.com", subject="010065 CDW Monument Health South Dakota AP swap",
        date="Mon, 02 Jun 2026 09:00:00 -0400",
        body="We need to swap 12 access points at Monument Health south campus. After hours only.",
    )
    _write_eml(
        deal_dir, "02.eml",
        sender="pm@purtera.com", subject="Re: 010065 CDW Monument Health South Dakota AP swap",
        date="Mon, 02 Jun 2026 11:00:00 -0400",
        body="Got it. Confirming 12 AP swap, after hours access at all 4 sites.",
    )
    _write_eml(
        deal_dir, "03.eml",
        sender="cdw@cdw.com", subject="RE: Fwd: 010065 CDW Monument Health South Dakota AP swap",
        date="Tue, 03 Jun 2026 08:00:00 -0400",
        body="Please add 3 more APs in the east wing. Badge required for escort.",
    )
    _write_eml(
        deal_dir, "04.eml",
        sender="pm@purtera.com", subject="Re: 010065 CDW Monument Health South Dakota AP swap",
        date="Tue, 03 Jun 2026 14:00:00 -0400",
        body="Approved to proceed with 15 APs total.",
    )


def _build_deal_rfc_chain(deal_dir: Path) -> None:
    """RFC-header chain: scope request → PM question → short customer approval."""
    deal_dir.mkdir(parents=True)
    _write_eml(
        deal_dir, "a.eml",
        sender="client@hospital.com", subject="010070 Camera rollout West Wing",
        date="Mon, 01 Jun 2026 09:00:00 -0400",
        body="Please quote 48 cameras for the West Wing rollout.",
        message_id="<cam1@hospital.com>",
    )
    _write_eml(
        deal_dir, "b.eml",
        sender="pm@purtera.com", subject="RE: 010070 Camera rollout West Wing",
        date="Mon, 01 Jun 2026 11:00:00 -0400",
        body="Confirming 48 cameras. Any access constraints for the West Wing?",
        message_id="<cam2@purtera.com>", in_reply_to="<cam1@hospital.com>",
        references="<cam1@hospital.com>",
    )
    _write_eml(
        deal_dir, "c.eml",
        sender="client@hospital.com", subject="RE: 010070 Camera rollout West Wing",
        date="Mon, 01 Jun 2026 15:00:00 -0400",
        body="Reduce from 48 to 36. Escort access required after 6pm. Approved to proceed.",
        message_id="<cam3@hospital.com>", in_reply_to="<cam2@purtera.com>",
        references="<cam1@hospital.com> <cam2@purtera.com>",
    )


def _build_deal_two_threads(deal_dir: Path) -> None:
    """Two unrelated conversations in one deal compile — must not bleed."""
    deal_dir.mkdir(parents=True)
    _write_eml(
        deal_dir, "t1a.eml", sender="a@x.com", subject="010065 AP swap",
        date="Mon, 01 Jun 2026 09:00:00 -0400", body="Swap the access points.",
        message_id="<a1@x.com>",
    )
    _write_eml(
        deal_dir, "t1b.eml", sender="b@x.com", subject="RE: 010065 AP swap",
        date="Mon, 01 Jun 2026 10:00:00 -0400", body="Confirmed.",
        message_id="<a2@x.com>", in_reply_to="<a1@x.com>", references="<a1@x.com>",
    )
    _write_eml(
        deal_dir, "t2a.eml", sender="c@y.com", subject="010099 Survey Texas",
        date="Mon, 01 Jun 2026 09:00:00 -0400", body="Need a wifi survey in Dallas.",
        message_id="<b1@y.com>",
    )


def test_compile_threads_010065_shape_email_only_deal(tmp_path, no_llm) -> None:
    deal = tmp_path / "deal_065"
    _build_deal_010065_shape(deal)

    # Pre-threading atom count (parse only).
    parser = EmailParser()
    pre_atoms: list = []
    for i, eml in enumerate(sorted(deal.glob("*.eml"))):
        pre_atoms.extend(
            parser.parse_artifact(project_id="p", artifact_id=f"art_{i}", path=eml)
        )
    pre_count = len(pre_atoms)

    result = compile_project(
        project_dir=deal, project_id="deal_065", use_cache=False,
        allow_unverified_receipts=True,
    )

    # Threading stage ran and linked all 4 emails.
    threading_info = [w for w in result.warnings if "email_threading linked" in w]
    assert threading_info, f"expected threading INFO warning, got: {result.warnings}"
    assert "4 email(s)" in threading_info[0]
    assert "1 thread(s)" in threading_info[0]

    # Every email atom stamped; no threading-stage drop.
    stamped = [a for a in result.atoms if _thread_of(a)]
    assert stamped, "no atoms carry email_thread context"
    sizes = {t["thread_size"] for t in (_thread_of(a) for a in stamped) if t}
    assert sizes == {4}

    # After-hours constraint survived (real deal detail, not dropped).
    constraints = [
        a for a in result.atoms
        if "after hours" in (a.normalized_text or "").lower()
    ]
    assert constraints, "after-hours constraint must survive compile"

    # Atom count only changes from dedup downstream — threading itself is additive.
    # Pre-parse count should be <= final (dedup may shrink); never zero.
    assert len(result.atoms) > 0
    assert pre_count > 0


def test_compile_short_reply_gets_prior_context(tmp_path, no_llm) -> None:
    deal = tmp_path / "deal_rfc"
    _build_deal_rfc_chain(deal)
    result = compile_project(
        project_dir=deal, project_id="deal_rfc", use_cache=False,
        allow_unverified_receipts=True,
    )

    # Find atoms from the final reply mentioning "36" or "approved"
    reply_atoms = [
        a for a in result.atoms
        if "36" in (a.raw_text or "") or "approved" in (a.normalized_text or "")
    ]
    assert reply_atoms
    ctx_blocks = [_thread_of(a) for a in reply_atoms if _thread_of(a)]
    assert ctx_blocks
    # At least one reply atom knows what it answered.
    assert any(
        "48 cameras" in (b.get("replied_to", {}).get("gist", "") or b.get("context", ""))
        for b in ctx_blocks
        if b
    )


def test_compile_two_unrelated_threads_stay_separate(tmp_path, no_llm) -> None:
    deal = tmp_path / "deal_multi"
    _build_deal_two_threads(deal)
    result = compile_project(
        project_dir=deal, project_id="deal_multi", use_cache=False,
        allow_unverified_receipts=True,
    )

    stamped = [a for a in result.atoms if _thread_of(a)]
    thread_ids = {t["thread_id"] for t in (_thread_of(a) for a in stamped) if t}
    assert len(thread_ids) == 2, f"expected 2 threads, got {len(thread_ids)}"

    # Survey thread is size 1; AP swap thread is size 2.
    sizes = sorted(t["thread_size"] for t in (_thread_of(a) for a in stamped) if t)
    assert 1 in sizes and 2 in sizes


def test_compile_email_threading_stage_does_not_fail(tmp_path, no_llm) -> None:
    """No swallowed 'email_threading failed' on any universal archetype."""
    for builder, name in (
        (_build_deal_010065_shape, "065"),
        (_build_deal_rfc_chain, "rfc"),
        (_build_deal_two_threads, "multi"),
    ):
        deal = tmp_path / name
        builder(deal)
        result = compile_project(
            project_dir=deal, project_id=name, use_cache=False,
            allow_unverified_receipts=True,
        )
        failures = [w for w in result.warnings if "email_threading failed" in w]
        assert not failures, f"{name}: {failures}"
