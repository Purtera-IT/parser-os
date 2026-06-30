"""Cross-email threading stage tests.

Covers the universal guarantees: messages in different .eml files are grouped
into one conversation (RFC headers first, subject fallback second), ordered
chronologically, short replies carry the gist of what they answer, two
unrelated threads in one compile stay separate, and — critically — the stage
drops nothing (every atom survives with the same id).
"""

from __future__ import annotations

from pathlib import Path

from app.core.email_threading import dedup_quoted_history, thread_emails
from app.parsers.email_parser import (
    EmailParser,
    _parse_date_epoch,
    normalize_email_subject,
)


def test_parse_date_epoch_handles_rfc2822_and_iso8601() -> None:
    # RFC 2822 (native mail clients).
    assert _parse_date_epoch("Mon, 01 Jun 2026 09:00:00 -0400") > 0
    # ISO 8601 with trailing Z (HubSpot export).
    assert _parse_date_epoch("2026-06-19T12:43:58Z") > 0
    # ISO 8601 with milliseconds.
    assert _parse_date_epoch("2026-06-19T12:15:09.532Z") > 0
    # Ordering is consistent across the two formats.
    iso_early = _parse_date_epoch("2026-06-19T12:15:09Z")
    iso_late = _parse_date_epoch("2026-06-19T12:43:58Z")
    assert iso_early < iso_late
    # Unparseable -> 0.0 (degrades to encounter order, never raises).
    assert _parse_date_epoch("not a date") == 0.0
    assert _parse_date_epoch("") == 0.0


def _write_eml(
    tmp_path: Path,
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
    p = tmp_path / name
    p.write_bytes(raw.encode("utf-8"))
    return p


def _parse(paths: list[Path]) -> list:
    parser = EmailParser()
    atoms: list = []
    for i, p in enumerate(paths):
        atoms.extend(
            parser.parse_artifact(
                project_id="proj", artifact_id=f"art_{i}", path=p
            )
        )
    return atoms


def _thread_of(atom):
    return (atom.value or {}).get("email_thread") if isinstance(atom.value, dict) else None


# --------------------------------------------------------------------------- #
# Subject normalisation                                                       #
# --------------------------------------------------------------------------- #

def test_normalize_subject_strips_reply_forward_prefixes() -> None:
    base = normalize_email_subject("010065 CDW Monument Health AP swap")
    assert normalize_email_subject("RE: 010065 CDW Monument Health AP swap") == base
    assert normalize_email_subject("Re: Fwd: 010065 CDW Monument Health AP swap") == base
    assert normalize_email_subject("FW:  010065   CDW Monument Health AP swap") == base
    assert base  # deal number kept as a strong thread key


# --------------------------------------------------------------------------- #
# RFC header threading                                                         #
# --------------------------------------------------------------------------- #

def test_references_headers_group_into_one_ordered_thread(tmp_path) -> None:
    a = _write_eml(
        tmp_path, "1.eml",
        sender="client@acme.com", subject="010070 Camera rollout",
        date="Mon, 01 Jun 2026 09:00:00 -0400",
        body="Please quote 48 cameras for the West Wing.",
        message_id="<m1@acme.com>",
    )
    b = _write_eml(
        tmp_path, "2.eml",
        sender="pm@purtera.com", subject="RE: 010070 Camera rollout",
        date="Mon, 01 Jun 2026 11:00:00 -0400",
        body="Confirming 48 cameras. Any access constraints?",
        message_id="<m2@purtera.com>", in_reply_to="<m1@acme.com>",
        references="<m1@acme.com>",
    )
    c = _write_eml(
        tmp_path, "3.eml",
        sender="client@acme.com", subject="RE: 010070 Camera rollout",
        date="Mon, 01 Jun 2026 15:00:00 -0400",
        body="Yes, reduce from 48 to 36. Escort access required after 6pm.",
        message_id="<m3@acme.com>", in_reply_to="<m2@purtera.com>",
        references="<m1@acme.com> <m2@purtera.com>",
    )
    atoms = _parse([a, b, c])
    threaded, summary = thread_emails(atoms, project_id="proj")

    assert summary["thread_count"] == 1
    assert summary["multi_message_threads"] == 1
    assert summary["threaded_message_count"] == 3

    blocks = [_thread_of(x) for x in threaded if _thread_of(x)]
    assert blocks, "every email atom should be stamped"
    sizes = {b["thread_size"] for b in blocks}
    assert sizes == {3}
    thread_ids = {b["thread_id"] for b in blocks}
    assert len(thread_ids) == 1
    # All three positions present, ordered by date.
    assert {b["thread_index"] for b in blocks} == {1, 2, 3}


def test_short_reply_carries_prior_message_context(tmp_path) -> None:
    a = _write_eml(
        tmp_path, "1.eml",
        sender="client@acme.com", subject="010070 Camera rollout",
        date="Mon, 01 Jun 2026 09:00:00 -0400",
        body="Please quote 48 cameras for the West Wing.",
        message_id="<m1@acme.com>",
    )
    b = _write_eml(
        tmp_path, "2.eml",
        sender="client@acme.com", subject="RE: 010070 Camera rollout",
        date="Mon, 01 Jun 2026 15:00:00 -0400",
        body="Approved, go ahead.",
        message_id="<m2@acme.com>", in_reply_to="<m1@acme.com>",
        references="<m1@acme.com>",
    )
    atoms = _parse([a, b])
    threaded, _ = thread_emails(atoms, project_id="proj")

    # The terse "Approved, go ahead." reply (art_1) must know what it answered.
    reply_blocks = [
        _thread_of(x)
        for x in threaded
        if x.artifact_id == "art_1" and _thread_of(x)
    ]
    assert reply_blocks
    blk = reply_blocks[0]
    assert blk["thread_index"] == 2
    assert "replied_to" in blk
    assert "48 cameras" in blk["replied_to"]["gist"]
    assert blk["context"].startswith("In reply to ")
    assert "48 cameras" in blk["context"]


# --------------------------------------------------------------------------- #
# Subject fallback (no References — common in HubSpot exports)                 #
# --------------------------------------------------------------------------- #

def test_subject_fallback_threads_without_references(tmp_path) -> None:
    a = _write_eml(
        tmp_path, "1.eml",
        sender="client@acme.com", subject="010065 CDW Monument Health AP swap",
        date="Tue, 02 Jun 2026 09:00:00 -0400",
        body="We need to swap 12 access points at the south campus.",
    )
    b = _write_eml(
        tmp_path, "2.eml",
        sender="pm@purtera.com", subject="Re: 010065 CDW Monument Health AP swap",
        date="Tue, 02 Jun 2026 12:00:00 -0400",
        body="Got it, scheduling the AP swap for next week.",
    )
    c = _write_eml(
        tmp_path, "3.eml",
        sender="client@acme.com", subject="RE: Fwd: 010065 CDW Monument Health AP swap",
        date="Wed, 03 Jun 2026 08:00:00 -0400",
        body="Please add 3 more APs in the east wing.",
    )
    atoms = _parse([a, b, c])
    _, summary = thread_emails(atoms, project_id="proj")

    assert summary["thread_count"] == 1
    assert summary["multi_message_threads"] == 1
    assert summary["threaded_message_count"] == 3


# --------------------------------------------------------------------------- #
# Universality: two unrelated threads in one compile stay separate            #
# --------------------------------------------------------------------------- #

def test_two_unrelated_threads_stay_separate(tmp_path) -> None:
    t1a = _write_eml(
        tmp_path, "t1a.eml", sender="a@x.com", subject="010065 AP swap",
        date="Mon, 01 Jun 2026 09:00:00 -0400", body="Swap the access points.",
        message_id="<a1@x.com>",
    )
    t1b = _write_eml(
        tmp_path, "t1b.eml", sender="b@x.com", subject="RE: 010065 AP swap",
        date="Mon, 01 Jun 2026 10:00:00 -0400", body="Confirmed.",
        message_id="<a2@x.com>", in_reply_to="<a1@x.com>", references="<a1@x.com>",
    )
    t2a = _write_eml(
        tmp_path, "t2a.eml", sender="c@y.com", subject="010099 Survey Texas",
        date="Mon, 01 Jun 2026 09:00:00 -0400", body="Need a wifi survey in Dallas.",
        message_id="<b1@y.com>",
    )
    atoms = _parse([t1a, t1b, t2a])
    _, summary = thread_emails(atoms, project_id="proj")

    assert summary["thread_count"] == 2
    sizes = sorted(t["size"] for t in summary["threads"])
    assert sizes == [1, 2]


# --------------------------------------------------------------------------- #
# No-drop guarantee — the load-bearing invariant                              #
# --------------------------------------------------------------------------- #

def test_threading_drops_nothing(tmp_path) -> None:
    a = _write_eml(
        tmp_path, "1.eml", sender="client@acme.com", subject="010070 Rollout",
        date="Mon, 01 Jun 2026 09:00:00 -0400",
        body=(
            "Please quote 48 cameras for the West Wing.\n"
            "Exclude the parking garage.\n"
            "Escort access required after 6pm.\n"
            "Can you confirm the timeline?"
        ),
        message_id="<m1@acme.com>",
    )
    b = _write_eml(
        tmp_path, "2.eml", sender="client@acme.com", subject="RE: 010070 Rollout",
        date="Mon, 01 Jun 2026 15:00:00 -0400",
        body="Reduce from 48 to 36. Approved to proceed.",
        message_id="<m2@acme.com>", in_reply_to="<m1@acme.com>", references="<m1@acme.com>",
    )
    atoms = _parse([a, b])
    before_ids = [x.id for x in atoms]
    before_texts = [x.raw_text for x in atoms]

    threaded, _ = thread_emails(atoms, project_id="proj")

    # Same objects, same ids, same texts — additive only.
    assert [x.id for x in threaded] == before_ids
    assert [x.raw_text for x in threaded] == before_texts
    assert len(threaded) == len(atoms)


def test_reply_gist_skips_greeting_and_signature(tmp_path) -> None:
    """The 'in reply to' gist must be substantive scope content, not a bare
    'Hi Hiran,' greeting or a signature line (real #010065 failure shape)."""
    a = _write_eml(
        tmp_path, "1.eml",
        sender="chase@purtera-it.com", subject="010065 AP swap",
        date="Mon, 01 Jun 2026 09:00:00 -0400",
        message_id="<g1@purtera-it.com>",
        body=(
            "Hi Hiran,\n"
            "We need to swap 12 access points at the south campus after hours.\n"
            "Thanks,\n"
            "Chase\n"
            "Office: 555-123-4567\n"
        ),
    )
    b = _write_eml(
        tmp_path, "2.eml",
        sender="hiran@cdw.com", subject="RE: 010065 AP swap",
        date="Mon, 01 Jun 2026 12:00:00 -0400",
        message_id="<g2@cdw.com>", in_reply_to="<g1@purtera-it.com>",
        references="<g1@purtera-it.com>",
        body="Approved.",
    )
    atoms = _parse([a, b])
    threaded, _ = thread_emails(atoms, project_id="proj")
    reply = [
        _thread_of(x) for x in threaded
        if x.artifact_id == "art_1" and _thread_of(x)
    ][0]
    gist = reply["replied_to"]["gist"]
    assert "access points" in gist
    assert not gist.lower().startswith("hi ")
    assert "thanks" not in gist.lower()
    assert "555-123-4567" not in gist


def test_no_email_atoms_is_noop() -> None:
    out, summary = thread_emails([], project_id="proj")
    assert out == []
    assert summary["thread_count"] == 0
    assert summary["threaded_message_count"] == 0


# --------------------------------------------------------------------------- #
# Parent-accurate context (In-Reply-To, not just chronological)               #
# --------------------------------------------------------------------------- #

def test_context_uses_true_parent_on_branching_thread(tmp_path) -> None:
    """Two replies both answer the ROOT (not each other). The later reply's
    context must point at the root via In-Reply-To, not at the chronologically
    previous sibling."""
    root = _write_eml(
        tmp_path, "root.eml",
        sender="client@acme.com", subject="010070 Camera rollout",
        date="Mon, 01 Jun 2026 09:00:00 -0400",
        body="Please quote 48 cameras for the West Wing rollout.",
        message_id="<root@acme.com>",
    )
    # Sibling A replies to root at 11:00.
    sib_a = _write_eml(
        tmp_path, "a.eml",
        sender="alice@purtera.com", subject="RE: 010070 Camera rollout",
        date="Mon, 01 Jun 2026 11:00:00 -0400",
        body="Alice here, I will own the BOM for this.",
        message_id="<a@purtera.com>", in_reply_to="<root@acme.com>",
        references="<root@acme.com>",
    )
    # Sibling B ALSO replies to root, but later at 15:00. Chronological prev
    # would be sibling A; true parent is root.
    sib_b = _write_eml(
        tmp_path, "b.eml",
        sender="bob@purtera.com", subject="RE: 010070 Camera rollout",
        date="Mon, 01 Jun 2026 15:00:00 -0400",
        body="Bob here, confirming the West Wing site survey.",
        message_id="<b@purtera.com>", in_reply_to="<root@acme.com>",
        references="<root@acme.com>",
    )
    atoms = _parse([root, sib_a, sib_b])
    threaded, _ = thread_emails(atoms, project_id="proj")

    b_block = [
        _thread_of(x) for x in threaded
        if x.artifact_id == "art_2" and _thread_of(x)
    ][0]
    assert b_block["replied_to"]["via"] == "in_reply_to"
    # Parent is the ROOT (48 cameras), NOT sibling A (the BOM line).
    assert "48 cameras" in b_block["replied_to"]["gist"]
    assert "BOM" not in b_block["replied_to"]["gist"]


def test_context_falls_back_to_chronological_without_headers(tmp_path) -> None:
    """No Message-ID/In-Reply-To (HubSpot export) → context still works via the
    chronological fallback."""
    a = _write_eml(
        tmp_path, "1.eml", sender="client@acme.com", subject="010065 AP swap",
        date="Mon, 01 Jun 2026 09:00:00 -0400",
        body="We need to swap 12 access points at the south campus.",
    )
    b = _write_eml(
        tmp_path, "2.eml", sender="pm@purtera.com", subject="RE: 010065 AP swap",
        date="Mon, 01 Jun 2026 12:00:00 -0400",
        body="Approved, scheduling now.",
    )
    atoms = _parse([a, b])
    threaded, _ = thread_emails(atoms, project_id="proj")
    b_block = [
        _thread_of(x) for x in threaded
        if x.artifact_id == "art_1" and _thread_of(x)
    ][0]
    assert b_block["replied_to"]["via"] == "chronological"
    assert "access points" in b_block["replied_to"]["gist"]


# --------------------------------------------------------------------------- #
# Thread-aware quoted-history dedup                                            #
# --------------------------------------------------------------------------- #

def test_quoted_history_dedup_drops_echoes_keeps_authored(tmp_path) -> None:
    """A reply that re-quotes the prior message must keep its own authored line
    and drop the quoted echo of content already authored in the thread."""
    a = _write_eml(
        tmp_path, "1.eml", sender="client@acme.com", subject="010070 Rollout",
        date="Mon, 01 Jun 2026 09:00:00 -0400",
        body="Please quote 48 cameras for the West Wing rollout.",
        message_id="<m1@acme.com>",
    )
    # Reply authors a new line AND quotes the original (parser marks '>' quoted).
    b = _write_eml(
        tmp_path, "2.eml", sender="pm@purtera.com", subject="RE: 010070 Rollout",
        date="Mon, 01 Jun 2026 15:00:00 -0400",
        message_id="<m2@purtera.com>", in_reply_to="<m1@acme.com>",
        references="<m1@acme.com>",
        body=(
            "Confirmed, scheduling the 48 camera install for next week.\n"
            "> Please quote 48 cameras for the West Wing rollout.\n"
        ),
    )
    atoms = _parse([a, b])
    threaded, _ = thread_emails(atoms, project_id="proj")
    kept, dropped = dedup_quoted_history(threaded, project_id="proj")

    kept_texts = [k.normalized_text for k in kept]
    # Authored original survives.
    assert any("please quote 48 cameras for the west wing rollout" in t for t in kept_texts)
    # The reply's own authored line survives.
    assert any("scheduling the 48 camera install" in t for t in kept_texts)
    # The quoted echo was diverted.
    assert dropped, "expected at least one quoted echo to be dropped"
    for d in dropped:
        assert (d.value or {}).get("quoted") is True


def test_quoted_history_dedup_keeps_unique_quotes(tmp_path) -> None:
    """A quoted line with NO authored counterpart in the thread (quote of an
    external email) must be kept — it may be the only record of that content."""
    a = _write_eml(
        tmp_path, "1.eml", sender="client@acme.com", subject="010070 Rollout",
        date="Mon, 01 Jun 2026 09:00:00 -0400",
        message_id="<m1@acme.com>",
        body=(
            "Forwarding the vendor note below.\n"
            "> Vendor requires a 30 percent restocking fee on returns.\n"
        ),
    )
    atoms = _parse([a])
    threaded, _ = thread_emails(atoms, project_id="proj")
    kept, dropped = dedup_quoted_history(threaded, project_id="proj")
    kept_texts = [k.normalized_text for k in kept]
    assert any("restocking fee" in t for t in kept_texts)
    assert not dropped


def test_quoted_history_dedup_is_lossless_partition(tmp_path) -> None:
    """kept + dropped == input (every atom accounted for; nothing vanishes)."""
    a = _write_eml(
        tmp_path, "1.eml", sender="client@acme.com", subject="010070 Rollout",
        date="Mon, 01 Jun 2026 09:00:00 -0400", message_id="<m1@acme.com>",
        body="Please quote 48 cameras for the West Wing rollout.",
    )
    b = _write_eml(
        tmp_path, "2.eml", sender="pm@purtera.com", subject="RE: 010070 Rollout",
        date="Mon, 01 Jun 2026 15:00:00 -0400",
        message_id="<m2@purtera.com>", in_reply_to="<m1@acme.com>",
        references="<m1@acme.com>",
        body=(
            "Confirmed, scheduling the install.\n"
            "> Please quote 48 cameras for the West Wing rollout.\n"
        ),
    )
    atoms = _parse([a, b])
    threaded, _ = thread_emails(atoms, project_id="proj")
    kept, dropped = dedup_quoted_history(threaded, project_id="proj")
    assert len(kept) + len(dropped) == len(threaded)
    kept_ids = {k.id for k in kept}
    dropped_ids = {d.id for d in dropped}
    assert kept_ids.isdisjoint(dropped_ids)


def test_quoted_history_dedup_noop_without_threads() -> None:
    out, dropped = dedup_quoted_history([], project_id="proj")
    assert out == []
    assert dropped == []
