"""Universal hybrid summary+transcript routing and conversation_meta tests.

Fixtures use generic names/vendors — never a real deal id, customer, or
person from production data.
"""

from __future__ import annotations

import re

from app.core.hybrid_summary_transcript import (
    CONVERSATION_META_KIND,
    classify_transcript_turn_role,
    collapse_greeting_clusters,
    detect_hybrid_summary_transcript,
    is_head_excluded_atom,
    is_non_deal_meta_atom,
    rewrite_hybrid_pdf_atoms,
    split_speaker_timestamp_turns,
    stamp_conversation_reply_adjacency,
)
from app.core.normalizers import detect_speaker, parse_timestamp, split_transcript_segments
from app.core.schemas import (
    AtomType,
    AuthorityClass,
    EvidenceAtom,
    ReviewStatus,
    SourceRef,
)
from app.core.span_admission import _is_protected_email_atom


def _mk_scope(text: str, *, page: int, filename: str = "Meeting_Summary_and_Full_Transcript.pdf") -> EvidenceAtom:
    return EvidenceAtom(
        id="atm_" + str(abs(hash((text, page))))[:12],
        project_id="proj_demo",
        artifact_id="art_demo",
        atom_type=AtomType.scope_item,
        raw_text=text,
        normalized_text=text.lower(),
        value={"kind": "paragraph", "text": text},
        entity_keys=[],
        source_refs=[
            SourceRef(
                id="src_demo",
                artifact_id="art_demo",
                artifact_type="pdf",
                filename=filename,
                locator={"page": page, "block_kind": "paragraph"},
                extraction_method="t",
                parser_version="t",
            )
        ],
        authority_class=AuthorityClass.meeting_note,
        confidence=0.7,
        review_status=ReviewStatus.needs_review,
        review_flags=[],
        parser_version="t",
    )


# ── detection ──


def test_detect_hybrid_from_filename_and_pages():
    plan = detect_hybrid_summary_transcript(
        filename="Kickoff_Meeting_Summary_and_Full_Transcript.pdf",
        title="Meeting Summary and Full Transcript",
        page_texts=[
            "Executive Summary\n- Badge readers and cameras are in scope.\n- SSO is required.",
            "Full Transcript\nAlex Rivera [00:04] Hey, how are you?\nSam Chen [00:12] Good to meet you.\n"
            "Alex Rivera [04:10] We need to configure badge zones and cameras.",
        ],
    )
    assert plan is not None
    assert plan.kind == "hybrid"
    assert plan.transcript_start_page == 1
    assert any("transcript" in r for r in plan.reasons)


def test_detect_transcript_only_from_density():
    plan = detect_hybrid_summary_transcript(
        filename="discovery_call_transcript.txt",
        page_texts=[
            "Alex Rivera [00:01] Hello everyone.\n"
            "Sam Chen [00:08] Thanks for joining.\n"
            "Alex Rivera [02:00] Let's install twelve cameras and two switches.\n"
            "Sam Chen [02:30] And exclude the firewall build-out.",
        ],
    )
    assert plan is not None
    assert plan.kind in {"hybrid", "transcript_only"}
    assert plan.transcript_start_page == 0


def test_non_transcript_doc_not_detected():
    plan = detect_hybrid_summary_transcript(
        filename="Statement_of_Work.pdf",
        page_texts=["This SOW covers installation of access points at three sites."],
    )
    assert plan is None


# ── speaker split ──


def test_split_speaker_timestamp_turns_generic():
    text = (
        "Full Transcript\n"
        "Alex Rivera [00:04] Hey, how are you doing?\n"
        "Sam Chen [00:12] Nice to meet you. I'm a systems engineer.\n"
        "Alex Rivera [05:20] We need SSO integration and camera configuration."
    )
    turns = split_speaker_timestamp_turns(text)
    assert len(turns) >= 3
    speakers = [t.speaker for t in turns if t.speaker]
    assert "Alex Rivera" in speakers
    assert "Sam Chen" in speakers
    dealish = [t for t in turns if "SSO" in (t.text or "") or "camera" in (t.text or "").lower()]
    assert dealish


def test_split_does_not_glue_sentence_end_into_speaker():
    """Regression: 'Hey.\\nTrent Torrence [00:56]' must not become speaker='Hey. Trent…'."""
    text = (
        "Alex Rivera [00:48] Well, hey, Patrick. How are you? Hey.\n"
        "Trent Torrence [00:56] Good.\n"
        "Alex Rivera [00:56] Dan, how are you doing?"
    )
    turns = split_speaker_timestamp_turns(text)
    speakers = [t.speaker for t in turns if t.speaker]
    assert speakers == ["Alex Rivera", "Trent Torrence", "Alex Rivera"]
    assert any(t.speaker == "Trent Torrence" and t.text.strip() == "Good." for t in turns)
    line = "Alex Rivera [03:15] We should configure the guest VLAN."
    assert detect_speaker(line) == "Alex Rivera"
    assert parse_timestamp(line) == "03:15"
    segs = split_transcript_segments(line)
    assert len(segs) == 1
    assert segs[0]["speaker"] == "Alex Rivera"
    assert "guest VLAN" in segs[0]["text"]


# ── turn role classification ──


def test_classify_greeting_intro_vs_deal():
    assert classify_transcript_turn_role("Hey, how you doing? Been a while.") == "greeting"
    assert (
        classify_transcript_turn_role(
            "I'm Sam Chen, systems engineer on the deployment team."
        )
        == "intro"
    )
    assert (
        classify_transcript_turn_role(
            "We're waiting for the customer to join from another call."
        )
        == "logistics"
    )
    assert classify_transcript_turn_role("Yeah.") == "acknowledgment"
    assert classify_transcript_turn_role("Good.") == "acknowledgment"
    assert classify_transcript_turn_role("Thanks.") == "acknowledgment"
    assert (
        classify_transcript_turn_role(
            "We need to configure badge readers and integrate with SSO."
        )
        == "deal"
    )
    # Substance wins over greeting opener
    assert (
        classify_transcript_turn_role(
            "Hey — can you confirm we should install twelve cameras?"
        )
        == "deal"
    )
    # Speaker stamp must not inflate soft-social into deal/scope.
    assert (
        classify_transcript_turn_role(
            "Jacob Vander-Plaats [00:41] I don't remember what that was..."
        )
        == "filler"
    )
    assert (
        classify_transcript_turn_role(
            "Daniel Peterson [00:48] We'll touch on this... Hey Patrick. How are you?"
        )
        == "greeting"
    )


# ── rewrite ──


def test_rewrite_hybrid_pdf_atoms_tags_greetings_keeps_summary():
    filename = "Acme_Meeting_Summary_and_Full_Transcript.pdf"
    structured = {
        "document": {"title": "Meeting Summary and Full Transcript"},
        "pages": [
            {
                "page": 0,
                "sections": [
                    {
                        "heading": "Executive Summary",
                        "blocks": [
                            {
                                "kind": "bullet_list",
                                "items": [
                                    {"text": "Badge access and camera configuration are primary focus areas."},
                                    {"text": "SSO integration is a hard requirement."},
                                ],
                            }
                        ],
                    }
                ],
            },
            {
                "page": 1,
                "sections": [
                    {
                        "heading": "Full Transcript",
                        "blocks": [
                            {
                                "kind": "paragraph",
                                "text": (
                                    "Alex Rivera [00:04] Hey, how are you?\n"
                                    "Sam Chen [00:20] Nice to meet you.\n"
                                    "Alex Rivera [06:10] We need to configure badge zones and cameras with SSO."
                                ),
                            }
                        ],
                    }
                ],
            },
        ],
    }
    # Simulate the bad PDF path: summary bullets + one glued transcript blob.
    atoms = [
        _mk_scope(
            "Badge access and camera configuration are primary focus areas.",
            page=0,
            filename=filename,
        ),
        _mk_scope(
            "SSO integration is a hard requirement.",
            page=0,
            filename=filename,
        ),
        _mk_scope(
            "Alex Rivera [00:04] Hey, how are you? Sam Chen [00:20] Nice to meet you. "
            "Alex Rivera [06:10] We need to configure badge zones and cameras with SSO.",
            page=1,
            filename=filename,
        ),
    ]
    # Mark page-0 as bullets so substance gate would keep them.
    for a in atoms[:2]:
        a.value = {"kind": "bullet", "depth": 1, "text": a.raw_text}
        a.source_refs[0].locator = {"page": 0, "block_kind": "bullet_list"}

    out = rewrite_hybrid_pdf_atoms(
        atoms=atoms,
        structured_doc=structured,
        filename=filename,
        project_id="proj_demo",
        artifact_id="art_demo",
        parser_version="test_v1",
    )
    # Summary bullets survive.
    summary_texts = {
        a.raw_text
        for a in out
        if (a.source_refs[0].locator or {}).get("page") == 0
        and a.atom_type == AtomType.scope_item
    }
    assert any("Badge access" in t for t in summary_texts)
    assert any("SSO" in t for t in summary_texts)

    meta = [
        a
        for a in out
        if a.atom_type == AtomType.deal_metadata
        and isinstance(a.value, dict)
        and a.value.get("kind") == CONVERSATION_META_KIND
    ]
    assert meta, "greeting/intro turns must become conversation_meta"
    assert all(is_non_deal_meta_atom(a) for a in meta)
    assert all(_is_protected_email_atom(a) for a in meta)

    # Glued blob must be gone; deal turn must exist as non-meta.
    glued = [a for a in out if "Hey, how are you? Sam Chen" in (a.raw_text or "")]
    assert glued == []
    deal_turns = [
        a
        for a in out
        if a.atom_type != AtomType.deal_metadata
        and ("badge" in (a.raw_text or "").lower() or "camera" in (a.raw_text or "").lower())
        and (a.source_refs[0].locator or {}).get("page") == 1
    ]
    assert deal_turns, "substantive transcript turn must remain a deal atom"


def test_rewrite_fragmented_pdf_blocks_like_text_rich_export():
    """Real PDF text-rich path emits speaker stamps as section headings and
    short utterance paragraphs — not one glued blob. Rewrite must still
    atomize turns and tag greetings as conversation_meta."""
    filename = "Discovery_Meeting_Summary_and_Full_Transcript.pdf"
    structured = {
        "document": {"title": "Meeting Summary and Full Transcript"},
        "pages": [
            {
                "page": 0,
                "metadata": [
                    "[text-rich page — heavyweight layout pipeline skipped; "
                    "prose extracted via lightweight text splitter]"
                ],
                "sections": [
                    {
                        "heading": "Executive Summary",
                        "blocks": [
                            {
                                "kind": "bullet_list",
                                "items": [
                                    {"text": "Badge access and camera configuration are primary focus areas."},
                                ],
                            }
                        ],
                    }
                ],
            },
            {
                "page": 1,
                "metadata": [
                    "[text-rich page — heavyweight layout pipeline skipped; "
                    "prose extracted via lightweight text splitter]"
                ],
                "sections": [
                    {
                        "heading": "Alex Rivera [00:04]",
                        "blocks": [
                            {"kind": "paragraph", "text": "Hey, how are you? Been a while."},
                        ],
                    },
                    {
                        "heading": "Sam Chen [00:20]",
                        "blocks": [
                            {"kind": "paragraph", "text": "Good."},
                            {
                                "kind": "paragraph",
                                "text": "Nice to meet you. I'm a systems engineer on the deployment team.",
                            },
                        ],
                    },
                    {
                        "heading": "Alex Rivera [06:10]",
                        "blocks": [
                            {
                                "kind": "paragraph",
                                "text": "We need to configure badge zones and cameras with SSO.",
                            },
                        ],
                    },
                ],
            },
        ],
    }
    atoms = [
        _mk_scope(
            "Badge access and camera configuration are primary focus areas.",
            page=0,
            filename=filename,
        ),
        _mk_scope("Hey, how are you? Been a while.", page=1, filename=filename),
        _mk_scope("Good.", page=1, filename=filename),
        _mk_scope(
            "Nice to meet you. I'm a systems engineer on the deployment team.",
            page=1,
            filename=filename,
        ),
        _mk_scope(
            "We need to configure badge zones and cameras with SSO.",
            page=1,
            filename=filename,
        ),
        # Junk quantity that the PDF path minted from a transcript page —
        # must be rebuilt, not kept as a duplicate contractual atom.
        EvidenceAtom(
            id="atm_qty_junk",
            project_id="proj_demo",
            artifact_id="art_demo",
            atom_type=AtomType.quantity,
            raw_text="And we have two 48 port switches.",
            normalized_text="and we have two 48 port switches.",
            value={"quantity": 2, "item": "switches"},
            entity_keys=[],
            source_refs=[
                SourceRef(
                    id="src_qty",
                    artifact_id="art_demo",
                    artifact_type="pdf",
                    filename=filename,
                    locator={"page": 1, "block_kind": "paragraph"},
                    extraction_method="t",
                    parser_version="t",
                )
            ],
            authority_class=AuthorityClass.contractual_scope,
            confidence=0.7,
            review_status=ReviewStatus.needs_review,
            review_flags=[],
            parser_version="t",
        ),
    ]
    atoms[0].value = {"kind": "bullet", "depth": 1, "text": atoms[0].raw_text}
    atoms[0].source_refs[0].locator = {"page": 0, "block_kind": "bullet_list"}

    out = rewrite_hybrid_pdf_atoms(
        atoms=atoms,
        structured_doc=structured,
        filename=filename,
        project_id="proj_demo",
        artifact_id="art_demo",
        parser_version="test_v1",
    )

    # Pipeline metadata must not leak into atoms.
    assert not any("text-rich page" in (a.raw_text or "") for a in out)

    meta = [
        a
        for a in out
        if a.atom_type == AtomType.deal_metadata
        and isinstance(a.value, dict)
        and a.value.get("kind") == CONVERSATION_META_KIND
    ]
    assert meta, "greeting/intro/filler turns must become conversation_meta"
    assert all(a.value.get("non_deal") is True for a in meta)
    assert all(_is_protected_email_atom(a) for a in meta)

    # Summary bullet survives on page 0.
    assert any(
        "Badge access" in (a.raw_text or "")
        and (a.source_refs[0].locator or {}).get("page") == 0
        for a in out
    )

    # Substantive deal turn on page 1.
    deal = [
        a
        for a in out
        if a.atom_type != AtomType.deal_metadata
        and ("badge" in (a.raw_text or "").lower() or "camera" in (a.raw_text or "").lower())
        and (a.source_refs[0].locator or {}).get("page") == 1
    ]
    assert deal
    # Original junk quantity atom must not survive as contractual_scope residue.
    assert not any(
        a.id == "atm_qty_junk" for a in out
    )


def test_split_does_not_absorb_section_chrome_across_newline():
    """Sticky 'Key Decisions' / 'Full Transcript' must not become the speaker."""
    from app.core.hybrid_summary_transcript import strip_transcript_section_chrome

    raw = (
        "Key Decisions\n"
        "Full Transcript Alex Rivera [00:04] See, Pat should be joining us shortly.\n"
        "Sam Chen [00:41] I don't remember what that was.\n"
        "Alex Rivera [01:20] Hey, how you doing?"
    )
    cleaned = strip_transcript_section_chrome(raw)
    assert "Key Decisions" not in cleaned
    assert not cleaned.lower().startswith("full transcript")
    turns = split_speaker_timestamp_turns(cleaned)
    speakers = [t.speaker for t in turns if t.speaker]
    assert speakers[0] == "Alex Rivera"
    assert "Full Transcript" not in (speakers[0] or "")
    assert all("\n" not in (s or "") for s in speakers)


def test_classify_call_logistics_voicemail_and_link():
    assert (
        classify_transcript_turn_role(
            "It. Your call has been forwarded to voicemail."
        )
        == "logistics"
    )
    assert (
        classify_transcript_turn_role(
            "Having trouble with the link. I don't know why. That would be everybody else's end."
        )
        == "logistics"
    )
    assert (
        classify_transcript_turn_role("I'm sending it to him this way.")
        == "logistics"
    )


def test_rewrite_collapses_duplicate_typed_atoms_per_turn():
    """One utterance must not mint both action_item and open_question."""
    filename = "Acme_Meeting_Summary_and_Full_Transcript.pdf"
    structured = {
        "document": {"title": "Meeting Summary and Full Transcript"},
        "pages": [
            {
                "page": 0,
                "sections": [
                    {
                        "heading": "Executive Summary",
                        "blocks": [
                            {
                                "kind": "bullet_list",
                                "items": [{"text": "Badge access is in scope."}],
                            }
                        ],
                    }
                ],
            },
            {
                "page": 1,
                "sections": [
                    {
                        "heading": "Full Transcript",
                        "blocks": [
                            {
                                "kind": "paragraph",
                                "text": (
                                    "Alex Rivera [01:59] Did you. Pat, did you send him that?\n"
                                    "Sam Chen [02:02] I did, sure.\n"
                                    "Alex Rivera [06:10] We need to configure badge zones and cameras with SSO."
                                ),
                            }
                        ],
                    }
                ],
            },
        ],
    }
    atoms = [
        _mk_scope("Badge access is in scope.", page=0, filename=filename),
        _mk_scope(
            "Alex Rivera [01:59] Did you. Pat, did you send him that?",
            page=1,
            filename=filename,
        ),
    ]
    atoms[0].value = {"kind": "bullet", "depth": 1, "text": atoms[0].raw_text}
    atoms[0].source_refs[0].locator = {"page": 0, "block_kind": "bullet_list"}

    out = rewrite_hybrid_pdf_atoms(
        atoms=atoms,
        structured_doc=structured,
        filename=filename,
        project_id="proj_demo",
        artifact_id="art_demo",
        parser_version="test_v1",
    )
    send_him = [
        a
        for a in out
        if "send him that" in (a.raw_text or "").lower()
        or "send him that" in str((a.value or {}).get("text") or "").lower()
    ]
    assert send_him, "question turn must survive"
    # Exactly one typed atom for that utterance (not action_item + open_question).
    assert len(send_him) == 1



def test_rewrite_strips_sticky_key_decisions_chrome():
    filename = "Acme_Meeting_Summary_and_Full_Transcript.pdf"
    structured = {
        "document": {"title": "Meeting Summary and Full Transcript"},
        "pages": [
            {
                "page": 0,
                "sections": [
                    {
                        "heading": "Executive Summary",
                        "blocks": [
                            {
                                "kind": "bullet_list",
                                "items": [
                                    {"text": "Badge access and camera configuration are primary focus areas."},
                                ],
                            }
                        ],
                    }
                ],
            },
            {
                "page": 1,
                "sections": [
                    {
                        # Sticky leftover from summary page — real PDF text-rich path.
                        "heading": "Key Decisions",
                        "blocks": [
                            {
                                "kind": "paragraph",
                                "text": (
                                    "Full Transcript Alex Rivera [00:04] See, Pat should be joining us shortly. "
                                    "Sam Chen [00:20] Hey, how you doing? Been a while. "
                                    "Alex Rivera [06:10] We need to configure badge zones and cameras with SSO."
                                ),
                            }
                        ],
                    }
                ],
            },
        ],
    }
    atoms = [
        _mk_scope(
            "Badge access and camera configuration are primary focus areas.",
            page=0,
            filename=filename,
        ),
        _mk_scope(
            "Full Transcript Alex Rivera [00:04] See, Pat should be joining us shortly.",
            page=1,
            filename=filename,
        ),
    ]
    atoms[0].value = {"kind": "bullet", "depth": 1, "text": atoms[0].raw_text}
    atoms[0].source_refs[0].locator = {"page": 0, "block_kind": "bullet_list"}

    out = rewrite_hybrid_pdf_atoms(
        atoms=atoms,
        structured_doc=structured,
        filename=filename,
        project_id="proj_demo",
        artifact_id="art_demo",
        parser_version="test_v1",
    )
    texts = [a.raw_text or "" for a in out]
    assert not any(t.startswith("Key Decisions") for t in texts)
    assert not any(t.startswith("Full Transcript") for t in texts)
    assert not any("Full Transcript Alex" in t for t in texts)
    meta = [
        a
        for a in out
        if a.atom_type == AtomType.deal_metadata
        and isinstance(a.value, dict)
        and a.value.get("kind") == CONVERSATION_META_KIND
    ]
    assert meta
    speakers = [
        (a.value or {}).get("speaker")
        for a in out
        if isinstance(a.value, dict) and a.value.get("speaker")
    ]
    assert "Alex Rivera" in speakers
    assert not any(s and ("Decision" in s or "Transcript" in s or "\n" in s) for s in speakers)


def test_no_gecko_hardcodes_in_hybrid_module():
    """Guard: hybrid module must stay deal-agnostic."""
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "app" / "core" / "hybrid_summary_transcript.py"
    text = src.read_text(encoding="utf-8").lower()
    for banned in (
        "gecko",
        "010058",
        "c2912e57",
        "ubiquiti",
        "vander-plaats",
        "daniel peterson",
        "trent torrence",
    ):
        assert banned not in text, f"deal-specific string leaked: {banned}"


def test_how_are_you_good_reply_adjacency():
    """'How are you?' → 'Good.' must stamp in_reply_to so heads never see an orphan."""
    from app.core.hybrid_summary_transcript import (
        collapse_greeting_clusters,
        is_head_excluded_atom,
        stamp_conversation_reply_adjacency,
    )

    filename = "Acme_Meeting_Summary_and_Full_Transcript.pdf"
    structured = {
        "document": {"title": "Meeting Summary and Full Transcript"},
        "pages": [
            {
                "page": 0,
                "sections": [
                    {
                        "heading": "Executive Summary",
                        "blocks": [
                            {
                                "kind": "bullet_list",
                                "items": [{"text": "Badge access is in scope."}],
                            }
                        ],
                    }
                ],
            },
            {
                "page": 1,
                "sections": [
                    {
                        "heading": "Full Transcript",
                        "blocks": [
                            {
                                "kind": "paragraph",
                                "text": (
                                    "Alex Rivera [00:48] We'll touch on this after the call. "
                                    "Well, hey, Patrick. How are you?\n"
                                    "Sam Chen [00:56] Good.\n"
                                    "Alex Rivera [00:56] Looks like we're just waiting for Eddie. "
                                    "Long time no talk.\n"
                                    "Alex Rivera [06:10] We need to configure badge zones and cameras with SSO."
                                ),
                            }
                        ],
                    }
                ],
            },
        ],
    }
    atoms = [
        _mk_scope("Badge access is in scope.", page=0, filename=filename),
        _mk_scope(
            "Alex Rivera [00:48] How are you? Sam Chen [00:56] Good.",
            page=1,
            filename=filename,
        ),
    ]
    atoms[0].value = {"kind": "bullet", "depth": 1, "text": atoms[0].raw_text}
    atoms[0].source_refs[0].locator = {"page": 0, "block_kind": "bullet_list"}

    out = rewrite_hybrid_pdf_atoms(
        atoms=atoms,
        structured_doc=structured,
        filename=filename,
        project_id="proj_demo",
        artifact_id="art_demo",
        parser_version="test_v1",
    )

    good = [
        a
        for a in out
        if a.atom_type == AtomType.deal_metadata
        and isinstance(a.value, dict)
        and a.value.get("kind") == CONVERSATION_META_KIND
        and re.search(r"\bGood\.?\b", a.raw_text or "", re.I)
    ]
    assert good, "acknowledgment 'Good.' must survive as conversation_meta"
    ack = good[0]
    assert is_head_excluded_atom(ack)
    assert ack.value.get("non_deal") is True
    assert ack.value.get("head_exclude") is True
    assert ack.value.get("role") in {"acknowledgment", "greeting_cluster"}
    # Reply-to context — what "Good." is answering.
    prev_text = (
        (ack.value.get("in_reply_to") or {}).get("text")
        or ack.value.get("previous_text")
        or ""
    )
    assert "how are you" in prev_text.lower(), f"expected How are you? context, got {prev_text!r}"
    assert ack.value.get("previous_speaker") or (ack.value.get("in_reply_to") or {}).get(
        "speaker"
    ), "previous speaker required"
    assert _is_protected_email_atom(ack)

    # Optional collapse: greeting → ack becomes one cluster with reply-to preserved.
    pair = [
        a
        for a in out
        if a.atom_type == AtomType.deal_metadata
        and isinstance(a.value, dict)
        and a.value.get("kind") == CONVERSATION_META_KIND
        and a.value.get("role") in {"greeting", "filler", "acknowledgment"}
        and (a.source_refs[0].locator or {}).get("page") == 1
    ]
    if len(pair) >= 2:
        stamp_conversation_reply_adjacency(pair)
        collapsed, n = collapse_greeting_clusters(list(pair))
        if n:
            clusters = [
                a
                for a in collapsed
                if isinstance(a.value, dict) and a.value.get("role") == "greeting_cluster"
            ]
            assert clusters
            assert clusters[0].value.get("in_reply_to") or clusters[0].value.get("previous_text")


def test_stitch_cross_page_speaker_turns_unit():
    """Page-break continuation without a new [mm:ss] merges into prior turn."""
    from app.core.hybrid_summary_transcript import (
        SpeakerTurn,
        stitch_cross_page_speaker_turns,
    )

    page_turns = [
        (
            1,
            [
                SpeakerTurn(
                    speaker="Alex Rivera",
                    timestamp="02:10",
                    text="",
                    char_start=0,
                    char_end=20,
                )
            ],
        ),
        (
            2,
            [
                SpeakerTurn(
                    speaker=None,
                    timestamp=None,
                    text="Call it, actually.",
                    char_start=0,
                    char_end=18,
                ),
                SpeakerTurn(
                    speaker="Sam Chen",
                    timestamp="02:11",
                    text="All right, next topic.",
                    char_start=19,
                    char_end=50,
                ),
            ],
        ),
    ]
    out = stitch_cross_page_speaker_turns(page_turns)
    assert len(out) == 2
    assert out[0].turn.speaker == "Alex Rivera"
    assert out[0].turn.timestamp == "02:10"
    assert "Call it, actually." in (out[0].turn.text or "")
    assert out[0].page_end == 2
    assert out[1].turn.speaker == "Sam Chen"
    assert out[1].page_end is None


def test_rewrite_stitches_cross_page_transcript_continuation():
    """Hybrid rewrite must emit one atom for a turn split across PDF pages."""
    filename = "Acme_Meeting_Summary_and_Full_Transcript.pdf"
    structured = {
        "document": {"title": "Meeting Summary and Full Transcript"},
        "pages": [
            {
                "page": 0,
                "sections": [
                    {
                        "heading": "Executive Summary",
                        "blocks": [
                            {"kind": "bullet_list", "items": [{"text": "SSO is in scope."}]}
                        ],
                    }
                ],
            },
            {
                "page": 1,
                "sections": [
                    {
                        "heading": "Full Transcript",
                        "blocks": [
                            {
                                "kind": "paragraph",
                                "text": (
                                    "Alex Rivera [02:05] Before we jump in I want to\n"
                                    "Alex Rivera [02:10]"
                                ),
                            }
                        ],
                    }
                ],
            },
            {
                "page": 2,
                "sections": [
                    {
                        "heading": "Full Transcript",
                        "blocks": [
                            {
                                "kind": "paragraph",
                                "text": (
                                    "Call it, actually.\n"
                                    "Sam Chen [02:11] All right, Jacob, while we're waiting."
                                ),
                            }
                        ],
                    }
                ],
            },
        ],
    }
    atoms = [
        _mk_scope("SSO is in scope.", page=0, filename=filename),
        _mk_scope("Alex Rivera [02:10]", page=1, filename=filename),
        _mk_scope("Call it, actually.", page=2, filename=filename),
    ]

    out = rewrite_hybrid_pdf_atoms(
        atoms=atoms,
        structured_doc=structured,
        filename=filename,
        project_id="proj_demo",
        artifact_id="art_demo",
        parser_version="test_v1",
    )

    # Orphan continuation must not survive as its own atom.
    orphans = [
        a
        for a in out
        if (a.raw_text or "").strip() == "Call it, actually."
        or (isinstance(a.value, dict) and (a.value.get("text") or "").strip() == "Call it, actually.")
    ]
    assert not orphans, f"continuation must be stitched, found orphan: {orphans}"

    stitched = [
        a
        for a in out
        if "Call it, actually." in (a.raw_text or "")
        and "Alex Rivera" in (a.raw_text or "")
    ]
    assert stitched, "expected one stitched Alex Rivera turn spanning pages"
    loc = stitched[0].source_refs[0].locator or {}
    assert loc.get("cross_page_stitched") is True
    assert loc.get("page_end") == 2
    assert loc.get("page") == 1


def test_pricing_and_contact_questions_are_deal_not_filler():
    """Short commercial questions must never fall to filler/head_exclude."""
    assert classify_transcript_turn_role("How much is it for?") == "deal"
    assert classify_transcript_turn_role("What is the price?") == "deal"
    assert (
        classify_transcript_turn_role(
            "Now that Morgan's not there, who should I be sending that to for signature?"
        )
        == "deal"
    )
    assert classify_transcript_turn_role("It's just time and materials,") == "deal"
    assert classify_transcript_turn_role("It's available to Tom. Tom Amble.") == "deal"
    # Social wellness stays greeting / acknowledgment.
    assert classify_transcript_turn_role("How are you?") == "greeting"
    assert classify_transcript_turn_role("Good.") == "acknowledgment"


def test_pricing_qa_reply_adjacency_and_head_eligible():
    """How much? → T&M answer must carry in_reply_to and stay head-eligible."""
    filename = "Acme_Meeting_Summary_and_Full_Transcript.pdf"
    structured = {
        "document": {"title": "Meeting Summary and Full Transcript"},
        "pages": [
            {
                "page": 0,
                "sections": [
                    {
                        "heading": "Executive Summary",
                        "blocks": [
                            {
                                "kind": "bullet_list",
                                "items": [{"text": "Support agreement is open."}],
                            }
                        ],
                    }
                ],
            },
            {
                "page": 1,
                "sections": [
                    {
                        "heading": "Full Transcript",
                        "blocks": [
                            {
                                "kind": "paragraph",
                                "text": (
                                    "Alex Rivera [02:50] Before we wrap, who should I be "
                                    "sending that to for signature?\n"
                                    "Sam Chen [02:53] How much is it for?\n"
                                    "Alex Rivera [02:55] It's just time and materials, "
                                    "so I'd be happy to put an agreement together.\n"
                                    "Sam Chen [03:02] It's available to Tom. Tom Amble.\n"
                                    "Alex Rivera [03:05] Okay."
                                ),
                            }
                        ],
                    }
                ],
            },
        ],
    }
    atoms = [
        _mk_scope("Support agreement is open.", page=0, filename=filename),
        _mk_scope(
            "Sam Chen [02:53] How much is it for?",
            page=1,
            filename=filename,
        ),
    ]
    atoms[0].value = {"kind": "bullet", "depth": 1, "text": atoms[0].raw_text}
    atoms[0].source_refs[0].locator = {"page": 0, "block_kind": "bullet_list"}

    out = rewrite_hybrid_pdf_atoms(
        atoms=atoms,
        structured_doc=structured,
        filename=filename,
        project_id="proj_demo",
        artifact_id="art_demo",
        parser_version="test_v1",
    )

    how_much = [
        a
        for a in out
        if re.search(r"how much is it for", a.raw_text or "", re.I)
    ]
    assert how_much, "pricing question must survive"
    hm = how_much[0]
    assert not is_head_excluded_atom(hm), "pricing question must be head-eligible"
    assert not (
        isinstance(hm.value, dict)
        and hm.value.get("kind") == CONVERSATION_META_KIND
        and hm.value.get("role") == "filler"
    ), "must not be conversation_meta filler"

    tm = [
        a
        for a in out
        if re.search(r"time and materials", a.raw_text or "", re.I)
    ]
    assert tm, "T&M answer must survive"
    ans = tm[0]
    assert not is_head_excluded_atom(ans), "T&M answer must be head-eligible"
    prev = (
        (ans.value.get("in_reply_to") or {}).get("text")
        if isinstance(ans.value, dict)
        else ""
    ) or (ans.value.get("previous_text") if isinstance(ans.value, dict) else "") or ""
    assert "how much" in prev.lower(), f"T&M must reply to pricing Q, got {prev!r}"

    contact = [
        a
        for a in out
        if re.search(r"tom amble", a.raw_text or "", re.I)
    ]
    assert contact, "contact answer must survive as deal substance"
    ct = contact[0]
    assert not is_head_excluded_atom(ct)
    assert not (
        isinstance(ct.value, dict)
        and ct.value.get("kind") == CONVERSATION_META_KIND
        and ct.value.get("role") == "filler"
    )
    cprev = (
        (ct.value.get("in_reply_to") or {}).get("text")
        if isinstance(ct.value, dict)
        else ""
    ) or ""
    assert "signature" in cprev.lower() or "sending" in cprev.lower(), (
        f"contact answer must reply to signature Q, got {cprev!r}"
    )

    okay = [
        a
        for a in out
        if isinstance(a.value, dict)
        and a.value.get("kind") == CONVERSATION_META_KIND
        and re.search(r"\bOkay\.?\b", a.raw_text or "", re.I)
    ]
    assert okay
    assert is_head_excluded_atom(okay[0])
    assert okay[0].value.get("in_reply_to") or okay[0].value.get("previous_text")


def test_stitch_incomplete_same_speaker_across_page():
    """Mid-sentence comma cut + same-speaker re-stamp on next page merges."""
    from app.core.hybrid_summary_transcript import (
        SpeakerTurn,
        stitch_cross_page_speaker_turns,
    )

    page_turns = [
        (
            2,
            [
                SpeakerTurn(
                    speaker="Alex Rivera",
                    timestamp="02:55",
                    text="It's just time and materials,",
                    char_start=0,
                    char_end=40,
                )
            ],
        ),
        (
            3,
            [
                SpeakerTurn(
                    speaker="Alex Rivera",
                    timestamp="02:55",
                    text="so I'd be happy to put an agreement together.",
                    char_start=0,
                    char_end=50,
                ),
                SpeakerTurn(
                    speaker="Sam Chen",
                    timestamp="03:02",
                    text="It's available to Tom.",
                    char_start=51,
                    char_end=80,
                ),
            ],
        ),
    ]
    out = stitch_cross_page_speaker_turns(page_turns)
    assert len(out) == 2
    assert out[0].turn.speaker == "Alex Rivera"
    assert "time and materials" in (out[0].turn.text or "")
    assert "agreement together" in (out[0].turn.text or "")
    assert out[0].page_end == 3
    assert out[1].turn.speaker == "Sam Chen"

