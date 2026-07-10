"""Universal hybrid summary+transcript routing and conversation_meta tests.

Fixtures use generic names/vendors — never a real deal id, customer, or
person from production data.
"""

from __future__ import annotations

from app.core.hybrid_summary_transcript import (
    CONVERSATION_META_KIND,
    classify_transcript_turn_role,
    detect_hybrid_summary_transcript,
    is_non_deal_meta_atom,
    rewrite_hybrid_pdf_atoms,
    split_speaker_timestamp_turns,
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
    assert classify_transcript_turn_role("Yeah.") == "filler"
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


def test_classify_co_founder_space_and_signoff():
    assert (
        classify_transcript_turn_role(
            "Yep. I can start. I'm Alex Rivera. I am one of the co founders of Acme."
        )
        == "intro"
    )
    assert (
        classify_transcript_turn_role("All right, I'll sit on my end. Thank you guys.")
        == "filler"
    )


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
