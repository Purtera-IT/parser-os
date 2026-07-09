from __future__ import annotations

from app.core.atom_substance_gate import (
    apply_substance_gate,
    drop_contextless_stakeholders,
    drop_nonsubstantive_fragments,
)
from app.core.schemas import (
    AtomType,
    AuthorityClass,
    EvidenceAtom,
    ReviewStatus,
    SourceRef,
)


def _mk(atom_type: str, text: str, *, value=None, entity_keys=None) -> EvidenceAtom:
    return EvidenceAtom(
        id="atm_" + str(abs(hash((atom_type, text))))[:12],
        project_id="p",
        artifact_id="art_t",
        atom_type=AtomType(atom_type),
        raw_text=text,
        normalized_text=text.lower(),
        value=value if value is not None else {"text": text},
        entity_keys=entity_keys or [],
        source_refs=[
            SourceRef(
                id="src_t",
                artifact_id="art_t",
                artifact_type="txt",
                filename="t.txt",
                locator={},
                extraction_method="t",
                parser_version="t",
            )
        ],
        authority_class=AuthorityClass.customer_current_authored,
        confidence=0.6,
        review_status=ReviewStatus.needs_review,
        review_flags=[],
        parser_version="t",
    )


def _texts(atoms):
    return [a.raw_text for a in atoms]


# ── contextless stakeholders (bare names / salutations / sign-offs) ──

def test_bare_name_stakeholders_dropped():
    atoms = [
        _mk("stakeholder", "Tom Amble."),
        _mk("stakeholder", "Eddie,"),
        _mk("stakeholder", "Patrick Kelly"),
    ]
    kept, dropped = drop_contextless_stakeholders(atoms)
    assert kept == []
    assert len(dropped) == 3


def test_stakeholder_with_role_kept():
    atoms = [_mk("stakeholder", "Renee Watkins, VP Engineering")]
    kept, dropped = drop_contextless_stakeholders(atoms)
    assert len(kept) == 1
    assert dropped == []


def test_stakeholder_with_email_kept():
    atoms = [_mk("stakeholder", "Jordan Ames (jordan@acme.com)")]
    kept, dropped = drop_contextless_stakeholders(atoms)
    assert len(kept) == 1 and dropped == []


def test_stakeholder_with_approval_relation_kept():
    atoms = [_mk("stakeholder", "Approved by Dana Cole")]
    kept, dropped = drop_contextless_stakeholders(atoms)
    assert len(kept) == 1 and dropped == []


def test_stakeholder_with_structured_role_field_kept():
    atoms = [
        _mk(
            "stakeholder",
            "Sara Chen",
            value={"text": "Sara Chen", "role": "Project Manager"},
        )
    ]
    kept, dropped = drop_contextless_stakeholders(atoms)
    assert len(kept) == 1 and dropped == []


def test_non_stakeholder_atoms_never_examined():
    atoms = [_mk("scope_item", "Patrick Kelly")]  # same bare-name text, not a stakeholder
    kept, dropped = drop_contextless_stakeholders(atoms)
    assert len(kept) == 1 and dropped == []


# ── non-substantive backchannel filler ──

def test_backchannel_filler_dropped():
    atoms = [
        _mk("scope_item", "Yeah."),
        _mk("scope_item", "Okay, sure."),
        _mk("scope_item", "Jacob Vander-Plaats [03:05] Yeah."),
    ]
    kept, dropped = drop_nonsubstantive_fragments(atoms)
    assert kept == []
    assert len(dropped) == 3


def test_real_scope_kept_even_when_short():
    atoms = [
        _mk("scope_item", "Okta integration"),
        _mk("scope_item", "Install 12 cameras"),
        _mk("scope_item", "Yeah, not too bad. Can't complain."),  # has real words
    ]
    kept, dropped = drop_nonsubstantive_fragments(atoms)
    assert len(kept) == 3
    assert dropped == []


def test_exclusion_and_quantity_types_not_filler_gated():
    atoms = [_mk("exclusion", "No."), _mk("quantity", "12 cameras")]
    kept, dropped = drop_nonsubstantive_fragments(atoms)
    assert len(kept) == 2 and dropped == []


def test_apply_substance_gate_combined():
    atoms = [
        _mk("stakeholder", "Tom Amble."),                    # drop
        _mk("stakeholder", "Renee Watkins, Director"),       # keep
        _mk("scope_item", "Yeah."),                          # drop
        _mk("scope_item", "Okta integration"),               # keep
    ]
    kept, dropped = apply_substance_gate(atoms)
    assert set(_texts(kept)) == {"Renee Watkins, Director", "Okta integration"}
    assert set(_texts(dropped)) == {"Tom Amble.", "Yeah."}


# ── transcript-PDF over-fragmentation: bare speaker-header atoms ──

def test_speaker_label_only_atoms_dropped():
    # When a transcript is delivered as a PDF, the color-driven segmenter emits
    # each speaker/timestamp header ("Daniel Peterson [00:48]") as its own
    # block, which the fail-open PDF atomizer turns into a standalone
    # scope_item. Those bare labels are chrome (speaker is already section_path
    # context) and must be gated.
    atoms = [
        _mk("scope_item", "Daniel Peterson [00:48]"),
        _mk("scope_item", "Jacob Vander-Plaats [07:16]"),
        _mk("scope_item", "Trent Torrence [16:24]"),
    ]
    kept, dropped = drop_nonsubstantive_fragments(atoms)
    assert kept == []
    assert len(dropped) == 3


def test_speaker_label_with_real_utterance_kept():
    # A speaker header glued to a substantive utterance is NOT a bare label —
    # it carries deal content and stays.
    atoms = [
        _mk(
            "scope_item",
            "Jacob Vander-Plaats [07:16] What we have on site are 12 cameras "
            "and six or seven badge readers.",
        )
    ]
    kept, dropped = drop_nonsubstantive_fragments(atoms)
    assert len(kept) == 1 and dropped == []


# ── conversational backchannel the bare-filler set alone would miss ──

def test_social_backchannel_dropped():
    atoms = [
        _mk("scope_item", "I see."),
        _mk("scope_item", "Thank you."),
        _mk("scope_item", "Thanks, guys."),
        _mk("scope_item", "Okay, sounds good."),
    ]
    kept, dropped = drop_nonsubstantive_fragments(atoms)
    assert kept == []
    assert len(dropped) == 4


def test_short_utterance_with_deal_word_kept():
    # A single real deal noun/verb keeps even a very short utterance — the gate
    # only fires when EVERY content token is non-substantive.
    atoms = [
        _mk("scope_item", "I see the switch."),
        _mk("scope_item", "Sounds like the firewall."),
        _mk("scope_item", "You handle Okta."),
    ]
    kept, dropped = drop_nonsubstantive_fragments(atoms)
    assert len(kept) == 3 and dropped == []


# ── section headers ──

def test_section_header_dropped():
    from app.core.atom_substance_gate import drop_section_headers

    atoms = [
        _mk("scope_item", "Meeting Summary and Full Transcript Executive Summary"),
        _mk("scope_item", "Executive Summary"),
        _mk("scope_item", "Full Transcript: Daniel Peterson [00:04]"),
    ]
    kept, dropped = drop_section_headers(atoms)
    assert kept == []
    assert len(dropped) == 3


def test_section_header_with_substance_kept():
    from app.core.atom_substance_gate import drop_section_headers

    atoms = [_mk("scope_item", "Okta integration is a significant requirement.")]
    kept, dropped = drop_section_headers(atoms)
    assert len(kept) == 1 and dropped == []


# ── email non-scope ──

def _mk_email(text, **value_kw):
    val = {"text": text, "kind": "email_body_line", **value_kw}
    return _mk("scope_item", text, value=val)


def test_email_label_only_dropped():
    from app.core.atom_substance_gate import drop_email_non_scope

    atoms = [
        _mk_email("Customer specifically said:"),
        _mk_email("By the end of the meeting customer clarified:"),
    ]
    kept, dropped = drop_email_non_scope(atoms)
    assert kept == []
    assert len(dropped) == 2


def test_email_pleasantry_scope_dropped_context_kept():
    from app.core.atom_substance_gate import drop_email_non_scope

    # Legacy / mis-typed scope_item pleasantry still drops.
    scope_pleasantry = _mk_email(
        "Appreciate you hopping on in such short notice. Attached is a summary."
    )
    # Intentional communication atoms must survive the gate.
    addressee = _mk(
        "deal_metadata",
        "Eddie,",
        value={"text": "Eddie,", "kind": "email_addressee", "role": "to_greeting"},
    )
    context = _mk(
        "deal_metadata",
        "Appreciate you hopping on in such short notice. Attached is a summary.",
        value={
            "text": "Appreciate you hopping on in such short notice. Attached is a summary.",
            "kind": "email_body_context",
            "role": "intro",
        },
    )
    kept, dropped = drop_email_non_scope([scope_pleasantry, addressee, context])
    assert scope_pleasantry in dropped
    assert addressee in kept
    assert context in kept


def test_email_include_list_item_kept():
    from app.core.atom_substance_gate import drop_email_non_scope

    atoms = [
        _mk_email("Okta integration", list_section="include"),
        _mk_email("Badge/access control setup", list_section="include"),
    ]
    kept, dropped = drop_email_non_scope(atoms)
    assert len(kept) == 2 and dropped == []


def test_email_header_metadata_dropped():
    from app.core.atom_substance_gate import drop_email_non_scope

    atoms = [
        _mk(
            "deal_metadata",
            "From: a@b.com | To: c@d.com | Subject: Test | Date: 2026-01-01",
            value={"kind": "email_header", "from": "a@b.com"},
        ),
    ]
    kept, dropped = drop_email_non_scope(atoms)
    assert kept == [] and len(dropped) == 1


# ── transcript conversational ──

def _mk_transcript(text, page=1, **loc_kw):
    atom = _mk("scope_item", text)
    loc = {"page": page, "block_kind": "paragraph", **loc_kw}
    atom.source_refs[0].locator = loc
    return atom


def test_transcript_greeting_dropped():
    from app.core.atom_substance_gate import drop_transcript_conversational

    atoms = [
        _mk_transcript("I know. It has been a while. Hope life's been treating you well.", page=1),
        _mk_transcript("Hey, how you doing? Been a while.", page=1),
        _mk_transcript("I'm not hearing very well. Can you repeat one more time?", page=6),
    ]
    kept, dropped = drop_transcript_conversational(atoms)
    assert kept == []
    assert len(dropped) == 3


def test_transcript_exec_summary_bullet_kept():
    from app.core.atom_substance_gate import drop_transcript_conversational

    atom = _mk("scope_item", "Okta integration is considered a significant requirement.")
    atom.source_refs[0].locator = {"page": 0, "block_kind": "bullet_list"}
    atom.value = {"kind": "bullet", "depth": 1}
    kept, dropped = drop_transcript_conversational([atom])
    assert len(kept) == 1 and dropped == []


def test_transcript_substantive_turn_kept():
    from app.core.atom_substance_gate import drop_transcript_conversational

    atoms = [
        _mk_transcript(
            "We need to set up badge zones and guest VLAN with Okta integration.",
            page=4,
        ),
    ]
    kept, dropped = drop_transcript_conversational(atoms)
    assert len(kept) == 1 and dropped == []


# ── risk fragments ──

def test_risk_mid_sentence_clip_dropped():
    from app.core.atom_substance_gate import drop_risk_fragments

    atoms = [
        _mk("risk", "consider it but we, if we can't we might."),
        _mk("risk", "there's going to be a lot more hesitation moving with any other option."),
    ]
    kept, dropped = drop_risk_fragments(atoms)
    assert kept == []
    assert len(dropped) == 2


def test_risk_complete_clause_kept():
    from app.core.atom_substance_gate import drop_risk_fragments

    atoms = [
        _mk(
            "risk",
            "Network build-out is excluded from the primary scope unless required later.",
            value={"kind": "bullet", "depth": 1},
        ),
        _mk(
            "risk",
            "If Okta integration cannot be completed that is a hard requirement "
            "and we would need to explore alternative options before proceeding.",
        ),
    ]
    kept, dropped = drop_risk_fragments(atoms)
    assert len(kept) == 2 and dropped == []


# ── ambiguous quantity collapse ──

def test_ambiguous_user_quantities_collapsed():
    from app.core.atom_substance_gate import collapse_ambiguous_user_quantities

    def _qty(q, noun="users", loc_page=5):
        a = _mk("quantity", f"may be simple since our {q} people.")
        a.value = {
            "kind": "quantity",
            "quantity": q,
            "noun": noun,
            "context": f"may be simple since our {q} people.",
        }
        a.source_refs[0].locator = {"page": loc_page, "block_id": "blk_same"}
        return a

    atoms = [_qty(20), _qty(50)]
    kept, dropped = collapse_ambiguous_user_quantities(atoms)
    assert len(kept) == 1
    assert kept[0].value["quantity"] == 50
    assert len(dropped) == 1


def test_different_noun_quantities_not_collapsed():
    from app.core.atom_substance_gate import collapse_ambiguous_user_quantities

    def _qty(q, noun):
        a = _mk("quantity", f"{q} {noun}")
        a.value = {"kind": "quantity", "quantity": q, "noun": noun}
        a.source_refs[0].locator = {"page": 4, "block_id": "blk_x"}
        return a

    atoms = [_qty(12, "cameras"), _qty(7, "badge readers")]
    kept, dropped = collapse_ambiguous_user_quantities(atoms)
    assert len(kept) == 2 and dropped == []
