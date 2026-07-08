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
