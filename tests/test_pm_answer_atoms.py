"""PM answers become first-class evidence, not a discarded string.

The product claim these pin: what a PM types when answering an open question is
usually deal truth that exists in no document, so it must reach the SOW *this*
compile and be gold for the heads *next* compile.
"""
from __future__ import annotations

from app.core.authority import authority_rank
from app.core.pm_answer_atoms import (
    build_claim,
    is_substantive_answer,
    pm_answer_to_atom,
    pm_answers_to_atoms,
)
from app.core.schemas import ArtifactType, AtomType, AuthorityClass, ReviewStatus


def _event(**over):
    base = {
        "action": "answered",
        "deal_id": "deal-1",
        "rule_id": "mode.av_install.drywall_ownership",
        "question_text": "Who owns drywall patching after cable conceal?",
        "edited_text": "Customer's GC patches and paints; PurTera cuts only.",
        "created_at": "2026-08-03T12:00:00Z",
        "actor": "pm@purtera.com",
    }
    base.update(over)
    return base


def test_answer_becomes_an_atom_the_sow_can_use():
    atom = pm_answer_to_atom(
        project_id="p1",
        question="Who owns drywall patching after cable conceal?",
        answer="Customer's GC patches and paints; PurTera cuts only.",
        rule_id="mode.av_install.drywall_ownership",
    )
    assert atom is not None
    assert atom.atom_type is AtomType.decision
    assert "patches and paints" in atom.raw_text
    # Self-contained: the SOW reads the atom without the card around it.
    assert "drywall patching" in atom.raw_text


def test_pm_answer_outranks_documents_but_not_signed_scope():
    """This is the whole point — a PM's answer must SETTLE a conflict with a
    stale doc line, not become one more competing claim."""
    pm = authority_rank(AuthorityClass.pm_confirmed)
    assert pm > authority_rank(AuthorityClass.customer_current_authored)
    assert pm > authority_rank(AuthorityClass.vendor_quote)
    assert pm > authority_rank(AuthorityClass.machine_extractor)
    assert pm < authority_rank(AuthorityClass.contractual_scope)


def test_atom_carries_provenance_back_to_the_card():
    atom = pm_answer_to_atom(
        project_id="p1",
        question="Who signs acceptance?",
        answer="The site lead, Dana Whitfield, signs per-site acceptance.",
        rule_id="mode.network_edge_install.acceptance_signer",
        answered_at="2026-08-03T12:00:00Z",
        actor="pm@purtera.com",
    )
    assert atom is not None
    src = atom.source_refs[0]
    assert src.artifact_type is ArtifactType.pm_answer
    assert src.locator["rule_id"] == "mode.network_edge_install.acceptance_signer"
    assert src.locator["actor"] == "pm@purtera.com"
    assert src.locator["answered_at"] == "2026-08-03T12:00:00Z"


def test_human_authored_so_it_does_not_re_enter_review():
    atom = pm_answer_to_atom(
        project_id="p1",
        question="Who signs acceptance?",
        answer="The site lead signs per-site acceptance.",
    )
    assert atom is not None
    assert atom.review_status is ReviewStatus.approved
    assert "pm_authored" in atom.review_flags
    assert atom.confidence >= 0.9


def test_contentless_answers_never_enter_evidence():
    # These still settle the card via the feedback ledger; they just carry no
    # facts, and an atom is read stripped of its question.
    for junk in ("yes", "no", "n/a", "TBD", "  ", "-", "?"):
        assert not is_substantive_answer(junk)
        assert (
            pm_answer_to_atom(project_id="p1", question="Anything?", answer=junk) is None
        )


def test_claim_joins_verbatim_and_never_paraphrases():
    claim = build_claim("Who patches the drywall?", "The customer does.")
    assert claim == "Who patches the drywall? PM answer: The customer does."
    # Answer text survives byte-for-byte — no rewriting into fluent prose.
    assert "The customer does." in claim


def test_ids_are_deterministic_so_recompiles_do_not_duplicate():
    kw = dict(
        project_id="p1",
        question="Who signs acceptance?",
        answer="The site lead signs.",
        rule_id="r1",
    )
    assert pm_answer_to_atom(**kw).id == pm_answer_to_atom(**kw).id


def test_re_answering_replaces_rather_than_contradicts():
    events = [
        _event(edited_text="PurTera patches."),
        _event(edited_text="Correction: the customer's GC patches."),
    ]
    atoms = pm_answers_to_atoms(events, project_id="p1")
    assert len(atoms) == 1
    assert "customer's GC patches" in atoms[0].raw_text


def test_only_answered_events_become_evidence():
    events = [
        _event(action="dismiss"),
        _event(action="wrong_for_project"),
        _event(action="add", edited_text="Confirm the after-hours rate?"),
        _event(),
    ]
    atoms = pm_answers_to_atoms(events, project_id="p1")
    assert len(atoms) == 1
    assert "patches and paints" in atoms[0].raw_text


def _doc_atom(authority: AuthorityClass, text: str):
    """A claim we merely READ out of a customer document."""
    from app.core.schemas import ArtifactType as AT
    from app.core.schemas import EvidenceAtom, SourceRef

    return EvidenceAtom(
        id="atm_doc_1",
        atom_id="atm_doc_1",
        project_id="p1",
        artifact_id="art_doc",
        atom_type=AtomType.scope_item,
        raw_text=text,
        normalized_text=text.lower(),
        source_refs=[
            SourceRef(
                id="src_doc_1",
                artifact_id="art_doc",
                artifact_type=AT.docx,
                filename="scope.docx",
                locator={"page": 4, "sender": "facilities@customer.com"},
                extraction_method="docx_rules",
                parser_version="docx_v1",
            )
        ],
        authority_class=authority,
        confidence=0.9,
        review_status=ReviewStatus.auto_accepted,
        parser_version="docx_v1",
    )


def test_pm_answer_governs_over_the_document_it_contradicts():
    """The product claim, end to end, through the real authority scorer.

    A doc says PurTera patches the drywall; the PM answers that the customer's
    GC does. The PM must GOVERN — otherwise answering the question changes
    nothing about the SOW, which is the whole point of asking.
    """
    from app.core.authority import score_authority

    doc = _doc_atom(
        AuthorityClass.customer_current_authored,
        "Contractor to patch and paint all wall penetrations.",
    )
    pm = pm_answer_to_atom(
        project_id="p1",
        question="Who owns drywall patching after cable conceal?",
        answer="Customer's GC patches and paints; PurTera cuts only.",
        rule_id="mode.av_install.drywall_ownership",
    )
    assert pm is not None
    both = [doc, pm]
    assert (
        score_authority(pm, both).final_score > score_authority(doc, both).final_score
    ), "a PM's explicit answer must beat a line we merely read"


def test_pm_answer_never_overrides_signed_contractual_scope():
    """The guardrail on the same claim: a PM note does not rewrite a contract."""
    from app.core.authority import score_authority

    contract = _doc_atom(
        AuthorityClass.contractual_scope,
        "Vendor shall patch and paint all penetrations per Exhibit B.",
    )
    pm = pm_answer_to_atom(
        project_id="p1",
        question="Who owns drywall patching?",
        answer="I think the customer handles it.",
        rule_id="mode.av_install.drywall_ownership",
    )
    assert pm is not None
    both = [contract, pm]
    assert (
        score_authority(contract, both).final_score
        > score_authority(pm, both).final_score
    )


def test_pm_answer_can_govern_a_scope_inclusion_packet():
    """Authority rank alone is not enough — the packetizer keeps its own
    'primary authority' allowlists, and a class missing from them is silently
    demoted to non-governing. A PM answer that cannot govern the scope packet
    would rank highest and still change nothing."""
    from app.core.packetizer import _can_act_as_scope_inclusion_governor

    pm = pm_answer_to_atom(
        project_id="p1",
        question="Are the Canada sites in this phase?",
        answer="Canada is deferred to phase 2; US sites only in this wave.",
        rule_id="mode.network_edge_install.phase_site_exclusions",
    )
    assert pm is not None
    assert _can_act_as_scope_inclusion_governor(pm)


def test_answer_without_a_question_still_stands_alone():
    atom = pm_answer_to_atom(
        project_id="p1",
        question="",
        answer="Union labor is required at the Chicago site after 6pm.",
    )
    assert atom is not None
    assert atom.raw_text.startswith("Union labor")
