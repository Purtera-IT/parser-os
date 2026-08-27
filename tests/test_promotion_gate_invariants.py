"""Promotion-gate invariants — the parser may not promote what it cannot support.

Four deterministic guardrails, pinned here against the two real deals that
exposed them (01491cca, 1e130077):

1. ``strip_unsupported_names``   — no manufactured display strings.
2. ``cap_authority_to_source``   — authority may not exceed its source.
3. roster / quantity misreads    — an account header is not a site roster,
                                   and "building 704" is not 704 of anything.
4. ``demote_unearned_contract_authority`` — a file format is not a contract.

Every test follows the non-vacuity discipline of
``tests/test_graph_packet_invariance.py``: a fixture must be asserted to
actually exercise the rule, so a regression that silently stops the rule
firing fails here instead of passing quietly.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import pytest

from app.core.atom_type_sanity import (
    cap_authority_to_source,
    classify_document_contract_evidence,
    demote_unearned_contract_authority,
    strip_unsupported_names,
    surface_headline_quantities,
)
from app.core.authority import AUTHORITY_RANKS
from app.core.entity_extraction import _emit_quantity_keys
from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    EvidenceAtom,
    ReviewStatus,
    SourceRef,
)
from app.parsers.site_roster_extractor import (
    _looks_like_billing_header,
    looks_like_site_roster,
)

MACHINE_RANK = AUTHORITY_RANKS[AuthorityClass.machine_extractor]
PM_RANK = AUTHORITY_RANKS[AuthorityClass.pm_confirmed]


def _atom(
    atom_id="atm_1",
    atom_type=AtomType.physical_site,
    text="text",
    *,
    value=None,
    authority=AuthorityClass.machine_extractor,
    artifact_id="art_real",
    source_artifact_id=None,
    entity_keys=None,
):
    return EvidenceAtom(
        id=atom_id,
        project_id="proj_1",
        artifact_id=artifact_id,
        atom_type=atom_type,
        raw_text=text,
        normalized_text=text.lower(),
        value=value if value is not None else {},
        entity_keys=entity_keys or [],
        source_refs=[
            SourceRef(
                id=f"src_{atom_id}",
                artifact_id=source_artifact_id or artifact_id,
                artifact_type=ArtifactType.txt,
                filename="fixture.txt",
                locator={},
                extraction_method="test",
                parser_version="test",
            )
        ],
        receipts=[],
        authority_class=authority,
        confidence=0.9,
        confidence_raw=0.9,
        calibrated_confidence=0.9,
        review_status=ReviewStatus.auto_accepted,
        review_flags=[],
        parser_version="test",
    )


def _snapshot(atoms):
    """Order-independent, comparable view of what the passes may change."""
    return {
        a.id: (
            str(getattr(a.atom_type, "value", a.atom_type)),
            str(getattr(a.authority_class, "value", a.authority_class)),
            json.dumps(a.value, sort_keys=True, default=str),
            tuple(sorted(a.review_flags or [])),
            tuple(sorted(a.entity_keys or [])),
        )
        for a in atoms
    }


# ══ 1. no manufactured display names ═══════════════════════════════════

# The real atom from deal 01491cca: a billing header out of "Cable Drop
# CALC.xlsx" promoted to physical_site with a name that appears in NO
# source document.
_GOLETA_TEXT = "Customer:: City/St: | Park Place Tech LLC:, Goleta, CA 93117"


def _goleta_atom():
    return _atom(
        atom_id="atm_goleta",
        atom_type=AtomType.physical_site,
        text=_GOLETA_TEXT,
        value={
            "kind": "physical_site",
            "id": "GOLETA-CA-93117",
            "site_id": "GOLETA-CA-93117",
            "name": "Goleta Office",
            "names": [
                "GOLETA-CA-93117",
                "Goleta Office",
                "Customer:: City/St: | Park Place Tech LLC:",
            ],
            "facility_name": "Goleta Office",
            "display_name": "Goleta Office",
            "city": "Goleta",
            "state": "CA",
            "facility_label": {"label": "keep_facility", "confidence": 0.82},
        },
    )


def test_fabricated_name_is_stripped_not_invented():
    atoms = [_goleta_atom()]
    # Non-vacuity: the fixture really does carry the fabricated name.
    assert atoms[0].value["name"] == "Goleta Office"
    assert "goleta office" not in atoms[0].raw_text.lower()

    assert strip_unsupported_names(atoms) == 1
    value = atoms[0].value

    # The invention is gone from every name-bearing field.
    assert "Goleta Office" not in json.dumps(value)
    # It fell back to a SUPPORTED alias — never a new string.
    assert value["name"] == "GOLETA-CA-93117"
    assert value["name"].lower().replace("-", " ") in atoms[0].raw_text.lower().replace(
        ",", ""
    ).replace("  ", " ")
    assert "Goleta Office" not in value["names"]
    assert "GOLETA-CA-93117" in value["names"]
    # Abstention on the NAME, not on the evidence.
    assert len(atoms) == 1
    assert atoms[0].raw_text == _GOLETA_TEXT
    assert "unsupported_name_stripped" in atoms[0].review_flags
    assert atoms[0].review_status == ReviewStatus.needs_review


def test_unsupported_name_with_no_alias_is_deleted_outright():
    atoms = [
        _atom(
            text="Invoice for services rendered.",
            value={"kind": "physical_site", "name": "Riverside Data Center"},
        )
    ]
    assert strip_unsupported_names(atoms) == 1
    assert "name" not in atoms[0].value  # unnamed, not renamed
    assert atoms[0].value["kind"] == "physical_site"


# A supported name must survive every dressing this repo treats as noise.
@pytest.mark.parametrize(
    "dress",
    [
        "Goleta Office",
        "goleta office",
        "GOLETA OFFICE",
        "Goleta  Office",
        "Goleta Office",  # NBSP
        "Goleta-Office",
        "goleta_office",
        "Goleta, Office.",
        "  Goleta Office  ",
    ],
)
def test_supported_name_survives_any_dress(dress):
    atoms = [
        _atom(
            text="Site walk scheduled at the Goleta Office next Tuesday.",
            value={"kind": "physical_site", "name": dress},
        )
    ]
    assert strip_unsupported_names(atoms) == 0
    assert atoms[0].value["name"] == dress
    assert atoms[0].review_flags == []


@pytest.mark.parametrize(
    "source_dress",
    [
        "Site walk at the Goleta Office.",
        "SITE WALK AT THE GOLETA OFFICE.",
        "Site walk at the goleta-office.",
        "Site walk at the goleta_office.",
        "Site walk at the Goleta Office.",
    ],
)
def test_name_support_is_blind_to_source_dress(source_dress):
    atoms = [_atom(text=source_dress, value={"name": "Goleta Office"})]
    assert strip_unsupported_names(atoms) == 0


def test_no_atom_keeps_a_display_name_absent_from_its_own_source():
    """The invariant itself, over a mixed population."""
    atoms = [
        _goleta_atom(),
        _atom(atom_id="atm_ok", text="The Jacksonville Campus is in scope.",
              value={"name": "Jacksonville Campus"}),
        _atom(atom_id="atm_bad2", text="Row 14 | 6500 Hollister Ave 210",
              value={"name": "Hollister Tech Park", "names": ["Hollister Tech Park"]}),
    ]
    changed = strip_unsupported_names(atoms)
    assert changed == 2, "fixture must exercise the rule on both bad atoms"

    for atom in atoms:
        support = atom.raw_text.lower()
        for field in ("name", "facility_name", "display_name", "site_id"):
            name = atom.value.get(field)
            if not isinstance(name, str):
                continue
            probe = "".join(c for c in name.lower().replace("-", " ") if c.isalnum() or c == " ")
            probe = " ".join(probe.split())
            haystack = "".join(
                c for c in support.replace("-", " ") if c.isalnum() or c == " "
            )
            haystack = " ".join(haystack.split())
            assert probe in haystack, f"{atom.id}: unsupported {field}={name!r}"


def test_strip_names_is_idempotent_and_order_invariant():
    def build():
        return [
            _goleta_atom(),
            _atom(atom_id="atm_ok", text="The Jacksonville Campus is in scope.",
                  value={"name": "Jacksonville Campus"}),
            _atom(atom_id="atm_bad2", text="Row 14 | 6500 Hollister Ave 210",
                  value={"name": "Hollister Tech Park"}),
        ]

    once = build()
    assert strip_unsupported_names(once) > 0, "fixture must exercise the rule"
    after_first = _snapshot(once)
    assert strip_unsupported_names(once) == 0  # idempotent: nothing left to strip
    assert _snapshot(once) == after_first

    shuffled = build()
    random.Random(7).shuffle(shuffled)
    strip_unsupported_names(shuffled)
    assert _snapshot(shuffled) == after_first


# ══ 2. authority may not exceed its source ═════════════════════════════

REAL_ARTIFACTS = {"art_real", "art_other"}


def test_unresolved_source_is_capped_to_machine_extractor():
    atoms = [
        _atom(atom_id="atm_ghost", text="Site is ready for cutover.",
              authority=AuthorityClass.customer_current_authored,
              artifact_id="art_nonexistent"),
    ]
    assert AUTHORITY_RANKS[atoms[0].authority_class] > MACHINE_RANK  # non-vacuity
    assert cap_authority_to_source(atoms, artifact_ids=REAL_ARTIFACTS) == 1
    assert atoms[0].authority_class == AuthorityClass.machine_extractor
    assert "authority_capped_unresolved_source" in atoms[0].review_flags


def test_resolved_source_keeps_its_authority():
    atoms = [
        _atom(text="Contractor shall provide 40 drops.",
              authority=AuthorityClass.customer_current_authored,
              artifact_id="art_real"),
    ]
    assert cap_authority_to_source(atoms, artifact_ids=REAL_ARTIFACTS) == 0
    assert atoms[0].authority_class == AuthorityClass.customer_current_authored


def test_pass_never_raises_authority():
    atoms = [
        _atom(atom_id="atm_low", text="stale quote text",
              authority=AuthorityClass.quoted_old_email,
              artifact_id="art_nonexistent"),
    ]
    cap_authority_to_source(atoms, artifact_ids=REAL_ARTIFACTS)
    assert atoms[0].authority_class == AuthorityClass.quoted_old_email


# The rank-laundering fixture: the pipeline's own prior output, carried in
# the manifest's `context` key and re-atomized at rank 90.
_LAUNDERED = [
    "context.prior_scope_process_v1.sowHandoff.assumptions[55].atom_id: atm_79fb",
    "context.prior_scope_process_v1.sowHandoff.scope_in[6].status: active",
    "artifacts[5].blob_url: https://example.blob.core.windows.net/x/y",
    "artifacts[3].metadata.hubspotNoteUpdatedAt: 2026-05-26T13:40:37.845Z",
]


@pytest.mark.parametrize("text", _LAUNDERED)
def test_serialized_structure_is_not_evidence(text):
    atoms = [
        _atom(atom_id="atm_ctx", atom_type=AtomType.scope_item, text=text,
              authority=AuthorityClass.customer_current_authored,
              artifact_id="art_real"),
    ]
    # Non-vacuity: the source DOES resolve, so only the structure check can fire.
    assert atoms[0].artifact_id in REAL_ARTIFACTS
    assert cap_authority_to_source(atoms, artifact_ids=REAL_ARTIFACTS) == 1
    assert atoms[0].authority_class == AuthorityClass.machine_extractor
    assert "authority_capped_serialized_structure" in atoms[0].review_flags
    assert "serialized_structure_not_evidence" in atoms[0].review_flags
    # Evidence is kept, only its standing drops.
    assert atoms[0].raw_text == text


@pytest.mark.parametrize(
    "text",
    [
        "Note: the site is ready.",
        "Customer: Acme Corp",
        "www.example.com: down",
        "Contractor shall install 48 ports. Refer to section 3.2.",
        "Rev. 2: issued for construction",
    ],
)
def test_document_prose_is_not_mistaken_for_structure(text):
    atoms = [
        _atom(text=text, authority=AuthorityClass.customer_current_authored,
              artifact_id="art_real"),
    ]
    assert cap_authority_to_source(atoms, artifact_ids=REAL_ARTIFACTS) == 0
    assert atoms[0].authority_class == AuthorityClass.customer_current_authored


def test_serialized_structure_is_floored_off_claim_types():
    atoms = [
        _atom(atom_id="atm_req", atom_type=AtomType.requirement,
              text=_LAUNDERED[0], authority=AuthorityClass.customer_current_authored),
        _atom(atom_id="atm_scope", atom_type=AtomType.scope_item,
              text=_LAUNDERED[1], authority=AuthorityClass.customer_current_authored),
    ]
    cap_authority_to_source(atoms, artifact_ids=REAL_ARTIFACTS)
    # A key path may not stand as a requirement...
    assert atoms[0].atom_type == AtomType.project_metadata
    # ...but a generic retained type is left alone (conservative).
    assert atoms[1].atom_type == AtomType.scope_item


def test_no_atom_lacking_a_real_artifact_exceeds_machine_rank():
    atoms = [
        _atom(atom_id=f"atm_{i}", text=t, artifact_id=aid,
              authority=AuthorityClass.customer_current_authored)
        for i, (t, aid) in enumerate([
            ("Contractor shall provide 40 drops.", "art_real"),
            ("Site is ready.", "art_missing"),
            (_LAUNDERED[0], "art_real"),
            ("Cutover window is 8pm-2am.", "art_other"),
            ("Ghost row", "art_gone"),
        ])
    ]
    capped = cap_authority_to_source(atoms, artifact_ids=REAL_ARTIFACTS)
    assert capped == 3, "fixture must exercise both halves of the rule"
    for atom in atoms:
        if atom.artifact_id not in REAL_ARTIFACTS:
            assert AUTHORITY_RANKS[atom.authority_class] <= MACHINE_RANK, atom.id


def test_cap_authority_is_idempotent_and_order_invariant():
    def build():
        return [
            _atom(atom_id="a1", text="Contractor shall provide 40 drops.",
                  artifact_id="art_real",
                  authority=AuthorityClass.customer_current_authored),
            _atom(atom_id="a2", text="Site is ready.", artifact_id="art_missing",
                  authority=AuthorityClass.customer_current_authored),
            _atom(atom_id="a3", atom_type=AtomType.requirement, text=_LAUNDERED[0],
                  artifact_id="art_real",
                  authority=AuthorityClass.customer_current_authored),
        ]

    once = build()
    assert cap_authority_to_source(once, artifact_ids=REAL_ARTIFACTS) == 2
    after_first = _snapshot(once)
    assert cap_authority_to_source(once, artifact_ids=REAL_ARTIFACTS) == 0
    assert _snapshot(once) == after_first

    shuffled = build()
    random.Random(11).shuffle(shuffled)
    cap_authority_to_source(shuffled, artifact_ids=REAL_ARTIFACTS)
    assert _snapshot(shuffled) == after_first


# ══ 3a. an account header is not a site roster ═════════════════════════


def test_billing_header_block_yields_no_site_roster():
    """The real "Cable Drop CALC.xlsx" block that minted "Goleta Office"."""
    columns = ["Date:", "Address:", "Exp Date:", "City/St:", "Account #", "Rqstd By:"]
    rows = [["", "6500 Hollister Ave 210", "", "Goleta, Ca 93117", "", ""]]
    surrounding = (
        "Date: | Address: | 6500 Hollister Ave 210 | Exp Date: | "
        "City/St: | Goleta, Ca 93117 | Account # | Rqstd By: |"
    )
    assert not looks_like_site_roster(
        columns=columns, rows=rows, surrounding_text=surrounding
    )


def test_address_plus_city_alone_is_not_a_roster():
    """The pair with no site-specific field — every letterhead matched it."""
    assert not looks_like_site_roster(
        columns=["Address", "City"],
        rows=[["6500 Hollister Ave", "Goleta"]],
        surrounding_text="",
    )


def test_address_plus_city_with_a_site_field_is_still_a_roster():
    """Non-vacuity: the gate must not have been broken shut."""
    assert looks_like_site_roster(
        columns=["Site ID", "Address", "City"],
        rows=[["ATL-HQ-01", "6500 Hollister Ave", "Goleta"]],
        surrounding_text="",
    )
    assert looks_like_site_roster(
        columns=["Facility Name", "Address", "City"],
        rows=[["Jacksonville Campus", "1 Main St", "Jacksonville"]],
        surrounding_text="",
    )


def test_real_roster_survives_a_single_incidental_billing_word():
    """One marker is not evidence — real SOW rosters say "Customer:"."""
    assert looks_like_site_roster(
        columns=["Site ID", "Facility Name", "Street Address"],
        rows=[["ATL-01", "Atlanta HQ", "1 Peachtree St"]],
        surrounding_text="Customer: Acme Corp. Site roster follows.",
    )


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("Account # 4471 Exp Date 05/26", True),
        ("Account# 4471 Exp  Date 05/26", True),  # whitespace is dress
        ("Account Number 4471 and Account No 4471", False),  # one concept, not two
        ("Customer: Acme Corp", False),  # one marker is not evidence
        ("", False),
    ],
)
def test_billing_header_needs_two_independent_markers(text, want):
    assert _looks_like_billing_header(text) is want


def test_explicit_site_declaration_beats_billing_markers():
    """Negative evidence weakens heuristics; it does not override a declaration."""
    assert looks_like_site_roster(
        columns=["Address", "City"],
        rows=[["1 Main St", "Goleta"]],
        surrounding_text="kind=physical_site — Account # 4471 Exp Date 05/26",
    )


# ══ 3b. a number in a naming context is not a count ════════════════════


_TOC_704 = "b-2 figure c-1 building 704 new drop locations ................."


def test_building_number_is_not_a_quantity():
    """The real TOC line from deal 01491cca that emitted ``quantity:704``.

    That key came from ``surface_headline_quantities`` (the atom carried
    ``{"quantity": 704, "noun": "drop", "inferred": true}``), not from
    ``_emit_quantity_keys`` — so the pin has to exercise *that* producer.
    """
    parent = _atom(atom_id="atm_toc", atom_type=AtomType.scope_item, text=_TOC_704)
    surfaced = surface_headline_quantities([parent], project_id="p")
    assert surfaced == []
    assert not any(
        "704" in k for a in surfaced for k in a.entity_keys
    )


def test_surface_headline_quantities_still_fires_on_a_real_count():
    """Non-vacuity: the naming guard must not have silenced the producer."""
    parent = _atom(
        atom_id="atm_prose",
        atom_type=AtomType.scope_item,
        text="Install 704 new drop locations across the campus.",
    )
    surfaced = surface_headline_quantities([parent], project_id="p")
    assert len(surfaced) == 1
    assert "quantity:704" in surfaced[0].entity_keys
    assert surfaced[0].value["quantity"] == 704


@pytest.mark.parametrize(
    ("text", "expect_count"),
    [
        ("building 704 new drop locations", False),
        ("bldg 704 new drop locations", False),
        ("suite 210 new drop locations", False),
        ("room 12 cameras", False),
        ("Install 704 new drop locations", True),
        ("704 new drop locations in building 12", True),
    ],
)
def test_headline_quantity_respects_naming_context(text, expect_count):
    parent = _atom(atom_id="atm_h", atom_type=AtomType.scope_item, text=text)
    surfaced = surface_headline_quantities([parent], project_id="p")
    assert bool(surfaced) is expect_count


@pytest.mark.parametrize(
    "text",
    [
        "building 704 drops",
        "bldg 704 drops",
        "bldg. 704 drops",
        "Building 704 drops",
        "suite 210 cameras",
        "room 12 cameras",
        "Floor 3 access points",
    ],
)
def test_naming_context_numbers_emit_no_quantity(text):
    assert not any(k.startswith("quantity:") for k in _emit_quantity_keys({}, text))


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("704 drops", "quantity:704"),
        ("install 704 drops", "quantity:704"),
        ("704 drops in building 12", "quantity:704"),
        ("Provide 48 ports", "quantity:48"),
    ],
)
def test_real_counts_still_emit(text, want):
    """Non-vacuity: the guard must not have silenced genuine counts."""
    assert want in _emit_quantity_keys({}, text)


def test_building_704_and_704_drops_are_distinguished():
    assert "quantity:704" not in _emit_quantity_keys({}, "building 704 drops")
    assert "quantity:704" in _emit_quantity_keys({}, "704 drops")


# ══ 4. a file format is not a contract ═════════════════════════════════

ROM_FILE = "010260 - ROM- Jacksonville Campus Workstation Deployment (8.27.26_V1).xlsx"
DEAL_KIT_FILE = "010260 Deal Kit.xlsx"
CONTRACT_FILE = "Master Services Agreement - Executed.pdf"


@pytest.mark.parametrize(
    "filename",
    [
        ROM_FILE,
        DEAL_KIT_FILE,
        "Cable Drop CALC.xlsx",
        "deal_kit.xlsx",
        "DEALKIT.XLSX",
        "Deal-Kit.xlsx",
        "budgetary estimate.xlsx",
        "DRAFT scope.docx",
        "Quotation 4471.pdf",
        "pricing sheet.xlsx",
        "rough order of magnitude.xlsx",
    ],
)
def test_non_contract_documents_are_classified_as_such(filename):
    assert classify_document_contract_evidence(filename) == "non_contract"


@pytest.mark.parametrize(
    "filename",
    [
        CONTRACT_FILE,
        "master_service_agreement.pdf",
        "MASTER SERVICES AGREEMENT.pdf",
        "Purchase Order 88213.pdf",
        "Terms and Conditions.pdf",
        "contract-2026.docx",
    ],
)
def test_contract_documents_are_recognised(filename):
    assert classify_document_contract_evidence(filename) == "contract"


def test_sow_needs_a_signature_block():
    assert classify_document_contract_evidence("Statement of Work.pdf") == "unknown"
    assert (
        classify_document_contract_evidence(
            "Statement of Work.xlsx", "Accepted by: ______  Signature: ______"
        )
        == "contract"
    )


def test_negative_evidence_beats_positive():
    """A draft agreement is a draft, not an executed agreement."""
    assert classify_document_contract_evidence("DRAFT Master Services Agreement.pdf") == (
        "non_contract"
    )


def test_eeprom_does_not_trip_the_rom_marker():
    """Word-boundary matching: markers are tokens, not substrings."""
    assert classify_document_contract_evidence("EEPROM programming spec.pdf") == "unknown"


def _contract_population():
    return [
        _atom(atom_id="atm_rom", text="Workstation deployment estimate row 14",
              atom_type=AtomType.bom_line,
              authority=AuthorityClass.contractual_scope, artifact_id="art_rom"),
        _atom(atom_id="atm_kit", text="Deal kit row 3",
              atom_type=AtomType.commercial_total,
              authority=AuthorityClass.contractual_scope, artifact_id="art_kit"),
        _atom(atom_id="atm_msa", text="Contractor shall maintain insurance.",
              atom_type=AtomType.requirement,
              authority=AuthorityClass.contractual_scope, artifact_id="art_msa"),
        _atom(atom_id="atm_mail", text="Confirming the campus list.",
              atom_type=AtomType.scope_item,
              authority=AuthorityClass.contractual_scope, artifact_id="art_mail"),
        _atom(atom_id="atm_pm", text="PM confirmed 40 drops.",
              atom_type=AtomType.quantity,
              authority=AuthorityClass.pm_confirmed, artifact_id="art_rom"),
    ]


_CONTRACT_DOCS = {
    "art_rom": ROM_FILE,
    "art_kit": DEAL_KIT_FILE,
    "art_msa": CONTRACT_FILE,
    "art_mail": "010261-hs-email-115682026798.eml",
}


def test_rom_and_deal_kit_lose_contract_authority():
    atoms = _contract_population()
    before = [a for a in atoms if a.authority_class == AuthorityClass.contractual_scope]
    assert len(before) == 4, "fixture must exercise the rule"

    assert demote_unearned_contract_authority(atoms, documents=_CONTRACT_DOCS) == 3

    by_id = {a.id: a for a in atoms}
    # Priced/estimated working documents land at vendor_quote (what they are).
    assert by_id["atm_rom"].authority_class == AuthorityClass.vendor_quote
    assert by_id["atm_kit"].authority_class == AuthorityClass.vendor_quote
    # An email is customer-authored, not a contract.
    assert by_id["atm_mail"].authority_class == AuthorityClass.customer_current_authored
    # The executed agreement keeps rank 100.
    assert by_id["atm_msa"].authority_class == AuthorityClass.contractual_scope

    for aid in ("atm_rom", "atm_kit", "atm_mail"):
        assert "unearned_contract_authority_demoted" in by_id[aid].review_flags
        assert by_id[aid].review_status == ReviewStatus.needs_review
    assert "unearned_contract_authority_demoted" not in by_id["atm_msa"].review_flags


def test_no_rank_100_atom_survives_from_a_non_contract_document():
    atoms = _contract_population()
    demote_unearned_contract_authority(atoms, documents=_CONTRACT_DOCS)
    for atom in atoms:
        if atom.authority_class != AuthorityClass.contractual_scope:
            continue
        filename = _CONTRACT_DOCS[atom.artifact_id]
        assert classify_document_contract_evidence(filename) == "contract", atom.id


def test_pm_confirmed_outranks_a_rom_after_the_pass():
    atoms = _contract_population()
    demote_unearned_contract_authority(atoms, documents=_CONTRACT_DOCS)
    by_id = {a.id: a for a in atoms}
    pm_rank = AUTHORITY_RANKS[by_id["atm_pm"].authority_class]
    assert pm_rank == PM_RANK
    for aid in ("atm_rom", "atm_kit"):
        assert AUTHORITY_RANKS[by_id[aid].authority_class] < pm_rank, (
            f"{aid} still outranks the PM"
        )


def test_unknown_documents_are_left_alone():
    """The pass never guesses about a document the caller did not describe."""
    atoms = _contract_population()
    assert demote_unearned_contract_authority(atoms, documents={}) == 0
    assert all(
        a.authority_class == AuthorityClass.contractual_scope
        for a in atoms
        if a.id != "atm_pm"
    )


def test_contract_demotion_is_idempotent_and_order_invariant():
    once = _contract_population()
    assert demote_unearned_contract_authority(once, documents=_CONTRACT_DOCS) == 3
    after_first = _snapshot(once)
    assert demote_unearned_contract_authority(once, documents=_CONTRACT_DOCS) == 0
    assert _snapshot(once) == after_first

    shuffled = _contract_population()
    random.Random(3).shuffle(shuffled)
    demote_unearned_contract_authority(shuffled, documents=_CONTRACT_DOCS)
    assert _snapshot(shuffled) == after_first


# ══ real-envelope pins ═════════════════════════════════════════════════
#
# Run against the actual deal envelopes when they are available locally:
#   ORBIT_ENVELOPE_01491CCA=/path/envelope.json
#   ORBIT_ENVELOPE_1E130077=/path/envelope2.json
# Skipped (not failed) otherwise, so CI stays hermetic.


def _atoms_from_envelope(path: Path):
    payload = json.loads(path.read_text())
    docs = {
        d["artifact_id"]: d.get("filename", "") for d in payload.get("documents", [])
    }
    atoms = []
    for row in payload["atoms"]:
        text = row.get("text") or ""
        try:
            atoms.append(
                EvidenceAtom(
                    id=row["id"],
                    project_id=payload.get("project_id", "p"),
                    artifact_id=row.get("artifact_id", ""),
                    atom_type=AtomType(row["atom_type"]),
                    raw_text=text,
                    normalized_text=text.lower(),
                    value=row.get("structured") or {},
                    entity_keys=list(row.get("entity_keys") or []),
                    source_refs=[
                        SourceRef(
                            id=f"src_{row['id']}",
                            artifact_id=row.get("artifact_id", ""),
                            artifact_type=ArtifactType.txt,
                            filename=docs.get(row.get("artifact_id", ""), ""),
                            locator=row.get("locator") or {},
                            extraction_method="envelope",
                            parser_version="envelope",
                        )
                    ],
                    receipts=[],
                    authority_class=AuthorityClass(row["authority_class"]),
                    confidence=row.get("confidence") or 0.5,
                    review_status=ReviewStatus(row.get("review_status") or "needs_review"),
                    review_flags=[],
                    parser_version="envelope",
                )
            )
        except Exception:  # pragma: no cover - envelope shape drift
            continue
    return atoms, docs


@pytest.mark.skipif(
    not os.environ.get("ORBIT_ENVELOPE_01491CCA"),
    reason="real envelope for deal 01491cca not available locally",
)
def test_real_envelope_01491cca_no_laundered_rank_90():
    path = Path(os.environ["ORBIT_ENVELOPE_01491CCA"])
    atoms, docs = _atoms_from_envelope(path)
    laundered = [a for a in atoms if a.raw_text.startswith("context.")]
    assert laundered, "fixture must contain the laundered atoms"
    assert any(
        a.authority_class == AuthorityClass.customer_current_authored for a in laundered
    ), "fixture must exercise the rule"

    cap_authority_to_source(atoms, artifact_ids=set(docs))
    assert not [
        a for a in laundered
        if a.authority_class == AuthorityClass.customer_current_authored
    ]
    assert all(AUTHORITY_RANKS[a.authority_class] <= MACHINE_RANK for a in laundered)


@pytest.mark.skipif(
    not os.environ.get("ORBIT_ENVELOPE_01491CCA"),
    reason="real envelope for deal 01491cca not available locally",
)
def test_real_envelope_01491cca_goleta_office_is_gone():
    path = Path(os.environ["ORBIT_ENVELOPE_01491CCA"])
    atoms, _ = _atoms_from_envelope(path)
    assert any("Goleta Office" in json.dumps(a.value) for a in atoms), (
        "fixture must contain the fabricated name"
    )
    strip_unsupported_names(atoms)
    assert not [a for a in atoms if "Goleta Office" in json.dumps(a.value)]


@pytest.mark.skipif(
    not os.environ.get("ORBIT_ENVELOPE_1E130077"),
    reason="real envelope for deal 1e130077 not available locally",
)
def test_real_envelope_1e130077_no_unearned_rank_100():
    path = Path(os.environ["ORBIT_ENVELOPE_1E130077"])
    atoms, docs = _atoms_from_envelope(path)
    before = [a for a in atoms if a.authority_class == AuthorityClass.contractual_scope]
    assert len(before) > 100, "fixture must exercise the rule"

    demote_unearned_contract_authority(atoms, documents=docs)
    for atom in atoms:
        if atom.authority_class != AuthorityClass.contractual_scope:
            continue
        assert classify_document_contract_evidence(docs.get(atom.artifact_id, "")) == (
            "contract"
        ), f"{atom.id} from {docs.get(atom.artifact_id)!r} kept rank 100"
