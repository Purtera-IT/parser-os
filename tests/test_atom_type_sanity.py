"""Tests for the deterministic atom type-sanity guardrail."""

from __future__ import annotations

from app.core.atom_type_sanity import (
    apply_type_sanity,
    demote_nondeliverable_quantities,
    scrub_nondeliverable_quantity_keys,
    surface_headline_quantities,
)
from app.core.schemas import (
    ArtifactType,
    AtomType,
    AuthorityClass,
    EvidenceAtom,
    ReviewStatus,
    SourceRef,
)
from app.core.training_log import TrainingLog, set_training_log


def _atom(atom_type, text, *, entity_keys=None, value=None, aid="art_x"):
    return EvidenceAtom(
        id=f"atm_{abs(hash((atom_type, text))) % (10**12):012x}",
        project_id="p",
        artifact_id=aid,
        atom_type=atom_type,
        raw_text=text,
        normalized_text=text.lower(),
        value=value or {},
        entity_keys=entity_keys or [],
        source_refs=[
            SourceRef(
                id="src_1",
                artifact_id=aid,
                artifact_type=ArtifactType.txt,
                filename="f.txt",
                locator={},
                extraction_method="test",
                parser_version="t",
            )
        ],
        receipts=[],
        authority_class=AuthorityClass.machine_extractor,
        confidence=0.8,
        confidence_raw=0.8,
        calibrated_confidence=0.8,
        review_status=ReviewStatus.auto_accepted,
        review_flags=[],
        parser_version="t",
    )


def test_financial_quantity_demoted():
    atoms = [_atom(AtomType.quantity, "28.57% margin", entity_keys=["quantity:28"])]
    n = demote_nondeliverable_quantities(atoms)
    assert n == 1
    assert atoms[0].atom_type == AtomType.pricing_assumption
    assert all(not k.startswith("quantity:") for k in atoms[0].entity_keys)
    assert "retyped_quantity_to_pricing_assumption" in atoms[0].review_flags
    assert atoms[0].review_status == ReviewStatus.needs_review


def test_pmo_cost_demoted():
    atoms = [_atom(AtomType.quantity, "260 PMO Cost", entity_keys=["quantity:260"])]
    assert demote_nondeliverable_quantities(atoms) == 1
    assert atoms[0].atom_type == AtomType.pricing_assumption


def test_meta_pricing_lines_demoted():
    atoms = [_atom(AtomType.quantity, "99 pricing lines", entity_keys=["quantity:99"])]
    assert demote_nondeliverable_quantities(atoms) == 1
    assert atoms[0].atom_type == AtomType.pricing_assumption


def test_real_deliverable_quantity_preserved():
    atoms = [
        _atom(AtomType.quantity, "23 dwellings", entity_keys=["quantity:23"]),
        _atom(AtomType.quantity, "110 displays", entity_keys=["quantity:110"]),
    ]
    assert demote_nondeliverable_quantities(atoms) == 0
    assert all(a.atom_type == AtomType.quantity for a in atoms)


def test_deliverable_with_price_token_preserved():
    # "5 switches at $200" has a deliverable noun -> must NOT be demoted.
    atoms = [_atom(AtomType.quantity, "5 switches at $200 each", entity_keys=["quantity:5"])]
    assert demote_nondeliverable_quantities(atoms) == 0
    assert atoms[0].atom_type == AtomType.quantity


def test_surface_headline_quantity_from_prose():
    atoms = [
        _atom(
            AtomType.requirement,
            "The customer requires onsite field services support to replace "
            "approximately 110 existing TVs and mounts across a resort property.",
        )
    ]
    surfaced = surface_headline_quantities(atoms, project_id="p")
    assert len(surfaced) == 1
    assert surfaced[0].atom_type == AtomType.quantity
    assert surfaced[0].value["quantity"] == 110
    assert "quantity:110" in surfaced[0].entity_keys
    assert "headline_quantity" in surfaced[0].review_flags


def test_no_duplicate_headline_when_quantity_exists():
    atoms = [
        _atom(AtomType.requirement, "replace approximately 110 existing TVs"),
        _atom(AtomType.quantity, "110 units", entity_keys=["quantity:110"]),
    ]
    surfaced = surface_headline_quantities(atoms, project_id="p")
    assert surfaced == []


def test_small_counts_not_surfaced():
    atoms = [_atom(AtomType.requirement, "bring 4 technicians to site")]
    # 4 < MIN_HEADLINE_COUNT and "technicians" is not a deliverable noun
    surfaced = surface_headline_quantities(atoms, project_id="p")
    assert surfaced == []


def test_measure_word_not_surfaced_as_quantity():
    # "65 inch display" is a SCREEN DIMENSION, not 65 displays.
    # "15 minutes per unit" is a CONFIG DURATION, not 15 units.
    # Both have a number >= MIN_HEADLINE_COUNT immediately followed by a
    # measurement word, so neither may surface a quantity. Guess-free.
    atoms = [
        _atom(AtomType.requirement, "Each unit is an LG 65 inch display mounted on the wall."),
        _atom(AtomType.scope_item, "Configuration takes approximately 15 minutes per unit."),
    ]
    surfaced = surface_headline_quantities(atoms, project_id="p")
    assert surfaced == []


def test_measure_word_does_not_block_real_count_in_same_corpus():
    # The measure-word guard must reject only the measurement phrase, not
    # suppress a genuine deliverable count elsewhere.
    atoms = [
        _atom(AtomType.requirement, "Install the LG 65 inch display in each room."),
        _atom(AtomType.requirement, "Replace approximately 110 existing TVs across the property."),
        _atom(AtomType.scope_item, "Deploy 50 wireless access points campus-wide."),
    ]
    surfaced = surface_headline_quantities(atoms, project_id="p")
    counts = sorted(a.value["quantity"] for a in surfaced)
    assert counts == [50, 110]
    assert 65 not in counts


def test_surface_config_equipment_counts_from_compact_notes():
    log = TrainingLog(":memory:")
    set_training_log(log)
    atoms = [
        _atom(
            AtomType.scope_item,
            "4E7 APs. 2 UDM beast for routers. 2 48 port switches and 2nvr. "
            "Six or seven badge readers. It is worth noting we have 1 spare AP.",
        )
    ]
    try:
        surfaced = surface_headline_quantities(atoms, project_id="p")
    finally:
        set_training_log(None)
    rows = {(a.value["quantity"], a.value["noun"]): a.value for a in surfaced}

    assert (4, "access points") in rows
    assert (2, "switches") in rows
    assert rows[(2, "switches")]["descriptor"] == "48 port"
    assert (2, "NVRs") in rows
    assert (7, "badge readers") in rows
    assert rows[(7, "badge readers")]["range_min"] == 6
    assert rows[(7, "badge readers")]["range_max"] == 7
    assert rows[(1, "access points")]["qualifier"] == "spare"
    assert (48, "switches") not in rows
    train_labels = {r.label.lower() for r in log.rows(relation="equipment_quantity_context")}
    assert {"access points", "switches", "nvrs", "badge readers"} <= train_labels


def test_surfaced_quantity_carries_context_sentence():
    # A surfaced quantity must NOT be an orphaned "<N> <noun>" — it carries the
    # subject and surrounding statement so it is actionable to a head and
    # verifies against source. Regression guard for the context-free quantity
    # atoms the user flagged ("12 cameras", "20 users" with no context).
    sentence = (
        "And so what we have on site are, I don't know, 12 cameras, "
        "I don't know, six or seven badge readers."
    )
    atoms = [_atom(AtomType.scope_item, sentence)]
    surfaced = surface_headline_quantities(atoms, project_id="p")
    by_noun = {a.value["noun"]: a for a in surfaced}
    cams = by_noun["cameras"]
    # The atom text is the context sentence, not a bare "12 cameras".
    assert cams.raw_text == sentence
    assert cams.raw_text != "12 cameras"
    assert cams.value["headline"] == "12 cameras"
    assert "12 cameras" in cams.value["context"]
    # Verification proxy: the important terms of the atom appear in the source.
    assert "cameras" in cams.raw_text.lower() and "12" in cams.raw_text


def test_scrub_strips_financial_quantity_key_off_commercial_atom():
    # A commercial_total atom carrying a junk quantity key from a price label.
    atoms = [
        _atom(
            AtomType.commercial_total,
            "PMO Cost line",
            entity_keys=["quantity:260_pmo_cost", "money:260"],
        )
    ]
    stripped = scrub_nondeliverable_quantity_keys(atoms)
    assert stripped == 1
    assert atoms[0].entity_keys == ["money:260"]
    assert "scrubbed_nondeliverable_quantity_key" in atoms[0].review_flags


def test_scrub_strips_margin_and_meta_keys():
    atoms = [
        _atom(AtomType.pricing_assumption, "margin row", entity_keys=["quantity:28_57_margin"]),
        _atom(AtomType.commercial_total, "pricing lines", entity_keys=["quantity:118_pricing_lines"]),
    ]
    stripped = scrub_nondeliverable_quantity_keys(atoms)
    assert stripped == 2
    assert all(not any(str(k).startswith("quantity:") for k in a.entity_keys) for a in atoms)


def test_scrub_preserves_bare_numeric_deliverable_key():
    atoms = [
        _atom(AtomType.quantity, "110 displays", entity_keys=["quantity:110", "device:display"]),
        _atom(AtomType.requirement, "23 dwellings", entity_keys=["quantity:23"]),
    ]
    stripped = scrub_nondeliverable_quantity_keys(atoms)
    assert stripped == 0
    assert "quantity:110" in atoms[0].entity_keys
    assert "quantity:23" in atoms[1].entity_keys


def test_scrub_runs_inside_apply_type_sanity():
    atoms = [
        _atom(
            AtomType.commercial_total,
            "PMO Cost",
            entity_keys=["quantity:260_pmo_cost"],
        ),
        _atom(AtomType.quantity, "110 displays", entity_keys=["quantity:110"]),
    ]
    out, _demoted, _surfaced = apply_type_sanity(atoms, project_id="p")
    commercial = next(a for a in out if a.atom_type == AtomType.commercial_total)
    assert all(not str(k).startswith("quantity:") for k in commercial.entity_keys)
    qty = next(a for a in out if a.atom_type == AtomType.quantity)
    assert "quantity:110" in qty.entity_keys


def test_apply_type_sanity_combined():
    atoms = [
        _atom(AtomType.quantity, "28.57% margin", entity_keys=["quantity:28"]),
        _atom(AtomType.requirement, "replace approximately 110 existing TVs"),
        _atom(AtomType.quantity, "23 dwellings", entity_keys=["quantity:23"]),
    ]
    out, demoted, surfaced = apply_type_sanity(atoms, project_id="p")
    assert demoted == 1
    assert surfaced == 1
    types = [a.atom_type for a in out]
    assert types.count(AtomType.quantity) == 2  # 23 dwellings + surfaced 110
    assert AtomType.pricing_assumption in types


def test_payment_term_demoted_from_quantity():
    """'Net 30 days' is a credit period, not a deliverable count -> payment_term."""
    atoms = [_atom(AtomType.quantity, "Net 30 days", entity_keys=["quantity:30"])]
    assert demote_nondeliverable_quantities(atoms) == 1
    assert atoms[0].atom_type == AtomType.payment_term
    assert "retyped_quantity_to_payment_term" in atoms[0].review_flags
    assert not any(str(k).startswith("quantity:") for k in atoms[0].entity_keys)


def test_time_window_demoted_from_quantity():
    """A business-hours / clock-time window is not a deliverable count."""
    atoms = [_atom(AtomType.quantity, "8:00 AM to 5:00 PM Business Hours",
                   entity_keys=["quantity:8"])]
    assert demote_nondeliverable_quantities(atoms) == 1
    assert atoms[0].atom_type in (AtomType.site_access_window,
                                  AtomType.site_implementation_note)


def test_deliverable_count_not_demoted_by_term_rules():
    """The headline deliverable count survives the new term/window rules."""
    atoms = [_atom(AtomType.quantity, "110 units", entity_keys=["quantity:110"])]
    assert demote_nondeliverable_quantities(atoms) == 0
    assert atoms[0].atom_type == AtomType.quantity
