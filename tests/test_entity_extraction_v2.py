"""Tests for the v2 entity extractors added on the OPTBOT integration
follow-up:

  - stakeholder / person entity extraction (named approver + role cue)
  - customer entity from "Company: X" / "Customer: X" labels
  - money / currency entity extraction (dollar / USD / K/M/B shorthand)
  - date and milestone entity extraction (ISO / US / long-form)
  - extended street-address suffix coverage
  - service-vs-device classification on BOM line items

Pinned by the OPTBOT mock deal as the regression baseline — every
named approver, dollar threshold, milestone date, and service line
item that appears in the OPTBOT fixture must be captured by these
emitters.
"""
from __future__ import annotations

import pytest

from app.core.entity_extraction import (
    _emit_customer_from_label,
    _emit_date_keys,
    _emit_money_keys,
    _emit_sites,
    _emit_stakeholders,
    _STREET_ADDRESS_REGEX,
)


# ─── A. extended street-address suffixes ───


@pytest.mark.parametrize(
    "text",
    [
        "4200 Global Gateway Connector, Building C",
        "1180 Peachtree Street NE",
        "976 Brady Avenue NW",
        "525 Innovation Corridor",
        "1 World Trade Plaza",
        "500 Town Square",
        "100 Independence Loop",
        "42 Riverside Crossing",
        "1700 Market Terrace",
        "300 Industrial Pike",
    ],
)
def test_extended_address_suffixes_match(text: str) -> None:
    """The address regex must capture commercial-deal-style street
    suffixes beyond the residential set (Connector, Corridor, Gateway,
    Crossing, Plaza, Square, Loop, Terrace, Pike, ...)."""
    matches = list(_STREET_ADDRESS_REGEX.finditer(text))
    assert matches, f"no address match for {text!r}"


# ─── B. customer-from-label extraction ───


@pytest.mark.parametrize(
    "text,expected_substring",
    [
        ("Company: OPTBOT, Inc. | Domain: optbot.example", "optbot_inc"),
        ("Customer: Acme Corp", "acme_corp"),
        ("Client: Globex LLC", "globex_llc"),
        ("Account: Initech, Inc.", "initech_inc"),
        ("End Client: Stark Industries", "stark_industries"),
        ("Buyer: Wonka Corporation", "wonka_corporation"),
        ("Organization: Hooli Holdings", "hooli_holdings"),
    ],
)
def test_customer_label_extraction(text: str, expected_substring: str) -> None:
    keys = _emit_customer_from_label(text)
    assert any(expected_substring in k for k in keys), (
        f"expected substring {expected_substring!r} in {sorted(keys)} from {text!r}"
    )


@pytest.mark.parametrize(
    "text",
    [
        # Generic / no corporate suffix → must NOT emit
        "Customer: ALL",
        "Client: TBD",
        "Company: Various",
        "Customer satisfaction is high",  # not a label
        "Company picnic on Friday",       # not a label
    ],
)
def test_customer_label_rejects_generics(text: str) -> None:
    keys = _emit_customer_from_label(text)
    assert not keys, f"generic value leaked as customer: {sorted(keys)} from {text!r}"


# ─── C. money / currency extraction ───


@pytest.mark.parametrize(
    "text,expected_amount",
    [
        ("Total mock deal amount: $1,847,250.00", 1847250),
        ("CFO approval required over $1,500,000", 1500000),
        ("$250K threshold for budget owner", 250000),
        ("$1.5M total commitment", 1500000),
        ("USD 1,847,250 final amount", 1847250),
        ("Hardware subtotal: $1,015,626.00", 1015626),
        ("$2.5B annual revenue", 2500000000),
        ("$5K device cost", 5000),
    ],
)
def test_money_amount_normalization(text: str, expected_amount: int) -> None:
    keys = _emit_money_keys(text)
    assert f"money:{expected_amount}" in keys, (
        f"expected money:{expected_amount} in {sorted(keys)} from {text!r}"
    )


@pytest.mark.parametrize(
    "text",
    [
        # Sub-$100 — likely noise (loose change, line numbers, page #s)
        "Quote item $5.50",
        "Page $99",
        # Currency-shape tokens that aren't money
        "$VAR is a placeholder",
        "$.00 trailing zero",
    ],
)
def test_money_rejects_implausible_amounts(text: str) -> None:
    keys = _emit_money_keys(text)
    assert not keys, f"implausible amount leaked: {sorted(keys)} from {text!r}"


# ─── D. date / milestone extraction ───


@pytest.mark.parametrize(
    "text,expected_date",
    [
        ("Close date: 2026-07-31", "2026-07-31"),
        ("Mobilization at ATL-AIR begins on 2026-05-20.", "2026-05-20"),
        ("Quote valid through 2026-06-14", "2026-06-14"),
        ("Cutover begins July 31, 2026", "2026-07-31"),
        ("Implementation end 08/14/2026", "2026-08-14"),
        ("Hypercare starts on 2026-08-15.", "2026-08-15"),
        ("Effective Jan 1, 2026", "2026-01-01"),
    ],
)
def test_iso_date_extraction(text: str, expected_date: str) -> None:
    keys = _emit_date_keys(text)
    assert f"date:{expected_date}" in keys, (
        f"expected date:{expected_date} in {sorted(keys)} from {text!r}"
    )


@pytest.mark.parametrize(
    "text,expected_date",
    [
        # All of these have a milestone-context cue near the date,
        # so both a date: AND a milestone: key should be emitted.
        ("Close date: 2026-07-31", "2026-07-31"),
        ("Mobilization start: 2026-05-20", "2026-05-20"),
        ("Cutover begins 2026-07-31", "2026-07-31"),
        ("Hypercare starts on 2026-08-15.", "2026-08-15"),
        ("Executive blackout window 2026-06-17 through 2026-06-21", "2026-06-17"),
        ("Phase 3 implementation start 2026-07-06", "2026-07-06"),
        ("Project go-live 2026-08-14", "2026-08-14"),
    ],
)
def test_milestone_extraction_with_context(text: str, expected_date: str) -> None:
    keys = _emit_date_keys(text)
    assert f"milestone:{expected_date}" in keys, (
        f"expected milestone:{expected_date} (with context cue) in "
        f"{sorted(keys)} from {text!r}"
    )


@pytest.mark.parametrize(
    "text",
    [
        # Dates WITHOUT milestone context — should emit date: but
        # NOT milestone:
        "The 2026-01-01 file was uploaded.",
        "Customer phoned on 2026-03-15 about pricing.",
    ],
)
def test_plain_date_no_milestone(text: str) -> None:
    keys = _emit_date_keys(text)
    date_keys = {k for k in keys if k.startswith("date:")}
    milestone_keys = {k for k in keys if k.startswith("milestone:")}
    assert date_keys, f"expected a date: in {sorted(keys)} from {text!r}"
    assert not milestone_keys, (
        f"plain date wrongly tagged as milestone: {sorted(milestone_keys)} from {text!r}"
    )


# ─── E. stakeholder / person extraction ───


@pytest.mark.parametrize(
    "text,expected_name",
    [
        # The six named OPTBOT approvers + one bonus (Noah Patel) —
        # each is mentioned with a role/title or approval verb nearby.
        ("Jordan Ames, VP Workplace Operations, is the executive sponsor.", "jordan_ames"),
        ("Priya Narang approves technical design pending ATL-WEST.", "priya_narang"),
        ("Camila Brooks: Approved security and data handling.", "camila_brooks"),
        ("Morgan Lee, CFO Delegate: Approval required.", "morgan_lee"),
        ("Elliot Tran approves procurement release.", "elliot_tran"),
        ("Renee Watkins accepts delivery governance artifacts.", "renee_watkins"),
        ("Noah Patel | Regional Facilities Manager | owns access windows.", "noah_patel"),
        # Common approval / sponsor patterns from real deal docs
        ("Approved by Sara Chen, CFO.", "sara_chen"),
        ("The project sponsor is Marcus Lee.", "marcus_lee"),
        ("Owned by Anika Patel, Director of Engineering.", "anika_patel"),
    ],
)
def test_stakeholder_extraction(text: str, expected_name: str) -> None:
    keys = _emit_stakeholders(text)
    assert f"stakeholder:{expected_name}" in keys, (
        f"expected stakeholder:{expected_name} in {sorted(keys)} from {text!r}"
    )


@pytest.mark.parametrize(
    "phrase",
    [
        # Two-word capitalized phrases that match the name shape but
        # aren't people — must NOT become stakeholders. These are
        # the false positives that appeared in the first stakeholder
        # implementation.
        "Help Desk approves access requests",
        "Checklist Item: approves the milestone",
        "Due Date approves the phase change",
        "Task Type signoff required",
        "Review Cadence is owned by the PMO",
        "Expected Output approved by lead",
        "Workplace Operations is the function",
        "Workforce Planning approves the budget",
        "Service Desk: approves the change",
    ],
)
def test_stakeholder_rejects_non_person_phrases(phrase: str) -> None:
    keys = _emit_stakeholders(phrase)
    assert not keys, (
        f"non-person phrase wrongly captured as stakeholder: {sorted(keys)} from {phrase!r}"
    )


@pytest.mark.parametrize(
    "text",
    [
        # Plain person names WITHOUT a role-context cue → must NOT
        # emit. Without the role cue we can't tell a name from any
        # other Capitalized two-word phrase.
        "Jordan Ames went to lunch.",
        "Priya Narang said hello.",
    ],
)
def test_stakeholder_requires_role_context(text: str) -> None:
    keys = _emit_stakeholders(text)
    assert not keys, (
        f"name without role context wrongly captured: {sorted(keys)} from {text!r}"
    )


# ─── F. OPTBOT end-to-end integration ───


def test_optbot_paragraph_extracts_all_value_categories() -> None:
    """A representative OPTBOT paragraph should yield clean entities
    in all five new categories: customer, money, date, milestone,
    stakeholder.
    """
    text = (
        "Company: OPTBOT, Inc. | Domain: optbot.example. "
        "Total mock deal amount: $1,847,250.00. CFO approval required "
        "over $1,500,000. Budget owner approval required over $250,000. "
        "Jordan Ames approves workplace outcome and business case. "
        "Priya Narang approves technical design. "
        "Close date: 2026-07-31. Mobilization at ATL-AIR begins on "
        "2026-05-20. Cutover completes 2026-08-14."
    )
    customer_keys = _emit_customer_from_label(text)
    money_keys = _emit_money_keys(text)
    date_keys = _emit_date_keys(text)
    stakeholder_keys = _emit_stakeholders(text)
    site_keys = _emit_sites(text)

    assert "customer:optbot_inc" in customer_keys
    assert "money:1847250" in money_keys
    assert "money:1500000" in money_keys
    assert "money:250000" in money_keys
    assert "date:2026-07-31" in date_keys
    assert "milestone:2026-07-31" in date_keys  # has "close date" cue
    assert "milestone:2026-05-20" in date_keys  # has "mobilization" cue
    assert "milestone:2026-08-14" in date_keys  # has "cutover" cue
    assert "stakeholder:jordan_ames" in stakeholder_keys
    assert "stakeholder:priya_narang" in stakeholder_keys
    assert "site:atl_air" in site_keys


# ─── G. service-vs-device classification ───


def test_service_classifier_recognizes_service_lines() -> None:
    """The xlsx/quote parser service-line classifier must correctly
    identify labor/support/training/hypercare/governance/etc. as
    services rather than devices.
    """
    from app.parsers.xlsx_parser import _looks_like_service_line
    from app.parsers.quote_parser import _looks_like_service_description

    service_descriptions = [
        "After-hours installation labor",
        "Hypercare support",
        "Training and adoption support",
        "Project management and weekly governance",
        "Discovery workshops and technical design",
        "Professional services engagement",
        "Managed services subscription",
        "Consulting hours",
        "Cutover commissioning",
        "UAT acceptance testing",
    ]
    for desc in service_descriptions:
        assert _looks_like_service_line(desc), (
            f"xlsx classifier missed service: {desc!r}"
        )
        assert _looks_like_service_description(desc), (
            f"quote classifier missed service: {desc!r}"
        )


def test_service_classifier_does_not_misclassify_devices() -> None:
    """Real hardware descriptions must NOT be misclassified as
    services."""
    from app.parsers.xlsx_parser import _looks_like_service_line

    device_descriptions = [
        "Access point Wi-Fi 7",
        "PoE++ network switch",
        "IP camera 4K",
        "UPS battery backup",
        "Video bar for medium rooms",
        "Docking station USB-C 180W",
        "Rugged logistics tablet",
        "Secure label printer kit",
        "Firewall appliance",
        "Network controller",
    ]
    for desc in device_descriptions:
        assert not _looks_like_service_line(desc), (
            f"xlsx classifier wrongly flagged device as service: {desc!r}"
        )
