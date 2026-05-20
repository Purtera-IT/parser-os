"""Universality tests for the v2 entity extractors.

These tests prove the extractors work on a wide variety of real-world
input shapes beyond the OPTBOT baseline:

  - Stakeholder: honorifics (Dr/Mr/Mrs/Ms/Prof), three-word names,
    name suffixes (Jr/Sr/III), apostrophe and hyphen surnames,
    last-name-first format, initial-with-period middle name,
    multiple stakeholders per sentence.
  - Customer: international corporate suffixes (GmbH, AG, K.K., Oy,
    AB, ApS, Pty Ltd, Pvt Ltd, S.A., S.p.A., N.V., B.V., ...).
  - Money: multi-currency ($/€/£/¥) + ISO codes (USD/EUR/GBP/JPY/
    CHF/CAD/AUD/SEK/NOK/...) in both prefix and suffix forms,
    K/M/B/T shorthand multipliers.
  - Date: ISO/US/long formats + quarter notation (Q3 2026 / 3Q26) +
    fiscal year (FY26 / Fiscal Year 2026).

A regression in any of these would mean the extractor is more
brittle than the OPTBOT baseline. The OPTBOT regression suite is
covered separately in test_entity_extraction_v2.py.
"""
from __future__ import annotations

import pytest

from app.core.entity_extraction import (
    _emit_customer_from_label,
    _emit_date_keys,
    _emit_money_keys,
    _emit_stakeholders,
)


# ─── A. Stakeholder universality ───


@pytest.mark.parametrize(
    "text,expected_slug",
    [
        # Honorifics — Dr/Mr/Mrs/Ms/Prof must be stripped from canonical
        ("Dr. Sara Chen approves the budget.", "sara_chen"),
        ("Mr. John Smith owns the project schedule.", "john_smith"),
        ("Mrs. Linda Park, CFO Delegate, signs off.", "linda_park"),
        ("Ms. Anika Patel approves all designs.", "anika_patel"),
        ("Prof. Daniel Wu owns the technical review.", "daniel_wu"),
        # Suffix names (Jr/Sr/II/III/IV/V) — suffix must be stripped
        ("Robert Brown Jr. signs off on procurement.", "robert_brown"),
        ("James Park Sr. approves the budget.", "james_park"),
        ("William Carter III is responsible for delivery.", "william_carter"),
        ("Alfred Vanderbilt IV approves the contract.", "alfred_vanderbilt"),
        # Apostrophe surnames
        ("John O'Brien is the project sponsor.", "john_o_brien"),
        ("Sarah D'Souza, CFO, approves the budget.", "sarah_d_souza"),
        # Hyphenated surnames
        ("Maria Garcia-Lopez approves the contract.", "maria_garcia_lopez"),
        ("Jean-Paul Dubois is the executive sponsor.", "jean_paul_dubois"),
        # Three-word names
        ("Mary Anne Smith is the executive sponsor.", "mary_anne_smith"),
        ("Jose Maria Rodriguez approves the design.", "jose_maria_rodriguez"),
        # Initial middle name
        ("Sara G. Chen owns the rollout.", "sara_g_chen"),
        ("James K. Polk approves all releases.", "james_k_polk"),
        # Asian-style two-name
        ("Li Wei approves the technical design.", "li_wei"),
        ("Park Sung approves the rollout.", "park_sung"),
        # Honorific + single name (Dr. Smith / Ms. Park)
        ("Dr. Park approves the design.", "park"),
        # Last-name-first format — requires an explicit field label
        # ("Name:", "Approver:", "Sponsor:", ...) to disambiguate from
        # comma-separated name lists.
        ("Approver: Smith, John approves all changes.", "john_smith"),
        ("Name: Patel, Anika approves the design.", "anika_patel"),
        # Role/title + name patterns
        ("Approved by Marcus Lee, the project sponsor.", "marcus_lee"),
        ("The CFO is Sara Chen.", "sara_chen"),
    ],
)
def test_stakeholder_universal_name_shapes(text: str, expected_slug: str) -> None:
    """Every reasonable name shape on the planet should fire when paired
    with a role-context cue.
    """
    keys = _emit_stakeholders(text)
    assert f"stakeholder:{expected_slug}" in keys, (
        f"expected stakeholder:{expected_slug} in {sorted(keys)} from {text!r}"
    )


def test_stakeholder_multiple_per_sentence() -> None:
    """Multiple stakeholders in one paragraph must all be captured."""
    text = (
        "Dr. Smith owns the budget. Ms. Park approves it. "
        "Mr. Lee signs off on procurement."
    )
    keys = _emit_stakeholders(text)
    for expected in ("stakeholder:smith", "stakeholder:park", "stakeholder:lee"):
        assert expected in keys, f"missing {expected} from {sorted(keys)}"


# ─── B. Customer international suffix universality ───


@pytest.mark.parametrize(
    "text,expected_substring",
    [
        # International corporate suffixes
        ("Customer: Siemens AG", "siemens_ag"),
        ("Client: Nokia Oyj", "nokia_oyj"),
        ("Account: Sony K.K.", "sony"),
        ("Buyer: BHP Pty Ltd", "bhp_pty_ltd"),
        ("Company: Volvo AB", "volvo_ab"),
        ("Customer: Philips BV", "philips_bv"),
        ("Client: Maersk ApS", "maersk_aps"),
        ("Account: Wipro Pvt Ltd", "wipro_pvt_ltd"),
        ("Customer: BASF SE", "basf"),
        ("Client: SAP SE", "sap"),
        # German GmbH
        ("Company: Bosch GmbH", "bosch_gmbh"),
        # US standard suffixes (regression — must still work)
        ("Customer: Acme Corp", "acme_corp"),
        ("Client: Globex LLC", "globex_llc"),
        ("Account: Initech, Inc.", "initech_inc"),
        ("Company: Stark Industries", "stark_industries"),
        ("Company: Wonka Corporation", "wonka_corporation"),
    ],
)
def test_customer_international_corporate_suffixes(
    text: str, expected_substring: str,
) -> None:
    keys = _emit_customer_from_label(text)
    assert any(expected_substring in k for k in keys), (
        f"expected substring {expected_substring!r} in {sorted(keys)} from {text!r}"
    )


# ─── C. Money multi-currency universality ───


@pytest.mark.parametrize(
    "text,expected_amount",
    [
        # USD (regression)
        ("$1,847,250 total deal", 1847250),
        ("$1.5M commitment", 1500000),
        ("$250K threshold", 250000),
        # Euro symbol
        ("€1,500,000 European budget", 1500000),
        ("€2.5M total", 2500000),
        # Pound symbol
        ("£1.5M UK contract", 1500000),
        ("£500K consulting fee", 500000),
        # Yen symbol
        ("¥100,000,000 Japan facility", 100000000),
        ("¥50M Tokyo office", 50000000),
        # ISO-code prefixed
        ("USD 1,847,250 final", 1847250),
        ("EUR 750000 European budget", 750000),
        ("GBP 1.5M UK contract", 1500000),
        ("CHF 250K Swiss budget", 250000),
        ("CAD 500000 Canada office", 500000),
        ("AUD 2M Sydney project", 2000000),
        ("JPY 100000000 Japan", 100000000),
        # ISO-code suffixed
        ("500000 EUR allocated", 500000),
        ("1,847,250 USD final", 1847250),
        ("250000 GBP UK budget", 250000),
        # Nordic / Eastern Europe
        ("SEK 100000 Sweden", 100000),
        ("NOK 50000 Norway", 50000),
        ("DKK 25000 Denmark", 25000),
        # Asia
        ("INR 10000000 India project", 10000000),
        ("CNY 5000000 China facility", 5000000),
    ],
)
def test_money_multi_currency(text: str, expected_amount: int) -> None:
    """Every major currency the parser will plausibly see in
    international deal documents must normalize to the absolute
    dollar-equivalent integer (no currency code in the slug — that's
    a downstream concern with FX rates)."""
    keys = _emit_money_keys(text)
    assert f"money:{expected_amount}" in keys, (
        f"expected money:{expected_amount} in {sorted(keys)} from {text!r}"
    )


# ─── D. Date quarter / fiscal-year universality ───


@pytest.mark.parametrize(
    "text,expected_quarter",
    [
        ("Target close Q3 2026", "2026-Q3"),
        ("Q1 FY26 kickoff", "2026-Q1"),
        ("Phase 2 lands Q4 2027", "2027-Q4"),
        ("3Q26 deliverable", "2026-Q3"),
        ("1Q2027 milestone", "2027-Q1"),
        ("Q2/2026 review", "2026-Q2"),
        ("Q3-2026 hypercare", "2026-Q3"),
    ],
)
def test_date_quarter_notation(text: str, expected_quarter: str) -> None:
    """Quarter notation in any common form must produce a
    `quarter:YYYY-Qn` AND a `milestone:YYYY-Qn` (quarters are
    inherently timeline markers)."""
    keys = _emit_date_keys(text)
    assert f"quarter:{expected_quarter}" in keys, (
        f"expected quarter:{expected_quarter} in {sorted(keys)} from {text!r}"
    )
    assert f"milestone:{expected_quarter}" in keys, (
        f"expected milestone:{expected_quarter} in {sorted(keys)} from {text!r}"
    )


@pytest.mark.parametrize(
    "text,expected_year",
    [
        ("FY26 budget approval", 2026),
        ("FY2027 final approval", 2027),
        ("FY-26 quarterly review", 2026),
        ("Fiscal Year 2026 final approval", 2026),
        ("fiscal year 2027 plan", 2027),
        ("Q1 FY26 kickoff", 2026),
    ],
)
def test_fiscal_year_notation(text: str, expected_year: int) -> None:
    """Fiscal year notation in common forms must produce a
    `fiscal_year:fyYYYY` key."""
    keys = _emit_date_keys(text)
    assert f"fiscal_year:fy{expected_year}" in keys, (
        f"expected fiscal_year:fy{expected_year} in {sorted(keys)} from {text!r}"
    )


# ─── E. Adversarial — extractors must NOT fire on these ───


@pytest.mark.parametrize(
    "text",
    [
        # Money: sub-$100 noise / variable placeholders
        "Page $5",
        "$VAR placeholder",
        "$.99 item",
        # Customer: generic / no suffix
        "Customer: TBD",
        "Company: Various",
        # Stakeholder: department / template phrases (regression check)
        "Workplace Operations approves",
        "Help Desk owns the issue",
        "Due Date approves the change",
        # Date: not actual dates
        "Status: 2026-XX-XX pending",
    ],
)
def test_extractors_do_not_fire_on_noise(text: str) -> None:
    """Concrete noise patterns that earlier versions of these
    extractors falsely matched. Pinning so they don't regress."""
    all_keys = (
        _emit_money_keys(text)
        | _emit_customer_from_label(text)
        | _emit_stakeholders(text)
        | _emit_date_keys(text)
    )
    # No real-money / no customer / no stakeholder for these.
    # (date noise like "2026-XX-XX" is allowed to emit no keys.)
    junk_keys = {
        k for k in all_keys
        if k.startswith(("money:", "customer:", "stakeholder:"))
    }
    assert not junk_keys, (
        f"adversarial input produced junk: {sorted(junk_keys)} from {text!r}"
    )
