"""Equipment is not a place, and recycled pipeline output is not evidence.

Two rejection rules pinned here, both found in production on 2026-08-27:

* deal 222b2173 minted a ``physical_site`` named ``"1 16u wall mounted"``
  from the scope line "install 1 vendor supplied 16u wall mounted hinged
  data rack ...". A rack specification was sold to a PM as a job site.
* the same atom's source text was ``context.prior_scope_process_v1...`` —
  the pipeline's own prior output. Authority capping demotes those to
  rank 40 but never stopped them being minted as places.

The rules must reject equipment WITHOUT rejecting genuinely named
facilities, so every case below carries its counter-case: "Data Center 3"
and "Cabinet Room" contain equipment-adjacent nouns and must survive.
"""
import pytest

from app.core import site_plausibility as P


REJECTED_EQUIPMENT = [
    "1 16u wall mounted",
    "install 1 vendor supplied 16u wall mounted hinged data rack",
    "48-port switch",
    "patch panel A",
    "12U cabinet",
    "firewall",
]

KEPT_REAL_SITES = [
    "Clayton Homes of Chillicothe",
    "Clayton Homes The Reserve on Sleepy Hollow",
    "Wind Creek Atmore",
    "Jacksonville Campus",
    "Building 900",
    "Goleta Office",
    # place nouns redeem an equipment-adjacent phrase
    "Data Center 3",
    "Cabinet Room",
    "Server Room",
]


@pytest.mark.parametrize("name", REJECTED_EQUIPMENT)
def test_equipment_is_not_a_site(name):
    assert P.is_equipment_shaped(name) is True


@pytest.mark.parametrize("name", KEPT_REAL_SITES)
def test_real_facilities_survive(name):
    assert P.is_equipment_shaped(name) is False


@pytest.mark.parametrize("dress", [
    "1 16U wall mounted",
    "1 16u WALL MOUNTED",
    "1  16u  wall  mounted",
    "1 16u wall mounted",   # NBSP
])
def test_equipment_rejection_is_dress_blind(dress):
    assert P.is_equipment_shaped(dress) is True


def test_fixture_is_non_vacuous():
    """The kept list must actually exercise the redemption branch, or this
    suite would pass with a rule that rejects nothing."""
    redeemed = [n for n in KEPT_REAL_SITES if P.has_equipment_token(n)]
    assert redeemed, "no kept case contains an equipment token — rule untested"


SERIALIZED = [
    "context.prior_scope_process_v1.orbitbriefAudit.evidenceMap.device:rack[5].text: install",
    "context.prior_scope_process_v1.sowHandoff.scope_in[6].status: active",
    "  context.crm.deal_name: 000043",
]

NOT_SERIALIZED = [
    "Customer: Acme Corp",
    "context clues in the manager office",
    "Contextual Networks Inc",
]


@pytest.mark.parametrize("text", SERIALIZED)
def test_recycled_pipeline_output_is_not_a_site(text):
    assert P.is_serialized_source(text) is True


@pytest.mark.parametrize("text", NOT_SERIALIZED)
def test_prose_that_merely_says_context_survives(text):
    assert P.is_serialized_source(text) is False


BOTH_MINTERS_REAL_CASES = [
    # every physical_site atom deal 222b2173 produced on 2026-08-27 22:03,
    # AFTER the first (incomplete) fix shipped — the reason this module exists
    ("1 16u wall mounted", "equipment_not_a_place"),
    ("1 28U Wall Mounted", "equipment_not_a_place"),
    ("12 pair of pendant speakers", "equipment_not_a_place"),
    ("3 sonance pendant subwoofer", "equipment_not_a_place"),
]


@pytest.mark.parametrize("name,reason", BOTH_MINTERS_REAL_CASES)
def test_production_cases_are_rejected_with_a_recorded_reason(name, reason):
    assert P.rejects_as_site(name) == reason


def test_recycled_output_rejected_via_anchor_text():
    assert P.rejects_as_site(
        "South Warehouse", "context.prior_scope_process_v1.sowHandoff.x: y"
    ) == "recycled_pipeline_output"


def test_a_kept_site_returns_empty_reason_not_false():
    """The contract is a reason string, so a rejection is always recordable."""
    assert P.rejects_as_site("Clayton Homes of Chillicothe") == ""


def test_rules_are_idempotent():
    name = "1 16u wall mounted"
    assert P.is_equipment_shaped(name) == P.is_equipment_shaped(name)
