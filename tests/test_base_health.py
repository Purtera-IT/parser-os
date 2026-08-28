"""Base health gates: the measurement logic, on synthetic envelopes only.

Never touches blob. Every envelope here is built in-process, so these tests
pin the RULES -- what counts as a defect, where each threshold flips, and
which direction drift fires in -- independently of what the corpus happens to
contain today.

Doctrine under test: **a broken audit must not look like a clean base**
(docs/IMAGE_GATE_LOOP.md, silent-zero-for-infra). The malformed-envelope
tests are the load-bearing ones: an envelope that cannot be measured must
raise, not return zeros, because zeros from a dead pipeline are
indistinguishable from zeros from a healthy one.
"""
from __future__ import annotations

import json

import pytest

from tools.base_health import (
    DRIFT_ABS_FLOOR,
    METRIC_BY_KEY,
    DealMeasure,
    EnvelopeError,
    build_baseline,
    display_names,
    drift_breaches,
    evaluate,
    filename_is_noncontract,
    is_composed_field_summary,
    looks_like_serialized_structure,
    measure_envelope,
    normalize,
    pick_sample,
)


# ── helpers ─────────────────────────────────────────────────────────

def atom(**kw):
    """One atom with sane defaults; override only what a test is about."""
    a = {
        "id": kw.pop("id", "atm_test"),
        "artifact_id": kw.pop("artifact_id", "art_1"),
        "atom_type": kw.pop("atom_type", "requirement"),
        "authority_class": kw.pop("authority_class", "machine_extractor"),
        "text": kw.pop("text", "some ordinary document prose"),
        "verified": kw.pop("verified", "verified"),
    }
    a.update(kw)
    return a


def envelope(atoms, documents=None):
    return {
        "project_id": "p",
        "documents": documents if documents is not None else [
            {"artifact_id": "art_1", "filename": "Master Services Agreement.pdf"}
        ],
        "atoms": atoms,
    }


def counts_of(atoms, documents=None):
    return measure_envelope(envelope(atoms, documents), deal="d").counts


# ── unmeasurable envelopes are ERRORS, never clean zeros ────────────

@pytest.mark.parametrize("bad, why", [
    (None, "null"),
    ([], "a list"),
    ("{}", "a string"),
    ({}, "no atoms key"),
    ({"atoms": None}, "atoms is null"),
    ({"atoms": {}}, "atoms is an object"),
    ({"atoms": []}, "atoms is empty"),
])
def test_unmeasurable_envelope_raises_rather_than_passing(bad, why):
    """The whole point: a broken audit must not look like a clean base."""
    with pytest.raises(EnvelopeError):
        measure_envelope(bad, deal="d")


def test_empty_atom_list_is_an_error_not_a_perfect_score():
    """Zero atoms is a broken compile. Scoring it 0/0 would be a silent pass."""
    with pytest.raises(EnvelopeError, match="zero atoms"):
        measure_envelope({"documents": [], "atoms": []}, deal="d")


def test_errored_deals_are_excluded_from_totals_and_never_baselined():
    good = measure_envelope(envelope([atom()]), deal="good")
    bad = DealMeasure(deal="bad", error="unparseable JSON")
    findings = {f.metric: f for f in evaluate([good, bad], None)}
    # The broken deal contributes no counts...
    assert findings["fabricated_names"].value == 0
    # ...and is not written into the baseline as if it had zero sites.
    assert list(build_baseline([good, bad])["deals"]) == ["good"]


# ── metric 1: unsupported contract authority ────────────────────────

@pytest.mark.parametrize("verified, expected", [
    ("verified", 0),
    ("unsupported", 1),
    ("failed", 1),
    ("partial", 1),
    (None, 1),
])
def test_rank100_requires_verified_support(verified, expected):
    c = counts_of([atom(authority_class="contractual_scope", verified=verified)])
    assert c["unsupported_contract_authority"] == expected


def test_lower_authority_may_be_unverified():
    """Only rank 100 has to prove itself; a machine extraction need not."""
    c = counts_of([atom(authority_class="machine_extractor", verified="unsupported")])
    assert c["unsupported_contract_authority"] == 0


# ── metric 2: rank-100 from a non-contract source ───────────────────

@pytest.mark.parametrize("filename, flagged", [
    ("Clayton Homes CALC.xlsx", True),
    ("ROM_Estimate_v3.xlsx", True),
    ("Deal Kit - pricing sheet.xlsx", True),
    ("Request for Proposal July2026.docx", True),
    ("Supplier Response Workbook.xlsx", True),
    ("Budgetary quote.pdf", True),
    ("Master Services Agreement.pdf", False),
    ("Signed SOW - executed.pdf", False),
    ("Purchase Order 4500123.pdf", False),
    # Contract evidence wins over an incidental non-contract word, so the
    # metric never trains people to ignore it.
    ("MSA Amendment 2 (draft cover).pdf", False),
    ("Statement of Work.pdf", False),
    ("", False),
])
def test_noncontract_filename_markers(filename, flagged):
    assert bool(filename_is_noncontract(filename)) is flagged


def test_marker_matching_is_whole_token_not_substring():
    """'rom' must not match inside 'Bathroom' or 'Promotion'."""
    assert filename_is_noncontract("Bathroom Promotion Schedule.pdf") is None


def test_rank100_from_noncontract_source_counts_only_rank100():
    docs = [{"artifact_id": "art_calc", "filename": "Deal Kit CALC.xlsx"}]
    c = counts_of([
        atom(id="a", artifact_id="art_calc", authority_class="contractual_scope"),
        atom(id="b", artifact_id="art_calc", authority_class="vendor_quote"),
    ], documents=docs)
    assert c["rank100_from_noncontract_source"] == 1


def test_unknown_artifact_is_not_flagged():
    """An atom whose document is missing has no filename evidence either way."""
    c = counts_of(
        [atom(artifact_id="art_missing", authority_class="contractual_scope")],
        documents=[],
    )
    assert c["rank100_from_noncontract_source"] == 0


# ── metric 3: fabricated names ──────────────────────────────────────

def test_name_present_in_source_text_is_supported():
    c = counts_of([atom(text="The Goleta Office needs a rack",
                        structured={"name": "Goleta Office"})])
    assert c["fabricated_names"] == 0


def test_name_absent_from_source_text_is_fabricated():
    c = counts_of([atom(text="309, Beaufort, SC 29901",
                        structured={"name": "Beaufort Office"})])
    assert c["fabricated_names"] == 1


def test_name_matching_is_dress_blind():
    """'ATL-HQ-01' and 'atl hq 01' are the same string, not a fabrication."""
    c = counts_of([atom(text="site atl_hq_01 rollout",
                        structured={"site_id": "ATL-HQ-01"})])
    assert c["fabricated_names"] == 0


def test_every_name_field_and_alias_list_is_checked():
    st = {
        "name": "Alpha", "facility_name": "Beta", "display_name": "Gamma",
        "site_name": "Delta", "site_id": "Epsilon",
        "names": ["Zeta"], "aliases": ["Eta"],
    }
    assert len(display_names(st)) == 7
    c = counts_of([atom(text="nothing here", structured=st)])
    assert c["fabricated_names"] == 7


def test_nested_labels_are_not_treated_as_names():
    """A nested classifier verdict is not a display name; counting it is noise."""
    st = {"name": "Goleta Office", "facility_label": {"label": "keep_facility"}}
    assert display_names(st) == ["Goleta Office"]
    c = counts_of([atom(text="Goleta Office", structured=st)])
    assert c["fabricated_names"] == 0


def test_non_string_and_empty_names_are_not_fabrications():
    c = counts_of([atom(text="x", structured={"name": None, "site_id": "", "names": [7]})])
    assert c["fabricated_names"] == 0


# ── the undecidable population is reported, never silently dropped ──

@pytest.mark.parametrize("text, composed", [
    ("site_id: HC-65 | facility: HC 65 | address: 6655 US 23", True),
    ("kind: physical_site | id: HC-2731", True),
    ("The Goleta Office needs a rack", False),
    ("309, Beaufort, SC 29901", False),
    ("Note: the site is closed", False),          # one pair, no pipe
    ("Alpha | Beta | Gamma", False),              # pipes, but no key: value
])
def test_composed_field_summary_detection(text, composed):
    assert is_composed_field_summary(text) is composed


def test_composed_summary_names_are_undecidable_not_fabricated():
    """The envelope does not serialize raw source text for roster atoms.

    Checking them anyway reported 1,518 real facility names as fabrications
    on one deal. They are excluded from the metric AND counted, so the
    exclusion is visible rather than a silent suppression.
    """
    m = measure_envelope(envelope([atom(
        atom_type="physical_site",
        text="site_id: HC-65 | facility: HC 65 | address: 6655 US 23",
        structured={"name": "Clayton Homes of Chillicothe",
                    "display_name": "Clayton Homes of Chillicothe"},
    )]), deal="d")
    assert m.counts["fabricated_names"] == 0
    assert m.names_undecidable == 2


# ── metric 4: street with no city ───────────────────────────────────

@pytest.mark.parametrize("structured, flagged", [
    ({"street_address": "6655 US 23", "city": ""}, True),
    ({"street_address": "6655 US 23"}, True),
    ({"street_address": "6655 US 23", "city": None}, True),
    ({"street_address": "6655 US 23", "city": "   "}, True),
    ({"street_address": "6655 US 23", "city": "Chillicothe"}, False),
    ({"street_address": "", "city": ""}, False),        # no street, no claim
    ({"city": "Chillicothe"}, False),
])
def test_street_without_city(structured, flagged):
    c = counts_of([atom(atom_type="physical_site", structured=structured)])
    assert c["street_without_city"] == (1 if flagged else 0)


def test_street_without_city_only_applies_to_sites():
    c = counts_of([atom(atom_type="requirement",
                        structured={"street_address": "6655 US 23", "city": ""})])
    assert c["street_without_city"] == 0


def test_site_atoms_are_counted_for_the_ratio_denominator():
    m = measure_envelope(envelope([
        atom(id="a", atom_type="physical_site", structured={"city": "X"}),
        atom(id="b", atom_type="physical_site", structured={"city": "Y"}),
        atom(id="c", atom_type="requirement"),
    ]), deal="d")
    assert m.site_atoms == 2


# ── metric 5: self-ingested structure at high authority ─────────────

@pytest.mark.parametrize("text, structural", [
    ("context.deal.sites[0]: something", True),
    ("site_registry[0].zip: 45601", True),
    ("a.b.c.d: value", True),
    ("report.final.pdf: attached", False),      # filename, not a key path
    ("www.example.com: see site", False),       # hostname, not a key path
    ("a.b: value", False),                      # too shallow to be structure
    ("The scope: install racks", False),
    ("", False),
])
def test_serialized_structure_shape_test(text, structural):
    assert looks_like_serialized_structure(text) is structural


@pytest.mark.parametrize("authority, flagged", [
    ("contractual_scope", True),
    ("pm_confirmed", True),
    ("customer_current_authored", True),
    ("approved_site_roster", True),
    ("vendor_quote", False),        # below the floor
    ("machine_extractor", False),
])
def test_self_ingestion_only_fires_at_or_above_the_authority_floor(authority, flagged):
    c = counts_of([atom(authority_class=authority,
                        text="context.deal.sites[0]: something")])
    assert c["self_ingested_high_authority"] == (1 if flagged else 0)


# ── thresholds: pass/fail boundaries ────────────────────────────────

def _corpus(**per_deal_counts):
    """One synthetic measured deal carrying exactly the counts given."""
    m = DealMeasure(deal="deal_x", atoms=100, site_atoms=per_deal_counts.pop("site_atoms", 0))
    m.counts = {k: 0 for k in METRIC_BY_KEY if k != "site_count_drift"}
    m.counts.update(per_deal_counts)
    return [m]


@pytest.mark.parametrize("value, breached", [(0, False), (5, False), (6, True)])
def test_unsupported_contract_authority_threshold_boundary(value, breached):
    """<= 5 passes, 6 fails. The bar is 5, not 4 or 6."""
    f = {x.metric: x for x in evaluate(_corpus(unsupported_contract_authority=value), None)}
    assert f["unsupported_contract_authority"].breached is breached


@pytest.mark.parametrize("key", [
    "rank100_from_noncontract_source",
    "fabricated_names",
    "self_ingested_high_authority",
])
@pytest.mark.parametrize("value, breached", [(0, False), (1, True)])
def test_zero_tolerance_metrics(key, value, breached):
    f = {x.metric: x for x in evaluate(_corpus(**{key: value}), None)}
    assert f[key].breached is breached
    assert METRIC_BY_KEY[key].threshold == 0


@pytest.mark.parametrize("bad, total, breached", [
    (64, 100, False),   # 64% — under the 65% placeholder
    (65, 100, False),   # exactly at it
    (66, 100, True),    # over
])
def test_street_without_city_ratio_boundary(bad, total, breached):
    f = {x.metric: x for x in
         evaluate(_corpus(street_without_city=bad, site_atoms=total), None)}
    assert f["street_without_city"].breached is breached


def test_street_ratio_with_no_sites_is_not_a_breach_but_reports_zero_denominator():
    """No site atoms means nothing was measured, not that the base is clean."""
    f = {x.metric: x for x in evaluate(_corpus(street_without_city=0, site_atoms=0), None)}
    assert f["street_without_city"].breached is False
    assert "0/0" in f["street_without_city"].display


# ── baseline drift: BOTH directions ─────────────────────────────────

def _measure(deal, sites):
    m = DealMeasure(deal=deal, atoms=10, site_atoms=sites)
    m.counts = {k: 0 for k in METRIC_BY_KEY if k != "site_count_drift"}
    return m


BASE = {"deals": {"d1": {"site_atoms": 438, "atoms": 2047}}}


def test_drift_catches_the_silent_site_loss():
    """The 438 -> 388 regression this file exists to catch."""
    offenders, checked = drift_breaches([_measure("d1", 388)], BASE)
    assert checked == 1
    assert len(offenders) == 1
    assert "LOST 50" in offenders[0]


def test_drift_catches_invented_sites_too():
    offenders, _ = drift_breaches([_measure("d1", 500)], BASE)
    assert len(offenders) == 1
    assert "GAINED 62" in offenders[0]


# 5% of 438 is 21.9 sites, so 21 is inside tolerance and 22 is outside --
# and it must be outside by the same margin in both directions.
@pytest.mark.parametrize("sites, breaches", [
    (438, False),   # exact
    (417, False),   # -21, inside
    (459, False),   # +21, inside
    (416, True),    # -22, outside
    (460, True),    # +22, outside
])
def test_drift_tolerance_boundary_is_symmetric(sites, breaches):
    offenders, _ = drift_breaches([_measure("d1", sites)], BASE)
    assert bool(offenders) is breaches


@pytest.mark.parametrize("sites, breaches", [(5, False), (6, True)])
def test_small_deals_get_an_absolute_floor(sites, breaches):
    """5% of 3 sites is 0.15, so percentage alone would fire on any change.

    The absolute floor is what keeps a 3 -> 4 site deal quiet while a real
    swing still reports.
    """
    base = {"deals": {"d1": {"site_atoms": 3, "atoms": 10}}}
    offenders, _ = drift_breaches([_measure("d1", sites)], base)
    assert bool(offenders) is breaches
    assert DRIFT_ABS_FLOOR == 2


def test_deal_absent_from_baseline_is_not_a_breach():
    """A new deal has no expectation to violate -- and is not silently checked."""
    offenders, checked = drift_breaches([_measure("brand_new", 999)], BASE)
    assert offenders == [] and checked == 0


def test_drift_is_skipped_entirely_without_a_baseline():
    f = {x.metric: x for x in evaluate(_measure_list(), None)}
    assert f["site_count_drift"].breached is False
    assert f["site_count_drift"].display == "no baseline"


def _measure_list():
    return [_measure("d1", 438)]


def test_baseline_roundtrips_through_json():
    payload = build_baseline([_measure("d1", 438), _measure("d2", 7)])
    reloaded = json.loads(json.dumps(payload))
    assert reloaded["deals"]["d1"]["site_atoms"] == 438
    offenders, checked = drift_breaches([_measure("d1", 300)], reloaded)
    assert checked == 1 and offenders


# ── misc invariants ─────────────────────────────────────────────────

def test_every_metric_carries_a_documented_rationale():
    for spec in METRIC_BY_KEY.values():
        assert spec.rationale.strip(), f"{spec.key} has no rationale"


def test_sampling_is_deterministic_and_spans_the_size_range():
    rows = [(f"deals/d{i:03d}/orbitbrief/latest/envelope.json", i * 1000) for i in range(100)]
    a, b = pick_sample(rows, 10), pick_sample(list(reversed(rows)), 10)
    assert a == b, "sample must not depend on listing order"
    assert len(a) == 10
    assert a[0][1] < a[-1][1], "sample must span small and large envelopes"
    assert pick_sample(rows, 0) == sorted(rows, key=lambda r: (r[1], r[0]))


def test_normalize_collapses_the_dressings_that_are_not_differences():
    assert normalize("St. Louis Office") == normalize("st_louis_office")
    assert normalize("ATL-HQ-01") == normalize("atl hq 01")
    assert normalize("  A  B  ") == "a b"
    assert normalize(None) == ""
