"""Pin the reconciliation that deal 010215 needed and nobody was doing.

The emails said "10" nine times and a quantity entity of 10 was extracted and
kept. The site layer resolved six addresses -- four of them two addresses fused,
one truncated, three sites missing. Both facts sat in the same envelope and
nothing compared them.
"""

from __future__ import annotations

from app.core.site_count_reconcile import reconcile_site_count, stated_site_counts


class A:
    def __init__(self, raw_text):
        self.raw_text = raw_text


REAL = [
    A("We need to have 10 timeclocks installed for Marion County SD in SC."),
    A("I have created SOW's  for each of the ten locations."),
    A("(I.E., $305 per site x 10 sites = $3,050.00)"),
    A("I will put all 10 sites on the same pSOW so we aren't drafting 10 individual pSOWs."),
    A("The plan would be to have the same tech knock out all 10 sites over the course of 3 days."),
]


def test_it_finds_the_count_the_documents_state():
    got = {n for n, _ in stated_site_counts(REAL)}
    assert 10 in got


def test_it_reads_the_word_ten_as_well_as_the_digit():
    assert 10 in {n for n, _ in stated_site_counts([A("SOW's for each of the ten locations.")])}


def test_the_010215_case_is_flagged():
    out = reconcile_site_count(REAL, resolved_sites=6)
    assert out["stated"] == 10
    assert out["resolved"] == 6
    assert out["agrees"] is False
    assert "10" in out["reason"] and "6" in out["reason"]
    # The finding must carry the sentence that supports it, not just a number.
    assert out["evidence"] and "10" in out["evidence"][0]


def test_agreement_is_reported_as_agreement():
    out = reconcile_site_count(REAL, resolved_sites=10)
    assert out["agrees"] is True


def test_no_stated_count_is_UNKNOWN_not_agreement():
    # The distinction this module exists for: silence is not consent. A deal that
    # never says how many sites it has has not confirmed our number.
    out = reconcile_site_count([A("Please install the clocks.")], resolved_sites=4)
    assert out["stated"] is None
    assert out["agrees"] is None


def test_the_most_repeated_claim_wins():
    atoms = [A("3 sites"), A("10 sites"), A("10 sites"), A("10 sites")]
    assert reconcile_site_count(atoms, 10)["stated"] == 10


def test_large_numbers_are_not_site_counts():
    # Part numbers and dollar figures sit next to the word "site" constantly.
    atoms = [A("Part 94575001 site kit"), A("$3,050 for the site")]
    assert reconcile_site_count(atoms, 2)["stated"] is None


def test_evidence_is_capped_but_present():
    long = A("x" * 900 + " 10 sites")
    out = reconcile_site_count([long], 6)
    assert out["agrees"] is False
    assert len(out["evidence"][0]) <= 240


# ── deal 010264: a national estate, and two counts of different things ──


class _A:
    def __init__(self, text):
        self.raw_text = text


def test_a_deal_may_have_more_than_two_hundred_sites():
    """The ceiling used to be 200, so a nationwide rollout could not state
    its own size. Deal 010264 says "795 locations" in four documents and
    the reconciler threw every one away as "a part number or a dollar
    figure", then reported `stated: 1` and called it agreed."""
    atoms = [_A("Up to 795 locations throughout the USA, see Exhibits")]
    assert stated_site_counts(atoms) == [
        (795, "Up to 795 locations throughout the USA, see Exhibits")]
    # four digits, and a thousands separator
    assert stated_site_counts([_A("The outreach to 1500 sites")])[0][0] == 1500
    assert stated_site_counts(
        [_A("rolled out to 1,200 stores")])[0][0] == 1200


def test_the_only_machine_readable_count_may_be_in_brackets():
    """The proposal writes "one thousand five hundred (795) United States
    locations" -- the words and the digits disagree, and only the digits
    are readable."""
    out = stated_site_counts(
        [_A("up to one thousand five hundred (795) United States locations.")])
    assert out and out[0][0] == 795


def test_a_clock_is_not_a_place():
    """1,500 clocks stand at ~795 locations because ~600 addresses carry
    more than one. A pattern that reads "1,500 time clocks" as a site count
    reproduces the error that put "one thousand five hundred (795)
    locations" into a customer-facing proposal, twice."""
    assert stated_site_counts([_A("Replace 1,400-1,500 time clocks at "
                                  "790-800 sites")])[0][0] == 800
    assert not stated_site_counts([_A("We need to have 10 timeclocks "
                                      "installed for Marion County")])
    # ...and a street number is not a count of anything
    assert not stated_site_counts([_A("1500 LOUISIANA ST | HOUSTON | TX")])


def test_documents_that_argue_with_each_other_are_not_reconciled():
    """Picking the most-repeated claim reports a winner where there is a
    dispute -- the same silence this module exists to break, moved one
    level up. On 010264 the corpus says 1500 nine times and 795 six, and
    those count different things."""
    atoms = ([_A("The outreach to 1500 sites is expensive")] * 9
             + [_A("Includes outreach to all 795 locations")] * 6)
    out = reconcile_site_count(atoms, resolved_sites=1)
    assert out["stated"] is None
    assert out["agrees"] is None
    assert out["rival_counts"] == {1500: 9, 795: 6}
    assert "disagree" in out["reason"]
    assert len(out["evidence"]) == 2

    # A small incidental count is not a rival. "Four locations" nine times
    # is a day's schedule, not a claim about a nationwide estate.
    atoms = ([_A("The outreach to 1500 sites is expensive")] * 9
             + [_A("we have four locations that day")] * 9)
    out = reconcile_site_count(atoms, resolved_sites=1)
    assert out["stated"] == 1500, "a day's schedule is not a rival claim"
