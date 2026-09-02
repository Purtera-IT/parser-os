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
