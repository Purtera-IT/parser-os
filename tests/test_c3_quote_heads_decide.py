"""The quote's numbers come from the super heads, not from wording rules.

Before this, ``ml/c3/quote_heads.py`` read none of the head answers parser-os sends it: every
line arrives with ``reads`` (read:labor_hours, read:crew_size, read:tech_level, read:billing_type,
read:work_window, col:supplier) and the quote derived hours, crew and the after-hours flag from
regexes over whatever the brief had put in ``scope_of_work``. Deal 010452 came back priced at a
family median while the hours head had marked 1019 lines as holding an hours statement.
"""
from __future__ import annotations

import json

import pytest

from ml.c3.quote_heads import QuoteHeads


@pytest.fixture
def heads(tmp_path):
    bundle = {
        "version": "test-1",
        "embed_model": "stub",
        "wording_model": None,
        "descriptions": {"survey": "site survey", "ap": "access points"},
        "description_embeddings": [[1.0, 0.0], [0.0, 1.0]],
        "families": {
            "survey": {"usual_techs": {"1": 9}, "median_hours_per_tech": 4,
                       "usual_rate_line": "L2 Networking", "usual_price": 115},
            "ap": {"usual_techs": {"2": 9}, "median_hours_per_tech": 6, "per_unit_crew_hours": 0.5,
                   "usual_rate_line": "L1 Cabling", "usual_price": 95},
        },
        "profiles": {},
        "quantity_weights": [1.0] * 10,
        "count_scaled_families": ["ap"],
    }
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(bundle))
    qh = QuoteHeads(str(path))
    qh._embed = lambda texts: [[1.0, 0.0] for _ in texts]   # always the survey family
    qh.wording = None
    return qh


def _payload(atoms, work_ids=(), site_count=1, deal_reads=None):
    return {"deal_id": "0001", "deal_name": "000001 - Test", "account": "", "atoms": list(atoms),
            "site_count": site_count,
            "heads": {"asked": True, "deal": deal_reads or {}},
            "c3_brief": {"sections": {"scope_of_work": [{"id": i} for i in work_ids]},
                         "job_details": [], "left_out": [], "also_mentioned": []}}


def _atom(aid, text, reads=None, **kw):
    a = {"id": aid, "text": text, "type": kw.pop("type", "scope_item")}
    if reads:
        a["reads"] = reads
    a.update(kw)
    return a


# ── the head vetoes a wording match ────────────────────────────────────────────

def test_an_hours_head_that_says_absent_vetoes_the_wording_rule(heads):
    """"Response within 24 hours" is not the job's effort, and the head already said so."""
    a = _atom("a1", "Support response within 24 hours per site",
              {"read:labor_hours": ["_absent", 0.97]})
    got = heads.quote(_payload([a], work_ids=["a1"]), word=False)
    assert got["lines"][0]["hours_per_tech"] == 4            # the family median, not 24
    assert "hours_per_tech" in got["heads"]["fell_back"]


def test_a_crew_head_that_says_absent_vetoes_the_crew_wording_rule(heads):
    a = _atom("a1", "Their 3 technicians will escort us", {"read:crew_size": ["_absent", 0.95]})
    got = heads.quote(_payload([a], work_ids=["a1"]), word=False)
    assert got["lines"][0]["techs"] == 1                     # the family's usual crew
    assert "techs" in got["heads"]["fell_back"]


# ── the head widens the search past the brief's section ───────────────────────

def test_a_line_the_hours_head_marked_is_read_when_the_work_section_states_nothing(heads):
    """010452: nothing in scope_of_work stated hours, so every number was a family median --
    while the hours head had marked lines that stated them outright."""
    work = _atom("w1", "Install the access points as scoped")
    marked = _atom("m1", "Each site takes approximately 6 hours", {"read:labor_hours": ["present", 0.93]})
    got = heads.quote(_payload([work, marked], work_ids=["w1"]), word=False)
    line = got["lines"][0]
    assert line["hours_per_tech"] == 6
    assert got["heads"]["used"]["hours_per_tech"] == {"head": "read:labor_hours", "line": "m1"}
    assert "read:labor_hours marked" in line["why_hours"]


def test_a_marked_line_the_brief_left_out_is_not_priced_from(heads):
    """Our own SOW and a left-out line still carry numbers; they are not the deal's statement."""
    work = _atom("w1", "Install the access points as scoped")
    ours = _atom("m1", "Each site takes approximately 6 hours",
                 {"read:labor_hours": ["present", 0.93]}, left_out=True)
    got = heads.quote(_payload([work, ours], work_ids=["w1"]), word=False)
    assert got["lines"][0]["hours_per_tech"] == 4
    assert "hours_per_tech" in got["heads"]["fell_back"]


def test_the_work_section_is_still_read_first(heads):
    """The head widens the search; it does not reorder a deal that already states its hours."""
    work = _atom("w1", "approximately 8 hours per site", {"read:labor_hours": ["present", 0.80]})
    other = _atom("m1", "roughly 2 hours each", {"read:labor_hours": ["present", 0.99]})
    got = heads.quote(_payload([work, other], work_ids=["w1"]), word=False)
    assert got["lines"][0]["hours_per_tech"] == 8
    assert got["heads"]["used"]["hours_per_tech"]["line"] == "w1"


# ── the day-type head replaces the after-hours wording rule ───────────────────

def test_the_window_head_replaces_the_after_hours_wording_rule(heads):
    a = _atom("a1", "Cutover runs overnight after the store closes",
              {"read:work_window": [["after_hours", "overnight"], 0.91]})
    got = heads.quote(_payload([a], work_ids=["a1"]), word=False)
    assert any("outside business hours" in f for f in got["flags"])
    assert got["heads"]["used"]["day_type"]["head"] == "read:work_window"
    assert got["work_window"]["after_hours"]["p"] == 0.91


def test_an_excluded_weekend_no_longer_flags_after_hours_work(heads):
    """The wording rule fires on "weekends" inside an exclusion. The head reads the window the
    work happens in, and answers business_hours here."""
    a = _atom("a1", "Weekends and holidays are excluded; work is during regular business hours",
              {"read:work_window": [["business_hours"], 0.88]})
    got = heads.quote(_payload([a], work_ids=["a1"]), word=False)
    assert not any("outside business hours" in f or "after-hours" in f for f in got["flags"])


def test_a_blackout_is_a_window_to_avoid_not_one_to_work_in(heads):
    a = _atom("a1", "No work during the December freeze", {"read:work_window": [["blackout"], 0.9]})
    got = heads.quote(_payload([a], work_ids=["a1"]), word=False)
    assert any("blackout window stated" in f for f in got["flags"])
    assert not any("outside business hours" in f for f in got["flags"])


def test_the_wording_rule_still_runs_when_the_window_head_did_not_answer(heads):
    a = _atom("a1", "Install after-hours once the floor is clear")
    got = heads.quote(_payload([a], work_ids=["a1"]), word=False)
    assert "after-hours work mentioned" in got["flags"]


# ── billing, level, supplier ──────────────────────────────────────────────────

def test_the_deal_billing_head_answers_once_and_that_answer_stands(heads):
    a = _atom("a1", "Install the access points", {"read:billing_type": ["fixed", 0.6]})
    got = heads.quote(_payload([a], work_ids=["a1"], deal_reads={"jdg:billing_type": ["t_and_m", 0.94]}),
                      word=False)
    assert got["billing"]["shape"] == "t_and_m"
    assert got["billing"]["by"] == "jdg:billing_type (the deal)"


def test_without_a_deal_answer_the_line_head_is_counted_over_the_work_lines(heads):
    atoms = [_atom("a1", "Billed time and materials", {"read:billing_type": ["t_and_m", 0.9]}),
             _atom("a2", "Fixed fee for the survey", {"read:billing_type": ["fixed", 0.4]}),
             _atom("a3", "Also time and materials", {"read:billing_type": ["t_and_m", 0.8]})]
    got = heads.quote(_payload(atoms, work_ids=["a1", "a2", "a3"]), word=False)
    assert got["billing"]["shape"] == "t_and_m"
    assert got["billing"]["by"] == "read:billing_type over the work lines"


def test_a_level_the_level_head_marked_sets_the_rate_lines_level(heads):
    a = _atom("a1", "Needs an L3 engineer onsite", {"read:tech_level": ["present", 0.92]})
    got = heads.quote(_payload([a], work_ids=["a1"]), word=False)
    assert got["lines"][0]["rate_line"] == "L3 Networking"
    assert got["heads"]["used"]["rate_line_level"]["level"] == 3


def test_a_level_nobody_marked_leaves_the_rate_line_alone(heads):
    a = _atom("a1", "Needs an L3 engineer onsite")
    got = heads.quote(_payload([a], work_ids=["a1"]), word=False)
    assert got["lines"][0]["rate_line"] == "L2 Networking"


def test_the_supplier_head_says_who_furnishes_the_parts(heads):
    a = _atom("a1", "Customer provides the access points", {"col:supplier": ["customer", 0.9]})
    got = heads.quote(_payload([a], work_ids=["a1"]), word=False)
    assert any("customer-furnished" in f for f in got["flags"])
    assert got["heads"]["used"]["customer_furnished"]["n"] == 1


# ── the audit trail ───────────────────────────────────────────────────────────

def test_a_quote_says_which_numbers_a_head_decided_and_which_fell_back(heads):
    atoms = [_atom("a1", "Each site takes approximately 6 hours", {"read:labor_hours": ["present", 0.93]})]
    got = heads.quote(_payload(atoms, work_ids=["a1"]), word=False)
    assert got["heads"]["used"]["hours_per_tech"]["head"] == "read:labor_hours"
    assert "techs" in got["heads"]["fell_back"]
    assert got["heads"]["asked"] == ["read:labor_hours"]


def test_a_number_head_value_is_taken_as_given(heads):
    """A NUMBER head sends its own value; nothing re-reads the line to second-guess it."""
    a = _atom("a1", "see the attached estimate", {"read:labor_hours": ["present", 0.9, 12]})
    got = heads.quote(_payload([a], work_ids=["a1"]), word=False)
    assert got["lines"][0]["hours_per_tech"] == 12


def test_a_multi_label_head_sends_a_probability_per_label(heads):
    """``read:work_window`` sends ``[labels, {label: p}]``. Reading that map with ``float()`` makes
    every multi-label confidence zero, which is what the first cut of this did: a window the head
    gave 0.58 was reported as "p 0.00"."""
    a = _atom("a1", "Cutover runs overnight",
              {"read:work_window": [["after_hours", "overnight"],
                                    {"_absent": 0.26, "business_hours": 0.02, "after_hours": 0.75,
                                     "overnight": 0.33, "weekend": 0.23, "blackout": 0.34}]})
    got = heads.quote(_payload([a], work_ids=["a1"]), word=False)
    assert got["work_window"]["after_hours"]["p"] == 0.75
    assert got["work_window"]["overnight"]["p"] == 0.33
    # blackout scored 0.34 but the head did not assert it: it is not a window of this deal's
    assert "blackout" not in got["work_window"]
    assert "p 0.75" in next(f for f in got["flags"] if "outside business hours" in f)


def test_a_split_billing_vote_is_reported_as_split_not_as_an_answer(heads):
    """010358's work lines split fixed 0.496 / t_and_m 0.44. Calling that "fixed" is how a summary
    came to read "Fixed fee" over a t_and_m deal: a head that is unsure must not sound sure."""
    atoms = [_atom("a1", "Fixed fee for the survey", {"read:billing_type": ["fixed", 0.52]}),
             _atom("a2", "Billed time and materials", {"read:billing_type": ["t_and_m", 0.48]})]
    got = heads.quote(_payload(atoms, work_ids=["a1", "a2"]), word=False)
    assert got["billing"]["shape"] is None
    assert "split" in got["billing"]["by"]
    assert any("billing shape unsettled" in f for f in got["flags"])
    assert "billing" in got["heads"]["fell_back"]


def test_a_window_read_outside_the_work_section_says_so(heads):
    """A blackout named in a quoted copy is not necessarily this job's window. The flag reports
    where the head read it instead of guessing a threshold."""
    work = _atom("w1", "Install the access points")
    aside = _atom("x1", "No work during the December freeze", {"read:work_window": [["blackout"], 0.9]})
    got = heads.quote(_payload([work, aside], work_ids=["w1"]), word=False)
    assert "on a line outside the work section" in next(f for f in got["flags"] if "blackout" in f)
