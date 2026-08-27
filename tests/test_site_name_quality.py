"""A site name a PM cannot read is a site the PM will not act on.

``site_naming`` recovers the document's own words, and the document's own
words are not always readable. Three failures reached production on deal
1e130077 (Jacksonville Campus ROM), compiled 2026-08-25:

    "building eight hundred 800"   the number, twice, in both forms
    "900"                          a name that is only a number
    seven pairs of identical names collapsing inside one deal

All three are pinned here, together with the invariant that makes the fix
safe: every word this module puts in a name is a word the evidence already
carried. Dress-invariance style follows ``test_head_invariance.py`` —
casing, parentheses, hyphens and NBSP are dress, and dress may not change
the decision. Non-vacuity is asserted throughout, so a regression that
guts the pass fails loudly instead of passing on empty output.
"""

from __future__ import annotations

import random

import pytest

from app.core.site_naming import (
    SITE_NAME_BARE_IDENTIFIER,
    SITE_NAME_DISAMBIGUATED,
    SITE_NAME_DUPLICATE,
    SITE_NAME_NUMBER_COLLAPSED,
    clean_site_name,
    collapse_number_word_duplication,
    recover_site_display_names,
    resolve_site_names,
    unsupported_name_tokens,
)

MILESTONE_LINE = (
    "Milestone: Building one hundred (100) Complete | Completed This Wave: "
    "nine hundred thirty eight (938) | Cumulative Completed: nine hundred "
    "thirty eight (938) | Remaining: two thousand nine hundred eighty "
    "seven (2,987)"
)


# ── 1. A number written twice is one number ─────────────────────────────

def test_parenthesised_spelling_collapses_to_the_digits():
    """"Building one hundred (100) Complete" is a careful document, not a
    name with two numbers in it. The digits are what is on the door."""
    assert collapse_number_word_duplication(
        "Building one hundred (100) Complete"
    ) == "Building 100 Complete"


def test_the_production_name_stops_being_gibberish():
    got = clean_site_name(
        "building eight hundred 800",
        source_text=MILESTONE_LINE,
        site_id="BUILDING-EIGHT-HUNDRED-800",
    )
    assert got.name == "Building 800"
    assert "eight hundred" not in got.name.casefold()
    assert SITE_NAME_NUMBER_COLLAPSED in got.flags


@pytest.mark.parametrize("dressed", [
    "eight hundred (800)",
    "Eight Hundred (800)",
    "EIGHT HUNDRED (800)",
    "eight-hundred (800)",
    "eight hundred (800)",
    "eight hundred [800]",
    "eight hundred  ( 800 )",
    "eight hundred 800",
    "800 (eight hundred)",
])
def test_duplication_collapses_in_every_dress(dressed):
    """Casing, parens, brackets, hyphens, NBSP and stray spaces are dress.
    The one number underneath is the decision."""
    got = collapse_number_word_duplication(dressed).replace(" ", " ").strip()
    assert got == "800", dressed


@pytest.mark.parametrize("phrase,value", [
    ("nine hundred thirty eight (938)", "938"),
    ("two thousand nine hundred eighty seven (2,987)", "2,987"),
    ("twenty-one (21)", "21"),
    ("four (4)", "4"),
    ("one hundred and five (105)", "105"),
])
def test_the_whole_spelled_run_goes_not_just_its_last_word(phrase, value):
    assert collapse_number_word_duplication(phrase) == value


def test_a_number_written_once_is_left_exactly_as_written():
    """The pass only ever deletes a redundant half. With nothing redundant
    there is nothing to delete — including for a name that merely contains
    a number word ("One Hundred Oaks")."""
    for untouched in (
        "Building 100",
        "Building one hundred",
        "One Hundred Oaks 100",
        "Prudential Center office",
        "Iron Mountain Data Centers AZS-1 - Scottsdale",
    ):
        assert collapse_number_word_duplication(untouched) == untouched


def test_a_mismatched_pair_is_two_facts_and_survives():
    """"one hundred (101)" is not a duplicated number, it is a typo or two
    different figures. Either way collapsing it would destroy evidence."""
    assert collapse_number_word_duplication(
        "Building one hundred (101)"
    ) == "Building one hundred (101)"


# ── 2. A bare number is not a name ──────────────────────────────────────

def test_the_source_type_word_names_the_number():
    """The word "Building" was sitting right next to the 900 the whole
    time. Production shipped "900"."""
    got = clean_site_name("900", source_text="Building: 900 |", site_id="900")
    assert got.name == "Building 900"
    assert got.is_named
    assert not got.flags


@pytest.mark.parametrize("source,expected", [
    ("Building: 900 |", "Building 900"),
    ("building 900", "building 900"),
    ("BUILDING 900", "Building 900"),
    ("Location: Building 900 |", "Building 900"),
    ("Building #900", "Building 900"),
    ("Building - 900", "Building 900"),
    ("occupies Building 900 through 2027.", "Building 900"),
])
def test_type_word_recovery_is_dress_invariant(source, expected):
    assert clean_site_name("900", source_text=source, site_id="900").name == expected


@pytest.mark.parametrize("source,name,expected", [
    ("Suite 210, 44 Wall St", "210", "Suite 210"),
    ("Store 43 remodel", "43", "Store 43"),
    ("Site 12 access window", "12", "Site 12"),
    ("Tower 3 riser", "3", "Tower 3"),
])
def test_the_type_word_is_whichever_one_the_source_used(source, name, expected):
    """Not "Building" every time — the module reads the word off the page.
    A list that only decides *whether* a word qualifies cannot leak a word
    of its own into a name."""
    assert clean_site_name(name, source_text=source, site_id=name).name == expected


def test_a_plural_type_word_names_one_building_not_several():
    got = clean_site_name(
        "400",
        source_text="Location: All Buildings | Wave 1 covers Buildings 400 and 500.",
        site_id="400",
    )
    assert got.name == "Building 400"


def test_a_number_with_no_type_word_keeps_the_identifier_and_says_so():
    """Abstention over invention. Nothing in the source puts a word next
    to this number, so nothing may be put next to it here."""
    got = clean_site_name(
        "900",
        source_text="Wave 4 runs July 1 to September 30, 2027.",
        site_id="900",
    )
    assert got.name == "900"
    assert SITE_NAME_BARE_IDENTIFIER in got.flags
    assert not got.is_named
    assert "building" not in got.name.casefold()


def test_the_word_building_is_never_conjured_from_the_type_list():
    """The strongest form of the rule: over sources that mention no place
    noun at all, no place noun may ever appear in the output."""
    for source in (
        "",
        "938 workstations replaced this wave.",
        "Wave 1C / Milestone 1",
        "Planned Dates: October 19 to November 1, 2026",
    ):
        got = clean_site_name("900", source_text=source, site_id="900")
        assert got.name == "900", source
        assert SITE_NAME_BARE_IDENTIFIER in got.flags


def test_the_identifier_may_supply_the_type_word_the_prose_omits():
    """``BUILDING-900`` is what the compile established, not a guess, so a
    row whose prose never says "building" can still be named from it."""
    got = clean_site_name(
        "900", source_text="Wave 4 delivery.", site_id="BUILDING-900",
    )
    assert got.name == "Building 900"
    assert SITE_NAME_BARE_IDENTIFIER not in got.flags
    assert not unsupported_name_tokens(got.name, "Wave 4 delivery.", "BUILDING-900")


def test_the_document_spelling_wins_over_the_identifier_round_trip():
    """``BUILDING-100`` comes back lowercased; the document printed
    "Building 100". Recovering the document's casing is the point of the
    module, not a style preference."""
    got = clean_site_name(
        "building 100",
        source_text="Phase: Wave 1A | Location: Building 100 | Planned Dates: September 8",
        site_id="BUILDING-100",
    )
    assert got.name == "Building 100"


def test_a_shouted_header_does_not_shout_back():
    """A source that prints a word in an all-caps header is not asking for
    the name to be all-caps. Short all-caps tokens stay — those are
    acronyms, and "HQ" is how the site is known."""
    assert clean_site_name(
        "annex 4", source_text="LOCATION AND ANNEX 4 SCHEDULE", site_id="ANNEX-4",
    ).name == "annex 4"
    assert clean_site_name(
        "hq campus", source_text="The HQ campus tour", site_id="HQ",
    ).name == "HQ campus"


# ── 3. Two rows, one name ───────────────────────────────────────────────

def test_identical_names_on_distinct_ids_are_flagged_not_merged():
    """Three atoms name the same building three ways. Merging them would
    drop rows the roster asserted; picking one silently would hide the
    duplication from the person who has to fix the source."""
    got = resolve_site_names([
        {"site_id": "900", "name": "900", "source_text": "Building: 900 |"},
        {"site_id": "BUILDING-900", "name": "Building 900",
         "source_text": "Location: Building 900 |"},
        {"site_id": "BUILDING-NINE-HUNDRED-900", "name": "Building Nine Hundred 900",
         "source_text": MILESTONE_LINE},
    ])
    assert {d.name for d in got.values()} == {"Building 900"}
    assert len(got) == 3
    for site_id, decision in got.items():
        assert SITE_NAME_DUPLICATE in decision.flags, site_id


def test_a_collision_is_split_only_by_a_token_the_evidence_holds():
    got = resolve_site_names([
        {"site_id": "BUILDING-100-AUSTIN", "name": "Building 100",
         "source_text": "Austin rollout", "city": "Austin"},
        {"site_id": "BUILDING-100-DENVER", "name": "Building 100",
         "source_text": "Denver rollout", "city": "Denver"},
    ])
    assert got["BUILDING-100-AUSTIN"].name == "Building 100 (Austin)"
    assert got["BUILDING-100-DENVER"].name == "Building 100 (Denver)"
    for decision in got.values():
        assert SITE_NAME_DISAMBIGUATED in decision.flags
        assert SITE_NAME_DUPLICATE not in decision.flags


def test_a_suffix_is_never_manufactured():
    """No counter, no "(2)", no "(duplicate)". A qualifier that is not in
    the evidence is a fact the compile does not have."""
    got = resolve_site_names([
        {"site_id": "s1", "name": "Pittsburgh Office", "source_text": "Pittsburgh Office"},
        {"site_id": "s2", "name": "Pittsburgh Office", "source_text": "Pittsburgh Office"},
    ])
    for decision in got.values():
        assert decision.name == "Pittsburgh Office"
        assert not any(ch.isdigit() for ch in decision.name)
        assert SITE_NAME_DUPLICATE in decision.flags


def test_prose_words_are_not_qualifiers():
    """The milestone line is full of tokens unique to it — "Cumulative",
    "Remaining". None of them tells a PM which building this is, and
    "Building 100 (Cumulative)" is worse than an honest flag."""
    got = resolve_site_names([
        {"site_id": "BUILDING-100", "name": "Building 100",
         "source_text": "Phase: Wave 1A | Location: Building 100 | September 8"},
        {"site_id": "BUILDING-ONE-HUNDRED-100", "name": "Building One Hundred 100",
         "source_text": MILESTONE_LINE},
    ])
    for decision in got.values():
        assert decision.name == "Building 100"
        assert "(" not in decision.name
        assert SITE_NAME_DUPLICATE in decision.flags


def test_a_half_disambiguated_group_is_refused_whole():
    """If only one member can be qualified, the other reads as canonical
    and the pair is more misleading than before. All or none."""
    got = resolve_site_names([
        {"site_id": "BUILDING-100", "name": "Building 100", "source_text": "x"},
        {"site_id": "BUILDING-100-AUSTIN", "name": "Building 100",
         "source_text": "x", "city": "Austin"},
    ])
    for decision in got.values():
        assert decision.name == "Building 100"
        assert SITE_NAME_DUPLICATE in decision.flags


def test_one_site_listed_twice_is_not_a_collision():
    got = resolve_site_names([
        {"site_id": "BUILDING-900", "name": "Building 900", "source_text": "Building 900"},
        {"site_id": "BUILDING-900", "name": "Building 900", "source_text": "Building 900"},
    ])
    assert got["BUILDING-900"].name == "Building 900"
    assert SITE_NAME_DUPLICATE not in got["BUILDING-900"].flags


# ── 4. The projection invariant ─────────────────────────────────────────

JACKSONVILLE = [
    {"site_id": "900", "name": "900", "source_text": "Building: 900 |"},
    {"site_id": "800", "name": "800", "source_text": "Building: 800 |"},
    {"site_id": "400", "name": "400", "source_text": "Building: 400 |"},
    {"site_id": "BUILDING-800", "name": "Building 800",
     "source_text": "Location: Building 800 |"},
    {"site_id": "BUILDING-900", "name": "Building 900",
     "source_text": "Phase: Wave 4 | Location: Building 900 | Planned Dates: July 1"},
    {"site_id": "BUILDING-100", "name": "building 100",
     "source_text": "Phase: Wave 1A | Location: Building 100 | Planned Dates: September 8"},
    {"site_id": "BUILDING-200", "name": "Building 200",
     "source_text": "Phase: Wave 1B | Location: Building 200 |"},
    {"site_id": "BUILDING-600", "name": "Building 600",
     "source_text": "Phase: Wave 1C | Location: Building 600 |"},
    {"site_id": "BUILDING-EIGHT-HUNDRED-800", "name": "building eight hundred 800",
     "source_text": MILESTONE_LINE},
    {"site_id": "BUILDING-SIX-HUNDRED-600", "name": "Building Six Hundred 600",
     "source_text": MILESTONE_LINE},
    {"site_id": "BUILDING-NINE-HUNDRED-900", "name": "Building Nine Hundred 900",
     "source_text": MILESTONE_LINE},
    {"site_id": "BUILDING-ONE-HUNDRED-100", "name": "Building One Hundred 100",
     "source_text": MILESTONE_LINE},
    {"site_id": "BUILDING-TWO-HUNDRED-200", "name": "Building Two Hundred 200",
     "source_text": MILESTONE_LINE},
    {"site_id": "BUILDING-FOUR-HUNDRED-400", "name": "building four hundred 400",
     "source_text": "Phase: Mobilization | Location: All Buildings | mobilizes four (4) technicians"},
]


def test_no_name_contains_a_token_the_evidence_does_not_carry():
    """The promotion gate's invariant, applied to the names this module
    produces: a name is a projection of the atom's own text and the
    identifier the compile established. Nothing else may appear in it.

    Non-vacuous by construction — the loop below asserts that names were
    in fact produced and in fact changed."""
    got = resolve_site_names(JACKSONVILLE)
    assert len(got) == 14
    changed = 0
    for entry in JACKSONVILLE:
        decision = got[entry["site_id"]]
        assert decision.name
        assert not unsupported_name_tokens(
            decision.name,
            entry["source_text"],
            entry["site_id"],
            entry["name"],
        ), (entry["site_id"], decision.name)
        changed += decision.changed
    assert changed >= 8


def test_the_fourteen_real_names_all_become_readable_or_flagged():
    got = resolve_site_names(JACKSONVILLE)
    for site_id, decision in got.items():
        readable = any(c.isalpha() for c in decision.name)
        assert readable or SITE_NAME_BARE_IDENTIFIER in decision.flags, site_id
    # Every one of the fourteen is nameable from its own evidence; none
    # needed the bare-identifier escape hatch.
    assert not any(
        SITE_NAME_BARE_IDENTIFIER in d.flags for d in got.values()
    )
    assert got["BUILDING-EIGHT-HUNDRED-800"].name == "Building 800"
    assert got["900"].name == "Building 900"
    # Six buildings, fourteen rows: every name collides with another.
    assert all(SITE_NAME_DUPLICATE in d.flags for d in got.values())
    assert len({d.name for d in got.values()}) == 6


# ── 5. Idempotency and order-invariance ─────────────────────────────────

def test_cleaning_a_cleaned_name_changes_nothing():
    for entry in JACKSONVILLE:
        once = clean_site_name(
            entry["name"], source_text=entry["source_text"], site_id=entry["site_id"],
        )
        twice = clean_site_name(
            once.name, source_text=entry["source_text"], site_id=entry["site_id"],
        )
        assert twice.name == once.name, entry["site_id"]
        assert not twice.changed


def test_the_deal_pass_is_idempotent_in_name_and_in_state():
    """Names are stable on a second pass, and so are the flags that
    describe the *state* of a name. ``number_word_collapsed`` is a
    provenance flag — it records what this run did, so it correctly stops
    firing once there is nothing left to collapse."""
    first = resolve_site_names(JACKSONVILLE)
    second = resolve_site_names([
        {**e, "name": first[e["site_id"]].name} for e in JACKSONVILLE
    ])
    assert {k: v.name for k, v in second.items()} == {k: v.name for k, v in first.items()}
    state = (SITE_NAME_BARE_IDENTIFIER, SITE_NAME_DUPLICATE)
    for site_id in first:
        assert (
            {f for f in second[site_id].flags if f in state}
            == {f for f in first[site_id].flags if f in state}
        ), site_id


def test_the_result_does_not_depend_on_row_order():
    expected = {k: (v.name, tuple(sorted(v.flags))) for k, v in
                resolve_site_names(JACKSONVILLE).items()}
    assert expected  # non-vacuous
    rng = random.Random(20260825)
    for _ in range(12):
        shuffled = JACKSONVILLE[:]
        rng.shuffle(shuffled)
        got = {k: (v.name, tuple(sorted(v.flags))) for k, v in
               resolve_site_names(shuffled).items()}
        assert got == expected


def test_disambiguation_does_not_depend_on_row_order():
    rows = [
        {"site_id": "B-100-AUSTIN", "name": "Building 100", "city": "Austin"},
        {"site_id": "B-100-DENVER", "name": "Building 100", "city": "Denver"},
        {"site_id": "B-100-RENO", "name": "Building 100", "city": "Reno"},
    ]
    expected = {k: v.name for k, v in resolve_site_names(rows).items()}
    assert expected == {
        "B-100-AUSTIN": "Building 100 (Austin)",
        "B-100-DENVER": "Building 100 (Denver)",
        "B-100-RENO": "Building 100 (Reno)",
    }
    for perm in ([2, 0, 1], [1, 2, 0], [2, 1, 0]):
        got = resolve_site_names([rows[i] for i in perm])
        assert {k: v.name for k, v in got.items()} == expected


# ── 6. Tier 1 may not stitch a name across a clause break ───────────────

def test_a_run_spanning_a_field_separator_is_not_a_name():
    """"…Building 600 | Planned Dates…" contains the token run "600
    Planned", and a site keyed ``site:600_planned`` would have been named
    by it. A name that steps over a bar is an artefact of the scan
    window."""
    text = "Location: Building 600 | Planned Dates: October 19"
    assert recover_site_display_names(
        sites=[{"site": "site:600_planned_dates", "aliases": []}],
        atoms=[{"text": text}],
        documents=[],
    ) == {}
    # Non-vacuity: the same run *inside* one clause still names the site.
    assert recover_site_display_names(
        sites=[{"site": "site:600_planned_dates", "aliases": []}],
        atoms=[{"text": "Location: Building 600 Planned Dates matter"}],
        documents=[],
    ) == {"site:600_planned_dates": "600 Planned Dates"}


def test_a_run_spanning_a_sentence_break_is_not_a_name():
    assert recover_site_display_names(
        sites=[{"site": "site:atmore_wind_creek", "aliases": []}],
        atoms=[{"text": "Work concludes at Atmore. Wind Creek follows."}],
        documents=[],
    ) == {}


def test_clause_bounding_leaves_real_prose_names_alone():
    """The names tier 1 exists to recover sit inside one clause, commas
    and all. Bounding the scan must not cost us any of them."""
    assert recover_site_display_names(
        sites=[{"site": "site:prudential_center_office", "aliases": []}],
        atoms=[{"text": "Staged at their Prudential Center office in Boston, MA."}],
        documents=[],
    ) == {"site:prudential_center_office": "Prudential Center office"}


def test_recovered_names_get_the_same_readability_pass():
    """A name recovered verbatim from the page is still the page's words,
    duplication and all. Tier 1 hands off to the same cleaner."""
    assert recover_site_display_names(
        sites=[{"site": "site:building_one_hundred_100", "aliases": []}],
        atoms=[{"text": "Milestone: Building one hundred (100) Complete"}],
        documents=[],
    ) == {"site:building_one_hundred_100": "Building 100"}
