"""Three ways a thing that is not a site became one.

All three are baseline extraction defects, not learned ones: every pseudo-site
in the corpus carried ``facility_label.source == "deterministic_fallback"``,
meaning no head ever decided any of them.
"""

from __future__ import annotations

from app.core.address_parse import looks_like_street_address
from app.parsers.hubspot_note_parser import _field_value_shape
from app.parsers.site_roster_extractor import extract_site_roster, looks_like_site_roster


class TestStreetAddressGrammar:
    """A street line has a grammar. Prose with digits in it does not."""

    def test_real_street_lines_are_addresses(self) -> None:
        for value in (
            "200 Keller Smithfield Rd",
            "550 West Grange Avenue",
            "625 W. Adams St, Chicago, IL 60661",
            "2 Camino Vuestro Espanola, NM 87532",  # no street suffix, but a real place
        ):
            assert looks_like_street_address(value) is True, value

    def test_prose_with_numbers_is_not_an_address(self) -> None:
        """Every one of these minted a physical_site atom in the live corpus."""
        for value in (
            "4 Stands) Let me know if you need anything else! My best",
            "10 minutes per device The process is straightforward",
            "7 support model cater to both cost",
            "City/state for this location? - City/state for this location?",
            "Estimated 20-30 hours per week, mostly planned work on weekends "
            "(Friday/Saturday nights), with occasional incident-driven activities.",
        ):
            assert looks_like_street_address(value) is False, value

    def test_a_transcript_is_never_an_address(self) -> None:
        assert looks_like_street_address("x " * 400) is False


class TestCrmNotesDoNotMintSites:
    def test_a_real_address_in_a_note_still_becomes_a_site(self) -> None:
        assert _field_value_shape("200 Keller Smithfield Rd") == "address"
        assert _field_value_shape("550 West Grange Avenue, Milwaukee, WI 53207") == "address"

    def test_a_scope_narrative_does_not(self) -> None:
        narrative = (
            "Estimated 20-30 hours per week, mostly planned work on weekends "
            "(Friday/Saturday nights), with occasional incident-driven activities."
        )
        assert _field_value_shape(narrative) != "address"

    def test_an_email_sign_off_does_not(self) -> None:
        assert _field_value_shape("4 Stands) Let me know if you need anything else! My best") != "address"


class TestRosterIdentifiedByAddress:
    """A letterhead has one address; a roster has a different one per row."""

    COLUMNS = ["Address Line 1", "Cable Drop Count", "City", "Materials $$$", "State", "Zip Code"]
    ROWS = [
        ["1200 Airway Blvd", "4", "El Paso", "1200", "TX", "79925"],
        ["55 Main St", "2", "Lake Placid", "900", "NY", "12946"],
        ["700 Harbor Dr", "6", "San Diego", "2100", "CA", "92101"],
    ]

    def test_many_distinct_places_read_as_a_roster(self) -> None:
        assert looks_like_site_roster(columns=self.COLUMNS, rows=self.ROWS, surrounding_text="") is True

    def test_and_the_geography_survives(self) -> None:
        """The whole point: these rows used to keep the city as a NAME and drop
        the address, state and ZIP sitting in the next three columns."""
        rows = extract_site_roster(columns=self.COLUMNS, rows=self.ROWS)
        assert len(rows) == 3
        first = rows[0].as_dict()
        assert first["street_address"] == "1200 Airway Blvd"
        assert first["city"] == "El Paso"
        assert first["state"] == "TX"
        assert first["zip"] == "79925"

    def test_a_single_address_block_is_still_a_letterhead(self) -> None:
        assert looks_like_site_roster(
            columns=self.COLUMNS, rows=self.ROWS[:1], surrounding_text="",
        ) is False

    def test_two_addresses_are_remit_to_and_ship_to(self) -> None:
        assert looks_like_site_roster(
            columns=self.COLUMNS, rows=self.ROWS[:2], surrounding_text="",
        ) is False


def test_the_facility_head_is_correctable() -> None:
    """It was consulted in code and absent from the registry, so it could never
    be trained and the "<City> Office" rule ran on every deal forever."""
    from app.core.pm_feedback import HEAD_REGISTRY
    from app.core.site_facility_head import SITE_FACILITY_RELATION, _CANDIDATES

    spec = HEAD_REGISTRY["facility"]
    assert spec.relation == SITE_FACILITY_RELATION
    assert set(spec.candidates) == set(_CANDIDATES)
