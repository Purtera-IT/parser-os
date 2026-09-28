"""Derive by role, not by name.

The first version of the drawing derivation matched "AR-WALL-N" literally.
That is BR Design's spelling. The standard says "A-WALL-N"; other firms ship
"A-WALL-NEWW". A hardcoded name finds nothing on the next deal's drawing and
reports NO new partition rather than reporting that it could not tell -- a
confident zero, which is the worst answer available.
"""
from app.parsers.cad_layers import (
    CABLING_DISCIPLINES, has_discipline, layers_with_role, parse_layer,
)

SP6 = ["AR-WALL-N", "AR-WALL-E", "AR-WALL-DEM-N", "AR-WALL-DEM-FUTURE",
       "AR-DOOR-N", "BC-WALL", "DEFPOINTS", "0", "FRN-FURN"]


def test_the_same_role_under_four_spellings():
    """Every one of these is new partition to the firm that drew it."""
    for name in ("A-WALL-N", "AR-WALL-N", "A-WALL-NEWW", "A_WALL_NEW"):
        lay = parse_layer(name)
        assert lay.role == "partition", name
        assert lay.intent == "new", name


def test_demolition_is_said_in_the_minor_field():
    """"AR-WALL-DEM-N" is major WALL, minor DEM, status N. Read strictly that
    is "new", and reading it strictly reported zero demolition on a sheet with
    144 feet of it."""
    lay = parse_layer("AR-WALL-DEM-N")
    assert lay.status == "new"        # the status field, read honestly
    assert lay.demolition is True
    assert lay.intent == "demolish"   # what the layer is actually FOR
    assert parse_layer("AR-WALL-DEM-FUTURE").intent == "future_demolition"


def test_roles_are_found_without_naming_a_layer():
    assert layers_with_role(SP6, "partition", "new") == ["AR-WALL-N"]
    assert layers_with_role(SP6, "partition", "demolish") == ["AR-WALL-DEM-N"]
    assert layers_with_role(SP6, "door") == ["AR-DOOR-N"]


def test_cad_furniture_is_not_content():
    for name in ("0", "DEFPOINTS", "DIMENSIONS"):
        assert parse_layer(name).is_content is False


def test_a_firms_own_code_is_not_mistaken_for_a_discipline():
    """"BC" is this firm's base-building code, not the NCS "C" for Civil.
    Guessing a discipline from a first letter puts base building in the wrong
    trade."""
    lay = parse_layer("BC-WALL")
    assert lay.discipline is None
    assert lay.role == "partition"    # the major group still reads


def test_absent_cabling_layers_are_detectable():
    assert has_discipline(SP6, CABLING_DISCIPLINES) is False
    assert has_discipline(SP6 + ["T-COMM-N"], CABLING_DISCIPLINES) is True


def test_a_file_that_ignores_the_standard_says_none_not_false():
    """Silence only means something when the file speaks the language. A
    drawing with no parseable discipline anywhere cannot be said to be missing
    one -- and `no telecom layer` must not fire on it."""
    assert has_discipline(["Layer1", "Walls", "STUFF"], CABLING_DISCIPLINES) is None
