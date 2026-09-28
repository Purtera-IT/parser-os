"""The corpus holds four CAD files and one is nothing like the others.

"BUMPER CONVEYOR MCP LOCATION.dwg" is an industrial electrical sheet: 75
layers named `$AUDIT-BAD-LAYER`, `001` and `0_...`, blocks called `eqklwmew`,
drawn in millimetres by a different CAD package. Every architectural rule in
`dwg_geometry` correctly finds nothing in it.

Finding nothing is the safe failure. It is not the right one, because that
sheet states a 480V service, three breaker capacities and eleven measured
lengths -- so these pin BOTH halves: the architectural rules must stay silent,
and the universal ones must still speak.
"""
from app.parsers.cad_layers import (
    CABLING_DISCIPLINES, follows_standard, has_discipline,
)

#: The real layer list, as this drawing ships.
JUNK = ["$AUDIT-BAD-LAYER", "-0", "0", "0 [SD 6]", "0-CENTER", "001", "01",
        "02J", "03", "05", "08", "0_", "121", "12F", "154487", "MCLBM1"]

#: SP-6's, which does follow the standard.
GOOD = ["AR-WALL-N", "AR-WALL-E", "AR-DOOR-N", "AR-GLAZ-N", "AR-MILL",
        "A-WALL", "AR-DOOR-E", "BC-WALL"]


def test_a_junk_layer_list_is_not_standard():
    assert follows_standard(JUNK) is False


def test_an_architectural_sheet_is_standard():
    assert follows_standard(GOOD) is True


def test_silence_from_an_unreadable_file_means_nothing():
    """The bug this exists for: two of the conveyor's 75 layers parsed as a
    discipline by coincidence, which was enough to certify the file as
    standards-following and then report 'no telecom or electrical information'
    about a MOTOR CONTROL PANEL drawing. Confident, and false."""
    assert has_discipline(JUNK, CABLING_DISCIPLINES) is None
    assert has_discipline(JUNK + ["A-WALL-N"], CABLING_DISCIPLINES) is None


def test_a_standard_sheet_can_still_report_an_absence():
    assert has_discipline(GOOD, CABLING_DISCIPLINES) is False
    assert has_discipline(GOOD + ["T-COMM-N"], CABLING_DISCIPLINES) is True


def test_a_handful_of_layers_is_not_enough_to_judge():
    """Three layers that happen to parse are a coincidence, not a standard."""
    assert follows_standard(["A-WALL-N", "A-DOOR-N"]) is False
