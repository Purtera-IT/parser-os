"""Where the deal IS, as a handful of lines instead of forty atoms.

On live 010180, 43 of 252 non-rejected atoms were the deal MOVING -- who does
the survey, which week, is Tuesday free, thanks-got-the-floorplans. Not one
line of it reaches a SOW, and as atoms they competed for attention with the
two numbers that move money.
"""
from app.core.deal_state import read_deal_state


class _A:
    def __init__(self, text):
        self.raw_text = text


def _state(*texts):
    return read_deal_state([_A(t) for t in texts])


ROM = "PurTera has already provided a budgetary ROM of approximately $110K."
ASK = "Wants us to do a site survey to dial in a true quote."


def test_an_unsurveyed_rom_blocks_the_deal_kit():
    """The point of the whole module. 010180's $110,108 came from a solutions
    architect listening to a call recording; nobody from PurTera had been on
    site. A deal kit built on that is a firm promise resting on a guess."""
    st = _state(ROM, ASK)
    assert "NOT survey-backed" in st.get("price_basis").value
    assert st.get("blocks").value.startswith("firm SOW / deal kit")
    assert st.get("stage").value == "awaiting site survey"


def test_a_planned_survey_is_not_a_completed_one():
    """The bug this exists for. Both of 010180's completion-shaped sentences
    were PLANS:

        "we can tackle the cabling ... once survey is completed"
        "wait until the wireless survey is complete so that we know where APs go"

    Read as completions they said the price had been checked against the site
    when nobody had been there -- switching OFF the one gate this module holds,
    silently. A future survey is the REASON the deal is waiting."""
    for planned in (
        "As a side note, we can tackle the cabling/installation for the wireless "
        "APs once survey is completed.",
        "It would make sense to wait until the wireless survey is complete so "
        "that we know where APs and cabling will be.",
        "We will send the firm quote after the survey is completed.",
        "Pending the site survey being complete, we can finalise.",
    ):
        st = _state(ROM, ASK, planned)
        assert "NOT survey-backed" in st.get("price_basis").value, planned
        assert st.get("blocks") is not None, planned


def test_a_reported_survey_lifts_the_block():
    """It has to be liftable, or it is not a gate, it is a wall."""
    st = _state(ROM, ASK, "The site survey is complete and the team is writing it up.")
    assert st.get("price_basis").value == "ROM, survey-backed"
    assert st.get("blocks") is None


def test_a_deal_with_no_rom_says_nothing_about_price_basis():
    """Most deals are not waiting on anything, and a module that always has an
    opinion is a module nobody reads."""
    st = _state("Two Cat6A drops per workstation.", "Six 48-port patch panels.")
    assert st.get("price_basis") is None
    assert st.get("blocks") is None
    assert st.lines == []


def test_coordination_is_gathered_not_judged():
    """Scheduling, acknowledgements and chasing are the deal moving. They are
    collected so they can be consolidated -- never deleted here."""
    st = _state(ROM, "Can you provide availability for the walkthrough?",
                "Thanks, got the floor plans.",
                "Following up on this one.",
                "Two Cat6A drops per workstation.")
    said = {getattr(a, "raw_text", "") for a in st.coordination}
    assert "Thanks, got the floor plans." in said
    assert "Two Cat6A drops per workstation." not in said   # that is content


def test_the_money_is_carried_into_the_line():
    st = _state("Gave them a ROM for $110k for the job.", ASK)
    assert "$110k" in st.get("price_basis").value
