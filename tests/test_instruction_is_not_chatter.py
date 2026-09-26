"""An instruction aimed at a colleague is coordination, not banter.

Chase's 11:31 message on 010288 says "AJ- get the location and well confirm if
we can do it." It is why anybody went looking for a site at all, and it reached
the compiled envelope in no form: the parser typed it scope_item, the substance
gate judged it conversational prose, and every copy was deleted. The coverage
report filed it under `unclaimed` -- the parser saying out loud that it read the
line and nobody claimed it.

It read as chatter because the prose test wants a capital letter after the first
word, and the only proper noun in the sentence is the person being told to do
something, which IS the first word.

The branch below the drop already handled this exact shape correctly -- keep it,
retype it out of scope, leave it auditable -- and cited the same live example the
drop branch cited. Both were right about the shape and only one could run.
"""
from __future__ import annotations

import app.core.atom_substance_gate as G
from app.core.deal_roster import read_signature


class _Ref:
    locator: dict = {}
    filename = "010288-hs-email-116256511315.eml"
    page = None
    artifact_type = None


class _Atom:
    def __init__(self, text: str, atom_type: str = "scope_item") -> None:
        self.id = "atm_test"
        self.raw_text = text
        self.normalized_text = text
        self.atom_type = atom_type
        self.value = {"kind": "email_body_line"}
        self.entity_keys: list[str] = []
        self.review_flags: list[str] = []
        self.source_refs = [_Ref()]
        self.confidence = 0.6


def test_an_instruction_to_a_colleague_survives_the_gate():
    """The sentence the deal turns on, deleted on every compile of 010288."""
    atom = _Atom("AJ- get the location and well confirm if we can do it.")
    kept, dropped = G.apply_substance_gate([atom])
    assert kept and not dropped
    # Kept out of scope, not kept AS scope: it is an instruction, not a line
    # item, and the heads that price scope should never see it.
    assert str(kept[0].atom_type).endswith("deal_metadata")
    assert kept[0].value.get("retagged_from") == "scope_item"


def test_the_shape_the_drop_branch_named_as_its_own_example():
    """"TT - clean up aisle Purtera." was cited by BOTH branches, one saying
    drop and one saying keep-and-retype. Initials count as a name."""
    kept, dropped = G.apply_substance_gate([_Atom("TT - clean up aisle Purtera.")])
    assert kept and not dropped


def test_a_promise_to_act_is_not_chatter_even_without_the_apostrophe():
    """It arrived as "well confirm", not "we'll confirm"."""
    assert not G._is_conversational_prose("well confirm if we can do it", [])
    assert not G._is_conversational_prose("we'll confirm if we can do it", [])


def test_banter_still_goes():
    """Keeping instructions is only right if the gate still does its job. These
    are the lines it exists for -- no deal content, no instruction, no promise."""
    for text in (
        "Excited to knock this out of the park with y'all.",
        "Thanks so much again!",
        "Looking forward to it.",
        "Hope all is well.",
        "This has been received and we are on it.",
    ):
        kept, dropped = G.apply_substance_gate([_Atom(text)])
        # Either dropped, or demoted out of scope -- never left as scope_item.
        assert dropped or str(kept[0].atom_type).endswith("deal_metadata"), text


def test_a_greeting_is_not_mistaken_for_an_instruction():
    """"Hi AJ," addresses somebody and tells them nothing. The delimiter has to
    follow the name immediately, or every greeting becomes a task."""
    assert G._DIRECTED_TASK_RE.match("Hi AJ, hope all is well.") is None
    assert G._DIRECTED_TASK_RE.match("AJ- get the location") is not None


def test_a_contact_handed_over_in_a_body_keeps_its_name():
    """Chase asked Trent for the access-control contact out of CA; Trent replied
    with `"Albert Arzate" <albert@rd-systems.com>`. The signature reader returned
    the mailbox and dropped the name, so the deal's only third-party contact
    would have shown as a bare address. Outlook writes it twice, and the doubled
    form is the one the compile sees."""
    for text in (
        '"Albert Arzate" <albert@rd-systems.com>',
        '"Albert Arzate" <albert@rd-systems.com<mailto:albert@rd-systems.com>>',
    ):
        got = read_signature(text)
        assert got.get("name") == "Albert Arzate", text
        assert got.get("email") == "albert@rd-systems.com", text


def test_a_real_signature_still_reads_as_before():
    got = read_signature(
        "Chase Smith | Director of Operations | chase@purtera-it.com | 770.500.5062")
    assert got["name"] == "Chase Smith"
    assert got["role"] == "Director of Operations"
    assert got["email"] == "chase@purtera-it.com"


def test_a_quoted_name_with_an_address_beside_it_is_a_person():
    """Chase asked Trent for the access-control contact out of CA. Trent answered
    with `"Albert Arzate" <albert@rd-systems.com>`, and the deal's only
    third-party contact appeared in no atom and no roster row.

    It was deleted for being quoted. The rule is right about contracts, where
    '("Customer Contact Person")' is a defined term rather than a human -- but
    Outlook hands a colleague's contact over in exactly the same quotes, with the
    address sitting next to the name. The address is what separates the two.
    """
    from app.core.atom_substance_gate import drop_contextless_stakeholders
    from app.core.schemas import AtomType

    def person(text, name):
        a = _Atom(text, AtomType.stakeholder)
        a.value = {"name": name, "kind": "email_body_line"}
        return a

    for text in (
        '"Albert Arzate" <albert@rd-systems.com>',
        '"Albert Arzate" <albert@rd-systems.com<mailto:albert@rd-systems.com>>',
        "Albert Arzate <albert@rd-systems.com>",
    ):
        kept, dropped = drop_contextless_stakeholders([person(text, "Albert Arzate")])
        assert kept and not dropped, text

    # The thing the rule was written for still goes: quotes, and nothing else.
    kept, dropped = drop_contextless_stakeholders(
        [person('("Customer Contact Person")', "Customer Contact Person")])
    assert dropped and not kept
