"""Orbit must not ask where a site is when the deal says where the site is.

On 010288 the PM's blocker list opened with:

    1. Where is this site located?
    2. @Trent Torrence whats that guys contact out of CA for access control?

The deal holds "Nesfield Performance Bethesda, 7832 Wisconsin Ave, Bethesda, MD
20814" and '"Albert Arzate" <albert@rd-systems.com>'. AJ asked the reseller and
the reseller answered; Chase asked Trent and Trent answered. Both questions are
the WORK of getting to a fact the deal already carries.

Neither linkage mechanism could catch them. The site question carries no entity
keys at all, so key-sharing has nothing to share; the address atom is
note-derived and carries no thread position, so it can never be the "direct next
reply" that cross-thread pairing requires. Both are plumbing. What a person does
is simpler: the deal states the site, so nobody asks where the site is.
"""
from __future__ import annotations

from app.core.orbitbrief_core import build_pm_dashboard


class _Atom:
    def __init__(self, text: str, atom_type: str, *, internal: bool = False) -> None:
        self.id = "atm_" + str(abs(hash(text)))[:10]
        self.artifact_id = "art_email"
        self.raw_text = text
        self.atom_type = atom_type
        self.value = {"internal_only": True} if internal else {}
        self.review_flags: list[str] = []
        self.review_status = None
        self.entity_keys: list[str] = []
        self.source_refs: list = []


SITE = "Nesfield Performance Bethesda, 7832 Wisconsin Ave, Bethesda, MD 20814"
CONTACT = '"Albert Arzate" <albert@rd-systems.com>'


def _summaries(atoms):
    d = build_pm_dashboard(atoms=atoms, packets=[], edges=[], entities=[])
    return [b.get("summary", "") for b in (d.get("blockers") or [])]


def test_where_is_the_site_is_not_a_blocker_when_the_deal_states_the_site():
    atoms = [
        _Atom("Where is this site located?", "open_question"),
        _Atom(SITE, "physical_site"),
    ]
    assert not any("Where is this site" in s for s in _summaries(atoms))


def test_it_is_still_a_blocker_when_the_deal_has_no_site():
    """The rule is "we already know", not "never ask"."""
    atoms = [_Atom("Where is this site located?", "open_question")]
    assert any("Where is this site" in s for s in _summaries(atoms))


def test_an_internal_question_is_never_a_blocker():
    """Purtera asking Purtera is coordination, not a gap to put to a customer."""
    atoms = [
        _Atom("@Trent Torrence whats that guys contact out of CA?",
              "open_question", internal=True),
    ]
    # The dashboard always adds its standard SRL gaps; what must not appear is
    # the internal question itself.
    assert not any("whats that guys contact" in s for s in _summaries(atoms))


def test_a_contact_question_is_not_a_blocker_once_somebody_is_named():
    atoms = [
        _Atom("Who is the contact for access control?", "open_question"),
        _Atom(CONTACT, "stakeholder"),
    ]
    assert not any("contact for access control" in s for s in _summaries(atoms))


def test_an_unrelated_question_still_blocks():
    """The suppression is narrow: it fires on the thing the deal actually holds."""
    atoms = [
        _Atom("What are the payment terms?", "open_question"),
        _Atom(SITE, "physical_site"),
    ]
    assert any("payment terms" in s for s in _summaries(atoms))
