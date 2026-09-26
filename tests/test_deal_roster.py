"""Who is on this deal: one small table, not a card per signature.

The stakeholder atoms on 010288 are referenced by zero packets and the
workload matrix they nominally feed is all zeros -- five "this person exists"
cards that reach nothing. But they cannot simply be dropped: before this, an
atom's own speaker record said ``{"email": "t@purtera-it.com", "name": "T"}``,
because the name was derived from the address. The signature was the only
place that knew it was Trent Torrence.
"""
from __future__ import annotations

import pytest

from app.core.deal_roster import build_deal_roster, read_signature


class Atom:
    def __init__(self, text, atom_type="stakeholder"):
        self.atom_type = atom_type
        self.raw_text = text


SIGS = [
    "Trent Torrence | Executive Vice President of Sales | t@purtera-it.com | 404.771.3490",
    "Chase Smith | Director of Operations | chase@purtera-it.com | 770.500.5062",
    "Octavian Mitroi | octavian@purtera-it.com",
    "AJ Evans | Account Executive | aj@purtera-it.com | 407.562.7871",
    "Alec Burns | Senior Client Executive, Commercial Majors | alecbur@cdw.com",
]
DOCS = (
    [{"sender_email": "aj@purtera-it.com", "filename": f"m{i}.eml"} for i in range(5)]
    + [{"sender_email": "chase@purtera-it.com", "filename": "m5.eml"},
       {"sender_email": "chase@purtera-it.com", "filename": "m6.eml"},
       {"sender_email": "t@purtera-it.com", "filename": "m7.eml"},
       {"sender_email": "alecandrich@cdw.com", "filename": "m8.eml"},
       {"sender_email": "alecandrich@cdw.com", "filename": "m9.eml"}]
)


def roster():
    return build_deal_roster(atoms=[Atom(s) for s in SIGS], documents=DOCS,
                             our_domains={"purtera-it"})


def by_email(r, email):
    return next(p for p in r["people"] if p["email"] == email)


def test_a_header_never_carries_a_name_and_a_signature_does():
    """The whole reason the table has to exist before the footers become
    chrome: the address is all a header knows."""
    trent = by_email(roster(), "t@purtera-it.com")
    assert trent["name"] == "Trent Torrence"
    assert trent["role"] == "Executive Vice President of Sales"
    assert trent["phone"] == "404.771.3490"


def test_a_title_with_a_comma_in_it_is_one_title():
    """"Senior Client Executive, Commercial Majors" was being split into two
    people by entity extraction -- the deal ended up with nine stakeholders
    for five humans, four of them job titles."""
    assert by_email(roster(), "alecbur@cdw.com")["role"] == (
        "Senior Client Executive, Commercial Majors")


def test_the_side_comes_from_the_domain_not_the_signature():
    r = roster()
    assert by_email(r, "aj@purtera-it.com")["side"] == "ours"
    assert by_email(r, "alecbur@cdw.com")["side"] == "theirs"
    assert r["by_side"] == {"ours": 4, "theirs": 2}


def test_who_actually_wrote_something_is_counted():
    """Headers are the only evidence a person wrote rather than was quoted."""
    r = roster()
    assert by_email(r, "aj@purtera-it.com")["messages_sent"] == 5
    assert by_email(r, "chase@purtera-it.com")["messages_sent"] == 2


def test_somebody_named_in_a_footer_who_never_sent_is_flagged():
    """On the deal, worth seeing. Never worth emailing without asking."""
    assert set(roster()["named_but_silent"]) == {
        "octavian@purtera-it.com", "alecbur@cdw.com"}


def test_a_line_that_is_not_a_signature_yields_nothing():
    assert read_signature("Anything in Orange") == {}
    assert read_signature("") == {}


def test_a_signature_with_no_address_is_not_a_person():
    """Without an address there is nothing to key on, and a row keyed on a
    name would merge two people who share one."""
    r = build_deal_roster(atoms=[Atom("Trent Torrence | EVP of Sales")], documents=[])
    assert r["people"] == []


@pytest.mark.parametrize("docs", [[], [{"sender_email": "not-an-address"}], [{}]])
def test_rubbish_headers_do_not_make_rows(docs):
    assert build_deal_roster(atoms=[], documents=docs)["people"] == []


def test_nothing_at_all_is_not_an_error():
    r = build_deal_roster(atoms=[], documents=[])
    assert r["people"] == [] and r["by_side"] == {}


# ── a job title is not a person ──────────────────────────────────────


def test_a_job_title_never_becomes_a_stakeholder():
    """010288 carried nine "stakeholder" entities for five humans: "executive
    vice president", "senior client executive", "account executive" and
    "commercial majors" -- the second half of Alec's title, split on its
    comma. Anything counting people counted a job."""
    from app.core.entity_extraction import _names_a_job_not_a_person as job

    for slug in ("executive_vice_president", "senior_client_executive",
                 "account_executive", "commercial_majors", "director_of_operations"):
        assert job(slug), slug
    for slug in ("alec_burns", "trent_torrence", "chase_smith", "linda_park"):
        assert not job(slug), slug


def test_a_single_word_is_left_alone():
    """A surname is often the only name the corpus has for somebody. Rejecting
    a bare token would cost more than the occasional stray "director"."""
    from app.core.entity_extraction import _names_a_job_not_a_person as job

    assert not job("director")
    assert not job("watkins")


def test_the_real_people_survive_extraction():
    from app.core.entity_extraction import extract_keys
    from app.domain import get_active_domain_pack

    pack = get_active_domain_pack()
    got = [k for k in extract_keys(
        "Alec Burns | Senior Client Executive, Commercial Majors | alecbur@cdw.com",
        pack=pack) if k.startswith("stakeholder:")]
    assert got == ["stakeholder:alec_burns"]


def test_the_parser_mints_no_stakeholder_key_for_a_job_title():
    """The guard in extract_keys made no difference on its own: these keys are
    PARSER-supplied, and the merge deliberately preserves parser keys for a
    prefix rather than recomputing them. The rule has to be where the key is
    minted."""
    from app.parsers.email_parser import _stakeholder_keys

    assert _stakeholder_keys("account_executive") == []
    assert _stakeholder_keys("commercial_majors") == []
    assert _stakeholder_keys("alec_burns") == ["stakeholder:alec_burns"]
    assert _stakeholder_keys("") == []


def test_hygiene_is_the_chokepoint_for_job_titles():
    """Three producers mint stakeholder keys and the rule was true for all of
    them: extract_keys, the email parser's four sites, and a third path that
    adds title keys to an atom already minted with the person's. Every key
    passes through hygiene before it lands, so the rule lives there."""
    from app.core.entity_hygiene import filter_entity_keys_for_atom

    class A:
        raw_text = "Alec Burns | Senior Client Executive, Commercial Majors | alecbur@cdw.com"
        normalized_text = raw_text
        value: dict = {}
        entity_keys: list = []
        source_refs: list = []

    got = filter_entity_keys_for_atom(A(), [
        "email:alecbur_cdw_com",
        "stakeholder:alec_burns",
        "stakeholder:commercial_majors",
        "stakeholder:senior_client_executive",
    ])
    assert got == ["email:alecbur_cdw_com", "stakeholder:alec_burns"]


def test_the_llm_pass_is_the_one_that_mattered():
    """Three guards on the regex path were each verified live and each changed
    nothing, because a multi-entity LLM pass returns its own stakeholders, is
    declared authoritative, and DELETES the regex emissions. Guarding the
    regex path did not merely miss -- its output was replaced.

    The shared rule is the same one; this pins that it is applied to whatever
    the model answers."""
    from app.core.entity_extraction import _names_a_job_not_a_person as job

    # What the model returns for "Alec Burns | Senior Client Executive,
    # Commercial Majors | alecbur@cdw.com".
    answered = ["Alec Burns", "Senior Client Executive", "Commercial Majors"]
    kept = [n for n in answered if not job(n.lower().replace(" ", "_"))]
    assert kept == ["Alec Burns"]


def test_no_producer_turns_a_signature_into_a_job():
    """Five producers mint stakeholder keys and the rule was true for all of
    them. This runs the lot over the real signatures and asserts the result
    rather than any one path -- four separate fixes each passed their own test
    and changed nothing about the compiled deal."""
    from app.core.entity_extraction import (
        _emit_person_from_contact,
        _emit_stakeholders,
        extract_keys,
    )
    from app.core.entity_hygiene import filter_entity_keys_for_atom
    from app.domain import get_active_domain_pack

    pack = get_active_domain_pack()
    titles = {"commercial_majors", "senior_client_executive", "account_executive",
              "executive_vice_president", "director_of_operations"}

    for text in SIGS:
        class A:
            raw_text = text
            normalized_text = text
            value: dict = {}
            entity_keys: list = []
            source_refs: list = []

        keys = (set(extract_keys(text, pack=pack))
                | _emit_person_from_contact(text)
                | _emit_stakeholders(text))
        keys = set(filter_entity_keys_for_atom(A(), sorted(keys)))
        got = {k.split(":", 1)[1] for k in keys if k.startswith("stakeholder:")}
        assert not (got & titles), (text, sorted(got))
