"""A roster row's stated Site No. is that place's identity, so one row is one site.

010447 priced 366 sites from 235 stated site numbers: 126 of those numbers carried
two keys, one from the number ("Site No.: dc 5509") and one from the location name
on the same row ("Site Location: San Diego MDO"). The rows are flattened pipe-joined
text, so `_id_columns` sees no table cells and `_join_same_place` could not tell the
two keys were one place.

The head that owns this judgment is `jdg:site` (answers same_site / distinct_site,
replaces "parser: entity_resolution site fusion"). It is asked by no read today, the
read path has no pair-size support, and its holdout sits at 0.984 against a guess of
1.0 with no swap test -- below guess level. Until it is promoted, the document's own
stated field is read as extraction and the fold is recorded in the rule ledger as a
removal target.
"""
import re
import types

from app.core.site_keys_follow_heads import _STATED_SITE_NO


def _slug(text):
    m = _STATED_SITE_NO.search(text)
    return re.sub(r"[^a-z0-9]+", "", m.group(1).lower()) if m else None


def test_the_number_is_read_whatever_else_the_row_says():
    assert _slug("Site No.: dc 5509 | net-new (y/n): Yes | replace/upgrade (y/n): No") == "dc5509"
    assert _slug("Site No.: dc 5883 | Site Location: Keasbey BDC | State: NJ") == "dc5883"


def test_the_separator_the_real_roster_uses():
    # The live file writes "dc  5441" with two spaces, and three on some rows.
    assert _slug("Site No.: dc  5441 | net-new") == "dc5441"
    assert _slug("Site No.: dc   5496 | x") == "dc5496"


def test_suffixed_and_hyphenated_numbers_keep_their_shape():
    assert _slug("Site No.: dc 5852a | Yes") == "dc5852a"
    assert _slug("Site No.: dc 7279-s dc | replace") == "dc7279s"


def test_a_store_number_with_no_letters():
    assert _slug("Store #: 4471") == "4471"


def test_a_count_is_not_a_place():
    # The place noun must be followed by a number WORD; "Qty"/"SqFt" are not.
    assert _slug("Qty: 4471") is None
    assert _slug("Bldg SqFt: 42648") is None


def test_a_part_number_is_not_a_place():
    assert _slug("Firewall Model: PA-3410") is None


def test_a_scope_line_about_locations_is_not_a_place_identity():
    # These three are 010447's junk roster entries; none states a site number.
    assert _slug("Install Palo Alto pair only - USA - 37 locations") is None
    assert _slug("145 Location Palo Alto deployment") is None
    assert _slug("COMPUTACENTER UNITED STATES INC. is registered in the State of Delaware") is None


def test_a_location_name_alone_names_no_number():
    assert _slug("Site Location: Keasbey BDC") is None


def test_place_ids_offers_the_stated_number_to_the_join():
    from app.core.site_keys_follow_heads import _place_ids

    atom = types.SimpleNamespace(id="a1", raw_text="Site No.: dc 5509 | Site Location: San Diego MDO",
                                 text="Site No.: dc 5509 | Site Location: San Diego MDO",
                                 atom_type="physical_site", entity_keys=["site:san_diego_mdo"],
                                 review_flags=[], value=None)
    assert "id:site_no:dc5509" in _place_ids(atom, {}, set())


def test_a_line_stating_no_number_offers_nothing():
    from app.core.site_keys_follow_heads import _place_ids

    atom = types.SimpleNamespace(id="a1", raw_text="Install Palo Alto pair only - USA - 37 locations",
                                 text="Install Palo Alto pair only - USA - 37 locations",
                                 atom_type="physical_site", entity_keys=["site:install_palo_alto_pair_only"],
                                 review_flags=[], value=None)
    assert _place_ids(atom, {}, set()) == []


def test_an_address_the_line_states_in_prose_is_read_for_the_join():
    """010171 names one place on two lines: "4250 Executive Square, Suite 900, La Jolla, CA 92037"
    and "San Diego Office - 4250 Executive Square, Suite 900, La Jolla, CA 92037". Neither carries
    a `value.street_address`, so the join saw no shared place."""
    from app.core.site_keys_follow_heads import _stated_address_id

    bare = _stated_address_id("4250 Executive Square, Suite 900, La Jolla, CA 92037")
    prose = _stated_address_id("San Diego Office - 4250 Executive Square, Suite 900, La Jolla, CA 92037")
    assert bare == prose == "addr:4250:executive:92037"


def test_a_zip_plus_four_still_names_the_place():
    from app.core.site_keys_follow_heads import _stated_address_id

    assert _stated_address_id("1 Main St, Akron, OH 44301-1234") == "addr:1:main:44301"


def test_a_count_earlier_in_the_sentence_is_not_the_street_number():
    """The street head is the LAST one before the ", ST ZIP" tail."""
    from app.core.site_keys_follow_heads import _stated_address_id

    got = _stated_address_id("37 locations in scope - ship to 1 University Ave, Westwood, MA 02090")
    assert got == "addr:1:university:02090"


def test_a_line_with_no_address_tail_names_no_place():
    from app.core.site_keys_follow_heads import _stated_address_id

    for text in ("Install Palo Alto pair only - USA - 37 locations",
                 "145 Location Palo Alto deployment",
                 "Site No.: dc 5509 | net-new (y/n): Yes",
                 "COMPUTACENTER UNITED STATES INC. is registered in the State of Delaware (US)"):
        assert _stated_address_id(text) is None, text


def test_the_stated_address_does_not_displace_what_the_parser_extracted():
    """`value.street_address` wins; the text is read only when the parser read nothing."""
    from app.core.site_keys_follow_heads import _place_ids

    atom = types.SimpleNamespace(
        id="a1", atom_type="physical_site", entity_keys=["site:x"], review_flags=[],
        value={"street_address": "1 Main St", "zip": "44301"},
        text="1 Main St, Akron, OH 44301", raw_text="1 Main St, Akron, OH 44301")
    ids = _place_ids(atom, {}, set())
    assert "addr:1:main:44301" in ids
    assert len([i for i in ids if i.startswith("addr:")]) == 1


def _site_atom(aid, atom_type, keys, text=""):
    return types.SimpleNamespace(id=aid, atom_type=atom_type, entity_keys=list(keys),
                                 review_flags=[], value=None, text=text, raw_text=text,
                                 decision_provenance=None)


def test_a_line_that_states_its_place_joins_whatever_the_head_types_it():
    """010447: 140 of its roster rows are typed `site_infrastructure` -- what is AT the site, not the
    site -- and those are the rows carrying the location-name keys. "Site No.: dc 5841 | Site
    Location: Hialeah Miami DFC" is keyed site:hialeah_miami_dfc while the bare row is keyed
    site:dc5841: same place, two keys, and 116 of 239 stated numbers were still in that state after
    the first pass, because the fold only looked at lines typed physical_site.

    Which place a line REFERS to is not the same question as what the line IS.
    """
    from app.core import rule_ledger, site_keys_follow_heads as sk

    bare = _site_atom("a1", "physical_site", ["site:dc5841"], "Site No.: dc 5841 | net-new (y/n): No")
    named = _site_atom("a2", "site_infrastructure", ["site:hialeah_miami_dfc"],
                       "Site No.: dc 5841 | Site Location: Hialeah Miami DFC | State: FL")
    atoms = [bare, named]
    types = {"a1": ["physical_site", 0.95], "a2": ["site_infrastructure", 0.95]}
    assert not rule_ledger.shadow("site_keys.join_same_place")
    joined = sk._join_same_place(atoms, types)
    keys = {k for a in atoms for k in sk._site_keys(a)}
    assert joined == 1, "the site_infrastructure row was not folded"
    assert keys == {"site:dc5841"}, keys


def test_two_different_numbers_are_two_places():
    from app.core import site_keys_follow_heads as sk

    a = _site_atom("a1", "physical_site", ["site:dc5841"], "Site No.: dc 5841 | net-new: No")
    b = _site_atom("a2", "physical_site", ["site:dc5842"], "Site No.: dc 5842 | net-new: Yes")
    atoms = [a, b]
    sk._join_same_place(atoms, {"a1": ["physical_site", 0.95], "a2": ["physical_site", 0.95]})
    assert {k for x in atoms for k in sk._site_keys(x)} == {"site:dc5841", "site:dc5842"}


def test_a_line_with_no_site_key_is_not_pulled_into_the_fold():
    from app.core import site_keys_follow_heads as sk

    a = _site_atom("a1", "physical_site", ["site:dc5841"], "Site No.: dc 5841 | net-new: No")
    chat = _site_atom("a2", "small_talk", [], "Site No.: dc 5841 was mentioned on the call")
    atoms = [a, chat]
    sk._join_same_place(atoms, {"a1": ["physical_site", 0.95], "a2": ["small_talk", 0.99]})
    assert sk._site_keys(chat) == []
    assert sk._site_keys(a) == ["site:dc5841"]
