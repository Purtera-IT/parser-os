"""A picture must be found wherever it was written, not only in a note field."""
from __future__ import annotations

from types import SimpleNamespace

from app.core.linked_pictures import stamp_linked_pictures

DIAGRAM = "https://huzzard.com/wp-content/uploads/2025/11/BPW061725-Rev-1-diagram.png"
WRAPPED = (
    "https://nam13.safelinks.protection.outlook.com/?url=https%3A%2F%2Fhuzzard.com%2Fwp-content"
    "%2Fuploads%2F2025%2F11%2FBPW061725-Rev-1-diagram.png&data=05%7C02%7C"
)


def atom(text: str, **value):
    return SimpleNamespace(text=text, value=dict(value))


def test_a_link_in_an_email_sentence_is_a_picture():
    a = atom(f"Diagram: {DIAGRAM} (rev 1)")
    assert stamp_linked_pictures([a]) == 1
    assert a.value["image_url"] == DIAGRAM  # the ") " did not come with it
    assert a.value["media_type"] == "image"


def test_a_gateway_wrapper_is_peeled_and_the_original_kept():
    a = atom(f"See the drawing here: {WRAPPED}")
    assert stamp_linked_pictures([a]) == 1
    assert a.value["image_url"] == DIAGRAM
    assert a.value["wrapped_url"] == WRAPPED


def test_a_picture_is_never_small_talk():
    a = atom(f"Thanks so much for this! Here you go: {DIAGRAM}", chatter=True)
    stamp_linked_pictures([a])
    assert "chatter" not in a.value


def test_links_that_are_not_pictures_are_left_alone():
    for text in ["See https://huzzard.com/products/bpw061725", "Our site: https://purtera-it.com/", "no link here"]:
        a = atom(text)
        assert stamp_linked_pictures([a]) == 0
        assert "image_url" not in a.value


def test_a_parser_that_already_found_it_is_not_second_guessed():
    a = atom(f"Diagram: {DIAGRAM}", image_url="https://huzzard.com/other.png", media_type="image")
    assert stamp_linked_pictures([a]) == 0
    assert a.value["image_url"] == "https://huzzard.com/other.png"


def test_a_broken_atom_cannot_cost_the_deal_its_parse():
    assert stamp_linked_pictures([None, SimpleNamespace(), atom(f"x {DIAGRAM}")]) == 1
