"""A store row's number is not a quantity: the head reading "0220: 6034 Baltimore National Pike" as a place clears the
normal bar, while "200 AP's" keeps the raised one (010035, 2026-11-05)."""
from __future__ import annotations

from types import SimpleNamespace as NS

from app.core import c3_type_base as m


def _atom(text, type_="bom_line"):
    return NS(id="a", raw_text=text, atom_type=type_, value={"quantity": 220, "unit": "count"}, entity_keys=[])


def test_a_store_row_with_a_street_takes_the_normal_bar():
    assert m._overwrite_bar(_atom("0220: 6034 Baltimore National Pike"), "physical_site", 0.5) == 0.5


def test_a_counted_item_keeps_the_raised_bar():
    assert m._overwrite_bar(_atom("200 AP's"), "physical_site", 0.5) > 0.5


def test_a_loop_is_a_street_too():
    assert m._overwrite_bar(_atom("2842: 250 McWhirt Loop"), "physical_site", 0.5) == 0.5


def test_a_street_word_first_is_a_street():
    assert m._overwrite_bar(_atom("1434: 4119 Avenue of the Cities"), "physical_site", 0.5) == 0.5


def test_a_store_number_then_a_house_number_is_a_street():
    assert m._overwrite_bar(_atom("6940: 12 N Broadway"), "physical_site", 0.5) == 0.5
    assert m._overwrite_bar(_atom("220: 24 Cat6 drops"), "physical_site", 0.5) > 0.5
