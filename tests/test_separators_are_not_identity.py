"""A customer spelling their own name two ways must not create two sites.

Live deal 010302 writes "SymphonyAI" three times and "Symphony AI" twice, in
its own documents. The extractor slugs whichever spelling it saw, so one
office became two keys — and whichever won a given compile became the row,
which read as the site renaming itself between runs.
"""

from __future__ import annotations

from app.core.entity_resolution import collect_site_alias_groups


class _Atom:
    _n = 0

    def __init__(self, entity_keys, text=""):
        _Atom._n += 1
        self.id = f"atm_{_Atom._n}"
        self.atom_type = "scope_item"
        self.entity_keys = entity_keys
        self.value = {}
        self.text = text
        self.raw_text = text
        self.artifact_id = "art_1"
        self.confidence = 0.8
        self.authority_class = "machine_extractor"


def _grouped(*keysets):
    groups = collect_site_alias_groups([_Atom(list(k)) for k in keysets])
    return [set(g) for g in groups]


def test_the_two_spellings_of_one_customer_are_one_site() -> None:
    groups = _grouped(
        ["site:symphonyai_hillview_office"],
        ["site:symphony_ai_hillview_office"],
    )
    assert any(
        {"site:symphonyai_hillview_office", "site:symphony_ai_hillview_office"} <= g
        for g in groups
    ), groups


def test_keys_carrying_digits_are_left_to_the_numbered_site_guard() -> None:
    """"atl_hq_01" and "atlhq01" are the same words, but a downstream hygiene
    pass deliberately keeps digit-bearing keys apart so "building_1" and
    "building_2" are never merged. Separator-insensitivity does not override
    that — the guard against merging numbered sites is worth more than the
    handful of unseparated codes it declines to join."""
    groups = _grouped(["site:atl_hq_01"], ["site:atlhq01"])
    assert not any({"site:atl_hq_01", "site:atlhq01"} <= g for g in groups), groups


def test_different_words_are_still_different_sites() -> None:
    """Moving separators is a spelling difference; different letters are not."""
    groups = _grouped(["site:palo_alto_office"], ["site:spokane_office"])
    assert not any(
        {"site:palo_alto_office", "site:spokane_office"} <= g for g in groups
    ), groups


def test_a_numeric_suffix_still_marks_a_different_site() -> None:
    """"building_1" and "building_2" differ in their digits, so they stay
    apart — the letters-and-digits comparison keeps the number."""
    groups = _grouped(["site:building_1"], ["site:building_2"])
    assert not any({"site:building_1", "site:building_2"} <= g for g in groups), groups


def test_one_key_alone_is_not_a_group() -> None:
    assert _grouped(["site:only_one"]) == [] or all(
        len(g) >= 2 for g in _grouped(["site:only_one"])
    )
