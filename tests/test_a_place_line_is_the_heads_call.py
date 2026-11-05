"""Whether a line names a place is the head's call, not the parser's veto.

`_is_site_line` gates every join in site_keys_follow_heads. It used to require the PARSER's type to
be physical_site AND the head to agree, so a line the two DISAGREED about was a place line to
neither, and invisible to all of them.

010171 is the case: one place named on two lines, and the read typed the PROSE line physical_site
and the bare address line as something else. The prose line failed the parser test, the address line
failed the head test, nothing folded them, and the deal priced site_count 0.
"""
import types

from app.core.site_keys_follow_heads import _is_site_line


def _atom(atom_type):
    return types.SimpleNamespace(id="a1", atom_type=atom_type, entity_keys=[], review_flags=[], value=None,
                                 text="", raw_text="")


def test_the_head_naming_a_place_is_enough_even_against_the_parser():
    # 010171's prose line: parser says scope_item, the read says physical_site.
    assert _is_site_line(_atom("scope_item"), {"a1": ["physical_site", 0.95]}) is True


def test_the_head_naming_something_else_settles_it_even_against_the_parser():
    # 010171's bare address line: parser says physical_site, the read says otherwise.
    assert _is_site_line(_atom("physical_site"), {"a1": ["work_scope_item", 0.72]}) is False


def test_keep_means_the_parsers_type_stands_by_the_heads_own_account():
    assert _is_site_line(_atom("physical_site"), {"a1": ["_keep", 0.99]}) is True
    assert _is_site_line(_atom("scope_item"), {"a1": ["_keep", 0.99]}) is False


def test_an_absent_answer_leaves_it_to_the_parser():
    assert _is_site_line(_atom("physical_site"), {"a1": ["_absent", 0.0]}) is True
    assert _is_site_line(_atom("scope_item"), {"a1": ["_absent", 0.0]}) is False


def test_a_line_the_read_never_saw_leaves_it_to_the_parser():
    assert _is_site_line(_atom("physical_site"), {}) is True
    assert _is_site_line(_atom("scope_item"), {}) is False


def test_both_agreeing_is_unchanged():
    assert _is_site_line(_atom("physical_site"), {"a1": ["physical_site", 0.99]}) is True
    assert _is_site_line(_atom("small_talk"), {"a1": ["small_talk", 0.99]}) is False
