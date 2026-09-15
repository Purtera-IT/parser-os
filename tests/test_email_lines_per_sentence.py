"""An email line that holds two sentences is two atoms, however the sentences end.

Live 010198 (2026-09-15): the customer's request line -- "We will be setting
just 1 Square register and 1 kitchen printer. I would also like to add a
Ubiquiti router with cellular backup - is that something you can help with?"
-- was one atom, typed as the question it ends with, so the work in its first
sentence never became a task and the Deal Kit's one priced line went unmatched.
"""
from app.parsers.email_parser import _expand_lines_to_sentences


def _pieces(line):
    return [t for _, t in _expand_lines_to_sentences([line], 7)]


def test_a_statement_followed_by_a_question_is_two_atoms():
    line = ("We will be setting just 1 Square register and 1 kitchen printer. I would also like to "
            "add a Ubiquiti router with cellular backup - is that something you can help with?")
    assert _pieces(line) == [
        "We will be setting just 1 Square register and 1 kitchen printer.",
        "I would also like to add a Ubiquiti router with cellular backup - is that something you can help with?",
    ]


def test_every_piece_keeps_the_line_it_came_from_and_its_quote_prefix():
    out = _expand_lines_to_sentences(["> Please mount the display on the north wall. Can your tech bring a lift?"], 3)
    assert [n for n, _ in out] == [3, 3]
    assert out[0][1].startswith("> Please mount") and out[1][1].startswith("> Can your tech")


def test_one_sentence_stays_one_line_whatever_it_ends_with():
    assert _pieces("Is that something you can help with?") == ["Is that something you can help with?"]
    assert _pieces("Thanks!") == ["Thanks!"]
    assert _pieces("Version 2.1.4 shipped") == ["Version 2.1.4 shipped"]  # dots inside a token are not endings


def test_table_rows_and_fragments_never_split():
    assert _pieces("Qty | Part no. | Desc.") == ["Qty | Part no. | Desc."]
    assert _pieces("Ok. Yes.") == ["Ok. Yes."]  # two endings, but no substantial sentence
