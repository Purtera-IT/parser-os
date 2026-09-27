"""One quoted message is one atom, however many replies carry it.

An email thread arrives as N files, each quoting the whole history. On 010180,
33 of 38 documents are replies, so the header of the 27 July message was minted
33 times and the 5 August one 33 times: 75 `email_quoted_header` atoms for 11
real messages, a quarter of the deal's 292-atom envelope.

They escaped dedup by accident. `_value_key`'s `deal_metadata` branch looks for
`field_name` or `value`, a quoted header carries neither, so the key came back
None and the atom was never a dedup candidate at all. Quoted BODY lines already
collapse across documents -- 204 atoms, 201 distinct -- so this is the same rule
reaching a kind it had been silently skipping.

The line these tests hold: the header is identified by the MESSAGE, never by the
file that quoted it, and no two messages are merged on a guess about a name.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.core.semantic_dedup import _value_key


@dataclass
class _Ref:
    id: str
    artifact_id: str


@dataclass
class _Atom:
    value: dict
    atom_type: str = "deal_metadata"
    raw_text: str = ""
    source_refs: list = field(default_factory=list)
    receipts: list = field(default_factory=list)
    entity_keys: list = field(default_factory=list)
    review_flags: list = field(default_factory=list)
    confidence: float = 0.45


def _hdr(sender: str, sent_at: str, *, artifact: str = "a1", index: int = 0) -> _Atom:
    return _Atom(
        value={"kind": "quoted_message_header", "sender": sender,
               "sent_at": sent_at, "message_index": index},
        raw_text=f"From: {sender} | Sent: {sent_at}",
        source_refs=[_Ref(id=f"src_{artifact}_{index}", artifact_id=artifact)],
    )


JUL27 = "Monday, July 27, 2026 2:24 PM"
AUG5 = "Wednesday, August 5, 2026 11:11 AM"


def test_the_same_message_quoted_in_two_files_is_one_identity():
    a = _hdr("Erick Villalobos", JUL27, artifact="art_1", index=8)
    b = _hdr("Erick Villalobos", JUL27, artifact="art_2", index=19)
    assert _value_key(a) == _value_key(b) is not None


def test_the_file_and_the_position_in_it_are_not_part_of_the_identity():
    """The bug in one line: 33 copies differed only by artifact and
    message_index, and each was allowed to be its own atom."""
    key = _value_key(_hdr("Erick Villalobos", JUL27, artifact="art_1", index=8))
    for i, art in enumerate(("art_2", "art_3", "art_4")):
        assert _value_key(_hdr("Erick Villalobos", JUL27, artifact=art, index=i)) == key


def test_two_messages_from_one_sender_stay_apart():
    assert _value_key(_hdr("Erick Villalobos", JUL27)) != _value_key(_hdr("Erick Villalobos", AUG5))


def test_a_display_name_is_not_folded_into_an_addressed_one():
    """Live 010180: "Erick Villalobos" at 11:11 and "Erick Villalobos
    <erick.villalobos@cdw.com>" at 11:12 are a minute apart. They may well be
    one message with a clock skew, but merging them requires a guess about
    whether a bare name and an address are the same person at a different
    minute, and a wrong guess deletes a message from the thread."""
    bare = _hdr("Erick Villalobos", AUG5)
    full = _hdr("Erick Villalobos <erick.villalobos@cdw.com>",
                "Wednesday, August 5, 2026 11:12 AM")
    assert _value_key(bare) != _value_key(full)


def test_a_header_with_no_timestamp_is_left_out_of_dedup():
    """Keying on the sender alone would collapse every undated message that
    person ever sent into one. Abstaining keeps the old behaviour."""
    assert _value_key(_hdr("Erick Villalobos", "")) is None


def test_an_ordinary_deal_metadata_atom_is_unaffected():
    """The branch this shares is the one that keys real metadata fields."""
    a = _Atom(value={"field_name": "po_number", "value": "4500123"})
    b = _Atom(value={"field_name": "po_number", "value": "4500999"})
    assert _value_key(a) == _value_key(b) is not None
    assert _value_key(_Atom(value={"kind": "something_else"})) is None


def test_the_collapse_keeps_every_file_that_quoted_it():
    """The whole safety argument. A collapsed header must still be able to say
    which documents carried it, or the thread's shape is gone."""
    from app.core.semantic_dedup import _merge_atom_metadata

    winner = _hdr("Erick Villalobos", JUL27, artifact="art_1", index=8)
    for i, art in enumerate(("art_2", "art_3"), start=1):
        _merge_atom_metadata(winner, _hdr("Erick Villalobos", JUL27, artifact=art, index=i))
    assert {r.artifact_id for r in winner.source_refs} == {"art_1", "art_2", "art_3"}
