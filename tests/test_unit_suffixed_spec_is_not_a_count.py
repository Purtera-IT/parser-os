"""A number with a unit glued to it is a spec, never a quantity.

Live 010353: "Sapphire PTZ 2MP Camera" surfaced a headline quantity atom of
2 cameras -- the 2 is megapixels.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.core.atom_type_sanity import _iter_quantity_mentions, surface_headline_quantities


@pytest.mark.parametrize("text", [
    "Sapphire PTZ 2MP Camera",
    "name: Sapphire PTZ 2MP Camera",
    "Install 4K cameras",
    "Install 5MP cameras",
    "Sapphire PTZ 2 MP camera",
    "12V cameras",
    "Mount 1080p cameras",
    "Replace 500GB servers",
    "Swap 65in displays",
    "Wire 60Hz monitors",
])
def test_spec_numbers_never_become_counts(text):
    assert _iter_quantity_mentions(text) == []


@pytest.mark.parametrize("text,count", [
    ("Install 3 cameras", 3),
    ("Install 2x cameras", 2),
    ("Install two 1080p cameras", 2),
    ("Install 12 cameras", 12),
])
def test_real_counts_still_read(text, count):
    assert [n for n, _noun, _m in _iter_quantity_mentions(text)] == [count]


def test_headline_surfacing_mints_no_quantity_from_a_spec():
    atom = SimpleNamespace(
        atom_type="scope_item", raw_text="Sapphire PTZ 2MP Camera", text="",
        value={}, entity_keys=[], artifact_id="a1", source_refs=[],
    )
    assert surface_headline_quantities([atom], project_id="p") == []
