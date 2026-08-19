"""Tests for single-site task anchoring."""

from __future__ import annotations

from app.core.site_task_anchor import anchor_orphan_atoms_to_confirmed_site


class _Site:
    atom_type = "physical_site"
    entity_keys = ["site:highland_park_mi_48203"]
    value = {"site_id": "HIGHLAND-PARK-MI-48203", "city": "Highland Park", "state": "MI"}


class _Task:
    def __init__(self, text: str, keys=None):
        self.atom_type = "task"
        self.raw_text = text
        self.entity_keys = keys or ["device:access_point"]
        self.review_flags = []


def test_single_site_links_orphan_tasks() -> None:
    atoms = [_Site(), _Task("Install wireless APs"), _Task("Site survey")]
    out, n = anchor_orphan_atoms_to_confirmed_site(atoms)
    assert n == 2
    for t in out[1:]:
        assert "site:highland_park_mi_48203" in t.entity_keys
        assert "single_site_anchor" in t.review_flags


def test_multi_site_noop() -> None:
    class _SiteB:
        atom_type = "physical_site"
        entity_keys = ["site:alpharetta_ga"]
        value = {}

    atoms = [_Site(), _SiteB(), _Task("Install APs")]
    _, n = anchor_orphan_atoms_to_confirmed_site(atoms)
    assert n == 0
    assert "site:highland_park_mi_48203" not in atoms[2].entity_keys


def test_task_with_existing_site_untouched() -> None:
    task = _Task("Work", keys=["site:other", "device:ap"])
    atoms = [_Site(), task]
    _, n = anchor_orphan_atoms_to_confirmed_site(atoms)
    assert n == 0
    assert task.entity_keys == ["site:other", "device:ap"]
