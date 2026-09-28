"""A site key must name the building, not the postal area it stands in.

Live 010288 carried one site entity — no fragmentation, which looked right —
called ``site:bethesda_md_20814``. The atom it came from says

    Nesfield Performance Bethesda, 7832 Wisconsin Ave, Bethesda, MD 20814

and its parsed value carries ``facility_name``, ``street_address``, ``city``,
``state`` and ``zip`` as separate fields. The key was built from
city_state_zip: the two things that identify the place were discarded by the
line that had them in hand.

That is worse than 010180's seven fragments. A fragmented site is visibly
wrong. A confidently-wrong canonical looks right — and "bethesda md 20814"
cannot be told apart from any other building in that ZIP, so the next Bethesda
deal MERGES into this one.
"""
from pathlib import Path

from app.core.schemas import AtomType


def _note(address: str) -> str:
    """A HubSpot note in the shape the backfill reads."""
    return "\n".join([
        f"HubSpot Note: {address}",
        "HubSpot Note ID: 1",
        "Date: 2026-09-02T10:33:00.000Z",
        "Author: AJ Evans",
        "Author-Email: aj@purtera-it.com",
        "",
        address,
    ])


def _site_atom(tmp_path: Path, address: str):
    from app.core.note_provenance_backfill import ensure_hubspot_note_provenance

    tmp_path.mkdir(parents=True, exist_ok=True)
    note = tmp_path / "010288-hs-note-1-Address_.txt"
    note.write_text(_note(address), encoding="utf-8")
    out, minted = ensure_hubspot_note_provenance(
        [], project_id="deal-x", artifact_paths={"art_note": note})
    assert minted == 1
    return next(a for a in out if a.atom_type == AtomType.physical_site)


def _sites(atom) -> set[str]:
    return {k for k in atom.entity_keys if k.startswith("site:")}


NESFIELD = "7832 Wisconsin Ave, Bethesda, MD 20814"


def test_the_street_names_the_site(tmp_path: Path):
    assert "site:7832_wisconsin_ave_bethesda_md" in _sites(
        _site_atom(tmp_path, NESFIELD))


def test_the_zip_is_not_the_identity(tmp_path: Path):
    """It stays on the atom — it is simply not what the site is CALLED."""
    site = _site_atom(tmp_path, NESFIELD)
    assert site.value["zip"] == "20814"
    assert "site:bethesda_md_20814" not in _sites(site)


def test_two_buildings_in_one_zip_are_two_sites(tmp_path: Path):
    """The failure the old key guaranteed: every building in a ZIP collided,
    so the second one to arrive was absorbed by the first."""
    a = _site_atom(tmp_path / "a", "7832 Wisconsin Ave, Bethesda, MD 20814")
    b = _site_atom(tmp_path / "b", "4800 Hampden Lane, Bethesda, MD 20814")
    assert _sites(a) and _sites(b)
    assert _sites(a) != _sites(b)


def test_the_same_street_in_two_towns_stays_apart(tmp_path: Path):
    """Which is why city and state stay in the key."""
    a = _site_atom(tmp_path / "a", "100 Main Street, Bethesda, MD 20814")
    b = _site_atom(tmp_path / "b", "100 Main Street, Tampa, FL 33602")
    assert _sites(a) != _sites(b)
